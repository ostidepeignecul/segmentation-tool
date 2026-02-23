"""Controller for anchored mask modification mode on annotation overlays."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from models.annotation_model import AnnotationModel
from models.overlay_data import OverlayData
from models.view_state_model import ViewStateModel
from services.mask_modification_service import MaskModificationService
from views.annotation_view import AnnotationView


class MaskModificationController:
    """Coordinate mask contour edition in `mod` tool mode."""

    def __init__(
        self,
        *,
        view_state_model: ViewStateModel,
        annotation_model: AnnotationModel,
        annotation_view: AnnotationView,
        mask_modification_service: MaskModificationService,
        refresh_overlay: Callable[..., None],
        restore_overlay: Callable[..., None],
        set_position_label: Callable[[int, int], None],
        status_message: Callable[..., None],
    ) -> None:
        self.view_state_model = view_state_model
        self.annotation_model = annotation_model
        self.annotation_view = annotation_view
        self.mask_modification_service = mask_modification_service
        self._refresh_overlay = refresh_overlay
        self._restore_overlay = restore_overlay
        self._set_position_label = set_position_label
        self._status_message = status_message

    def reset(self, *, restore_overlay: bool = False) -> None:
        self.mask_modification_service.reset()
        self.annotation_view.clear_mod_anchor_points()
        if restore_overlay:
            self._restore_overlay()

    def on_tool_mode_changed(self, mode: str) -> None:
        if str(mode) != "mod":
            self.mask_modification_service.end_drag()
            self.annotation_view.clear_mod_anchor_points()
            self._restore_overlay()
            return

        if not self._ensure_context():
            self.annotation_view.clear_mod_anchor_points()
            return
        self.sync_anchors()
        self.refresh_preview()

    def on_active_label_changed(self, _label_id: int) -> None:
        if not self._is_mod_mode():
            self.annotation_view.clear_mod_anchor_points()
            return
        self.mask_modification_service.clear_active_component()
        self.sync_anchors()

    def on_slice_changed(self, _slice_idx: int) -> None:
        if not self._is_mod_mode():
            self.annotation_view.clear_mod_anchor_points()
            return
        self.mask_modification_service.clear_active_component()
        self.sync_anchors()

    def on_drag_started(self, pos: Any) -> None:
        if not self._is_mod_mode():
            return
        point = self._parse_pos(pos)
        if point is None:
            return
        if not self._ensure_context():
            return

        label = self._active_label_for_mod()
        if label is None:
            self._status_message("Selectionne un label > 0 pour modifier le masque.", timeout_ms=1800)
            return
        x, y = point
        started = self.mask_modification_service.start_drag(
            slice_idx=int(self.view_state_model.current_slice),
            label=label,
            x_pos=x,
            y_pos=y,
        )
        if not started:
            self._status_message("Clique directement sur un point d'ancrage.", timeout_ms=1400)
            return
        self._set_position_label(x, y)
        self.sync_anchors()

    def on_drag_moved(self, pos: Any) -> None:
        if not self._is_mod_mode():
            return
        point = self._parse_pos(pos)
        if point is None:
            return

        x, y = point
        moved = self.mask_modification_service.drag_to(
            slice_idx=int(self.view_state_model.current_slice),
            x_pos=x,
            y_pos=y,
        )
        if not moved:
            return
        self._set_position_label(x, y)
        self.refresh_preview()
        self.sync_anchors()

    def on_drag_finished(self, _pos: Any) -> None:
        self.mask_modification_service.end_drag()
        self.sync_anchors()

    def on_double_clicked(self, pos: Any) -> None:
        if not self._is_mod_mode():
            return
        point = self._parse_pos(pos)
        if point is None:
            return
        if not self._ensure_context():
            return

        label = self._active_label_for_mod()
        if label is None:
            self._status_message("Selectionne un label > 0 pour modifier le masque.", timeout_ms=1800)
            return
        x, y = point
        created = self.mask_modification_service.add_anchor_on_contour(
            slice_idx=int(self.view_state_model.current_slice),
            label=label,
            x_pos=x,
            y_pos=y,
        )
        if not created:
            if self.mask_modification_service.has_full_density_anchors(
                slice_idx=int(self.view_state_model.current_slice),
                label=label,
            ):
                self._status_message(
                    "Tous les pixels du contour sont deja ancres.",
                    timeout_ms=1800,
                )
            else:
                self._status_message("Double-clique proche du contour pour ajouter un ancrage.", timeout_ms=1800)
            return
        self.sync_anchors()
        self._status_message("Ancrage ajoute.", timeout_ms=1000)

    def on_selection_cancel_requested(self) -> bool:
        if not self._is_mod_mode() and not self.mask_modification_service.has_pending_edits():
            return False
        had_dirty = self.mask_modification_service.cancel_pending()
        self.annotation_view.clear_mod_anchor_points()
        self._restore_overlay()
        if had_dirty:
            self._status_message("Modifications masque annulees.", timeout_ms=1800)
        return True

    def has_pending_edits(self) -> bool:
        return self.mask_modification_service.has_pending_edits()

    def commit_pending_edits(self) -> bool:
        payload = self.mask_modification_service.commit()
        if payload is None:
            return False
        self.annotation_model.set_mask_volume(payload.mask_volume, preserve_labels=True)
        self.annotation_view.clear_mod_anchor_points()
        self._refresh_overlay(defer_volume=True, rebuild=True)
        self._status_message("Modifications masque appliquees.", timeout_ms=1800)
        return True

    def refresh_preview(self) -> None:
        if not self._is_mod_mode():
            return
        mask_volume = self.mask_modification_service.preview_mask_volume()
        if mask_volume is None:
            return
        palette = self.annotation_model.get_label_palette()
        visible_labels = self.annotation_model.get_visible_labels()
        overlay = OverlayData(
            mask_volume=np.asarray(mask_volume, dtype=np.uint8),
            palette=dict(palette),
            label_volumes=None,
        )
        self.annotation_view.set_overlay(overlay, visible_labels=visible_labels)

    def sync_anchors(self) -> None:
        if not self._is_mod_mode():
            self.annotation_view.clear_mod_anchor_points()
            return
        label = self._active_label_for_mod()
        if label is None:
            self.annotation_view.clear_mod_anchor_points()
            return
        points = self.mask_modification_service.anchor_points(
            slice_idx=int(self.view_state_model.current_slice),
            label=label,
        )
        if not points:
            self.annotation_view.clear_mod_anchor_points()
            return
        self.annotation_view.set_mod_anchor_points(
            points,
            active=True,
            active_index=self.mask_modification_service.active_anchor_index(),
        )

    def _ensure_context(self) -> bool:
        mask_volume = self.annotation_model.get_mask_volume()
        if mask_volume is None:
            return False
        return self.mask_modification_service.ensure_loaded(mask_volume)

    def _active_label_for_mod(self) -> Optional[int]:
        label = self.view_state_model.active_label
        if label is None:
            return None
        try:
            value = int(label)
        except Exception:
            return None
        if value <= 0:
            return None
        return value

    def _is_mod_mode(self) -> bool:
        return str(getattr(self.view_state_model, "tool_mode", "") or "") == "mod"

    @staticmethod
    def _parse_pos(pos: Any) -> Optional[tuple[int, int]]:
        if not isinstance(pos, (tuple, list)) or len(pos) != 2:
            return None
        try:
            return int(pos[0]), int(pos[1])
        except Exception:
            return None
