"""Controller for corrosion profile editing flow in corrosion Endview."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from models.annotation_model import AnnotationModel
from models.view_state_model import ViewStateModel
from services.corrosion_profile_edit_service import CorrosionProfileEditService
from services.cscan_corrosion_service import CScanCorrosionService
from controllers.endview_controller import EndviewController


class CorrosionProfileController:
    """Coordinates corrosion profile interactions between view/controller/service."""

    def __init__(
        self,
        *,
        view_state_model: ViewStateModel,
        annotation_model: AnnotationModel,
        endview_controller: EndviewController,
        annotation_controller: Any,
        cscan_controller: Any,
        cscan_corrosion_service: CScanCorrosionService,
        corrosion_profile_edit_service: CorrosionProfileEditService,
        get_volume: Callable[[], Optional[np.ndarray]],
        set_position_label: Callable[[int, int], None],
        status_message: Callable[..., None],
        apply_roi_fallback: Callable[[], None],
    ) -> None:
        self.view_state_model = view_state_model
        self.annotation_model = annotation_model
        self.endview_controller = endview_controller
        self.annotation_controller = annotation_controller
        self.cscan_controller = cscan_controller
        self.cscan_corrosion_service = cscan_corrosion_service
        self.corrosion_profile_edit_service = corrosion_profile_edit_service
        self._get_volume = get_volume
        self._set_position_label = set_position_label
        self._status_message = status_message
        self._apply_roi_fallback = apply_roi_fallback

    def on_active_label_changed(self, _label_id: int) -> None:
        self.sync_anchors()

    def on_apply_roi_requested(self) -> None:
        """Apply ROI normally, or commit pending corrosion profile edits."""
        if self.view_state_model.corrosion_active and self.corrosion_profile_edit_service.has_pending_edits():
            self.commit_pending_edits()
            return
        self._apply_roi_fallback()

    def on_drag_started(self, pos: Any) -> None:
        if not self.view_state_model.corrosion_active:
            return
        point = self._parse_pos(pos)
        if point is None:
            return
        x, y = point
        if not self.ensure_context():
            return

        target_is_a = self.corrosion_profile_edit_service.resolve_target_from_active_label(
            self.view_state_model.active_label
        )
        if target_is_a is None:
            self._status_message("Selectionne le label corrosion a editer (A ou B).", timeout_ms=2000)
            return
        if not self.corrosion_profile_edit_service.start_drag(
            slice_idx=int(self.view_state_model.current_slice),
            target_is_a=target_is_a,
            x_pos=x,
            y_pos=y,
        ):
            return
        self._set_position_label(x, y)
        self.sync_anchors()

    def on_drag_moved(self, pos: Any) -> None:
        if not self.view_state_model.corrosion_active:
            return
        point = self._parse_pos(pos)
        if point is None:
            return
        x, y = point
        if not self.ensure_context():
            return
        moved = self.corrosion_profile_edit_service.drag_to(
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
        self.corrosion_profile_edit_service.end_drag()
        self.sync_anchors()

    def on_double_clicked(self, pos: Any) -> None:
        if not self.view_state_model.corrosion_active:
            return
        point = self._parse_pos(pos)
        if point is None:
            return
        x, y = point
        if not self.ensure_context():
            return

        target_is_a = self.corrosion_profile_edit_service.resolve_target_from_active_label(
            self.view_state_model.active_label
        )
        if target_is_a is None:
            self._status_message("Selectionne le label corrosion a editer (A ou B).", timeout_ms=2000)
            return
        created = self.corrosion_profile_edit_service.add_anchor_on_line(
            slice_idx=int(self.view_state_model.current_slice),
            target_is_a=target_is_a,
            x_pos=x,
            y_pos=y,
        )
        if not created:
            self._status_message(
                "Double-clique directement sur la ligne active pour ajouter un ancrage.",
                timeout_ms=1800,
            )
            return
        self.refresh_preview()
        self.sync_anchors()
        self._status_message("Ancrage ajoute.", timeout_ms=1000)

    def ensure_context(self) -> bool:
        volume = self._get_volume()
        if volume is None or getattr(volume, "ndim", 0) != 3:
            return False
        label_ids = self.view_state_model.corrosion_overlay_label_ids
        peak_map_a = self.view_state_model.corrosion_peak_index_map_a
        peak_map_b = self.view_state_model.corrosion_peak_index_map_b
        if label_ids is None or peak_map_a is None or peak_map_b is None:
            return False
        return self.corrosion_profile_edit_service.ensure_context(
            peak_map_a=peak_map_a,
            peak_map_b=peak_map_b,
            label_ids=label_ids,
            image_shape=(int(volume.shape[1]), int(volume.shape[2])),
            cscan_corrosion_service=self.cscan_corrosion_service,
        )

    def refresh_preview(self) -> None:
        """Show temporary corrosion profile edits in corrosion endview only."""
        overlay = self.corrosion_profile_edit_service.preview_overlay()
        if overlay is None:
            return
        palette = self.view_state_model.corrosion_overlay_palette or self.annotation_model.get_label_palette()
        visible_labels = self.annotation_model.get_visible_labels()
        self.endview_controller.set_corrosion_profile_preview_overlay(
            overlay,
            palette=dict(palette),
            visible_labels=visible_labels,
        )

    def sync_anchors(self) -> None:
        if not self.view_state_model.corrosion_active:
            self.endview_controller.clear_corrosion_profile_anchors()
            return
        if not self.ensure_context():
            self.endview_controller.clear_corrosion_profile_anchors()
            return
        target_is_a = self.corrosion_profile_edit_service.resolve_target_from_active_label(
            self.view_state_model.active_label
        )
        if target_is_a is None:
            self.endview_controller.clear_corrosion_profile_anchors()
            return
        points = self.corrosion_profile_edit_service.anchor_points(
            slice_idx=int(self.view_state_model.current_slice),
            target_is_a=target_is_a,
        )
        self.endview_controller.set_corrosion_profile_anchors(points, active=True)

    def commit_pending_edits(self) -> bool:
        """Commit temporary corrosion profile edits (triggered by Apply ROI)."""
        if not self.ensure_context():
            return False
        payload = self.corrosion_profile_edit_service.commit(
            cscan_corrosion_service=self.cscan_corrosion_service,
            rebuild_projection=True,
        )
        if payload is None:
            return False

        self.view_state_model.corrosion_peak_index_map_a = payload.peak_map_a
        self.view_state_model.corrosion_peak_index_map_b = payload.peak_map_b
        self.view_state_model.corrosion_overlay_volume = payload.overlay_volume
        if payload.projection is not None and payload.value_range is not None:
            self.view_state_model.corrosion_projection = (payload.projection, payload.value_range)
            self.view_state_model.corrosion_interpolated_projection = (
                payload.projection,
                payload.value_range,
            )
            self.view_state_model.corrosion_active = True

        self.annotation_model.set_mask_volume(payload.overlay_volume)
        palette = self.view_state_model.corrosion_overlay_palette or {}
        if palette:
            self.annotation_model.label_palette = dict(palette)
            self.annotation_model.label_visibility = {int(lbl): True for lbl in palette.keys()}
            self.annotation_model.ensure_persistent_labels()

        volume = self._get_volume()
        if volume is not None:
            self.cscan_controller.update_views(volume)
        self.annotation_controller.refresh_overlay(defer_volume=False, rebuild=True)
        self.sync_anchors()
        self._status_message("Profil corrosion applique.", timeout_ms=2000)
        return True

    @staticmethod
    def _parse_pos(pos: Any) -> Optional[tuple[int, int]]:
        if not isinstance(pos, (tuple, list)) or len(pos) != 2:
            return None
        try:
            return int(pos[0]), int(pos[1])
        except Exception:
            return None
