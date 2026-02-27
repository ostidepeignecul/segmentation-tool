"""Controller for anchored mask modification mode on annotation overlays."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from models.annotation_model import AnnotationModel
from models.temp_mask_model import TempMaskModel
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
        temp_mask_model: TempMaskModel,
        annotation_view: AnnotationView,
        mask_modification_service: MaskModificationService,
        refresh_overlay: Callable[..., None],
        refresh_roi_overlay_for_slice: Callable[[int], None],
        set_position_label: Callable[[int, int], None],
        status_message: Callable[..., None],
    ) -> None:
        self.view_state_model = view_state_model
        self.annotation_model = annotation_model
        self.temp_mask_model = temp_mask_model
        self.annotation_view = annotation_view
        self.mask_modification_service = mask_modification_service
        self._refresh_overlay = refresh_overlay
        self._refresh_roi_overlay_for_slice = refresh_roi_overlay_for_slice
        self._set_position_label = set_position_label
        self._status_message = status_message
        self._mod_preview_slices: set[int] = set()
        self._mod_preview_base: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._mod_preview_current: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def reset(self, *, restore_overlay: bool = False) -> None:
        self.mask_modification_service.reset()
        self.annotation_view.clear_mod_anchor_points()
        had_preview = self._clear_mod_preview_from_temp()
        if restore_overlay or had_preview:
            self._refresh_current_roi_overlay()

    def on_tool_mode_changed(self, mode: str) -> None:
        if str(mode) != "mod":
            self.mask_modification_service.end_drag()
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
        slice_idx = int(self.view_state_model.current_slice)
        effective_slice = self._effective_slice_mask(slice_idx)
        if effective_slice is None:
            return
        started = self.mask_modification_service.start_drag(
            slice_idx=slice_idx,
            label=label,
            x_pos=x,
            y_pos=y,
            slice_mask=effective_slice,
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
        slice_idx = int(self.view_state_model.current_slice)
        updated_slice = self.mask_modification_service.drag_to(
            slice_idx=slice_idx,
            x_pos=x,
            y_pos=y,
        )
        if updated_slice is None:
            return
        self._set_position_label(x, y)
        self._store_mod_preview_slice(slice_idx=slice_idx, updated_slice=updated_slice)
        self.refresh_preview(changed_slice=slice_idx)
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
        slice_idx = int(self.view_state_model.current_slice)
        effective_slice = self._effective_slice_mask(slice_idx)
        if effective_slice is None:
            return
        created = self.mask_modification_service.add_anchor_on_contour(
            slice_idx=slice_idx,
            label=label,
            x_pos=x,
            y_pos=y,
            slice_mask=effective_slice,
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
        cleared_preview = self._clear_mod_preview_from_temp()
        if had_dirty:
            self._status_message("Modifications masque annulees.", timeout_ms=1800)
        if had_dirty or cleared_preview:
            self._refresh_current_roi_overlay()
        return bool(self._is_mod_mode() or had_dirty or cleared_preview)

    def on_roi_delete_requested(self) -> None:
        """Keep mod pending state consistent when ROI temp deletion is requested."""
        had_dirty = self.mask_modification_service.cancel_pending()
        self.mask_modification_service.end_drag()
        self.annotation_view.clear_mod_anchor_points()
        self._mod_preview_slices.clear()
        self._mod_preview_base.clear()
        self._mod_preview_current.clear()
        if had_dirty:
            self._status_message("Modifications masque supprimees.", timeout_ms=1800)

    def has_pending_edits(self) -> bool:
        return self.mask_modification_service.has_pending_edits()

    def commit_pending_edits(self) -> bool:
        changed = self.mask_modification_service.commit()
        if not changed:
            return False
        self.annotation_view.clear_mod_anchor_points()
        self._mod_preview_slices.clear()
        self._mod_preview_base.clear()
        self._mod_preview_current.clear()
        # Temp data is already in TempMaskModel; apply/clear is handled by the standard pipeline.
        self._status_message("Modifications masque appliquees.", timeout_ms=1800)
        return True

    def refresh_preview(self, *, changed_slice: Optional[int] = None) -> None:
        if not self._is_mod_mode() and not self.mask_modification_service.has_pending_edits():
            return
        if changed_slice is None:
            self._refresh_current_roi_overlay()
            return
        self._refresh_roi_overlay_for_slice(int(changed_slice))

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

    def _volume_depth(self) -> int:
        ann = self.annotation_model.get_mask_volume()
        if ann is not None:
            return int(ann.shape[0])
        temp = self.temp_mask_model.get_mask_volume()
        if temp is not None:
            return int(temp.shape[0])
        return 0

    def _annotation_slice(self, slice_idx: int) -> Optional[np.ndarray]:
        mask_volume = self.annotation_model.get_mask_volume()
        if mask_volume is None:
            return None
        idx = int(slice_idx)
        if idx < 0 or idx >= mask_volume.shape[0]:
            return None
        return np.asarray(mask_volume[idx], dtype=np.uint8)

    def _effective_slice_mask(self, slice_idx: int) -> Optional[np.ndarray]:
        base_slice = self._annotation_slice(slice_idx)
        if base_slice is None:
            return None
        effective = np.array(base_slice, copy=True)
        temp_slice = self.temp_mask_model.get_slice_mask(slice_idx)
        temp_cov = self.temp_mask_model.get_slice_coverage(slice_idx)
        if temp_slice is not None and temp_cov is not None:
            arr = np.asarray(temp_slice, dtype=np.uint8)
            cov = np.asarray(temp_cov, dtype=bool)
            if arr.shape == effective.shape and cov.shape == effective.shape:
                effective[cov] = arr[cov]
        return effective

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

    def _ensure_temp_model_shape(self, shape: tuple[int, int, int]) -> bool:
        temp_volume = self.temp_mask_model.get_mask_volume()
        if temp_volume is None or temp_volume.shape != tuple(shape):
            self.temp_mask_model.initialize(shape)
            self._mod_preview_slices.clear()
            self._mod_preview_base.clear()
            temp_volume = self.temp_mask_model.get_mask_volume()
        return temp_volume is not None and temp_volume.shape == tuple(shape)

    def _sync_temp_palette_from_annotation(self) -> None:
        for label, color in self.annotation_model.get_label_palette().items():
            self.temp_mask_model.ensure_label(int(label), tuple(int(c) for c in color), visible=True)

    def _store_mod_preview_slice(self, *, slice_idx: int, updated_slice: np.ndarray) -> None:
        idx = int(slice_idx)
        base_slice = self._annotation_slice(idx)
        if base_slice is None:
            return
        if not self._ensure_temp_model_shape((self._volume_depth(), base_slice.shape[0], base_slice.shape[1])):
            return
        self._sync_temp_palette_from_annotation()

        if idx not in self._mod_preview_base:
            existing_slice = self.temp_mask_model.get_slice_mask(idx)
            existing_cov = self.temp_mask_model.get_slice_coverage(idx)
            if existing_slice is None:
                base_temp_slice = np.zeros(base_slice.shape, dtype=np.uint8)
            else:
                base_temp_slice = np.asarray(existing_slice, dtype=np.uint8).copy()
            if existing_cov is None:
                base_temp_cov = np.zeros(base_slice.shape, dtype=bool)
            else:
                base_temp_cov = np.asarray(existing_cov, dtype=bool).copy()
            self._mod_preview_base[idx] = (base_temp_slice, base_temp_cov)
        self._rebase_mod_preview_baseline(slice_idx=idx)

        updated = np.asarray(updated_slice, dtype=np.uint8)
        if updated.shape != base_slice.shape:
            return
        coverage = updated != base_slice
        self.temp_mask_model.set_slice_data(idx, updated, coverage)
        self._mod_preview_slices.add(idx)
        self._mod_preview_current[idx] = (np.array(updated, copy=True), np.array(coverage, copy=True))

    def _rebase_mod_preview_baseline(self, *, slice_idx: int) -> None:
        idx = int(slice_idx)
        if idx not in self._mod_preview_base:
            return
        expected = self._mod_preview_current.get(idx)
        if expected is None:
            return
        current_slice = self.temp_mask_model.get_slice_mask(idx)
        current_cov = self.temp_mask_model.get_slice_coverage(idx)
        if current_slice is None or current_cov is None:
            return
        expected_slice, expected_cov = expected
        cur_slice = np.asarray(current_slice, dtype=np.uint8)
        cur_cov = np.asarray(current_cov, dtype=bool)
        if cur_slice.shape != expected_slice.shape or cur_cov.shape != expected_cov.shape:
            return
        external_mask = np.logical_or(cur_slice != expected_slice, cur_cov != expected_cov)
        if not np.any(external_mask):
            return
        base_slice, base_cov = self._mod_preview_base[idx]
        rebased_slice = np.array(base_slice, copy=True)
        rebased_cov = np.array(base_cov, copy=True)
        rebased_slice[external_mask] = cur_slice[external_mask]
        rebased_cov[external_mask] = cur_cov[external_mask]
        self._mod_preview_base[idx] = (rebased_slice, rebased_cov)

    def _clear_mod_preview_from_temp(self) -> bool:
        tracked = (
            set(self._mod_preview_slices)
            | set(self._mod_preview_base.keys())
            | set(self._mod_preview_current.keys())
        )
        if not tracked:
            return False
        for idx in sorted(int(s) for s in tracked):
            base = self._mod_preview_base.get(idx)
            if base is None:
                self.temp_mask_model.clear_slice(idx)
                continue
            self.temp_mask_model.set_slice_data(idx, base[0], base[1])
        self._mod_preview_slices.clear()
        self._mod_preview_base.clear()
        self._mod_preview_current.clear()
        return True

    def _refresh_current_roi_overlay(self) -> None:
        self._refresh_roi_overlay_for_slice(int(self.view_state_model.current_slice))

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
