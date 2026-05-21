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

    MOD_DRAG_START_DISTANCE_PX = 3

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
        self._mod_press_pos: Optional[tuple[int, int]] = None
        self._mod_press_slice_idx: Optional[int] = None
        self._mod_press_label: Optional[int] = None
        self._mod_drag_started: bool = False

    def reset(self, *, restore_overlay: bool = False) -> None:
        self.mask_modification_service.reset()
        self._reset_mod_pointer_state()
        self.annotation_view.clear_mod_anchor_points()
        had_preview = self._clear_mod_preview_from_temp()
        if restore_overlay or had_preview:
            self._refresh_current_roi_overlay()

    def on_tool_mode_changed(self, mode: str) -> None:
        if str(mode) != "mod":
            self.mask_modification_service.clear_active_component()
            self._reset_mod_pointer_state()
            self.annotation_view.clear_mod_anchor_points()
            return
        if self._is_corrosion_layer_mode():
            self.mask_modification_service.clear_active_component()
            self._reset_mod_pointer_state()
            self.annotation_view.clear_mod_anchor_points()
            return

        if not self._ensure_context():
            self._reset_mod_pointer_state()
            self.annotation_view.clear_mod_anchor_points()
            return
        self.sync_anchors()
        self.refresh_preview()

    def on_active_label_changed(self, _label_id: int) -> None:
        if self._is_corrosion_layer_mode():
            self.mask_modification_service.clear_active_component()
            self._reset_mod_pointer_state()
            self.annotation_view.clear_mod_anchor_points()
            return
        if not self._is_mod_mode():
            self.mask_modification_service.clear_active_component()
            self._reset_mod_pointer_state()
            self.annotation_view.clear_mod_anchor_points()
            return
        self.sync_anchors()

    def on_slice_changed(self, _slice_idx: int) -> None:
        if self._is_corrosion_layer_mode():
            self.mask_modification_service.clear_active_component()
            self._reset_mod_pointer_state()
            self.annotation_view.clear_mod_anchor_points()
            return
        if not self._is_mod_mode():
            self.mask_modification_service.clear_active_component()
            self._reset_mod_pointer_state()
            self.annotation_view.clear_mod_anchor_points()
            return
        self.mask_modification_service.clear_active_component()
        self._reset_mod_pointer_state()
        self.sync_anchors()

    def on_drag_started(self, pos: Any) -> None:
        if not self._is_mod_mode():
            return
        point = self._parse_pos(pos)
        if point is None:
            return
        if not self._ensure_context():
            return

        slice_idx = int(self.view_state_model.current_slice)
        effective_slice = self._effective_slice_mask(slice_idx)
        if effective_slice is None:
            return

        label = self._resolve_mod_label_at_point(point=point, slice_mask=effective_slice)
        if label is None:
            self._status_message("Clique sur un mask pour le modifier.", timeout_ms=1800)
            return
        self._mod_press_pos = point
        self._mod_press_slice_idx = slice_idx
        self._mod_press_label = int(label)
        self._mod_drag_started = False

    def on_drag_moved(self, pos: Any) -> None:
        if not self._is_mod_mode():
            return
        point = self._parse_pos(pos)
        if point is None:
            return

        if self._mod_press_pos is None or self._mod_press_slice_idx is None or self._mod_press_label is None:
            return

        x, y = point
        slice_idx = int(self._mod_press_slice_idx)
        if int(self.view_state_model.current_slice) != slice_idx:
            self._reset_mod_pointer_state()
            return

        if not self._mod_drag_started:
            press_x, press_y = self._mod_press_pos
            dx = int(x) - int(press_x)
            dy = int(y) - int(press_y)
            threshold_sq = int(self.MOD_DRAG_START_DISTANCE_PX) * int(self.MOD_DRAG_START_DISTANCE_PX)
            if (dx * dx) + (dy * dy) < threshold_sq:
                return
            effective_slice = self._effective_slice_mask(slice_idx)
            if effective_slice is None:
                self._reset_mod_pointer_state()
                return
            started = self.mask_modification_service.start_drag(
                slice_idx=slice_idx,
                label=int(self._mod_press_label),
                x_pos=int(press_x),
                y_pos=int(press_y),
                slice_mask=effective_slice,
            )
            if not started:
                return
            self._mod_drag_started = True

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

    def on_drag_finished(self, pos: Any) -> None:
        point = self._parse_pos(pos)
        if (
            self._mod_press_pos is not None
            and self._mod_press_slice_idx is not None
            and self._mod_press_label is not None
            and not self._mod_drag_started
        ):
            release_point = self._mod_press_pos
            effective_slice = self._effective_slice_mask(int(self._mod_press_slice_idx))
            if effective_slice is not None:
                selected_idx = self.mask_modification_service.select_component(
                    slice_idx=int(self._mod_press_slice_idx),
                    label=int(self._mod_press_label),
                    x_pos=int(release_point[0]),
                    y_pos=int(release_point[1]),
                    slice_mask=effective_slice,
                )
                if selected_idx is not None:
                    self._set_position_label(int(release_point[0]), int(release_point[1]))
        self.mask_modification_service.end_drag()
        self._reset_mod_pointer_state()
        self.sync_anchors()

    def on_double_clicked(self, pos: Any) -> None:
        if not self._is_mod_mode():
            return
        point = self._parse_pos(pos)
        if point is None:
            return
        if not self._ensure_context():
            return

        x, y = point
        slice_idx = int(self.view_state_model.current_slice)
        effective_slice = self._effective_slice_mask(slice_idx)
        if effective_slice is None:
            return
        label = self._resolve_mod_label_at_point(point=point, slice_mask=effective_slice)
        if label is None:
            self._status_message("Double-click near a mask contour to add an anchor.", timeout_ms=1800)
            return
        self.mask_modification_service.select_component(
            slice_idx=slice_idx,
            label=label,
            x_pos=x,
            y_pos=y,
            slice_mask=effective_slice,
        )
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
                    "All contour pixels are already anchored.",
                    timeout_ms=1800,
                )
            else:
                self._status_message("Double-click near the contour to add an anchor.", timeout_ms=1800)
            return
        self.sync_anchors()
        self._status_message("Anchor added.", timeout_ms=1000)

    def on_context_menu_requested(self, payload: Any) -> bool:
        if not self._is_mod_mode():
            return False
        request = self._parse_context_request(payload)
        if request is None:
            return False
        point, global_pos = request
        if not self._ensure_context():
            return False

        slice_idx = int(self.view_state_model.current_slice)
        effective_slice = self._effective_slice_mask(slice_idx)
        if effective_slice is None:
            return False
        label = self._resolve_mod_label_at_point(point=point, slice_mask=effective_slice)
        if label is None:
            self._status_message(
                "Ctrl+right-click directly on a mask to open the mod menu.",
                timeout_ms=1800,
            )
            return False

        selected_idx = self.mask_modification_service.select_component(
            slice_idx=slice_idx,
            label=label,
            x_pos=int(point[0]),
            y_pos=int(point[1]),
            slice_mask=effective_slice,
        )
        if selected_idx is None:
            self._status_message(
                "Ctrl+right-click directly on a mask to open the mod menu.",
                timeout_ms=1800,
            )
            return False

        self._set_position_label(int(point[0]), int(point[1]))
        self.sync_anchors()
        palette = self.annotation_model.get_label_palette()
        action = self.annotation_view.show_mod_context_menu(
            global_pos=global_pos,
            current_label=label,
            label_choices=sorted(int(lbl) for lbl in palette.keys() if int(lbl) > 0),
        )
        if action is None:
            return False

        action_kind, target_label = action
        if action_kind == "delete":
            return self.on_delete_selected_component_requested()
        if action_kind == "relabel" and target_label is not None:
            return self.on_relabel_selected_component_requested(target_label)
        return False

    def on_selection_cancel_requested(self) -> bool:
        if not self._is_mod_mode() and not self.mask_modification_service.has_pending_edits():
            return False
        had_dirty = self.mask_modification_service.cancel_pending()
        self._reset_mod_pointer_state()
        self.annotation_view.clear_mod_anchor_points()
        cleared_preview = self._clear_mod_preview_from_temp()
        if had_dirty:
            self._status_message("Mask edits canceled.", timeout_ms=1800)
        if had_dirty or cleared_preview:
            self._refresh_current_roi_overlay()
        return bool(self._is_mod_mode() or had_dirty or cleared_preview)

    def on_roi_delete_requested(self) -> None:
        """Keep mod pending state consistent when ROI temp deletion is requested."""
        had_dirty = self.mask_modification_service.cancel_pending()
        self.mask_modification_service.end_drag()
        self._reset_mod_pointer_state()
        self.annotation_view.clear_mod_anchor_points()
        self._mod_preview_slices.clear()
        self._mod_preview_base.clear()
        self._mod_preview_current.clear()
        if had_dirty:
            self._status_message("Mask edits deleted.", timeout_ms=1800)

    def on_delete_selected_component_requested(self) -> bool:
        """Delete the currently selected mod components and stage them for standard apply."""
        if not self._is_mod_mode():
            return False
        if not self._ensure_context():
            return False

        deleted = self.mask_modification_service.delete_selected_components()
        if deleted is None:
            self._status_message(
                "Selectionne un ou plusieurs masks en mode mod avant d'appuyer sur Delete.",
                timeout_ms=1800,
            )
            return False

        slice_idx, updated_slice = deleted
        self.annotation_view.clear_mod_anchor_points()
        self._store_mod_preview_slice(slice_idx=slice_idx, updated_slice=updated_slice)
        self.refresh_preview(changed_slice=slice_idx)
        if not self._is_apply_auto_enabled():
            self._status_message("Suppression previsualisee. Applique pour valider.", timeout_ms=1800)
        return True

    def on_relabel_selected_component_requested(self, target_label: int) -> bool:
        """Reassign the selected mod components to another label and stage the preview."""
        if not self._is_mod_mode():
            return False
        if not self._ensure_context():
            return False

        changed = self.mask_modification_service.relabel_selected_components(target_label=int(target_label))
        if changed is None:
            self._status_message(
                "Selectionne un ou plusieurs masks puis choisis un label cible different.",
                timeout_ms=1800,
            )
            return False

        slice_idx, updated_slice = changed
        self.annotation_view.clear_mod_anchor_points()
        self._store_mod_preview_slice(slice_idx=slice_idx, updated_slice=updated_slice)
        self.refresh_preview(changed_slice=slice_idx)
        if not self._is_apply_auto_enabled():
            self._status_message("Changement de label previsualise. Applique pour valider.", timeout_ms=1800)
        return True

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
        self._status_message("Mask edits applied.", timeout_ms=1800)
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
        active_slice_idx, active_label = self.mask_modification_service.active_context()
        if active_slice_idx is None or active_label is None:
            self.annotation_view.clear_mod_anchor_points()
            return
        if int(active_slice_idx) != int(self.view_state_model.current_slice):
            self.annotation_view.clear_mod_anchor_points()
            return
        groups = self.mask_modification_service.selected_anchor_groups(
            slice_idx=int(self.view_state_model.current_slice),
            label=int(active_label),
        )
        if not groups:
            self.annotation_view.clear_mod_anchor_points()
            return
        drag_component_idx, drag_anchor_idx = self.mask_modification_service.active_drag_state()
        self.annotation_view.set_mod_anchor_groups(
            groups,
            drag_component_index=drag_component_idx,
            drag_anchor_index=drag_anchor_idx,
        )

    def _reset_mod_pointer_state(self) -> None:
        self._mod_press_pos = None
        self._mod_press_slice_idx = None
        self._mod_press_label = None
        self._mod_drag_started = False

    def _parse_context_request(
        self,
        payload: Any,
    ) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
        if not isinstance(payload, tuple) or len(payload) != 2:
            return None
        point = self._parse_pos(payload[0])
        global_pos = payload[1]
        if point is None or not isinstance(global_pos, tuple) or len(global_pos) != 2:
            return None
        try:
            gx = int(global_pos[0])
            gy = int(global_pos[1])
        except Exception:
            return None
        return point, (gx, gy)

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

    def _resolve_mod_label_at_point(
        self,
        *,
        point: tuple[int, int],
        slice_mask: np.ndarray,
    ) -> Optional[int]:
        return self.mask_modification_service.detect_label_at_point(
            slice_mask=slice_mask,
            x_pos=int(point[0]),
            y_pos=int(point[1]),
        )

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
        return (
            str(getattr(self.view_state_model, "tool_mode", "") or "") == "mod"
            and not self._is_corrosion_layer_mode()
        )

    def _is_apply_auto_enabled(self) -> bool:
        return bool(getattr(self.view_state_model, "apply_auto", False))

    def _is_corrosion_layer_mode(self) -> bool:
        return bool(getattr(self.view_state_model, "corrosion_active", False))

    @staticmethod
    def _parse_pos(pos: Any) -> Optional[tuple[int, int]]:
        if not isinstance(pos, (tuple, list)) or len(pos) != 2:
            return None
        try:
            return int(pos[0]), int(pos[1])
        except Exception:
            return None
