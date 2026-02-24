"""Service dedicated to corrosion profile anchor editing (temporary + commit)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from services.cscan_corrosion_service import CScanCorrosionService


@dataclass
class CorrosionCommitPayload:
    """Committed corrosion editing payload."""

    peak_map_a: np.ndarray
    peak_map_b: np.ndarray
    overlay_volume: np.ndarray
    projection: Optional[np.ndarray]
    value_range: Optional[Tuple[float, float]]


class CorrosionProfileEditService:
    """Handles anchor-based editing of corrosion BW/FW profiles."""

    ANCHOR_SPACING_PX = 1
    ANCHOR_HIT_RADIUS_PX = 3
    LINE_HIT_TOLERANCE_PX = 6
    LINE_THICKNESS = 1

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._pending_peak_map_a: Optional[np.ndarray] = None
        self._pending_peak_map_b: Optional[np.ndarray] = None
        self._label_ids: Optional[tuple[int, int]] = None
        self._image_shape: Optional[tuple[int, int]] = None  # (H, W)
        self._controls_a: Dict[int, List[Tuple[int, int]]] = {}
        self._controls_b: Dict[int, List[Tuple[int, int]]] = {}
        self._overlay_cache: Optional[np.ndarray] = None
        self._drag_active: bool = False
        self._drag_target_is_a: Optional[bool] = None
        self._drag_slice_idx: Optional[int] = None
        self._drag_anchor_idx: Optional[int] = None
        self._dirty: bool = False

    def has_pending_edits(self) -> bool:
        return bool(self._dirty)

    def ensure_context(
        self,
        *,
        peak_map_a: np.ndarray,
        peak_map_b: np.ndarray,
        label_ids: tuple[int, int],
        image_shape: tuple[int, int],
        cscan_corrosion_service: CScanCorrosionService,
    ) -> bool:
        data_a = np.asarray(peak_map_a, dtype=np.int32)
        data_b = np.asarray(peak_map_b, dtype=np.int32)
        if data_a.ndim != 2 or data_b.ndim != 2:
            return False

        shape_key = (
            data_a.shape,
            data_b.shape,
            tuple(int(x) for x in label_ids),
            (int(image_shape[0]), int(image_shape[1])),
        )
        current_key = None
        if (
            self._pending_peak_map_a is not None
            and self._pending_peak_map_b is not None
            and self._label_ids is not None
            and self._image_shape is not None
        ):
            current_key = (
                self._pending_peak_map_a.shape,
                self._pending_peak_map_b.shape,
                self._label_ids,
                self._image_shape,
            )
        if current_key == shape_key:
            return True

        self._pending_peak_map_a = np.array(data_a, dtype=np.int32, copy=True)
        self._pending_peak_map_b = np.array(data_b, dtype=np.int32, copy=True)
        self._label_ids = tuple(int(x) for x in label_ids)
        self._image_shape = (int(image_shape[0]), int(image_shape[1]))
        self._controls_a.clear()
        self._controls_b.clear()
        self._drag_active = False
        self._drag_target_is_a = None
        self._drag_slice_idx = None
        self._drag_anchor_idx = None
        self._dirty = False

        self._overlay_cache = cscan_corrosion_service.build_overlay_from_peak_maps(
            peak_map_a=self._pending_peak_map_a,
            peak_map_b=self._pending_peak_map_b,
            image_shape=self._image_shape,
            class_A_id=self._label_ids[0],
            class_B_id=self._label_ids[1],
            line_thickness=self.LINE_THICKNESS,
        )
        return True

    def resolve_target_from_active_label(self, active_label: Optional[int]) -> Optional[bool]:
        if active_label is None or self._label_ids is None:
            return None
        try:
            label = int(active_label)
        except Exception:
            return None
        if label == int(self._label_ids[0]):
            return True
        if label == int(self._label_ids[1]):
            return False
        return None

    def anchor_points(self, *, slice_idx: int, target_is_a: bool) -> list[tuple[int, int]]:
        controls = self._controls_for(slice_idx=slice_idx, target_is_a=target_is_a, create=True)
        return [(int(x), int(y)) for x, y in controls]

    def start_drag(
        self,
        *,
        slice_idx: int,
        target_is_a: bool,
        x_pos: int,
        y_pos: int,
    ) -> bool:
        controls = self._controls_for(slice_idx=slice_idx, target_is_a=target_is_a, create=True)
        if not controls:
            return False
        anchor_idx = self._find_anchor_index(
            controls=controls,
            x_pos=int(x_pos),
            y_pos=int(y_pos),
            radius_px=self.ANCHOR_HIT_RADIUS_PX,
        )
        if anchor_idx is None:
            return False
        self._drag_active = True
        self._drag_target_is_a = bool(target_is_a)
        self._drag_slice_idx = int(slice_idx)
        self._drag_anchor_idx = int(anchor_idx)
        return True

    def drag_to(
        self,
        *,
        slice_idx: int,
        x_pos: int,
        y_pos: int,
    ) -> bool:
        if not self._drag_active:
            return False
        if self._drag_target_is_a is None or self._drag_slice_idx is None or self._drag_anchor_idx is None:
            return False
        if int(slice_idx) != int(self._drag_slice_idx):
            return False
        if self._image_shape is None:
            return False
        height, width = self._image_shape

        controls = self._controls_for(
            slice_idx=self._drag_slice_idx,
            target_is_a=self._drag_target_is_a,
            create=True,
        )
        if not controls:
            return False
        idx = int(max(0, min(len(controls) - 1, self._drag_anchor_idx)))
        old_x, old_y = controls[idx]

        x_new = max(0, min(width - 1, int(x_pos)))
        y_new = max(0, min(height - 1, int(y_pos)))
        if idx > 0:
            x_new = max(x_new, controls[idx - 1][0] + 1)
        if idx < len(controls) - 1:
            x_new = min(x_new, controls[idx + 1][0] - 1)

        if x_new == old_x and y_new == old_y:
            return False
        controls[idx] = (int(x_new), int(y_new))
        self._write_slice_from_controls(
            slice_idx=self._drag_slice_idx,
            target_is_a=self._drag_target_is_a,
            controls=controls,
        )
        self._dirty = True
        return True

    def end_drag(self) -> None:
        self._drag_active = False
        self._drag_target_is_a = None
        self._drag_slice_idx = None
        self._drag_anchor_idx = None

    def add_anchor_on_line(
        self,
        *,
        slice_idx: int,
        target_is_a: bool,
        x_pos: int,
        y_pos: int,
    ) -> bool:
        controls = self._controls_for(slice_idx=slice_idx, target_is_a=target_is_a, create=True)
        if not controls:
            return False
        if self._image_shape is None:
            return False
        height, width = self._image_shape
        x = max(0, min(width - 1, int(x_pos)))
        y = max(0, min(height - 1, int(y_pos)))

        existing = self._find_anchor_index(
            controls=controls,
            x_pos=x,
            y_pos=y,
            radius_px=self.ANCHOR_HIT_RADIUS_PX,
        )
        if existing is not None:
            return False

        row = self._row_for(slice_idx=slice_idx, target_is_a=target_is_a)
        if row is None or x >= row.shape[0]:
            return False
        y_line = int(row[x])
        if y_line < 0:
            return False
        if abs(y_line - y) > self.LINE_HIT_TOLERANCE_PX:
            return False

        insert_at = 0
        while insert_at < len(controls) and controls[insert_at][0] < x:
            insert_at += 1
        controls.insert(insert_at, (int(x), int(y)))
        self._write_slice_from_controls(
            slice_idx=slice_idx,
            target_is_a=target_is_a,
            controls=controls,
        )
        self._dirty = True
        return True

    def preview_overlay(self) -> Optional[np.ndarray]:
        if self._overlay_cache is None:
            return None
        return self._overlay_cache

    def commit(
        self,
        *,
        cscan_corrosion_service: CScanCorrosionService,
        rebuild_projection: bool = True,
    ) -> Optional[CorrosionCommitPayload]:
        if (
            self._pending_peak_map_a is None
            or self._pending_peak_map_b is None
            or self._image_shape is None
            or self._label_ids is None
        ):
            return None

        height, _width = self._image_shape
        committed_a = cscan_corrosion_service.interpolate_peak_map_1d_dual_axis(
            self._pending_peak_map_a,
            height=height,
        )
        committed_b = cscan_corrosion_service.interpolate_peak_map_1d_dual_axis(
            self._pending_peak_map_b,
            height=height,
        )
        overlay = cscan_corrosion_service.build_overlay_from_peak_maps(
            peak_map_a=committed_a,
            peak_map_b=committed_b,
            image_shape=self._image_shape,
            class_A_id=int(self._label_ids[0]),
            class_B_id=int(self._label_ids[1]),
            line_thickness=self.LINE_THICKNESS,
        )

        projection: Optional[np.ndarray] = None
        value_range: Optional[Tuple[float, float]] = None
        if rebuild_projection:
            distance_map = cscan_corrosion_service.build_interpolated_distance_map(
                overlay=overlay,
                class_A_value=int(self._label_ids[0]),
                class_B_value=int(self._label_ids[1]),
                use_mm=False,
                resolution_ultrasound_mm=1.0,
            )
            if distance_map.size > 0:
                projection, value_range = cscan_corrosion_service.compute_corrosion_projection(distance_map)

        self._pending_peak_map_a = np.array(committed_a, dtype=np.int32, copy=True)
        self._pending_peak_map_b = np.array(committed_b, dtype=np.int32, copy=True)
        self._overlay_cache = np.array(overlay, copy=True)
        self._controls_a.clear()
        self._controls_b.clear()
        self._dirty = False
        self.end_drag()

        return CorrosionCommitPayload(
            peak_map_a=committed_a,
            peak_map_b=committed_b,
            overlay_volume=overlay,
            projection=projection,
            value_range=value_range,
        )

    def _controls_for(
        self,
        *,
        slice_idx: int,
        target_is_a: bool,
        create: bool,
    ) -> list[tuple[int, int]]:
        target = self._controls_a if bool(target_is_a) else self._controls_b
        key = int(slice_idx)
        if key in target:
            return target[key]
        if not create:
            return []

        row = self._row_for(slice_idx=key, target_is_a=target_is_a)
        if row is None:
            target[key] = []
            return target[key]
        controls = self._build_default_controls(row=row)
        target[key] = controls
        return controls

    def _row_for(self, *, slice_idx: int, target_is_a: bool) -> Optional[np.ndarray]:
        data = self._pending_peak_map_a if bool(target_is_a) else self._pending_peak_map_b
        if data is None:
            return None
        z = int(slice_idx)
        if z < 0 or z >= data.shape[0]:
            return None
        return data[z]

    def _build_default_controls(self, *, row: np.ndarray) -> list[tuple[int, int]]:
        if self._image_shape is None:
            return []
        height, _width = self._image_shape
        valid_x = np.where((row >= 0) & (row < height))[0]
        if valid_x.size == 0:
            return []

        controls: list[tuple[int, int]] = []
        last_x = -10_000
        for x in valid_x.tolist():
            if not controls or (int(x) - int(last_x)) >= self.ANCHOR_SPACING_PX:
                controls.append((int(x), int(row[x])))
                last_x = int(x)

        tail_x = int(valid_x[-1])
        if controls[-1][0] != tail_x:
            controls.append((tail_x, int(row[tail_x])))

        if len(controls) == 1 and valid_x.size >= 2:
            mid_x = int(valid_x[valid_x.size // 2])
            controls.insert(1, (mid_x, int(row[mid_x])))
        return controls

    @staticmethod
    def _find_anchor_index(
        *,
        controls: list[tuple[int, int]],
        x_pos: int,
        y_pos: int,
        radius_px: int,
    ) -> Optional[int]:
        if not controls:
            return None
        r_sq = int(radius_px) * int(radius_px)
        best_idx: Optional[int] = None
        best_dist = r_sq + 1
        for idx, (ax, ay) in enumerate(controls):
            dx = int(ax) - int(x_pos)
            dy = int(ay) - int(y_pos)
            dist = (dx * dx) + (dy * dy)
            if dist <= r_sq and dist < best_dist:
                best_idx = int(idx)
                best_dist = int(dist)
        return best_idx

    def _write_slice_from_controls(
        self,
        *,
        slice_idx: int,
        target_is_a: bool,
        controls: list[tuple[int, int]],
    ) -> None:
        if self._image_shape is None:
            return
        height, width = self._image_shape
        row = np.full((width,), -1, dtype=np.int32)
        if not controls:
            self._set_row(slice_idx=slice_idx, target_is_a=target_is_a, row=row)
            self._update_overlay_slice(slice_idx=slice_idx)
            return

        controls_sorted = sorted((int(x), int(y)) for x, y in controls)
        if len(controls_sorted) == 1:
            x0, y0 = controls_sorted[0]
            if 0 <= x0 < width:
                row[x0] = int(max(0, min(height - 1, y0)))
        else:
            for idx in range(len(controls_sorted) - 1):
                x0, y0 = controls_sorted[idx]
                x1, y1 = controls_sorted[idx + 1]
                if x0 == x1:
                    if 0 <= x0 < width:
                        row[x0] = int(max(0, min(height - 1, y0)))
                    continue
                xs = np.arange(min(x0, x1), max(x0, x1) + 1, dtype=np.int32)
                ys = np.rint(np.interp(xs, [x0, x1], [y0, y1])).astype(np.int32)
                ys = np.clip(ys, 0, height - 1)
                row[xs] = ys

        self._set_row(slice_idx=slice_idx, target_is_a=target_is_a, row=row)
        self._update_overlay_slice(slice_idx=slice_idx)

    def _set_row(self, *, slice_idx: int, target_is_a: bool, row: np.ndarray) -> None:
        if target_is_a:
            if self._pending_peak_map_a is None:
                return
            if 0 <= int(slice_idx) < self._pending_peak_map_a.shape[0]:
                self._pending_peak_map_a[int(slice_idx)] = row
        else:
            if self._pending_peak_map_b is None:
                return
            if 0 <= int(slice_idx) < self._pending_peak_map_b.shape[0]:
                self._pending_peak_map_b[int(slice_idx)] = row

    def _update_overlay_slice(self, *, slice_idx: int) -> None:
        if (
            self._overlay_cache is None
            or self._pending_peak_map_a is None
            or self._pending_peak_map_b is None
            or self._label_ids is None
            or self._image_shape is None
        ):
            return
        z = int(slice_idx)
        if z < 0 or z >= self._overlay_cache.shape[0]:
            return

        h, w = self._image_shape
        color_a = int(self._label_ids[0])
        color_b = int(self._label_ids[1])
        row_a = self._pending_peak_map_a[z]
        row_b = self._pending_peak_map_b[z]
        rendered = np.zeros((h, w), dtype=np.uint8)

        self._draw_row_polyline(
            canvas=rendered,
            row=row_a,
            color=color_a,
            height=h,
            line_thickness=self.LINE_THICKNESS,
        )
        self._draw_row_polyline(
            canvas=rendered,
            row=row_b,
            color=color_b,
            height=h,
            line_thickness=self.LINE_THICKNESS,
        )
        self._overlay_cache[z] = rendered

    @staticmethod
    def _draw_row_polyline(
        *,
        canvas: np.ndarray,
        row: np.ndarray,
        color: int,
        height: int,
        line_thickness: int,
    ) -> None:
        valid = np.where((row >= 0) & (row < int(height)))[0]
        if valid.size == 0:
            return
        pts = [(int(x), int(row[x])) for x in valid.tolist()]
        if len(pts) == 1:
            x0, y0 = pts[0]
            canvas[y0, x0] = int(color)
            return
        for idx in range(len(pts) - 1):
            cv2.line(canvas, pts[idx], pts[idx + 1], color=int(color), thickness=int(line_thickness))
