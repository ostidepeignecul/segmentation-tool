"""Service dedicated to anchored mask editing in pending mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class _SelectedComponent:
    kind: str
    component_mask: np.ndarray
    contour_points: np.ndarray
    anchor_indices: list[int]


class MaskModificationService:
    """Handle multi-component selection, anchor dragging, and deferred mask commit."""

    ANCHOR_HIT_RADIUS_PX = 7
    CONTOUR_HIT_TOLERANCE_PX = 8

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._image_shape: Optional[tuple[int, int]] = None
        self._dirty_slices: set[int] = set()
        self._drag_active = False
        self._drag_component_list_idx: Optional[int] = None
        self._drag_anchor_list_idx: Optional[int] = None
        self._active_slice_idx: Optional[int] = None
        self._active_label: Optional[int] = None
        self._active_slice_mask: Optional[np.ndarray] = None
        self._selected_components: list[_SelectedComponent] = []

    def has_pending_edits(self) -> bool:
        return bool(self._dirty_slices)

    def dirty_slices(self) -> set[int]:
        """Return the set of slices currently marked as modified in pending mode."""
        return {int(idx) for idx in self._dirty_slices}

    def ensure_loaded(self, mask_volume: np.ndarray) -> bool:
        data = np.asarray(mask_volume, dtype=np.uint8)
        if data.ndim != 3:
            return False
        shape = (int(data.shape[1]), int(data.shape[2]))
        if self._image_shape != shape:
            self._image_shape = shape
            self.clear_active_component()
        return True

    def load_from_mask(self, mask_volume: np.ndarray) -> bool:
        return self.ensure_loaded(mask_volume)

    def clear_active_component(self) -> None:
        self._clear_active_component()
        self._drag_active = False
        self._drag_component_list_idx = None
        self._drag_anchor_list_idx = None

    def active_context(self) -> tuple[Optional[int], Optional[int]]:
        """Return the currently loaded `(slice_idx, label)` context for mod editing."""
        return self._active_slice_idx, self._active_label

    def detect_label_at_point(self, *, slice_mask: np.ndarray, x_pos: int, y_pos: int) -> Optional[int]:
        """Resolve the label under the cursor, or the nearest contour label within tolerance."""
        slice_arr = np.asarray(slice_mask, dtype=np.uint8)
        if slice_arr.ndim != 2:
            return None
        height, width = slice_arr.shape
        if height <= 0 or width <= 0:
            return None

        x = int(max(0, min(width - 1, int(x_pos))))
        y = int(max(0, min(height - 1, int(y_pos))))
        direct_label = int(slice_arr[y, x])
        if direct_label > 0:
            return direct_label

        labels = [int(lbl) for lbl in np.unique(slice_arr) if int(lbl) > 0]
        if not labels:
            return None

        best_label: Optional[int] = None
        best_d2: Optional[float] = None
        max_d2 = float(self.CONTOUR_HIT_TOLERANCE_PX * self.CONTOUR_HIT_TOLERANCE_PX)
        for label in labels:
            label_binary = (slice_arr == int(label)).astype(np.uint8)
            if not np.any(label_binary):
                continue
            contours, _ = cv2.findContours(label_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if contour is None or contour.size == 0:
                    continue
                pts = contour.reshape((-1, 2))
                d2 = float(np.min((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2))
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_label = int(label)
        if best_d2 is None or best_d2 > max_d2:
            return None
        return best_label

    def selected_anchor_groups(self, *, slice_idx: int, label: int) -> list[list[tuple[int, int]]]:
        if not self._context_matches(slice_idx=slice_idx, label=label):
            return []
        groups: list[list[tuple[int, int]]] = []
        for component in self._selected_components:
            points: list[tuple[int, int]] = []
            for contour_idx in component.anchor_indices:
                x, y = component.contour_points[int(contour_idx)]
                points.append((int(round(float(x))), int(round(float(y)))))
            if points:
                groups.append(points)
        return groups

    def active_drag_state(self) -> tuple[Optional[int], Optional[int]]:
        if not self._drag_active:
            return None, None
        if self._drag_component_list_idx is None or self._drag_anchor_list_idx is None:
            return None, None
        if self._drag_component_list_idx < 0 or self._drag_component_list_idx >= len(self._selected_components):
            return None, None
        component = self._selected_components[int(self._drag_component_list_idx)]
        if not component.anchor_indices:
            return None, None
        anchor_idx = int(
            max(0, min(len(component.anchor_indices) - 1, int(self._drag_anchor_list_idx)))
        )
        return int(self._drag_component_list_idx), anchor_idx

    def has_full_density_anchors(self, *, slice_idx: int, label: int) -> bool:
        if not self._context_matches(slice_idx=slice_idx, label=label):
            return False
        if not self._selected_components:
            return False
        for component in self._selected_components:
            if component.kind in {"point", "segment"}:
                continue
            contour = component.contour_points
            if contour.size == 0:
                return False
            if len(component.anchor_indices) < int(contour.shape[0]):
                return False
            pts = np.rint(contour).astype(np.int32)
            n_pts = int(pts.shape[0])
            if n_pts < 2:
                continue
            for idx in range(n_pts):
                nxt = (idx + 1) % n_pts
                dx = abs(int(pts[nxt, 0]) - int(pts[idx, 0]))
                dy = abs(int(pts[nxt, 1]) - int(pts[idx, 1]))
                if max(dx, dy) > 1:
                    return False
        return True

    def select_component(
        self,
        *,
        slice_idx: int,
        label: int,
        x_pos: int,
        y_pos: int,
        slice_mask: np.ndarray,
    ) -> Optional[int]:
        source_slice = self._ensure_context(
            slice_idx=slice_idx,
            label=label,
            slice_mask=slice_mask,
        )
        if source_slice is None:
            return None
        component = self._extract_component_at_point(
            slice_idx=int(slice_idx),
            label=int(label),
            x_pos=int(x_pos),
            y_pos=int(y_pos),
            slice_mask=source_slice,
        )
        if component is None:
            return None
        existing_idx = self._find_selected_component_index(component.component_mask)
        if existing_idx is not None:
            return existing_idx
        self._selected_components.append(component)
        return len(self._selected_components) - 1

    def start_drag(
        self,
        *,
        slice_idx: int,
        label: int,
        x_pos: int,
        y_pos: int,
        slice_mask: np.ndarray,
    ) -> bool:
        source_slice = self._ensure_context(
            slice_idx=slice_idx,
            label=label,
            slice_mask=slice_mask,
        )
        if source_slice is None:
            return False
        hit = self._find_anchor_hit(
            x_pos=int(x_pos),
            y_pos=int(y_pos),
            radius_px=self.ANCHOR_HIT_RADIUS_PX,
        )
        if hit is None:
            return False
        component_idx, anchor_idx = hit
        self._drag_active = True
        self._drag_component_list_idx = int(component_idx)
        self._drag_anchor_list_idx = int(anchor_idx)
        return True

    def drag_to(self, *, slice_idx: int, x_pos: int, y_pos: int) -> Optional[np.ndarray]:
        if not self._drag_active:
            return None
        if self._drag_component_list_idx is None or self._drag_anchor_list_idx is None:
            return None
        if self._image_shape is None:
            return None
        if self._active_slice_idx is None or self._active_label is None:
            return None
        if self._active_slice_mask is None:
            return None
        if int(slice_idx) != int(self._active_slice_idx):
            return None
        if self._drag_component_list_idx < 0 or self._drag_component_list_idx >= len(self._selected_components):
            return None

        component_idx = int(self._drag_component_list_idx)
        component = self._selected_components[component_idx]
        if not component.anchor_indices:
            return None

        anchor_list_idx = int(
            max(0, min(len(component.anchor_indices) - 1, int(self._drag_anchor_list_idx)))
        )
        anchor_point_idx = int(component.anchor_indices[anchor_list_idx])
        contour = np.array(component.contour_points, dtype=np.float32, copy=True)
        if contour.ndim != 2 or contour.shape[0] <= 0:
            return None
        if anchor_point_idx < 0 or anchor_point_idx >= contour.shape[0]:
            return None

        height, width = self._image_shape
        x_new = float(max(0, min(width - 1, int(x_pos))))
        y_new = float(max(0, min(height - 1, int(y_pos))))
        x_old = float(contour[anchor_point_idx, 0])
        y_old = float(contour[anchor_point_idx, 1])
        if abs(x_new - x_old) < 0.01 and abs(y_new - y_old) < 0.01:
            return None

        contour[anchor_point_idx, 0] = x_new
        contour[anchor_point_idx, 1] = y_new
        contour[:, 0] = np.clip(contour[:, 0], 0.0, float(width - 1))
        contour[:, 1] = np.clip(contour[:, 1], 0.0, float(height - 1))
        if component.kind == "polygon":
            contour = self._normalize_contour_points(np.rint(contour).astype(np.float32))
            if contour.shape[0] < 3:
                return None
            contour_int = np.rint(contour).astype(np.int32)
            if np.unique(contour_int, axis=0).shape[0] < 3:
                return None

        component_bool = self._rasterize_component_mask(
            kind=component.kind,
            contour_points=contour,
            height=height,
            width=width,
        )
        if component_bool is None:
            return None
        if not np.any(component_bool):
            return None

        updated_slice = np.array(self._active_slice_mask, dtype=np.uint8, copy=True)
        updated_slice[component.component_mask] = 0
        blocked = updated_slice != 0
        if np.any(blocked):
            component_bool = np.logical_and(component_bool, np.logical_not(blocked))
            if not np.any(component_bool):
                return None
            refreshed = self._build_component_from_mask(
                component_mask=component_bool,
                ref_x=int(round(float(x_new))),
                ref_y=int(round(float(y_new))),
            )
            if refreshed is None:
                return None
            component_bool = refreshed.component_mask
        else:
            refreshed = self._build_component_from_mask(
                component_mask=component_bool,
                ref_x=int(round(float(x_new))),
                ref_y=int(round(float(y_new))),
            )
            if refreshed is None:
                return None

        label = int(self._active_label)
        updated_slice[refreshed.component_mask] = label
        self._active_slice_mask = np.array(updated_slice, dtype=np.uint8, copy=True)
        self._selected_components[component_idx] = refreshed
        nearest_active = self._nearest_anchor_list_index(
            component=refreshed,
            x_pos=int(round(float(x_new))),
            y_pos=int(round(float(y_new))),
        )
        if nearest_active is not None:
            self._drag_anchor_list_idx = int(nearest_active)
        self._dirty_slices.add(int(self._active_slice_idx))
        return np.array(updated_slice, dtype=np.uint8, copy=True)

    def end_drag(self) -> None:
        self._drag_active = False
        self._drag_component_list_idx = None
        self._drag_anchor_list_idx = None

    def add_anchor_on_contour(
        self,
        *,
        slice_idx: int,
        label: int,
        x_pos: int,
        y_pos: int,
        slice_mask: np.ndarray,
    ) -> bool:
        component_idx = self.select_component(
            slice_idx=slice_idx,
            label=label,
            x_pos=x_pos,
            y_pos=y_pos,
            slice_mask=slice_mask,
        )
        if component_idx is None:
            return False
        if (
            component_idx < 0
            or component_idx >= len(self._selected_components)
            or self._selected_components[int(component_idx)].kind != "polygon"
        ):
            return False
        inserted = self._insert_anchor_near_point(
            component_idx=int(component_idx),
            x_pos=int(x_pos),
            y_pos=int(y_pos),
            max_dist_px=self.CONTOUR_HIT_TOLERANCE_PX,
        )
        return inserted is not None

    def delete_selected_components(self) -> Optional[tuple[int, np.ndarray]]:
        """Remove all currently selected components from the active slice."""
        if self._active_slice_idx is None or self._active_slice_mask is None:
            return None
        if not self._selected_components:
            return None

        slice_idx = int(self._active_slice_idx)
        updated_slice = np.array(self._active_slice_mask, dtype=np.uint8, copy=True)
        for component in self._selected_components:
            updated_slice[np.asarray(component.component_mask, dtype=bool)] = 0
        self._dirty_slices.add(slice_idx)
        self.clear_active_component()
        return slice_idx, updated_slice

    def relabel_selected_components(self, *, target_label: int) -> Optional[tuple[int, np.ndarray]]:
        """Reassign all selected components to another label on the active slice."""
        if self._active_slice_idx is None or self._active_slice_mask is None:
            return None
        if self._active_label is None or not self._selected_components:
            return None
        label = int(target_label)
        if label <= 0 or label == int(self._active_label):
            return None

        slice_idx = int(self._active_slice_idx)
        updated_slice = np.array(self._active_slice_mask, dtype=np.uint8, copy=True)
        for component in self._selected_components:
            component_mask = np.asarray(component.component_mask, dtype=bool)
            updated_slice[component_mask] = label
        self._dirty_slices.add(slice_idx)
        self.clear_active_component()
        return slice_idx, updated_slice

    def preview_mask_volume(self) -> Optional[np.ndarray]:
        return None

    def cancel_pending(self) -> bool:
        had_dirty = bool(self._dirty_slices)
        self._dirty_slices.clear()
        self.clear_active_component()
        return had_dirty

    def commit(self) -> Optional[set[int]]:
        if not self._dirty_slices:
            return None
        changed = set(int(v) for v in self._dirty_slices)
        self._dirty_slices.clear()
        self.clear_active_component()
        return changed

    def _context_matches(self, *, slice_idx: int, label: int) -> bool:
        return (
            self._active_slice_idx is not None
            and self._active_label is not None
            and int(self._active_slice_idx) == int(slice_idx)
            and int(self._active_label) == int(label)
        )

    def _ensure_context(
        self,
        *,
        slice_idx: int,
        label: int,
        slice_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        if label <= 0:
            return None
        slice_arr = np.asarray(slice_mask, dtype=np.uint8)
        if slice_arr.ndim != 2:
            return None
        shape = (int(slice_arr.shape[0]), int(slice_arr.shape[1]))
        if self._image_shape is None:
            self._image_shape = shape
        elif self._image_shape != shape:
            return None

        z = int(slice_idx)
        lbl = int(label)
        if not self._context_matches(slice_idx=z, label=lbl):
            self.clear_active_component()
            self._active_slice_idx = z
            self._active_label = lbl
            self._active_slice_mask = np.array(slice_arr, dtype=np.uint8, copy=True)
            return np.array(self._active_slice_mask, dtype=np.uint8, copy=False)

        if self._active_slice_mask is None or (not self._selected_components and not self._drag_active):
            self._active_slice_mask = np.array(slice_arr, dtype=np.uint8, copy=True)
        return np.array(self._active_slice_mask, dtype=np.uint8, copy=False)

    def _extract_component_at_point(
        self,
        *,
        slice_idx: int,
        label: int,
        x_pos: int,
        y_pos: int,
        slice_mask: np.ndarray,
    ) -> Optional[_SelectedComponent]:
        if self._image_shape is None:
            return None
        z = int(slice_idx)
        lbl = int(label)
        if not self._context_matches(slice_idx=z, label=lbl):
            return None
        slice_arr = np.asarray(slice_mask, dtype=np.uint8)
        if slice_arr.ndim != 2:
            return None
        height, width = self._image_shape
        x = int(max(0, min(width - 1, int(x_pos))))
        y = int(max(0, min(height - 1, int(y_pos))))

        label_binary = (slice_arr == lbl).astype(np.uint8)
        if not np.any(label_binary):
            return None

        n_components, component_map = cv2.connectedComponents(label_binary, connectivity=8)
        if n_components <= 1:
            return None

        component_id = int(component_map[y, x]) if 0 <= y < height and 0 <= x < width else 0
        if component_id <= 0:
            contours, _ = cv2.findContours(label_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            best_d2 = None
            best_component = 0
            for contour in contours:
                if contour is None or contour.size == 0:
                    continue
                pts = contour.reshape((-1, 2))
                d2 = np.min((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2)
                px, py = int(pts[0][0]), int(pts[0][1])
                cid = int(component_map[py, px]) if 0 <= py < height and 0 <= px < width else 0
                if cid <= 0:
                    continue
                if best_d2 is None or float(d2) < float(best_d2):
                    best_d2 = float(d2)
                    best_component = int(cid)
            if best_component <= 0:
                return None
            max_d2 = float(self.CONTOUR_HIT_TOLERANCE_PX * self.CONTOUR_HIT_TOLERANCE_PX)
            if best_d2 is not None and float(best_d2) > max_d2:
                return None
            component_id = int(best_component)

        component_bool = component_map == int(component_id)
        if not np.any(component_bool):
            return None
        return self._build_component_from_mask(
            component_mask=component_bool,
            ref_x=x,
            ref_y=y,
        )

    def _build_component_from_mask(
        self,
        *,
        component_mask: np.ndarray,
        ref_x: int,
        ref_y: int,
    ) -> Optional[_SelectedComponent]:
        component_bool = np.asarray(component_mask, dtype=bool)
        if component_bool.ndim != 2 or not np.any(component_bool):
            return None
        ys, xs = np.nonzero(component_bool)
        pixel_count = int(xs.size)
        if pixel_count == 1:
            contour_pts = np.asarray([[float(xs[0]), float(ys[0])]], dtype=np.float32)
            return _SelectedComponent(
                kind="point",
                component_mask=np.array(component_bool, dtype=bool, copy=True),
                contour_points=contour_pts,
                anchor_indices=[0],
            )
        if pixel_count == 2:
            coords = np.column_stack((xs, ys)).astype(np.float32)
            contour_pts = self._order_segment_points(coords, ref_x=int(ref_x), ref_y=int(ref_y))
            return _SelectedComponent(
                kind="segment",
                component_mask=np.array(component_bool, dtype=bool, copy=True),
                contour_points=np.array(contour_pts, dtype=np.float32, copy=True),
                anchor_indices=[0, 1],
            )
        contours, _ = cv2.findContours(
            component_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        contour_pts = self._extract_component_contour_points(
            contours=contours,
            ref_x=int(ref_x),
            ref_y=int(ref_y),
        )
        if contour_pts is None or contour_pts.shape[0] < 3:
            return None
        anchor_indices = self._build_anchor_indices(contour_pts)
        if not anchor_indices:
            return None
        return _SelectedComponent(
            kind="polygon",
            component_mask=np.array(component_bool, dtype=bool, copy=True),
            contour_points=np.array(contour_pts, dtype=np.float32, copy=True),
            anchor_indices=list(anchor_indices),
        )

    def _find_selected_component_index(self, component_mask: np.ndarray) -> Optional[int]:
        candidate = np.asarray(component_mask, dtype=bool)
        for idx, component in enumerate(self._selected_components):
            current = np.asarray(component.component_mask, dtype=bool)
            if current.shape == candidate.shape and np.array_equal(current, candidate):
                return int(idx)
        return None

    @staticmethod
    def _order_segment_points(coords: np.ndarray, *, ref_x: int, ref_y: int) -> np.ndarray:
        if coords.ndim != 2 or coords.shape[0] != 2 or coords.shape[1] != 2:
            return np.asarray(coords, dtype=np.float32)
        ref = np.asarray([float(ref_x), float(ref_y)], dtype=np.float32)
        distances = np.sum((coords - ref) ** 2, axis=1)
        order = np.lexsort((coords[:, 1], coords[:, 0], distances))
        return np.asarray(coords[order], dtype=np.float32)

    @staticmethod
    def _rasterize_component_mask(
        *,
        kind: str,
        contour_points: np.ndarray,
        height: int,
        width: int,
    ) -> Optional[np.ndarray]:
        contour = np.asarray(contour_points, dtype=np.float32)
        if contour.ndim != 2 or contour.shape[1] != 2:
            return None
        mask = np.zeros((height, width), dtype=bool)
        if kind == "point":
            if contour.shape[0] < 1:
                return None
            x = int(round(float(contour[0, 0])))
            y = int(round(float(contour[0, 1])))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            mask[y, x] = True
            return mask
        if kind == "segment":
            if contour.shape[0] < 2:
                return None
            p0 = np.rint(contour[0]).astype(np.int32)
            p1 = np.rint(contour[1]).astype(np.int32)
            line_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.line(
                line_mask,
                (int(p0[0]), int(p0[1])),
                (int(p1[0]), int(p1[1])),
                1,
                thickness=1,
            )
            return line_mask.astype(bool)
        if contour.shape[0] < 3:
            return None
        contour_int = np.rint(contour).astype(np.int32)
        if np.unique(contour_int, axis=0).shape[0] < 3:
            return None
        filled_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(filled_mask, [contour_int.reshape((-1, 1, 2))], 1)
        return filled_mask.astype(bool)

    def _find_anchor_hit(
        self,
        *,
        x_pos: int,
        y_pos: int,
        radius_px: int,
    ) -> Optional[tuple[int, int]]:
        if not self._selected_components:
            return None
        r_sq = float(int(radius_px) * int(radius_px))
        best_hit: Optional[tuple[int, int]] = None
        best_dist = r_sq + 1.0
        for component_idx, component in enumerate(self._selected_components):
            for anchor_list_idx, contour_idx in enumerate(component.anchor_indices):
                x, y = component.contour_points[int(contour_idx)]
                dx = float(x) - float(x_pos)
                dy = float(y) - float(y_pos)
                dist = (dx * dx) + (dy * dy)
                if dist <= r_sq and dist < best_dist:
                    best_dist = dist
                    best_hit = (int(component_idx), int(anchor_list_idx))
        return best_hit

    def _nearest_anchor_list_index(
        self,
        *,
        component: _SelectedComponent,
        x_pos: int,
        y_pos: int,
    ) -> Optional[int]:
        if not component.anchor_indices:
            return None
        best_idx: Optional[int] = None
        best_dist = None
        for list_idx, contour_idx in enumerate(component.anchor_indices):
            x, y = component.contour_points[int(contour_idx)]
            dx = float(x) - float(x_pos)
            dy = float(y) - float(y_pos)
            dist = (dx * dx) + (dy * dy)
            if best_dist is None or dist < best_dist:
                best_idx = int(list_idx)
                best_dist = float(dist)
        return best_idx

    def _insert_anchor_near_point(
        self,
        *,
        component_idx: int,
        x_pos: int,
        y_pos: int,
        max_dist_px: int,
    ) -> Optional[int]:
        if component_idx < 0 or component_idx >= len(self._selected_components):
            return None
        component = self._selected_components[int(component_idx)]
        contour = np.asarray(component.contour_points, dtype=np.float32)
        if contour.ndim != 2 or contour.shape[0] < 2:
            return None
        n_pts = int(contour.shape[0])

        px = float(x_pos)
        py = float(y_pos)
        max_dist_sq = float(max_dist_px) * float(max_dist_px)
        best_seg_idx: Optional[int] = None
        best_dist_sq = max_dist_sq + 1.0
        best_proj_x = 0.0
        best_proj_y = 0.0

        for idx in range(n_pts):
            nxt = (idx + 1) % n_pts
            ax = float(contour[idx, 0])
            ay = float(contour[idx, 1])
            bx = float(contour[nxt, 0])
            by = float(contour[nxt, 1])
            vx = bx - ax
            vy = by - ay
            seg_len_sq = (vx * vx) + (vy * vy)
            if seg_len_sq <= 1e-8:
                t = 0.0
            else:
                t = ((px - ax) * vx + (py - ay) * vy) / seg_len_sq
                t = max(0.0, min(1.0, t))
            proj_x = ax + (t * vx)
            proj_y = ay + (t * vy)
            dx = px - proj_x
            dy = py - proj_y
            dist_sq = (dx * dx) + (dy * dy)
            if dist_sq < best_dist_sq:
                best_dist_sq = float(dist_sq)
                best_seg_idx = int(idx)
                best_proj_x = float(proj_x)
                best_proj_y = float(proj_y)

        if best_seg_idx is None or best_dist_sq > max_dist_sq:
            return None
        ix = int(round(best_proj_x))
        iy = int(round(best_proj_y))
        if self._image_shape is not None:
            height, width = self._image_shape
            ix = max(0, min(width - 1, ix))
            iy = max(0, min(height - 1, iy))

        contour_int = np.rint(contour).astype(np.int32)
        if np.any(np.logical_and(contour_int[:, 0] == ix, contour_int[:, 1] == iy)):
            return None

        insert_at = int(best_seg_idx + 1)
        new_point = np.asarray([[float(ix), float(iy)]], dtype=np.float32)
        new_contour = np.concatenate((contour[:insert_at], new_point, contour[insert_at:]), axis=0)
        new_contour = self._normalize_contour_points(new_contour)
        if int(new_contour.shape[0]) <= n_pts:
            return None

        updated_component = _SelectedComponent(
            kind=component.kind,
            component_mask=np.array(component.component_mask, dtype=bool, copy=True),
            contour_points=np.array(new_contour, dtype=np.float32, copy=True),
            anchor_indices=self._build_anchor_indices(new_contour),
        )
        self._selected_components[int(component_idx)] = updated_component
        return self._nearest_anchor_list_index(
            component=updated_component,
            x_pos=ix,
            y_pos=iy,
        )

    def _build_anchor_indices(self, contour_pts: np.ndarray) -> list[int]:
        """Create one anchor per contour pixel (full-density anchors)."""
        n_pts = int(contour_pts.shape[0])
        if n_pts <= 0:
            return []
        return list(range(n_pts))

    @staticmethod
    def _extract_component_contour_points(
        *,
        contours: list[np.ndarray],
        ref_x: int,
        ref_y: int,
    ) -> Optional[np.ndarray]:
        """Extract the most relevant contour and normalize it as polygon points."""
        if not contours:
            return None

        best_pts: Optional[np.ndarray] = None
        best_score: Optional[tuple[float, float]] = None
        for contour in contours:
            if contour is None or contour.size == 0:
                continue
            pts = contour.reshape((-1, 2)).astype(np.float32)
            pts = MaskModificationService._normalize_contour_points(pts)
            if pts.shape[0] < 3:
                continue

            dx = pts[:, 0] - float(ref_x)
            dy = pts[:, 1] - float(ref_y)
            min_d2 = float(np.min((dx * dx) + (dy * dy)))
            area = float(
                abs(
                    cv2.contourArea(
                        np.rint(pts).astype(np.int32).reshape((-1, 1, 2))
                    )
                )
            )
            score = (min_d2, -area)
            if best_score is None or score < best_score:
                best_score = score
                best_pts = pts
        return best_pts

    @staticmethod
    def _normalize_contour_points(contour_pts: np.ndarray) -> np.ndarray:
        """Remove duplicated successive points and trailing closure duplicate."""
        if contour_pts.ndim != 2 or contour_pts.shape[0] <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        pts = np.rint(contour_pts).astype(np.int32)
        if pts.shape[0] >= 2 and np.array_equal(pts[0], pts[-1]):
            pts = pts[:-1]
        if pts.shape[0] <= 1:
            return pts.astype(np.float32)

        keep = np.ones((pts.shape[0],), dtype=bool)
        keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
        pts = pts[keep]

        if pts.shape[0] >= 2:
            unique_pts: list[np.ndarray] = []
            seen: set[tuple[int, int]] = set()
            for point in pts:
                key = (int(point[0]), int(point[1]))
                if key in seen:
                    continue
                seen.add(key)
                unique_pts.append(point)
            if unique_pts:
                pts = np.asarray(unique_pts, dtype=np.int32)

        if pts.shape[0] >= 2 and np.array_equal(pts[0], pts[-1]):
            pts = pts[:-1]
        return pts.astype(np.float32)

    def _clear_active_component(self) -> None:
        self._active_slice_idx = None
        self._active_label = None
        self._active_slice_mask = None
        self._selected_components = []
