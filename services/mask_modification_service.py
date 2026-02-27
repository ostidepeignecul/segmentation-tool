"""Service dedicated to anchored mask contour editing in pending mode."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class MaskModificationService:
    """Handle component selection, anchor dragging, and deferred mask commit."""

    ANCHOR_HIT_RADIUS_PX = 7
    CONTOUR_HIT_TOLERANCE_PX = 8

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._image_shape: Optional[tuple[int, int]] = None
        self._dirty_slices: set[int] = set()
        self._drag_active = False
        self._drag_anchor_list_idx: Optional[int] = None
        self._active_slice_idx: Optional[int] = None
        self._active_label: Optional[int] = None
        self._active_slice_mask: Optional[np.ndarray] = None
        self._other_label_mask: Optional[np.ndarray] = None
        self._working_component_mask: Optional[np.ndarray] = None
        self._contour_points: Optional[np.ndarray] = None  # float32 (N,2), x/y
        self._anchor_indices: list[int] = []  # indices in _contour_points

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
        self._drag_anchor_list_idx = None

    def anchor_points(self, *, slice_idx: int, label: int) -> list[tuple[int, int]]:
        if (
            self._active_slice_idx is None
            or self._active_label is None
            or self._contour_points is None
            or self._active_slice_idx != int(slice_idx)
            or self._active_label != int(label)
            or not self._anchor_indices
        ):
            return []
        out: list[tuple[int, int]] = []
        for contour_idx in self._anchor_indices:
            x, y = self._contour_points[int(contour_idx)]
            out.append((int(round(float(x))), int(round(float(y)))))
        return out

    def active_anchor_index(self) -> Optional[int]:
        if not self._drag_active or self._drag_anchor_list_idx is None:
            return None
        if not self._anchor_indices:
            return None
        return int(max(0, min(len(self._anchor_indices) - 1, self._drag_anchor_list_idx)))

    def has_full_density_anchors(self, *, slice_idx: int, label: int) -> bool:
        if (
            self._active_slice_idx is None
            or self._active_label is None
            or self._contour_points is None
            or self._active_slice_idx != int(slice_idx)
            or self._active_label != int(label)
        ):
            return False
        if self._contour_points.size == 0:
            return False
        if len(self._anchor_indices) < int(self._contour_points.shape[0]):
            return False
        pts = np.rint(self._contour_points).astype(np.int32)
        n_pts = int(pts.shape[0])
        if n_pts < 2:
            return True
        for idx in range(n_pts):
            nxt = (idx + 1) % n_pts
            dx = abs(int(pts[nxt, 0]) - int(pts[idx, 0]))
            dy = abs(int(pts[nxt, 1]) - int(pts[idx, 1]))
            if max(dx, dy) > 1:
                return False
        return True

    def start_drag(
        self,
        *,
        slice_idx: int,
        label: int,
        x_pos: int,
        y_pos: int,
        slice_mask: np.ndarray,
    ) -> bool:
        if label <= 0:
            return False
        slice_arr = np.asarray(slice_mask, dtype=np.uint8)
        if slice_arr.ndim != 2:
            return False
        shape = (int(slice_arr.shape[0]), int(slice_arr.shape[1]))
        if self._image_shape is None:
            self._image_shape = shape
        elif self._image_shape != shape:
            return False

        z = int(slice_idx)
        lbl = int(label)
        if (
            self._active_slice_idx != z
            or self._active_label != lbl
            or self._contour_points is None
            or self._working_component_mask is None
            or self._other_label_mask is None
            or self._active_slice_mask is None
        ):
            if not self._select_component(
                slice_idx=z,
                label=lbl,
                x_pos=x_pos,
                y_pos=y_pos,
                slice_mask=slice_arr,
            ):
                return False

        if self._contour_points is None or not self._anchor_indices:
            return False
        anchor_list_idx = self._find_anchor_list_index(
            x_pos=int(x_pos),
            y_pos=int(y_pos),
            radius_px=self.ANCHOR_HIT_RADIUS_PX,
        )
        if anchor_list_idx is None:
            return False

        self._drag_active = True
        self._drag_anchor_list_idx = int(anchor_list_idx)
        return True

    def drag_to(self, *, slice_idx: int, x_pos: int, y_pos: int) -> Optional[np.ndarray]:
        if not self._drag_active:
            return None
        if self._drag_anchor_list_idx is None:
            return None
        if self._image_shape is None:
            return None
        if self._active_slice_idx is None or self._active_label is None:
            return None
        if self._contour_points is None or self._working_component_mask is None or self._other_label_mask is None:
            return None
        if self._active_slice_mask is None:
            return None
        if int(slice_idx) != int(self._active_slice_idx):
            return None
        if not self._anchor_indices:
            return None

        anchor_list_idx = int(max(0, min(len(self._anchor_indices) - 1, self._drag_anchor_list_idx)))
        anchor_point_idx = int(self._anchor_indices[anchor_list_idx])
        contour = np.array(self._contour_points, dtype=np.float32, copy=True)
        if contour.ndim != 2 or contour.shape[0] < 3:
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

        # Move only the selected anchor vertex: no neighborhood deformation.
        contour[anchor_point_idx, 0] = x_new
        contour[anchor_point_idx, 1] = y_new
        target_x = int(round(float(x_new)))
        target_y = int(round(float(y_new)))
        contour[:, 0] = np.clip(contour[:, 0], 0.0, float(width - 1))
        contour[:, 1] = np.clip(contour[:, 1], 0.0, float(height - 1))
        contour = self._normalize_contour_points(np.rint(contour).astype(np.float32))
        if contour.shape[0] < 3:
            return None

        contour_int = np.rint(contour).astype(np.int32)
        if np.unique(contour_int, axis=0).shape[0] < 3:
            return None

        component_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(component_mask, [contour_int.reshape((-1, 1, 2))], 1)
        if not np.any(component_mask):
            return None
        component_bool = component_mask.astype(bool)

        label = int(self._active_label)
        updated_slice = np.array(self._active_slice_mask, dtype=np.uint8, copy=True)
        updated_slice[self._working_component_mask] = 0
        updated_slice[self._other_label_mask] = label
        blocked = np.logical_and(updated_slice != 0, updated_slice != label)
        if np.any(blocked):
            component_bool = np.logical_and(component_bool, np.logical_not(blocked))
            if not np.any(component_bool):
                return None
        updated_slice[component_bool] = label
        self._active_slice_mask = np.array(updated_slice, dtype=np.uint8, copy=True)
        self._working_component_mask = component_bool
        self._contour_points = contour
        self._anchor_indices = self._build_anchor_indices(contour)
        nearest_active = self._nearest_anchor_list_index(
            x_pos=int(target_x),
            y_pos=int(target_y),
        )
        if nearest_active is not None:
            self._drag_anchor_list_idx = int(nearest_active)
        self._dirty_slices.add(int(self._active_slice_idx))
        return np.array(updated_slice, dtype=np.uint8, copy=True)

    def end_drag(self) -> None:
        self._drag_active = False
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
        if label <= 0:
            return False
        slice_arr = np.asarray(slice_mask, dtype=np.uint8)
        if slice_arr.ndim != 2:
            return False

        z = int(slice_idx)
        lbl = int(label)
        if (
            self._active_slice_idx != z
            or self._active_label != lbl
            or self._contour_points is None
            or self._working_component_mask is None
            or self._other_label_mask is None
            or self._active_slice_mask is None
        ):
            if not self._select_component(
                slice_idx=z,
                label=lbl,
                x_pos=x_pos,
                y_pos=y_pos,
                slice_mask=slice_arr,
            ):
                return False

        inserted = self._insert_anchor_near_point(
            x_pos=int(x_pos),
            y_pos=int(y_pos),
            max_dist_px=self.CONTOUR_HIT_TOLERANCE_PX,
        )
        return inserted is not None

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

    def _select_component(
        self,
        *,
        slice_idx: int,
        label: int,
        x_pos: int,
        y_pos: int,
        slice_mask: np.ndarray,
    ) -> bool:
        if self._image_shape is None:
            return False
        z = int(slice_idx)
        lbl = int(label)
        slice_arr = np.asarray(slice_mask, dtype=np.uint8)
        if slice_arr.ndim != 2:
            return False
        height, width = self._image_shape
        x = int(max(0, min(width - 1, int(x_pos))))
        y = int(max(0, min(height - 1, int(y_pos))))

        label_binary = (slice_arr == lbl).astype(np.uint8)
        if not np.any(label_binary):
            return False

        n_components, component_map = cv2.connectedComponents(label_binary, connectivity=8)
        if n_components <= 1:
            return False

        component_id = int(component_map[y, x]) if 0 <= y < height and 0 <= x < width else 0
        if component_id <= 0:
            contours, _ = cv2.findContours(label_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            best_d2 = None
            best_component = 0
            for cnt in contours:
                if cnt is None or cnt.size == 0:
                    continue
                pts = cnt.reshape((-1, 2))
                d2 = np.min((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2)
                px, py = int(pts[0][0]), int(pts[0][1])
                cid = int(component_map[py, px]) if 0 <= py < height and 0 <= px < width else 0
                if cid <= 0:
                    continue
                if best_d2 is None or float(d2) < float(best_d2):
                    best_d2 = float(d2)
                    best_component = int(cid)
            if best_component <= 0:
                return False
            max_d2 = float(self.CONTOUR_HIT_TOLERANCE_PX * self.CONTOUR_HIT_TOLERANCE_PX)
            if best_d2 is not None and float(best_d2) > max_d2:
                return False
            component_id = int(best_component)

        component_bool = component_map == int(component_id)
        if not np.any(component_bool):
            return False
        contours, _ = cv2.findContours(component_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_pts = self._extract_component_contour_points(contours=contours, ref_x=x, ref_y=y)
        if contour_pts is None:
            return False
        if contour_pts.shape[0] < 3:
            return False

        other_label_mask = np.logical_and(slice_arr == lbl, np.logical_not(component_bool))
        self._active_slice_idx = z
        self._active_label = lbl
        self._active_slice_mask = np.array(slice_arr, dtype=np.uint8, copy=True)
        self._other_label_mask = other_label_mask
        self._working_component_mask = component_bool
        self._contour_points = contour_pts
        self._anchor_indices = self._build_anchor_indices(contour_pts)
        if not self._anchor_indices:
            return False
        return True

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

        # Remove any repeated pixel globally to avoid duplicated anchors.
        if pts.shape[0] >= 2:
            unique_pts: list[np.ndarray] = []
            seen: set[tuple[int, int]] = set()
            for p in pts:
                key = (int(p[0]), int(p[1]))
                if key in seen:
                    continue
                seen.add(key)
                unique_pts.append(p)
            if unique_pts:
                pts = np.asarray(unique_pts, dtype=np.int32)

        if pts.shape[0] >= 2 and np.array_equal(pts[0], pts[-1]):
            pts = pts[:-1]
        return pts.astype(np.float32)

    def _find_anchor_list_index(self, *, x_pos: int, y_pos: int, radius_px: int) -> Optional[int]:
        if self._contour_points is None or not self._anchor_indices:
            return None
        r_sq = float(int(radius_px) * int(radius_px))
        best_idx: Optional[int] = None
        best_dist = r_sq + 1.0
        for list_idx, contour_idx in enumerate(self._anchor_indices):
            x, y = self._contour_points[int(contour_idx)]
            dx = float(x) - float(x_pos)
            dy = float(y) - float(y_pos)
            dist = (dx * dx) + (dy * dy)
            if dist <= r_sq and dist < best_dist:
                best_dist = dist
                best_idx = int(list_idx)
        return best_idx

    def _nearest_anchor_list_index(self, x_pos: int, y_pos: int) -> Optional[int]:
        if self._contour_points is None or not self._anchor_indices:
            return None
        best_idx: Optional[int] = None
        best_dist = None
        for list_idx, contour_idx in enumerate(self._anchor_indices):
            x, y = self._contour_points[int(contour_idx)]
            dx = float(x) - float(x_pos)
            dy = float(y) - float(y_pos)
            dist = (dx * dx) + (dy * dy)
            if best_dist is None or dist < best_dist:
                best_idx = int(list_idx)
                best_dist = float(dist)
        return best_idx

    def _insert_anchor_near_point(self, *, x_pos: int, y_pos: int, max_dist_px: int) -> Optional[int]:
        if self._contour_points is None:
            return None
        contour = np.asarray(self._contour_points, dtype=np.float32)
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

        self._contour_points = new_contour
        self._anchor_indices = self._build_anchor_indices(new_contour)
        return self._nearest_anchor_list_index(ix, iy)

    def _clear_active_component(self) -> None:
        self._active_slice_idx = None
        self._active_label = None
        self._active_slice_mask = None
        self._other_label_mask = None
        self._working_component_mask = None
        self._contour_points = None
        self._anchor_indices = []
