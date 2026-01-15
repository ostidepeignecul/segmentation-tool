"""Service métier pour les opérations d'annotation/ROI (stubs)."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
from skimage.morphology import flood
from scipy.ndimage import label as scipy_label

from config.constants import MASK_COLORS_BGRA
from models.annotation_model import AnnotationModel
from models.roi_model import RoiModel
from models.temp_mask_model import TempMaskModel


class AnnotationService:
    """Encapsule la logique ROI, seuils et propagation (placeholder)."""

    def compute_threshold(
        self,
        gray_slice: Any,
        free_hand_points: Sequence[Tuple[int, int]] | None,
        *,
        auto: bool,
    ) -> Optional[float]:
        """Calcule ou retourne un seuil pour la ROI courante (stub)."""
        return None

    def build_roi_mask(
        self,
        shape: Tuple[int, int],
        free_hand_points: Sequence[Tuple[int, int]] | None = None,
        box: Tuple[int, int, int, int] | None = None,
        point: Tuple[int, int] | None = None,
    ) -> Optional[Any]:
        """Construit un masque binaire pour la ROI courante (stub)."""
        return None

    def apply_label_on_slice(
        self,
        mask: Any,
        label: int,
        *,
        persistence: bool,
    ) -> Optional[Any]:
        """Applique un label sur une slice à partir d'un masque ROI (stub)."""
        return None

    def propagate_volume(
        self,
        slice_mask: Any,
        target_depth: int,
        *,
        persistence: bool,
    ) -> Optional[Any]:
        """Propage un masque de slice dans tout le volume (stub)."""
        return None

    # ------------------------------------------------------------------ #
    # Rectangle helpers
    # ------------------------------------------------------------------ #
    def build_box_mask(self, shape: Tuple[int, int], box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Build a binary mask for a box defined by two opposite corners."""
        try:
            h, w = int(shape[0]), int(shape[1])
        except Exception:
            return None
        if h <= 0 or w <= 0:
            return None
        x1, y1, x2, y2 = [int(v) for v in box]
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        xmin = max(0, min(w - 1, xmin))
        xmax = max(0, min(w - 1, xmax))
        ymin = max(0, min(h - 1, ymin))
        ymax = max(0, min(h - 1, ymax))
        if xmax < xmin or ymax < ymin:
            return None
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[ymin : ymax + 1, xmin : xmax + 1] = 1
        return mask

    def apply_temp_box(
        self,
        temp_model: TempMaskModel,
        slice_idx: int,
        box_mask: np.ndarray,
        label: int,
        *,
        persistent: bool,
    ) -> None:
        """Apply a box mask into the temporary mask model."""
        temp_model.set_slice_mask(slice_idx, box_mask, label=label, persistent=persistent)

    def build_disk_mask(
        self,
        shape: Tuple[int, int],
        center: Tuple[int, int],
        radius: int,
    ) -> Optional[np.ndarray]:
        """Build a binary disk mask inside given (H, W)."""
        try:
            h, w = int(shape[0]), int(shape[1])
            cx = int(center[0])
            cy = int(center[1])
            r = max(1, int(radius))
        except Exception:
            return None
        if h <= 0 or w <= 0 or r <= 0:
            return None

        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        xmin = max(0, cx - r)
        xmax = min(w - 1, cx + r)
        ymin = max(0, cy - r)
        ymax = min(h - 1, cy + r)
        if xmin > xmax or ymin > ymax:
            return None

        mask = np.zeros((h, w), dtype=np.uint8)
        yy, xx = np.ogrid[ymin : ymax + 1, xmin : xmax + 1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        mask[ymin : ymax + 1, xmin : xmax + 1][disk] = 1
        if not np.any(mask):
            return None
        return mask

    # ------------------------------------------------------------------ #
    # Grow helpers
    # ------------------------------------------------------------------ #
    def _normalize_restriction_mask(
        self,
        restriction_mask: Optional[np.ndarray],
        shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if restriction_mask is None:
            return None
        try:
            mask = np.asarray(restriction_mask, dtype=bool)
        except Exception:
            return None
        try:
            h, w = int(shape[0]), int(shape[1])
        except Exception:
            return None
        if mask.shape != (h, w):
            return None
        return mask

    def _normalize_blocked_mask(
        self,
        blocked_mask: Optional[np.ndarray],
        shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if blocked_mask is None:
            return None
        try:
            mask = np.asarray(blocked_mask, dtype=bool)
        except Exception:
            return None
        try:
            h, w = int(shape[0]), int(shape[1])
        except Exception:
            return None
        if mask.shape != (h, w):
            return None
        return mask

    def _normalize_slice_to_uint8(self, slice_data: np.ndarray) -> Optional[np.ndarray]:
        """Normalize arbitrary slice data to uint8 [0, 255]."""
        try:
            data = np.asarray(slice_data, dtype=np.float32)
        except Exception:
            return None
        if data.ndim == 3 and data.shape[2] in (3, 4):
            data = data[..., 0]
        if data.ndim != 2 or data.size == 0:
            return None
        dmin = float(data.min())
        dmax = float(data.max())
        if dmax <= dmin:
            return np.zeros_like(data, dtype=np.uint8)
        norm = (data - dmin) / (dmax - dmin) * 255.0
        return norm.astype(np.uint8)

    def build_grow_mask(
        self,
        shape: Tuple[int, int],
        seed: Tuple[int, int],
        slice_data: np.ndarray,
        threshold: Optional[float],
        restriction_mask: Optional[np.ndarray] = None,
        blocked_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Region growing (4-connexe) optimisé avec skimage.morphology.flood."""
        try:
            h, w = int(shape[0]), int(shape[1])
        except Exception:
            return None
        if h <= 0 or w <= 0:
            return None

        norm = self._normalize_slice_to_uint8(slice_data)
        if norm is None or norm.shape[0] < h or norm.shape[1] < w:
            return None

        x = max(0, min(w - 1, int(seed[0])))
        y = max(0, min(h - 1, int(seed[1])))

        restriction = self._normalize_restriction_mask(restriction_mask, (h, w))
        if restriction is not None and not restriction[y, x]:
            return None
        
        blocked = self._normalize_blocked_mask(blocked_mask, (h, w))
        if blocked is not None and blocked[y, x]:
            return None

        thr = float(threshold) if threshold is not None else 0.0
        if norm[y, x] < thr:
            return None

        valid_mask = (norm >= thr)
        if restriction is not None:
            valid_mask &= restriction.astype(bool)
        if blocked is not None:
            valid_mask &= ~blocked.astype(bool)

        if not valid_mask[y, x]:
            return None

        try:
            flooded = flood(valid_mask, (y, x), connectivity=1)
        except Exception:
            return None
            
        if not np.any(flooded):
            return None

        return flooded.astype(np.uint8)

    def _rasterize_line_segment(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        *,
        width: int,
        height: int,
    ) -> list[Tuple[int, int]]:
        """Rasterize a line segment into integer pixel coordinates."""
        x1 = max(0, min(width - 1, int(start[0])))
        y1 = max(0, min(height - 1, int(start[1])))
        x2 = max(0, min(width - 1, int(end[0])))
        y2 = max(0, min(height - 1, int(end[1])))
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        points: list[Tuple[int, int]] = []
        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return points

    def _rasterize_polyline(
        self,
        points: Sequence[Tuple[int, int]],
        shape: Tuple[int, int],
    ) -> list[Tuple[int, int]]:
        """Convert a polyline into a list of unique pixel seeds."""
        try:
            h, w = int(shape[0]), int(shape[1])
        except Exception:
            return []
        if h <= 0 or w <= 0:
            return []
        if not points:
            return []
        raw = [(int(x), int(y)) for x, y in points]
        seeds: list[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        if len(raw) == 1:
            x = max(0, min(w - 1, raw[0][0]))
            y = max(0, min(h - 1, raw[0][1]))
            return [(x, y)]
        prev = raw[0]
        for cur in raw[1:]:
            for x, y in self._rasterize_line_segment(prev, cur, width=w, height=h):
                if (x, y) not in seen:
                    seeds.append((x, y))
                    seen.add((x, y))
            prev = cur
        return seeds

    def build_grow_mask_from_seeds(
        self,
        shape: Tuple[int, int],
        seeds: Sequence[Tuple[int, int]],
        slice_data: np.ndarray,
        threshold: Optional[float],
        restriction_mask: Optional[np.ndarray] = None,
        blocked_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Region growing (4-connexe) optimisé depuis plusieurs seeds."""
        try:
            h, w = int(shape[0]), int(shape[1])
        except Exception:
            return None
        if h <= 0 or w <= 0:
            return None
        if not seeds:
            return None

        norm = self._normalize_slice_to_uint8(slice_data)
        if norm is None or norm.shape[0] < h or norm.shape[1] < w:
            return None

        thr = float(threshold) if threshold is not None else 0.0

        valid_mask = (norm >= thr)
        restriction = self._normalize_restriction_mask(restriction_mask, (h, w))
        if restriction is not None:
            valid_mask &= restriction.astype(bool)
        
        blocked = self._normalize_blocked_mask(blocked_mask, (h, w))
        if blocked is not None:
            valid_mask &= ~blocked.astype(bool)
        
        if not np.any(valid_mask):
            return None

        labeled_array, num_features = scipy_label(valid_mask, structure=[[0,1,0],[1,1,1],[0,1,0]])
        if num_features == 0:
            return None

        seed_indices = tuple(np.array(seeds).T[::-1])
        
        ys, xs = seed_indices
        valid_seeds_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(valid_seeds_mask):
            return None
            
        ys = ys[valid_seeds_mask]
        xs = xs[valid_seeds_mask]
        
        touched_labels = labeled_array[ys, xs]
        unique_labels = np.unique(touched_labels)
        unique_labels = unique_labels[unique_labels != 0]
        
        if unique_labels.size == 0:
            return None
            
        mask = np.isin(labeled_array, unique_labels).astype(np.uint8)
        return mask

    def apply_grow_roi(
        self,
        *,
        slice_idx: int,
        point: Tuple[int, int],
        shape: Tuple[int, int],
        slice_data: np.ndarray,
        label: int,
        threshold: Optional[float],
        persistent: bool,
        roi_model: RoiModel,
        temp_mask_model: TempMaskModel,
        palette: Optional[dict[int, tuple[int, int, int, int]]] = None,
        restriction_mask: Optional[np.ndarray] = None,
        blocked_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Region growing ROI: build mask from seed/threshold, store ROI and apply to temp model.
        """
        grow_mask = self.build_grow_mask(
            shape,
            point,
            slice_data,
            threshold,
            restriction_mask=restriction_mask,
            blocked_mask=blocked_mask,
        )
        if grow_mask is None:
            return None

        color = None
        if palette is not None:
            color = palette.get(label)
        if color is None:
            color = MASK_COLORS_BGRA.get(label, (255, 0, 255, 160))

        roi_model.add_grow(
            slice_idx,
            point,
            label=label,
            threshold=threshold,
            persistent=persistent,
        )
        temp_mask_model.ensure_label(label, color, visible=True)
        temp_mask_model.set_slice_mask(slice_idx, grow_mask, label=label, persistent=persistent)
        return grow_mask

    def apply_line_roi(
        self,
        *,
        slice_idx: int,
        points: Sequence[Tuple[int, int]],
        shape: Tuple[int, int],
        slice_data: np.ndarray,
        label: int,
        threshold: Optional[float],
        persistent: bool,
        roi_model: RoiModel,
        temp_mask_model: TempMaskModel,
        palette: Optional[dict[int, tuple[int, int, int, int]]] = None,
        restriction_mask: Optional[np.ndarray] = None,
        blocked_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Region growing ROI from a freehand line (multi-seed), store ROI and apply to temp model.
        """
        seeds = self._rasterize_polyline(points, shape)
        if not seeds:
            return None

        line_mask = self.build_grow_mask_from_seeds(
            shape,
            seeds,
            slice_data,
            threshold,
            restriction_mask=restriction_mask,
            blocked_mask=blocked_mask,
        )
        if line_mask is None:
            return None

        color = None
        if palette is not None:
            color = palette.get(label)
        if color is None:
            color = MASK_COLORS_BGRA.get(label, (255, 0, 255, 160))

        roi_model.add_line(
            slice_idx,
            points,
            label=label,
            threshold=threshold,
            persistent=persistent,
        )
        temp_mask_model.ensure_label(label, color, visible=True)
        temp_mask_model.set_slice_mask(slice_idx, line_mask, label=label, persistent=persistent)
        return line_mask

    def rebuild_temp_masks_for_slice(
        self,
        *,
        rois: Sequence,
        shape: Tuple[int, int],
        slice_idx: int,
        temp_mask_model: TempMaskModel,
        palette: Optional[dict[int, tuple[int, int, int, int]]] = None,
        clear_slice: bool = True,
        slice_data: Optional[np.ndarray] = None,
        restriction_mask: Optional[np.ndarray] = None,
        blocked_mask: Optional[np.ndarray] = None,
    ) -> list[Tuple[int, int, int, int]]:
        """
        Rebuild all temporary masks for a slice from ROI definitions.

        Returns the list of box tuples (x1, y1, x2, y2) for display.
        """
        boxes: list[Tuple[int, int, int, int]] = []
        if clear_slice:
            temp_mask_model.clear_slice(slice_idx)
        restriction = self._normalize_restriction_mask(restriction_mask, shape)
        blocked = self._normalize_blocked_mask(blocked_mask, shape)

        for roi in rois:
            if getattr(roi, "roi_type", None) != "box" or len(getattr(roi, "points", [])) < 2:
                continue
            box_tuple = roi.points[0] + roi.points[1]
            box_mask = self.build_box_mask(shape, box_tuple)
            if box_mask is None:
                continue
            if restriction is not None:
                box_mask = np.where(restriction, box_mask, 0)
                if not np.any(box_mask):
                    continue
            label = getattr(roi, "label", 1)
            color = (palette or {}).get(label) or MASK_COLORS_BGRA.get(label, (255, 0, 255, 160))
            temp_mask_model.ensure_label(label, color, visible=True)
            mask_to_apply = box_mask
            roi_threshold = getattr(roi, "threshold", None)
            if slice_data is not None and roi_threshold is not None:
                mask_to_apply = self.build_thresholded_mask(box_mask, slice_data, roi_threshold)
            if blocked is not None and int(label) != 0:
                mask_to_apply = np.where(blocked, 0, mask_to_apply)
            self.apply_temp_box(
                temp_mask_model,
                slice_idx,
                mask_to_apply,
                label,
                persistent=getattr(roi, "persistent", False),
            )
            boxes.append(box_tuple)
        # Grow ROIs
        for roi in rois:
            if getattr(roi, "roi_type", None) != "grow" or len(getattr(roi, "points", [])) < 1:
                continue
            if slice_data is None:
                continue
            seed = roi.points[0]
            label = getattr(roi, "label", 1)
            color = (palette or {}).get(label) or MASK_COLORS_BGRA.get(label, (255, 0, 255, 160))
            temp_mask_model.ensure_label(label, color, visible=True)
            grow_mask = self.build_grow_mask(
                shape,
                seed,
                slice_data,
                getattr(roi, "threshold", None),
                restriction_mask=restriction,
                blocked_mask=blocked if int(label) != 0 else None,
            )
            if grow_mask is None:
                continue
            self.apply_temp_box(
                temp_mask_model,
                slice_idx,
                grow_mask,
                label,
                persistent=getattr(roi, "persistent", False),
            )
        # Line ROIs (multi-seed grow)
        for roi in rois:
            if getattr(roi, "roi_type", None) != "line" or len(getattr(roi, "points", [])) < 1:
                continue
            if slice_data is None:
                continue
            label = getattr(roi, "label", 1)
            color = (palette or {}).get(label) or MASK_COLORS_BGRA.get(label, (255, 0, 255, 160))
            temp_mask_model.ensure_label(label, color, visible=True)
            seeds = self._rasterize_polyline(roi.points, shape)
            if not seeds:
                continue
            line_mask = self.build_grow_mask_from_seeds(
                shape,
                seeds,
                slice_data,
                getattr(roi, "threshold", None),
                restriction_mask=restriction,
                blocked_mask=blocked if int(label) != 0 else None,
            )
            if line_mask is None:
                continue
            self.apply_temp_box(
                temp_mask_model,
                slice_idx,
                line_mask,
                label,
                persistent=getattr(roi, "persistent", False),
            )
        return boxes

    # ------------------------------------------------------------------ #
    # Paint helpers (brush)
    # ------------------------------------------------------------------ #
    def paint_disk_on_slice(
        self,
        annotation_model: AnnotationModel,
        slice_idx: int,
        center: Tuple[int, int],
        *,
        label: int,
        radius: int,
    ) -> bool:
        """
        Paint a disk with the given label on a slice of the mask volume.
        Supports label=0 to erase.
        Returns True if the slice was modified.
        """
        mask_volume = annotation_model.get_mask_volume()
        if mask_volume is None:
            return False
        try:
            z = max(0, min(mask_volume.shape[0] - 1, int(slice_idx)))
            cx = int(center[0])
            cy = int(center[1])
            r = max(1, int(radius))
            lbl = int(label)
        except Exception:
            return False

        if r <= 0:
            return False
        h, w = mask_volume.shape[1], mask_volume.shape[2]
        if h <= 0 or w <= 0:
            return False

        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        xmin = max(0, cx - r)
        xmax = min(w - 1, cx + r)
        ymin = max(0, cy - r)
        ymax = min(h - 1, cy + r)
        if xmin > xmax or ymin > ymax:
            return False

        current_slice = mask_volume[z]
        yy, xx = np.ogrid[ymin : ymax + 1, xmin : xmax + 1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        if not np.any(disk):
            return False

        updated = current_slice.copy()
        updated[ymin : ymax + 1, xmin : xmax + 1][disk] = lbl
        if np.array_equal(updated, current_slice):
            return False

        annotation_model.set_slice_mask(z, updated)
        return True

    def rebuild_volume_preview_from_rois(
        self,
        *,
        depth: int,
        mask_shape: Tuple[int, int],
        roi_model: RoiModel,
        temp_mask_model: TempMaskModel,
        palette: Optional[dict[int, tuple[int, int, int, int]]] = None,
        slice_data_provider: Callable[[int], Optional[np.ndarray]],
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        restriction_mask: Optional[np.ndarray] = None,
        blocked_mask_provider: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    ) -> None:
        """
        Rebuild temporary masks for the whole volume from stored ROIs.
        """
        if temp_mask_model.get_mask_volume() is None:
            temp_mask_model.initialize((depth, mask_shape[0], mask_shape[1]))
        else:
            temp_mask_model.clear()

        min_idx = 0 if start_idx is None else max(0, int(start_idx))
        max_idx = (int(depth) - 1) if end_idx is None else min(int(end_idx), int(depth) - 1)
        if max_idx < min_idx:
            return

        for idx in range(min_idx, max_idx + 1):
            rois = list(roi_model.list_on_slice(idx))
            persistent_rois = roi_model.list_persistent()
            if persistent_rois:
                seen_ids = {roi.id for roi in rois}
                rois.extend([roi for roi in persistent_rois if roi.id not in seen_ids])
            if not rois:
                continue
            slice_data = slice_data_provider(idx)
            blocked_mask = (
                blocked_mask_provider(idx) if blocked_mask_provider is not None else None
            )
            self.rebuild_temp_masks_for_slice(
                rois=rois,
                shape=mask_shape,
                slice_idx=idx,
                temp_mask_model=temp_mask_model,
                palette=palette,
                clear_slice=True,
                slice_data=slice_data,
                restriction_mask=restriction_mask,
                blocked_mask=blocked_mask,
            )

    def build_thresholded_mask(
        self,
        box_mask: np.ndarray,
        slice_data: np.ndarray,
        threshold: Optional[float],
    ) -> np.ndarray:
        """
        Apply a threshold inside the box mask to select pixels for the temp mask.
        """
        if threshold is None:
            return box_mask
        try:
            data = np.asarray(slice_data, dtype=np.float32)
        except Exception:
            return box_mask
        if data.ndim == 3 and data.shape[2] in (3, 4):
            data = data[..., 0]
        mask = np.zeros_like(box_mask, dtype=np.uint8)
        inside = box_mask > 0
        if not np.any(inside):
            return mask
        data_inside = data[inside]
        if data_inside.size == 0:
            return mask
        finite = np.isfinite(data_inside)
        if not np.all(finite):
            data_inside = data_inside[finite]
            if data_inside.size == 0:
                return mask
        dmin = float(data_inside.min())
        dmax = float(data_inside.max())
        lo, hi = dmin, dmax
        try:
            lo_p, hi_p = np.percentile(data_inside, (1.0, 99.0))
            if np.isfinite(lo_p) and np.isfinite(hi_p) and hi_p > lo_p:
                lo, hi = float(lo_p), float(hi_p)
        except Exception:
            pass
        if hi <= lo:
            if dmax <= dmin:
                norm = np.zeros_like(data_inside, dtype=np.float32)
            else:
                lo, hi = dmin, dmax
                clipped = np.clip(data_inside, lo, hi)
                norm = (clipped - lo) / (hi - lo) * 255.0
        else:
            clipped = np.clip(data_inside, lo, hi)
            norm = (clipped - lo) / (hi - lo) * 255.0
        thr = float(threshold)
        mask_inside = (norm >= thr).astype(np.uint8)
        mask[inside] = mask_inside
        return mask

    # ------------------------------------------------------------------ #
    # Rectangle end-to-end
    # ------------------------------------------------------------------ #
    def apply_box_roi(
        self,
        *,
        slice_idx: int,
        box: Any,
        shape: Tuple[int, int],
        label: int,
        threshold: Optional[float],
        persistent: bool,
        roi_model: RoiModel,
        temp_mask_model: TempMaskModel,
        palette: Optional[dict[int, tuple[int, int, int, int]]] = None,
        slice_data: Optional[np.ndarray] = None,
        restriction_mask: Optional[np.ndarray] = None,
        blocked_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Normalize a box, build its mask, store ROI metadata and apply mask to temp model.
        Returns the mask for preview.
        """
        box_tuple = self._normalize_box_input(box)
        if box_tuple is None:
            return None

        box_mask = self.build_box_mask(shape, box_tuple)
        if box_mask is None:
            return None
        restriction = self._normalize_restriction_mask(restriction_mask, shape)
        if restriction is not None:
            box_mask = np.where(restriction, box_mask, 0)
            if not np.any(box_mask):
                return None

        color = None
        if palette is not None:
            color = palette.get(label)
        if color is None:
            color = MASK_COLORS_BGRA.get(label, (255, 0, 255, 160))

        roi_model.add_box(
            slice_idx,
            (box_tuple[0], box_tuple[1]),
            (box_tuple[2], box_tuple[3]),
            label=label,
            threshold=threshold,
            persistent=persistent,
        )
        temp_mask_model.ensure_label(label, color, visible=True)
        mask_to_apply = box_mask
        if slice_data is not None and threshold is not None:
            mask_to_apply = self.build_thresholded_mask(box_mask, slice_data, threshold)
        if blocked_mask is not None and int(label) != 0:
            blocked = np.asarray(blocked_mask, dtype=bool)
            if blocked.shape == mask_to_apply.shape:
                mask_to_apply = np.where(blocked, 0, mask_to_apply)

        self.apply_temp_box(
            temp_mask_model, slice_idx, mask_to_apply, label, persistent=persistent
        )
        return mask_to_apply

    @staticmethod
    def _normalize_box_input(box: Any) -> Optional[Tuple[int, int, int, int]]:
        """Convert various box formats to (x1, y1, x2, y2)."""
        if box is None:
            return None
        if isinstance(box, (list, tuple)):
            if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                x1, y1, x2, y2 = box
                return (int(x1), int(y1), int(x2), int(y2))
            if len(box) == 2 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box):
                (x1, y1), (x2, y2) = box
                return (int(x1), int(y1), int(x2), int(y2))
        return None

    # ------------------------------------------------------------------ #
    # Volume helpers
    # ------------------------------------------------------------------ #
    def apply_temp_volume_to_model(
        self,
        *,
        temp_mask_model: TempMaskModel,
        annotation_model: AnnotationModel,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> None:
        """Apply the entire temp mask volume into the annotation model."""
        temp_volume = temp_mask_model.get_mask_volume()
        if temp_volume is None:
            return

        mask_volume = annotation_model.get_mask_volume()
        if mask_volume is None:
            annotation_model.initialize(temp_volume.shape)
            mask_volume = annotation_model.get_mask_volume()
        if mask_volume is None:
            return

        depth = min(temp_volume.shape[0], mask_volume.shape[0])
        min_idx = 0 if start_idx is None else max(0, int(start_idx))
        max_idx = (depth - 1) if end_idx is None else min(int(end_idx), depth - 1)
        if max_idx < min_idx:
            return
        coverage_volume = temp_mask_model.get_coverage_volume()
        for idx in range(min_idx, max_idx + 1):
            temp_slice = temp_volume[idx]
            current_slice = mask_volume[idx]
            if current_slice.shape != temp_slice.shape:
                continue

            if coverage_volume is not None:
                coverage_slice = coverage_volume[idx]
                if coverage_slice is None or not np.any(coverage_slice):
                    continue
                updated = np.array(current_slice, copy=True)
                updated[coverage_slice] = temp_slice[coverage_slice]
                annotation_model.set_slice_mask(idx, updated)
            else:
                if temp_slice is None or not np.any(temp_slice):
                    continue
                updated = np.array(current_slice, copy=True)
                updated[temp_slice > 0] = temp_slice[temp_slice > 0]
                annotation_model.set_slice_mask(idx, updated)

    def propagate_grow_volume_from_slice(
        self,
        *,
        start_slice: int,
        point: Tuple[int, int],
        shape: Tuple[int, int],
        threshold: float,
        label: int,
        depth: int,
        roi_model: RoiModel,
        temp_mask_model: TempMaskModel,
        palette: Optional[dict[int, tuple[int, int, int, int]]] = None,
        slice_data_provider: Callable[[int], Optional[np.ndarray]],
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        restriction_mask: Optional[np.ndarray] = None,
        blocked_mask_provider: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    ) -> None:
        """
        Apply grow ROI forward/backward from a slice until threshold fails.
        """
        if depth <= 0:
            return

        if temp_mask_model.get_mask_volume() is None:
            temp_mask_model.initialize((depth, shape[0], shape[1]))

        for idx in range(int(depth)):
            # Ensure label visibility consistency even if no mask is applied
            if palette is not None and label in palette:
                temp_mask_model.ensure_label(label, palette[label], visible=True)
                break

        min_idx = 0 if start_idx is None else max(0, int(start_idx))
        max_idx = (int(depth) - 1) if end_idx is None else min(int(end_idx), int(depth) - 1)
        if max_idx < min_idx:
            return
        if start_slice < min_idx or start_slice > max_idx:
            return

        def apply_once(idx: int) -> bool:
            slice_data = slice_data_provider(idx)
            if slice_data is None:
                return False
            blocked_mask = (
                blocked_mask_provider(idx) if blocked_mask_provider is not None else None
            )
            mask = self.apply_grow_roi(
                slice_idx=idx,
                point=point,
                shape=shape,
                slice_data=slice_data,
                label=label,
                threshold=threshold,
                persistent=False,
                roi_model=roi_model,
                temp_mask_model=temp_mask_model,
                palette=palette,
                restriction_mask=restriction_mask,
                blocked_mask=blocked_mask,
            )
            return mask is not None

        if not apply_once(start_slice):
            return

        for idx in range(start_slice + 1, max_idx + 1):
            if not apply_once(idx):
                break

        for idx in range(start_slice - 1, min_idx - 1, -1):
            if not apply_once(idx):
                break

    def propagate_line_volume_from_slice(
        self,
        *,
        start_slice: int,
        points: Sequence[Tuple[int, int]],
        shape: Tuple[int, int],
        threshold: float,
        label: int,
        depth: int,
        roi_model: RoiModel,
        temp_mask_model: TempMaskModel,
        palette: Optional[dict[int, tuple[int, int, int, int]]] = None,
        slice_data_provider: Callable[[int], Optional[np.ndarray]],
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        restriction_mask: Optional[np.ndarray] = None,
        blocked_mask_provider: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    ) -> None:
        """
        Apply line ROI forward/backward from a slice until threshold fails.
        """
        if depth <= 0:
            return

        if temp_mask_model.get_mask_volume() is None:
            temp_mask_model.initialize((depth, shape[0], shape[1]))

        for idx in range(int(depth)):
            if palette is not None and label in palette:
                temp_mask_model.ensure_label(label, palette[label], visible=True)
                break

        min_idx = 0 if start_idx is None else max(0, int(start_idx))
        max_idx = (int(depth) - 1) if end_idx is None else min(int(end_idx), int(depth) - 1)
        if max_idx < min_idx:
            return
        if start_slice < min_idx or start_slice > max_idx:
            return

        def apply_once(idx: int) -> bool:
            slice_data = slice_data_provider(idx)
            if slice_data is None:
                return False
            blocked_mask = (
                blocked_mask_provider(idx) if blocked_mask_provider is not None else None
            )
            mask = self.apply_line_roi(
                slice_idx=idx,
                points=points,
                shape=shape,
                slice_data=slice_data,
                label=label,
                threshold=threshold,
                persistent=False,
                roi_model=roi_model,
                temp_mask_model=temp_mask_model,
                palette=palette,
                restriction_mask=restriction_mask,
                blocked_mask=blocked_mask,
            )
            return mask is not None

        if not apply_once(start_slice):
            return

        for idx in range(start_slice + 1, max_idx + 1):
            if not apply_once(idx):
                break

        for idx in range(start_slice - 1, min_idx - 1, -1):
            if not apply_once(idx):
                break

    def apply_box_roi_to_range(
        self,
        *,
        start_idx: int,
        end_idx: int,
        box: Any,
        shape: Tuple[int, int],
        label: int,
        threshold: Optional[float],
        persistent: bool,
        roi_model: RoiModel,
        temp_mask_model: TempMaskModel,
        palette: Optional[dict[int, tuple[int, int, int, int]]] = None,
        slice_data_provider: Optional[Callable[[int], Optional[np.ndarray]]] = None,
        restriction_mask: Optional[np.ndarray] = None,
        blocked_mask_provider: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    ) -> None:
        """Apply a box ROI across a slice range."""
        min_idx = int(min(start_idx, end_idx))
        max_idx = int(max(start_idx, end_idx))
        for idx in range(min_idx, max_idx + 1):
            slice_data = slice_data_provider(idx) if slice_data_provider is not None else None
            blocked_mask = (
                blocked_mask_provider(idx) if blocked_mask_provider is not None else None
            )
            self.apply_box_roi(
                slice_idx=idx,
                box=box,
                shape=shape,
                label=label,
                threshold=threshold,
                persistent=persistent,
                roi_model=roi_model,
                temp_mask_model=temp_mask_model,
                palette=palette,
                slice_data=slice_data,
                restriction_mask=restriction_mask,
                blocked_mask=blocked_mask,
            )
