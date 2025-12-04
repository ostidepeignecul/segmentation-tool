"""Service métier pour les opérations d'annotation/ROI (stubs)."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from config.constants import MASK_COLORS_BGRA
from models.roi_model import RoiModel
from models.temp_mask_model import TempMaskModel


class AnnotationService:
    """Encapsule la logique ROI, seuils et propagation (placeholder)."""

    def compute_threshold(
        self,
        gray_slice: Any,
        polygon: Sequence[Tuple[int, int]] | None,
        *,
        auto: bool,
    ) -> Optional[float]:
        """Calcule ou retourne un seuil pour la ROI courante (stub)."""
        return None

    def build_roi_mask(
        self,
        shape: Tuple[int, int],
        polygon: Sequence[Tuple[int, int]] | None = None,
        rectangle: Tuple[int, int, int, int] | None = None,
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
    def build_rectangle_mask(self, shape: Tuple[int, int], rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Build a binary mask for a rectangle defined by two opposite corners."""
        try:
            h, w = int(shape[0]), int(shape[1])
        except Exception:
            return None
        if h <= 0 or w <= 0:
            return None
        x1, y1, x2, y2 = [int(v) for v in rect]
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

    def apply_temp_rectangle(
        self,
        temp_model: TempMaskModel,
        slice_idx: int,
        rect_mask: np.ndarray,
        label: int,
        *,
        persistent: bool,
    ) -> None:
        """Apply a rectangle mask into the temporary mask model."""
        temp_model.set_slice_mask(slice_idx, rect_mask, label=label, persistent=persistent)

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
    ) -> list[Tuple[int, int, int, int]]:
        """
        Rebuild all temporary masks for a slice from ROI definitions.

        Returns the list of rectangle tuples (x1, y1, x2, y2) for display.
        """
        rects: list[Tuple[int, int, int, int]] = []
        if clear_slice:
            temp_mask_model.clear_slice(slice_idx)

        for roi in rois:
            if getattr(roi, "roi_type", None) != "rectangle" or len(getattr(roi, "points", [])) < 2:
                continue
            rect_tuple = roi.points[0] + roi.points[1]
            rect_mask = self.build_rectangle_mask(shape, rect_tuple)
            if rect_mask is None:
                continue
            label = getattr(roi, "label", 1)
            color = (palette or {}).get(label) or MASK_COLORS_BGRA.get(label, (255, 0, 255, 160))
            temp_mask_model.ensure_label(label, color, visible=True)
            mask_to_apply = rect_mask
            roi_threshold = getattr(roi, "threshold", None)
            if slice_data is not None and roi_threshold is not None:
                mask_to_apply = self.build_thresholded_mask(rect_mask, slice_data, roi_threshold)
            self.apply_temp_rectangle(
                temp_mask_model,
                slice_idx,
                mask_to_apply,
                label,
                persistent=getattr(roi, "persistent", False),
            )
            rects.append(rect_tuple)
        return rects

    def build_thresholded_mask(
        self,
        rect_mask: np.ndarray,
        slice_data: np.ndarray,
        threshold: Optional[float],
    ) -> np.ndarray:
        """
        Apply a threshold inside the rectangle mask to select pixels for the temp polygon.
        """
        if threshold is None:
            return rect_mask
        try:
            data = np.asarray(slice_data, dtype=np.float32)
        except Exception:
            return rect_mask
        if data.ndim == 3 and data.shape[2] in (3, 4):
            data = data[..., 0]
        mask = np.zeros_like(rect_mask, dtype=np.uint8)
        inside = rect_mask > 0
        if not np.any(inside):
            return mask
        data_inside = data[inside]
        dmin = float(data_inside.min()) if data_inside.size else 0.0
        dmax = float(data_inside.max()) if data_inside.size else 1.0
        if dmax <= dmin:
            norm = np.zeros_like(data_inside, dtype=np.float32)
        else:
            norm = (data_inside - dmin) / (dmax - dmin) * 255.0
        thr = float(threshold)
        mask_inside = (norm >= thr).astype(np.uint8)
        mask[inside] = mask_inside
        return mask

    # ------------------------------------------------------------------ #
    # Rectangle end-to-end
    # ------------------------------------------------------------------ #
    def apply_rectangle_roi(
        self,
        *,
        slice_idx: int,
        rect: Any,
        shape: Tuple[int, int],
        label: int,
        threshold: Optional[float],
        persistent: bool,
        roi_model: RoiModel,
        temp_mask_model: TempMaskModel,
        palette: Optional[dict[int, tuple[int, int, int, int]]] = None,
        slice_data: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Normalize a rectangle, build its mask, store ROI metadata and apply mask to temp model.
        Returns the mask for preview.
        """
        rect_tuple = self._normalize_rect_input(rect)
        if rect_tuple is None:
            return None

        rect_mask = self.build_rectangle_mask(shape, rect_tuple)
        if rect_mask is None:
            return None

        color = None
        if palette is not None:
            color = palette.get(label)
        if color is None:
            color = MASK_COLORS_BGRA.get(label, (255, 0, 255, 160))

        roi_model.add_rectangle(
            slice_idx,
            (rect_tuple[0], rect_tuple[1]),
            (rect_tuple[2], rect_tuple[3]),
            label=label,
            threshold=threshold,
            persistent=persistent,
        )
        temp_mask_model.ensure_label(label, color, visible=True)
        mask_to_apply = rect_mask
        if slice_data is not None and threshold is not None:
            mask_to_apply = self.build_thresholded_mask(rect_mask, slice_data, threshold)

        self.apply_temp_rectangle(
            temp_mask_model, slice_idx, mask_to_apply, label, persistent=persistent
        )
        return mask_to_apply

    @staticmethod
    def _normalize_rect_input(rect: Any) -> Optional[Tuple[int, int, int, int]]:
        """Convert various rectangle formats to (x1, y1, x2, y2)."""
        if rect is None:
            return None
        if isinstance(rect, (list, tuple)):
            if len(rect) == 4 and all(isinstance(v, (int, float)) for v in rect):
                x1, y1, x2, y2 = rect
                return (int(x1), int(y1), int(x2), int(y2))
            if len(rect) == 2 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in rect):
                (x1, y1), (x2, y2) = rect
                return (int(x1), int(y1), int(x2), int(y2))
        return None
