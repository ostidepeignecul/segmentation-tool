"""Service métier pour les opérations d'annotation/ROI (stubs)."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

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

    # ------------------------------------------------------------------ #
    # Grow helpers
    # ------------------------------------------------------------------ #
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
    ) -> Optional[np.ndarray]:
        """Region growing (4-connexe) depuis un seed si la valeur dépasse le threshold."""
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
        thr = float(threshold) if threshold is not None else 0.0
        if norm[y, x] < thr:
            return None

        mask = np.zeros((h, w), dtype=np.uint8)
        q: deque[Tuple[int, int]] = deque()
        q.append((x, y))
        mask[y, x] = 1
        while q:
            cx, cy = q.popleft()
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                if mask[ny, nx]:
                    continue
                if norm[ny, nx] < thr:
                    continue
                mask[ny, nx] = 1
                q.append((nx, ny))
        if not np.any(mask):
            return None
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
    ) -> Optional[np.ndarray]:
        """
        Region growing ROI: build mask from seed/threshold, store ROI and apply to temp model.
        """
        grow_mask = self.build_grow_mask(shape, point, slice_data, threshold)
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

        Returns the list of box tuples (x1, y1, x2, y2) for display.
        """
        boxes: list[Tuple[int, int, int, int]] = []
        if clear_slice:
            temp_mask_model.clear_slice(slice_idx)

        for roi in rois:
            if getattr(roi, "roi_type", None) != "box" or len(getattr(roi, "points", [])) < 2:
                continue
            box_tuple = roi.points[0] + roi.points[1]
            box_mask = self.build_box_mask(shape, box_tuple)
            if box_mask is None:
                continue
            label = getattr(roi, "label", 1)
            color = (palette or {}).get(label) or MASK_COLORS_BGRA.get(label, (255, 0, 255, 160))
            temp_mask_model.ensure_label(label, color, visible=True)
            mask_to_apply = box_mask
            roi_threshold = getattr(roi, "threshold", None)
            if slice_data is not None and roi_threshold is not None:
                mask_to_apply = self.build_thresholded_mask(box_mask, slice_data, roi_threshold)
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
                shape, seed, slice_data, getattr(roi, "threshold", None)
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
        return boxes

    def rebuild_volume_preview_from_rois(
        self,
        *,
        depth: int,
        mask_shape: Tuple[int, int],
        roi_model: RoiModel,
        temp_mask_model: TempMaskModel,
        palette: Optional[dict[int, tuple[int, int, int, int]]] = None,
        slice_data_provider: Callable[[int], Optional[np.ndarray]],
    ) -> None:
        """
        Rebuild temporary masks for the whole volume from stored ROIs.
        """
        if temp_mask_model.get_mask_volume() is None:
            temp_mask_model.initialize((depth, mask_shape[0], mask_shape[1]))
        else:
            temp_mask_model.clear()

        for idx in range(int(depth)):
            rois = list(roi_model.list_on_slice(idx))
            persistent_rois = roi_model.list_persistent()
            if persistent_rois:
                seen_ids = {roi.id for roi in rois}
                rois.extend([roi for roi in persistent_rois if roi.id not in seen_ids])
            if not rois:
                continue
            slice_data = slice_data_provider(idx)
            self.rebuild_temp_masks_for_slice(
                rois=rois,
                shape=mask_shape,
                slice_idx=idx,
                temp_mask_model=temp_mask_model,
                palette=palette,
                clear_slice=True,
                slice_data=slice_data,
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
        for idx in range(depth):
            temp_slice = temp_volume[idx]
            if temp_slice is None or not np.any(temp_slice):
                continue
            current_slice = mask_volume[idx]
            if current_slice.shape != temp_slice.shape:
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

        def apply_once(idx: int) -> bool:
            slice_data = slice_data_provider(idx)
            if slice_data is None:
                return False
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
            )
            return mask is not None

        if not apply_once(start_slice):
            return

        for idx in range(start_slice + 1, int(depth)):
            if not apply_once(idx):
                break

        for idx in range(start_slice - 1, -1, -1):
            if not apply_once(idx):
                break
