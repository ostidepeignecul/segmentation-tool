from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from config.constants import MASK_COLORS_BGRA


class TempMaskModel:
    """
    Stores a temporary mask volume (preview before committing to the overlay).
    Similar structure to AnnotationModel but kept separate to avoid side effects.
    """

    def __init__(self) -> None:
        self.mask_volume: Optional[np.ndarray] = None
        self.label_palette: Dict[int, Tuple[int, int, int, int]] = {}
        self.label_visibility: Dict[int, bool] = {}
        self.coverage_volume: Optional[np.ndarray] = None

    def initialize(self, shape: Any) -> None:
        """Prepare an empty mask volume with the given shape."""
        prev_palette = dict(self.label_palette)
        prev_visibility = dict(self.label_visibility)
        try:
            depth, height, width = shape  # type: ignore[misc]
        except Exception:
            self.mask_volume = None
            return
        self.mask_volume = np.zeros((int(depth), int(height), int(width)), dtype=np.uint8)
        self.coverage_volume = np.zeros((int(depth), int(height), int(width)), dtype=bool)
        self.label_palette = prev_palette
        self.label_visibility = prev_visibility

    def clear(self) -> None:
        """Reset masks to zero; keep label palette/visibility."""
        if self.mask_volume is not None:
            self.mask_volume[:] = 0
        if self.coverage_volume is not None:
            self.coverage_volume[:] = False

    def ensure_label(self, label_id: int, color: Tuple[int, int, int, int], visible: bool = True) -> None:
        """Ensure a label exists in the palette/visibility maps."""
        key = int(label_id)
        self.label_palette.setdefault(key, tuple(int(c) for c in color))
        self.label_visibility.setdefault(key, bool(visible))

    def set_label_color(self, label_id: int, color: Tuple[int, int, int, int]) -> None:
        """Set/update label color."""
        key = int(label_id)
        self.label_palette[key] = tuple(int(c) for c in color)

    def set_slice_mask(self, slice_idx: int, mask: Any, label: int, *, persistent: bool = False) -> None:
        """Set/merge the mask for a specific slice."""
        if self.mask_volume is None:
            return
        slice_idx = max(0, min(self.mask_volume.shape[0] - 1, int(slice_idx)))
        mask_array = np.asarray(mask, dtype=np.uint8)
        if mask_array.shape != self.mask_volume[slice_idx].shape:
            return

        if self.coverage_volume is None:
            self.coverage_volume = np.zeros_like(self.mask_volume, dtype=bool)

        coverage_slice = self.coverage_volume[slice_idx]
        coverage = mask_array > 0
        if persistent:
            updated = np.where(mask_array > 0, int(label), self.mask_volume[slice_idx])
        else:
            # Replace only where mask > 0
            updated = self.mask_volume[slice_idx].copy()
            updated[coverage] = int(label)
        self.mask_volume[slice_idx] = updated
        self.coverage_volume[slice_idx] = np.logical_or(coverage_slice, coverage)

    def clear_slice(self, slice_idx: int) -> None:
        """Clear a slice in the temporary mask volume."""
        if self.mask_volume is None:
            return
        slice_idx = max(0, min(self.mask_volume.shape[0] - 1, int(slice_idx)))
        self.mask_volume[slice_idx] = 0
        if self.coverage_volume is not None:
            self.coverage_volume[slice_idx] = False

    def clear_slice_label(self, slice_idx: int, label: int) -> None:
        """Clear only a specific label from a slice."""
        if self.mask_volume is None:
            return
        slice_idx = max(0, min(self.mask_volume.shape[0] - 1, int(slice_idx)))
        lbl = int(label)
        slice_mask = self.mask_volume[slice_idx]
        slice_mask[slice_mask == lbl] = 0
        self.mask_volume[slice_idx] = slice_mask

    def get_slice_mask(self, slice_idx: int) -> Optional[np.ndarray]:
        """Return a slice mask (preview) if available."""
        if self.mask_volume is None:
            return None
        slice_idx = max(0, min(self.mask_volume.shape[0] - 1, int(slice_idx)))
        return self.mask_volume[slice_idx]

    def get_slice_coverage(self, slice_idx: int) -> Optional[np.ndarray]:
        """Return coverage (where temp mask should apply) for a slice."""
        if self.coverage_volume is None:
            return None
        slice_idx = max(0, min(self.coverage_volume.shape[0] - 1, int(slice_idx)))
        return self.coverage_volume[slice_idx]

    def get_mask_volume(self) -> Optional[np.ndarray]:
        return self.mask_volume

    def get_coverage_volume(self) -> Optional[np.ndarray]:
        return self.coverage_volume

    def mask_shape_hw(self) -> Optional[tuple[int, int]]:
        if self.mask_volume is None:
            return None
        try:
            return (self.mask_volume.shape[1], self.mask_volume.shape[2])
        except Exception:
            return None
