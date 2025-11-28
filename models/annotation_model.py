from typing import Any, Optional

import numpy as np


class AnnotationModel:
    """Holds the 3D annotation mask aligned with the NDE volume."""

    def __init__(self) -> None:
        self.mask_volume: Optional[np.ndarray] = None
        self.class_labels: list[int] = []

    def initialize(self, shape: Any) -> None:
        """Prepare an empty mask volume with the given shape."""
        try:
            depth, height, width = shape  # type: ignore[misc]
        except Exception:
            self.mask_volume = None
            return
        self.mask_volume = np.zeros((int(depth), int(height), int(width)), dtype=np.uint8)

    def set_slice_mask(self, slice_idx: int, mask: Any) -> None:
        """Set or replace the mask for a specific slice."""
        if self.mask_volume is None:
            return
        slice_idx = max(0, min(self.mask_volume.shape[0] - 1, int(slice_idx)))
        mask_array = np.asarray(mask, dtype=np.uint8)
        if mask_array.shape != self.mask_volume[slice_idx].shape:
            return
        self.mask_volume[slice_idx] = mask_array

    def clear(self) -> None:
        """Reset all stored annotations."""
        self.mask_volume = None
        self.class_labels = []

    def apply_draw_operation(self, *args: Any, **kwargs: Any) -> None:
        """Apply a drawing operation placeholder."""
        # Drawing logic is UI/Controller-driven; model stays as a data container.
        return
