from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from config.constants import MASK_COLORS_BGRA

FallbackColor = (255, 0, 255, 160)  # Magenta semi-transparent


class AnnotationModel:
    """Stores masks, label palette and visibility (no rendering logic)."""

    def __init__(self) -> None:
        self.mask_volume: Optional[np.ndarray] = None
        self.label_palette: Dict[int, Tuple[int, int, int, int]] = {}
        self.label_visibility: Dict[int, bool] = {}

    # ------------------------------------------------------------------ #
    # Mask handling
    # ------------------------------------------------------------------ #
    def initialize(self, shape: Any) -> None:
        """Prepare an empty mask volume with the given shape."""
        try:
            depth, height, width = shape  # type: ignore[misc]
        except Exception:
            self.mask_volume = None
            return
        self.mask_volume = np.zeros((int(depth), int(height), int(width)), dtype=np.uint8)

    def set_mask_volume(self, mask_volume: Any) -> None:
        """Assign a full mask volume (uint8, shape (Z,H,W))."""
        arr = np.asarray(mask_volume, dtype=np.uint8)
        if arr.ndim != 3:
            raise ValueError("Mask volume must be 3D (Z,H,W).")
        self.mask_volume = arr
        # Reset palette/visibilité et auto-register labels présents
        self.label_palette = {}
        self.label_visibility = {}
        for cls_value in np.unique(arr):
            cls_int = int(cls_value)
            if cls_int == 0:
                continue
            self.label_palette.setdefault(cls_int, FallbackColor)
            self.label_visibility.setdefault(cls_int, True)

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
        """Reset masks and label state."""
        self.mask_volume = None
        self.label_palette = {}
        self.label_visibility = {}

    # ------------------------------------------------------------------ #
    # Labels (palette + visibility)
    # ------------------------------------------------------------------ #
    def ensure_label(self, label_id: int, color: Tuple[int, int, int, int], visible: bool = True) -> None:
        """Create the label entry if missing and set initial visibility."""
        key = int(label_id)
        self.label_palette.setdefault(key, tuple(int(c) for c in color))
        self.label_visibility.setdefault(key, bool(visible))

    def set_label_color(self, label_id: int, color: Tuple[int, int, int, int]) -> None:
        """Assign or update a label color (BGRA)."""
        self.label_palette[int(label_id)] = tuple(int(c) for c in color)

    def set_label_visibility(self, label_id: int, visible: bool) -> None:
        """Toggle visibility for a given label id."""
        self.label_visibility[int(label_id)] = bool(visible)

    def visible_labels(self) -> Optional[set[int]]:
        """Return the set of labels currently marked visible (None = all)."""
        if not self.label_visibility:
            return None
        if all(self.label_visibility.values()):
            return None
        return {lbl for lbl, vis in self.label_visibility.items() if vis}

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    def get_mask_volume(self) -> Optional[np.ndarray]:
        """Expose the current mask volume."""
        return self.mask_volume

    def get_visible_labels(self) -> Optional[set[int]]:
        """Return the set of labels currently marked visible (None = all)."""
        return self.visible_labels()

    def get_label_palette(self) -> Dict[int, Tuple[int, int, int, int]]:
        """Return the current label palette (BGRA), with defaults if empty."""
        return self.label_palette or dict(MASK_COLORS_BGRA)
