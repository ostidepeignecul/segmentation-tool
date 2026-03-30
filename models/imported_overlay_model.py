from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ImportedOverlayModel:
    """Runtime-only source snapshot for a non-destructive imported NPZ overlay."""

    source_path: Optional[str] = None
    original_mask_volume: Optional[np.ndarray] = None
    original_label_ids: tuple[int, ...] = ()
    current_mapping: dict[int, int] = field(default_factory=dict)
    applied_mask_volume: Optional[np.ndarray] = None

    def clear(self) -> None:
        """Drop the imported overlay source and its last applied state."""
        self.source_path = None
        self.original_mask_volume = None
        self.original_label_ids = ()
        self.current_mapping = {}
        self.applied_mask_volume = None

    def has_source(self) -> bool:
        """Return True when an imported overlay source is available."""
        return self.original_mask_volume is not None and bool(self.original_label_ids)

    def set_imported_overlay(
        self,
        *,
        source_path: str,
        original_mask_volume: np.ndarray,
        original_label_ids: tuple[int, ...],
    ) -> None:
        """Store the original imported overlay and reset the remap state."""
        normalized_ids = tuple(int(label_id) for label_id in original_label_ids)
        self.source_path = str(source_path)
        self.original_mask_volume = np.asarray(original_mask_volume, dtype=np.uint8).copy()
        self.original_label_ids = normalized_ids
        self.current_mapping = {int(label_id): int(label_id) for label_id in normalized_ids}
        self.applied_mask_volume = self.original_mask_volume.copy()

    def update_after_remap(
        self,
        *,
        mapping: dict[int, int],
        applied_mask_volume: np.ndarray,
    ) -> None:
        """Update the current remap rules and the last applied mask snapshot."""
        self.current_mapping = {int(source): int(target) for source, target in mapping.items()}
        self.applied_mask_volume = np.asarray(applied_mask_volume, dtype=np.uint8).copy()

    def current_mask_matches(self, mask_volume: Optional[np.ndarray]) -> bool:
        """Return True when the current annotation mask matches the last applied import/remap."""
        if mask_volume is None or self.applied_mask_volume is None:
            return False
        candidate = np.asarray(mask_volume, dtype=np.uint8)
        return candidate.shape == self.applied_mask_volume.shape and np.array_equal(
            candidate,
            self.applied_mask_volume,
        )
