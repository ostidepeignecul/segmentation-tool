from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class AppliedAnnotationHistoryEntry:
    """Store the previous and applied content of slices changed by one apply action."""

    previous_slices: Dict[int, np.ndarray]
    applied_slices: Dict[int, np.ndarray]

    @property
    def slice_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.previous_slices.keys()))


class AppliedAnnotationHistoryModel:
    """Keep a bounded undo stack for committed annotation writes."""

    def __init__(self, *, max_entries: int = 20) -> None:
        maxlen = max(1, int(max_entries))
        self._undo_entries: Deque[AppliedAnnotationHistoryEntry] = deque(maxlen=maxlen)
        self._redo_entries: Deque[AppliedAnnotationHistoryEntry] = deque(maxlen=maxlen)

    def clear(self) -> None:
        self._undo_entries.clear()
        self._redo_entries.clear()

    def can_undo(self) -> bool:
        return bool(self._undo_entries)

    def can_redo(self) -> bool:
        return bool(self._redo_entries)

    def push(self, *, previous_slices: Mapping[int, Any], applied_slices: Mapping[int, Any]) -> bool:
        previous = self._normalize_slices(previous_slices)
        applied = self._normalize_slices(applied_slices)
        if not previous or not applied or set(previous.keys()) != set(applied.keys()):
            return False
        self._undo_entries.append(
            AppliedAnnotationHistoryEntry(
                previous_slices=previous,
                applied_slices=applied,
            )
        )
        self._redo_entries.clear()
        return True

    def pop_undo(self) -> Optional[AppliedAnnotationHistoryEntry]:
        if not self._undo_entries:
            return None
        entry = self._undo_entries.pop()
        self._redo_entries.append(self._clone_entry(entry))
        return self._clone_entry(entry)

    def pop_redo(self) -> Optional[AppliedAnnotationHistoryEntry]:
        if not self._redo_entries:
            return None
        entry = self._redo_entries.pop()
        self._undo_entries.append(self._clone_entry(entry))
        return self._clone_entry(entry)

    @staticmethod
    def _normalize_slices(slices: Mapping[int, Any]) -> Dict[int, np.ndarray]:
        normalized: Dict[int, np.ndarray] = {}
        for slice_idx, slice_mask in dict(slices).items():
            try:
                idx = int(slice_idx)
                arr = np.asarray(slice_mask, dtype=np.uint8)
            except Exception:
                continue
            if arr.ndim != 2:
                continue
            normalized[idx] = np.array(arr, copy=True)
        return normalized

    @staticmethod
    def _clone_entry(entry: AppliedAnnotationHistoryEntry) -> AppliedAnnotationHistoryEntry:
        return AppliedAnnotationHistoryEntry(
            previous_slices={
                int(slice_idx): np.array(slice_mask, copy=True)
                for slice_idx, slice_mask in entry.previous_slices.items()
            },
            applied_slices={
                int(slice_idx): np.array(slice_mask, copy=True)
                for slice_idx, slice_mask in entry.applied_slices.items()
            },
        )
