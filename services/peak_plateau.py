from __future__ import annotations

from typing import Optional

import numpy as np


def midpoint_index(start: int | np.ndarray, end: int | np.ndarray) -> int | np.ndarray:
    """Return the discrete midpoint between two non-negative sample indices."""
    start_arr = np.asarray(start, dtype=np.int64)
    end_arr = np.asarray(end, dtype=np.int64)
    mid = (start_arr + end_arr + 1) // 2
    if np.isscalar(start) and np.isscalar(end):
        return int(mid)
    return mid.astype(np.int32, copy=False)


def pick_plateau_peak_index(candidates: np.ndarray, values: np.ndarray) -> Optional[int]:
    """Pick the midpoint of the max-value plateau among candidate sample indices."""
    candidate_arr = np.asarray(candidates, dtype=np.int32).reshape(-1)
    value_arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if candidate_arr.size == 0 or value_arr.size == 0 or candidate_arr.size != value_arr.size:
        return None

    finite_mask = np.isfinite(value_arr)
    if not np.any(finite_mask):
        return None

    valid_candidates = candidate_arr[finite_mask]
    valid_values = value_arr[finite_mask]
    max_value = float(np.max(valid_values))
    plateau_candidates = valid_candidates[valid_values == max_value]
    if plateau_candidates.size == 0:
        return None

    return int(midpoint_index(int(plateau_candidates[0]), int(plateau_candidates[-1])))


def peak_indices_from_masked_max(slice_values: np.ndarray, slice_mask: np.ndarray, class_id: int) -> np.ndarray:
    """Return one peak index per column, using the midpoint of any saturated plateau."""
    values = np.asarray(slice_values, dtype=np.float32)
    mask = np.asarray(slice_mask)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError(
            f"Peak extraction attendu sur une slice 2D identique, recu {values.shape} et {mask.shape}"
        )

    height, width = values.shape
    peaks = np.full(width, -1, dtype=np.int32)
    class_mask = (mask == class_id) & np.isfinite(values)
    if not np.any(class_mask):
        return peaks

    masked_values = np.where(class_mask, values, -np.inf)
    max_values = np.max(masked_values, axis=0)
    valid = np.isfinite(max_values)
    if not np.any(valid):
        return peaks

    peak_hits = valid[None, :] & (masked_values == max_values[None, :])
    first = np.argmax(peak_hits, axis=0).astype(np.int32, copy=False)
    last = (height - 1 - np.argmax(peak_hits[::-1], axis=0)).astype(np.int32, copy=False)
    peaks[valid] = midpoint_index(first[valid], last[valid])
    return peaks
