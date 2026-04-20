from __future__ import annotations

from typing import Optional

import numpy as np

from config.constants import normalize_corrosion_peak_selection_mode


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


def _contiguous_runs(mask_column: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive [start, end] runs for a 1D boolean mask column."""
    true_indices = np.flatnonzero(mask_column)
    if true_indices.size == 0:
        return []

    gap_positions = np.flatnonzero(np.diff(true_indices) > 1)
    starts = np.concatenate((true_indices[:1], true_indices[gap_positions + 1]))
    ends = np.concatenate((true_indices[gap_positions], true_indices[-1:]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _pick_peak_index_in_run(values_column: np.ndarray, start: int, end: int) -> Optional[int]:
    candidates = np.arange(int(start), int(end) + 1, dtype=np.int32)
    if candidates.size == 0:
        return None
    return pick_plateau_peak_index(candidates, values_column[candidates])


def _select_run_for_reference(
    values_column: np.ndarray,
    runs: list[tuple[int, int]],
    reference_peak: int,
    selection_mode: str,
) -> int:
    best_idx = 0
    best_key: Optional[tuple[float, ...]] = None
    ref = float(reference_peak)

    for idx, (start, end) in enumerate(runs):
        centroid = (float(start) + float(end)) / 2.0
        centroid_distance = abs(centroid - ref)
        peak_idx = _pick_peak_index_in_run(values_column, start, end)
        if peak_idx is None:
            peak_idx = int(start)
        peak_distance = abs(float(peak_idx) - ref)
        local_max = float(np.max(values_column[start : end + 1]))

        if selection_mode == "pessimistic":
            key = (centroid_distance, peak_distance, -local_max, float(start))
        else:
            key = (-centroid_distance, -peak_distance, -local_max, float(start))

        if best_key is None or key < best_key:
            best_key = key
            best_idx = idx

    return best_idx


def _resolve_peak_indices_from_reference(
    slice_values: np.ndarray,
    label_mask: np.ndarray,
    fallback_peaks: np.ndarray,
    reference_peaks: np.ndarray,
    selection_mode: str,
) -> np.ndarray:
    peaks = np.array(fallback_peaks, dtype=np.int32, copy=True)
    if selection_mode == "max_peak":
        return peaks

    _, width = label_mask.shape
    for x in range(width):
        if peaks[x] < 0:
            continue
        reference_peak = int(reference_peaks[x])
        if reference_peak < 0:
            continue

        runs = _contiguous_runs(label_mask[:, x])
        if len(runs) <= 1:
            continue

        run_idx = _select_run_for_reference(
            slice_values[:, x],
            runs,
            reference_peak=reference_peak,
            selection_mode=selection_mode,
        )
        start, end = runs[run_idx]
        peak_idx = _pick_peak_index_in_run(slice_values[:, x], start, end)
        if peak_idx is not None:
            peaks[x] = int(peak_idx)

    return peaks


def peak_indices_from_masked_pair(
    slice_values: np.ndarray,
    slice_mask: np.ndarray,
    class_a: int,
    class_b: int,
    *,
    selection_mode_a: Optional[str] = None,
    selection_mode_b: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return one peak index per column for a label pair.

    `max_peak` keeps the historical behavior. `optimistic` and `pessimistic`
    only change columns where a label has multiple disconnected Y bands, using
    the other label's raw `max_peak` as a stable reference to avoid circular
    dependencies between both labels. Each label can use its own selection mode.
    """
    values = np.asarray(slice_values, dtype=np.float32)
    mask = np.asarray(slice_mask)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError(
            f"Peak extraction attendu sur une slice 2D identique, recu {values.shape} et {mask.shape}"
        )

    normalized_mode_a = normalize_corrosion_peak_selection_mode(selection_mode_a)
    normalized_mode_b = normalize_corrosion_peak_selection_mode(selection_mode_b)
    base_peaks_a = peak_indices_from_masked_max(values, mask, class_a)
    base_peaks_b = peak_indices_from_masked_max(values, mask, class_b)
    if normalized_mode_a == "max_peak" and normalized_mode_b == "max_peak":
        return base_peaks_a, base_peaks_b

    finite_mask = np.isfinite(values)
    label_mask_a = (mask == class_a) & finite_mask
    label_mask_b = (mask == class_b) & finite_mask

    if normalized_mode_a == "max_peak":
        resolved_a = np.array(base_peaks_a, dtype=np.int32, copy=True)
    else:
        resolved_a = _resolve_peak_indices_from_reference(
            values,
            label_mask_a,
            fallback_peaks=base_peaks_a,
            reference_peaks=base_peaks_b,
            selection_mode=normalized_mode_a,
        )

    if normalized_mode_b == "max_peak":
        resolved_b = np.array(base_peaks_b, dtype=np.int32, copy=True)
    else:
        resolved_b = _resolve_peak_indices_from_reference(
            values,
            label_mask_b,
            fallback_peaks=base_peaks_b,
            reference_peaks=base_peaks_a,
            selection_mode=normalized_mode_b,
        )
    return resolved_a, resolved_b
