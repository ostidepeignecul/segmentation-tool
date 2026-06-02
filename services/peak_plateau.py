from __future__ import annotations

from typing import Optional

import numpy as np

from config.constants import normalize_corrosion_analysis_mode, normalize_corrosion_peak_selection_mode


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


def _distance_from_reference_to_run(start: int, end: int, reference_peak: int) -> float:
    """Return the shortest Y distance between one run interval and the reference peak."""
    ref = float(reference_peak)
    run_start = float(start)
    run_end = float(end)
    if ref < run_start:
        return run_start - ref
    if ref > run_end:
        return ref - run_end
    return 0.0


def _select_a_candidate_by_position(
    candidate_positions: np.ndarray,
    candidate_values: np.ndarray,
    reference_peak: Optional[int],
) -> int:
    """Pick the deepest A candidate, preferably still above the B reference."""
    candidates = np.asarray(candidate_positions, dtype=np.int32).reshape(-1)
    values = np.asarray(candidate_values, dtype=np.float32).reshape(-1)
    reference = None if reference_peak is None or int(reference_peak) < 0 else int(reference_peak)

    best_idx = 0
    best_key: Optional[tuple[float, ...]] = None
    for idx, peak_y in enumerate(candidates):
        y = int(peak_y)
        amp = float(values[idx]) if idx < values.size else float("-inf")
        if reference is not None and y < reference:
            # Best valid A is the lowest peak that still remains above B.
            key = (0.0, -float(y), -amp)
        elif reference is not None:
            key = (1.0, abs(float(y) - float(reference)), float(y), -amp)
        else:
            key = (2.0, -float(y), -amp)

        if best_key is None or key < best_key:
            best_key = key
            best_idx = int(idx)
    return best_idx


def _select_b_candidate_by_position(
    candidate_positions: np.ndarray,
    candidate_values: np.ndarray,
    reference_peak: Optional[int],
) -> int:
    """Pick the highest B candidate, preferably still below the A reference."""
    candidates = np.asarray(candidate_positions, dtype=np.int32).reshape(-1)
    values = np.asarray(candidate_values, dtype=np.float32).reshape(-1)
    reference = None if reference_peak is None or int(reference_peak) < 0 else int(reference_peak)

    best_idx = 0
    best_key: Optional[tuple[float, ...]] = None
    for idx, peak_y in enumerate(candidates):
        y = int(peak_y)
        amp = float(values[idx]) if idx < values.size else float("-inf")
        if reference is not None and y > reference:
            # Best valid B is the highest peak that still remains below A.
            key = (0.0, float(y), -amp)
        elif reference is not None:
            key = (1.0, abs(float(y) - float(reference)), -float(y), -amp)
        else:
            key = (2.0, float(y), -amp)

        if best_key is None or key < best_key:
            best_key = key
            best_idx = int(idx)
    return best_idx


def _top_local_peak_candidates_in_mask(
    values_column: np.ndarray,
    mask_column: np.ndarray,
    *,
    limit: int = 2,
) -> np.ndarray:
    sample_indices = np.flatnonzero(np.asarray(mask_column, dtype=bool))
    if sample_indices.size == 0:
        return np.empty((0,), dtype=np.int32)

    sample_values = np.asarray(values_column[sample_indices], dtype=np.float32)
    finite_mask = np.isfinite(sample_values)
    if not np.any(finite_mask):
        return np.empty((0,), dtype=np.int32)

    sample_indices = sample_indices[finite_mask].astype(np.int32, copy=False)
    sample_values = sample_values[finite_mask]
    sample_count = int(sample_indices.size)
    if sample_count == 0:
        return np.empty((0,), dtype=np.int32)
    if sample_count == 1:
        return sample_indices[:1]

    smoothed = np.array(sample_values, dtype=np.float32, copy=True)
    if sample_count > 2:
        smoothed[1:-1] = (
            sample_values[:-2] + sample_values[1:-1] + sample_values[2:]
        ) / 3.0

    peak_flags = np.zeros(sample_count, dtype=bool)
    if float(smoothed[0]) > float(smoothed[1]):
        peak_flags[0] = True
    if sample_count > 2:
        mid = smoothed[1:-1]
        left = smoothed[:-2]
        right = smoothed[2:]
        peak_flags[1:-1] = ((mid >= left) & (mid > right)) | ((mid > left) & (mid >= right))
    if float(smoothed[-1]) > float(smoothed[-2]):
        peak_flags[-1] = True

    peak_positions = np.flatnonzero(peak_flags)
    if peak_positions.size == 0:
        peak_positions = np.asarray([int(np.argmax(smoothed))], dtype=np.int32)

    candidate_indices = sample_indices[peak_positions]
    candidate_values = sample_values[peak_positions]
    order = np.lexsort((candidate_indices, -candidate_values))
    ranked = candidate_indices[order]
    return ranked[: max(1, int(limit))]


def _resolve_ac_ab_peak_indices(
    slice_values: np.ndarray,
    label_mask: np.ndarray,
    fallback_peaks: np.ndarray,
    reference_peaks: np.ndarray,
) -> np.ndarray:
    peaks = np.array(fallback_peaks, dtype=np.int32, copy=True)
    candidate_columns = np.flatnonzero(fallback_peaks >= 0)
    if candidate_columns.size == 0:
        return peaks

    for x in candidate_columns:
        local_candidates = _top_local_peak_candidates_in_mask(
            slice_values[:, x],
            label_mask[:, x],
            limit=2,
        )
        if local_candidates.size < 2:
            continue

        candidate_values = np.asarray(slice_values[local_candidates, x], dtype=np.float32)
        reference_peak = int(reference_peaks[x]) if int(reference_peaks[x]) >= 0 else None
        chosen_idx = _select_a_candidate_by_position(
            local_candidates,
            candidate_values,
            reference_peak,
        )
        peaks[x] = int(local_candidates[chosen_idx])

    return peaks


def _select_run_for_reference(
    values_column: np.ndarray,
    runs: list[tuple[int, int]],
    reference_peak: Optional[int],
    selection_mode: str,
    resolved_label: str,
) -> int:
    best_idx = 0
    best_key: Optional[tuple[float, ...]] = None
    reference = None if reference_peak is None or int(reference_peak) < 0 else int(reference_peak)

    for idx, (start, end) in enumerate(runs):
        peak_idx = _pick_peak_index_in_run(values_column, start, end)
        if peak_idx is None:
            peak_idx = int(start)
        local_max = float(np.max(values_column[start : end + 1]))

        if selection_mode == "pessimistic":
            if str(resolved_label).strip().casefold() == "b":
                chosen_pos = float(start)
                if reference is not None and start > reference:
                    # For B, keep the highest band that still remains below A.
                    key = (0.0, chosen_pos, -local_max, float(start))
                elif reference is not None:
                    run_distance = _distance_from_reference_to_run(start, end, reference)
                    key = (1.0, run_distance, chosen_pos, -local_max, float(start))
                else:
                    key = (2.0, chosen_pos, -local_max, float(start))
            else:
                chosen_pos = float(end)
                if reference is not None and end < reference:
                    # For A, keep the lowest band that still remains above B.
                    key = (0.0, -chosen_pos, -local_max, float(start))
                elif reference is not None:
                    run_distance = _distance_from_reference_to_run(start, end, reference)
                    key = (1.0, run_distance, -chosen_pos, -local_max, float(start))
                else:
                    key = (2.0, -chosen_pos, -local_max, float(start))
        else:
            ref = float(reference if reference is not None else peak_idx)
            centroid = (float(start) + float(end)) / 2.0
            centroid_distance = abs(centroid - ref)
            peak_distance = abs(float(peak_idx) - ref)
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
    resolved_label: str,
) -> np.ndarray:
    peaks = np.array(fallback_peaks, dtype=np.int32, copy=True)
    if selection_mode == "max_peak":
        return peaks

    run_starts = label_mask[0].astype(np.int32, copy=False)
    if label_mask.shape[0] > 1:
        run_starts = run_starts + np.sum(label_mask[1:] & ~label_mask[:-1], axis=0, dtype=np.int32)
    if selection_mode == "pessimistic":
        ambiguous_columns = np.flatnonzero((peaks >= 0) & (run_starts > 1))
    else:
        ambiguous_columns = np.flatnonzero((peaks >= 0) & (reference_peaks >= 0) & (run_starts > 1))
    if ambiguous_columns.size == 0:
        return peaks

    for x in ambiguous_columns:
        reference_peak = int(reference_peaks[x]) if int(reference_peaks[x]) >= 0 else None

        runs = _contiguous_runs(label_mask[:, x])

        run_idx = _select_run_for_reference(
            slice_values[:, x],
            runs,
            reference_peak=reference_peak,
            selection_mode=selection_mode,
            resolved_label=resolved_label,
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
    analysis_mode: Optional[str] = None,
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

    normalized_analysis_mode = normalize_corrosion_analysis_mode(analysis_mode)
    normalized_mode_a = normalize_corrosion_peak_selection_mode(selection_mode_a)
    normalized_mode_b = normalize_corrosion_peak_selection_mode(selection_mode_b)
    base_peaks_b = peak_indices_from_masked_max(values, mask, class_b)
    base_peaks_a = peak_indices_from_masked_max(values, mask, class_a)

    finite_mask = np.isfinite(values)
    label_mask_a = (mask == class_a) & finite_mask
    label_mask_b = (mask == class_b) & finite_mask

    if normalized_analysis_mode == "ac_ab":
        base_peaks_a = _resolve_ac_ab_peak_indices(
            values,
            label_mask_a,
            fallback_peaks=base_peaks_a,
            reference_peaks=base_peaks_b,
        )

    if normalized_mode_a == "max_peak" and normalized_mode_b == "max_peak":
        return base_peaks_a, base_peaks_b

    if normalized_mode_a == "max_peak":
        resolved_a = np.array(base_peaks_a, dtype=np.int32, copy=True)
    else:
        resolved_a = _resolve_peak_indices_from_reference(
            values,
            label_mask_a,
            fallback_peaks=base_peaks_a,
            reference_peaks=base_peaks_b,
            selection_mode=normalized_mode_a,
            resolved_label="a",
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
            resolved_label="b",
        )
    return resolved_a, resolved_b
