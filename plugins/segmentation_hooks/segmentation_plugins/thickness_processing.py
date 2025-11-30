from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ------------- configurable thresholds for thickness processing -------------
FRONT_INTERPOLATION_MAX_GAP = 0  # allow filling single-column gaps for frontwall
BACK_INTERPOLATION_MAX_GAP = 0   # allow filling single-column gaps for backwall
MAX_BACK_DISTANCE_FROM_FRONT = 60  # max allowed gap between front/back picks; set <0 to disable
NEIGHBOR_DISTANCE_RELAX_RADIUS = 1  # columns (±) allowed to rescue distant backwall
NEIGHBOR_DISTANCE_RELAX_TOLERANCE = 10  # additional pixels allowed when neighbor is within limit (-1 disables)


def extract_ascan_peak_positions(
    mask: np.ndarray,
    volume_data: np.ndarray,
    labels: dict[str, int],
) -> dict[str, dict[tuple[int, int], int]]:
    """
    Extrait les positions Y des pics A-scan en favorisant la cohérence géométrique
    entre frontwall et backwall. Les classes restantes (ex. flaw) utilisent encore
    l'amplitude maximum.
    """
    ascan_peaks: dict[str, dict[tuple[int, int], int]] = {}

    frontwall_value = labels.get("frontwall", 1)
    backwall_value = labels.get("backwall", 2)
    z_len, x_len, _ = mask.shape

    front_candidates: dict[tuple[int, int], np.ndarray] = {}
    back_candidates: dict[tuple[int, int], np.ndarray] = {}
    other_class_peaks: dict[str, dict[tuple[int, int], int]] = {}

    for class_name, class_value in labels.items():
        if class_name == "background" or class_value == 0:
            continue

        logger.info(">>> Processing class '%s' (value=%s)", class_name, class_value)

        is_frontwall = class_value == frontwall_value
        is_backwall = class_value == backwall_value

        class_peaks: dict[tuple[int, int], int] = {}

        for z in range(z_len):
            slice_mask = mask[z] == class_value
            if not slice_mask.any():
                continue

            for x in range(x_len):
                y_positions = np.where(slice_mask[x, :])[0]
                if y_positions.size == 0:
                    continue

                if is_frontwall:
                    front_candidates[(z, x)] = y_positions
                elif is_backwall:
                    back_candidates[(z, x)] = y_positions
                else:
                    amplitudes = volume_data[z, x, y_positions]
                    sel_idx = int(np.argmax(amplitudes))
                    class_peaks[(z, x)] = int(y_positions[sel_idx])

        if not is_frontwall and not is_backwall:
            other_class_peaks[class_name] = class_peaks
            logger.info(
                "    Extracted %d A-scan peaks for class '%s' (value=%s)",
                len(class_peaks),
                class_name,
                class_value,
            )

    def _select_front_candidate(candidates: np.ndarray) -> int:
        if candidates.size == 0:
            return -1
        return int(candidates.max())

    def _select_back_candidate(front_idx: int, candidates: np.ndarray) -> int:
        if candidates.size == 0:
            return -1
        sorted_candidates = np.sort(candidates)
        if front_idx >= 0:
            forward = sorted_candidates[sorted_candidates > front_idx]
            if forward.size:
                return int(forward[0])
        return int(sorted_candidates.min())

    front_map = np.full((z_len, x_len), -1, dtype=np.int16)
    back_map = np.full((z_len, x_len), -1, dtype=np.int16)
    front_available = np.zeros((z_len, x_len), dtype=bool)
    back_available = np.zeros((z_len, x_len), dtype=bool)

    for (z, x), candidates in front_candidates.items():
        val = _select_front_candidate(candidates)
        front_map[z, x] = val
        if val >= 0:
            front_available[z, x] = True

    for (z, x), candidates in back_candidates.items():
        front_idx = front_map[z, x]
        val = _select_back_candidate(front_idx, candidates)
        back_map[z, x] = val
        if val >= 0:
            back_available[z, x] = True

    front_map = _median_fill_missing(front_map, radius=1)
    back_map = _median_fill_missing(back_map, radius=1)

    front_map, removed_front = _limit_interpolation_runs(
        front_map,
        front_available,
        FRONT_INTERPOLATION_MAX_GAP,
    )
    back_map, removed_back = _limit_interpolation_runs(
        back_map,
        back_available,
        BACK_INTERPOLATION_MAX_GAP,
    )

    if removed_front:
        logger.info(
            "Removed %d interpolated frontwall positions (gap > %d)",
            removed_front,
            FRONT_INTERPOLATION_MAX_GAP,
        )
    if removed_back:
        logger.info(
            "Removed %d interpolated backwall positions (gap > %d)",
            removed_back,
            BACK_INTERPOLATION_MAX_GAP,
        )

    thickness_delta = back_map.astype(np.int32) - front_map.astype(np.int32)
    invalid_mask = (front_map < 0) | (back_map < 0) | (back_map <= front_map)

    removed_far = 0
    relaxed_far = 0
    if MAX_BACK_DISTANCE_FROM_FRONT >= 0:
        base_valid = (~invalid_mask) & (thickness_delta >= 0)
        ok_mask = base_valid & (thickness_delta <= MAX_BACK_DISTANCE_FROM_FRONT)
        far_mask = base_valid & (thickness_delta > MAX_BACK_DISTANCE_FROM_FRONT)

        if NEIGHBOR_DISTANCE_RELAX_RADIUS > 0 and np.any(far_mask):
            keep_mask = np.zeros_like(far_mask)
            tolerance = NEIGHBOR_DISTANCE_RELAX_TOLERANCE
            for z in range(z_len):
                far_cols = np.where(far_mask[z])[0]
                if far_cols.size == 0:
                    continue
                for x in far_cols:
                    current_delta = thickness_delta[z, x]
                    keep = False
                    for dx in range(-NEIGHBOR_DISTANCE_RELAX_RADIUS, NEIGHBOR_DISTANCE_RELAX_RADIUS + 1):
                        if dx == 0:
                            continue
                        nx = x + dx
                        if 0 <= nx < x_len and ok_mask[z, nx]:
                            neighbor_delta = thickness_delta[z, nx]
                            diff = current_delta - neighbor_delta
                            if tolerance < 0 or diff <= tolerance:
                                keep = True
                                break
                    if keep:
                        keep_mask[z, x] = True
            relaxed_far = int(np.count_nonzero(keep_mask))
            far_mask &= ~keep_mask
        removed_far = int(np.count_nonzero(far_mask))
        invalid_mask |= far_mask

    front_map[invalid_mask] = -1
    back_map[invalid_mask] = -1

    if removed_far:
        logger.info(
            "Removed %d backwall positions farther than %d pixels from frontwall",
            removed_far,
            MAX_BACK_DISTANCE_FROM_FRONT,
        )
    if relaxed_far:
        logger.info(
            "Kept %d backwall positions beyond %d pixels due to neighboring lines",
            relaxed_far,
            MAX_BACK_DISTANCE_FROM_FRONT,
        )

    front_peaks: dict[tuple[int, int], int] = {}
    back_peaks: dict[tuple[int, int], int] = {}

    for z in range(z_len):
        for x in range(x_len):
            y_front = int(front_map[z, x])
            y_back = int(back_map[z, x])
            if y_front >= 0:
                front_peaks[(z, x)] = y_front
            if y_back >= 0:
                back_peaks[(z, x)] = y_back

    if "frontwall" in labels:
        ascan_peaks["frontwall"] = front_peaks
    if "backwall" in labels:
        ascan_peaks["backwall"] = back_peaks

    for class_name, peaks in other_class_peaks.items():
        ascan_peaks[class_name] = peaks

    logger.info(
        "Outlier filtering is DISABLED - returning boundary-aware peaks for nnUNet classes"
    )
    return ascan_peaks


def create_thickness_lines(
    mask_shape: tuple[int, int, int],
    frontwall_peaks: dict[tuple[int, int], int],
    backwall_peaks: dict[tuple[int, int], int],
) -> np.ndarray:
    """
    Crée un masque avec des lignes verticales (en Y) entre frontwall et backwall.
    """
    thickness_mask = np.zeros(mask_shape, dtype=np.uint8)

    lines_created = 0
    total_pixels = 0

    logger.info(
        ">>> create_thickness_lines: mask_shape=%s (Z, X=largeur, Y=profondeur)",
        mask_shape,
    )
    logger.info(
        ">>> create_thickness_lines: frontwall_peaks count=%d",
        len(frontwall_peaks),
    )
    logger.info(
        ">>> create_thickness_lines: backwall_peaks count=%d",
        len(backwall_peaks),
    )

    for (z, x), y_front in frontwall_peaks.items():
        if (z, x) in backwall_peaks:
            y_back = backwall_peaks[(z, x)]
            y_min = min(y_front, y_back)
            y_max = max(y_front, y_back)

            if y_min != y_max:
                thickness_mask[z, x, y_min:y_max + 1] = 1
                lines_created += 1
                total_pixels += (y_max - y_min + 1)

    logger.info(
        "Created %d thickness lines with %d total pixels",
        lines_created,
        total_pixels,
    )

    return thickness_mask


def build_thickness_measurements(
    frontwall_peaks: dict[tuple[int, int], int],
    backwall_peaks: dict[tuple[int, int], int],
) -> dict[str, Any] | None:
    """
    Collecte les distances frontwall->backwall sous forme exploitable par Sentinel.
    """
    entries: list[dict[str, int]] = []
    by_position: dict[str, dict[str, int]] = {}

    for (z, x), y_front in frontwall_peaks.items():
        if (z, x) not in backwall_peaks:
            continue
        y_back = backwall_peaks[(z, x)]
        distance_px = abs(int(y_back) - int(y_front))
        if distance_px <= 0:
            continue

        entry = {
            "z": int(z),
            "x": int(x),
            "front_y": int(y_front),
            "back_y": int(y_back),
            "distance_pixels": int(distance_px),
        }
        entries.append(entry)
        key = f"z{int(z)}_x{int(x)}"
        by_position[key] = {
            "distance_pixels": entry["distance_pixels"],
            "front_y": entry["front_y"],
            "back_y": entry["back_y"],
        }

    if not entries:
        return None

    distances = np.array([entry["distance_pixels"] for entry in entries], dtype=np.float32)
    stats = {
        "count": int(distances.size),
        "min_pixels": int(distances.min()),
        "max_pixels": int(distances.max()),
        "mean_pixels": float(distances.mean()),
        "median_pixels": float(np.median(distances)),
    }

    return {
        "unit": "pixel",
        "entries": entries,
        "positions": by_position,
        "stats": stats,
    }


def _median_fill_missing(map_arr: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Remplit les valeurs -1 par la médiane locale des voisins valides.
    Utilisé pour combler les colonnes orphelines après sélection frontwall/backwall.
    """
    filled = map_arr.copy()
    z_len, x_len = map_arr.shape
    for z in range(z_len):
        z0 = max(0, z - radius)
        z1 = min(z_len, z + radius + 1)
        for x in range(x_len):
            if filled[z, x] >= 0:
                continue
            x0 = max(0, x - radius)
            x1 = min(x_len, x + radius + 1)
            window = map_arr[z0:z1, x0:x1]
            valid = window[window >= 0]
            if valid.size:
                filled[z, x] = int(np.median(valid))
    return filled


def _limit_interpolation_runs(
    map_arr: np.ndarray,
    available_mask: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, int]:
    """
    Annule les interpolations lorsque la longueur du trou dépasse `max_gap`.
    Retourne le tableau ajusté et le nombre de cellules remises à -1.
    """
    if max_gap < 0:
        return map_arr, 0

    limited = map_arr.copy()
    removed = 0
    z_len, x_len = map_arr.shape

    for z in range(z_len):
        row_values = limited[z]
        row_available = available_mask[z]
        x = 0
        while x < x_len:
            if row_available[x] or row_values[x] < 0:
                x += 1
                continue
            start = x
            while x < x_len and (not row_available[x]) and row_values[x] >= 0:
                x += 1
            gap_len = x - start
            if gap_len > max_gap:
                row_values[start:x] = -1
                removed += gap_len
    return limited, removed
