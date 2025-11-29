"""Business logic to extract and normalize A-Scan profiles from an NDE volume."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from models.nde_model import NdeModel
from services.ascan_debug_logger import ascan_debug_logger


@dataclass
class AScanProfile:
    """Represents a ready-to-display A-Scan profile."""

    signal_percent: np.ndarray
    positions: np.ndarray
    marker_index: int
    crosshair: Tuple[int, int]


class AScanService:
    """Encapsulates A-Scan extraction logic to keep controllers/view layers lean."""

    def build_profile(
        self,
        model: NdeModel,
        slice_idx: int,
        point_hint: Optional[Tuple[int, int]] = None,
    ) -> Optional[AScanProfile]:
        """Return a normalized profile and metadata for the requested point."""
        volume = model.get_active_volume()
        if volume is None or getattr(volume, "ndim", 0) != 3:
            return None

        num_slices, height, width = volume.shape
        slice_idx = self._clamp(slice_idx, num_slices)

        if point_hint is not None:
            px, py = point_hint
        else:
            px, py = width // 2, height // 2
        px = self._clamp(px, width)
        py = self._clamp(py, height)

        ultrasound_axis = self._ultrasound_axis_index(model, volume.shape)

        profile: Optional[np.ndarray]
        marker_index: int

        if ultrasound_axis == 2:
            profile = volume[slice_idx, py, :]
            marker_index = px
        elif ultrasound_axis == 1:
            profile = volume[slice_idx, :, px]
            marker_index = py
        elif ultrasound_axis == 0:
            profile = volume[:, py, px]
            marker_index = slice_idx
        else:
            profile = volume[slice_idx, :, px]
            marker_index = py

        profile = np.asarray(profile, dtype=np.float32)
        if profile.size == 0:
            return None

        normalized = self._normalize_profile(profile, model)
        positions = self._axis_positions_for_profile(
            model,
            normalized.size,
            ultrasound_axis,
        )

        return AScanProfile(
            signal_percent=normalized * 100.0,
            positions=positions,
            marker_index=marker_index,
            crosshair=(px, py),
        )

    def log_preview(
        self,
        logger,
        model: NdeModel,
        volume: Optional[np.ndarray],
        *,
        slice_idx: Optional[int] = None,
        point: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Log both normalized and raw A-Scan stats for a diagnostic snapshot."""
        if logger is None or model is None or volume is None or getattr(volume, "ndim", 0) != 3:
            return
        depth, height, width = volume.shape[:3]
        if depth == 0 or height == 0 or width == 0:
            return
        s_idx = depth // 2 if slice_idx is None else self._clamp(slice_idx, depth)
        px = width // 2
        py = height // 2
        if point is not None and len(point) == 2:
            px = self._clamp(point[0], width)
            py = self._clamp(point[1], height)

        profile = self.build_profile(model, slice_idx=s_idx, point_hint=(px, py))
        if profile is None or profile.signal_percent.size == 0:
            logger.info("AScan preview: unavailable (empty profile)")
            ascan_debug_logger.log_preview(s_idx, (px, py), None, None)
            return

        sig = profile.signal_percent
        stats = {
            "len": int(sig.size),
            "min": float(sig.min()),
            "max": float(sig.max()),
            "mean": float(sig.mean()),
            "head5": [float(x) for x in sig[:5]],
            "marker": profile.marker_index,
            "crosshair": profile.crosshair,
        }
        logger.info("AScan preview (slice=%d, x=%d, y=%d): %s", s_idx, px, py, stats)

        raw_volume = getattr(model, "volume", None)
        if raw_volume is None or getattr(raw_volume, "ndim", 0) != 3:
            return
        d, h, w = raw_volume.shape[:3]
        if d == 0 or h == 0 or w == 0:
            return
        s_idx_raw = self._clamp(s_idx, d)
        px_raw = self._clamp(px, w)
        py_raw = self._clamp(py, h)
        ultrasound_axis = self._ultrasound_axis_index(model, raw_volume.shape)

        if ultrasound_axis == 2:
            raw_profile = raw_volume[s_idx_raw, py_raw, :]
        elif ultrasound_axis == 1:
            raw_profile = raw_volume[s_idx_raw, :, px_raw]
        else:
            raw_profile = raw_volume[:, py_raw, px_raw]

        raw_arr = np.asarray(raw_profile, dtype=np.float32)
        if raw_arr.size == 0:
            ascan_debug_logger.log_preview(s_idx_raw, (px_raw, py_raw), stats, None)
            return
        raw_stats = {
            "ultrasound_axis": ultrasound_axis,
            "len": int(raw_arr.size),
            "min": float(raw_arr.min()),
            "max": float(raw_arr.max()),
            "mean": float(raw_arr.mean()),
            "head5": [float(x) for x in raw_arr[:5]],
        }
        logger.info("AScan raw preview (slice=%d, x=%d, y=%d): %s", s_idx_raw, px_raw, py_raw, raw_stats)
        ascan_debug_logger.log_preview(s_idx_raw, (px_raw, py_raw), stats, raw_stats)

    def map_profile_index_to_point(
        self,
        model: NdeModel,
        profile_idx: int,
        current_point: Optional[Tuple[int, int]],
        slice_idx: int,
    ) -> Optional[Tuple[Tuple[int, int], Optional[int]]]:
        """Convert an A-Scan marker index back to (x, y) coordinates."""
        volume = model.get_active_volume()
        if volume is None or getattr(volume, "ndim", 0) != 3:
            return None

        ultrasound_axis = self._ultrasound_axis_index(model, volume.shape)
        width = volume.shape[2]
        height = volume.shape[1]
        num_slices = volume.shape[0]

        if ultrasound_axis == 2:
            x = self._clamp(profile_idx, width)
            y = current_point[1] if current_point else height // 2
            return (x, y), None
        if ultrasound_axis == 1:
            y = self._clamp(profile_idx, height)
            x = current_point[0] if current_point else width // 2
            return (x, y), None
        if ultrasound_axis == 0:
            x = current_point[0] if current_point else width // 2
            y = current_point[1] if current_point else height // 2
            new_slice = self._clamp(profile_idx, num_slices)
            return (x, y), new_slice
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _clamp(self, value: int, axis_length: int) -> int:
        if axis_length <= 0:
            return 0
        return max(0, min(axis_length - 1, int(value)))

    def _normalize_profile(self, profile: np.ndarray, model: NdeModel) -> np.ndarray:
        if model.normalized_volume is not None:
            return np.clip(profile, 0.0, 1.0)
        min_value = model.metadata.get("min_value")
        max_value = model.metadata.get("max_value")
        if min_value is None or max_value is None or max_value <= min_value:
            return np.zeros_like(profile, dtype=np.float32)
        normalized = (profile - min_value) / (max_value - min_value)
        return np.clip(normalized.astype(np.float32, copy=False), 0.0, 1.0)

    def _axis_positions_for_profile(
        self,
        model: NdeModel,
        expected_len: int,
        ultrasound_axis: int,
    ) -> np.ndarray:
        axis_order = model.metadata.get("axis_order", [])
        axis_name = None
        if 0 <= ultrasound_axis < len(axis_order):
            axis_name = axis_order[ultrasound_axis]
        target_axis = axis_name or "ultrasound"
        positions = model.get_axis_positions(target_axis)
        if positions is None:
            return np.arange(expected_len, dtype=np.float32)
        pos_array = np.asarray(positions, dtype=np.float32)
        if pos_array.size != expected_len:
            return np.arange(expected_len, dtype=np.float32)
        return pos_array

    def _ultrasound_axis_index(self, model: NdeModel, shape: Tuple[int, ...]) -> int:
        axis_order = model.metadata.get("axis_order", [])
        for idx, name in enumerate(axis_order):
            if isinstance(name, str) and name.lower() == "ultrasound":
                return idx
        # Heuristic fallback: pick the longest non-slice axis (after index 0)
        if len(shape) >= 3:
            tail_axes = list(enumerate(shape[1:], start=1))
            tail_axes.sort(key=lambda item: item[1], reverse=True)
            return tail_axes[0][0]
        return max(0, len(shape) - 1)
