"""Business logic to extract and normalize A-Scan profiles from an NDE volume."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from models.simple_nde_model import SimpleNDEModel


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
        model: SimpleNDEModel,
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

    def map_profile_index_to_point(
        self,
        model: SimpleNDEModel,
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

    def _normalize_profile(self, profile: np.ndarray, model: SimpleNDEModel) -> np.ndarray:
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
        model: SimpleNDEModel,
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

    def _ultrasound_axis_index(self, model: SimpleNDEModel, shape: Tuple[int, ...]) -> int:
        axis_order = model.metadata.get("axis_order", [])
        for idx, name in enumerate(axis_order):
            if isinstance(name, str) and name.lower() == "ultrasound":
                return idx
        # Default to last axis if metadata is missing or does not specify an ultrasound axis
        return max(0, len(shape) - 1)
