"""Centralize display-unit conversion for shared ruler widgets."""

from __future__ import annotations

from typing import Optional

import numpy as np


class RulerDisplayService:
    """Build display-domain values shared by all ruler-enabled views."""

    DISPLAY_UNIT_PIXELS = "px"
    DISPLAY_UNIT_MM = "mm"

    @classmethod
    def normalize_display_unit(cls, display_unit: Optional[str]) -> str:
        normalized = str(display_unit or cls.DISPLAY_UNIT_PIXELS).strip().casefold()
        if normalized == cls.DISPLAY_UNIT_MM:
            return cls.DISPLAY_UNIT_MM
        return cls.DISPLAY_UNIT_PIXELS

    @classmethod
    def build_axis_values(
        cls,
        *,
        sample_count: int,
        source_positions: Optional[np.ndarray] = None,
        resolution_mm: Optional[float] = None,
        display_unit: Optional[str] = None,
    ) -> np.ndarray:
        if int(sample_count) <= 0:
            return np.array([], dtype=np.float32)

        unit = cls.normalize_display_unit(display_unit)
        if unit == cls.DISPLAY_UNIT_MM:
            resolution_axis = cls._build_mm_axis_values(
                sample_count=sample_count,
                resolution_mm=resolution_mm,
            )
            if resolution_axis is not None:
                return resolution_axis
            positions = cls._coerce_positions(source_positions, expected_len=sample_count)
            if positions is not None:
                return positions

        return np.arange(int(sample_count), dtype=np.float32)

    @classmethod
    def build_content_range(
        cls,
        *,
        sample_count: int,
        source_positions: Optional[np.ndarray] = None,
        resolution_mm: Optional[float] = None,
        display_unit: Optional[str] = None,
    ) -> tuple[float, float]:
        axis_values = cls.build_axis_values(
            sample_count=sample_count,
            source_positions=source_positions,
            resolution_mm=resolution_mm,
            display_unit=display_unit,
        )
        if axis_values.size == 0:
            return 0.0, 0.0
        return float(axis_values[0]), float(axis_values[-1])

    @classmethod
    def axis_value_for_display(
        cls,
        value_px: Optional[float],
        *,
        display_unit: Optional[str] = None,
        resolution_mm: Optional[float] = None,
    ) -> float:
        value = cls._coerce_scalar(value_px)
        if value is None:
            return 0.0

        unit = cls.normalize_display_unit(display_unit)
        if unit == cls.DISPLAY_UNIT_MM:
            mm_per_px = cls.normalize_resolution_mm(resolution_mm)
            if mm_per_px is not None:
                return value * mm_per_px
        return value

    @classmethod
    def normalize_resolution_mm(cls, resolution_mm: Optional[float]) -> Optional[float]:
        try:
            value = float(resolution_mm)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value) or value <= 0.0:
            return None
        return value

    @classmethod
    def distance_for_display(
        cls,
        distance_px: Optional[float],
        *,
        display_unit: Optional[str] = None,
        resolution_mm: Optional[float] = None,
    ) -> tuple[Optional[float], str]:
        value_px = cls._coerce_scalar(distance_px)
        if value_px is None:
            return None, cls.DISPLAY_UNIT_PIXELS

        unit = cls.normalize_display_unit(display_unit)
        if unit == cls.DISPLAY_UNIT_MM:
            mm_per_px = cls.normalize_resolution_mm(resolution_mm)
            if mm_per_px is not None:
                return value_px * mm_per_px, cls.DISPLAY_UNIT_MM

        return value_px, cls.DISPLAY_UNIT_PIXELS

    @classmethod
    def format_distance(
        cls,
        distance_px: Optional[float],
        *,
        display_unit: Optional[str] = None,
        resolution_mm: Optional[float] = None,
        px_decimals: int = 1,
        mm_decimals: int = 1,
        fallback_text: str = "-",
    ) -> str:
        value, unit = cls.distance_for_display(
            distance_px,
            display_unit=display_unit,
            resolution_mm=resolution_mm,
        )
        if value is None:
            return str(fallback_text)

        decimals = mm_decimals if unit == cls.DISPLAY_UNIT_MM else px_decimals
        return f"{value:.{max(0, int(decimals))}f} {unit}"

    @staticmethod
    def build_sample_edges(axis_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        samples = np.asarray(axis_values, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        if samples.size == 1:
            center = float(samples[0])
            return (
                np.asarray([center - 0.5], dtype=np.float32),
                np.asarray([center + 0.5], dtype=np.float32),
            )

        midpoints = (samples[:-1] + samples[1:]) / 2.0
        left_edges = np.empty_like(samples, dtype=np.float32)
        right_edges = np.empty_like(samples, dtype=np.float32)
        left_edges[0] = samples[0] - (midpoints[0] - samples[0])
        left_edges[1:] = midpoints
        right_edges[:-1] = midpoints
        right_edges[-1] = samples[-1] + (samples[-1] - midpoints[-1])
        return left_edges, right_edges

    @staticmethod
    def _coerce_positions(
        source_positions: Optional[np.ndarray],
        *,
        expected_len: int,
    ) -> Optional[np.ndarray]:
        if source_positions is None:
            return None
        positions = np.asarray(source_positions, dtype=np.float32).reshape(-1)
        if positions.size != int(expected_len):
            return None
        if not np.all(np.isfinite(positions)):
            return None
        return positions

    @classmethod
    def _build_mm_axis_values(
        cls,
        *,
        sample_count: int,
        resolution_mm: Optional[float],
    ) -> Optional[np.ndarray]:
        mm_per_px = cls.normalize_resolution_mm(resolution_mm)
        if mm_per_px is None:
            return None
        indices = np.arange(int(sample_count), dtype=np.float32)
        return indices * np.float32(mm_per_px)

    @staticmethod
    def _coerce_scalar(value: Optional[float]) -> Optional[float]:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(scalar):
            return None
        return scalar
