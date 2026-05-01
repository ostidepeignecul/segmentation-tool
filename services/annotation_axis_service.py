"""Business rules for annotation axis selection and secondary orthogonal view."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from config.constants import MASK_COLORS_BGRA
from models.nde_model import NdeModel
from models.overlay_data import OverlayData


@dataclass(frozen=True)
class CoordinateDockTitles:
    """Computed titles and axis names for U/V coordinate docks."""

    primary_axis_name: str
    secondary_axis_name: str
    primary_title: str
    secondary_title: str


class AnnotationAxisService:
    """Encapsulate axis/orthogonal-view business logic away from controllers."""

    _DISPLAY_AXIS_NAMES = {
        "UCoordinate": "B-Scan",
        "VCoordinate": "D-Scan",
    }

    @staticmethod
    def axis_mode_choices(model: Optional[NdeModel]) -> list[str]:
        """Return allowed annotation plane modes based on model axis metadata."""
        if model is None:
            return ["Auto"]
        axis_order = list(model.metadata.get("axis_order") or [])
        available = {str(name) for name in axis_order}
        choices = ["Auto"]
        if "UCoordinate" in available:
            choices.append("UCoordinate")
        if "VCoordinate" in available:
            choices.append("VCoordinate")
        return choices

    @staticmethod
    def normalize_axis_mode(current_mode: Optional[str], choices: Sequence[str]) -> str:
        """Return a valid axis mode against a list of available choices."""
        if current_mode is None:
            return "Auto"
        mode = str(current_mode)
        return mode if mode in choices else "Auto"

    def apply_axis_mode(self, model: Optional[NdeModel], axis_mode: str) -> Optional[str]:
        """
        Apply requested annotation axis mode directly on model volumes.

        Returns a warning message if the requested mode cannot be applied.
        """
        if model is None:
            return None
        requested = str(axis_mode)
        if requested == "Auto":
            return None

        axis_order = list(model.metadata.get("axis_order") or [])
        if len(axis_order) < 3:
            return None

        primary = str(axis_order[0])
        secondary = str(axis_order[2])
        if primary == requested:
            return None
        if secondary != requested:
            return (
                f"Requested axis {requested} not available as secondary axis "
                f"(order={axis_order}). Keeping auto order."
            )

        for attr in (
            "volume",
            "normalized_volume",
            "processed_volume",
            "processed_normalized_volume",
        ):
            data = getattr(model, attr, None)
            if data is None or getattr(data, "ndim", 0) != 3:
                continue
            setattr(model, attr, np.transpose(np.asarray(data), (2, 1, 0)))

        axis_order[0], axis_order[2] = axis_order[2], axis_order[0]
        model.metadata["axis_order"] = axis_order
        return None

    @staticmethod
    def build_coordinate_dock_titles(
        model: Optional[NdeModel],
        *,
        axis_mode: str,
    ) -> CoordinateDockTitles:
        """Build display titles for coordinate docks from model metadata."""
        axis_order = []
        if model is not None:
            axis_order = list(model.metadata.get("axis_order") or [])
        primary_axis = str(axis_order[0]) if len(axis_order) >= 1 else "UCoordinate"
        secondary_axis = str(axis_order[2]) if len(axis_order) >= 3 else "VCoordinate"
        primary = AnnotationAxisService.display_axis_name(primary_axis)
        secondary = AnnotationAxisService.display_axis_name(secondary_axis)
        auto_suffix = " [Auto]" if str(axis_mode) == "Auto" else ""
        return CoordinateDockTitles(
            primary_axis_name=primary,
            secondary_axis_name=secondary,
            primary_title=f"{primary} (annotation){auto_suffix}",
            secondary_title=f"{secondary} (read-only)",
        )

    @classmethod
    def display_axis_name(cls, axis_name: Optional[str]) -> str:
        """Return the user-facing label for a logical annotation axis."""
        name = str(axis_name or "").strip()
        if not name:
            return "Unknown axis"
        return cls._DISPLAY_AXIS_NAMES.get(name, name)

    @staticmethod
    def build_secondary_volume(volume: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Return orthogonal view volume shaped (other_axis, height, primary_axis)."""
        if volume is None or getattr(volume, "ndim", 0) != 3:
            return None
        return np.transpose(np.asarray(volume), (2, 1, 0))

    @staticmethod
    def build_secondary_overlay_data(overlay_data: Optional[OverlayData]) -> Optional[OverlayData]:
        """Build overlay payload transposed for the secondary orthogonal view."""
        if overlay_data is None or overlay_data.mask_volume is None:
            return None
        try:
            secondary_mask = np.transpose(np.asarray(overlay_data.mask_volume), (2, 1, 0))
        except Exception:
            return None
        return OverlayData(
            mask_volume=secondary_mask,
            palette=dict(overlay_data.palette),
            label_volumes=None,
        )

    @staticmethod
    def build_temp_preview_slice(
        *,
        slice_mask: Optional[np.ndarray],
        coverage: Optional[np.ndarray],
        label_palette: Mapping[int, tuple[int, int, int, int]],
    ) -> tuple[Optional[np.ndarray], dict[int, tuple[int, int, int, int]]]:
        """Build a 2D preview mask from temp mask data and explicit coverage."""
        palette = {
            int(label): tuple(int(channel) for channel in color)
            for label, color in dict(label_palette).items()
        }
        if slice_mask is None or coverage is None:
            return None, palette

        try:
            mask_arr = np.asarray(slice_mask, dtype=np.uint8)
            coverage_arr = np.asarray(coverage, dtype=bool)
        except Exception:
            return None, palette

        if mask_arr.ndim != 2 or coverage_arr.shape != mask_arr.shape or not np.any(coverage_arr):
            return None, palette

        overlay_mask = np.array(mask_arr, copy=True)
        zero_area = coverage_arr & (overlay_mask == 0)
        if np.any(zero_area):
            overlay_mask[zero_area] = 255
            if 0 in palette:
                palette[255] = palette[0]
            else:
                palette[255] = MASK_COLORS_BGRA.get(0, (180, 180, 180, 160))
        return overlay_mask, palette

    @staticmethod
    def build_secondary_temp_preview_slice(
        *,
        temp_mask_volume: Optional[np.ndarray],
        coverage_volume: Optional[np.ndarray],
        secondary_slice: int,
        label_palette: Mapping[int, tuple[int, int, int, int]],
    ) -> tuple[Optional[np.ndarray], dict[int, tuple[int, int, int, int]]]:
        """Extract the orthogonal temp preview slice for the secondary endview."""
        palette = {
            int(label): tuple(int(channel) for channel in color)
            for label, color in dict(label_palette).items()
        }
        if temp_mask_volume is None or coverage_volume is None:
            return None, palette

        try:
            mask_volume = np.asarray(temp_mask_volume, dtype=np.uint8)
            coverage_data = np.asarray(coverage_volume, dtype=bool)
        except Exception:
            return None, palette

        if mask_volume.ndim != 3 or coverage_data.shape != mask_volume.shape:
            return None, palette

        max_idx = int(mask_volume.shape[2]) - 1
        if max_idx < 0:
            return None, palette
        clamped_idx = max(0, min(max_idx, int(secondary_slice)))

        slice_mask = np.transpose(mask_volume[:, :, clamped_idx], (1, 0))
        coverage = np.transpose(coverage_data[:, :, clamped_idx], (1, 0))
        return AnnotationAxisService.build_temp_preview_slice(
            slice_mask=slice_mask,
            coverage=coverage,
            label_palette=palette,
        )

    @staticmethod
    def secondary_crosshair(
        *,
        current_slice: int,
        current_point: Optional[tuple[int, int]],
    ) -> Optional[tuple[int, int]]:
        """Map main-view cursor state to secondary-view crosshair coordinates."""
        if current_point is None:
            return None
        return int(current_slice), int(current_point[1])
