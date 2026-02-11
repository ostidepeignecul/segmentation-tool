"""Business rules for annotation axis selection and secondary orthogonal view."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

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

        for attr in ("volume", "normalized_volume"):
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
        primary = str(axis_order[0]) if len(axis_order) >= 1 else "Primary"
        secondary = str(axis_order[2]) if len(axis_order) >= 3 else "Secondary"
        auto_suffix = " [Auto]" if str(axis_mode) == "Auto" else ""
        return CoordinateDockTitles(
            primary_axis_name=primary,
            secondary_axis_name=secondary,
            primary_title=f"{primary} (annotation){auto_suffix}",
            secondary_title=f"{secondary} (read-only)",
        )

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
    def secondary_crosshair(
        *,
        current_slice: int,
        current_point: Optional[tuple[int, int]],
    ) -> Optional[tuple[int, int]]:
        """Map main-view cursor state to secondary-view crosshair coordinates."""
        if current_point is None:
            return None
        return int(current_slice), int(current_point[1])
