"""Utilities to derive 2D heatmaps (C-Scan) from oriented NDE volumes."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import numpy as np


ReductionKind = Literal["max", "mean"]


class CScanService:
    """Compute top-view projections (X,Z) from a normalized volume."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def compute_top_projection(
        self,
        volume: np.ndarray,
        *,
        y_range: Optional[Tuple[int, int]] = None,
        reduction: ReductionKind = "max",
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Project the volume along the Y axis to obtain a (Z, X) heatmap.

        Args:
            volume: Normalized volume shaped (num_slices, height, width)
            y_range: Optional (start, end) indices along height to restrict the projection.
            reduction: Aggregation function along Y ('max' or 'mean')

        Returns:
            projection: 2D array shaped (num_slices, width)
            value_range: (min, max) of the projection for downstream display
        """
        if volume is None:
            raise ValueError("A normalized volume is required to compute the C-Scan.")

        data = np.asarray(volume)
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D array, got shape {data.shape}.")

        num_slices, height, _ = data.shape

        start, end = self._sanitize_range(y_range, height)
        slice_block = data[:, start:end, :]

        if reduction == "mean":
            projection = slice_block.mean(axis=1)
        elif reduction == "max":
            projection = slice_block.max(axis=1)
        else:
            raise ValueError(f"Unsupported reduction mode: {reduction}")

        projection = projection.astype(np.float32, copy=False)
        value_range = (float(projection.min()), float(projection.max()))

        self.logger.debug(
            "C-Scan projection computed: shape=(%d, %d), y_range=(%d, %d), reduction=%s",
            projection.shape[0],
            projection.shape[1],
            start,
            end,
            reduction,
        )
        return projection, value_range

    @staticmethod
    def _sanitize_range(
        y_range: Optional[Tuple[int, int]],
        height: int,
    ) -> Tuple[int, int]:
        if y_range is None:
            return 0, height
        start, end = y_range
        start = max(0, min(height - 1, start))
        end = max(start + 1, min(height, end))
        return start, end
