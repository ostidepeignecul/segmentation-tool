from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from models.overlay_data import OverlayLayerData


@dataclass(frozen=True)
class VectorOverlayPathData:
    """One polyline ready to be rendered for a specific label."""

    label: int
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class VectorOverlayLayerData:
    """Vectorized geometry for one overlay layer and one displayed slice."""

    layer_id: str
    paths: tuple[VectorOverlayPathData, ...]


class OverlayVectorizationService:
    """Build cached vector polylines from overlay mask slices without any Qt dependency."""

    LEFT_EXTENSION_PX = 0.5
    RIGHT_EXTENSION_PX = 0.5
    CACHE_LIMIT = 512

    def __init__(self) -> None:
        self._layer_payload_cache: dict[
            tuple[str, int, Optional[tuple[int, ...]], int],
            VectorOverlayLayerData,
        ] = {}

    def clear_cache(self) -> None:
        self._layer_payload_cache.clear()

    def build_layer_payload(
        self,
        layer: OverlayLayerData,
        *,
        slice_idx: int,
    ) -> Optional[VectorOverlayLayerData]:
        overlay = layer.overlay
        if overlay is None or overlay.mask_volume is None:
            return None
        try:
            depth = int(overlay.mask_volume.shape[0])
        except Exception:
            return None
        if slice_idx < 0 or slice_idx >= depth:
            return None

        slice_mask = np.asarray(overlay.mask_volume[int(slice_idx)], dtype=np.int32)
        if slice_mask.ndim != 2 or slice_mask.size == 0:
            return None

        visible_labels_key = (
            tuple(sorted(int(label_id) for label_id in layer.visible_labels))
            if layer.visible_labels is not None
            else None
        )
        cache_key = (
            str(layer.layer_id),
            int(slice_idx),
            visible_labels_key,
            id(overlay),
        )
        cached = self._layer_payload_cache.get(cache_key)
        if cached is not None:
            return cached

        labels_to_draw = (
            set(int(label_id) for label_id in layer.visible_labels)
            if layer.visible_labels is not None
            else {int(label_id) for label_id in np.unique(slice_mask) if int(label_id) > 0}
        )
        paths: list[VectorOverlayPathData] = []
        height, width = slice_mask.shape
        for label in sorted(labels_to_draw):
            if label <= 0:
                continue
            paths.extend(
                self._paths_for_label(
                    slice_mask,
                    label=label,
                    width=width,
                    height=height,
                )
            )

        payload = VectorOverlayLayerData(
            layer_id=str(layer.layer_id),
            paths=tuple(paths),
        )
        if len(self._layer_payload_cache) >= self.CACHE_LIMIT:
            self._layer_payload_cache.clear()
        self._layer_payload_cache[cache_key] = payload
        return payload

    @classmethod
    def _paths_for_label(
        cls,
        slice_mask: np.ndarray,
        *,
        label: int,
        width: int,
        height: int,
    ) -> list[VectorOverlayPathData]:
        y_coords, x_coords = np.nonzero(slice_mask == int(label))
        if y_coords.size == 0:
            return []

        sum_y = np.bincount(x_coords, weights=y_coords, minlength=width)
        count_y = np.bincount(x_coords, minlength=width)
        valid = count_y > 0
        x_valid = np.nonzero(valid)[0]
        if x_valid.size == 0:
            return []

        mean_y = np.zeros(width, dtype=np.float32)
        mean_y[valid] = sum_y[valid] / count_y[valid]

        paths: list[VectorOverlayPathData] = []
        start = 0
        while start < x_valid.size:
            end = start
            while end + 1 < x_valid.size and x_valid[end + 1] == x_valid[end] + 1:
                end += 1
            segment = x_valid[start : end + 1]
            points = cls._segment_points(segment=segment, mean_y=mean_y, height=height, width=width)
            if points:
                paths.append(
                    VectorOverlayPathData(
                        label=int(label),
                        points=points,
                    )
                )
            start = end + 1
        return paths

    @classmethod
    def _segment_points(
        cls,
        *,
        segment: np.ndarray,
        mean_y: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[tuple[float, float], ...]:
        if segment.size == 0:
            return ()
        if segment.size == 1:
            x = int(segment[0])
            y = int(round(float(mean_y[x])))
            if not (0 <= x < width and 0 <= y < height):
                return ()
            center_x = x + 0.5
            y_pos = y + 0.5
            return (
                (center_x - cls.LEFT_EXTENSION_PX, y_pos),
                (center_x + cls.RIGHT_EXTENSION_PX, y_pos),
            )

        x0 = int(segment[0])
        y0 = int(round(float(mean_y[x0])))
        y0 = max(0, min(height - 1, y0))
        points: list[tuple[float, float]] = [
            ((x0 + 0.5) - cls.LEFT_EXTENSION_PX, y0 + 0.5)
        ]
        last_x = x0
        last_y = y0
        for x in segment[1:]:
            xi = int(x)
            yi = int(round(float(mean_y[xi])))
            yi = max(0, min(height - 1, yi))
            points.append((xi + 0.5, yi + 0.5))
            last_x = xi
            last_y = yi
        points.append(((last_x + 0.5) + cls.RIGHT_EXTENSION_PX, last_y + 0.5))
        return tuple(points)
