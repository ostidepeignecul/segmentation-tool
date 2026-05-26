from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from models.overlay_data import CorrosionProfileData, OverlayLayerData


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
            tuple[object, ...],
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

        profile_payload = self._build_profile_payload(
            layer,
            slice_idx=int(slice_idx),
            visible_labels_key=visible_labels_key,
        )
        if profile_payload is not None:
            self._cache_payload(cache_key, profile_payload)
            return profile_payload

        slice_mask = np.asarray(overlay.mask_volume[int(slice_idx)], dtype=np.int32)
        if slice_mask.ndim != 2 or slice_mask.size == 0:
            return None

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
        self._cache_payload(cache_key, payload)
        return payload

    def _cache_payload(
        self,
        cache_key: tuple[object, ...],
        payload: VectorOverlayLayerData,
    ) -> None:
        if len(self._layer_payload_cache) >= self.CACHE_LIMIT:
            self._layer_payload_cache.clear()
        self._layer_payload_cache[cache_key] = payload

    def _build_profile_payload(
        self,
        layer: OverlayLayerData,
        *,
        slice_idx: int,
        visible_labels_key: Optional[tuple[int, ...]],
    ) -> Optional[VectorOverlayLayerData]:
        profile = getattr(layer, "corrosion_profile", None)
        if profile is None:
            return None
        if not isinstance(profile, CorrosionProfileData):
            return None

        cache_key = (
            "profile",
            str(layer.layer_id),
            int(slice_idx),
            visible_labels_key,
            id(profile.peak_map_a),
            id(profile.peak_map_b),
            tuple(int(v) for v in profile.label_ids),
            tuple(int(v) for v in profile.image_shape),
            bool(profile.connect_points),
            None if profile.max_gap_px is None else int(profile.max_gap_px),
        )
        cached = self._layer_payload_cache.get(cache_key)
        if cached is not None:
            return cached

        labels_to_draw = (
            set(int(label_id) for label_id in layer.visible_labels)
            if layer.visible_labels is not None
            else {int(profile.label_ids[0]), int(profile.label_ids[1])}
        )
        height, width = (int(profile.image_shape[0]), int(profile.image_shape[1]))
        label_sources = (
            (int(profile.label_ids[0]), np.asarray(profile.peak_map_a, dtype=np.int32)),
            (int(profile.label_ids[1]), np.asarray(profile.peak_map_b, dtype=np.int32)),
        )
        paths: list[VectorOverlayPathData] = []
        for label, peak_map in label_sources:
            if label not in labels_to_draw:
                continue
            paths.extend(
                self._paths_for_peak_row(
                    peak_map,
                    label=label,
                    slice_idx=slice_idx,
                    width=width,
                    height=height,
                    connect_points=bool(profile.connect_points),
                    max_gap_px=profile.max_gap_px,
                )
            )

        payload = VectorOverlayLayerData(
            layer_id=str(layer.layer_id),
            paths=tuple(paths),
        )
        self._cache_payload(cache_key, payload)
        return payload

    @classmethod
    def _paths_for_peak_row(
        cls,
        peak_map: np.ndarray,
        *,
        label: int,
        slice_idx: int,
        width: int,
        height: int,
        connect_points: bool,
        max_gap_px: Optional[int],
    ) -> list[VectorOverlayPathData]:
        data = np.asarray(peak_map, dtype=np.int32)
        if data.ndim != 2 or data.size == 0:
            return []
        if slice_idx < 0 or slice_idx >= data.shape[0]:
            return []

        width_map = min(int(width), int(data.shape[1]))
        if width_map <= 0 or height <= 0:
            return []
        row = data[int(slice_idx), :width_map]
        valid = np.where((row >= 0) & (row < int(height)))[0]
        if valid.size == 0:
            return []

        if not connect_points:
            return [
                VectorOverlayPathData(
                    label=int(label),
                    points=cls._point_segment(x=int(x), y=int(row[int(x)]), width=width, height=height),
                )
                for x in valid.tolist()
            ]

        paths: list[VectorOverlayPathData] = []
        max_gap = 0 if max_gap_px is None else max(0, int(max_gap_px))
        start = 0
        while start < valid.size:
            end = start
            while end + 1 < valid.size and int(valid[end + 1]) - int(valid[end]) - 1 <= max_gap:
                end += 1
            segment = valid[start : end + 1]
            points = cls._peak_segment_points(segment=segment, row=row, height=height, width=width)
            if points:
                paths.append(VectorOverlayPathData(label=int(label), points=points))
            start = end + 1
        return paths

    @classmethod
    def _point_segment(
        cls,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> tuple[tuple[float, float], ...]:
        if not (0 <= x < width and 0 <= y < height):
            return ()
        center_x = int(x) + 0.5
        y_pos = int(y) + 0.5
        return (
            (center_x - cls.LEFT_EXTENSION_PX, y_pos),
            (center_x + cls.RIGHT_EXTENSION_PX, y_pos),
        )

    @classmethod
    def _peak_segment_points(
        cls,
        *,
        segment: np.ndarray,
        row: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[tuple[float, float], ...]:
        if segment.size == 0:
            return ()
        if segment.size == 1:
            x = int(segment[0])
            return cls._point_segment(x=x, y=int(row[x]), width=width, height=height)

        x0 = int(segment[0])
        y0 = int(row[x0])
        y0 = max(0, min(int(height) - 1, y0))
        points: list[tuple[float, float]] = [
            ((x0 + 0.5) - cls.LEFT_EXTENSION_PX, y0 + 0.5)
        ]
        last_x = x0
        last_y = y0
        for x in segment[1:]:
            xi = int(x)
            yi = int(row[xi])
            yi = max(0, min(int(height) - 1, yi))
            points.append((xi + 0.5, yi + 0.5))
            last_x = xi
            last_y = yi
        points.append(((last_x + 0.5) + cls.RIGHT_EXTENSION_PX, last_y + 0.5))
        return tuple(points)

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

        order = np.lexsort((y_coords, x_coords))
        sorted_x = x_coords[order]
        sorted_y = y_coords[order]
        x_valid, first_indices = np.unique(sorted_x, return_index=True)
        if x_valid.size == 0:
            return []
        y_by_x = np.full(width, -1, dtype=np.int32)
        y_by_x[x_valid] = sorted_y[first_indices].astype(np.int32, copy=False)

        paths: list[VectorOverlayPathData] = []
        start = 0
        while start < x_valid.size:
            end = start
            while end + 1 < x_valid.size and x_valid[end + 1] == x_valid[end] + 1:
                end += 1
            segment = x_valid[start : end + 1]
            points = cls._segment_points(segment=segment, y_by_x=y_by_x, height=height, width=width)
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
        y_by_x: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[tuple[float, float], ...]:
        if segment.size == 0:
            return ()
        if segment.size == 1:
            x = int(segment[0])
            y = int(y_by_x[x])
            if not (0 <= x < width and 0 <= y < height):
                return ()
            center_x = x + 0.5
            y_pos = y + 0.5
            return (
                (center_x - cls.LEFT_EXTENSION_PX, y_pos),
                (center_x + cls.RIGHT_EXTENSION_PX, y_pos),
            )

        x0 = int(segment[0])
        y0 = int(y_by_x[x0])
        y0 = max(0, min(height - 1, y0))
        points: list[tuple[float, float]] = [
            ((x0 + 0.5) - cls.LEFT_EXTENSION_PX, y0 + 0.5)
        ]
        last_x = x0
        last_y = y0
        for x in segment[1:]:
            xi = int(x)
            yi = int(y_by_x[xi])
            yi = max(0, min(height - 1, yi))
            points.append((xi + 0.5, yi + 0.5))
            last_x = xi
            last_y = yi
        points.append(((last_x + 0.5) + cls.RIGHT_EXTENSION_PX, last_y + 0.5))
        return tuple(points)
