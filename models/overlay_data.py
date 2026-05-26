from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class OverlayData:
    """Overlay payload containing full mask volume and palette."""
    mask_volume: Optional[np.ndarray]  # uint8 volume (Z,H,W)
    palette: Dict[int, Tuple[int, int, int, int]]  # BGRA colors per label

    # DEPRECATED: Kept for temporary compatibility if needed, but we aim to remove.
    label_volumes: Mapping[int, np.ndarray] = None  # label -> float32 alpha volume


@dataclass(frozen=True)
class CorrosionProfileData:
    """Peak-map driven corrosion profile payload for render-only consumers."""

    peak_map_a: np.ndarray
    peak_map_b: np.ndarray
    label_ids: Tuple[int, int]
    image_shape: Tuple[int, int]
    connect_points: bool = True
    max_gap_px: Optional[int] = None


@dataclass(frozen=True)
class OverlayLayerData:
    """One renderable overlay layer with its own opacity and visible labels."""

    layer_id: str
    name: str
    overlay: Optional[OverlayData]
    visible_labels: Optional[frozenset[int]] = None
    opacity: float = 1.0
    corrosion_profile: Optional[CorrosionProfileData] = None


@dataclass(frozen=True)
class OverlayStackData:
    """Ordered overlay stack pushed to 2D/3D views."""

    layers: tuple[OverlayLayerData, ...]
    active_layer_id: Optional[str] = None
