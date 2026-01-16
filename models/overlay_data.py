from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class OverlayData:
    """Overlay payload containing full mask volume and palette."""
    mask_volume: Optional[np.ndarray]  # uint8 volume (Z,H,W)
    palette: Dict[int, Tuple[int, int, int, int]]  # BGRA colors per label

    # DEPRECATED: Kept for temporary compatibility if needed, but we aim to remove.
    label_volumes: Mapping[int, np.ndarray] = None  # label -> float32 alpha volume
