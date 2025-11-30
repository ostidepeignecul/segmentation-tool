from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class OverlayData:
    """Overlay payload containing per-label alpha volumes and palette."""

    label_volumes: Mapping[int, np.ndarray]  # label -> float32 alpha volume (Z,H,W) in [0,1]
    palette: Dict[int, Tuple[int, int, int, int]]  # BGRA colors per label
