from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from config.constants import MASK_COLORS_BGRA
from models.overlay_data import OverlayData

FallbackColor = (255, 0, 255, 160)  # Magenta semi-transparent


class OverlayService:
    """Construit des overlays compacts à partir d'un volume de masques et d'une palette BGRA."""

    def build_overlay_data(
        self,
        mask_volume: Optional[np.ndarray],
        label_palette: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
    ) -> Optional[OverlayData]:
        """Retourne un OverlayData avec un volume alpha par label (tous labels présents)."""
        if mask_volume is None:
            return None

        masks = np.asarray(mask_volume, dtype=np.uint8)
        if masks.ndim != 3:
            raise ValueError("Mask volume must be 3D (Z,H,W).")

        palette: Dict[int, Tuple[int, int, int, int]] = dict(label_palette or MASK_COLORS_BGRA)
        labels_present = [int(v) for v in np.unique(masks) if int(v) != 0]
        if not labels_present:
            return None

        label_volumes: Dict[int, np.ndarray] = {}
        for cls_int in labels_present:
            color = palette.get(cls_int, FallbackColor)
            a = float(color[3]) / 255.0
            if a <= 0.0:
                continue
            # Bool mask -> alpha float32
            mask_cls = masks == cls_int
            if not np.any(mask_cls):
                continue
            alpha_vol = np.zeros_like(masks, dtype=np.float32)
            alpha_vol[mask_cls] = a
            label_volumes[cls_int] = alpha_vol

        if not label_volumes:
            return None

        return OverlayData(label_volumes=label_volumes, palette=palette)
