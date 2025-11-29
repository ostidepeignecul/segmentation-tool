from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from config.constants import MASK_COLORS_BGRA

FallbackColor = (255, 0, 255, 160)  # Magenta semi-transparent


class OverlayService:
    """Construit des overlays RGBA à partir d'un volume de masques et d'une palette BGRA."""

    def build_overlay_rgba(
        self,
        mask_volume: Optional[np.ndarray],
        label_palette: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
        visible_labels: Optional[Iterable[int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Retourne un tableau (Z, H, W, 4) BGRA respectant la visibilité.
        Ne doit pas dépendre de PyQt6.
        """
        if mask_volume is None:
            return None

        masks = np.asarray(mask_volume, dtype=np.uint8)
        if masks.ndim != 3:
            raise ValueError("Mask volume must be 3D (Z,H,W).")

        overlay = np.zeros((*masks.shape, 4), dtype=np.uint8)
        palette: Dict[int, Tuple[int, int, int, int]] = dict(label_palette or MASK_COLORS_BGRA)
        visible = set(visible_labels) if visible_labels is not None else None

        for cls_value in np.unique(masks):
            cls_int = int(cls_value)
            if cls_int == 0:
                continue
            if visible is not None and cls_int not in visible:
                continue
            color = palette.get(cls_int, FallbackColor)
            overlay[masks == cls_int] = color

        return overlay
