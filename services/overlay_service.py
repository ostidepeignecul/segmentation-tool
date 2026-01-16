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
        """Retourne un OverlayData avec le volume de masque complet."""
        if mask_volume is None:
            return None

        # Just pass the uint8 mask directly.
        # We no longer split into float32 volumes per label.
        if label_palette is None:
            palette: Dict[int, Tuple[int, int, int, int]] = dict(MASK_COLORS_BGRA)
        else:
            palette = dict(label_palette)

        return OverlayData(
            mask_volume=np.asarray(mask_volume, dtype=np.uint8),
            palette=palette,
            label_volumes={} # Empty to save memory
        )

    def update_overlay_slice(
        self,
        *,
        mask_volume: Optional[np.ndarray],
        label_palette: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
        overlay_cache: Optional[OverlayData],
        slice_idx: int,
    ) -> Optional[OverlayData]:
        """
        Update only one slice of the overlay.
        Since we now hold the full mask reference, this is trivial/instant
        assuming mask_volume is the source of truth.
        """
        return self.build_overlay_data(mask_volume, label_palette)
