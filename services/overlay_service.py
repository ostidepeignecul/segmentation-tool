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

        if label_palette is None:
            palette: Dict[int, Tuple[int, int, int, int]] = dict(MASK_COLORS_BGRA)
        else:
            palette = dict(label_palette)
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

    def update_overlay_slice(
        self,
        *,
        mask_volume: Optional[np.ndarray],
        label_palette: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
        overlay_cache: Optional[OverlayData],
        slice_idx: int,
    ) -> Optional[OverlayData]:
        """
        Update only one slice of the overlay using an existing cache.

        Falls back to full rebuild if cache/shape is missing or inconsistent.
        """
        if mask_volume is None:
            return None
        masks = np.asarray(mask_volume, dtype=np.uint8)
        if masks.ndim != 3:
            raise ValueError("Mask volume must be 3D (Z,H,W).")
        depth, height, width = masks.shape
        if slice_idx < 0 or slice_idx >= depth:
            return self.build_overlay_data(mask_volume, label_palette)

        palette: Dict[int, Tuple[int, int, int, int]] = (
            dict(label_palette) if label_palette is not None else dict(MASK_COLORS_BGRA)
        )

        cached_volumes: Dict[int, np.ndarray] = {}
        if overlay_cache is not None:
            for label, vol in overlay_cache.label_volumes.items():
                arr = np.asarray(vol, dtype=np.float32)
                if arr.shape != masks.shape:
                    cached_volumes = {}
                    break
                cached_volumes[int(label)] = arr

        # If the cache is unusable, rebuild everything.
        if overlay_cache is None or not cached_volumes:
            return self.build_overlay_data(mask_volume, palette)

        label_volumes: Dict[int, np.ndarray] = dict(cached_volumes)
        slice_mask = masks[int(slice_idx)]
        labels_in_slice = [int(v) for v in np.unique(slice_mask) if int(v) != 0]
        # Preserve cached volumes, but copy only what needs to change on this slice.
        for label, arr in cached_volumes.items():
            try:
                slice_view = arr[int(slice_idx)]
            except Exception:
                slice_view = None
            needs_change = label in labels_in_slice or (slice_view is not None and np.any(slice_view))
            if needs_change:
                vol = np.array(arr, dtype=np.float32, copy=True)
                vol[int(slice_idx)] = 0.0
                label_volumes[label] = vol
            else:
                label_volumes[label] = np.asarray(arr, dtype=np.float32)

        # Apply current slice data to the relevant labels.
        for label in labels_in_slice:
            color = palette.get(label, FallbackColor)
            alpha = float(color[3]) / 255.0
            if alpha <= 0.0:
                continue
            vol = label_volumes.get(label)
            if vol is None:
                vol = np.zeros((depth, height, width), dtype=np.float32)
            mask_lbl = slice_mask == label
            if np.any(mask_lbl):
                vol[int(slice_idx)] = 0.0
                vol[int(slice_idx)][mask_lbl] = alpha
            label_volumes[label] = vol

        # Drop labels that are now empty.
        for label in list(label_volumes.keys()):
            if not np.any(label_volumes[label]):
                del label_volumes[label]

        if not label_volumes:
            return None

        return OverlayData(label_volumes=label_volumes, palette=palette)
