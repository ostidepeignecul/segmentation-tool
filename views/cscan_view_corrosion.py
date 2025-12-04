"""C-Scan corrosion view with dedicated colormap."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from views.cscan_view import CScanView


class CscanViewCorrosion(CScanView):
    """Displays corrosion distance maps with a red→orange→yellow→blue gradient."""

    _LUT_CACHE: Optional[np.ndarray] = None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        if CscanViewCorrosion._LUT_CACHE is None:
            CscanViewCorrosion._LUT_CACHE = self._build_lut()

    @staticmethod
    def _build_lut() -> np.ndarray:
        """Build a 256x3 LUT with linear segments."""
        stops = [
            (0.0, (255, 0, 0)),      # red (small distance)
            (0.33, (255, 128, 0)),   # orange
            (0.66, (255, 255, 0)),   # yellow
            (1.0, (0, 128, 255)),    # blue (large distance)
        ]
        lut = np.zeros((256, 3), dtype=np.uint8)
        for idx in range(len(stops) - 1):
            start_pos, start_col = stops[idx]
            end_pos, end_col = stops[idx + 1]
            start_idx = int(round(start_pos * 255))
            end_idx = int(round(end_pos * 255))
            span = max(1, end_idx - start_idx)
            for channel in range(3):
                start_val = start_col[channel]
                end_val = end_col[channel]
                lut[start_idx:end_idx + 1, channel] = np.linspace(
                    start_val, end_val, span + 1, dtype=np.uint8
                )
        # Garantir dernière valeur
        lut[-1, :] = stops[-1][1]
        return lut

    @staticmethod
    def _to_rgb(data: np.ndarray, value_range: Tuple[float, float], _unused_lut=None) -> np.ndarray:
        vmin, vmax = value_range
        if vmax <= vmin:
            return np.zeros((*data.shape, 3), dtype=np.uint8)
        normalized = (data - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0.0, 1.0)
        indices = (normalized * 255.0).astype(np.int32)
        lut = CscanViewCorrosion._LUT_CACHE
        if lut is None:
            lut = CscanViewCorrosion._build_lut()
            CscanViewCorrosion._LUT_CACHE = lut
        rgb = lut[indices]
        return rgb
