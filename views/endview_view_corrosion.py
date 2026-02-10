"""Corrosion-specific Endview with cosmetic interpolated lines."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtGui import QColor, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsPathItem

from models.overlay_data import OverlayData
from views.endview_view import EndviewView


class EndviewViewCorrosion(EndviewView):
    """Render corrosion overlays as cosmetic lines independent from zoom."""

    _COSMETIC_LINE_WIDTH = 5
    _LEFT_EXTENSION_PX = 0.5
    _RIGHT_EXTENSION_PX = 0.5

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cosmetic_line_items: list[QGraphicsPathItem] = []
        self._overlay_item.setPixmap(QPixmap())

    def set_overlay(
        self,
        overlay: Optional[OverlayData],
        *,
        visible_labels: Optional[set[int]] = None,
    ) -> None:
        super().set_overlay(overlay, visible_labels=visible_labels)
        if overlay is None:
            self._clear_cosmetic_lines()

    def set_overlay_opacity(self, opacity: float) -> None:
        super().set_overlay_opacity(opacity)
        if self._mask_volume is not None and self._volume is not None:
            self._refresh_overlay_pixmap()

    def _refresh_overlay_pixmap(self) -> None:
        """Build cosmetic line items instead of a scaled RGBA pixmap."""
        self._overlay_item.setPixmap(QPixmap())
        self._clear_cosmetic_lines()
        if self._mask_volume is None or self._volume is None:
            return
        depth = int(self._mask_volume.shape[0]) if self._mask_volume.ndim == 3 else 0
        if depth <= 0 or self._current_slice < 0 or self._current_slice >= depth:
            return

        slice_mask = np.asarray(self._mask_volume[self._current_slice], dtype=np.int32)
        if slice_mask.ndim != 2:
            return
        height, width = slice_mask.shape
        labels_to_draw = (
            set(self._visible_labels)
            if self._visible_labels is not None
            else {int(lbl) for lbl in np.unique(slice_mask) if int(lbl) > 0}
        )

        for label in sorted(labels_to_draw):
            if label <= 0:
                continue
            paths = self._paths_for_label(slice_mask, label=label, width=width, height=height)
            if not paths:
                continue

            pen = self._pen_for_label(label)
            for path in paths:
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setZValue(6)
                self._scene.addItem(item)
                self._cosmetic_line_items.append(item)

    def _clear_cosmetic_lines(self) -> None:
        for item in self._cosmetic_line_items:
            self._scene.removeItem(item)
        self._cosmetic_line_items.clear()

    def _pen_for_label(self, label: int) -> QPen:
        b, g, r, a = self._overlay_palette.get(int(label), (255, 0, 255, 200))
        alpha = int(max(0, min(255, round(float(a) * float(self._overlay_opacity)))))
        color = QColor(int(r), int(g), int(b), alpha)
        pen = QPen(color)
        pen.setWidth(self._COSMETIC_LINE_WIDTH)
        pen.setCosmetic(True)
        return pen

    @staticmethod
    def _paths_for_label(
        slice_mask: np.ndarray,
        *,
        label: int,
        width: int,
        height: int,
    ) -> list[QPainterPath]:
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

        paths: list[QPainterPath] = []
        start = 0
        while start < x_valid.size:
            end = start
            while end + 1 < x_valid.size and x_valid[end + 1] == x_valid[end] + 1:
                end += 1
            segment = x_valid[start : end + 1]
            if segment.size == 1:
                x = int(segment[0])
                y = int(round(float(mean_y[x])))
                if 0 <= x < width and 0 <= y < height:
                    path = QPainterPath()
                    center_x = x + 0.5
                    y_pos = y + 0.5
                    path.moveTo(center_x - EndviewViewCorrosion._LEFT_EXTENSION_PX, y_pos)
                    path.lineTo(center_x + EndviewViewCorrosion._RIGHT_EXTENSION_PX, y_pos)
                    paths.append(path)
            elif segment.size >= 2:
                path = QPainterPath()
                x0 = int(segment[0])
                y0 = int(round(float(mean_y[x0])))
                y0 = max(0, min(height - 1, y0))
                path.moveTo(
                    (x0 + 0.5) - EndviewViewCorrosion._LEFT_EXTENSION_PX,
                    y0 + 0.5,
                )
                last_x = x0
                last_y = y0
                for x in segment[1:]:
                    xi = int(x)
                    yi = int(round(float(mean_y[xi])))
                    yi = max(0, min(height - 1, yi))
                    path.lineTo(xi + 0.5, yi + 0.5)
                    last_x = xi
                    last_y = yi
                path.lineTo(
                    (last_x + 0.5) + EndviewViewCorrosion._RIGHT_EXTENSION_PX,
                    last_y + 0.5,
                )
                paths.append(path)
            start = end + 1
        return paths
