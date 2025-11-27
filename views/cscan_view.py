"""Interactive C-Scan heatmap view."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class CScanView(QFrame):
    """Displays a (Z, X) projection, exposes LUT selection and slice picking."""

    crosshair_changed = pyqtSignal(int, int)
    slice_requested = pyqtSignal(int)
    colormap_changed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._projection: Optional[np.ndarray] = None
        self._value_range: Tuple[float, float] = (0.0, 1.0)

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.viewport().installEventFilter(self)
        self._view.setMouseTracking(True)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        pen = QPen(Qt.GlobalColor.white)
        pen.setWidth(1)
        self._cursor_h = QGraphicsLineItem()
        self._cursor_v = QGraphicsLineItem()
        self._cursor_h.setPen(pen)
        self._cursor_v.setPen(pen)
        self._cursor_h.setZValue(10)
        self._cursor_v.setZValue(10)
        self._scene.addItem(self._cursor_h)
        self._scene.addItem(self._cursor_v)

        # Header with status + LUT selection
        header = QHBoxLayout()
        self._status = QLabel("C-scan non disponible")
        self._status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        header.addWidget(self._status, 1)

        self._lut_combo = QComboBox()
        self._lut_combo.currentTextChanged.connect(self.colormap_changed.emit)
        header.addWidget(self._lut_combo, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addLayout(header)
        layout.addWidget(self._view, 1)

        self.setStyleSheet("background-color: #181818; color: #cccccc;")

    def set_projection(
        self,
        projection: np.ndarray,
        value_range: Optional[Tuple[float, float]] = None,
        colormaps: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """Display the projection (Z, X)."""
        if projection is None or projection.size == 0:
            self._projection = None
            self._status.setText("C-scan vide")
            self._pixmap_item.setPixmap(QPixmap())
            return

        self._projection = np.asarray(projection, dtype=np.float32)
        if value_range is None:
            value_range = (float(self._projection.min()), float(self._projection.max()))
        self._value_range = value_range
        self._status.setText(f"Z={projection.shape[0]} · X={projection.shape[1]}")
        self._render_pixmap()

        if colormaps:
            self._lut_combo.blockSignals(True)
            self._lut_combo.clear()
            self._lut_combo.addItems(colormaps)
            self._lut_combo.blockSignals(False)

    def highlight_slice(self, slice_idx: int) -> None:
        if self._projection is None:
            return
        slice_idx = max(0, min(self._projection.shape[0] - 1, slice_idx))
        self._update_cursor(slice_idx, self._projection.shape[1] // 2)

    def eventFilter(self, obj, event) -> bool:
        if obj is self._view.viewport() and self._projection is not None:
            if event.type() == QEvent.Type.MouseMove:
                coords = self._coords_from_event(event.position().toPoint())
                if coords:
                    z, x = coords
                    self._update_cursor(z, x)
                    self.crosshair_changed.emit(z, x)
                return True
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                coords = self._coords_from_event(event.position().toPoint())
                if coords:
                    z, _ = coords
                    self.slice_requested.emit(z)
                return True
        return super().eventFilter(obj, event)

    def _render_pixmap(self) -> None:
        heatmap = self._to_rgb(self._projection, self._value_range)
        h, w, _ = heatmap.shape
        image = QImage(heatmap.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap_item.setPixmap(QPixmap.fromImage(image.copy()))
        self._scene.setSceneRect(0, 0, w, h)

    @staticmethod
    def _to_rgb(data: np.ndarray, value_range: Tuple[float, float]) -> np.ndarray:
        vmin, vmax = value_range
        if vmax <= vmin:
            normalized = np.zeros_like(data, dtype=np.uint8)
        else:
            normalized = (data - vmin) / (vmax - vmin)
            normalized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        red = normalized
        blue = 255 - normalized
        green = (normalized // 2 + 64).clip(0, 255)
        return np.stack([red, green, blue], axis=-1)

    def _coords_from_event(self, pos) -> Optional[Tuple[int, int]]:
        if self._projection is None:
            return None
        scene_pos = self._view.mapToScene(pos)
        x = int(scene_pos.x())
        z = int(scene_pos.y())
        if 0 <= z < self._projection.shape[0] and 0 <= x < self._projection.shape[1]:
            return z, x
        return None

    def _update_cursor(self, z: int, x: int) -> None:
        self._cursor_h.setLine(0, z, self._projection.shape[1], z)
        self._cursor_v.setLine(x, 0, x, self._projection.shape[0])
