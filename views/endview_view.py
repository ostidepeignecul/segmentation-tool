"""Interactive endview renderer using QGraphicsView (zoom, pan, overlays)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QVBoxLayout,
)


@dataclass
class _PixmapBundle:
    """Stores the base and overlay pixmaps for quicker refresh."""

    base: Optional[QPixmap] = None
    overlay: Optional[QPixmap] = None


class EndviewView(QFrame):
    """Displays a slice of the NDE volume with basic interactions."""

    slice_changed = pyqtSignal(int)
    mouse_clicked = pyqtSignal(object, object)
    polygon_started = pyqtSignal(object)
    polygon_point_added = pyqtSignal(object)
    polygon_completed = pyqtSignal(object)
    rectangle_drawn = pyqtSignal(object)
    point_selected = pyqtSignal(object)
    drag_update = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._volume: Optional[np.ndarray] = None
        self._current_slice: int = 0
        self._overlay: Optional[np.ndarray] = None
        self._pixmaps = _PixmapBundle()

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._view.viewport().installEventFilter(self)
        self._view.setMouseTracking(True)

        self._image_item = QGraphicsPixmapItem()
        self._scene.addItem(self._image_item)
        self._overlay_item = QGraphicsPixmapItem()
        self._overlay_item.setOpacity(0.4)
        self._scene.addItem(self._overlay_item)

        pen = QPen(Qt.GlobalColor.red)
        self._crosshair_h = QGraphicsLineItem()
        self._crosshair_v = QGraphicsLineItem()
        self._crosshair_h.setPen(pen)
        self._crosshair_v.setPen(pen)
        self._crosshair_h.setZValue(10)
        self._crosshair_v.setZValue(10)
        self._scene.addItem(self._crosshair_h)
        self._scene.addItem(self._crosshair_v)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self.setStyleSheet("background-color: #202020; color: #bbbbbb;")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_volume(self, volume: np.ndarray) -> None:
        """Assign the oriented/normalized volume (shape: num_slices, H, W)."""
        if volume is None or volume.size == 0:
            self._volume = None
            self._image_item.setPixmap(QPixmap())
            self._overlay_item.setPixmap(QPixmap())
            return
        self._volume = np.asarray(volume)
        self._current_slice = min(self._current_slice, self._volume.shape[0] - 1)
        self._refresh_pixmaps()

    def set_slice(self, index: int) -> None:
        """Update the currently displayed slice index."""
        if self._volume is None:
            return
        index = int(max(0, min(self._volume.shape[0] - 1, index)))
        if index == self._current_slice:
            return
        self._current_slice = index
        self._refresh_pixmaps()

    def set_overlay(self, overlay: Optional[np.ndarray]) -> None:
        """Set an overlay (same shape as slice, values in [0,1])."""
        if overlay is None:
            self._overlay = None
            self._overlay_item.setPixmap(QPixmap())
            return
        self._overlay = np.asarray(overlay)
        self._refresh_overlay_pixmap()

    def update_image(self) -> None:
        """Force re-rendering the base slice."""
        self._refresh_pixmaps()

    # ------------------------------------------------------------------ #
    # Event handling
    # ------------------------------------------------------------------ #

    def eventFilter(self, obj: Any, event) -> bool:
        if obj is self._view.viewport():
            if isinstance(event, QMouseEvent):
                if event.type() == QMouseEvent.Type.MouseButtonPress:
                    return self._handle_mouse_press(event)
                if event.type() == QMouseEvent.Type.MouseMove:
                    self._handle_mouse_move(event)
        return super().eventFilter(obj, event)

    def wheelEvent(self, event) -> None:
        if not self._scene.items():
            return
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 1 / 1.15
        self._view.scale(factor, factor)

    def _handle_mouse_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        coords = self._scene_coords_from_event(event)
        if coords is None:
            return False
        self.point_selected.emit(coords)
        self.mouse_clicked.emit(coords, event.button())
        return True

    def _handle_mouse_move(self, event: QMouseEvent) -> None:
        coords = self._scene_coords_from_event(event)
        if coords is None:
            return
        x, y = coords
        self._update_crosshair(x, y)
        self.drag_update.emit(coords)

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #

    def _refresh_pixmaps(self) -> None:
        if self._volume is None:
            return
        slice_data = self._volume[self._current_slice]
        pixmap = self._array_to_pixmap(slice_data)
        self._pixmaps.base = pixmap
        self._image_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._image_item.boundingRect())
        self._refresh_overlay_pixmap()

    def _refresh_overlay_pixmap(self) -> None:
        if self._overlay is None or self._volume is None:
            self._overlay_item.setPixmap(QPixmap())
            return
        if self._overlay.ndim == 3:
            overlay_slice = self._overlay[self._current_slice]
        else:
            overlay_slice = self._overlay
        overlay_pixmap = self._mask_to_pixmap(overlay_slice)
        self._pixmaps.overlay = overlay_pixmap
        self._overlay_item.setPixmap(overlay_pixmap)

    @staticmethod
    def _array_to_pixmap(array: np.ndarray) -> QPixmap:
        data = np.asarray(array, dtype=np.float32)
        if data.size == 0:
            return QPixmap()
        min_val = float(data.min())
        max_val = float(data.max())
        if max_val <= min_val:
            normalized = np.zeros_like(data, dtype=np.uint8)
        else:
            normalized = (data - min_val) / (max_val - min_val)
            normalized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        h, w = normalized.shape
        qimage = QImage(
            normalized.data,
            w,
            h,
            w,
            QImage.Format.Format_Grayscale8,
        )
        return QPixmap.fromImage(qimage.copy())

    @staticmethod
    def _mask_to_pixmap(mask: np.ndarray) -> QPixmap:
        data = np.asarray(mask, dtype=np.float32)
        if data.size == 0:
            return QPixmap()
        normalized = np.clip(data, 0.0, 1.0)
        rgba = np.zeros((*normalized.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255  # red channel
        rgba[..., 3] = (normalized * 200).astype(np.uint8)
        h, w, _ = rgba.shape
        qimage = QImage(
            rgba.data,
            w,
            h,
            w * 4,
            QImage.Format.Format_RGBA8888,
        )
        return QPixmap.fromImage(qimage.copy())

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def _scene_coords_from_event(self, event: QMouseEvent) -> Optional[Tuple[int, int]]:
        if self._volume is None:
            return None
        scene_pos = self._view.mapToScene(event.position().toPoint())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        height, width = self._volume.shape[1:]
        if 0 <= x < width and 0 <= y < height:
            return (x, y)
        return None

    def _update_crosshair(self, x: int, y: int) -> None:
        if self._volume is None:
            return
        height, width = self._volume.shape[1:]
        self._crosshair_h.setLine(0, y, width, y)
        self._crosshair_v.setLine(x, 0, x, height)
