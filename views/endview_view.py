"""Interactive endview renderer using QGraphicsView (zoom, pan, overlays)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Mapping

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

from models.overlay_data import OverlayData


@dataclass
class _PixmapBundle:
    """Stores the base and overlay pixmaps for quicker refresh."""

    base: Optional[QPixmap] = None
    overlay: Optional[QPixmap] = None


class EndviewView(QFrame):
    """Displays a slice of the NDE volume with basic interactions."""

    slice_changed = pyqtSignal(int)
    mouse_clicked = pyqtSignal(object, object)
    freehand_started = pyqtSignal(object)
    freehand_point_added = pyqtSignal(object)
    freehand_completed = pyqtSignal(object)
    box_drawn = pyqtSignal(object)
    point_selected = pyqtSignal(object)
    drag_update = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._volume: Optional[np.ndarray] = None
        self._current_slice: int = 0
        self._label_volumes: Dict[int, np.ndarray] = {}
        self._overlay_palette: Dict[int, Tuple[int, int, int, int]] = {}
        self._visible_labels: Optional[set[int]] = None
        self._pixmaps = _PixmapBundle()

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.viewport().installEventFilter(self)
        self._view.installEventFilter(self)
        self._view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._view.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._view.setMouseTracking(True)
        self._panning: bool = False
        self._pan_last = QPointF()

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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocusProxy(self._view)

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

    def set_overlay(
        self,
        overlay: Optional[OverlayData],
        *,
        visible_labels: Optional[set[int]] = None,
    ) -> None:
        """Set an overlay using per-label volumes/palette."""
        if overlay is None:
            self._label_volumes = {}
            self._visible_labels = None
            self._overlay_palette = {}
            self._overlay_item.setPixmap(QPixmap())
            return
        self._label_volumes = {
            int(label): np.asarray(vol) for label, vol in overlay.label_volumes.items()
        }
        self._overlay_palette = dict(overlay.palette)
        self._visible_labels = set(visible_labels) if visible_labels is not None else None
        self._refresh_overlay_pixmap()

    def update_image(self) -> None:
        """Force re-rendering the base slice."""
        self._refresh_pixmaps()

    def set_crosshair(self, x: int, y: int) -> None:
        """Expose crosshair updates to controllers for synchronized highlighting."""
        if self._volume is None:
            return
        height, width = self._volume.shape[1:]
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        self._update_crosshair(x, y)

    # ------------------------------------------------------------------ #
    # Event handling
    # ------------------------------------------------------------------ #

    def eventFilter(self, obj: Any, event) -> bool:
        if obj is self._view.viewport():
            if isinstance(event, QMouseEvent):
                if event.type() == QMouseEvent.Type.MouseButtonPress:
                    if self._handle_pan_press(event):
                        return True
                    return self._handle_mouse_press(event)
                if event.type() == QMouseEvent.Type.MouseMove:
                    if self._handle_pan_move(event):
                        return True
                    return self._handle_mouse_move(event)
                if event.type() == QMouseEvent.Type.MouseButtonRelease:
                    if self._handle_pan_release(event):
                        return True
        return super().eventFilter(obj, event)

    def wheelEvent(self, event) -> None:
        if not self._scene.items():
            return
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 1 / 1.15
        self._view.scale(factor, factor)
        event.accept()

    def _handle_mouse_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            return False
        coords = self._scene_coords_from_event(event)
        if coords is None:
            return False
        self._view.setFocus(Qt.FocusReason.MouseFocusReason)
        x, y = coords
        self._update_crosshair(x, y)
        self.point_selected.emit(coords)
        self.mouse_clicked.emit(coords, event.button())
        return True

    def _handle_mouse_move(self, event: QMouseEvent) -> bool:
        coords = self._scene_coords_from_event(event)
        if coords is None:
            return False
        self.drag_update.emit(coords)
        return False

    # ------------------------------------------------------------------ #
    # Panning helpers (right-click)
    # ------------------------------------------------------------------ #
    def _handle_pan_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.RightButton:
            return False
        if not self._scene.items():
            return False
        self._view.setFocus(Qt.FocusReason.MouseFocusReason)
        self._panning = True
        self._pan_last = event.position()
        self._view.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()
        return True

    def _handle_pan_move(self, event: QMouseEvent) -> bool:
        if not self._panning:
            return False
        delta = event.position() - self._pan_last
        self._pan_last = event.position()
        hbar = self._view.horizontalScrollBar()
        vbar = self._view.verticalScrollBar()
        hbar.setValue(hbar.value() - int(delta.x()))
        vbar.setValue(vbar.value() - int(delta.y()))
        event.accept()
        return True

    def _handle_pan_release(self, event: QMouseEvent) -> bool:
        if not self._panning:
            return False
        if event.button() != Qt.MouseButton.RightButton:
            return False
        self._panning = False
        self._view.setCursor(Qt.CursorShape.ArrowCursor)
        event.accept()
        return True

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
        if not self._label_volumes or self._volume is None:
            self._overlay_item.setPixmap(QPixmap())
            return
        overlay_slice = self._compose_slice_rgba(self._current_slice)
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
        normalized = np.ascontiguousarray(normalized, dtype=np.uint8)
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
        data = np.asarray(mask)
        if data.size == 0:
            return QPixmap()
        # Support soit une mask 2D (valeurs 0..1/uint8), soit une image RGBA prête (H,W,4)
        if data.ndim == 2:
            normalized = np.clip(data.astype(np.float32), 0.0, 1.0)
            rgba = np.zeros((*normalized.shape, 4), dtype=np.uint8)
            rgba[..., 0] = 255  # red
            rgba[..., 3] = (normalized * 200).astype(np.uint8)
        elif data.ndim == 3 and data.shape[2] == 4:
            rgba = data.astype(np.uint8, copy=False)
        else:
            return QPixmap()
        h, w, _ = rgba.shape
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        qimage = QImage(
            rgba.data,
            w,
            h,
            w * 4,
            QImage.Format.Format_RGBA8888,
        )
        return QPixmap.fromImage(qimage.copy())

    @staticmethod
    def _colorize_overlay_slice(
        mask_slice: np.ndarray, palette: Dict[int, Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Transforme une slice de labels en image RGBA via la palette BGRA."""
        labels = np.asarray(mask_slice, dtype=np.uint8)
        if labels.ndim != 2:
            return np.zeros((0, 0, 4), dtype=np.uint8)
        rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
        for cls_value in np.unique(labels):
            cls_int = int(cls_value)
            if cls_int == 0:
                continue
            b, g, r, a = palette.get(cls_int, (255, 0, 255, 160))
            rgba[labels == cls_int] = (r, g, b, a)
        return rgba

    def _compose_slice_rgba(self, index: int) -> np.ndarray:
        """Compose une slice RGBA à partir des volumes par label et des labels visibles."""
        if self._volume is None:
            return np.zeros((0, 0, 4), dtype=np.uint8)
        depth, height, width = self._volume.shape[:3]
        if index < 0 or index >= depth:
            return np.zeros((0, 0, 4), dtype=np.uint8)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        labels_to_draw = (
            self._visible_labels if self._visible_labels is not None else self._label_volumes.keys()
        )
        for label in labels_to_draw:
            vol = self._label_volumes.get(label)
            if vol is None or vol.shape[0] <= index:
                continue
            slice_alpha = vol[index]
            if slice_alpha.ndim != 2 or not np.any(slice_alpha):
                continue
            mask = slice_alpha > 0
            if not np.any(mask):
                continue
            b, g, r, a = self._overlay_palette.get(label, (255, 0, 255, 160))
            rgba_slice = np.zeros((height, width, 4), dtype=np.uint8)
            rgba_slice[..., 0] = r
            rgba_slice[..., 1] = g
            rgba_slice[..., 2] = b
            rgba_slice[..., 3] = np.clip(slice_alpha * (a / 255.0) * 255.0, 0, 255).astype(
                np.uint8
            )
            rgba[mask] = rgba_slice[mask]
        return rgba

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

    def set_cross_visible(self, visible: bool) -> None:
        """Show or hide the crosshair lines."""
        self._crosshair_h.setVisible(visible)
        self._crosshair_v.setVisible(visible)

    def _emit_slice_scroll(self, delta: int) -> None:
        if self._volume is None or delta == 0:
            return
        step = -1 if delta > 0 else 1
        new_index = max(0, min(self._volume.shape[0] - 1, self._current_slice + step))
        if new_index != self._current_slice:
            self.slice_changed.emit(new_index)
