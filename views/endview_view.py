"""Interactive endview renderer using QGraphicsView (zoom, pan, overlays)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Mapping

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, Qt, pyqtSignal
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
        self._mask_volume: Optional[np.ndarray] = None  # New single mask volume
        self._overlay_palette: Dict[int, Tuple[int, int, int, int]] = {}
        self._visible_labels: Optional[set[int]] = None
        self._colormap_name: str = "Gris"
        self._colormap_lut: Optional[np.ndarray] = None
        self._pixmaps = _PixmapBundle()
        self._display_size: Optional[Tuple[int, int]] = None
        self._zoom_factor: float = 1.0

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.viewport().installEventFilter(self)
        self._view.installEventFilter(self)
        self._view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._view.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._view.setMouseTracking(True)
        self._panning: bool = False
        self._pan_last = QPointF()
        self._pan_center_scene: Optional[QPointF] = None
        self._default_view_min_size = self._view.minimumSize()
        self._default_view_max_size = self._view.maximumSize()
        self._default_self_min_size = self.minimumSize()
        self._default_self_max_size = self.maximumSize()

        self._image_item = QGraphicsPixmapItem()
        self._nde_opacity: float = 1.0
        self._image_item.setOpacity(self._nde_opacity)
        self._scene.addItem(self._image_item)
        self._overlay_opacity: float = 1.0
        self._overlay_item = QGraphicsPixmapItem()
        self._overlay_item.setOpacity(self._overlay_opacity)
        self._scene.addItem(self._overlay_item)

        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(1)
        pen.setCosmetic(True)
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
    def set_colormap(self, name: str, lut: Optional[np.ndarray]) -> None:
        """Set the base image colormap (expects lut shape (256,3) floats 0-1)."""
        self._colormap_name = str(name)
        if lut is not None and lut.shape == (256, 3):
            self._colormap_lut = np.asarray(lut, dtype=np.float32)
        else:
            self._colormap_lut = None
        self._refresh_pixmaps()

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
        """Set an overlay using full mask volume and palette (LUT)."""
        if overlay is None:
            self._mask_volume = None
            self._overlay_palette = {}
            self._visible_labels = None
            self._overlay_item.setPixmap(QPixmap())
            return
        
        self._mask_volume = overlay.mask_volume
        self._overlay_palette = dict(overlay.palette)
        self._visible_labels = set(visible_labels) if visible_labels is not None else None
        self._refresh_overlay_pixmap()

    def set_overlay_opacity(self, opacity: float) -> None:
        """Set global overlay opacity (0.0 - 1.0)."""
        try:
            value = float(opacity)
        except (TypeError, ValueError):
            value = 1.0
        self._overlay_opacity = max(0.0, min(1.0, value))
        self._overlay_item.setOpacity(self._overlay_opacity)

    def set_nde_opacity(self, opacity: float) -> None:
        """Set base NDE slice opacity (0.0 - 1.0)."""
        try:
            value = float(opacity)
        except (TypeError, ValueError):
            value = 1.0
        self._nde_opacity = max(0.0, min(1.0, value))
        self._image_item.setOpacity(self._nde_opacity)

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
            if event.type() == QEvent.Type.Wheel:
                if not self._scene.items():
                    return False
                zoom_in = event.angleDelta().y() > 0
                factor = 1.15 if zoom_in else 1 / 1.15
                self._apply_zoom(factor, event.position())
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def wheelEvent(self, event) -> None:
        if not self._scene.items():
            return
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 1 / 1.15
        anchor = self._view.viewport().mapFrom(self, event.position().toPoint())
        self._apply_zoom(factor, QPointF(anchor))
        event.accept()

    def _handle_mouse_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        coords = self._scene_coords_from_event(event)
        if coords is None:
            return False
        self._view.setFocus(Qt.FocusReason.MouseFocusReason)
        x, y = coords
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Shift+clic = crosshair uniquement
            self._update_crosshair(x, y)
            self.point_selected.emit(coords)
        else:
            # Clic gauche normal = interaction annotation (grow, etc.)
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
        self._ensure_pan_center()
        self._view.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()
        return True

    def _handle_pan_move(self, event: QMouseEvent) -> bool:
        if not self._panning:
            return False
        delta = event.position() - self._pan_last
        self._pan_last = event.position()
        self._ensure_pan_center()
        scale_x, scale_y = self._current_scale()
        if abs(scale_x) < 1e-6:
            scale_x = 1.0
        if abs(scale_y) < 1e-6:
            scale_y = 1.0
        delta_x = delta.x() / scale_x
        delta_y = delta.y() / scale_y
        self._pan_center_scene = QPointF(
            self._pan_center_scene.x() - delta_x,
            self._pan_center_scene.y() - delta_y,
        )
        self._apply_view_transform()
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
        if self._pan_center_scene is None:
            rect = self._image_item.boundingRect()
            if not rect.isEmpty():
                self._pan_center_scene = rect.center()
        self._update_scene_padding()
        self._apply_display_scale()
        self._refresh_overlay_pixmap()

    def _refresh_overlay_pixmap(self) -> None:
        if self._mask_volume is None or self._volume is None:
            self._overlay_item.setPixmap(QPixmap())
            return
        overlay_slice = self._compose_slice_rgba(self._current_slice)
        overlay_pixmap = self._mask_to_pixmap(overlay_slice)
        self._pixmaps.overlay = overlay_pixmap
        self._overlay_item.setPixmap(overlay_pixmap)

    @staticmethod
    def _array_to_pixmap_gray(array: np.ndarray) -> QPixmap:
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

    def _array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        data = np.asarray(array, dtype=np.float32)
        if self._colormap_lut is None:
            return self._array_to_pixmap_gray(data)
        if data.size == 0:
            return QPixmap()
        min_val = float(data.min())
        max_val = float(data.max())
        if max_val <= min_val:
            idx = np.zeros_like(data, dtype=np.uint8)
        else:
            normalized = (data - min_val) / (max_val - min_val)
            idx = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        rgb = (self._colormap_lut[idx] * 255.0).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w, _ = rgb.shape
        qimage = QImage(
            rgb.data,
            w,
            h,
            w * 3,
            QImage.Format.Format_RGB888,
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
        """Compose une slice RGBA à partir du masque uint8 et de la palette (LUT)."""
        if self._mask_volume is None:
            return np.zeros((0, 0, 4), dtype=np.uint8)

        depth, height, width = self._mask_volume.shape
        if index < 0 or index >= depth:
            return np.zeros((0, 0, 4), dtype=np.uint8)

        # Build LUT (256 entries for uint8)
        # We recreate it here for simplicity, but could cache it. 
        # Given it's 256 it's negligible.
        lut = np.zeros((256, 4), dtype=np.uint8)

        labels_to_draw = (
            self._visible_labels if self._visible_labels is not None
            else self._overlay_palette.keys()
        )

        for label in labels_to_draw:
            if not (0 < label < 256):
                continue
            color = self._overlay_palette.get(label)
            if color:
                b, g, r, a = color
                # Apply alpha scaling logic if needed, or just use palette alpha
                # The original code did: np.clip(slice_alpha * (a / 255.0) * 255.0, 0, 255)
                # Since our mask is binary per label (it's an ID), alpha is just 'a'.
                lut[label] = [r, g, b, a]

        # Get slice (H, W)
        try:
            slice_indices = self._mask_volume[index]
        except IndexError:
             return np.zeros((height, width, 4), dtype=np.uint8)

        # Apply LUT -> (H, W, 4)
        rgba = lut[slice_indices]
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

    def get_display_size(self) -> Tuple[int, int]:
        """Return the requested display size (viewport), defaults to current if unset."""
        if self._display_size is not None:
            return self._display_size
        size = self._view.viewport().size()
        return size.width(), size.height()

    def set_display_size(self, width: int, height: int) -> None:
        """Resize and scale the view so the image is visually stretched to the given size."""
        width = max(1, int(width))
        height = max(1, int(height))
        self._display_size = (width, height)
        # Only change the visual scale; do not resize the container widget.
        self._view.setMinimumSize(self._default_view_min_size)
        self._view.setMaximumSize(self._default_view_max_size)
        self.setMinimumSize(self._default_self_min_size)
        self.setMaximumSize(self._default_self_max_size)
        self.updateGeometry()
        self._apply_display_scale()

    def reset_display_size(self) -> None:
        """Reset display size, zoom, and pan to defaults."""
        self._display_size = None
        self._zoom_factor = 1.0
        self._panning = False
        self._pan_center_scene = None
        self._view.setMinimumSize(self._default_view_min_size)
        self._view.setMaximumSize(self._default_view_max_size)
        self.setMinimumSize(self._default_self_min_size)
        self.setMaximumSize(self._default_self_max_size)
        self.updateGeometry()
        if not self._scene.sceneRect().isEmpty():
            self._pan_center_scene = self._scene.sceneRect().center()
            self._apply_view_transform()
        else:
            self._view.resetTransform()
        if hasattr(self, "_apply_tool_cursor"):
            try:
                self._apply_tool_cursor()
            except Exception:
                pass

    def _apply_display_scale(self) -> None:
        """Apply a transform so the scene fills the requested display size (can deform)."""
        if self._display_size is None:
            return
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            return
        self._apply_view_transform()
        self._update_scene_padding()

    def _apply_zoom(self, factor: float, anchor_pos: Optional[QPointF] = None) -> None:
        """Scale view while tracking user zoom so resize keeps proportion."""
        if factor == 0:
            return
        self._ensure_pan_center()
        if anchor_pos is None:
            anchor_pos = self._viewport_center()
        anchor_scene = self._view.mapToScene(anchor_pos.toPoint())
        self._zoom_factor *= factor
        scale_x, scale_y = self._current_scale()
        if abs(scale_x) < 1e-6:
            scale_x = 1.0
        if abs(scale_y) < 1e-6:
            scale_y = 1.0
        view_center = self._viewport_center()
        new_center_x = anchor_scene.x() - (anchor_pos.x() - view_center.x()) / scale_x
        new_center_y = anchor_scene.y() - (anchor_pos.y() - view_center.y()) / scale_y
        self._pan_center_scene = QPointF(new_center_x, new_center_y)
        self._update_scene_padding()
        self._apply_view_transform()

    def _viewport_center(self) -> QPointF:
        rect = self._view.viewport().rect()
        return QPointF(rect.center())

    def _display_scale(self) -> Tuple[float, float]:
        if self._display_size is None:
            return (1.0, 1.0)
        rect = self._image_item.boundingRect()
        if rect.isEmpty():
            return (1.0, 1.0)
        base_w = max(1.0, rect.width())
        base_h = max(1.0, rect.height())
        target_w, target_h = self._display_size
        return (float(target_w) / base_w, float(target_h) / base_h)

    def _current_scale(self) -> Tuple[float, float]:
        scale_x, scale_y = self._display_scale()
        scale_x *= self._zoom_factor
        scale_y *= self._zoom_factor
        return (scale_x, scale_y)

    def _ensure_pan_center(self) -> QPointF:
        if self._pan_center_scene is None:
            rect = self._image_item.boundingRect()
            if not rect.isEmpty():
                self._pan_center_scene = rect.center()
            else:
                view_center = self._viewport_center()
                self._pan_center_scene = self._view.mapToScene(view_center.toPoint())
        return self._pan_center_scene

    def _apply_view_transform(self) -> None:
        scale_x, scale_y = self._current_scale()
        if abs(scale_x) < 1e-6 or abs(scale_y) < 1e-6:
            return
        center_scene = self._ensure_pan_center()
        self._view.resetTransform()
        self._view.scale(scale_x, scale_y)
        self._view.centerOn(center_scene)

    def _update_scene_padding(self) -> None:
        rect = self._image_item.boundingRect()
        if rect.isEmpty():
            return
        view_size = self._view.viewport().size()
        if view_size.isEmpty():
            self._scene.setSceneRect(rect)
            return
        scale_x, scale_y = self._current_scale()
        if abs(scale_x) < 1e-6 or abs(scale_y) < 1e-6:
            self._scene.setSceneRect(rect)
            return
        pad_x = view_size.width() / scale_x
        pad_y = view_size.height() / scale_y
        padded = rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)
        self._scene.setSceneRect(padded)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._scene.items():
            return
        self._apply_view_transform()
        self._update_scene_padding()

    def _emit_slice_scroll(self, delta: int) -> None:
        if self._volume is None or delta == 0:
            return
        step = -1 if delta > 0 else 1
        new_index = max(0, min(self._volume.shape[0] - 1, self._current_slice + step))
        if new_index != self._current_slice:
            self.slice_changed.emit(new_index)
