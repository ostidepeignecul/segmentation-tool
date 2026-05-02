"""Interactive C-Scan heatmap view."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from views.color_axis_ruler import ColorAxisRuler


class CScanView(QFrame):
    """Displays a (Z, X) projection, exposes LUT selection and slice picking."""

    crosshair_changed = pyqtSignal(int, int)
    slice_requested = pyqtSignal(int)
    colormap_changed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._projection: Optional[np.ndarray] = None
        self._value_range: Tuple[float, float] = (0.0, 1.0)
        self._current_crosshair: Optional[Tuple[int, int]] = None
        self._colormap_name: str = "Gris"
        self._colormap_lut: Optional[np.ndarray] = None
        self._value_scale_mm: Optional[float] = None
        self._panning: bool = False
        self._pan_last = QPointF()
        self._pan_center_scene: Optional[QPointF] = None
        self._display_size: Optional[Tuple[int, int]] = None
        self._zoom_factor: float = 1.0
        self._display_axis_x_name: str = ""
        self._display_axis_y_name: str = ""

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.viewport().installEventFilter(self)
        self._view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._view.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._view.setMouseTracking(True)
        self._default_view_min_size = self._view.minimumSize()
        self._default_view_max_size = self._view.maximumSize()
        self._default_self_min_size = self.minimumSize()
        self._default_self_max_size = self.maximumSize()

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(1)
        pen.setCosmetic(True)
        self._cursor_h = QGraphicsLineItem()
        self._cursor_v = QGraphicsLineItem()
        self._cursor_h.setPen(pen)
        self._cursor_v.setPen(pen)
        self._cursor_h.setZValue(10)
        self._cursor_v.setZValue(10)
        self._scene.addItem(self._cursor_h)
        self._scene.addItem(self._cursor_v)

        self._horizontal_ruler = ColorAxisRuler(Qt.Orientation.Horizontal, self)
        self._vertical_ruler = ColorAxisRuler(Qt.Orientation.Vertical, self)
        self._ruler_corner = QWidget(self)
        self._ruler_corner.setFixedSize(
            self._vertical_ruler.width(),
            self._horizontal_ruler.height(),
        )
        self._ruler_corner.setStyleSheet("background-color: #171717;")

        header = QHBoxLayout()
        self._header_layout = header
        self._status = QLabel("C-scan non disponible")
        self._status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        header.addWidget(self._status, 1)

        self._lut_combo = QComboBox()
        self._lut_combo.currentTextChanged.connect(self.colormap_changed.emit)
        header.addWidget(self._lut_combo, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addLayout(header)
        self._view_ruler_layout = QGridLayout()
        self._view_ruler_layout.setContentsMargins(0, 0, 0, 0)
        self._view_ruler_layout.setSpacing(0)
        self._view_ruler_layout.addWidget(self._vertical_ruler, 0, 0)
        self._view_ruler_layout.addWidget(self._view, 0, 1)
        self._view_ruler_layout.addWidget(self._ruler_corner, 1, 0)
        self._view_ruler_layout.addWidget(self._horizontal_ruler, 1, 1)
        self._view_ruler_layout.setColumnStretch(1, 1)
        self._view_ruler_layout.setRowStretch(0, 1)
        layout.addLayout(self._view_ruler_layout, 1)

        self.setStyleSheet("background-color: #181818; color: #cccccc;")
        self.set_ruler_axis_names(horizontal="", vertical="")

    def add_header_widget(self, widget: QWidget, stretch: int = 0) -> None:
        """Append a widget to the header row."""
        self._header_layout.addWidget(widget, int(stretch))

    def set_colormap(self, name: str, lut: Optional[np.ndarray]) -> None:
        """Set colormap to apply to the projection (lut shape (256,3) floats 0-1)."""
        self._colormap_name = str(name)
        if lut is not None and lut.shape == (256, 3):
            self._colormap_lut = np.asarray(lut, dtype=np.float32)
        else:
            self._colormap_lut = None
            self._colormap_name = "Gris"
        self._set_combo_current(self._colormap_name)
        self._render_pixmap()

    def set_ruler_axis_names(self, *, horizontal: str, vertical: str) -> None:
        """Display the X/Y axis names used by the pixel rulers."""
        self._display_axis_x_name = str(horizontal or "").strip()
        self._display_axis_y_name = str(vertical or "").strip()
        self._horizontal_ruler.set_axis_name(self._display_axis_x_name)
        self._vertical_ruler.set_axis_name(self._display_axis_y_name)

    def set_projection(
        self,
        projection: np.ndarray,
        value_range: Optional[Tuple[float, float]] = None,
        colormaps: Optional[Tuple[str, ...]] = None,
        value_scale_mm: Optional[float] = None,
    ) -> None:
        """Display the projection (Z, X)."""
        if projection is None or projection.size == 0:
            self._projection = None
            self._current_crosshair = None
            self._value_scale_mm = None
            self._panning = False
            self._pan_center_scene = None
            self._zoom_factor = 1.0
            self._status.setText("C-scan vide")
            self._pixmap_item.setPixmap(QPixmap())
            self._clear_rulers()
            return

        self._projection = np.asarray(projection, dtype=np.float32)
        self._value_scale_mm = value_scale_mm
        if value_range is None:
            value_range = (float(self._projection.min()), float(self._projection.max()))
        self._value_range = value_range
        self._panning = False
        self._pan_center_scene = None
        self._zoom_factor = 1.0
        self._current_crosshair = (
            self._projection.shape[0] // 2,
            self._projection.shape[1] // 2,
        )
        self._render_pixmap()
        self._update_cursor(*self._current_crosshair)

        if colormaps:
            self._lut_combo.blockSignals(True)
            self._lut_combo.clear()
            self._lut_combo.addItems(colormaps)
            self._lut_combo.blockSignals(False)
            self._set_combo_current(self._colormap_name)

    def highlight_slice(self, slice_idx: int) -> None:
        if self._projection is None:
            return
        slice_idx = max(0, min(self._projection.shape[0] - 1, slice_idx))
        _, last_x = self._current_crosshair or (
            slice_idx,
            self._projection.shape[1] // 2,
        )
        self._update_cursor(slice_idx, last_x)

    def set_crosshair(self, slice_idx: int, x: int) -> None:
        """Synchronize the crosshair position programmatically."""
        if self._projection is None:
            return
        z = max(0, min(self._projection.shape[0] - 1, int(slice_idx)))
        x_clamped = max(0, min(self._projection.shape[1] - 1, int(x)))
        self._update_cursor(z, x_clamped)

    def eventFilter(self, obj, event) -> bool:
        if obj is self._view.viewport() and self._projection is not None:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    coords = self._coords_from_event(event.position().toPoint())
                    if coords is not None:
                        z, x = coords
                        self._update_cursor(z, x)
                        self.crosshair_changed.emit(z, x)
                        self.slice_requested.emit(z)
                    return True
            if isinstance(event, QMouseEvent):
                if event.type() == QEvent.Type.MouseButtonPress:
                    if self._handle_pan_press(event):
                        return True
                elif event.type() == QEvent.Type.MouseMove:
                    if self._handle_pan_move(event):
                        return True
                elif event.type() == QEvent.Type.MouseButtonRelease:
                    if self._handle_pan_release(event):
                        return True
            if event.type() == QEvent.Type.Wheel:
                delta = event.angleDelta().y()
                if delta == 0:
                    return False
                zoom_in = delta > 0
                factor = 1.15 if zoom_in else 1 / 1.15
                self._apply_zoom(factor, event.position())
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def _render_pixmap(self) -> None:
        if self._projection is None:
            self._clear_rulers()
            return
        heatmap = self._to_rgb(self._projection, self._value_range, self._colormap_lut)
        heatmap = np.ascontiguousarray(heatmap, dtype=np.uint8)
        h, w, _ = heatmap.shape
        bytes_per_line = heatmap.strides[0]
        image = QImage(
            heatmap.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        self._pixmap_item.setPixmap(QPixmap.fromImage(image.copy()))
        if self._pan_center_scene is None:
            rect = self._pixmap_item.boundingRect()
            if not rect.isEmpty():
                self._pan_center_scene = rect.center()
        self._update_scene_padding()
        self._apply_display_scale()
        self._refresh_rulers()

    def _set_combo_current(self, name: str) -> None:
        idx = self._lut_combo.findText(name)
        if idx >= 0:
            self._lut_combo.blockSignals(True)
            self._lut_combo.setCurrentIndex(idx)
            self._lut_combo.blockSignals(False)

    @staticmethod
    def _to_rgb(data: np.ndarray, value_range: Tuple[float, float], lut: Optional[np.ndarray]) -> np.ndarray:
        vmin, vmax = value_range
        if vmax <= vmin:
            normalized = np.zeros_like(data, dtype=np.float32)
        else:
            normalized = (data - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0.0, 1.0)
        idx = (normalized * 255.0).astype(np.uint8)
        if lut is None:
            gray = (normalized * 255.0).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)
        rgb = (lut[idx] * 255.0).astype(np.uint8)
        return rgb

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
        if self._projection is None:
            return
        max_z = self._projection.shape[0] - 1
        max_x = self._projection.shape[1] - 1
        z_clamped = max(0, min(max_z, z))
        x_clamped = max(0, min(max_x, x))
        self._cursor_h.setLine(0, z_clamped, self._projection.shape[1], z_clamped)
        self._cursor_v.setLine(x_clamped, 0, x_clamped, self._projection.shape[0])
        self._current_crosshair = (z_clamped, x_clamped)
        self._update_status(z_clamped, x_clamped)

    def set_cross_visible(self, visible: bool) -> None:
        """Show or hide the crosshair lines."""
        self._cursor_h.setVisible(visible)
        self._cursor_v.setVisible(visible)

    def get_display_size(self) -> Tuple[int, int]:
        """Return the requested display size, defaults to current viewport size."""
        if self._display_size is not None:
            return self._display_size
        size = self._view.viewport().size()
        return size.width(), size.height()

    def set_display_size(self, width: int, height: int) -> None:
        """Apply visual stretch on the C-scan view without changing underlying data."""
        width = max(1, int(width))
        height = max(1, int(height))
        self._display_size = (width, height)
        self._view.setMinimumSize(self._default_view_min_size)
        self._view.setMaximumSize(self._default_view_max_size)
        self.setMinimumSize(self._default_self_min_size)
        self.setMaximumSize(self._default_self_max_size)
        self.updateGeometry()
        self._apply_display_scale()

    def reset_display_size(self) -> None:
        """Reset visual stretch, zoom, and pan to defaults."""
        self._display_size = None
        self._zoom_factor = 1.0
        self._panning = False
        self._pan_center_scene = None
        self._view.setMinimumSize(self._default_view_min_size)
        self._view.setMaximumSize(self._default_view_max_size)
        self.setMinimumSize(self._default_self_min_size)
        self.setMaximumSize(self._default_self_max_size)
        self.updateGeometry()
        if self._projection is not None:
            rect = self._pixmap_item.boundingRect()
            if not rect.isEmpty():
                self._pan_center_scene = rect.center()
                self._apply_view_transform()
                self._update_scene_padding()
                self._refresh_rulers()
                return
        self._view.resetTransform()
        self._clear_rulers()

    def _apply_display_scale(self) -> None:
        rect = self._pixmap_item.boundingRect()
        if rect.isEmpty():
            return
        self._apply_view_transform()
        self._update_scene_padding()

    def _display_scale(self) -> Tuple[float, float]:
        if self._display_size is None:
            return (1.0, 1.0)
        rect = self._pixmap_item.boundingRect()
        if rect.isEmpty():
            return (1.0, 1.0)
        target_w, target_h = self._display_size
        base_w = max(1.0, rect.width())
        base_h = max(1.0, rect.height())
        return (float(target_w) / base_w, float(target_h) / base_h)

    def _current_scale(self) -> Tuple[float, float]:
        scale_x, scale_y = self._display_scale()
        scale_x *= self._zoom_factor
        scale_y *= self._zoom_factor
        return (scale_x, scale_y)

    def _ensure_pan_center(self) -> QPointF:
        if self._pan_center_scene is None:
            rect = self._pixmap_item.boundingRect()
            if not rect.isEmpty():
                self._pan_center_scene = rect.center()
            else:
                self._pan_center_scene = self._view.mapToScene(self._view.viewport().rect().center())
        return self._pan_center_scene

    def _apply_view_transform(self) -> None:
        scale_x, scale_y = self._current_scale()
        if abs(scale_x) < 1e-6 or abs(scale_y) < 1e-6:
            return
        center_scene = self._ensure_pan_center()
        self._view.resetTransform()
        self._view.scale(scale_x, scale_y)
        self._view.centerOn(center_scene)
        self._refresh_rulers()

    def _update_scene_padding(self) -> None:
        rect = self._pixmap_item.boundingRect()
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

    def _apply_zoom(self, factor: float, anchor_pos: Optional[QPointF] = None) -> None:
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
        return QPointF(self._view.viewport().rect().center())

    def _handle_pan_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.RightButton or self._projection is None:
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
        self._pan_center_scene = QPointF(
            self._pan_center_scene.x() - (delta.x() / scale_x),
            self._pan_center_scene.y() - (delta.y() / scale_y),
        )
        self._apply_view_transform()
        event.accept()
        return True

    def _handle_pan_release(self, event: QMouseEvent) -> bool:
        if not self._panning or event.button() != Qt.MouseButton.RightButton:
            return False
        self._panning = False
        self._view.setCursor(Qt.CursorShape.ArrowCursor)
        event.accept()
        return True

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._projection is None:
            self._clear_rulers()
            return
        self._apply_view_transform()
        self._update_scene_padding()
        self._refresh_rulers()

    def _clear_rulers(self) -> None:
        self._horizontal_ruler.clear_range()
        self._vertical_ruler.clear_range()

    def _refresh_rulers(self) -> None:
        if self._projection is None:
            self._clear_rulers()
            return

        visible_rect = self._visible_scene_rect()
        if visible_rect is None or visible_rect.isEmpty():
            self._clear_rulers()
            return

        height, width = self._projection.shape
        self._horizontal_ruler.set_view_range(
            view_min=visible_rect.left(),
            view_max=visible_rect.right(),
            content_min=0.0,
            content_max=float(max(0, width - 1)),
        )
        self._vertical_ruler.set_view_range(
            view_min=visible_rect.top(),
            view_max=visible_rect.bottom(),
            content_min=0.0,
            content_max=float(max(0, height - 1)),
        )

    def _visible_scene_rect(self) -> Optional[QRectF]:
        viewport_rect = self._view.viewport().rect()
        if viewport_rect.isEmpty():
            return None
        top_left = self._view.mapToScene(viewport_rect.topLeft())
        bottom_right = self._view.mapToScene(viewport_rect.bottomRight())
        return QRectF(top_left, bottom_right).normalized()

    def _update_status(self, z: int, x: int) -> None:
        """Refresh header label with current coordinates and pixel value."""
        if self._projection is None:
            self._status.setText("C-scan non disponible")
            return
        try:
            value = float(self._projection[z, x])
        except Exception:
            value = float("nan")

        text_value = "-"
        if np.isfinite(value):
            if self._value_scale_mm is not None:
                mm = value * float(self._value_scale_mm)
                text_value = f"{mm:.2f} mm ({value:.2f} px)"
            else:
                text_value = f"{value:.2f} px"

        self._status.setText(f"Z={z} | X={x} | dist={text_value}")
