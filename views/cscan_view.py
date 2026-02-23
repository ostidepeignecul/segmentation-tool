"""Interactive C-Scan heatmap view."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPen, QPixmap, QTransform
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
        self._current_crosshair: Optional[Tuple[int, int]] = None
        self._colormap_name: str = "Gris"
        self._colormap_lut: Optional[np.ndarray] = None
        self._value_scale_mm: Optional[float] = None
        self._panning: bool = False
        self._pan_last = QPointF()
        self._display_size: Optional[Tuple[int, int]] = None

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.viewport().installEventFilter(self)
        self._view.setMouseTracking(True)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
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
            self._status.setText("C-scan vide")
            self._pixmap_item.setPixmap(QPixmap())
            return

        self._projection = np.asarray(projection, dtype=np.float32)
        self._value_scale_mm = value_scale_mm
        if value_range is None:
            value_range = (float(self._projection.min()), float(self._projection.max()))
        self._value_range = value_range
        self._view.setTransform(QTransform())  # reset zoom when updating projection
        self._current_crosshair = (
            projection.shape[0] // 2,
            projection.shape[1] // 2,
        )
        self._render_pixmap()
        self._apply_display_scale()
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
                    if coords:
                        z, x = coords
                        self._update_cursor(z, x)
                        self.crosshair_changed.emit(z, x)
                        self.slice_requested.emit(z)
                    return True
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
                if not self._scene.items():
                    return False
                self._view.setFocus(Qt.FocusReason.MouseFocusReason)
                self._panning = True
                self._pan_last = event.position()
                self._view.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return True
            if event.type() == QEvent.Type.MouseMove and self._panning:
                delta = event.position() - self._pan_last
                self._pan_last = event.position()
                hbar = self._view.horizontalScrollBar()
                vbar = self._view.verticalScrollBar()
                hbar.setValue(hbar.value() - int(delta.x()))
                vbar.setValue(vbar.value() - int(delta.y()))
                event.accept()
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and self._panning:
                if event.button() == Qt.MouseButton.RightButton:
                    self._panning = False
                    self._view.setCursor(Qt.CursorShape.ArrowCursor)
                    event.accept()
                    return True
            if event.type() == QEvent.Type.Wheel:
                delta = event.angleDelta().y()
                if delta == 0:
                    return False
                zoom_in = delta > 0
                factor = 1.15 if zoom_in else 1 / 1.15
                self._view.scale(factor, factor)
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def _render_pixmap(self) -> None:
        if self._projection is None:
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
        self._scene.setSceneRect(0, 0, w, h)

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
        """Reset visual stretch and return to default transform."""
        self._display_size = None
        self._view.setMinimumSize(self._default_view_min_size)
        self._view.setMaximumSize(self._default_view_max_size)
        self.setMinimumSize(self._default_self_min_size)
        self.setMaximumSize(self._default_self_max_size)
        self.updateGeometry()
        self._view.setTransform(QTransform())
        if self._projection is not None:
            self._view.centerOn(self._pixmap_item)

    def _apply_display_scale(self) -> None:
        if self._display_size is None:
            return
        rect = self._pixmap_item.boundingRect()
        if rect.isEmpty():
            return
        target_w, target_h = self._display_size
        base_w = max(1.0, rect.width())
        base_h = max(1.0, rect.height())
        scale_x = float(target_w) / base_w
        scale_y = float(target_h) / base_h
        if abs(scale_x) < 1e-6 or abs(scale_y) < 1e-6:
            return
        self._view.setTransform(QTransform())
        self._view.scale(scale_x, scale_y)
        self._view.centerOn(rect.center())

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

        self._status.setText(f"Z={z} · X={x} · dist={text_value}")
