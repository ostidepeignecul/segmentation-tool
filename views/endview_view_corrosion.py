"""Corrosion-specific Endview with cosmetic interpolated lines."""

from __future__ import annotations

from typing import Optional
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsPathItem

from models.overlay_data import OverlayData, OverlayStackData
from views.endview_view import EndviewView


class EndviewViewCorrosion(EndviewView):
    """Render corrosion overlays as cosmetic lines independent from zoom."""

    profile_drag_started = pyqtSignal(object)
    profile_drag_moved = pyqtSignal(object)
    profile_drag_finished = pyqtSignal(object)
    profile_double_clicked = pyqtSignal(object)

    _COSMETIC_LINE_WIDTH = 5
    _ANCHOR_SIZE_PX = 1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cosmetic_line_items: list[QGraphicsPathItem] = []
        self._anchor_items: list[QGraphicsEllipseItem] = []
        self._overlay_item.setPixmap(QPixmap())
        self._anchor_pen_active = QPen(QColor(255, 230, 90, 230))
        self._anchor_pen_active.setWidth(1)
        self._anchor_pen_active.setCosmetic(True)
        self._anchor_pen_inactive = QPen(QColor(180, 180, 180, 180))
        self._anchor_pen_inactive.setWidth(1)
        self._anchor_pen_inactive.setCosmetic(True)

    def set_overlay(
        self,
        overlay: Optional[OverlayData],
        *,
        visible_labels: Optional[set[int]] = None,
    ) -> None:
        super().set_overlay(overlay, visible_labels=visible_labels)
        if overlay is None:
            self._clear_cosmetic_lines()
            self.clear_anchor_points()

    def set_overlay_stack(self, overlay_stack: Optional[OverlayStackData]) -> None:
        super().set_overlay_stack(overlay_stack)
        if overlay_stack is None or not overlay_stack.layers:
            self._clear_cosmetic_lines()
            self.clear_anchor_points()

    def set_overlay_opacity(self, opacity: float) -> None:
        super().set_overlay_opacity(opacity)
        if self._overlay_stack is not None and self._volume is not None:
            self._refresh_overlay_pixmap()

    def _refresh_overlay_pixmap(self) -> None:
        """Build cosmetic line items instead of a scaled RGBA pixmap."""
        self._overlay_item.setPixmap(QPixmap())
        self._clear_cosmetic_lines()
        if self._overlay_stack is None or not self._overlay_stack.layers or self._volume is None:
            return
        for layer_index, layer in enumerate(self._overlay_stack.layers):
            overlay = layer.overlay
            if overlay is None:
                continue
            payload = self._overlay_vectorization_service.build_layer_payload(
                layer,
                slice_idx=self._current_slice,
            )
            if payload is None or not payload.paths:
                continue
            for path_data in payload.paths:
                pen = self._pen_for_label(
                    path_data.label,
                    palette=overlay.palette,
                    layer_opacity=float(layer.opacity),
                )
                item = QGraphicsPathItem(self._qpath_from_points(path_data.points))
                item.setPen(pen)
                item.setZValue(6.0 + (layer_index * 0.01))
                self._scene.addItem(item)
                self._cosmetic_line_items.append(item)

    def _clear_cosmetic_lines(self) -> None:
        for item in self._cosmetic_line_items:
            self._scene.removeItem(item)
        self._cosmetic_line_items.clear()

    def set_anchor_points(
        self,
        points: list[tuple[int, int]],
        *,
        active: bool = True,
    ) -> None:
        """Display draggable anchor points over corrosion profile."""
        self.clear_anchor_points()
        if not points:
            return
        pen = self._anchor_pen_active if bool(active) else self._anchor_pen_inactive
        size = int(self._ANCHOR_SIZE_PX)
        half = float(size) / 2.0
        brush_color = QColor(pen.color())
        brush_color.setAlpha(min(255, pen.color().alpha() + 20))

        for x, y in points:
            item = QGraphicsEllipseItem(float(x) + 0.5 - half, float(y) + 0.5 - half, float(size), float(size))
            item.setPen(pen)
            item.setBrush(brush_color)
            item.setZValue(9)
            self._scene.addItem(item)
            self._anchor_items.append(item)

    def clear_anchor_points(self) -> None:
        for item in self._anchor_items:
            self._scene.removeItem(item)
        self._anchor_items.clear()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._view.viewport() and isinstance(event, QMouseEvent):
            coords = self._scene_coords_from_event(event)
            has_modifier = bool(
                event.modifiers()
                & (
                    Qt.KeyboardModifier.ShiftModifier
                    | Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                )
            )
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if coords is not None and not has_modifier:
                    self.profile_drag_started.emit(coords)
                    return True
            elif event.type() == QEvent.Type.MouseMove and (event.buttons() & Qt.MouseButton.LeftButton):
                if coords is not None and not has_modifier:
                    self.profile_drag_moved.emit(coords)
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                if not has_modifier:
                    self.profile_drag_finished.emit(coords)
                    return True
            elif event.type() == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
                if coords is not None and not has_modifier:
                    self.profile_double_clicked.emit(coords)
                    return True
        return super().eventFilter(obj, event)

    def _pen_for_label(
        self,
        label: int,
        *,
        palette: dict[int, tuple[int, int, int, int]],
        layer_opacity: float,
    ) -> QPen:
        b, g, r, a = palette.get(int(label), (255, 0, 255, 200))
        alpha = int(
            max(
                0,
                min(
                    255,
                    round(float(a) * float(layer_opacity) * float(self._overlay_opacity)),
                ),
            )
        )
        color = QColor(int(r), int(g), int(b), alpha)
        pen = QPen(color)
        pen.setWidth(self._COSMETIC_LINE_WIDTH)
        pen.setCosmetic(True)
        return pen

