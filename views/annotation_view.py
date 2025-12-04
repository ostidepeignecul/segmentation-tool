"""Annotation-specific view extending EndviewView with ROI/drawing hooks."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QEvent, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem

from config.constants import MASK_COLORS_BGRA
from views.endview_view import EndviewView


class AnnotationView(EndviewView):
    """Extends the base endview renderer with placeholders for ROI rendering."""

    selection_cancel_requested = pyqtSignal()
    apply_temp_mask_requested = pyqtSignal()
    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()
    apply_roi_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._temp_polygon: list[Tuple[int, int]] = []
        self._temp_rectangle: Optional[Tuple[int, int, int, int]] = None
        self._roi_overlay: Optional[Any] = None
        self._rect_start: Optional[Tuple[int, int]] = None
        self._roi_item = QGraphicsPixmapItem()
        self._roi_item.setOpacity(0.35)
        self._roi_item.setZValue(5)
        self._scene.addItem(self._roi_item)
        self._temp_rect_item = QGraphicsRectItem()
        self._temp_rect_item.setPen(QPen(Qt.GlobalColor.yellow))
        self._temp_rect_item.setZValue(6)
        self._scene.addItem(self._temp_rect_item)
        self._roi_rect_items: list[QGraphicsRectItem] = []
        self._roi_pen = QPen(Qt.GlobalColor.white)
        self._roi_pen.setWidth(2)

    # ------------------------------------------------------------------ #
    # Temporary shapes (stubs)
    # ------------------------------------------------------------------ #
    def set_temp_polygon(self, points: Sequence[Tuple[int, int]]) -> None:
        """Placeholder to display a polygon in progress."""
        self._temp_polygon = [(int(x), int(y)) for x, y in points]

    def set_temp_rectangle(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        """Placeholder to display a rectangle in progress."""
        self._temp_rectangle = rect if rect is None else tuple(int(v) for v in rect)
        if rect is None:
            self._temp_rect_item.setRect(0, 0, 0, 0)
            return
        x1, y1, x2, y2 = self._temp_rectangle
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        self._temp_rect_item.setRect(xmin, ymin, xmax - xmin, ymax - ymin)

    def clear_temp_shapes(self) -> None:
        """Clear any temporary polygon/rectangle."""
        self._temp_polygon = []
        self._temp_rectangle = None
        self._rect_start = None
        self._temp_rect_item.setRect(0, 0, 0, 0)

    # ------------------------------------------------------------------ #
    # ROI overlay (stub)
    # ------------------------------------------------------------------ #
    def set_roi_overlay(self, roi_mask: Any, palette: Optional[dict[int, tuple[int, int, int, int]]] = None) -> None:
        """Display a ROI mask overlay."""
        self._roi_overlay = roi_mask
        pixmap = self._roi_to_pixmap(roi_mask, palette=palette)
        self._roi_item.setPixmap(pixmap)

    def clear_roi_overlay(self) -> None:
        """Remove any ROI overlay."""
        self._roi_overlay = None
        self._roi_item.setPixmap(self._blank_pixmap())
        self.clear_roi_rectangles()

    def set_roi_rectangle(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        """Backwards compatible single rectangle setter."""
        if rect is None:
            self.clear_roi_rectangles()
            return
        self.set_roi_rectangles([rect])

    def set_roi_rectangles(self, rects: list[Tuple[int, int, int, int]]) -> None:
        """Display multiple ROI rectangles outlines."""
        self.clear_roi_rectangles()
        for rect in rects:
            x1, y1, x2, y2 = rect
            xmin, xmax = sorted((x1, x2))
            ymin, ymax = sorted((y1, y2))
            item = QGraphicsRectItem()
            item.setPen(self._roi_pen)
            item.setZValue(7)
            item.setRect(xmin, ymin, xmax - xmin, ymax - ymin)
            self._scene.addItem(item)
            self._roi_rect_items.append(item)

    def clear_roi_rectangles(self) -> None:
        """Remove all ROI rectangle outlines."""
        for it in self._roi_rect_items:
            self._scene.removeItem(it)
        self._roi_rect_items.clear()

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def eventFilter(self, obj: Any, event) -> bool:
        if obj is self._view.viewport():
            if isinstance(event, QMouseEvent):
                if event.type() == QMouseEvent.Type.MouseButtonPress:
                    handled = self._handle_rectangle_press(event)
                    if handled:
                        return True
                if event.type() == QMouseEvent.Type.MouseMove:
                    self._handle_rectangle_move(event)
        return super().eventFilter(obj, event)

    def _roi_to_pixmap(self, mask: Any, palette: Optional[dict[int, tuple[int, int, int, int]]] = None):
        arr = np.asarray(mask) if mask is not None else None
        if arr is None or arr.ndim != 2:
            return self._blank_pixmap()
        rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
        palette = palette or {}
        for lbl in np.unique(arr):
            lbl_int = int(lbl)
            if lbl_int == 0:
                continue
            b, g, r, a = palette.get(lbl_int, MASK_COLORS_BGRA.get(lbl_int, (255, 0, 255, 160)))
            mask_lbl = arr == lbl_int
            rgba[..., 0][mask_lbl] = r
            rgba[..., 1][mask_lbl] = g
            rgba[..., 2][mask_lbl] = b
            rgba[..., 3][mask_lbl] = a
        return self._mask_to_pixmap(rgba)

    @staticmethod
    def _blank_pixmap():
        return QPixmap()

    def _handle_rectangle_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        # Ne pas intercepter Shift+clic : réservé au déplacement de la crosshair
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return False
        coords = self._clamped_scene_coords_from_event(event)
        if coords is None:
            return False
        self._view.setFocus(Qt.FocusReason.MouseFocusReason)
        if self._rect_start is None:
            self._rect_start = coords
            self.set_temp_rectangle((coords[0], coords[1], coords[0], coords[1]))
        else:
            rect = (self._rect_start[0], self._rect_start[1], coords[0], coords[1])
            self.rectangle_drawn.emit(rect)
            self.clear_temp_shapes()
        # Do not consume to allow base handler to update crosshair/drag if needed
        return False

    def _handle_rectangle_move(self, event: QMouseEvent) -> None:
        if self._rect_start is None:
            return
        coords = self._clamped_scene_coords_from_event(event)
        if coords is None:
            return
        rect = (self._rect_start[0], self._rect_start[1], coords[0], coords[1])
        self.set_temp_rectangle(rect)

    def _clamped_scene_coords_from_event(self, event: QMouseEvent) -> Optional[Tuple[int, int]]:
        if self._volume is None:
            return None
        scene_pos = self._view.mapToScene(event.position().toPoint())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        height, width = self._volume.shape[1:]
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        return (x, y)
