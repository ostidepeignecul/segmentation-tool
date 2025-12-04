"""Annotation-specific view extending EndviewView with ROI/drawing hooks."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QEvent, pyqtSignal
from PyQt6.QtGui import QCursor, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QFileDialog, QGraphicsPixmapItem, QGraphicsRectItem

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
        self._temp_mask_points: list[Tuple[int, int]] = []
        self._temp_box: Optional[Tuple[int, int, int, int]] = None
        self._roi_overlay: Optional[Any] = None
        self._box_start: Optional[Tuple[int, int]] = None
        self._roi_item = QGraphicsPixmapItem()
        self._roi_item.setOpacity(0.35)
        self._roi_item.setZValue(5)
        self._scene.addItem(self._roi_item)
        self._temp_box_item = QGraphicsRectItem()
        self._temp_box_item.setPen(QPen(Qt.GlobalColor.yellow))
        self._temp_box_item.setZValue(6)
        self._scene.addItem(self._temp_box_item)
        self._roi_box_items: list[QGraphicsRectItem] = []
        self._roi_pen = QPen(Qt.GlobalColor.white)
        self._roi_pen.setWidth(2)
        self._roi_point_items: list[QGraphicsRectItem] = []
        self._roi_point_pen = QPen(Qt.GlobalColor.white)
        self._roi_point_pen.setWidth(2)
        self._tool_mode: Optional[str] = None
        self._paint_cursor: Optional[QCursor] = None
        self._paint_cursor_radius: int = 8

    # ------------------------------------------------------------------ #
    # Temporary shapes (stubs)
    # ------------------------------------------------------------------ #
    def set_temp_mask(self, points: Sequence[Tuple[int, int]]) -> None:
        """Placeholder to display temp mask outline points in progress."""
        self._temp_mask_points = [(int(x), int(y)) for x, y in points]

    def set_temp_box(self, box: Optional[Tuple[int, int, int, int]]) -> None:
        """Placeholder to display a box in progress."""
        self._temp_box = box if box is None else tuple(int(v) for v in box)
        if box is None:
            self._temp_box_item.setRect(0, 0, 0, 0)
            return
        x1, y1, x2, y2 = self._temp_box
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        self._temp_box_item.setRect(xmin, ymin, xmax - xmin, ymax - ymin)

    def clear_temp_shapes(self) -> None:
        """Clear any temporary mask/box."""
        self._temp_mask_points = []
        self._temp_box = None
        self._box_start = None
        self._temp_box_item.setRect(0, 0, 0, 0)

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
        self.clear_roi_boxes()
        self.clear_roi_points()

    def select_overlay_save_path(self, parent: Any) -> Optional[str]:
        """Open a save dialog for overlay export and return the chosen path."""
        file_path, _ = QFileDialog.getSaveFileName(
            parent,
            "Sauvegarder l'overlay (.npz)",
            "",
            "Overlay NPZ (*.npz);;All Files (*)",
        )
        return file_path or None

    def set_roi_box(self, box: Optional[Tuple[int, int, int, int]]) -> None:
        """Backwards compatible single box setter."""
        if box is None:
            self.clear_roi_boxes()
            return
        self.set_roi_boxes([box])

    def set_roi_boxes(self, boxes: list[Tuple[int, int, int, int]]) -> None:
        """Display multiple ROI boxes outlines."""
        self.clear_roi_boxes()
        for box in boxes:
            x1, y1, x2, y2 = box
            xmin, xmax = sorted((x1, x2))
            ymin, ymax = sorted((y1, y2))
            item = QGraphicsRectItem()
            item.setPen(self._roi_pen)
            item.setZValue(7)
            item.setRect(xmin, ymin, xmax - xmin, ymax - ymin)
            self._scene.addItem(item)
            self._roi_box_items.append(item)

    def clear_roi_boxes(self) -> None:
        """Remove all ROI box outlines."""
        for it in self._roi_box_items:
            self._scene.removeItem(it)
        self._roi_box_items.clear()

    def set_roi_points(self, points: list[Tuple[int, int]]) -> None:
        """Display ROI grow seeds as small white points."""
        self.clear_roi_points()
        for x, y in points:
            item = QGraphicsRectItem()
            size = 6
            half = size // 2
            item.setRect(x - half, y - half, size, size)
            item.setPen(self._roi_point_pen)
            item.setZValue(7)
            self._scene.addItem(item)
            self._roi_point_items.append(item)

    def clear_roi_points(self) -> None:
        """Remove all ROI grow seed points."""
        for it in self._roi_point_items:
            self._scene.removeItem(it)
        self._roi_point_items.clear()

    # ------------------------------------------------------------------ #
    # Tool mode
    # ------------------------------------------------------------------ #
    def set_tool_mode(self, mode: Optional[str]) -> None:
        """Synchronize active tool to enable/disable box drawing."""
        self._tool_mode = mode
        self._apply_tool_cursor()
        if mode != "box":
            self.clear_temp_shapes()

    def _apply_tool_cursor(self) -> None:
        """Set a cursor matching the active tool (paint shows a hollow circle)."""
        if self._tool_mode == "paint":
            cursor = self._ensure_paint_cursor()
            self._view.setCursor(cursor)
            self._view.viewport().setCursor(cursor)
        else:
            self._view.setCursor(Qt.CursorShape.ArrowCursor)
            self._view.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def _ensure_paint_cursor(self) -> QCursor:
        """Build (or reuse) a hollow-circle cursor for the paint tool."""
        if self._paint_cursor is not None:
            return self._paint_cursor
        diameter = self._paint_cursor_radius * 2
        size = diameter + 4
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(Qt.GlobalColor.white)
        pen.setWidth(2)
        painter.setPen(pen)
        offset = (size - diameter) // 2
        painter.drawEllipse(offset, offset, diameter, diameter)
        painter.end()

        self._paint_cursor = QCursor(pixmap)
        return self._paint_cursor

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def eventFilter(self, obj: Any, event) -> bool:
        if obj is self._view.viewport():
            if isinstance(event, QMouseEvent):
                if event.type() == QMouseEvent.Type.MouseButtonPress:
                    if self._tool_mode == "box":
                        handled = self._handle_box_press(event)
                        if handled:
                            return True
                if event.type() == QMouseEvent.Type.MouseMove:
                    if self._tool_mode == "box":
                        self._handle_box_move(event)
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

    def _handle_box_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        # Ne pas intercepter Shift+clic : réservé au déplacement de la crosshair
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return False
        coords = self._clamped_scene_coords_from_event(event)
        if coords is None:
            return False
        self._view.setFocus(Qt.FocusReason.MouseFocusReason)
        if self._box_start is None:
            self._box_start = coords
            self.set_temp_box((coords[0], coords[1], coords[0], coords[1]))
        else:
            box = (self._box_start[0], self._box_start[1], coords[0], coords[1])
            self.box_drawn.emit(box)
            self.clear_temp_shapes()
        # Do not consume to allow base handler to update crosshair/drag if needed
        return False

    def _handle_box_move(self, event: QMouseEvent) -> None:
        if self._box_start is None:
            return
        coords = self._clamped_scene_coords_from_event(event)
        if coords is None:
            return
        box = (self._box_start[0], self._box_start[1], coords[0], coords[1])
        self.set_temp_box(box)

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
