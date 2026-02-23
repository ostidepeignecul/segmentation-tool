"""Annotation-specific view extending EndviewView with ROI/drawing hooks."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QEvent, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QFileDialog, QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsRectItem

from config.constants import MASK_COLORS_BGRA
from views.endview_view import EndviewView


class AnnotationView(EndviewView):
    """Extends the base endview renderer with placeholders for ROI rendering."""

    selection_cancel_requested = pyqtSignal()
    apply_temp_mask_requested = pyqtSignal()
    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()
    apply_roi_requested = pyqtSignal()
    line_drawn = pyqtSignal(object)
    restriction_rect_changed = pyqtSignal(object)
    mod_drag_started = pyqtSignal(object)
    mod_drag_moved = pyqtSignal(object)
    mod_drag_finished = pyqtSignal(object)
    mod_double_clicked = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._restriction_rect: Optional[Tuple[int, int, int, int]] = None
        self._temp_mask_points: list[Tuple[int, int]] = []
        self._temp_box: Optional[Tuple[int, int, int, int]] = None
        self._temp_line_points: list[Tuple[int, int]] = []
        self._freehand_drawing: bool = False
        self._line_drawing: bool = False
        self._roi_overlay: Optional[Any] = None
        self._box_start: Optional[Tuple[int, int]] = None
        self._roi_item = QGraphicsPixmapItem()
        self._roi_item.setOpacity(0.35)
        self._roi_item.setZValue(5)
        self._scene.addItem(self._roi_item)
        self._temp_box_item = QGraphicsRectItem()
        temp_box_pen = QPen(Qt.GlobalColor.yellow)
        temp_box_pen.setWidth(1)
        temp_box_pen.setCosmetic(True)
        self._temp_box_item.setPen(temp_box_pen)
        self._temp_box_item.setZValue(6)
        self._scene.addItem(self._temp_box_item)
        self._temp_line_item = QGraphicsPathItem()
        temp_line_pen = QPen(Qt.GlobalColor.yellow)
        temp_line_pen.setWidth(1)
        temp_line_pen.setCosmetic(True)
        self._temp_line_item.setPen(temp_line_pen)
        self._temp_line_item.setZValue(6)
        self._scene.addItem(self._temp_line_item)
        self._restriction_item = QGraphicsRectItem()
        restriction_pen = QPen(Qt.GlobalColor.cyan)
        restriction_pen.setWidth(1)
        restriction_pen.setCosmetic(True)
        restriction_pen.setStyle(Qt.PenStyle.DashLine)
        self._restriction_item.setPen(restriction_pen)
        self._restriction_item.setZValue(8)
        self._restriction_item.setVisible(False)
        self._scene.addItem(self._restriction_item)
        self._roi_box_items: list[QGraphicsRectItem] = []
        self._roi_pen = QPen(Qt.GlobalColor.white)
        self._roi_pen.setWidth(1)
        self._roi_pen.setCosmetic(True)
        self._roi_point_items: list[QGraphicsRectItem] = []
        self._roi_point_pen = QPen(Qt.GlobalColor.white)
        self._roi_point_pen.setWidth(1)
        self._roi_point_pen.setCosmetic(True)
        self._mod_anchor_items: list[QGraphicsEllipseItem] = []
        self._mod_anchor_pen_active = QPen(Qt.GlobalColor.cyan)
        self._mod_anchor_pen_active.setWidth(1)
        self._mod_anchor_pen_active.setCosmetic(True)
        self._mod_anchor_pen_inactive = QPen(Qt.GlobalColor.gray)
        self._mod_anchor_pen_inactive.setWidth(1)
        self._mod_anchor_pen_inactive.setCosmetic(True)
        self._tool_mode: Optional[str] = None
        self._paint_cursor: Optional[QCursor] = None
        self._paint_cursor_radius: int = 8
        self._restriction_dragging: bool = False
        self._restriction_drag_mode: Optional[str] = None
        self._restriction_drag_start: Optional[Tuple[int, int]] = None
        self._restriction_drag_rect: Optional[Tuple[int, int, int, int]] = None
        self._restriction_edge_grab: int = 6
        self._restriction_min_size: int = 10
        self._update_roi_outline_color()

    def set_colormap(self, name: str, lut: Optional[np.ndarray]) -> None:
        """Set base colormap and adapt ROI box contour color for readability."""
        super().set_colormap(name, lut)
        self._update_roi_outline_color()

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
        self._temp_box_item.setRect(xmin, ymin, (xmax - xmin) + 1, (ymax - ymin) + 1)

    def set_temp_line(self, points: Sequence[Tuple[int, int]]) -> None:
        """Display a temporary freehand line in progress."""
        self._temp_line_points = [(int(x), int(y)) for x, y in points]
        path = QPainterPath()
        if self._temp_line_points:
            x0, y0 = self._temp_line_points[0]
            path.moveTo(x0 + 0.5, y0 + 0.5)
            for x, y in self._temp_line_points[1:]:
                path.lineTo(x + 0.5, y + 0.5)
        self._temp_line_item.setPath(path)

    def clear_temp_shapes(self) -> None:
        """Clear any temporary mask/box/line."""
        self._temp_mask_points = []
        self._temp_box = None
        self._box_start = None
        self._temp_box_item.setRect(0, 0, 0, 0)
        self._temp_line_points = []
        self._freehand_drawing = False
        self._line_drawing = False
        self._temp_line_item.setPath(QPainterPath())

    def set_restriction_rect(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        """Display the global restriction rectangle."""
        if rect is None:
            self._restriction_rect = None
            self._restriction_item.setRect(0, 0, 0, 0)
            self._restriction_item.setVisible(False)
            return
        x1, y1, x2, y2 = (int(v) for v in rect)
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        if self._volume is not None:
            height, width = self._volume.shape[1:]
            xmax = max(0, min(width - 1, xmax))
            xmin = max(0, min(width - 1, xmin))
            ymax = max(0, min(height - 1, ymax))
            ymin = max(0, min(height - 1, ymin))
        self._restriction_rect = (xmin, ymin, xmax, ymax)
        self._restriction_item.setRect(xmin, ymin, (xmax - xmin) + 1, (ymax - ymin) + 1)
        self._restriction_item.setVisible(True)

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
            item.setRect(xmin, ymin, (xmax - xmin) + 1, (ymax - ymin) + 1)
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
        """Synchronize active tool and clear transient shapes when switching."""
        prev_mode = self._tool_mode
        self._tool_mode = mode
        self._apply_tool_cursor()
        if prev_mode != mode:
            self.clear_temp_shapes()
            if prev_mode == "mod" and mode != "mod":
                self.clear_mod_anchor_points()

    def set_mod_anchor_points(
        self,
        points: list[tuple[int, int]],
        *,
        active: bool = True,
        active_index: Optional[int] = None,
    ) -> None:
        """Display anchor points for mask modification mode."""
        self.clear_mod_anchor_points()
        if not points:
            return
        size = 1.0
        half = size / 2.0
        for idx, (x, y) in enumerate(points):
            is_active = bool(active) and (active_index is None or int(active_index) == int(idx))
            pen = self._mod_anchor_pen_active if is_active else self._mod_anchor_pen_inactive
            brush = QColor(pen.color())
            brush.setAlpha(min(255, pen.color().alpha() + 20))
            item = QGraphicsEllipseItem(float(x) + 0.5 - half, float(y) + 0.5 - half, size, size)
            item.setPen(pen)
            item.setBrush(brush)
            item.setZValue(9)
            self._scene.addItem(item)
            self._mod_anchor_items.append(item)

    def clear_mod_anchor_points(self) -> None:
        for item in self._mod_anchor_items:
            self._scene.removeItem(item)
        self._mod_anchor_items.clear()

    def _apply_tool_cursor(self) -> None:
        """Set a cursor matching the active tool (paint shows a hollow circle)."""
        if self._tool_mode == "paint":
            cursor = self._ensure_paint_cursor()
            self._view.setCursor(cursor)
            self._view.viewport().setCursor(cursor)
        else:
            self._view.setCursor(Qt.CursorShape.ArrowCursor)
            self._view.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def set_paint_radius(self, radius: int) -> None:
        """Update paint cursor radius and refresh cursor if active."""
        radius = max(1, int(radius))
        if radius == self._paint_cursor_radius and self._paint_cursor is not None:
            return
        self._paint_cursor_radius = radius
        self._paint_cursor = None  # force rebuild
        if self._tool_mode == "paint":
            self._apply_tool_cursor()

    def _ensure_paint_cursor(self) -> QCursor:
        """Build (or reuse) a fixed-size cross cursor for the paint tool."""
        if self._paint_cursor is not None:
            return self._paint_cursor
        size = 13
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        center = size // 2
        outline = QPen(Qt.GlobalColor.black)
        outline.setWidth(3)
        painter.setPen(outline)
        painter.drawLine(0, center, size - 1, center)
        painter.drawLine(center, 0, center, size - 1)
        cross = QPen(Qt.GlobalColor.white)
        cross.setWidth(1)
        painter.setPen(cross)
        painter.drawLine(0, center, size - 1, center)
        painter.drawLine(center, 0, center, size - 1)
        painter.end()

        self._paint_cursor = QCursor(pixmap)
        return self._paint_cursor

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def eventFilter(self, obj: Any, event) -> bool:
        if obj is self._view.viewport():
            if isinstance(event, QMouseEvent):
                if self._tool_mode == "mod":
                    coords = self._clamped_scene_coords_from_event(event)
                    has_modifier = bool(
                        event.modifiers()
                        & (
                            Qt.KeyboardModifier.ShiftModifier
                            | Qt.KeyboardModifier.ControlModifier
                            | Qt.KeyboardModifier.AltModifier
                        )
                    )
                    if (
                        event.type() == QEvent.Type.MouseButtonPress
                        and event.button() == Qt.MouseButton.LeftButton
                        and coords is not None
                        and not has_modifier
                    ):
                        self.mod_drag_started.emit(coords)
                        return True
                    if (
                        event.type() == QEvent.Type.MouseMove
                        and (event.buttons() & Qt.MouseButton.LeftButton)
                        and coords is not None
                        and not has_modifier
                    ):
                        self.mod_drag_moved.emit(coords)
                        return True
                    if (
                        event.type() == QEvent.Type.MouseButtonRelease
                        and event.button() == Qt.MouseButton.LeftButton
                        and not has_modifier
                    ):
                        self.mod_drag_finished.emit(coords)
                        return True
                    if (
                        event.type() == QEvent.Type.MouseButtonDblClick
                        and event.button() == Qt.MouseButton.LeftButton
                        and coords is not None
                        and not has_modifier
                    ):
                        self.mod_double_clicked.emit(coords)
                        return True
                if event.type() == QMouseEvent.Type.MouseButtonPress:
                    if (
                        event.modifiers() & Qt.KeyboardModifier.AltModifier
                        and event.button() == Qt.MouseButton.LeftButton
                    ):
                        self._handle_restriction_press(event)
                        return True
                    if self._tool_mode == "line":
                        handled = self._handle_line_press(event)
                        if handled:
                            return True
                    if self._tool_mode == "free_hand":
                        handled = self._handle_freehand_press(event)
                        if handled:
                            return True
                    if self._tool_mode == "box":
                        handled = self._handle_box_press(event)
                        if handled:
                            return True
                if event.type() == QMouseEvent.Type.MouseMove:
                    if self._restriction_dragging:
                        self._handle_restriction_move(event)
                        return True
                    if self._tool_mode == "line":
                        self._handle_line_move(event)
                    if self._tool_mode == "free_hand":
                        self._handle_freehand_move(event)
                    if self._tool_mode == "box":
                        self._handle_box_move(event)
                if event.type() == QMouseEvent.Type.MouseButtonRelease:
                    if self._restriction_dragging:
                        self._handle_restriction_release(event)
                        return True
                    if (
                        event.modifiers() & Qt.KeyboardModifier.AltModifier
                        and event.button() == Qt.MouseButton.LeftButton
                    ):
                        return True
                    if self._tool_mode == "line":
                        handled = self._handle_line_release(event)
                        if handled:
                            return True
                    if self._tool_mode == "free_hand":
                        handled = self._handle_freehand_release(event)
                        if handled:
                            return True
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

    def _handle_line_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return False
        coords = self._clamped_scene_coords_from_event(event)
        if coords is None:
            return False
        self._view.setFocus(Qt.FocusReason.MouseFocusReason)
        self._line_drawing = True
        self._temp_line_points = [coords]
        self.set_temp_line(self._temp_line_points)
        return False

    def _handle_line_move(self, event: QMouseEvent) -> None:
        if not self._line_drawing:
            return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        coords = self._clamped_scene_coords_from_event(event)
        if coords is None:
            return
        if self._temp_line_points and coords == self._temp_line_points[-1]:
            return
        self._temp_line_points.append(coords)
        self.set_temp_line(self._temp_line_points)

    def _handle_line_release(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if not self._line_drawing:
            return False
        coords = self._clamped_scene_coords_from_event(event)
        if coords is not None and (
            not self._temp_line_points or coords != self._temp_line_points[-1]
        ):
            self._temp_line_points.append(coords)
        points = list(self._temp_line_points)
        self.clear_temp_shapes()
        if points:
            self.line_drawn.emit(points)
        return False

    def _handle_freehand_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return False
        coords = self._clamped_scene_coords_from_event(event)
        if coords is None:
            return False
        self._view.setFocus(Qt.FocusReason.MouseFocusReason)
        self._freehand_drawing = True
        self._temp_mask_points = [coords]
        self.set_temp_mask(self._temp_mask_points)
        self.set_temp_line(self._temp_mask_points)
        self.freehand_started.emit(coords)
        return False

    def _handle_freehand_move(self, event: QMouseEvent) -> None:
        if not self._freehand_drawing:
            return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        coords = self._clamped_scene_coords_from_event(event)
        if coords is None:
            return
        if self._temp_mask_points and coords == self._temp_mask_points[-1]:
            return
        self._temp_mask_points.append(coords)
        self.set_temp_mask(self._temp_mask_points)
        self.set_temp_line(self._temp_mask_points)
        self.freehand_point_added.emit(coords)

    def _handle_freehand_release(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if not self._freehand_drawing:
            return False
        coords = self._clamped_scene_coords_from_event(event)
        if coords is not None and (
            not self._temp_mask_points or coords != self._temp_mask_points[-1]
        ):
            self._temp_mask_points.append(coords)
        points = list(self._temp_mask_points)
        self.clear_temp_shapes()
        if points:
            self.freehand_completed.emit(points)
        return False

    def _handle_restriction_press(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._restriction_rect is None:
            return
        coords = self._clamped_scene_coords_from_event(event)
        if coords is None:
            return
        mode = self._restriction_hit_test(coords)
        if mode is None:
            return
        self._restriction_dragging = True
        self._restriction_drag_mode = mode
        self._restriction_drag_start = coords
        self._restriction_drag_rect = self._restriction_rect
        self._view.setFocus(Qt.FocusReason.MouseFocusReason)

    def _handle_restriction_move(self, event: QMouseEvent) -> None:
        if not self._restriction_dragging:
            return
        if self._restriction_drag_rect is None or self._restriction_drag_mode is None:
            return
        coords = self._clamped_scene_coords_from_event(event)
        if coords is None or self._volume is None:
            return
        x, y = coords
        start_x, start_y = self._restriction_drag_start or coords
        x1, y1, x2, y2 = self._restriction_drag_rect
        height, width = self._volume.shape[1:]
        min_w = min(self._restriction_min_size, max(0, width - 1))
        min_h = min(self._restriction_min_size, max(0, height - 1))

        mode = self._restriction_drag_mode
        if mode == "move":
            dx = x - start_x
            dy = y - start_y
            rect_w = x2 - x1
            rect_h = y2 - y1
            new_x1 = x1 + dx
            new_y1 = y1 + dy
            max_x1 = max(0, (width - 1) - rect_w)
            max_y1 = max(0, (height - 1) - rect_h)
            new_x1 = max(0, min(max_x1, new_x1))
            new_y1 = max(0, min(max_y1, new_y1))
            new_x2 = new_x1 + rect_w
            new_y2 = new_y1 + rect_h
        elif mode == "resize_left":
            limit = x2 - min_w
            new_x1 = max(0, min(limit, x))
            new_x2 = x2
            new_y1, new_y2 = y1, y2
        elif mode == "resize_right":
            limit = x1 + min_w
            new_x2 = min(width - 1, max(limit, x))
            new_x1 = x1
            new_y1, new_y2 = y1, y2
        elif mode == "resize_top":
            limit = y2 - min_h
            new_y1 = max(0, min(limit, y))
            new_y2 = y2
            new_x1, new_x2 = x1, x2
        elif mode == "resize_bottom":
            limit = y1 + min_h
            new_y2 = min(height - 1, max(limit, y))
            new_y1 = y1
            new_x1, new_x2 = x1, x2
        else:
            return

        new_rect = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
        if new_rect != self._restriction_rect:
            self.set_restriction_rect(new_rect)
            self.restriction_rect_changed.emit(new_rect)

    def _handle_restriction_release(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._restriction_dragging = False
        self._restriction_drag_mode = None
        self._restriction_drag_start = None
        self._restriction_drag_rect = None

    def _restriction_hit_test(self, coords: Tuple[int, int]) -> Optional[str]:
        if self._restriction_rect is None:
            return None
        x, y = coords
        x1, y1, x2, y2 = self._restriction_rect
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        edge = self._restriction_edge_grab
        if x < xmin - edge or x > xmax + edge or y < ymin - edge or y > ymax + edge:
            return None
        if abs(x - xmin) <= edge and ymin <= y <= ymax:
            return "resize_left"
        if abs(x - xmax) <= edge and ymin <= y <= ymax:
            return "resize_right"
        if abs(y - ymin) <= edge and xmin <= x <= xmax:
            return "resize_top"
        if abs(y - ymax) <= edge and xmin <= x <= xmax:
            return "resize_bottom"
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return "move"
        return None

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

    def _update_roi_outline_color(self) -> None:
        """Keep ROI outlines visible on selected base colormap."""
        colormap_name = (self._colormap_name or "").strip().casefold()
        outline_color = Qt.GlobalColor.black if colormap_name == "omniscan" else Qt.GlobalColor.white
        self._roi_pen.setColor(outline_color)
        self._roi_point_pen.setColor(outline_color)
        temp_box_pen = self._temp_box_item.pen()
        temp_box_pen.setColor(outline_color)
        self._temp_box_item.setPen(temp_box_pen)
        temp_line_pen = self._temp_line_item.pen()
        temp_line_pen.setColor(outline_color)
        self._temp_line_item.setPen(temp_line_pen)
        for item in self._roi_box_items:
            item.setPen(self._roi_pen)
        for item in self._roi_point_items:
            item.setPen(self._roi_point_pen)
