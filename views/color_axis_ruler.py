"""Pixel rulers that follow the visible endview zoom and pan window."""

from __future__ import annotations

import math

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class ColorAxisRuler(QWidget):
    """Draw a horizontal or vertical ruler from scene pixel coordinates."""

    _THICKNESS = 30
    _TARGET_MAJOR_SPACING_PX = 72
    _BACKGROUND = QColor("#171717")
    _BORDER = QColor("#303030")
    _TEXT = QColor("#d0d0d0")
    _SUBTEXT = QColor("#8c8c8c")
    _TICK = QColor("#a8a8a8")
    _AXIS_COLORS = {
        "d-scan": QColor("#ff6a00"),
        "profondeur": QColor("#9b649b"),
        "b-scan": QColor("#00b4b4"),
        "amplitude": QColor("#cfff53"),
    }

    def __init__(self, orientation: Qt.Orientation, parent=None) -> None:
        super().__init__(parent)
        self._orientation = orientation
        self._axis_name: str = ""
        self._view_min: float = 0.0
        self._view_max: float = 0.0
        self._content_min: float = 0.0
        self._content_max: float = 0.0
        self._has_range: bool = False

        if self._orientation == Qt.Orientation.Horizontal:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.setFixedHeight(self._THICKNESS)
        else:
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            self.setFixedWidth(self._THICKNESS + 18)

    def sizeHint(self) -> QSize:
        if self._orientation == Qt.Orientation.Horizontal:
            return QSize(160, self.height())
        return QSize(self.width(), 160)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    @classmethod
    def axis_color_for_name(cls, axis_name: str) -> QColor | None:
        """Return the configured color for a user-facing axis name."""
        normalized = str(axis_name or "").strip().casefold()
        color = cls._AXIS_COLORS.get(normalized)
        if color is None:
            return None
        return QColor(color)

    def set_axis_name(self, name: str) -> None:
        """Update the displayed axis label."""
        axis_name = str(name or "").strip()
        if axis_name == self._axis_name:
            return
        self._axis_name = axis_name
        self.update()

    def clear_range(self) -> None:
        """Hide tick marks when no scene is displayed."""
        if not self._has_range:
            return
        self._has_range = False
        self.update()

    def set_view_range(
        self,
        *,
        view_min: float,
        view_max: float,
        content_min: float,
        content_max: float,
    ) -> None:
        """Set the visible scene range and the valid content limits."""
        self._view_min = float(view_min)
        self._view_max = float(view_max)
        self._content_min = float(content_min)
        self._content_max = float(content_max)
        self._has_range = True
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), self._BACKGROUND)

        if self._orientation == Qt.Orientation.Horizontal:
            self._draw_horizontal(painter)
        else:
            self._draw_vertical(painter)

    def _draw_horizontal(self, painter: QPainter) -> None:
        rect = self.rect()
        baseline_y = 0
        tick_pen = QPen(self._resolved_tick_color())
        painter.setPen(QPen(self._resolved_border_color()))
        painter.drawLine(rect.left(), baseline_y, rect.right(), baseline_y)

        if self._axis_name:
            painter.setPen(self._resolved_text_color())
            painter.drawText(rect.adjusted(6, 3, -6, -2), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, self._axis_name)

        if not self._has_drawable_range():
            return

        label_y = 14
        for value, position in self._iter_major_ticks(length=rect.width()):
            x_pos = rect.left() + position
            painter.setPen(tick_pen)
            painter.drawLine(x_pos, baseline_y, x_pos, baseline_y + 8)
            painter.setPen(self._resolved_text_color())
            painter.drawText(x_pos + 2, label_y, str(int(value)))

    def _draw_vertical(self, painter: QPainter) -> None:
        rect = self.rect()
        baseline_x = rect.right()
        tick_pen = QPen(self._resolved_tick_color())
        painter.setPen(QPen(self._resolved_border_color()))
        painter.drawLine(baseline_x, rect.top(), baseline_x, rect.bottom())

        if self._axis_name:
            painter.save()
            painter.setPen(self._resolved_text_color())
            painter.translate(0, rect.height())
            painter.rotate(-90)
            painter.drawText(
                4,
                12,
                rect.height() - 8,
                12,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._axis_name,
            )
            painter.restore()

        if not self._has_drawable_range():
            return

        metrics = QFontMetrics(painter.font())
        for value, position in self._iter_major_ticks(length=rect.height()):
            y_pos = rect.top() + position
            painter.setPen(tick_pen)
            painter.drawLine(baseline_x - 8, y_pos, baseline_x, y_pos)
            text = str(int(value))
            text_width = metrics.horizontalAdvance(text)
            painter.setPen(self._resolved_text_color())
            painter.drawText(baseline_x - 12 - text_width, y_pos + metrics.ascent() // 2, text)

    def _has_drawable_range(self) -> bool:
        if not self._has_range:
            return False
        span = abs(self._view_max - self._view_min)
        return math.isfinite(span) and span > 1e-6

    def _iter_major_ticks(self, *, length: int) -> list[tuple[int, int]]:
        visible_min = min(self._view_min, self._view_max)
        visible_max = max(self._view_min, self._view_max)
        content_min = min(self._content_min, self._content_max)
        content_max = max(self._content_min, self._content_max)
        draw_min = max(visible_min, content_min)
        draw_max = min(visible_max, content_max)
        if draw_max < draw_min:
            return []

        visible_span = visible_max - visible_min
        if visible_span <= 1e-6:
            return []

        pixels_per_scene_unit = max(1e-6, float(length) / float(visible_span))
        major_step = self._choose_major_step(self._TARGET_MAJOR_SPACING_PX / pixels_per_scene_unit)

        first_tick = int(math.floor(draw_min / major_step) * major_step)
        if first_tick < content_min:
            first_tick = int(math.ceil(content_min / major_step) * major_step)

        ticks: list[tuple[int, int]] = []
        value = first_tick
        max_tick = int(math.floor(draw_max))
        while value <= max_tick:
            ratio = (float(value) - visible_min) / visible_span
            position = int(round(ratio * float(length - 1)))
            if 0 <= position < length:
                ticks.append((int(value), position))
            value += major_step
        return ticks

    @staticmethod
    def _choose_major_step(target_scene_units: float) -> int:
        target = max(1.0, float(target_scene_units))
        exponent = int(math.floor(math.log10(target)))
        base = 10 ** exponent
        for factor in (1, 2, 5, 10):
            step = factor * base
            if step >= target:
                return int(max(1, step))
        return int(max(1, 10 * base))

    def _resolved_axis_color(self) -> QColor | None:
        return self.axis_color_for_name(self._axis_name)

    def _resolved_border_color(self) -> QColor:
        color = self._resolved_axis_color()
        if color is None:
            return QColor(self._BORDER)
        color.setAlpha(210)
        return color

    def _resolved_tick_color(self) -> QColor:
        color = self._resolved_axis_color()
        if color is None:
            return QColor(self._TICK)
        color.setAlpha(235)
        return color

    def _resolved_text_color(self) -> QColor:
        color = self._resolved_axis_color()
        if color is None:
            return QColor(self._TEXT)
        return color
