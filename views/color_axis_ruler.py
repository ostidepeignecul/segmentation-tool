"""Shared rulers that follow the visible window of ruler-enabled views."""

from __future__ import annotations

import math

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class AxisTitleLabel(QWidget):
    """Display an axis title outside the colored ruler body."""

    _TITLE_BAND = 18
    _BACKGROUND = QColor("#171717")
    _TEXT = QColor("#d0d0d0")

    def __init__(self, orientation: Qt.Orientation, parent=None) -> None:
        super().__init__(parent)
        self._orientation = orientation
        self._axis_name: str = ""

        if self._orientation == Qt.Orientation.Horizontal:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.setFixedHeight(self._TITLE_BAND)
        else:
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            self.setFixedWidth(self._TITLE_BAND)

    def sizeHint(self) -> QSize:
        if self._orientation == Qt.Orientation.Horizontal:
            return QSize(160, self.height())
        return QSize(self.width(), 160)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def set_axis_name(self, name: str) -> None:
        axis_name = str(name or "").strip()
        if axis_name == self._axis_name:
            return
        self._axis_name = axis_name
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), self._BACKGROUND)

        if not self._axis_name:
            return

        text_color = self._resolved_text_color()
        painter.setPen(text_color)
        rect = self.rect()

        if self._orientation == Qt.Orientation.Horizontal:
            painter.drawText(
                rect.adjusted(4, 0, -4, 0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                self._axis_name,
            )
            return

        metrics = QFontMetrics(painter.font())
        text_height = metrics.height()
        text_width = metrics.horizontalAdvance(self._axis_name)
        text_x = rect.left() + max(0, (rect.width() - text_height) // 2)
        text_y = rect.top() + max(0, (rect.height() - text_width) // 2) + text_width

        painter.save()
        painter.translate(text_x, text_y)
        painter.rotate(-90)
        painter.drawText(0, metrics.ascent(), self._axis_name)
        painter.restore()

    def _resolved_text_color(self) -> QColor:
        color = ColorAxisRuler.axis_color_for_name(self._axis_name)
        if color is None:
            return QColor(self._TEXT)
        return QColor(color)


class ColorAxisRuler(QWidget):
    """Draw a horizontal or vertical ruler from scene coordinates."""

    _THICKNESS = 30
    _TARGET_MAJOR_SPACING_PX = 72
    _MIN_MINOR_SPACING_PX = 6
    _BACKGROUND = QColor("#171717")
    _BORDER = QColor("#303030")
    _DARK_FOREGROUND = QColor("#111111")
    _LIGHT_FOREGROUND = QColor("#f2f2f2")
    _TEXT = QColor("#d0d0d0")
    _SUBTEXT = QColor("#8c8c8c")
    _TICK = QColor("#a8a8a8")
    _AXIS_COLORS = {
        "d-scan": QColor("#ff6a00"),
        "profondeur": QColor("#9b649b"),
        "b-scan": QColor("#00b4b4"),
        "amplitude": QColor("#cfff53"),
        "amplitude (%)": QColor("#cfff53"),
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
            self.setFixedWidth(self._THICKNESS)

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
        painter.fillRect(self.rect(), self._resolved_fill_color())

        if self._orientation == Qt.Orientation.Horizontal:
            self._draw_horizontal(painter)
        else:
            self._draw_vertical(painter)

    def _draw_horizontal(self, painter: QPainter) -> None:
        rect = self.rect()
        metrics = QFontMetrics(painter.font())
        border_pen = QPen(self._resolved_border_color())
        medium_tick_pen = QPen(self._resolved_medium_tick_color())
        minor_tick_pen = QPen(self._resolved_minor_tick_color())
        major_tick_pen = QPen(self._resolved_tick_color())
        text_color = self._resolved_text_color()
        anchor_y = rect.top()
        major_tick_length, medium_tick_length, minor_tick_length = self._tick_lengths()
        label_baseline_y = min(
            rect.bottom() - 4,
            anchor_y + major_tick_length + metrics.ascent() + 2,
        )
        label_step = 1.0

        if self._has_drawable_range():
            major_ticks, medium_ticks, minor_ticks, label_step = self._iter_ticks(length=rect.width())
            for _, position in minor_ticks:
                x_pos = rect.left() + position
                painter.setPen(minor_tick_pen)
                painter.drawLine(x_pos, anchor_y, x_pos, anchor_y + minor_tick_length)
            for _, position in medium_ticks:
                x_pos = rect.left() + position
                painter.setPen(medium_tick_pen)
                painter.drawLine(x_pos, anchor_y, x_pos, anchor_y + medium_tick_length)
            for value, position in major_ticks:
                x_pos = rect.left() + position
                painter.setPen(major_tick_pen)
                painter.drawLine(x_pos, anchor_y, x_pos, anchor_y + major_tick_length)
                text = self._format_tick_value(value, label_step)
                text_width = metrics.horizontalAdvance(text)
                label_x = max(
                    rect.left() + 2,
                    min(x_pos - text_width // 2, rect.right() - text_width - 2),
                )
                painter.setPen(text_color)
                painter.drawText(label_x, label_baseline_y, text)

        painter.setPen(border_pen)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

    def _draw_vertical(self, painter: QPainter) -> None:
        rect = self.rect()
        metrics = QFontMetrics(painter.font())
        border_pen = QPen(self._resolved_border_color())
        medium_tick_pen = QPen(self._resolved_medium_tick_color())
        minor_tick_pen = QPen(self._resolved_minor_tick_color())
        major_tick_pen = QPen(self._resolved_tick_color())
        text_color = self._resolved_text_color()
        anchor_x = rect.right()
        major_tick_length, medium_tick_length, minor_tick_length = self._tick_lengths()
        label_origin_x = rect.left() + 2
        label_step = 1.0

        if self._has_drawable_range():
            major_ticks, medium_ticks, minor_ticks, label_step = self._iter_ticks(length=rect.height())
            for _, position in minor_ticks:
                y_pos = rect.top() + position
                painter.setPen(minor_tick_pen)
                painter.drawLine(anchor_x - minor_tick_length, y_pos, anchor_x, y_pos)
            for _, position in medium_ticks:
                y_pos = rect.top() + position
                painter.setPen(medium_tick_pen)
                painter.drawLine(anchor_x - medium_tick_length, y_pos, anchor_x, y_pos)
            for value, position in major_ticks:
                y_pos = rect.top() + position
                painter.setPen(major_tick_pen)
                painter.drawLine(anchor_x - major_tick_length, y_pos, anchor_x, y_pos)
                text = self._format_tick_value(value, label_step)
                text_width = metrics.horizontalAdvance(text)
                text_x = label_origin_x
                text_y = y_pos + text_width // 2

                if text_y > rect.bottom() - 2:
                    text_y = rect.bottom() - 2
                if text_y < rect.top() + text_width:
                    text_y = rect.top() + text_width

                painter.save()
                painter.setPen(text_color)
                painter.translate(text_x, text_y)
                painter.rotate(-90)
                painter.drawText(0, metrics.ascent(), text)
                painter.restore()

        painter.setPen(border_pen)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

    def _has_drawable_range(self) -> bool:
        if not self._has_range:
            return False
        span = abs(self._view_max - self._view_min)
        return math.isfinite(span) and span > 1e-6

    def _iter_ticks(
        self,
        *,
        length: int,
    ) -> tuple[
        list[tuple[float, int]],
        list[tuple[float, int]],
        list[tuple[float, int]],
        float,
    ]:
        visible_span = abs(self._view_max - self._view_min)
        pixels_per_scene_unit = max(1e-6, float(length) / max(1e-6, visible_span))
        major_step = self._choose_major_step(self._TARGET_MAJOR_SPACING_PX / pixels_per_scene_unit)
        visible_min = min(self._view_min, self._view_max)
        visible_max = max(self._view_min, self._view_max)
        content_min = min(self._content_min, self._content_max)
        content_max = max(self._content_min, self._content_max)
        draw_min = max(visible_min, content_min)
        draw_max = min(visible_max, content_max)
        if draw_max < draw_min:
            return [], [], [], major_step

        subdivision_count = self._choose_subdivision_count(major_step * pixels_per_scene_unit)
        minor_step = major_step / float(subdivision_count)
        first_index = int(math.ceil((draw_min / minor_step) - 1e-9))
        last_index = int(math.floor((draw_max / minor_step) + 1e-9))

        major_ticks: list[tuple[float, int]] = []
        medium_ticks: list[tuple[float, int]] = []
        minor_ticks: list[tuple[float, int]] = []

        for index in range(first_index, last_index + 1):
            value = float(index) * minor_step
            position = self._position_for_value(
                value=value,
                length=length,
                visible_min=visible_min,
                visible_max=visible_max,
            )
            if position is None:
                continue

            remainder = index % subdivision_count
            if remainder == 0:
                major_ticks.append((value, position))
                continue
            if subdivision_count % 2 == 0 and remainder == (subdivision_count // 2):
                medium_ticks.append((value, position))
                continue
            minor_ticks.append((value, position))

        return major_ticks, medium_ticks, minor_ticks, major_step

    def _position_for_value(
        self,
        *,
        value: float,
        length: int,
        visible_min: float,
        visible_max: float,
    ) -> int | None:
        visible_span = visible_max - visible_min
        if visible_span <= 1e-6 or length <= 0:
            return None
        ratio = (value - visible_min) / visible_span
        position = int(round(ratio * float(length - 1)))
        if 0 <= position < length:
            return position
        return None

    @staticmethod
    def _choose_major_step(target_scene_units: float) -> float:
        target = max(1e-6, float(target_scene_units))
        exponent = int(math.floor(math.log10(target)))
        base = 10.0 ** exponent
        for factor in (1.0, 2.0, 5.0, 10.0):
            step = factor * base
            if step >= target:
                return float(step)
        return float(10.0 * base)

    @classmethod
    def _choose_subdivision_count(cls, major_spacing_px: float) -> int:
        for subdivision_count in (10, 5, 4, 2):
            if (major_spacing_px / float(subdivision_count)) >= float(cls._MIN_MINOR_SPACING_PX):
                return subdivision_count
        return 1

    def _tick_lengths(self) -> tuple[int, int, int]:
        major = max(9, int(round(float(self._THICKNESS) * 0.42)))
        medium = max(6, int(round(float(major) * 0.72)))
        minor = max(3, int(round(float(major) * 0.45)))
        return major, medium, minor

    @staticmethod
    def _format_tick_value(value: float, step: float) -> str:
        abs_step = abs(float(step))
        if abs_step >= 1.0:
            decimals = 0
        else:
            decimals = min(3, max(1, int(math.ceil(-math.log10(abs_step)))))
        rounded_value = round(float(value), decimals)
        text = f"{rounded_value:.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if text == "-0":
            return "0"
        return text

    def _resolved_axis_color(self) -> QColor | None:
        return self.axis_color_for_name(self._axis_name)

    def _resolved_fill_color(self) -> QColor:
        color = self._resolved_axis_color()
        if color is None:
            return QColor(self._BACKGROUND)
        return QColor(color)

    def _resolved_border_color(self) -> QColor:
        color = self._resolved_axis_color()
        if color is None:
            return QColor(self._BORDER)
        foreground = self._resolved_contrast_color()
        foreground.setAlpha(185)
        return foreground

    def _resolved_tick_color(self) -> QColor:
        color = self._resolved_axis_color()
        if color is None:
            return QColor(self._TICK)
        foreground = self._resolved_contrast_color()
        foreground.setAlpha(215)
        return foreground

    def _resolved_medium_tick_color(self) -> QColor:
        color = self._resolved_axis_color()
        if color is None:
            return QColor("#8c8c8c")
        foreground = self._resolved_contrast_color()
        foreground.setAlpha(150)
        return foreground

    def _resolved_minor_tick_color(self) -> QColor:
        color = self._resolved_axis_color()
        if color is None:
            return QColor("#6a6a6a")
        foreground = self._resolved_contrast_color()
        foreground.setAlpha(110)
        return foreground

    def _resolved_text_color(self) -> QColor:
        color = self._resolved_axis_color()
        if color is None:
            return QColor(self._TEXT)
        return self._resolved_contrast_color()

    def _resolved_contrast_color(self) -> QColor:
        fill = self._resolved_axis_color()
        if fill is None:
            return QColor(self._TEXT)
        luminance = (
            0.2126 * float(fill.red())
            + 0.7152 * float(fill.green())
            + 0.0722 * float(fill.blue())
        )
        if luminance >= 110.0:
            return QColor(self._DARK_FOREGROUND)
        return QColor(self._LIGHT_FOREGROUND)
