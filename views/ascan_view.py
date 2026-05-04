"""A-scan view built with PyQtGraph."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import QFrame, QGridLayout, QVBoxLayout, QWidget

from services.ruler_display_service import RulerDisplayService
from views.color_axis_ruler import AxisTitleLabel, ColorAxisRuler


class AScanView(QFrame):
    """Displays a 1D signal with interactive position marker."""

    position_changed = pyqtSignal(int)
    cursor_moved = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._signal: Optional[np.ndarray] = None
        self._marker_idx: Optional[int] = None
        self._suspend_marker_signal: bool = False
        self._positions: Optional[np.ndarray] = None
        self._source_positions: Optional[np.ndarray] = None
        self._horizontal_axis_resolution_mm: Optional[float] = None
        self._overlay_spans: tuple[tuple[int, int, int], ...] = ()
        self._overlay_palette: dict[int, tuple[int, int, int, int]] = {}
        self._overlay_opacity: float = 0.4
        self._display_axis_x_name: str = "Profondeur"
        self._display_axis_y_name: str = "Amplitude (%)"
        self._ruler_display_unit: str = RulerDisplayService.DISPLAY_UNIT_PIXELS

        self._plot_widget = pg.PlotWidget(background="#111111")
        self._plot_widget.setMenuEnabled(False)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self._plot_item = self._plot_widget.getPlotItem()
        self._view_box = self._plot_item.getViewBox()
        self._plot_item.hideAxis("bottom")
        self._plot_item.hideAxis("left")
        self._overlay_bars = pg.BarGraphItem(x0=[], x1=[], y0=[], height=[], pen=None, brush=None)
        self._overlay_bars.setZValue(-10)
        self._plot_widget.addItem(self._overlay_bars)
        self._plot = self._plot_widget.plot([], [], pen=pg.mkPen("#00d7ff", width=2))

        self._marker = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen("#ff4d4d", width=1, style=Qt.PenStyle.DashLine),
        )
        self._marker.sigPositionChanged.connect(self._on_marker_move)
        self._plot_widget.addItem(self._marker)

        self._horizontal_ruler = ColorAxisRuler(Qt.Orientation.Horizontal, self)
        self._vertical_ruler = ColorAxisRuler(Qt.Orientation.Vertical, self)
        self._horizontal_axis_title = AxisTitleLabel(Qt.Orientation.Horizontal, self)
        self._vertical_axis_title = AxisTitleLabel(Qt.Orientation.Vertical, self)
        self._ruler_title_corner = QWidget(self)
        self._ruler_corner = QWidget(self)
        self._bottom_left_spacer = QWidget(self)
        self._ruler_title_corner.setFixedSize(
            self._vertical_axis_title.width(),
            self._horizontal_ruler.height(),
        )
        self._ruler_corner.setFixedSize(
            self._vertical_ruler.width(),
            self._horizontal_ruler.height(),
        )
        self._bottom_left_spacer.setFixedHeight(self._horizontal_axis_title.height())
        self._ruler_title_corner.setStyleSheet("background-color: #171717;")
        self._ruler_corner.setStyleSheet("background-color: #171717;")
        self._bottom_left_spacer.setStyleSheet("background-color: #171717;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._view_ruler_layout = QGridLayout()
        self._view_ruler_layout.setContentsMargins(0, 0, 0, 0)
        self._view_ruler_layout.setSpacing(0)
        self._view_ruler_layout.addWidget(self._vertical_axis_title, 0, 0)
        self._view_ruler_layout.addWidget(self._vertical_ruler, 0, 1)
        self._view_ruler_layout.addWidget(self._plot_widget, 0, 2)
        self._view_ruler_layout.addWidget(self._ruler_title_corner, 1, 0)
        self._view_ruler_layout.addWidget(self._ruler_corner, 1, 1)
        self._view_ruler_layout.addWidget(self._horizontal_ruler, 1, 2)
        self._view_ruler_layout.addWidget(self._bottom_left_spacer, 2, 0, 1, 2)
        self._view_ruler_layout.addWidget(self._horizontal_axis_title, 2, 2)
        self._view_ruler_layout.setColumnStretch(2, 1)
        self._view_ruler_layout.setRowStretch(0, 1)
        layout.addLayout(self._view_ruler_layout, 1)

        self.setStyleSheet("background-color: #181818; color: #cccccc;")
        self.set_ruler_axis_names(
            horizontal=self._display_axis_x_name,
            vertical=self._display_axis_y_name,
        )
        self._view_box.sigRangeChanged.connect(self._on_plot_range_changed)
        self._clear_rulers()

    def set_signal(self, signal: np.ndarray, positions: Optional[np.ndarray] = None) -> None:
        """Assign the signal expressed in percentage against optional position samples."""
        if signal is None or signal.size == 0:
            self._signal = None
            self._positions = None
            self._source_positions = None
            self._plot.setData([])
            self._refresh_overlay_bars()
            self._clear_rulers()
            return
        data = np.nan_to_num(np.asarray(signal, dtype=np.float32), nan=0.0)
        if positions is not None:
            pos_array = np.asarray(positions, dtype=np.float32)
            self._source_positions = pos_array if pos_array.size == data.size else None
        else:
            self._source_positions = None
        self._signal = data
        self._rebuild_plot_axis()

    def set_marker(self, index: Optional[int]) -> None:
        if index is None or self._signal is None:
            return
        index = max(0, min(len(self._signal) - 1, index))
        if self._positions is not None and 0 <= index < len(self._positions):
            axis_value = float(self._positions[index])
        else:
            axis_value = float(index)
        self._set_marker_value(axis_value)
        self._marker_idx = index

    def clear(self) -> None:
        self._signal = None
        self._positions = None
        self._source_positions = None
        self._plot.setData([])
        self.set_overlay_segments(())
        self._clear_rulers()

    def set_marker_visible(self, visible: bool) -> None:
        """Show or hide the marker line."""
        self._marker.setVisible(visible)

    def set_overlay_segments(
        self,
        spans: Sequence[tuple[int, int, int]] | None,
        *,
        palette: Optional[dict[int, tuple[int, int, int, int]]] = None,
    ) -> None:
        """Assign projected mask spans rendered as translucent vertical bands."""
        self._overlay_spans = tuple(
            (int(start_idx), int(end_idx), int(label_id))
            for start_idx, end_idx, label_id in (spans or ())
        )
        self._overlay_palette = dict(palette or {})
        self._refresh_overlay_bars()

    def set_overlay_opacity(self, opacity: float) -> None:
        """Set the global overlay opacity applied to A-scan bands."""
        try:
            value = float(opacity)
        except (TypeError, ValueError):
            value = 1.0
        self._overlay_opacity = max(0.0, min(1.0, value))
        self._refresh_overlay_bars()

    def set_horizontal_axis_resolution_mm(self, resolution_mm: Optional[float]) -> None:
        """Store the NDE-specific sampling step used for future display conversion."""
        normalized = RulerDisplayService.normalize_resolution_mm(resolution_mm)
        if normalized == self._horizontal_axis_resolution_mm:
            return
        self._horizontal_axis_resolution_mm = normalized
        if self._signal is not None and self._signal.size > 0:
            self._rebuild_plot_axis()

    def set_ruler_display_unit(self, display_unit: Optional[str]) -> None:
        """Switch the horizontal display axis between pixels and millimeters."""
        normalized = RulerDisplayService.normalize_display_unit(display_unit)
        if normalized == self._ruler_display_unit:
            return
        self._ruler_display_unit = normalized
        if self._signal is not None and self._signal.size > 0:
            self._rebuild_plot_axis()

    def set_ruler_axis_names(self, *, horizontal: str, vertical: str) -> None:
        """Display the X/Y axis names used by the A-scan rulers."""
        self._display_axis_x_name = str(horizontal or "").strip()
        self._display_axis_y_name = str(vertical or "").strip()
        self._horizontal_ruler.set_axis_name(self._display_axis_x_name)
        self._vertical_ruler.set_axis_name(self._display_axis_y_name)
        self._horizontal_axis_title.set_axis_name(self._display_axis_x_name)
        self._vertical_axis_title.set_axis_name(self._display_axis_y_name)

    def _on_marker_move(self) -> None:
        if self._signal is None or self._suspend_marker_signal:
            return
        value = self._marker.value()
        if self._positions is not None and len(self._positions) == len(self._signal):
            idx = int(np.abs(self._positions - value).argmin())
        else:
            idx = int(round(value))
        idx = max(0, min(len(self._signal) - 1, idx))
        self._marker_idx = idx
        self.position_changed.emit(idx)

    def _set_marker_value(self, value: float) -> None:
        self._suspend_marker_signal = True
        try:
            self._marker.setValue(value)
        finally:
            self._suspend_marker_signal = False

    def _on_plot_range_changed(self, *_args) -> None:
        self._refresh_rulers()

    def _refresh_overlay_bars(self) -> None:
        if self._signal is None or self._signal.size == 0 or not self._overlay_spans:
            self._overlay_bars.setOpts(x0=[], x1=[], y0=[], height=[], brushes=[], pen=None)
            return

        left_edges, right_edges = self._sample_edges()
        if left_edges.size == 0 or right_edges.size == 0:
            self._overlay_bars.setOpts(x0=[], x1=[], y0=[], height=[], brushes=[], pen=None)
            return

        max_idx = int(self._signal.size) - 1
        x0_values: list[float] = []
        x1_values: list[float] = []
        brushes: list[QBrush] = []

        for raw_start, raw_end, raw_label in self._overlay_spans:
            start_idx = max(0, min(max_idx, int(raw_start)))
            end_idx = max(start_idx, min(max_idx, int(raw_end)))
            brush = self._brush_for_label(int(raw_label))
            if brush.color().alpha() <= 0:
                continue
            x0_values.append(float(left_edges[start_idx]))
            x1_values.append(float(right_edges[end_idx]))
            brushes.append(brush)

        if not x0_values:
            self._overlay_bars.setOpts(x0=[], x1=[], y0=[], height=[], brushes=[], pen=None)
            return

        count = len(x0_values)
        self._overlay_bars.setOpts(
            x0=np.asarray(x0_values, dtype=np.float32),
            x1=np.asarray(x1_values, dtype=np.float32),
            y0=np.zeros(count, dtype=np.float32),
            height=np.full(count, 100.0, dtype=np.float32),
            brushes=brushes,
            pen=None,
        )

    def _sample_edges(self) -> tuple[np.ndarray, np.ndarray]:
        if self._signal is None or self._signal.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        samples = (
            np.asarray(self._positions, dtype=np.float32)
            if self._positions is not None and self._positions.size == self._signal.size
            else RulerDisplayService.build_axis_values(
                sample_count=self._signal.size,
                source_positions=self._source_positions,
                resolution_mm=self._horizontal_axis_resolution_mm,
                display_unit=self._ruler_display_unit,
            )
        )
        return RulerDisplayService.build_sample_edges(samples)

    def _rebuild_plot_axis(self) -> None:
        if self._signal is None or self._signal.size == 0:
            self._positions = None
            self._plot.setData([], [])
            self._refresh_overlay_bars()
            self._clear_rulers()
            return

        x_axis = RulerDisplayService.build_axis_values(
            sample_count=len(self._signal),
            source_positions=self._source_positions,
            resolution_mm=self._horizontal_axis_resolution_mm,
            display_unit=self._ruler_display_unit,
        )
        self._positions = x_axis
        self._plot.setData(x_axis, self._signal)
        if len(x_axis) > 0:
            self._plot_widget.setXRange(float(x_axis[0]), float(x_axis[-1]), padding=0)
        self._plot_widget.setYRange(0, 100, padding=0.05)
        self._refresh_overlay_bars()
        self._refresh_rulers()

    def _brush_for_label(self, label_id: int) -> QBrush:
        bgra = self._overlay_palette.get(int(label_id), (255, 0, 255, 160))
        b, g, r, a = (int(value) for value in bgra)
        alpha = max(0, min(255, int(round(a * self._overlay_opacity))))
        return pg.mkBrush(QColor(r, g, b, alpha))

    def _clear_rulers(self) -> None:
        if not hasattr(self, "_horizontal_ruler") or not hasattr(self, "_vertical_ruler"):
            return
        self._horizontal_ruler.clear_range()
        self._vertical_ruler.clear_range()

    def _refresh_rulers(self) -> None:
        if not hasattr(self, "_horizontal_ruler") or not hasattr(self, "_vertical_ruler"):
            return
        if self._signal is None or self._signal.size == 0:
            self._clear_rulers()
            return

        try:
            x_range, y_range = self._view_box.viewRange()
        except Exception:
            self._clear_rulers()
            return

        left_edges, right_edges = self._sample_edges()
        if left_edges.size == 0 or right_edges.size == 0:
            self._clear_rulers()
            return

        content_min_x = float(min(left_edges[0], right_edges[-1]))
        content_max_x = float(max(left_edges[0], right_edges[-1]))
        self._horizontal_ruler.set_view_range(
            view_min=float(x_range[0]),
            view_max=float(x_range[1]),
            content_min=content_min_x,
            content_max=content_max_x,
        )
        self._vertical_ruler.set_view_range(
            view_min=float(y_range[0]),
            view_max=float(y_range[1]),
            content_min=0.0,
            content_max=100.0,
        )
