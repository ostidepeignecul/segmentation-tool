"""A-scan view built with PyQtGraph."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import QFrame, QVBoxLayout

from views.color_axis_ruler import ColorAxisRuler


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
        self._overlay_spans: tuple[tuple[int, int, int], ...] = ()
        self._overlay_palette: dict[int, tuple[int, int, int, int]] = {}
        self._overlay_opacity: float = 0.4

        self._plot_widget = pg.PlotWidget(background="#111111")
        self._plot_widget.setMenuEnabled(False)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self._plot_widget.setLabel("bottom", "Profondeur")
        self._plot_widget.setLabel("left", "Amplitude (%)")
        self._apply_axis_colors()
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_widget)

    def set_signal(self, signal: np.ndarray, positions: Optional[np.ndarray] = None) -> None:
        """Assign the signal expressed in percentage against optional position samples."""
        if signal is None or signal.size == 0:
            self._signal = None
            self._positions = None
            self._plot.setData([])
            self._refresh_overlay_bars()
            return
        data = np.nan_to_num(np.asarray(signal, dtype=np.float32), nan=0.0)
        if positions is not None:
            pos_array = np.asarray(positions, dtype=np.float32)
            self._positions = pos_array if pos_array.size == data.size else None
        else:
            self._positions = None
        x_axis = (
            self._positions
            if self._positions is not None
            else np.arange(len(data), dtype=np.float32)
        )
        self._signal = data
        self._plot.setData(x_axis, data)
        if len(x_axis) > 0:
            self._plot_widget.setXRange(float(x_axis[0]), float(x_axis[-1]), padding=0)
        self._plot_widget.setYRange(0, 100, padding=0.05)
        self._refresh_overlay_bars()

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
        self._plot.setData([])
        self.set_overlay_segments(())

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

        if self._positions is not None and self._positions.size == self._signal.size:
            samples = np.asarray(self._positions, dtype=np.float32)
            if not np.all(np.isfinite(samples)):
                samples = np.arange(self._signal.size, dtype=np.float32)
        else:
            samples = np.arange(self._signal.size, dtype=np.float32)

        if samples.size == 1:
            center = float(samples[0])
            return (
                np.asarray([center - 0.5], dtype=np.float32),
                np.asarray([center + 0.5], dtype=np.float32),
            )

        midpoints = (samples[:-1] + samples[1:]) / 2.0
        left_edges = np.empty_like(samples, dtype=np.float32)
        right_edges = np.empty_like(samples, dtype=np.float32)
        left_edges[0] = samples[0] - (midpoints[0] - samples[0])
        left_edges[1:] = midpoints
        right_edges[:-1] = midpoints
        right_edges[-1] = samples[-1] + (samples[-1] - midpoints[-1])
        return left_edges, right_edges

    def _brush_for_label(self, label_id: int) -> QBrush:
        bgra = self._overlay_palette.get(int(label_id), (255, 0, 255, 160))
        b, g, r, a = (int(value) for value in bgra)
        alpha = max(0, min(255, int(round(a * self._overlay_opacity))))
        return pg.mkBrush(QColor(r, g, b, alpha))

    def _apply_axis_colors(self) -> None:
        """Align A-scan axis styling with the shared ruler color mapping."""
        bottom_axis = self._plot_widget.getAxis("bottom")
        left_axis = self._plot_widget.getAxis("left")

        depth_color = ColorAxisRuler.axis_color_for_name("Profondeur") or QColor("#9b649b")
        amplitude_color = ColorAxisRuler.axis_color_for_name("Amplitude") or QColor("#cfff53")

        bottom_axis.setLabel("Profondeur", color=depth_color.name())
        bottom_axis.setPen(pg.mkPen(depth_color))
        bottom_axis.setTextPen(pg.mkPen(depth_color))

        left_axis.setLabel("Amplitude (%)", color=amplitude_color.name())
        left_axis.setPen(pg.mkPen(amplitude_color))
        left_axis.setTextPen(pg.mkPen(amplitude_color))
