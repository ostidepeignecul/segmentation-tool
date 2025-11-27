"""A-scan view built with PyQtGraph."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QVBoxLayout


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

        self._plot_widget = pg.PlotWidget(background="#111111")
        self._plot_widget.setMenuEnabled(False)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self._plot_widget.setLabel("bottom", "Position Y")
        self._plot_widget.setLabel("left", "Amplitude (%)")
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
