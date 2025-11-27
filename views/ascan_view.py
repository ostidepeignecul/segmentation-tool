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

        self._plot_widget = pg.PlotWidget(background="#111111")
        self._plot_widget.setMenuEnabled(False)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self._plot = self._plot_widget.plot([], [], pen=pg.mkPen("#00d7ff", width=2))

        self._marker = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen("#ff4d4d", width=1, style=Qt.PenStyle.DashLine))
        self._marker.sigPositionChanged.connect(self._on_marker_move)
        self._plot_widget.addItem(self._marker)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_widget)

    def set_signal(self, signal: np.ndarray) -> None:
        """Assign the signal (float array)."""
        if signal is None or signal.size == 0:
            self._signal = None
            self._plot.setData([])
            return
        data = np.asarray(signal, dtype=np.float32)
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())
        else:
            data = np.zeros_like(data)
        self._signal = data
        self._plot.setData(data)
        self._marker.setValue(len(data) // 2)

    def set_marker(self, index: Optional[int]) -> None:
        if index is None or self._signal is None:
            return
        index = max(0, min(len(self._signal) - 1, index))
        self._marker.setValue(index)
        self._marker_idx = index

    def clear(self) -> None:
        self._signal = None
        self._plot.setData([])

    def _on_marker_move(self) -> None:
        if self._signal is None:
            return
        idx = int(round(self._marker.value()))
        idx = max(0, min(len(self._signal) - 1, idx))
        self._marker_idx = idx
        self.position_changed.emit(idx)
