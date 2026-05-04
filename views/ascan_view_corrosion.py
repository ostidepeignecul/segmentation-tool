"""A-Scan corrosion view with distance overlay."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from services.ruler_display_service import RulerDisplayService
from views.ascan_view import AScanView


class AScanViewCorrosion(AScanView):
    """Displays A-Scan with a measurement line between corrosion peaks."""

    _MEASURE_COLOR = "#ffb347"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        pen = pg.mkPen(self._MEASURE_COLOR, width=2)
        self._measurement_line_a = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self._measurement_line_b = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self._measurement_span = pg.PlotDataItem([], [], pen=pen)
        self._measurement_label = pg.TextItem(
            "",
            color=self._MEASURE_COLOR,
            anchor=(0.5, 1.0),
        )

        self._plot_widget.addItem(self._measurement_line_a)
        self._plot_widget.addItem(self._measurement_line_b)
        self._plot_widget.addItem(self._measurement_span)
        self._plot_widget.addItem(self._measurement_label)
        self.clear_measurement()

    def set_signal(self, signal: np.ndarray, positions: np.ndarray | None = None) -> None:  # type: ignore[override]
        super().set_signal(signal, positions=positions)
        if self._signal is None or self._signal.size == 0:
            self.clear_measurement()

    def set_measurement_indices(
        self,
        idx_a: int,
        idx_b: int,
        *,
        distance_px: float | None = None,
    ) -> None:
        """Set the measurement line using sample indices along the A-Scan."""
        if self._signal is None or self._signal.size == 0:
            self.clear_measurement()
            return

        max_idx = int(self._signal.size) - 1
        if max_idx < 0:
            self.clear_measurement()
            return

        a_idx = max(0, min(max_idx, int(round(idx_a))))
        b_idx = max(0, min(max_idx, int(round(idx_b))))

        if self._positions is not None and self._positions.size == self._signal.size:
            x_axis = self._positions
        else:
            x_axis = np.arange(self._signal.size, dtype=np.float32)

        x1 = float(x_axis[a_idx])
        x2 = float(x_axis[b_idx])
        y1 = float(self._signal[a_idx])
        y2 = float(self._signal[b_idx])

        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_measure = min(100.0, max(y1, y2) + 5.0)

        self._measurement_line_a.setValue(x1)
        self._measurement_line_b.setValue(x2)
        self._measurement_span.setData([x_min, x_max], [y_measure, y_measure])

        label_distance_px = distance_px if distance_px is not None and np.isfinite(distance_px) else abs(b_idx - a_idx)
        label = RulerDisplayService.format_distance(
            label_distance_px,
            display_unit=getattr(self, "_ruler_display_unit", None),
            resolution_mm=getattr(self, "_horizontal_axis_resolution_mm", None),
            px_decimals=1,
            mm_decimals=1,
        )

        label_x = (x_min + x_max) / 2.0
        label_y = max(0.0, y_measure - 10.0)
        self._measurement_label.setText(label)
        self._measurement_label.setPos(label_x, label_y)
        self._set_measurement_visible(True)

    def clear_measurement(self) -> None:
        """Hide the corrosion measurement overlay."""
        self._measurement_span.setData([], [])
        self._measurement_label.setText("")
        self._set_measurement_visible(False)

    def _set_measurement_visible(self, visible: bool) -> None:
        self._measurement_line_a.setVisible(visible)
        self._measurement_line_b.setVisible(visible)
        self._measurement_span.setVisible(visible)
        self._measurement_label.setVisible(visible)
