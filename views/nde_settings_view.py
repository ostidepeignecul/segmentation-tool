"""Simple settings dialog to pick colormaps for views."""

from __future__ import annotations

from typing import Iterable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class NdeSettingsView(QDialog):
    """Floating window to select colormaps for endview/3D and C-scan."""

    endview_colormap_changed = pyqtSignal(str)
    cscan_colormap_changed = pyqtSignal(str)
    apply_volume_range_changed = pyqtSignal(int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Paramètres")
        self.setModal(False)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._endview_combo = QComboBox(self)
        self._cscan_combo = QComboBox(self)
        form.addRow(QLabel("Colormap Endview + 3D"), self._endview_combo)
        form.addRow(QLabel("Colormap C-scan"), self._cscan_combo)

        self._apply_volume_start = QSpinBox(self)
        self._apply_volume_end = QSpinBox(self)
        for box in (self._apply_volume_start, self._apply_volume_end):
            box.setMinimum(0)
            box.setMaximum(0)
            box.setValue(0)
        form.addRow(QLabel("Appliquer au volume (de)"), self._apply_volume_start)
        form.addRow(QLabel("Appliquer au volume (à)"), self._apply_volume_end)

        layout.addLayout(form)

        close_btn = QPushButton("Fermer", self)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, 0, alignment=Qt.AlignmentFlag.AlignRight)

        self._populate(["Gris", "OmniScan"])
        self._wire_signals()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_colormaps(self, *, endview: str, cscan: str) -> None:
        """Populate combos with the provided current values."""
        self._set_current(self._endview_combo, endview)
        self._set_current(self._cscan_combo, cscan)

    def set_apply_volume_bounds(self, minimum: int, maximum: int) -> None:
        """Set min/max bounds for apply-to-volume range."""
        min_idx = int(minimum)
        max_idx = int(maximum)
        if max_idx < min_idx:
            max_idx = min_idx
        for box in (self._apply_volume_start, self._apply_volume_end):
            box.blockSignals(True)
            box.setMinimum(min_idx)
            box.setMaximum(max_idx)
            box.blockSignals(False)

    def set_apply_volume_range(self, start: int, end: int) -> None:
        """Update apply-to-volume range without emitting signals."""
        start_idx = int(start)
        end_idx = int(end)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        self._apply_volume_start.blockSignals(True)
        self._apply_volume_end.blockSignals(True)
        self._apply_volume_start.setValue(start_idx)
        self._apply_volume_end.setValue(end_idx)
        self._apply_volume_start.blockSignals(False)
        self._apply_volume_end.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _populate(self, choices: Iterable[str]) -> None:
        self._endview_combo.clear()
        self._cscan_combo.clear()
        for choice in choices:
            self._endview_combo.addItem(choice)
            self._cscan_combo.addItem(choice)

    def _wire_signals(self) -> None:
        self._endview_combo.currentTextChanged.connect(self.endview_colormap_changed.emit)
        self._cscan_combo.currentTextChanged.connect(self.cscan_colormap_changed.emit)
        self._apply_volume_start.valueChanged.connect(self._on_apply_volume_start_changed)
        self._apply_volume_end.valueChanged.connect(self._on_apply_volume_end_changed)

    def _emit_apply_volume_range(self) -> None:
        self.apply_volume_range_changed.emit(
            int(self._apply_volume_start.value()),
            int(self._apply_volume_end.value()),
        )

    def _on_apply_volume_start_changed(self, value: int) -> None:
        if value > self._apply_volume_end.value():
            self._apply_volume_end.blockSignals(True)
            self._apply_volume_end.setValue(int(value))
            self._apply_volume_end.blockSignals(False)
        self._emit_apply_volume_range()

    def _on_apply_volume_end_changed(self, value: int) -> None:
        if value < self._apply_volume_start.value():
            self._apply_volume_start.blockSignals(True)
            self._apply_volume_start.setValue(int(value))
            self._apply_volume_start.blockSignals(False)
        self._emit_apply_volume_range()

    @staticmethod
    def _set_current(combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx < 0:
            combo.addItem(value)
            idx = combo.findText(value)
        combo.setCurrentIndex(idx)
