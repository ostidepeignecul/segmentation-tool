"""Small dialog to choose the display size of the endview viewport."""

from __future__ import annotations

from typing import Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class EndviewResizeDialog(QDialog):
    """Modal dialog that lets the user pick a display width/height in pixels."""

    def __init__(self, current_size: Tuple[int, int], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Redimensionner l'endview")
        self.setModal(True)

        width, height = current_size

        self._width_spin = QSpinBox(self)
        self._width_spin.setRange(100, 5000)
        self._width_spin.setValue(max(1, int(width)))

        self._height_spin = QSpinBox(self)
        self._height_spin.setRange(100, 5000)
        self._height_spin.setValue(max(1, int(height)))

        self._lock_square = QCheckBox("Forcer carré", self)
        self._lock_square.stateChanged.connect(self._sync_square_state)
        self._width_spin.valueChanged.connect(self._on_width_changed)

        form = QFormLayout()
        form.addRow(QLabel("Largeur (px)"), self._width_spin)
        form.addRow(QLabel("Hauteur (px)"), self._height_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addLayout(form)

        lock_row = QHBoxLayout()
        lock_row.addWidget(self._lock_square)
        lock_row.addStretch()
        layout.addLayout(lock_row)

        layout.addWidget(buttons, 0, alignment=Qt.AlignmentFlag.AlignRight)

    def get_size(self) -> Tuple[int, int]:
        """Return the chosen (width, height)."""
        return int(self._width_spin.value()), int(self._height_spin.value())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _sync_square_state(self, state: int) -> None:
        if state == Qt.CheckState.Checked.value:
            self._height_spin.setValue(self._width_spin.value())

    def _on_width_changed(self, value: int) -> None:
        if self._lock_square.isChecked():
            self._height_spin.setValue(int(value))
