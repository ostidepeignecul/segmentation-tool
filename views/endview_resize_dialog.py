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
        self._reset_requested = False
        self._square_sync_guard = False

        width, height = current_size

        self._width_spin = QSpinBox(self)
        self._width_spin.setRange(100, 5000)
        self._width_spin.setValue(max(1, int(width)))

        self._height_spin = QSpinBox(self)
        self._height_spin.setRange(100, 5000)
        self._height_spin.setValue(max(1, int(height)))

        self._lock_square = QCheckBox("Forcer carre", self)
        self._lock_square.stateChanged.connect(self._sync_square_state)
        self._width_spin.valueChanged.connect(self._on_width_changed)
        self._height_spin.valueChanged.connect(self._on_height_changed)

        form = QFormLayout()
        form.addRow(QLabel("Largeur (px)"), self._width_spin)
        form.addRow(QLabel("Hauteur (px)"), self._height_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        reset_button = buttons.addButton("Par defaut", QDialogButtonBox.ButtonRole.ResetRole)
        reset_button.clicked.connect(self._on_reset_clicked)
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

    def wants_reset(self) -> bool:
        """Return True if the user requested a reset."""
        return self._reset_requested

    def is_square_locked(self) -> bool:
        """Return True when force-square is enabled."""
        return self._lock_square.isChecked()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _sync_square_state(self, state: int) -> None:
        if state == Qt.CheckState.Checked.value:
            self._sync_square_dimension(source="width", value=int(self._width_spin.value()))

    def _on_width_changed(self, value: int) -> None:
        self._sync_square_dimension(source="width", value=int(value))

    def _on_height_changed(self, value: int) -> None:
        self._sync_square_dimension(source="height", value=int(value))

    def _sync_square_dimension(self, *, source: str, value: int) -> None:
        if not self._lock_square.isChecked() or self._square_sync_guard:
            return
        self._square_sync_guard = True
        try:
            if source == "width":
                self._height_spin.setValue(int(value))
            else:
                self._width_spin.setValue(int(value))
        finally:
            self._square_sync_guard = False

    def _on_reset_clicked(self) -> None:
        self._reset_requested = True
        self.accept()
