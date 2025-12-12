"""Dialog for NPZ overlay export options."""

from __future__ import annotations

from typing import NamedTuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout


class OverlayExportOptions(NamedTuple):
    """Holds user-selected export options."""

    mirror_vertical: bool = False
    rotation_degrees: int = 0


class OverlayExportDialog(QDialog):
    """Modal dialog letting the user choose export options before saving."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Options d'export overlay")
        self.setModal(True)

        self._mirror_vertical = QCheckBox("Miroir sur l'axe vertical (gauche/droite)", self)
        self._rotation_label = QLabel("Rotation (sens antihoraire)")
        self._rotation = QComboBox(self)
        self._rotation.addItems(["0°", "90°", "180°", "270°"])
        self._rotation.setCurrentIndex(0)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self._rotation_label)
        layout.addWidget(self._rotation)
        layout.addWidget(self._mirror_vertical)
        layout.addStretch()
        layout.addWidget(buttons, 0, alignment=Qt.AlignmentFlag.AlignRight)

    def get_options(self) -> OverlayExportOptions:
        """Return the selected export options."""
        rotation_text = self._rotation.currentText().replace("°", "").strip()
        try:
            rotation_value = int(rotation_text)
        except Exception:
            rotation_value = 0
        if rotation_value not in (0, 90, 180, 270):
            rotation_value = 0
        return OverlayExportOptions(
            mirror_vertical=bool(self._mirror_vertical.isChecked()),
            rotation_degrees=rotation_value,
        )
