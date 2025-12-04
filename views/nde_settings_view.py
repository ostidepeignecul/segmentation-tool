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
    QVBoxLayout,
    QWidget,
)


class NdeSettingsView(QDialog):
    """Floating window to select colormaps for endview/3D and C-scan."""

    endview_colormap_changed = pyqtSignal(str)
    cscan_colormap_changed = pyqtSignal(str)

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

    @staticmethod
    def _set_current(combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx < 0:
            combo.addItem(value)
            idx = combo.findText(value)
        combo.setCurrentIndex(idx)
