from __future__ import annotations

from typing import Iterable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QDialog, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class CorrosionSettingsView(QDialog):
    """Floating window to select label pair for corrosion analysis."""

    label_a_changed = pyqtSignal(object)
    label_b_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Parametres corrosion")
        self.setModal(False)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._label_a_combo = QComboBox(self)
        self._label_b_combo = QComboBox(self)
        form.addRow(QLabel("Label A"), self._label_a_combo)
        form.addRow(QLabel("Label B"), self._label_b_combo)

        layout.addLayout(form)

        close_btn = QPushButton("Fermer", self)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, 0, alignment=Qt.AlignmentFlag.AlignRight)

        self._wire_signals()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_label_choices(
        self,
        labels: Iterable[int],
        *,
        current_a: Optional[int],
        current_b: Optional[int],
    ) -> None:
        """Populate label selectors and set current selections without emitting signals."""
        self._label_a_combo.blockSignals(True)
        self._label_b_combo.blockSignals(True)
        self._label_a_combo.clear()
        self._label_b_combo.clear()
        self._label_a_combo.addItem("Aucun", None)
        self._label_b_combo.addItem("Aucun", None)
        for label_id in labels:
            lbl = int(label_id)
            self._label_a_combo.addItem(f"Label {lbl}", lbl)
            self._label_b_combo.addItem(f"Label {lbl}", lbl)
        self._set_current_data(self._label_a_combo, current_a)
        self._set_current_data(self._label_b_combo, current_b)
        self._label_a_combo.blockSignals(False)
        self._label_b_combo.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _wire_signals(self) -> None:
        self._label_a_combo.currentIndexChanged.connect(self._on_label_a_changed)
        self._label_b_combo.currentIndexChanged.connect(self._on_label_b_changed)

    def _on_label_a_changed(self, _index: int) -> None:
        self.label_a_changed.emit(self._label_a_combo.currentData())

    def _on_label_b_changed(self, _index: int) -> None:
        self.label_b_changed.emit(self._label_b_combo.currentData())

    @staticmethod
    def _set_current_data(combo: QComboBox, value: Optional[int]) -> None:
        if value is None:
            combo.setCurrentIndex(0)
            return
        target = int(value)
        for idx in range(combo.count()):
            data = combo.itemData(idx)
            if data is None:
                continue
            try:
                if int(data) == target:
                    combo.setCurrentIndex(idx)
                    return
            except Exception:
                continue
        combo.setCurrentIndex(0)
