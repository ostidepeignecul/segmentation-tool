"""Dialog used to choose NDE opening options."""

from __future__ import annotations

from typing import Sequence, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QVBoxLayout,
)


class NdeOpenOptionsDialog(QDialog):
    """Modal dialog for axis mode and optional signal processing selection."""

    def __init__(
        self,
        *,
        axis_choices: Sequence[str],
        current_axis_mode: str,
        detected_title: str,
        detected_lines: Sequence[str],
        default_apply_hilbert: bool,
        default_apply_smoothing: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ouverture NDE")
        self.setModal(True)

        self._axis_combo = QComboBox(self)
        for choice in axis_choices:
            self._axis_combo.addItem(str(choice))
        current_idx = self._axis_combo.findText(str(current_axis_mode))
        if current_idx >= 0:
            self._axis_combo.setCurrentIndex(current_idx)

        self._hilbert_checkbox = QCheckBox("Appliquer l'enveloppe de Hilbert", self)
        self._hilbert_checkbox.setChecked(bool(default_apply_hilbert))

        self._smoothing_checkbox = QCheckBox("Appliquer le lissage general", self)
        self._smoothing_checkbox.setChecked(bool(default_apply_smoothing))

        summary_title = QLabel(str(detected_title), self)
        summary_title.setStyleSheet("font-weight: 600;")

        summary_body = QLabel("\n".join(str(line) for line in detected_lines), self)
        summary_body.setWordWrap(True)
        summary_body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        hint = QLabel(
            "Laissez les deux options decochees pour conserver le signal tel qu'il apparait dans le NDE.",
            self,
        )
        hint.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Plan d'annotation", self._axis_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addLayout(form)
        layout.addWidget(separator)
        layout.addWidget(summary_title)
        layout.addWidget(summary_body)
        layout.addWidget(hint)
        layout.addWidget(self._hilbert_checkbox)
        layout.addWidget(self._smoothing_checkbox)
        layout.addWidget(buttons, 0, alignment=Qt.AlignmentFlag.AlignRight)

    def get_selection(self) -> Tuple[str, bool, bool]:
        """Return (axis_mode, apply_hilbert, apply_smoothing)."""
        return (
            str(self._axis_combo.currentText()),
            bool(self._hilbert_checkbox.isChecked()),
            bool(self._smoothing_checkbox.isChecked()),
        )
