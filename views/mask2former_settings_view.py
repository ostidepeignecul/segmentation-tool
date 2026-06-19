"""Floating window to configure Mask2Former inference settings."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class Mask2FormerSettingsView(QDialog):
    """Floating window to configure the Mask2Former model path."""

    model_path_changed = pyqtSignal(str)
    choose_directory_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mask2Former settings")
        self.setModal(False)
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._model_path_edit = QLineEdit(self)
        self._model_path_edit.setPlaceholderText(
            "Choose a local Mask2Former model folder"
        )
        form.addRow(QLabel("Model path"), self._model_path_edit)
        layout.addLayout(form)

        hint = QLabel("Accepted inputs: local pre-trained Mask2Former model folder containing config.json.", self)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        browse_row = QHBoxLayout()
        self._choose_folder_btn = QPushButton("Choose folder", self)
        self._clear_btn = QPushButton("Clear", self)
        browse_row.addWidget(self._choose_folder_btn)
        browse_row.addWidget(self._clear_btn)
        browse_row.addStretch(1)
        layout.addLayout(browse_row)

        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, 0, alignment=Qt.AlignmentFlag.AlignRight)

        self._wire_signals()

    def set_model_path(self, path: str) -> None:
        """Update the displayed model path without emitting change signals."""
        normalized = str(path or "").strip()
        self._model_path_edit.blockSignals(True)
        self._model_path_edit.setText(normalized)
        self._model_path_edit.blockSignals(False)

    def model_path(self) -> str:
        """Return the current model path."""
        return self._model_path_edit.text().strip()

    def _wire_signals(self) -> None:
        self._model_path_edit.textChanged.connect(self._on_model_path_text_changed)
        self._choose_folder_btn.clicked.connect(self.choose_directory_requested.emit)
        self._clear_btn.clicked.connect(self._model_path_edit.clear)

    def _on_model_path_text_changed(self, value: str) -> None:
        self.model_path_changed.emit(str(value or "").strip())
