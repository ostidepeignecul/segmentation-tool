from __future__ import annotations

from typing import Mapping

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.constants import format_label_text


class OverlayClassRemapDialog(QDialog):
    """Dialog used to remap imported NPZ class ids in memory."""

    def __init__(
        self,
        *,
        source_path: str,
        source_classes: tuple[int, ...],
        current_mapping: Mapping[int, int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Remap classes")
        self.setModal(True)
        self.resize(520, 420)

        self._source_classes = tuple(int(value) for value in source_classes)
        self._target_inputs: dict[int, QSpinBox] = {}
        self._target_preview_labels: dict[int, QLabel] = {}

        info_label = QLabel(
            "Remap en memoire depuis le NPZ importe. Le fichier source n'est jamais modifie.",
            self,
        )
        info_label.setWordWrap(True)

        path_label = QLabel(f"Source: {source_path}", self)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        path_label.setWordWrap(True)

        self._table = QTableWidget(len(self._source_classes), 3, self)
        self._table.setHorizontalHeaderLabels(["Classe source", "Classe cible", "Libelle cible"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        for row, source in enumerate(self._source_classes):
            source_item = QTableWidgetItem(str(int(source)))
            source_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 0, source_item)

            target_input = QSpinBox(self._table)
            target_input.setRange(0, 255)
            target_input.setValue(int(current_mapping.get(source, source)))
            if int(source) == 0:
                target_input.setEnabled(False)
            target_input.valueChanged.connect(
                lambda value, source_label=source: self._on_target_value_changed(source_label, value)
            )
            self._table.setCellWidget(row, 1, target_input)
            self._target_inputs[int(source)] = target_input

            preview_label = QLabel(self._target_label_text(int(target_input.value())), self._table)
            preview_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._table.setCellWidget(row, 2, preview_label)
            self._target_preview_labels[int(source)] = preview_label

        self._table.resizeColumnsToContents()
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)

        reset_button = QPushButton("Reset identite", self)
        reset_button.clicked.connect(self._reset_identity_mapping)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        footer = QHBoxLayout()
        footer.addWidget(reset_button)
        footer.addStretch()
        footer.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(info_label)
        layout.addWidget(path_label)
        layout.addWidget(self._table, 1)
        layout.addLayout(footer)

    def get_mapping(self) -> dict[int, int]:
        """Return the current source->target mapping selected in the dialog."""
        return {
            int(source): int(widget.value())
            for source, widget in self._target_inputs.items()
        }

    def _on_target_value_changed(self, source_class: int, value: int) -> None:
        preview_label = self._target_preview_labels.get(int(source_class))
        if preview_label is not None:
            preview_label.setText(self._target_label_text(int(value)))

    def _reset_identity_mapping(self) -> None:
        for source_class, target_input in self._target_inputs.items():
            target_input.blockSignals(True)
            target_input.setValue(int(source_class))
            target_input.blockSignals(False)
            self._on_target_value_changed(int(source_class), int(source_class))

    @staticmethod
    def _target_label_text(label_id: int) -> str:
        return format_label_text(int(label_id))
