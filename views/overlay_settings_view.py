from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class OverlaySettingsView(QDialog):
    """Floating window to manage overlay label visibility and colors."""

    label_visibility_changed = pyqtSignal(int, bool)
    label_color_changed = pyqtSignal(int, QColor)
    label_added = pyqtSignal(int, QColor)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Paramètres overlay")
        self.setModal(False)
        self.setMinimumWidth(340)

        self._labels: Dict[int, _LabelRow] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._list_layout = QVBoxLayout(self._container)
        self._list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll, 1)

        buttons_layout = QHBoxLayout()
        self._add_zero_button = QPushButton("Ajouter label 0", self)
        self._add_zero_button.clicked.connect(self._on_add_zero_label)
        buttons_layout.addWidget(self._add_zero_button, 1)

        self._add_button = QPushButton("Ajouter un label", self)
        self._add_button.clicked.connect(self._on_add_label)
        buttons_layout.addWidget(self._add_button, 1)

        layout.addLayout(buttons_layout)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def clear_labels(self) -> None:
        """Remove all label rows."""
        for row in self._labels.values():
            row.setParent(None)
        self._labels.clear()

    def ensure_label(self, label_id: int, color: QColor, *, visible: bool = True) -> None:
        """Add the label row if absent, or refresh its color/visibility."""
        label_id = int(label_id)
        if label_id in self._labels:
            self._labels[label_id].set_color(color)
            self._labels[label_id].set_checked(visible)
            return
        row = _LabelRow(label_id, color, visible, parent=self)
        row.visibility_toggled.connect(self.label_visibility_changed)
        row.color_changed.connect(self.label_color_changed)
        self._labels[label_id] = row
        self._list_layout.addWidget(row)

    def set_labels(self, entries: Iterable[Tuple[int, QColor, bool]]) -> None:
        """Sync the view from an external list of (id, color, visible)."""
        self.clear_labels()
        seen = set()
        for label_id, color, visible in entries:
            self.ensure_label(label_id, color, visible=visible)
            seen.add(int(label_id))

    # ------------------------------------------------------------------ #
    # Color helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def bgra_to_qcolor(color: Tuple[int, int, int, int]) -> QColor:
        b, g, r, a = color
        return QColor(r, g, b, a)

    @staticmethod
    def qcolor_to_bgra(color: QColor) -> Tuple[int, int, int, int]:
        r, g, b, a = color.getRgb()
        return (b, g, r, a)

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _on_add_label(self) -> None:
        new_id = self._next_free_label_id()
        color = self._generate_color(len(self._labels))
        self.ensure_label(new_id, color, visible=True)
        self.label_added.emit(new_id, color)

    def _on_add_zero_label(self) -> None:
        if 0 in self._labels:
            return
        color = QColor(180, 180, 180, 200)
        self.ensure_label(0, color, visible=True)
        self.label_added.emit(0, color)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _next_free_label_id(self) -> int:
        return max(self._labels.keys(), default=0) + 1

    @staticmethod
    def _generate_color(index: int) -> QColor:
        """Generate a distinct color using a golden-angle hue wheel."""
        hue = int((index * 137.508) % 360)
        return QColor.fromHsv(hue, 255, 255)


class _LabelRow(QWidget):
    """Row widget holding a checkbox + color picker for one label."""

    visibility_toggled = pyqtSignal(int, bool)
    color_changed = pyqtSignal(int, QColor)

    def __init__(self, label_id: int, color: QColor, visible: bool, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.label_id = int(label_id)
        self._color = QColor(color)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._checkbox = QCheckBox(f"Label {self.label_id}", self)
        self._checkbox.setChecked(visible)
        self._checkbox.toggled.connect(self._on_visibility_toggled)
        layout.addWidget(self._checkbox, 1)

        self._color_button = QPushButton(self._color_name(), self)
        self._color_button.setFixedWidth(90)
        self._apply_color_style()
        self._color_button.clicked.connect(self._on_pick_color)
        layout.addWidget(self._color_button, 0)

    # ------------------------------------------------------------------ #
    # Public setters
    # ------------------------------------------------------------------ #
    def set_color(self, color: QColor) -> None:
        self._color = QColor(color)
        self._color_button.setText(self._color_name())
        self._apply_color_style()

    def set_checked(self, checked: bool) -> None:
        self._checkbox.blockSignals(True)
        self._checkbox.setChecked(checked)
        self._checkbox.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _on_visibility_toggled(self, checked: bool) -> None:
        self.visibility_toggled.emit(self.label_id, checked)

    def _on_pick_color(self) -> None:
        picked = QColorDialog.getColor(self._color, self, f"Couleur du label {self.label_id}")
        if picked.isValid():
            self.set_color(picked)
            self.color_changed.emit(self.label_id, picked)

    def _apply_color_style(self) -> None:
        self._color_button.setStyleSheet(
            f"background-color: {self._color.name()}; color: #000; font-weight: bold;"
        )

    def _color_name(self) -> str:
        return self._color.name()
