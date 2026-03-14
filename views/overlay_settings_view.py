from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from config.constants import (
    MASK_COLORS_BGRA,
    PERSISTENT_LABEL_IDS,
    USER_LABEL_START,
    format_label_text,
)


class OverlaySettingsView(QDialog):
    """Floating window to manage overlay label visibility and colors."""

    label_visibility_changed = pyqtSignal(int, bool)
    label_color_changed = pyqtSignal(int, QColor)
    label_added = pyqtSignal(int, QColor)
    label_deleted = pyqtSignal(int)
    overlay_opacity_changed = pyqtSignal(float)

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

        opacity_layout = QHBoxLayout()
        self._opacity_label = QLabel("Opacité overlay", self)
        opacity_layout.addWidget(self._opacity_label, 0)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self._opacity_slider, 1)
        self._opacity_value = QLabel("100%", self)
        self._opacity_value.setFixedWidth(48)
        self._opacity_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        opacity_layout.addWidget(self._opacity_value, 0)
        layout.addLayout(opacity_layout)

        buttons_layout = QHBoxLayout()
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
        row = _LabelRow(
            label_id,
            color,
            visible,
            parent=self,
            allow_delete=label_id not in PERSISTENT_LABEL_IDS,
        )
        row.visibility_toggled.connect(self.label_visibility_changed)
        row.color_changed.connect(self.label_color_changed)
        row.deleted.connect(self._on_label_deleted)
        self._labels[label_id] = row
        self._list_layout.addWidget(row)

    def set_labels(self, entries: Iterable[Tuple[int, QColor, bool]]) -> None:
        """Sync the view from an external list of (id, color, visible)."""
        self.clear_labels()
        for label_id, color, visible in entries:
            self.ensure_label(label_id, color, visible=visible)

    def set_overlay_opacity(self, value: float) -> None:
        """Set the overlay opacity slider (0.0 - 1.0)."""
        opacity = max(0.0, min(1.0, float(value)))
        percent = int(round(opacity * 100.0))
        self._opacity_slider.blockSignals(True)
        self._opacity_slider.setValue(percent)
        self._opacity_slider.blockSignals(False)
        self._update_opacity_label(percent)

    def _on_label_deleted(self, label_id: int) -> None:
        """Remove row then propagate deletion event."""
        lbl = int(label_id)
        if lbl in PERSISTENT_LABEL_IDS:
            return
        row = self._labels.pop(lbl, None)
        if row is not None:
            row.setParent(None)
            row.deleteLater()
        self.label_deleted.emit(lbl)

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
        palette_color = self._palette_color_for_label(new_id)
        color = palette_color or self._generate_color(len(self._labels))
        self.ensure_label(new_id, color, visible=True)
        self.label_added.emit(new_id, color)

    def _on_opacity_changed(self, value: int) -> None:
        self._update_opacity_label(value)
        self.overlay_opacity_changed.emit(float(value) / 100.0)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _next_free_label_id(self) -> int:
        candidate = int(USER_LABEL_START)
        while candidate in self._labels:
            candidate += 1
        return candidate

    @staticmethod
    def _palette_color_for_label(label_id: int) -> Optional[QColor]:
        """Return palette color for a label if defined in MASK_COLORS_BGRA."""
        color = MASK_COLORS_BGRA.get(int(label_id))
        if color is None:
            return None
        b, g, r, a = color
        return QColor(r, g, b, a)

    @staticmethod
    def _generate_color(index: int) -> QColor:
        """Generate a distinct color using a golden-angle hue wheel."""
        hue = int((index * 137.508) % 360)
        return QColor.fromHsv(hue, 255, 255)

    def _update_opacity_label(self, value: int) -> None:
        self._opacity_value.setText(f"{int(value)}%")


class _LabelRow(QWidget):
    """Row widget holding a checkbox + color picker for one label."""

    visibility_toggled = pyqtSignal(int, bool)
    color_changed = pyqtSignal(int, QColor)
    deleted = pyqtSignal(int)

    def __init__(
        self,
        label_id: int,
        color: QColor,
        visible: bool,
        parent: Optional[QWidget],
        *,
        allow_delete: bool = True,
    ) -> None:
        super().__init__(parent)
        self.label_id = int(label_id)
        self._color = QColor(color)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._checkbox = QCheckBox(format_label_text(self.label_id), self)
        self._checkbox.setChecked(visible)
        self._checkbox.toggled.connect(self._on_visibility_toggled)
        layout.addWidget(self._checkbox, 1)

        self._color_button = QPushButton(self._color_name(), self)
        self._color_button.setFixedWidth(90)
        self._apply_color_style()
        self._color_button.clicked.connect(self._on_pick_color)
        layout.addWidget(self._color_button, 0)

        self._delete_button: Optional[QPushButton] = None
        if allow_delete:
            self._delete_button = QPushButton("Supprimer", self)
            self._delete_button.setFixedWidth(90)
            self._delete_button.clicked.connect(self._on_delete_clicked)
            layout.addWidget(self._delete_button, 0)

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
        picked = QColorDialog.getColor(self._color, self, f"Couleur pour {format_label_text(self.label_id)}")
        if picked.isValid():
            self.set_color(picked)
            self.color_changed.emit(self.label_id, picked)

    def _apply_color_style(self) -> None:
        self._color_button.setStyleSheet(
            f"background-color: {self._color.name()}; color: #000; font-weight: bold;"
        )

    def _color_name(self) -> str:
        return self._color.name()

    def _on_delete_clicked(self) -> None:
        self.deleted.emit(self.label_id)
