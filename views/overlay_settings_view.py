from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
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

    layer_selected = pyqtSignal(str)
    layer_visibility_changed = pyqtSignal(str, bool)
    layer_created = pyqtSignal()
    layer_duplicated = pyqtSignal()
    layer_deleted = pyqtSignal(str)
    label_visibility_changed = pyqtSignal(int, bool)
    label_color_changed = pyqtSignal(int, QColor)
    label_added = pyqtSignal(int, QColor)
    label_deleted = pyqtSignal(int)
    overlay_opacity_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Overlay settings")
        self.setModal(False)
        self.setMinimumWidth(340)

        self._active_layer_id: Optional[str] = None
        self._layers: Dict[str, _LayerRow] = {}
        self._labels: Dict[int, _LabelRow] = {}
        self._layer_group = QButtonGroup(self)
        self._layer_group.setExclusive(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._list_layout = QVBoxLayout(self._container)
        self._list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._layers_title = QLabel("Layers", self._container)
        self._layers_title.setStyleSheet("font-weight: bold;")
        self._list_layout.addWidget(self._layers_title)
        self._layers_layout = QVBoxLayout()
        self._layers_layout.setContentsMargins(0, 0, 0, 0)
        self._layers_layout.setSpacing(4)
        self._list_layout.addLayout(self._layers_layout)

        self._layer_buttons_row = QHBoxLayout()
        self._layer_add_button = QPushButton("Add layer", self._container)
        self._layer_duplicate_button = QPushButton("Duplicate", self._container)
        self._layer_delete_button = QPushButton("Delete", self._container)
        self._layer_add_button.clicked.connect(lambda: self.layer_created.emit())
        self._layer_duplicate_button.clicked.connect(lambda: self.layer_duplicated.emit())
        self._layer_delete_button.clicked.connect(self._on_delete_active_layer)
        self._layer_buttons_row.addWidget(self._layer_add_button, 1)
        self._layer_buttons_row.addWidget(self._layer_duplicate_button, 1)
        self._layer_buttons_row.addWidget(self._layer_delete_button, 1)
        self._list_layout.addLayout(self._layer_buttons_row)

        self._separator = QFrame(self._container)
        self._separator.setFrameShape(QFrame.Shape.HLine)
        self._separator.setFrameShadow(QFrame.Shadow.Sunken)
        self._list_layout.addWidget(self._separator)

        self._labels_title = QLabel("Labels", self._container)
        self._labels_title.setStyleSheet("font-weight: bold;")
        self._list_layout.addWidget(self._labels_title)
        self._labels_layout = QVBoxLayout()
        self._labels_layout.setContentsMargins(0, 0, 0, 0)
        self._labels_layout.setSpacing(4)
        self._list_layout.addLayout(self._labels_layout)

        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll, 1)

        opacity_layout = QHBoxLayout()
        self._opacity_label = QLabel("Overlay opacity", self)
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
        self._add_button = QPushButton("Add label", self)
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
            row.deleteLater()
        self._labels.clear()

    def clear_layers(self) -> None:
        """Remove all layer rows."""
        self._active_layer_id = None
        self._layer_group.setExclusive(False)
        for row in self._layers.values():
            self._layer_group.removeButton(row.active_button())
            row.setParent(None)
            row.deleteLater()
        self._layer_group.setExclusive(True)
        self._layers.clear()
        self._update_layer_buttons_enabled()

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
        self._labels_layout.addWidget(row)

    def set_labels(self, entries: Iterable[Tuple[int, QColor, bool]]) -> None:
        """Sync the view from an external list of (id, color, visible)."""
        self.clear_labels()
        for label_id, color, visible in entries:
            self.ensure_label(label_id, color, visible=visible)

    def set_layers(self, entries: Iterable[Tuple[str, str, bool, bool]]) -> None:
        """Sync the view from an external list of (id, name, visible, is_active)."""
        normalized_entries: list[Tuple[str, str, bool, bool]] = []
        for layer_id, name, visible, is_active in entries:
            normalized_id = str(layer_id or "").strip()
            if not normalized_id:
                continue
            normalized_entries.append(
                (
                    normalized_id,
                    str(name or "Layer"),
                    bool(visible),
                    bool(is_active),
                )
            )

        ordered_ids = [layer_id for layer_id, _name, _visible, _is_active in normalized_entries]
        if list(self._layers.keys()) != ordered_ids:
            self.clear_layers()
        self._active_layer_id = None
        for layer_id, name, visible, is_active in normalized_entries:
            self.ensure_layer(layer_id, name, visible=visible, active=is_active)
        self._update_layer_buttons_enabled()

    def set_layer_controls_visible(self, visible: bool) -> None:
        """Show or hide demo-only layer management buttons."""
        is_visible = bool(visible)
        self._layer_add_button.setVisible(is_visible)
        self._layer_duplicate_button.setVisible(is_visible)
        self._layer_delete_button.setVisible(is_visible)

    def ensure_layer(
        self,
        layer_id: str,
        name: str,
        *,
        visible: bool = True,
        active: bool = False,
    ) -> None:
        """Add one layer row if absent, or refresh its state."""
        normalized_id = str(layer_id or "").strip()
        if not normalized_id:
            return
        if normalized_id in self._layers:
            row = self._layers[normalized_id]
            row.set_name(name)
            row.set_visible_checked(visible)
            row.set_active_checked(active)
            if active:
                self._active_layer_id = normalized_id
            self._update_layer_buttons_enabled()
            return

        row = _LayerRow(
            layer_id=normalized_id,
            name=name,
            visible=visible,
            active=active,
            parent=self._container,
        )
        row.active_selected.connect(self._on_layer_selected)
        row.visibility_toggled.connect(self.layer_visibility_changed)
        self._layers[normalized_id] = row
        self._layer_group.addButton(row.active_button())
        self._layers_layout.addWidget(row)
        if active:
            self._active_layer_id = normalized_id
        self._update_layer_buttons_enabled()

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

    def _on_layer_selected(self, layer_id: str) -> None:
        normalized_id = str(layer_id or "").strip()
        if not normalized_id:
            return
        self._active_layer_id = normalized_id
        self._update_layer_buttons_enabled()
        self.layer_selected.emit(normalized_id)

    def _on_delete_active_layer(self) -> None:
        active_id = str(self._active_layer_id or "").strip()
        if not active_id:
            return
        self.layer_deleted.emit(active_id)

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

    def _update_layer_buttons_enabled(self) -> None:
        has_active = bool(self._active_layer_id)
        layer_count = len(self._layers)
        self._layer_duplicate_button.setEnabled(has_active)
        self._layer_delete_button.setEnabled(has_active and layer_count > 1)


class _LayerRow(QWidget):
    """Row widget holding active selection and visibility for one layer."""

    active_selected = pyqtSignal(str)
    visibility_toggled = pyqtSignal(str, bool)

    def __init__(
        self,
        *,
        layer_id: str,
        name: str,
        visible: bool,
        active: bool,
        parent: Optional[QWidget],
    ) -> None:
        super().__init__(parent)
        self.layer_id = str(layer_id)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._active = QRadioButton(self)
        self._active.setChecked(bool(active))
        self._active.toggled.connect(self._on_active_toggled)
        layout.addWidget(self._active, 0)

        self._name = QLabel(str(name or "Layer"), self)
        self._name.setMinimumWidth(120)
        layout.addWidget(self._name, 1)

        self._visible = QCheckBox("Visible", self)
        self._visible.setChecked(bool(visible))
        self._visible.toggled.connect(self._on_visible_toggled)
        layout.addWidget(self._visible, 0)

    def active_button(self) -> QRadioButton:
        return self._active

    def set_name(self, name: str) -> None:
        self._name.setText(str(name or "Layer"))

    def set_active_checked(self, checked: bool) -> None:
        self._active.blockSignals(True)
        self._active.setChecked(bool(checked))
        self._active.blockSignals(False)

    def set_visible_checked(self, checked: bool) -> None:
        self._visible.blockSignals(True)
        self._visible.setChecked(bool(checked))
        self._visible.blockSignals(False)

    def _on_active_toggled(self, checked: bool) -> None:
        if checked:
            self.active_selected.emit(self.layer_id)

    def _on_visible_toggled(self, checked: bool) -> None:
        self.visibility_toggled.emit(self.layer_id, checked)


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
            self._delete_button = QPushButton("Delete", self)
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
        picked = QColorDialog.getColor(self._color, self, f"Color for {format_label_text(self.label_id)}")
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
