from __future__ import annotations

from typing import Optional, Dict

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QButtonGroup,
    QRadioButton as QBtn,
)
from PyQt6.QtWidgets import QInputDialog, QButtonGroup


class ToolsPanel(QFrame):
    """Docked tools panel exposing user interactions as signals (view only)."""

    slice_changed = pyqtSignal(int)
    goto_requested = pyqtSignal(int)
    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()
    apply_roi_requested = pyqtSignal()
    tool_mode_changed = pyqtSignal(str)
    threshold_changed = pyqtSignal(int)
    threshold_auto_toggled = pyqtSignal(bool)
    apply_volume_toggled = pyqtSignal(bool)
    roi_persistence_toggled = pyqtSignal(bool)
    roi_recompute_requested = pyqtSignal()
    roi_delete_requested = pyqtSignal()
    selection_cancel_requested = pyqtSignal()
    overlay_toggled = pyqtSignal(bool)
    cross_toggled = pyqtSignal(bool)
    label_selected = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._slice_slider: Optional[QSlider] = None
        self._slice_label: Optional[QLabel] = None
        self._position_label: Optional[QLabel] = None
        self._goto_button: Optional[QPushButton] = None
        self._threshold_slider: Optional[QSlider] = None
        self._free_hand_radio: Optional[QRadioButton] = None
        self._box_radio: Optional[QRadioButton] = None
        self._grow_radio: Optional[QRadioButton] = None
        self._apply_volume_checkbox: Optional[QCheckBox] = None
        self._threshold_auto_checkbox: Optional[QCheckBox] = None
        self._overlay_checkbox: Optional[QCheckBox] = None
        self._cross_checkbox: Optional[QCheckBox] = None
        self._roi_persistence_checkbox: Optional[QCheckBox] = None
        self._roi_recompute_button: Optional[QPushButton] = None
        self._roi_delete_button: Optional[QPushButton] = None
        self._selection_cancel_button: Optional[QPushButton] = None
        self._previous_button: Optional[QPushButton] = None
        self._next_button: Optional[QPushButton] = None
        self._apply_roi_button: Optional[QPushButton] = None
        self._label_container: Optional[QWidget] = None
        self._label_layout: Optional[QVBoxLayout] = None
        self._label_group: Optional[QButtonGroup] = None
        self._label_buttons: Dict[int, QBtn] = {}

        self._slice_min: int = 0
        self._slice_max: int = 0
        self._wired = False

    def attach_designer_widgets(
        self,
        *,
        slice_slider: QSlider,
        slice_label: QLabel,
        goto_button: QPushButton,
        threshold_slider: QSlider,
        free_hand_radio: QRadioButton,
        box_radio: QRadioButton,
        grow_radio: QRadioButton,
        position_label: QLabel,
        overlay_checkbox: QCheckBox,
        cross_checkbox: QCheckBox,
        apply_volume_checkbox: QCheckBox,
        threshold_auto_checkbox: QCheckBox,
        roi_persistence_checkbox: QCheckBox,
        roi_recompute_button: QPushButton,
        roi_delete_button: QPushButton,
        selection_cancel_button: QPushButton,
        previous_button: QPushButton,
        next_button: QPushButton,
        apply_roi_button: QPushButton,
        label_container: QWidget,
    ) -> None:
        """Receive Designer-created widgets and wire them to the exposed signals."""
        if self._wired:
            return

        self._slice_slider = slice_slider
        self._slice_label = slice_label
        self._position_label = position_label
        self._goto_button = goto_button
        self._threshold_slider = threshold_slider
        self._free_hand_radio = free_hand_radio
        self._box_radio = box_radio
        self._grow_radio = grow_radio
        self._overlay_checkbox = overlay_checkbox
        self._cross_checkbox = cross_checkbox
        self._apply_volume_checkbox = apply_volume_checkbox
        self._threshold_auto_checkbox = threshold_auto_checkbox
        self._roi_persistence_checkbox = roi_persistence_checkbox
        self._roi_recompute_button = roi_recompute_button
        self._roi_delete_button = roi_delete_button
        self._selection_cancel_button = selection_cancel_button
        self._previous_button = previous_button
        self._next_button = next_button
        self._apply_roi_button = apply_roi_button
        self._label_container = label_container
        self._ensure_label_layout()

        self._slice_slider.valueChanged.connect(self._on_slider_changed)
        self._goto_button.clicked.connect(self._emit_goto_requested)
        if self._previous_button is not None:
            self._previous_button.clicked.connect(self.previous_requested)
        if self._next_button is not None:
            self._next_button.clicked.connect(self.next_requested)
        if self._apply_roi_button is not None:
            self._apply_roi_button.clicked.connect(self.apply_roi_requested)
        self._threshold_slider.valueChanged.connect(self.threshold_changed.emit)
        self._threshold_auto_checkbox.toggled.connect(self.threshold_auto_toggled.emit)
        self._apply_volume_checkbox.toggled.connect(self.apply_volume_toggled.emit)
        self._overlay_checkbox.toggled.connect(self.overlay_toggled.emit)
        self._cross_checkbox.toggled.connect(self.cross_toggled.emit)
        self._roi_persistence_checkbox.toggled.connect(self.roi_persistence_toggled.emit)
        self._roi_recompute_button.clicked.connect(self.roi_recompute_requested)
        self._roi_delete_button.clicked.connect(self.roi_delete_requested)
        self._selection_cancel_button.clicked.connect(self.selection_cancel_requested)

        self._free_hand_radio.toggled.connect(
            lambda checked: checked and self.tool_mode_changed.emit("free_hand")
        )
        self._box_radio.toggled.connect(
            lambda checked: checked and self.tool_mode_changed.emit("box")
        )
        self._grow_radio.toggled.connect(
            lambda checked: checked and self.tool_mode_changed.emit("grow")
        )

        self._wired = True

    def _ensure_label_layout(self) -> None:
        if self._label_container is None:
            return
        if self._label_layout is None:
            self._label_layout = QVBoxLayout(self._label_container)
            self._label_layout.setContentsMargins(0, 0, 0, 0)
            self._label_layout.setSpacing(4)
        if self._label_group is None:
            self._label_group = QButtonGroup(self)
            self._label_group.idClicked.connect(self.label_selected.emit)

    def set_labels(self, labels: list[int], *, current: Optional[int] = None) -> None:
        """Populate the label list and select the requested/current label if possible."""
        self._ensure_label_layout()
        if self._label_layout is None or self._label_group is None:
            return
        # Clear old buttons
        for btn in self._label_buttons.values():
            self._label_group.removeButton(btn)
            btn.setParent(None)
        self._label_buttons.clear()

        for lbl in labels:
            btn = QBtn(f"Label {lbl}", self._label_container)
            btn.setCheckable(True)
            self._label_group.addButton(btn, lbl)
            self._label_layout.addWidget(btn)
            self._label_buttons[lbl] = btn

        target = current if current in labels else (labels[0] if labels else None)
        if target is not None:
            self.select_label(target)
        else:
            # No selection if no labels
            pass

    def select_label(self, label_id: int) -> None:
        """Programmatically select a label button."""
        btn = self._label_buttons.get(int(label_id))
        if btn:
            btn.setChecked(True)

    def set_slice_bounds(self, minimum: int, maximum: int) -> None:
        """Configure slider bounds without emitting change signals."""
        if not self._slice_slider:
            return
        self._slice_slider.blockSignals(True)
        self._slice_slider.setMinimum(minimum)
        self._slice_slider.setMaximum(maximum)
        self._slice_slider.blockSignals(False)
        self._slice_min = minimum
        self._slice_max = maximum

    def set_slice_value(self, slice_idx: int) -> None:
        """Update the slice slider/label without re-emitting signals."""
        if not self._slice_slider:
            return
        self._slice_slider.blockSignals(True)
        self._slice_slider.setValue(slice_idx)
        self._update_slice_label(slice_idx)
        self._slice_slider.blockSignals(False)

    def set_threshold_value(self, threshold: int) -> None:
        """Update the threshold slider without re-emitting signals."""
        if not self._threshold_slider:
            return
        self._threshold_slider.blockSignals(True)
        self._threshold_slider.setValue(threshold)
        self._threshold_slider.blockSignals(False)

    def set_overlay_checked(self, enabled: bool) -> None:
        """Set overlay checkbox state without emitting signals."""
        if not self._overlay_checkbox:
            return
        self._overlay_checkbox.blockSignals(True)
        self._overlay_checkbox.setChecked(enabled)
        self._overlay_checkbox.blockSignals(False)

    def set_cross_checked(self, enabled: bool) -> None:
        """Set cross checkbox state without emitting signals."""
        if not self._cross_checkbox:
            return
        self._cross_checkbox.blockSignals(True)
        self._cross_checkbox.setChecked(enabled)
        self._cross_checkbox.blockSignals(False)

    def select_tool_mode(self, mode: str) -> None:
        """Select a tool radio button without emitting tool_mode_changed."""
        mapping = {
            "free_hand": self._free_hand_radio,
            "box": self._box_radio,
            "grow": self._grow_radio,
        }
        target = mapping.get(mode)
        if not target:
            return
        target.blockSignals(True)
        target.setChecked(True)
        target.blockSignals(False)

    def _emit_goto_requested(self) -> None:
        """Emit goto with the current spinbox value."""
        if not self._slice_slider:
            return
        current = self._slice_slider.value()
        slice_idx, ok = QInputDialog.getInt(
            self,
            "Se rendre à la tranche",
            f"Index de tranche (0 - {self._slice_max})",
            current,
            self._slice_min,
            self._slice_max,
            1,
        )
        if ok:
            self.goto_requested.emit(slice_idx)

    def _on_slider_changed(self, value: int) -> None:
        """Handle slider movements and emit slice_changed."""
        self._update_slice_label(value)
        self.slice_changed.emit(value)

    def _update_slice_label(self, value: int) -> None:
        """Reflect the current slider value into the label."""
        if not self._slice_label:
            return
        if self._slice_max:
            self._slice_label.setText(f"{value} / {self._slice_max}")
        else:
            self._slice_label.setText(str(value))

    def set_position_label(self, x: int, y: int) -> None:
        """Update the position label with the latest cursor coordinates."""
        if not self._position_label:
            return
        self._position_label.setText(f"position x = {x} ; y = {y}")
