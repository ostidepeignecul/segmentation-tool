from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
)


class ToolsPanel(QFrame):
    """Docked tools panel exposing user interactions as signals (view only)."""

    slice_changed = pyqtSignal(int)
    goto_requested = pyqtSignal(int)
    tool_mode_changed = pyqtSignal(str)
    threshold_changed = pyqtSignal(int)
    threshold_auto_toggled = pyqtSignal(bool)
    apply_volume_toggled = pyqtSignal(bool)
    roi_persistence_toggled = pyqtSignal(bool)
    roi_recompute_requested = pyqtSignal()
    roi_delete_requested = pyqtSignal()
    selection_cancel_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._slice_spinbox: Optional[QSpinBox] = None
        self._goto_button: Optional[QPushButton] = None
        self._threshold_slider: Optional[QSlider] = None
        self._polygon_radio: Optional[QRadioButton] = None
        self._rectangle_radio: Optional[QRadioButton] = None
        self._point_radio: Optional[QRadioButton] = None
        self._apply_volume_checkbox: Optional[QCheckBox] = None
        self._threshold_auto_checkbox: Optional[QCheckBox] = None
        self._roi_persistence_checkbox: Optional[QCheckBox] = None
        self._roi_recompute_button: Optional[QPushButton] = None
        self._roi_delete_button: Optional[QPushButton] = None
        self._selection_cancel_button: Optional[QPushButton] = None

        self._wired = False

    def attach_designer_widgets(
        self,
        *,
        slice_spinbox: QSpinBox,
        goto_button: QPushButton,
        threshold_slider: QSlider,
        polygon_radio: QRadioButton,
        rectangle_radio: QRadioButton,
        point_radio: QRadioButton,
        apply_volume_checkbox: QCheckBox,
        threshold_auto_checkbox: QCheckBox,
        roi_persistence_checkbox: QCheckBox,
        roi_recompute_button: QPushButton,
        roi_delete_button: QPushButton,
        selection_cancel_button: QPushButton,
    ) -> None:
        """Receive Designer-created widgets and wire them to the exposed signals."""
        if self._wired:
            return

        self._slice_spinbox = slice_spinbox
        self._goto_button = goto_button
        self._threshold_slider = threshold_slider
        self._polygon_radio = polygon_radio
        self._rectangle_radio = rectangle_radio
        self._point_radio = point_radio
        self._apply_volume_checkbox = apply_volume_checkbox
        self._threshold_auto_checkbox = threshold_auto_checkbox
        self._roi_persistence_checkbox = roi_persistence_checkbox
        self._roi_recompute_button = roi_recompute_button
        self._roi_delete_button = roi_delete_button
        self._selection_cancel_button = selection_cancel_button

        self._slice_spinbox.valueChanged.connect(self.slice_changed.emit)
        self._goto_button.clicked.connect(self._emit_goto_requested)
        self._threshold_slider.valueChanged.connect(self.threshold_changed.emit)
        self._threshold_auto_checkbox.toggled.connect(self.threshold_auto_toggled.emit)
        self._apply_volume_checkbox.toggled.connect(self.apply_volume_toggled.emit)
        self._roi_persistence_checkbox.toggled.connect(self.roi_persistence_toggled.emit)
        self._roi_recompute_button.clicked.connect(self.roi_recompute_requested)
        self._roi_delete_button.clicked.connect(self.roi_delete_requested)
        self._selection_cancel_button.clicked.connect(self.selection_cancel_requested)

        self._polygon_radio.toggled.connect(
            lambda checked: checked and self.tool_mode_changed.emit("polygon")
        )
        self._rectangle_radio.toggled.connect(
            lambda checked: checked and self.tool_mode_changed.emit("rectangle")
        )
        self._point_radio.toggled.connect(
            lambda checked: checked and self.tool_mode_changed.emit("point")
        )

        self._slice_spinbox.editingFinished.connect(self._emit_goto_requested)
        self._wired = True

    def set_slice_value(self, slice_idx: int) -> None:
        """Update the slice spinbox without re-emitting signals."""
        if not self._slice_spinbox:
            return
        self._slice_spinbox.blockSignals(True)
        self._slice_spinbox.setValue(slice_idx)
        self._slice_spinbox.blockSignals(False)

    def set_threshold_value(self, threshold: int) -> None:
        """Update the threshold slider without re-emitting signals."""
        if not self._threshold_slider:
            return
        self._threshold_slider.blockSignals(True)
        self._threshold_slider.setValue(threshold)
        self._threshold_slider.blockSignals(False)

    def select_tool_mode(self, mode: str) -> None:
        """Select a tool radio button without emitting tool_mode_changed."""
        mapping = {
            "polygon": self._polygon_radio,
            "rectangle": self._rectangle_radio,
            "point": self._point_radio,
        }
        target = mapping.get(mode)
        if not target:
            return
        target.blockSignals(True)
        target.setChecked(True)
        target.blockSignals(False)

    def _emit_goto_requested(self) -> None:
        """Emit goto with the current spinbox value."""
        if not self._slice_spinbox:
            return
        self.goto_requested.emit(self._slice_spinbox.value())
