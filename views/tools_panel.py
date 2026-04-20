from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QFrame,
    QLabel,
    QPushButton,
    QRadioButton as QBtn,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config.constants import format_label_text


class ToolsPanel(QFrame):
    """Docked tools panel exposing user interactions as signals (view only)."""

    slice_changed = pyqtSignal(int)
    secondary_slice_changed = pyqtSignal(int)
    apply_roi_requested = pyqtSignal()
    tool_mode_changed = pyqtSignal(str)
    annotation_action_changed = pyqtSignal(str)
    threshold_changed = pyqtSignal(int)
    force_threshold_erase_toggled = pyqtSignal(bool)
    apply_auto_toggled = pyqtSignal(bool)
    threshold_auto_toggled = pyqtSignal(bool)
    apply_volume_toggled = pyqtSignal(bool)
    roi_persistence_toggled = pyqtSignal(bool)
    closing_mask_toggled = pyqtSignal(bool)
    clean_outliers_toggled = pyqtSignal(bool)
    volume_view_overlay_toggled = pyqtSignal(bool)
    roi_recompute_requested = pyqtSignal()
    roi_delete_requested = pyqtSignal()
    selection_cancel_requested = pyqtSignal()
    label_selected = pyqtSignal(int)
    label_color_changed = pyqtSignal(int, QColor)
    paint_size_changed = pyqtSignal(int)
    overlay_opacity_changed = pyqtSignal(float)
    nde_opacity_changed = pyqtSignal(float)
    nde_contrast_changed = pyqtSignal(float)
    endview_colormap_changed = pyqtSignal(str)
    _TOOL_MODE_BY_TEXT = {
        "free hand": "free_hand",
        "box": "box",
        "grow": "grow",
        "line": "line",
        "paint": "paint",
        "mod": "mod",
        "peak": "peak",
    }
    _ANNOTATION_ACTION_BY_TEXT = {
        "draw": "draw",
        "erase": "erase",
    }
    _COLORMAP_BY_TEXT = {
        "omniscan": "OmniScan",
        "gray": "Gris",
        "gris": "Gris",
    }
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._primary_axis_label: Optional[QLabel] = None
        self._secondary_axis_label: Optional[QLabel] = None
        self._primary_slider: Optional[QSlider] = None
        self._secondary_slider: Optional[QSlider] = None
        self._primary_spinbox: Optional[QSpinBox] = None
        self._secondary_spinbox: Optional[QSpinBox] = None
        self._nde_label: Optional[QLabel] = None
        self._endview_label: Optional[QLabel] = None
        self._position_label: Optional[QLabel] = None
        self._tool_combo: Optional[QComboBox] = None
        self._action_combo: Optional[QComboBox] = None
        self._colormap_combo: Optional[QComboBox] = None
        self._threshold_slider: Optional[QSlider] = None
        self._threshold_label: Optional[QLabel] = None
        self._paint_size_slider: Optional[QSlider] = None
        self._overlay_opacity_slider: Optional[QSlider] = None
        self._overlay_opacity_spinbox: Optional[QSpinBox] = None
        self._nde_opacity_slider: Optional[QSlider] = None
        self._nde_opacity_spinbox: Optional[QSpinBox] = None
        self._nde_opacity_label: Optional[QLabel] = None
        self._nde_contrast_slider: Optional[QSlider] = None
        self._nde_contrast_spinbox: Optional[QSpinBox] = None
        self._nde_contrast_label: Optional[QLabel] = None
        self._apply_volume_checkbox: Optional[QCheckBox] = None
        self._apply_auto_checkbox: Optional[QCheckBox] = None
        self._force_threshold_erase_checkbox: Optional[QCheckBox] = None
        self._threshold_auto_checkbox: Optional[QCheckBox] = None
        self._roi_persistence_checkbox: Optional[QCheckBox] = None
        self._closing_mask_checkbox: Optional[QCheckBox] = None
        self._clean_outliers_checkbox: Optional[QCheckBox] = None
        self._volume_view_checkbox: Optional[QCheckBox] = None
        self._roi_recompute_button: Optional[QPushButton] = None
        self._roi_delete_button: Optional[QPushButton] = None
        self._selection_cancel_button: Optional[QPushButton] = None
        self._apply_roi_button: Optional[QPushButton] = None
        self._label_text_container: Optional[QWidget] = None
        self._label_color_container: Optional[QWidget] = None
        self._label_text_layout: Optional[QVBoxLayout] = None
        self._label_color_layout: Optional[QVBoxLayout] = None
        self._label_group: Optional[QButtonGroup] = None
        self._label_buttons: Dict[int, QBtn] = {}
        self._label_color_buttons: Dict[int, QPushButton] = {}
        self._label_colors: Dict[int, QColor] = {}

        self._primary_min: int = 0
        self._primary_max: int = 0
        self._secondary_min: int = 0
        self._secondary_max: int = 0
        self._wired = False

    def attach_designer_widgets(
        self,
        *,
        tool_combo: QComboBox,
        action_combo: Optional[QComboBox],
        colormap_combo: QComboBox,
        threshold_slider: QSlider,
        threshold_label: QLabel,
        paint_slider: QSlider,
        overlay_opacity_slider: QSlider,
        overlay_opacity_spinbox: QSpinBox,
        nde_opacity_slider: QSlider,
        nde_opacity_spinbox: QSpinBox,
        nde_contrast_slider: QSlider,
        nde_contrast_spinbox: QSpinBox,
        apply_auto_checkbox: Optional[QCheckBox],
        force_threshold_erase_checkbox: Optional[QCheckBox],
        apply_volume_checkbox: QCheckBox,
        threshold_auto_checkbox: QCheckBox,
        roi_persistence_checkbox: QCheckBox,
        closing_mask_checkbox: Optional[QCheckBox],
        clean_outliers_checkbox: Optional[QCheckBox],
        volume_view_checkbox: Optional[QCheckBox],
        roi_recompute_button: QPushButton,
        roi_delete_button: QPushButton,
        selection_cancel_button: QPushButton,
        apply_roi_button: QPushButton,
        label_text_container: QWidget,
        label_color_container: QWidget,
        nde_opacity_label: Optional[QLabel] = None,
        nde_contrast_label: Optional[QLabel] = None,
    ) -> None:
        """Receive Designer-created widgets and wire them to the exposed signals."""
        if self._wired:
            return

        self._tool_combo = tool_combo
        self._action_combo = action_combo
        self._colormap_combo = colormap_combo
        self._threshold_slider = threshold_slider
        self._threshold_label = threshold_label
        self._paint_size_slider = paint_slider
        self._overlay_opacity_slider = overlay_opacity_slider
        self._overlay_opacity_spinbox = overlay_opacity_spinbox
        self._nde_opacity_slider = nde_opacity_slider
        self._nde_opacity_spinbox = nde_opacity_spinbox
        self._nde_opacity_label = nde_opacity_label
        self._nde_contrast_slider = nde_contrast_slider
        self._nde_contrast_spinbox = nde_contrast_spinbox
        self._nde_contrast_label = nde_contrast_label
        self._apply_auto_checkbox = apply_auto_checkbox
        self._force_threshold_erase_checkbox = force_threshold_erase_checkbox
        self._apply_volume_checkbox = apply_volume_checkbox
        self._threshold_auto_checkbox = threshold_auto_checkbox
        self._roi_persistence_checkbox = roi_persistence_checkbox
        self._closing_mask_checkbox = closing_mask_checkbox
        self._clean_outliers_checkbox = clean_outliers_checkbox
        self._volume_view_checkbox = volume_view_checkbox
        self._roi_recompute_button = roi_recompute_button
        self._roi_delete_button = roi_delete_button
        self._selection_cancel_button = selection_cancel_button
        self._apply_roi_button = apply_roi_button
        self._label_text_container = label_text_container
        self._label_color_container = label_color_container
        self._ensure_label_layouts()

        self._prepare_tool_combo()
        self._tool_combo.currentIndexChanged.connect(self._on_tool_combo_changed)
        self._prepare_action_combo()
        if self._action_combo is not None:
            self._action_combo.currentIndexChanged.connect(self._on_action_combo_changed)
        self._prepare_colormap_combo()
        self._colormap_combo.currentIndexChanged.connect(self._on_colormap_combo_changed)

        self._threshold_slider.setMinimum(0)
        self._threshold_slider.setMaximum(255)
        self._threshold_slider.setValue(50)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self._configure_percentage_controls(
            slider=self._overlay_opacity_slider,
            spinbox=self._overlay_opacity_spinbox,
            handler=self._on_overlay_opacity_value_changed,
        )
        self._configure_percentage_controls(
            slider=self._nde_opacity_slider,
            spinbox=self._nde_opacity_spinbox,
            handler=self._on_nde_opacity_value_changed,
        )
        self._configure_percentage_controls(
            slider=self._nde_contrast_slider,
            spinbox=self._nde_contrast_spinbox,
            handler=self._on_nde_contrast_value_changed,
            maximum=200,
        )

        if self._force_threshold_erase_checkbox is not None:
            self._force_threshold_erase_checkbox.toggled.connect(
                self.force_threshold_erase_toggled.emit
            )
        if self._apply_auto_checkbox is not None:
            self._apply_auto_checkbox.toggled.connect(self.apply_auto_toggled.emit)
        self._threshold_auto_checkbox.toggled.connect(self.threshold_auto_toggled.emit)
        self._apply_volume_checkbox.toggled.connect(self.apply_volume_toggled.emit)
        self._roi_persistence_checkbox.toggled.connect(self.roi_persistence_toggled.emit)
        if self._closing_mask_checkbox is not None:
            self._closing_mask_checkbox.toggled.connect(self.closing_mask_toggled.emit)
        if self._clean_outliers_checkbox is not None:
            self._clean_outliers_checkbox.toggled.connect(self.clean_outliers_toggled.emit)
        if self._volume_view_checkbox is not None:
            self._volume_view_checkbox.toggled.connect(self.volume_view_overlay_toggled.emit)
        self._roi_recompute_button.clicked.connect(self.roi_recompute_requested)
        self._roi_delete_button.clicked.connect(self.roi_delete_requested)
        self._selection_cancel_button.clicked.connect(self.selection_cancel_requested)
        self._apply_roi_button.clicked.connect(self.apply_roi_requested)

        if self._paint_size_slider is not None:
            self._paint_size_slider.setMinimum(1)
            self._paint_size_slider.setMaximum(50)
            self._paint_size_slider.setValue(8)
            self._paint_size_slider.valueChanged.connect(self.paint_size_changed.emit)

        self._wired = True
        self.set_nde_opacity_available(False)

    def _configure_slice_controls(
        self,
        *,
        slider: QSlider,
        spinbox: QSpinBox,
        handler,
    ) -> None:
        slider.setMinimum(0)
        slider.setMaximum(0)
        spinbox.setMinimum(0)
        spinbox.setMaximum(0)
        slider.valueChanged.connect(handler)
        spinbox.valueChanged.connect(handler)

    def _prepare_tool_combo(self) -> None:
        if self._tool_combo is None:
            return
        for idx in range(self._tool_combo.count()):
            text = self._tool_combo.itemText(idx).strip().lower()
            mode = self._TOOL_MODE_BY_TEXT.get(text)
            if mode is not None:
                self._tool_combo.setItemData(idx, mode)

    def _prepare_action_combo(self) -> None:
        if self._action_combo is None:
            return
        for idx in range(self._action_combo.count()):
            text = self._action_combo.itemText(idx).strip().lower()
            action = self._ANNOTATION_ACTION_BY_TEXT.get(text)
            if action is not None:
                self._action_combo.setItemData(idx, action)

    def _prepare_colormap_combo(self) -> None:
        if self._colormap_combo is None:
            return
        for idx in range(self._colormap_combo.count()):
            text = self._colormap_combo.itemText(idx).strip()
            normalized = self._normalize_colormap_name(text)
            self._colormap_combo.setItemData(idx, normalized)
            self._colormap_combo.setItemText(idx, normalized)

    def _configure_percentage_controls(
        self,
        *,
        slider: Optional[QSlider],
        spinbox: Optional[QSpinBox],
        handler,
        maximum: int = 100,
    ) -> None:
        if slider is None or spinbox is None:
            return
        slider.setMinimum(0)
        slider.setMaximum(int(maximum))
        spinbox.setMinimum(0)
        spinbox.setMaximum(int(maximum))
        spinbox.setKeyboardTracking(False)
        slider.valueChanged.connect(handler)
        spinbox.valueChanged.connect(handler)

    def _ensure_label_layouts(self) -> None:
        if self._label_text_container is None or self._label_color_container is None:
            return

        if self._label_text_layout is None:
            existing = self._label_text_container.layout()
            if isinstance(existing, QVBoxLayout):
                self._label_text_layout = existing
            else:
                self._label_text_layout = QVBoxLayout(self._label_text_container)
            self._label_text_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        if self._label_color_layout is None:
            existing = self._label_color_container.layout()
            if isinstance(existing, QVBoxLayout):
                self._label_color_layout = existing
            else:
                self._label_color_layout = QVBoxLayout(self._label_color_container)
            self._label_color_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        if self._label_group is None:
            self._label_group = QButtonGroup(self)
            self._label_group.idClicked.connect(self.label_selected.emit)

    def _clear_label_widgets(self) -> None:
        if self._label_group is not None:
            for btn in self._label_buttons.values():
                self._label_group.removeButton(btn)
                btn.setParent(None)
                btn.deleteLater()

        for btn in self._label_color_buttons.values():
            btn.setParent(None)
            btn.deleteLater()

        self._label_buttons.clear()
        self._label_color_buttons.clear()
        self._label_colors.clear()

    def set_labels(
        self,
        entries: Iterable[Tuple[int, QColor]],
        *,
        current: Optional[int] = None,
    ) -> None:
        """Populate the label list with editable colors and restore selection."""
        self._ensure_label_layouts()
        if (
            self._label_text_layout is None
            or self._label_color_layout is None
            or self._label_group is None
        ):
            return

        self._clear_label_widgets()

        label_ids: list[int] = []
        max_label_width = 0
        max_row_height = 0

        for label_id, color in entries:
            lbl = int(label_id)
            qcolor = QColor(color)

            label_button = QBtn(format_label_text(lbl), self._label_text_container)
            label_button.setCheckable(True)
            self._label_group.addButton(label_button, lbl)
            self._label_text_layout.addWidget(
                label_button,
                0,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )

            color_button = QPushButton(qcolor.name(), self._label_color_container)
            color_button.setFixedWidth(90)
            color_button.clicked.connect(
                lambda _checked=False, label_id=lbl: self._on_pick_label_color(label_id)
            )
            self._label_color_layout.addWidget(
                color_button,
                0,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )

            self._label_buttons[lbl] = label_button
            self._label_color_buttons[lbl] = color_button
            self._label_colors[lbl] = qcolor
            label_ids.append(lbl)
            max_label_width = max(max_label_width, label_button.sizeHint().width())
            max_row_height = max(
                max_row_height,
                label_button.sizeHint().height(),
                color_button.sizeHint().height(),
            )

        for lbl in label_ids:
            label_button = self._label_buttons[lbl]
            color_button = self._label_color_buttons[lbl]
            label_button.setFixedWidth(max_label_width)
            label_button.setFixedHeight(max_row_height)
            color_button.setFixedHeight(max_row_height)
            self._apply_color_button_style(lbl)

        target = current if current in label_ids else (label_ids[0] if label_ids else None)
        if target is not None:
            self.select_label(target)

    def select_label(self, label_id: int) -> None:
        """Programmatically select a label button."""
        btn = self._label_buttons.get(int(label_id))
        if btn is not None:
            btn.setChecked(True)

    def set_label_color(self, label_id: int, color: QColor) -> None:
        """Update a label color button without rebuilding the full list."""
        label = int(label_id)
        if label not in self._label_color_buttons:
            return
        self._label_colors[label] = QColor(color)
        self._apply_color_button_style(label)

    def _on_pick_label_color(self, label_id: int) -> None:
        current = QColor(self._label_colors.get(int(label_id), QColor("#ff00ff")))
        picked = QColorDialog.getColor(
            current,
            self,
            f"Couleur pour {format_label_text(int(label_id))}",
        )
        if picked.isValid():
            self.set_label_color(int(label_id), picked)
            self.label_color_changed.emit(int(label_id), picked)

    def _apply_color_button_style(self, label_id: int) -> None:
        label = int(label_id)
        button = self._label_color_buttons.get(label)
        color = self._label_colors.get(label)
        if button is None or color is None:
            return
        text_color = "#000000" if color.lightness() >= 140 else "#ffffff"
        button.setText(color.name())
        button.setStyleSheet(
            f"background-color: {color.name()}; color: {text_color}; font-weight: bold;"
        )
        button.setToolTip(
            f"Changer la couleur de {format_label_text(label)} ({color.name()})"
        )

    def set_slice_bounds(self, minimum: int, maximum: int) -> None:
        """Backward-compatible wrapper for the primary coordinate bounds."""
        self.set_primary_slice_bounds(minimum, maximum)

    def set_slice_value(self, slice_idx: int) -> None:
        """Backward-compatible wrapper for the primary coordinate value."""
        self.set_primary_slice_value(slice_idx)

    def set_primary_slice_bounds(self, minimum: int, maximum: int) -> None:
        """Configure primary coordinate bounds without emitting change signals."""
        self._set_pair_bounds(self._primary_slider, self._primary_spinbox, minimum, maximum)
        self._primary_min = int(minimum)
        self._primary_max = int(maximum)

    def set_secondary_slice_bounds(self, minimum: int, maximum: int) -> None:
        """Configure secondary coordinate bounds without emitting change signals."""
        self._set_pair_bounds(self._secondary_slider, self._secondary_spinbox, minimum, maximum)
        self._secondary_min = int(minimum)
        self._secondary_max = int(maximum)

    def set_primary_slice_value(self, slice_idx: int) -> None:
        """Update the primary coordinate widgets without re-emitting signals."""
        self._set_pair_value(self._primary_slider, self._primary_spinbox, slice_idx)

    def set_secondary_slice_value(self, slice_idx: int) -> None:
        """Update the secondary coordinate widgets without re-emitting signals."""
        self._set_pair_value(self._secondary_slider, self._secondary_spinbox, slice_idx)

    def _set_pair_bounds(
        self,
        slider: Optional[QSlider],
        spinbox: Optional[QSpinBox],
        minimum: int,
        maximum: int,
    ) -> None:
        if slider is None or spinbox is None:
            return
        for widget in (slider, spinbox):
            widget.blockSignals(True)
            widget.setMinimum(int(minimum))
            widget.setMaximum(int(maximum))
            widget.blockSignals(False)

    def _set_pair_value(
        self,
        slider: Optional[QSlider],
        spinbox: Optional[QSpinBox],
        value: int,
    ) -> None:
        if slider is None or spinbox is None:
            return
        clamped = max(int(slider.minimum()), min(int(slider.maximum()), int(value)))
        slider.blockSignals(True)
        spinbox.blockSignals(True)
        slider.setValue(clamped)
        spinbox.setValue(clamped)
        slider.blockSignals(False)
        spinbox.blockSignals(False)

    def set_threshold_value(self, threshold: int) -> None:
        """Update the threshold slider without re-emitting signals."""
        if self._threshold_slider is None:
            return
        self._threshold_slider.blockSignals(True)
        self._threshold_slider.setValue(int(threshold))
        self._threshold_slider.blockSignals(False)
        self._update_threshold_label(threshold)

    def set_threshold_auto_checked(self, enabled: bool) -> None:
        """Set threshold-auto checkbox state without emitting signals."""
        if self._threshold_auto_checkbox is None:
            return
        self._threshold_auto_checkbox.blockSignals(True)
        self._threshold_auto_checkbox.setChecked(bool(enabled))
        self._threshold_auto_checkbox.blockSignals(False)

    def set_force_threshold_erase_checked(self, enabled: bool) -> None:
        """Set erase-threshold checkbox state without emitting signals."""
        if self._force_threshold_erase_checkbox is None:
            return
        self._force_threshold_erase_checkbox.blockSignals(True)
        self._force_threshold_erase_checkbox.setChecked(bool(enabled))
        self._force_threshold_erase_checkbox.blockSignals(False)

    def set_apply_auto_checked(self, enabled: bool) -> None:
        """Set apply-auto checkbox state without emitting signals."""
        if self._apply_auto_checkbox is None:
            return
        self._apply_auto_checkbox.blockSignals(True)
        self._apply_auto_checkbox.setChecked(bool(enabled))
        self._apply_auto_checkbox.blockSignals(False)

    def set_apply_volume_checked(self, enabled: bool) -> None:
        """Set apply-volume checkbox state without emitting signals."""
        if self._apply_volume_checkbox is None:
            return
        self._apply_volume_checkbox.blockSignals(True)
        self._apply_volume_checkbox.setChecked(bool(enabled))
        self._apply_volume_checkbox.blockSignals(False)

    def set_roi_persistence_checked(self, enabled: bool) -> None:
        """Set ROI persistence checkbox state without emitting signals."""
        if self._roi_persistence_checkbox is None:
            return
        self._roi_persistence_checkbox.blockSignals(True)
        self._roi_persistence_checkbox.setChecked(bool(enabled))
        self._roi_persistence_checkbox.blockSignals(False)

    def set_closing_mask_checked(self, enabled: bool) -> None:
        """Set closing-mask checkbox state without emitting signals."""
        if self._closing_mask_checkbox is None:
            return
        self._closing_mask_checkbox.blockSignals(True)
        self._closing_mask_checkbox.setChecked(bool(enabled))
        self._closing_mask_checkbox.blockSignals(False)

    def set_clean_outliers_checked(self, enabled: bool) -> None:
        """Set clean-outliers checkbox state without emitting signals."""
        if self._clean_outliers_checkbox is None:
            return
        self._clean_outliers_checkbox.blockSignals(True)
        self._clean_outliers_checkbox.setChecked(bool(enabled))
        self._clean_outliers_checkbox.blockSignals(False)

    def set_volume_view_overlay_checked(self, enabled: bool) -> None:
        """Set volume-view overlay checkbox state without emitting signals."""
        if self._volume_view_checkbox is None:
            return
        self._volume_view_checkbox.blockSignals(True)
        self._volume_view_checkbox.setChecked(bool(enabled))
        self._volume_view_checkbox.blockSignals(False)

    def current_tool_mode(self) -> Optional[str]:
        """Return the currently selected drawing tool mode."""
        if self._tool_combo is None:
            return None
        data = self._tool_combo.currentData()
        if data is not None:
            return str(data)
        text = self._tool_combo.currentText().strip().lower()
        return self._TOOL_MODE_BY_TEXT.get(text)

    def select_tool_mode(self, mode: str) -> None:
        """Select the active tool in the combo box without emitting signals."""
        if self._tool_combo is None:
            return
        target_mode = str(mode)
        target_index = self._tool_combo.findData(target_mode)
        if target_index < 0:
            for idx in range(self._tool_combo.count()):
                text = self._tool_combo.itemText(idx).strip().lower()
                if self._TOOL_MODE_BY_TEXT.get(text) == target_mode:
                    target_index = idx
                    break
        if target_index < 0:
            return
        self._tool_combo.blockSignals(True)
        self._tool_combo.setCurrentIndex(target_index)
        self._tool_combo.blockSignals(False)

    def current_annotation_action(self) -> str:
        """Return the current annotation action (`draw` or `erase`)."""
        if self._action_combo is None:
            return "draw"
        data = self._action_combo.currentData()
        if data is not None:
            return str(data)
        text = self._action_combo.currentText().strip().lower()
        return self._ANNOTATION_ACTION_BY_TEXT.get(text, "draw")

    def set_annotation_action(self, action: str) -> None:
        """Select the current annotation action without re-emitting signals."""
        if self._action_combo is None:
            return
        target_action = str(action).strip().lower() or "draw"
        target_index = self._action_combo.findData(target_action)
        if target_index < 0:
            for idx in range(self._action_combo.count()):
                text = self._action_combo.itemText(idx).strip().lower()
                if self._ANNOTATION_ACTION_BY_TEXT.get(text) == target_action:
                    target_index = idx
                    break
        if target_index < 0:
            return
        self._action_combo.blockSignals(True)
        self._action_combo.setCurrentIndex(target_index)
        self._action_combo.blockSignals(False)

    def _on_tool_combo_changed(self, _index: int) -> None:
        mode = self.current_tool_mode()
        if mode:
            self.tool_mode_changed.emit(mode)

    def _on_action_combo_changed(self, _index: int) -> None:
        self.annotation_action_changed.emit(self.current_annotation_action())

    def current_endview_colormap(self) -> Optional[str]:
        """Return the selected endview/3D colormap."""
        if self._colormap_combo is None:
            return None
        data = self._colormap_combo.currentData()
        if data is not None:
            return str(data)
        return self._normalize_colormap_name(self._colormap_combo.currentText())

    def set_endview_colormap(self, name: str) -> None:
        """Select the current endview/3D colormap without re-emitting signals."""
        if self._colormap_combo is None:
            return
        normalized = self._normalize_colormap_name(name)
        target_index = self._colormap_combo.findData(normalized)
        if target_index < 0:
            for idx in range(self._colormap_combo.count()):
                text = self._normalize_colormap_name(self._colormap_combo.itemText(idx))
                if text == normalized:
                    target_index = idx
                    break
        if target_index < 0:
            self._colormap_combo.addItem(normalized, normalized)
            target_index = self._colormap_combo.findData(normalized)
        self._colormap_combo.blockSignals(True)
        self._colormap_combo.setCurrentIndex(target_index)
        self._colormap_combo.blockSignals(False)

    def set_overlay_opacity(self, opacity: float) -> None:
        """Update the overlay opacity widgets without re-emitting signals."""
        percent = int(round(max(0.0, min(1.0, float(opacity))) * 100.0))
        self._set_pair_value(
            self._overlay_opacity_slider,
            self._overlay_opacity_spinbox,
            percent,
        )

    def set_nde_opacity(self, opacity: float) -> None:
        """Update the NDE opacity widgets without re-emitting signals."""
        percent = int(round(max(0.0, min(1.0, float(opacity))) * 100.0))
        self._set_pair_value(
            self._nde_opacity_slider,
            self._nde_opacity_spinbox,
            percent,
        )

    def set_nde_contrast(self, contrast: float) -> None:
        """Update the NDE contrast widgets without re-emitting signals."""
        percent = int(round(max(0.0, min(2.0, float(contrast))) * 100.0))
        self._set_pair_value(
            self._nde_contrast_slider,
            self._nde_contrast_spinbox,
            percent,
        )

    def _on_threshold_changed(self, value: int) -> None:
        """Update threshold label and emit value."""
        self._update_threshold_label(value)
        self.threshold_changed.emit(int(value))

    def _update_threshold_label(self, value: int) -> None:
        if self._threshold_label is None:
            return
        self._threshold_label.setText(f"Threshold : {int(value)}")

    def _on_colormap_combo_changed(self, _index: int) -> None:
        name = self.current_endview_colormap()
        if name:
            self.endview_colormap_changed.emit(name)

    def _on_overlay_opacity_value_changed(self, value: int) -> None:
        self._sync_pair_from_source(
            source_value=int(value),
            slider=self._overlay_opacity_slider,
            spinbox=self._overlay_opacity_spinbox,
        )
        self.overlay_opacity_changed.emit(float(int(value)) / 100.0)

    def _on_nde_opacity_value_changed(self, value: int) -> None:
        self._sync_pair_from_source(
            source_value=int(value),
            slider=self._nde_opacity_slider,
            spinbox=self._nde_opacity_spinbox,
        )
        self.nde_opacity_changed.emit(float(int(value)) / 100.0)

    def _on_nde_contrast_value_changed(self, value: int) -> None:
        self._sync_pair_from_source(
            source_value=int(value),
            slider=self._nde_contrast_slider,
            spinbox=self._nde_contrast_spinbox,
        )
        self.nde_contrast_changed.emit(float(int(value)) / 100.0)

    def _on_primary_slice_changed(self, value: int) -> None:
        self._sync_pair_from_source(
            source_value=int(value),
            slider=self._primary_slider,
            spinbox=self._primary_spinbox,
        )
        self.slice_changed.emit(int(value))

    def _on_secondary_slice_changed(self, value: int) -> None:
        self._sync_pair_from_source(
            source_value=int(value),
            slider=self._secondary_slider,
            spinbox=self._secondary_spinbox,
        )
        self.secondary_slice_changed.emit(int(value))

    def _sync_pair_from_source(
        self,
        *,
        source_value: int,
        slider: Optional[QSlider],
        spinbox: Optional[QSpinBox],
    ) -> None:
        if slider is None or spinbox is None:
            return
        clamped = max(int(slider.minimum()), min(int(slider.maximum()), int(source_value)))
        if slider.value() != clamped:
            slider.blockSignals(True)
            slider.setValue(clamped)
            slider.blockSignals(False)
        if spinbox.value() != clamped:
            spinbox.blockSignals(True)
            spinbox.setValue(clamped)
            spinbox.blockSignals(False)

    def set_position_label(self, x: int, y: int) -> None:
        """Update the position label with the latest cursor coordinates."""
        if self._position_label is None:
            return
        self._position_label.setText(f"position x = {int(x)} ; y = {int(y)}")

    def set_paint_size(self, radius: int) -> None:
        """Update the paint size slider without emitting signals."""
        if self._paint_size_slider is None:
            return
        clamped = max(
            int(self._paint_size_slider.minimum()),
            min(int(self._paint_size_slider.maximum()), int(radius)),
        )
        self._paint_size_slider.blockSignals(True)
        self._paint_size_slider.setValue(clamped)
        self._paint_size_slider.blockSignals(False)

    def set_nde_name(self, name: str) -> None:
        """Display the opened NDE file name."""
        if self._nde_label is None:
            return
        suffix = str(name).strip() if name else "-"
        self._nde_label.setText(f"NDE: {suffix}")

    def set_endview_name(self, name: str) -> None:
        """Display the current endview identifier."""
        if self._endview_label is None:
            return
        suffix = str(name).strip() if name else "-"
        self._endview_label.setText(f"Endview: {suffix}")

    def set_primary_axis_name(self, name: str) -> None:
        """Display the primary coordinate axis label."""
        if self._primary_axis_label is None:
            return
        self._primary_axis_label.setText(str(name).strip() if name else "U-Coordinate")

    def set_secondary_axis_name(self, name: str) -> None:
        """Display the secondary coordinate axis label."""
        if self._secondary_axis_label is None:
            return
        self._secondary_axis_label.setText(str(name).strip() if name else "V-Coordinate")

    def set_nde_opacity_available(self, available: bool) -> None:
        enabled = bool(available)
        tooltip = "" if enabled else "Affichage NDE non disponible pour l'instant."
        for widget in (
            self._nde_opacity_slider,
            self._nde_opacity_spinbox,
            self._nde_opacity_label,
            self._nde_contrast_slider,
            self._nde_contrast_spinbox,
            self._nde_contrast_label,
        ):
            if widget is None:
                continue
            widget.setEnabled(enabled)
            widget.setToolTip(tooltip)

    @classmethod
    def _normalize_colormap_name(cls, value: str) -> str:
        text = str(value).strip()
        return cls._COLORMAP_BY_TEXT.get(text.lower(), text or "Gris")
