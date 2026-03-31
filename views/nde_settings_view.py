"""Simple settings dialog to pick colormaps for views."""

from __future__ import annotations

from typing import Iterable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config.constants import format_label_text


class NdeSettingsView(QDialog):
    """Floating window to select colormaps for endview/3D and C-scan."""

    endview_colormap_changed = pyqtSignal(str)
    cscan_colormap_changed = pyqtSignal(str)
    apply_volume_range_changed = pyqtSignal(int, int)
    erase_label_target_changed = pyqtSignal(object)
    roi_thin_line_width_changed = pyqtSignal(int)
    roi_peak_preference_changed = pyqtSignal(bool)
    roi_peak_ignore_position_changed = pyqtSignal(bool)
    roi_peak_vertical_min_changed = pyqtSignal(int)
    roi_peak_vertical_max_changed = pyqtSignal(int)
    closing_mask_tolerance_changed = pyqtSignal(int)
    closing_mask_merge_distance_changed = pyqtSignal(int)
    clean_outliers_tolerance_changed = pyqtSignal(int)
    clean_outliers_thin_line_width_changed = pyqtSignal(int)
    clean_outliers_thin_gap_width_changed = pyqtSignal(int)
    clean_outliers_contour_smoothing_changed = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Paramètres")
        self.setModal(False)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._endview_combo = QComboBox(self)
        self._cscan_combo = QComboBox(self)
        form.addRow(QLabel("Colormap Endview + 3D"), self._endview_combo)
        form.addRow(QLabel("Colormap C-scan"), self._cscan_combo)

        self._apply_volume_start = QSpinBox(self)
        self._apply_volume_end = QSpinBox(self)
        for box in (self._apply_volume_start, self._apply_volume_end):
            box.setMinimum(0)
            box.setMaximum(0)
            box.setValue(0)
            box.setKeyboardTracking(False)
        form.addRow(QLabel("Appliquer au volume (de)"), self._apply_volume_start)
        form.addRow(QLabel("Appliquer au volume (à)"), self._apply_volume_end)

        self._erase_label_combo = QComboBox(self)
        form.addRow(QLabel("Effacement label 0"), self._erase_label_combo)

        self._roi_thin_line_width = QSpinBox(self)
        self._roi_thin_line_width.setMinimum(0)
        self._roi_thin_line_width.setMaximum(20)
        self._roi_thin_line_width.setValue(2)
        form.addRow(QLabel("Bloquer lignes <= (px)"), self._roi_thin_line_width)

        self._roi_peak_combo = QComboBox(self)
        self._roi_peak_combo.addItem("Premier pic", False)
        self._roi_peak_combo.addItem("Deuxieme pic", True)
        form.addRow(QLabel("ROI Peak - choix du pic"), self._roi_peak_combo)

        self._roi_peak_ignore_combo = QComboBox(self)
        self._roi_peak_ignore_combo.addItem("Avec position", False)
        self._roi_peak_ignore_combo.addItem("Plus fort (sans position)", True)
        form.addRow(QLabel("ROI Peak - mode du pic"), self._roi_peak_ignore_combo)

        self._roi_peak_vertical_min = QSpinBox(self)
        self._roi_peak_vertical_min.setMinimum(1)
        self._roi_peak_vertical_min.setMaximum(999)
        self._roi_peak_vertical_min.setValue(1)
        form.addRow(QLabel("ROI Peak - min vertical"), self._roi_peak_vertical_min)

        self._roi_peak_vertical_max = QSpinBox(self)
        self._roi_peak_vertical_max.setMinimum(0)
        self._roi_peak_vertical_max.setMaximum(999)
        self._roi_peak_vertical_max.setValue(0)
        self._roi_peak_vertical_max.setSpecialValueText("Illimite")
        form.addRow(QLabel("ROI Peak - max vertical"), self._roi_peak_vertical_max)

        self._closing_mask_tolerance = QSpinBox(self)
        self._closing_mask_tolerance.setMinimum(0)
        self._closing_mask_tolerance.setMaximum(99999)
        self._closing_mask_tolerance.setValue(64)
        form.addRow(QLabel("Closing mask - aire max trou (px2)"), self._closing_mask_tolerance)

        self._closing_mask_merge_distance = QSpinBox(self)
        self._closing_mask_merge_distance.setMinimum(0)
        self._closing_mask_merge_distance.setMaximum(9999)
        self._closing_mask_merge_distance.setValue(0)
        form.addRow(QLabel("Closing mask - distance fusion (px)"), self._closing_mask_merge_distance)

        self._clean_outliers_tolerance = QSpinBox(self)
        self._clean_outliers_tolerance.setMinimum(0)
        self._clean_outliers_tolerance.setMaximum(99999)
        self._clean_outliers_tolerance.setValue(64)
        form.addRow(QLabel("Mask cleanup - aire max ilot (px2)"), self._clean_outliers_tolerance)

        self._clean_outliers_thin_line_width = QSpinBox(self)
        self._clean_outliers_thin_line_width.setMinimum(0)
        self._clean_outliers_thin_line_width.setMaximum(20)
        self._clean_outliers_thin_line_width.setValue(1)
        form.addRow(
            QLabel("Mask cleanup - largeur max excroissance (px)"),
            self._clean_outliers_thin_line_width,
        )

        self._clean_outliers_thin_gap_width = QSpinBox(self)
        self._clean_outliers_thin_gap_width.setMinimum(0)
        self._clean_outliers_thin_gap_width.setMaximum(20)
        self._clean_outliers_thin_gap_width.setValue(0)
        form.addRow(
            QLabel("Mask cleanup - largeur max entaille (px)"),
            self._clean_outliers_thin_gap_width,
        )

        self._clean_outliers_contour_smoothing = QSpinBox(self)
        self._clean_outliers_contour_smoothing.setMinimum(0)
        self._clean_outliers_contour_smoothing.setMaximum(20)
        self._clean_outliers_contour_smoothing.setValue(0)
        form.addRow(
            QLabel("Mask cleanup - lissage contour (px)"),
            self._clean_outliers_contour_smoothing,
        )

        layout.addLayout(form)

        close_btn = QPushButton("Fermer", self)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, 0, alignment=Qt.AlignmentFlag.AlignRight)

        self._populate(["Gris", "OmniScan"])
        self._wire_signals()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_colormaps(self, *, endview: str, cscan: str) -> None:
        """Populate combos with the provided current values."""
        self.set_endview_colormap(endview)
        self.set_cscan_colormap(cscan)

    def set_endview_colormap(self, value: str) -> None:
        """Update the endview/3D colormap combo without emitting signals."""
        self._set_current_blocked(self._endview_combo, value)

    def set_cscan_colormap(self, value: str) -> None:
        """Update the C-scan colormap combo without emitting signals."""
        self._set_current_blocked(self._cscan_combo, value)

    def set_apply_volume_bounds(self, minimum: int, maximum: int) -> None:
        """Set min/max bounds for apply-to-volume range."""
        min_idx = int(minimum)
        max_idx = int(maximum)
        if max_idx < min_idx:
            max_idx = min_idx
        for box in (self._apply_volume_start, self._apply_volume_end):
            box.blockSignals(True)
            box.setMinimum(min_idx)
            box.setMaximum(max_idx)
            box.blockSignals(False)

    def set_apply_volume_range(self, start: int, end: int) -> None:
        """Update apply-to-volume range without emitting signals."""
        start_idx = int(start)
        end_idx = int(end)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        self._apply_volume_start.blockSignals(True)
        self._apply_volume_end.blockSignals(True)
        self._apply_volume_start.setValue(start_idx)
        self._apply_volume_end.setValue(end_idx)
        self._apply_volume_start.blockSignals(False)
        self._apply_volume_end.blockSignals(False)

    def set_erase_label_choices(self, labels: Iterable[int], *, current: Optional[int]) -> None:
        """Populate the erase-target combo (None = Tous)."""
        self._erase_label_combo.blockSignals(True)
        self._erase_label_combo.clear()
        self._erase_label_combo.addItem("Tous", None)
        for label_id in labels:
            lbl = int(label_id)
            self._erase_label_combo.addItem(format_label_text(lbl), lbl)
        self._set_current_data(self._erase_label_combo, current)
        self._erase_label_combo.blockSignals(False)

    def set_roi_thin_line_max_width(self, value: int) -> None:
        """Update the thin-line pruning width without emitting signals."""
        try:
            width = int(value)
        except Exception:
            width = 0
        if width < 0:
            width = 0
        self._roi_thin_line_width.blockSignals(True)
        self._roi_thin_line_width.setValue(width)
        self._roi_thin_line_width.blockSignals(False)

    def set_roi_peak_prefer_second(self, enabled: bool) -> None:
        """Update the Peak ROI preference without emitting signals."""
        target = bool(enabled)
        self._roi_peak_combo.blockSignals(True)
        idx = self._roi_peak_combo.findData(target)
        if idx < 0:
            idx = 1 if target else 0
        self._roi_peak_combo.setCurrentIndex(idx)
        self._roi_peak_combo.blockSignals(False)

    def set_roi_peak_ignore_position(self, enabled: bool) -> None:
        """Update strongest-peak-only mode without emitting signals."""
        target = bool(enabled)
        self._roi_peak_ignore_combo.blockSignals(True)
        idx = self._roi_peak_ignore_combo.findData(target)
        if idx < 0:
            idx = 1 if target else 0
        self._roi_peak_ignore_combo.setCurrentIndex(idx)
        self._roi_peak_ignore_combo.blockSignals(False)

    def set_roi_peak_vertical_min_length(self, value: int) -> None:
        """Update minimum vertical peak length without emitting signals."""
        try:
            min_len = int(value)
        except Exception:
            min_len = 1
        if min_len < 1:
            min_len = 1
        self._roi_peak_vertical_min.blockSignals(True)
        self._roi_peak_vertical_min.setValue(min_len)
        self._roi_peak_vertical_min.blockSignals(False)

    def set_roi_peak_vertical_max_length(self, value: int) -> None:
        """Update maximum vertical peak length without emitting signals (0 = unlimited)."""
        try:
            max_len = int(value)
        except Exception:
            max_len = 0
        if max_len < 0:
            max_len = 0
        self._roi_peak_vertical_max.blockSignals(True)
        self._roi_peak_vertical_max.setValue(max_len)
        self._roi_peak_vertical_max.blockSignals(False)

    def set_closing_mask_tolerance(self, value: int) -> None:
        """Update closing-mask hole tolerance without emitting signals."""
        try:
            tolerance = int(value)
        except Exception:
            tolerance = 0
        if tolerance < 0:
            tolerance = 0
        self._closing_mask_tolerance.blockSignals(True)
        self._closing_mask_tolerance.setValue(tolerance)
        self._closing_mask_tolerance.blockSignals(False)

    def set_closing_mask_merge_distance(self, value: int) -> None:
        """Update closing-mask merge distance without emitting signals."""
        try:
            distance = int(value)
        except Exception:
            distance = 0
        if distance < 0:
            distance = 0
        self._closing_mask_merge_distance.blockSignals(True)
        self._closing_mask_merge_distance.setValue(distance)
        self._closing_mask_merge_distance.blockSignals(False)

    def set_clean_outliers_tolerance(self, value: int) -> None:
        """Update clean-outliers tolerance without emitting signals."""
        try:
            tolerance = int(value)
        except Exception:
            tolerance = 0
        if tolerance < 0:
            tolerance = 0
        self._clean_outliers_tolerance.blockSignals(True)
        self._clean_outliers_tolerance.setValue(tolerance)
        self._clean_outliers_tolerance.blockSignals(False)

    def set_clean_outliers_thin_line_max_width(self, value: int) -> None:
        """Update mask-cleanup foreground protrusion width without emitting signals."""
        try:
            width = int(value)
        except Exception:
            width = 0
        if width < 0:
            width = 0
        self._clean_outliers_thin_line_width.blockSignals(True)
        self._clean_outliers_thin_line_width.setValue(width)
        self._clean_outliers_thin_line_width.blockSignals(False)

    def set_clean_outliers_thin_gap_max_width(self, value: int) -> None:
        """Update mask-cleanup background notch width without emitting signals."""
        try:
            width = int(value)
        except Exception:
            width = 0
        if width < 0:
            width = 0
        self._clean_outliers_thin_gap_width.blockSignals(True)
        self._clean_outliers_thin_gap_width.setValue(width)
        self._clean_outliers_thin_gap_width.blockSignals(False)

    def set_clean_outliers_contour_smoothing(self, value: int) -> None:
        """Update mask-cleanup contour smoothing without emitting signals."""
        try:
            smoothing = int(value)
        except Exception:
            smoothing = 0
        if smoothing < 0:
            smoothing = 0
        self._clean_outliers_contour_smoothing.blockSignals(True)
        self._clean_outliers_contour_smoothing.setValue(smoothing)
        self._clean_outliers_contour_smoothing.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _populate(self, choices: Iterable[str]) -> None:
        self._endview_combo.clear()
        self._cscan_combo.clear()
        for choice in choices:
            self._endview_combo.addItem(choice)
            self._cscan_combo.addItem(choice)

    def _wire_signals(self) -> None:
        self._endview_combo.currentTextChanged.connect(self.endview_colormap_changed.emit)
        self._cscan_combo.currentTextChanged.connect(self.cscan_colormap_changed.emit)
        self._apply_volume_start.valueChanged.connect(self._on_apply_volume_start_changed)
        self._apply_volume_end.valueChanged.connect(self._on_apply_volume_end_changed)
        self._erase_label_combo.currentIndexChanged.connect(self._on_erase_label_target_changed)
        self._roi_thin_line_width.valueChanged.connect(self._on_roi_thin_line_width_changed)
        self._roi_peak_combo.currentIndexChanged.connect(self._on_roi_peak_preference_changed)
        self._roi_peak_ignore_combo.currentIndexChanged.connect(
            self._on_roi_peak_ignore_position_changed
        )
        self._roi_peak_vertical_min.valueChanged.connect(self._on_roi_peak_vertical_min_changed)
        self._roi_peak_vertical_max.valueChanged.connect(self._on_roi_peak_vertical_max_changed)
        self._closing_mask_tolerance.valueChanged.connect(self._on_closing_mask_tolerance_changed)
        self._closing_mask_merge_distance.valueChanged.connect(
            self._on_closing_mask_merge_distance_changed
        )
        self._clean_outliers_tolerance.valueChanged.connect(
            self._on_clean_outliers_tolerance_changed
        )
        self._clean_outliers_thin_line_width.valueChanged.connect(
            self._on_clean_outliers_thin_line_width_changed
        )
        self._clean_outliers_thin_gap_width.valueChanged.connect(
            self._on_clean_outliers_thin_gap_width_changed
        )
        self._clean_outliers_contour_smoothing.valueChanged.connect(
            self._on_clean_outliers_contour_smoothing_changed
        )

    def _emit_apply_volume_range(self) -> None:
        self.apply_volume_range_changed.emit(
            int(self._apply_volume_start.value()),
            int(self._apply_volume_end.value()),
        )

    def _on_apply_volume_start_changed(self, value: int) -> None:
        if value > self._apply_volume_end.value():
            self._apply_volume_end.blockSignals(True)
            self._apply_volume_end.setValue(int(value))
            self._apply_volume_end.blockSignals(False)
        self._emit_apply_volume_range()

    def _on_apply_volume_end_changed(self, value: int) -> None:
        if value < self._apply_volume_start.value():
            self._apply_volume_start.blockSignals(True)
            self._apply_volume_start.setValue(int(value))
            self._apply_volume_start.blockSignals(False)
        self._emit_apply_volume_range()

    def _on_erase_label_target_changed(self, _index: int) -> None:
        value = self._erase_label_combo.currentData()
        if value is None:
            self.erase_label_target_changed.emit(None)
            return
        try:
            self.erase_label_target_changed.emit(int(value))
        except Exception:
            self.erase_label_target_changed.emit(None)

    def _on_roi_thin_line_width_changed(self, value: int) -> None:
        try:
            self.roi_thin_line_width_changed.emit(int(value))
        except Exception:
            self.roi_thin_line_width_changed.emit(0)

    def _on_roi_peak_preference_changed(self, _index: int) -> None:
        data = self._roi_peak_combo.currentData()
        self.roi_peak_preference_changed.emit(bool(data))

    def _on_roi_peak_ignore_position_changed(self, _index: int) -> None:
        data = self._roi_peak_ignore_combo.currentData()
        self.roi_peak_ignore_position_changed.emit(bool(data))

    def _on_roi_peak_vertical_min_changed(self, value: int) -> None:
        min_len = max(1, int(value))
        max_len = int(self._roi_peak_vertical_max.value())
        if max_len > 0 and min_len > max_len:
            self._roi_peak_vertical_max.blockSignals(True)
            self._roi_peak_vertical_max.setValue(min_len)
            self._roi_peak_vertical_max.blockSignals(False)
            max_len = min_len
        self.roi_peak_vertical_min_changed.emit(min_len)
        if max_len > 0:
            self.roi_peak_vertical_max_changed.emit(max_len)

    def _on_roi_peak_vertical_max_changed(self, value: int) -> None:
        max_len = max(0, int(value))
        min_len = int(self._roi_peak_vertical_min.value())
        if max_len > 0 and max_len < min_len:
            self._roi_peak_vertical_min.blockSignals(True)
            self._roi_peak_vertical_min.setValue(max_len)
            self._roi_peak_vertical_min.blockSignals(False)
            min_len = max_len
            self.roi_peak_vertical_min_changed.emit(min_len)
        self.roi_peak_vertical_max_changed.emit(max_len)

    def _on_closing_mask_tolerance_changed(self, value: int) -> None:
        self.closing_mask_tolerance_changed.emit(max(0, int(value)))

    def _on_closing_mask_merge_distance_changed(self, value: int) -> None:
        self.closing_mask_merge_distance_changed.emit(max(0, int(value)))

    def _on_clean_outliers_tolerance_changed(self, value: int) -> None:
        self.clean_outliers_tolerance_changed.emit(max(0, int(value)))

    def _on_clean_outliers_thin_line_width_changed(self, value: int) -> None:
        self.clean_outliers_thin_line_width_changed.emit(max(0, int(value)))

    def _on_clean_outliers_thin_gap_width_changed(self, value: int) -> None:
        self.clean_outliers_thin_gap_width_changed.emit(max(0, int(value)))

    def _on_clean_outliers_contour_smoothing_changed(self, value: int) -> None:
        self.clean_outliers_contour_smoothing_changed.emit(max(0, int(value)))

    @staticmethod
    def _set_current(combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx < 0:
            combo.addItem(value)
            idx = combo.findText(value)
        combo.setCurrentIndex(idx)

    @classmethod
    def _set_current_blocked(cls, combo: QComboBox, value: str) -> None:
        combo.blockSignals(True)
        cls._set_current(combo, value)
        combo.blockSignals(False)

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
