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

from config.constants import (
    format_label_text,
    normalize_corrosion_analysis_mode,
    normalize_corrosion_peak_selection_mode,
    normalize_interpolation_algo,
    CORROSION_STAGE_BASE,
    CORROSION_STAGE_RAW,
    CORROSION_STAGE_INTERPOLATED,
)


class CorrosionSettingsView(QDialog):
    """Floating window to configure corrosion analysis and interpolation."""

    label_a_changed = pyqtSignal(object)
    label_b_changed = pyqtSignal(object)
    analysis_mode_changed = pyqtSignal(str)
    peak_mode_a_changed = pyqtSignal(str)
    peak_mode_b_changed = pyqtSignal(str)
    interpolation_algo_changed = pyqtSignal(str)
    interpolation_gap_changed = pyqtSignal(int)

    _DEFAULT_ANALYSIS_MODE = "normal"
    _DEFAULT_PEAK_MODE = "max_peak"
    _DEFAULT_INTERP_ALGO = "1d_dual_axis"
    _MAX_INTERP_GAP_PX = 9999

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Corrosion settings")
        self.setModal(False)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._label_a_combo = QComboBox(self)
        self._analysis_mode_combo = QComboBox(self)
        self._peak_mode_a_combo = QComboBox(self)
        self._label_b_combo = QComboBox(self)
        self._peak_mode_b_combo = QComboBox(self)
        self._interp_algo_combo = QComboBox(self)
        self._interp_gap_spin = QSpinBox(self)

        self._analysis_mode_combo.addItems(["Normal", "AC-AB"])
        self._peak_mode_a_combo.addItems(
            ["Max peak", "Farthest from paired peak", "Closest to paired peak"]
        )
        self._peak_mode_b_combo.addItems(
            ["Max peak", "Farthest from paired peak", "Closest to paired peak"]
        )
        self._interp_algo_combo.addItems(
            [
                "1D Dual-Axis",
                "1D PCHIP Dual-Axis",
                "2D Linear ND",
                "2D Clough-Tocher",
                "1D Makima Dual-Axis",
                "2D RBF Thin-Plate",
                "2D Gaussian-Fill",
            ]
        )
        self._interp_gap_spin.setMinimum(0)
        self._interp_gap_spin.setMaximum(self._MAX_INTERP_GAP_PX)
        self._interp_gap_spin.setValue(0)
        self._interp_gap_spin.setSuffix(" px")
        self._interp_gap_spin.setToolTip(
            "Maximum consecutive empty A-scan columns bridged during interpolation. "
            "0 leaves empty A-scan gaps open."
        )

        form.addRow(QLabel("Label A"), self._label_a_combo)
        form.addRow(QLabel("Analysis mode"), self._analysis_mode_combo)
        form.addRow(QLabel("Label A mode"), self._peak_mode_a_combo)
        form.addRow(QLabel("Label B"), self._label_b_combo)
        form.addRow(QLabel("Label B mode"), self._peak_mode_b_combo)
        form.addRow(QLabel("Interpolation"), self._interp_algo_combo)
        form.addRow(QLabel("Empty A-scan gap"), self._interp_gap_spin)

        layout.addLayout(form)

        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, 0, alignment=Qt.AlignmentFlag.AlignRight)

        self._prepare_analysis_mode_combo(self._analysis_mode_combo)
        self._prepare_peak_mode_combo(self._peak_mode_a_combo)
        self._prepare_peak_mode_combo(self._peak_mode_b_combo)
        self._prepare_interpolation_combo(self._interp_algo_combo)
        self._wire_signals()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_label_choices(
        self,
        labels: Iterable[int],
        *,
        current_a: Optional[int],
        current_b: Optional[int],
    ) -> None:
        """Populate label selectors and set current selections without emitting signals."""
        self._label_a_combo.blockSignals(True)
        self._label_b_combo.blockSignals(True)
        self._label_a_combo.clear()
        self._label_b_combo.clear()
        self._label_a_combo.addItem("None", None)
        self._label_b_combo.addItem("None", None)
        for label_id in labels:
            lbl = int(label_id)
            self._label_a_combo.addItem(format_label_text(lbl), lbl)
            self._label_b_combo.addItem(format_label_text(lbl), lbl)
        self._set_current_data(self._label_a_combo, current_a)
        self._set_current_data(self._label_b_combo, current_b)
        self._label_a_combo.blockSignals(False)
        self._label_b_combo.blockSignals(False)

    def set_peak_selection_modes(
        self,
        *,
        current_a: str,
        current_b: str,
    ) -> None:
        self._set_mode_combo_value(self._peak_mode_a_combo, current_a, self._DEFAULT_PEAK_MODE)
        self._set_mode_combo_value(self._peak_mode_b_combo, current_b, self._DEFAULT_PEAK_MODE)

    def set_interpolation_algo(self, algo: str) -> None:
        self._set_mode_combo_value(
            self._interp_algo_combo,
            algo,
            self._DEFAULT_INTERP_ALGO,
        )

    def set_interpolation_gap_px(self, value: int) -> None:
        gap = self._normalize_gap_px(value, maximum=self._MAX_INTERP_GAP_PX)
        self._interp_gap_spin.blockSignals(True)
        self._interp_gap_spin.setValue(gap)
        self._interp_gap_spin.blockSignals(False)

    def set_analysis_mode(self, mode: str) -> None:
        self._set_mode_combo_value(
            self._analysis_mode_combo,
            mode,
            self._DEFAULT_ANALYSIS_MODE,
        )

    def current_analysis_mode(self) -> str:
        return self._combo_mode_value(
            self._analysis_mode_combo,
            normalize_corrosion_analysis_mode,
        )

    def current_peak_selection_mode_a(self) -> str:
        return self._combo_mode_value(self._peak_mode_a_combo, normalize_corrosion_peak_selection_mode)

    def current_peak_selection_mode_b(self) -> str:
        return self._combo_mode_value(self._peak_mode_b_combo, normalize_corrosion_peak_selection_mode)

    def current_interpolation_algo(self) -> str:
        return self._combo_mode_value(self._interp_algo_combo, normalize_interpolation_algo)

    def current_interpolation_gap_px(self) -> int:
        return int(self._interp_gap_spin.value())

    def set_workflow_state(self, stage: str) -> None:
        normalized = str(stage or "").strip().lower()
        analysis_enabled = normalized == CORROSION_STAGE_BASE
        interpolation_enabled = normalized == CORROSION_STAGE_RAW
        gap_enabled = normalized in (CORROSION_STAGE_BASE, CORROSION_STAGE_RAW)

        analysis_tooltip = ""
        interpolation_tooltip = ""
        gap_tooltip = (
            "Maximum consecutive empty A-scan columns bridged during interpolation. "
            "0 leaves empty A-scan gaps open."
        )
        if normalized == CORROSION_STAGE_RAW:
            analysis_tooltip = "The raw session has already been analyzed."
        elif normalized == CORROSION_STAGE_INTERPOLATED:
            analysis_tooltip = "The interpolated session is finalized."
            interpolation_tooltip = "The interpolated session cannot be interpolated again."
            gap_tooltip = "The interpolated session is finalized."
        else:
            interpolation_tooltip = "Run Analyze first to create a raw session."

        for widget in (
            self._label_a_combo,
            self._analysis_mode_combo,
            self._peak_mode_a_combo,
            self._label_b_combo,
            self._peak_mode_b_combo,
        ):
            widget.setEnabled(analysis_enabled)
            widget.setToolTip(analysis_tooltip)

        self._interp_algo_combo.setEnabled(interpolation_enabled)
        self._interp_algo_combo.setToolTip(interpolation_tooltip)
        self._interp_gap_spin.setEnabled(gap_enabled)
        self._interp_gap_spin.setToolTip(gap_tooltip)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _wire_signals(self) -> None:
        self._label_a_combo.currentIndexChanged.connect(self._on_label_a_changed)
        self._label_b_combo.currentIndexChanged.connect(self._on_label_b_changed)
        self._analysis_mode_combo.currentIndexChanged.connect(self._on_analysis_mode_changed)
        self._peak_mode_a_combo.currentIndexChanged.connect(self._on_peak_mode_a_changed)
        self._peak_mode_b_combo.currentIndexChanged.connect(self._on_peak_mode_b_changed)
        self._interp_algo_combo.currentIndexChanged.connect(self._on_interpolation_algo_changed)
        self._interp_gap_spin.valueChanged.connect(self._on_interpolation_gap_changed)

    def _on_label_a_changed(self, _index: int) -> None:
        self.label_a_changed.emit(self._label_a_combo.currentData())

    def _on_label_b_changed(self, _index: int) -> None:
        self.label_b_changed.emit(self._label_b_combo.currentData())

    def _on_analysis_mode_changed(self, _index: int) -> None:
        self.analysis_mode_changed.emit(self.current_analysis_mode())

    def _on_peak_mode_a_changed(self, _index: int) -> None:
        self.peak_mode_a_changed.emit(self.current_peak_selection_mode_a())

    def _on_peak_mode_b_changed(self, _index: int) -> None:
        self.peak_mode_b_changed.emit(self.current_peak_selection_mode_b())

    def _on_interpolation_algo_changed(self, _index: int) -> None:
        self.interpolation_algo_changed.emit(self.current_interpolation_algo())

    def _on_interpolation_gap_changed(self, value: int) -> None:
        self.interpolation_gap_changed.emit(self._normalize_gap_px(value))

    def _prepare_peak_mode_combo(self, combo: QComboBox) -> None:
        for idx in range(combo.count()):
            mode = normalize_corrosion_peak_selection_mode(combo.itemText(idx))
            combo.setItemData(idx, mode)

    def _prepare_analysis_mode_combo(self, combo: QComboBox) -> None:
        for idx in range(combo.count()):
            mode = normalize_corrosion_analysis_mode(combo.itemText(idx))
            combo.setItemData(idx, mode)

    def _prepare_interpolation_combo(self, combo: QComboBox) -> None:
        for idx in range(combo.count()):
            algo = normalize_interpolation_algo(combo.itemText(idx))
            combo.setItemData(idx, algo)

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

    @staticmethod
    def _combo_mode_value(
        combo: QComboBox,
        normalizer,
    ) -> str:
        data = combo.currentData()
        if data is not None:
            return str(data)
        return normalizer(combo.currentText())

    @staticmethod
    def _set_mode_combo_value(
        combo: QComboBox,
        value: str,
        default: str,
    ) -> None:
        normalized = str(value or "").strip().lower() or default
        target_index = combo.findData(normalized)
        if target_index < 0 and combo.count() > 0:
            target_index = 0
        if target_index < 0:
            return
        combo.blockSignals(True)
        combo.setCurrentIndex(target_index)
        combo.blockSignals(False)

    @staticmethod
    def _normalize_gap_px(value: int, *, maximum: Optional[int] = None) -> int:
        try:
            gap = int(value)
        except Exception:
            gap = 0
        if gap < 0:
            gap = 0
        if maximum is not None:
            gap = min(gap, int(maximum))
        return gap
