from __future__ import annotations

from typing import Iterable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config.constants import (
    format_label_text,
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
    peak_mode_a_changed = pyqtSignal(str)
    peak_mode_b_changed = pyqtSignal(str)
    interpolation_algo_changed = pyqtSignal(str)

    _DEFAULT_PEAK_MODE = "max_peak"
    _DEFAULT_INTERP_ALGO = "1d_dual_axis"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Parametres corrosion")
        self.setModal(False)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._label_a_combo = QComboBox(self)
        self._peak_mode_a_combo = QComboBox(self)
        self._label_b_combo = QComboBox(self)
        self._peak_mode_b_combo = QComboBox(self)
        self._interp_algo_combo = QComboBox(self)

        self._peak_mode_a_combo.addItems(["Max peak", "Optimiste", "Pessimiste"])
        self._peak_mode_b_combo.addItems(["Max peak", "Optimiste", "Pessimiste"])
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

        form.addRow(QLabel("Label A"), self._label_a_combo)
        form.addRow(QLabel("Mode label A"), self._peak_mode_a_combo)
        form.addRow(QLabel("Label B"), self._label_b_combo)
        form.addRow(QLabel("Mode label B"), self._peak_mode_b_combo)
        form.addRow(QLabel("Interpolation"), self._interp_algo_combo)

        layout.addLayout(form)

        close_btn = QPushButton("Fermer", self)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, 0, alignment=Qt.AlignmentFlag.AlignRight)

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
        self._label_a_combo.addItem("Aucun", None)
        self._label_b_combo.addItem("Aucun", None)
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

    def current_peak_selection_mode_a(self) -> str:
        return self._combo_mode_value(self._peak_mode_a_combo, normalize_corrosion_peak_selection_mode)

    def current_peak_selection_mode_b(self) -> str:
        return self._combo_mode_value(self._peak_mode_b_combo, normalize_corrosion_peak_selection_mode)

    def current_interpolation_algo(self) -> str:
        return self._combo_mode_value(self._interp_algo_combo, normalize_interpolation_algo)

    def set_workflow_state(self, stage: str) -> None:
        normalized = str(stage or "").strip().lower()
        analysis_enabled = normalized == CORROSION_STAGE_BASE
        interpolation_enabled = normalized == CORROSION_STAGE_RAW

        analysis_tooltip = ""
        interpolation_tooltip = ""
        if normalized == CORROSION_STAGE_RAW:
            analysis_tooltip = "La session brute est deja analysee."
        elif normalized == CORROSION_STAGE_INTERPOLATED:
            analysis_tooltip = "La session interpolee est finalisee."
            interpolation_tooltip = "La session interpolee ne peut pas etre re-interpolee."
        else:
            interpolation_tooltip = "Lance d'abord Analyze pour creer une session brute."

        for widget in (
            self._label_a_combo,
            self._peak_mode_a_combo,
            self._label_b_combo,
            self._peak_mode_b_combo,
        ):
            widget.setEnabled(analysis_enabled)
            widget.setToolTip(analysis_tooltip)

        self._interp_algo_combo.setEnabled(interpolation_enabled)
        self._interp_algo_combo.setToolTip(interpolation_tooltip)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _wire_signals(self) -> None:
        self._label_a_combo.currentIndexChanged.connect(self._on_label_a_changed)
        self._label_b_combo.currentIndexChanged.connect(self._on_label_b_changed)
        self._peak_mode_a_combo.currentIndexChanged.connect(self._on_peak_mode_a_changed)
        self._peak_mode_b_combo.currentIndexChanged.connect(self._on_peak_mode_b_changed)
        self._interp_algo_combo.currentIndexChanged.connect(self._on_interpolation_algo_changed)

    def _on_label_a_changed(self, _index: int) -> None:
        self.label_a_changed.emit(self._label_a_combo.currentData())

    def _on_label_b_changed(self, _index: int) -> None:
        self.label_b_changed.emit(self._label_b_combo.currentData())

    def _on_peak_mode_a_changed(self, _index: int) -> None:
        self.peak_mode_a_changed.emit(self.current_peak_selection_mode_a())

    def _on_peak_mode_b_changed(self, _index: int) -> None:
        self.peak_mode_b_changed.emit(self.current_peak_selection_mode_b())

    def _on_interpolation_algo_changed(self, _index: int) -> None:
        self.interpolation_algo_changed.emit(self.current_interpolation_algo())

    def _prepare_peak_mode_combo(self, combo: QComboBox) -> None:
        for idx in range(combo.count()):
            mode = normalize_corrosion_peak_selection_mode(combo.itemText(idx))
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
