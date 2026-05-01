"""Dialog for NPZ overlay export options."""

from __future__ import annotations

from typing import NamedTuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)


class OverlayExportOptions(NamedTuple):
    """Holds user-selected export options."""

    export_target: str = "normal"
    rotation_degrees: int = 0
    rotation_axes: str = ""
    transpose_axes: str = ""
    output_suffix: str = ""
    mirror_horizontal: bool = False
    mirror_vertical: bool = False
    mirror_z: bool = False
    strict_mode: bool = False


class OverlayExportDialog(QDialog):
    """Modal dialog letting the user choose export options before saving."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Options d'export overlay")
        self.setModal(True)

        self._export_target = QComboBox(self)
        self._export_target.addItem("Normal", "normal")
        self._export_target.addItem("Sentinel", "sentinel")

        self._rotation = QComboBox(self)
        self._rotation.addItems(["0°", "90°", "180°", "270°"])
        self._rotation.setCurrentText("270°")

        self._rotation_axes = QLineEdit(self)
        self._rotation_axes.setText("-2, -1")
        self._transpose_axes = QLineEdit(self)
        self._transpose_axes.setPlaceholderText("0,2,1")
        self._output_suffix = QLineEdit(self)
        self._output_suffix.setText("_sentinel")

        self._mirror_horizontal = QCheckBox("Miroir horizontal", self)
        self._mirror_vertical = QCheckBox("Miroir vertical", self)
        self._mirror_vertical.setChecked(True)
        self._mirror_z = QCheckBox("Miroir Z (axe 0)", self)
        self._mirror_z.setChecked(True)
        self._recursive_mode = QCheckBox("Recursif (mode dossier)", self)
        self._recursive_mode.setEnabled(False)
        self._overwrite_original = QCheckBox("Ecraser l'original (--inplace)", self)
        self._overwrite_original.setEnabled(False)
        self._strict_mode = QCheckBox("Mode strict", self)

        sentinel_box = QGroupBox("Transformations Sentinel", self)
        sentinel_form = QFormLayout(sentinel_box)
        sentinel_form.addRow("Rotation", self._rotation)
        sentinel_form.addRow("Axes rotation (ex: -2,-1)", self._rotation_axes)
        sentinel_form.addRow("Transpose (ex: 0,2,1)", self._transpose_axes)
        sentinel_form.addRow("Suffixe sortie", self._output_suffix)
        sentinel_form.addRow(self._mirror_horizontal)
        sentinel_form.addRow(self._mirror_vertical)
        sentinel_form.addRow(self._mirror_z)
        sentinel_form.addRow(self._recursive_mode)
        sentinel_form.addRow(self._overwrite_original)
        sentinel_form.addRow(self._strict_mode)

        order_hint = QLabel(
            "Ordre applique: transpose -> rotate -> mirror-h -> mirror-v -> mirror-z",
            self,
        )
        order_hint.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(QLabel("Type d'export", self))
        layout.addWidget(self._export_target)
        layout.addWidget(sentinel_box)
        layout.addWidget(order_hint)
        layout.addStretch()
        layout.addWidget(buttons, 0, alignment=Qt.AlignmentFlag.AlignRight)

        self._sentinel_box = sentinel_box
        self._export_target.currentIndexChanged.connect(self._sync_mode_widgets)
        self._sync_mode_widgets()

    def get_options(self) -> OverlayExportOptions:
        """Return the selected export options."""
        rotation_text = self._rotation.currentText().replace("°", "").strip()
        try:
            rotation_value = int(rotation_text)
        except Exception:
            rotation_value = 0
        if rotation_value not in (0, 90, 180, 270):
            rotation_value = 0

        export_target = str(self._export_target.currentData() or "normal")
        return OverlayExportOptions(
            export_target=export_target,
            rotation_degrees=rotation_value,
            rotation_axes=self._rotation_axes.text().strip(),
            transpose_axes=self._transpose_axes.text().strip(),
            output_suffix=self._output_suffix.text().strip(),
            mirror_horizontal=bool(self._mirror_horizontal.isChecked()),
            mirror_vertical=bool(self._mirror_vertical.isChecked()),
            mirror_z=bool(self._mirror_z.isChecked()),
            strict_mode=bool(self._strict_mode.isChecked()),
        )

    def _sync_mode_widgets(self) -> None:
        """Enable Sentinel-only controls when that export target is selected."""
        is_sentinel = str(self._export_target.currentData() or "") == "sentinel"
        self._sentinel_box.setEnabled(is_sentinel)
