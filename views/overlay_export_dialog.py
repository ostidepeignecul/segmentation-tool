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
    sentinel_source_view: str = "dscan"
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

    def __init__(
        self,
        parent=None,
        *,
        default_sentinel_source_view: str = "dscan",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Overlay export options")
        self.setModal(True)

        self._export_target = QComboBox(self)
        self._export_target.addItem("Normal", "normal")
        self._export_target.addItem("Sentinel", "sentinel")

        self._sentinel_source_view = QComboBox(self)
        self._sentinel_source_view.addItem("B-Scan", "bscan")
        self._sentinel_source_view.addItem("D-Scan", "dscan")
        default_source_index = self._sentinel_source_view.findData(
            str(default_sentinel_source_view or "").strip().lower()
        )
        if default_source_index < 0:
            default_source_index = self._sentinel_source_view.findData("dscan")
        if default_source_index >= 0:
            self._sentinel_source_view.setCurrentIndex(default_source_index)

        self._rotation = QComboBox(self)
        self._rotation.addItems(["0 deg", "90 deg", "180 deg", "270 deg"])
        self._rotation.setCurrentText("270 deg")

        self._rotation_axes = QLineEdit(self)
        self._rotation_axes.setText("-2, -1")
        self._transpose_axes = QLineEdit(self)
        self._transpose_axes.setPlaceholderText("0,2,1")
        self._output_suffix = QLineEdit(self)
        self._output_suffix.setText("_sentinel")

        self._mirror_horizontal = QCheckBox("Horizontal mirror", self)
        self._mirror_vertical = QCheckBox("Vertical mirror", self)
        self._mirror_vertical.setChecked(True)
        self._mirror_z = QCheckBox("Z mirror (axis 0)", self)
        self._mirror_z.setChecked(True)
        self._recursive_mode = QCheckBox("Recursive (folder mode)", self)
        self._recursive_mode.setEnabled(False)
        self._overwrite_original = QCheckBox("Overwrite original (--inplace)", self)
        self._overwrite_original.setEnabled(False)
        self._strict_mode = QCheckBox("Strict mode", self)

        sentinel_box = QGroupBox("Sentinel transforms", self)
        sentinel_form = QFormLayout(sentinel_box)
        sentinel_form.addRow("Source view", self._sentinel_source_view)
        sentinel_form.addRow("Rotation", self._rotation)
        sentinel_form.addRow("Rotation axes (e.g. -2,-1)", self._rotation_axes)
        sentinel_form.addRow("Transpose (e.g. 0,2,1)", self._transpose_axes)
        sentinel_form.addRow("Output suffix", self._output_suffix)
        sentinel_form.addRow(self._mirror_horizontal)
        sentinel_form.addRow(self._mirror_vertical)
        sentinel_form.addRow(self._mirror_z)
        sentinel_form.addRow(self._recursive_mode)
        sentinel_form.addRow(self._overwrite_original)
        sentinel_form.addRow(self._strict_mode)

        order_hint = QLabel(
            "Applied order: transpose -> rotate -> mirror-h -> mirror-v -> mirror-z",
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
        layout.addWidget(QLabel("Export type", self))
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
        rotation_text = (
            self._rotation.currentText()
            .replace("deg", "")
            .replace("°", "")
            .strip()
        )
        try:
            rotation_value = int(rotation_text)
        except Exception:
            rotation_value = 0
        if rotation_value not in (0, 90, 180, 270):
            rotation_value = 0

        export_target = str(self._export_target.currentData() or "normal")
        return OverlayExportOptions(
            export_target=export_target,
            sentinel_source_view=str(self._sentinel_source_view.currentData() or "dscan"),
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
