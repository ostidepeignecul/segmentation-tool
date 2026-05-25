"""Dialog for the `Export all` bundle flow."""

from __future__ import annotations

from typing import NamedTuple, Sequence

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


class SessionBundleExportOptions(NamedTuple):
    """User-selected options for the export bundle."""

    export_session_npz: bool = True
    export_sentinel_npz: bool = True
    export_nnunet_pngs: bool = True
    export_corrosion_cscan: bool = True
    main_layer_id: str = ""
    sentinel_source_view: str = "dscan"
    rotation_degrees: int = 0
    rotation_axes: str = ""
    transpose_axes: str = ""
    output_suffix: str = "_sentinel"
    mirror_horizontal: bool = False
    mirror_vertical: bool = False
    mirror_z: bool = False
    strict_mode: bool = False


class SessionBundleExportDialog(QDialog):
    """Collect the options required by the `Export all` action."""

    def __init__(
        self,
        parent=None,
        *,
        layer_choices: Sequence[tuple[str, str]],
        default_main_layer_id: str = "",
        default_sentinel_source_view: str = "dscan",
        has_corrosion_layers: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export all")
        self.setModal(True)

        self._export_session_npz = QCheckBox("Session NPZ for all layers", self)
        self._export_session_npz.setChecked(True)
        self._export_sentinel_npz = QCheckBox("Sentinel NPZ for main layer", self)
        self._export_sentinel_npz.setChecked(True)
        self._export_nnunet_pngs = QCheckBox("nnU-Net PNGs for main layer", self)
        self._export_nnunet_pngs.setChecked(True)
        self._export_corrosion_cscan = QCheckBox(
            "Corrosion C-scan for raw/interpolated layers",
            self,
        )
        self._export_corrosion_cscan.setChecked(bool(has_corrosion_layers))
        self._export_corrosion_cscan.setEnabled(bool(has_corrosion_layers))

        bundle_box = QGroupBox("Bundle content", self)
        bundle_layout = QVBoxLayout(bundle_box)
        bundle_layout.addWidget(self._export_session_npz)
        bundle_layout.addWidget(self._export_sentinel_npz)
        bundle_layout.addWidget(self._export_nnunet_pngs)
        bundle_layout.addWidget(self._export_corrosion_cscan)

        self._main_layer = QComboBox(self)
        for layer_id, label in layer_choices:
            self._main_layer.addItem(label, layer_id)
        default_layer_index = self._main_layer.findData(default_main_layer_id)
        if default_layer_index < 0 and self._main_layer.count() > 0:
            default_layer_index = 0
        if default_layer_index >= 0:
            self._main_layer.setCurrentIndex(default_layer_index)
        self._main_layer.setEnabled(self._main_layer.count() > 1)

        main_layer_box = QGroupBox("Main layer", self)
        main_layer_form = QFormLayout(main_layer_box)
        main_layer_form.addRow(
            QLabel("Used by Sentinel and nnU-Net exports.", self),
        )
        main_layer_form.addRow("Layer", self._main_layer)

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
        self._strict_mode = QCheckBox("Strict mode", self)

        self._sentinel_box = QGroupBox("Sentinel transforms", self)
        sentinel_form = QFormLayout(self._sentinel_box)
        sentinel_form.addRow("Source view", self._sentinel_source_view)
        sentinel_form.addRow("Rotation", self._rotation)
        sentinel_form.addRow("Rotation axes (e.g. -2,-1)", self._rotation_axes)
        sentinel_form.addRow("Transpose (e.g. 0,2,1)", self._transpose_axes)
        sentinel_form.addRow("Output suffix", self._output_suffix)
        sentinel_form.addRow(self._mirror_horizontal)
        sentinel_form.addRow(self._mirror_vertical)
        sentinel_form.addRow(self._mirror_z)
        sentinel_form.addRow(self._strict_mode)

        order_hint = QLabel(
            "Sentinel order: transpose -> rotate -> mirror-h -> mirror-v -> mirror-z",
            self,
        )
        order_hint.setWordWrap(True)

        no_corrosion_hint = None
        if not has_corrosion_layers:
            no_corrosion_hint = QLabel(
                "No raw/interpolated corrosion layer is available in the active session.",
                self,
            )
            no_corrosion_hint.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(bundle_box)
        layout.addWidget(main_layer_box)
        layout.addWidget(self._sentinel_box)
        layout.addWidget(order_hint)
        if no_corrosion_hint is not None:
            layout.addWidget(no_corrosion_hint)
        layout.addStretch()
        layout.addWidget(buttons, 0, alignment=Qt.AlignmentFlag.AlignRight)

        self._export_sentinel_npz.toggled.connect(self._sync_widget_state)
        self._export_nnunet_pngs.toggled.connect(self._sync_widget_state)
        self._sync_widget_state()

    def get_options(self) -> SessionBundleExportOptions:
        """Return the selected export options."""
        rotation_text = (
            self._rotation.currentText()
            .replace("deg", "")
            .strip()
        )
        try:
            rotation_value = int(rotation_text)
        except Exception:
            rotation_value = 0
        if rotation_value not in (0, 90, 180, 270):
            rotation_value = 0

        return SessionBundleExportOptions(
            export_session_npz=bool(self._export_session_npz.isChecked()),
            export_sentinel_npz=bool(self._export_sentinel_npz.isChecked()),
            export_nnunet_pngs=bool(self._export_nnunet_pngs.isChecked()),
            export_corrosion_cscan=bool(self._export_corrosion_cscan.isChecked()),
            main_layer_id=str(self._main_layer.currentData() or ""),
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

    def _sync_widget_state(self) -> None:
        """Keep dependent widgets aligned with the selected export categories."""
        needs_main_layer = bool(
            self._export_sentinel_npz.isChecked() or self._export_nnunet_pngs.isChecked()
        )
        self._main_layer.setEnabled(needs_main_layer and self._main_layer.count() > 1)
        self._sentinel_box.setEnabled(bool(self._export_sentinel_npz.isChecked()))
