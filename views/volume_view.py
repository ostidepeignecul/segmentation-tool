"""VisPy-based 3D volume placeholder with simple controls."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QLabel, QSlider, QVBoxLayout, QWidget
from vispy import scene
from vispy.color import get_colormap


class VolumeView(QFrame):
    """Displays a basic 3D volume using VisPy."""

    volume_needs_update = pyqtSignal()
    overlay_updated = pyqtSignal()
    camera_changed = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._volume: Optional[np.ndarray] = None
        self._canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=False)
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.TurntableCamera(fov=45, elevation=30, azimuth=45)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        canvas_layout = QVBoxLayout(container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self._canvas.native)
        layout.addWidget(container, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._emit_slice_change)
        layout.addWidget(self._slider)

        self._status = QLabel("Volume non chargé")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        self._volume_visual = None

    def set_volume(self, volume: np.ndarray, cmap: str = "viridis") -> None:
        """Assign the 3D volume to render."""
        if volume is None or volume.size == 0:
            self._volume = None
            self._status.setText("Volume vide")
            return

        self._volume = np.asarray(volume, dtype=np.float32)
        self._status.setText(f"Volume chargé: {self._volume.shape}")
        self._slider.setMaximum(self._volume.shape[0] - 1)
        self._render_volume(cmap)

    def _render_volume(self, cmap: str) -> None:
        for child in list(self._view.scene.children):
            if child is not self._view.camera:
                child.parent = None
        data = self._volume
        if data is None:
            return

        if data.max() > data.min():
            norm = (data - data.min()) / (data.max() - data.min())
        else:
            norm = np.zeros_like(data, dtype=np.float32)

        try:
            colormap = get_colormap(cmap)
        except Exception:
            colormap = get_colormap("viridis")
        volume_visual = scene.visuals.Volume(
            norm,
            parent=self._view.scene,
            method="mip",
            cmap=colormap,
        )
        self._volume_visual = volume_visual
        self._view.camera.set_range()

    def update_volume(self) -> None:
        self.volume_needs_update.emit()

    def update_overlay(self) -> None:
        self.overlay_updated.emit()

    def _emit_slice_change(self, value: int) -> None:
        self.camera_changed.emit({"slice": value})
