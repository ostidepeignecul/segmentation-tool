"""VisPy-based 3D volume placeholder with simple controls."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QLabel, QSlider, QVBoxLayout, QWidget
from vispy import scene
from vispy.visuals.transforms import STTransform


class VolumeView(QFrame):
    """Displays a basic 3D volume using VisPy."""

    volume_needs_update = pyqtSignal()
    overlay_updated = pyqtSignal()
    slice_changed = pyqtSignal(int)
    camera_changed = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._volume: Optional[np.ndarray] = None
        self._norm_volume: Optional[np.ndarray] = None
        self._volume_min: Optional[float] = None
        self._volume_max: Optional[float] = None
        self._current_slice: int = 0
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
        self._slice_highlight_visual = None
        self._slice_image = None

    def set_volume(self, volume: np.ndarray, cmap: str = "viridis", *, slice_idx: Optional[int] = None) -> None:
        """Assign the 3D volume to render."""
        if volume is None or volume.size == 0:
            self._volume = None
            self._status.setText("Volume vide")
            return

        self._volume = np.asarray(volume, dtype=np.float32)
        self._volume_min = float(np.min(self._volume))
        self._volume_max = float(np.max(self._volume))
        if self._volume_max > self._volume_min:
            self._norm_volume = (self._volume - self._volume_min) / (self._volume_max - self._volume_min)
        else:
            self._norm_volume = np.zeros_like(self._volume, dtype=np.float32)
        self._status.setText(f"Volume chargé: {self._volume.shape}")
        self._slider.setMaximum(self._volume.shape[0] - 1)
        self._build_scene(cmap)
        target_slice = slice_idx if slice_idx is not None else self._current_slice
        self.set_slice_index(target_slice, update_slider=True, emit=False)

    def set_slice_index(self, index: int, *, update_slider: bool = False, emit: bool = False) -> None:
        """Sync the slice indicator (slider + plane + image) with external selection."""
        if self._volume is None:
            return
        clamped = max(0, min(self._volume.shape[0] - 1, int(index)))
        self._current_slice = clamped
        self._update_slice_plane()
        self._update_slice_image()
        if update_slider:
            self._slider.blockSignals(True)
            self._slider.setValue(clamped)
            self._slider.blockSignals(False)
        if emit:
            self.slice_changed.emit(clamped)
            self.camera_changed.emit({"slice": clamped})

    def _build_scene(self, cmap: str) -> None:
        for child in list(self._view.scene.children):
            if child is not self._view.camera:
                child.parent = None
        self._volume_visual = None
        self._slice_highlight_visual = None
        self._slice_image = None

        if self._norm_volume is None:
            return

        colormap_name = cmap if isinstance(cmap, str) else "viridis"
        try:
            depth, height, width = self._norm_volume.shape
            self._volume_visual = scene.visuals.Volume(
                self._norm_volume,
                parent=self._view.scene,
                method="mip",
                cmap=colormap_name,
            )
            plane_z = max(0, min(depth - 1, int(self._current_slice)))
            self._slice_highlight_visual = scene.visuals.Volume(
                self._norm_volume,
                parent=self._view.scene,
                method="mip",
                cmap=colormap_name,
                raycasting_mode="plane",
                plane_normal=(1, 0, 0),
                plane_thickness=1.0,
                plane_position=(float(plane_z), height / 2.0, width / 2.0),
            )
            self._view.camera.set_range()
        except Exception as exc:
            self._status.setText(f"Volume non rendu ({exc})")
            self._volume_visual = None
            self._slice_highlight_visual = None

        self._add_slice_image()

    def update_volume(self) -> None:
        self.volume_needs_update.emit()

    def update_overlay(self) -> None:
        self.overlay_updated.emit()

    def _emit_slice_change(self, value: int) -> None:
        self.set_slice_index(value, update_slider=False, emit=True)

    def _add_slice_image(self) -> None:
        if self._volume is None:
            return
        data = self._get_slice_data(self._current_slice)
        if data is None:
            return
        self._slice_image = scene.visuals.Image(
            data,
            parent=self._view.scene,
            cmap="gray",
            method="auto",
        )
        self._slice_image.order = 9
        self._slice_image.interpolation = "nearest"
        self._update_slice_image()

    def _update_slice_plane(self) -> None:
        if self._slice_highlight_visual is None or self._volume is None:
            return
        depth, height, width = self._volume.shape
        clamped = max(0, min(depth - 1, int(self._current_slice)))
        self._slice_highlight_visual.plane_position = (
            float(clamped),
            height / 2.0,
            width / 2.0,
        )

    def _update_slice_image(self) -> None:
        if self._slice_image is None:
            return
        data = self._get_slice_data(self._current_slice)
        if data is not None:
            self._slice_image.set_data(data)
        self._slice_image.transform = STTransform(
            translate=(0.0, 0.0, float(self._current_slice))
        )

    def _get_slice_data(self, index: int) -> Optional[np.ndarray]:
        if self._norm_volume is None:
            return None
        clamped = max(0, min(self._norm_volume.shape[0] - 1, int(index)))
        return self._norm_volume[clamped]
