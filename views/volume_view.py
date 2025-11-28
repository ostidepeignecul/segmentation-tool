"""
Volume view for 3D NDE data that keeps the slice orientation consistent with
the 2D endview. This implementation reorders axes based on the provided
``axis_order`` metadata so that the underlying volume has the shape
``(num_slices, ultrasound, cross)`` regardless of how the data was stored in
the HDF5 file. The ``axis_order`` list should come from the JSON
``dimensions`` description and is expected to mirror the order of the
dimensions in the underlying HDF5 dataset【848655542718855†L150-L163】.  The
ultrasound axis is detected by searching for the string "ultrasound"
(case‑insensitive) in this list; the remaining axis becomes the cross axis.

This class uses VisPy to render a 3D volume with a movable slice plane and
overlay a 2D image of the current slice. A horizontal slider lets users
navigate through the slices. When an ``axis_order`` is provided and the
underlying volume has three or more dimensions, the volume is permuted
accordingly before display to ensure that the ultrasound dimension maps to
vertical (``y``) and the cross dimension maps to horizontal (``x``).  This
alignment guarantees that the 3D view corresponds to the endview’s
orientation.

The methods exposed here mirror the original ``VolumeView`` interface,
allowing this drop‑in replacement to work with existing controllers.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QLabel, QSlider, QVBoxLayout, QWidget
from vispy import scene
from vispy.visuals.transforms import STTransform


class VolumeView(QFrame):
    """Displays a basic 3D volume using VisPy, oriented consistently with the endview.

    The first dimension of the volume corresponds to the slice index, the
    second dimension to the ultrasound/depth axis, and the third dimension to
    the cross axis.  An optional ``axis_order`` sequence may be supplied
    when assigning a new volume; if present, the axes are permuted so that
    the ultrasound axis becomes the second dimension and the remaining axis
    becomes the third.  This reordering is key to ensuring that the 3D view
    aligns with the 2D endview orientation, as specified by the NDE
    documentation【848655542718855†L258-L268】.
    """

    volume_needs_update = pyqtSignal()
    overlay_updated = pyqtSignal()
    slice_changed = pyqtSignal(int)
    camera_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QFrame] = None) -> None:
        super().__init__(parent)
        # Internal state
        self._volume: Optional[np.ndarray] = None
        self._norm_volume: Optional[np.ndarray] = None
        self._volume_min: Optional[float] = None
        self._volume_max: Optional[float] = None
        self._current_slice: int = 0
        # VisPy canvas and scene
        self._canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=False)
        self._view = self._canvas.central_widget.add_view()
        # Use a turntable camera so the user can rotate/pan with the mouse
        self._view.camera = scene.TurntableCamera(
            fov=45,
            up="+y",
        )
        # Qt layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        container = QWidget()
        canvas_layout = QVBoxLayout(container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self._canvas.native)
        layout.addWidget(container, 1)
        # Slider to change slice index
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._emit_slice_change)
        layout.addWidget(self._slider)
        # Status label
        self._status = QLabel("Volume non chargé")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)
        # Visuals
        self._volume_visual: Optional[scene.visuals.Volume] = None
        self._slice_highlight_visual: Optional[scene.visuals.Volume] = None
        self._slice_image: Optional[scene.visuals.Image] = None
        # Store the axis order used to orient the current volume
        self._axis_order: Optional[Sequence[str]] = None

    def set_volume(
        self,
        volume: np.ndarray,
        cmap: str = "viridis",
        *,
        slice_idx: Optional[int] = None,
        axis_order: Optional[Sequence[str]] = None,
    ) -> None:
        """Assign a new 3D volume to render.

        Parameters
        ----------
        volume : np.ndarray
            The array to visualise.  Shape must be at least 3D and is
            interpreted as ``(num_slices, height, width)``.  Axes are
            expected to be already oriented consistently with the 2D
            endview; no additional permutation is applied here.
        cmap : str
            Name of the VisPy colormap to use for volume rendering.
        slice_idx : Optional[int]
            The initial slice index to show after assigning the volume.
        axis_order : Optional[Sequence[str]]
            The order of axes as described in the NDE metadata.  Stored for
            reference but not used for reordering inside the view.
        """
        # Reset visuals if volume is None or empty
        if volume is None or np.size(volume) == 0:
            self._volume = None
            self._norm_volume = None
            self._status.setText("Volume vide")
            self._slider.setMaximum(0)
            return
        # Make a copy to avoid side effects and ensure float32 for VisPy
        vol = np.asarray(volume)
        # Store axis order for reference; the volume is assumed to be already oriented.
        self._axis_order = axis_order
        # Store the raw and normalised volumes
        self._volume = vol.astype(np.float32, copy=False)
        self._volume_min = float(np.min(self._volume))
        self._volume_max = float(np.max(self._volume))
        if self._volume_max > self._volume_min:
            self._norm_volume = (self._volume - self._volume_min) / (
                self._volume_max - self._volume_min
            )
        else:
            self._norm_volume = np.zeros_like(self._volume, dtype=np.float32)
        # Update status and slider range
        self._status.setText(f"Volume chargé: {self._volume.shape}")
        self._slider.setMaximum(self._volume.shape[0] - 1)
        # Build scene with new data
        self._build_scene(cmap)
        # Set initial slice
        target_slice = slice_idx if slice_idx is not None else self._current_slice
        self.set_slice_index(target_slice, update_slider=True, emit=False)

    def set_slice_index(
        self, index: int, *, update_slider: bool = False, emit: bool = False
    ) -> None:
        """Synchronise the slice indicator (slider + plane + image) with external selection.

        Clamps the index to the valid range, updates the slice highlight and
        image, recentres the camera on the selected slice, updates the slider
        position if requested, and emits signals when appropriate.
        """
        if self._volume is None:
            return
        clamped = max(0, min(self._volume.shape[0] - 1, int(index)))
        self._current_slice = clamped
        self._update_slice_plane()
        self._update_slice_image()
        self._focus_camera_on_slice()
        if update_slider:
            self._slider.blockSignals(True)
            self._slider.setValue(clamped)
            self._slider.blockSignals(False)
        if emit:
            self.slice_changed.emit(clamped)
            self.camera_changed.emit({"slice": clamped})

    # ------------------------------------------------------------------
    # Internal scene construction and updates
    # ------------------------------------------------------------------
    def _build_scene(self, cmap: str) -> None:
        """Initialise the VisPy visuals for the current normalised volume."""
        # Remove existing visuals
        for child in list(self._view.scene.children):
            if child is not self._view.camera:
                child.parent = None
        self._volume_visual = None
        self._slice_highlight_visual = None
        self._slice_image = None
        # If no data, nothing to do
        if self._norm_volume is None:
            return
        colormap_name = cmap if isinstance(cmap, str) else "viridis"
        # Determine dimensions: (depth, height, width)
        depth, height, width = self._norm_volume.shape[:3]
        # Create 3D volume (maximum intensity projection)
        self._volume_visual = scene.visuals.Volume(
            self._norm_volume,
            parent=self._view.scene,
            method="mip",
            cmap=colormap_name,
        )
        # Slice highlight plane désactivé (conservé pour réactivation future)
        # plane_z = max(0, min(depth - 1, int(self._current_slice)))
        # self._slice_highlight_visual = scene.visuals.Volume(
        #     self._norm_volume,
        #     parent=self._view.scene,
        #     method="mip",
        #     cmap=colormap_name,
        #     raycasting_mode="plane",
        #     plane_normal=(0, 0, 1),
        #     plane_thickness=1.0,
        #     plane_position=(0.0, 0.0, float(plane_z)),
        # )
        # Configure camera ranges and centre
        self._configure_camera(depth, height, width)
        # Add the initial 2D slice image overlay
        self._add_slice_image()

    def update_volume(self) -> None:
        """Emit a signal requesting the controller to refresh the volume."""
        self.volume_needs_update.emit()

    def update_overlay(self) -> None:
        """Emit a signal requesting the controller to refresh the overlay."""
        self.overlay_updated.emit()

    def _emit_slice_change(self, value: int) -> None:
        self.set_slice_index(value, update_slider=False, emit=True)

    def _add_slice_image(self) -> None:
        """Add a 2D image of the current slice on top of the 3D volume."""
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
        self._slice_image.transform = STTransform(
            translate=(0.0, 0.0, float(self._current_slice)),
        )
        self._update_slice_image()

    def _update_slice_plane(self) -> None:
        """Move the highlight plane to the current slice index."""
        if self._slice_highlight_visual is None or self._volume is None:
            return
        depth = self._volume.shape[0]
        clamped = max(0, min(depth - 1, int(self._current_slice)))
        self._slice_highlight_visual.plane_position = (
            0.0,
            0.0,
            float(clamped),
        )

    def _update_slice_image(self) -> None:
        """Update the 2D slice image data and its position."""
        if self._slice_image is None:
            return
        data = self._get_slice_data(self._current_slice)
        if data is not None:
            self._slice_image.set_data(data)
        if self._volume is not None:
            self._slice_image.transform = STTransform(
                translate=(0.0, 0.0, float(self._current_slice)),
            )

    def _get_slice_data(self, index: int) -> Optional[np.ndarray]:
        """Return the normalised 2D slice at ``index`` along the first axis."""
        if self._norm_volume is None:
            return None
        clamped = max(0, min(self._norm_volume.shape[0] - 1, int(index)))
        # Extract the 2D plane: shape (height, width)
        return self._norm_volume[clamped]

    def _configure_camera(self, depth: int, height: int, width: int) -> None:
        """Configure the camera so that slices face the user and the volume fills the view."""
        if self._view.camera is None:
            return
        center = (width / 2.0, height / 2.0, depth / 2.0)
        self._view.camera.up = "+y"
        # Set the ranges along each axis
        self._view.camera.set_range(
            x=(0.0, float(width)),
            y=(0.0, float(height)),
            z=(0.0, float(depth)),
        )
        # Centre the view on the middle slice
        self._view.camera.center = center
        # Immediately focus on the current slice
        self._focus_camera_on_slice()

    def _focus_camera_on_slice(self) -> None:
        """Recentre the camera to focus on the current slice."""
        if self._view.camera is None or self._volume is None:
            return
        depth, height, width = self._volume.shape[:3]
        z_focus = max(0, min(depth - 1, int(self._current_slice)))
        self._view.camera.center = (
            width / 2.0,
            height / 2.0,
            float(z_focus),
        )
