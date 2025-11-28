"""
Volume view for 3D NDE data that keeps the slice orientation consistent with
the 2D endview. The loader is expected to deliver an already oriented volume
of shape ``(num_slices, height, width)``; the view does not permute axes but
applies a display-only flip in the XY plane so that the 3D rendering matches
the 2D endview.

This class uses VisPy to render a 3D volume with a movable slice image. A
horizontal slider lets users navigate through the slices, and the camera is
recentred on the active slice for coherent navigation.

The methods exposed here mirror the original ``VolumeView`` interface,
allowing this drop-in replacement to work with existing controllers.
"""


from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QLabel, QSlider, QVBoxLayout, QWidget
from vispy import scene
from vispy.color import BaseColormap
from vispy.visuals.transforms import STTransform


class _TranslucentMask(BaseColormap):
    """Simple translucent fire-like colormap for overlay volume."""

    glsl_map = """
    vec4 translucent_mask(float t) {
        float a = clamp(t, 0.0, 1.0);
        return vec4(pow(a, 0.5), a, a * a, max(0.0, a * 1.05 - 0.05));
    }
    """


class VolumeView(QFrame):
    """Displays a basic 3D volume using VisPy, oriented consistently with the endview.

    The first dimension of the volume corresponds to the slice index, the
    second to height, and the third to width. Axes are assumed already
    oriented by the loader; the view only applies a display flip in XY so
    the 3D render matches the 2D endview orientation.
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
        self._slice_overlay: Optional[scene.visuals.Image] = None
        self._overlay_volume_visual: Optional[scene.visuals.Volume] = None
        # Store the axis order used to orient the current volume
        self._axis_order: Optional[Sequence[str]] = None
        self._overlay_volume: Optional[np.ndarray] = None
        self._overlay_alpha_volume: Optional[np.ndarray] = None
        self._overlay_colormap = _TranslucentMask()

    def set_volume(
        self,
        volume: np.ndarray,
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
        self._build_scene()
        # Set initial slice
        target_slice = slice_idx if slice_idx is not None else self._current_slice
        self.set_slice_index(target_slice, update_slider=True, emit=False)

    def set_overlay(self, overlay: Optional[np.ndarray]) -> None:
        """Assign an RGBA overlay volume shaped (num_slices, H, W, 4)."""
        if overlay is None:
            self._overlay_volume = None
            self._overlay_alpha_volume = None
            if self._slice_overlay is not None:
                self._slice_overlay.parent = None
                self._slice_overlay = None
            if self._overlay_volume_visual is not None:
                self._overlay_volume_visual.parent = None
                self._overlay_volume_visual = None
            return
        arr = np.asarray(overlay)
        if arr.ndim != 4 or arr.shape[0] == 0:
            return
        self._overlay_volume = arr
        self._overlay_alpha_volume = self._extract_overlay_alpha(arr)
        if self._overlay_volume_visual is None and self._view.scene is not None and self._volume is not None:
            self._add_overlay_volume()
        elif self._overlay_volume_visual is not None and self._overlay_alpha_volume is not None:
            self._overlay_volume_visual.set_data(self._overlay_alpha_volume)
            self._overlay_volume_visual.cmap = self._overlay_colormap
            self._overlay_volume_visual.set_gl_state(depth_test=False, blend=True)
            self._apply_visual_transform()
        # If scene already exists, add/update overlay visual
        if self._slice_overlay is None and self._view.scene is not None:
            self._add_slice_overlay()
        self._update_overlay_image()

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
    def _build_scene(self) -> None:
        """Initialise the VisPy visuals for the current normalised volume."""
        # Remove existing visuals
        for child in list(self._view.scene.children):
            if child is not self._view.camera:
                child.parent = None
        self._volume_visual = None
        self._slice_highlight_visual = None
        self._slice_image = None
        self._slice_overlay = None
        self._overlay_volume_visual = None
        # If no data, nothing to do
        if self._norm_volume is None:
            return
        # Determine dimensions: (depth, height, width)
        depth, height, width = self._norm_volume.shape[:3]
        # Create 3D volume (maximum intensity projection)
        self._volume_visual = scene.visuals.Volume(
            self._norm_volume,
            parent=self._view.scene,
            method="mip",
            cmap="gray",
        )

        # Configure camera ranges and centre
        self._configure_camera(depth, height, width)
        self._apply_visual_transform()
        # Add the initial 2D slice image overlay
        self._add_slice_image()
        self._add_overlay_volume()

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
        self._apply_visual_transform()
        self._update_slice_image()
        self._add_slice_overlay()

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
            height = self._volume.shape[1]
            width = self._volume.shape[2]
            self._slice_image.transform = STTransform(
                scale=(-1.0, -1.0, 1.0),
                translate=(
                    float(width),
                    float(height),
                    float(self._current_slice),
                ),
            )
        self._update_overlay_image()

    def _add_slice_overlay(self) -> None:
        """Add a colored overlay image for the current slice."""
        if self._overlay_volume is None or self._volume is None:
            return
        slice_rgba = self._get_overlay_slice(self._current_slice)
        if slice_rgba is None:
            return
        self._slice_overlay = scene.visuals.Image(
            slice_rgba,
            parent=self._view.scene,
            method="auto",
        )
        self._slice_overlay.order = 20
        self._slice_overlay.interpolation = "nearest"
        # Ensure the overlay draws on top of the volume without depth test
        self._slice_overlay.set_gl_state(depth_test=False, blend=True)
        self._slice_overlay.transform = self._overlay_transform()

    def _update_overlay_image(self) -> None:
        """Update overlay image data/transform for current slice."""
        if self._slice_overlay is None or self._overlay_volume is None:
            return
        slice_rgba = self._get_overlay_slice(self._current_slice)
        if slice_rgba is not None:
            self._slice_overlay.set_data(slice_rgba)
        self._slice_overlay.transform = self._overlay_transform()
        self._slice_overlay.set_gl_state(depth_test=False, blend=True)

    def _get_overlay_slice(self, index: int) -> Optional[np.ndarray]:
        if self._overlay_volume is None:
            return None
        if index < 0 or index >= self._overlay_volume.shape[0]:
            return None
        return self._overlay_volume[index]

    def _add_overlay_volume(self) -> None:
        """Add the overlay as a translucent 3D volume."""
        if self._overlay_alpha_volume is None or self._volume is None:
            return
        self._overlay_volume_visual = scene.visuals.Volume(
            self._overlay_alpha_volume,
            parent=self._view.scene,
            method="translucent",
            cmap=self._overlay_colormap,
            threshold=0.1,
        )
        self._overlay_volume_visual.set_gl_state(depth_test=False, blend=True)
        self._apply_visual_transform()

    def _extract_overlay_alpha(self, overlay_rgba: np.ndarray) -> Optional[np.ndarray]:
        """Return a float32 alpha volume in [0,1] from RGBA overlay."""
        if overlay_rgba is None or overlay_rgba.ndim != 4 or overlay_rgba.shape[0] == 0:
            return None
        alpha = overlay_rgba[..., 3].astype(np.float32, copy=False)
        if alpha.size == 0 or float(alpha.max()) <= 0.0:
            return None
        return alpha / 255.0

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
        # Centre initial: milieu du volume, avec z sur la slice courante
        center = (width / 2.0, height / 2.0, float(self._current_slice))
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
        # Focus : centre de la slice courante (x/y au centre, z = index de slice)
        z_focus = max(0, min(depth - 1, int(self._current_slice)))
        self._view.camera.center = (
            width / 2.0,
            height / 2.0,
            float(z_focus),
        )

    def _apply_visual_transform(self) -> None:
        """Apply a 180° flip in the XY plane to align 3D orientation with Endview."""
        if self._volume is None:
            return
        height = self._volume.shape[1]
        width = self._volume.shape[2]
        flip = STTransform(
            scale=(-1.0, -1.0, 1.0),
            translate=(float(width), float(height), 0.0),
        )
        if self._volume_visual is not None:
            self._volume_visual.transform = flip
        if self._overlay_volume_visual is not None:
            self._overlay_volume_visual.transform = flip
        if self._slice_image is not None:
            self._slice_image.transform = STTransform(
                scale=(-1.0, -1.0, 1.0),
                translate=(
                    float(width),
                    float(height),
                    float(self._current_slice),
                ),
            )
        if self._slice_overlay is not None:
            self._slice_overlay.transform = self._overlay_transform()

    def _overlay_transform(self) -> STTransform:
        """Return transform for overlay aligned with the base slice."""
        if self._volume is None:
            return STTransform()
        height = self._volume.shape[1]
        width = self._volume.shape[2]
        return STTransform(
            scale=(-1.0, -1.0, 1.0),
            translate=(
                float(width),
                float(height),
                float(self._current_slice),
            ),
        )
