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

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QFrame, QLabel, QSlider, QVBoxLayout, QWidget
from vispy import scene
from vispy.scene.cameras.perspective import PerspectiveCamera
from vispy.color import BaseColormap, Colormap
from vispy.util import keys
from vispy.visuals.transforms import STTransform

from models.overlay_data import OverlayData


class _TranslucentMask(BaseColormap):
    """Simple translucent fire-like colormap for overlay volume."""

    glsl_map = """
    vec4 translucent_mask(float t) {
        float a = clamp(t, 0.0, 1.0);
        return vec4(pow(a, 0.5), a, a * a, max(0.0, a * 1.05 - 0.05));
    }
    """


class _AnchorMoveTurntableCamera(scene.TurntableCamera):
    """Turntable camera where right-drag moves the camera anchor instead of zoom."""

    def viewbox_mouse_event(self, event):
        if event.handled or not self.interactive:
            return

        PerspectiveCamera.viewbox_mouse_event(self, event)

        if event.type == "mouse_release":
            self._event_value = None
            return
        if event.type == "mouse_press":
            event.handled = True
            return
        if event.type != "mouse_move" or event.press_event is None:
            return
        if 1 in event.buttons and 2 in event.buttons:
            return

        modifiers = event.mouse_event.modifiers
        p1 = event.mouse_event.press_event.pos
        p2 = event.mouse_event.pos
        d = p2 - p1

        if 1 in event.buttons and not modifiers:
            # Left drag keeps the default orbit behavior.
            self._update_rotation(event)
            return

        if 2 in event.buttons and not modifiers:
            # Right drag now repositions the pivot (camera center) in XY.
            norm = np.mean(self._viewbox.size)
            if self._event_value is None or len(self._event_value) == 2:
                self._event_value = self.center
            dist = (p1 - p2) / norm * self._scale_factor
            dist[1] *= -1
            dx, dy, dz = self._dist_to_trans(dist)
            ff = self._flip_factors
            up, forward, right = self._get_dim_vectors()
            dx, dy, dz = right * dx + forward * dy + up * dz
            dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
            c = self._event_value
            self.center = c[0] + dx, c[1] + dy, c[2]
            return

        if 1 in event.buttons and keys.SHIFT in modifiers:
            norm = np.mean(self._viewbox.size)
            if self._event_value is None or len(self._event_value) == 2:
                self._event_value = self.center
            dist = (p1 - p2) / norm * self._scale_factor
            dist[1] *= -1
            dx, dy, dz = self._dist_to_trans(dist)
            ff = self._flip_factors
            up, forward, right = self._get_dim_vectors()
            dx, dy, dz = right * dx + forward * dy + up * dz
            dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
            c = self._event_value
            self.center = c[0] + dx, c[1] + dy, c[2] + dz
            return

        if 2 in event.buttons and keys.SHIFT in modifiers:
            if self._event_value is None:
                self._event_value = self._fov
            fov = self._event_value - d[1] / 5.0
            self.fov = min(180.0, max(0.0, fov))


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
    secondary_slice_changed = pyqtSignal(int)
    camera_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QFrame] = None) -> None:
        super().__init__(parent)
        # Internal state
        self._volume: Optional[np.ndarray] = None
        self._norm_volume: Optional[np.ndarray] = None
        self._volume_min: Optional[float] = None
        self._volume_max: Optional[float] = None
        self._current_slice: int = 0
        self._secondary_slice: int = 0
        self._recenter_on_slice_change: bool = False
        # VisPy canvas and scene
        self._canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=False)
        self._view = self._canvas.central_widget.add_view()
        # Use a turntable camera so the user can rotate/pan with the mouse
        self._view.camera = _AnchorMoveTurntableCamera(
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
        self._canvas.native.installEventFilter(self)
        layout.addWidget(container, 1)
        # Slider to change slice index
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._emit_slice_change)
        layout.addWidget(self._slider)
        # Slider to change the secondary orthogonal slice index
        self._secondary_slider = QSlider(Qt.Orientation.Horizontal)
        self._secondary_slider.setMinimum(0)
        self._secondary_slider.setMaximum(0)
        self._secondary_slider.valueChanged.connect(self._emit_secondary_slice_change)
        layout.addWidget(self._secondary_slider)
        # Status label
        self._status = QLabel("Volume non chargé")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)
        # Visuals
        self._volume_visual: Optional[scene.visuals.Volume] = None
        self._slice_highlight_visual: Optional[scene.visuals.Volume] = None
        self._slice_image: Optional[scene.visuals.Image] = None
        self._primary_slice_line: Optional[scene.visuals.Line] = None
        self._secondary_slice_plane: Optional[scene.visuals.Mesh] = None
        self._secondary_slice_line: Optional[scene.visuals.Line] = None
        self._slice_overlay: Optional[scene.visuals.Image] = None
        self._overlay_volume_visual: Optional[scene.visuals.Volume] = None
        self._base_colormap_name: str = "Gris"
        self._base_colormap: Optional[BaseColormap | str] = "gray"
        # Store the axis order used to orient the current volume
        self._axis_order: Optional[Sequence[str]] = None
        self._overlay_mask: Optional[np.ndarray] = None
        self._overlay_alpha_volume: Optional[np.ndarray] = None
        self._mask_volume: Optional[np.ndarray] = None # Single uint8 mask
        self._overlay_palette: Dict[int, Tuple[int, int, int, int]] = {}
        self._overlay_colormap: Optional[Colormap] = None
        self._overlay_opacity: float = 1.0
        self._label_visuals: Dict[int, scene.visuals.Volume] = {}
        self._uploaded_volumes: Dict[int, np.ndarray] = {}
        self._visible_labels: Optional[set[int]] = None
        # Debounce overlay volume uploads to avoid GPU spam on rapid toggles
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.timeout.connect(self._apply_pending_overlay_volume)
        self._pending_overlay_apply = False
        self._pending_overlay_labels: Optional[set[int]] = None
        self._display_size: Optional[Tuple[int, int]] = None
        # Rebuild scene after dock/undock transitions to avoid stale GL state.
        self._dock_rebuild_timer = QTimer(self)
        self._dock_rebuild_timer.setSingleShot(True)
        self._dock_rebuild_timer.timeout.connect(self._rebuild_scene_after_dock_change)

    def eventFilter(self, watched, event) -> bool:
        """Recentre the camera on double-click inside the 3D canvas."""
        if (
            watched is self._canvas.native
            and event.type() == QEvent.Type.MouseButtonDblClick
            and hasattr(event, "button")
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._focus_camera_on_slice()
            return True
        return super().eventFilter(watched, event)

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
            self._secondary_slider.setMaximum(0)
            self._clear_overlay_visuals()
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
        self._secondary_slider.setMaximum(self._volume.shape[2] - 1)
        # Clear any existing overlay visuals before rebuilding the scene
        self._clear_overlay_visuals()
        # Build scene with new data
        self._build_scene()
        # Set initial slice
        target_slice = slice_idx if slice_idx is not None else self._current_slice
        self.set_slice_index(target_slice, update_slider=True, emit=False)
        self.set_secondary_slice_index(self._secondary_slice, update_slider=True, emit=False)

    def set_base_colormap(self, name: str, lut: Optional[np.ndarray]) -> None:
        """Set the base volume colormap (lut expected shape (256,3) floats 0-1)."""
        self._base_colormap_name = str(name)
        if lut is not None and lut.shape == (256, 3):
            self._base_colormap = Colormap(np.asarray(lut, dtype=np.float32))
        else:
            self._base_colormap = "gray"
            self._base_colormap_name = "Gris"
        if self._volume_visual is not None:
            self._volume_visual.cmap = self._base_colormap
        # self._slice_image uses direct RGBA, so no colormap needed.
        pass

    def get_display_size(self) -> Tuple[int, int]:
        """Return requested display size or current viewport size."""
        if self._display_size is not None:
            return self._display_size
        size = self._canvas.native.size()
        return size.width(), size.height()

    def set_display_size(self, width: int, height: int) -> None:
        """Apply visual deformation in XY without changing underlying volume data."""
        width = max(1, int(width))
        height = max(1, int(height))
        previous_scale = self._display_scale_factors()
        self._display_size = (width, height)
        self._on_display_scale_changed(previous_scale=previous_scale)

    def reset_display_size(self) -> None:
        """Reset 3D display deformation to default proportions."""
        previous_scale = self._display_scale_factors()
        self._display_size = None
        self._on_display_scale_changed(previous_scale=previous_scale)

    def notify_dock_topology_changed(self) -> None:
        """Schedule a scene rebuild after docking/floating transitions."""
        self._dock_rebuild_timer.start(75)

    def _rebuild_scene_after_dock_change(self) -> None:
        """Recreate visuals after reparent/context changes triggered by docking."""
        if self._volume is None or self._norm_volume is None:
            return
        current_slice = int(self._current_slice)
        current_secondary_slice = int(self._secondary_slice)
        self._overlay_timer.stop()
        self._pending_overlay_apply = False
        self._pending_overlay_labels = None
        self._build_scene()
        self.set_slice_index(current_slice, update_slider=True, emit=False)
        self.set_secondary_slice_index(
            current_secondary_slice,
            update_slider=True,
            emit=False,
        )
        try:
            self._canvas.update()
        except Exception:
            pass

    def set_overlay(
        self,
        overlay: Optional[OverlayData],
        *,
        visible_labels: Optional[set[int]] = None,
        defer_3d: bool = False,
        changed_slice: Optional[int] = None,
        changed_labels: Optional[set[int]] = None,
    ) -> None:
        """Assign overlay mask and update visuals."""
        if overlay is None:
            self._clear_overlay_visuals()
            return

        self._visible_labels = set(visible_labels) if visible_labels is not None else None
        self._overlay_palette = dict(overlay.palette)
        
        # Check consistency
        if overlay.mask_volume is None:
            self._clear_overlay_visuals()
            return
            
        if self._volume is not None:
             if overlay.mask_volume.shape != self._volume.shape:
                 # If shapes mismatch, ignore overlay
                 self._clear_overlay_visuals()
                 return

        new_mask = overlay.mask_volume
        mask_changed = (
            self._mask_volume is None 
            or not np.shares_memory(new_mask, self._mask_volume)
        )
        self._mask_volume = new_mask

        # If only visibility/palette changed, we just update the colormap.
        # If mask data changed (new object), we need to set_data on visual.
        
        if defer_3d:
            # For defer_3d (often used during painting), we update 2D visuals immediately
            if self._slice_overlay is None and self._view.scene is not None:
                self._add_slice_overlay()
            self._update_overlay_image()
            
            # Defer 3D update
            self._schedule_overlay_volume_update(labels=changed_labels)
            return

        # Immediate update
        self._apply_overlay_volume_now()
        
        # Update 2D stuff
        if self._slice_overlay is None and self._view.scene is not None:
            self._add_slice_overlay()
        self._update_overlay_image()

    def set_overlay_opacity(self, opacity: float) -> None:
        """Set global overlay opacity (0.0 - 1.0)."""
        try:
            value = float(opacity)
        except (TypeError, ValueError):
            value = 1.0
        self._overlay_opacity = max(0.0, min(1.0, value))
        if self._overlay_volume_visual is not None:
            self._update_overlay_colormap()
            self._overlay_volume_visual.cmap = self._overlay_colormap
        self._update_overlay_image()

    def _clear_overlay_visuals(self) -> None:
        """Remove overlay visuals and reset related state."""
        self._overlay_timer.stop()
        self._mask_volume = None
        self._overlay_palette = {}
        
        # Clear the single volume visual
        if self._overlay_volume_visual is not None:
            self._overlay_volume_visual.parent = None
            self._overlay_volume_visual = None
            
        if self._slice_overlay is not None:
            self._slice_overlay.parent = None
            self._slice_overlay = None
            
        self._pending_overlay_apply = False
        self._pending_overlay_labels = None

    def _schedule_overlay_volume_update(self, *, labels: Optional[set[int]] = None) -> None:
        """Coalesce overlay volume uploads to reduce GPU churn."""
        self._pending_overlay_apply = True
        self._pending_overlay_labels = set(labels) if labels is not None else None
        # Short delay to batch multiple toggles; 120ms keeps UI responsive
        self._overlay_timer.start(120)

    def _apply_pending_overlay_volume(self) -> None:
        if not self._pending_overlay_apply:
            return
        self._pending_overlay_apply = False
        pending_labels = self._pending_overlay_labels
        self._pending_overlay_labels = None
        self._apply_overlay_volume_now(labels_to_push=pending_labels)

    def _apply_overlay_volume_now(self, *, labels_to_push: Optional[set[int]] = None) -> None:
        # labels_to_push is ignored now; we update the whole colormap or volume.
        
        if self._mask_volume is None:
            if self._overlay_volume_visual is not None:
                self._overlay_volume_visual.parent = None
                self._overlay_volume_visual = None
            return

        if self._view.scene is None:
            return

        # 1. Update Colormap (Lightweight)
        self._update_overlay_colormap()

        if self._overlay_volume_visual is None:
            # Create fresh
            # We use uint8 directly to save memory (float32 is too heavy).
            # VisPy handles uint8 by interpreting them as 0..255 if we set clim=(0, 255).
            self._overlay_volume_visual = scene.visuals.Volume(
                self._mask_volume,
                parent=self._view.scene,
                method="translucent", # or 'iso'
                cmap=self._overlay_colormap,
                clim=(0, 255),
                interpolation="nearest", # CRITICAL: No interpolation between labels
            )
            self._overlay_volume_visual.order = 10  # Ensure it draws AFTER the base volume
            self._overlay_volume_visual.set_gl_state(depth_test=True, blend=True)
            # Apply transform
            self._apply_visual_transform()
        else:
            # Update data if needed (if pointer changed)
            self._overlay_volume_visual.set_data(self._mask_volume)
            self._overlay_volume_visual.cmap = self._overlay_colormap

        # Ensure correct GL state
        self._overlay_volume_visual.visible = True

    def _update_overlay_colormap(self) -> None:
        """Build a 256-entry colormap for the current palette/visibility."""
        # 256 colors for uint8
        colors = np.zeros((256, 4), dtype=np.float32)
        
        labels_to_draw = (
            self._visible_labels if self._visible_labels is not None 
            else self._overlay_palette.keys()
        )

        for label in labels_to_draw:
            if not (0 < label < 256):
                 continue
            c_bgra = self._overlay_palette.get(label)
            if c_bgra:
                b, g, r, a = c_bgra
                # Normalize 0..1
                alpha = (a / 255.0) * self._overlay_opacity
                colors[label] = [r/255.0, g/255.0, b/255.0, alpha]

        # Use 'from_array' directly? No, Colormap takes list of colors and controls.
        # But for exact matching we want controls at 0, 1/255, etc.
        # VisPy Colormap accepts 'colors' generic list.
        # A simple way to ensure 1-to-1 mapping for uint8 is to provide all 256 colors
        # and rely on linear interpolation between centroids which are same color?
        # Or better: just pass the array.
        self._overlay_colormap = Colormap(colors, controls=np.linspace(0, 1, 256))

    @staticmethod
    def _label_colormap(color: Optional[Tuple[int, int, int, int]]) -> Colormap:
        """Return a simple 2-stop colormap for a given BGRA color."""
        if color is None:
            color = (255, 0, 255, 160)
        b, g, r, a = color
        return Colormap(
            colors=[
                (0.0, 0.0, 0.0, 0.0),
                (r / 255.0, g / 255.0, b / 255.0, a / 255.0),
            ]
        )

    def set_slice_index(
        self, index: int, *, update_slider: bool = False, emit: bool = False
    ) -> None:
        """Synchronise the slice indicator (slider + plane + image) with external selection.

        Clamps the index to the valid range, updates the slice highlight and
        image, optionally recentres the camera, updates the slider position if
        requested, and emits signals when appropriate.
        """
        if self._volume is None:
            return
        if self._pending_overlay_apply:
            # Ensure pending overlay upload happens before moving slices to keep 3D in sync
            self._apply_pending_overlay_volume()
        clamped = max(0, min(self._volume.shape[0] - 1, int(index)))
        self._current_slice = clamped
        self._update_slice_plane()
        self._update_slice_image()
        self._update_primary_slice_outline()
        if self._recenter_on_slice_change:
            self._focus_camera_on_slice()
        if update_slider:
            self._slider.blockSignals(True)
            self._slider.setValue(clamped)
            self._slider.blockSignals(False)
        if emit:
            self.slice_changed.emit(clamped)
            self.camera_changed.emit({"slice": clamped})

    def set_secondary_slice_index(
        self, index: int, *, update_slider: bool = False, emit: bool = False
    ) -> None:
        """Synchronise the orthogonal slicing plane along axis X."""
        if self._volume is None:
            return
        clamped = max(0, min(self._volume.shape[2] - 1, int(index)))
        self._secondary_slice = clamped
        self._update_secondary_slice_plane()
        if update_slider:
            self._secondary_slider.blockSignals(True)
            self._secondary_slider.setValue(clamped)
            self._secondary_slider.blockSignals(False)
        if emit:
            self.secondary_slice_changed.emit(clamped)

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
        self._primary_slice_line = None
        self._secondary_slice_plane = None
        self._secondary_slice_line = None
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
            cmap=self._base_colormap,
        )

        # Configure camera ranges and centre
        self._configure_camera(depth, height, width)
        self._apply_visual_transform()
        # Add the initial 2D slice image overlay
        self._add_slice_image()
        self._add_primary_slice_line()
        self._add_secondary_slice_plane()
        self._add_secondary_slice_line()
        self._add_overlay_volume()

    def update_volume(self) -> None:
        """Emit a signal requesting the controller to refresh the volume."""
        self.volume_needs_update.emit()

    def update_overlay(self) -> None:
        """Emit a signal requesting the controller to refresh the overlay."""
        self.overlay_updated.emit()

    def _emit_slice_change(self, value: int) -> None:
        self.set_slice_index(value, update_slider=False, emit=True)

    def _emit_secondary_slice_change(self, value: int) -> None:
        self.set_secondary_slice_index(value, update_slider=False, emit=True)

    def _add_slice_image(self) -> None:
        """Add a static translucent black plane to indicate the current slice."""
        if self._volume is None:
            return
        # Alpha channel (0-255): Lower = More transparent
        alpha = 100 
        data = np.array([[[0, 0, 0, alpha]]], dtype=np.uint8)
        self._slice_image = scene.visuals.Image(
            data,
            parent=self._view.scene,
            method="auto",
        )
        self._slice_image.order = 11
        self._slice_image.interpolation = "linear"
        # Disable depth testing to force visibility over the overlay
        self._slice_image.set_gl_state(depth_test=False, blend=True)
        self._apply_visual_transform()
        self._update_slice_image()
        self._add_slice_overlay()

    def _add_primary_slice_line(self) -> None:
        """Add a green frame around the active primary slice."""
        if self._volume is None:
            return
        self._primary_slice_line = scene.visuals.Line(
            np.zeros((5, 3), dtype=np.float32),
            color=(0.1, 1.0, 0.1, 0.95),
            width=2.0,
            parent=self._view.scene,
        )
        self._primary_slice_line.order = 13
        self._primary_slice_line.set_gl_state(depth_test=False, blend=True)
        self._apply_visual_transform()
        self._update_primary_slice_outline()

    def _update_primary_slice_outline(self) -> None:
        """Move the primary green frame to the selected Z slice."""
        if self._primary_slice_line is None or self._volume is None:
            return
        depth, height, width = self._volume.shape[:3]
        z_pos = max(0, min(depth - 1, int(self._current_slice)))
        points = np.array(
            [
                [0.0, 0.0, float(z_pos)],
                [float(width), 0.0, float(z_pos)],
                [float(width), float(height), float(z_pos)],
                [0.0, float(height), float(z_pos)],
                [0.0, 0.0, float(z_pos)],
            ],
            dtype=np.float32,
        )
        self._primary_slice_line.set_data(pos=points)

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
        """Update the position of the slice indicator (static black plane)."""
        if self._slice_image is None or self._volume is None:
            return
        # No data update needed (it's a constant 1x1 pixel)
        # Just update transform to place it at the correct Z
        height = self._volume.shape[1]
        width = self._volume.shape[2]
        scale_x, scale_y = self._display_scale_factors()
        width_scaled = float(width) * float(scale_x)
        height_scaled = float(height) * float(scale_y)
        # Scale the 1x1 pixel to cover the full width/height
        # scale=(-width, -height) because of the existing flip logic
        self._slice_image.transform = STTransform(
            scale=(-width_scaled, -height_scaled, 1.0),
            translate=(
                width_scaled,
                height_scaled,
                float(self._current_slice),
            ),
        )
        self._update_overlay_image()

    def _add_secondary_slice_plane(self) -> None:
        """Add a translucent wall for the current secondary orthogonal slice."""
        if self._volume is None:
            return
        self._secondary_slice_plane = scene.visuals.Mesh(
            vertices=np.zeros((4, 3), dtype=np.float32),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
            color=(0.0, 0.0, 0.0, 100.0 / 255.0),
            parent=self._view.scene,
        )
        self._secondary_slice_plane.order = 11
        self._secondary_slice_plane.set_gl_state(depth_test=False, blend=True)
        self._apply_visual_transform()
        self._update_secondary_slice_plane()

    def _add_secondary_slice_line(self) -> None:
        """Add an orthogonal rectangle showing the secondary slicing plane."""
        if self._volume is None:
            return
        self._secondary_slice_line = scene.visuals.Line(
            np.zeros((5, 3), dtype=np.float32),
            color=(1.0, 0.85, 0.1, 0.95),
            width=2.0,
            parent=self._view.scene,
        )
        self._secondary_slice_line.order = 12
        self._secondary_slice_line.set_gl_state(depth_test=False, blend=True)
        self._apply_visual_transform()
        self._update_secondary_slice_plane()

    def _update_secondary_slice_plane(self) -> None:
        """Move secondary translucent plane and yellow frame along the X axis."""
        if self._volume is None:
            return
        depth, height, width = self._volume.shape[:3]
        x_pos = max(0, min(width - 1, int(self._secondary_slice)))
        z_max = max(0, depth - 1)
        plane_vertices = np.array(
            [
                [float(x_pos), 0.0, 0.0],
                [float(x_pos), float(height), 0.0],
                [float(x_pos), float(height), float(z_max)],
                [float(x_pos), 0.0, float(z_max)],
            ],
            dtype=np.float32,
        )
        if self._secondary_slice_plane is not None:
            self._secondary_slice_plane.set_data(
                vertices=plane_vertices,
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
                color=(0.0, 0.0, 0.0, 100.0 / 255.0),
            )
        points = np.array(
            [
                [float(x_pos), 0.0, 0.0],
                [float(x_pos), float(height), 0.0],
                [float(x_pos), float(height), float(z_max)],
                [float(x_pos), 0.0, float(z_max)],
                [float(x_pos), 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        if self._secondary_slice_line is not None:
            self._secondary_slice_line.set_data(pos=points)

    def _add_slice_overlay(self) -> None:
        """Add a colored overlay image for the current slice."""
        if self._mask_volume is None or self._volume is None:
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
        if self._slice_overlay is None or self._mask_volume is None:
            return
        slice_rgba = self._get_overlay_slice(self._current_slice)
        if slice_rgba is not None:
            self._slice_overlay.set_data(slice_rgba)
        self._slice_overlay.transform = self._overlay_transform()
        self._slice_overlay.set_gl_state(depth_test=False, blend=True)

    def _get_overlay_slice(self, index: int) -> Optional[np.ndarray]:
        if self._mask_volume is None:
            return None
        if self._volume is None:
            return None
        depth = self._volume.shape[0]
        if index < 0 or index >= depth:
            return None
        height = self._volume.shape[1]
        width = self._volume.shape[2]

        # 1. Extract slice from mask (H, W)
        try:
             slice_indices = self._mask_volume[index]
        except IndexError:
             return None

        # 2. Build LUT (could be cached, but fast enough)
        lut = np.zeros((256, 4), dtype=np.uint8)
        labels_to_draw = (
            self._visible_labels
            if self._visible_labels is not None
            else self._overlay_palette.keys()
        )
        for label in labels_to_draw:
            if not (0 < label < 256):
                 continue
            c = self._overlay_palette.get(label)
            if c:
                scaled_alpha = int(round(c[3] * self._overlay_opacity))
                scaled_alpha = max(0, min(255, scaled_alpha))
                lut[label] = [c[2], c[1], c[0], scaled_alpha] # BGRA -> RGBA?
                # Palette is BGRA. VisPy Image might want RGBA or BGRA?
                # Usually VisPy expects RGBA.
                # EndviewView code used: r, g, b, a logic.
                # My snippet above: c[2]=r, c[1]=g, c[0]=b. Correct.

        # 3. Apply LUT -> (H, W, 4)
        rgba = lut[slice_indices]
        return rgba

    def _add_overlay_volume(self) -> None:
        """Add the overlay as a translucent 3D volume."""
        self._apply_overlay_volume_now()

    def _get_slice_data(self, index: int) -> Optional[np.ndarray]:
        """Return the normalised 2D slice at ``index`` along the first axis."""
        if self._norm_volume is None:
            return None
        clamped = max(0, min(self._norm_volume.shape[0] - 1, int(index)))
        # Extract the 2D plane: shape (height, width)
        return self._norm_volume[clamped]

    def _display_scale_factors(self) -> Tuple[float, float]:
        """Compute XY deformation factors from the requested display size."""
        if self._volume is None or self._display_size is None:
            return (1.0, 1.0)
        target_w, target_h = self._display_size
        base_w = max(1.0, float(self._volume.shape[2]))
        base_h = max(1.0, float(self._volume.shape[1]))
        return (float(target_w) / base_w, float(target_h) / base_h)

    def _on_display_scale_changed(self, *, previous_scale: Tuple[float, float]) -> None:
        if self._volume is None:
            return
        new_scale = self._display_scale_factors()
        self._apply_visual_transform()
        self._rescale_camera_for_display_scale(
            previous_scale=previous_scale,
            new_scale=new_scale,
        )
        self._update_overlay_image()
        try:
            self._canvas.update()
        except Exception:
            pass

    def _rescale_camera_for_display_scale(
        self,
        *,
        previous_scale: Tuple[float, float],
        new_scale: Tuple[float, float],
    ) -> None:
        if self._view.camera is None or self._volume is None:
            return
        depth, height, width = self._volume.shape[:3]
        new_x_max = float(width) * float(new_scale[0])
        new_y_max = float(height) * float(new_scale[1])
        if new_x_max <= 0.0 or new_y_max <= 0.0:
            return

        prev_x = max(1e-6, float(previous_scale[0]))
        prev_y = max(1e-6, float(previous_scale[1]))
        ratio_x = float(new_scale[0]) / prev_x
        ratio_y = float(new_scale[1]) / prev_y

        center = getattr(self._view.camera, "center", None)
        if center is None:
            cx = new_x_max / 2.0
            cy = new_y_max / 2.0
            cz = float(max(0, min(depth - 1, int(self._current_slice))))
        else:
            try:
                cx = float(center[0]) * ratio_x
                cy = float(center[1]) * ratio_y
                cz = float(center[2])
            except Exception:
                cx = new_x_max / 2.0
                cy = new_y_max / 2.0
                cz = float(max(0, min(depth - 1, int(self._current_slice))))
        cx = max(0.0, min(new_x_max, cx))
        cy = max(0.0, min(new_y_max, cy))
        cz = max(0.0, min(float(depth - 1), cz))

        try:
            self._view.camera.set_range(
                x=(0.0, new_x_max),
                y=(0.0, new_y_max),
                z=(0.0, float(depth)),
            )
        except Exception:
            return
        try:
            self._view.camera.center = (cx, cy, cz)
        except Exception:
            pass

    def _configure_camera(self, depth: int, height: int, width: int) -> None:
        """Configure the camera so that slices face the user and the volume fills the view."""
        if self._view.camera is None:
            return
        scale_x, scale_y = self._display_scale_factors()
        width_scaled = float(width) * float(scale_x)
        height_scaled = float(height) * float(scale_y)
        # Centre initial: milieu du volume, avec z sur la slice courante
        center = (width_scaled / 2.0, height_scaled / 2.0, float(self._current_slice))
        self._view.camera.up = "+y"
        # Set the ranges along each axis
        self._view.camera.set_range(
            x=(0.0, width_scaled),
            y=(0.0, height_scaled),
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
        scale_x, scale_y = self._display_scale_factors()
        width_scaled = float(width) * float(scale_x)
        height_scaled = float(height) * float(scale_y)
        # Focus : centre de la slice courante (x/y au centre, z = index de slice)
        z_focus = max(0, min(depth - 1, int(self._current_slice)))
        self._view.camera.center = (
            width_scaled / 2.0,
            height_scaled / 2.0,
            float(z_focus),
        )

    def _apply_visual_transform(self) -> None:
        """Apply a 180° flip in the XY plane to align 3D orientation with Endview."""
        if self._volume is None:
            return
        height = self._volume.shape[1]
        width = self._volume.shape[2]
        scale_x, scale_y = self._display_scale_factors()
        width_scaled = float(width) * float(scale_x)
        height_scaled = float(height) * float(scale_y)
        flip = STTransform(
            scale=(-float(scale_x), -float(scale_y), 1.0),
            translate=(width_scaled, height_scaled, 0.0),
        )
        if self._volume_visual is not None:
            self._volume_visual.transform = flip
            
        if self._overlay_volume_visual is not None:
            self._overlay_volume_visual.transform = flip
        if self._primary_slice_line is not None:
            self._primary_slice_line.transform = flip
        if self._secondary_slice_plane is not None:
            self._secondary_slice_plane.transform = flip
        if self._slice_image is not None:
            # Scale 1x1 pixel to full size
            self._slice_image.transform = STTransform(
                scale=(-width_scaled, -height_scaled, 1.0),
                translate=(
                    width_scaled,
                    height_scaled,
                    float(self._current_slice),
                ),
            )
        if self._slice_overlay is not None:
            self._slice_overlay.transform = self._overlay_transform()
        if self._secondary_slice_line is not None:
            self._secondary_slice_line.transform = flip

    def _overlay_transform(self) -> STTransform:
        """Return transform for overlay aligned with the base slice."""
        if self._volume is None:
            return STTransform()
        height = self._volume.shape[1]
        width = self._volume.shape[2]
        scale_x, scale_y = self._display_scale_factors()
        width_scaled = float(width) * float(scale_x)
        height_scaled = float(height) * float(scale_y)
        return STTransform(
            scale=(-float(scale_x), -float(scale_y), 1.0),
            translate=(
                width_scaled,
                height_scaled,
                float(self._current_slice),
            ),
        )
