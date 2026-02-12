from __future__ import annotations

import numpy as np
from enum import Enum, auto
from PyQt6.QtWidgets import QWidget, QMenu
from PyQt6.QtGui import QAction, QActionGroup
from vispy import scene

from views.volume_view import VolumeView


class AnchorMode(Enum):
    SLICE = auto()
    VOLUME_CENTER = auto()


class Piece3DView(VolumeView):
    """Vue dédiée pour afficher la pièce corrosion en rendu iso métallique."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._recenter_on_slice_change = True
        self._canvas.bgcolor = "#0b0b0b"
        self._slider.hide()
        self._iso_threshold: float = 0.5
        self._shading_profile = "standard"
        self._shading_lut_boost: bool = False
        self.set_base_colormap("Metal", self._metal_lut())
        self._status.setText("Pièce 3D non chargée")
        self._anchor_mode = AnchorMode.VOLUME_CENTER
        self._anchor_point: tuple[float, float, float] | None = None

    def set_piece_volume(self, volume: np.ndarray) -> None:
        """Charge le volume solide (0/1) et reconstruit la scène iso."""
        self._iso_threshold = 0.5
        self.set_volume(volume)
        if self._volume_visual is not None:
            self._volume_visual.threshold = self._iso_threshold
            self._apply_shading()

    def set_anchor_point(self, anchor: tuple[float, float, float] | None) -> None:
        """Set the data-space anchor point (x, y, z) used for camera centering."""
        self._anchor_point = anchor
        if self._anchor_mode == AnchorMode.VOLUME_CENTER:
            self._focus_camera_on_slice()

    def set_overlay(self, *args, **kwargs) -> None:  # type: ignore[override]
        """Désactive les overlays pour cette vue (non utilisés)."""
        return

    def contextMenuEvent(self, event) -> None:
        """Affiche un menu contextuel pour changer le mode d'ancrage."""
        menu = QMenu(self)

        group = QActionGroup(self)
        group.setExclusive(True)

        shading_menu = menu.addMenu("Ombrage")
        action_std = QAction("Standard", self)
        action_std.setCheckable(True)
        action_std.setActionGroup(group)
        action_std.setChecked(self._shading_profile == "standard")
        action_std.triggered.connect(lambda: self._set_shading_profile("standard"))
        shading_menu.addAction(action_std)

        action_boost = QAction("Renforcé", self)
        action_boost.setCheckable(True)
        action_boost.setActionGroup(group)
        action_boost.setChecked(self._shading_profile == "boosted")
        action_boost.triggered.connect(lambda: self._set_shading_profile("boosted"))
        shading_menu.addAction(action_boost)

        action_slice = QAction("Ancrage: Slice", self)
        action_slice.setCheckable(True)
        action_slice.setChecked(self._anchor_mode == AnchorMode.SLICE)
        action_slice.triggered.connect(lambda: self._set_anchor_mode(AnchorMode.SLICE))
        menu.addAction(action_slice)

        action_center = QAction("Ancrage: Centre du Volume", self)
        action_center.setCheckable(True)
        action_center.setChecked(self._anchor_mode == AnchorMode.VOLUME_CENTER)
        action_center.triggered.connect(lambda: self._set_anchor_mode(AnchorMode.VOLUME_CENTER))
        menu.addAction(action_center)

        menu.exec(event.globalPos())

    def _set_anchor_mode(self, mode: AnchorMode) -> None:
        if self._anchor_mode != mode:
            self._anchor_mode = mode
            if mode == AnchorMode.SLICE:
                self._slider.show()
            else:
                self._slider.hide()
            self._focus_camera_on_slice()

    def _focus_camera_on_slice(self) -> None:
        """Recentre la caméra selon le mode d'ancrage choisi."""
        if self._anchor_mode == AnchorMode.SLICE:
            super()._focus_camera_on_slice()
        elif self._anchor_mode == AnchorMode.VOLUME_CENTER:
            if self._view.camera is None or self._volume is None:
                return
            depth, height, width = self._volume.shape[:3]
            if self._anchor_point is not None:
                x, y, z = self._anchor_point
                x = max(0.0, min(float(width - 1), float(x)))
                y = max(0.0, min(float(height - 1), float(y)))
                z = max(0.0, min(float(depth - 1), float(z)))
                # Apply the same XY flip as the volume visual.
                self._view.camera.center = (
                    float(width) - x,
                    float(height) - y,
                    z,
                )
            else:
                # Fallback : centre absolu du volume
                self._view.camera.center = (
                    width / 2.0,
                    height / 2.0,
                    depth / 2.0,
                )

    def _build_scene(self) -> None:
        """Recrée la scène VisPy en mode iso, sans overlay 2D."""
        for child in list(self._view.scene.children):
            if child is not self._view.camera:
                child.parent = None
        self._volume_visual = None
        self._slice_highlight_visual = None
        self._slice_image = None
        self._slice_overlay = None
        self._overlay_volume_visual = None

        if self._norm_volume is None:
            return

        depth, height, width = self._norm_volume.shape[:3]
        self._volume_visual = scene.visuals.Volume(
            self._norm_volume,
            parent=self._view.scene,
            method="iso",
            cmap=self._base_colormap,
            threshold=self._iso_threshold,
            relative_step_size=0.7,
        )
        # Activer le depth test pour un rendu plus solide
        self._volume_visual.set_gl_state(depth_test=True, cull_face=False)
        self._apply_shading()
        self._configure_camera(depth, height, width)
        self._apply_visual_transform()

    def _set_shading_profile(self, profile: str) -> None:
        if profile not in {"standard", "boosted"}:
            return
        if self._shading_profile == profile:
            return
        self._shading_profile = profile
        self._shading_lut_boost = profile == "boosted"
        self._apply_shading()

    def _apply_shading(self) -> None:
        if self._volume_visual is None:
            return
        params = self._shading_params(self._shading_profile)
        # VisPy iso shader uses const lighting, so we vary gamma/contrast/step for visible modes.
        self._volume_visual.gamma = params["gamma"]
        self._volume_visual.clim = (params["clim_low"], params["clim_high"])
        self._volume_visual.relative_step_size = params["relative_step_size"]
        self._volume_visual.interpolation = params["interpolation"]
        lut = self._metal_lut(boosted=self._shading_lut_boost)
        self.set_base_colormap("Metal", lut)

    @staticmethod
    def _shading_params(profile: str) -> dict[str, float | str]:
        if profile == "boosted":
            return {
                "gamma": 1.25,
                "clim_low": 0.12,
                "clim_high": 0.86,
                "relative_step_size": 0.45,
                "interpolation": "linear",
            }
        return {
            "gamma": 1.0,
            "clim_low": 0.0,
            "clim_high": 1.0,
            "relative_step_size": 0.7,
            "interpolation": "linear",
        }

    @staticmethod
    def _metal_lut(*, boosted: bool = False) -> np.ndarray:
        """Construit un dégradé cuivré pour l'iso surface."""
        if boosted:
            dark = np.array([20, 10, 6], dtype=np.float32)
            mid = np.array([90, 45, 25], dtype=np.float32)
            bright = np.array([190, 130, 90], dtype=np.float32)
        else:
            dark = np.array([40, 20, 10], dtype=np.float32)
            mid = np.array([140, 70, 35], dtype=np.float32)
            bright = np.array([230, 170, 120], dtype=np.float32)
        t = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None]
        # Accentue le contraste autour des valeurs intermédiaires
        if boosted:
            t = np.clip(0.5 + 1.35 * (t - 0.5), 0.0, 1.0)
        else:
            t = np.clip(0.5 + 1.15 * (t - 0.5), 0.0, 1.0)
        lut = np.where(
            t < 0.5,
            dark + (mid - dark) * (t * 2.0),
            mid + (bright - mid) * ((t - 0.5) * 2.0),
        )
        return np.clip(lut / 255.0, 0.0, 1.0)
