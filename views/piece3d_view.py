from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QWidget
from vispy import scene

from views.volume_view import VolumeView


class Piece3DView(VolumeView):
    """Vue dédiée pour afficher la pièce corrosion en rendu iso métallique."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._canvas.bgcolor = "#0b0b0b"
        self._slider.hide()
        self._iso_threshold: float = 0.5
        self.set_base_colormap("Metal", self._metal_lut())
        self._status.setText("Pièce 3D non chargée")

    def set_piece_volume(self, volume: np.ndarray) -> None:
        """Charge le volume solide (0/1) et reconstruit la scène iso."""
        self._iso_threshold = 0.5
        self.set_volume(volume)
        if self._volume_visual is not None:
            self._volume_visual.threshold = self._iso_threshold

    def set_overlay(self, *args, **kwargs) -> None:  # type: ignore[override]
        """Désactive les overlays pour cette vue (non utilisés)."""
        return

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
        )
        # Activer le depth test pour un rendu plus solide
        self._volume_visual.set_gl_state(depth_test=True, cull_face=False)
        self._configure_camera(depth, height, width)
        self._apply_visual_transform()

    @staticmethod
    def _metal_lut() -> np.ndarray:
        """Construit un dégradé cuivré pour l'iso surface."""
        dark = np.array([40, 20, 10], dtype=np.float32)
        mid = np.array([140, 70, 35], dtype=np.float32)
        bright = np.array([230, 170, 120], dtype=np.float32)
        t = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None]
        lut = np.where(
            t < 0.5,
            dark + (mid - dark) * (t * 2.0),
            mid + (bright - mid) * ((t - 0.5) * 2.0),
        )
        return np.clip(lut / 255.0, 0.0, 1.0)
