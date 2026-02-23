from __future__ import annotations

from enum import Enum, auto

import numpy as np
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QMenu, QWidget
from vispy import scene

from views.volume_view import VolumeView


class AnchorMode(Enum):
    SLICE = auto()
    VOLUME_CENTER = auto()


class PieceGeometrySource(Enum):
    DISTANCE_PRISM = auto()
    LEGACY_BWFW = auto()


class Piece3DView(VolumeView):
    """Vue dediee pour afficher la piece corrosion en rendu iso metallique."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._recenter_on_slice_change = True
        self._canvas.bgcolor = "#0b0b0b"
        self._slider.hide()
        self._iso_threshold: float = 0.5
        self._shading_profile = "standard"
        self._shading_lut_boost: bool = False
        self.set_base_colormap("Metal", self._metal_lut())
        self._status.setText("Piece 3D non chargee")
        self._anchor_mode = AnchorMode.VOLUME_CENTER
        self._anchor_point: tuple[float, float, float] | None = None

        # Source switching: distance prism (default) or legacy BW/FW solid.
        self._geometry_source = PieceGeometrySource.DISTANCE_PRISM
        self._show_interpolated: bool = True
        self._piece_distance_raw: np.ndarray | None = None
        self._piece_distance_interpolated: np.ndarray | None = None
        self._piece_legacy_raw: np.ndarray | None = None
        self._piece_legacy_interpolated: np.ndarray | None = None

    def set_piece_volume(self, volume: np.ndarray) -> None:
        """Compatibilite: charge un volume unique dans tous les slots."""
        data = None if volume is None else np.asarray(volume, dtype=np.float32)
        self.set_piece_volume_sources(
            distance_raw=data,
            distance_interpolated=data,
            legacy_raw=data,
            legacy_interpolated=data,
        )

    def set_piece_volume_sources(
        self,
        *,
        distance_raw: np.ndarray | None,
        distance_interpolated: np.ndarray | None,
        legacy_raw: np.ndarray | None,
        legacy_interpolated: np.ndarray | None,
    ) -> None:
        """Assigne tous les volumes disponibles et rafraichit l affichage."""
        self._piece_distance_raw = None if distance_raw is None else np.asarray(distance_raw, dtype=np.float32)
        self._piece_distance_interpolated = (
            None if distance_interpolated is None else np.asarray(distance_interpolated, dtype=np.float32)
        )
        self._piece_legacy_raw = None if legacy_raw is None else np.asarray(legacy_raw, dtype=np.float32)
        self._piece_legacy_interpolated = (
            None if legacy_interpolated is None else np.asarray(legacy_interpolated, dtype=np.float32)
        )
        self._refresh_piece_volume()

    def set_piece_show_interpolated(self, enabled: bool) -> None:
        """Choisit entre volume brut et interpole pour la source active."""
        self._show_interpolated = bool(enabled)
        self._refresh_piece_volume()

    def set_anchor_point(self, anchor: tuple[float, float, float] | None) -> None:
        """Set the data-space anchor point (x, y, z) used for camera centering."""
        self._anchor_point = anchor
        if self._anchor_mode == AnchorMode.VOLUME_CENTER:
            self._focus_camera_on_slice()

    def set_overlay(self, *args, **kwargs) -> None:  # type: ignore[override]
        """Desactive les overlays pour cette vue dediee."""
        return

    def contextMenuEvent(self, event) -> None:
        """Affiche le menu contextuel (ombrage, geometrie, ancrage)."""
        menu = QMenu(self)

        shading_group = QActionGroup(self)
        shading_group.setExclusive(True)

        shading_menu = menu.addMenu("Ombrage")
        action_std = QAction("Standard", self)
        action_std.setCheckable(True)
        action_std.setActionGroup(shading_group)
        action_std.setChecked(self._shading_profile == "standard")
        action_std.triggered.connect(lambda: self._set_shading_profile("standard"))
        shading_menu.addAction(action_std)

        action_boost = QAction("Renforce", self)
        action_boost.setCheckable(True)
        action_boost.setActionGroup(shading_group)
        action_boost.setChecked(self._shading_profile == "boosted")
        action_boost.triggered.connect(lambda: self._set_shading_profile("boosted"))
        shading_menu.addAction(action_boost)

        geometry_group = QActionGroup(self)
        geometry_group.setExclusive(True)

        geometry_menu = menu.addMenu("Geometrie")
        action_distance = QAction("Prisme distance", self)
        action_distance.setCheckable(True)
        action_distance.setActionGroup(geometry_group)
        action_distance.setChecked(self._geometry_source == PieceGeometrySource.DISTANCE_PRISM)
        action_distance.triggered.connect(
            lambda: self._set_geometry_source(PieceGeometrySource.DISTANCE_PRISM)
        )
        geometry_menu.addAction(action_distance)

        action_legacy = QAction("Volume BW/FW", self)
        action_legacy.setCheckable(True)
        action_legacy.setActionGroup(geometry_group)
        action_legacy.setChecked(self._geometry_source == PieceGeometrySource.LEGACY_BWFW)
        action_legacy.triggered.connect(
            lambda: self._set_geometry_source(PieceGeometrySource.LEGACY_BWFW)
        )
        geometry_menu.addAction(action_legacy)

        action_slice = QAction("Ancrage: Slice", self)
        action_slice.setCheckable(True)
        action_slice.setChecked(self._anchor_mode == AnchorMode.SLICE)
        action_slice.triggered.connect(lambda: self._set_anchor_mode(AnchorMode.SLICE))
        menu.addAction(action_slice)

        action_center = QAction("Ancrage: Centre du volume", self)
        action_center.setCheckable(True)
        action_center.setChecked(self._anchor_mode == AnchorMode.VOLUME_CENTER)
        action_center.triggered.connect(lambda: self._set_anchor_mode(AnchorMode.VOLUME_CENTER))
        menu.addAction(action_center)

        menu.exec(event.globalPos())

    def _set_geometry_source(self, source: PieceGeometrySource) -> None:
        if self._geometry_source == source:
            return
        self._geometry_source = source
        self._refresh_piece_volume()

    def _set_anchor_mode(self, mode: AnchorMode) -> None:
        if self._anchor_mode != mode:
            self._anchor_mode = mode
            if mode == AnchorMode.SLICE:
                self._slider.show()
            else:
                self._slider.hide()
            self._focus_camera_on_slice()

    def _resolve_current_piece_volume(self) -> np.ndarray | None:
        if self._geometry_source == PieceGeometrySource.DISTANCE_PRISM:
            primary_raw = self._piece_distance_raw
            primary_interp = self._piece_distance_interpolated
            fallback_raw = self._piece_legacy_raw
            fallback_interp = self._piece_legacy_interpolated
        else:
            primary_raw = self._piece_legacy_raw
            primary_interp = self._piece_legacy_interpolated
            fallback_raw = self._piece_distance_raw
            fallback_interp = self._piece_distance_interpolated

        preferred: list[np.ndarray | None]
        if self._show_interpolated:
            preferred = [primary_interp, primary_raw, fallback_interp, fallback_raw]
        else:
            preferred = [primary_raw, primary_interp, fallback_raw, fallback_interp]

        for candidate in preferred:
            if candidate is not None and candidate.size > 0:
                return candidate
        return None

    def _refresh_piece_volume(self) -> None:
        volume = self._resolve_current_piece_volume()
        if volume is None:
            return
        self._iso_threshold = 0.5
        self.set_volume(volume)
        if self._volume_visual is not None:
            self._volume_visual.threshold = self._iso_threshold
            self._apply_shading()
        if self._anchor_mode == AnchorMode.VOLUME_CENTER:
            self._focus_camera_on_slice()

        source_label = "distance" if self._geometry_source == PieceGeometrySource.DISTANCE_PRISM else "bw/fw"
        interp_label = "interpole" if self._show_interpolated else "brut"
        self._status.setText(f"Piece 3D ({source_label}, {interp_label})")

    def _focus_camera_on_slice(self) -> None:
        """Recentre la camera selon le mode d ancrage."""
        if self._anchor_mode == AnchorMode.SLICE:
            super()._focus_camera_on_slice()
            return

        if self._view.camera is None or self._volume is None:
            return
        depth, height, width = self._volume.shape[:3]
        scale_x, scale_y = self._display_scale_factors()
        width_scaled = float(width) * float(scale_x)
        height_scaled = float(height) * float(scale_y)
        if self._anchor_point is not None:
            x, y, z = self._anchor_point
            x = max(0.0, min(float(width - 1), float(x)))
            y = max(0.0, min(float(height - 1), float(y)))
            z = max(0.0, min(float(depth - 1), float(z)))
            # Apply the same XY flip as the volume visual.
            self._view.camera.center = (
                width_scaled - (x * float(scale_x)),
                height_scaled - (y * float(scale_y)),
                z,
            )
        else:
            self._view.camera.center = (
                width_scaled / 2.0,
                height_scaled / 2.0,
                depth / 2.0,
            )

    def _build_scene(self) -> None:
        """Reconstruit la scene VisPy en mode iso, sans overlay 2D."""
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
        """Build a copper-like gradient for iso surface rendering."""
        if boosted:
            dark = np.array([20, 10, 6], dtype=np.float32)
            mid = np.array([90, 45, 25], dtype=np.float32)
            bright = np.array([190, 130, 90], dtype=np.float32)
        else:
            dark = np.array([40, 20, 10], dtype=np.float32)
            mid = np.array([140, 70, 35], dtype=np.float32)
            bright = np.array([230, 170, 120], dtype=np.float32)
        t = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None]
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
