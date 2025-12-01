"""Controller dédié à la gestion des annotations et overlays."""

from __future__ import annotations

import logging
from typing import Any, Optional

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QFileDialog

from config.constants import MASK_COLORS_BGRA
from models.annotation_model import AnnotationModel
from models.overlay_data import OverlayData
from models.view_state_model import ViewStateModel
from services.overlay_service import OverlayService
from services.overlay_export import OverlayExport
from views.endview_view import EndviewView
from views.overlay_settings_view import OverlaySettingsView
from views.volume_view import VolumeView


class AnnotationController:
    """Gère les annotations, labels et overlays (visibilité, couleurs, synchronisation)."""

    def __init__(
        self,
        *,
        annotation_model: AnnotationModel,
        view_state_model: ViewStateModel,
        overlay_service: OverlayService,
        overlay_export: OverlayExport,
        endview_view: EndviewView,
        volume_view: VolumeView,
        overlay_settings_view: OverlaySettingsView,
        logger: logging.Logger,
    ) -> None:
        self.annotation_model = annotation_model
        self.view_state_model = view_state_model
        self.overlay_service = overlay_service
        self.overlay_export = overlay_export
        self.endview_view = endview_view
        self.volume_view = volume_view
        self.overlay_settings_view = overlay_settings_view
        self.logger = logger
        self._overlay_cache: OverlayData | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def open_overlay_settings(self) -> None:
        """Ouvre la fenêtre de paramètres overlay et synchronise les labels actuels."""
        self._sync_overlay_settings_with_model()
        self.overlay_settings_view.show()
        self.overlay_settings_view.raise_()
        self.overlay_settings_view.activateWindow()

    def on_label_visibility_changed(self, label_id: int, visible: bool) -> None:
        """Gère les changements de visibilité des labels overlay."""
        if label_id not in self.annotation_model.label_palette:
            default_color = MASK_COLORS_BGRA.get(label_id, (255, 0, 255, 160))
            self.annotation_model.ensure_label(label_id, default_color, visible=visible)
        self.annotation_model.set_label_visibility(label_id, visible)
        self.refresh_overlay(defer_volume=True, rebuild=False)

    def on_label_color_changed(self, label_id: int, color: QColor) -> None:
        """Gère les changements de couleur des labels overlay."""
        bgra = self._qcolor_to_bgra(color)
        self.annotation_model.set_label_color(label_id, bgra)
        self.annotation_model.set_label_visibility(label_id, True)
        self.refresh_overlay(defer_volume=True, rebuild=False)

    def on_label_added(self, label_id: int, color: QColor) -> None:
        """Gère l'ajout d'un nouveau label depuis les paramètres overlay."""
        bgra = self._qcolor_to_bgra(color)
        self.annotation_model.set_label_color(label_id, bgra)
        self.annotation_model.set_label_visibility(label_id, True)
        self.refresh_overlay(defer_volume=True, rebuild=False)

    def on_overlay_toggled(self, enabled: bool) -> None:
        """Gère le toggle de visibilité de l'overlay."""
        self.view_state_model.toggle_overlay(enabled)
        self.refresh_overlay(rebuild=False)

    def refresh_overlay(self, *, defer_volume: bool = False, rebuild: bool = True) -> None:
        """Recalcule et pousse l'overlay vers les vues selon l'état actuel."""
        if not self.view_state_model.show_overlay:
            self.logger.info("Overlay hidden by toggle; clearing views.")
            self.endview_view.set_overlay(None)
            self.volume_view.set_overlay(None)
            return

        mask_volume = self.annotation_model.get_mask_volume()
        palette = self.annotation_model.get_label_palette()
        visible_labels = self.annotation_model.get_visible_labels()

        overlay_data = None
        if not rebuild and self._overlay_cache is not None:
            overlay_data = OverlayData(
                label_volumes=self._overlay_cache.label_volumes,
                palette=palette,
            )
        else:
            overlay_data = self.overlay_service.build_overlay_data(mask_volume, palette)
            self._overlay_cache = overlay_data

        if overlay_data is None:
            self.logger.info("No overlay available to push; clearing views.")
            self.endview_view.set_overlay(None)
            self.volume_view.set_overlay(None)
            return

        mask_label_count = len(overlay_data.label_volumes)
        palette_count = len(palette)
        visible_count = len(visible_labels) if visible_labels is not None else palette_count
        self.logger.info(
            "Pushing overlay to views | mask_labels=%d | palette=%d | visible=%s",
            mask_label_count,
            palette_count,
            visible_count if visible_labels is not None else "all",
        )

        self.endview_view.set_overlay(overlay_data, visible_labels=visible_labels)
        self.volume_view.set_overlay(
            overlay_data,
            visible_labels=visible_labels,
            defer_3d=defer_volume,
        )

    def clear_labels(self) -> None:
        """Efface tous les labels de la vue de paramètres overlay."""
        self.overlay_settings_view.clear_labels()

    def sync_overlay_settings(self) -> None:
        """Synchronise la vue de paramètres overlay avec le modèle d'annotation."""
        self._sync_overlay_settings_with_model()

    def reset_overlay_state(self) -> None:
        """Réinitialise le cache et nettoie les overlays (ex: lors du chargement d'un nouveau NDE)."""
        self._overlay_cache = None
        self.endview_view.set_overlay(None)
        self.volume_view.set_overlay(None)
        self.overlay_settings_view.clear_labels()

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #

    def save_overlay_via_dialog(self, *, parent: Any, volume_shape: Optional[tuple[int, int, int]]) -> Optional[str]:
        """
        Ouvre une boîte de dialogue et sauvegarde l'overlay courant en NPZ.

        Returns le chemin sauvegardé ou None si annulé.
        """
        mask_volume = self.annotation_model.get_mask_volume()
        if mask_volume is None:
            raise ValueError("Aucun overlay à sauvegarder.")

        # Validation de shape si fournie
        if volume_shape is not None:
            try:
                tgt_shape = tuple(int(x) for x in volume_shape)
            except Exception:
                tgt_shape = None
            else:
                if mask_volume.shape != tgt_shape:
                    raise ValueError(f"Overlay shape {mask_volume.shape} différent du volume {tgt_shape}.")

        file_path, _ = QFileDialog.getSaveFileName(
            parent,
            "Sauvegarder l'overlay (.npz)",
            "",
            "Overlay NPZ (*.npz);;All Files (*)",
        )
        if not file_path:
            return None

        saved_path = self.overlay_export.save_npz(mask_volume, file_path, expected_shape=volume_shape)
        return saved_path

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _sync_overlay_settings_with_model(self) -> None:
        """Remplit la fenêtre de paramètres overlay avec la palette/visibilité actuelle du modèle."""
        entries = []
        palette = self.annotation_model.label_palette
        visibility = self.annotation_model.label_visibility
        for label_id in sorted(palette.keys()):
            color = palette[label_id]
            qcolor = self._bgra_to_qcolor(color)
            visible = visibility.get(label_id, True)
            entries.append((label_id, qcolor, visible))
        self.overlay_settings_view.set_labels(entries)

    @staticmethod
    def _bgra_to_qcolor(color: tuple[int, int, int, int]) -> QColor:
        """Convert BGRA tuple to QColor."""
        b, g, r, a = color
        return QColor(r, g, b, a)

    @staticmethod
    def _qcolor_to_bgra(color: QColor) -> tuple[int, int, int, int]:
        """Convert QColor to BGRA tuple."""
        r, g, b, a = color.getRgb()
        return (b, g, r, a)
