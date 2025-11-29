"""Controller dédié à la gestion des annotations et overlays."""

from __future__ import annotations

import logging
from PyQt6.QtGui import QColor

from config.constants import MASK_COLORS_BGRA
from models.annotation_model import AnnotationModel
from models.view_state_model import ViewStateModel
from services.overlay_service import OverlayService
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
        endview_view: EndviewView,
        volume_view: VolumeView,
        overlay_settings_view: OverlaySettingsView,
        logger: logging.Logger,
    ) -> None:
        self.annotation_model = annotation_model
        self.view_state_model = view_state_model
        self.overlay_service = overlay_service
        self.endview_view = endview_view
        self.volume_view = volume_view
        self.overlay_settings_view = overlay_settings_view
        self.logger = logger

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
        self.refresh_overlay()

    def on_label_color_changed(self, label_id: int, color: QColor) -> None:
        """Gère les changements de couleur des labels overlay."""
        bgra = self._qcolor_to_bgra(color)
        self.annotation_model.set_label_color(label_id, bgra)
        self.annotation_model.set_label_visibility(label_id, True)
        self.refresh_overlay()

    def on_label_added(self, label_id: int, color: QColor) -> None:
        """Gère l'ajout d'un nouveau label depuis les paramètres overlay."""
        bgra = self._qcolor_to_bgra(color)
        self.annotation_model.set_label_color(label_id, bgra)
        self.annotation_model.set_label_visibility(label_id, True)
        self.refresh_overlay()

    def on_overlay_toggled(self, enabled: bool) -> None:
        """Gère le toggle de visibilité de l'overlay."""
        self.view_state_model.toggle_overlay(enabled)
        self.refresh_overlay()

    def refresh_overlay(self) -> None:
        """Recalcule et pousse l'overlay vers les vues selon l'état actuel."""
        if not self.view_state_model.show_overlay:
            self.logger.info("Overlay hidden by toggle; clearing views.")
            self.endview_view.set_overlay(None)
            self.volume_view.set_overlay(None)
            return

        mask_volume = self.annotation_model.get_mask_volume()
        palette = self.annotation_model.get_label_palette()
        visible_labels = self.annotation_model.get_visible_labels()
        overlay = self.overlay_service.build_overlay_rgba(mask_volume, palette, visible_labels)

        # Force refresh in the views to avoid stale overlays when visibility changes
        self.endview_view.set_overlay(None)
        self.volume_view.set_overlay(None)

        if overlay is not None:
            self.logger.info("Pushing overlay to views | shape=%s dtype=%s", overlay.shape, overlay.dtype)
        else:
            self.logger.info("No overlay available to push; clearing views.")

        self.endview_view.set_overlay(overlay)
        self.volume_view.set_overlay(overlay)

    def clear_labels(self) -> None:
        """Efface tous les labels de la vue de paramètres overlay."""
        self.overlay_settings_view.clear_labels()

    def sync_overlay_settings(self) -> None:
        """Synchronise la vue de paramètres overlay avec le modèle d'annotation."""
        self._sync_overlay_settings_with_model()

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
