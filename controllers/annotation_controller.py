"""Controller dédié à la gestion des annotations et overlays."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QFileDialog

from config.constants import MASK_COLORS_BGRA
from models.annotation_model import AnnotationModel
from models.roi_model import RoiModel
from models.temp_mask_model import TempMaskModel
from models.overlay_data import OverlayData
from models.view_state_model import ViewStateModel
from services.annotation_service import AnnotationService
from services.overlay_service import OverlayService
from services.overlay_export import OverlayExport
from views.annotation_view import AnnotationView
from views.overlay_settings_view import OverlaySettingsView
from views.volume_view import VolumeView


class AnnotationController:
    """Gère les annotations, labels et overlays (visibilité, couleurs, synchronisation)."""

    def __init__(
        self,
        *,
        annotation_model: AnnotationModel,
        view_state_model: ViewStateModel,
        roi_model: RoiModel,
        temp_mask_model: TempMaskModel,
        annotation_service: AnnotationService,
        overlay_service: OverlayService,
        overlay_export: OverlayExport,
        annotation_view: AnnotationView,
        volume_view: VolumeView,
        overlay_settings_view: OverlaySettingsView,
        logger: logging.Logger,
        get_volume: Optional[callable] = None,
    ) -> None:
        self.annotation_model = annotation_model
        self.view_state_model = view_state_model
        self.roi_model = roi_model
        self.temp_mask_model = temp_mask_model
        self.annotation_service = annotation_service
        self.overlay_service = overlay_service
        self.overlay_export = overlay_export
        self.annotation_view = annotation_view
        self.volume_view = volume_view
        self.overlay_settings_view = overlay_settings_view
        self.logger = logger
        self._overlay_cache: OverlayData | None = None
        self._get_volume = get_volume

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
        self.temp_mask_model.set_label_color(label_id, bgra)
        self.refresh_overlay(defer_volume=True, rebuild=False)
        # Rafraîchir la preview ROI si le label actif change de couleur
        self.refresh_roi_overlay_for_slice(self.view_state_model.current_slice)

    def on_label_added(self, label_id: int, color: QColor) -> None:
        """Gère l'ajout d'un nouveau label depuis les paramètres overlay."""
        bgra = self._qcolor_to_bgra(color)
        self.annotation_model.set_label_color(label_id, bgra)
        self.annotation_model.set_label_visibility(label_id, True)
        self.temp_mask_model.ensure_label(label_id, bgra, visible=True)
        self.view_state_model.set_active_label(label_id)
        self.refresh_overlay(defer_volume=True, rebuild=False)

    def on_overlay_toggled(self, enabled: bool) -> None:
        """Gère le toggle de visibilité de l'overlay."""
        self.view_state_model.toggle_overlay(enabled)
        self.refresh_overlay(rebuild=False)

    def refresh_overlay(self, *, defer_volume: bool = False, rebuild: bool = True) -> None:
        """Recalcule et pousse l'overlay vers les vues selon l'état actuel."""
        if not self.view_state_model.show_overlay:
            self.logger.info("Overlay hidden by toggle; clearing views.")
            self.annotation_view.set_overlay(None)
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
            self.annotation_view.set_overlay(None)
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

        self.annotation_view.set_overlay(overlay_data, visible_labels=visible_labels)
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
        self.annotation_view.set_overlay(None)
        self.annotation_view.clear_roi_overlay()
        self.annotation_view.clear_temp_shapes()
        self.volume_view.set_overlay(None)
        self.overlay_settings_view.clear_labels()
        self.temp_mask_model.clear()
        self.roi_model.clear()

    # ------------------------------------------------------------------ #
    # Interaction handlers (stubs)
    # ------------------------------------------------------------------ #
    def on_tool_mode_changed(self, mode: str) -> None:
        """Handle drawing tool changes (stub)."""
        self.view_state_model.set_tool_mode(mode)

    def on_threshold_changed(self, value: int) -> None:
        """Handle manual threshold change (stub)."""
        self.view_state_model.set_threshold(value)

    def on_threshold_auto_toggled(self, enabled: bool) -> None:
        """Handle auto-threshold toggle (stub)."""
        self.view_state_model.set_threshold_auto(enabled)

    def on_label_selected(self, label_id: int) -> None:
        """Handle active label selection from the tools panel."""
        self.view_state_model.set_active_label(label_id)

    def on_apply_volume_toggled(self, enabled: bool) -> None:
        """Handle apply-to-volume toggle (stub)."""
        self.view_state_model.set_apply_volume(enabled)

    def on_roi_persistence_toggled(self, enabled: bool) -> None:
        """Handle ROI persistence toggle (stub)."""
        self.view_state_model.set_roi_persistence(enabled)

    def on_roi_recompute_requested(self) -> None:
        """Handle ROI recomputation request (stub)."""
        slice_idx = self.view_state_model.current_slice
        self._rebuild_slice_preview(slice_idx)
        self.refresh_roi_overlay_for_slice(slice_idx)

    def on_roi_delete_requested(self) -> None:
        """Handle ROI deletion request (stub)."""
        # Supprime toutes les ROI (toutes slices) et nettoie les previews
        self.roi_model.clear()
        self.temp_mask_model.clear()
        self.annotation_view.clear_roi_overlay()
        self.annotation_view.clear_roi_boxes()
        self.annotation_view.clear_temp_shapes()
        self.refresh_roi_overlay_for_slice(self.view_state_model.current_slice)

    def on_selection_cancel_requested(self) -> None:
        """Handle selection cancel request (stub)."""
        slice_idx = self.view_state_model.current_slice
        self.temp_mask_model.clear_slice(slice_idx)
        self.annotation_view.clear_roi_overlay()
        self.annotation_view.clear_temp_shapes()
        self.refresh_roi_overlay_for_slice(slice_idx)

    def on_annotation_mouse_clicked(self, pos: Any, button: Any) -> None:
        """Handle mouse click in annotation view (stub)."""
        pass

    def on_annotation_freehand_started(self, pos: Any) -> None:
        """Handle free-hand start (stub)."""
        pass

    def on_annotation_freehand_point_added(self, pos: Any) -> None:
        """Handle free-hand point addition (stub)."""
        pass

    def on_annotation_freehand_completed(self, points: Any) -> None:
        """Handle free-hand completion (stub)."""
        pass

    def on_annotation_box_drawn(self, box: Any) -> None:
        """Handle box draw completion (stub)."""
        mask_volume = self.annotation_model.get_mask_volume()
        if mask_volume is None:
            mask_volume = self.temp_mask_model.get_mask_volume()
        if mask_volume is None:
            return
        if self.view_state_model.active_label is None:
            return
        label = self.view_state_model.active_label

        try:
            slice_idx = int(self.view_state_model.current_slice)
            h, w = mask_volume.shape[1], mask_volume.shape[2]
        except Exception:
            return

        palette = self.annotation_model.get_label_palette()
        self.annotation_service.apply_box_roi(
            slice_idx=slice_idx,
            box=box,
            shape=(h, w),
            label=label,
            threshold=self.view_state_model.threshold,
            persistent=self.view_state_model.roi_persistence,
            roi_model=self.roi_model,
            temp_mask_model=self.temp_mask_model,
            palette=palette,
            slice_data=self._slice_data(slice_idx),
        )
        self.refresh_roi_overlay_for_slice(slice_idx)

    def refresh_roi_overlay_for_slice(self, slice_idx: int) -> None:
        """Refresh ROI preview overlay for the given slice."""
        slice_mask = self.temp_mask_model.get_slice_mask(slice_idx)
        if slice_mask is None or not np.any(slice_mask):
            self.annotation_view.clear_roi_overlay()
        else:
            self.annotation_view.set_roi_overlay(slice_mask, palette=self.temp_mask_model.label_palette)

        boxes = self.roi_model.boxes_for_slice(slice_idx, include_persistent=True)
        if boxes:
            self.annotation_view.set_roi_boxes(boxes)
        else:
            self.annotation_view.clear_roi_boxes()

    def on_annotation_point_selected(self, pos: Any) -> None:
        """Handle point selection (stub)."""
        pass

    def on_annotation_drag_update(self, pos: Any) -> None:
        """Handle drag update (stub)."""
        pass

    def on_apply_temp_mask_requested(self) -> None:
        """Apply the current temporary mask (free-hand/ROI) into the annotation model."""
        slice_idx = self.view_state_model.current_slice
        mask_volume = self.annotation_model.get_mask_volume()
        temp_slice = self.temp_mask_model.get_slice_mask(slice_idx)

        if mask_volume is None or temp_slice is None:
            return
        if not np.any(temp_slice):
            return
        try:
            current_slice = mask_volume[int(slice_idx)]
        except Exception:
            return
        if current_slice.shape != temp_slice.shape:
            return

        updated = np.array(current_slice, copy=True)
        updated[temp_slice > 0] = temp_slice[temp_slice > 0]
        self.annotation_model.set_slice_mask(slice_idx, updated)

        for label, color in self.temp_mask_model.label_palette.items():
            self.annotation_model.ensure_label(label, color, visible=True)

        self.temp_mask_model.clear_slice(slice_idx)
        self.annotation_view.clear_temp_shapes()
        self.annotation_view.clear_roi_overlay()
        self.annotation_view.clear_roi_boxes()

        self.refresh_overlay(defer_volume=True, rebuild=True)
        self.refresh_roi_overlay_for_slice(slice_idx)

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

    def _slice_data(self, slice_idx: int):
        """Return raw slice data (H, W) from the current volume if available."""
        if self._get_volume is None:
            return None
        volume = self._get_volume()
        if volume is None:
            return None
        try:
            return volume[int(slice_idx)]
        except Exception:
            return None

    def _rebuild_slice_preview(self, slice_idx: int) -> None:
        """Rebuild temporary mask and box for a given slice from stored ROIs."""
        mask_shape = self.annotation_model.mask_shape_hw() or self.temp_mask_model.mask_shape_hw()
        if mask_shape is None:
            return

        rois = self.roi_model.list_on_slice(slice_idx) or self.roi_model.list_persistent()
        boxes = self.annotation_service.rebuild_temp_masks_for_slice(
            rois=rois,
            shape=mask_shape,
            slice_idx=slice_idx,
            temp_mask_model=self.temp_mask_model,
            palette=self.annotation_model.get_label_palette(),
            clear_slice=True,
            slice_data=self._slice_data(slice_idx),
        )

        slice_mask = self.temp_mask_model.get_slice_mask(slice_idx)
        if slice_mask is not None and np.any(slice_mask):
            self.annotation_view.set_roi_overlay(slice_mask, palette=self.temp_mask_model.label_palette)
        else:
            self.annotation_view.clear_roi_overlay()

        if boxes:
            self.annotation_view.set_roi_boxes(boxes)
        else:
            self.annotation_view.clear_roi_boxes()
