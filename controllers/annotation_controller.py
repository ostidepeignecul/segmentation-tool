"""Controller dédié à la gestion des annotations et overlays."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QDialog

from config.constants import MASK_COLORS_BGRA
from models.annotation_model import AnnotationModel
from models.roi_model import RoiModel
from models.temp_mask_model import TempMaskModel
from models.overlay_data import OverlayData
from models.view_state_model import ViewStateModel
from services.annotation_axis_service import AnnotationAxisService
from services.annotation_service import AnnotationService
from services.overlay_service import OverlayService
from services.overlay_export import OverlayExport
from views.annotation_view import AnnotationView
from views.endview_view_corrosion import EndviewViewCorrosion
from views.overlay_settings_view import OverlaySettingsView
from views.overlay_export_dialog import OverlayExportDialog
from views.volume_view import VolumeView


class AnnotationController:
    """Gère les annotations, labels et overlays (visibilité, couleurs, synchronisation)."""

    PAINT_RADIUS_DEFAULT = 8
    RESTRICTION_MIN_SIZE = 10

    def __init__(
        self,
        *,
        annotation_model: AnnotationModel,
        view_state_model: ViewStateModel,
        roi_model: RoiModel,
        temp_mask_model: TempMaskModel,
        annotation_axis_service: AnnotationAxisService,
        annotation_service: AnnotationService,
        overlay_service: OverlayService,
        overlay_export: OverlayExport,
        annotation_view: AnnotationView,
        annotation_corrosion_view: Optional[EndviewViewCorrosion],
        annotation_secondary_view: Optional[AnnotationView],
        annotation_secondary_corrosion_view: Optional[EndviewViewCorrosion],
        volume_view: VolumeView,
        overlay_settings_view: OverlaySettingsView,
        logger: logging.Logger,
        get_volume: Optional[callable] = None,
    ) -> None:
        self.annotation_model = annotation_model
        self.view_state_model = view_state_model
        self.roi_model = roi_model
        self.temp_mask_model = temp_mask_model
        self.annotation_axis_service = annotation_axis_service
        self.annotation_service = annotation_service
        self.overlay_service = overlay_service
        self.overlay_export = overlay_export
        self.annotation_view = annotation_view
        self.annotation_corrosion_view = annotation_corrosion_view
        self.annotation_secondary_view = annotation_secondary_view
        self.annotation_secondary_corrosion_view = annotation_secondary_corrosion_view
        self.volume_view = volume_view
        self.overlay_settings_view = overlay_settings_view
        self.logger = logger
        self._get_volume = get_volume
        self.on_paint_size_changed(self.view_state_model.paint_radius)

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
        bgra = self.overlay_settings_view.qcolor_to_bgra(color)
        self.annotation_model.set_label_color(label_id, bgra)
        self.annotation_model.set_label_visibility(label_id, True)
        self.temp_mask_model.set_label_color(label_id, bgra)
        self.refresh_overlay(defer_volume=True, rebuild=False)
        # Rafraîchir la preview ROI si le label actif change de couleur
        self.refresh_roi_overlay_for_slice(self.view_state_model.current_slice)

    def on_label_added(self, label_id: int, color: QColor) -> None:
        """Gère l'ajout d'un nouveau label depuis les paramètres overlay."""
        bgra = self.overlay_settings_view.qcolor_to_bgra(color)
        self.annotation_model.set_label_color(label_id, bgra)
        self.annotation_model.set_label_visibility(label_id, True)
        self.temp_mask_model.ensure_label(label_id, bgra, visible=True)
        self.view_state_model.set_active_label(label_id)
        self.refresh_overlay(defer_volume=True, rebuild=False)

    def on_label_deleted(self, label_id: int) -> None:
        """Supprime un label (palette + masques + ROIs) et rafraîchit les vues."""
        lbl = int(label_id)
        self.annotation_model.remove_label(lbl)
        self.temp_mask_model.remove_label(lbl)
        self.roi_model.remove_label(lbl)
        if self.view_state_model.active_label == lbl:
            self.view_state_model.set_active_label(None)
        self.annotation_view.clear_temp_shapes()
        self.annotation_view.clear_roi_overlay()
        self.annotation_view.clear_roi_boxes()
        self.annotation_view.clear_roi_points()
        self.refresh_overlay(defer_volume=True, rebuild=False)
        self.refresh_roi_overlay_for_slice(self.view_state_model.current_slice)

    def on_overlay_opacity_changed(self, opacity: float) -> None:
        """Handle overlay opacity changes from the settings dialog."""
        self.view_state_model.set_overlay_alpha(opacity)
        self.apply_overlay_opacity()

    def on_overlay_toggled(self, enabled: bool) -> None:
        """Gère le toggle de visibilité de l'overlay."""
        self.view_state_model.toggle_overlay(enabled)
        self.refresh_overlay(rebuild=False)

    def refresh_overlay(
        self, *, defer_volume: bool = False, rebuild: bool = True, changed_slice: Optional[int] = None
    ) -> None:
        """Recalcule et pousse l'overlay vers les vues selon l'état actuel."""
        show_volume_overlay = self.view_state_model.show_overlay
        if not show_volume_overlay:
            self.logger.info("Overlay hidden by toggle; clearing 2D/3D views.")
            self.annotation_view.set_overlay(None)
            if self.annotation_secondary_view is not None:
                self.annotation_secondary_view.set_overlay(None)
            if self.annotation_secondary_corrosion_view is not None:
                self.annotation_secondary_corrosion_view.set_overlay(None)
            if self.annotation_corrosion_view is not None:
                self.annotation_corrosion_view.set_overlay(None)
            self.volume_view.set_overlay(None)

        mask_volume = self.annotation_model.get_mask_volume()
        palette = self.annotation_model.get_label_palette()
        visible_labels = self.annotation_model.get_visible_labels()

        overlay_data = None
        cached = self.annotation_model.get_overlay_cache()
        if changed_slice is not None:
            overlay_data = self.overlay_service.update_overlay_slice(
                mask_volume=mask_volume,
                label_palette=palette,
                overlay_cache=cached,
                slice_idx=changed_slice,
            )
        elif not rebuild and cached is not None:
            overlay_data = OverlayData(
                mask_volume=mask_volume,  # Added missing required field
                label_volumes=cached.label_volumes,
                palette=palette,
            )
        if overlay_data is None:
            overlay_data = self.overlay_service.build_overlay_data(mask_volume, palette)

        changed_labels = None
        # if changed_slice is not None: ... (removed optimization)

        self.annotation_model.set_overlay_cache(overlay_data)

        if overlay_data is None:
            self.logger.info("No overlay available to push; clearing views.")
            self.annotation_view.set_overlay(None)
            if self.annotation_secondary_view is not None:
                self.annotation_secondary_view.set_overlay(None)
            if self.annotation_secondary_corrosion_view is not None:
                self.annotation_secondary_corrosion_view.set_overlay(None)
            if self.annotation_corrosion_view is not None:
                self.annotation_corrosion_view.set_overlay(None)
            self.volume_view.set_overlay(None)
            return

        mask_label_count = len(palette) # approximate
        palette_count = len(palette)
        visible_count = len(visible_labels) if visible_labels is not None else palette_count
        self.logger.info(
            "Pushing overlay to views | mask_labels=%d | palette=%d | visible=%s",
            mask_label_count,
            palette_count,
            visible_count if visible_labels is not None else "all",
        )

        if show_volume_overlay:
            self.annotation_view.set_overlay(overlay_data, visible_labels=visible_labels)
            secondary_overlay = self.annotation_axis_service.build_secondary_overlay_data(overlay_data)
            if self.annotation_secondary_view is not None:
                self.annotation_secondary_view.set_overlay(
                    secondary_overlay,
                    visible_labels=visible_labels,
                )
            if self.annotation_secondary_corrosion_view is not None:
                self.annotation_secondary_corrosion_view.set_overlay(
                    secondary_overlay,
                    visible_labels=visible_labels,
                )
            if self.annotation_corrosion_view is not None:
                self.annotation_corrosion_view.set_overlay(overlay_data, visible_labels=visible_labels)
            self.volume_view.set_overlay(
                overlay_data,
                visible_labels=visible_labels,
                defer_3d=defer_volume,
                changed_slice=changed_slice,
                changed_labels=changed_labels,
            )

    def clear_labels(self) -> None:
        """Efface tous les labels de la vue de paramètres overlay."""
        self.overlay_settings_view.clear_labels()

    def sync_overlay_settings(self) -> None:
        """Synchronise la vue de paramètres overlay avec le modèle d'annotation."""
        self._sync_overlay_settings_with_model()

    def reset_overlay_state(self, *, preserve_labels: bool = False) -> None:
        """Réinitialise le cache et nettoie les overlays (ex: lors du chargement d'un nouveau NDE)."""
        self.annotation_model.clear_overlay_cache()
        self.annotation_view.set_overlay(None)
        if self.annotation_secondary_view is not None:
            self.annotation_secondary_view.set_overlay(None)
        if self.annotation_secondary_corrosion_view is not None:
            self.annotation_secondary_corrosion_view.set_overlay(None)
        if self.annotation_corrosion_view is not None:
            self.annotation_corrosion_view.set_overlay(None)
        self.annotation_view.clear_roi_overlay()
        self.annotation_view.clear_temp_shapes()
        if self.annotation_secondary_view is not None:
            self.annotation_secondary_view.clear_roi_overlay()
            self.annotation_secondary_view.clear_temp_shapes()
        self.volume_view.set_overlay(None)
        if not preserve_labels:
            self.overlay_settings_view.clear_labels()
        self.temp_mask_model.clear()
        self.roi_model.clear()

    def apply_overlay_opacity(self) -> None:
        """Apply the current overlay opacity to 2D and 3D views."""
        alpha = float(self.view_state_model.overlay_alpha)
        self.annotation_view.set_overlay_opacity(alpha)
        if self.annotation_secondary_view is not None:
            self.annotation_secondary_view.set_overlay_opacity(alpha)
        if self.annotation_secondary_corrosion_view is not None:
            self.annotation_secondary_corrosion_view.set_overlay_opacity(alpha)
        if self.annotation_corrosion_view is not None:
            self.annotation_corrosion_view.set_overlay_opacity(alpha)
        self.volume_view.set_overlay_opacity(alpha)

    # ------------------------------------------------------------------ #
    # Interaction handlers (stubs)
    # ------------------------------------------------------------------ #
    def on_tool_mode_changed(self, mode: str) -> None:
        """Handle drawing tool changes (stub)."""
        self.view_state_model.set_tool_mode(mode)
        self.annotation_view.set_tool_mode(mode)

    def on_paint_size_changed(self, radius: int) -> None:
        """Handle brush size updates from the tools panel."""
        self.view_state_model.set_paint_radius(radius)
        self.annotation_view.set_paint_radius(self.view_state_model.paint_radius)

    def on_threshold_changed(self, value: int) -> None:
        """Handle manual threshold change (stub)."""
        self.view_state_model.set_threshold(value)

    def on_threshold_auto_toggled(self, enabled: bool) -> None:
        """Handle auto-threshold toggle (stub)."""
        self.view_state_model.set_threshold_auto(enabled)

    def on_ascan_roi_mode_toggled(self, enabled: bool) -> None:
        """Handle A-scan max ROI mode toggle."""
        self.view_state_model.set_roi_ascan_max_mode(enabled)

    def on_ascan_peak_second_toggled(self, enabled: bool) -> None:
        """Handle preference toggle between first and second A-scan peaks."""
        self.view_state_model.set_roi_ascan_prefer_second_peak(enabled)

    def on_label_selected(self, label_id: int) -> None:
        """Handle active label selection from the tools panel."""
        self.view_state_model.set_active_label(label_id)

    def on_apply_volume_toggled(self, enabled: bool) -> None:
        """Handle apply-to-volume toggle (stub)."""
        self.view_state_model.set_apply_volume(enabled)

    def on_roi_persistence_toggled(self, enabled: bool) -> None:
        """Handle ROI persistence toggle (stub)."""
        self.view_state_model.set_roi_persistence(enabled)

    def on_slice_changed(self, slice_idx: int) -> None:
        """Refresh ROI preview for the requested slice."""
        self.refresh_roi_overlay_for_slice(int(slice_idx))

    def on_restriction_rect_changed(self, rect: Any) -> None:
        """Update the global restriction rectangle from the view."""
        norm = self._normalize_rect_input(rect)
        if norm is None:
            return
        shape = self.annotation_model.mask_shape_hw() or self.temp_mask_model.mask_shape_hw()
        if shape is None:
            _, shape = self._resolve_volume_dimensions()
        if shape is None:
            return
        clamped = self._clamp_rect_to_shape(norm, shape)
        self.view_state_model.set_restriction_rect(clamped)
        if clamped != norm:
            self.annotation_view.set_restriction_rect(clamped)

    def ensure_restriction_rect(self, *, shape: Optional[tuple[int, int]] = None) -> None:
        """Ensure a valid global restriction rectangle exists for the current view."""
        if shape is None:
            _, shape = self._resolve_volume_dimensions()
        if shape is None:
            return
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            return
        default_rect = (0, 0, width - 1, height - 1)
        rect = self.view_state_model.restriction_rect or default_rect
        rect = self._clamp_rect_to_shape(rect, shape)
        self.view_state_model.set_restriction_rect(rect)
        self.annotation_view.set_restriction_rect(rect)

    def on_roi_recompute_requested(self) -> None:
        """Handle ROI recomputation request (stub)."""
        if self.view_state_model.apply_volume:
            depth, mask_shape = self._resolve_volume_dimensions()
            if depth is None or mask_shape is None:
                return
            start_idx, end_idx = self._resolve_apply_volume_range(depth)
            restriction_mask = self._restriction_mask(mask_shape)
            blocked_mask_provider = lambda idx, _shape=mask_shape: self._build_blocked_mask(
                idx, _shape, include_temp=False
            )
            blocked_mask_for_label_provider = lambda idx, label_id, _shape=mask_shape: (
                self._build_label0_blocked_mask(idx, _shape, include_temp=True)
                if int(label_id) == 0
                else None
            )
            self.annotation_service.rebuild_volume_preview_from_rois(
                depth=depth,
                mask_shape=mask_shape,
                roi_model=self.roi_model,
                temp_mask_model=self.temp_mask_model,
                palette=self.annotation_model.get_label_palette(),
                slice_data_provider=self._slice_data,
                thin_line_max_width=self.view_state_model.roi_thin_line_max_width,
                start_idx=start_idx,
                end_idx=end_idx,
                restriction_mask=restriction_mask,
                blocked_mask_provider=blocked_mask_provider,
                blocked_mask_for_label_provider=blocked_mask_for_label_provider,
                use_box_percentiles=self.view_state_model.threshold_auto,
                use_ascan_max_in_roi=self.view_state_model.roi_ascan_max_mode,
                prefer_second_peak=self.view_state_model.roi_ascan_prefer_second_peak,
            )
            self.refresh_roi_overlay_for_slice(self.view_state_model.current_slice)
        else:
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
        """Handle mouse click in annotation view (grow tool or paint brush)."""
        if not isinstance(pos, (tuple, list)) or len(pos) != 2:
            return

        if self.view_state_model.tool_mode == "paint":
            self._handle_paint_click(pos)
            return

        if self.view_state_model.tool_mode != "grow":
            return
        if self.view_state_model.active_label is None:
            return
        point = (int(pos[0]), int(pos[1]))
        label = self.view_state_model.active_label
        threshold = self.view_state_model.threshold if self.view_state_model.threshold is not None else 0
        thin_line_width = self.view_state_model.roi_thin_line_max_width

        try:
            slice_idx = int(self.view_state_model.current_slice)
        except Exception:
            return

        shape = (
            self.annotation_model.mask_shape_hw()
            or self.temp_mask_model.mask_shape_hw()
        )
        slice_data = self._slice_data(slice_idx)
        if slice_data is not None and shape is None and slice_data.ndim >= 2:
            shape = (int(slice_data.shape[0]), int(slice_data.shape[1]))
        if slice_data is None or shape is None:
            return
        restriction_mask = self._restriction_mask(shape)
        if restriction_mask is not None and not restriction_mask[point[1], point[0]]:
            return
        if int(label) == 0:
            blocked_mask = self._build_label0_blocked_mask(
                slice_idx, shape, include_temp=True
            )
        else:
            blocked_mask = self._build_blocked_mask(slice_idx, shape, include_temp=True)

        if self.view_state_model.apply_volume and not self.view_state_model.roi_persistence:
            depth, _ = self._resolve_volume_dimensions()
            if depth is None:
                return
            start_idx, end_idx = self._resolve_apply_volume_range(depth)
            blocked_mask_provider = None
            if int(label) == 0:
                blocked_mask_provider = lambda idx, _shape=shape: self._build_label0_blocked_mask(
                    idx, _shape, include_temp=True
                )
            else:
                blocked_mask_provider = lambda idx, _shape=shape: self._build_blocked_mask(
                    idx, _shape, include_temp=True
                )
            self.annotation_service.propagate_grow_volume_from_slice(
                start_slice=slice_idx,
                point=point,
                shape=shape,
                threshold=threshold,
                label=label,
                depth=depth,
                roi_model=self.roi_model,
                temp_mask_model=self.temp_mask_model,
                palette=self.annotation_model.get_label_palette(),
                slice_data_provider=self._slice_data,
                thin_line_max_width=thin_line_width,
                start_idx=start_idx,
                end_idx=end_idx,
                restriction_mask=restriction_mask,
                blocked_mask_provider=blocked_mask_provider,
            )
        else:
            grow_mask = self.annotation_service.apply_grow_roi(
                slice_idx=slice_idx,
                point=point,
                shape=shape,
                slice_data=slice_data,
                label=label,
                threshold=threshold,
                thin_line_max_width=thin_line_width,
                persistent=self.view_state_model.roi_persistence,
                roi_model=self.roi_model,
                temp_mask_model=self.temp_mask_model,
                palette=self.annotation_model.get_label_palette(),
                restriction_mask=restriction_mask,
                blocked_mask=blocked_mask,
            )
            if grow_mask is None:
                return

        self.refresh_roi_overlay_for_slice(slice_idx)

    def on_annotation_line_drawn(self, points: Any) -> None:
        """Handle freehand line completion (line grow tool)."""
        if self.view_state_model.tool_mode != "line":
            return
        if self.view_state_model.active_label is None:
            return
        if not isinstance(points, (list, tuple)):
            return
        clean_points: list[tuple[int, int]] = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                continue
            try:
                clean_points.append((int(pt[0]), int(pt[1])))
            except Exception:
                continue
        if not clean_points:
            return

        label = self.view_state_model.active_label
        threshold = self.view_state_model.threshold if self.view_state_model.threshold is not None else 0
        thin_line_width = self.view_state_model.roi_thin_line_max_width

        try:
            slice_idx = int(self.view_state_model.current_slice)
        except Exception:
            return

        shape = (
            self.annotation_model.mask_shape_hw()
            or self.temp_mask_model.mask_shape_hw()
        )
        slice_data = self._slice_data(slice_idx)
        if slice_data is not None and shape is None and slice_data.ndim >= 2:
            shape = (int(slice_data.shape[0]), int(slice_data.shape[1]))
        if slice_data is None or shape is None:
            return
        restriction_mask = self._restriction_mask(shape)
        if int(label) == 0:
            blocked_mask = self._build_label0_blocked_mask(
                slice_idx, shape, include_temp=True
            )
        else:
            blocked_mask = self._build_blocked_mask(slice_idx, shape, include_temp=True)

        palette = self.annotation_model.get_label_palette()
        if self.view_state_model.apply_volume and not self.view_state_model.roi_persistence:
            depth, _ = self._resolve_volume_dimensions()
            if depth is None:
                return
            start_idx, end_idx = self._resolve_apply_volume_range(depth)
            blocked_mask_provider = None
            if int(label) == 0:
                blocked_mask_provider = lambda idx, _shape=shape: self._build_label0_blocked_mask(
                    idx, _shape, include_temp=True
                )
            else:
                blocked_mask_provider = lambda idx, _shape=shape: self._build_blocked_mask(
                    idx, _shape, include_temp=True
                )
            self.annotation_service.propagate_line_volume_from_slice(
                start_slice=slice_idx,
                points=clean_points,
                shape=shape,
                threshold=threshold,
                label=label,
                depth=depth,
                roi_model=self.roi_model,
                temp_mask_model=self.temp_mask_model,
                palette=palette,
                slice_data_provider=self._slice_data,
                thin_line_max_width=thin_line_width,
                start_idx=start_idx,
                end_idx=end_idx,
                restriction_mask=restriction_mask,
                blocked_mask_provider=blocked_mask_provider,
            )
        else:
            line_mask = self.annotation_service.apply_line_roi(
                slice_idx=slice_idx,
                points=clean_points,
                shape=shape,
                slice_data=slice_data,
                label=label,
                threshold=threshold,
                thin_line_max_width=thin_line_width,
                persistent=self.view_state_model.roi_persistence,
                roi_model=self.roi_model,
                temp_mask_model=self.temp_mask_model,
                palette=palette,
                restriction_mask=restriction_mask,
                blocked_mask=blocked_mask,
            )
            if line_mask is None:
                return

        self.refresh_roi_overlay_for_slice(slice_idx)

    def on_annotation_freehand_started(self, pos: Any) -> None:
        """Handle free-hand start."""
        if self.view_state_model.tool_mode != "free_hand":
            return
        if not isinstance(pos, (tuple, list)) or len(pos) != 2:
            return

    def on_annotation_freehand_point_added(self, pos: Any) -> None:
        """Handle free-hand point addition."""
        if self.view_state_model.tool_mode != "free_hand":
            return
        if not isinstance(pos, (tuple, list)) or len(pos) != 2:
            return

    def on_annotation_freehand_completed(self, points: Any) -> None:
        """Handle free-hand completion."""
        if self.view_state_model.tool_mode != "free_hand":
            return
        if self.view_state_model.active_label is None:
            return
        if not isinstance(points, (list, tuple)):
            return

        clean_points: list[tuple[int, int]] = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                continue
            try:
                clean_points.append((int(pt[0]), int(pt[1])))
            except Exception:
                continue
        if len(clean_points) < 3:
            return

        label = self.view_state_model.active_label
        try:
            slice_idx = int(self.view_state_model.current_slice)
        except Exception:
            return

        shape = self.annotation_model.mask_shape_hw() or self.temp_mask_model.mask_shape_hw()
        slice_data = self._slice_data(slice_idx)
        if slice_data is not None and shape is None and slice_data.ndim >= 2:
            shape = (int(slice_data.shape[0]), int(slice_data.shape[1]))
        if shape is None:
            _, shape = self._resolve_volume_dimensions()
        if shape is None:
            return

        restriction_mask = self._restriction_mask(shape)
        if int(label) == 0:
            blocked_mask = self._build_label0_blocked_mask(
                slice_idx, shape, include_temp=True
            )
        else:
            blocked_mask = self._build_blocked_mask(slice_idx, shape, include_temp=True)
        palette = self.annotation_model.get_label_palette()

        if self.view_state_model.apply_volume and not self.view_state_model.roi_persistence:
            depth, _ = self._resolve_volume_dimensions()
            if depth is None:
                return
            start_idx, end_idx = self._resolve_apply_volume_range(depth)
            blocked_mask_provider = None
            if int(label) == 0:
                blocked_mask_provider = lambda idx, _shape=shape: self._build_label0_blocked_mask(
                    idx, _shape, include_temp=True
                )
            else:
                blocked_mask_provider = lambda idx, _shape=shape: self._build_blocked_mask(
                    idx, _shape, include_temp=True
                )
            self.annotation_service.apply_free_hand_roi_to_range(
                start_idx=start_idx,
                end_idx=end_idx,
                points=clean_points,
                shape=shape,
                label=label,
                threshold=self.view_state_model.threshold,
                persistent=self.view_state_model.roi_persistence,
                roi_model=self.roi_model,
                temp_mask_model=self.temp_mask_model,
                palette=palette,
                slice_data_provider=self._slice_data,
                restriction_mask=restriction_mask,
                blocked_mask_provider=blocked_mask_provider,
                use_box_percentiles=self.view_state_model.threshold_auto,
                use_ascan_max_in_roi=self.view_state_model.roi_ascan_max_mode,
                prefer_second_peak=self.view_state_model.roi_ascan_prefer_second_peak,
            )
        else:
            free_hand_mask = self.annotation_service.apply_free_hand_roi(
                slice_idx=slice_idx,
                points=clean_points,
                shape=shape,
                label=label,
                threshold=self.view_state_model.threshold,
                persistent=self.view_state_model.roi_persistence,
                roi_model=self.roi_model,
                temp_mask_model=self.temp_mask_model,
                palette=palette,
                slice_data=slice_data,
                restriction_mask=restriction_mask,
                blocked_mask=blocked_mask,
                use_box_percentiles=self.view_state_model.threshold_auto,
                use_ascan_max_in_roi=self.view_state_model.roi_ascan_max_mode,
                prefer_second_peak=self.view_state_model.roi_ascan_prefer_second_peak,
            )
            if free_hand_mask is None:
                return

        self.refresh_roi_overlay_for_slice(slice_idx)

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

        box_tuple = self._normalize_rect_input(box)
        if box_tuple is None:
            return
        restriction_rect = self.view_state_model.restriction_rect
        if restriction_rect is not None:
            box_tuple = self._intersect_rects(box_tuple, restriction_rect)
            if box_tuple is None:
                return
        restriction_mask = self._restriction_mask((h, w))
        if int(label) == 0:
            blocked_mask = self._build_label0_blocked_mask(
                slice_idx, (h, w), include_temp=True
            )
        else:
            blocked_mask = self._build_blocked_mask(slice_idx, (h, w), include_temp=True)
        palette = self.annotation_model.get_label_palette()
        if self.view_state_model.apply_volume and not self.view_state_model.roi_persistence:
            depth = mask_volume.shape[0]
            start_idx, end_idx = self._resolve_apply_volume_range(depth)
            blocked_mask_provider = None
            if int(label) == 0:
                blocked_mask_provider = lambda idx, _shape=(h, w): self._build_label0_blocked_mask(
                    idx, _shape, include_temp=True
                )
            else:
                blocked_mask_provider = lambda idx, _shape=(h, w): self._build_blocked_mask(
                    idx, _shape, include_temp=True
                )
            self.annotation_service.apply_box_roi_to_range(
                start_idx=start_idx,
                end_idx=end_idx,
                box=box_tuple,
                shape=(h, w),
                label=label,
                threshold=self.view_state_model.threshold,
                persistent=self.view_state_model.roi_persistence,
                roi_model=self.roi_model,
                temp_mask_model=self.temp_mask_model,
                palette=palette,
                slice_data_provider=self._slice_data,
                restriction_mask=restriction_mask,
                blocked_mask_provider=blocked_mask_provider,
                use_box_percentiles=self.view_state_model.threshold_auto,
                use_ascan_max_in_roi=self.view_state_model.roi_ascan_max_mode,
                prefer_second_peak=self.view_state_model.roi_ascan_prefer_second_peak,
            )
        else:
            self.annotation_service.apply_box_roi(
                slice_idx=slice_idx,
                box=box_tuple,
                shape=(h, w),
                label=label,
                threshold=self.view_state_model.threshold,
                persistent=self.view_state_model.roi_persistence,
                roi_model=self.roi_model,
                temp_mask_model=self.temp_mask_model,
                palette=palette,
                slice_data=self._slice_data(slice_idx),
                restriction_mask=restriction_mask,
                blocked_mask=blocked_mask,
                use_box_percentiles=self.view_state_model.threshold_auto,
                use_ascan_max_in_roi=self.view_state_model.roi_ascan_max_mode,
                prefer_second_peak=self.view_state_model.roi_ascan_prefer_second_peak,
            )
        self.refresh_roi_overlay_for_slice(slice_idx)

    def _handle_paint_click(self, pos: tuple[Any, Any]) -> None:
        """Paint the active label (including 0) into the temp mask (requires Apply)."""
        if self.view_state_model.active_label is None:
            return
        label = int(self.view_state_model.active_label)
        try:
            slice_idx = int(self.view_state_model.current_slice)
        except Exception:
            return

        # Determine shape/depth
        mask_shape = self.annotation_model.mask_shape_hw() or self.temp_mask_model.mask_shape_hw()
        if mask_shape is None:
            slice_data = self._slice_data(slice_idx)
            if slice_data is not None and slice_data.ndim >= 2:
                mask_shape = (int(slice_data.shape[0]), int(slice_data.shape[1]))
        depth, _ = self._resolve_volume_dimensions()
        if depth is None or mask_shape is None:
            return
        restriction_mask = self._restriction_mask(mask_shape)
        if restriction_mask is not None and not restriction_mask[int(pos[1]), int(pos[0])]:
            return

        # Ensure temp volume exists and matches shape
        temp_vol = self.temp_mask_model.get_mask_volume()
        need_init = False
        if temp_vol is None:
            need_init = True
        else:
            th, tw = temp_vol.shape[1], temp_vol.shape[2]
            if (th, tw) != (mask_shape[0], mask_shape[1]) or temp_vol.shape[0] != depth:
                need_init = True
        if need_init:
            self.temp_mask_model.initialize((depth, mask_shape[0], mask_shape[1]))

        radius = self.view_state_model.paint_radius or self.PAINT_RADIUS_DEFAULT
        effective_radius = max(0, int(radius) - 1)
        disk = self.annotation_service.build_disk_mask(mask_shape, (int(pos[0]), int(pos[1])), effective_radius)
        if disk is None:
            return
        if restriction_mask is not None:
            disk = np.logical_and(disk > 0, restriction_mask).astype(np.uint8)
            if not np.any(disk):
                return
        if label == 0:
            blocked_mask = self._build_label0_blocked_mask(
                slice_idx, mask_shape, include_temp=True
            )
        else:
            blocked_mask = self._build_blocked_mask(slice_idx, mask_shape, include_temp=True)
        if blocked_mask is not None:
            disk = np.where(blocked_mask, 0, disk)
            if not np.any(disk):
                return

        color = self.annotation_model.get_label_palette().get(label) or MASK_COLORS_BGRA.get(label, (255, 0, 255, 160))
        self.temp_mask_model.ensure_label(label, color, visible=True)
        self.temp_mask_model.set_slice_mask(slice_idx, disk, label=label, persistent=False)
        self.refresh_roi_overlay_for_slice(slice_idx)

    def refresh_roi_overlay_for_slice(self, slice_idx: int) -> None:
        """Refresh ROI preview overlay for the given slice."""
        slice_mask = self.temp_mask_model.get_slice_mask(slice_idx)
        coverage = self.temp_mask_model.get_slice_coverage(slice_idx)
        palette = dict(self.temp_mask_model.label_palette)
        overlay_mask = None
        if slice_mask is not None and coverage is not None and np.any(coverage):
            overlay_mask = np.array(slice_mask, copy=True)
            zero_area = coverage & (overlay_mask == 0)
            if np.any(zero_area):
                overlay_mask[zero_area] = 255
                if 0 in palette:
                    palette[255] = palette[0]
                else:
                    palette[255] = MASK_COLORS_BGRA.get(0, (180, 180, 180, 160))

        if overlay_mask is None:
            self.annotation_view.clear_roi_overlay()
        else:
            self.annotation_view.set_roi_overlay(overlay_mask, palette=palette)

        boxes = self.roi_model.boxes_for_slice(slice_idx, include_persistent=True)
        if boxes:
            self.annotation_view.set_roi_boxes(boxes)
        else:
            self.annotation_view.clear_roi_boxes()
        restriction_mask = None
        if slice_mask is not None:
            restriction_mask = self._restriction_mask(slice_mask.shape)
        seeds = self.roi_model.seeds_for_slice(slice_idx, include_persistent=True)
        seeds = self._filter_points_by_restriction(seeds, restriction_mask)
        if seeds:
            self.annotation_view.set_roi_points(seeds)
        else:
            self.annotation_view.clear_roi_points()

    def on_apply_temp_mask_requested(self) -> None:
        """Apply the current temporary mask (free-hand/ROI) into the annotation model."""
        if self.view_state_model.apply_volume:
            depth, mask_shape = self._resolve_volume_dimensions()
            if depth is None or mask_shape is None:
                return
            start_idx, end_idx = self._resolve_apply_volume_range(depth)
            restriction_mask = self._restriction_mask(mask_shape)
            # Preserve existing temp (e.g., paint) before rebuild from ROIs
            prev_temp = self.temp_mask_model.get_mask_volume()
            prev_cov = self.temp_mask_model.get_coverage_volume()
            blocked_mask_provider = lambda idx, _shape=mask_shape: self._build_blocked_mask(
                idx, _shape, include_temp=False
            )
            blocked_mask_for_label_provider = lambda idx, label_id, _shape=mask_shape: (
                self._build_label0_blocked_mask(idx, _shape, include_temp=True)
                if int(label_id) == 0
                else None
            )
            self.annotation_service.rebuild_volume_preview_from_rois(
                depth=depth,
                mask_shape=mask_shape,
                roi_model=self.roi_model,
                temp_mask_model=self.temp_mask_model,
                palette=self.annotation_model.get_label_palette(),
                slice_data_provider=self._slice_data,
                thin_line_max_width=self.view_state_model.roi_thin_line_max_width,
                start_idx=start_idx,
                end_idx=end_idx,
                restriction_mask=restriction_mask,
                blocked_mask_provider=blocked_mask_provider,
                blocked_mask_for_label_provider=blocked_mask_for_label_provider,
                use_box_percentiles=self.view_state_model.threshold_auto,
                use_ascan_max_in_roi=self.view_state_model.roi_ascan_max_mode,
                prefer_second_peak=self.view_state_model.roi_ascan_prefer_second_peak,
            )
            if prev_temp is not None and prev_cov is not None:
                new_temp = self.temp_mask_model.get_mask_volume()
                new_cov = self.temp_mask_model.get_coverage_volume()
                if (
                    new_temp is not None
                    and new_cov is not None
                    and new_temp.shape == prev_temp.shape
                    and new_cov.shape == prev_cov.shape
                ):
                    merged = np.array(new_temp, copy=True)
                    merged_cov = np.array(new_cov, copy=True)
                    merged[prev_cov] = prev_temp[prev_cov]
                    merged_cov = np.logical_or(merged_cov, prev_cov)
                    self.temp_mask_model.mask_volume = merged  # type: ignore[attr-defined]
                    self.temp_mask_model.coverage_volume = merged_cov  # type: ignore[attr-defined]
            self.annotation_service.apply_temp_volume_to_model(
                temp_mask_model=self.temp_mask_model,
                annotation_model=self.annotation_model,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            target_slice = self.view_state_model.current_slice
        else:
            target_slice = self.view_state_model.current_slice
            mask_volume = self.annotation_model.get_mask_volume()
            temp_slice = self.temp_mask_model.get_slice_mask(target_slice)
            coverage = self.temp_mask_model.get_slice_coverage(target_slice)

            if mask_volume is None or temp_slice is None or coverage is None:
                return
            if not np.any(coverage):
                return
            try:
                current_slice = mask_volume[int(target_slice)]
            except Exception:
                return
            if current_slice.shape != temp_slice.shape:
                return

            updated = np.array(current_slice, copy=True)
            updated[coverage] = temp_slice[coverage]
            self.annotation_model.set_slice_mask(
                target_slice,
                updated,
                invalidate_cache=False,
            )

        for label, color in self.temp_mask_model.label_palette.items():
            self.annotation_model.ensure_label(label, color, visible=True)

        if self.view_state_model.apply_volume:
            self.temp_mask_model.clear()
        else:
            self.temp_mask_model.clear_slice(target_slice)
        if not self.view_state_model.roi_persistence:
            # Clear transient ROIs once applied to the mask
            self.roi_model.clear_non_persistent()

        self.annotation_view.clear_temp_shapes()
        self.annotation_view.clear_roi_overlay()
        self.annotation_view.clear_roi_boxes()
        self.annotation_view.clear_roi_points()

        self.refresh_overlay(
            defer_volume=True,
            rebuild=self.view_state_model.apply_volume,
            changed_slice=None if self.view_state_model.apply_volume else target_slice,
        )
        self.refresh_roi_overlay_for_slice(target_slice)

    def on_apply_all_temp_masks_requested(self) -> None:
        """Apply temp masks across the whole volume regardless of the current slice mode."""
        prev_apply_volume = self.view_state_model.apply_volume
        prev_range = (self.view_state_model.apply_volume_start, self.view_state_model.apply_volume_end)
        self.view_state_model.set_apply_volume(True)
        depth, _ = self._resolve_volume_dimensions()
        if depth is not None and depth > 0:
            self.view_state_model.set_apply_volume_range(0, depth - 1, include_current=False)
        try:
            self.on_apply_temp_mask_requested()
        finally:
            self.view_state_model.set_apply_volume(prev_apply_volume)
            self.view_state_model.apply_volume_start = prev_range[0]
            self.view_state_model.apply_volume_end = prev_range[1]

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

        options_dialog = OverlayExportDialog(parent)
        result = options_dialog.exec()
        if result != int(QDialog.DialogCode.Accepted):
            return None
        options = options_dialog.get_options()

        file_path = self.annotation_view.select_overlay_save_path(parent)
        if not file_path:
            return None

        saved_path = self.overlay_export.save_npz(
            mask_volume,
            file_path,
            expected_shape=volume_shape,
            mirror_vertical=options.mirror_vertical,
            rotation_degrees=options.rotation_degrees,
        )
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
            qcolor = self.overlay_settings_view.bgra_to_qcolor(color)
            visible = visibility.get(label_id, True)
            entries.append((label_id, qcolor, visible))
        self.overlay_settings_view.set_labels(entries)
        self.overlay_settings_view.set_overlay_opacity(self.view_state_model.overlay_alpha)

    def _resolve_volume_dimensions(self) -> tuple[Optional[int], Optional[tuple[int, int]]]:
        """Return (depth, (H, W)) from annotation/temp models or underlying volume."""
        mask_shape = self.annotation_model.mask_shape_hw() or self.temp_mask_model.mask_shape_hw()
        depth = None

        mask_volume = self.annotation_model.get_mask_volume()
        if mask_volume is not None:
            depth = mask_volume.shape[0]
            if mask_shape is None:
                mask_shape = (mask_volume.shape[1], mask_volume.shape[2])

        temp_volume = self.temp_mask_model.get_mask_volume()
        if depth is None and temp_volume is not None:
            depth = temp_volume.shape[0]
            if mask_shape is None:
                mask_shape = (temp_volume.shape[1], temp_volume.shape[2])

        if (depth is None or mask_shape is None) and self._get_volume is not None:
            volume = self._get_volume()
            if volume is not None:
                depth = volume.shape[0]
                if mask_shape is None:
                    mask_shape = (volume.shape[1], volume.shape[2])

        return depth, mask_shape

    def _resolve_apply_volume_range(self, depth: int) -> tuple[int, int]:
        """Clamp apply-to-volume range to depth and include current slice."""
        max_idx = max(0, int(depth) - 1)
        start_idx = int(getattr(self.view_state_model, "apply_volume_start", 0))
        end_idx = int(getattr(self.view_state_model, "apply_volume_end", max_idx))
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        start_idx = max(0, min(max_idx, start_idx))
        end_idx = max(0, min(max_idx, end_idx))
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        current = int(self.view_state_model.current_slice)
        if current < start_idx:
            start_idx = current
        elif current > end_idx:
            end_idx = current
        start_idx = max(0, min(max_idx, start_idx))
        end_idx = max(0, min(max_idx, end_idx))
        return start_idx, end_idx

    @staticmethod
    def _normalize_rect_input(rect: Any) -> Optional[tuple[int, int, int, int]]:
        if rect is None:
            return None
        if isinstance(rect, (list, tuple)):
            if len(rect) == 4 and all(isinstance(v, (int, float)) for v in rect):
                x1, y1, x2, y2 = rect
                return (int(x1), int(y1), int(x2), int(y2))
            if (
                len(rect) == 2
                and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in rect)
            ):
                (x1, y1), (x2, y2) = rect
                return (int(x1), int(y1), int(x2), int(y2))
        return None

    @staticmethod
    def _intersect_rects(
        rect_a: tuple[int, int, int, int],
        rect_b: tuple[int, int, int, int],
    ) -> Optional[tuple[int, int, int, int]]:
        ax1, ay1, ax2, ay2 = rect_a
        bx1, by1, bx2, by2 = rect_b
        a_xmin, a_xmax = sorted((ax1, ax2))
        a_ymin, a_ymax = sorted((ay1, ay2))
        b_xmin, b_xmax = sorted((bx1, bx2))
        b_ymin, b_ymax = sorted((by1, by2))
        xmin = max(a_xmin, b_xmin)
        xmax = min(a_xmax, b_xmax)
        ymin = max(a_ymin, b_ymin)
        ymax = min(a_ymax, b_ymax)
        if xmax < xmin or ymax < ymin:
            return None
        return (int(xmin), int(ymin), int(xmax), int(ymax))

    def _clamp_rect_to_shape(
        self,
        rect: tuple[int, int, int, int],
        shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        height, width = int(shape[0]), int(shape[1])
        x1, y1, x2, y2 = rect
        xmin, xmax = sorted((int(x1), int(x2)))
        ymin, ymax = sorted((int(y1), int(y2)))
        if width <= 0 or height <= 0:
            return (xmin, ymin, xmax, ymax)
        xmin = max(0, min(width - 1, xmin))
        xmax = max(0, min(width - 1, xmax))
        ymin = max(0, min(height - 1, ymin))
        ymax = max(0, min(height - 1, ymax))
        min_w = min(self.RESTRICTION_MIN_SIZE, max(0, width - 1))
        min_h = min(self.RESTRICTION_MIN_SIZE, max(0, height - 1))
        if xmax - xmin < min_w:
            xmax = min(width - 1, xmin + min_w)
            if xmax - xmin < min_w:
                xmin = max(0, xmax - min_w)
        if ymax - ymin < min_h:
            ymax = min(height - 1, ymin + min_h)
            if ymax - ymin < min_h:
                ymin = max(0, ymax - min_h)
        return (int(xmin), int(ymin), int(xmax), int(ymax))

    def _restriction_mask(self, shape: tuple[int, int]) -> Optional[np.ndarray]:
        rect = self.view_state_model.restriction_rect
        if rect is None:
            return None
        mask = self.annotation_service.build_box_mask(shape, rect)
        if mask is None:
            return None
        return mask.astype(bool)

    def _build_blocked_mask(
        self,
        slice_idx: int,
        shape: tuple[int, int],
        *,
        include_temp: bool = True,
    ) -> Optional[np.ndarray]:
        """Return a boolean mask of existing labels (annotation + optional temp)."""
        try:
            h, w = int(shape[0]), int(shape[1])
        except Exception:
            return None
        blocked = None

        mask_volume = self.annotation_model.get_mask_volume()
        if mask_volume is not None:
            try:
                ann_slice = mask_volume[int(slice_idx)]
            except Exception:
                ann_slice = None
            if ann_slice is not None and ann_slice.shape == (h, w):
                blocked = ann_slice > 0

        if include_temp:
            coverage = self.temp_mask_model.get_slice_coverage(slice_idx)
            if coverage is not None and coverage.shape == (h, w):
                blocked = coverage if blocked is None else np.logical_or(blocked, coverage)

        if blocked is None or not np.any(blocked):
            return None
        return blocked

    def _build_label0_blocked_mask(
        self,
        slice_idx: int,
        shape: tuple[int, int],
        *,
        include_temp: bool = True,
    ) -> Optional[np.ndarray]:
        """Return a blocked mask for label 0 erasing only a target label."""
        target = getattr(self.view_state_model, "label0_erase_target", None)
        if target is None:
            return None
        try:
            target = int(target)
        except Exception:
            return None
        try:
            h, w = int(shape[0]), int(shape[1])
        except Exception:
            return None
        if h <= 0 or w <= 0:
            return None

        allowed = None
        mask_volume = self.annotation_model.get_mask_volume()
        if mask_volume is not None:
            try:
                ann_slice = mask_volume[int(slice_idx)]
            except Exception:
                ann_slice = None
            if ann_slice is not None and ann_slice.shape == (h, w):
                allowed = ann_slice == target

        if include_temp:
            temp_slice = self.temp_mask_model.get_slice_mask(slice_idx)
            if temp_slice is not None and temp_slice.shape == (h, w):
                temp_allowed = temp_slice == target
                allowed = temp_allowed if allowed is None else np.logical_or(allowed, temp_allowed)

        if allowed is None:
            return np.ones((h, w), dtype=bool)

        return np.logical_not(allowed)

    @staticmethod
    def _filter_points_by_restriction(
        points: list[tuple[int, int]],
        restriction_mask: Optional[np.ndarray],
    ) -> list[tuple[int, int]]:
        if restriction_mask is None:
            return points
        h, w = restriction_mask.shape
        filtered: list[tuple[int, int]] = []
        for x, y in points:
            if 0 <= x < w and 0 <= y < h and restriction_mask[y, x]:
                filtered.append((int(x), int(y)))
        return filtered

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

    def _compute_changed_labels_for_slice(
        self,
        *,
        cached_overlay: Optional[OverlayData],
        new_overlay: Optional[OverlayData],
        slice_idx: int,
    ) -> Optional[set[int]]:
        """Compare cached/new overlay data on a slice to limit GPU uploads."""
        if new_overlay is None or new_overlay.label_volumes is None:
            return None
        if slice_idx < 0:
            return None
        changed: set[int] = set()
        cached_volumes = cached_overlay.label_volumes if cached_overlay is not None else {}
        new_volumes = new_overlay.label_volumes if new_overlay is not None else {}

        for label, vol in new_volumes.items():
            try:
                new_slice = np.asarray(vol, dtype=np.float32)[int(slice_idx)]
            except Exception:
                changed.add(int(label))
                continue
            old_vol = cached_volumes.get(label)
            if old_vol is None:
                changed.add(int(label))
                continue
            try:
                old_slice = np.asarray(old_vol, dtype=np.float32)[int(slice_idx)]
            except Exception:
                changed.add(int(label))
                continue
            if old_slice.shape != new_slice.shape or not np.array_equal(old_slice, new_slice):
                changed.add(int(label))

        for label in cached_volumes.keys():
            if label not in new_volumes:
                changed.add(int(label))

        return changed if changed else None

    def _rebuild_slice_preview(self, slice_idx: int) -> None:
        """Rebuild temporary mask and box for a given slice from stored ROIs."""
        mask_shape = self.annotation_model.mask_shape_hw() or self.temp_mask_model.mask_shape_hw()
        if mask_shape is None:
            return

        rois = list(self.roi_model.list_on_slice(slice_idx))
        persistent_rois = self.roi_model.list_persistent()
        if persistent_rois:
            seen_ids = {roi.id for roi in rois}
            rois.extend([roi for roi in persistent_rois if roi.id not in seen_ids])

        blocked_mask = self._build_blocked_mask(
            slice_idx,
            mask_shape,
            include_temp=False,
        )
        blocked_mask_for_label = lambda label_id, _shape=mask_shape: (
            self._build_label0_blocked_mask(slice_idx, _shape, include_temp=True)
            if int(label_id) == 0
            else None
        )
        boxes = self.annotation_service.rebuild_temp_masks_for_slice(
            rois=rois,
            shape=mask_shape,
            slice_idx=slice_idx,
            temp_mask_model=self.temp_mask_model,
            palette=self.annotation_model.get_label_palette(),
            clear_slice=True,
            slice_data=self._slice_data(slice_idx),
            thin_line_max_width=self.view_state_model.roi_thin_line_max_width,
            restriction_mask=self._restriction_mask(mask_shape),
            blocked_mask=blocked_mask,
            blocked_mask_for_label=blocked_mask_for_label,
            use_box_percentiles=self.view_state_model.threshold_auto,
            use_ascan_max_in_roi=self.view_state_model.roi_ascan_max_mode,
            prefer_second_peak=self.view_state_model.roi_ascan_prefer_second_peak,
        )

        slice_mask = self.temp_mask_model.get_slice_mask(slice_idx)
        coverage = self.temp_mask_model.get_slice_coverage(slice_idx)
        palette = dict(self.temp_mask_model.label_palette)
        overlay_mask = None
        if slice_mask is not None and coverage is not None and np.any(coverage):
            overlay_mask = np.array(slice_mask, copy=True)
            zero_area = coverage & (overlay_mask == 0)
            if np.any(zero_area):
                overlay_mask[zero_area] = 255
                if 0 in palette:
                    palette[255] = palette[0]
                else:
                    palette[255] = MASK_COLORS_BGRA.get(0, (180, 180, 180, 160))

        if overlay_mask is not None:
            self.annotation_view.set_roi_overlay(overlay_mask, palette=palette)
        else:
            self.annotation_view.clear_roi_overlay()

        if boxes:
            self.annotation_view.set_roi_boxes(boxes)
        else:
            self.annotation_view.clear_roi_boxes()

        seeds = self.roi_model.seeds_for_slice(slice_idx, include_persistent=True)
        seeds = self._filter_points_by_restriction(seeds, self._restriction_mask(mask_shape))
        if seeds:
            self.annotation_view.set_roi_points(seeds)
        else:
            self.annotation_view.clear_roi_points()

