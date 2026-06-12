"""Controller dédié à la zone C-scan (standard + corrosion)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PyQt6.QtWidgets import QStackedLayout

from models.annotation_model import AnnotationModel
from models.nde_model import NdeModel
from models.view_state_model import ViewStateModel
from services.cscan_corrosion_service import (
    CScanCorrosionService,
    CorrosionWorkflowResult,
    CorrosionWorkflowService,
)
from services.cscan_service import CScanService
from views.cscan_view import CScanView
from views.cscan_view_corrosion import CscanViewCorrosion


@dataclass(frozen=True)
class CScanLayerChoice:
    """One selectable layer source for the C-scan display."""

    layer_id: str
    name: str
    layer_kind: str = "annotation"
    is_active: bool = False
    corrosion_projection: Optional[Tuple[Any, Tuple[float, float]]] = None


class CScanController:
    """Gère la pile de vues C-scan et le workflow d'analyse corrosion."""

    def __init__(
        self,
        *,
        standard_view: Optional[CScanView],
        corrosion_view: Optional[CscanViewCorrosion],
        stacked_layout: Optional[QStackedLayout],
        view_state_model: ViewStateModel,
        annotation_model: AnnotationModel,
        get_volume: Callable[[], Optional[np.ndarray]],
        get_nde_model: Callable[[], Optional[NdeModel]],
        status_callback: Callable[[str, int], None],
        logger: logging.Logger,
        get_layer_choices: Optional[Callable[[], list[CScanLayerChoice]]] = None,
        corrosion_workflow_service: Optional[CorrosionWorkflowService] = None,
        on_corrosion_completed: Optional[Callable[[CorrosionWorkflowResult], None]] = None,
    ) -> None:
        self.standard_view: Optional[CScanView] = standard_view
        self.corrosion_view: Optional[CscanViewCorrosion] = corrosion_view
        self._stack: Optional[QStackedLayout] = stacked_layout
        self.view_state_model = view_state_model
        self.annotation_model = annotation_model
        self.get_volume = get_volume
        self.get_nde_model = get_nde_model
        self.status_callback = status_callback
        self.logger = logger
        self.get_layer_choices = get_layer_choices or (lambda: [])

        self.last_corrosion_result: Optional[CorrosionWorkflowResult] = None
        self.on_corrosion_completed = on_corrosion_completed
        self._displaying_corrosion: bool = False
        self._displayed_corrosion_projection: Optional[Tuple[Any, Tuple[float, float]]] = None

        self.cscan_service = CScanService()
        self.corrosion_service = CScanCorrosionService()

        if corrosion_workflow_service is None:
            corrosion_workflow_service = CorrosionWorkflowService(
                cscan_corrosion_service=self.corrosion_service
            )
        else:
            self.corrosion_service = corrosion_workflow_service.cscan_corrosion_service
        self.corrosion_workflow = corrosion_workflow_service

        self._cached_standard_projection: Optional[np.ndarray] = None
        self._cached_standard_range: Optional[Tuple[float, float]] = None
        self._cached_volume_shape: Optional[Tuple[int, int, int]] = None
        self._bind_layer_selector_signals()

    # --- Stack & visibility ---------------------------------------------------------
    def show_standard(self) -> None:
        if self._stack is not None and self.standard_view is not None:
            self._stack.setCurrentWidget(self.standard_view)

    def show_corrosion(self) -> None:
        if self._stack is not None and self.corrosion_view is not None:
            self._stack.setCurrentWidget(self.corrosion_view)

    def is_showing_corrosion(self) -> bool:
        return bool(self._displaying_corrosion)

    # --- Crosshair helpers ---------------------------------------------------------
    def highlight_slice(self, slice_idx: int) -> None:
        if self.standard_view is not None:
            self.standard_view.highlight_slice(slice_idx)
        if self.corrosion_view is not None:
            self.corrosion_view.highlight_slice(slice_idx)

    def set_crosshair(self, slice_idx: int, x: int) -> None:
        if self.standard_view is not None:
            self.standard_view.set_crosshair(slice_idx, x)
        if self.corrosion_view is not None:
            self.corrosion_view.set_crosshair(slice_idx, x)

    def set_cross_visible(self, visible: bool) -> None:
        if self.standard_view is not None:
            self.standard_view.set_cross_visible(visible)
        if self.corrosion_view is not None:
            self.corrosion_view.set_cross_visible(visible)

    def set_ruler_axis_names(self, *, horizontal: str, vertical: str) -> None:
        """Push displayed X/Y axis names into the C-scan rulers."""
        if self.standard_view is not None:
            self.standard_view.set_ruler_axis_names(
                horizontal=horizontal,
                vertical=vertical,
            )
        if self.corrosion_view is not None:
            self.corrosion_view.set_ruler_axis_names(
                horizontal=horizontal,
                vertical=vertical,
            )

    def set_ruler_display_unit(self, display_unit: Optional[str]) -> None:
        """Push the shared ruler display unit into both C-scan views."""
        if self.standard_view is not None:
            self.standard_view.set_ruler_display_unit(display_unit)
        if self.corrosion_view is not None:
            self.corrosion_view.set_ruler_display_unit(display_unit)

    def set_ruler_axis_resolutions_mm(
        self,
        *,
        horizontal: Optional[float],
        vertical: Optional[float],
    ) -> None:
        """Push per-axis mm calibration into both C-scan views."""
        if self.standard_view is not None:
            self.standard_view.set_ruler_axis_resolutions_mm(
                horizontal_resolution_mm=horizontal,
                vertical_resolution_mm=vertical,
            )
        if self.corrosion_view is not None:
            self.corrosion_view.set_ruler_axis_resolutions_mm(
                horizontal_resolution_mm=horizontal,
                vertical_resolution_mm=vertical,
            )

    def set_colormap(self, name: str, lut: Optional[np.ndarray]) -> None:
        """Apply colormap on the standard C-scan view."""
        if self.standard_view is not None:
            self.standard_view.set_colormap(name, lut)

    def sync_layer_choices(self) -> None:
        choices = self._current_layer_choices()
        selected = self._selected_layer_choice(choices)
        selected_id = selected.layer_id if selected is not None else None
        self._sync_layer_selectors(choices, selected_id)

    def reset_display_size(self) -> None:
        if self.standard_view is not None:
            self.standard_view.reset_display_size()
        if self.corrosion_view is not None:
            self.corrosion_view.reset_display_size()

    def set_display_size(self, width: int, height: int) -> None:
        w = int(width)
        h = int(height)
        if self.standard_view is not None:
            self.standard_view.set_display_size(w, h)
        if self.corrosion_view is not None:
            self.corrosion_view.set_display_size(w, h)

    def on_crosshair_changed(
        self,
        *,
        slice_idx: int,
        x: int,
        volume_shape: Tuple[int, int, int],
        current_point: Optional[Tuple[int, int]],
    ) -> tuple[int, int]:
        """Handle crosshair move from C-scan and update shared state.

        Returns:
            The point (x, y) to use for A-scan refresh.
        """
        self.view_state_model.set_slice(int(slice_idx))
        clamped_slice = int(self.view_state_model.current_slice)
        self.highlight_slice(clamped_slice)
        self.set_crosshair(clamped_slice, int(x))

        y_default = int(volume_shape[1]) // 2 if len(volume_shape) >= 2 else 0
        current_y = int(current_point[1]) if current_point is not None else y_default
        self.view_state_model.update_crosshair(int(x), current_y)
        y = int(self.view_state_model.current_point[1]) if self.view_state_model.current_point else y_default
        return int(x), y

    # --- Update projections --------------------------------------------------------
    def update_views(self, volume: Optional[np.ndarray], *, preserve_view: bool = False) -> None:
        """Met à jour la projection standard et corrosion selon l'état courant."""
        if volume is None:
            self.sync_layer_choices()
            return

        choices = self._current_layer_choices()
        selected_choice = self._selected_layer_choice(choices)
        selected_id = selected_choice.layer_id if selected_choice is not None else None
        self._sync_layer_selectors(choices, selected_id)
        navigation_state = self._capture_current_navigation_state() if preserve_view else None

        nde_model = self.get_nde_model()
        self.set_ruler_display_unit(getattr(self.view_state_model, "ruler_display_unit", "px"))
        ultrasound_resolution_mm = (
            nde_model.get_axis_resolution_mm("Ultrasound", fallback_axis_index=1)
            if nde_model is not None
            else None
        )

        use_cache = (
            self._cached_standard_projection is not None
            and self._cached_standard_range is not None
            and self._cached_volume_shape == tuple(volume.shape)
        )
        if use_cache:
            standard_projection, standard_range = (
                self._cached_standard_projection,
                self._cached_standard_range,
            )
        else:
            standard_projection, standard_range = self.cscan_service.compute_top_projection(volume)
            self._cached_standard_projection = standard_projection
            self._cached_standard_range = standard_range
            self._cached_volume_shape = tuple(volume.shape)

        corrosion_payload = (
            selected_choice.corrosion_projection
            if selected_choice is not None
            and str(selected_choice.layer_kind).strip().casefold() == "corrosion"
            else None
        )
        if corrosion_payload is not None and self.corrosion_view is not None:
            projection, value_range = corrosion_payload
            self._displaying_corrosion = True
            self._displayed_corrosion_projection = corrosion_payload
            self.show_corrosion()
            self.corrosion_view.set_projection(
                projection,
                value_range,
                value_scale_mm=ultrasound_resolution_mm,
                preserve_view=preserve_view,
            )
            if navigation_state is not None:
                self.corrosion_view.restore_navigation_state(navigation_state)
        else:
            self._displaying_corrosion = False
            self._displayed_corrosion_projection = None
            self.show_standard()
            if self.standard_view is not None:
                self.standard_view.set_projection(
                    standard_projection,
                    standard_range,
                    preserve_view=preserve_view,
                )
                if navigation_state is not None:
                    self.standard_view.restore_navigation_state(navigation_state)

    # --- Corrosion workflow --------------------------------------------------------
    def reset_corrosion(self) -> None:
        self.view_state_model.deactivate_corrosion()
        self._displaying_corrosion = False
        self._displayed_corrosion_projection = None
        self._cached_standard_projection = None
        self._cached_standard_range = None
        self._cached_volume_shape = None

    def run_corrosion_analysis(self) -> None:
        """Execute corrosion analysis using the selected label pair."""
        volume = self.get_volume()
        nde_model = self.get_nde_model()
        if volume is None or nde_model is None:
            self.logger.error("Corrosion analysis aborted: volume or NDE model missing.")
            self.view_state_model.deactivate_corrosion()
            return

        self.logger.info("Corrosion analysis: started")
        self.status_callback("Corrosion analysis in progress...", 2000)
        label_a = getattr(self.view_state_model, "corrosion_label_a", None)
        label_b = getattr(self.view_state_model, "corrosion_label_b", None)
        analysis_mode = self.view_state_model.get_corrosion_analysis_mode()
        peak_selection_mode_a = self.view_state_model.get_corrosion_peak_selection_mode_a()
        peak_selection_mode_b = self.view_state_model.get_corrosion_peak_selection_mode_b()
        result = self.corrosion_workflow.run(
            nde_model=nde_model,
            annotation_model=self.annotation_model,
            volume=volume,
            label_a=label_a,
            label_b=label_b,
            analysis_mode=analysis_mode,
            peak_selection_mode_a=peak_selection_mode_a,
            peak_selection_mode_b=peak_selection_mode_b,
        )
        self.last_corrosion_result = result

        if not result.ok:
            self.logger.error(result.message)
            self.status_callback(result.message, 5000)
            self.view_state_model.deactivate_corrosion()
            return

        self.view_state_model.activate_corrosion(result.projection, result.value_range)
        self.update_views(volume)
        self.status_callback(result.message, 3000)
        self.logger.info("Corrosion analysis: completed")

        if self.on_corrosion_completed is not None:
            try:
                self.on_corrosion_completed(result)
            except Exception:  # noqa: BLE001
                self.logger.exception("Corrosion completion callback failed")

    def export_corrosion_projection(self, *, output_directory: str) -> tuple[str, str]:
        """Export the corrosion C-scan currently displayed in the corrosion view."""
        if not output_directory:
            raise ValueError("No output folder selected.")

        nde_model = self.get_nde_model()
        if nde_model is None:
            raise ValueError("Load an NDE before exporting the corrosion C-scan.")

        projection_payload = self._displayed_corrosion_projection
        if projection_payload is None:
            raise ValueError("Select a corrosion mapping before exporting the C-scan.")

        projection, value_range = projection_payload
        nde_filename = str((nde_model.metadata or {}).get("path", "unknown"))
        saved_paths = self.corrosion_service.save_cscan_projection(
            output_directory=output_directory,
            nde_filename=nde_filename,
            projection=projection,
            value_range=value_range,
        )
        self.logger.info("Corrosion C-scan exported: %s | %s", saved_paths[0], saved_paths[1])
        return saved_paths

    def _bind_layer_selector_signals(self) -> None:
        for view in (self.standard_view, self.corrosion_view):
            if view is not None:
                view.layer_selected.connect(self._on_display_layer_selected)

    def _on_display_layer_selected(self, layer_id: str) -> None:
        normalized = self.view_state_model.set_cscan_display_layer_id(layer_id)
        volume = self.get_volume()
        if volume is not None:
            self.update_views(volume, preserve_view=True)
        else:
            self.sync_layer_choices()
        if normalized:
            self.logger.debug("C-scan display layer selected: %s", normalized)

    def _capture_current_navigation_state(self) -> Optional[dict[str, object]]:
        current_view = self._current_display_view()
        if current_view is None:
            return None
        return current_view.capture_navigation_state()

    def _current_display_view(self) -> Optional[CScanView]:
        if self._stack is not None:
            current = self._stack.currentWidget()
            if current is self.standard_view:
                return self.standard_view
            if current is self.corrosion_view:
                return self.corrosion_view
        if self._displaying_corrosion and self.corrosion_view is not None:
            return self.corrosion_view
        return self.standard_view

    def _current_layer_choices(self) -> list[CScanLayerChoice]:
        try:
            choices = self.get_layer_choices()
        except Exception:  # noqa: BLE001
            self.logger.exception("Unable to list C-scan layer choices")
            return []
        return [
            choice
            for choice in choices
            if str(getattr(choice, "layer_id", "") or "").strip()
        ]

    def _selected_layer_choice(
        self,
        choices: list[CScanLayerChoice],
    ) -> Optional[CScanLayerChoice]:
        if not choices:
            self.view_state_model.set_cscan_display_layer_id(None)
            return None

        selected_id = str(
            getattr(self.view_state_model, "cscan_display_layer_id", None) or ""
        ).strip()
        if selected_id:
            for choice in choices:
                if str(choice.layer_id) == selected_id:
                    return choice

        fallback = next((choice for choice in choices if bool(choice.is_active)), choices[0])
        self.view_state_model.set_cscan_display_layer_id(fallback.layer_id)
        return fallback

    def _sync_layer_selectors(
        self,
        choices: list[CScanLayerChoice],
        selected_layer_id: Optional[str],
    ) -> None:
        items = [(choice.layer_id, choice.name) for choice in choices]
        if self.standard_view is not None:
            self.standard_view.set_layer_choices(items, selected_layer_id)
        if self.corrosion_view is not None:
            self.corrosion_view.set_layer_choices(items, selected_layer_id)
