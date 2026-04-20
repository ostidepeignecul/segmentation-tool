"""Controller dédié à la zone C-scan (standard + corrosion)."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

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

        self.last_corrosion_result: Optional[CorrosionWorkflowResult] = None
        self.on_corrosion_completed = on_corrosion_completed

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
    # --- Stack & visibility ---------------------------------------------------------
    def show_standard(self) -> None:
        if self._stack is not None and self.standard_view is not None:
            self._stack.setCurrentWidget(self.standard_view)

    def show_corrosion(self) -> None:
        if self._stack is not None and self.corrosion_view is not None:
            self._stack.setCurrentWidget(self.corrosion_view)

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

    def set_colormap(self, name: str, lut: Optional[np.ndarray]) -> None:
        """Apply colormap on the standard C-scan view."""
        if self.standard_view is not None:
            self.standard_view.set_colormap(name, lut)

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
    def update_views(self, volume: Optional[np.ndarray]) -> None:
        """Met à jour la projection standard et corrosion selon l'état courant."""
        if volume is None:
            return

        nde_model = self.get_nde_model()
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

        if (
            self.view_state_model.corrosion_active
            and self.view_state_model.corrosion_projection is not None
            and self.corrosion_view
        ):
            projection, value_range = self.view_state_model.corrosion_projection
            self.show_corrosion()
            self.corrosion_view.set_projection(
                projection,
                value_range,
                colormaps=("Corrosion",),
                value_scale_mm=ultrasound_resolution_mm,
            )
        else:
            self.view_state_model.deactivate_corrosion()
            self.show_standard()
            if self.standard_view is not None:
                self.standard_view.set_projection(
                    standard_projection,
                    standard_range,
                    colormaps=("Gris", "OmniScan"),
                )

    # --- Corrosion workflow --------------------------------------------------------
    def reset_corrosion(self) -> None:
        self.view_state_model.deactivate_corrosion()
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
        self.status_callback("Analyse corrosion en cours...", 2000)
        label_a = getattr(self.view_state_model, "corrosion_label_a", None)
        label_b = getattr(self.view_state_model, "corrosion_label_b", None)
        peak_selection_mode_a = self.view_state_model.get_corrosion_peak_selection_mode_a()
        peak_selection_mode_b = self.view_state_model.get_corrosion_peak_selection_mode_b()
        result = self.corrosion_workflow.run(
            nde_model=nde_model,
            annotation_model=self.annotation_model,
            volume=volume,
            label_a=label_a,
            label_b=label_b,
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
            raise ValueError("Aucun dossier de sortie sÃ©lectionnÃ©.")

        nde_model = self.get_nde_model()
        if nde_model is None:
            raise ValueError("Chargez un NDE avant d'exporter le C-scan corrosion.")

        projection_payload = self.view_state_model.corrosion_projection
        if not self.view_state_model.corrosion_active or projection_payload is None:
            raise ValueError("Lancez une analyse corrosion avant d'exporter le C-scan.")

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
