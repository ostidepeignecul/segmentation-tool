"""Controller dédié à la zone C-scan (standard + corrosion)."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from PyQt6.QtWidgets import QStackedLayout

from models.annotation_model import AnnotationModel
from models.nde_model import NdeModel
from models.view_state_model import ViewStateModel
from services.cscan_corrosion_service import CScanCorrosionService, CorrosionWorkflowService
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

        self.cscan_service = CScanService()
        self.corrosion_service = CScanCorrosionService()

        if corrosion_workflow_service is None:
            corrosion_workflow_service = CorrosionWorkflowService(
                cscan_corrosion_service=self.corrosion_service
            )
        else:
            self.corrosion_service = corrosion_workflow_service.cscan_corrosion_service
        self.corrosion_workflow = corrosion_workflow_service

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

    # --- Update projections --------------------------------------------------------
    def update_views(self, volume: Optional[np.ndarray]) -> None:
        """Met à jour la projection standard et corrosion selon l'état courant."""
        if volume is None:
            return

        standard_projection, standard_range = self.cscan_service.compute_top_projection(volume)

        if (
            self.view_state_model.corrosion_active
            and self.view_state_model.corrosion_projection is not None
            and self.corrosion_view
        ):
            projection, value_range = self.view_state_model.corrosion_projection
            self.show_corrosion()
            self.corrosion_view.set_projection(projection, value_range, colormaps=("Corrosion",))
        else:
            self.view_state_model.deactivate_corrosion()
            self.show_standard()
            if self.standard_view is not None:
                self.standard_view.set_projection(standard_projection, standard_range)

    # --- Corrosion workflow --------------------------------------------------------
    def reset_corrosion(self) -> None:
        self.view_state_model.deactivate_corrosion()

    def run_corrosion_analysis(self) -> None:
        """Execute corrosion analysis using exactly two visible labels."""
        volume = self.get_volume()
        nde_model = self.get_nde_model()
        if volume is None or nde_model is None:
            self.logger.error("Corrosion analysis aborted: volume or NDE model missing.")
            self.view_state_model.deactivate_corrosion()
            return

        result = self.corrosion_workflow.run(
            nde_model=nde_model,
            annotation_model=self.annotation_model,
            volume=volume,
        )

        if not result.ok:
            self.logger.error(result.message)
            self.status_callback(result.message, 5000)
            self.view_state_model.deactivate_corrosion()
            return

        self.view_state_model.activate_corrosion(result.projection, result.value_range)
        self.update_views(volume)
        self.status_callback(result.message, 3000)
