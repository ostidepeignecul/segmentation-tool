"""Controller dédié à la zone C-scan (standard + corrosion)."""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Tuple

import numpy as np
from PyQt6.QtWidgets import QStackedLayout, QWidget

from models.annotation_model import AnnotationModel
from models.nde_model import NdeModel
from models.view_state_model import ViewStateModel
from services.cscan_corrosion_service import CScanCorrosionService
from services.cscan_service import CScanService
from views.cscan_view import CScanView
from views.cscan_view_corrosion import CscanViewCorrosion


class CScanController:
    """Gère la pile de vues C-scan et le workflow d'analyse corrosion."""

    def __init__(
        self,
        *,
        ui,
        view_state_model: ViewStateModel,
        annotation_model: AnnotationModel,
        get_volume: Callable[[], Optional[np.ndarray]],
        get_nde_model: Callable[[], Optional[NdeModel]],
        status_callback: Callable[[str, int], None],
        logger: logging.Logger,
    ) -> None:
        self.ui = ui
        self.view_state_model = view_state_model
        self.annotation_model = annotation_model
        self.get_volume = get_volume
        self.get_nde_model = get_nde_model
        self.status_callback = status_callback
        self.logger = logger

        self.cscan_service = CScanService()
        self.corrosion_service = CScanCorrosionService()

        self._stack: Optional[QStackedLayout] = None
        self.standard_view: Optional[CScanView] = None
        self.corrosion_view: Optional[CscanViewCorrosion] = None

        self._setup_stack()

    # --- Stack & visibility ---------------------------------------------------------
    def _setup_stack(self) -> None:
        """Wrap the standard C-scan view with the corrosion view into a stacked container."""
        try:
            splitter = self.ui.splitter
            original_view = self.ui.frame_4
        except Exception:
            return

        idx = splitter.indexOf(original_view)
        container = QWidget(parent=splitter)
        layout = QStackedLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        original_view.setParent(container)
        layout.addWidget(original_view)

        corrosion_view = CscanViewCorrosion(parent=container)
        layout.addWidget(corrosion_view)

        layout.setCurrentWidget(original_view)
        if idx >= 0:
            splitter.replaceWidget(idx, container)
        else:
            splitter.addWidget(container)

        self._stack = layout
        self.standard_view = original_view
        self.corrosion_view = corrosion_view

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
        mask_volume = self.annotation_model.mask_volume

        if volume is None or mask_volume is None:
            self.logger.error("Corrosion analysis aborted: volume or masks missing.")
            self.view_state_model.deactivate_corrosion()
            return

        if mask_volume.shape[0] != volume.shape[0]:
            self.logger.error(
                "Corrosion analysis aborted: mask depth %s != volume depth %s",
                mask_volume.shape[0],
                volume.shape[0],
            )
            self.view_state_model.deactivate_corrosion()
            return

        visible_labels = [
            lbl for lbl, vis in (self.annotation_model.label_visibility or {}).items() if vis and int(lbl) > 0
        ]
        if len(visible_labels) != 2:
            self.logger.error(
                "Corrosion analysis requires exactly 2 visible labels; found %d.",
                len(visible_labels),
            )
            self.view_state_model.deactivate_corrosion()
            return

        class_A_id, class_B_id = sorted(int(x) for x in visible_labels[:2])
        self.logger.info("Corrosion analysis started with labels %s and %s", class_A_id, class_B_id)

        global_masks = [mask_volume[idx] for idx in range(mask_volume.shape[0])]
        resolution_cross, resolution_ultra = self._extract_resolutions(nde_model)
        nde_filename = "unknown"
        if nde_model is not None:
            nde_filename = str(nde_model.metadata.get("path", "unknown"))
        output_directory = os.path.dirname(nde_filename) if nde_filename not in ("", "unknown") else "."

        try:
            result = self.corrosion_service.run_analysis(
                global_masks=global_masks,
                volume_data=volume,
                nde_data=nde_model.metadata if nde_model else {},
                nde_filename=nde_filename,
                orientation="lengthwise",
                transpose=False,
                rotation_applied=False,
                resolution_crosswise_mm=resolution_cross,
                resolution_ultrasound_mm=resolution_ultra,
                output_directory=output_directory,
                class_A_id=class_A_id,
                class_B_id=class_B_id,
                use_mm=False,
            )
        except Exception as exc:
            self.logger.error("Corrosion analysis failed: %s", exc)
            self.view_state_model.deactivate_corrosion()
            return

        projection, value_range = self.corrosion_service.compute_corrosion_projection(
            result.distance_map,
            value_range=result.distance_value_range,
        )
        self.view_state_model.activate_corrosion(projection, value_range)
        self.update_views(volume)
        self.status_callback("Analyse corrosion terminée", 3000)

    # --- Internals -----------------------------------------------------------------
    def _extract_resolutions(self, nde_model: Optional[NdeModel]) -> tuple[float, float]:
        """Get crosswise and ultrasound resolutions from metadata, defaults to 1.0."""
        if nde_model is None:
            return 1.0, 1.0
        dimensions = nde_model.metadata.get("dimensions", [])
        try:
            cross = float(dimensions[1].get("resolution", 1.0)) if len(dimensions) >= 3 else 1.0
            ultra = float(dimensions[2].get("resolution", 1.0)) if len(dimensions) >= 3 else 1.0
        except Exception:
            cross, ultra = 1.0, 1.0
        return cross, ultra
