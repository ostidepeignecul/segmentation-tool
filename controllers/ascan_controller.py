"""Contrôleur dédié à l’A-Scan (profil + synchronisation crosshair)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from PyQt6.QtWidgets import QStackedLayout

from models.nde_model import NdeModel
from models.view_state_model import ViewStateModel
from services.ascan_service import AScanService, AScanProfile
from views.ascan_view import AScanView
from views.ascan_view_corrosion import AScanViewCorrosion
from views.endview_view import EndviewView


class AScanController:
    """Encapsule la logique d’affichage A-Scan pour décharger le MasterController."""

    def __init__(
        self,
        *,
        ascan_service: AScanService,
        standard_view: Optional[AScanView],
        corrosion_view: Optional[AScanViewCorrosion],
        stacked_layout: Optional[QStackedLayout],
        endview_view: EndviewView,
        view_state_model: ViewStateModel,
        set_cscan_crosshair: Callable[[int, int], None],
    ) -> None:
        self.ascan_service = ascan_service
        self.standard_view = standard_view
        self.corrosion_view = corrosion_view
        self._stack = stacked_layout
        self.endview_view = endview_view
        self.view_state_model = view_state_model
        self.set_cscan_crosshair = set_cscan_crosshair

    def update_trace(
        self,
        nde_model: Optional[NdeModel],
        volume: Optional[np.ndarray],
        *,
        point: Optional[tuple[int, int]] = None,
    ) -> None:
        if nde_model is None or volume is None:
            self.clear()
            return

        profile: Optional[AScanProfile] = self.ascan_service.build_profile(
            nde_model,
            slice_idx=self.view_state_model.current_slice,
            point_hint=point or self.view_state_model.current_point,
        )
        if profile is None:
            self.clear()
            return

        for view in (self.standard_view, self.corrosion_view):
            if view is None:
                continue
            view.set_signal(profile.signal_percent, positions=profile.positions)
            view.set_marker(profile.marker_index)

        if self.view_state_model.corrosion_active and self.corrosion_view is not None:
            self.show_corrosion()
            self._update_corrosion_measurement(nde_model, volume, profile)
        else:
            self.show_standard()
            if self.corrosion_view is not None:
                self.corrosion_view.clear_measurement()

        self.endview_view.set_crosshair(*profile.crosshair)
        slice_idx = self.view_state_model.current_slice
        self.set_cscan_crosshair(slice_idx, profile.crosshair[0])
        self.view_state_model.update_crosshair(*profile.crosshair)

    def clear(self) -> None:
        if self.standard_view is not None:
            self.standard_view.clear()
        if self.corrosion_view is not None:
            self.corrosion_view.clear()
            self.corrosion_view.clear_measurement()
        self.show_standard()
        self.view_state_model.set_current_point(None)
        self.view_state_model.cursor_position = None

    def show_standard(self) -> None:
        if self._stack is not None and self.standard_view is not None:
            self._stack.setCurrentWidget(self.standard_view)

    def show_corrosion(self) -> None:
        if self._stack is not None and self.corrosion_view is not None:
            self._stack.setCurrentWidget(self.corrosion_view)

    def set_marker_visible(self, visible: bool) -> None:
        if self.standard_view is not None:
            self.standard_view.set_marker_visible(visible)
        if self.corrosion_view is not None:
            self.corrosion_view.set_marker_visible(visible)

    def _update_corrosion_measurement(
        self,
        nde_model: NdeModel,
        volume: np.ndarray,
        profile: AScanProfile,
    ) -> None:
        if self.corrosion_view is None:
            return

        overlay = self.view_state_model.corrosion_overlay_volume
        label_ids = self.view_state_model.corrosion_overlay_label_ids
        if overlay is None or label_ids is None:
            self.corrosion_view.clear_measurement()
            return

        axis_index = self.ascan_service.get_ultrasound_axis_index(nde_model, volume.shape)

        indices = None
        peak_map_a = self.view_state_model.corrosion_peak_index_map_a
        peak_map_b = self.view_state_model.corrosion_peak_index_map_b
        if peak_map_a is not None and peak_map_b is not None:
            indices = self.ascan_service.resolve_corrosion_indices_from_peak_maps(
                peak_map_a=peak_map_a,
                peak_map_b=peak_map_b,
                slice_idx=self.view_state_model.current_slice,
                x_pos=profile.crosshair[0],
                axis_index=axis_index,
            )

        if indices is None:
            indices = self.ascan_service.resolve_corrosion_indices(
                overlay=overlay,
                label_ids=label_ids,
                slice_idx=self.view_state_model.current_slice,
                x_pos=profile.crosshair[0],
                axis_index=axis_index,
                signal=profile.signal_percent,
            )
        if indices is None:
            self.corrosion_view.clear_measurement()
            return

        idx_a, idx_b = indices
        distance_px = None
        distance_mm = None
        projection_data = self.view_state_model.corrosion_projection
        if projection_data is not None:
            projection = projection_data[0]
            if projection is not None:
                result = self.ascan_service.resolve_corrosion_distance(
                    projection=projection,
                    slice_idx=self.view_state_model.current_slice,
                    x_pos=profile.crosshair[0],
                    nde_model=nde_model,
                )
                if result is not None:
                    distance_px, distance_mm = result

        self.corrosion_view.set_measurement_indices(
            idx_a,
            idx_b,
            distance_px=distance_px,
            distance_mm=distance_mm,
        )

    def map_profile_index_to_point(
        self,
        nde_model: Optional[NdeModel],
        profile_idx: int,
        current_point: Optional[Tuple[int, int]],
        slice_idx: int,
    ) -> Optional[Tuple[Tuple[int, int], Optional[int]]]:
        if nde_model is None:
            return None
        return self.ascan_service.map_profile_index_to_point(
            nde_model,
            profile_idx,
            current_point,
            slice_idx,
        )
