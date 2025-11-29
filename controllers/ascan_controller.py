"""Contrôleur dédié à l’A-Scan (profil + synchronisation crosshair)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from models.nde_model import NdeModel
from models.view_state_model import ViewStateModel
from services.ascan_service import AScanService, AScanProfile
from views.ascan_view import AScanView
from views.endview_view import EndviewView


class AScanController:
    """Encapsule la logique d’affichage A-Scan pour décharger le MasterController."""

    def __init__(
        self,
        *,
        ascan_service: AScanService,
        ascan_view: AScanView,
        endview_view: EndviewView,
        view_state_model: ViewStateModel,
        set_cscan_crosshair: Callable[[int, int], None],
    ) -> None:
        self.ascan_service = ascan_service
        self.ascan_view = ascan_view
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

        self.ascan_view.set_signal(profile.signal_percent, positions=profile.positions)
        self.ascan_view.set_marker(profile.marker_index)
        self.endview_view.set_crosshair(*profile.crosshair)
        slice_idx = self.view_state_model.current_slice
        self.set_cscan_crosshair(slice_idx, profile.crosshair[0])
        self.view_state_model.update_crosshair(*profile.crosshair)

    def clear(self) -> None:
        self.ascan_view.clear()
        self.view_state_model.set_current_point(None)
        self.view_state_model.cursor_position = None

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
