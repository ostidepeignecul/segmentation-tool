"""Controller dedicated to Endview orchestration (standard + corrosion)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from PyQt6.QtWidgets import QStackedLayout

from models.view_state_model import ViewStateModel
from views.endview_view import EndviewView


class EndviewController:
    """Encapsulates Endview synchronization to keep MasterController lean."""

    def __init__(
        self,
        *,
        standard_view: EndviewView,
        corrosion_view: Optional[EndviewView],
        secondary_view: Optional[EndviewView],
        secondary_corrosion_view: Optional[EndviewView],
        stacked_layout: Optional[QStackedLayout],
        secondary_stacked_layout: Optional[QStackedLayout],
        view_state_model: ViewStateModel,
    ) -> None:
        self.standard_view = standard_view
        self.corrosion_view = corrosion_view
        self.secondary_view = secondary_view
        self.secondary_corrosion_view = secondary_corrosion_view
        self._stack = stacked_layout
        self._secondary_stack = secondary_stacked_layout
        self.view_state_model = view_state_model

    # --- Stack mode ----------------------------------------------------------------
    def show_standard(self) -> None:
        if self._stack is not None and self.standard_view is not None:
            self._stack.setCurrentWidget(self.standard_view)
        if self._secondary_stack is not None and self.secondary_view is not None:
            self._secondary_stack.setCurrentWidget(self.secondary_view)

    def show_corrosion(self) -> None:
        if self._stack is not None and self.corrosion_view is not None:
            self._stack.setCurrentWidget(self.corrosion_view)
        if self._secondary_stack is not None:
            if self.secondary_corrosion_view is not None:
                self._secondary_stack.setCurrentWidget(self.secondary_corrosion_view)
            elif self.secondary_view is not None:
                self._secondary_stack.setCurrentWidget(self.secondary_view)

    def sync_mode(self) -> None:
        if self.view_state_model.corrosion_active and self.corrosion_view is not None:
            self.show_corrosion()
            return
        self.show_standard()

    # --- View synchronization -------------------------------------------------------
    def set_volume(self, volume: np.ndarray) -> None:
        self.standard_view.set_volume(volume)
        if self.corrosion_view is not None:
            self.corrosion_view.set_volume(volume)

    def set_secondary_volume(self, volume: np.ndarray) -> None:
        if self.secondary_view is not None:
            self.secondary_view.set_volume(volume)
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.set_volume(volume)

    def set_slice(self, slice_idx: int) -> None:
        self.standard_view.set_slice(int(slice_idx))
        if self.corrosion_view is not None:
            self.corrosion_view.set_slice(int(slice_idx))

    def set_secondary_slice(self, slice_idx: int) -> None:
        if self.secondary_view is not None:
            self.secondary_view.set_slice(int(slice_idx))
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.set_slice(int(slice_idx))

    def set_cross_visible(self, visible: bool) -> None:
        self.standard_view.set_cross_visible(bool(visible))
        if self.corrosion_view is not None:
            self.corrosion_view.set_cross_visible(bool(visible))
        if self.secondary_view is not None:
            self.secondary_view.set_cross_visible(bool(visible))
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.set_cross_visible(bool(visible))

    def set_crosshair(self, x: int, y: int) -> None:
        self.standard_view.set_crosshair(int(x), int(y))
        if self.corrosion_view is not None:
            self.corrosion_view.set_crosshair(int(x), int(y))

    def set_secondary_crosshair(self, x: int, y: int) -> None:
        if self.secondary_view is not None:
            self.secondary_view.set_crosshair(int(x), int(y))
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.set_crosshair(int(x), int(y))

    def set_colormap(self, name: str, lut: Optional[np.ndarray]) -> None:
        self.standard_view.set_colormap(name, lut)
        if self.corrosion_view is not None:
            self.corrosion_view.set_colormap(name, lut)
        if self.secondary_view is not None:
            self.secondary_view.set_colormap(name, lut)
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.set_colormap(name, lut)

    def reset_display_size(self) -> None:
        self.standard_view.reset_display_size()
        if self.corrosion_view is not None:
            self.corrosion_view.reset_display_size()
        if self.secondary_view is not None:
            self.secondary_view.reset_display_size()
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.reset_display_size()

    def set_display_size(self, width: int, height: int) -> None:
        self.standard_view.set_display_size(int(width), int(height))
        if self.corrosion_view is not None:
            self.corrosion_view.set_display_size(int(width), int(height))
        if self.secondary_view is not None:
            self.secondary_view.set_display_size(int(width), int(height))
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.set_display_size(int(width), int(height))

    # --- Input/state helpers --------------------------------------------------------
    def on_point_selected(self, pos: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(pos, (tuple, list)) or len(pos) != 2:
            return None
        try:
            x, y = int(pos[0]), int(pos[1])
        except Exception:
            return None
        self.view_state_model.update_crosshair(x, y)
        return x, y

    def on_drag_update(self, pos: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(pos, (tuple, list)) or len(pos) != 2:
            return None
        try:
            x, y = int(pos[0]), int(pos[1])
        except Exception:
            return None
        self.view_state_model.set_cursor_position(x, y)
        return x, y
