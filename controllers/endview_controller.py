"""Controller dedicated to Endview orchestration (standard + corrosion)."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtWidgets import QStackedLayout

from models.overlay_data import OverlayData
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

    def set_slice_bounds(self, minimum: int, maximum: int) -> None:
        self.standard_view.set_navigation_bounds(int(minimum), int(maximum))
        if self.corrosion_view is not None:
            self.corrosion_view.set_navigation_bounds(int(minimum), int(maximum))

    def set_secondary_slice_bounds(self, minimum: int, maximum: int) -> None:
        if self.secondary_view is not None:
            self.secondary_view.set_navigation_bounds(int(minimum), int(maximum))
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.set_navigation_bounds(int(minimum), int(maximum))

    def set_axis_names(self, *, primary: str, secondary: str) -> None:
        self.standard_view.set_navigation_axis_name(primary)
        if self.corrosion_view is not None:
            self.corrosion_view.set_navigation_axis_name(primary)
        if self.secondary_view is not None:
            self.secondary_view.set_navigation_axis_name(secondary)
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.set_navigation_axis_name(secondary)

    def set_primary_endview_name(self, name: str) -> None:
        self.standard_view.set_endview_name(name)
        if self.corrosion_view is not None:
            self.corrosion_view.set_endview_name(name)

    def set_secondary_endview_name(self, name: str) -> None:
        if self.secondary_view is not None:
            self.secondary_view.set_endview_name(name)
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.set_endview_name(name)

    def set_primary_status_position(self, x: int, y: int) -> None:
        self.standard_view.set_status_position(int(x), int(y))
        if self.corrosion_view is not None:
            self.corrosion_view.set_status_position(int(x), int(y))

    def clear_primary_status_position(self) -> None:
        self.standard_view.clear_status_position()
        if self.corrosion_view is not None:
            self.corrosion_view.clear_status_position()

    def set_primary_status_position_visible(self, visible: bool) -> None:
        self.standard_view.set_status_position_visible(bool(visible))
        if self.corrosion_view is not None:
            self.corrosion_view.set_status_position_visible(bool(visible))

    def set_secondary_status_position_visible(self, visible: bool) -> None:
        if self.secondary_view is not None:
            self.secondary_view.set_status_position_visible(bool(visible))
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.set_status_position_visible(bool(visible))

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

    def set_nde_opacity(self, opacity: float) -> None:
        self.standard_view.set_nde_opacity(float(opacity))
        if self.corrosion_view is not None:
            self.corrosion_view.set_nde_opacity(float(opacity))
        if self.secondary_view is not None:
            self.secondary_view.set_nde_opacity(float(opacity))
        if self.secondary_corrosion_view is not None:
            self.secondary_corrosion_view.set_nde_opacity(float(opacity))

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

    # --- Corrosion profile interaction ---------------------------------------------
    def bind_corrosion_profile_signals(
        self,
        *,
        on_drag_started: Callable[[Any], None],
        on_drag_moved: Callable[[Any], None],
        on_drag_finished: Callable[[Any], None],
        on_double_clicked: Callable[[Any], None],
    ) -> None:
        """Connect corrosion profile edit signals when the view supports them."""
        view = self.corrosion_view
        if view is None:
            return

        started = getattr(view, "profile_drag_started", None)
        moved = getattr(view, "profile_drag_moved", None)
        finished = getattr(view, "profile_drag_finished", None)
        double_clicked = getattr(view, "profile_double_clicked", None)
        if started is not None:
            started.connect(on_drag_started)
        if moved is not None:
            moved.connect(on_drag_moved)
        if finished is not None:
            finished.connect(on_drag_finished)
        if double_clicked is not None:
            double_clicked.connect(on_double_clicked)

    def set_corrosion_profile_preview_overlay(
        self,
        mask_volume: np.ndarray,
        *,
        palette: dict[int, tuple[int, int, int, int]],
        visible_labels: Optional[set[int]] = None,
    ) -> None:
        """Push temporary profile edit overlay to the corrosion endview only."""
        view = self.corrosion_view
        if view is None:
            return
        overlay = OverlayData(
            mask_volume=np.asarray(mask_volume),
            palette=dict(palette),
            label_volumes=None,
        )
        view.set_overlay(overlay, visible_labels=visible_labels)

    def clear_corrosion_profile_anchors(self) -> None:
        """Hide corrosion profile anchors when the view supports them."""
        view = self.corrosion_view
        if view is None:
            return
        clear_anchors = getattr(view, "clear_anchor_points", None)
        if callable(clear_anchors):
            clear_anchors()

    def set_corrosion_profile_anchors(
        self,
        points: Sequence[tuple[int, int]],
        *,
        active: bool,
    ) -> None:
        """Show corrosion profile anchors when the view supports them."""
        view = self.corrosion_view
        if view is None:
            return
        set_anchors = getattr(view, "set_anchor_points", None)
        if callable(set_anchors):
            set_anchors(list(points), active=bool(active))

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
