from typing import Optional, Tuple, Any


class ViewStateModel:
    """
    Stores UI-related state: slice navigation, crosshair, overlay visibility,
    drawing tools, thresholding, corrosion mode, etc.
    Pure model — no UI, no Qt, no services.
    """

    def __init__(self) -> None:

        # --- Overlay & Display ---
        self.overlay_alpha: float = 0.4
        self.colormap: Optional[str] = None
        self.endview_colormap: str = "Gris"
        self.cscan_colormap: str = "Gris"
        self.show_overlay: bool = True
        self.show_volume: bool = True
        self.show_cross: bool = True

        # --- Tools / Interaction ---
        self.tool_mode: Optional[str] = None
        self.threshold: Optional[int] = 50
        self.threshold_auto: bool = False
        self.roi_peak_prefer_second: bool = False
        self.apply_volume: bool = False
        self.roi_persistence: bool = False
        self.active_label: Optional[int] = None
        self.label0_erase_target: Optional[int] = None
        self.roi_thin_line_max_width: int = 2
        self.apply_volume_start: int = 0
        self.apply_volume_end: int = 0
        self.restriction_rect: Optional[Tuple[int, int, int, int]] = None

        # --- Navigation ---
        self.cursor_position: Optional[Tuple[int, int]] = None
        self.current_slice: int = 0
        self.slice_min: int = 0
        self.slice_max: int = 0
        self.secondary_slice: int = 0
        self.secondary_slice_min: int = 0
        self.secondary_slice_max: int = 0
        self.current_point: Optional[Tuple[int, int]] = None

        # --- Metadata for Views ---
        self.axis_order: Optional[list[str]] = None
        self.camera_state: dict = {}
        self.paint_radius: int = 8

        # --- Corrosion Mode ---
        self.corrosion_active: bool = False
        self.corrosion_projection: Optional[Tuple[Any, Tuple[float, float]]] = None
        self.corrosion_interpolated_projection: Optional[Tuple[Any, Tuple[float, float]]] = None
        self.corrosion_overlay_volume: Optional[Any] = None
        self.corrosion_overlay_palette: Optional[dict] = None
        self.corrosion_overlay_label_ids: Optional[tuple[int, int]] = None
        self.corrosion_peak_index_map_a: Optional[Any] = None
        self.corrosion_peak_index_map_b: Optional[Any] = None
        self.corrosion_label_a: Optional[int] = None
        self.corrosion_label_b: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Slice control
    # ------------------------------------------------------------------ #
    def set_slice_bounds(self, min_idx: int, max_idx: int) -> None:
        """Define valid slice range."""
        self.slice_min = int(min_idx)
        self.slice_max = int(max_idx)

    def clamp_slice(self, index: int) -> int:
        """Clamp slice index inside defined bounds."""
        index = int(index)
        return max(self.slice_min, min(self.slice_max, index))

    def set_slice(self, index: int) -> None:
        """Update current slice using clamping rules."""
        self.current_slice = self.clamp_slice(index)

    def set_secondary_slice_bounds(self, min_idx: int, max_idx: int) -> None:
        """Define valid range for the secondary orthogonal slice."""
        self.secondary_slice_min = int(min_idx)
        self.secondary_slice_max = int(max_idx)

    def clamp_secondary_slice(self, index: int) -> int:
        """Clamp secondary slice index inside defined bounds."""
        index = int(index)
        return max(self.secondary_slice_min, min(self.secondary_slice_max, index))

    def set_secondary_slice(self, index: int) -> None:
        """Update the secondary slice using clamping rules."""
        self.secondary_slice = self.clamp_secondary_slice(index)

    # ------------------------------------------------------------------ #
    # Crosshair & point control
    # ------------------------------------------------------------------ #
    def update_crosshair(self, x: int, y: int) -> None:
        """Store (x, y) as the active cursor AND crosshair point."""
        p = (int(x), int(y))
        self.cursor_position = p
        self.current_point = p

    def set_cursor_position(self, x: int, y: int) -> None:
        """Only update the cursor position (used during drag)."""
        self.cursor_position = (int(x), int(y))

    def set_current_point(self, point: Optional[Tuple[int, int]]) -> None:
        """Update the main crosshair point or clear it."""
        if point is None:
            self.current_point = None
        else:
            self.current_point = (int(point[0]), int(point[1]))

    # ------------------------------------------------------------------ #
    # Tool & threshold
    # ------------------------------------------------------------------ #
    def set_tool_mode(self, mode: str) -> None:
        self.tool_mode = mode

    def set_threshold(self, threshold: int) -> None:
        self.threshold = int(threshold)

    def set_threshold_auto(self, enabled: bool) -> None:
        self.threshold_auto = bool(enabled)

    def set_roi_peak_prefer_second(self, enabled: bool) -> None:
        """Prefer second A-scan peak in Peak ROI mode."""
        self.roi_peak_prefer_second = bool(enabled)

    def set_apply_volume(self, enabled: bool) -> None:
        self.apply_volume = bool(enabled)

    def set_roi_persistence(self, enabled: bool) -> None:
        self.roi_persistence = bool(enabled)

    def set_active_label(self, label_id: Optional[int]) -> None:
        if label_id is None:
            self.active_label = None
        else:
            self.active_label = int(label_id)

    def set_label0_erase_target(self, label_id: Optional[int]) -> None:
        """Set the target label that label 0 is allowed to erase (None = all)."""
        if label_id is None:
            self.label0_erase_target = None
        else:
            self.label0_erase_target = int(label_id)

    def set_roi_thin_line_max_width(self, value: int) -> None:
        """Set max width (px) for thin-line pruning in grow/line ROIs (0 disables)."""
        try:
            width = int(value)
        except Exception:
            return
        if width < 0:
            width = 0
        self.roi_thin_line_max_width = width

    def set_restriction_rect(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        """Set the global restriction rectangle (x1, y1, x2, y2) or clear it."""
        if rect is None:
            self.restriction_rect = None
            return
        x1, y1, x2, y2 = rect
        self.restriction_rect = (int(x1), int(y1), int(x2), int(y2))

    def set_apply_volume_range(self, start: int, end: int, *, include_current: bool = True) -> tuple[int, int]:
        """Set slice range for apply-to-volume, optionally enforcing current slice inclusion."""
        start_idx = int(start)
        end_idx = int(end)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        start_idx = max(self.slice_min, min(self.slice_max, start_idx))
        end_idx = max(self.slice_min, min(self.slice_max, end_idx))
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        if include_current:
            cur = int(self.current_slice)
            if cur < start_idx:
                start_idx = cur
            elif cur > end_idx:
                end_idx = cur
        self.apply_volume_start = start_idx
        self.apply_volume_end = end_idx
        return start_idx, end_idx

    def set_paint_radius(self, radius: int) -> None:
        """Update brush radius (in pixels) for paint tool."""
        self.paint_radius = max(1, int(radius))

    # ------------------------------------------------------------------ #
    # Visibility toggles
    # ------------------------------------------------------------------ #
    def toggle_overlay(self, visible: bool) -> None:
        self.show_overlay = bool(visible)

    def set_overlay_alpha(self, value: float) -> None:
        """Set global overlay opacity (0.0 - 1.0)."""
        try:
            alpha = float(value)
        except (TypeError, ValueError):
            alpha = 1.0
        self.overlay_alpha = max(0.0, min(1.0, alpha))

    def toggle_volume(self, visible: bool) -> None:
        self.show_volume = bool(visible)

    def set_show_cross(self, visible: bool) -> None:
        self.show_cross = bool(visible)

    def set_endview_colormap(self, name: str) -> None:
        self.endview_colormap = str(name)

    def set_cscan_colormap(self, name: str) -> None:
        self.cscan_colormap = str(name)

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    def set_axis_order(self, order) -> None:
        self.axis_order = list(order) if order else None

    def set_camera_state(self, params: dict) -> None:
        """Store arbitrary view navigation parameters."""
        self.camera_state = dict(params)

    # ------------------------------------------------------------------ #
    # Corrosion state
    # ------------------------------------------------------------------ #
    def activate_corrosion(self, projection, value_range) -> None:
        self.corrosion_active = True
        self.corrosion_projection = (projection, value_range)

    def deactivate_corrosion(self) -> None:
        self.corrosion_active = False
        self.corrosion_projection = None
        self.corrosion_interpolated_projection = None
        self.corrosion_overlay_volume = None
        self.corrosion_overlay_palette = None
        self.corrosion_overlay_label_ids = None
        self.corrosion_peak_index_map_a = None
        self.corrosion_peak_index_map_b = None

    def set_corrosion_label_a(self, label_id: Optional[int]) -> None:
        if label_id is None:
            self.corrosion_label_a = None
        else:
            self.corrosion_label_a = int(label_id)

    def set_corrosion_label_b(self, label_id: Optional[int]) -> None:
        if label_id is None:
            self.corrosion_label_b = None
        else:
            self.corrosion_label_b = int(label_id)

    def set_corrosion_label_pair(
        self,
        label_a: Optional[int],
        label_b: Optional[int],
    ) -> None:
        self.set_corrosion_label_a(label_a)
        self.set_corrosion_label_b(label_b)
