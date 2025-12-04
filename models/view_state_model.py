from typing import Optional, Tuple, Any


class ViewStateModel:
    """
    Stores UI-related state: slice navigation, crosshair, overlay visibility,
    drawing tools, thresholding, corrosion mode, etc.
    Pure model — no UI, no Qt, no services.
    """

    def __init__(self) -> None:

        # --- Overlay & Display ---
        self.overlay_alpha: float = 1.0
        self.colormap: Optional[str] = None
        self.endview_colormap: str = "Gris"
        self.cscan_colormap: str = "Gris"
        self.show_overlay: bool = True
        self.show_volume: bool = True
        self.show_cross: bool = True

        # --- Tools / Interaction ---
        self.tool_mode: Optional[str] = None
        self.threshold: Optional[int] = None
        self.threshold_auto: bool = False
        self.apply_volume: bool = False
        self.roi_persistence: bool = False
        self.active_label: Optional[int] = None

        # --- Navigation ---
        self.cursor_position: Optional[Tuple[int, int]] = None
        self.current_slice: int = 0
        self.slice_min: int = 0
        self.slice_max: int = 0
        self.current_point: Optional[Tuple[int, int]] = None

        # --- Metadata for Views ---
        self.axis_order: Optional[list[str]] = None
        self.camera_state: dict = {}

        # --- Corrosion Mode ---
        self.corrosion_active: bool = False
        self.corrosion_projection: Optional[Tuple[Any, Tuple[float, float]]] = None

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

    def set_apply_volume(self, enabled: bool) -> None:
        self.apply_volume = bool(enabled)

    def set_roi_persistence(self, enabled: bool) -> None:
        self.roi_persistence = bool(enabled)

    def set_active_label(self, label_id: Optional[int]) -> None:
        if label_id is None:
            self.active_label = None
        else:
            self.active_label = int(label_id)

    # ------------------------------------------------------------------ #
    # Visibility toggles
    # ------------------------------------------------------------------ #
    def toggle_overlay(self, visible: bool) -> None:
        self.show_overlay = bool(visible)

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
