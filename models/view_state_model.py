from typing import Optional, Tuple, Any, Iterable

from config.constants import DEFAULT_ACTIVE_LABEL_ID


class ViewStateModel:
    """
    Stores UI-related state: slice navigation, crosshair, overlay visibility,
    drawing tools, thresholding, corrosion mode, etc.
    Pure model — no UI, no Qt, no services.
    """

    def __init__(self) -> None:

        # --- Overlay & Display ---
        self.overlay_alpha: float = 0.4
        self.nde_alpha: float = 1.0
        self.nde_contrast: float = 1.0
        self.colormap: Optional[str] = None
        self.endview_colormap: str = "Gris"
        self.cscan_colormap: str = "Gris"
        self.show_overlay: bool = True
        self.show_outline_only: bool = False
        self.show_volume_view_overlay: bool = True
        self.show_volume: bool = True
        self.show_cross: bool = True

        # --- Tools / Interaction ---
        self.tool_mode: Optional[str] = None
        self.threshold: Optional[int] = 50
        self.threshold_auto: bool = False
        self.roi_peak_prefer_second: bool = False
        self.roi_peak_ignore_position: bool = False
        self.roi_peak_vertical_min_length: int = 1
        self.roi_peak_vertical_max_length: int = 0
        self.apply_volume: bool = False
        self.roi_persistence: bool = False
        self.closing_mask_enabled: bool = False
        self.closing_mask_tolerance: int = 64
        self.closing_mask_merge_distance: int = 0
        self.clean_outliers_enabled: bool = False
        self.clean_outliers_tolerance: int = 64
        self.clean_outliers_thin_line_max_width: int = 1
        self.clean_outliers_thin_gap_max_width: int = 0
        self.clean_outliers_contour_smoothing: int = 0
        self.annotation_action: str = "draw"
        self.force_threshold_erase: bool = False
        self.apply_auto: bool = False
        self.active_label: Optional[int] = int(DEFAULT_ACTIVE_LABEL_ID)
        self.label_overwrite_targets: dict[int, Optional[int]] = {0: None}
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
        self.corrosion_ascan_support_map: Optional[Any] = None
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
    @staticmethod
    def normalize_annotation_action(action: Optional[str]) -> str:
        value = str(action or "").strip().casefold()
        if value == "erase":
            return "erase"
        return "draw"

    def set_tool_mode(self, mode: str) -> None:
        self.tool_mode = mode

    def set_annotation_action(self, action: str) -> str:
        normalized = self.normalize_annotation_action(action)
        self.annotation_action = normalized
        return normalized

    def set_force_threshold_erase(self, enabled: bool) -> None:
        self.force_threshold_erase = bool(enabled)

    def set_apply_auto(self, enabled: bool) -> None:
        self.apply_auto = bool(enabled)

    def set_threshold(self, threshold: int) -> None:
        self.threshold = int(threshold)

    def set_threshold_auto(self, enabled: bool) -> None:
        self.threshold_auto = bool(enabled)

    def set_roi_peak_prefer_second(self, enabled: bool) -> None:
        """Prefer second A-scan peak in Peak ROI mode."""
        self.roi_peak_prefer_second = bool(enabled)

    def set_roi_peak_ignore_position(self, enabled: bool) -> None:
        """Force strongest A-scan peak selection regardless of depth position."""
        self.roi_peak_ignore_position = bool(enabled)

    def set_roi_peak_vertical_min_length(self, value: int) -> None:
        """Set minimum vertical length kept for Peak ROI germination."""
        try:
            min_len = int(value)
        except Exception:
            min_len = 1
        if min_len < 1:
            min_len = 1
        self.roi_peak_vertical_min_length = min_len

    def set_roi_peak_vertical_max_length(self, value: int) -> None:
        """Set maximum vertical length for Peak ROI germination (0 = unlimited)."""
        try:
            max_len = int(value)
        except Exception:
            max_len = 0
        if max_len < 0:
            max_len = 0
        self.roi_peak_vertical_max_length = max_len

    def set_apply_volume(self, enabled: bool) -> None:
        self.apply_volume = bool(enabled)

    def set_roi_persistence(self, enabled: bool) -> None:
        self.roi_persistence = bool(enabled)

    def set_closing_mask_enabled(self, enabled: bool) -> None:
        self.closing_mask_enabled = bool(enabled)

    def set_closing_mask_tolerance(self, value: int) -> None:
        try:
            tolerance = int(value)
        except Exception:
            tolerance = 0
        if tolerance < 0:
            tolerance = 0
        self.closing_mask_tolerance = tolerance

    def set_closing_mask_merge_distance(self, value: int) -> None:
        try:
            distance = int(value)
        except Exception:
            distance = 0
        if distance < 0:
            distance = 0
        self.closing_mask_merge_distance = distance

    def set_clean_outliers_enabled(self, enabled: bool) -> None:
        self.clean_outliers_enabled = bool(enabled)

    def set_clean_outliers_tolerance(self, value: int) -> None:
        try:
            tolerance = int(value)
        except Exception:
            tolerance = 0
        if tolerance < 0:
            tolerance = 0
        self.clean_outliers_tolerance = tolerance

    def set_clean_outliers_thin_line_max_width(self, value: int) -> None:
        try:
            width = int(value)
        except Exception:
            width = 0
        if width < 0:
            width = 0
        self.clean_outliers_thin_line_max_width = width

    def set_clean_outliers_thin_gap_max_width(self, value: int) -> None:
        try:
            width = int(value)
        except Exception:
            width = 0
        if width < 0:
            width = 0
        self.clean_outliers_thin_gap_max_width = width

    def set_clean_outliers_contour_smoothing(self, value: int) -> None:
        try:
            smoothing = int(value)
        except Exception:
            smoothing = 0
        if smoothing < 0:
            smoothing = 0
        self.clean_outliers_contour_smoothing = smoothing

    def set_active_label(self, label_id: Optional[int]) -> None:
        if label_id is None:
            self.active_label = None
        else:
            self.active_label = int(label_id)

    def is_erase_action(self) -> bool:
        return self.normalize_annotation_action(self.annotation_action) == "erase"

    def effective_annotation_label(self) -> Optional[int]:
        if self.is_erase_action():
            return 0
        return self.active_label

    def effective_annotation_threshold(self) -> Optional[int]:
        if self.is_erase_action() and not self.force_threshold_erase:
            return 0
        return self.threshold

    @property
    def label0_erase_target(self) -> Optional[int]:
        """Backward-compatible alias for the label-0 overwrite target."""
        _has_rule, target = self.get_label_overwrite_target(0)
        return target

    def get_label_overwrite_target(self, source_label: int) -> tuple[bool, Optional[int]]:
        """Return whether a source label has an explicit overwrite rule and its target."""
        try:
            source = int(source_label)
        except Exception:
            return False, None
        if source not in self.label_overwrite_targets:
            return False, None
        target = self.label_overwrite_targets.get(source)
        return True, (None if target is None else int(target))

    def set_label_overwrite_target(self, source_label: int, target_label: Optional[int]) -> None:
        """Set an explicit overwrite rule for a source label (None = overwrite all)."""
        source = int(source_label)
        self.label_overwrite_targets[source] = (
            None if target_label is None else int(target_label)
        )

    def clear_label_overwrite_target(self, source_label: int) -> None:
        """Remove the explicit overwrite rule for a source label."""
        try:
            source = int(source_label)
        except Exception:
            return
        self.label_overwrite_targets.pop(source, None)

    def prune_label_overwrite_targets(
        self,
        *,
        source_labels: Iterable[int],
        target_labels: Iterable[int],
    ) -> None:
        """Drop overwrite rules that reference deleted or unavailable labels."""
        valid_sources = {int(label_id) for label_id in source_labels}
        valid_targets = {int(label_id) for label_id in target_labels}
        cleaned: dict[int, Optional[int]] = {}
        for raw_source, raw_target in self.label_overwrite_targets.items():
            try:
                source = int(raw_source)
            except Exception:
                continue
            if source not in valid_sources:
                continue
            if raw_target is None:
                cleaned[source] = None
                continue
            try:
                target = int(raw_target)
            except Exception:
                continue
            if target in valid_targets:
                cleaned[source] = target
        self.label_overwrite_targets = cleaned

    def set_label0_erase_target(self, label_id: Optional[int]) -> None:
        """Backward-compatible helper: None keeps label 0 allowed on all labels."""
        self.set_label_overwrite_target(0, label_id)

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

    def set_show_outline_only(self, visible: bool) -> None:
        self.show_outline_only = bool(visible)

    def set_show_volume_view_overlay(self, visible: bool) -> None:
        self.show_volume_view_overlay = bool(visible)

    def set_overlay_alpha(self, value: float) -> None:
        """Set global overlay opacity (0.0 - 1.0)."""
        try:
            alpha = float(value)
        except (TypeError, ValueError):
            alpha = 1.0
        self.overlay_alpha = max(0.0, min(1.0, alpha))

    def set_nde_alpha(self, value: float) -> None:
        """Set global NDE opacity (0.0 - 1.0)."""
        try:
            alpha = float(value)
        except (TypeError, ValueError):
            alpha = 1.0
        self.nde_alpha = max(0.0, min(1.0, alpha))

    def set_nde_contrast(self, value: float) -> None:
        """Set global NDE contrast factor (1.0 = neutral)."""
        try:
            contrast = float(value)
        except (TypeError, ValueError):
            contrast = 1.0
        self.nde_contrast = max(0.0, min(2.0, contrast))

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
        self.corrosion_ascan_support_map = None

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
