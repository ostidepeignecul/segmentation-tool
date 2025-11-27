from typing import Optional


class ViewStateModel:
    """Stores UI state such as slice index, overlays, and color settings."""

    def __init__(self) -> None:
        self.current_slice: Optional[int] = None
        self.overlay_alpha: float = 1.0
        self.colormap: Optional[str] = None
        self.show_overlay: bool = True
        self.show_volume: bool = True
        self.tool_mode: Optional[str] = None
        self.threshold: Optional[int] = None
        self.threshold_auto: bool = False
        self.apply_volume: bool = False
        self.roi_persistence: bool = False

    def set_slice(self, index: int) -> None:
        """Set the current slice index."""
        self.current_slice = index

    def set_alpha(self, alpha: float) -> None:
        """Adjust overlay alpha value."""
        self.overlay_alpha = max(0.0, min(1.0, alpha))

    def set_colormap(self, name: str) -> None:
        """Change the active colormap."""
        self.colormap = name

    def toggle_overlay(self, visible: bool) -> None:
        """Show or hide overlays."""
        self.show_overlay = visible

    def toggle_volume(self, visible: bool) -> None:
        """Show or hide the 3D volume."""
        self.show_volume = visible

    def set_tool_mode(self, mode: str) -> None:
        """Update the active drawing tool."""
        self.tool_mode = mode

    def set_threshold(self, threshold: int) -> None:
        """Update the threshold value."""
        self.threshold = threshold

    def set_threshold_auto(self, enabled: bool) -> None:
        """Set whether automatic thresholding is enabled."""
        self.threshold_auto = enabled

    def set_apply_volume(self, enabled: bool) -> None:
        """Set whether operations apply to the full volume."""
        self.apply_volume = enabled

    def set_roi_persistence(self, enabled: bool) -> None:
        """Set whether ROI persistence is enabled."""
        self.roi_persistence = enabled
