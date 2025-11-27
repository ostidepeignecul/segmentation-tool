from typing import Optional


class ViewStateModel:
    """Stores UI state such as slice index, overlays, and color settings."""

    def __init__(self) -> None:
        self.current_slice: Optional[int] = None
        self.overlay_alpha: float = 1.0
        self.colormap: Optional[str] = None
        self.show_overlay: bool = True
        self.show_volume: bool = True

    def set_slice(self, index: int) -> None:
        """Set the current slice index."""
        pass

    def set_alpha(self, alpha: float) -> None:
        """Adjust overlay alpha value."""
        pass

    def set_colormap(self, name: str) -> None:
        """Change the active colormap."""
        pass

    def toggle_overlay(self, visible: bool) -> None:
        """Show or hide overlays."""
        pass
