from typing import Optional


class ViewStateModel:
    """Stores UI state such as slice index and tool selection."""

    def __init__(self) -> None:
        self.current_slice: Optional[int] = None
        self.slice_range: Optional[range] = None
        self.zoom_level: float = 1.0
        self.overlay_opacity: float = 1.0
        self.active_tool: Optional[str] = None

    def set_slice(self, i: int) -> None:
        """Set the current slice index."""
        pass

    def set_tool(self, name: str) -> None:
        """Set the active tool."""
        pass
