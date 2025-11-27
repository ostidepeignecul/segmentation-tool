from typing import Any, Dict, Optional


class NDEModel:
    """Stores raw NDE data (volume, A-scan, metadata) without UI logic."""

    def __init__(self) -> None:
        self.volume: Optional[Any] = None
        self.a_scan: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.current_slice: Optional[int] = None

    def set_volume(self, volume: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Assign the volume and optional metadata."""
        pass

    def set_a_scan(self, a_scan: Any) -> None:
        """Assign the raw A-scan data."""
        pass

    def set_current_slice(self, index: int) -> None:
        """Update the currently selected slice index."""
        pass

    def clear(self) -> None:
        """Reset stored NDE data."""
        pass
