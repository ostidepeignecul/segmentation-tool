from typing import Any, Dict, Optional


class NDEModel:
    """Stores oriented NDE data (volume, A-scan, metadata) without UI logic."""

    def __init__(self) -> None:
        self.volume: Optional[Any] = None
        self.normalized_volume: Optional[Any] = None
        self.a_scan: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.current_slice: Optional[int] = None

    def set_volume(
        self,
        volume: Any,
        metadata: Optional[Dict[str, Any]] = None,
        normalized_volume: Optional[Any] = None,
    ) -> None:
        """Assign the oriented volume, optional normalized copy, and optional metadata."""
        self.volume = volume
        self.normalized_volume = normalized_volume
        if metadata:
            self.metadata = dict(metadata)

    def set_a_scan(self, a_scan: Any) -> None:
        """Assign the raw A-scan data."""
        self.a_scan = a_scan

    def set_current_slice(self, index: int) -> None:
        """Update the currently selected slice index."""
        self.current_slice = index

    def set_normalized_volume(self, normalized_volume: Any) -> None:
        """Assign the normalized volume representation (0-1)."""
        self.normalized_volume = normalized_volume

    def clear(self) -> None:
        """Reset stored NDE data."""
        self.volume = None
        self.normalized_volume = None
        self.a_scan = None
        self.metadata = {}
        self.current_slice = None
