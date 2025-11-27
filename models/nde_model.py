from typing import Any, Dict, Optional, Tuple


class NDEModel:
    """Represents the loaded NDE data without any UI logic."""

    def __init__(self) -> None:
        self.raw_volume: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.file_path: Optional[str] = None
        self.volume_shape: Optional[Tuple[int, int, int]] = None

    def load_from_array(self, array: Any) -> None:
        """Load NDE data from an existing array."""
        pass

    def clear(self) -> None:
        """Reset all stored NDE data."""
        pass
