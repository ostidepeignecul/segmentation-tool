from typing import Any, Optional


class AnnotationModel:
    """Holds the 3D annotation mask aligned with the NDE volume."""

    def __init__(self) -> None:
        self.mask_volume: Optional[Any] = None
        self.class_labels: list[int] = []

    def initialize(self, shape: Any) -> None:
        """Prepare an empty mask volume with the given shape."""
        pass

    def set_slice_mask(self, slice_idx: int, mask: Any) -> None:
        """Set or replace the mask for a specific slice."""
        pass

    def clear(self) -> None:
        """Reset all stored annotations."""
        pass

    def apply_draw_operation(self, *args: Any, **kwargs: Any) -> None:
        """Apply a drawing operation placeholder."""
        pass
