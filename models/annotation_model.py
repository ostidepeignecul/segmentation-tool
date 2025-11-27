from typing import Any, Optional


class AnnotationModel:
    """Stores the annotation mask volume."""

    def __init__(self) -> None:
        self.mask_volume: Optional[Any] = None
        self.class_values = [0, 1, 2, 3, 4]

    def set_mask(self, slice_idx: int, mask: Any) -> None:
        """Set the mask for a given slice."""
        pass

    def clear(self) -> None:
        """Reset the annotation mask volume."""
        pass

    def apply_draw_operation(self, *args: Any, **kwargs: Any) -> None:
        """Apply a drawing operation to the mask."""
        pass
