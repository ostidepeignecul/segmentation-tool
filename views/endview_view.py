from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame


class EndviewView(QFrame):
    """Displays a 2D slice of the NDE volume."""

    slice_changed = pyqtSignal(int)
    mouse_clicked = pyqtSignal(object, object)
    polygon_started = pyqtSignal(object)
    polygon_point_added = pyqtSignal(object)
    polygon_completed = pyqtSignal(object)
    rectangle_drawn = pyqtSignal(object)
    point_selected = pyqtSignal(object)
    drag_update = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_slice: Optional[int] = None

    def set_slice(self, index: int) -> None:
        """Update the slice shown in the view."""
        self._current_slice = index

    def update_image(self) -> None:
        """Refresh the displayed image."""
        pass

    def update_overlay(self) -> None:
        """Refresh the overlay."""
        pass

    def current_slice(self) -> Optional[int]:
        """Return the last slice index applied to the view."""
        return self._current_slice
