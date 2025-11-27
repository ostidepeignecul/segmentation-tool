from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame


class EndviewView(QFrame):
    """Displays a 2D slice of the NDE volume."""

    poly_started = pyqtSignal()
    poly_finished = pyqtSignal()
    slice_changed = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def set_slice(self, index: int) -> None:
        """Update the slice shown in the view."""
        pass

    def update_image(self) -> None:
        """Refresh the displayed image."""
        pass

    def update_overlay(self) -> None:
        """Refresh the overlay."""
        pass
