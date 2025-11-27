from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame


class EndviewView(QFrame):
    sliceClicked = pyqtSignal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def update_image(self) -> None:
        """Refresh the displayed image."""
        pass

    def update_overlay(self) -> None:
        """Refresh the overlay."""
        pass
