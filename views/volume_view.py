from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame


class VolumeView(QFrame):
    """Displays a 3D representation of the NDE volume."""

    volume_needs_update = pyqtSignal()
    overlay_updated = pyqtSignal()
    camera_changed = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def set_volume(self, volume) -> None:
        """Assign the volume data to render."""
        pass

    def update_volume(self) -> None:
        """Refresh the volume display."""
        pass

    def update_overlay(self) -> None:
        """Refresh the overlay displayed on the volume."""
        pass
