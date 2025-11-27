from PyQt6.QtWidgets import QFrame


class VolumeView(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def update_volume(self) -> None:
        """Refresh the volume display."""
        pass

    def update_overlay(self) -> None:
        """Refresh the overlay displayed on the volume."""
        pass
