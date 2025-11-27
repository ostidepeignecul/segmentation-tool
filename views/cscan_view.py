from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame


class CScanView(QFrame):
    """Displays a 2D heatmap (C-Scan) view."""

    crosshair_changed = pyqtSignal(int, int)
    slice_requested = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def set_data(self, data) -> None:
        """Assign the C-Scan data to display."""
        pass

    def update_cscan(self) -> None:
        """Refresh the C-Scan display."""
        pass
