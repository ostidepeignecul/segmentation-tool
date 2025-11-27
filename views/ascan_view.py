from typing import Any

from PyQt6.QtWidgets import QFrame


class AScanView(QFrame):
    """Displays the raw A-Scan signal."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def set_data(self, curve: Any) -> None:
        """Assign the A-Scan data to display."""
        pass

    def update_ascan(self, curve: Any) -> None:
        """Refresh the A-Scan display."""
        pass
