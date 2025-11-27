from typing import Any

from PyQt6.QtWidgets import QFrame


class AScanView(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def update_ascan(self, curve: Any) -> None:
        """Refresh the A-Scan display."""
        pass
