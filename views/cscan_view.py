from PyQt6.QtWidgets import QFrame


class CScanView(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def update_cscan(self) -> None:
        """Refresh the C-Scan display."""
        pass
