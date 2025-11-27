from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget


class ToolsPanel(QWidget):
    toolSelected = pyqtSignal(str)
    maskClassChanged = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
