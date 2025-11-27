from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QPushButton, QVBoxLayout


class ToolsPanel(QFrame):
    """Simple tools container with placeholder signals."""

    tool_selected = pyqtSignal(str)
    mask_class_changed = pyqtSignal(int)
    alpha_changed = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Fictive buttons to illustrate available signals.
        self._pencil_btn = QPushButton("Pencil", self)
        self._eraser_btn = QPushButton("Eraser", self)
        self._class_one_btn = QPushButton("Class 1", self)

        layout = QVBoxLayout(self)
        layout.addWidget(self._pencil_btn)
        layout.addWidget(self._eraser_btn)
        layout.addWidget(self._class_one_btn)

        self._pencil_btn.clicked.connect(lambda: self.tool_selected.emit("pencil"))
        self._eraser_btn.clicked.connect(lambda: self.tool_selected.emit("eraser"))
        self._class_one_btn.clicked.connect(lambda: self.mask_class_changed.emit(1))

    def set_alpha(self, alpha: float) -> None:
        """Emit alpha changes for overlays."""
        self.alpha_changed.emit(alpha)
