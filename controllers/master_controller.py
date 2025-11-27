from typing import Any, Optional

from PyQt6.QtWidgets import QMainWindow

from models.annotation_model import AnnotationModel
from models.nde_model import NDEModel
from models.view_state_model import ViewStateModel
from ui_mainwindow import Ui_MainWindow


class MasterController:
    """Coordinates models and main window wiring without business logic."""

    def __init__(self, main_window: Optional[QMainWindow] = None) -> None:
        self.main_window = main_window or QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)

        self.nde_model = NDEModel()
        self.annotation_model = AnnotationModel()
        self.view_state_model = ViewStateModel()

        self._connect_actions()
        self._connect_view_signals()

    def _connect_actions(self) -> None:
        """Wire menu actions to controller handlers."""
        self.ui.actionopen_nde.triggered.connect(self._on_open_nde)
        self.ui.actioncharger_npz.triggered.connect(self._on_load_npz)

    def _connect_view_signals(self) -> None:
        """Wire view signals to controller handlers."""
        self.ui.frame_3.sliceClicked.connect(lambda x, y: self._on_slice_changed(y))
        self.ui.dockWidgetContents_2.toolSelected.connect(self._on_draw_operation)
        self.ui.dockWidgetContents_2.maskClassChanged.connect(self._on_draw_operation)

    def _on_open_nde(self) -> None:
        """Handle opening an NDE file."""
        pass

    def _on_load_npz(self) -> None:
        """Handle loading an NPZ overlay."""
        pass

    def _on_slice_changed(self, i: int) -> None:
        """Handle slice change events."""
        pass

    def _on_draw_operation(self, *args: Any, **kwargs: Any) -> None:
        """Handle draw operations from views or tools."""
        pass

    def run(self) -> None:
        """Launch the main window."""
        self.main_window.show()
