from typing import Any, Optional

from PyQt6.QtWidgets import QMainWindow

from models.annotation_model import AnnotationModel
from models.nde_model import NDEModel
from models.view_state_model import ViewStateModel
from ui_mainwindow import Ui_MainWindow


class MasterController:
    """Coordinates models and the Designer-built UI without embedding business logic."""

    def __init__(self, main_window: Optional[QMainWindow] = None) -> None:
        self.main_window = main_window or QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)

        self.nde_model = NDEModel()
        self.annotation_model = AnnotationModel()
        self.view_state_model = ViewStateModel()

        # References to Designer-created views.
        self.endview_view = self.ui.frame_3
        self.volume_view = self.ui.frame_4
        self.cscan_view = self.ui.frame_5
        self.ascan_view = self.ui.frame_7
        self.tools_panel = self.ui.dockWidgetContents_2

        self._connect_actions()
        self._connect_signals()

    def _connect_actions(self) -> None:
        """Wire menu actions to controller handlers."""
        self.ui.actionopen_nde.triggered.connect(self._on_open_nde)
        self.ui.actioncharger_npz.triggered.connect(self._on_load_npz)

    def _connect_signals(self) -> None:
        """Wire view signals to controller handlers."""
        self.endview_view.poly_started.connect(self._on_poly_started)
        self.endview_view.poly_finished.connect(self._on_poly_finished)
        self.endview_view.slice_changed.connect(self._on_slice_changed)

        self.tools_panel.tool_selected.connect(self._on_tool_selected)
        self.tools_panel.mask_class_changed.connect(self._on_mask_class_changed)
        self.tools_panel.alpha_changed.connect(self._on_alpha_changed)

    def _on_open_nde(self) -> None:
        """Handle opening an NDE file."""
        pass

    def _on_load_npz(self) -> None:
        """Handle loading an NPZ overlay."""
        pass

    def _on_poly_started(self) -> None:
        """Handle polygon start."""
        pass

    def _on_poly_finished(self) -> None:
        """Handle polygon completion."""
        pass

    def _on_slice_changed(self, index: int) -> None:
        """Handle slice change events."""
        pass

    def _on_tool_selected(self, tool_name: str) -> None:
        """Handle tool selection from tools panel."""
        pass

    def _on_mask_class_changed(self, class_idx: int) -> None:
        """Handle mask class changes from tools panel."""
        pass

    def _on_alpha_changed(self, alpha: float) -> None:
        """Handle overlay alpha changes."""
        pass

    def run(self) -> None:
        """Launch the main window."""
        self.main_window.show()
