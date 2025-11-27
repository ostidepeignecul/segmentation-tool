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
        self.tools_panel.attach_designer_widgets(
            slice_spinbox=self.ui.spinBox,
            goto_button=self.ui.pushButton,
            threshold_slider=self.ui.horizontalSlider,
            polygon_radio=self.ui.radioButton,
            rectangle_radio=self.ui.radioButton_2,
            point_radio=self.ui.radioButton_3,
            apply_volume_checkbox=self.ui.checkBox,
            threshold_auto_checkbox=self.ui.checkBox_2,
            roi_persistence_checkbox=self.ui.checkBox_3,
            roi_recompute_button=self.ui.pushButton_2,
            roi_delete_button=self.ui.pushButton_3,
            selection_cancel_button=self.ui.pushButton_4,
        )

        self.tools_panel.slice_changed.connect(self._on_slice_changed)
        self.tools_panel.goto_requested.connect(self._on_goto_requested)
        self.tools_panel.tool_mode_changed.connect(self._on_tool_mode_changed)
        self.tools_panel.threshold_changed.connect(self._on_threshold_changed)
        self.tools_panel.threshold_auto_toggled.connect(self._on_threshold_auto_toggled)
        self.tools_panel.apply_volume_toggled.connect(self._on_apply_volume_toggled)
        self.tools_panel.roi_persistence_toggled.connect(self._on_roi_persistence_toggled)
        self.tools_panel.roi_recompute_requested.connect(self._on_roi_recompute_requested)
        self.tools_panel.roi_delete_requested.connect(self._on_roi_delete_requested)
        self.tools_panel.selection_cancel_requested.connect(self._on_selection_cancel_requested)

        self.endview_view.slice_changed.connect(self._on_slice_changed)
        self.endview_view.mouse_clicked.connect(self._on_endview_mouse_clicked)
        self.endview_view.polygon_started.connect(self._on_endview_polygon_started)
        self.endview_view.polygon_point_added.connect(self._on_endview_polygon_point_added)
        self.endview_view.polygon_completed.connect(self._on_endview_polygon_completed)
        self.endview_view.rectangle_drawn.connect(self._on_endview_rectangle_drawn)
        self.endview_view.point_selected.connect(self._on_endview_point_selected)
        self.endview_view.drag_update.connect(self._on_endview_drag_update)

        self.cscan_view.crosshair_changed.connect(self._on_cscan_crosshair_changed)
        self.cscan_view.slice_requested.connect(self._on_cscan_slice_requested)
        self.ascan_view.position_changed.connect(self._on_ascan_position_changed)
        self.ascan_view.cursor_moved.connect(self._on_ascan_cursor_moved)

        self.volume_view.volume_needs_update.connect(self._on_volume_needs_update)
        self.volume_view.camera_changed.connect(self._on_camera_changed)

    def _on_open_nde(self) -> None:
        """Handle opening an NDE file."""
        pass

    def _on_load_npz(self) -> None:
        """Handle loading an NPZ overlay."""
        pass

    def _on_slice_changed(self, index: int) -> None:
        """Handle slice change events."""
        self.view_state_model.set_slice(index)
        self.tools_panel.set_slice_value(index)
        self.nde_model.set_current_slice(index)
        self.endview_view.set_slice(index)

    def _on_goto_requested(self, slice_idx: int) -> None:
        """Handle explicit goto action from tools panel."""
        self._on_slice_changed(slice_idx)

    def _on_tool_mode_changed(self, mode: str) -> None:
        """Handle drawing mode changes."""
        self.view_state_model.set_tool_mode(mode)

    def _on_threshold_changed(self, value: int) -> None:
        """Handle manual threshold changes."""
        self.view_state_model.set_threshold(value)

    def _on_threshold_auto_toggled(self, enabled: bool) -> None:
        """Handle auto-threshold toggle."""
        self.view_state_model.set_threshold_auto(enabled)

    def _on_apply_volume_toggled(self, enabled: bool) -> None:
        """Handle volume application toggle."""
        self.view_state_model.set_apply_volume(enabled)

    def _on_roi_persistence_toggled(self, enabled: bool) -> None:
        """Handle ROI persistence toggle."""
        self.view_state_model.set_roi_persistence(enabled)

    def _on_roi_recompute_requested(self) -> None:
        """Handle ROI recomputation requests."""
        pass

    def _on_roi_delete_requested(self) -> None:
        """Handle ROI deletion requests."""
        pass

    def _on_selection_cancel_requested(self) -> None:
        """Handle cancellation of current selection."""
        pass

    def _on_endview_mouse_clicked(self, pos: Any, button: Any) -> None:
        """Handle mouse clicks in the endview."""
        pass

    def _on_endview_polygon_started(self, pos: Any) -> None:
        """Handle polygon start."""
        pass

    def _on_endview_polygon_point_added(self, pos: Any) -> None:
        """Handle polygon point addition."""
        pass

    def _on_endview_polygon_completed(self, points: Any) -> None:
        """Handle polygon completion."""
        pass

    def _on_endview_rectangle_drawn(self, rect: Any) -> None:
        """Handle rectangle draw completion."""
        pass

    def _on_endview_point_selected(self, pos: Any) -> None:
        """Handle point selection."""
        pass

    def _on_endview_drag_update(self, pos: Any) -> None:
        """Handle drag updates during drawing."""
        pass

    def _on_cscan_crosshair_changed(self, x: int, y: int) -> None:
        """Handle crosshair movement on the C-Scan view."""
        pass

    def _on_cscan_slice_requested(self, z: int) -> None:
        """Handle slice requests originating from the C-Scan view."""
        self._on_slice_changed(z)

    def _on_ascan_position_changed(self, x: int, y: int, z: int) -> None:
        """Handle A-Scan position changes."""
        pass

    def _on_ascan_cursor_moved(self, t: float) -> None:
        """Handle cursor moves within the A-Scan."""
        pass

    def _on_volume_needs_update(self) -> None:
        """Handle volume refresh requests."""
        pass

    def _on_camera_changed(self, view_params: Any) -> None:
        """Handle camera/navigation changes in the volume view."""
        pass

    def run(self) -> None:
        """Launch the main window."""
        self.main_window.show()
