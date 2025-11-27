from typing import Any, Optional

from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox

from models.annotation_model import AnnotationModel
from models.nde_model import NDEModel
from models.view_state_model import ViewStateModel
from services.ascan_service import AScanService
from services.cscan_service import CScanService
from services.nde_loader import NdeLoaderService
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
        self.nde_loader = NdeLoaderService()
        self.cscan_service = CScanService()
        self.ascan_service = AScanService()

        # References to Designer-created views.
        self.endview_view = self.ui.frame_3
        self.volume_view = self.ui.frame_4
        self.cscan_view = self.ui.frame_5
        self.ascan_view = self.ui.frame_7
        self.tools_panel = self.ui.dockWidgetContents_2
        self._current_point: Optional[tuple[int, int]] = None

        self._connect_actions()
        self._connect_signals()

    def _connect_actions(self) -> None:
        """Wire menu actions to controller handlers."""
        self.ui.actionopen_nde.triggered.connect(self._on_open_nde)
        self.ui.actioncharger_npz.triggered.connect(self._on_load_npz)
        self.ui.actionSauvegarder.triggered.connect(self._on_save)
        self.ui.actionParam_tres.triggered.connect(self._on_open_settings)
        self.ui.actionQuitter.triggered.connect(self._on_quit)

    def _connect_signals(self) -> None:
        """Wire view signals to controller handlers."""
        self.tools_panel.attach_designer_widgets(
            slice_slider=self.ui.horizontalSlider_2,
            slice_label=self.ui.label_3,
            position_label=self.ui.label_4,
            goto_button=self.ui.pushButton,
            threshold_slider=self.ui.horizontalSlider,
            polygon_radio=self.ui.radioButton,
            rectangle_radio=self.ui.radioButton_2,
            point_radio=self.ui.radioButton_3,
            overlay_checkbox=self.ui.checkBox_5,
            cross_checkbox=self.ui.checkBox_4,
            apply_volume_checkbox=self.ui.checkBox,
            threshold_auto_checkbox=self.ui.checkBox_2,
            roi_persistence_checkbox=self.ui.checkBox_3,
            roi_recompute_button=self.ui.pushButton_2,
            roi_delete_button=self.ui.pushButton_3,
            selection_cancel_button=self.ui.pushButton_4,
        )

        self.tools_panel.set_overlay_checked(self.view_state_model.show_overlay)
        self.tools_panel.set_cross_checked(self.view_state_model.show_cross)

        self.tools_panel.slice_changed.connect(self._on_slice_changed)
        self.tools_panel.goto_requested.connect(self._on_goto_requested)
        self.tools_panel.tool_mode_changed.connect(self._on_tool_mode_changed)
        self.tools_panel.threshold_changed.connect(self._on_threshold_changed)
        self.tools_panel.threshold_auto_toggled.connect(self._on_threshold_auto_toggled)
        self.tools_panel.apply_volume_toggled.connect(self._on_apply_volume_toggled)
        self.tools_panel.overlay_toggled.connect(self._on_overlay_toggled)
        self.tools_panel.cross_toggled.connect(self._on_cross_toggled)
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
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Ouvrir un fichier .nde",
            "",
            "NDE Files (*.nde);;All Files (*)",
        )
        if not file_path:
            return

        try:
            loaded_model = self.nde_loader.load_nde_model(file_path)
            self.nde_model = loaded_model
            self.nde_model.set_current_slice(0)

            volume = loaded_model.volume
            num_slices = volume.shape[0] if volume is not None else 0
            if num_slices > 0:
                self.tools_panel.set_slice_bounds(0, num_slices - 1)
                self.tools_panel.set_slice_value(0)

            self._refresh_views()

            self.status_message(f"NDE chargé: {file_path}")

        except Exception as exc:
            QMessageBox.critical(self.main_window, "Erreur NDE", str(exc))

    def _on_load_npz(self) -> None:
        """Handle loading an NPZ overlay."""
        pass

    def _on_save(self) -> None:
        """Handle saving current session or annotations."""
        pass

    def _on_open_settings(self) -> None:
        """Open the settings dialog."""
        pass

    def _on_quit(self) -> None:
        """Quit the application."""
        self.main_window.close()

    def _on_slice_changed(self, index: int) -> None:
        """Handle slice change events."""
        self.tools_panel.set_slice_value(index)
        self.nde_model.set_current_slice(index)
        self.endview_view.set_slice(index)
        self.cscan_view.highlight_slice(index)
        self._update_ascan_trace()

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

    def _on_overlay_toggled(self, enabled: bool) -> None:
        """Handle overlay visibility toggle."""
        self.view_state_model.toggle_overlay(enabled)

    def _on_cross_toggled(self, enabled: bool) -> None:
        """Handle crosshair visibility toggle."""
        self.view_state_model.set_show_cross(enabled)
        self.endview_view.set_cross_visible(enabled)
        self.cscan_view.set_cross_visible(enabled)
        self.ascan_view.set_marker_visible(enabled)

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
        if not isinstance(pos, tuple) or len(pos) != 2:
            return
        x, y = int(pos[0]), int(pos[1])
        self.view_state_model.set_cursor_position(x, y)
        self.tools_panel.set_position_label(x, y)
        self._update_ascan_trace(point=(x, y))

    def _on_endview_drag_update(self, pos: Any) -> None:
        """Handle drag updates during drawing."""
        if not isinstance(pos, tuple) or len(pos) != 2:
            return
        x, y = int(pos[0]), int(pos[1])
        self.view_state_model.set_cursor_position(x, y)
        self.tools_panel.set_position_label(x, y)

    def _on_cscan_crosshair_changed(self, slice_idx: int, x: int) -> None:
        """Handle crosshair movement on the C-Scan view."""
        volume = self._current_volume()
        if volume is None:
            return
        self.nde_model.set_current_slice(slice_idx)
        self.tools_panel.set_slice_value(slice_idx)
        self.endview_view.set_slice(slice_idx)
        self.cscan_view.highlight_slice(slice_idx)
        y = self._current_point[1] if self._current_point else volume.shape[1] // 2
        self._update_ascan_trace(point=(x, y))

    def _on_cscan_slice_requested(self, z: int) -> None:
        """Handle slice requests originating from the C-Scan view."""
        self._on_slice_changed(z)

    def _on_ascan_position_changed(self, profile_idx: int) -> None:
        """Handle A-Scan position changes."""
        point = self.ascan_service.map_profile_index_to_point(
            self.nde_model,
            profile_idx,
            self._current_point,
        )
        if point is None:
            return
        self._update_ascan_trace(point=point)

    def _on_ascan_cursor_moved(self, t: float) -> None:
        """Handle cursor moves within the A-Scan."""
        pass

    def _on_volume_needs_update(self) -> None:
        """Handle volume refresh requests."""
        self._refresh_views()

    def _on_camera_changed(self, view_params: Any) -> None:
        """Handle camera/navigation changes in the volume view."""
        if isinstance(view_params, dict) and "slice" in view_params:
            self._on_slice_changed(int(view_params["slice"]))

    def run(self) -> None:
        """Launch the main window."""
        self.main_window.show()

    def status_message(self, message: str, timeout_ms: int = 3000) -> None:
        """Display a transient status message."""
        if hasattr(self.ui, "statusbar") and self.ui.statusbar:
            self.ui.statusbar.showMessage(message, timeout_ms)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _refresh_views(self) -> None:
        """Push the current volume state into all views."""
        volume = self._current_volume()
        if volume is None:
            return

        slice_idx = self.nde_model.current_slice or 0
        self.endview_view.set_volume(volume)
        self.endview_view.set_slice(slice_idx)

        projection, value_range = self.cscan_service.compute_top_projection(volume)
        self.cscan_view.set_projection(projection, value_range)

        self.volume_view.set_volume(volume)
        self._update_ascan_trace()
        self.endview_view.set_cross_visible(self.view_state_model.show_cross)
        self.cscan_view.set_cross_visible(self.view_state_model.show_cross)
        self.ascan_view.set_marker_visible(self.view_state_model.show_cross)

    def _current_volume(self) -> Optional[Any]:
        return self.nde_model.get_active_volume()

    def _update_ascan_trace(self, point: Optional[tuple[int, int]] = None) -> None:
        profile = self.ascan_service.build_profile(
            self.nde_model,
            point_hint=point or self._current_point,
        )
        if profile is None:
            self.ascan_view.clear()
            self._current_point = None
            return

        self.ascan_view.set_signal(profile.signal_percent, positions=profile.positions)
        self.ascan_view.set_marker(profile.marker_index)
        self.endview_view.set_crosshair(*profile.crosshair)
        slice_idx = self.nde_model.current_slice or 0
        self.cscan_view.set_crosshair(slice_idx, profile.crosshair[0])
        self._current_point = profile.crosshair
