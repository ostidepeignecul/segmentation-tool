import logging
import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QWidget,
    QVBoxLayout,
)

from config.constants import MASK_COLORS_BGRA, PERSISTENT_LABEL_IDS
from controllers.annotation_controller import AnnotationController
from controllers.ascan_controller import AScanController
from controllers.cscan_controller import CScanController
from controllers.corrosion_profile_controller import CorrosionProfileController
from controllers.dock_layout_controller import DockLayoutController
from controllers.endview_controller import EndviewController
from controllers.mask_modification_controller import MaskModificationController
from controllers.session_workspace_controller import SessionWorkspaceController
from models.annotation_model import AnnotationModel
from models.applied_annotation_history_model import AppliedAnnotationHistoryModel
from models.nde_model import NdeModel
from models.view_state_model import ViewStateModel
from models.roi_model import RoiModel
from models.temp_mask_model import TempMaskModel
from services.annotation_axis_service import AnnotationAxisService
from services.annotation_session_manager import AnnotationSessionManager
from services.annotation_service import AnnotationService
from services.ascan_service import AScanService
from services.nnunet_service import NnUnetResult, NnUnetService
from services.overlay_loader import OverlayLoader
from services.overlay_service import OverlayService
from services.overlay_export import OverlayExport
from services.endview_export import EndviewExportService
from services.project_persistence import ProjectPersistence
from services.split_service import SplitFlawNoflawService
from services.nde_loader import NdeLoader
from services.nde_signal_processing_service import (
    NdeSignalProcessingOptions,
    NdeSignalProcessingService,
)
from services.cscan_corrosion_service import CScanCorrosionService, CorrosionWorkflowService
from services.corrosion_profile_edit_service import CorrosionProfileEditService
from services.corrosion_label_service import CorrosionLabelService
from services.mask_modification_service import MaskModificationService
from ui_mainwindow import Ui_MainWindow
from views.annotation_view import AnnotationView
from views.corrosion_settings_view import CorrosionSettingsView
from views.nde_settings_view import NdeSettingsView
from views.nde_open_options_dialog import NdeOpenOptionsDialog
from views.endview_resize_dialog import EndviewResizeDialog
from views.overlay_settings_view import OverlaySettingsView
from views.piece3d_view import Piece3DView


class MasterController:
    """Coordinates models and the Designer-built UI without embedding business logic."""

    def __init__(self, main_window: Optional[QMainWindow] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.main_window = main_window or QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)
        self.dock_layout_controller = DockLayoutController(main_window=self.main_window)

        self.nde_model: Optional[NdeModel] = None
        self.annotation_model = AnnotationModel()
        self.applied_annotation_history_model = AppliedAnnotationHistoryModel()
        self.view_state_model = ViewStateModel()
        self.roi_model = RoiModel()
        self.temp_mask_model = TempMaskModel()
        self.nde_loader = NdeLoader()
        self.nde_signal_processing_service = NdeSignalProcessingService()
        self.annotation_axis_service = AnnotationAxisService()
        self.session_manager = AnnotationSessionManager()
        self.project_persistence = ProjectPersistence()
        self._nde_path: Optional[str] = None
        self.overlay_loader = OverlayLoader()
        self.overlay_service = OverlayService()
        self.overlay_export = OverlayExport()
        self.endview_export_service = EndviewExportService(nde_loader=self.nde_loader)
        self.split_flaw_noflaw_service = SplitFlawNoflawService(
            endview_export_service=self.endview_export_service
        )
        self.nnunet_service = NnUnetService(logger=self.logger)
        self.cscan_corrosion_service = CScanCorrosionService()
        self.corrosion_workflow_service = CorrosionWorkflowService(
            cscan_corrosion_service=self.cscan_corrosion_service
        )
        self.corrosion_profile_edit_service = CorrosionProfileEditService()
        self.mask_modification_service = MaskModificationService()
        self.overlay_settings_view = OverlaySettingsView(self.main_window)
        self.nde_settings_view = NdeSettingsView(self.main_window)
        self.corrosion_settings_view = CorrosionSettingsView(self.main_window)
        self._piece3d_window: Optional[QWidget] = None
        self._piece3d_view: Optional[Piece3DView] = None
        self._piece_toggle_btn: Optional[QPushButton] = None
        self._piece_volume_raw: Optional[np.ndarray] = None
        self._piece_volume_interpolated: Optional[np.ndarray] = None
        self._piece_volume_legacy_raw: Optional[np.ndarray] = None
        self._piece_volume_legacy_interpolated: Optional[np.ndarray] = None
        self._piece_anchor: Optional[tuple[float, float, float]] = None
        self._piece_show_interpolated: bool = True
        self._shortcuts: list[QShortcut] = []
        self._omniscan_lut: Optional[np.ndarray] = None
        self._pre_corrosion_session_state = None
        self._pre_corrosion_session_id: Optional[str] = None
        self._annotation_axis_mode: str = "Auto"
        self._annotation_axis_name: str = "UCoordinate"
        self._secondary_axis_name: str = "VCoordinate"
        self.main_window.closeEvent = self._on_main_window_close_event  # type: ignore[method-assign]
        self._app = QApplication.instance()
        if self._app is not None:
            self._app.aboutToQuit.connect(self._on_app_about_to_quit)

        # References to Designer-created views.
        self.annotation_view: AnnotationView = self.dock_layout_controller.annotation_view
        self.secondary_annotation_view: AnnotationView = self.dock_layout_controller.secondary_annotation_view
        self.secondary_annotation_view_corrosion = (
            self.dock_layout_controller.secondary_annotation_view_corrosion
        )
        self.cscan_view = self.dock_layout_controller.cscan_view
        self.volume_view = self.dock_layout_controller.volume_view
        self.ascan_view = self.dock_layout_controller.ascan_view
        self.tools_panel = self.dock_layout_controller.tools_panel
        self._tools_ui = self.dock_layout_controller.tools_ui
        self.tools_dock = self.dock_layout_controller.tools_dock
        self.ucoordinate_dock = self.dock_layout_controller.ucoordinate_dock
        self.vcoordinate_dock = self.dock_layout_controller.vcoordinate_dock
        self.annotation_view_corrosion = self.dock_layout_controller.annotation_view_corrosion
        self.annotation_stack = self.dock_layout_controller.annotation_stack
        self.secondary_annotation_stack = self.dock_layout_controller.secondary_annotation_stack
        self.cscan_view_corrosion = self.dock_layout_controller.cscan_view_corrosion
        self.ascan_view_corrosion = self.dock_layout_controller.ascan_view_corrosion
        self.cscan_stack = self.dock_layout_controller.cscan_stack
        self.ascan_stack = self.dock_layout_controller.ascan_stack

        self.endview_controller = EndviewController(
            standard_view=self.annotation_view,
            corrosion_view=self.annotation_view_corrosion,
            secondary_view=self.secondary_annotation_view,
            secondary_corrosion_view=self.secondary_annotation_view_corrosion,
            stacked_layout=self.annotation_stack,
            secondary_stacked_layout=self.secondary_annotation_stack,
            view_state_model=self.view_state_model,
        )

        self.annotation_service = AnnotationService()

        # Apply default colormaps
        self.endview_controller.set_colormap(self.view_state_model.endview_colormap, None)
        self.volume_view.set_base_colormap(self.view_state_model.endview_colormap, None)
        if self.cscan_view is not None:
            self.cscan_view.set_colormap(self.view_state_model.cscan_colormap, None)

        # Annotation controller (overlays)
        self.annotation_controller = AnnotationController(
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
            roi_model=self.roi_model,
            temp_mask_model=self.temp_mask_model,
            annotation_axis_service=self.annotation_axis_service,
            annotation_service=self.annotation_service,
            overlay_service=self.overlay_service,
            overlay_export=self.overlay_export,
            annotation_view=self.annotation_view,
            annotation_corrosion_view=self.annotation_view_corrosion,
            annotation_secondary_view=self.secondary_annotation_view,
            annotation_secondary_corrosion_view=self.secondary_annotation_view_corrosion,
            volume_view=self.volume_view,
            overlay_settings_view=self.overlay_settings_view,
            applied_annotation_history_model=self.applied_annotation_history_model,
            logger=self.logger,
            get_volume=self._current_volume,
        )
        self.mask_modification_controller = MaskModificationController(
            view_state_model=self.view_state_model,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            annotation_view=self.annotation_view,
            mask_modification_service=self.mask_modification_service,
            refresh_overlay=self.annotation_controller.refresh_overlay,
            refresh_roi_overlay_for_slice=self.annotation_controller.refresh_roi_overlay_for_slice,
            set_position_label=self.tools_panel.set_position_label,
            status_message=self.status_message,
        )
        self.session_workspace_controller = SessionWorkspaceController(
            main_window=self.main_window,
            logger=self.logger,
            session_manager=self.session_manager,
            project_persistence=self.project_persistence,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
            current_nde_path=lambda: self._nde_path,
            require_nde_path=self._require_nde_path,
            get_annotation_axis_mode=lambda: self._annotation_axis_mode,
            get_signal_processing_selection=self._current_signal_processing_selection,
            load_nde_file=self._load_nde_file,
            after_session_switch=self._after_session_switch,
            status_message=self.status_message,
            has_pending_mask_edits=self.mask_modification_controller.has_pending_edits,
            commit_pending_mask_edits=self.mask_modification_controller.commit_pending_edits,
            has_pending_corrosion_edits=self.corrosion_profile_edit_service.has_pending_edits,
            commit_pending_corrosion_edits=lambda: bool(
                getattr(self, "corrosion_profile_controller", None)
                and self.corrosion_profile_controller.commit_pending_edits()
            ),
        )

        self.cscan_controller = CScanController(
            standard_view=self.cscan_view,
            corrosion_view=self.cscan_view_corrosion,
            stacked_layout=self.cscan_stack,
            view_state_model=self.view_state_model,
            annotation_model=self.annotation_model,
            get_volume=self._current_volume,
            get_nde_model=lambda: self.nde_model,
            status_callback=self.status_message,
            logger=self.logger,
            corrosion_workflow_service=self.corrosion_workflow_service,
            on_corrosion_completed=self._on_corrosion_completed,
        )

        self.ascan_service = AScanService()
        self.ascan_controller = AScanController(
            ascan_service=self.ascan_service,
            standard_view=self.ascan_view,
            corrosion_view=self.ascan_view_corrosion,
            stacked_layout=self.ascan_stack,
            view_state_model=self.view_state_model,
            set_cscan_crosshair=self.cscan_controller.set_crosshair,
            set_endview_crosshair=self.endview_controller.set_crosshair,
        )
        self.corrosion_profile_controller = CorrosionProfileController(
            view_state_model=self.view_state_model,
            annotation_model=self.annotation_model,
            endview_controller=self.endview_controller,
            annotation_controller=self.annotation_controller,
            cscan_controller=self.cscan_controller,
            cscan_corrosion_service=self.cscan_corrosion_service,
            corrosion_profile_edit_service=self.corrosion_profile_edit_service,
            get_volume=self._current_volume,
            set_position_label=self.tools_panel.set_position_label,
            status_message=self.status_message,
            apply_roi_fallback=self._apply_roi_non_corrosion,
            on_session_changed=self._mark_active_session_dirty,
        )

        self._connect_actions()
        self._connect_signals()
        self._register_shortcuts()
        self.annotation_controller.apply_overlay_opacity()
        self._sync_tools_labels()
        self._update_nde_label()
        self._update_endview_label()
        self._sync_display_toggle_actions()
        self.session_manager.ensure_default(
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
        )

    def _connect_actions(self) -> None:
        """Wire menu actions to controller handlers."""
        self.ui.actionopen_nde.triggered.connect(self._on_open_nde)
        if hasattr(self.ui, "actionOuvrir_une_session"):
            self.ui.actionOuvrir_une_session.triggered.connect(self._on_open_session)
        self.ui.actioncharger_npz.triggered.connect(self._on_load_npz)
        if hasattr(self.ui, "actionSauvegarder"):
            self.ui.actionSauvegarder.triggered.connect(self._on_save_session)
        if hasattr(self.ui, "actionEnregistrer_sous"):
            self.ui.actionEnregistrer_sous.triggered.connect(self._on_save_session_as)
        self.ui.actionExporter_npz.triggered.connect(self._on_export_overlay_npz)
        if hasattr(self.ui, "actionExporter_endviews"):
            self.ui.actionExporter_endviews.triggered.connect(self._on_export_endviews)
        if hasattr(self.ui, "actionSplit_flaw_noflaw"):
            self.ui.actionSplit_flaw_noflaw.triggered.connect(self._on_split_flaw_noflaw)
        self.ui.actionParam_tres.triggered.connect(self._on_open_settings)
        self.ui.actionParam_tres_2.triggered.connect(self.annotation_controller.open_overlay_settings)
        if hasattr(self.ui, "actionParam_tres_3"):
            self.ui.actionParam_tres_3.setText("Parametres corrosion")
            self.ui.actionParam_tres_3.triggered.connect(self._on_open_corrosion_settings)
        self.ui.actionCorrosion_analyse.triggered.connect(self._on_run_corrosion_analysis)
        if hasattr(self.ui, "actionnnunet"):
            self.ui.actionnnunet.triggered.connect(self._on_run_nnunet)
        self.ui.actionQuitter.triggered.connect(self._on_quit)
        if hasattr(self.ui, "actionSession_selector"):
            self.ui.actionSession_selector.triggered.connect(self._open_session_dialog)
        if hasattr(self.ui, "actionToggle_tools_panel"):
            self.dock_layout_controller.bind_tools_toggle_action(self.ui.actionToggle_tools_panel)
        if hasattr(self.ui, "actionToggle_ucoord"):
            self.dock_layout_controller.bind_dock_toggle_action(
                self.ucoordinate_dock,
                self.ui.actionToggle_ucoord,
            )
        if hasattr(self.ui, "actionToggle_vcoord"):
            self.dock_layout_controller.bind_dock_toggle_action(
                self.vcoordinate_dock,
                self.ui.actionToggle_vcoord,
            )
        if hasattr(self.ui, "actionToggle_A_Scan"):
            self.dock_layout_controller.bind_dock_toggle_action(
                self.dock_layout_controller.ascan_dock,
                self.ui.actionToggle_A_Scan,
            )
        if hasattr(self.ui, "actionToggle_C_Scan"):
            self.dock_layout_controller.bind_dock_toggle_action(
                self.dock_layout_controller.cscan_dock,
                self.ui.actionToggle_C_Scan,
            )
        if hasattr(self.ui, "actionToggle_Volume"):
            self.dock_layout_controller.bind_dock_toggle_action(
                self.dock_layout_controller.volume_dock,
                self.ui.actionToggle_Volume,
            )
        if hasattr(self.ui, "actionToggle_cross"):
            self.ui.actionToggle_cross.setCheckable(True)
            self.ui.actionToggle_cross.toggled.connect(self._on_cross_toggled)
        if hasattr(self.ui, "actionToggle_overlay"):
            self.ui.actionToggle_overlay.setCheckable(True)
            self.ui.actionToggle_overlay.toggled.connect(self._on_overlay_toggled)
        if hasattr(self.ui, "actionResize_endview"):
            self.ui.actionResize_endview.triggered.connect(self._on_resize_endview)
        if hasattr(self.ui, "actionR_initialisation_docks"):
            self.ui.actionR_initialisation_docks.setText("Réinitialisation docks")
            self.ui.actionR_initialisation_docks.triggered.connect(
                self.dock_layout_controller.reset_layout_to_default
            )
        if hasattr(self.ui, "actionAfficher_solide_3d"):
            action = self.ui.actionAfficher_solide_3d
            action.setCheckable(True)
            action.setChecked(False)
            action.toggled.connect(self._on_piece3d_toggled)

    def _connect_signals(self) -> None:
        """Wire view signals to controller handlers."""
        self.tools_panel.attach_designer_widgets(
            primary_axis_label=self._tools_ui.label_3,
            primary_slice_slider=self._tools_ui.horizontalSlider_2,
            primary_slice_spinbox=self._tools_ui.spinBox_2,
            secondary_axis_label=self._tools_ui.label_7,
            secondary_slice_slider=self._tools_ui.horizontalSlider_4,
            secondary_slice_spinbox=self._tools_ui.spinBox,
            tool_combo=self._tools_ui.comboBox,
            colormap_combo=self._tools_ui.comboBox_2,
            threshold_slider=self._tools_ui.horizontalSlider,
            threshold_label=self._tools_ui.label_2,
            paint_slider=self._tools_ui.horizontalSlider_3,
            overlay_opacity_slider=self._tools_ui.horizontalSlider_6,
            overlay_opacity_spinbox=self._tools_ui.spinBox_4,
            nde_opacity_slider=self._tools_ui.horizontalSlider_5,
            nde_opacity_spinbox=self._tools_ui.spinBox_3,
            nde_label=self._tools_ui.label,
            endview_label=self._tools_ui.label_5,
            position_label=self._tools_ui.label_4,
            apply_volume_checkbox=self._tools_ui.checkBox,
            threshold_auto_checkbox=self._tools_ui.checkBox_2,
            roi_persistence_checkbox=self._tools_ui.checkBox_3,
            roi_recompute_button=self._tools_ui.pushButton_2,
            roi_delete_button=self._tools_ui.pushButton_3,
            selection_cancel_button=self._tools_ui.pushButton_4,
            apply_roi_button=self._tools_ui.pushButton_7,
            label_container=self._tools_ui.scrollAreaWidgetContents,
            overlay_checkbox=getattr(self._tools_ui, "checkBox_5", None),
            cross_checkbox=getattr(self._tools_ui, "checkBox_4", None),
            nde_opacity_label=getattr(self._tools_ui, "label_10", None),
        )

        self.tools_panel.set_overlay_checked(self.view_state_model.show_overlay)
        self.tools_panel.set_cross_checked(self.view_state_model.show_cross)
        if self.view_state_model.threshold is not None:
            self.tools_panel.set_threshold_value(int(self.view_state_model.threshold))
        self.tools_panel.set_paint_size(self.view_state_model.paint_radius)
        self.tools_panel.set_overlay_opacity(self.view_state_model.overlay_alpha)
        self.tools_panel.set_endview_colormap(self.view_state_model.endview_colormap)
        self._sync_tools_coordinate_labels()
        initial_tool_mode = self.view_state_model.tool_mode or self.tools_panel.current_tool_mode()
        if initial_tool_mode:
            self.tools_panel.select_tool_mode(initial_tool_mode)
            self.annotation_controller.on_tool_mode_changed(initial_tool_mode)
            self.mask_modification_controller.on_tool_mode_changed(initial_tool_mode)

        self.tools_panel.slice_changed.connect(self._on_slice_changed)
        self.tools_panel.secondary_slice_changed.connect(self._on_secondary_slice_changed)
        self.tools_panel.tool_mode_changed.connect(self.annotation_controller.on_tool_mode_changed)
        self.tools_panel.tool_mode_changed.connect(self.mask_modification_controller.on_tool_mode_changed)
        self.tools_panel.paint_size_changed.connect(self.annotation_controller.on_paint_size_changed)
        self.tools_panel.threshold_changed.connect(self.annotation_controller.on_threshold_changed)
        self.tools_panel.threshold_auto_toggled.connect(self.annotation_controller.on_threshold_auto_toggled)
        self.tools_panel.apply_volume_toggled.connect(self.annotation_controller.on_apply_volume_toggled)
        self.tools_panel.overlay_opacity_changed.connect(self._on_overlay_opacity_changed)
        self.tools_panel.endview_colormap_changed.connect(self._on_endview_colormap_changed)
        self.tools_panel.overlay_toggled.connect(self._on_overlay_toggled)
        self.tools_panel.cross_toggled.connect(self._on_cross_toggled)
        self.tools_panel.roi_persistence_toggled.connect(self.annotation_controller.on_roi_persistence_toggled)
        self.tools_panel.roi_recompute_requested.connect(self.annotation_controller.on_roi_recompute_requested)
        self.tools_panel.roi_delete_requested.connect(self._on_roi_delete_requested)
        self.tools_panel.selection_cancel_requested.connect(self._on_selection_cancel_requested)
        self.tools_panel.apply_roi_requested.connect(
            self.corrosion_profile_controller.on_apply_roi_requested
        )
        self.tools_panel.label_selected.connect(self.annotation_controller.on_label_selected)
        self.tools_panel.label_selected.connect(
            self.corrosion_profile_controller.on_active_label_changed
        )
        self.tools_panel.label_selected.connect(
            self.mask_modification_controller.on_active_label_changed
        )

        self.annotation_view.slice_changed.connect(self._on_slice_changed)
        self.annotation_view.mouse_clicked.connect(self.annotation_controller.on_annotation_mouse_clicked)
        self.annotation_view.freehand_started.connect(self.annotation_controller.on_annotation_freehand_started)
        self.annotation_view.freehand_point_added.connect(self.annotation_controller.on_annotation_freehand_point_added)
        self.annotation_view.freehand_completed.connect(self.annotation_controller.on_annotation_freehand_completed)
        self.annotation_view.line_drawn.connect(self.annotation_controller.on_annotation_line_drawn)
        self.annotation_view.box_drawn.connect(self.annotation_controller.on_annotation_box_drawn)
        self.annotation_view.mod_drag_started.connect(self.mask_modification_controller.on_drag_started)
        self.annotation_view.mod_drag_moved.connect(self.mask_modification_controller.on_drag_moved)
        self.annotation_view.mod_drag_finished.connect(self.mask_modification_controller.on_drag_finished)
        self.annotation_view.mod_double_clicked.connect(self.mask_modification_controller.on_double_clicked)
        self.annotation_view.restriction_rect_changed.connect(
            self.annotation_controller.on_restriction_rect_changed
        )
        self.annotation_view.selection_cancel_requested.connect(self._on_selection_cancel_requested)
        self.annotation_view.point_selected.connect(self._on_endview_point_selected)
        self.annotation_view.drag_update.connect(self._on_endview_drag_update)
        self.annotation_view.apply_roi_requested.connect(
            self.corrosion_profile_controller.on_apply_roi_requested
        )
        self.annotation_view.previous_requested.connect(self._on_previous_slice)
        self.annotation_view.next_requested.connect(self._on_next_slice)
        self.secondary_annotation_view.slice_changed.connect(self._on_secondary_slice_changed)
        if self.secondary_annotation_view_corrosion is not None:
            self.secondary_annotation_view_corrosion.slice_changed.connect(
                self._on_secondary_slice_changed
            )
        if self.annotation_view_corrosion is not None:
            self.annotation_view_corrosion.point_selected.connect(self._on_endview_point_selected)
            self.annotation_view_corrosion.drag_update.connect(self._on_endview_drag_update)
        self.endview_controller.bind_corrosion_profile_signals(
            on_drag_started=self.corrosion_profile_controller.on_drag_started,
            on_drag_moved=self.corrosion_profile_controller.on_drag_moved,
            on_drag_finished=self.corrosion_profile_controller.on_drag_finished,
            on_double_clicked=self.corrosion_profile_controller.on_double_clicked,
        )

        if self.cscan_view is not None:
            self.cscan_view.crosshair_changed.connect(self._on_cscan_crosshair_changed)
            self.cscan_view.slice_requested.connect(self._on_cscan_slice_requested)
        self.ascan_view.position_changed.connect(self._on_ascan_position_changed)
        self.ascan_view.cursor_moved.connect(self._on_ascan_cursor_moved)
        if self.ascan_view_corrosion is not None:
            self.ascan_view_corrosion.position_changed.connect(self._on_ascan_position_changed)
            self.ascan_view_corrosion.cursor_moved.connect(self._on_ascan_cursor_moved)
        if self.cscan_view_corrosion is not None:
            self.cscan_view_corrosion.crosshair_changed.connect(self._on_cscan_crosshair_changed)
            self.cscan_view_corrosion.slice_requested.connect(self._on_cscan_slice_requested)
            self.cscan_view_corrosion.export_requested.connect(self._on_export_corrosion_cscan)

        self.volume_view.volume_needs_update.connect(self._on_volume_needs_update)
        self.volume_view.secondary_slice_changed.connect(self._on_secondary_slice_changed)
        self.volume_view.camera_changed.connect(self._on_camera_changed)
        self.overlay_settings_view.label_visibility_changed.connect(
            self.annotation_controller.on_label_visibility_changed
        )
        self.overlay_settings_view.label_color_changed.connect(
            self.annotation_controller.on_label_color_changed
        )
        self.overlay_settings_view.overlay_opacity_changed.connect(
            self._on_overlay_opacity_changed
        )
        self.overlay_settings_view.label_added.connect(self._on_label_added)
        self.overlay_settings_view.label_deleted.connect(self._on_label_deleted)
        self.nde_settings_view.endview_colormap_changed.connect(self._on_endview_colormap_changed)
        self.nde_settings_view.cscan_colormap_changed.connect(self._on_cscan_colormap_changed)
        self.nde_settings_view.apply_volume_range_changed.connect(
            self._on_apply_volume_range_changed
        )
        self.nde_settings_view.erase_label_target_changed.connect(
            self._on_erase_label_target_changed
        )
        self.nde_settings_view.roi_thin_line_width_changed.connect(
            self._on_roi_thin_line_width_changed
        )
        self.nde_settings_view.roi_peak_preference_changed.connect(
            self._on_roi_peak_preference_changed
        )
        self.nde_settings_view.roi_peak_ignore_position_changed.connect(
            self._on_roi_peak_ignore_position_changed
        )
        self.nde_settings_view.roi_peak_vertical_min_changed.connect(
            self._on_roi_peak_vertical_min_changed
        )
        self.nde_settings_view.roi_peak_vertical_max_changed.connect(
            self._on_roi_peak_vertical_max_changed
        )
        self.corrosion_settings_view.label_a_changed.connect(self._on_corrosion_label_a_changed)
        self.corrosion_settings_view.label_b_changed.connect(self._on_corrosion_label_b_changed)

    def _register_shortcuts(self) -> None:
        """Global keyboard shortcuts (active anywhere in the window)."""
        parent = self.main_window
        mapping = [
            (QKeySequence(QKeySequence.StandardKey.Save), self._on_save_session),
            (QKeySequence(Qt.Key.Key_A), self._on_previous_slice),
            (QKeySequence(Qt.Key.Key_D), self._on_next_slice),
            (QKeySequence(Qt.Key.Key_W), self._apply_roi_non_corrosion),
            (QKeySequence(QKeySequence.StandardKey.Undo), self._on_annotation_undo_requested),
            (QKeySequence("Ctrl+Shift+Z"), self._on_annotation_redo_requested),
            (QKeySequence(Qt.Key.Key_Escape), self._on_selection_cancel_requested),
            (QKeySequence(Qt.Key.Key_Return), self.annotation_controller.on_apply_all_temp_masks_requested),
            (QKeySequence(Qt.Key.Key_Enter), self.annotation_controller.on_apply_all_temp_masks_requested),
        ]
        for seq, handler in mapping:
            sc = QShortcut(seq, parent)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(handler)
            self._shortcuts.append(sc)

    def _on_resize_endview(self) -> None:
        """Open a dialog to resize displayed endview, C-scan, and 3D volume views."""
        if self.annotation_view is None:
            return
        current_size = self.annotation_view.get_display_size()
        dialog = EndviewResizeDialog(current_size, parent=self.main_window)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            force_square = dialog.is_square_locked()
            if dialog.wants_reset():
                self.endview_controller.reset_display_size()
                self.cscan_controller.reset_display_size()
                if self.volume_view is not None:
                    self.volume_view.reset_display_size()
            else:
                width, height = dialog.get_size()
                self.endview_controller.set_display_size(width, height)
                self.cscan_controller.set_display_size(width, height)
                if self.volume_view is not None:
                    self.volume_view.set_display_size(
                        width,
                        height,
                        force_square=force_square,
                    )

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
        if not self._confirm_unsaved_sessions_before_reset("ouvrir un autre fichier NDE"):
            return

        try:
            if not self._load_nde_file(file_path, prompt_open_options=True):
                return
        except Exception as exc:
            QMessageBox.critical(self.main_window, "Erreur NDE", str(exc))

    def _on_open_session(self) -> None:
        """Open a persisted `.session` file and restore the contained active session."""
        self.session_workspace_controller.open_session_via_dialog()

    def _on_load_npz(self) -> None:
        """Handle loading an NPZ overlay."""
        if self.nde_model is None:
            QMessageBox.warning(self.main_window, "Overlay", "Chargez un NDE avant l'overlay.")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Charger un overlay (.npz/.npy)",
            "",
            "Overlay Files (*.npz *.npy);;All Files (*)",
        )
        if not file_path:
            return
        volume = self._current_volume()
        if volume is None:
            QMessageBox.warning(self.main_window, "Overlay", "Volume NDE indisponible.")
            return
        try:
            self.annotation_controller.reset_overlay_state(preserve_labels=True)
            self.temp_mask_model.clear()
            self.temp_mask_model.initialize(volume.shape)
            self.roi_model.clear()
            mask_volume = self.overlay_loader.load(file_path, target_shape=volume.shape)
            self.annotation_model.set_mask_volume(mask_volume, preserve_labels=True)
            self.cscan_controller.reset_corrosion()
            self.mask_modification_controller.reset()
            self.annotation_controller.sync_overlay_settings()
            self.annotation_controller.refresh_overlay()
            self._sync_tools_labels()
            self._mark_active_session_dirty()
            self.status_message(f"Overlay chargé: {file_path}")
        except Exception as exc:
            QMessageBox.critical(self.main_window, "Erreur overlay", str(exc))

    def _on_run_nnunet(self) -> None:
        """Lance l'inférence nnUNet sur le volume NDE chargé."""
        if self.nde_model is None:
            QMessageBox.warning(self.main_window, "nnUNet", "Chargez un NDE avant de lancer l'inférence.")
            return

        volume = self._current_volume()
        if volume is None:
            QMessageBox.warning(self.main_window, "nnUNet", "Volume NDE indisponible.")
            return

        model_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Choisir le modèle nnUNet (zip ou dossier)",
            "",
            "Modèles nnUNet (*.zip);;Tous les fichiers (*)",
        )
        if not model_path:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Enregistrer le résultat nnUNet (.npz)",
            "",
            "NPZ Files (*.npz)",
        )
        if not save_path:
            return

        raw_volume = getattr(self.nde_model, "volume", None)
        dataset_id = self.nde_model.metadata.get("path") if self.nde_model.metadata else "current"

        def _on_success(result: NnUnetResult) -> None:
            def _apply() -> None:
                try:
                    self.annotation_controller.reset_overlay_state()
                    self.mask_modification_controller.reset()
                    self.annotation_model.clear()
                    self.annotation_model.set_mask_volume(result.mask)
                    labels = (
                        result.labels_mapping.get("labels", {})
                        if isinstance(result.labels_mapping, dict)
                        else {}
                    )
                    for _, label_id in labels.items():
                        color = MASK_COLORS_BGRA.get(int(label_id), (255, 0, 255, 160))
                        self.annotation_model.ensure_label(int(label_id), color, visible=True)
                    self.view_state_model.toggle_overlay(True)
                    self.annotation_controller.clear_labels()
                    self.annotation_controller.sync_overlay_settings()
                    self.annotation_controller.refresh_overlay()
                    self._sync_tools_labels()
                    self._mark_active_session_dirty()
                    self.status_message(
                        f"nnUNet terminé, masque sauvegardé : {result.output_path}", timeout_ms=5000
                    )
                    QMessageBox.information(
                        self.main_window,
                        "nnUNet",
                        f"Résultat enregistré dans :\n{result.output_path}",
                    )
                except Exception as exc:
                    QMessageBox.critical(self.main_window, "nnUNet", str(exc))

            QTimer.singleShot(0, _apply)

        def _on_error(exc: Exception) -> None:
            def _show() -> None:
                QMessageBox.critical(self.main_window, "nnUNet", str(exc))
                self.status_message("Echec de l'inférence nnUNet", timeout_ms=5000)

            QTimer.singleShot(0, _show)

        try:
            self.status_message("Inférence nnUNet en cours...", timeout_ms=4000)
            self.nnunet_service.run_inference(
                volume=volume,
                raw_volume=raw_volume,
                model_path=model_path,
                output_path=save_path,
                dataset_id=str(dataset_id) if dataset_id else "current",
                on_success=_on_success,
                on_error=_on_error,
            )
        except Exception as exc:
            QMessageBox.critical(self.main_window, "nnUNet", str(exc))

    def _on_export_overlay_npz(self) -> None:
        """Export the current overlay to a standalone NPZ file."""
        volume = self._current_volume()
        if volume is None:
            QMessageBox.warning(self.main_window, "Overlay", "Chargez un NDE avant de sauvegarder.")
            return
        try:
            saved_path = self.annotation_controller.save_overlay_via_dialog(
                parent=self.main_window,
                volume_shape=volume.shape,
            )
            if not saved_path:
                return
            self.status_message(f"Overlay sauvegarde: {saved_path}")
        except Exception as exc:
            QMessageBox.critical(self.main_window, "Erreur sauvegarde overlay", str(exc))

    def _on_save_session(self) -> None:
        """Persist the active annotation session to its bound `.session` file."""
        self.session_workspace_controller.save_active_session_via_ui(force_dialog=False)

    def _on_save_session_as(self) -> None:
        """Persist the active annotation session to a user-chosen `.session` file."""
        self.session_workspace_controller.save_active_session_via_ui(force_dialog=True)

    def _on_export_endviews(self) -> None:
        """Export endviews (RGB + UINT8) via le service dédié."""
        if self.nde_model is None:
            QMessageBox.warning(self.main_window, "Export endviews", "Chargez un NDE avant d'exporter les endviews.")
            return

        base_dir = QFileDialog.getExistingDirectory(
            self.main_window,
            "Choisir le dossier de sortie (sera utilisé comme racine endviews)",
            "",
        )
        if not base_dir:
            return

        base_path = Path(base_dir)
        rgb_dir = base_path / "endviews_rgb24" / "complete"
        uint8_dir = base_path / "endviews_uint8" / "complete"

        success_rgb, message_rgb = self.endview_export_service.export_endviews(
            nde_file=None,
            nde_model=self.nde_model,
            output_folder=str(rgb_dir),
            export_format="rgb",
        )
        if not success_rgb:
            QMessageBox.critical(self.main_window, "Export endviews", message_rgb)
            return

        success_uint8, message_uint8 = self.endview_export_service.export_endviews(
            nde_file=None,
            nde_model=self.nde_model,
            output_folder=str(uint8_dir),
            export_format="uint8",
        )
        if not success_uint8:
            QMessageBox.critical(self.main_window, "Export endviews", message_uint8)
            return

        final_message = f"{message_rgb}\n\n{message_uint8}"
        self.status_message("Export endviews terminé", timeout_ms=4000)
        QMessageBox.information(self.main_window, "Export endviews", final_message)

    def _on_export_corrosion_cscan(self) -> None:
        """Open the native Windows folder picker and export the current corrosion C-scan."""
        if self.nde_model is None:
            QMessageBox.warning(
                self.main_window,
                "Export C-scan corrosion",
                "Chargez un NDE avant d'exporter le C-scan corrosion.",
            )
            return

        nde_path = None
        try:
            nde_path = (self.nde_model.metadata or {}).get("path")
        except Exception:
            nde_path = None

        initial_dir = str(Path(nde_path).parent) if nde_path else ""
        output_dir = QFileDialog.getExistingDirectory(
            self.main_window,
            "Choisir le dossier d'export du C-scan corrosion",
            initial_dir,
        )
        if not output_dir:
            return

        try:
            npz_path, png_path = self.cscan_controller.export_corrosion_projection(
                output_directory=output_dir,
            )
        except ValueError as exc:
            QMessageBox.warning(self.main_window, "Export C-scan corrosion", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self.main_window, "Export C-scan corrosion", str(exc))
            return

        saved_path = f"{npz_path} | {png_path}"
        self.status_message(f"C-scan corrosion sauvegardé: {saved_path}", timeout_ms=5000)

    def _on_split_flaw_noflaw(self) -> None:
        """Lance le split flaw/noflaw (export + tri) via le service dédié."""
        if self.nde_model is None:
            QMessageBox.warning(self.main_window, "Split flaw/noflaw", "Chargez un NDE avant de lancer le split.")
            return

        nde_path = None
        try:
            nde_path = (self.nde_model.metadata or {}).get("path")
        except Exception:
            nde_path = None

        initial_dir = str(Path(nde_path).parent) if nde_path else ""
        output_root = QFileDialog.getExistingDirectory(
            self.main_window,
            "Choisir le dossier parent pour l'export endviews split",
            initial_dir,
        )
        if not output_root:
            return

        prefix, ok = QInputDialog.getText(
            self.main_window,
            "Split flaw/noflaw",
            "Préfixe des images exportées (optionnel) :",
        )
        if not ok:
            return
        suffix, ok = QInputDialog.getText(
            self.main_window,
            "Split flaw/noflaw",
            "Suffixe des images exportées (optionnel) :",
        )
        if not ok:
            return

        prefix = (prefix or "").strip()
        suffix = (suffix or "").strip()

        # Retrieve current signal processing selection from model metadata
        processing_options = None
        if self.nde_model is not None:
            sel = (self.nde_model.metadata or {}).get("signal_processing_selection")
            if isinstance(sel, dict):
                processing_options = NdeSignalProcessingOptions(
                    apply_hilbert=bool(sel.get("apply_hilbert", False)),
                    apply_smoothing=bool(sel.get("apply_smoothing", False)),
                )

        self.status_message("Split flaw/noflaw en cours...", timeout_ms=2000)
        success, message = self.split_flaw_noflaw_service.split_endviews(
            nde_model=self.nde_model,
            annotation_model=self.annotation_model,
            nde_file=nde_path,
            output_root=output_root,
            filename_prefix=prefix,
            filename_suffix=suffix,
            signal_processing_options=processing_options,
        )

        if success:
            self.status_message("Split flaw/noflaw terminé", timeout_ms=5000)
            QMessageBox.information(self.main_window, "Split flaw/noflaw", message)
        else:
            QMessageBox.critical(self.main_window, "Split flaw/noflaw", message)

    def _on_open_settings(self) -> None:
        """Open the settings dialog."""
        self.nde_settings_view.set_colormaps(
            endview=self.view_state_model.endview_colormap,
            cscan=self.view_state_model.cscan_colormap,
        )
        self._sync_apply_volume_range_view()
        self._sync_erase_label_choices()
        self.nde_settings_view.set_roi_thin_line_max_width(
            self.view_state_model.roi_thin_line_max_width
        )
        self.nde_settings_view.set_roi_peak_prefer_second(
            self.view_state_model.roi_peak_prefer_second
        )
        self.nde_settings_view.set_roi_peak_ignore_position(
            self.view_state_model.roi_peak_ignore_position
        )
        self.nde_settings_view.set_roi_peak_vertical_min_length(
            self.view_state_model.roi_peak_vertical_min_length
        )
        self.nde_settings_view.set_roi_peak_vertical_max_length(
            self.view_state_model.roi_peak_vertical_max_length
        )
        self.nde_settings_view.show()
        self.nde_settings_view.raise_()
        self.nde_settings_view.activateWindow()

    def _on_open_corrosion_settings(self) -> None:
        """Open the corrosion settings dialog."""
        self._sync_corrosion_label_choices()
        self.corrosion_settings_view.show()
        self.corrosion_settings_view.raise_()
        self.corrosion_settings_view.activateWindow()

    def _on_run_corrosion_analysis(self) -> None:
        """Capture active session state before launching corrosion analysis."""
        self.corrosion_profile_edit_service.reset()
        self.mask_modification_controller.reset(restore_overlay=True)
        active_id = self.session_manager._active_id  # noqa: SLF001
        if active_id is not None:
            try:
                active_name = self.session_manager._sessions[active_id].name  # noqa: SLF001
                self._pre_corrosion_session_state = self.session_manager._snapshot(  # noqa: SLF001
                    name=active_name,
                    annotation_model=self.annotation_model,
                    temp_mask_model=self.temp_mask_model,
                    roi_model=self.roi_model,
                    view_state_model=self.view_state_model,
                )
                self._pre_corrosion_session_id = active_id
            except Exception:
                self._pre_corrosion_session_state = None
                self._pre_corrosion_session_id = None
        else:
            self._pre_corrosion_session_state = None
            self._pre_corrosion_session_id = None
        self.cscan_controller.run_corrosion_analysis()
        if not self.view_state_model.corrosion_active:
            self._pre_corrosion_session_state = None
            self._pre_corrosion_session_id = None

    def _sync_apply_volume_range_view(self) -> None:
        """Sync apply-to-volume range bounds/values into the settings dialog."""
        volume = self._current_volume()
        if volume is None or getattr(volume, "shape", None) is None:
            self.nde_settings_view.set_apply_volume_bounds(0, 0)
            self.nde_settings_view.set_apply_volume_range(0, 0)
            return
        max_idx = max(0, int(volume.shape[0]) - 1)
        start_idx, end_idx = self.view_state_model.set_apply_volume_range(
            self.view_state_model.apply_volume_start,
            self.view_state_model.apply_volume_end,
            include_current=True,
        )
        self.nde_settings_view.set_apply_volume_bounds(0, max_idx)
        self.nde_settings_view.set_apply_volume_range(start_idx, end_idx)

    def _on_quit(self) -> None:
        """Quit the application."""
        self.main_window.close()

    def _on_main_window_close_event(self, event: Any) -> None:
        """Intercept window close to protect unsaved sessions."""
        self.session_workspace_controller.on_main_window_close_event(event)

    def _on_app_about_to_quit(self) -> None:
        """Persist UI docking layout before Qt application shutdown."""
        try:
            self.dock_layout_controller.save_layout_state()
        except Exception:
            self.logger.exception("Unable to save dock layout state.")

    def _on_overlay_opacity_changed(self, opacity: float) -> None:
        """Keep overlay opacity synchronized across the tools panel and settings dialog."""
        self.annotation_controller.on_overlay_opacity_changed(opacity)
        alpha = float(self.view_state_model.overlay_alpha)
        self.overlay_settings_view.set_overlay_opacity(alpha)
        self.tools_panel.set_overlay_opacity(alpha)

    def _on_endview_colormap_changed(self, name: str) -> None:
        normalized = self._normalize_colormap_name(name)
        lut = self._get_colormap_lut(normalized)
        self.view_state_model.set_endview_colormap(normalized)
        self.endview_controller.set_colormap(normalized, lut)
        self.volume_view.set_base_colormap(normalized, lut)
        self.tools_panel.set_endview_colormap(normalized)
        self.nde_settings_view.set_endview_colormap(normalized)

    def _on_cscan_colormap_changed(self, name: str) -> None:
        normalized = self._normalize_colormap_name(name)
        lut = self._get_colormap_lut(normalized)
        self.view_state_model.set_cscan_colormap(normalized)
        self.cscan_controller.set_colormap(normalized, lut)
        self.nde_settings_view.set_cscan_colormap(normalized)

    def _on_apply_volume_range_changed(self, start: int, end: int) -> None:
        """Handle apply-to-volume range updates from settings."""
        volume = self._current_volume()
        if volume is None:
            return
        start_idx, end_idx = self.view_state_model.set_apply_volume_range(
            start, end, include_current=False
        )
        self.nde_settings_view.set_apply_volume_range(start_idx, end_idx)

    def _on_erase_label_target_changed(self, label_id: Optional[int]) -> None:
        """Handle changes to the label-0 erase target setting."""
        if label_id is None:
            self.view_state_model.set_label0_erase_target(None)
            return
        try:
            self.view_state_model.set_label0_erase_target(int(label_id))
        except Exception:
            self.view_state_model.set_label0_erase_target(None)

    def _on_roi_thin_line_width_changed(self, value: int) -> None:
        """Handle changes to thin-line pruning width for grow/line ROIs."""
        try:
            self.view_state_model.set_roi_thin_line_max_width(int(value))
        except Exception:
            self.view_state_model.set_roi_thin_line_max_width(0)

    def _on_roi_peak_preference_changed(self, prefer_second: bool) -> None:
        """Handle first/second peak preference for Peak ROI mode."""
        self.view_state_model.set_roi_peak_prefer_second(bool(prefer_second))

    def _on_roi_peak_ignore_position_changed(self, enabled: bool) -> None:
        """Handle strongest-peak-only preference for Peak ROI mode."""
        self.view_state_model.set_roi_peak_ignore_position(bool(enabled))

    def _on_roi_peak_vertical_min_changed(self, value: int) -> None:
        """Handle minimum vertical length for Peak ROI germination."""
        self.view_state_model.set_roi_peak_vertical_min_length(int(value))
        min_len = self.view_state_model.roi_peak_vertical_min_length
        max_len = self.view_state_model.roi_peak_vertical_max_length
        if max_len > 0 and min_len > max_len:
            self.view_state_model.set_roi_peak_vertical_max_length(min_len)
            self.nde_settings_view.set_roi_peak_vertical_max_length(
                self.view_state_model.roi_peak_vertical_max_length
            )

    def _on_roi_peak_vertical_max_changed(self, value: int) -> None:
        """Handle maximum vertical length for Peak ROI germination."""
        self.view_state_model.set_roi_peak_vertical_max_length(int(value))
        max_len = self.view_state_model.roi_peak_vertical_max_length
        min_len = self.view_state_model.roi_peak_vertical_min_length
        if max_len > 0 and max_len < min_len:
            self.view_state_model.set_roi_peak_vertical_min_length(max_len)
            self.nde_settings_view.set_roi_peak_vertical_min_length(
                self.view_state_model.roi_peak_vertical_min_length
            )

    def _apply_roi_non_corrosion(self) -> None:
        """Apply all temporary masks through the standard pipeline."""
        self.mask_modification_controller.commit_pending_edits()
        if self.annotation_controller.on_apply_temp_mask_requested():
            self._mark_active_session_dirty()

    def _on_selection_cancel_requested(self) -> None:
        """Cancel mod pending edits first, then fallback to ROI/temp cancel."""
        if self.mask_modification_controller.on_selection_cancel_requested():
            return
        self.annotation_controller.on_selection_cancel_requested()

    def _on_annotation_undo_requested(self) -> None:
        """Undo the last committed annotation apply action."""
        if self.annotation_controller.on_undo_last_applied_annotation_requested():
            self._mark_active_session_dirty()
            self.status_message("Derniere annotation appliquee annulee.", timeout_ms=1800)
            return
        self.status_message("Aucune annotation appliquee a annuler.", timeout_ms=1800)

    def _on_annotation_redo_requested(self) -> None:
        """Redo the last committed annotation undo action."""
        if self.annotation_controller.on_redo_last_applied_annotation_requested():
            self._mark_active_session_dirty()
            self.status_message("Derniere annotation reappliquee.", timeout_ms=1800)
            return
        self.status_message("Aucune annotation a reappliquer.", timeout_ms=1800)

    def _on_roi_delete_requested(self) -> None:
        """Delete ROI/temp previews and clear mod pending edits consistently."""
        self.mask_modification_controller.on_roi_delete_requested()
        self.annotation_controller.on_roi_delete_requested()

    @staticmethod
    def _normalize_colormap_name(name: str) -> str:
        text = str(name).strip()
        lowered = text.casefold()
        if lowered == "omniscan":
            return "OmniScan"
        if lowered in {"gray", "gris"}:
            return "Gris"
        return text or "Gris"

    def _get_colormap_lut(self, name: str) -> Optional[np.ndarray]:
        """Return LUT (256x3 float) for known colormap names."""
        if str(name).lower() != "omniscan":
            return None
        if self._omniscan_lut is not None:
            return self._omniscan_lut
        self._omniscan_lut = self._load_omniscan_colormap()
        return self._omniscan_lut

    def _load_omniscan_colormap(self) -> Optional[np.ndarray]:
        lut_path = Path(__file__).resolve().parent.parent / "OmniScanColorMap.npy"
        try:
            if not lut_path.exists():
                self.logger.warning("Colormap file not found: %s", lut_path)
                return None
            arr = np.load(lut_path)
            if arr.shape != (256, 3):
                self.logger.warning("Unexpected colormap shape %s at %s", arr.shape, lut_path)
                return None
            arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
            return arr
        except Exception as exc:
            self.logger.error("Failed to load OmniScan colormap: %s", exc)
            return None

    def _on_slice_changed(self, index: int) -> None:
        """Handle slice change events."""
        volume = self._current_volume()
        if volume is None:
            return
        clamped = max(0, min(volume.shape[0] - 1, int(index)))
        self.view_state_model.set_slice(index)
        clamped = self.view_state_model.current_slice
        self.tools_panel.set_primary_slice_value(clamped)
        self.endview_controller.set_slice(clamped)
        self.annotation_controller.on_slice_changed(clamped)
        self.mask_modification_controller.on_slice_changed(clamped)
        self.cscan_controller.highlight_slice(clamped)
        self.volume_view.set_slice_index(clamped, update_slider=True)
        self._update_ascan_trace()
        self._update_endview_label()
        self.corrosion_profile_controller.sync_anchors()

    def _on_previous_slice(self) -> None:
        """Navigate to the previous slice via buttons/shortcuts."""
        self._navigate_slice_delta(-1)

    def _on_next_slice(self) -> None:
        """Navigate to the next slice via buttons/shortcuts."""
        self._navigate_slice_delta(1)

    def _navigate_slice_delta(self, delta: int) -> None:
        volume = self._current_volume()
        if volume is None:
            return
        current = self.view_state_model.current_slice
        self._on_slice_changed(current + int(delta))

    def _on_secondary_slice_changed(self, index: int) -> None:
        """Handle secondary orthogonal slice changes."""
        volume = self._current_volume()
        if volume is None or getattr(volume, "ndim", 0) != 3:
            return
        self.view_state_model.set_secondary_slice_bounds(0, volume.shape[2] - 1)
        self.view_state_model.set_secondary_slice(index)
        clamped = self.view_state_model.secondary_slice
        self.tools_panel.set_secondary_slice_value(clamped)
        self.volume_view.set_secondary_slice_index(clamped, update_slider=True, emit=False)
        current_point = self.view_state_model.current_point
        current_x = int(current_point[0]) if current_point is not None else None
        current_y = (
            int(current_point[1])
            if current_point is not None
            else int(volume.shape[1]) // 2
        )
        if current_x == clamped:
            self._sync_secondary_endview_state()
            self.annotation_controller.refresh_secondary_roi_overlay()
            return

        # Keep A-scan/C-scan/Main endview in sync when X changes from the secondary view.
        self._update_ascan_trace(point=(clamped, current_y))
        synced_point = self.view_state_model.current_point
        if synced_point is not None:
            self.tools_panel.set_position_label(synced_point[0], synced_point[1])
        else:
            self.tools_panel.set_position_label(clamped, current_y)
        self.annotation_controller.refresh_secondary_roi_overlay()

    def _on_cross_toggled(self, enabled: bool) -> None:
        """Handle crosshair visibility toggle."""
        self.view_state_model.set_show_cross(enabled)
        self.tools_panel.set_cross_checked(enabled)
        self._set_action_checked(getattr(self.ui, "actionToggle_cross", None), enabled)
        self.endview_controller.set_cross_visible(enabled)
        self.cscan_controller.set_cross_visible(enabled)
        self.ascan_controller.set_marker_visible(enabled)

    def _on_overlay_toggled(self, enabled: bool) -> None:
        """Handle overlay visibility toggle from menu or tools panel."""
        self.annotation_controller.on_overlay_toggled(enabled)
        self.tools_panel.set_overlay_checked(enabled)
        self._set_action_checked(getattr(self.ui, "actionToggle_overlay", None), enabled)

    def _on_endview_point_selected(self, pos: Any) -> None:
        """Handle point selection for crosshair sync."""
        point = self.endview_controller.on_point_selected(pos)
        if point is None:
            return
        x, y = point
        self.tools_panel.set_position_label(x, y)
        self._update_ascan_trace(point=(x, y))
        self._on_secondary_slice_changed(x)

    def _on_endview_drag_update(self, pos: Any) -> None:
        """Handle drag updates during drawing (cursor label only)."""
        point = self.endview_controller.on_drag_update(pos)
        if point is None:
            return
        x, y = point
        self.tools_panel.set_position_label(x, y)

    def _on_cscan_crosshair_changed(self, slice_idx: int, x: int) -> None:
        """Handle crosshair movement on the C-Scan view."""
        volume = self._current_volume()
        if volume is None:
            return
        point = self.cscan_controller.on_crosshair_changed(
            slice_idx=slice_idx,
            x=x,
            volume_shape=tuple(volume.shape),
            current_point=self.view_state_model.current_point,
        )
        clamped_slice = self.view_state_model.current_slice
        self.tools_panel.set_primary_slice_value(clamped_slice)
        self.endview_controller.set_slice(clamped_slice)
        self.annotation_controller.on_slice_changed(clamped_slice)
        self.mask_modification_controller.on_slice_changed(clamped_slice)
        self._update_ascan_trace(point=point)
        if point is not None:
            self._on_secondary_slice_changed(point[0])

    def _on_cscan_slice_requested(self, z: int) -> None:
        """Handle slice requests originating from the C-Scan view."""
        self._on_slice_changed(z)

    def _on_ascan_position_changed(self, profile_idx: int) -> None:
        """Handle A-Scan position changes."""
        selection = self.ascan_controller.on_position_changed(self.nde_model, profile_idx)
        if selection is None:
            return
        point, new_slice = selection
        if new_slice is not None:
            self._on_slice_changed(new_slice)
        self._update_ascan_trace(point=point)
        if point is not None:
            self._on_secondary_slice_changed(point[0])

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
        if isinstance(view_params, dict):
            self.view_state_model.set_camera_state(view_params)

    def _on_label_added(self, label_id: int, color: Any) -> None:
        """Forward label addition then resync tool panel labels."""
        self.annotation_controller.on_label_added(label_id, color)
        self.view_state_model.set_active_label(label_id)
        self._sync_tools_labels(select_label_id=label_id)

    def _on_label_deleted(self, label_id: int) -> None:
        """Forward label deletion then resync tool panel labels."""
        if int(label_id) in PERSISTENT_LABEL_IDS:
            self._sync_tools_labels(select_label_id=self.view_state_model.active_label)
            return
        self.annotation_controller.on_label_deleted(label_id)
        if self.view_state_model.active_label == int(label_id):
            self.view_state_model.set_active_label(None)
        self._sync_tools_labels(select_label_id=None)

    def _sync_tools_labels(self, select_label_id: Optional[int] = None) -> None:
        """Sync the label list in the tools panel with the annotation model."""
        self.annotation_model.ensure_persistent_labels()
        self.temp_mask_model.ensure_persistent_labels()
        palette = self.annotation_model.get_label_palette()
        labels = sorted(palette.keys()) if palette else []
        current = select_label_id if select_label_id is not None else self.view_state_model.active_label
        if current not in labels:
            current = next((label_id for label_id in PERSISTENT_LABEL_IDS if label_id in labels), None)
        self.view_state_model.set_active_label(current)
        self.tools_panel.set_labels(labels, current=current)
        self.mask_modification_controller.on_active_label_changed(-1 if current is None else int(current))
        self._sync_erase_label_choices()
        self._sync_corrosion_label_choices()

    def _sync_erase_label_choices(self) -> None:
        """Sync the label-0 erase target choices with the current label palette."""
        palette = self.annotation_model.get_label_palette()
        labels = sorted(lbl for lbl in palette.keys() if int(lbl) != 0) if palette else []
        current = getattr(self.view_state_model, "label0_erase_target", None)
        if current not in labels:
            current = None
            self.view_state_model.set_label0_erase_target(None)
        self.nde_settings_view.set_erase_label_choices(labels, current=current)

    def _sync_corrosion_label_choices(self) -> None:
        """Sync corrosion label choices with current labels and defaults."""
        labels = self._get_corrosion_labels()
        current_a = getattr(self.view_state_model, "corrosion_label_a", None)
        current_b = getattr(self.view_state_model, "corrosion_label_b", None)
        label_a, label_b = CorrosionLabelService.normalize_pair(
            labels,
            label_a=current_a,
            label_b=current_b,
        )
        self.view_state_model.set_corrosion_label_pair(label_a, label_b)
        self.corrosion_settings_view.set_label_choices(
            labels,
            current_a=label_a,
            current_b=label_b,
        )

    def _get_corrosion_labels(self) -> list[int]:
        palette = self.annotation_model.get_label_palette()
        if not palette:
            return []
        return [int(lbl) for lbl in palette.keys() if int(lbl) > 0]

    def _on_corrosion_label_a_changed(self, value: Optional[int]) -> None:
        try:
            label_a = int(value) if value is not None else None
        except Exception:
            label_a = None
        self.view_state_model.set_corrosion_label_a(label_a)
        self._sync_corrosion_label_choices()

    def _on_corrosion_label_b_changed(self, value: Optional[int]) -> None:
        try:
            label_b = int(value) if value is not None else None
        except Exception:
            label_b = None
        self.view_state_model.set_corrosion_label_b(label_b)
        self._sync_corrosion_label_choices()

    def run(self) -> None:
        """Launch the main window."""
        self.main_window.show()

    def status_message(self, message: str, timeout_ms: int = 3000) -> None:
        """Display a transient status message."""
        if hasattr(self.ui, "statusbar") and self.ui.statusbar:
            self.ui.statusbar.showMessage(message, timeout_ms)

    def _set_action_checked(self, action: Optional[Any], enabled: bool) -> None:
        """Update a toggle action without retriggering its slot."""
        if action is None:
            return
        action.blockSignals(True)
        action.setChecked(bool(enabled))
        action.blockSignals(False)

    def _sync_display_toggle_actions(self) -> None:
        """Sync menu toggle actions with the current overlay/cross state."""
        self._set_action_checked(
            getattr(self.ui, "actionToggle_cross", None),
            self.view_state_model.show_cross,
        )
        self._set_action_checked(
            getattr(self.ui, "actionToggle_overlay", None),
            self.view_state_model.show_overlay,
        )

    def _sync_tools_coordinate_labels(self) -> None:
        """Push primary/secondary axis names into the tools panel."""
        self.tools_panel.set_primary_axis_name(self._annotation_axis_name)
        self.tools_panel.set_secondary_axis_name(self._secondary_axis_name)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _prompt_nde_open_options(
        self,
        model: NdeModel,
    ) -> Optional[tuple[str, NdeSignalProcessingOptions]]:
        """Ask for axis mode and optional signal processing before opening the NDE."""
        choices = self.annotation_axis_service.axis_mode_choices(model)
        current = self.annotation_axis_service.normalize_axis_mode(
            self._annotation_axis_mode,
            choices,
        )
        transform_info = self.nde_signal_processing_service.coerce_transform_info(
            model.metadata.get("signal_transform_info")
        )
        defaults = self.nde_signal_processing_service.default_processing_options(
            transform_info
        )

        dialog = NdeOpenOptionsDialog(
            axis_choices=choices,
            current_axis_mode=current,
            detected_title=self.nde_signal_processing_service.build_detection_title(
                transform_info
            ),
            detected_lines=self.nde_signal_processing_service.build_detection_lines(
                transform_info
            ),
            default_apply_hilbert=defaults.apply_hilbert,
            default_apply_smoothing=defaults.apply_smoothing,
            parent=self.main_window,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        axis_mode, apply_hilbert, apply_smoothing = dialog.get_selection()
        return axis_mode, NdeSignalProcessingOptions(
            apply_hilbert=apply_hilbert,
            apply_smoothing=apply_smoothing,
        )

    def _prompt_annotation_axis_mode(self, model: NdeModel) -> Optional[str]:
        """Ask which coordinate axis should be the editable annotation plane."""
        choices = self.annotation_axis_service.axis_mode_choices(model)
        if len(choices) <= 1:
            return "Auto"

        current = self.annotation_axis_service.normalize_axis_mode(
            self._annotation_axis_mode,
            choices,
        )
        current_idx = choices.index(current)
        selection, ok = QInputDialog.getItem(
            self.main_window,
            "Plan d'annotation",
            "Plan autorise pour annotation:",
            choices,
            current_idx,
            False,
        )
        if not ok:
            return None
        return str(selection)

    def _apply_annotation_axis_mode(self, model: NdeModel, axis_mode: str) -> None:
        """Force U/V as primary slice axis when requested by the user."""
        warning_message = self.annotation_axis_service.apply_axis_mode(model, axis_mode)
        if warning_message:
            self.logger.warning("%s", warning_message)

    def _load_nde_file(
        self,
        file_path: str,
        *,
        prompt_open_options: bool,
        axis_mode: Optional[str] = None,
        processing_options: Optional[NdeSignalProcessingOptions] = None,
    ) -> bool:
        """Load an NDE, optionally reusing persisted open parameters."""
        loaded_model = self.nde_loader.load(file_path)
        if prompt_open_options:
            open_selection = self._prompt_nde_open_options(loaded_model)
            if open_selection is None:
                return False
            axis_mode, processing_options = open_selection
        else:
            choices = self.annotation_axis_service.axis_mode_choices(loaded_model)
            axis_mode = self.annotation_axis_service.normalize_axis_mode(axis_mode, choices)
            processing_options = processing_options or NdeSignalProcessingOptions()

        self.nde_signal_processing_service.apply_processing_to_model(
            loaded_model,
            processing_options,
        )
        self._annotation_axis_mode = str(axis_mode or "Auto")
        self._apply_annotation_axis_mode(loaded_model, self._annotation_axis_mode)
        volume = loaded_model.get_active_volume()
        if volume is None or getattr(volume, "ndim", 0) != 3:
            raise ValueError("Le fichier NDE ne contient pas de volume 3D exploitable.")

        self._update_coordinate_dock_titles_from_model(loaded_model)
        self._initialize_ascan_logger(file_path)

        self.nde_model = loaded_model
        num_slices = volume.shape[0]
        self.view_state_model.set_slice_bounds(0, num_slices - 1)
        self.view_state_model.set_slice(0)
        self.view_state_model.set_secondary_slice_bounds(0, volume.shape[2] - 1)
        self.view_state_model.set_secondary_slice(volume.shape[2] // 2)
        self.view_state_model.set_current_point(None)
        self.view_state_model.set_apply_volume_range(0, num_slices - 1, include_current=True)
        self.annotation_controller.reset_overlay_state(preserve_labels=True)
        self.annotation_model.initialize(volume.shape)
        self.temp_mask_model.clear()
        self.temp_mask_model.initialize(volume.shape)
        self.roi_model.clear()
        self.annotation_controller.sync_overlay_settings()
        self.cscan_controller.reset_corrosion()
        self.mask_modification_controller.reset()

        self.session_manager.reset_for_new_dataset(
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
        )
        self._clear_session_runtime_state(remove_autosaves=True)
        self._refresh_session_dialog()

        self.tools_panel.set_primary_slice_bounds(0, num_slices - 1)
        self.tools_panel.set_primary_slice_value(0)
        self.tools_panel.set_secondary_slice_bounds(0, volume.shape[2] - 1)
        self.tools_panel.set_secondary_slice_value(volume.shape[2] // 2)
        self._sync_tools_labels()

        axis_order = loaded_model.metadata.get("axis_order", [])
        positions = loaded_model.metadata.get("positions") or {}
        self.view_state_model.set_axis_order(axis_order)
        axes_info = []
        for idx, name in enumerate(axis_order):
            shape_len = volume.shape[idx] if idx < len(volume.shape) else "?"
            pos_len = len(positions.get(name, [])) if positions.get(name) is not None else "?"
            axes_info.append(f"{name}: shape={shape_len}, positions={pos_len}")
        self.logger.info(
            "NDE loaded | structure=%s | shape=%s | axes=%s | path=%s",
            loaded_model.metadata.get("structure"),
            volume.shape,
            "; ".join(axes_info) if axes_info else "n/a",
            loaded_model.metadata.get("path"),
        )
        self.ascan_service.log_preview(self.logger, self.nde_model, volume)

        self._refresh_views()
        self._nde_path = file_path
        self._update_nde_label()
        self._update_endview_label()
        self._sync_apply_volume_range_view()

        processing_label = self.nde_signal_processing_service.describe_selection(
            processing_options
        )
        self.status_message(f"NDE charge: {file_path} | {processing_label}")
        return True

    def _current_signal_processing_selection(self) -> dict[str, bool]:
        """Return the current NDE processing flags stored in model metadata."""
        if self.nde_model is None:
            return {
                "apply_hilbert": False,
                "apply_smoothing": False,
            }
        selection = (self.nde_model.metadata or {}).get("signal_processing_selection")
        if not isinstance(selection, dict):
            return {
                "apply_hilbert": False,
                "apply_smoothing": False,
            }
        return {
            "apply_hilbert": bool(selection.get("apply_hilbert", False)),
            "apply_smoothing": bool(selection.get("apply_smoothing", False)),
        }

    def _save_active_session(self, *, force_dialog: bool) -> Optional[str]:
        """Save the active session, optionally forcing the save dialog."""
        return self.session_workspace_controller.save_active_session(force_dialog=force_dialog)

    def _save_session(
        self,
        session_id: str,
        *,
        force_dialog: bool,
        clean_after_save: bool,
    ) -> Optional[str]:
        """Save one session by id, optionally forcing the save dialog."""
        return self.session_workspace_controller.save_session(
            session_id,
            force_dialog=force_dialog,
            clean_after_save=clean_after_save,
        )

    def _prepare_active_session_for_persistence(self) -> bool:
        """Apply pending live edits so the saved session reflects the visible document."""
        return self.session_workspace_controller._prepare_active_session_for_persistence()

    def _persist_session_to_destination(self, session_id: str, destination: str) -> str:
        """Serialize one session to the provided destination path."""
        return self.session_workspace_controller._persist_session_to_destination(
            session_id,
            destination,
        )

    def _set_session_dirty(
        self,
        session_id: str,
        dirty: bool,
        *,
        schedule_autosave: bool = True,
    ) -> None:
        self.session_workspace_controller._set_session_dirty(
            session_id,
            dirty,
            schedule_autosave=schedule_autosave,
        )

    def _mark_active_session_dirty(self) -> None:
        self.session_workspace_controller.mark_active_session_dirty()

    def _is_session_dirty(self, session_id: str) -> bool:
        return self.session_workspace_controller._is_session_dirty(session_id)

    def _active_session_has_pending_runtime_edits(self) -> bool:
        return self.session_workspace_controller._active_session_has_pending_runtime_edits()

    def _session_has_unsaved_changes(self, session_id: str) -> bool:
        return self.session_workspace_controller._session_has_unsaved_changes(session_id)

    def _ordered_unsaved_session_ids(self) -> list[str]:
        return self.session_workspace_controller._ordered_unsaved_session_ids()

    def _confirm_unsaved_sessions_before_reset(self, action_label: str) -> bool:
        """Prompt the user about unsaved sessions before replacing the dataset/session set."""
        return self.session_workspace_controller.confirm_unsaved_sessions_before_reset(
            action_label,
        )

    def _confirm_unsaved_sessions(self, session_ids: list[str], *, action_label: str) -> bool:
        return self.session_workspace_controller._confirm_unsaved_sessions(
            session_ids,
            action_label=action_label,
        )

    def _confirm_unsaved_session(self, session_id: str, *, action_label: str) -> bool:
        return self.session_workspace_controller._confirm_unsaved_session(
            session_id,
            action_label=action_label,
        )

    def _schedule_session_autosave(self, session_id: str) -> None:
        self.session_workspace_controller._schedule_session_autosave(session_id)

    def _flush_session_autosaves(self) -> None:
        self.session_workspace_controller._flush_session_autosaves()

    def _autosave_session_path(self, session_id: str) -> str:
        nde_path = self._require_nde_path()
        return self.session_workspace_controller._autosave_session_path(
            session_id,
            nde_path=nde_path,
        )

    def _cleanup_session_autosave(self, session_id: str) -> None:
        self.session_workspace_controller._cleanup_session_autosave(session_id)

    def _clear_session_runtime_state(self, *, remove_autosaves: bool) -> None:
        self.session_workspace_controller.clear_runtime_state(remove_autosaves=remove_autosaves)

    def _suggest_session_save_path(
        self,
        *,
        session_name: Optional[str],
        current_path: Optional[str] = None,
    ) -> str:
        """Return the default destination proposed by the save dialog."""
        return self.session_workspace_controller._suggest_session_save_path(
            session_name=session_name,
            current_path=current_path,
        )

    def _extract_single_session_dump(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Reduce any persisted dump to a single active session payload."""
        return self.session_workspace_controller._extract_single_session_dump(payload)

    @staticmethod
    def _normalize_session_name(session_name: Optional[str], *, fallback: str = "New session") -> str:
        return SessionWorkspaceController._normalize_session_name(
            session_name,
            fallback=fallback,
        )

    @staticmethod
    def _session_name_from_path(file_path: str, *, fallback: Optional[str] = None) -> str:
        return SessionWorkspaceController._session_name_from_path(
            file_path,
            fallback=fallback,
        )

    def _require_nde_path(self) -> str:
        """Return the current NDE path or fail if none is available."""
        nde_path = self._nde_path
        if not nde_path and self.nde_model is not None:
            nde_path = str((self.nde_model.metadata or {}).get("path") or "")
        if not nde_path:
            raise ValueError("Chemin NDE indisponible pour la session courante.")
        return nde_path

    def _update_coordinate_dock_titles_from_model(self, model: NdeModel) -> None:
        """Reflect active and secondary axes in dock titles."""
        titles = self.annotation_axis_service.build_coordinate_dock_titles(
            model,
            axis_mode=self._annotation_axis_mode,
        )
        self._annotation_axis_name = titles.primary_axis_name
        self._secondary_axis_name = titles.secondary_axis_name
        self.ucoordinate_dock.setWindowTitle(titles.primary_title)
        self.vcoordinate_dock.setWindowTitle(titles.secondary_title)
        self._sync_tools_coordinate_labels()

    def _sync_secondary_endview_state(self) -> None:
        """Apply secondary slice/crosshair state to the secondary endview."""
        volume = self._current_volume()
        if volume is None or getattr(volume, "ndim", 0) != 3:
            return
        self.view_state_model.set_secondary_slice_bounds(0, volume.shape[2] - 1)
        self.view_state_model.set_secondary_slice(self.view_state_model.secondary_slice)
        self.endview_controller.set_secondary_slice(self.view_state_model.secondary_slice)
        crosshair = self.annotation_axis_service.secondary_crosshair(
            current_slice=self.view_state_model.current_slice,
            current_point=self.view_state_model.current_point,
        )
        if crosshair is not None:
            self.endview_controller.set_secondary_crosshair(*crosshair)

    def _refresh_views(self) -> None:
        """Push the current volume state into all views."""
        volume = self._current_volume()
        if volume is None:
            return
        self.view_state_model.set_slice_bounds(0, volume.shape[0] - 1)
        self.view_state_model.set_secondary_slice_bounds(0, volume.shape[2] - 1)

        # Sélectionne l’indice de tranche courant dans les bornes valides
        slice_idx = self.view_state_model.clamp_slice(self.view_state_model.current_slice)
        self.view_state_model.set_secondary_slice(self.view_state_model.secondary_slice)
        secondary_slice_idx = self.view_state_model.secondary_slice
        self.tools_panel.set_primary_slice_bounds(0, volume.shape[0] - 1)
        self.tools_panel.set_secondary_slice_bounds(0, volume.shape[2] - 1)
        self.tools_panel.set_primary_slice_value(slice_idx)
        self.tools_panel.set_secondary_slice_value(secondary_slice_idx)

        # Met à jour l’Endview (pas de changement ici)
        self.endview_controller.set_volume(volume)
        secondary_volume = self.annotation_axis_service.build_secondary_volume(volume)
        if secondary_volume is not None:
            self.endview_controller.set_secondary_volume(secondary_volume)
        self.endview_controller.set_slice(slice_idx)
        self.endview_controller.set_secondary_slice(secondary_slice_idx)
        self.annotation_controller.on_slice_changed(slice_idx)
        self.mask_modification_controller.on_slice_changed(slice_idx)
        self.annotation_controller.ensure_restriction_rect(
            shape=(volume.shape[1], volume.shape[2])
        )

        # Met à jour la C‑scan (standard ou corrosion)
        self.cscan_controller.update_views(volume)
        self.endview_controller.sync_mode()

        # Récupère l'ordre des axes depuis le modèle, s'il existe
        axis_order = self.view_state_model.axis_order

        # Envoie le volume à la vue 3D en précisant l’ordre des axes
        self.volume_view.set_volume(volume, slice_idx=slice_idx, axis_order=axis_order)
        self.volume_view.set_secondary_slice_index(
            secondary_slice_idx,
            update_slider=True,
            emit=False,
        )

        # Applique l’overlay après la (re)construction de la scène 3D
        self.annotation_controller.refresh_overlay(rebuild=False)

        # Reste des mises à jour
        self._update_ascan_trace()
        self.endview_controller.set_cross_visible(self.view_state_model.show_cross)
        self.cscan_controller.set_cross_visible(self.view_state_model.show_cross)
        self.ascan_controller.set_marker_visible(self.view_state_model.show_cross)


    # ------------------------------------------------------------------ #
    # Session handling
    # ------------------------------------------------------------------ #
    def _open_session_dialog(self) -> None:
        """Affiche le gestionnaire de sessions et connecte les signaux."""
        self.session_workspace_controller.open_session_dialog()

    def _refresh_session_dialog(self) -> None:
        self.session_workspace_controller.refresh_session_dialog()

    def _on_session_created(self, name: str) -> None:
        self.session_workspace_controller.on_session_created(name)

    def _on_session_duplicated(self, name: str) -> None:
        self.session_workspace_controller.on_session_duplicated(name)

    def _on_session_selected(self, session_id: str) -> None:
        self.session_workspace_controller.on_session_selected(session_id)

    def _on_session_deleted(self, session_id: str) -> None:
        self.session_workspace_controller.on_session_deleted(session_id)

    def _after_session_switch(self) -> None:
        """Synchronise l'état du modèle actif vers les vues."""
        self.annotation_controller.clear_apply_history()
        self.tools_panel.set_overlay_checked(self.view_state_model.show_overlay)
        self.tools_panel.set_cross_checked(self.view_state_model.show_cross)
        self._sync_display_toggle_actions()
        self._sync_tools_coordinate_labels()
        current_tool_mode = self.view_state_model.tool_mode or self.tools_panel.current_tool_mode()
        if current_tool_mode:
            self.tools_panel.select_tool_mode(current_tool_mode)
            self.annotation_controller.on_tool_mode_changed(current_tool_mode)
            self.mask_modification_controller.on_tool_mode_changed(current_tool_mode)
        self.mask_modification_controller.reset()
        self.endview_controller.set_cross_visible(self.view_state_model.show_cross)
        self.cscan_controller.set_cross_visible(self.view_state_model.show_cross)
        self.ascan_controller.set_marker_visible(self.view_state_model.show_cross)

        # Colormaps
        self.endview_controller.set_colormap(self.view_state_model.endview_colormap, None)
        self.volume_view.set_base_colormap(self.view_state_model.endview_colormap, None)
        self.cscan_controller.set_colormap(self.view_state_model.cscan_colormap, None)
        self.tools_panel.set_endview_colormap(self.view_state_model.endview_colormap)
        self.nde_settings_view.set_colormaps(
            endview=self.view_state_model.endview_colormap,
            cscan=self.view_state_model.cscan_colormap,
        )

        self._sync_tools_labels()
        self.annotation_controller.sync_overlay_settings()
        self.annotation_controller.apply_overlay_opacity()
        self.tools_panel.set_overlay_opacity(self.view_state_model.overlay_alpha)
        # Rafraîchir le volume puis réappliquer l'overlay pour forcer le push 3D
        self._refresh_views()
        self.corrosion_profile_controller.sync_anchors()

    # ------------------------------------------------------------------ #
    # Corrosion completion handling
    # ------------------------------------------------------------------ #
    def _on_corrosion_completed(self, result) -> None:
        """Crée une session interpolée sans écraser l'état brut."""
        self.corrosion_profile_edit_service.reset()
        # 1) Met à jour la session active avec le résultat brut (projection déjà active)
        self.mask_modification_controller.reset()
        self.view_state_model.corrosion_interpolated_projection = None
        self.view_state_model.corrosion_overlay_volume = result.overlay_volume
        self.view_state_model.corrosion_overlay_palette = result.overlay_palette
        self.view_state_model.corrosion_overlay_label_ids = result.overlay_label_ids
        self.view_state_model.corrosion_peak_index_map_a = result.peak_index_map_a
        self.view_state_model.corrosion_peak_index_map_b = result.peak_index_map_b
        self.view_state_model.corrosion_ascan_support_map = result.ascan_support_map

        origin_id = self._pre_corrosion_session_id
        origin_state = self._pre_corrosion_session_state
        if (
            origin_id is not None
            and origin_state is not None
            and origin_id in self.session_manager._sessions  # noqa: SLF001
        ):
            try:
                self.session_manager._sessions[origin_id] = copy.deepcopy(origin_state)  # noqa: SLF001
            except Exception:
                self.session_manager._sessions[origin_id] = origin_state  # noqa: SLF001

        # 2) Prépare un état interpolé pour la nouvelle session
        interpolated_view_state = copy.deepcopy(self.view_state_model)
        if result.interpolated_projection is not None and result.interpolated_value_range is not None:
            interpolated_view_state.corrosion_projection = (
                result.interpolated_projection,
                result.interpolated_value_range,
            )
            interpolated_view_state.corrosion_interpolated_projection = (
                result.interpolated_projection,
                result.interpolated_value_range,
            )

        # Remplace les masques de la session active par l'overlay interpolé pour la nouvelle session
        if result.overlay_volume is not None:
            self.annotation_model.set_mask_volume(result.overlay_volume)
            palette = result.overlay_palette or {}
            self.annotation_model.label_palette = dict(palette)
            vis = {lbl: True for lbl in palette.keys()}
            self.annotation_model.label_visibility = vis
            self.annotation_model.ensure_persistent_labels()

        # Nettoie les masques/temp et ROIs pour la session interpolée
        self.temp_mask_model.clear()
        if result.overlay_volume is not None:
            self.temp_mask_model.initialize(result.overlay_volume.shape)
        self.roi_model.clear()

        # Applique la projection interpolée sur le modèle de vue actif
        if result.interpolated_projection is not None and result.interpolated_value_range is not None:
            self.view_state_model.corrosion_projection = (
                result.interpolated_projection,
                result.interpolated_value_range,
            )
            self.view_state_model.corrosion_interpolated_projection = (
                result.interpolated_projection,
                result.interpolated_value_range,
            )
            self.view_state_model.corrosion_active = True

        origin_name = getattr(origin_state, "name", None)
        name = f"{self._normalize_session_name(origin_name)} corrosion"

        # 3) Crée la nouvelle session sans sauvegarder l'active (déjà mise à jour), puis bascule dessus
        new_session_id = self.session_manager.create_from_models(
            name=name,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=interpolated_view_state,
            set_active=True,
            save_active=False,
        )
        self.session_workspace_controller.register_unsaved_session(new_session_id, dirty=True)

        self._pre_corrosion_session_state = None
        self._pre_corrosion_session_id = None

        self._after_session_switch()
        self.annotation_controller.refresh_overlay(defer_volume=False, rebuild=True)
        has_distance = (
            result.piece_volume_raw is not None and result.piece_volume_raw.size > 0
        ) or (
            result.piece_volume_interpolated is not None and result.piece_volume_interpolated.size > 0
        )
        has_legacy = (
            result.piece_volume_legacy_raw is not None and result.piece_volume_legacy_raw.size > 0
        ) or (
            result.piece_volume_legacy_interpolated is not None
            and result.piece_volume_legacy_interpolated.size > 0
        )
        if has_distance or has_legacy:
            self._piece_anchor = result.piece_anchor
            self._show_piece3d_volume(
                raw_volume=result.piece_volume_raw,
                interpolated_volume=result.piece_volume_interpolated,
                legacy_raw_volume=result.piece_volume_legacy_raw,
                legacy_interpolated_volume=result.piece_volume_legacy_interpolated,
            )

    def _current_volume(self) -> Optional[Any]:
        if self.nde_model is None:
            return None
        return self.nde_model.get_active_volume()

    def _on_piece3d_toggled(self, checked: bool) -> None:
        """Show/hide the piece3D window from the Analyse menu."""
        if checked:
            self._show_piece3d_window()
        else:
            self._close_piece3d_window()

    def _sync_piece3d_action(self, checked: bool) -> None:
        """Update the menu action without re-triggering the toggle handler."""
        action = getattr(self.ui, "actionAfficher_solide_3d", None)
        if action is None:
            return
        action.blockSignals(True)
        action.setChecked(bool(checked))
        action.blockSignals(False)

    def _ensure_piece3d_window(self) -> None:
        """Create the piece3D window lazily and wire close tracking."""
        if self._piece3d_window is not None:
            return
        self._piece3d_window = QDialog(self.main_window)
        self._piece3d_window.setWindowTitle("Pièce 3D corrosion")
        self._piece3d_window.finished.connect(self._on_piece3d_window_closed)
        layout = QVBoxLayout(self._piece3d_window)
        self._piece3d_view = Piece3DView(parent=self._piece3d_window)
        self._piece_toggle_btn = QPushButton("Afficher version brute", parent=self._piece3d_window)
        self._piece_toggle_btn.clicked.connect(self._toggle_piece_volume)
        layout.addWidget(self._piece3d_view)
        layout.addWidget(self._piece_toggle_btn)

    def _show_piece3d_window(self) -> None:
        """Open the piece3D window using the latest available volume."""
        self._ensure_piece3d_window()
        if self._piece3d_view is None or self._piece3d_window is None:
            return

        has_interp = (
            self._piece_volume_interpolated is not None and self._piece_volume_interpolated.size > 0
        ) or (
            self._piece_volume_legacy_interpolated is not None
            and self._piece_volume_legacy_interpolated.size > 0
        )
        has_raw = (self._piece_volume_raw is not None and self._piece_volume_raw.size > 0) or (
            self._piece_volume_legacy_raw is not None and self._piece_volume_legacy_raw.size > 0
        )
        if has_interp:
            self._piece_show_interpolated = True
        elif has_raw:
            self._piece_show_interpolated = False

        self._piece3d_view.set_piece_volume_sources(
            distance_raw=self._piece_volume_raw,
            distance_interpolated=self._piece_volume_interpolated,
            legacy_raw=self._piece_volume_legacy_raw,
            legacy_interpolated=self._piece_volume_legacy_interpolated,
        )
        self._piece3d_view.set_piece_show_interpolated(self._piece_show_interpolated)
        self._piece3d_view.set_anchor_point(self._piece_anchor)
        self._update_piece_toggle_label()

        self._piece3d_window.show()
        self._piece3d_window.raise_()
        self._piece3d_window.activateWindow()
        self._sync_piece3d_action(True)

    def _close_piece3d_window(self) -> None:
        """Close the piece3D window if open."""
        if self._piece3d_window is None:
            return
        self._piece3d_window.close()

    def _on_piece3d_window_closed(self, *_args) -> None:
        """Keep the menu action in sync when the window is closed."""
        self._sync_piece3d_action(False)

    def _show_piece3d_volume(
        self,
        *,
        raw_volume: Optional[np.ndarray],
        interpolated_volume: Optional[np.ndarray],
        legacy_raw_volume: Optional[np.ndarray],
        legacy_interpolated_volume: Optional[np.ndarray],
    ) -> None:
        """Affiche ou met à jour la fenêtre flottante de la pièce 3D corrosion."""
        self._ensure_piece3d_window()
        if self._piece3d_view is None:
            return

        self._piece_volume_raw = raw_volume.astype(np.float32) if raw_volume is not None else None
        self._piece_volume_interpolated = (
            interpolated_volume.astype(np.float32) if interpolated_volume is not None else None
        )
        self._piece_volume_legacy_raw = (
            legacy_raw_volume.astype(np.float32) if legacy_raw_volume is not None else None
        )
        self._piece_volume_legacy_interpolated = (
            legacy_interpolated_volume.astype(np.float32)
            if legacy_interpolated_volume is not None
            else None
        )

        has_interp = (
            self._piece_volume_interpolated is not None and self._piece_volume_interpolated.size > 0
        ) or (
            self._piece_volume_legacy_interpolated is not None
            and self._piece_volume_legacy_interpolated.size > 0
        )
        has_raw = (self._piece_volume_raw is not None and self._piece_volume_raw.size > 0) or (
            self._piece_volume_legacy_raw is not None and self._piece_volume_legacy_raw.size > 0
        )
        if has_interp:
            self._piece_show_interpolated = True
        elif has_raw:
            self._piece_show_interpolated = False

        self._show_piece3d_window()

    def _toggle_piece_volume(self) -> None:
        """Bascule entre volume brut et volume interpolé si les deux sont disponibles."""
        if self._piece3d_view is None:
            return

        has_interp = (
            self._piece_volume_interpolated is not None and self._piece_volume_interpolated.size > 0
        ) or (
            self._piece_volume_legacy_interpolated is not None
            and self._piece_volume_legacy_interpolated.size > 0
        )
        has_raw = (self._piece_volume_raw is not None and self._piece_volume_raw.size > 0) or (
            self._piece_volume_legacy_raw is not None and self._piece_volume_legacy_raw.size > 0
        )
        if not (has_interp and has_raw):
            return

        self._piece_show_interpolated = not self._piece_show_interpolated
        self._piece3d_view.set_piece_show_interpolated(self._piece_show_interpolated)
        self._piece3d_view.set_anchor_point(self._piece_anchor)
        self._update_piece_toggle_label()

    def _update_piece_toggle_label(self) -> None:
        """Met à jour le texte du bouton selon le volume affiché."""
        if self._piece_toggle_btn is None:
            return
        if self._piece_show_interpolated:
            self._piece_toggle_btn.setText("Afficher version brute")
        else:
            self._piece_toggle_btn.setText("Afficher version interpolee")

    def _update_nde_label(self) -> None:
        """Reflect the opened NDE file into the tools panel label."""
        name = Path(self._nde_path).name if self._nde_path else "-"
        self.tools_panel.set_nde_name(name)

    def _update_endview_label(self) -> None:
        """Reflect the current slice as an endview identifier."""
        volume = self._current_volume()
        if volume is None:
            self.tools_panel.set_endview_name("-")
            return
        slice_idx = int(self.view_state_model.current_slice)
        name = f"endview_{slice_idx * 1500:012d}.png"
        self.tools_panel.set_endview_name(name)

    def _initialize_ascan_logger(self, source: str) -> None:
        """Initialize the A-Scan debug logging session."""
        try:
            from services.ascan_debug_logger import ascan_debug_logger

            ascan_debug_logger.start_session(source)
        except Exception:
            # Fail silently if logger cannot start
            return

    def _update_ascan_trace(self, point: Optional[tuple[int, int]] = None) -> None:
        volume = self._current_volume()
        self.ascan_controller.update_trace(
            self.nde_model,
            volume,
            point=point,
        )
        self._sync_secondary_endview_state()
