import logging
import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedLayout,
    QWidget,
)

from config.constants import DEFAULT_ACTIVE_LABEL_ID, PERSISTENT_LABEL_IDS, CORROSION_STAGE_BASE, CORROSION_STAGE_RAW, CORROSION_STAGE_INTERPOLATED
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
from models.imported_overlay_model import ImportedOverlayModel
from models.nde_model import NdeModel
from models.view_state_model import ViewStateModel
from models.roi_model import RoiModel
from models.temp_mask_model import TempMaskModel
from services.annotation_axis_service import AnnotationAxisService
from services.annotation_session_manager import AnnotationSessionManager
from services.annotation_service import AnnotationService
from services.ascan_service import AScanService
from services.nnunet_service import NnUnetResult, NnUnetService
from services.overlay_class_remap_service import OverlayClassRemapService
from services.overlay_loader import OverlayLoader
from services.overlay_service import OverlayService
from services.overlay_export import OverlayExport
from services.endview_export import EndviewExportService
from services.project_persistence import ProjectPersistence
from services.session_bundle_export import (
    BundleSentinelExportOptions,
    SessionBundleExportRequest,
    SessionBundleExportService,
)
from services.split_service import SplitFlawNoflawService
from services.nde_loader import NdeLoader
from services.nde_signal_processing_service import (
    NdeSignalProcessingOptions,
    NdeSignalProcessingService,
)
from services.cscan_corrosion_service import (
    CScanCorrosionService,
    CorrosionWorkflowResult,
    CorrosionWorkflowService,
)
from services.corrosion_profile_edit_service import CorrosionProfileEditService
from services.corrosion_label_service import CorrosionLabelService
from services.mask_modification_service import MaskModificationService
from ui_mainwindow import Ui_MainWindow
from views.annotation_view import AnnotationView
from views.corrosion_settings_view import CorrosionSettingsView
from views.nde_settings_view import NdeSettingsView
from views.nde_open_options_dialog import NdeOpenOptionsDialog
from views.endview_resize_dialog import EndviewResizeDialog
from views.nnunet_settings_view import NnUnetSettingsView
from views.overlay_class_remap_dialog import OverlayClassRemapDialog
from views.overlay_settings_view import OverlaySettingsView
from views.piece3d_view import Piece3DView
from views.session_bundle_export_dialog import SessionBundleExportDialog
from utils.filename_utils import sanitize_filename_component


class NnUnetUiSignals(QObject):
    """Bridge nnUNet completion callbacks back onto the UI thread."""

    success = pyqtSignal(object)
    error = pyqtSignal(object)


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
        self.imported_overlay_model = ImportedOverlayModel()
        self.view_state_model = ViewStateModel()
        self.roi_model = RoiModel()
        self.temp_mask_model = TempMaskModel()
        self.nde_loader = NdeLoader()
        self.nde_signal_processing_service = NdeSignalProcessingService()
        self.annotation_axis_service = AnnotationAxisService()
        self.session_manager = AnnotationSessionManager()
        self.project_persistence = ProjectPersistence()
        self._nde_path: Optional[str] = None
        self.overlay_class_remap_service = OverlayClassRemapService()
        self.overlay_loader = OverlayLoader()
        self.overlay_service = OverlayService()
        self.overlay_export = OverlayExport()
        self.endview_export_service = EndviewExportService(nde_loader=self.nde_loader)
        self.split_flaw_noflaw_service = SplitFlawNoflawService(
            endview_export_service=self.endview_export_service
        )
        self.nnunet_service = NnUnetService(logger=self.logger)
        self.cscan_corrosion_service = CScanCorrosionService()
        self.session_bundle_export_service = SessionBundleExportService(
            overlay_export=self.overlay_export,
            split_export_service=self.split_flaw_noflaw_service,
            cscan_corrosion_service=self.cscan_corrosion_service,
        )
        self.corrosion_workflow_service = CorrosionWorkflowService(
            cscan_corrosion_service=self.cscan_corrosion_service
        )
        self.corrosion_profile_edit_service = CorrosionProfileEditService()
        self.mask_modification_service = MaskModificationService()
        self.overlay_settings_view = OverlaySettingsView(self.main_window)
        self.nde_settings_view = NdeSettingsView(self.main_window)
        self.corrosion_settings_view = CorrosionSettingsView(self.main_window)
        self.nnunet_settings_view = NnUnetSettingsView(self.main_window)
        self._volume_stack: Optional[QStackedLayout] = None
        self._piece3d_page: Optional[QWidget] = None
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
        self._annotation_axis_mode: str = "Auto"
        self._primary_view_name: str = "D-Scan"
        self._secondary_view_name: str = "B-Scan"
        self.main_window.closeEvent = self._on_main_window_close_event  # type: ignore[method-assign]
        self._app = QApplication.instance()
        self._nnunet_ui_signals = NnUnetUiSignals()
        self._nnunet_ui_signals.success.connect(self._handle_nnunet_success)
        self._nnunet_ui_signals.error.connect(self._handle_nnunet_error)
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
        self._volume_stack = self.dock_layout_controller.volume_stack
        self._piece3d_page = self.dock_layout_controller.piece3d_page
        self._piece3d_view = self.dock_layout_controller.piece3d_view
        self._piece_toggle_btn = self.dock_layout_controller.piece3d_toggle_button
        if self._piece_toggle_btn is not None:
            self._piece_toggle_btn.clicked.connect(self._toggle_piece_volume)

        self.endview_controller = EndviewController(
            standard_view=self.annotation_view,
            corrosion_view=self.annotation_view_corrosion,
            secondary_view=self.secondary_annotation_view,
            secondary_corrosion_view=self.secondary_annotation_view_corrosion,
            stacked_layout=self.annotation_stack,
            secondary_stacked_layout=self.secondary_annotation_stack,
            view_state_model=self.view_state_model,
        )
        self.endview_controller.set_primary_status_position_visible(True)
        self.endview_controller.set_secondary_status_position_visible(False)
        self.endview_controller.set_navigation_view_names(
            primary=self._primary_view_name,
            secondary=self._secondary_view_name,
        )
        self.endview_controller.set_smooth_enabled(
            getattr(self.view_state_model, "show_endview_smooth", True)
        )

        self.annotation_service = AnnotationService()

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
            session_manager=self.session_manager,
            get_volume=self._current_volume,
            on_overlay_updated=self._update_ascan_trace,
        )
        self.mask_modification_controller = MaskModificationController(
            view_state_model=self.view_state_model,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            annotation_view=self.annotation_view,
            mask_modification_service=self.mask_modification_service,
            refresh_overlay=self.annotation_controller.refresh_overlay,
            refresh_roi_overlay_for_slice=self.annotation_controller.refresh_roi_overlay_for_slice,
            set_position_label=self._set_annotation_position_label,
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
            annotation_model=self.annotation_model,
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
            set_position_label=self._set_annotation_position_label,
            status_message=self.status_message,
            apply_roi_fallback=self._apply_roi_non_corrosion,
            on_session_changed=self._on_corrosion_session_changed,
        )
        self._apply_saved_colormaps()
        self.volume_view.set_volume_planes_visible(
            getattr(self.view_state_model, "show_volume_planes", True)
        )

        self._connect_actions()
        self._connect_signals()
        self._register_shortcuts()
        self.annotation_controller.apply_overlay_opacity()
        self.ascan_controller.set_overlay_opacity(self.view_state_model.overlay_alpha)
        self._apply_annotation_action(getattr(self.view_state_model, "annotation_action", "draw"))
        self._update_main_window_title()
        self._update_endview_label()
        self._sync_display_toggle_actions()
        self._sync_cscan_labels()
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
        if hasattr(self.ui, "actionRemap_classes"):
            self.ui.actionRemap_classes.triggered.connect(self._on_remap_classes)
        if hasattr(self.ui, "actionSauvegarder"):
            self.ui.actionSauvegarder.triggered.connect(self._on_save_session)
        if hasattr(self.ui, "actionEnregistrer_sous"):
            self.ui.actionEnregistrer_sous.triggered.connect(self._on_save_session_as)
        self.ui.actionExporter_npz.triggered.connect(self._on_export_overlay_npz)
        if hasattr(self.ui, "actionExporter_endviews"):
            self.ui.actionExporter_endviews.triggered.connect(self._on_export_endviews)
        if hasattr(self.ui, "actionExport_all"):
            self.ui.actionExport_all.triggered.connect(self._on_export_all)
        if hasattr(self.ui, "actionSplit_flaw_noflaw"):
            self.ui.actionSplit_flaw_noflaw.triggered.connect(self._on_dataset_export_requested)
        self.ui.actionParam_tres.triggered.connect(self._on_open_settings)
        self.ui.actionParam_tres_2.triggered.connect(self.annotation_controller.open_overlay_settings)
        if hasattr(self.ui, "actionParam_tres_3"):
            self.ui.actionParam_tres_3.triggered.connect(self._on_open_corrosion_settings)
        if hasattr(self.ui, "actionCorrosion_analyse"):
            self.ui.actionCorrosion_analyse.triggered.connect(self._on_run_corrosion_analysis)
        if hasattr(self.ui, "actionInterpolate"):
            self.ui.actionInterpolate.triggered.connect(self._on_corrosion_interpolation_requested_from_menu)
        if hasattr(self.ui, "actionnnunet"):
            self.ui.actionnnunet.triggered.connect(self._on_run_nnunet)
        if hasattr(self.ui, "actionSettings"):
            self.ui.actionSettings.triggered.connect(self._on_open_nnunet_settings)
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
        if hasattr(self.ui, "actionToggle_overlay_ascan"):
            self.ui.actionToggle_overlay_ascan.setCheckable(True)
            self.ui.actionToggle_overlay_ascan.toggled.connect(self._on_overlay_ascan_toggled)
        if hasattr(self.ui, "actionToggle_outline_only"):
            self.ui.actionToggle_outline_only.setCheckable(True)
            self.ui.actionToggle_outline_only.toggled.connect(self._on_outline_only_toggled)
        if hasattr(self.ui, "actionToggle_plans_volume"):
            self.ui.actionToggle_plans_volume.setCheckable(True)
            self.ui.actionToggle_plans_volume.toggled.connect(self._on_volume_planes_toggled)
        if hasattr(self.ui, "actionToggle_pixel_mm"):
            self.ui.actionToggle_pixel_mm.setCheckable(True)
            self.ui.actionToggle_pixel_mm.toggled.connect(self._on_ruler_display_unit_toggled)
        if hasattr(self.ui, "actionToggle_restriction"):
            self.ui.actionToggle_restriction.setCheckable(True)
            self.ui.actionToggle_restriction.toggled.connect(self._on_restriction_toggled)
        if hasattr(self.ui, "actionToggle_Smooth"):
            self.ui.actionToggle_Smooth.setCheckable(True)
            self.ui.actionToggle_Smooth.toggled.connect(self._on_endview_smooth_toggled)
        if hasattr(self.ui, "actionToggle_Vectorise"):
            self.ui.actionToggle_Vectorise.setCheckable(True)
            self.ui.actionToggle_Vectorise.toggled.connect(self._on_interpolated_profile_vectorized_toggled)
        if hasattr(self.ui, "actionResize_endview"):
            self.ui.actionResize_endview.triggered.connect(self._on_resize_endview)
        if hasattr(self.ui, "actionR_initialisation_docks"):
            self.ui.actionR_initialisation_docks.setText("Reset docks")
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
            tool_combo=self._tools_ui.comboBox,
            action_combo=getattr(self._tools_ui, "comboBox_3", None),
            colormap_combo=self._tools_ui.comboBox_2,
            threshold_slider=self._tools_ui.horizontalSlider,
            threshold_label=self._tools_ui.label_2,
            paint_slider=self._tools_ui.horizontalSlider_3,
            overlay_opacity_slider=self._tools_ui.horizontalSlider_6,
            overlay_opacity_spinbox=self._tools_ui.spinBox_4,
            nde_opacity_slider=self._tools_ui.horizontalSlider_5,
            nde_opacity_spinbox=self._tools_ui.spinBox_3,
            nde_contrast_slider=self._tools_ui.horizontalSlider_7,
            nde_contrast_spinbox=self._tools_ui.spinBox_5,
            apply_auto_checkbox=getattr(self._tools_ui, "checkBox_5", None),
            mod_apply_auto_checkbox=getattr(self._tools_ui, "checkBox_9", None),
            force_threshold_erase_checkbox=getattr(self._tools_ui, "checkBox_4", None),
            apply_volume_checkbox=self._tools_ui.checkBox,
            threshold_auto_checkbox=self._tools_ui.checkBox_2,
            roi_persistence_checkbox=self._tools_ui.checkBox_3,
            closing_mask_checkbox=getattr(self._tools_ui, "checkBox_7", None),
            clean_outliers_checkbox=getattr(self._tools_ui, "checkBox_8", None),
            volume_view_checkbox=getattr(self._tools_ui, "checkBox_6", None),
            roi_recompute_button=self._tools_ui.pushButton_2,
            roi_delete_button=self._tools_ui.pushButton_3,
            selection_cancel_button=self._tools_ui.pushButton_4,
            apply_roi_button=self._tools_ui.pushButton_7,
            label_text_container=self._tools_ui.frame_5,
            label_color_container=self._tools_ui.frame_6,
            nde_opacity_label=getattr(self._tools_ui, "label_10", None),
            nde_contrast_label=getattr(self._tools_ui, "label_12", None),
        )

        if self.view_state_model.threshold is not None:
            self.tools_panel.set_threshold_value(int(self.view_state_model.threshold))
        self.tools_panel.set_force_threshold_erase_checked(
            getattr(self.view_state_model, "force_threshold_erase", False)
        )
        self.tools_panel.set_apply_auto_checked(getattr(self.view_state_model, "apply_auto", False))
        self.tools_panel.set_mod_apply_auto_checked(
            getattr(self.view_state_model, "mod_apply_auto", False)
        )
        self.tools_panel.set_threshold_auto_checked(self.view_state_model.threshold_auto)
        self.tools_panel.set_apply_volume_checked(self.view_state_model.apply_volume)
        self.tools_panel.set_roi_persistence_checked(self.view_state_model.roi_persistence)
        self.tools_panel.set_closing_mask_checked(
            getattr(self.view_state_model, "closing_mask_enabled", False)
        )
        self.tools_panel.set_clean_outliers_checked(
            getattr(self.view_state_model, "clean_outliers_enabled", False)
        )
        self.tools_panel.set_volume_view_overlay_checked(
            getattr(self.view_state_model, "show_volume_view_overlay", True)
        )
        self.tools_panel.set_paint_size(self.view_state_model.paint_radius)
        self.tools_panel.set_overlay_opacity(self.view_state_model.overlay_alpha)
        self.tools_panel.set_nde_opacity(self.view_state_model.nde_alpha)
        self.tools_panel.set_nde_contrast(self.view_state_model.nde_contrast)
        self.tools_panel.set_nde_opacity_available(self._current_volume() is not None)
        self._sync_apply_volume_range_view()
        self.tools_panel.set_endview_colormap(self.view_state_model.endview_colormap)
        self._sync_corrosion_label_choices()
        self._sync_corrosion_workflow_controls()
        self.tools_panel.set_annotation_action(
            getattr(self.view_state_model, "annotation_action", "draw")
        )
        self._sync_coordinate_view_labels()
        initial_tool_mode = self.view_state_model.tool_mode or self.tools_panel.current_tool_mode()
        if initial_tool_mode:
            self.tools_panel.select_tool_mode(initial_tool_mode)
            self._on_tool_mode_changed(initial_tool_mode)

        self.tools_panel.tool_mode_changed.connect(self._on_tool_mode_changed)
        self.tools_panel.annotation_action_changed.connect(self._apply_annotation_action)
        self.tools_panel.paint_size_changed.connect(self.annotation_controller.on_paint_size_changed)
        self.tools_panel.threshold_changed.connect(self.annotation_controller.on_threshold_changed)
        self.tools_panel.force_threshold_erase_toggled.connect(
            self.annotation_controller.on_force_threshold_erase_toggled
        )
        self.tools_panel.apply_auto_toggled.connect(self.annotation_controller.on_apply_auto_toggled)
        self.tools_panel.mod_apply_auto_toggled.connect(
            self.mask_modification_controller.on_mod_apply_auto_toggled
        )
        self.tools_panel.threshold_auto_toggled.connect(self.annotation_controller.on_threshold_auto_toggled)
        self.tools_panel.apply_volume_toggled.connect(self.annotation_controller.on_apply_volume_toggled)
        self.tools_panel.closing_mask_toggled.connect(self._on_closing_mask_toggled)
        self.tools_panel.clean_outliers_toggled.connect(self._on_clean_outliers_toggled)
        self.tools_panel.volume_view_overlay_toggled.connect(
            self.annotation_controller.on_volume_view_overlay_toggled
        )
        self.tools_panel.overlay_opacity_changed.connect(self._on_overlay_opacity_changed)
        self.tools_panel.nde_opacity_changed.connect(self._on_nde_opacity_changed)
        self.tools_panel.nde_contrast_changed.connect(self._on_nde_contrast_changed)
        self.tools_panel.endview_colormap_changed.connect(self._on_endview_colormap_changed)
        self.tools_panel.roi_persistence_toggled.connect(self.annotation_controller.on_roi_persistence_toggled)
        self.tools_panel.roi_recompute_requested.connect(self.annotation_controller.on_roi_recompute_requested)
        self.tools_panel.roi_delete_requested.connect(self._on_roi_delete_requested)
        self.tools_panel.selection_cancel_requested.connect(self._on_selection_cancel_requested)
        self.tools_panel.apply_roi_requested.connect(
            self.corrosion_profile_controller.on_apply_roi_requested
        )
        self.tools_panel.layer_selected.connect(self._on_layer_selected)
        self.tools_panel.layer_visibility_changed.connect(self._on_layer_visibility_changed)
        self.tools_panel.layer_created.connect(self._on_layer_created)
        self.tools_panel.layer_duplicated.connect(self._on_layer_duplicated)
        self.tools_panel.layer_deleted.connect(self._on_layer_deleted)
        self.tools_panel.label_selected.connect(self.annotation_controller.on_label_selected)
        self.tools_panel.label_color_changed.connect(self._on_label_color_changed)
        self.tools_panel.label_selected.connect(
            self.corrosion_profile_controller.on_active_label_changed
        )
        self.tools_panel.label_selected.connect(
            self.mask_modification_controller.on_active_label_changed
        )
        self.corrosion_settings_view.label_a_changed.connect(self._on_corrosion_label_a_changed)
        self.corrosion_settings_view.peak_mode_a_changed.connect(
            self._on_corrosion_peak_selection_mode_a_changed
        )
        self.corrosion_settings_view.label_b_changed.connect(self._on_corrosion_label_b_changed)
        self.corrosion_settings_view.peak_mode_b_changed.connect(
            self._on_corrosion_peak_selection_mode_b_changed
        )
        self.corrosion_settings_view.interpolation_algo_changed.connect(
            self._on_corrosion_algo_changed
        )

        self.annotation_view.slice_changed.connect(self._on_slice_changed)
        self.annotation_view.mouse_clicked.connect(self._on_annotation_mouse_clicked)
        self.annotation_view.paint_stroke_started.connect(self.annotation_controller.on_annotation_paint_stroke_started)
        self.annotation_view.paint_stroke_moved.connect(self.annotation_controller.on_annotation_paint_stroke_moved)
        self.annotation_view.paint_stroke_finished.connect(self._on_annotation_paint_stroke_finished)
        self.annotation_view.freehand_started.connect(self.annotation_controller.on_annotation_freehand_started)
        self.annotation_view.freehand_point_added.connect(self.annotation_controller.on_annotation_freehand_point_added)
        self.annotation_view.freehand_completed.connect(self._on_annotation_freehand_completed)
        self.annotation_view.line_drawn.connect(self._on_annotation_line_drawn)
        self.annotation_view.box_drawn.connect(self._on_annotation_box_drawn)
        self.annotation_view.mod_drag_started.connect(self._on_mod_drag_started)
        self.annotation_view.mod_drag_moved.connect(self._on_mod_drag_moved)
        self.annotation_view.mod_drag_finished.connect(self._on_mod_drag_finished)
        self.annotation_view.mod_double_clicked.connect(self._on_mod_double_clicked)
        self.annotation_view.mod_context_requested.connect(self._on_mod_context_requested)
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
            self.annotation_view_corrosion.slice_changed.connect(self._on_slice_changed)
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
        self.overlay_settings_view.layer_selected.connect(self._on_layer_selected)
        self.overlay_settings_view.layer_visibility_changed.connect(
            self._on_layer_visibility_changed
        )
        self.overlay_settings_view.layer_created.connect(self._on_layer_created)
        self.overlay_settings_view.layer_duplicated.connect(self._on_layer_duplicated)
        self.overlay_settings_view.layer_deleted.connect(self._on_layer_deleted)
        self.overlay_settings_view.label_visibility_changed.connect(
            self.annotation_controller.on_label_visibility_changed
        )
        self.overlay_settings_view.label_color_changed.connect(
            self._on_label_color_changed
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
        self.nde_settings_view.overwrite_source_changed.connect(
            self._on_overwrite_source_changed
        )
        self.nde_settings_view.overwrite_target_changed.connect(
            self._on_overwrite_target_changed
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
        self.nde_settings_view.prune_label_a_changed.connect(self._on_prune_label_a_changed)
        self.nde_settings_view.prune_label_b_changed.connect(self._on_prune_label_b_changed)
        self.nde_settings_view.prune_peak_selection_mode_changed.connect(
            self._on_prune_peak_selection_mode_changed
        )
        self.nde_settings_view.closing_mask_tolerance_changed.connect(
            self._on_closing_mask_tolerance_changed
        )
        self.nde_settings_view.closing_mask_merge_distance_changed.connect(
            self._on_closing_mask_merge_distance_changed
        )
        self.nde_settings_view.clean_outliers_tolerance_changed.connect(
            self._on_clean_outliers_tolerance_changed
        )
        self.nde_settings_view.clean_outliers_thin_line_width_changed.connect(
            self._on_clean_outliers_thin_line_width_changed
        )
        self.nde_settings_view.clean_outliers_thin_gap_width_changed.connect(
            self._on_clean_outliers_thin_gap_width_changed
        )
        self.nde_settings_view.clean_outliers_contour_smoothing_changed.connect(
            self._on_clean_outliers_contour_smoothing_changed
        )
        self.nnunet_settings_view.model_path_changed.connect(
            self._on_nnunet_model_path_changed
        )
        self.nnunet_settings_view.choose_zip_requested.connect(
            self._on_choose_nnunet_model_zip_requested
        )
        self.nnunet_settings_view.choose_directory_requested.connect(
            self._on_choose_nnunet_model_directory_requested
        )

    def _register_shortcuts(self) -> None:
        """Global keyboard shortcuts (active anywhere in the window)."""
        parent = self.main_window
        mapping = [
            (QKeySequence(QKeySequence.StandardKey.Save), self._on_save_session),
            (QKeySequence(Qt.Key.Key_A), self._on_previous_slice),
            (QKeySequence(Qt.Key.Key_D), self._on_next_slice),
            (QKeySequence(Qt.Key.Key_W), self._apply_roi_non_corrosion),
            (QKeySequence(Qt.Key.Key_Delete), self._on_mod_delete_requested),
            (QKeySequence(Qt.Key.Key_R), self._select_draw_action),
            (QKeySequence(Qt.Key.Key_E), self._select_erase_action),
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
            "Open .nde file",
            "",
            "NDE Files (*.nde);;All Files (*)",
        )
        if not file_path:
            return
        if not self._confirm_unsaved_sessions_before_reset("open another NDE file"):
            return

        try:
            if not self._load_nde_file(file_path, prompt_open_options=True):
                return
        except Exception as exc:
            QMessageBox.critical(self.main_window, "NDE error", str(exc))

    def _on_open_session(self) -> None:
        """Open a persisted `.session` file and restore the contained active session."""
        self.session_workspace_controller.open_session_via_dialog()

    def _on_load_npz(self) -> None:
        """Handle loading an NPZ overlay."""
        if self.nde_model is None:
            QMessageBox.warning(self.main_window, "Overlay", "Load an NDE before using the overlay.")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Load overlay (.npz/.npy)",
            "",
            "Overlay Files (*.npz *.npy);;All Files (*)",
        )
        if not file_path:
            return
        try:
            self._load_overlay_from_file(
                file_path,
                preserve_labels=True,
                force_visible=False,
            )
            self.status_message(f"Overlay loaded: {file_path}")
        except Exception as exc:
            QMessageBox.critical(self.main_window, "Overlay error", str(exc))

    def _load_overlay_from_file(
        self,
        file_path: str | Path,
        *,
        preserve_labels: bool,
        force_visible: bool,
    ) -> np.ndarray:
        """Load an overlay file and apply it through the standard MVC refresh pipeline."""
        volume = self._current_volume()
        if volume is None:
            raise ValueError("NDE volume unavailable.")

        overlay_path = Path(file_path)
        axis_order = list((getattr(self.nde_model, "metadata", {}) or {}).get("axis_order") or [])
        primary_axis_name = str(axis_order[0]) if axis_order else None
        mask_volume = self.overlay_loader.load(
            str(overlay_path),
            target_shape=volume.shape,
            preferred_primary_axis=primary_axis_name,
        )
        self._apply_overlay_mask_volume(mask_volume, preserve_labels=preserve_labels)
        self.imported_overlay_model.set_imported_overlay(
            source_path=str(overlay_path),
            original_mask_volume=mask_volume,
            original_label_ids=self.annotation_model.get_detected_label_ids(),
        )
        if force_visible:
            self._on_overlay_toggled(True)
        return mask_volume

    def _handle_nnunet_success(self, payload: Any) -> None:
        """Load the saved nnUNet NPZ through the standard overlay-import path."""
        if not isinstance(payload, NnUnetResult):
            self._handle_nnunet_error(RuntimeError("Invalid nnUNet result."))
            return
        try:
            self._load_overlay_from_file(
                payload.output_path,
                preserve_labels=True,
                force_visible=True,
            )
            self._rename_active_layer("AI result")
            self.status_message(
                f"nnUNet completed, NPZ displayed: {payload.output_path}",
                timeout_ms=5000,
            )
            QMessageBox.information(
                self.main_window,
                "nnUNet",
                f"Result saved and displayed:\n{payload.output_path}",
            )
        except Exception as exc:
            self._handle_nnunet_error(exc)

    def _handle_nnunet_error(self, payload: Any) -> None:
        """Display nnUNet errors once they have been marshalled back to the UI thread."""
        exc = payload if isinstance(payload, Exception) else RuntimeError(str(payload))
        QMessageBox.critical(self.main_window, "nnUNet", str(exc))
        self.status_message("nnUNet inference failed", timeout_ms=5000)

    def _on_remap_classes(self) -> None:
        """Open the in-memory class remap dialog for the last imported NPZ overlay."""
        if self.nde_model is None:
            QMessageBox.warning(self.main_window, "Remap classes", "Load an NDE before remapping an overlay.")
            return
        if not self.imported_overlay_model.has_source():
            QMessageBox.warning(
                self.main_window,
                "Remap classes",
                "Load a .npz/.npy file first from the Overlay menu.",
            )
            return

        current_mask = self.annotation_model.get_mask_volume()
        if not self.imported_overlay_model.current_mask_matches(current_mask):
            answer = QMessageBox.question(
                self.main_window,
                "Remap classes",
                (
                    "The current overlay has been modified since the NPZ import.\n\n"
                    "The remap will restart from the original imported NPZ and replace the current mask.\n"
                    "Continue?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        dialog = OverlayClassRemapDialog(
            source_path=self.imported_overlay_model.source_path or "",
            source_classes=self.imported_overlay_model.original_label_ids,
            current_mapping=self.imported_overlay_model.current_mapping,
            parent=self.main_window,
        )
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return

        try:
            original_mask = self.imported_overlay_model.original_mask_volume
            if original_mask is None:
                raise ValueError("Overlay source not found for remap.")
            mapping = self.overlay_class_remap_service.normalize_mapping(
                dialog.get_mapping(),
                allowed_sources=self.imported_overlay_model.original_label_ids,
            )
            remapped_volume = self.overlay_class_remap_service.remap_mask_volume(original_mask, mapping)
            self._apply_overlay_mask_volume(remapped_volume, preserve_labels=False)
            self.imported_overlay_model.update_after_remap(
                mapping=mapping,
                applied_mask_volume=remapped_volume,
            )
            final_classes = self.overlay_class_remap_service.extract_classes(remapped_volume)
            self.status_message(
                "Remap classes applied: "
                f"{self.imported_overlay_model.source_path} | final classes: {list(final_classes)}",
                timeout_ms=5000,
            )
        except Exception as exc:
            QMessageBox.critical(self.main_window, "Remap classes", str(exc))

    def _on_run_nnunet(self) -> None:
        """Lance l'inférence nnUNet sur le volume NDE chargé."""
        if self.nde_model is None:
            QMessageBox.warning(self.main_window, "nnUNet", "Load an NDE before starting inference.")
            return

        volume = self._current_volume()
        if volume is None:
            QMessageBox.warning(self.main_window, "nnUNet", "NDE volume unavailable.")
            return

        model_path = self.view_state_model.set_nnunet_model_path(
            getattr(self.view_state_model, "nnunet_model_path", "")
        )
        if not model_path:
            QMessageBox.warning(
                self.main_window,
                "nnUNet",
                "Configure a model first from Inference > Settings.",
            )
            return
        if not Path(model_path).exists():
            QMessageBox.warning(
                self.main_window,
                "nnUNet",
                "The configured nnUNet model was not found. Update it from Inference > Settings.",
            )
            return

        raw_volume = getattr(self.nde_model, "volume", None)
        dataset_id = self.nde_model.metadata.get("path") if self.nde_model.metadata else "current"
        save_path = self._suggest_nnunet_output_path(model_path)

        def _on_success(result: NnUnetResult) -> None:
            self._nnunet_ui_signals.success.emit(result)

        def _on_error(exc: Exception) -> None:
            self._nnunet_ui_signals.error.emit(exc)

        try:
            self.status_message("nnUNet inference in progress...", timeout_ms=4000)
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

    def _suggest_nnunet_output_path(self, model_path: str | Path) -> str:
        """Build the default nnUNet NPZ path from the current NDE and selected model."""
        model_file = Path(model_path)
        try:
            nde_file = Path(self._require_nde_path())
        except Exception:
            nde_file = Path.cwd() / "current.nde"

        nde_stem = self._sanitize_output_stem(nde_file.stem, fallback="nde")
        model_name = model_file.stem if model_file.suffix else model_file.name
        model_stem = self._sanitize_output_stem(model_name, fallback="model")
        output_path = nde_file.with_name(f"{nde_stem}_{model_stem}.npz")
        return str(self._next_available_output_path(output_path))

    @staticmethod
    def _sanitize_output_stem(value: str, *, fallback: str) -> str:
        """Sanitize a filename stem for a Windows-friendly NPZ output path."""
        return sanitize_filename_component(value, fallback=fallback)

    @staticmethod
    def _next_available_output_path(path: Path) -> Path:
        """Return a non-conflicting NPZ path by appending an increment when needed."""
        candidate = Path(path)
        if not candidate.exists():
            return candidate
        index = 2
        while True:
            suffixed = candidate.with_name(f"{candidate.stem}_{index}{candidate.suffix}")
            if not suffixed.exists():
                return suffixed
            index += 1

    def _on_export_overlay_npz(self) -> None:
        """Export the current overlay to a standalone NPZ file."""
        volume = self._current_volume()
        if volume is None:
            QMessageBox.warning(self.main_window, "Overlay", "Load an NDE before saving.")
            return
        axis_order = list((getattr(self.nde_model, "metadata", {}) or {}).get("axis_order") or [])
        primary_axis_name = str(axis_order[0]) if axis_order else None
        try:
            saved_path = self.annotation_controller.save_overlay_via_dialog(
                parent=self.main_window,
                volume_shape=volume.shape,
                primary_axis_name=primary_axis_name,
            )
            if not saved_path:
                return
            self.status_message(f"Overlay saved: {saved_path}")
        except Exception as exc:
            QMessageBox.critical(self.main_window, "Overlay save error", str(exc))

    def _on_save_session(self) -> None:
        """Persist the active annotation session to its bound `.session` file."""
        self.session_workspace_controller.save_active_session_via_ui(force_dialog=False)

    def _on_save_session_as(self) -> None:
        """Persist the active annotation session to a user-chosen `.session` file."""
        self.session_workspace_controller.save_active_session_via_ui(force_dialog=True)

    def _on_export_endviews(self) -> None:
        """Export endviews (RGB + UINT8) via le service dédié."""
        if self.nde_model is None:
            QMessageBox.warning(self.main_window, "Export endviews", "Load an NDE before exporting endviews.")
            return

        base_dir = QFileDialog.getExistingDirectory(
            self.main_window,
            "Choose output folder (used as the endview root)",
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
        self.status_message("Endviews export completed", timeout_ms=4000)
        QMessageBox.information(self.main_window, "Export endviews", final_message)

    def _on_export_all(self) -> None:
        """Export the full active-session bundle from one dialog."""
        if self.nde_model is None:
            QMessageBox.warning(self.main_window, "Export all", "Load an NDE before exporting.")
            return
        if not self._prepare_active_session_for_persistence():
            return

        self.session_manager.sync_active_layer_from_model(
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
        )
        layer_stack = self.session_manager.get_active_layer_stack()
        if layer_stack is None or not layer_stack.layers:
            QMessageBox.warning(
                self.main_window,
                "Export all",
                "No layer is available in the active session.",
            )
            return

        active_layer_id = str(layer_stack.active_layer_id or "")
        main_layer_choices: list[tuple[str, str]] = []
        default_main_layer_id = ""

        for layer in layer_stack.layers:
            if layer.mask_volume is None:
                continue
            if str(getattr(layer, "layer_kind", "")).strip().casefold() == "corrosion":
                continue
            label = str(layer.name or "Layer")
            if str(layer.id) == active_layer_id:
                label = f"{label} (active)"
                default_main_layer_id = str(layer.id)
            elif not default_main_layer_id:
                default_main_layer_id = str(layer.id)
            main_layer_choices.append((str(layer.id), label))

        if not main_layer_choices:
            for layer in layer_stack.layers:
                if layer.mask_volume is None:
                    continue
                label = str(layer.name or "Layer")
                if str(layer.id) == active_layer_id:
                    label = f"{label} (active)"
                    default_main_layer_id = str(layer.id)
                elif not default_main_layer_id:
                    default_main_layer_id = str(layer.id)
                main_layer_choices.append((str(layer.id), label))

        if not main_layer_choices:
            QMessageBox.warning(
                self.main_window,
                "Export all",
                "No layer with a mask volume is available in the active session.",
            )
            return

        has_corrosion_layers = any(
            str(getattr(layer, "layer_kind", "")).strip().casefold() == "corrosion"
            and ViewStateModel.normalize_corrosion_session_stage(
                getattr(getattr(layer, "corrosion_state", None), "stage", None)
            )
            in {CORROSION_STAGE_RAW, CORROSION_STAGE_INTERPOLATED}
            for layer in layer_stack.layers
        )

        axis_order = list((getattr(self.nde_model, "metadata", {}) or {}).get("axis_order") or [])
        primary_axis_name = str(axis_order[0]) if axis_order else None
        dialog = SessionBundleExportDialog(
            self.main_window,
            layer_choices=main_layer_choices,
            default_main_layer_id=default_main_layer_id,
            default_sentinel_source_view=self.overlay_export.suggested_sentinel_source_view(
                primary_axis_name
            ),
            has_corrosion_layers=has_corrosion_layers,
        )
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        options = dialog.get_options()

        nde_path = None
        try:
            nde_path = (self.nde_model.metadata or {}).get("path")
        except Exception:
            nde_path = None
        initial_dir = str(Path(nde_path).parent) if nde_path else ""
        output_root = QFileDialog.getExistingDirectory(
            self.main_window,
            "Choose the export root folder",
            initial_dir,
        )
        if not output_root:
            return

        request = SessionBundleExportRequest(
            output_root=output_root,
            primary_axis_name=primary_axis_name,
            main_layer_id=options.main_layer_id,
            export_session_npz=options.export_session_npz,
            export_sentinel_npz=options.export_sentinel_npz,
            export_nnunet_pngs=options.export_nnunet_pngs,
            export_corrosion_cscan=options.export_corrosion_cscan,
            sentinel=BundleSentinelExportOptions(
                sentinel_source_view=options.sentinel_source_view,
                rotation_degrees=options.rotation_degrees,
                rotation_axes=options.rotation_axes,
                transpose_axes=options.transpose_axes,
                output_suffix=options.output_suffix,
                mirror_horizontal=options.mirror_horizontal,
                mirror_vertical=options.mirror_vertical,
                mirror_z=options.mirror_z,
                strict_mode=options.strict_mode,
            ),
        )

        try:
            self.status_message("Bundle export in progress...", timeout_ms=2000)
            result = self.session_bundle_export_service.export_bundle(
                session_manager=self.session_manager,
                nde_model=self.nde_model,
                nde_file=str(nde_path) if nde_path else None,
                request=request,
                signal_processing_options=self._current_signal_processing_options(),
            )
        except ValueError as exc:
            QMessageBox.warning(self.main_window, "Export all", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self.main_window, "Export all", str(exc))
            return

        self.status_message("Bundle export completed", timeout_ms=5000)
        QMessageBox.information(self.main_window, "Export all", result.build_summary())

    def _on_export_corrosion_cscan(self) -> None:
        """Open the native Windows folder picker and export the current corrosion C-scan."""
        if self.nde_model is None:
            QMessageBox.warning(
                self.main_window,
                "Export C-scan corrosion",
                "Load an NDE before exporting the corrosion C-scan.",
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
            "Choose corrosion C-scan export folder",
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
        self.status_message(f"Corrosion C-scan saved: {saved_path}", timeout_ms=5000)

    def _on_split_flaw_noflaw(self) -> None:
        """Lance le split flaw/noflaw (export + tri) via le service dédié."""
        if self.nde_model is None:
            QMessageBox.warning(self.main_window, "Split flaw/noflaw", "Load an NDE before starting the split.")
            return

        nde_path = None
        try:
            nde_path = (self.nde_model.metadata or {}).get("path")
        except Exception:
            nde_path = None

        initial_dir = str(Path(nde_path).parent) if nde_path else ""
        output_root = QFileDialog.getExistingDirectory(
            self.main_window,
            "Choose the parent folder for split endview export",
            initial_dir,
        )
        if not output_root:
            return

        prefix, ok = QInputDialog.getText(
            self.main_window,
            "Split flaw/noflaw",
            "Exported image prefix (optional):",
        )
        if not ok:
            return
        suffix, ok = QInputDialog.getText(
            self.main_window,
            "Split flaw/noflaw",
            "Exported image suffix (optional):",
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

        self.status_message("flaw/noflaw split in progress...", timeout_ms=2000)
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
            self.status_message("flaw/noflaw split completed", timeout_ms=5000)
            QMessageBox.information(self.main_window, "Split flaw/noflaw", message)
        else:
            QMessageBox.critical(self.main_window, "Split flaw/noflaw", message)

    def _on_dataset_export_requested(self) -> None:
        """Launch the dataset export flow selected by the user."""
        if self.nde_model is None:
            QMessageBox.warning(self.main_window, "Split flaw/noflaw", "Load an NDE before starting the split.")
            return

        export_mode = self._prompt_split_export_mode()
        if export_mode is None:
            return

        nde_path = None
        try:
            nde_path = (self.nde_model.metadata or {}).get("path")
        except Exception:
            nde_path = None

        initial_dir = str(Path(nde_path).parent) if nde_path else ""
        dialog_title = "Export nnU-Net" if export_mode == "nnunet" else "Split flaw/noflaw"
        output_prompt = (
            "Choose the parent folder for the nnU-Net dataset"
            if export_mode == "nnunet"
            else "Choose the parent folder for split endview export"
        )
        output_root = QFileDialog.getExistingDirectory(
            self.main_window,
            output_prompt,
            initial_dir,
        )
        if not output_root:
            return

        affixes = self._prompt_export_filename_affixes(dialog_title=dialog_title)
        if affixes is None:
            return
        prefix, suffix = affixes
        processing_options = self._current_signal_processing_options()

        if export_mode == "nnunet":
            self.status_message("nnU-Net export in progress...", timeout_ms=2000)
            success, message = self.split_flaw_noflaw_service.export_nnunet_dataset(
                nde_model=self.nde_model,
                annotation_model=self.annotation_model,
                nde_file=nde_path,
                output_root=output_root,
                filename_prefix=prefix,
                filename_suffix=suffix,
                signal_processing_options=processing_options,
            )
            if success:
                self.status_message("nnU-Net export completed", timeout_ms=5000)
                QMessageBox.information(self.main_window, dialog_title, message)
            else:
                QMessageBox.critical(self.main_window, dialog_title, message)
            return

        self.status_message("flaw/noflaw split in progress...", timeout_ms=2000)
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
            self.status_message("flaw/noflaw split completed", timeout_ms=5000)
            QMessageBox.information(self.main_window, dialog_title, message)
        else:
            QMessageBox.critical(self.main_window, dialog_title, message)

    def _prompt_split_export_mode(self) -> Optional[str]:
        """Ask whether the dataset export should target flaw/noflaw or nnU-Net."""
        choices = [
            "flaw/noflaw",
            "nnU-Net (imagesTr + labelsTr)",
        ]
        selection, ok = QInputDialog.getItem(
            self.main_window,
            "Export type",
            "Dataset to generate:",
            choices,
            0,
            False,
        )
        if not ok:
            return None
        return "nnunet" if str(selection).startswith("nnU-Net") else "flaw_noflaw"

    def _prompt_export_filename_affixes(self, *, dialog_title: str) -> Optional[tuple[str, str]]:
        """Collect optional filename prefix/suffix from the user."""
        prefix, ok = QInputDialog.getText(
            self.main_window,
            dialog_title,
            "Exported image prefix (optional):",
        )
        if not ok:
            return None

        suffix, ok = QInputDialog.getText(
            self.main_window,
            dialog_title,
            "Exported image suffix (optional):",
        )
        if not ok:
            return None

        return (prefix or "").strip(), (suffix or "").strip()

    def _current_signal_processing_options(self) -> NdeSignalProcessingOptions:
        """Return the currently selected signal-processing options."""
        selection = self._current_signal_processing_selection()
        return NdeSignalProcessingOptions(
            apply_hilbert=bool(selection.get("apply_hilbert", False)),
            apply_smoothing=bool(selection.get("apply_smoothing", False)),
        )

    def _on_open_settings(self) -> None:
        """Open the settings dialog."""
        self.nde_settings_view.set_colormaps(
            endview=self.view_state_model.endview_colormap,
            cscan=self.view_state_model.cscan_colormap,
        )
        self._sync_apply_volume_range_view()
        self._sync_overwrite_rule_editor(
            preferred_source=self._default_overwrite_source_label()
        )
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
        self._sync_prune_label_choices()
        self.nde_settings_view.set_prune_peak_selection_mode(
            getattr(self.view_state_model, "prune_peak_selection_mode", "max_peak")
        )
        self.nde_settings_view.set_closing_mask_tolerance(
            getattr(self.view_state_model, "closing_mask_tolerance", 0)
        )
        self.nde_settings_view.set_closing_mask_merge_distance(
            getattr(self.view_state_model, "closing_mask_merge_distance", 0)
        )
        self.nde_settings_view.set_clean_outliers_tolerance(
            getattr(self.view_state_model, "clean_outliers_tolerance", 0)
        )
        self.nde_settings_view.set_clean_outliers_thin_line_max_width(
            getattr(self.view_state_model, "clean_outliers_thin_line_max_width", 0)
        )
        self.nde_settings_view.set_clean_outliers_thin_gap_max_width(
            getattr(self.view_state_model, "clean_outliers_thin_gap_max_width", 0)
        )
        self.nde_settings_view.set_clean_outliers_contour_smoothing(
            getattr(self.view_state_model, "clean_outliers_contour_smoothing", 0)
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

    def _on_open_nnunet_settings(self) -> None:
        """Open the nnUNet settings dialog."""
        self.nnunet_settings_view.set_model_path(
            getattr(self.view_state_model, "nnunet_model_path", "")
        )
        self.nnunet_settings_view.show()
        self.nnunet_settings_view.raise_()
        self.nnunet_settings_view.activateWindow()

    def _on_nnunet_model_path_changed(self, value: str) -> None:
        """Persist the configured nnUNet model path in the view-state model."""
        normalized = self.view_state_model.set_nnunet_model_path(value)
        if normalized != value:
            self.nnunet_settings_view.set_model_path(normalized)

    def _on_choose_nnunet_model_zip_requested(self) -> None:
        """Open a file dialog to configure an exported nnUNet model archive."""
        selected_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Choose nnUNet model zip",
            self._nnunet_model_dialog_start_path(),
            "nnUNet models (*.zip);;All files (*)",
        )
        if not selected_path:
            return
        normalized = self.view_state_model.set_nnunet_model_path(selected_path)
        self.nnunet_settings_view.set_model_path(normalized)

    def _on_choose_nnunet_model_directory_requested(self) -> None:
        """Open a directory dialog to configure an extracted nnUNet model folder."""
        selected_path = QFileDialog.getExistingDirectory(
            self.main_window,
            "Choose nnUNet model folder",
            self._nnunet_model_dialog_start_path(),
        )
        if not selected_path:
            return
        normalized = self.view_state_model.set_nnunet_model_path(selected_path)
        self.nnunet_settings_view.set_model_path(normalized)

    def _nnunet_model_dialog_start_path(self) -> str:
        """Return the best-effort starting path for nnUNet model dialogs."""
        configured_path = str(getattr(self.view_state_model, "nnunet_model_path", "") or "").strip()
        if configured_path:
            path = Path(configured_path).expanduser()
            if path.exists():
                return str(path if path.is_dir() else path.parent)
            fallback_parent = path.parent if path.suffix else path
            fallback_text = str(fallback_parent).strip()
            if fallback_text:
                return fallback_text
        try:
            return str(Path(self._require_nde_path()).parent)
        except Exception:
            return str(Path.cwd())

    def _on_corrosion_interpolation_requested_from_menu(self) -> None:
        algo = getattr(self.view_state_model, "corrosion_interpolation_algo", "1d_dual_axis")
        self._on_corrosion_interpolation_requested(algo)

    def _on_run_corrosion_analysis(self) -> None:
        """Persist the active layer before launching corrosion analysis."""
        if not self.view_state_model.can_run_corrosion_analysis():
            self.status_message(
                "Analyze est disponible uniquement depuis un layer standard.",
                3000,
            )
            self._sync_corrosion_workflow_controls()
            return

        self.corrosion_profile_edit_service.reset()
        self.mask_modification_controller.reset(restore_overlay=True)
        self.session_manager.sync_active_layer_from_model(
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
        )
        self.cscan_controller.run_corrosion_analysis()
        self._sync_cscan_labels()

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
        self.ascan_controller.set_overlay_opacity(alpha)
        self.overlay_settings_view.set_overlay_opacity(alpha)
        self.tools_panel.set_overlay_opacity(alpha)

    def _on_label_color_changed(self, label_id: int, color: Any) -> None:
        """Keep label colors synchronized across both label editors and render views."""
        qcolor = QColor(color)
        if not qcolor.isValid():
            return
        label = int(label_id)
        self.annotation_controller.on_label_color_changed(label, qcolor)
        visible = self.annotation_model.label_visibility.get(label, True)
        self.overlay_settings_view.ensure_label(label, qcolor, visible=visible)
        self.tools_panel.set_label_color(label, qcolor)

    def _on_nde_opacity_changed(self, opacity: float) -> None:
        """Keep NDE opacity synchronized across the tools panel and render views."""
        self.view_state_model.set_nde_alpha(opacity)
        alpha = float(self.view_state_model.nde_alpha)
        self.endview_controller.set_nde_opacity(alpha)
        self.volume_view.set_nde_opacity(alpha)
        self.tools_panel.set_nde_opacity(alpha)
        self.tools_panel.set_nde_opacity_available(self._current_volume() is not None)

    def _on_nde_contrast_changed(self, contrast: float) -> None:
        """Keep NDE contrast synchronized across the tools panel and render views."""
        self.view_state_model.set_nde_contrast(contrast)
        factor = float(self.view_state_model.nde_contrast)
        self.endview_controller.set_nde_contrast(factor)
        self.volume_view.set_nde_contrast(factor)
        self.tools_panel.set_nde_contrast(factor)
        self.tools_panel.set_nde_opacity_available(self._current_volume() is not None)

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

    def _on_corrosion_algo_changed(self, algo: str) -> None:
        normalized = self.view_state_model.set_corrosion_interpolation_algo(algo)
        self.corrosion_settings_view.set_interpolation_algo(normalized)

    def _on_corrosion_peak_selection_mode_a_changed(self, mode: str) -> None:
        normalized = self.view_state_model.set_corrosion_peak_selection_mode_a(mode)
        self.corrosion_settings_view.set_peak_selection_modes(
            current_a=normalized,
            current_b=self.view_state_model.get_corrosion_peak_selection_mode_b(),
        )

    def _on_corrosion_peak_selection_mode_b_changed(self, mode: str) -> None:
        normalized = self.view_state_model.set_corrosion_peak_selection_mode_b(mode)
        self.corrosion_settings_view.set_peak_selection_modes(
            current_a=self.view_state_model.get_corrosion_peak_selection_mode_a(),
            current_b=normalized,
        )

    def _on_corrosion_session_changed(self) -> None:
        self.session_manager.sync_active_layer_from_model(
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
        )
        self._mark_active_session_dirty()
        self._restore_piece3d_state_from_view_state(sync_action=False)

    def _on_apply_volume_range_changed(self, start: int, end: int) -> None:
        """Handle apply-to-volume range updates from settings."""
        volume = self._current_volume()
        if volume is None:
            return
        start_idx, end_idx = self.view_state_model.set_apply_volume_range(
            start, end, include_current=False
        )
        self.nde_settings_view.set_apply_volume_range(start_idx, end_idx)

    def _apply_annotation_action(self, action: str) -> None:
        """Synchronize the draw/erase action between the tools panel and the view state."""
        normalized = self.view_state_model.set_annotation_action(action)
        self.tools_panel.set_annotation_action(normalized)
        self._sync_tools_labels(select_label_id=self.view_state_model.active_label)

    def _select_draw_action(self) -> None:
        self._apply_annotation_action("draw")

    def _select_erase_action(self) -> None:
        self._apply_annotation_action("erase")

    def _on_overwrite_source_changed(self, label_id: Optional[int]) -> None:
        """Refresh overwrite-target choices when the source label changes."""
        self._sync_overwrite_target_choices(current_source=label_id)

    def _on_overwrite_target_changed(self, value: Any) -> None:
        """Persist the overwrite rule for the currently selected source label."""
        source_label = self.nde_settings_view.current_overwrite_source_label()
        if source_label is None:
            return
        mode = "default"
        target_label = None
        if isinstance(value, (tuple, list)) and len(value) == 2:
            mode = str(value[0])
            target_label = value[1]
        if mode == "all":
            self.view_state_model.set_label_overwrite_target(source_label, None)
        elif mode == "label":
            try:
                self.view_state_model.set_label_overwrite_target(
                    source_label,
                    int(target_label),
                )
            except Exception:
                self.view_state_model.clear_label_overwrite_target(source_label)
        else:
            self.view_state_model.clear_label_overwrite_target(source_label)
        self._sync_overwrite_target_choices(current_source=source_label)
        self.annotation_controller.on_roi_recompute_requested()

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

    def _on_prune_peak_selection_mode_changed(self, mode: str) -> None:
        """Handle prune peak-selection mode from annotation settings."""
        normalized = self.view_state_model.set_prune_peak_selection_mode(mode)
        self.nde_settings_view.set_prune_peak_selection_mode(normalized)

    def _on_prune_label_a_changed(self, value: Optional[int]) -> None:
        try:
            label_a = int(value) if value is not None else None
        except Exception:
            label_a = None
        self.view_state_model.set_prune_label_a(label_a)
        self._sync_prune_label_choices()

    def _on_prune_label_b_changed(self, value: Optional[int]) -> None:
        try:
            label_b = int(value) if value is not None else None
        except Exception:
            label_b = None
        self.view_state_model.set_prune_label_b(label_b)
        self._sync_prune_label_choices()

    def _on_closing_mask_toggled(self, enabled: bool) -> None:
        """Handle closing-mask toggle from the tools panel."""
        self.view_state_model.set_closing_mask_enabled(bool(enabled))
        self.tools_panel.set_closing_mask_checked(self.view_state_model.closing_mask_enabled)

    def _on_clean_outliers_toggled(self, enabled: bool) -> None:
        """Handle clean-outliers toggle from the tools panel."""
        self.view_state_model.set_clean_outliers_enabled(bool(enabled))
        self.tools_panel.set_clean_outliers_checked(
            self.view_state_model.clean_outliers_enabled
        )

    def _on_closing_mask_tolerance_changed(self, value: int) -> None:
        """Handle closing-mask hole tolerance from settings."""
        self.view_state_model.set_closing_mask_tolerance(int(value))
        self.nde_settings_view.set_closing_mask_tolerance(
            self.view_state_model.closing_mask_tolerance
        )

    def _on_closing_mask_merge_distance_changed(self, value: int) -> None:
        """Handle closing-mask merge distance from settings."""
        self.view_state_model.set_closing_mask_merge_distance(int(value))
        self.nde_settings_view.set_closing_mask_merge_distance(
            self.view_state_model.closing_mask_merge_distance
        )

    def _on_clean_outliers_tolerance_changed(self, value: int) -> None:
        """Handle clean-outliers component tolerance from settings."""
        self.view_state_model.set_clean_outliers_tolerance(int(value))
        self.nde_settings_view.set_clean_outliers_tolerance(
            self.view_state_model.clean_outliers_tolerance
        )

    def _on_clean_outliers_thin_line_width_changed(self, value: int) -> None:
        """Handle mask-cleanup thin-line width from settings."""
        self.view_state_model.set_clean_outliers_thin_line_max_width(int(value))
        self.nde_settings_view.set_clean_outliers_thin_line_max_width(
            self.view_state_model.clean_outliers_thin_line_max_width
        )

    def _on_clean_outliers_thin_gap_width_changed(self, value: int) -> None:
        """Handle mask-cleanup thin-gap width from settings."""
        self.view_state_model.set_clean_outliers_thin_gap_max_width(int(value))
        self.nde_settings_view.set_clean_outliers_thin_gap_max_width(
            self.view_state_model.clean_outliers_thin_gap_max_width
        )

    def _on_clean_outliers_contour_smoothing_changed(self, value: int) -> None:
        """Handle mask-cleanup contour smoothing from settings."""
        self.view_state_model.set_clean_outliers_contour_smoothing(int(value))
        self.nde_settings_view.set_clean_outliers_contour_smoothing(
            self.view_state_model.clean_outliers_contour_smoothing
        )

    def _apply_roi_non_corrosion(self) -> None:
        """Apply all temporary masks through the standard pipeline."""
        self.mask_modification_controller.commit_pending_edits()
        if self.annotation_controller.on_apply_temp_mask_requested():
            self._mark_active_session_dirty()

    def _on_selection_cancel_requested(self) -> None:
        """Cancel mod pending edits first, then fallback to ROI/temp cancel."""
        if self.corrosion_profile_controller.on_selection_cancel_requested():
            return
        if self.mask_modification_controller.on_selection_cancel_requested():
            return
        self.annotation_controller.on_selection_cancel_requested()

    def _on_annotation_undo_requested(self) -> None:
        """Undo the last committed annotation apply action."""
        if self.annotation_controller.on_undo_last_applied_annotation_requested():
            self._mark_active_session_dirty()
            self.status_message("Last applied annotation undone.", timeout_ms=1800)
            return
        self.status_message("No applied annotation to undo.", timeout_ms=1800)

    def _on_annotation_redo_requested(self) -> None:
        """Redo the last committed annotation undo action."""
        if self.annotation_controller.on_redo_last_applied_annotation_requested():
            self._mark_active_session_dirty()
            self.status_message("Last annotation reapplied.", timeout_ms=1800)
            return
        self.status_message("No annotation to reapply.", timeout_ms=1800)

    def _on_roi_delete_requested(self) -> None:
        """Delete ROI/temp previews and clear mod pending edits consistently."""
        self.mask_modification_controller.on_roi_delete_requested()
        self.annotation_controller.on_roi_delete_requested()

    def _on_mod_delete_requested(self) -> None:
        """Delete the selected mod component in normal mode, then auto-apply if enabled."""
        if self.corrosion_profile_controller.is_profile_mod_active():
            return
        preview_created = self.mask_modification_controller.on_delete_selected_component_requested()
        self._apply_annotation_preview_if_needed(preview_created)

    @staticmethod
    def _normalize_colormap_name(name: str) -> str:
        text = str(name).strip()
        lowered = text.casefold()
        if lowered == "omniscan":
            return "OmniScan"
        if lowered in {"gray", "gris"}:
            return "Gray"
        return text or "Gray"

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
            self._set_annotation_position_label(synced_point[0], synced_point[1])
        else:
            self._set_annotation_position_label(clamped, current_y)
        self.annotation_controller.refresh_secondary_roi_overlay()
        self._update_endview_label()

    def _on_cross_toggled(self, enabled: bool) -> None:
        """Handle crosshair visibility toggle."""
        self.view_state_model.set_show_cross(enabled)
        self._set_action_checked(getattr(self.ui, "actionToggle_cross", None), enabled)
        self.endview_controller.set_cross_visible(enabled)
        self.cscan_controller.set_cross_visible(enabled)
        self.ascan_controller.set_marker_visible(enabled)

    def _on_overlay_toggled(self, enabled: bool) -> None:
        """Handle overlay visibility toggle from menu or tools panel."""
        self.annotation_controller.on_overlay_toggled(enabled)
        self._set_action_checked(getattr(self.ui, "actionToggle_overlay", None), enabled)

    def _on_overlay_ascan_toggled(self, enabled: bool) -> None:
        """Handle A-scan overlay visibility toggle from the display menu."""
        self.view_state_model.set_show_overlay_ascan(enabled)
        self._set_action_checked(getattr(self.ui, "actionToggle_overlay_ascan", None), enabled)
        self._update_ascan_trace()

    def _on_outline_only_toggled(self, enabled: bool) -> None:
        """Handle outline-only overlay rendering toggle."""
        self.annotation_controller.set_outline_only(enabled)
        self._set_action_checked(getattr(self.ui, "actionToggle_outline_only", None), enabled)

    def _on_volume_planes_toggled(self, enabled: bool) -> None:
        """Handle the visibility of moving slicing planes in the 3D volume view."""
        self.view_state_model.set_show_volume_planes(enabled)
        self._set_action_checked(getattr(self.ui, "actionToggle_plans_volume", None), enabled)
        self.volume_view.set_volume_planes_visible(enabled)

    def _on_ruler_display_unit_toggled(self, enabled: bool) -> None:
        """Handle the global ruler display unit toggle."""
        unit = "mm" if enabled else "px"
        active_unit = self.set_ruler_display_unit(unit)
        self._set_action_checked(
            getattr(self.ui, "actionToggle_pixel_mm", None),
            active_unit == "mm",
        )
        self.status_message(f"Ruler display: {active_unit}", 2000)

    def _on_restriction_toggled(self, enabled: bool) -> None:
        """Handle the restriction-outline visibility toggle."""
        self.annotation_controller.set_restriction_visible(enabled)
        self._set_action_checked(getattr(self.ui, "actionToggle_restriction", None), enabled)

    def _on_endview_smooth_toggled(self, enabled: bool) -> None:
        """Handle visual smoothing on the endview base image."""
        self.view_state_model.set_show_endview_smooth(enabled)
        self._set_action_checked(getattr(self.ui, "actionToggle_Smooth", None), enabled)
        self.endview_controller.set_smooth_enabled(enabled)

    def _on_interpolated_profile_vectorized_toggled(self, enabled: bool) -> None:
        """Toggle vector rendering for visible interpolated corrosion profiles."""
        self.view_state_model.set_show_interpolated_profile_vectorized(enabled)
        self._set_action_checked(getattr(self.ui, "actionToggle_Vectorise", None), enabled)
        self.annotation_controller.refresh_overlay(defer_volume=True, rebuild=False)

    def _on_tool_mode_changed(self, mode: str) -> None:
        """Route the shared tool mode to the controller that owns the current context."""
        self.annotation_controller.on_tool_mode_changed(mode)
        if self.corrosion_profile_controller.is_profile_mod_active():
            self.mask_modification_controller.on_tool_mode_changed(mode)
            self.corrosion_profile_controller.on_tool_mode_changed(mode)
            return
        self.corrosion_profile_controller.on_tool_mode_changed(mode)
        self.mask_modification_controller.on_tool_mode_changed(mode)

    def _on_mod_drag_started(self, pos: Any) -> None:
        """Start either contour mod or raw corrosion profile mod."""
        if self.corrosion_profile_controller.is_profile_mod_active():
            self.corrosion_profile_controller.on_drag_started(pos)
            return
        self.mask_modification_controller.on_drag_started(pos)

    def _on_mod_drag_moved(self, pos: Any) -> None:
        """Move either contour mod or raw corrosion profile mod."""
        if self.corrosion_profile_controller.is_profile_mod_active():
            self.corrosion_profile_controller.on_drag_moved(pos)
            return
        self.mask_modification_controller.on_drag_moved(pos)

    def _on_mod_double_clicked(self, pos: Any) -> None:
        """Add an anchor to either the selected contour or selected corrosion line."""
        if self.corrosion_profile_controller.is_profile_mod_active():
            self.corrosion_profile_controller.on_double_clicked(pos)
            return
        self.mask_modification_controller.on_double_clicked(pos)

    def _on_mod_context_requested(self, payload: Any) -> None:
        """Open the contour-mod context menu and auto-apply the chosen action when enabled."""
        if self.corrosion_profile_controller.is_profile_mod_active():
            return
        preview_created = self.mask_modification_controller.on_context_menu_requested(payload)
        self._apply_annotation_preview_if_needed(preview_created)

    def _on_annotation_freehand_completed(self, points: Any) -> None:
        """Create the ROI preview, then optionally apply it immediately."""
        preview_created = self.annotation_controller.on_annotation_freehand_completed(points)
        self._apply_annotation_preview_if_needed(preview_created)

    def _on_annotation_mouse_clicked(self, pos: Any, button: Any) -> None:
        """Handle point-based annotation tools, then auto-apply when enabled."""
        preview_created = self.annotation_controller.on_annotation_mouse_clicked(pos, button)
        self._apply_annotation_preview_if_needed(preview_created)

    def _on_annotation_paint_stroke_finished(self, pos: Any) -> None:
        """Apply a completed paint stroke once when auto-apply is enabled."""
        preview_created = self.annotation_controller.on_annotation_paint_stroke_finished(pos)
        self._apply_annotation_preview_if_needed(preview_created)

    def _on_annotation_line_drawn(self, points: Any) -> None:
        """Handle line ROI completion, then auto-apply when enabled."""
        preview_created = self.annotation_controller.on_annotation_line_drawn(points)
        self._apply_annotation_preview_if_needed(preview_created)

    def _on_annotation_box_drawn(self, box: Any) -> None:
        """Handle box ROI completion, then auto-apply when enabled."""
        preview_created = self.annotation_controller.on_annotation_box_drawn(box)
        self._apply_annotation_preview_if_needed(preview_created)

    def _on_mod_drag_finished(self, pos: Any) -> None:
        """Finish mod drag and auto-apply its pending temp preview when requested."""
        if self.corrosion_profile_controller.is_profile_mod_active():
            self.corrosion_profile_controller.on_drag_finished(pos)
            return
        preview_created = self.mask_modification_controller.on_drag_finished(pos)
        self._apply_annotation_preview_if_needed(preview_created)

    def _apply_annotation_preview_if_needed(self, preview_created: bool) -> None:
        """Auto-commit the temp annotation preview when the toggle is enabled."""
        if not preview_created or not getattr(self.view_state_model, "apply_auto", False):
            return
        self._apply_roi_non_corrosion()

    def _on_endview_point_selected(self, pos: Any) -> None:
        """Handle point selection for crosshair sync."""
        point = self.endview_controller.on_point_selected(pos)
        if point is None:
            return
        x, y = point
        self._set_annotation_position_label(x, y)
        self._update_ascan_trace(point=(x, y))
        self._on_secondary_slice_changed(x)

    def _on_endview_drag_update(self, pos: Any) -> None:
        """Handle drag updates during drawing (cursor label only)."""
        point = self.endview_controller.on_drag_update(pos)
        if point is None:
            return
        x, y = point
        self._set_annotation_position_label(x, y)

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

    def _on_layer_selected(self, layer_id: str) -> None:
        """Switch the editable layer and rebuild UI state around it."""
        if not self.session_manager.switch_active_layer(
            layer_id,
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
        ):
            self.annotation_controller.sync_overlay_settings()
            return
        self._clear_active_layer_runtime_edits()
        self._after_layer_stack_changed(defer_volume=True)

    def _on_layer_visibility_changed(self, layer_id: str, visible: bool) -> None:
        """Toggle layer visibility and refresh the composed overlay."""
        if not self.session_manager.set_layer_visibility(layer_id, visible):
            self.annotation_controller.sync_overlay_settings()
            return
        self.annotation_controller.sync_overlay_settings()
        self.annotation_controller.refresh_overlay(defer_volume=True, rebuild=False)
        self._mark_active_session_dirty()

    def _on_layer_created(self) -> None:
        """Create a new empty layer above the current label list."""
        created_layer_id = self.session_manager.create_empty_layer(
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
        )
        if created_layer_id is None:
            self.annotation_controller.sync_overlay_settings()
            return
        self._clear_active_layer_runtime_edits()
        self._after_layer_stack_changed()

    def _on_layer_duplicated(self) -> None:
        """Duplicate the active layer and switch editing to the copy."""
        duplicated_layer_id = self.session_manager.duplicate_active_layer(
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
        )
        if duplicated_layer_id is None:
            self.annotation_controller.sync_overlay_settings()
            return
        self._clear_active_layer_runtime_edits()
        self._after_layer_stack_changed()

    def _on_layer_deleted(self, layer_id: str) -> None:
        """Delete one layer while keeping the session stack consistent."""
        if not self.session_manager.delete_layer(
            layer_id,
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
        ):
            self.annotation_controller.sync_overlay_settings()
            return
        self._clear_active_layer_runtime_edits()
        self._after_layer_stack_changed()

    def _clear_active_layer_runtime_edits(self) -> None:
        """Discard temporary previews and per-layer apply history after a layer switch."""
        self.annotation_controller.clear_apply_history()
        self.mask_modification_controller.reset()
        self.corrosion_profile_edit_service.reset()
        self.temp_mask_model.clear()
        if self.annotation_view is not None:
            self.annotation_view.clear_temp_shapes()
        self.annotation_controller.refresh_roi_overlay_for_slice(self.view_state_model.current_slice)

    def _after_layer_stack_changed(self, *, defer_volume: bool = False) -> None:
        """Resync overlay UI, rendered overlay and active-label selectors after a layer change."""
        self._after_layer_switch(defer_volume=defer_volume)
        self._mark_active_session_dirty()

    def _rename_active_layer(self, name: str) -> None:
        """Rename the active layer and refresh layer selectors without rebuilding the overlay."""
        active_layer = self.session_manager.get_active_layer()
        if active_layer is None:
            return
        if not self.session_manager.rename_layer(str(active_layer.id), name):
            return
        self.annotation_controller.sync_overlay_settings(force=True)
        self._sync_tools_labels(select_label_id=self.view_state_model.active_label)
        self._mark_active_session_dirty()

    def _sync_tools_labels(self, select_label_id: Optional[int] = None) -> None:
        """Sync the label list in the tools panel with the annotation model."""
        self.tools_panel.set_layers(self.session_manager.list_active_layers())
        self.annotation_model.ensure_persistent_labels()
        self.temp_mask_model.ensure_persistent_labels()
        palette = self.annotation_model.get_label_palette()
        entries = [
            (
                int(label_id),
                QColor(int(color[2]), int(color[1]), int(color[0]), int(color[3])),
            )
            for label_id, color in sorted(palette.items())
            if int(label_id) != 0
        ] if palette else []
        labels = [label_id for label_id, _color in entries]
        current = select_label_id if select_label_id is not None else self.view_state_model.active_label
        try:
            current = None if current is None else int(current)
        except Exception:
            current = None
        if current == 0:
            current = None
        if current not in labels:
            if int(DEFAULT_ACTIVE_LABEL_ID) in labels:
                current = int(DEFAULT_ACTIVE_LABEL_ID)
            else:
                current = next(
                    (label_id for label_id in PERSISTENT_LABEL_IDS if int(label_id) != 0 and label_id in labels),
                    (labels[0] if labels else None),
                )
        self.view_state_model.set_active_label(current)
        self.tools_panel.set_labels(entries, current=current)
        self.mask_modification_controller.on_active_label_changed(-1 if current is None else int(current))
        self._sync_overwrite_rule_editor()
        self._sync_prune_label_choices()
        self._sync_corrosion_label_choices()

    def _default_overwrite_source_label(self) -> Optional[int]:
        """Return the label that should be selected by default in overwrite settings."""
        current = self.view_state_model.effective_annotation_label()
        try:
            return None if current is None else int(current)
        except Exception:
            return None

    def _sync_overwrite_rule_editor(self, *, preferred_source: Optional[int] = None) -> None:
        """Sync overwrite source/target editors with the current label palette."""
        palette = self.annotation_model.get_label_palette()
        labels = sorted(lbl for lbl in palette.keys() if int(lbl) != 0) if palette else []
        source_labels = [0, *labels]
        target_labels = [0, *labels]
        self.view_state_model.prune_label_overwrite_targets(
            source_labels=source_labels,
            target_labels=target_labels,
        )
        current_source = preferred_source
        if current_source is None:
            current_source = self.nde_settings_view.current_overwrite_source_label()
        try:
            current_source = None if current_source is None else int(current_source)
        except Exception:
            current_source = None
        if current_source not in source_labels:
            current_source = self._default_overwrite_source_label()
        if current_source not in source_labels:
            current_source = source_labels[0] if source_labels else None
        self.nde_settings_view.set_overwrite_source_choices(
            source_labels,
            current=current_source,
        )
        self._sync_overwrite_target_choices(current_source=current_source, labels=target_labels)

    def _sync_overwrite_target_choices(
        self,
        *,
        current_source: Optional[int],
        labels: Optional[list[int]] = None,
    ) -> None:
        """Sync overwrite-target choices for the selected source label."""
        if labels is None:
            palette = self.annotation_model.get_label_palette()
            labels = [0, *sorted(lbl for lbl in palette.keys() if int(lbl) != 0)] if palette else [0]
        try:
            source_label = None if current_source is None else int(current_source)
        except Exception:
            source_label = None
        mode = "default"
        target = None
        if source_label is not None:
            has_rule, explicit_target = self.view_state_model.get_label_overwrite_target(
                source_label
            )
            if has_rule:
                if explicit_target is None:
                    mode = "all"
                elif explicit_target in labels:
                    mode = "label"
                    target = explicit_target
        self.nde_settings_view.set_overwrite_target_choices(
            labels,
            current_mode=mode,
            current_target=target,
        )

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
        self.corrosion_settings_view.set_peak_selection_modes(
            current_a=self.view_state_model.get_corrosion_peak_selection_mode_a(),
            current_b=self.view_state_model.get_corrosion_peak_selection_mode_b(),
        )
        self.corrosion_settings_view.set_interpolation_algo(
            getattr(self.view_state_model, "corrosion_interpolation_algo", "1d_dual_axis")
        )
        self.corrosion_settings_view.set_workflow_state(self._current_corrosion_session_stage())

    def _sync_prune_label_choices(self) -> None:
        """Sync prune companion-label choices with current labels and defaults."""
        labels = self._get_corrosion_labels()
        current_a = getattr(self.view_state_model, "prune_label_a", None)
        current_b = getattr(self.view_state_model, "prune_label_b", None)
        label_a, label_b = CorrosionLabelService.normalize_pair(
            labels,
            label_a=current_a,
            label_b=current_b,
        )
        self.view_state_model.set_prune_label_pair(label_a, label_b)
        self.nde_settings_view.set_prune_label_choices(
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
        """Sync menu toggle actions with the current display state."""
        self._set_action_checked(
            getattr(self.ui, "actionToggle_cross", None),
            self.view_state_model.show_cross,
        )
        self._set_action_checked(
            getattr(self.ui, "actionToggle_overlay", None),
            self.view_state_model.show_overlay,
        )
        self._set_action_checked(
            getattr(self.ui, "actionToggle_overlay_ascan", None),
            getattr(self.view_state_model, "show_overlay_ascan", True),
        )
        self._set_action_checked(
            getattr(self.ui, "actionToggle_outline_only", None),
            getattr(self.view_state_model, "show_outline_only", False),
        )
        self._set_action_checked(
            getattr(self.ui, "actionToggle_plans_volume", None),
            getattr(self.view_state_model, "show_volume_planes", True),
        )
        self._set_action_checked(
            getattr(self.ui, "actionToggle_pixel_mm", None),
            getattr(self.view_state_model, "ruler_display_unit", "px") == "mm",
        )
        self._set_action_checked(
            getattr(self.ui, "actionToggle_restriction", None),
            getattr(self.view_state_model, "show_restriction", True),
        )
        self._set_action_checked(
            getattr(self.ui, "actionToggle_Smooth", None),
            getattr(self.view_state_model, "show_endview_smooth", True),
        )
        self._set_action_checked(
            getattr(self.ui, "actionToggle_Vectorise", None),
            getattr(self.view_state_model, "show_interpolated_profile_vectorized", True),
        )

    def _apply_saved_colormaps(self) -> None:
        """Reapply persisted colormap names with their resolved LUTs."""
        endview_name = self._normalize_colormap_name(
            getattr(self.view_state_model, "endview_colormap", "Gray")
        )
        cscan_name = self._normalize_colormap_name(
            getattr(self.view_state_model, "cscan_colormap", "Gray")
        )
        endview_lut = self._get_colormap_lut(endview_name)
        cscan_lut = self._get_colormap_lut(cscan_name)
        self.view_state_model.set_endview_colormap(endview_name)
        self.view_state_model.set_cscan_colormap(cscan_name)
        self.endview_controller.set_colormap(endview_name, endview_lut)
        self.volume_view.set_base_colormap(endview_name, endview_lut)
        self.cscan_controller.set_colormap(cscan_name, cscan_lut)
        self.tools_panel.set_endview_colormap(endview_name)
        self.nde_settings_view.set_colormaps(endview=endview_name, cscan=cscan_name)

    def _sync_coordinate_view_labels(self) -> None:
        """Push view names and ruler axis labels into the endview-local controls."""
        self.endview_controller.set_navigation_view_names(
            primary=self._primary_view_name,
            secondary=self._secondary_view_name,
        )
        ruler_axes = self.annotation_axis_service.build_endview_ruler_axes(self.nde_model)
        self.endview_controller.set_ruler_axis_names(
            primary_horizontal=ruler_axes.primary_horizontal_axis_name,
            primary_vertical=ruler_axes.primary_vertical_axis_name,
            secondary_horizontal=ruler_axes.secondary_horizontal_axis_name,
            secondary_vertical=ruler_axes.secondary_vertical_axis_name,
        )
        cscan_ruler_axes = self.annotation_axis_service.build_cscan_ruler_axes(self.nde_model)
        self.cscan_controller.set_ruler_axis_names(
            horizontal=cscan_ruler_axes.horizontal_axis_name,
            vertical=cscan_ruler_axes.vertical_axis_name,
        )
        self.tools_panel.set_primary_view_name(self._primary_view_name)
        self.tools_panel.set_secondary_view_name(self._secondary_view_name)
        self._sync_ruler_display_settings(
            endview_ruler_axes=ruler_axes,
            cscan_ruler_axes=cscan_ruler_axes,
        )

    def _axis_resolution_mm(self, axis_name: Optional[str]) -> Optional[float]:
        """Return the per-NDE sampling step for a logical axis when available."""
        if self.nde_model is None:
            return None
        name = str(axis_name or "").strip()
        if not name:
            return None
        return self.nde_model.get_axis_resolution_mm(name)

    def _sync_ruler_display_settings(
        self,
        *,
        endview_ruler_axes=None,
        cscan_ruler_axes=None,
    ) -> None:
        """Push the global ruler unit and per-axis calibration to all ruler-enabled views."""
        display_unit = self.view_state_model.set_ruler_display_unit(
            getattr(self.view_state_model, "ruler_display_unit", "px")
        )
        self.endview_controller.set_ruler_display_unit(display_unit)
        self.cscan_controller.set_ruler_display_unit(display_unit)
        self.ascan_controller.set_ruler_display_unit(display_unit)

        if endview_ruler_axes is None:
            endview_ruler_axes = self.annotation_axis_service.build_endview_ruler_axes(self.nde_model)
        if cscan_ruler_axes is None:
            cscan_ruler_axes = self.annotation_axis_service.build_cscan_ruler_axes(self.nde_model)

        self.endview_controller.set_ruler_axis_resolutions_mm(
            primary_horizontal=self._axis_resolution_mm(endview_ruler_axes.primary_horizontal_axis_key),
            primary_vertical=self._axis_resolution_mm(endview_ruler_axes.primary_vertical_axis_key),
            secondary_horizontal=self._axis_resolution_mm(endview_ruler_axes.secondary_horizontal_axis_key),
            secondary_vertical=self._axis_resolution_mm(endview_ruler_axes.secondary_vertical_axis_key),
        )
        self.cscan_controller.set_ruler_axis_resolutions_mm(
            horizontal=self._axis_resolution_mm(cscan_ruler_axes.horizontal_axis_key),
            vertical=self._axis_resolution_mm(cscan_ruler_axes.vertical_axis_key),
        )

    def set_ruler_display_unit(self, unit: Optional[str]) -> str:
        """Switch all ruler-enabled views between pixel and millimeter display."""
        normalized = self.view_state_model.set_ruler_display_unit(unit)
        self._sync_ruler_display_settings()
        self._set_action_checked(
            getattr(self.ui, "actionToggle_pixel_mm", None),
            normalized == "mm",
        )
        return normalized

    def _sync_cscan_labels(self) -> None:
        """Keep the C-scan dock and toggle action aligned with corrosion state."""
        is_corrosion = bool(self.view_state_model.corrosion_active)
        dock_title = "Thickness CScan" if is_corrosion else "C-Scan"
        action_text = "Toggle Thickness CScan" if is_corrosion else "Toggle c-scan"
        self.dock_layout_controller.cscan_dock.setWindowTitle(dock_title)
        if hasattr(self.ui, "actionToggle_C_Scan"):
            self.ui.actionToggle_C_Scan.setText(action_text)

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
        annotatable_view_by_axis_mode = {
            "Auto": "Auto",
            "UCoordinate": "D-Scan",
            "VCoordinate": "B-Scan",
        }
        axis_choice_items = [
            (choice, annotatable_view_by_axis_mode.get(str(choice), str(choice)))
            for choice in choices
        ]

        dialog = NdeOpenOptionsDialog(
            axis_choices=axis_choice_items,
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
        annotatable_view_by_axis_mode = {
            "Auto": "Auto",
            "UCoordinate": "D-Scan",
            "VCoordinate": "B-Scan",
        }
        display_choices = [
            annotatable_view_by_axis_mode.get(str(choice), str(choice))
            for choice in choices
        ]
        current_display = annotatable_view_by_axis_mode.get(str(current), str(current))
        current_idx = display_choices.index(current_display)
        selection, ok = QInputDialog.getItem(
            self.main_window,
            "Plan d'annotation",
            "Plan autorise pour annotation:",
            display_choices,
            current_idx,
            False,
        )
        if not ok:
            return None
        resolved = None
        selection_text = str(selection)
        for choice in choices:
            if annotatable_view_by_axis_mode.get(str(choice), str(choice)) == selection_text:
                resolved = str(choice)
                break
        return resolved or current

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
        self.imported_overlay_model.clear()
        self.annotation_model.initialize(volume.shape)
        self.temp_mask_model.clear()
        self.temp_mask_model.initialize(volume.shape)
        self.roi_model.clear()
        self.annotation_controller.clear_restriction_rect()
        self.annotation_controller.sync_overlay_settings()
        self.cscan_controller.reset_corrosion()
        self._reset_piece3d_state(sync_action=True)
        self.mask_modification_controller.reset()

        self.session_manager.reset_for_new_dataset(
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
        )
        self._clear_session_runtime_state(remove_autosaves=True)
        self._refresh_session_dialog()

        axis_order = loaded_model.metadata.get("axis_order", [])
        positions = loaded_model.metadata.get("positions") or {}
        self.view_state_model.set_axis_order(axis_order)
        self._nde_path = file_path
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

        self._after_session_switch(rebuild_volume_view=True)
        self._update_main_window_title()
        self._sync_apply_volume_range_view()

        processing_label = self.nde_signal_processing_service.describe_selection(
            processing_options
        )
        self.status_message(f"NDE charge: {file_path} | {processing_label}")
        return True

    def _apply_overlay_mask_volume(self, mask_volume: np.ndarray, *, preserve_labels: bool) -> None:
        """Apply a full overlay mask volume through the standard MVC refresh pipeline."""
        volume = self._current_volume()
        if volume is None:
            raise ValueError("NDE volume unavailable.")

        self.annotation_controller.reset_overlay_state(preserve_labels=preserve_labels)
        if not preserve_labels:
            self.annotation_model.clear()
            self.temp_mask_model.label_palette = {}
            self.temp_mask_model.label_visibility = {}
        self.temp_mask_model.clear()
        self.temp_mask_model.initialize(volume.shape)
        self.roi_model.clear()
        self.annotation_model.set_mask_volume(mask_volume, preserve_labels=preserve_labels)
        self.session_manager.sync_active_layer_from_model(
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
        )
        self.cscan_controller.reset_corrosion()
        self._reset_piece3d_state(sync_action=True)
        self.mask_modification_controller.reset()
        self.annotation_controller.sync_overlay_settings()
        self.annotation_controller.refresh_overlay()
        self._sync_tools_labels()
        self._mark_active_session_dirty()

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
        self._reset_piece3d_state(sync_action=True)

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

    def _current_corrosion_session_stage(self) -> str:
        return self.view_state_model.get_corrosion_session_stage()

    def _ensure_corrosion_runtime_cache(self, *, include_piece: bool = False) -> None:
        """Rebuild missing corrosion caches from the active layer sources."""
        if not bool(getattr(self.view_state_model, "corrosion_active", False)):
            return
        stage = self.view_state_model.get_corrosion_session_stage()
        if stage not in {CORROSION_STAGE_RAW, CORROSION_STAGE_INTERPOLATED}:
            return

        has_projection = self.view_state_model.corrosion_projection is not None
        has_piece = self._workflow_result_has_piece3d_data(
            CorrosionWorkflowResult(
                ok=True,
                piece_volume_raw=getattr(self.view_state_model, "corrosion_piece_volume_raw", None),
                piece_volume_interpolated=getattr(
                    self.view_state_model,
                    "corrosion_piece_volume_interpolated",
                    None,
                ),
                piece_volume_legacy_raw=getattr(
                    self.view_state_model,
                    "corrosion_piece_volume_legacy_raw",
                    None,
                ),
                piece_volume_legacy_interpolated=getattr(
                    self.view_state_model,
                    "corrosion_piece_volume_legacy_interpolated",
                    None,
                ),
            )
        )
        wants_piece = bool(
            include_piece or getattr(self.view_state_model, "corrosion_piece_view_enabled", False)
        )
        if has_projection and (has_piece or not wants_piece):
            return

        peak_a = self.view_state_model.corrosion_peak_index_map_a
        peak_b = self.view_state_model.corrosion_peak_index_map_b
        if peak_a is None or peak_b is None:
            peak_a = self.view_state_model.corrosion_raw_peak_index_map_a
            peak_b = self.view_state_model.corrosion_raw_peak_index_map_b
        if peak_a is None or peak_b is None:
            return

        try:
            distance_map = self.cscan_corrosion_service.build_distance_map_from_peak_maps(
                peak_map_a=peak_a,
                peak_map_b=peak_b,
                use_mm=False,
                resolution_ultrasound_mm=1.0,
            )
            projection, value_range = self.cscan_corrosion_service.compute_corrosion_projection(
                distance_map
            )
        except Exception:
            return

        self.view_state_model.corrosion_projection = (projection, value_range)
        if stage == CORROSION_STAGE_INTERPOLATED:
            self.view_state_model.corrosion_interpolated_projection = (
                projection,
                value_range,
            )
        else:
            self.view_state_model.corrosion_interpolated_projection = None

        raw_peak_a = self.view_state_model.corrosion_raw_peak_index_map_a
        raw_peak_b = self.view_state_model.corrosion_raw_peak_index_map_b
        raw_distance_map = None
        if raw_peak_a is not None and raw_peak_b is not None:
            try:
                raw_distance_map = self.cscan_corrosion_service.build_distance_map_from_peak_maps(
                    peak_map_a=raw_peak_a,
                    peak_map_b=raw_peak_b,
                    use_mm=False,
                    resolution_ultrasound_mm=1.0,
                )
            except Exception:
                raw_distance_map = None
        if raw_distance_map is None and stage == CORROSION_STAGE_RAW:
            raw_distance_map = distance_map
        self.view_state_model.corrosion_raw_distance_map = raw_distance_map

        if wants_piece and not has_piece:
            overlay_volume = self.annotation_model.get_mask_volume()
            label_ids = self.view_state_model.corrosion_overlay_label_ids
            try:
                piece_volume_current = (
                    self.cscan_corrosion_service.build_piece_volume_from_distance_map(
                        distance_map
                    )
                )
                piece_volume_raw = (
                    self.cscan_corrosion_service.build_piece_volume_from_distance_map(
                        raw_distance_map
                    )
                    if raw_distance_map is not None
                    else None
                )
                legacy_current = None
                if (
                    overlay_volume is not None
                    and getattr(overlay_volume, "ndim", 0) == 3
                    and label_ids is not None
                ):
                    legacy_current = self.cscan_corrosion_service.build_legacy_piece_volume(
                        mask_stack=overlay_volume,
                        class_A_id=int(label_ids[0]),
                        class_B_id=int(label_ids[1]),
                    )
            except Exception:
                piece_volume_current = None
                piece_volume_raw = None
                legacy_current = None

            if stage == CORROSION_STAGE_RAW:
                self.view_state_model.corrosion_piece_volume_raw = piece_volume_raw
                self.view_state_model.corrosion_piece_volume_interpolated = None
                self.view_state_model.corrosion_piece_volume_legacy_raw = legacy_current
                self.view_state_model.corrosion_piece_volume_legacy_interpolated = None
            else:
                self.view_state_model.corrosion_piece_volume_raw = piece_volume_raw
                self.view_state_model.corrosion_piece_volume_interpolated = piece_volume_current
                self.view_state_model.corrosion_piece_volume_legacy_interpolated = legacy_current
            self.view_state_model.corrosion_piece_anchor = (
                self.cscan_corrosion_service.compute_piece_anchor(
                    self.view_state_model.corrosion_piece_volume_interpolated,
                    self.view_state_model.corrosion_piece_volume_raw,
                    self.view_state_model.corrosion_piece_volume_legacy_interpolated,
                    self.view_state_model.corrosion_piece_volume_legacy_raw,
                )
            )

        self.view_state_model.corrosion_overlay_volume = self.annotation_model.get_mask_volume()
        self.session_manager.sync_active_layer_from_model(
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
        )

    def _sync_corrosion_workflow_controls(self) -> None:
        stage = self._current_corrosion_session_stage()
        analyze_enabled = stage == CORROSION_STAGE_BASE
        interpolate_enabled = stage == CORROSION_STAGE_RAW

        analyze_tip = ""
        interpolate_tip = ""
        if stage == CORROSION_STAGE_RAW:
            analyze_tip = "The raw corrosion layer has already been analyzed."
        elif stage == CORROSION_STAGE_INTERPOLATED:
            analyze_tip = "The interpolated corrosion layer is finalized."
            interpolate_tip = "The interpolated corrosion layer cannot be interpolated again."
        else:
            interpolate_tip = "Run Analyze first to create a raw corrosion layer."

        analyze_action = getattr(self.ui, "actionCorrosion_analyse", None)
        if analyze_action is not None:
            analyze_action.setEnabled(analyze_enabled)
            analyze_action.setToolTip(analyze_tip)
            analyze_action.setStatusTip(analyze_tip)

        interpolate_action = getattr(self.ui, "actionInterpolate", None)
        if interpolate_action is not None:
            interpolate_action.setEnabled(interpolate_enabled)
            interpolate_action.setToolTip(interpolate_tip)
            interpolate_action.setStatusTip(interpolate_tip)

        self.corrosion_settings_view.set_workflow_state(stage)

    @staticmethod
    def _corrosion_raw_display_name() -> str:
        return "Corrosion profile"

    @staticmethod
    def _corrosion_interpolated_display_name() -> str:
        return "Interpolated profile"

    @classmethod
    def _build_corrosion_raw_session_name(cls) -> str:
        return cls._corrosion_raw_display_name()

    @classmethod
    def _build_corrosion_interpolated_session_name(cls) -> str:
        return cls._corrosion_interpolated_display_name()

    def _build_correction_layer_name(self, base_name: str) -> str:
        normalized_base = str(base_name or "").strip() or "Layer"
        layer_stack = self.session_manager.get_active_layer_stack()
        if layer_stack is None:
            return normalized_base
        existing_names = {
            str(layer.name or "").strip().casefold()
            for layer in layer_stack.layers
        }
        if normalized_base.casefold() not in existing_names:
            return normalized_base
        correction_index = 1
        while True:
            candidate = f"{normalized_base} (correction {correction_index})"
            if candidate.casefold() not in existing_names:
                return candidate
            correction_index += 1

    def _build_corrosion_raw_layer_name(self) -> str:
        return self._build_correction_layer_name(self._corrosion_raw_display_name())

    def _build_corrosion_interpolated_layer_name(self) -> str:
        return self._build_correction_layer_name(
            self._corrosion_interpolated_display_name()
        )

    def _snapshot_active_session_in_manager(self) -> tuple[Optional[str], Optional[str]]:
        active_id = self.session_manager.get_active_session_id()
        if active_id is None or active_id not in self.session_manager._sessions:  # noqa: SLF001
            return active_id, None

        active_name = self.session_manager._sessions[active_id].name  # noqa: SLF001
        self.session_manager._sessions[active_id] = self.session_manager._snapshot(  # noqa: SLF001
            name=active_name,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
        )
        return active_id, active_name

    def _rebuild_raw_corrosion_result_from_active_mask(
        self,
    ) -> Optional[CorrosionWorkflowResult]:
        label_ids = self.view_state_model.corrosion_overlay_label_ids
        support_map = self.view_state_model.corrosion_ascan_support_map
        if label_ids is None or support_map is None:
            return None

        overlay_volume = self.annotation_model.get_mask_volume()
        if overlay_volume is None or getattr(overlay_volume, "ndim", 0) != 3:
            return None

        try:
            class_a_id = int(label_ids[0])
            class_b_id = int(label_ids[1])
        except Exception:
            return None

        try:
            raw_peak_a, raw_peak_b = self.cscan_corrosion_service.build_peak_maps_from_overlay_mask(
                mask_stack=overlay_volume,
                class_A_id=class_a_id,
                class_B_id=class_b_id,
                support_map=support_map,
            )
            raw_distance_map = self.cscan_corrosion_service.build_distance_map_from_peak_maps(
                peak_map_a=raw_peak_a,
                peak_map_b=raw_peak_b,
                use_mm=False,
                resolution_ultrasound_mm=1.0,
            )
            projection, value_range = self.cscan_corrosion_service.compute_corrosion_projection(
                raw_distance_map
            )
            piece_volume_legacy_raw = self.cscan_corrosion_service.build_legacy_piece_volume(
                mask_stack=overlay_volume,
                class_A_id=class_a_id,
                class_B_id=class_b_id,
            )
            piece_volume_raw = self.cscan_corrosion_service.build_piece_volume_from_distance_map(
                raw_distance_map
            )
            piece_anchor = self.cscan_corrosion_service.compute_piece_anchor(
                piece_volume_raw,
                piece_volume_legacy_raw,
            )
        except Exception:
            return None

        self.view_state_model.corrosion_projection = (projection, value_range)
        self.view_state_model.corrosion_peak_index_map_a = raw_peak_a
        self.view_state_model.corrosion_peak_index_map_b = raw_peak_b
        self.view_state_model.corrosion_raw_peak_index_map_a = raw_peak_a
        self.view_state_model.corrosion_raw_peak_index_map_b = raw_peak_b
        self.view_state_model.corrosion_raw_distance_map = raw_distance_map
        self.view_state_model.corrosion_piece_volume_raw = self._copy_piece_volume(piece_volume_raw)
        self.view_state_model.corrosion_piece_volume_legacy_raw = self._copy_piece_volume(
            piece_volume_legacy_raw
        )
        self.view_state_model.corrosion_piece_anchor = self._copy_piece_anchor(piece_anchor)

        return CorrosionWorkflowResult(
            ok=True,
            message="Corrosion analysis completed (raw data)",
            projection=projection,
            value_range=value_range,
            raw_distance_map=raw_distance_map,
            peak_index_map_a=raw_peak_a,
            peak_index_map_b=raw_peak_b,
            raw_peak_index_map_a=raw_peak_a,
            raw_peak_index_map_b=raw_peak_b,
            ascan_support_map=support_map,
            overlay_volume=overlay_volume,
            overlay_label_ids=(class_a_id, class_b_id),
            overlay_palette=self.view_state_model.corrosion_overlay_palette,
            mask_height=int(overlay_volume.shape[1]),
            piece_volume_raw=piece_volume_raw,
            piece_volume_legacy_raw=piece_volume_legacy_raw,
            piece_anchor=piece_anchor,
        )

    def _build_raw_corrosion_workflow_result_from_active_session(
        self,
    ) -> Optional[CorrosionWorkflowResult]:
        if self._current_corrosion_session_stage() != CORROSION_STAGE_RAW:
            return None

        rebuilt = self._rebuild_raw_corrosion_result_from_active_mask()
        if rebuilt is not None:
            return rebuilt

        raw_peak_a = self.view_state_model.corrosion_raw_peak_index_map_a
        raw_peak_b = self.view_state_model.corrosion_raw_peak_index_map_b
        support_map = self.view_state_model.corrosion_ascan_support_map
        label_ids = self.view_state_model.corrosion_overlay_label_ids
        if (
            raw_peak_a is None
            or raw_peak_b is None
            or support_map is None
            or label_ids is None
        ):
            return None

        overlay_volume = self.annotation_model.get_mask_volume()
        if overlay_volume is None or getattr(overlay_volume, "ndim", 0) != 3:
            return None
        mask_height = int(overlay_volume.shape[1])

        projection = None
        value_range = None
        projection_payload = self.view_state_model.corrosion_projection
        if projection_payload is not None:
            try:
                projection, value_range = projection_payload
            except Exception:
                projection = None
                value_range = None

        return CorrosionWorkflowResult(
            ok=True,
            message="Corrosion analysis completed (raw data)",
            projection=projection,
            value_range=value_range,
            raw_distance_map=self.view_state_model.corrosion_raw_distance_map,
            peak_index_map_a=self.view_state_model.corrosion_peak_index_map_a,
            peak_index_map_b=self.view_state_model.corrosion_peak_index_map_b,
            raw_peak_index_map_a=raw_peak_a,
            raw_peak_index_map_b=raw_peak_b,
            ascan_support_map=support_map,
            overlay_volume=overlay_volume,
            overlay_label_ids=label_ids,
            overlay_palette=self.view_state_model.corrosion_overlay_palette,
            mask_height=mask_height,
            piece_volume_raw=getattr(self.view_state_model, "corrosion_piece_volume_raw", None),
            piece_volume_legacy_raw=getattr(
                self.view_state_model,
                "corrosion_piece_volume_legacy_raw",
                None,
            ),
            piece_anchor=getattr(self.view_state_model, "corrosion_piece_anchor", None),
        )

    def _apply_corrosion_session_result(
        self,
        result: CorrosionWorkflowResult,
        *,
        stage: str,
    ) -> None:
        self.corrosion_profile_edit_service.reset()
        self.mask_modification_controller.reset()

        if result.projection is not None and result.value_range is not None:
            self.view_state_model.activate_corrosion(
                result.projection,
                result.value_range,
            )
        else:
            self.view_state_model.corrosion_active = bool(result.overlay_volume is not None)
            self.view_state_model.corrosion_projection = None

        self.view_state_model.corrosion_interpolated_projection = None
        if (
            stage == "interpolated"
            and result.interpolated_projection is not None
            and result.interpolated_value_range is not None
        ):
            self.view_state_model.corrosion_interpolated_projection = (
                result.interpolated_projection,
                result.interpolated_value_range,
            )

        self.view_state_model.corrosion_overlay_palette = result.overlay_palette
        self.view_state_model.corrosion_overlay_label_ids = result.overlay_label_ids
        self.view_state_model.corrosion_peak_index_map_a = result.peak_index_map_a
        self.view_state_model.corrosion_peak_index_map_b = result.peak_index_map_b
        self.view_state_model.corrosion_raw_peak_index_map_a = result.raw_peak_index_map_a
        self.view_state_model.corrosion_raw_peak_index_map_b = result.raw_peak_index_map_b
        self.view_state_model.corrosion_raw_distance_map = result.raw_distance_map
        self.view_state_model.corrosion_ascan_support_map = result.ascan_support_map
        self.view_state_model.corrosion_piece_volume_raw = self._copy_piece_volume(
            result.piece_volume_raw
        )
        self.view_state_model.corrosion_piece_volume_interpolated = self._copy_piece_volume(
            result.piece_volume_interpolated
        )
        self.view_state_model.corrosion_piece_volume_legacy_raw = self._copy_piece_volume(
            result.piece_volume_legacy_raw
        )
        self.view_state_model.corrosion_piece_volume_legacy_interpolated = self._copy_piece_volume(
            result.piece_volume_legacy_interpolated
        )
        self.view_state_model.corrosion_piece_anchor = self._copy_piece_anchor(
            result.piece_anchor
        )
        self.view_state_model.corrosion_piece_show_interpolated = bool(
            stage == "interpolated"
            and (
                result.piece_volume_interpolated is not None
                or result.piece_volume_legacy_interpolated is not None
            )
        )
        self.view_state_model.set_corrosion_session_stage(stage)
        self.view_state_model.corrosion_active = True
        self._sync_cscan_labels()

        if result.overlay_volume is not None:
            self.annotation_model.set_mask_volume(result.overlay_volume)
            self.view_state_model.corrosion_overlay_volume = self.annotation_model.get_mask_volume()
            palette = result.overlay_palette or {}
            self.annotation_model.label_palette = dict(palette)
            self.annotation_model.label_visibility = {lbl: True for lbl in palette.keys()}
            self.annotation_model.ensure_persistent_labels()

        self.temp_mask_model.clear()
        if result.overlay_volume is not None:
            self.temp_mask_model.initialize(result.overlay_volume.shape)
        self.roi_model.clear()

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
        self._primary_view_name = titles.primary_view_name
        self._secondary_view_name = titles.secondary_view_name
        self.ucoordinate_dock.setWindowTitle(titles.primary_title)
        self.vcoordinate_dock.setWindowTitle(titles.secondary_title)
        self._sync_coordinate_view_labels()

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

    def _refresh_views(self, *, rebuild_volume_view: bool = True) -> None:
        """Push the current volume state into all views."""
        volume = self._current_volume()
        self._sync_cscan_labels()
        if volume is None:
            self.tools_panel.set_nde_opacity_available(False)
            self.endview_controller.set_primary_endview_name("-")
            self.endview_controller.set_secondary_endview_name("-")
            self._clear_annotation_position_label()
            return
        self.view_state_model.set_slice_bounds(0, volume.shape[0] - 1)
        self.view_state_model.set_secondary_slice_bounds(0, volume.shape[2] - 1)

        # Sélectionne l’indice de tranche courant dans les bornes valides
        slice_idx = self.view_state_model.clamp_slice(self.view_state_model.current_slice)
        self.view_state_model.set_secondary_slice(self.view_state_model.secondary_slice)
        secondary_slice_idx = self.view_state_model.secondary_slice
        self.endview_controller.set_slice_bounds(0, volume.shape[0] - 1)
        self.endview_controller.set_secondary_slice_bounds(0, volume.shape[2] - 1)
        self._sync_coordinate_view_labels()

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
        self._sync_cscan_labels()
        self.endview_controller.sync_mode()

        # Le dataset NDE ne change pas au switch de session: on ne reconstruit
        # la scène VisPy que lors d'un vrai changement de volume.
        if rebuild_volume_view:
            axis_order = self.view_state_model.axis_order
            self.volume_view.set_volume(volume, slice_idx=slice_idx, axis_order=axis_order)
        else:
            self.volume_view.set_slice_index(slice_idx, update_slider=True, emit=False)
        self.volume_view.set_nde_opacity(self.view_state_model.nde_alpha)
        self.volume_view.set_nde_contrast(self.view_state_model.nde_contrast)
        self.volume_view.set_secondary_slice_index(
            secondary_slice_idx,
            update_slider=True,
            emit=False,
        )
        self.endview_controller.set_nde_opacity(self.view_state_model.nde_alpha)
        self.endview_controller.set_nde_contrast(self.view_state_model.nde_contrast)
        self.endview_controller.set_smooth_enabled(
            getattr(self.view_state_model, "show_endview_smooth", True)
        )
        self.tools_panel.set_nde_opacity(self.view_state_model.nde_alpha)
        self.tools_panel.set_nde_contrast(self.view_state_model.nde_contrast)
        self.tools_panel.set_nde_opacity_available(True)

        # Applique l’overlay après la (re)construction de la scène 3D
        self.annotation_controller.refresh_overlay(rebuild=False)

        # Reste des mises à jour
        self._update_ascan_trace()
        self.endview_controller.set_cross_visible(self.view_state_model.show_cross)
        self.cscan_controller.set_cross_visible(self.view_state_model.show_cross)
        self.ascan_controller.set_marker_visible(self.view_state_model.show_cross)
        self._update_endview_label()
        current_point = self.view_state_model.current_point or self.view_state_model.cursor_position
        if current_point is not None:
            self._set_annotation_position_label(*current_point)
        else:
            self._clear_annotation_position_label()


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

    def _after_layer_switch(self, *, defer_volume: bool) -> None:
        """Synchronize only the UI state affected by an intra-session layer change."""
        volume = self._current_volume()
        self._ensure_corrosion_runtime_cache()
        if volume is not None:
            self.cscan_controller.update_views(volume)
        self._sync_cscan_labels()
        self._sync_corrosion_workflow_controls()
        self.annotation_controller.sync_overlay_settings()
        self._sync_tools_labels(select_label_id=self.view_state_model.active_label)
        self.annotation_controller.refresh_overlay(
            defer_volume=defer_volume,
            rebuild=False,
        )
        self.corrosion_profile_controller.sync_anchors()
        self._restore_piece3d_state_from_view_state(sync_action=True)

    def _after_session_switch(self, *, rebuild_volume_view: bool = False) -> None:
        """Synchronise l'état du modèle actif vers les vues."""
        self.annotation_controller.clear_apply_history()
        self._ensure_corrosion_runtime_cache()
        if self.view_state_model.threshold is not None:
            self.tools_panel.set_threshold_value(int(self.view_state_model.threshold))
        self.tools_panel.set_force_threshold_erase_checked(
            getattr(self.view_state_model, "force_threshold_erase", False)
        )
        self.tools_panel.set_apply_auto_checked(getattr(self.view_state_model, "apply_auto", False))
        self.tools_panel.set_mod_apply_auto_checked(
            getattr(self.view_state_model, "mod_apply_auto", False)
        )
        self.tools_panel.set_threshold_auto_checked(self.view_state_model.threshold_auto)
        self.tools_panel.set_apply_volume_checked(self.view_state_model.apply_volume)
        self.tools_panel.set_roi_persistence_checked(self.view_state_model.roi_persistence)
        self.tools_panel.set_closing_mask_checked(
            getattr(self.view_state_model, "closing_mask_enabled", False)
        )
        self.tools_panel.set_clean_outliers_checked(
            getattr(self.view_state_model, "clean_outliers_enabled", False)
        )
        self.tools_panel.set_volume_view_overlay_checked(
            getattr(self.view_state_model, "show_volume_view_overlay", True)
        )
        self.tools_panel.set_paint_size(self.view_state_model.paint_radius)
        self._sync_display_toggle_actions()
        self._sync_coordinate_view_labels()
        self.endview_controller.set_smooth_enabled(
            getattr(self.view_state_model, "show_endview_smooth", True)
        )
        current_tool_mode = self.view_state_model.tool_mode or self.tools_panel.current_tool_mode()
        if current_tool_mode:
            self.tools_panel.select_tool_mode(current_tool_mode)
            self._on_tool_mode_changed(current_tool_mode)
        self.mask_modification_controller.reset()
        self.endview_controller.set_cross_visible(self.view_state_model.show_cross)
        self.cscan_controller.set_cross_visible(self.view_state_model.show_cross)
        self.ascan_controller.set_marker_visible(self.view_state_model.show_cross)
        self.ascan_controller.set_overlay_opacity(self.view_state_model.overlay_alpha)
        self.annotation_controller.set_outline_only(
            getattr(self.view_state_model, "show_outline_only", False)
        )
        self.annotation_controller.set_restriction_visible(
            getattr(self.view_state_model, "show_restriction", True)
        )

        # Colormaps
        self._apply_saved_colormaps()
        self.volume_view.set_volume_planes_visible(
            getattr(self.view_state_model, "show_volume_planes", True)
        )
        self._sync_prune_label_choices()
        self._sync_corrosion_label_choices()
        self._sync_corrosion_workflow_controls()
        self.nde_settings_view.set_closing_mask_tolerance(
            getattr(self.view_state_model, "closing_mask_tolerance", 0)
        )
        self.nde_settings_view.set_closing_mask_merge_distance(
            getattr(self.view_state_model, "closing_mask_merge_distance", 0)
        )
        self.nde_settings_view.set_clean_outliers_tolerance(
            getattr(self.view_state_model, "clean_outliers_tolerance", 0)
        )
        self.nde_settings_view.set_clean_outliers_thin_line_max_width(
            getattr(self.view_state_model, "clean_outliers_thin_line_max_width", 0)
        )
        self.nde_settings_view.set_clean_outliers_thin_gap_max_width(
            getattr(self.view_state_model, "clean_outliers_thin_gap_max_width", 0)
        )
        self.nde_settings_view.set_clean_outliers_contour_smoothing(
            getattr(self.view_state_model, "clean_outliers_contour_smoothing", 0)
        )
        self.nde_settings_view.set_prune_peak_selection_mode(
            getattr(self.view_state_model, "prune_peak_selection_mode", "max_peak")
        )

        self._apply_annotation_action(getattr(self.view_state_model, "annotation_action", "draw"))
        self.annotation_controller.sync_overlay_settings()
        self._sync_tools_labels(select_label_id=self.view_state_model.active_label)
        self.annotation_controller.apply_overlay_opacity()
        self.tools_panel.set_overlay_opacity(self.view_state_model.overlay_alpha)
        self.tools_panel.set_nde_opacity(self.view_state_model.nde_alpha)
        self.tools_panel.set_nde_contrast(self.view_state_model.nde_contrast)
        self.tools_panel.set_nde_opacity_available(self._current_volume() is not None)
        self._refresh_views(rebuild_volume_view=rebuild_volume_view)
        self.corrosion_profile_controller.sync_anchors()
        self._restore_piece3d_state_from_view_state(sync_action=True)

    # ------------------------------------------------------------------ #
    # Corrosion completion handling
    # ------------------------------------------------------------------ #
    def _on_corrosion_completed_legacy_session_workflow(self, result) -> None:
        """Crée une session corrosion avec les données brutes (sans interpolation).

        L'interpolation sera déclenchée manuellement via le bouton Calculer du ToolsPanel.
        """
        self.corrosion_profile_edit_service.reset()
        self.mask_modification_controller.reset()

        # Stocker le résultat brut pour réutilisation lors de l'interpolation
        self._apply_corrosion_session_result(result, stage="raw")

        # Met à jour l'état courant avec les données brutes

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

        # Prépare la session avec les données brutes (pas d'interpolation)
        name = self._build_corrosion_raw_session_name()
        self.view_state_model.corrosion_piece_view_enabled = self._workflow_result_has_piece3d_data(
            result
        )

        new_session_id = self.session_manager.create_from_models(
            name=name,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
            set_active=True,
            save_active=False,
        )
        self.session_workspace_controller.register_unsaved_session(new_session_id, dirty=True)

        self._pre_corrosion_session_state = None
        self._pre_corrosion_session_id = None

        self._after_session_switch()

    def _on_corrosion_interpolation_requested_legacy_session_workflow(self, algo: str) -> None:
        """Applique l'algorithme d'interpolation choisi sur les données brutes."""
        algo = self.view_state_model.set_corrosion_interpolation_algo(algo)
        self.corrosion_settings_view.set_interpolation_algo(algo)
        if not self.view_state_model.can_run_corrosion_interpolation():
            self.status_message(
                "Interpolate is available only from a raw corrosion layer.",
                3000,
            )
            self._sync_corrosion_workflow_controls()
            return

        if self.corrosion_profile_edit_service.has_pending_edits():
            if not self.corrosion_profile_controller.commit_pending_edits():
                self.status_message(
                    "Unable to apply the pending corrosion edits.",
                    5000,
                )
                return

        raw = self._build_raw_corrosion_workflow_result_from_active_session()
        if raw is None or not raw.ok:
            self.status_message("No raw corrosion analysis available.", 3000)
            self._sync_corrosion_workflow_controls()
            return

        self._snapshot_active_session_in_manager()

        nde_model = self.nde_model if hasattr(self, "nde_model") else None
        self.status_message(f"Interpolation ({algo}) in progress...", 2000)

        interp_result = self.corrosion_workflow_service.run_interpolation(
            raw_result=raw,
            algo=algo,
            nde_model=nde_model,
        )

        if not interp_result.ok:
            self.status_message(interp_result.message, 5000)
            return

        # Met à jour le view_state avec les données interpolées
        self._apply_corrosion_session_result(interp_result, stage="interpolated")

        self.view_state_model.corrosion_piece_view_enabled = self._workflow_result_has_piece3d_data(
            interp_result
        )
        new_session_id = self.session_manager.create_from_models(
            name=self._build_corrosion_interpolated_session_name(),
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
            set_active=True,
            save_active=False,
        )
        self.session_workspace_controller.register_unsaved_session(new_session_id, dirty=True)


        # Remplace les masques par l'overlay interpolé

        # Rafraîchit toutes les vues
        self._after_session_switch()

        self.status_message(interp_result.message, 3000)

    def _on_corrosion_completed(self, result) -> None:
        """Create a raw corrosion layer inside the active session."""
        self.corrosion_profile_edit_service.reset()
        self.mask_modification_controller.reset()
        self._apply_corrosion_session_result(result, stage="raw")
        self.view_state_model.corrosion_piece_view_enabled = self._workflow_result_has_piece3d_data(
            result
        )
        created_layer_id = self.session_manager.create_layer_from_model_state(
            name=self._build_corrosion_raw_layer_name(),
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
            set_active=True,
            save_current=False,
            layer_kind="corrosion",
        )
        if created_layer_id is None:
            self.status_message("Unable to create the corrosion raw layer.", 5000)
            return
        self._after_layer_stack_changed()

    def _on_corrosion_interpolation_requested(self, algo: str) -> None:
        """Apply the selected interpolation algorithm and create a derived layer."""
        algo = self.view_state_model.set_corrosion_interpolation_algo(algo)
        self.corrosion_settings_view.set_interpolation_algo(algo)
        if not self.view_state_model.can_run_corrosion_interpolation():
            self.status_message(
                "Interpolate is available only from a raw corrosion layer.",
                3000,
            )
            self._sync_corrosion_workflow_controls()
            return

        if self.corrosion_profile_edit_service.has_pending_edits():
            if not self.corrosion_profile_controller.commit_pending_edits():
                self.status_message(
                    "Unable to apply the pending corrosion edits.",
                    5000,
                )
                return

        raw = self._build_raw_corrosion_workflow_result_from_active_session()
        if raw is None or not raw.ok:
            self.status_message("No raw corrosion analysis available.", 3000)
            self._sync_corrosion_workflow_controls()
            return

        self.session_manager.sync_active_layer_from_model(
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
        )
        nde_model = self.nde_model if hasattr(self, "nde_model") else None
        self.status_message(f"Interpolation ({algo}) in progress...", 2000)

        interp_result = self.corrosion_workflow_service.run_interpolation(
            raw_result=raw,
            algo=algo,
            nde_model=nde_model,
        )

        if not interp_result.ok:
            self.status_message(interp_result.message, 5000)
            return

        self._apply_corrosion_session_result(interp_result, stage="interpolated")
        self.view_state_model.corrosion_piece_view_enabled = self._workflow_result_has_piece3d_data(
            interp_result
        )
        created_layer_id = self.session_manager.create_layer_from_model_state(
            name=self._build_corrosion_interpolated_layer_name(),
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
            set_active=True,
            save_current=False,
            layer_kind="corrosion",
        )
        if created_layer_id is None:
            self.status_message("Unable to create the corrosion interpolated layer.", 5000)
            return
        self._after_layer_stack_changed()
        self.status_message(interp_result.message, 3000)

    def _current_volume(self) -> Optional[Any]:
        if self.nde_model is None:
            return None
        return self.nde_model.get_active_volume()

    @staticmethod
    def _workflow_result_has_piece3d_data(result: CorrosionWorkflowResult) -> bool:
        volumes = (
            result.piece_volume_raw,
            result.piece_volume_interpolated,
            result.piece_volume_legacy_raw,
            result.piece_volume_legacy_interpolated,
        )
        return any(volume is not None and volume.size > 0 for volume in volumes)

    def _on_piece3d_toggled(self, checked: bool) -> None:
        """Show/hide the embedded piece3D view inside the Volume dock."""
        self.view_state_model.corrosion_piece_view_enabled = bool(checked)
        if checked:
            self._ensure_corrosion_runtime_cache(include_piece=True)
            self._restore_piece3d_state_from_view_state(sync_action=False)
            if not self._has_piece3d_data():
                self.status_message("No corrosion 3D solid available.", 3000)
                self._show_standard_volume_view(sync_action=True)
                return
            self._show_piece3d_view(sync_action=True)
        else:
            self._show_standard_volume_view(sync_action=False)

    def _sync_piece3d_action(self, checked: bool) -> None:
        """Update the menu action without re-triggering the toggle handler."""
        action = getattr(self.ui, "actionAfficher_solide_3d", None)
        if action is None:
            return
        action.blockSignals(True)
        action.setChecked(bool(checked))
        action.blockSignals(False)

    def _has_piece3d_interpolated(self) -> bool:
        return bool(
            (self._piece_volume_interpolated is not None and self._piece_volume_interpolated.size > 0)
            or (
                self._piece_volume_legacy_interpolated is not None
                and self._piece_volume_legacy_interpolated.size > 0
            )
        )

    def _has_piece3d_raw(self) -> bool:
        return bool(
            (self._piece_volume_raw is not None and self._piece_volume_raw.size > 0)
            or (self._piece_volume_legacy_raw is not None and self._piece_volume_legacy_raw.size > 0)
        )

    def _has_piece3d_data(self) -> bool:
        return self._has_piece3d_interpolated() or self._has_piece3d_raw()

    @staticmethod
    def _copy_piece_volume(volume: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if volume is None:
            return None
        return np.array(volume, dtype=np.float32, copy=True)

    @staticmethod
    def _copy_piece_anchor(
        anchor: Optional[tuple[float, float, float]],
    ) -> Optional[tuple[float, float, float]]:
        if anchor is None:
            return None
        try:
            x, y, z = anchor
        except Exception:
            return None
        return (float(x), float(y), float(z))

    def _persist_piece3d_state_to_view_state(self) -> None:
        self.view_state_model.corrosion_piece_volume_raw = self._piece_volume_raw
        self.view_state_model.corrosion_piece_volume_interpolated = (
            self._piece_volume_interpolated
        )
        self.view_state_model.corrosion_piece_volume_legacy_raw = self._piece_volume_legacy_raw
        self.view_state_model.corrosion_piece_volume_legacy_interpolated = (
            self._piece_volume_legacy_interpolated
        )
        self.view_state_model.corrosion_piece_anchor = self._copy_piece_anchor(
            self._piece_anchor
        )
        self.view_state_model.corrosion_piece_show_interpolated = bool(
            self._piece_show_interpolated
        )

    def _restore_piece3d_state_from_view_state(self, *, sync_action: bool) -> None:
        self._piece_volume_raw = getattr(self.view_state_model, "corrosion_piece_volume_raw", None)
        self._piece_volume_interpolated = getattr(
            self.view_state_model,
            "corrosion_piece_volume_interpolated",
            None,
        )
        self._piece_volume_legacy_raw = getattr(
            self.view_state_model,
            "corrosion_piece_volume_legacy_raw",
            None,
        )
        self._piece_volume_legacy_interpolated = getattr(
            self.view_state_model,
            "corrosion_piece_volume_legacy_interpolated",
            None,
        )
        self._piece_anchor = self._copy_piece_anchor(
            getattr(self.view_state_model, "corrosion_piece_anchor", None)
        )
        self._piece_show_interpolated = bool(
            getattr(self.view_state_model, "corrosion_piece_show_interpolated", True)
        )
        restore_piece3d_view = bool(
            getattr(self.view_state_model, "corrosion_piece_view_enabled", False)
        )
        if restore_piece3d_view and self._has_piece3d_data():
            self._show_piece3d_view(sync_action=sync_action)
            return
        self._show_standard_volume_view(sync_action=sync_action)

    def _sync_piece3d_view(self) -> None:
        if self._piece3d_view is None:
            return
        if self._piece_show_interpolated and not self._has_piece3d_interpolated():
            self._piece_show_interpolated = False
        elif not self._piece_show_interpolated and not self._has_piece3d_raw():
            self._piece_show_interpolated = True
        self.view_state_model.corrosion_piece_show_interpolated = bool(
            self._piece_show_interpolated
        )

        self._piece3d_view.sync_piece_state(
            distance_raw=self._piece_volume_raw,
            distance_interpolated=self._piece_volume_interpolated,
            legacy_raw=self._piece_volume_legacy_raw,
            legacy_interpolated=self._piece_volume_legacy_interpolated,
            show_interpolated=self._piece_show_interpolated,
            anchor=self._piece_anchor,
        )
        self._update_piece_toggle_label()

    def _show_piece3d_view(self, *, sync_action: bool) -> None:
        """Display the corrosion piece view inside the Volume dock stack."""
        if self._volume_stack is None or self._piece3d_page is None:
            return
        self._sync_piece3d_view()
        self._volume_stack.setCurrentWidget(self._piece3d_page)
        if sync_action:
            self._sync_piece3d_action(True)

    def _show_standard_volume_view(self, *, sync_action: bool) -> None:
        """Display the standard NDE volume view inside the Volume dock stack."""
        if self._volume_stack is not None:
            self._volume_stack.setCurrentWidget(self.volume_view)
        if sync_action:
            self._sync_piece3d_action(False)

    def _reset_piece3d_state(self, *, sync_action: bool) -> None:
        """Clear cached corrosion piece data and restore the standard volume page."""
        self._piece_volume_raw = None
        self._piece_volume_interpolated = None
        self._piece_volume_legacy_raw = None
        self._piece_volume_legacy_interpolated = None
        self._piece_anchor = None
        self._piece_show_interpolated = True
        self.view_state_model.clear_corrosion_piece_state()
        self._sync_piece3d_view()
        self._show_standard_volume_view(sync_action=sync_action)

    def _show_piece3d_volume(
        self,
        *,
        raw_volume: Optional[np.ndarray],
        interpolated_volume: Optional[np.ndarray],
        legacy_raw_volume: Optional[np.ndarray],
        legacy_interpolated_volume: Optional[np.ndarray],
    ) -> None:
        """Cache et rafraîchit les données de la pièce 3D corrosion."""
        self._piece_volume_raw = self._copy_piece_volume(raw_volume)
        self._piece_volume_interpolated = self._copy_piece_volume(interpolated_volume)
        self._piece_volume_legacy_raw = self._copy_piece_volume(legacy_raw_volume)
        self._piece_volume_legacy_interpolated = self._copy_piece_volume(
            legacy_interpolated_volume
        )
        self._piece_show_interpolated = self._has_piece3d_interpolated()
        self._persist_piece3d_state_to_view_state()
        piece3d_visible = bool(
            self._volume_stack is not None
            and self._piece3d_page is not None
            and self._volume_stack.currentWidget() is self._piece3d_page
        )
        action = getattr(self.ui, "actionAfficher_solide_3d", None)
        if piece3d_visible or (action is not None and action.isChecked()):
            self._show_piece3d_view(sync_action=False)
            return
        self._update_piece_toggle_label()

    def _toggle_piece_volume(self) -> None:
        """Bascule entre volume brut et volume interpolé si les deux sont disponibles."""
        if self._piece3d_view is None:
            return

        if not (self._has_piece3d_interpolated() and self._has_piece3d_raw()):
            return

        self._piece_show_interpolated = not self._piece_show_interpolated
        self._persist_piece3d_state_to_view_state()
        self._piece3d_view.set_piece_show_interpolated(self._piece_show_interpolated)
        self._piece3d_view.set_anchor_point(self._piece_anchor)
        self._update_piece_toggle_label()

    def _update_piece_toggle_label(self) -> None:
        """Met à jour le texte du bouton selon le volume affiché."""
        if self._piece_toggle_btn is None:
            return
        if self._piece_show_interpolated:
            self._piece_toggle_btn.setText("Show raw version")
        else:
            self._piece_toggle_btn.setText("Show interpolated version")

    def _update_main_window_title(self) -> None:
        """Reflect the current NDE absolute path in the main window title bar."""
        nde_path = str(
            self._nde_path
            or ((self.nde_model.metadata or {}).get("path") if self.nde_model is not None else "")
            or ""
        ).strip()
        title = "Corrosion Mapping"
        if nde_path:
            title = f"{title} - {nde_path}"
        self.main_window.setWindowTitle(title)

    def _update_endview_label(self) -> None:
        """Reflect the current primary and secondary endview identifiers."""
        volume = self._current_volume()
        if volume is None:
            self.endview_controller.set_primary_endview_name("-")
            self.endview_controller.set_secondary_endview_name("-")
            return
        primary_slice_idx = int(self.view_state_model.current_slice)
        secondary_slice_idx = int(self.view_state_model.secondary_slice)
        primary_name = f"endview_{primary_slice_idx * 1500:012d}.png"
        secondary_name = f"endview_{secondary_slice_idx * 1500:012d}.png"
        self.endview_controller.set_primary_endview_name(primary_name)
        self.endview_controller.set_secondary_endview_name(secondary_name)

    def _set_annotation_position_label(self, x: int, y: int) -> None:
        """Reflect the active annotation cursor into the primary endview status."""
        self.endview_controller.set_primary_status_position(int(x), int(y))

    def _clear_annotation_position_label(self) -> None:
        """Clear the primary endview cursor text when no position is active."""
        self.endview_controller.clear_primary_status_position()

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
