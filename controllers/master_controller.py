import logging
import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedLayout,
    QWidget,
    QVBoxLayout,
)

from config.constants import MASK_COLORS_BGRA
from controllers.annotation_controller import AnnotationController
from controllers.ascan_controller import AScanController
from controllers.cscan_controller import CScanController
from models.annotation_model import AnnotationModel
from models.nde_model import NdeModel
from models.view_state_model import ViewStateModel
from models.roi_model import RoiModel
from models.temp_mask_model import TempMaskModel
from services.annotation_session_manager import AnnotationSessionManager
from services.annotation_service import AnnotationService
from services.ascan_service import AScanService
from services.nnunet_service import NnUnetResult, NnUnetService
from services.overlay_loader import OverlayLoader
from services.overlay_service import OverlayService
from services.overlay_export import OverlayExport
from services.endview_export import EndviewExportService
from services.split_service import SplitFlawNoflawService
from services.nde_loader import NdeLoader
from services.cscan_corrosion_service import CScanCorrosionService, CorrosionWorkflowService
from services.corrosion_label_service import CorrosionLabelService
from ui_mainwindow import Ui_MainWindow
from views.annotation_view import AnnotationView
from views.cscan_view_corrosion import CscanViewCorrosion
from views.corrosion_settings_view import CorrosionSettingsView
from views.nde_settings_view import NdeSettingsView
from views.endview_resize_dialog import EndviewResizeDialog
from views.overlay_settings_view import OverlaySettingsView
from views.piece3d_view import Piece3DView
from views.session_manager_dialog import SessionManagerDialog


class MasterController:
    """Coordinates models and the Designer-built UI without embedding business logic."""

    def __init__(self, main_window: Optional[QMainWindow] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.main_window = main_window or QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)

        self.nde_model: Optional[NdeModel] = None
        self.annotation_model = AnnotationModel()
        self.view_state_model = ViewStateModel()
        self.roi_model = RoiModel()
        self.temp_mask_model = TempMaskModel()
        self.nde_loader = NdeLoader()
        self.session_manager = AnnotationSessionManager()
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
        self.overlay_settings_view = OverlaySettingsView(self.main_window)
        self.nde_settings_view = NdeSettingsView(self.main_window)
        self.corrosion_settings_view = CorrosionSettingsView(self.main_window)
        self._piece3d_window: Optional[QWidget] = None
        self._piece3d_view: Optional[Piece3DView] = None
        self._piece_toggle_btn: Optional[QPushButton] = None
        self._piece_volume_raw: Optional[np.ndarray] = None
        self._piece_volume_interpolated: Optional[np.ndarray] = None
        self._piece_anchor: Optional[tuple[float, float, float]] = None
        self._piece_show_interpolated: bool = True
        self._shortcuts: list[QShortcut] = []
        self._omniscan_lut: Optional[np.ndarray] = None
        self._session_dialog: Optional[SessionManagerDialog] = None

        # References to Designer-created views.
        self.annotation_view: AnnotationView = self.ui.frame_3
        self.volume_view = self.ui.frame_5
        self.ascan_view = self.ui.frame_7
        self.tools_panel = self.ui.dockWidgetContents_2

        # C-scan stacked layout (standard + corrosion)
        self.cscan_view = self.ui.frame_4
        self.cscan_view_corrosion = None
        self.cscan_stack = None
        try:
            splitter = self.ui.splitter
        except Exception:
            splitter = None
        if splitter is not None and self.cscan_view is not None:
            container = QWidget(parent=splitter)
            stack = QStackedLayout(container)
            stack.setContentsMargins(0, 0, 0, 0)
            idx = splitter.indexOf(self.cscan_view)

            self.cscan_view.setParent(container)
            stack.addWidget(self.cscan_view)

            self.cscan_view_corrosion = CscanViewCorrosion(parent=container)
            stack.addWidget(self.cscan_view_corrosion)
            stack.setCurrentWidget(self.cscan_view)

            if idx >= 0:
                splitter.replaceWidget(idx, container)
            else:
                splitter.addWidget(container)

            self.cscan_stack = stack

        self.annotation_service = AnnotationService()

        # Apply default colormaps
        self.annotation_view.set_colormap(self.view_state_model.endview_colormap, None)
        self.volume_view.set_base_colormap(self.view_state_model.endview_colormap, None)
        if self.cscan_view is not None:
            self.cscan_view.set_colormap(self.view_state_model.cscan_colormap, None)

        # Annotation controller (overlays)
        self.annotation_controller = AnnotationController(
            annotation_model=self.annotation_model,
            view_state_model=self.view_state_model,
            roi_model=self.roi_model,
            temp_mask_model=self.temp_mask_model,
            annotation_service=self.annotation_service,
            overlay_service=self.overlay_service,
            overlay_export=self.overlay_export,
            annotation_view=self.annotation_view,
            volume_view=self.volume_view,
            overlay_settings_view=self.overlay_settings_view,
            logger=self.logger,
            get_volume=self._current_volume,
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
            ascan_view=self.ascan_view,
            endview_view=self.annotation_view,
            view_state_model=self.view_state_model,
            set_cscan_crosshair=self.cscan_controller.set_crosshair,
        )

        self._connect_actions()
        self._connect_signals()
        self._register_shortcuts()
        self.annotation_controller.apply_overlay_opacity()
        self._sync_tools_labels()
        self._update_nde_label()
        self._update_endview_label()
        self.session_manager.ensure_default(
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
        )

    def _connect_actions(self) -> None:
        """Wire menu actions to controller handlers."""
        self.ui.actionopen_nde.triggered.connect(self._on_open_nde)
        self.ui.actioncharger_npz.triggered.connect(self._on_load_npz)
        if hasattr(self.ui, "actionSauvegarder"):
            self.ui.actionSauvegarder.triggered.connect(self._on_save)
        self.ui.actionExporter_npz.triggered.connect(self._on_save)
        if hasattr(self.ui, "actionExporter_endviews"):
            self.ui.actionExporter_endviews.triggered.connect(self._on_export_endviews)
        if hasattr(self.ui, "actionSplit_flaw_noflaw"):
            self.ui.actionSplit_flaw_noflaw.triggered.connect(self._on_split_flaw_noflaw)
        self.ui.actionParam_tres.triggered.connect(self._on_open_settings)
        self.ui.actionParam_tres_2.triggered.connect(self.annotation_controller.open_overlay_settings)
        if hasattr(self.ui, "actionParam_tres_3"):
            self.ui.actionParam_tres_3.setText("Parametres corrosion")
            self.ui.actionParam_tres_3.triggered.connect(self._on_open_corrosion_settings)
        self.ui.actionCorrosion_analyse.triggered.connect(self.cscan_controller.run_corrosion_analysis)
        if hasattr(self.ui, "actionnnunet"):
            self.ui.actionnnunet.triggered.connect(self._on_run_nnunet)
        self.ui.actionQuitter.triggered.connect(self._on_quit)
        if hasattr(self.ui, "actionSession_selector"):
            self.ui.actionSession_selector.triggered.connect(self._open_session_dialog)
        if hasattr(self.ui, "actionToggle_tools_panel"):
            action = self.ui.actionToggle_tools_panel
            action.setCheckable(True)
            action.triggered.connect(self._on_toggle_tools_panel)
            dock = getattr(self.ui, "dockWidget_2", None)
            if dock is not None:
                action.setChecked(dock.isVisible())
                dock.visibilityChanged.connect(self._on_tools_panel_visibility_changed)
        if hasattr(self.ui, "actionResize_endview"):
            self.ui.actionResize_endview.triggered.connect(self._on_resize_endview)
        if hasattr(self.ui, "actionAfficher_solide_3d"):
            action = self.ui.actionAfficher_solide_3d
            action.setCheckable(True)
            action.setChecked(False)
            action.toggled.connect(self._on_piece3d_toggled)

    def _connect_signals(self) -> None:
        """Wire view signals to controller handlers."""
        self.tools_panel.attach_designer_widgets(
            slice_slider=self.ui.horizontalSlider_2,
            slice_label=self.ui.label_3,
            goto_button=self.ui.pushButton,
            threshold_slider=self.ui.horizontalSlider,
            threshold_label=self.ui.label_2,
            free_hand_radio=self.ui.radioButton,
            box_radio=self.ui.radioButton_2,
            grow_radio=self.ui.radioButton_3,
            line_radio=self.ui.radioButton_5,
            paint_radio=self.ui.radioButton_4,
            paint_slider=self.ui.horizontalSlider_3,
            nde_label=self.ui.label,
            endview_label=self.ui.label_5,
            position_label=self.ui.label_4,
            overlay_checkbox=self.ui.checkBox_5,
            cross_checkbox=self.ui.checkBox_4,
            apply_volume_checkbox=self.ui.checkBox,
            threshold_auto_checkbox=self.ui.checkBox_2,
            roi_persistence_checkbox=self.ui.checkBox_3,
            roi_recompute_button=self.ui.pushButton_2,
            roi_delete_button=self.ui.pushButton_3,
            selection_cancel_button=self.ui.pushButton_4,
            previous_button=self.ui.pushButton_5,
            next_button=self.ui.pushButton_6,
            apply_roi_button=self.ui.pushButton_7,
            label_container=self.ui.scrollAreaWidgetContents,
        )

        self.tools_panel.set_overlay_checked(self.view_state_model.show_overlay)
        self.tools_panel.set_cross_checked(self.view_state_model.show_cross)
        if self.view_state_model.threshold is not None:
            self.tools_panel.set_threshold_value(int(self.view_state_model.threshold))
        self.tools_panel.set_paint_size(self.view_state_model.paint_radius)

        self.tools_panel.slice_changed.connect(self._on_slice_changed)
        self.tools_panel.goto_requested.connect(self._on_goto_requested)
        self.tools_panel.tool_mode_changed.connect(self.annotation_controller.on_tool_mode_changed)
        self.tools_panel.paint_size_changed.connect(self.annotation_controller.on_paint_size_changed)
        self.tools_panel.threshold_changed.connect(self.annotation_controller.on_threshold_changed)
        self.tools_panel.threshold_auto_toggled.connect(self.annotation_controller.on_threshold_auto_toggled)
        self.tools_panel.apply_volume_toggled.connect(self.annotation_controller.on_apply_volume_toggled)
        self.tools_panel.overlay_toggled.connect(self.annotation_controller.on_overlay_toggled)
        self.tools_panel.cross_toggled.connect(self._on_cross_toggled)
        self.tools_panel.roi_persistence_toggled.connect(self.annotation_controller.on_roi_persistence_toggled)
        self.tools_panel.roi_recompute_requested.connect(self.annotation_controller.on_roi_recompute_requested)
        self.tools_panel.roi_delete_requested.connect(self.annotation_controller.on_roi_delete_requested)
        self.tools_panel.selection_cancel_requested.connect(self.annotation_controller.on_selection_cancel_requested)
        self.tools_panel.previous_requested.connect(self._on_previous_slice)
        self.tools_panel.next_requested.connect(self._on_next_slice)
        self.tools_panel.apply_roi_requested.connect(self.annotation_controller.on_apply_temp_mask_requested)
        self.tools_panel.label_selected.connect(self.annotation_controller.on_label_selected)

        self.annotation_view.slice_changed.connect(self._on_slice_changed)
        self.annotation_view.mouse_clicked.connect(self.annotation_controller.on_annotation_mouse_clicked)
        self.annotation_view.freehand_started.connect(self.annotation_controller.on_annotation_freehand_started)
        self.annotation_view.freehand_point_added.connect(self.annotation_controller.on_annotation_freehand_point_added)
        self.annotation_view.freehand_completed.connect(self.annotation_controller.on_annotation_freehand_completed)
        self.annotation_view.line_drawn.connect(self.annotation_controller.on_annotation_line_drawn)
        self.annotation_view.box_drawn.connect(self.annotation_controller.on_annotation_box_drawn)
        self.annotation_view.restriction_rect_changed.connect(
            self.annotation_controller.on_restriction_rect_changed
        )
        self.annotation_view.selection_cancel_requested.connect(self.annotation_controller.on_selection_cancel_requested)
        self.annotation_view.point_selected.connect(self._on_endview_point_selected)
        self.annotation_view.drag_update.connect(self._on_endview_drag_update)
        self.annotation_view.apply_roi_requested.connect(
            self.annotation_controller.on_apply_temp_mask_requested
        )
        self.annotation_view.previous_requested.connect(self._on_previous_slice)
        self.annotation_view.next_requested.connect(self._on_next_slice)

        if self.cscan_view is not None:
            self.cscan_view.crosshair_changed.connect(self._on_cscan_crosshair_changed)
            self.cscan_view.slice_requested.connect(self._on_cscan_slice_requested)
        self.ascan_view.position_changed.connect(self._on_ascan_position_changed)
        self.ascan_view.cursor_moved.connect(self._on_ascan_cursor_moved)
        if self.cscan_view_corrosion is not None:
            self.cscan_view_corrosion.crosshair_changed.connect(self._on_cscan_crosshair_changed)
            self.cscan_view_corrosion.slice_requested.connect(self._on_cscan_slice_requested)

        self.volume_view.volume_needs_update.connect(self._on_volume_needs_update)
        self.volume_view.camera_changed.connect(self._on_camera_changed)
        self.overlay_settings_view.label_visibility_changed.connect(
            self.annotation_controller.on_label_visibility_changed
        )
        self.overlay_settings_view.label_color_changed.connect(
            self.annotation_controller.on_label_color_changed
        )
        self.overlay_settings_view.overlay_opacity_changed.connect(
            self.annotation_controller.on_overlay_opacity_changed
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
        self.corrosion_settings_view.label_a_changed.connect(self._on_corrosion_label_a_changed)
        self.corrosion_settings_view.label_b_changed.connect(self._on_corrosion_label_b_changed)

    def _register_shortcuts(self) -> None:
        """Global keyboard shortcuts (active anywhere in the window)."""
        parent = self.main_window
        mapping = [
            (QKeySequence(Qt.Key.Key_A), self._on_previous_slice),
            (QKeySequence(Qt.Key.Key_D), self._on_next_slice),
            (QKeySequence(Qt.Key.Key_W), self.annotation_controller.on_apply_temp_mask_requested),
            (QKeySequence(Qt.Key.Key_Escape), self.annotation_controller.on_selection_cancel_requested),
            (QKeySequence(Qt.Key.Key_Return), self.annotation_controller.on_apply_all_temp_masks_requested),
            (QKeySequence(Qt.Key.Key_Enter), self.annotation_controller.on_apply_all_temp_masks_requested),
        ]
        for seq, handler in mapping:
            sc = QShortcut(seq, parent)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(handler)
            self._shortcuts.append(sc)

    def _on_resize_endview(self) -> None:
        """Open a dialog to resize only the displayed endview viewport."""
        if self.annotation_view is None:
            return
        current_size = self.annotation_view.get_display_size()
        dialog = EndviewResizeDialog(current_size, parent=self.main_window)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            if dialog.wants_reset():
                self.annotation_view.reset_display_size()
            else:
                width, height = dialog.get_size()
                self.annotation_view.set_display_size(width, height)

    def _on_toggle_tools_panel(self, checked: bool) -> None:
        """Show/hide the tools dock when the menu action is toggled."""
        dock = getattr(self.ui, "dockWidget_2", None)
        if dock is None:
            return
        dock.setVisible(bool(checked))

    def _on_tools_panel_visibility_changed(self, visible: bool) -> None:
        """Keep the toggle action state in sync with the dock visibility."""
        action = getattr(self.ui, "actionToggle_tools_panel", None)
        if action is None:
            return
        action.blockSignals(True)
        action.setChecked(bool(visible))
        action.blockSignals(False)

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
            loaded_model = self.nde_loader.load(file_path)
            volume = loaded_model.get_active_volume()
            if volume is None or getattr(volume, "ndim", 0) != 3:
                raise ValueError("Le fichier NDE ne contient pas de volume 3D exploitable.")

            self._initialize_ascan_logger(file_path)

            self.nde_model = loaded_model
            num_slices = volume.shape[0]
            self.view_state_model.set_slice_bounds(0, num_slices - 1)
            self.view_state_model.set_slice(0)
            self.view_state_model.set_current_point(None)
            self.view_state_model.set_apply_volume_range(0, num_slices - 1, include_current=True)
            self.annotation_controller.reset_overlay_state(preserve_labels=True)
            self.annotation_model.initialize(volume.shape)
            self.temp_mask_model.clear()
            self.temp_mask_model.initialize(volume.shape)
            self.roi_model.clear()
            self.annotation_controller.sync_overlay_settings()
            self.cscan_controller.reset_corrosion()

            # Réinitialise les sessions pour éviter les incohérences de shape
            self.session_manager.reset_for_new_dataset(
                annotation_model=self.annotation_model,
                temp_mask_model=self.temp_mask_model,
                roi_model=self.roi_model,
                view_state_model=self.view_state_model,
            )

            self.tools_panel.set_slice_bounds(0, num_slices - 1)
            self.tools_panel.set_slice_value(0)
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

            self.status_message(f"NDE chargé: {file_path}")

        except Exception as exc:
            QMessageBox.critical(self.main_window, "Erreur NDE", str(exc))

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
            self.annotation_controller.sync_overlay_settings()
            self.annotation_controller.refresh_overlay()
            self._sync_tools_labels()
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

    def _on_save(self) -> None:
        """Handle saving current session or annotations."""
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
            self.status_message(f"Overlay sauvegardé: {saved_path}")
        except Exception as exc:
            QMessageBox.critical(self.main_window, "Erreur sauvegarde overlay", str(exc))

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

        self.status_message("Split flaw/noflaw en cours...", timeout_ms=2000)
        success, message = self.split_flaw_noflaw_service.split_endviews(
            nde_model=self.nde_model,
            annotation_model=self.annotation_model,
            nde_file=nde_path,
            output_root=output_root,
            filename_prefix=prefix,
            filename_suffix=suffix,
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
        self.nde_settings_view.show()
        self.nde_settings_view.raise_()
        self.nde_settings_view.activateWindow()

    def _on_open_corrosion_settings(self) -> None:
        """Open the corrosion settings dialog."""
        self._sync_corrosion_label_choices()
        self.corrosion_settings_view.show()
        self.corrosion_settings_view.raise_()
        self.corrosion_settings_view.activateWindow()

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

    def _on_endview_colormap_changed(self, name: str) -> None:
        lut = self._get_colormap_lut(name)
        self.view_state_model.set_endview_colormap(name)
        self.annotation_view.set_colormap(name, lut)
        self.volume_view.set_base_colormap(name, lut)

    def _on_cscan_colormap_changed(self, name: str) -> None:
        lut = self._get_colormap_lut(name)
        self.view_state_model.set_cscan_colormap(name)
        if self.cscan_view is not None:
            self.cscan_view.set_colormap(name, lut)

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
        self.tools_panel.set_slice_value(clamped)
        self.annotation_view.set_slice(clamped)
        self.annotation_controller.refresh_roi_overlay_for_slice(clamped)
        self.cscan_controller.highlight_slice(clamped)
        self.volume_view.set_slice_index(clamped, update_slider=True)
        self._update_ascan_trace()
        self._update_endview_label()

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

    def _on_goto_requested(self, slice_idx: int) -> None:
        """Handle explicit goto action from tools panel."""
        self._on_slice_changed(slice_idx)

    def _on_cross_toggled(self, enabled: bool) -> None:
        """Handle crosshair visibility toggle."""
        self.view_state_model.set_show_cross(enabled)
        self.annotation_view.set_cross_visible(enabled)
        self.cscan_controller.set_cross_visible(enabled)
        self.ascan_view.set_marker_visible(enabled)

    def _on_endview_point_selected(self, pos: Any) -> None:
        """Handle point selection for crosshair sync."""
        if not isinstance(pos, tuple) or len(pos) != 2:
            return
        x, y = int(pos[0]), int(pos[1])
        self.view_state_model.update_crosshair(x, y)
        self.tools_panel.set_position_label(x, y)
        self._update_ascan_trace(point=(x, y))

    def _on_endview_drag_update(self, pos: Any) -> None:
        """Handle drag updates during drawing (cursor label only)."""
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
        self.view_state_model.set_slice(slice_idx)
        clamped_slice = self.view_state_model.current_slice
        self.tools_panel.set_slice_value(clamped_slice)
        self.annotation_view.set_slice(clamped_slice)
        self.cscan_controller.highlight_slice(clamped_slice)
        self.cscan_controller.set_crosshair(clamped_slice, x)
        y_default = volume.shape[1] // 2
        current_y = self.view_state_model.current_point[1] if self.view_state_model.current_point else y_default
        self.view_state_model.update_crosshair(x, current_y)
        y = self.view_state_model.current_point[1] if self.view_state_model.current_point else y_default
        self._update_ascan_trace(point=(x, y))

    def _on_cscan_slice_requested(self, z: int) -> None:
        """Handle slice requests originating from the C-Scan view."""
        self._on_slice_changed(z)

    def _on_ascan_position_changed(self, profile_idx: int) -> None:
        """Handle A-Scan position changes."""
        selection = self.ascan_controller.map_profile_index_to_point(
            self.nde_model,
            profile_idx,
            self.view_state_model.current_point,
            self.view_state_model.current_slice,
        )
        if selection is None:
            return
        point, new_slice = selection
        if new_slice is not None:
            self._on_slice_changed(new_slice)
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
        if isinstance(view_params, dict):
            self.view_state_model.set_camera_state(view_params)

    def _on_label_added(self, label_id: int, color: Any) -> None:
        """Forward label addition then resync tool panel labels."""
        self.annotation_controller.on_label_added(label_id, color)
        self.view_state_model.set_active_label(label_id)
        self._sync_tools_labels(select_label_id=label_id)

    def _on_label_deleted(self, label_id: int) -> None:
        """Forward label deletion then resync tool panel labels."""
        self.annotation_controller.on_label_deleted(label_id)
        if self.view_state_model.active_label == int(label_id):
            self.view_state_model.set_active_label(None)
        self._sync_tools_labels(select_label_id=None)

    def _sync_tools_labels(self, select_label_id: Optional[int] = None) -> None:
        """Sync the label list in the tools panel with the annotation model."""
        palette = self.annotation_model.get_label_palette()
        labels = sorted(palette.keys()) if palette else []
        current = select_label_id if select_label_id is not None else self.view_state_model.active_label
        if current not in labels:
            current = None
        self.view_state_model.set_active_label(current)
        self.tools_panel.set_labels(labels, current=current)
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

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _refresh_views(self) -> None:
        """Push the current volume state into all views."""
        volume = self._current_volume()
        if volume is None:
            return
        self.view_state_model.set_slice_bounds(0, volume.shape[0] - 1)

        # Sélectionne l’indice de tranche courant dans les bornes valides
        slice_idx = self.view_state_model.clamp_slice(self.view_state_model.current_slice)

        # Met à jour l’Endview (pas de changement ici)
        self.annotation_view.set_volume(volume)
        self.annotation_view.set_slice(slice_idx)
        self.annotation_controller.ensure_restriction_rect(
            shape=(volume.shape[1], volume.shape[2])
        )

        # Met à jour la C‑scan (standard ou corrosion)
        self.cscan_controller.update_views(volume)

        # Récupère l'ordre des axes depuis le modèle, s'il existe
        axis_order = self.view_state_model.axis_order

        # Envoie le volume à la vue 3D en précisant l’ordre des axes
        self.volume_view.set_volume(volume, slice_idx=slice_idx, axis_order=axis_order)

        # Applique l’overlay après la (re)construction de la scène 3D
        self.annotation_controller.refresh_overlay(rebuild=False)

        # Reste des mises à jour
        self._update_ascan_trace()
        self.annotation_view.set_cross_visible(self.view_state_model.show_cross)
        self.cscan_controller.set_cross_visible(self.view_state_model.show_cross)
        self.ascan_view.set_marker_visible(self.view_state_model.show_cross)


    # ------------------------------------------------------------------ #
    # Session handling
    # ------------------------------------------------------------------ #
    def _open_session_dialog(self) -> None:
        """Affiche le gestionnaire de sessions et connecte les signaux."""
        if self._session_dialog is None:
            self._session_dialog = SessionManagerDialog(parent=self.main_window)
            self._session_dialog.session_selected.connect(self._on_session_selected)
            self._session_dialog.session_created.connect(self._on_session_created)
            self._session_dialog.session_deleted.connect(self._on_session_deleted)
        self._refresh_session_dialog()
        self._session_dialog.show()
        self._session_dialog.raise_()
        self._session_dialog.activateWindow()

    def _refresh_session_dialog(self) -> None:
        if self._session_dialog is None:
            return
        sessions = self.session_manager.list_sessions()
        self._session_dialog.set_sessions(sessions)

    def _on_session_created(self, name: str) -> None:
        self.session_manager.create_from_models(
            name=name,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
            set_active=True,
            save_active=True,
        )
        self._refresh_session_dialog()
        self._after_session_switch()

    def _on_session_selected(self, session_id: str) -> None:
        self.session_manager.switch_session(
            session_id,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
            save_current=True,
        )
        self._after_session_switch()

    def _on_session_deleted(self, session_id: str) -> None:
        self.session_manager.delete_session(
            session_id,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
        )
        self._refresh_session_dialog()
        self._after_session_switch()

    def _after_session_switch(self) -> None:
        """Synchronise l'état du modèle actif vers les vues."""
        self.tools_panel.set_overlay_checked(self.view_state_model.show_overlay)
        self.tools_panel.set_cross_checked(self.view_state_model.show_cross)
        self.annotation_view.set_cross_visible(self.view_state_model.show_cross)
        self.cscan_controller.set_cross_visible(self.view_state_model.show_cross)
        self.ascan_view.set_marker_visible(self.view_state_model.show_cross)

        # Colormaps
        self.annotation_view.set_colormap(self.view_state_model.endview_colormap, None)
        self.volume_view.set_base_colormap(self.view_state_model.endview_colormap, None)
        if self.cscan_view is not None:
            self.cscan_view.set_colormap(self.view_state_model.cscan_colormap, None)

        self._sync_tools_labels()
        self.annotation_controller.sync_overlay_settings()
        self.annotation_controller.apply_overlay_opacity()
        # Rafraîchir le volume puis réappliquer l'overlay pour forcer le push 3D
        self._refresh_views()

    # ------------------------------------------------------------------ #
    # Corrosion completion handling
    # ------------------------------------------------------------------ #
    def _on_corrosion_completed(self, result) -> None:
        """Crée une session interpolée sans écraser l'état brut."""
        # 1) Met à jour la session active avec le résultat brut (projection déjà active)
        self.view_state_model.corrosion_interpolated_projection = None
        self.view_state_model.corrosion_overlay_volume = result.overlay_volume
        self.view_state_model.corrosion_overlay_palette = result.overlay_palette
        self.view_state_model.corrosion_overlay_label_ids = result.overlay_label_ids

        if self.session_manager._active_id is not None:  # noqa: SLF001
            raw_state = self.session_manager._snapshot(  # noqa: SLF001
                name=self.session_manager._sessions[self.session_manager._active_id].name,  # noqa: SLF001
                annotation_model=self.annotation_model,
                temp_mask_model=self.temp_mask_model,
                roi_model=self.roi_model,
                view_state_model=self.view_state_model,
            )
            self.session_manager._sessions[self.session_manager._active_id] = raw_state  # noqa: SLF001

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

        name = "Corrosion (interpolé)" if self.nde_model is None else f"Corrosion (interpolé) - {self.nde_model.metadata.get('path', 'NDE')}"

        # 3) Crée la nouvelle session sans sauvegarder l'active (déjà mise à jour), puis bascule dessus
        self.session_manager.create_from_models(
            name=name,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=interpolated_view_state,
            set_active=True,
            save_active=False,
        )

        self._after_session_switch()
        self.annotation_controller.refresh_overlay(defer_volume=False, rebuild=True)
        if (result.piece_volume_raw is not None and result.piece_volume_raw.size > 0) or (
            result.piece_volume_interpolated is not None and result.piece_volume_interpolated.size > 0
        ):
            self._piece_anchor = result.piece_anchor
            self._show_piece3d_volume(
                raw_volume=result.piece_volume_raw,
                interpolated_volume=result.piece_volume_interpolated,
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
        self._piece_toggle_btn = QPushButton("Montrer volume interpolé", parent=self._piece3d_window)
        self._piece_toggle_btn.clicked.connect(self._toggle_piece_volume)
        layout.addWidget(self._piece3d_view)
        layout.addWidget(self._piece_toggle_btn)

    def _show_piece3d_window(self) -> None:
        """Open the piece3D window using the latest available volume."""
        self._ensure_piece3d_window()
        if self._piece3d_view is None or self._piece3d_window is None:
            return

        volume = None
        if self._piece_show_interpolated and self._piece_volume_interpolated is not None:
            volume = self._piece_volume_interpolated
        elif (not self._piece_show_interpolated) and self._piece_volume_raw is not None:
            volume = self._piece_volume_raw
        elif self._piece_volume_interpolated is not None:
            self._piece_show_interpolated = True
            volume = self._piece_volume_interpolated
        elif self._piece_volume_raw is not None:
            self._piece_show_interpolated = False
            volume = self._piece_volume_raw

        if volume is not None:
            self._piece3d_view.set_anchor_point(self._piece_anchor)
            self._piece3d_view.set_piece_volume(volume)
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
    ) -> None:
        """Affiche ou met à jour la fenêtre flottante de la pièce 3D corrosion."""
        self._ensure_piece3d_window()
        if self._piece3d_view is None:
            return

        self._piece_volume_raw = raw_volume.astype(np.float32) if raw_volume is not None else None
        self._piece_volume_interpolated = (
            interpolated_volume.astype(np.float32) if interpolated_volume is not None else None
        )

        # Choix initial : interpolé si disponible, sinon brut
        if self._piece_volume_interpolated is not None:
            self._piece_show_interpolated = True
            current = self._piece_volume_interpolated
        else:
            self._piece_show_interpolated = False
            current = self._piece_volume_raw

        if current is not None:
            self._piece3d_view.set_anchor_point(self._piece_anchor)
            self._piece3d_view.set_piece_volume(current)
            self._update_piece_toggle_label()
            self._show_piece3d_window()

    def _toggle_piece_volume(self) -> None:
        """Bascule entre volume brut et volume interpolé si les deux sont disponibles."""
        if self._piece3d_view is None:
            return

        has_interp = self._piece_volume_interpolated is not None and self._piece_volume_interpolated.size > 0
        has_raw = self._piece_volume_raw is not None and self._piece_volume_raw.size > 0
        if not (has_interp and has_raw):
            return

        self._piece_show_interpolated = not self._piece_show_interpolated
        volume = self._piece_volume_interpolated if self._piece_show_interpolated else self._piece_volume_raw
        if volume is not None:
            self._piece3d_view.set_anchor_point(self._piece_anchor)
            self._piece3d_view.set_piece_volume(volume)
            self._update_piece_toggle_label()

    def _update_piece_toggle_label(self) -> None:
        """Met à jour le texte du bouton selon le volume affiché."""
        if self._piece_toggle_btn is None:
            return
        if self._piece_show_interpolated:
            self._piece_toggle_btn.setText("Afficher volume brut BW/FW")
        else:
            self._piece_toggle_btn.setText("Afficher volume interpolé BW/FW")

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

