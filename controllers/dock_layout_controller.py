"""Dock layout controller using ADS for UI orchestration only."""

from __future__ import annotations

from typing import Optional

import PyQt6Ads as ads
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMainWindow, QStackedLayout, QWidget

from ui_ascan import Ui_DockWidget as Ui_AScanDockWidget
from ui_cscan import Ui_DockWidget as Ui_CScanDockWidget
from ui_toolspanel import Ui_DockWidget as Ui_ToolsPanelDockWidget
from ui_ucoordinate import Ui_DockWidget as Ui_UCoordinateDockWidget
from ui_vcoordinate import Ui_DockWidget as Ui_VCoordinateDockWidget
from ui_volume import Ui_DockWidget as Ui_VolumeDockWidget
from views.annotation_view import AnnotationView
from views.ascan_view import AScanView
from views.ascan_view_corrosion import AScanViewCorrosion
from views.cscan_view import CScanView
from views.cscan_view_corrosion import CscanViewCorrosion
from views.endview_view_corrosion import EndviewViewCorrosion
from views.tools_panel import ToolsPanel
from views.volume_view import VolumeView


class DockLayoutController:
    """Build and maintain ADS docks and related stacked view containers."""
    _DEFAULT_ROOT_SPLITTER_SIZES = [220, 420, 420, 520]
    _DEFAULT_ASCAN_CSCAN_SPLITTER_SIZES = [1, 1]

    def __init__(self, *, main_window: QMainWindow) -> None:
        self.main_window = main_window
        self.dock_manager = ads.CDockManager(self.main_window)

        self.ucoordinate_ui = Ui_UCoordinateDockWidget()
        self.vcoordinate_ui = Ui_VCoordinateDockWidget()
        self.cscan_ui = Ui_CScanDockWidget()
        self.ascan_ui = Ui_AScanDockWidget()
        self.volume_ui = Ui_VolumeDockWidget()
        self.tools_ui = Ui_ToolsPanelDockWidget()

        self.ucoordinate_dock = ads.CDockWidget("U-Coordinate")
        self.vcoordinate_dock = ads.CDockWidget("V-Coordinate")
        self.cscan_dock = ads.CDockWidget("C-Scan")
        self.ascan_dock = ads.CDockWidget("A-Scan")
        self.volume_dock = ads.CDockWidget("Volume")
        self.tools_dock = ads.CDockWidget("Tools")

        self.ucoordinate_ui.setupUi(self.ucoordinate_dock)
        self.vcoordinate_ui.setupUi(self.vcoordinate_dock)
        self.cscan_ui.setupUi(self.cscan_dock)
        self.ascan_ui.setupUi(self.ascan_dock)
        self.volume_ui.setupUi(self.volume_dock)
        self.tools_ui.setupUi(self.tools_dock)

        self.ucoordinate_view: AnnotationView = self.ucoordinate_ui.frame
        self.vcoordinate_view: AnnotationView = self.vcoordinate_ui.frame
        # Compatibility aliases used by existing controllers.
        self.annotation_view: AnnotationView = self.ucoordinate_view
        self.secondary_annotation_view: AnnotationView = self.vcoordinate_view
        self.cscan_view: CScanView = self.cscan_ui.frame
        self.ascan_view: AScanView = self.ascan_ui.frame
        self.volume_view: VolumeView = self.volume_ui.frame
        self.tools_panel: ToolsPanel = self.tools_ui.dockWidgetContents

        self.cscan_view_corrosion: Optional[CscanViewCorrosion] = None
        self.ascan_view_corrosion: Optional[AScanViewCorrosion] = None
        self.annotation_view_corrosion: Optional[EndviewViewCorrosion] = None
        self.secondary_annotation_view_corrosion: Optional[EndviewViewCorrosion] = None
        self.cscan_stack: Optional[QStackedLayout] = None
        self.ascan_stack: Optional[QStackedLayout] = None
        self.annotation_stack: Optional[QStackedLayout] = None
        self.secondary_annotation_stack: Optional[QStackedLayout] = None
        self._tools_toggle_action: Optional[QAction] = None

        self._configure_docks()
        self._build_default_layout()
        self._build_corrosion_stacks()
        self.tools_dock.viewToggled.connect(self._on_tools_view_toggled)
        self.volume_dock.topLevelChanged.connect(self._on_volume_top_level_changed)

    def bind_tools_toggle_action(self, action: QAction) -> None:
        """Bind menu action to ADS tools dock visibility."""
        if self._tools_toggle_action is action:
            return
        if self._tools_toggle_action is not None:
            try:
                self._tools_toggle_action.toggled.disconnect(self.set_tools_visible)
            except Exception:
                pass
        self._tools_toggle_action = action
        action.setCheckable(True)
        action.toggled.connect(self.set_tools_visible)
        action.setChecked(not self.tools_dock.isClosed())

    def set_tools_visible(self, visible: bool) -> None:
        """Show or hide the tools dock."""
        self.tools_dock.toggleView(bool(visible))

    def _configure_docks(self) -> None:
        self.ucoordinate_dock.setWindowTitle("U-Coordinate")
        self.vcoordinate_dock.setWindowTitle("V-Coordinate")
        self.cscan_dock.setWindowTitle("C-Scan")
        self.ascan_dock.setWindowTitle("A-Scan")
        self.volume_dock.setWindowTitle("Volume")
        self.tools_dock.setObjectName("dockWidget_2")
        self.tools_dock.setWindowTitle("Tools")
        self.tools_dock.setFeatures(ads.CDockWidget.DockWidgetFeature.DefaultDockWidgetFeatures)

    def _build_default_layout(self) -> None:
        """Default layout:
        tools | (U/V) | volume
              |      | (ascan | cscan)
        """
        tools_area = self.dock_manager.addDockWidget(
            ads.DockWidgetArea.LeftDockWidgetArea,
            self.tools_dock,
        )
        ucoordinate_area = self.dock_manager.addDockWidget(
            ads.DockWidgetArea.RightDockWidgetArea,
            self.ucoordinate_dock,
            tools_area,
        )
        vcoordinate_area = self.dock_manager.addDockWidget(
            ads.DockWidgetArea.RightDockWidgetArea,
            self.vcoordinate_dock,
            ucoordinate_area,
        )
        right_top_area = self.dock_manager.addDockWidget(
            ads.DockWidgetArea.RightDockWidgetArea,
            self.volume_dock,
            vcoordinate_area,
        )
        right_mid_area = self.dock_manager.addDockWidget(
            ads.DockWidgetArea.BottomDockWidgetArea,
            self.ascan_dock,
            right_top_area,
        )
        self.dock_manager.addDockWidget(
            ads.DockWidgetArea.RightDockWidgetArea,
            self.cscan_dock,
            right_mid_area,
        )
        self._apply_default_splitter_sizes()

    def _apply_default_splitter_sizes(self) -> None:
        """Tune default dock proportions for initial startup layout."""
        root_reference_area = self.ucoordinate_dock.dockAreaWidget()
        if root_reference_area is not None:
            # tools | U | V | (volume + ascan/cscan)
            self.dock_manager.setSplitterSizes(
                root_reference_area,
                self._DEFAULT_ROOT_SPLITTER_SIZES,
            )

        ascan_cscan_reference_area = self.cscan_dock.dockAreaWidget()
        if ascan_cscan_reference_area is not None:
            self.dock_manager.setSplitterSizes(
                ascan_cscan_reference_area,
                self._DEFAULT_ASCAN_CSCAN_SPLITTER_SIZES,
            )

    def _build_corrosion_stacks(self) -> None:
        self.annotation_stack, self.annotation_view_corrosion = self._build_annotation_stack()
        (
            self.secondary_annotation_stack,
            self.secondary_annotation_view_corrosion,
        ) = self._build_secondary_annotation_stack()
        self.cscan_stack, self.cscan_view_corrosion = self._build_cscan_stack()
        self.ascan_stack, self.ascan_view_corrosion = self._build_ascan_stack()

    def _build_annotation_stack(self) -> tuple[Optional[QStackedLayout], Optional[EndviewViewCorrosion]]:
        parent_widget = self.annotation_view.parentWidget()
        parent_layout = parent_widget.layout() if parent_widget is not None else None
        if parent_widget is None or parent_layout is None:
            return None, None

        container = QWidget(parent=parent_widget)
        stack = QStackedLayout(container)
        stack.setContentsMargins(0, 0, 0, 0)
        parent_layout.replaceWidget(self.annotation_view, container)
        self.annotation_view.setParent(container)
        stack.addWidget(self.annotation_view)

        corrosion_view = EndviewViewCorrosion(parent=container)
        stack.addWidget(corrosion_view)
        stack.setCurrentWidget(self.annotation_view)
        return stack, corrosion_view

    def _build_secondary_annotation_stack(self) -> tuple[Optional[QStackedLayout], Optional[EndviewViewCorrosion]]:
        parent_widget = self.secondary_annotation_view.parentWidget()
        parent_layout = parent_widget.layout() if parent_widget is not None else None
        if parent_widget is None or parent_layout is None:
            return None, None

        container = QWidget(parent=parent_widget)
        stack = QStackedLayout(container)
        stack.setContentsMargins(0, 0, 0, 0)
        parent_layout.replaceWidget(self.secondary_annotation_view, container)
        self.secondary_annotation_view.setParent(container)
        stack.addWidget(self.secondary_annotation_view)

        corrosion_view = EndviewViewCorrosion(parent=container)
        stack.addWidget(corrosion_view)
        stack.setCurrentWidget(self.secondary_annotation_view)
        return stack, corrosion_view

    def _build_cscan_stack(self) -> tuple[Optional[QStackedLayout], Optional[CscanViewCorrosion]]:
        parent_widget = self.cscan_view.parentWidget()
        parent_layout = parent_widget.layout() if parent_widget is not None else None
        if parent_widget is None or parent_layout is None:
            return None, None

        container = QWidget(parent=parent_widget)
        stack = QStackedLayout(container)
        stack.setContentsMargins(0, 0, 0, 0)
        parent_layout.replaceWidget(self.cscan_view, container)
        self.cscan_view.setParent(container)
        stack.addWidget(self.cscan_view)

        corrosion_view = CscanViewCorrosion(parent=container)
        stack.addWidget(corrosion_view)
        stack.setCurrentWidget(self.cscan_view)
        return stack, corrosion_view

    def _build_ascan_stack(self) -> tuple[Optional[QStackedLayout], Optional[AScanViewCorrosion]]:
        parent_widget = self.ascan_view.parentWidget()
        parent_layout = parent_widget.layout() if parent_widget is not None else None
        if parent_widget is None or parent_layout is None:
            return None, None

        container = QWidget(parent=parent_widget)
        stack = QStackedLayout(container)
        stack.setContentsMargins(0, 0, 0, 0)
        parent_layout.replaceWidget(self.ascan_view, container)
        self.ascan_view.setParent(container)
        stack.addWidget(self.ascan_view)

        corrosion_view = AScanViewCorrosion(parent=container)
        stack.addWidget(corrosion_view)
        stack.setCurrentWidget(self.ascan_view)
        return stack, corrosion_view

    def _on_tools_view_toggled(self, visible: bool) -> None:
        """Keep toggle action state in sync with dock visibility."""
        action = self._tools_toggle_action
        if action is None:
            return
        action.blockSignals(True)
        action.setChecked(bool(visible))
        action.blockSignals(False)

    def _on_volume_top_level_changed(self, _floating: bool) -> None:
        """Rebuild VolumeView GL scene after dock/undock transitions."""
        if self.volume_view is None:
            return
        self.volume_view.notify_dock_topology_changed()
