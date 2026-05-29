"""Non-blocking popup that displays the current AI pipeline state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class PipelineStatusDialog(QDialog):
    """Display the current stage of the AI inference pipeline."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Etat du pipeline IA")
        self.setModal(False)
        self.setMinimumWidth(420)

        self._stage_order: list[str] = []
        self._stage_labels: dict[str, str] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._summary_label = QLabel("Inference inactive.", self)
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._steps_list = QListWidget(self)
        self._steps_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._steps_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        layout.addWidget(self._steps_list)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._detail_label = QLabel("", self)
        self._detail_label.setWordWrap(True)
        layout.addWidget(self._detail_label)

    def configure_stages(self, stages: Sequence[tuple[str, str]]) -> None:
        """Reset the visible pipeline steps."""
        self._stage_order = [str(stage_key) for stage_key, _ in stages]
        self._stage_labels = {str(stage_key): str(label) for stage_key, label in stages}

        self._steps_list.clear()
        for _, label in stages:
            item = QListWidgetItem(f"[ ] {label}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._steps_list.addItem(item)

    def start(self, message: str) -> None:
        """Show the popup and reset it to a running state."""
        self._summary_label.setText(str(message or "Inference in progress..."))
        self._detail_label.setText("Pipeline running...")
        self._progress_bar.setRange(0, 0)
        self._mark_stage(None)
        self.show()
        self.raise_()
        self.activateWindow()

    def set_status(
        self,
        stage_key: str,
        message: str,
        *,
        current: Optional[int] = None,
        total: Optional[int] = None,
        eta_seconds: Optional[float] = None,
    ) -> None:
        """Update the popup with the latest pipeline stage and progress."""
        self._summary_label.setText(str(message or "Inference in progress..."))
        self._mark_stage(stage_key)

        if current is not None and total is not None and int(total) > 0:
            total_value = max(int(total), 1)
            current_value = max(0, min(int(current), total_value))
            self._progress_bar.setRange(0, total_value)
            self._progress_bar.setValue(current_value)
            detail = f"{current_value}/{total_value}"
            if eta_seconds is not None and float(eta_seconds) >= 0:
                detail = f"{detail} | ETA {float(eta_seconds):.1f}s"
            self._detail_label.setText(detail)
            return

        self._progress_bar.setRange(0, 0)
        self._detail_label.setText(self._stage_labels.get(stage_key, "Pipeline running..."))

    def finish(self, message: str) -> None:
        """Display a completed state before the controller closes the popup."""
        self._summary_label.setText(str(message or "Inference completed."))
        self._detail_label.setText("Pipeline completed.")
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(1)
        self._mark_stage("__all__")

    def fail(self, message: str) -> None:
        """Display a failed state before the controller closes the popup."""
        self._summary_label.setText(str(message or "Inference failed."))
        self._detail_label.setText("Pipeline interrupted.")
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)

    def _mark_stage(self, active_stage_key: Optional[str]) -> None:
        if active_stage_key == "__all__":
            active_index = len(self._stage_order)
        elif active_stage_key in self._stage_labels:
            active_index = self._stage_order.index(str(active_stage_key))
        else:
            active_index = None

        for index, stage_key in enumerate(self._stage_order):
            prefix = "[ ]"
            if active_index is not None:
                if active_stage_key == "__all__" or index < active_index:
                    prefix = "[x]"
                elif index == active_index:
                    prefix = "[>]"

            item = self._steps_list.item(index)
            if item is not None:
                item.setText(f"{prefix} {self._stage_labels.get(stage_key, stage_key)}")
