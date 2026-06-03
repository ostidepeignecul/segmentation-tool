from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class LoadingPopupView(QDialog):
    """Simple modal popup used to indicate a long-running task."""

    log_line_received = pyqtSignal(str)

    def __init__(
        self,
        *,
        title: str,
        message: str,
        parent: Optional[QWidget] = None,
        show_log: bool = False,
        max_log_lines: int = 12,
    ) -> None:
        super().__init__(parent)
        self._allow_close = False
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(480 if show_log else 340)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self._message_label = QLabel(message, self)
        self._message_label.setWordWrap(True)
        layout.addWidget(self._message_label)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        layout.addWidget(self._progress_bar)

        self._log_view = QPlainTextEdit(self)
        self._log_view.setReadOnly(True)
        self._log_view.setMinimumHeight(160)
        self._log_view.document().setMaximumBlockCount(max_log_lines)
        layout.addWidget(self._log_view)

        self.log_line_received.connect(self.append_log_line)
        self.set_log_visible(show_log)

    def set_message(self, message: str) -> None:
        self._message_label.setText(str(message or "").strip())

    def set_log_visible(self, visible: bool) -> None:
        self._log_view.setVisible(bool(visible))

    def append_log_line(self, line: str) -> None:
        text = str(line or "").strip()
        if not text:
            return
        self._log_view.appendPlainText(text)

    def finish(self) -> None:
        self._allow_close = True
        self.close()

    def reject(self) -> None:
        if self._allow_close:
            super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            super().closeEvent(event)
            return
        event.ignore()


class LoadingPopupLogHandler(logging.Handler):
    """Forward selected log records into a loading popup log area."""

    def __init__(
        self,
        popup: LoadingPopupView,
        *,
        allowed_logger_prefixes: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(level=logging.INFO)
        self._popup: Optional[LoadingPopupView] = popup
        self._allowed_prefixes = tuple(
            prefix.strip()
            for prefix in (allowed_logger_prefixes or ())
            if str(prefix or "").strip()
        )
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        popup = self._popup
        if popup is None or not self._should_forward(record.name):
            return
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        cleaned = _ANSI_ESCAPE_RE.sub("", str(message or "")).strip()
        if not cleaned:
            return
        if record.levelno >= logging.WARNING:
            cleaned = f"{record.levelname}: {cleaned}"
        try:
            popup.log_line_received.emit(cleaned)
        except RuntimeError:
            self._popup = None

    def close(self) -> None:
        self._popup = None
        super().close()

    def _should_forward(self, logger_name: str) -> bool:
        if not self._allowed_prefixes:
            return True
        normalized = str(logger_name or "").strip()
        for prefix in self._allowed_prefixes:
            if normalized == prefix or normalized.startswith(f"{prefix}."):
                return True
        return False
