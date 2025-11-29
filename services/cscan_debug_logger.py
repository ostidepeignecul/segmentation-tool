#!/usr/bin/env python3
"""
Placeholder debug logger for C-Scan diagnostics.
"""

from __future__ import annotations

import time
from typing import Optional


class CScanDebugLogger:
    """Minimal singleton for future C-Scan logging."""

    _instance: Optional["CScanDebugLogger"] = None

    def __new__(cls) -> "CScanDebugLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._log_file = None
        self._start_time: Optional[float] = None
        self._session_active = False
        self._initialized = True

    def start_session(self, source: str) -> None:
        """Start a new logging session for a given NDE source path."""
        self.end_session()
        try:
            self._log_file = open("cscan_debug_log.txt", "w", encoding="utf-8")
            self._start_time = time.time()
            self._session_active = True
            self._write(f"=== C-SCAN DEBUG SESSION: {source} ===\n")
        except Exception:
            self._session_active = False
            self._log_file = None

    def ensure_session(self, source: str) -> None:
        """Start a session if none is active."""
        if not self._session_active:
            self.start_session(source)

    def end_session(self) -> None:
        """Close the current session."""
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
        self._log_file = None
        self._session_active = False
        self._start_time = None

    def _timestamp(self) -> str:
        if self._start_time is None:
            return "0.000s"
        return f"{(time.time() - self._start_time):.3f}s"

    def _write(self, text: str) -> None:
        if self._log_file and not self._log_file.closed:
            self._log_file.write(text)
            self._log_file.flush()


cscan_debug_logger = CScanDebugLogger()
