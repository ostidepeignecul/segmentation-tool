#!/usr/bin/env python3
"""
Lightweight logger for A-Scan diagnostics.

Writes normalized and raw profile stats to ascan_debug_log.txt with a simple
session header per loaded file.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class AScanDebugLogger:
    """Singleton debug logger for A-Scan traces."""

    _instance: Optional["AScanDebugLogger"] = None

    def __new__(cls) -> "AScanDebugLogger":
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
            self._log_file = open("ascan_debug_log.txt", "w", encoding="utf-8")
            self._start_time = time.time()
            self._session_active = True
            self._write(f"=== A-SCAN DEBUG SESSION: {source} ===\n")
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

    def log_preview(
        self,
        slice_idx: int,
        point: tuple[int, int],
        normalized_stats: Optional[Dict[str, Any]],
        raw_stats: Optional[Dict[str, Any]],
    ) -> None:
        """Log normalized/raw profile stats."""
        if not self._session_active or self._log_file is None:
            return
        timestamp = self._timestamp()
        self._write(f"\n[{timestamp}] A-SCAN PREVIEW slice={slice_idx} x={point[0]} y={point[1]}\n")
        if normalized_stats is not None:
            self._write(f"  normalized: {normalized_stats}\n")
        else:
            self._write("  normalized: unavailable\n")
        if raw_stats is not None:
            self._write(f"  raw: {raw_stats}\n")
        else:
            self._write("  raw: unavailable\n")

    def _timestamp(self) -> str:
        if self._start_time is None:
            return "0.000s"
        return f"{(time.time() - self._start_time):.3f}s"

    def _write(self, text: str) -> None:
        if self._log_file and not self._log_file.closed:
            self._log_file.write(text)
            self._log_file.flush()


ascan_debug_logger = AScanDebugLogger()
