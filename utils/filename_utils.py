"""Helpers to sanitize auto-generated export filename components."""

from __future__ import annotations


WINDOWS_FORBIDDEN_FILENAME_CHARS = '<>:"/\\|?*'


def sanitize_filename_component(value: object, *, fallback: str = "") -> str:
    """Return a Windows-friendly filename fragment with spaces normalized to dashes."""
    cleaned = "".join(
        "_" if char in WINDOWS_FORBIDDEN_FILENAME_CHARS else char
        for char in str(value or "")
    )
    cleaned = "-".join(cleaned.split()).strip().strip(".")
    return cleaned or fallback
