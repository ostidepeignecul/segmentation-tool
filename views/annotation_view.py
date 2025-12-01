"""Annotation-specific view extending EndviewView with ROI/drawing hooks."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from views.endview_view import EndviewView


class AnnotationView(EndviewView):
    """Extends the base endview renderer with placeholders for ROI rendering."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._temp_polygon: list[Tuple[int, int]] = []
        self._temp_rectangle: Optional[Tuple[int, int, int, int]] = None
        self._roi_overlay: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # Temporary shapes (stubs)
    # ------------------------------------------------------------------ #
    def set_temp_polygon(self, points: Sequence[Tuple[int, int]]) -> None:
        """Placeholder to display a polygon in progress."""
        self._temp_polygon = [(int(x), int(y)) for x, y in points]

    def set_temp_rectangle(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        """Placeholder to display a rectangle in progress."""
        self._temp_rectangle = rect if rect is None else tuple(int(v) for v in rect)

    def clear_temp_shapes(self) -> None:
        """Clear any temporary polygon/rectangle."""
        self._temp_polygon = []
        self._temp_rectangle = None

    # ------------------------------------------------------------------ #
    # ROI overlay (stub)
    # ------------------------------------------------------------------ #
    def set_roi_overlay(self, roi_mask: Any) -> None:
        """Placeholder to display a ROI mask overlay."""
        self._roi_overlay = roi_mask

    def clear_roi_overlay(self) -> None:
        """Remove any ROI overlay."""
        self._roi_overlay = None
