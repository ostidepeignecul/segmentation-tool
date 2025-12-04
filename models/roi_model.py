from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class ROI:
    """Represents a region of interest on a given slice."""

    id: int
    roi_type: str  # "box", "free_hand", "grow"
    slice_idx: int
    points: List[Tuple[int, int]] = field(default_factory=list)
    label: int = 1
    threshold: Optional[float] = None
    persistent: bool = False


class RoiModel:
    """Stores ROIs independently from the overlay mask."""

    def __init__(self) -> None:
        self._rois: List[ROI] = []
        self._next_id: int = 1

    def clear(self) -> None:
        """Remove all ROIs."""
        self._rois.clear()
        self._next_id = 1

    def add_box(
        self,
        slice_idx: int,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        *,
        label: int = 1,
        threshold: Optional[float] = None,
        persistent: bool = False,
    ) -> ROI:
        roi = ROI(
            id=self._next_id,
            roi_type="box",
            slice_idx=int(slice_idx),
            points=[(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))],
            label=int(label),
            threshold=threshold,
            persistent=bool(persistent),
        )
        self._next_id += 1
        self._rois.append(roi)
        return roi

    def list(self) -> List[ROI]:
        """Return all ROIs."""
        return list(self._rois)

    def list_on_slice(self, slice_idx: int) -> List[ROI]:
        """Return ROIs for a specific slice."""
        idx = int(slice_idx)
        return [r for r in self._rois if r.slice_idx == idx]

    def list_persistent(self) -> List[ROI]:
        """Return ROIs marked persistent (any slice)."""
        return [r for r in self._rois if r.persistent]

    def remove_on_slice(self, slice_idx: int) -> None:
        """Remove all ROIs belonging to a slice."""
        idx = int(slice_idx)
        self._rois = [r for r in self._rois if r.slice_idx != idx]

    def boxes_for_slice(self, slice_idx: int, *, include_persistent: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        Return box coordinates (x1, y1, x2, y2) for a slice.
        If no ROI on the slice and include_persistent=True, fall back to persistent ROIs.
        """
        rois = self.list_on_slice(slice_idx)
        if not rois and include_persistent:
            rois = self.list_persistent()
        rects: List[Tuple[int, int, int, int]] = []
        for roi in rois:
            if roi.roi_type == "box" and len(roi.points) >= 2:
                rects.append(roi.points[0] + roi.points[1])
        return rects
