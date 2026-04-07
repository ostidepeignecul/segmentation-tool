from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class ROI:
    """Represents a region of interest on a given slice."""

    id: int
    roi_type: str  # "box", "free_hand", "grow", "line", "peak"
    slice_idx: int
    points: List[Tuple[int, int]] = field(default_factory=list)
    label: int = 1
    threshold: Optional[float] = None
    persistent: bool = False
    erase_cleanup: bool = False


class RoiModel:
    """Stores ROIs independently from the overlay mask."""

    def __init__(self) -> None:
        self._rois: List[ROI] = []
        self._next_id: int = 1

    def clear(self) -> None:
        """Remove all ROIs."""
        self._rois.clear()
        self._next_id = 1

    def clear_non_persistent(self) -> None:
        """Remove only ROIs that are not marked persistent."""
        self._rois = [roi for roi in self._rois if roi.persistent]
        # Recompute next id to keep it monotonic and avoid collisions
        if self._rois:
            self._next_id = max(roi.id for roi in self._rois) + 1
        else:
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
        erase_cleanup: bool = False,
    ) -> ROI:
        roi = ROI(
            id=self._next_id,
            roi_type="box",
            slice_idx=int(slice_idx),
            points=[(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))],
            label=int(label),
            threshold=threshold,
            persistent=bool(persistent),
            erase_cleanup=bool(erase_cleanup),
        )
        self._next_id += 1
        self._rois.append(roi)
        return roi

    def add_grow(
        self,
        slice_idx: int,
        seed: Tuple[int, int],
        *,
        label: int = 1,
        threshold: Optional[float] = None,
        persistent: bool = False,
        erase_cleanup: bool = False,
    ) -> ROI:
        roi = ROI(
            id=self._next_id,
            roi_type="grow",
            slice_idx=int(slice_idx),
            points=[(int(seed[0]), int(seed[1]))],
            label=int(label),
            threshold=threshold,
            persistent=bool(persistent),
            erase_cleanup=bool(erase_cleanup),
        )
        self._next_id += 1
        self._rois.append(roi)
        return roi

    def add_line(
        self,
        slice_idx: int,
        points: Sequence[Tuple[int, int]],
        *,
        label: int = 1,
        threshold: Optional[float] = None,
        persistent: bool = False,
        erase_cleanup: bool = False,
    ) -> ROI:
        roi = ROI(
            id=self._next_id,
            roi_type="line",
            slice_idx=int(slice_idx),
            points=[(int(x), int(y)) for x, y in (points or [])],
            label=int(label),
            threshold=threshold,
            persistent=bool(persistent),
            erase_cleanup=bool(erase_cleanup),
        )
        self._next_id += 1
        self._rois.append(roi)
        return roi

    def add_free_hand(
        self,
        slice_idx: int,
        points: Sequence[Tuple[int, int]],
        *,
        label: int = 1,
        threshold: Optional[float] = None,
        persistent: bool = False,
        erase_cleanup: bool = False,
    ) -> ROI:
        roi = ROI(
            id=self._next_id,
            roi_type="free_hand",
            slice_idx=int(slice_idx),
            points=[(int(x), int(y)) for x, y in (points or [])],
            label=int(label),
            threshold=threshold,
            persistent=bool(persistent),
            erase_cleanup=bool(erase_cleanup),
        )
        self._next_id += 1
        self._rois.append(roi)
        return roi

    def add_peak(
        self,
        slice_idx: int,
        points: Sequence[Tuple[int, int]],
        *,
        label: int = 1,
        threshold: Optional[float] = None,
        persistent: bool = False,
        erase_cleanup: bool = False,
    ) -> ROI:
        roi = ROI(
            id=self._next_id,
            roi_type="peak",
            slice_idx=int(slice_idx),
            points=[(int(x), int(y)) for x, y in (points or [])],
            label=int(label),
            threshold=threshold,
            persistent=bool(persistent),
            erase_cleanup=bool(erase_cleanup),
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

    def remove_label(self, label_id: int) -> None:
        """Remove all ROIs associated with a given label."""
        lbl = int(label_id)
        self._rois = [r for r in self._rois if r.label != lbl]

    def boxes_for_slice(self, slice_idx: int, *, include_persistent: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        Return box coordinates (x1, y1, x2, y2) for a slice.
        If include_persistent=True, merge slice ROIs with persistent ones (no duplicates).
        """
        rois = self.list_on_slice(slice_idx)
        if include_persistent:
            seen_ids = {roi.id for roi in rois}
            for roi in self.list_persistent():
                if roi.id not in seen_ids:
                    rois.append(roi)
                    seen_ids.add(roi.id)

        rects: List[Tuple[int, int, int, int]] = []
        for roi in rois:
            if roi.roi_type == "box" and len(roi.points) >= 2:
                rects.append(roi.points[0] + roi.points[1])
        return rects

    def seeds_for_slice(self, slice_idx: int, *, include_persistent: bool = True) -> List[Tuple[int, int]]:
        """
        Return seed points (x, y) for grow ROIs on a slice.
        If include_persistent=True, merge slice ROIs with persistent ones (no duplicates).
        """
        rois = self.list_on_slice(slice_idx)
        if include_persistent:
            seen_ids = {roi.id for roi in rois}
            for roi in self.list_persistent():
                if roi.id not in seen_ids:
                    rois.append(roi)
                    seen_ids.add(roi.id)

        seeds: List[Tuple[int, int]] = []
        for roi in rois:
            if roi.roi_type == "grow" and len(roi.points) >= 1:
                seeds.append(roi.points[0])
        return seeds
