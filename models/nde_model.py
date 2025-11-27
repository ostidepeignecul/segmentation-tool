from typing import Any, Dict, Optional, Tuple


class NDEModel:
    """Stores oriented NDE data (volume, A-scan, metadata) without UI logic."""

    def __init__(self) -> None:
        self.volume: Optional[Any] = None
        self.normalized_volume: Optional[Any] = None
        self.a_scan: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.current_slice: Optional[int] = None
        self._axis_map: Dict[str, int] = {}
        self._axis_map_shape: Optional[Tuple[int, ...]] = None

    def set_volume(
        self,
        volume: Any,
        metadata: Optional[Dict[str, Any]] = None,
        normalized_volume: Optional[Any] = None,
    ) -> None:
        """Assign the oriented volume, optional normalized copy, and optional metadata."""
        self.volume = volume
        self.normalized_volume = normalized_volume
        if metadata:
            self.metadata = dict(metadata)
        self._reset_axis_cache()

    def set_a_scan(self, a_scan: Any) -> None:
        """Assign the raw A-scan data."""
        self.a_scan = a_scan

    def set_current_slice(self, index: int) -> None:
        """Update the currently selected slice index."""
        target_index = int(index)
        active_volume = self.get_active_volume()
        if active_volume is not None and hasattr(active_volume, "shape") and len(active_volume.shape) >= 1:
            max_idx = max(0, active_volume.shape[0] - 1)
            target_index = max(0, min(max_idx, target_index))
        self.current_slice = target_index

    def set_normalized_volume(self, normalized_volume: Any) -> None:
        """Assign the normalized volume representation (0-1)."""
        self.normalized_volume = normalized_volume
        self._reset_axis_cache()

    def get_active_volume(self) -> Optional[Any]:
        """Return the preferred volume (normalized when available)."""
        if self.normalized_volume is not None:
            return self.normalized_volume
        return self.volume

    def get_axis_map(self) -> Dict[str, int]:
        """Infer how metadata axes map onto the current volume shape."""
        volume = self.get_active_volume()
        if volume is None or not hasattr(volume, "shape"):
            return {}

        shape: Tuple[int, ...] = volume.shape  # type: ignore[assignment]
        if self._axis_map and self._axis_map_shape == shape:
            return self._axis_map

        positions = self.metadata.get("positions") or {}
        axis_map: Dict[str, int] = {}
        used_axes: set[int] = set()
        for axis_name, coords in positions.items():
            qty = len(coords)
            idx = next(
                (
                    axis_idx
                    for axis_idx, dim in enumerate(shape)
                    if dim == qty and axis_idx not in used_axes
                ),
                None,
            )
            if idx is not None:
                axis_map[axis_name] = idx
                used_axes.add(idx)

        self._axis_map = axis_map
        self._axis_map_shape = shape
        return axis_map

    def get_axis_positions(self, axis_name: str) -> Optional[Any]:
        """Return the physical positions array for a given axis if available."""
        positions = self.metadata.get("positions") or {}
        return positions.get(axis_name)

    def get_trace(
        self,
        *,
        slice_idx: Optional[int] = None,
        x: Optional[int] = None,
    ) -> Optional[Any]:
        """Return a 1D trace (depth axis) for the requested slice/x coordinate."""
        volume = self.get_active_volume()
        if volume is None:
            return None

        num_slices, height, width = volume.shape
        if slice_idx is None:
            slice_idx = self.current_slice if self.current_slice is not None else 0
        slice_idx = max(0, min(num_slices - 1, int(slice_idx)))

        if x is None:
            x = width // 2
        x = max(0, min(width - 1, int(x)))

        trace = volume[slice_idx, :, x]
        return trace.copy() if hasattr(trace, "copy") else trace

    def clear(self) -> None:
        """Reset stored NDE data."""
        self.volume = None
        self.normalized_volume = None
        self.a_scan = None
        self.metadata = {}
        self.current_slice = None
        self._reset_axis_cache()

    def _reset_axis_cache(self) -> None:
        self._axis_map = {}
        self._axis_map_shape = None
