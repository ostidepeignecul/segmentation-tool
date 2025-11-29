"""Minimal NDE model for holding NDE inspection data.

This simple model stores a 3‑D volume extracted from a .nde file along with
basic metadata such as axis order and physical positions.  It also keeps a
normalized copy of the data (scaled to the range 0‑1) when min and max
values are provided.  The class exposes helper methods to retrieve the
active volume (normalized when available), to obtain the positions along a
specified axis, and to extract a one‑dimensional trace along any axis while
fixing the other axes to either their midpoint or a provided index.

The implementation follows the official NDE Open File Format documentation:

* Each dataset lists its dimensions as an array of objects.  The order
  of these dimension objects is the same as the order of the HDF5 dataset
  dimensions【848655542718855†L258-L268】.  Each dimension object may include an
  ``axis`` name (e.g. ``UCoordinate``, ``VCoordinate`` or ``Ultrasound``), an
  ``offset`` and a ``resolution``【848655542718855†L256-L268】.  These values are used
  here to compute physical positions along each axis.
* The metadata may also include a mapping called ``axis_order``, which is a
  list of axis names in the same order as the volume dimensions.  When
  provided, this mapping is used to look up axes in a case‑insensitive
  manner.  Otherwise, the positions dictionary is examined and the model
  attempts to match axes based on the length of each coordinate array.

This minimal model does not implement any UI logic.  It merely stores
volume data and exposes basic accessors required by higher‑level services.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


class NdeModel:
    """Lightweight container for NDE volume data and metadata.

    The model stores the raw 3‑D volume extracted from a .nde file, an
    optional normalized copy, and metadata describing the data.  It offers
    methods to retrieve the active volume (normalized when available),
    obtain physical positions along a given axis, and extract a 1‑D trace
    along any axis while fixing the remaining axes to either the midpoint
    or supplied indices.
    """

    def __init__(self) -> None:
        # Raw volume data (e.g. amplitude values)
        self.volume: Optional[np.ndarray] = None
        # Normalized volume in the range 0–1 when min/max values are known
        self.normalized_volume: Optional[np.ndarray] = None
        # Arbitrary metadata, including axis_order and positions
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Volume management
    # ------------------------------------------------------------------
    def set_volume(self, volume: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Assign the volume and associated metadata.

        Parameters
        ----------
        volume:
            A 3‑D NumPy array representing the inspection data.
        metadata:
            A dictionary containing at least ``axis_order`` (list of axis
            names in the same order as ``volume.shape``), ``positions``
            (mapping of axis names to physical coordinate arrays) and
            ``min_value``/``max_value`` for normalization.  Additional
            fields are preserved as‑is.
        """
        self.volume = volume
        self.metadata = dict(metadata)

        # Compute a normalized copy when possible
        min_val = self.metadata.get("min_value")
        max_val = self.metadata.get("max_value")
        try:
            if min_val is not None and max_val is not None and float(max_val) != float(min_val):
                norm = (volume.astype(np.float32) - float(min_val)) / (float(max_val) - float(min_val))
                self.normalized_volume = np.clip(norm, 0.0, 1.0)
            else:
                self.normalized_volume = None
        except Exception:
            # Fall back to no normalization if casting fails
            self.normalized_volume = None

    # ------------------------------------------------------------------
    # Volume accessors
    # ------------------------------------------------------------------
    def get_active_volume(self) -> Optional[np.ndarray]:
        """Return the normalized volume when available, else the raw volume."""
        return self.normalized_volume if self.normalized_volume is not None else self.volume

    # ------------------------------------------------------------------
    # Axis utilities
    # ------------------------------------------------------------------
    def get_axis_positions(self, axis_name: str) -> Optional[np.ndarray]:
        """Return the physical positions array for a given axis.

        The lookup is case‑insensitive: both the original axis name and its
        lower‑case version are checked against the ``positions`` dictionary in
        metadata.  If no matching key is found, ``None`` is returned.
        """
        positions = self.metadata.get("positions", {})
        if axis_name in positions:
            return positions[axis_name]
        lowered = axis_name.lower()
        for key, value in positions.items():
            if isinstance(key, str) and key.lower() == lowered:
                return value
        return None

    def get_trace_along_axis(
        self,
        axis_name: str,
        fixed_indices: Optional[Dict[str, int]] = None,
    ) -> Optional[np.ndarray]:
        """Extract a 1‑D trace along the requested axis.

        Parameters
        ----------
        axis_name:
            Name of the axis along which to extract the trace.  Names are
            compared in a case‑insensitive manner using the ``axis_order``
            metadata field.
        fixed_indices:
            Optional mapping of axis names to index values.  For axes not
            listed in this mapping, the midpoint of the corresponding axis
            length is used.

        Returns
        -------
        A NumPy array representing the 1‑D trace, or ``None`` if the axis
        cannot be identified or the volume is not loaded.
        """
        volume = self.get_active_volume()
        if volume is None:
            return None
        axis_order: Sequence[str] = self.metadata.get("axis_order", [])
        # Determine the index of the target axis in a case‑insensitive way
        target_idx: Optional[int] = None
        lower_name = axis_name.lower()
        for idx, name in enumerate(axis_order):
            if name == axis_name or name.lower() == lower_name:
                target_idx = idx
                break
        if target_idx is None:
            return None
        # Build slicing indices for each dimension
        indices: list[Any] = []
        fixed_normalized: Dict[str, int] = {
            (k.lower() if isinstance(k, str) else k): v
            for k, v in (fixed_indices or {}).items()
        }
        for idx, name in enumerate(axis_order):
            if idx == target_idx:
                indices.append(slice(None))
            else:
                axis_len = volume.shape[idx]
                # Default to middle index
                i = axis_len // 2
                # Override with fixed index if provided
                key = name.lower() if isinstance(name, str) else name
                if key in fixed_normalized:
                    val = fixed_normalized[key]
                    i = max(0, min(axis_len - 1, int(val)))
                indices.append(i)
        # Extract the trace
        trace = volume[tuple(indices)]
        return trace.copy() if hasattr(trace, "copy") else trace
