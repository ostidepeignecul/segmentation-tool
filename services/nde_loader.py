"""Minimal loader for NDE Open File Format files.

This loader implements a minimal subset of the NDE Open File Format
specification.  It is designed to extract a single three‑dimensional dataset
from the ``Public`` or ``Domain`` section of a .nde file, compute basic
metadata and physical coordinates, and return a :class:`~nde_model.NdeModel`
instance containing the data.  The loader follows the official
documentation guidelines:

* Each group in the JSON ``Setup`` metadata may contain a ``datasets`` array
  describing the available datasets with their ``id``, ``dataClass``, ``path``
  and ``dimensions``【848655542718855†L150-L160】.
* The ``dimensions`` array lists dimension objects in the same order as the
  HDF5 dataset dimensions【848655542718855†L258-L268】.  Each object may include an
  ``axis`` name (e.g. ``UCoordinate``, ``VCoordinate`` or ``Ultrasound``), an
  ``offset``, a ``quantity`` and a ``resolution``【848655542718855†L256-L268】.
* The ``dataValue`` object provides the minimum and maximum amplitude
  values【848655542718855†L185-L196】; these values are used for normalization.

This simplified loader selects the first dataset with ``dataClass`` equal
to ``AScanAmplitude`` when multiple datasets are present.  If no such
dataset exists, it falls back to the first available dataset.  It computes
physical positions along each axis based on the dimension definitions and
builds a minimal metadata dictionary for the model.

Limitations:
    - Only the first group (``group_idx=1``) is supported.
    - Only the first dataset is extracted.
    - Only three axes are expected (e.g. ``UCoordinate``, ``VCoordinate`` and
      ``Ultrasound``).  Additional axes will receive generic names.
    - No orientation or visualization heuristics are applied; the data is
      returned exactly as stored in the file.

Despite its simplicity, this loader demonstrates how to navigate the JSON
metadata and HDF5 structure defined by the NDE specification and should
serve as a clear starting point for more advanced implementations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from models.nde_model import NdeModel

logger = logging.getLogger(__name__)


class NdeLoader:
    """Simple reader for .nde files.

    This loader opens a .nde file using :mod:`h5py`, identifies whether it
    contains the ``Public`` or ``Domain`` layout, and extracts a single
    dataset (preferably an ``AScanAmplitude`` dataset) along with its
    metadata.  The result is wrapped into a :class:`~nde_model.NdeModel`.
    """

    def load(self, nde_file: str, group_idx: int = 1) -> NdeModel:
        """Load a .nde file and return a model containing one dataset.

        Parameters
        ----------
        nde_file:
            Path to the .nde file on disk.
        group_idx:
            1‑based index of the group to load (only 1 is supported in this
            minimal implementation).

        Returns
        -------
        NdeModel
            A model containing the extracted volume and metadata.
        """
        with h5py.File(nde_file, "r") as handle:
            root_keys = set(handle.keys())
            if "Public" in root_keys:
                return self._load_public(handle, group_idx)
            if "Domain" in root_keys:
                return self._load_domain(handle, group_idx)
            raise ValueError(f"Unknown NDE structure: {root_keys}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_public(self, handle: h5py.File, group_idx: int) -> NdeModel:
        """Load a dataset from the ``Public`` structure."""
        # Read and decode the JSON Setup metadata
        json_str = handle["Public/Setup"][()]
        json_decoded = json.loads(json_str)
        groups = json_decoded.get("groups") or []
        if not groups or group_idx - 1 >= len(groups):
            raise ValueError(f"Group index {group_idx} out of range (found {len(groups)} groups)")
        group_info: Dict[str, Any] = groups[group_idx - 1]

        dataset_entry = self._select_dataset_entry(group_info.get("datasets", []))
        if dataset_entry is None:
            raise ValueError("No datasets available in the selected group")

        path: str = dataset_entry.get("path", "").lstrip("/")
        if not path:
            raise ValueError("Dataset entry is missing its 'path' field")
        # Ensure the path exists in the file; try with and without leading slash
        if path in handle:
            data = handle[path][:]
        elif f"/{path}" in handle:
            data = handle[f"/{path}"][:]
        else:
            raise ValueError(f"Dataset path not found: {path}")

        # Compute axis order and physical positions
        dimensions: List[Dict[str, Any]] = dataset_entry.get("dimensions") or []
        axis_order, positions = self._compute_axes(dimensions, data.shape)
        self._log_dimensions("public", dimensions, data.shape)
        data, axis_order, positions = self._reorder_axes_by_metadata(
            data,
            axis_order,
            positions,
            dimensions=dimensions,
            setup_json=json_decoded,
        )
        data, axis_order, positions = self._orient_if_missing_metadata(
            data,
            axis_order,
            positions,
            structure="public",
        )
        data, axis_order, positions = self._maybe_rotate(
            data,
            axis_order,
            positions,
            structure="public",
            setup_json=json_decoded,
        )

        # Retrieve min/max values from dataValue if present
        data_value: Dict[str, Any] = dataset_entry.get("dataValue") or {}
        min_value = data_value.get("min")
        max_value = data_value.get("max")
        try:
            min_val = float(min_value) if min_value is not None else None
            max_val = float(max_value) if max_value is not None else None
        except (TypeError, ValueError):
            min_val = None
            max_val = None
        if min_val is None or max_val is None:
            # Fall back to data min/max
            min_val = float(np.min(data))
            max_val = float(np.max(data))

        metadata = {
            "structure": "public",
            "group_idx": group_idx,
            "path": path,
            "axis_order": axis_order,
            "positions": positions,
            "min_value": min_val,
            "max_value": max_val,
            "data_class": dataset_entry.get("dataClass"),
        }

        model = NdeModel()
        model.set_volume(data, metadata)
        return model

    def _load_domain(self, handle: h5py.File, group_idx: int) -> NdeModel:
        """Load a dataset from the ``Domain`` structure.

        This implementation attempts to read the first dataset path listed in
        the JSON metadata.  If no path is provided, it falls back to the
        legacy HDF5 location ``Domain/DataGroups/{i}/Datasets/0/Amplitude``.
        """
        json_str = handle["Domain/Setup"][()]
        json_decoded = json.loads(json_str)
        groups = json_decoded.get("groups") or []
        if not groups or group_idx - 1 >= len(groups):
            raise ValueError(f"Group index {group_idx} out of range (found {len(groups)} groups)")
        group_info: Dict[str, Any] = groups[group_idx - 1]

        dataset_entry = self._select_dataset_entry(group_info.get("datasets", []))
        # Determine dataset path from entry or fall back to legacy path
        if dataset_entry and dataset_entry.get("path"):
            path = dataset_entry["path"].lstrip("/")
        else:
            path = f"Domain/DataGroups/{group_idx - 1}/Datasets/0/Amplitude"
        # Ensure path exists
        if path in handle:
            data = handle[path][:]
        elif f"/{path}" in handle:
            data = handle[f"/{path}"][:]
        else:
            raise ValueError(f"Dataset path not found: {path}")

        dimensions: List[Dict[str, Any]] = []
        if dataset_entry:
            dimensions = dataset_entry.get("dimensions") or []
        axis_order, positions = self._compute_axes(dimensions, data.shape)
        self._log_dimensions("domain", dimensions, data.shape)
        data, axis_order, positions = self._reorder_axes_by_metadata(
            data,
            axis_order,
            positions,
            dimensions=dimensions,
            setup_json=json_decoded,
        )
        data, axis_order, positions = self._orient_if_missing_metadata(
            data,
            axis_order,
            positions,
            structure="domain",
        )
        data, axis_order, positions = self._maybe_rotate(
            data,
            axis_order,
            positions,
            structure="domain",
            setup_json=json_decoded,
        )

        # Determine min/max values
        data_value: Dict[str, Any] = dataset_entry.get("dataValue") if dataset_entry else {}
        min_value = None
        max_value = None
        if data_value:
            min_val = data_value.get("min")
            max_val = data_value.get("max")
            try:
                min_value = float(min_val) if min_val is not None else None
                max_value = float(max_val) if max_val is not None else None
            except (TypeError, ValueError):
                min_value, max_value = None, None
        if min_value is None or max_value is None:
            min_value = float(np.min(data))
            max_value = float(np.max(data))

        metadata = {
            "structure": "domain",
            "group_idx": group_idx,
            "path": path,
            "axis_order": axis_order,
            "positions": positions,
            "min_value": min_value,
            "max_value": max_value,
            "data_class": dataset_entry.get("dataClass") if dataset_entry else None,
        }

        model = NdeModel()
        model.set_volume(data, metadata)
        return model

    def _select_dataset_entry(self, entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the first dataset entry matching AScanAmplitude or return first."""
        for entry in entries:
            if entry.get("dataClass") == "AScanAmplitude":
                return entry
        return entries[0] if entries else None

    def _compute_axes(
        self, dimensions: List[Dict[str, Any]], shape: Tuple[int, ...]
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Compute axis order and positions from dimension definitions.

        Parameters
        ----------
        dimensions:
            List of dimension definition dictionaries from the JSON metadata.
        shape:
            Shape of the NumPy array loaded from the HDF5 dataset.

        Returns
        -------
        tuple
            A two‑tuple containing the list of axis names and a mapping of
            axis names to physical position arrays.
        """
        axis_order: List[str] = []
        positions: Dict[str, np.ndarray] = {}
        # If dimensions list is empty or its length mismatches the data shape, generate defaults
        dims = dimensions
        if not dims or len(dims) != len(shape):
            dims = []
            for idx, axis_len in enumerate(shape):
                dims.append(
                    {
                        "axis": f"axis_{idx}",
                        "offset": 0.0,
                        "resolution": 1.0,
                        "quantity": axis_len,
                    }
                )
        # Build positions and axis names
        for dim, axis_len in zip(dims, shape):
            axis_name: str = dim.get("axis") or f"axis_{len(axis_order)}"
            offset = dim.get("offset", 0.0)
            resolution = dim.get("resolution", 1.0)
            # The number of samples along this axis in the file
            qty = dim.get("quantity", axis_len)
            axis_order.append(axis_name)
            # Create position array with one entry per sample; ensure it matches axis_len
            coords = np.array(
                [offset + i * resolution for i in range(axis_len)], dtype=float
            )
            positions[axis_name] = coords
        return axis_order, positions

    def _orient_if_missing_metadata(
        self,
        data: np.ndarray,
        axis_order: List[str],
        positions: Dict[str, np.ndarray],
        *,
        structure: str,
    ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
        """
        When dimensions are missing, re-orient so that the slice axis is the largest one.

        Rationale: Domain files without metadata may store slices on a non-zero axis;
        endviews are expected to iterate over the largest dimension (number of slices),
        while keeping the presumed ultrasound axis as the last axis.
        """
        if axis_order and not all(name.startswith("axis_") for name in axis_order):
            # Metadata existed; keep original order.
            return data, axis_order, positions

        shape = data.shape
        if len(shape) < 3:
            return data, axis_order, positions

        slice_axis = int(np.argmax(shape))
        if slice_axis == 0:
            return data, axis_order, positions

        reordered_data = np.moveaxis(data, slice_axis, 0)
        reordered_axis_order = [axis_order[slice_axis], *axis_order[:slice_axis], *axis_order[slice_axis + 1 :]]

        reordered_positions: Dict[str, np.ndarray] = {}
        for name in reordered_axis_order:
            reordered_positions[name] = positions.get(name)

        logger.info(
            "[%s] fallback orientation: moved axis %s to slice axis | old shape=%s | new shape=%s",
            structure,
            slice_axis,
            shape,
            reordered_data.shape,
        )
        return reordered_data, reordered_axis_order, reordered_positions

    def _log_dimensions(self, source: str, dimensions: List[Dict[str, Any]], shape: Tuple[int, ...]) -> None:
        """Log axis names and quantities to help diagnose orientation issues."""
        if not logger.isEnabledFor(logging.INFO):
            return
        if not dimensions:
            logger.info("[%s] dimensions missing in metadata; data shape=%s", source, shape)
            return
        parts = []
        for idx, dim in enumerate(dimensions):
            name = dim.get("axis") or f"axis_{idx}"
            qty = dim.get("quantity")
            parts.append(f"{name}={qty} (shape[{idx}]={shape[idx] if idx < len(shape) else '?'})")
        logger.info("[%s] dimensions: %s", source, "; ".join(parts))

    def _reorder_axes_by_metadata(
        self,
        data: np.ndarray,
        axis_order: List[str],
        positions: Dict[str, np.ndarray],
        *,
        dimensions: List[Dict[str, Any]],
        setup_json: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
        """
        Reorder axes using Setup metadata:
        - Ultrasound reste l'axe Y (profondeur)
        - Axe slice choisi via discreteGrid.uCoordinateOrientation ou nom "Index", sinon plus grand entre U/V
        - L'autre axe spatial devient X
        """
        if data.ndim < 3 or not axis_order or len(axis_order) != data.ndim:
            return data, axis_order, positions

        orientation = None
        grid_dims: List[Dict[str, Any]] = []
        try:
            mappings = (setup_json or {}).get("dataMappings") or []
            if mappings:
                mapping = mappings[0]
                grid = mapping.get("discreteGrid") or {}
                orientation = grid.get("uCoordinateOrientation")
                grid_dims = grid.get("dimensions") or []
        except Exception:
            orientation = None
            grid_dims = []

        orientation_l = str(orientation).lower() if orientation is not None else None
        slice_axis_name: Optional[str] = None
        if orientation_l == "around":
            slice_axis_name = "UCoordinate"
        elif orientation_l == "length":
            slice_axis_name = "VCoordinate"

        if slice_axis_name is None:
            for dim in grid_dims:
                if str(dim.get("name", "")).lower() == "index" and dim.get("axis"):
                    slice_axis_name = dim.get("axis")
                    break

        if slice_axis_name is None:
            quantities = {}
            for dim in dimensions:
                axis_name = dim.get("axis")
                qty = dim.get("quantity")
                if axis_name is not None and qty is not None:
                    quantities[str(axis_name)] = qty
            candidates = {k: v for k, v in quantities.items() if str(k).lower() in {"ucoordinate", "vcoordinate"}}
            if candidates:
                slice_axis_name = max(candidates.items(), key=lambda item: item[1])[0]

        normalized = [str(name).lower() for name in axis_order]
        ultrasound_idx = normalized.index("ultrasound") if "ultrasound" in normalized else None
        slice_idx = None
        if slice_axis_name is not None and slice_axis_name.lower() in normalized:
            slice_idx = normalized.index(slice_axis_name.lower())

        desired_indices: List[int] = []
        if slice_idx is not None:
            desired_indices.append(slice_idx)
        if ultrasound_idx is not None and ultrasound_idx not in desired_indices:
            desired_indices.append(ultrasound_idx)
        for idx in range(len(axis_order)):
            if idx not in desired_indices:
                desired_indices.append(idx)

        if desired_indices == list(range(len(axis_order))):
            return data, axis_order, positions

        reordered = np.transpose(data, axes=desired_indices)
        new_axis_order = [axis_order[i] for i in desired_indices]

        new_positions: Dict[str, np.ndarray] = {}
        for name in new_axis_order:
            new_positions[name] = positions.get(name)

        logger.info(
            "Reordered axes from metadata | orientation=%s | slice_axis=%s | old_order=%s | new_order=%s | old_shape=%s | new_shape=%s",
            orientation,
            slice_axis_name,
            axis_order,
            new_axis_order,
            data.shape,
            reordered.shape,
        )
        return reordered, new_axis_order, new_positions

    def _rotate_clockwise(
        self,
        data: np.ndarray,
        axis_order: List[str],
        positions: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
        """
        Rotate each slice 90° clockwise in the (Y, X) plane.

        Rotation is applied as a display-ready orientation so views stay passive.
        Axis names/positions are swapped accordingly; the axis coming from the
        original X (third dim) becomes the new Y and is reversed, while the
        original Y becomes the new X.
        """
        if data.ndim < 3:
            return data, axis_order, positions

        rotated = np.rot90(data, k=-1, axes=(1, 2))  # (Z, H, W) -> (Z, W, H)

        if axis_order and len(axis_order) >= 3:
            a0, a1, a2, *rest = axis_order
            new_axis_order = [a0, a2, a1, *rest]
        else:
            new_axis_order = axis_order

        new_positions: Dict[str, np.ndarray] = {}
        for name, coords in positions.items():
            new_positions[name] = coords
        if axis_order and len(axis_order) >= 3:
            a0, a1, a2, *rest = axis_order
            # new axis1 comes from old axis2 but reversed by rot90
            if a2 in positions:
                new_positions[a2] = positions[a2][::-1]
            # new axis2 comes from old axis1 (orientation preserved)
            if a1 in positions:
                new_positions[a1] = positions[a1]

        return rotated, new_axis_order, new_positions

    def _maybe_rotate(
        self,
        data: np.ndarray,
        axis_order: List[str],
        positions: Dict[str, np.ndarray],
        *,
        structure: str,
        setup_json: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
        """Apply rot90 only for fallback or legacy Domain encodings without dataMappings."""
        if not self._should_apply_rotation(axis_order, setup_json, structure):
            return data, axis_order, positions
        rotated, new_order, new_positions = self._rotate_clockwise(data, axis_order, positions)
        logger.info(
            "[%s] applied fallback rotation 90° CW | old_shape=%s | new_shape=%s | old_order=%s | new_order=%s",
            structure,
            data.shape,
            rotated.shape,
            axis_order,
            new_order,
        )
        return rotated, new_order, new_positions

    def _should_apply_rotation(
        self,
        axis_order: List[str],
        setup_json: Optional[Dict[str, Any]],
        structure: str,
    ) -> bool:
        """Detect cases where we lacked explicit orientation and need the legacy rot90."""
        if axis_order and all(name.startswith("axis_") for name in axis_order):
            return True

        mappings = (setup_json or {}).get("dataMappings") or []
        encodings = (setup_json or {}).get("dataEncodings") or []

        if structure == "domain" and not mappings and encodings:
            # Legacy Domain files (3.x) expose discreteGrid via dataEncodings only.
            return True

        return False
