"""NDE Loader - Complete implementation using utils modules.

This loader implements full support for all NDE data types by leveraging
the specialized utilities in the utils package:
- extract_data_from_nde.py: Data extraction and type detection
- nde_versions_helper.py: Version-specific metadata access
- sectorial_nde.py: Sectorial scan processing

Supported data types:
- Zero-degree UT (standard ultrasonic inspection)
- Sectorial scans (S-scans with beam angles)
- TFM (Total Focusing Method)
- FMC (Full Matrix Capture)

The loader returns NdeModel instances compatible with the rest of the application.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from models.nde_model import NdeModel

# Import from utils - Version helpers
from utils.nde_versions_helper import (
    Unistatus_NDE_3_0_0,
    Unistatus_NDE_4_0_0_Dev,
    get_unistatus,
    isVersionHigherOrEqualThan,
)

# Import from utils - Sectorial scan processing
from utils.sectorial_nde import (
    SectorialScanNDE,
    SScanSliceInfo,
    s_scan_to_cartesian_image_extremes_fast,
    get_physical_radian_sscan,
    get_virtual_radian_sscan,
)

logger = logging.getLogger(__name__)

# Type alias for unistatus objects
Unistatus = Union[Unistatus_NDE_3_0_0, Unistatus_NDE_4_0_0_Dev]


class NDEDataTypeCheck:
    """Static methods for detecting NDE data types."""

    @staticmethod
    def is_sectorial_scan(setup: dict, group_index: int = 0) -> bool:
        """Check if the NDE file contains sectorial scan data.
        
        Args:
            setup: Decoded JSON setup from the NDE file.
            group_index: Index of the group to check.
            
        Returns:
            True if the group contains sectorial scan data.
        """
        if "groups" not in setup:
            return False
        if len(setup["groups"]) == 0:
            return False
        if group_index > len(setup["groups"]) - 1:
            raise ValueError(
                f"Group index {group_index} is out of bounds for the number of groups in setup"
            )
        if "processes" not in setup["groups"][group_index]:
            return False
            
        processes = setup["groups"][group_index]["processes"]
        if not processes:
            return False
            
        if "ultrasonicPhasedArray" not in processes[0]:
            return False
            
        ultrasound_phased_array = processes[0]["ultrasonicPhasedArray"]
        if "pulseEcho" not in ultrasound_phased_array:
            return False
            
        pulse_echo = ultrasound_phased_array["pulseEcho"]
        if "sectorialFormation" in pulse_echo:
            return True
        if "compoundFormation" in pulse_echo:
            return True
            
        return False

    @staticmethod
    def is_tfm(setup: dict, group_index: int = 0) -> bool:
        """Check if the NDE file contains TFM (Total Focusing Method) data.
        
        Args:
            setup: Decoded JSON setup from the NDE file.
            group_index: Index of the group to check.
            
        Returns:
            True if the group contains TFM data.
        """
        valid_datasets = {
            idx: group
            for idx, group in enumerate(setup.get("groups", []))
            if "datasets" in group
        }

        supported_classes = ["AScanAmplitude", "TfmValue"]
        group_classes = []

        for idx, group in valid_datasets.items():
            if len(group["datasets"]) >= 1:
                data_class = group["datasets"][0].get("dataClass")
            else:
                data_class = group["datasets"].get("dataClass")

            if data_class in supported_classes:
                group_classes.append(data_class)

        return "TfmValue" in group_classes

    @staticmethod
    def is_fmc(setup: dict, group_index: int = 0) -> bool:
        """Check if the NDE file contains FMC (Full Matrix Capture) data.
        
        Args:
            setup: Decoded JSON setup from the NDE file.
            group_index: Index of the group to check.
            
        Returns:
            True if the group contains FMC data.
        """
        valid_datasets = {
            idx: group
            for idx, group in enumerate(setup.get("groups", []))
            if "datasets" in group
        }

        for idx, group in valid_datasets.items():
            if "processes" not in group:
                continue
            for process in group["processes"]:
                if "ultrasonicMatrixCapture" not in process:
                    continue
                if "acquisitionPattern" not in process["ultrasonicMatrixCapture"]:
                    continue
                if "FMC" in process["ultrasonicMatrixCapture"]["acquisitionPattern"]:
                    logger.info("FMC dataset detected")
                    return True
        return False


class NDEGroupData:
    """Base class for extracting data from an NDE group.
    
    This class handles the common logic for loading data arrays and
    extracting status information from NDE files.
    """

    def __init__(
        self,
        f: h5py.File,
        unistatus: Unistatus,
        group_id: int,
    ):
        """Initialize the group data extractor.
        
        Args:
            f: Open HDF5 file handle.
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
        """
        self.set_data_array(f, unistatus, group_id)
        self.set_status_info(unistatus, group_id)

    def set_data_array(
        self,
        f: h5py.File,
        unistatus: Unistatus,
        group_id: int,
    ) -> None:
        """Load the data array from the HDF5 file.
        
        Args:
            f: Open HDF5 file handle.
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
        """
        path = unistatus.get_path_to_data(group_id)
        if path is None:
            raise ValueError(f"No data path found for group {group_id}")
        self.data_array = f[path][:]

    def set_status_info(
        self, 
        unistatus: Unistatus, 
        group_id: int
    ) -> dict:
        """Extract metadata and status information for the group.
        
        Args:
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
            
        Returns:
            Dictionary containing status information.
        """
        self.status_info = {}
        self.status_info["gr"] = group_id
        self.status_info["group_name"] = unistatus.get_group_name(group_id)
        self.status_info["min_value"] = unistatus.get_amplitude_min(group_id)
        self.status_info["max_value"] = unistatus.get_amplitude_max(group_id)
        
        # Get data shape
        (
            self.status_info["number_files_input"],
            self.status_info["img_height_px"],
            self.status_info["img_width_px"],
        ) = np.shape(self.data_array)
        
        # Get axis registration info
        self.status_info["lengthwise"] = unistatus.get_u_axis_reg_info(group_id)
        self.status_info["crosswise"] = unistatus.get_v_axis_reg_info(group_id)
        self.status_info["ultrasound"] = unistatus.get_w_axis_reg_info(group_id)

        # Convert seconds (round-trip) to meters for ultrasound infos
        ut_velocity = unistatus.get_ut_velocity()
        self.status_info["ultrasound"]["velocity"] = ut_velocity
        self.status_info["ultrasound"]["resolution"] = (
            self.status_info["ultrasound"]["resolution"] * ut_velocity / 2
        )
        try:
            self.status_info["ultrasound"]["offset"] = (
                self.status_info["ultrasound"]["offset"] * ut_velocity / 2
            )
        except Exception:
            self.status_info["ultrasound"]["offset"] = 0

        # Static mapping of position/index for each axis
        for axis in ["lengthwise", "crosswise", "ultrasound"]:
            positions_m = []
            if axis not in self.status_info:
                continue
            if "quantity" not in self.status_info[axis]:
                continue
            for idx in range(self.status_info[axis]["quantity"]):
                positions_m.append(
                    self.status_info[axis].get("offset", 0)
                    + idx * self.status_info[axis]["resolution"]
                )
            self.status_info[axis]["positions_m"] = np.array(positions_m)

        return self.status_info


class NDEGroupDataZeroDegUT(NDEGroupData):
    """Group data extractor for standard zero-degree UT data.
    
    This class inherits from NDEGroupData without modifications,
    as zero-degree UT data uses the standard extraction logic.
    """
    pass


class NDEGroupDataSectorialScan(NDEGroupData):
    """Group data extractor for sectorial scan (S-scan) data.
    
    This class handles the specialized processing required for
    sectorial scans, including cartesian reconstruction.
    """

    def __init__(
        self,
        f: h5py.File,
        unistatus: Unistatus,
        group_id: int,
        setup_json: dict,
    ):
        """Initialize the sectorial scan extractor.
        
        Args:
            f: Open HDF5 file handle.
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
            setup_json: Decoded JSON setup from the NDE file.
        """
        self.unistatus = unistatus
        self.group_id = group_id
        self.setup = setup_json
        self.slice_info_lookup: Dict[int, SScanSliceInfo] = {}
        self.file = f
        self.radian_sscan_data: Optional[dict] = None
        
        self.set_data_array(f, unistatus, group_id)
        self.set_status_info(unistatus, group_id)

    def get_data_array(
        self,
        f: h5py.File,
        unistatus: Unistatus,
        group_id: int,
    ) -> np.ndarray:
        """Load raw data array from HDF5 file.
        
        Args:
            f: Open HDF5 file handle.
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
            
        Returns:
            Raw data array.
        """
        path = unistatus.get_path_to_data(group_id)
        if path is None:
            raise ValueError(f"No data path found for group {group_id}")
        return np.array(f[path][:])

    def set_data_array(
        self,
        f: h5py.File,
        unistatus: Unistatus,
        group_id: int,
    ) -> None:
        """Process sectorial scan data with cartesian reconstruction.
        
        Args:
            f: Open HDF5 file handle.
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
        """
        import time
        
        group_data = self.get_data_array(f, unistatus, group_id)

        time1 = time.time()
        logger.info("Sectorial Scan Data Reconstruction Started")

        # Use SectorialScanNDE for parsing and reconstruction
        sectorial_scan_parser = SectorialScanNDE()
        parsed_setup = sectorial_scan_parser.parse_setup(self.setup)
        group_setup = parsed_setup[group_id]
        parsed_group = sectorial_scan_parser.parse_group(
            group_setup=group_setup, group_data=group_data
        )

        self.radian_sscan_data = parsed_group
        self.data_array, self.slice_info_lookup = (
            sectorial_scan_parser.create_cartesian_sscan_array(
                sscans=parsed_group["s_scan"], parsed_group=parsed_group
            )
        )
        
        logger.info(
            f"Sectorial Scan Data Reconstruction Finished in {time.time() - time1:.2f} seconds"
        )

    def set_status_info(
        self, 
        unistatus: Unistatus, 
        group_id: int
    ) -> dict:
        """Extract status info with sectorial-specific additions.
        
        Args:
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
            
        Returns:
            Dictionary containing status information.
        """
        # Call parent to get base status info
        self.status_info = {}
        self.status_info["gr"] = group_id
        self.status_info["group_name"] = unistatus.get_group_name(group_id)
        self.status_info["min_value"] = unistatus.get_amplitude_min(group_id)
        self.status_info["max_value"] = unistatus.get_amplitude_max(group_id)
        
        # Get data shape from reconstructed array
        if self.data_array is not None and len(self.data_array.shape) >= 3:
            (
                self.status_info["number_files_input"],
                self.status_info["img_height_px"],
                self.status_info["img_width_px"],
            ) = np.shape(self.data_array)
        
        # Get axis registration info
        self.status_info["lengthwise"] = unistatus.get_u_axis_reg_info(group_id)
        self.status_info["crosswise"] = unistatus.get_v_axis_reg_info(group_id)
        self.status_info["ultrasound"] = unistatus.get_w_axis_reg_info(group_id)

        # Add sectorial-specific information
        # The slice info uses the z (in 3D coordinates) axis as the index key
        if self.slice_info_lookup and 0 in self.slice_info_lookup:
            self.status_info["sscan_slice_info"] = self.slice_info_lookup[0]
        
        # Store the radial sscan data for A-scan extraction
        if self.radian_sscan_data is not None:
            self.status_info["radian_sscan_data"] = self.radian_sscan_data

        return self.status_info


class NDEGroupDataTFM(NDEGroupData):
    """Group data extractor for TFM (Total Focusing Method) data.
    
    TFM data requires transposition as it's stored with swapped axes.
    """

    def get_data_array(
        self,
        f: h5py.File,
        unistatus: Unistatus,
        group_id: int,
    ) -> np.ndarray:
        """Load and transpose TFM data.
        
        Args:
            f: Open HDF5 file handle.
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
            
        Returns:
            Transposed data array (z,y,x → z,x,y).
        """
        path = unistatus.get_path_to_data(group_id)
        if path is None:
            raise ValueError(f"No data path found for group {group_id}")
        arr = np.array(f[path][:])
        # TFM data is stored as z,y,x - transpose to z,x,y
        return np.transpose(arr, (0, 2, 1))

    def set_data_array(
        self,
        f: h5py.File,
        unistatus: Unistatus,
        group_id: int,
    ) -> None:
        """Load the transposed TFM data array.
        
        Args:
            f: Open HDF5 file handle.
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
        """
        self.data_array = self.get_data_array(f, unistatus, group_id)


class NDEGroupDataFMC(NDEGroupData):
    """Group data extractor for FMC (Full Matrix Capture) data.
    
    FMC data requires reshaping from stacked A-scans to a proper
    pulser/receiver organization.
    """

    def __init__(
        self,
        f: h5py.File,
        unistatus: Unistatus,
        group_id: int,
        setup_json: dict,
    ):
        """Initialize the FMC extractor.
        
        Args:
            f: Open HDF5 file handle.
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
            setup_json: Decoded JSON setup for FMC beam configuration.
        """
        self.setup_json = setup_json
        self.group_setups = self._get_group_dict(setup_json)
        super().__init__(f, unistatus, group_id)

    def _get_group_dict(self, setup_json_dict: dict) -> list:
        """Extract groups configuration from setup JSON.
        
        Args:
            setup_json_dict: Decoded JSON setup.
            
        Returns:
            List of group configurations.
        """
        return setup_json_dict.get("groups", [])

    def _reshape_pulser_receiver(
        self, 
        arr: np.ndarray, 
        len_pulsers: int, 
        len_receivers: int
    ) -> np.ndarray:
        """Reshape FMC data from stacked format to pulser/receiver organization.
        
        Args:
            arr: Raw FMC data array.
            len_pulsers: Number of pulsers.
            len_receivers: Number of receivers per pulser.
            
        Returns:
            Reshaped array organized by pulser/receiver.
        """
        # Reshape logic for FMC data
        data_shape = arr.shape
        if len(data_shape) == 2:
            n_positions = data_shape[0]
            stacked_length = data_shape[1]
            ascan_length = stacked_length // (len_pulsers * len_receivers)
            
            reshaped = arr.reshape(n_positions, len_pulsers, len_receivers, ascan_length)
            return reshaped
        return arr

    def get_data_array(
        self,
        f: h5py.File,
        unistatus: Unistatus,
        group_id: int,
    ) -> np.ndarray:
        """Load and reshape FMC data.
        
        Args:
            f: Open HDF5 file handle.
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
            
        Returns:
            Reshaped FMC data array.
        """
        path = unistatus.get_path_to_data(group_id)
        if path is None:
            raise ValueError(f"No data path found for group {group_id}")
        arr = np.array(f[path][:])

        # Get beam configuration for reshaping
        try:
            beams = self.group_setups[group_id]["processes"][0]["ultrasonicMatrixCapture"]["beams"]
            
            tx_rx_mapping = {}
            for beam in beams:
                tx_rx_mapping[beam["pulsers"][0]["id"]] = beam["receivers"]

            len_receivers = len(tx_rx_mapping[0])
            len_pulsers = len(tx_rx_mapping)

            # Reshape the array
            stacked_array = self._reshape_pulser_receiver(arr, len_pulsers, len_receivers)
            
            # Stack by removing U axis
            if stacked_array.ndim == 4:
                stacked_array = np.vstack(stacked_array).astype(np.int16)
            
            return stacked_array
            
        except (KeyError, IndexError) as e:
            logger.warning(f"Could not reshape FMC data: {e}. Returning raw array.")
            return arr

    def set_data_array(
        self,
        f: h5py.File,
        unistatus: Unistatus,
        group_id: int,
    ) -> None:
        """Load the reshaped FMC data array.
        
        Args:
            f: Open HDF5 file handle.
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
        """
        self.data_array = self.get_data_array(f, unistatus, group_id)

    def set_status_info(
        self, 
        unistatus: Unistatus, 
        group_id: int
    ) -> dict:
        """Extract status info for FMC data.
        
        Args:
            unistatus: Version-specific metadata accessor.
            group_id: 1-based group index.
            
        Returns:
            Dictionary containing status information.
        """
        self.status_info = {}
        self.status_info["gr"] = group_id
        self.status_info["group_name"] = unistatus.get_group_name(group_id)
        self.status_info["min_value"] = unistatus.get_amplitude_min(group_id)
        self.status_info["max_value"] = unistatus.get_amplitude_max(group_id)

        return self.status_info


class NdeLoader:
    """Complete NDE file loader supporting all data types.
    
    This loader uses the utility modules to handle:
    - Zero-degree UT data
    - Sectorial scans (S-scans)
    - TFM (Total Focusing Method)
    - FMC (Full Matrix Capture)
    
    It returns NdeModel instances compatible with the application.
    """

    def load(self, nde_file: str, group_idx: int = 1) -> NdeModel:
        """Load a single group from an NDE file.
        
        Args:
            nde_file: Path to the .nde file.
            group_idx: 1-based index of the group to load.
            
        Returns:
            NdeModel containing the extracted volume and metadata.
        """
        nde_path = Path(nde_file)
        
        with h5py.File(nde_file, "r") as f:
            # Get and decode the JSON setup
            json_decoded = self._read_setup_json(f)
            
            # Get version-specific accessor
            unistatus = get_unistatus(json_decoded)
            if unistatus is None:
                raise ValueError(
                    f"Unsupported NDE version: {json_decoded.get('version', 'unknown')}"
                )
            
            # Get valid groups
            group_list = unistatus.get_n_group()
            if not group_list:
                raise ValueError("No valid groups found in NDE file")
            
            # Map group_idx to actual group ID
            if group_idx < 1 or group_idx > len(group_list):
                raise ValueError(
                    f"Group index {group_idx} out of range (1-{len(group_list)})"
                )
            actual_group_id = group_list[group_idx - 1]
            
            # Detect data type and load accordingly
            group_data = self._load_group(
                f, unistatus, actual_group_id, json_decoded
            )
            
            # Build and return NdeModel
            return self._build_model(
                group_data, 
                json_decoded, 
                nde_path,
                group_idx
            )

    def load_all_groups(self, nde_file: str) -> Dict[int, NdeModel]:
        """Load all valid groups from an NDE file.
        
        Args:
            nde_file: Path to the .nde file.
            
        Returns:
            Dictionary mapping 1-based group indices to NdeModel instances.
        """
        nde_path = Path(nde_file)
        models: Dict[int, NdeModel] = {}
        
        with h5py.File(nde_file, "r") as f:
            json_decoded = self._read_setup_json(f)
            
            unistatus = get_unistatus(json_decoded)
            if unistatus is None:
                raise ValueError(
                    f"Unsupported NDE version: {json_decoded.get('version', 'unknown')}"
                )
            
            group_list = unistatus.get_n_group()
            
            for idx, actual_group_id in enumerate(group_list, start=1):
                try:
                    group_data = self._load_group(
                        f, unistatus, actual_group_id, json_decoded
                    )
                    models[idx] = self._build_model(
                        group_data, json_decoded, nde_path, idx
                    )
                except Exception as e:
                    logger.error(f"Failed to load group {idx}: {e}")
                    continue
        
        return models

    def _read_setup_json(self, f: h5py.File) -> dict:
        """Read and decode the setup JSON from the NDE file.
        
        Args:
            f: Open HDF5 file handle.
            
        Returns:
            Decoded JSON setup dictionary.
        """
        path_to_json = "Public/Setup" if "Public" in f.keys() else "Domain/Setup"
        json_str = f[path_to_json][()]
        return json.loads(json_str)

    def _detect_data_type(
        self, 
        json_decoded: dict, 
        group_index: int = 0
    ) -> str:
        """Detect the type of NDE data.
        
        Args:
            json_decoded: Decoded JSON setup.
            group_index: 0-based group index for type detection.
            
        Returns:
            String identifier for the data type.
        """
        if NDEDataTypeCheck.is_sectorial_scan(json_decoded, group_index):
            return "sectorial_scan"
        if NDEDataTypeCheck.is_tfm(json_decoded, group_index):
            return "tfm"
        if NDEDataTypeCheck.is_fmc(json_decoded, group_index):
            return "fmc"
        return "zero_deg"

    def _load_group(
        self,
        f: h5py.File,
        unistatus: Unistatus,
        group_id: int,
        json_decoded: dict,
    ) -> NDEGroupData:
        """Load a single group with type-specific handling.
        
        Args:
            f: Open HDF5 file handle.
            unistatus: Version-specific metadata accessor.
            group_id: Actual group ID from the file.
            json_decoded: Decoded JSON setup.
            
        Returns:
            NDEGroupData instance with loaded data.
        """
        # Detect data type (use 0-based index for detection)
        data_type = self._detect_data_type(json_decoded, 0)
        
        logger.info(f"Loading group {group_id} as {data_type}")
        
        if data_type == "sectorial_scan":
            return NDEGroupDataSectorialScan(
                f=f,
                unistatus=unistatus,
                group_id=group_id,
                setup_json=json_decoded,
            )
        elif data_type == "tfm":
            return NDEGroupDataTFM(
                f=f, 
                unistatus=unistatus, 
                group_id=group_id
            )
        elif data_type == "fmc":
            return NDEGroupDataFMC(
                f=f, 
                unistatus=unistatus, 
                group_id=group_id,
                setup_json=json_decoded,
            )
        else:
            return NDEGroupDataZeroDegUT(
                f=f, 
                unistatus=unistatus, 
                group_id=group_id
            )

    def _build_model(
        self,
        group_data: NDEGroupData,
        json_decoded: dict,
        nde_path: Path,
        group_idx: int,
    ) -> NdeModel:
        """Build an NdeModel from extracted group data.
        
        Args:
            group_data: Extracted NDEGroupData instance.
            json_decoded: Decoded JSON setup.
            nde_path: Path to the source NDE file.
            group_idx: 1-based group index.
            
        Returns:
            Configured NdeModel instance.
        """
        data_array = group_data.data_array
        status_info = group_data.status_info
        
        # Determine data type for metadata
        data_type = self._detect_data_type(json_decoded, 0)
        
        # Build axis order and positions from status_info
        axis_order, positions = self._build_axis_info(status_info, data_array.shape)
        
        # Apply orientation correction based on metadata
        data_array, axis_order, positions = self._reorder_axes_by_metadata(
            data_array, axis_order, positions, status_info, json_decoded
        )
        
        # Apply 90° clockwise rotation on each slice (axes 1 and 2)
        data_array, axis_order, positions = self._rotate_clockwise(
            data_array, axis_order, positions
        )
        
        # Get min/max values
        min_value = status_info.get("min_value", float(np.min(data_array)))
        max_value = status_info.get("max_value", float(np.max(data_array)))
        
        # Build metadata dictionary
        metadata: Dict[str, Any] = {
            "structure": "Public" if "Public" in str(nde_path) else "auto",
            "group_idx": group_idx,
            "group_name": status_info.get("group_name", ""),
            "axis_order": axis_order,
            "positions": positions,
            "min_value": min_value,
            "max_value": max_value,
            "data_type": data_type,
            "acquisition_attributes": {
                "processes": [data_type],
            },
            "nde_version": json_decoded.get("version", "unknown"),
            "source_file": str(nde_path),
        }
        
        # Add type-specific metadata
        if data_type == "sectorial_scan":
            if "sscan_slice_info" in status_info:
                metadata["sscan_slice_info"] = status_info["sscan_slice_info"]
            if "radian_sscan_data" in status_info:
                metadata["radian_sscan_data"] = status_info["radian_sscan_data"]
        
        # Add raw status info for reference
        metadata["status_info"] = status_info
        
        # Create and configure model
        model = NdeModel()
        model.set_volume(data_array, metadata)
        
        return model

    def _build_axis_info(
        self,
        status_info: dict,
        data_shape: Tuple[int, ...],
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Build axis order and positions from status info.
        
        Args:
            status_info: Extracted status information.
            data_shape: Shape of the data array.
            
        Returns:
            Tuple of (axis_order list, positions dict).
        """
        axis_order: List[str] = []
        positions: Dict[str, np.ndarray] = {}
        
        # Map internal names to standard names
        axis_mapping = {
            "lengthwise": "UCoordinate",
            "crosswise": "VCoordinate", 
            "ultrasound": "Ultrasound",
        }
        
        # Build axis info in order: lengthwise (U), crosswise (V), ultrasound (W)
        for internal_name, standard_name in axis_mapping.items():
            if internal_name in status_info:
                axis_info = status_info[internal_name]
                axis_order.append(standard_name)
                
                if "positions_m" in axis_info:
                    positions[standard_name] = axis_info["positions_m"]
                else:
                    # Generate positions from offset/resolution/quantity
                    offset = axis_info.get("offset", 0.0)
                    resolution = axis_info.get("resolution", 1.0)
                    quantity = axis_info.get("quantity", 1)
                    positions[standard_name] = np.array([
                        offset + i * resolution for i in range(quantity)
                    ])
        
        # Handle case where axis info doesn't match data shape
        while len(axis_order) < len(data_shape):
            idx = len(axis_order)
            name = f"axis_{idx}"
            axis_order.append(name)
            positions[name] = np.arange(data_shape[idx], dtype=float)
        
        return axis_order, positions

    def _reorder_axes_by_metadata(
        self,
        data: np.ndarray,
        axis_order: List[str],
        positions: Dict[str, np.ndarray],
        status_info: dict,
        setup_json: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
        """Reorder axes so that the slice axis (largest spatial dimension) comes first.
        
        The slice axis should be the one with the most elements between U and V coordinates.
        Ultrasound axis always stays last (depth/Y axis).
        
        Strategy:
        1. Check uCoordinateOrientation in dataMappings for explicit guidance
        2. Otherwise, use the largest axis between U and V as the slice axis
        3. Ultrasound always remains the last axis
        
        Args:
            data: The volume data array.
            axis_order: Current axis order names.
            positions: Current positions dict.
            status_info: Extracted status information with axis details.
            setup_json: Decoded JSON setup for orientation hints.
            
        Returns:
            Tuple of (reordered_data, new_axis_order, new_positions).
        """
        if data.ndim < 3 or not axis_order or len(axis_order) != data.ndim:
            return data, axis_order, positions

        # Try to get orientation hint from dataMappings
        orientation = None
        try:
            mappings = (setup_json or {}).get("dataMappings") or []
            if mappings:
                mapping = mappings[0]
                grid = mapping.get("discreteGrid") or {}
                orientation = grid.get("uCoordinateOrientation")
        except Exception:
            orientation = None

        orientation_l = str(orientation).lower() if orientation is not None else None
        
        # Determine slice axis based on orientation or quantity
        # uCoordinateOrientation indicates how U is oriented:
        # - "around" = U is circumferential (around a pipe) = U is the slice axis
        # - "length" = U is along the length = U is the slice axis
        # When orientation is explicitly set, U is always the slice axis
        slice_axis_name: Optional[str] = None
        
        if orientation_l in ("around", "length"):
            # Explicit orientation: U is the slice axis
            slice_axis_name = "UCoordinate"
            logger.debug(f"Using UCoordinate as slice axis based on orientation={orientation}")
        
        # If no explicit orientation, use the axis with the most elements
        if slice_axis_name is None:
            u_qty = status_info.get("lengthwise", {}).get("quantity", 0)
            v_qty = status_info.get("crosswise", {}).get("quantity", 0)
            
            if v_qty > u_qty:
                slice_axis_name = "VCoordinate"
                logger.debug(f"Using VCoordinate as slice axis (V={v_qty} > U={u_qty})")
            else:
                slice_axis_name = "UCoordinate"
                logger.debug(f"Using UCoordinate as slice axis (U={u_qty} >= V={v_qty})")
        
        # Find current indices
        normalized = [str(name).lower() for name in axis_order]
        
        ultrasound_idx = None
        if "ultrasound" in normalized:
            ultrasound_idx = normalized.index("ultrasound")
        
        slice_idx = None
        if slice_axis_name and slice_axis_name.lower() in normalized:
            slice_idx = normalized.index(slice_axis_name.lower())
        
        # Build desired axis order: [slice_axis, other_spatial, ultrasound]
        desired_indices: List[int] = []
        
        # First: slice axis
        if slice_idx is not None:
            desired_indices.append(slice_idx)
        
        # Second: other spatial axes (not slice, not ultrasound)
        for idx in range(len(axis_order)):
            if idx not in desired_indices and idx != ultrasound_idx:
                desired_indices.append(idx)
        
        # Last: ultrasound axis
        if ultrasound_idx is not None and ultrasound_idx not in desired_indices:
            desired_indices.append(ultrasound_idx)
        
        # Fill any remaining
        for idx in range(len(axis_order)):
            if idx not in desired_indices:
                desired_indices.append(idx)
        
        # Check if reordering is needed
        if desired_indices == list(range(len(axis_order))):
            logger.debug(
                "No axis reordering needed | shape=%s | order=%s",
                data.shape, axis_order
            )
            return data, axis_order, positions
        
        # Apply transpose
        reordered = np.transpose(data, axes=desired_indices)
        new_axis_order = [axis_order[i] for i in desired_indices]
        
        new_positions: Dict[str, np.ndarray] = {}
        for name in new_axis_order:
            if name in positions:
                new_positions[name] = positions[name]
        
        logger.info(
            "Reordered axes | orientation=%s | slice_axis=%s | old_order=%s | new_order=%s | old_shape=%s | new_shape=%s",
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
        """Rotate each slice 90° clockwise in the (Y, X) plane.
        
        This rotation is applied as a display-ready orientation so views 
        stay passive. Axis names and positions are swapped accordingly:
        - The axis from original X (dim 2) becomes the new Y (dim 1) and is reversed
        - The original Y (dim 1) becomes the new X (dim 2)
        
        Args:
            data: The volume data array with shape (slices, Y, X).
            axis_order: Current axis order names.
            positions: Current positions dict.
            
        Returns:
            Tuple of (rotated_data, new_axis_order, new_positions).
        """
        if data.ndim < 3:
            return data, axis_order, positions

        # Apply 90° clockwise rotation on axes (1, 2) = (Y, X) plane
        # k=-1 means clockwise rotation
        rotated = np.rot90(data, k=-1, axes=(1, 2))

        # Swap axis names for dimensions 1 and 2
        if axis_order and len(axis_order) >= 3:
            a0, a1, a2, *rest = axis_order
            new_axis_order = [a0, a2, a1, *rest]
        else:
            new_axis_order = axis_order

        # Update positions: swap and reverse as needed
        new_positions: Dict[str, np.ndarray] = {}
        for name, coords in positions.items():
            new_positions[name] = coords
            
        # After rot90 clockwise:
        # - new axis1 comes from old axis2 (reversed)
        # - new axis2 comes from old axis1 (preserved)
        if axis_order and len(axis_order) >= 3:
            a0, a1, a2, *rest = axis_order
            if a2 in positions:
                new_positions[a2] = positions[a2][::-1]
            if a1 in positions:
                new_positions[a1] = positions[a1]

        logger.info(
            "Applied 90° CW rotation | old_shape=%s | new_shape=%s | old_order=%s | new_order=%s",
            data.shape,
            rotated.shape,
            axis_order,
            new_axis_order,
        )

        return rotated, new_axis_order, new_positions


# Convenience function for direct loading
def load_nde(nde_file: str, group_idx: int = 1) -> NdeModel:
    """Load a single group from an NDE file.
    
    This is a convenience function that creates an NdeLoader instance
    and loads the specified group.
    
    Args:
        nde_file: Path to the .nde file.
        group_idx: 1-based index of the group to load.
        
    Returns:
        NdeModel containing the extracted volume and metadata.
    """
    loader = NdeLoader()
    return loader.load(nde_file, group_idx)


def load_all_nde_groups(nde_file: str) -> Dict[int, NdeModel]:
    """Load all valid groups from an NDE file.
    
    This is a convenience function that creates an NdeLoader instance
    and loads all groups.
    
    Args:
        nde_file: Path to the .nde file.
        
    Returns:
        Dictionary mapping 1-based group indices to NdeModel instances.
    """
    loader = NdeLoader()
    return loader.load_all_groups(nde_file)
