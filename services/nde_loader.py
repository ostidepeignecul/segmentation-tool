"""NDE Loader - Complete implementation.

This loader implements full support for all NDE data types and includes
the version and sectorial scan helpers required for processing NDE files.

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
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from models.nde_model import NdeModel

logger = logging.getLogger(__name__)

supported_speciments = ["plateGeometry", "pipeGeometry"]


class Unistatus_NDE_4_0_0_Dev:
    version = "4.0.0"

    def __init__(self, json_decoded) -> None:
        self.json_decoded = json_decoded

        self.n_groups = len(self.json_decoded["groups"])
        self.raw_ascan_dataset_idx = {}

        # We want to discard any groups that dont have data
        # without modifying the json decoded
        # This is because groups can exist that only describe a process
        # or for organizing acquisitions without data
        # the original index is the key to the dict, and the group itself is the value
        valid_datasets = {
            idx: group
            for idx, group in enumerate(self.json_decoded["groups"])
            if "datasets" in group
        }
        self.valid_groups = list(valid_datasets.values())

        # sometimes, for example in sectorial scans,
        # the first group has an id of 0, whereas most of the
        # calling functions assume an 1-based index
        # so we need to check if any valid group has an id of 0
        # and if so, we need to adjust the index (and the index of all other groups)

        # if any([group["id"] == 0 for group in self.valid_groups]):
        #     # we only modify their keys, we dont want to modify the original json
        #     valid_datasets = {idx + 1: group for idx, group in valid_datasets.items()}

        group_classes = []

        supported_classes = ["AScanAmplitude", "TfmValue"]

        for idx, group in valid_datasets.items():
            if len(group["datasets"]) >= 1:
                data_class = group["datasets"][0]["dataClass"]
            else:
                data_class = group["datasets"]["dataClass"]

            if data_class in supported_classes:
                group_classes.append(data_class)

            # for gr, dataset in enumerate(self.valid_groups[idx - 1]["datasets"]):
            # We want the frist dataset
            self.raw_ascan_dataset_idx[idx] = 0
        self.group_classes = group_classes

        # Creates a list with the indexes of the groups.
        # This allows to know the real group id based on the index in the self.valid_groups list
        # For example, with mapping [1,3,5,7], we can know that the group with index 0 in self.valid_groups
        # has an id of 1, the group with index 1 in self.valid_groups has an id of 3, and so on.
        # This is necessary for NDES that have groups that dont have data mixed with groups that do have data
        self.group_mapping = self.get_n_group()

    def get_n_group(self) -> list[int]:
        return [group["id"] for group in self.valid_groups]

    def get_path_to_data(self, gr: int) -> str | None:
        """
        Return the path in the nde file containing the complete data.
        Args:
          gr: Int from 1 to n_group"""

        for group in self.valid_groups:
            if group["id"] == gr:
                return group["datasets"][self.raw_ascan_dataset_idx[gr]]["path"]

        # return self.valid_groups[gr - 1]["datasets"][self.raw_ascan_dataset_idx[gr]][
        #     "path"
        # ]

    def get_group_name(self, gr: int) -> str:
        """
        Return the group name.
        Args:
          gr: Int from 1 to n_group"""
        try:
            gr_index = self.group_mapping.index(gr)
            group_name = self.valid_groups[gr_index]["name"]
        except KeyError:
            logger.debug(f"Group name not fund at path groups/{gr - 1}/name")
            group_name = ""

        return group_name

    def get_amplitude_min(self, gr: int) -> float:
        """
        Return the minimum amplitude of the complete data.
        Args:
          gr: Int from 1 to n_group"""

        gr_index = self.group_mapping.index(gr)
        return self.valid_groups[gr_index]["datasets"][self.raw_ascan_dataset_idx[gr]][
            "dataValue"
        ]["min"]

    def get_amplitude_max(self, gr: int) -> float:
        """
        Return the maximum amplitude of the complete data.
        Args:
          gr: Int from 1 to n_group"""
        gr_index = self.group_mapping.index(gr)
        return self.valid_groups[gr_index]["datasets"][self.raw_ascan_dataset_idx[gr]][
            "dataValue"
        ]["max"]

    def get_u_axis_reg_info(self, gr: int) -> dict:
        """
        Return useful information about u axis to register position from index.
        Args:
          gr: Int from 1 to n_group"""
        try:
            gr_index = self.group_mapping.index(gr)
            u_axis_reg_info = self.valid_groups[gr_index]["datasets"][
                self.raw_ascan_dataset_idx[gr]
            ]["dimensions"][0]
        except KeyError:
            logger.debug(
                "No u axis registration info in json setup at path "
                f"groups/{self.raw_ascan_dataset_idx[gr]}/dimensions/0"
            )
            u_axis_reg_info = {
                "axis": "UCoordinate",
                "quantity": 0,
                "resolution": 0.0,
                "offset": 0.0,
            }
        return u_axis_reg_info

    def get_v_axis_reg_info(self, gr: int) -> dict:
        """
        Return useful information about v axis to register position from index.
        Args:
          gr: Int from 1 to n_group"""
        try:
            gr_index = self.group_mapping.index(gr)
            v_axis_reg_info = self.valid_groups[gr_index]["datasets"][
                self.raw_ascan_dataset_idx[gr]
            ]["dimensions"][1]
        except KeyError:
            logger.debug(
                "No v axis registration info in json setup at path "
                f"groups/{self.raw_ascan_dataset_idx[gr]}/dimensions/1"
            )
            v_axis_reg_info = {
                "axis": "VCoordinate",
                "quantity": 0,
                "resolution": 0.0,
                "offset": 0.0,
            }
        return v_axis_reg_info

    def get_w_axis_reg_info(self, gr: int) -> dict:
        """
        Return useful information about ut axis to register position from index.
        Args:
          gr: Int from 1 to n_group"""
        try:
            gr_index = self.group_mapping.index(gr)
            w_axis_reg_info = self.valid_groups[gr_index]["datasets"][
                self.raw_ascan_dataset_idx[gr]
            ]["dimensions"][2]
        except KeyError:
            logger.debug(
                "No w axis registration info in json setup at path "
                f"groups/{self.raw_ascan_dataset_idx[gr]}/dimensions/2"
            )
            w_axis_reg_info = {
                "axis": "Ultrasound",
                "quantity": 0,
                "resolution": 0.0,
                "offset": 0.0,
            }
        return w_axis_reg_info

    def get_ut_velocity(self) -> float:
        try:
            specimens = self.json_decoded["specimens"][0]
            if len(specimens) == 0:
                velocity = 0.0
                return velocity
            # Check if one of the supported specimens is in the json
            for specimen_type in supported_speciments:
                if specimen_type in specimens:
                    velocity = specimens[specimen_type]["material"]["longitudinalWave"][
                        "nominalVelocity"
                    ]
                    return velocity

            velocity = self.json_decoded["specimens"][0]["plateGeometry"]["material"][
                "longitudinalWave"
            ]["nominalVelocity"]
        except KeyError:
            logger.debug(
                "No velocity available in json setup at path "
                "specimens/0/plateGeometry/material/longitudinalWave/nominalVelocity"
            )
            velocity = 0.0
        return velocity


class Unistatus_NDE_3_0_0:
    version = "3.0.0"

    def __init__(self, json_decoded) -> None:
        self.json_decoded = json_decoded

    def get_n_group(self) -> list[int]:
        return list(range(1, len(self.json_decoded["groups"]) + 1))

    def get_path_to_data(self, gr: int) -> str:
        """
        Return the path in the nde file containing the complete data.
        Args:
          gr: Int from 1 to n_group"""
        return self.json_decoded["groups"][gr - 1]["dataset"]["ascan"]["amplitude"][
            "path"
        ]

    def get_group_name(self, gr: int) -> str:
        """
        Return the group name.
        Args:
          gr: Int from 1 to n_group"""
        try:
            group_name = self.json_decoded["groups"][gr - 1]["name"]
        except KeyError:
            logger.debug(f"Group name not fund at path groups/{gr - 1}/name")
            group_name = ""
        return group_name

    def get_amplitude_min(self, gr: int) -> float:
        """
        Return the minimum amplitude of the complete data.
        Args:
          gr: Int from 1 to n_group"""
        return self.json_decoded["groups"][gr - 1]["dataset"]["ascan"]["amplitude"][
            "dataSampling"
        ]["min"]

    def get_amplitude_max(self, gr: int) -> float:
        """
        Return the maximum amplitude of the complete data.
        Args:
          gr: Int from 1 to n_group"""
        return self.json_decoded["groups"][gr - 1]["dataset"]["ascan"]["amplitude"][
            "dataSampling"
        ]["max"]

    def get_u_axis_reg_info(self, gr: int) -> dict:
        """
        Return useful information about u axis to register position from index.
        Args:
          gr: Int from 1 to n_group"""
        try:
            u_axis_reg_info = self.json_decoded["groups"][gr - 1]["dataset"]["ascan"][
                "amplitude"
            ]["dimensions"][0]
        except KeyError:
            logger.debug(
                f"No u axis registration info in json setup at path groups/{gr - 1}/dataset/ascan/amplitude/dimensions/0"
            )
            u_axis_reg_info = {
                "axis": "UCoordinate",
                "quantity": 0,
                "resolution": 0.0,
                "offset": 0.0,
            }
        return u_axis_reg_info

    def get_v_axis_reg_info(self, gr: int) -> dict:
        """
        Return useful information about v axis to register position from index.
        Args:
          gr: Int from 1 to n_group"""
        try:
            v_axis_reg_info = self.json_decoded["groups"][gr - 1]["dataset"]["ascan"][
                "amplitude"
            ]["dimensions"][1]
        except KeyError:
            logger.debug(
                f"No v axis registration info in json setup at path groups/{gr - 1}/dataset/ascan/amplitude/dimensions/1"
            )
            v_axis_reg_info = {
                "axis": "VCoordinate",
                "quantity": 0,
                "resolution": 0.0,
                "offset": 0.0,
            }
        return v_axis_reg_info

    def get_w_axis_reg_info(self, gr: int) -> dict:
        """
        Return useful information about ut axis to register position from index.
        Args:
          gr: Int from 1 to n_group"""
        try:
            w_axis_reg_info = self.json_decoded["groups"][gr - 1]["dataset"]["ascan"][
                "amplitude"
            ]["dimensions"][2]
        except KeyError:
            logger.debug(
                f"No w axis registration info in json setup at path groups/{gr - 1}/dataset/ascan/amplitude/dimensions/2"
            )
            w_axis_reg_info = {
                "axis": "Ultrasound",
                "quantity": 0,
                "resolution": 0.0,
                "offset": 0.0,
            }
        return w_axis_reg_info

    def get_ut_velocity(self) -> float:
        try:
            # 2 differents values of velocity in nde
            # velocity = self.json_decoded["specimens"][0]["plateGeometry"]["material"][
            #     "longitudinalWave"
            # ]["nominalVelocity"]
            velocity = self.json_decoded["groups"][0]["paut"]["velocity"]
        except KeyError:
            logger.debug(
                "No velocity available in json setup at path "
                "specimens/0/plateGeometry/material/longitudinalWave/nominalVelocity"
            )
            velocity = 1.0
        return velocity


def get_unistatus(json_decoded) -> Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev | None:
    """This function return an object that standardize the way to retreive information in the setup json accross
    different version of ndes in order to visualize the data."""
    version = json_decoded["version"]

    if not isVersionHigherOrEqualThan(version, "2.2.13"):
        # raise Exception("This version is not supported.")
        return None
    elif (
        (version == "3.0.0")
        | (version == "3.1.0")
        | (version == "3.2.0")
        | (version == "3.3.0")
        | (version == "3.0.0.ReducedForAI")
    ):
        return Unistatus_NDE_3_0_0(json_decoded)
    elif version in ["4.0.0-Dev", "4.0.0", "4.1.0"]:
        return Unistatus_NDE_4_0_0_Dev(json_decoded)


def isVersionHigherOrEqualThan(inputVersion: str, refVersion: str) -> bool:
    in_list = inputVersion.split(".")
    ref_list = refVersion.split(".")

    if len(in_list) <= len(ref_list):
        P = len(in_list)
    else:
        P = len(ref_list)

    isHigher = False
    for p in range(P):
        if int(in_list[p]) >= int(ref_list[p]):
            isHigher = True
            break

    return isHigher


class SScanSliceInfo(TypedDict):
    Nx: int
    Nz: int
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    flip_z: bool
    angle_min_deg: float
    angle_max_deg: float
    r_min: float
    r_max: float
    beam_starts: list[float]
    beam_ranges: list[float]
    beam_angles: list[float]
    beam_coords: dict[int, dict[str, dict[str, float]]]


class SectorialScanNDE:
    def __init__(self, filename: str | None | Path = None):
        # Take filename as class import, run through main sequence

        self.filename = filename
        self.file = None
        # self.parse_sectorial_scan()

    @staticmethod
    def is_sectorial_scan(setup: dict, group_index: int = 0) -> bool:
        #!Not sure if this can reliably always detect a sectorial scan
        #! Maybe multiple types of formations or sectorialX can exist
        #! Need to check the NDE spec
        if "groups" in setup:
            if len(setup["groups"]) > 0:
                if group_index > len(setup["groups"]) - 1:
                    raise ValueError(
                        f"Group index {group_index} is out of bounds for the number of groups in setup"
                    )
                if "processes" in setup["groups"][group_index]:
                    processes = setup["groups"][0]["processes"]
                    if "ultrasonicPhasedArray" in processes[0]:
                        ultrasoundPhasedArray = processes[0]["ultrasonicPhasedArray"]
                        if "pulseEcho" in ultrasoundPhasedArray:
                            pulseEcho = ultrasoundPhasedArray["pulseEcho"]
                            if "sectorialFormation" or "compoundFormation" in pulseEcho:
                                return True
        return False

    def read_h5(self, file=None):
        # Useful for testing, but we might want to remove any ability for this
        # class to read files directlu, as the rest of the data input processing handles it
        # and provides fallbacks for compatibility
        # Load h5 file with h5py package

        if file is not None:
            self.file = file
            return
        if self.filename is None:
            raise ValueError("No filename provided")
        self.file = h5py.File(self.filename, "r")
        return self.file

    def parse_setup(self, setup: dict | None = None):
        """If a setup json dict is provided, it will be used instead of the one in the h5 file.
        This is useful for providing compatibility with older versions of the NDE file format.
        """

        if setup is None:
            path_to_json = (
                "Public/Setup" if "Public" in self.file.keys() else "Domain/Setup"
            )
            json_str = self.file[path_to_json][()]  # type:ignore
            setup = json.loads(json_str)  # type:ignore

        if setup is None:
            raise ValueError("No setup provided and no setup file found in h5 file")

        self.setup = setup

        groups = setup["groups"]
        self.groups = groups

        self.inspection_setup = [group["processes"] for group in groups]
        return groups
        # self.scanplan = setup["boost_serialization"]["m_scanPlan"]["px"]

    def extract_scan_parameters(self):
        # retrive the mechanical scanning parameters, they are the same across all groups

        self.shared_scan_parameters = {}

        data_mappings = self.setup["dataMappings"][0]["discreteGrid"]

        scan_length = float(data_mappings["dimensions"][0]["quantity"])
        scan_origin = float(data_mappings["dimensions"][0]["offset"])
        scan_res = float(data_mappings["dimensions"][0]["resolution"])

        self.shared_scan_parameters["Scan Length"] = scan_length
        self.shared_scan_parameters["Scan Origin"] = scan_origin
        self.shared_scan_parameters["Scan Resolution"] = scan_res
        self.shared_scan_parameters["Unit"] = "mm"

    def extract_beam_parameters(self, group_params: dict):
        # retrive the beam parameters needed for S-scan construction
        # a dict is set up to store all these useful parameters that will be structured by group then by beam index

        beam_params = {}

        # group_probe_offset = float(
        #      group["datasets"][0]["dimensions"][1]["beams"][0]["ultrasoundOffset"]
        # )
        group_velocity = float(
            group_params["processes"][0]["ultrasonicPhasedArray"]["velocity"]
        )
        ultrasound_axis = [
            ut
            for ut in group_params["datasets"][0]["dimensions"]
            if ut["axis"] == "Ultrasound"
        ]
        if len(ultrasound_axis) == 0:
            raise ValueError("Ultrasound axis not found")
        else:
            ultrasound_axis = ultrasound_axis[0]
        group_time_res = float(ultrasound_axis["resolution"])

        group_beam = group_params["processes"][0]["ultrasonicPhasedArray"].get("beams")
        if group_beam is None:
            raise ValueError(
                "No beam data found in group. This indicates that the group is not a sectorial scan group."
            )

        for j in np.arange(len(group_beam)):
            #!We dont compensate for beam delay, this
            #! Does not seem to cause any issues for simple extraction of
            #! S-scan data, but may be needed for more advanced analysis

            beam_key = group_beam[j]["id"]

            beam_params[beam_key] = {}

            refracted_ang = group_beam[j]["refractedAngle"]

            beam_params[beam_key]["refracted_angle"] = refracted_ang
            beam_params[beam_key]["start"] = (
                float(group_beam[j]["ascanStart"])  # in mm
                / 2
                * (group_velocity * 1e3)  # velocity in mm/s
                * np.cos(np.deg2rad(refracted_ang))
            )

            beam_params[beam_key]["range"] = (
                float(group_beam[j]["ascanLength"])
                / 2
                * (group_velocity * 1e3)
                * np.cos(np.deg2rad(refracted_ang))
            )
            beam_params[beam_key]["UltrasoundIndexLength"] = ultrasound_axis["quantity"]
            beam_params[beam_key]["Unit"] = "mm / degrees"
            beam_params[beam_key]["Time Resolution"] = group_time_res
            beam_params[beam_key]["ultrasound_resolution"] = (
                float(ultrasound_axis["resolution"]) * 1e3
            )
            beam_params[beam_key]["raw_params"] = group_beam[j]

            beam_params[beam_key]["dimension_params"] = group_params["datasets"][0][
                "dimensions"
            ][1]["beams"][j]
            beam_params[beam_key]["skew_angle"] = beam_params[beam_key][
                "dimension_params"
            ]["skewAngle"]
        return beam_params

    def retrieve_ascans(
        self, group_setup: dict, beam_params: dict, group_data: np.ndarray
    ):
        # Ascan is retrived from the data section of the h5 file, a reshape is done to remove the index axis (not used in oneline scanning scenario)
        # Ascans are organized to share the same group/beam index layout as the beam parameter dict
        # Quanitiy of Ascan points as well as global max and min amplitudes are also calculated here
        # Ascan QTY is needed for S-scan recontruction, the global min/max will be needed for normalization of the data

        ascans = {}
        # self.UT_param = {}

        global_max = 0
        global_min = 100
        # Createa dict from the data with the name as key

        data_shape = np.shape(group_data[:])

        for j in range(len(beam_params)):
            beam_name = list(beam_params.keys())[j]

            local_max = np.max(group_data[:, beam_name, :])
            local_min = np.min(group_data[:, beam_name, :])

            if local_max > global_max:
                global_max = local_max

            if local_min < global_min:
                global_min = local_min

            # Currently, we have for example 45 z points, that each contain for example
            # the data for each beam (say, 32). In each of those 32, there is the a scan data
            # However, we want to make it so we reorginize the data so that we
            # get the data by beam, for example 32 beams, with thei 45 z points and the a scan data at each z point
            # The data is currently in [z, beams, ascan] format, we want to change it to [beams, z, ascan]
            # since the beam name is the same as the beam index, we want the value at the key to be [z, ascan]
            ascans[beam_name] = group_data[:, beam_name, :]
        return ascans

    def build_uniform_radial_axis(self, beam_params, num_radial_samples=512):
        """
        Gather the min "Start" among all beams and the max "end" among all beams,
        then build a uniform radial axis from min_start to max_end.
        """
        beam_keys = list(beam_params.keys())

        # track the global min and max radial extents
        global_min_r = np.inf
        global_max_r = -np.inf

        for k in beam_keys:
            start_k = beam_params[k]["start"]  # mm
            range_k = beam_params[k]["range"]  # mm
            end_k = start_k + range_k

            if start_k < global_min_r:
                global_min_r = start_k
            if end_k > global_max_r:
                global_max_r = end_k

        # build the uniform radial axis
        radial_global = np.linspace(global_min_r, global_max_r, num_radial_samples)
        return radial_global

    def build_local_radial_axis(self, beam_index, beam_params, num_radial_samples=512):
        """
        Build a local radial axis for a specific beam index.
        """
        start_k = beam_params[beam_index]["start"]
        range_k = beam_params[beam_index]["range"]
        radial_local = np.linspace(start_k, start_k + range_k, num_radial_samples)
        return radial_local

    def build_uniform_angle_axis(self, beam_params):
        """
        Gather all refracted angles from beam_params
        and build a sorted *uniform* angle array from the min to max beam angle.

        Returns:
            theta_global: 1D array of angles in degrees (or radians, if you prefer).
        """
        beam_keys = list(beam_params.keys())

        # Get each beam's angle
        angles = [beam_params[k]["refracted_angle"] for k in beam_keys]

        # Decide how finely spaced you want your final angle axis
        # For instance, if beams are at 0°, 1°, 2°, ... 45°, you might keep them as is
        # or you might want a finer or coarser interpolation.
        min_angle = np.min(angles)
        max_angle = np.max(angles)

        # e.g. if your original beams are 46 angles, you might do the same count:
        num_angles = len(angles)  # or pick more or fewer, e.g. 2*len(angles)

        # create uniform angle array
        theta_global = np.linspace(min_angle, max_angle, num_angles)

        return theta_global

    def interpolate_beam_ascan_to_radial(
        self, beam_radial, beam_amplitude, radial_global
    ):
        """
        1D interpolation of beam's amplitude vs. beam_radial
        onto the radial_global axis.
        """

        # create a 1D interpolator
        f = interp1d(
            beam_radial,
            beam_amplitude,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        return f(radial_global)

    def construct_s_scan(
        self, ascans: dict, beam_params: dict, num_radial_samples=None
    ):
        """
        Build a 3D array of rectified S-scans, one per Z index.
        Shape of the final array: (N_z, N_r, N_theta).
        - N_z     = number of scanning positions (Z)
        - N_r     = number of interpolated radial samples
        - N_theta = number of interpolated angles (or beam angles)

        Steps:
        1) Build uniform angle axis (angles_global).
        2) Build uniform radial axis (radial_global).
        3) Determine how many Z indices there are from the first beam.
        4) For each z in range(N_z):
            For each beam, do 1D radial interpolation of that z's A-scan
            Place the result in the correct location in S_scan_3d[z, :, angleCol].
        """
        import numpy as np

        # We'll define final array shape = (N_z, N_r, N_theta)
        # First, let's find out how many Z positions we have.
        # We'll pick the first beam as reference.
        beam_keys = list(beam_params.keys())
        beam_keys.sort(key=lambda bk: beam_params[bk]["refracted_angle"])
        sample_beam = beam_keys[0]
        ascan_2d_ref = ascans[sample_beam]
        # We also define the num_radial_samples by the quantity of radial samples in the beam
        # meaning the ultrasound "quantity" in the beam
        if num_radial_samples is None:
            num_radial_samples = beam_params[sample_beam]["UltrasoundIndexLength"]

        # Typically shape = (N_z, N_ascanSamples)
        if ascan_2d_ref.ndim != 2:
            # Edge case: if you truly have no scanning axis, N_z would be 1.
            N_z = 1
        else:
            N_z = ascan_2d_ref.shape[0]

        # 1) Build a uniform angle axis
        angles_global = self.build_uniform_angle_axis(beam_params)
        # 2) Build a uniform radial axis
        radial_global = self.build_uniform_radial_axis(
            beam_params, num_radial_samples=num_radial_samples
        )

        N_theta = len(angles_global)
        N_r = len(radial_global)

        # Allocate the 3D array
        # S_scan_3d[z, r, theta]
        s_scan_3d = np.zeros((N_z, N_r, N_theta), dtype=np.float32)
        radial_locals = {}

        # Extract the Start/Range for the radial axis
        # start_k = beam_params[sample_beam]["start"]  # mm
        # range_k = beam_params[sample_beam]["range"]  # mm
        # Loop over beams
        for beam_key in beam_keys:
            beam_angle_deg = beam_params[beam_key]["refracted_angle"]

            radial_locals[beam_key] = self.build_local_radial_axis(
                beam_key, beam_params, num_radial_samples=num_radial_samples
            )
            # find the nearest angle index in angles_global
            angle_col = np.argmin(np.abs(angles_global - beam_angle_deg))

            start_k = beam_params[beam_key]["start"]  # mm
            range_k = beam_params[beam_key]["range"]  # mm
            # Loop over beams
            # ascan_2d: shape (N_z, N_samples)
            ascan_2d = ascans[beam_key]
            if ascan_2d.ndim == 1:
                # means we have no scanning axis, just do a single z
                ascan_2d = ascan_2d[np.newaxis, :]  # shape (1, N_samples)

            n_samples = ascan_2d.shape[1]

            beam_radial = np.linspace(0, start_k + range_k, n_samples, endpoint=False)

            # For each z, we interpolate that z's A-scan onto radial_global
            for z_idx in range(N_z):
                ascan_1d = ascan_2d[z_idx, :]
                f = interp1d(
                    beam_radial,
                    ascan_1d,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                beam_interp = f(radial_locals[beam_key])  # shape (N_r,)

                # Place in the final 3D array
                s_scan_3d[z_idx, :, angle_col] = beam_interp

        # we then invert the radial axis to have the near field at the top
        # radial_global = radial_global[::-1]

        # if beam_params[sample_beam]["raw_params"]["skewAngle"] == 0:
        #     # The default is 90, so we need to account for different skews
        #     s_scan_3d = s_scan_3d[:, ::-1, ::-1]
        #     # s_scan_3d = s_scan_3d[:, :, ::-1]

        return angles_global, radial_locals, s_scan_3d

    def parse_all_groups(
        self,
        groups_setup: list,
        groups_data: list[np.ndarray],
    ):
        # We shouldideally remove all of this logic from this function but it is currently useful for
        # prototyping this class as a standalone parser
        # THis is not reflected in the type annotations, as we dont want to enforce this behaviour
        if groups_data is None:
            try:
                # Attempt to load the data from the h5 file if not provided

                if self.file is None:
                    try:
                        self.file = self.read_h5()
                    except Exception as e:
                        raise ValueError(
                            f"Failed to load Data from NDE file: {e}. Directly passing a group data array is recommended"
                        )

                if groups_setup is None:
                    try:
                        groups_setup = self.parse_setup()
                        if groups_setup is None:
                            raise ValueError(
                                "Failed to load group setup from the h5 file. Directly passing a group data array is recommended"
                            )
                        groups_data = [
                            np.array(self.file[f"{group['datasets'][0]['path']}"])
                            for group in groups_setup
                        ]
                    except Exception as e:
                        raise ValueError(
                            f"Failed to load group setup from the h5 file: {e}. Directly passing a group data array is recommended"
                        )

            except Exception as e:
                raise ValueError(
                    f"Failed to load group data from the h5 file: {e}. Directly passing a group data array is recommended"
                )
        if groups_setup is None:
            return
        if groups_data is None:
            return

        self.amplitude_data_arrays = []
        self.s_scans_data_arrays = []
        self.extracted_group_data = []

        for group_setup in groups_setup:
            group_name = group_setup["name"]
            # The 'group_data' in HDF5 form (not yet converted to np.array).
            group_data = groups_data.pop(0)

            # We still store the HDF5 dataset object in amplitude_data_arrays
            self.amplitude_data_arrays.append(group_data)

            # Now call parse_group to do the central logic for a single group.
            parsed_dict = self.parse_group(group_setup, group_data)

            # We want to keep self.s_scans_data_arrays updated, so we append the "s_scan" from the parsed dict.
            self.s_scans_data_arrays.append(parsed_dict["s_scan"])

            # We also append the entire parsed_dict to extracted_group_data
            self.extracted_group_data.append(parsed_dict)

        return self.extracted_group_data

    # self.retrive_weld_param()
    # self.retrive_TOFD_param()
    # self.retrive_TOFD_Ascan()

    def parse_group(self, group_setup: dict, group_data: np.ndarray):
        """
        Parse a SINGLE group, returning a dictionary with all the data fields
        required to construct the S-scan and other useful information.



        """
        group_name = group_setup["name"]

        group_data_array = np.array(group_data)

        # Extract beam parameters, ascans, s-scan, etc.
        beam_param = self.extract_beam_parameters(group_setup)
        ascans = self.retrieve_ascans(group_setup, beam_param, group_data_array)
        angles, distances, s_scan = self.construct_s_scan(ascans, beam_param)

        specimen_characteristics = self.setup["specimens"]

        real_ascan_max = np.max(group_data_array)

        return {
            "id": group_setup["id"],
            "group_name": group_name,
            "ascans": ascans,
            "beam_param": beam_param,
            "s_scan": s_scan,
            "angles": angles,
            "distances": distances,
            "group_setup": group_setup,
            "real_amplitude_max": real_ascan_max,
        }

    def get_scan_array(self, ascans: dict, beam_param: dict, num_radial_samples=None):
        angles, distances, s_scan = self.construct_s_scan(
            ascans=ascans, beam_params=beam_param, num_radial_samples=num_radial_samples
        )
        # self.data_array = s_scan
        return s_scan

    @staticmethod
    def get_beam_position_attributes(parsed_group: dict) -> dict:
        # amplitude_data = parsed_group["ascans"]
        beam_angles = parsed_group["angles"]
        # step_res_mm = parsed_group["beam_param"][0]["ultrasound_resolution"]
        # radial_global = parsed_group["distances"]

        amplitude_data = parsed_group["ascans"]
        beam_angles = parsed_group["angles"]
        step_res_mm = parsed_group["beam_param"][0]["ultrasound_resolution"]
        radial_global = parsed_group["distances"]

        beam_starts = []
        beam_ranges = []
        beam_x_pos = []
        beam_params = parsed_group["beam_param"]
        beam_total_lengths = []
        for j in range(len(beam_angles)):
            params = beam_params[j]
            beam_starts.append(beam_params[j]["start"])  # e.g. 5.0 mm
            beam_ranges.append(beam_params[j]["range"])  # e.g. 35.0 mm
            if j == 0:
                beam_x_pos.append(0)
            else:
                beam_pos_range = (
                    abs(beam_params[0]["dimension_params"]["vCoordinateOffset"])
                    - abs(beam_params[j]["dimension_params"]["vCoordinateOffset"])
                ) * 1e3  # in mm
                beam_x_pos.append(beam_pos_range)
            beam_total_lengths.append(beam_ranges[j] + beam_starts[j])

        beam_attributes = {
            "beam_starts": beam_starts,
            "beam_ranges": beam_ranges,
            "beam_x_pos": beam_x_pos,
            "beam_total_lengths": beam_total_lengths,
            "beam_angles": beam_angles,
            "step_res_mm": step_res_mm,
        }
        return beam_attributes

    @staticmethod
    def create_cartesian_sscan_array(
        sscans: np.ndarray, parsed_group: dict
    ) -> tuple[np.ndarray, dict[int | str, SScanSliceInfo]]:
        cartesian_sscans_list = []
        slice_parameters_lookup: dict[str | int, SScanSliceInfo] = {}
        # invert the sscans so that the first beam is at the top

        # sscans = sscans[:, ::-1, :]+
        sscans = sscans[:, :, ::-1]
        for i in range(sscans.shape[0]):
            beam_positions_attributes = SectorialScanNDE.get_beam_position_attributes(
                parsed_group
            )

            s_scan_2d = sscans[i, :, :]

            # R, TH_deg = build_r_theta_mesh(
            #     s_scan_2d, angles_global, step_res_mm, r_start_mm=0.0
            # )

            # X, Z = polar_to_cartesian(R, TH_deg)

            # grid_x, grid_z, cart_img = s_scan_to_cartesian_image(
            #     s_scan_2d, X, Z, Nx=512, Nz=512
            # )

            beam_angles = beam_positions_attributes["beam_angles"]
            beam_ranges = beam_positions_attributes["beam_ranges"]
            beam_starts = beam_positions_attributes["beam_starts"]
            step_res_mm = beam_positions_attributes["step_res_mm"]
            grid_x, grid_z, cart_image, slice_info = (
                s_scan_to_cartesian_image_extremes_fast(
                    s_scan_2d,
                    beam_angles,
                    beam_ranges,
                    beam_starts,
                    mm_per_pixel=step_res_mm * 1e3,
                )
            )
            cartesian_sscans_list.append(cart_image)
            slice_parameters_lookup[i] = slice_info

        cartesian_sscans = np.array(cartesian_sscans_list)

        if cartesian_sscans.dtype != np.uint8:
            cartesian_sscans = (
                (cartesian_sscans - np.min(cartesian_sscans))
                / (np.max(cartesian_sscans) - np.min(cartesian_sscans))
                * 255
            )
            cartesian_sscans = cartesian_sscans.astype(np.uint8)
        return cartesian_sscans, slice_parameters_lookup


def s_scan_to_cartesian_image_extremes_fast(
    s_scan_2d,
    beam_angles: list,
    beam_ranges: list,
    beam_starts: list,  # not used in this snippet, but kept for signature
    mm_per_pixel=0.1,
    fill_value=0.0,
    method="linear",
    flip_z=False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SScanSliceInfo]:
    #!This function is messy, but highly optimized for speed
    #!More optmizations are possibe, as we currently
    #!Use a naive but prudent approach to mapping the data
    #!Original implementation speed on example dataset was 93 secs,
    #!this is now 8secs
    """
    Build a Cartesian image from an S-scan by:
      1) Building (X,Z) for each radial sample and beam (like before)
      2) Using mm_per_pixel to define a real-size grid in x,z
      3) Interpolating amplitude onto that grid
         BUT using map_coordinates instead of griddata for speed.

    Args:
      s_scan_2d : array of shape (N_r, N_theta)
                  The amplitude S-scan data in (radial, beam) form
      beam_angles: list of angles (deg) for each beam
      beam_ranges: list of ranges, used to get min/max for radius
      beam_starts: not used here, but kept for function signature compatibility
      mm_per_pixel: float, resolution in mm for both x and z
      fill_value  : amplitude for out-of-bounds
      method      : 'linear', 'nearest', or 'cubic' => map to interpolation order
      flip_z      : if True, we flip sign on z for typical UT usage

    Returns:
      grid_x  : 1D array of x-coordinates (length Nx)
      grid_z  : 1D array of z-coordinates (length Nz)
      cart_img: 2D array (Nz, Nx), amplitude in real-size Cartesian space
      slice_info: dict with the params that were used to create the image
    """

    # -- 1) Derive shape from s_scan_2d
    N_r, N_theta = s_scan_2d.shape

    # asuumes adius goes from min(beam_ranges) to max(beam_ranges),
    # and angles from min(beam_angles) to max(beam_angles).

    min_beam_range = min(beam_ranges)
    max_beam_range = max(beam_ranges)

    angle_min_deg = min(beam_angles)
    angle_max_deg = max(beam_angles)

    # 2)  build an approximate "inverse polar" mapping for each point (x,z).
    #    i.e. row_r = (r - r_min)/(r_max-r_min)*(N_r-1)
    #         col_t = (theta_deg - angle_min_deg)/(angle_max_deg - angle_min_deg)*(N_theta-1)

    r_min = float(min_beam_range)
    r_max = float(max_beam_range)
    if r_max <= r_min:
        raise ValueError("Invalid beam_ranges, cannot proceed (r_max <= r_min)")

    th_min_deg = float(angle_min_deg)
    th_max_deg = float(angle_max_deg)
    if th_max_deg <= th_min_deg:
        raise ValueError("Invalid beam_angles, cannot proceed (angle_max <= angle_min)")

    # 3) Determine bounding box in x,z
    #    assuming x ∈ [ -r_max, +r_max ], z ∈ [ -r_max, +r_max

    X = np.zeros((N_r, N_theta), dtype=np.float32)
    Z = np.zeros((N_r, N_theta), dtype=np.float32)

    beam_coords = {}

    for i in range(N_theta):
        angle_i_deg = beam_angles[i]
        angle_i_rad = np.deg2rad(angle_i_deg)

        # radial distances from min_beam_range..max_beam_range
        r_samples = np.linspace(r_min, r_max, N_r)

        # Explanation: For each beam, we calculate the x and z coordinates for each radial sample
        # which are derived from the beam's angle and the radial distance from the start of the beam

        for j in range(N_r):
            r_j = r_samples[j]

            X[j, i] = r_j * np.cos(angle_i_rad)
            Z[j, i] = r_j * np.sin(angle_i_rad)

        beam_start_coords = np.array([X[0, i], Z[0, i]])
        beam_end_coords = np.array([X[-1, i], Z[-1, i]])

        beam_coords[i] = {
            "start": {"x": beam_start_coords[0], "z": beam_start_coords[1]},
            "end": {"x": beam_end_coords[0], "z": beam_end_coords[1]},
        }

    x_min, x_max = X.min(), X.max()
    z_min, z_max = Z.min(), Z.max()

    dx = x_max - x_min
    dz = z_max - z_min
    if dx <= 0 or dz <= 0:
        raise ValueError("Invalid bounding box. Check your data or angles.")

    # If flip_z => we invert z bounding box
    if flip_z:
        z_min, z_max = -z_max, -z_min

    # Nx, Nz from the bounding box + mm_per_pixel
    Nx = max(1, int(dx / mm_per_pixel))
    Nz = max(1, int(dz / mm_per_pixel))

    grid_x = np.linspace(x_min, x_max, Nx)
    grid_z = np.linspace(z_min, z_max, Nz)

    # Create 2D mesh => shape (Nz, Nx) in typical "ij" indexing
    # working slow gtriddata impl uses indexing="ij" => mesh_x.shape=(Nx, Nz).
    # We'll do standard (mesh_x[r,c], mesh_z[r,c]) => shape=(Nz, Nx) if indexing='xy'
    mesh_x, mesh_z = np.meshgrid(grid_x, grid_z, indexing="ij")
    # shape => (Nz, Nx)

    # Flatten
    flat_x = mesh_x.ravel()
    flat_z = mesh_z.ravel()

    if flip_z:
        flat_z = -flat_z  # we flip the sign
    # 3) uses an 'inverse polar' approach to sample s_scan_2d via map_coordinates
    # but we must figure out how r, angle map to array indices (row=r, col=theta).

    # approx radial: r = sqrt(x^2 + z^2)
    rpix = np.sqrt(flat_x**2 + flat_z**2)

    # approx angle in deg: angle_deg = atan2(z, x)*180/pi
    # original code uses X= r*cos(angle), Z=r*sin(angle)
    # => angle= arctan2(Z, X). We'll do that in degrees:
    angle_pix_deg = np.degrees(np.arctan2(flat_z, flat_x))

    # map to row, col => row_r in [0..N_r-1], col_t in [0..N_theta-1]
    # row_r = (rpix - min_beam_range)/(max_beam_range - min_beam_range)*(N_r -1)
    row_r = (rpix - min_beam_range) / (max_beam_range - min_beam_range) * (N_r - 1)

    # col_t = (angle_pix_deg - angle_min_deg)/(angle_max_deg-angle_min_deg)*(N_theta-1)
    col_t = (
        (angle_pix_deg - angle_min_deg)
        / (angle_max_deg - angle_min_deg)
        * (N_theta - 1)
    )

    coords = np.vstack((row_r, col_t))

    # convert 'method' => map_coordinates order
    if method == "nearest":
        interp_order = 0
    elif method == "linear":
        interp_order = 1
    elif method == "cubic":
        interp_order = 3
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")

    # mask out-of-bounds
    valid_mask = (
        (row_r >= 0) & (row_r <= N_r - 1) & (col_t >= 0) & (col_t <= N_theta - 1)
    )
    out_vals = np.full_like(flat_x, fill_value, dtype=s_scan_2d.dtype)

    valid_idx = np.where(valid_mask)[0]
    valid_coords = coords[:, valid_idx]

    val_subset = map_coordinates(
        s_scan_2d, valid_coords, order=interp_order, cval=fill_value, mode="constant"
    )
    out_vals[valid_idx] = val_subset

    cart_img = out_vals.reshape(Nx, Nz)

    # reshape => (Nx, Nz)
    info: SScanSliceInfo = {
        "Nx": Nx,
        "Nz": Nz,
        "x_min": x_min,
        "x_max": x_max,
        "z_min": z_min,
        "z_max": z_max,
        "flip_z": flip_z,
        "angle_min_deg": angle_min_deg,
        "angle_max_deg": angle_max_deg,
        "r_min": r_min,
        "r_max": r_max,
        "beam_angles": beam_angles,
        "beam_ranges": beam_ranges,
        "beam_starts": beam_starts,
        "beam_coords": beam_coords,
    }
    # Return the 1D arrays + final image
    return grid_x, grid_z, cart_img, info


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

            # Heuristic: If V is significantly larger (>2x) than U, it's likely the intended scan axis
            # Otherwise prefer U to respect file structure for similar dimensions
            if v_qty > u_qty * 2.0:
                slice_axis_name = "VCoordinate"
                logger.debug(f"Using VCoordinate as slice axis (V={v_qty} >> U={u_qty})")
            else:
                slice_axis_name = "UCoordinate"
                logger.debug(f"Using UCoordinate as slice axis (Default preference)")
        
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
