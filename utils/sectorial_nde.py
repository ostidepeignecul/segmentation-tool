import json
from pathlib import Path
from typing import TypedDict

import h5py  # package for reading hdf5 files
import numpy as np
from scipy.interpolate import interp1d

#!TODO: Currently, the sectorial scan is simply interpolated to a Nz, Nx grid (see s_scan_to_cartesian_image func) without
#!TODO: Any consideration for the actual geometry of the scan. The cartesian mapping is also very slow and needs to be optimized
#!TODO: For any real use case.
from scipy.ndimage import map_coordinates


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


def get_physical_radian_sscan(pos_x: int, pos_y: int, info: SScanSliceInfo):
    """
    Given a pixel index (pos_x, pos_y) in the final Cartesian array of shape (Nx, Nz) in mm,
    plus the typed-dict 'info' that was used in 's_scan_to_cartesian_image_extremes_fast',
    return:

      - x_mm, y_mm:  the physical coordinates in mm
      - angle_deg:   the approximate beam angle at that pixel
      - r_mm:        the radial distance, if you want it

    This is the 'reverse lookup' of the forward transform.

    Args:
      pos_x, pos_y: integer indices in [0..info["Nx"]-1], [0..info["Nz"]-1],
                    basically the index position in the slice
      info        : typed dict SScanSliceInfo containing:
                      Nx, Nz,
                      x_min, x_max,
                      z_min, z_max,
                      flip_z,
                      angle_min_deg, angle_max_deg,
                      r_min, r_max
    Returns:
      (x_mm, y_mm), (angle_pix_deg), r_pix_mm
    """

    Nx = info["Nx"]
    Nz = info["Nz"]
    x_min = info["x_min"]
    x_max = info["x_max"]
    z_min = info["z_min"]
    z_max = info["z_max"]
    flip_z = info["flip_z"]
    angle_min_deg = info["angle_min_deg"]
    angle_max_deg = info["angle_max_deg"]
    r_min = info["r_min"]
    r_max = info["r_max"]

    # 1) compute x, y in mm from pixel indices
    #    Nx -> horizontal dimension, Nz -> vertical dimension
    #    pos_x in [0..Nx-1], pos_y in [0..Nz-1]
    if Nx > 1:
        x_mm = x_min + (pos_x / (Nx - 1)) * (x_max - x_min)
    else:
        x_mm = x_min

    if Nz > 1:
        y_mm = z_min + (pos_y / (Nz - 1)) * (z_max - z_min)
    else:
        y_mm = z_min

    # If we had flip_z, we invert y_mm
    if flip_z:
        y_mm = -y_mm

    # 2) approximate the angle via atan2(y, x) => degrees
    angle_pix_deg = np.degrees(np.arctan2(y_mm, x_mm))

    # 3) approximate radial distance
    r_pix_mm = np.sqrt(x_mm**2 + y_mm**2)

    # Optionally clamp angle to [angle_min_deg..angle_max_deg]
    angle_pix_deg = np.clip(angle_pix_deg, angle_min_deg, angle_max_deg)

    return (x_mm, y_mm), (angle_pix_deg), r_pix_mm


def get_virtual_radian_sscan(pos_x: int, pos_y: int, slice_info: SScanSliceInfo):
    """
    Similar to the physical version, but returns a radial min and max coordinates in terms of index space
    that allows to draw the radial distance regardless of the physical distance.
    and the returned x,y are the input x, y (both are index wise, not in mm) but clipped to the closest actual pixel.
    It also returns a total radial span in terms of start-end of the entire beam at that angle,
    but in (x0, y0), (x1, y1) format. This is useful to draw the extent of the beam at that angle.



    Returns:
      dict: {
        "clipped_position": (clipped_x_idx, clipped_y_idx),
        "angle": (int(angle_pix_deg)),
        "beam_extent": {
            "start": (x0_beam, y0_beam),
            "end": (x1_beam, y1_beam),
      }

    """

    Nx = slice_info["Nx"]
    Nz = slice_info["Nz"]
    x_min = slice_info["x_min"]
    x_max = slice_info["x_max"]
    z_min = slice_info["z_min"]
    z_max = slice_info["z_max"]
    flip_z = slice_info["flip_z"]
    angle_min_deg = slice_info["angle_min_deg"]
    angle_max_deg = slice_info["angle_max_deg"]
    r_min = slice_info["r_min"]
    r_max = slice_info["r_max"]
    beam_starts = slice_info["beam_starts"]
    beam_ranges = slice_info["beam_ranges"]
    beam_angles = slice_info["beam_angles"]

    beam_coords = slice_info["beam_coords"]

    # 1) compute x, y in mm from pixel indices
    #    Nx -> horizontal dimension, Nz -> vertical dimension
    #    pos_x in [0..Nx-1], pos_y in [0..Nz-1]
    if Nx > 1:
        x_mm = x_min + (pos_x / (Nx - 1)) * (x_max - x_min)
    else:
        x_mm = x_min

    if Nz > 1:
        y_mm = z_min + (pos_y / (Nz - 1)) * (z_max - z_min)
    else:
        y_mm = z_min

    # If we had flip_z, we invert y_mm
    if flip_z:
        y_mm = -y_mm

    # 2) approximate the angle via atan2(y, x) => degrees
    angle_pix_deg = np.degrees(np.arctan2(y_mm, x_mm))

    # 3) approximate radial distance
    r_pix_mm = np.sqrt(x_mm**2 + y_mm**2)

    # Optionally clamp angle to [angle_min_deg..angle_max_deg]
    angle_pix_deg = int(np.clip(angle_pix_deg, angle_min_deg, angle_max_deg))
    indx_angle = list(beam_angles).index(angle_pix_deg)

    beam_start = beam_starts[indx_angle]
    beam_range = beam_ranges[indx_angle]

    # We then convert the x_mm and y_mm to the closest pixel index
    clipped_x_idx = np.clip(int((x_mm - x_min) / (x_max - x_min) * (Nx - 1)), 0, Nx - 1)
    clipped_y_idx = np.clip(int((y_mm - z_min) / (z_max - z_min) * (Nz - 1)), 0, Nz - 1)

    # We get the radial distance of that pos in indx
    r_pix_mm = np.sqrt(x_mm**2 + y_mm**2)
    r_pix_mm = np.clip(r_pix_mm, r_min, r_max)
    # we want the start and end of the r_pix_mm in index space
    # the r_pix_mm is the radian distance to that specific position, starting from 0.
    # where as the x0_beam, y0_beam, x1_beam, y1_beam are the start and end of the entire beam at that angle

    beam_start_coords = beam_coords[indx_angle]["start"]
    beam_end_coords = beam_coords[indx_angle]["end"]
    x_start_mm = beam_start_coords["x"]
    y_start_mm = beam_start_coords["z"]

    x_end_mm = beam_end_coords["x"]
    y_end_mm = beam_end_coords["z"]

    x0_beam = np.clip(int((x_start_mm - x_min) / (x_max - x_min) * (Nx - 1)), 0, Nx - 1)

    y0_beam = np.clip(int((y_start_mm - z_min) / (z_max - z_min) * (Nz - 1)), 0, Nz - 1)

    x1_beam = np.clip(int((x_end_mm - x_min) / (x_max - x_min) * (Nx - 1)), 0, Nx - 1)

    y1_beam = np.clip(int((y_end_mm - z_min) / (z_max - z_min) * (Nz - 1)), 0, Nz - 1)

    index_sscan_data = {
        "clipped_position": (clipped_x_idx, clipped_y_idx),
        "angle": (int(angle_pix_deg)),
        "beam_extent": {
            "start": (x0_beam, y0_beam),
            "end": (x1_beam, y1_beam),
        },
    }

    return index_sscan_data
