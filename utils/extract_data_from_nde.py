import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
from h5py import File

from sentinelai.utils.nde_processing.fmc_nde import reshape_pulser_receiver
from sentinelai.utils.nde_processing.nde_versions_helper import (
    Unistatus_NDE_3_0_0,
    Unistatus_NDE_4_0_0_Dev,
    get_unistatus,
)
from sentinelai.utils.nde_processing.sectorial_nde import (
    SectorialScanNDE,
)

logger = logging.getLogger(__name__)
Unistatus = Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev


# if twe start supporting more than just zero deg and sextorial scan
# we might need to implement a  more standardized way to access the NDEGroupData
# output, so that we can do any processing we want for each case
class NDEGroupData:
    def __init__(
        self,
        f: File,
        unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev,
        group_id: int,
    ):
        self.set_data_array(f, unistatus, group_id)
        self.set_status_info(unistatus, group_id)

    def set_data_array(
        self,
        f: File,
        unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev,
        group_id: int,
    ):
        path = unistatus.get_path_to_data(group_id)
        self.data_array = f[path][:]  # type: ignore

    def set_status_info(
        self, unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev, group_id: int
    ):
        self.status_info = {}
        self.status_info["gr"] = group_id
        self.status_info["group_name"] = unistatus.get_group_name(group_id)
        self.status_info["min_value"] = unistatus.get_amplitude_min(group_id)
        self.status_info["max_value"] = unistatus.get_amplitude_max(group_id)
        (
            self.status_info["number_files_input"],
            self.status_info["img_height_px"],
            self.status_info["img_width_px"],
        ) = np.shape(
            self.data_array  # type: ignore
        )  # type: ignore
        self.status_info["lengthwise"] = unistatus.get_u_axis_reg_info(group_id)
        self.status_info["crosswise"] = unistatus.get_v_axis_reg_info(group_id)
        self.status_info["ultrasound"] = unistatus.get_w_axis_reg_info(group_id)

        # Conversion s (round-trip) -> meters for ultrasound infos
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

        # Static mapping of position/index
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
    # Empty as it is just the current GroupData, but this allows us to differentiate between the two
    # WHile still allowing the set_status_info to be shared

    pass


class NDEGroupDataSectorialScan(NDEGroupData):
    def __init__(
        self,
        f: File,
        unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev,
        group_id: int,
        setup_json: dict,
    ):
        self.unistatus = unistatus
        self.group_id = group_id
        self.setup = setup_json
        self.slice_info_lookup = {}
        self.file = f
        self.set_data_array(f, unistatus, group_id)
        self.set_status_info(unistatus, group_id)

    def get_group_dict(self, setup_json_dict: dict) -> dict:
        self.setup = setup_json_dict

        groups = setup_json_dict["groups"]

        return groups

    def get_data_array(
        self,
        f: File,
        unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev,
        group_id: int,
    ) -> np.ndarray:
        # The sectorial scan ndes we have seem to have their real data group in id=0m
        # whereas we usually start from one. so we need to decrement the group_id
        path = unistatus.get_path_to_data(group_id)
        return np.array(f[path][:])  # type: ignore

    def set_data_array(
        self,
        f: File,
        unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev,
        group_id: int,
    ):
        """Implements the custom data processing logic for sectorial scans"""

        # the function to get the group data is from the 1 indexed group_id
        # so we dont need to decrement it for self.get_data_array

        group_data = self.get_data_array(f, unistatus, group_id)

        # However, It seems like  the NDE itself is 0 indexed,
        # So the sectorial parsing was built with that in mind.
        # but the other part of the input pipeline are 1 indexed. So we need to subtract 1 from the group_id
        # if group_id != 0:
        #     # This is a bit wonky, as it silently decrements the group_id but
        #     # still implicitly accepts a 0 indexed group_id if its 0. Whereas
        #     # we should rather either make it 0 everuwjere, or enforce that the input index in this func
        #     # is 1 indexed (by not accepting 0 as a valid input)
        #     group_id -= 1

        time1 = time.time()
        logger.info(" Sectorial Scan Data Reconstruction Started")

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
            f" Sectorial Scan Data Reconstruction Finished in {time.time() - time1:.2f} seconds"
        )

    def set_status_info(
        self, unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev, group_id: int
    ):
        status_info = super().set_status_info(unistatus, group_id)
        # the slice info uses the z (in 3d coordinates) axis as the index key
        # This can be provided as
        # We currently assume that all sscans in an acquisition have the same slice info
        # meaning  the same beams, beam properties, etc
        status_info["sscan_slice_info"] = self.slice_info_lookup[0]
        # In the radians sscan data, the ascans are organized by beam. Meaning that each beam
        # has a dict, with the z index (lengthwise) as the key, and the ascan data as the value for that index
        status_info["radian_sscan_data"] = self.radian_sscan_data

        self.status_info = status_info

        return self.status_info


class NDEGroupDataTFM(NDEGroupData):
    def get_data_array(
        self,
        f: File,
        unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev,
        group_id: int,
    ) -> np.ndarray:
        path = unistatus.get_path_to_data(group_id)
        arr = np.array(f[path][:])  # type: ignore
        # The tfm seems to have an array that is z,y,x instead of z,x,y
        # So we need to transpose it
        return np.transpose(arr, (0, 2, 1))

    def set_data_array(
        self,
        f: File,
        unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev,
        group_id: int,
    ):
        self.data_array = self.get_data_array(f, unistatus, group_id)
        # return self.data_array


class NDEGroupDataFMC(NDEGroupData):
    def __init__(
        self,
        f: File,
        unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev,
        group_id: int,
    ):
        self.group_setups = self.get_group_dict(unistatus.json_decoded)
        super().__init__(f, unistatus, group_id)

    def get_group_dict(self, setup_json_dict: dict) -> dict:
        self.setup = setup_json_dict

        groups = setup_json_dict["groups"]

        return groups

    def get_data_array(
        self,
        f: File,
        unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev,
        group_id: int,
    ) -> np.ndarray:
        path = unistatus.get_path_to_data(group_id)
        arr = np.array(f[path][:])  # type: ignore

        # FMC data is not 3dimensional, but is instead composed of
        # a 2d array with stacked ascans. That means that the first axis
        # is the number of physical positions (typically 1),
        # and the second position is a stacked ascan that is constructed as such:
        #  First Pulser (Supports different receivers for each pulser)-> Its receivers-> Every receivers Ascans
        #  Second Pulser -> Its receivers -> Every receivers Ascans
        # All stacked in a single dimension.
        beams = self.group_setups[group_id]["processes"][0]["ultrasonicMatrixCapture"][
            "beams"
        ]
        ascan_qtty = self.group_setups[group_id]["datasets"][0]["dimensions"][1][
            "quantity"
        ]
        tx_rx_mapping = {}
        for beam in beams:
            tx_rx_mapping[beam["pulsers"][0]["id"]] = beam["receivers"]

        len_receivers = len(tx_rx_mapping[0])
        len_pulsers = len(tx_rx_mapping)

        # Since the  array is shaped as U*StackedAscans, we need to reshape it to U*Pulsers*Receivers*AscanLength
        # Eventually, we can apply  transformations to turn FMC into TFM or other acquisition methods.
        # For now, we reshape the array so that it is a 3d array that contains "blocks" of pulser*receiver*ascans
        # so the first axis is pulsers, receivers, and ascan length but with aall the pulsers frm all U

        data_shape = arr.shape
        # # create a new array that has the shape
        # stacked_array = np.zeros(
        #     (data_shape[0], len_pulsers, len_receivers, ascan_qtty)
        # )
        # for u in range(data_shape[0]):
        #     stacked_array[u] = reshape_pulser_receiver(
        #         arr[u], len_pulsers, len_receivers
        #     )

        stacked_array_list = reshape_pulser_receiver(arr, len_pulsers, len_receivers)
        stacked_array = np.array(stacked_array_list, dtype=np.int16)

        # by stacking the arrays. meaning that if there are 60 pulsers and 3 u,
        # the first  pulser of the 3rd u will be at index 180, and the u axis wil lbe remove
        stacked_array = np.vstack(stacked_array, dtype=np.int16)  # type: ignore

        return stacked_array

    def set_data_array(
        self,
        f: File,
        unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev,
        group_id: int,
    ):
        self.data_array = self.get_data_array(f, unistatus, group_id)

    def set_status_info(
        self, unistatus: Unistatus_NDE_3_0_0 | Unistatus_NDE_4_0_0_Dev, group_id: int
    ):
        self.status_info = {}
        self.status_info["gr"] = group_id
        self.status_info["group_name"] = unistatus.get_group_name(group_id)
        self.status_info["min_value"] = unistatus.get_amplitude_min(group_id)
        self.status_info["max_value"] = unistatus.get_amplitude_max(group_id)

        # Conversion s (round-trip) -> meters for ultrasound infos

        return self.status_info


class NDEDataTypeCheck:
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
                            if "sectorialFormation" in pulseEcho:
                                return True
                            elif "compoundFormation" in pulseEcho:
                                return True
        return False

    @staticmethod
    def is_tfm(setup: dict, group_index: int = 0) -> bool:
        valid_datasets = {
            idx: group
            for idx, group in enumerate(setup["groups"])
            if "datasets" in group
        }

        group_classes = []

        supported_classes = ["AScanAmplitude", "TfmValue"]

        for idx, group in valid_datasets.items():
            # if there are more than one datasets in the group, we take the first one
            if len(group["datasets"]) >= 1:
                data_class = group["datasets"][0]["dataClass"]
            else:
                data_class = group["datasets"]["dataClass"]

            if data_class in supported_classes:
                group_classes.append(data_class)

        if "TfmValue" in group_classes:
            return True
        return False

    @staticmethod
    def is_fmc(setup: dict, group_index: int = 0) -> bool:
        valid_datasets = {
            idx: group
            for idx, group in enumerate(setup["groups"])
            if "datasets" in group
        }

        for idx, group in valid_datasets.items():
            if "processes" in group:
                for process in group["processes"]:
                    if "ultrasonicMatrixCapture" in process:
                        if "acquisitionPattern" in process["ultrasonicMatrixCapture"]:
                            if (
                                "FMC"
                                in process["ultrasonicMatrixCapture"][
                                    "acquisitionPattern"
                                ]
                            ):
                                logger.info("FMC dataset detected")
                                return True
        return False


# Function to obtain a dictionary with infos that we need in all the nde_path in the list
def data_from_nde(list_of_nde_path: list, group_list=[], verbose=True):
    # Creation of a main dictionary of all nde_files infos that we need

    files_dict = {}

    for nde_file in list_of_nde_path:
        # If the nde_file is a Path object, convert it to a string
        # this is useful as most of the codebase that uses this function uses Path objects
        if isinstance(nde_file, Path):
            nde_file = nde_file.as_posix()

        print(f"nde file : {nde_file}. (key : {nde_file[-11:-4]})")
        with h5py.File(nde_file, "r") as f:
            # get and decode the json file about the configuration
            path_to_json = "Public/Setup" if "Public" in f.keys() else "Domain/Setup"
            json_str = f[path_to_json][()]  # type: ignore
            json_decoded = json.loads(json_str)  # type: ignore

            is_sectorial_scan = NDEDataTypeCheck.is_sectorial_scan(json_decoded)
            is_tfm = NDEDataTypeCheck.is_tfm(json_decoded)
            is_fmc = NDEDataTypeCheck.is_fmc(json_decoded)

            unistatus = get_unistatus(json_decoded)

            if unistatus:
                data = {}

                if not group_list:
                    group_list = unistatus.get_n_group()

                # the rest  of the app expects a 1 indexed group list
                i = 1

                for gr in group_list:
                    if is_sectorial_scan:
                        nde_groupdata = NDEGroupDataSectorialScan(
                            f=f,
                            unistatus=unistatus,
                            group_id=gr,
                            setup_json=json_decoded,
                        )
                        acquisition_attributes = {
                            "processes": ["sectorial_scan"],
                        }
                    elif is_tfm:
                        nde_groupdata = NDEGroupDataTFM(
                            f=f, unistatus=unistatus, group_id=gr
                        )
                        acquisition_attributes = {
                            "processes": ["tfm"],
                        }
                    elif is_fmc:
                        nde_groupdata = NDEGroupDataFMC(
                            f=f, unistatus=unistatus, group_id=gr
                        )
                        acquisition_attributes = {
                            "processes": ["fmc"],
                        }
                    else:
                        nde_groupdata = NDEGroupDataZeroDegUT(
                            f=f, unistatus=unistatus, group_id=gr
                        )
                        acquisition_attributes = {
                            "processes": ["zero_deg"],
                        }

                    # Since most of the rest of the app only cares about
                    # the real groups (groups with data),
                    # we dont use the gr from the group_list,
                    # as it is a list with the real indexes (for example [1,3,5,7])
                    # of only groups with data.
                    data[i] = {}
                    data[i]["data_array"] = nde_groupdata.data_array
                    data[i]["status_info"] = nde_groupdata.status_info
                    data[i]["status_info"]["acquisition_attributes"] = (
                        acquisition_attributes
                    )
                    i += 1
                    # data[gr]["nde_json"] = json_decoded

                # # Keep data dictionary into main dictionary of all nde_files
                # data["status"] = json_decoded
                # data["version"] = json_decoded["version"]
                nde_file_path = Path(nde_file)
                files_dict[f"{nde_file_path.stem}"] = data
            else:
                logger.error(
                    f"Unable to get unistatus from {nde_file}. Please check the file format version."
                )
                continue

    return files_dict
