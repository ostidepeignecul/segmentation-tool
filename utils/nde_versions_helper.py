import logging

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
                f"No u axis registration info in json setup at path groups/{self.raw_ascan_dataset_idx[gr]}/dimensions/0"
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
                f"No v axis registration info in json setup at path groups/{self.raw_ascan_dataset_idx[gr]}/dimensions/1"
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
                f"No w axis registration info in json setup at path groups/{self.raw_ascan_dataset_idx[gr]}/dimensions/2"
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
                "No velocity available in json setup at path specimens/0/plateGeometry/material/longitudinalWave/nominalVelocity"
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
                "No velocity available in json setup at path specimens/0/plateGeometry/material/longitudinalWave/nominalVelocity"
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
