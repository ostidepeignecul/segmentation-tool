import logging
import operator
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import spatial

from sentinelai.iohandler.io_module import IOModule
from sentinelai.models.Swin_UNETR.data_processing.flaw_sizing_methods.FlawSizingTools import (
    SizingDataset,
    extract_coordinates,
    get_blob,
)
from sentinelai.models.Swin_UNETR.swin_unetr_util import (
    add_flaw_names,
    generate_indication_table_and_labeled_array,
    one_hot_encode_numpy,
)
from sentinelai.plugins.inference_postprocess_hooks.postprocessing_plugin_manager import (
    FlawSizingData,
    PostprocessInput,
    PostprocessOutput,
    PostprocessPlugin,
)
from sentinelai.utils.generate_uid import UniqueIdentifier

logger = logging.getLogger(__name__)


# @register_postprocess_plugin(postprocess_plugin_registry)
class FlawSizingBase(PostprocessPlugin):
    def __init__(
        self, mask_adjustment_method: Callable, inference_id: str, csv_output: str
    ):
        self.mask_adjustment_method = mask_adjustment_method
        self.inference_id = inference_id
        self.exported_csv_filename = csv_output
        self.io_module = IOModule()

    def process_inference_result(
        self, post_process_input: PostprocessInput
    ) -> PostprocessOutput:
        """Process the inference result and return a modified inference array and indication table."""
        self.base_data_array = post_process_input.base_data_array
        base_inference_array = post_process_input.base_inference_array
        base_indication_table = post_process_input.base_indication_table
        status_info = post_process_input.status_info
        group_index = post_process_input.group_index
        metadata = post_process_input.metadata
        classes_names = post_process_input.classes_names
        self.sizing_stats = []

        self.adjusted_mask, flaw_sizing_log = self.generate_adjusted_mask(
            base_data_array=self.base_data_array,
            base_inference_array=base_inference_array,
            base_indication_table=base_indication_table,
            classes_names=classes_names,
        )
        adjusted_indications_table, labeled_array, dict_flaw_sizing_data = (
            self.generate_adjusted_indications(
                base_data_array=self.base_data_array,
                base_inference_array=base_inference_array,
                base_indication_table=base_indication_table,
                status_info=status_info,
                group_index=group_index,
                adjusted_mask=self.adjusted_mask,
                classes_names=classes_names,
            )
        )
        if post_process_input.calibration_flaw_ids is not None:
            if self._validate_calibration(post_process_input.calibration_flaw_ids):
                try:
                    adjusted_indications_table = add_flaw_names(
                        adjusted_indications_table,
                        post_process_input.calibration_flaw_ids,
                        status_info.get("gr", group_index),
                        status_info,
                    )
                    adjusted_indications_table = self.add_flaw_sizing_evaluation(
                        adjusted_indications_table
                    )
                    evaluation_columns = [
                        "flaw_names",
                        "SNR",
                        "Area (in2)",
                        "Area from px qty (in2)",
                        "Theoretical Area (in2)",
                        "Theoretical u size (in)",
                        "Target Area (in2)",
                        "Area from px qty (in2) Evaluation",
                        "Area (in2) Evaluation",
                        "Lengthwise Size (in) Evaluation",
                        "Lengthwise Start (in)",
                        "Crosswise Start (in)",
                        "Ultrasound Start (in)",
                        "Lengthwise Size (in)",
                        "Crosswise Size (in)",
                        "Ultrasound Size (in)",
                    ]
                    self.export_csvfile(
                        adjusted_indications_table[evaluation_columns],
                        "results_" + self.exported_csv_filename,
                        group_index=group_index,
                    )
                except Exception as e:
                    logger.error(f"Error in flaw sizing evaluation: {e}")
        else:
            adjusted_indications_table["flaw_names"] = adjusted_indications_table[
                "flaw_uid"
            ]

        self.export_csvfile(
            adjusted_indications_table,
            self.exported_csv_filename,
            group_index=group_index,
        )

        adjusted_indications_table.sort_values(
            by=["Lengthwise Start (mm)", "Lengthwise End (mm)"],
            inplace=True,
            ignore_index=True,
        )
        # If the adjusted mask has more than 3 dimensions, remove the first dimension
        # Hacky solution to the the fact that the adjust_mask function takes in a 4D array

        if len(self.adjusted_mask.shape) > 3:
            self.adjusted_mask = self.adjusted_mask[0]

        postprocess_output = PostprocessOutput(
            group_index=group_index,
            modified_array=self.adjusted_mask,
            modified_table=adjusted_indications_table,
            inference_id=self.inference_id,
            modified_labeled_array_lst=labeled_array,
            flaw_sizing_data=dict_flaw_sizing_data,
            flaw_sizing_log=flaw_sizing_log,
            metadata=metadata,
            classes_names=classes_names,
            dataset_id=post_process_input.dataset_id,
        )
        return postprocess_output

    def generate_adjusted_mask(
        self,
        base_data_array,
        base_inference_array,
        base_indication_table,
        classes_names,
    ):
        """Generate an adjusted mask."""
        # Add a dimension to the prediction array, so that the current array [x,y,z] becomes [1,x,y,z]
        # base_inference_array = np.expand_dims(base_inference_array, axis=0)
        n_classes = len(classes_names)
        base_inference_array = one_hot_encode_numpy(base_inference_array, n_classes)
        adjusted_mask = np.zeros_like(base_inference_array)
        flaw_sizing_log = ""

        for class_id, class_name in enumerate(classes_names):
            (
                adjusted_mask[class_id],
                class_specific_sizing_stats,
                class_specific_flaw_sizing_log,
            ) = self.mask_adjustment_method(
                data_array=base_data_array,
                pred_array=base_inference_array[class_id],
                indication_table=base_indication_table[
                    base_indication_table["label"] == class_name
                ],
                return_stats=True,
                class_name=class_name,
            )
            # Check if adjusted mask is not a tuple
            # if isinstance(adjusted_mask, tuple):
            #     adjusted_mask = adjusted_mask[0]
            self.sizing_stats.extend(class_specific_sizing_stats)
            flaw_sizing_log += class_specific_flaw_sizing_log + "\n"

        return adjusted_mask, flaw_sizing_log

    def generate_adjusted_indications(
        self,
        base_data_array,
        base_inference_array,
        base_indication_table,
        status_info,
        group_index,
        adjusted_mask,
        classes_names,
    ):
        """Process the inputs and return a modified inference array and indication table."""

        adjusted_indication_table, labeled = (
            generate_indication_table_and_labeled_array(
                status_info=status_info,
                one_hot_cleanedMask=adjusted_mask,
                data_array=base_data_array,
                group_index=group_index,
                classes_names=classes_names,
                add_stats=True,
            )
        )
        dict_flaw_sizing_data = None
        if self.sizing_stats:
            adjusted_indication_table, dict_flaw_sizing_data = self.add_analysis_stats(
                indication_table=adjusted_indication_table,
                analysis_stats=self.sizing_stats,
                classes_names=classes_names,
            )

        if adjusted_indication_table is None:
            adjusted_indication_table = base_indication_table

        return adjusted_indication_table, labeled, dict_flaw_sizing_data

    def add_analysis_stats(self, indication_table, analysis_stats, classes_names):
        # Complex merging of two tables based on the distance between old blob centroid and new blob centroid
        flaw_sizing_data = {}

        analysis_stats_to_add = [
            "FAA_SNR",
            "blob_min_value",
            "blob_max_value",
            "noise mean",
            "noise std",
            "noise median",
            "noise mad",
            "cscan_blob_min_value",
            "cscan_blob_max_value",
            "cscan noise mean",
            "cscan noise std",
            "cscan noise median",
            "cscan noise mad",
            "Boeing_SNR",
            "Boeing_SNR_mad",
            "Hypo_max_Boeing_SNR",
            "Hypo_max_Boeing_SNR_mad",
            "SNR_analysis_mean_ref_area",
            "SNR_analysis_std_ref_area",
            "SNR_analysis_median_ref_area",
            "SNR_analysis_mad_ref_area",
            "SNR_analysis_threshold",
            "SNR_analysis_mean_ref_area%",
            "SNR_analysis_std_ref_area%",
            "SNR_analysis_median_ref_area%",
            "SNR_analysis_mad_ref_area%",
            "SNR_analysis_threshold%",
        ]

        df_analysis_stats = pd.DataFrame.from_dict(analysis_stats)

        new_adjusted_indication_table_lst = []

        for class_id, class_name in enumerate(classes_names):
            class_specific_all_centroids = df_analysis_stats[
                df_analysis_stats["label"] == class_name
            ]["centroids"].to_list()
            tree = spatial.KDTree(class_specific_all_centroids)

            class_specific_new_indication_table = indication_table[
                indication_table["label"] == class_name
            ].copy()
            class_specific_new_indication_table.reset_index(drop=True, inplace=True)
            for stat_name in analysis_stats_to_add:
                if stat_name in analysis_stats[0].keys():
                    class_specific_new_indication_table[stat_name] = np.nan

            new_ds = SizingDataset(
                data_array=self.base_data_array,
                pred_array=self.adjusted_mask[class_id],
                indication_table=class_specific_new_indication_table,
                compute_dilated=False,
            )

            for new_blob_id in class_specific_new_indication_table.index:
                print("new blob id :", new_blob_id)
                blob_to_identify = (
                    class_specific_new_indication_table.at[new_blob_id, "center_x"],
                    class_specific_new_indication_table.at[new_blob_id, "center_y"],
                    class_specific_new_indication_table.at[new_blob_id, "center_z"],
                )
                _, old_blob_id = tree.query(blob_to_identify)
                for stat_name in analysis_stats_to_add:
                    if stat_name in list(df_analysis_stats.columns):
                        class_specific_new_indication_table.at[
                            new_blob_id, stat_name
                        ] = round(df_analysis_stats.at[old_blob_id, stat_name], 3)

                new_blob, new_blob_location = get_blob(
                    flaw_id=new_blob_id,
                    ds=new_ds,
                    dilated=False,
                    fill_outside_with_nan=True,
                )
                new_blob_coords = extract_coordinates(
                    new_blob,
                    [
                        new_blob_location["u_start"],
                        new_blob_location["v_start"],
                        new_blob_location["w_start"],
                    ],
                )

                flaw_uid = UniqueIdentifier.get_uid()
                class_specific_new_indication_table.at[new_blob_id, "flaw_uid"] = (
                    flaw_uid
                )
                stat = self.sizing_stats[old_blob_id]
                try:
                    flaw_sizing_data[flaw_uid] = self.get_flaw_sizing_data(
                        stat,
                        new_blob,
                        (
                            new_blob_location["u_start"],
                            new_blob_location["v_start"],
                            new_blob_location["w_start"],
                        ),
                        new_blob_coords,
                    )
                except:
                    print("Did not find noise_array key.")
                    print("stat keys :", stat.keys())
            new_adjusted_indication_table_lst.append(
                class_specific_new_indication_table
            )

        # Join tables of each class into a unique table
        new_adjusted_indication_table = pd.concat(new_adjusted_indication_table_lst)
        new_adjusted_indication_table.reset_index(drop=True, inplace=True)

        return new_adjusted_indication_table, flaw_sizing_data

    def get_flaw_sizing_data(
        self, stat, new_blob, new_blob_location, new_blob_coords
    ) -> FlawSizingData:
        return FlawSizingData(
            parent_flaw_id=stat["parent_flaw_uid"],
            blob_array=stat["blob_array"],
            blob_coordinates=stat["blob_coordinates"],
            blob_location_absolute=stat["blob_location"],
            # Patches  a bug where the noise array seems to be missing
            noise_array=stat.get("noise_array", np.zeros_like(stat["blob_array"])),
            noise_coordinates=stat["noise_coordinates"],
            noise_location_absolute=stat["noise_location"],
            revised_blob_array=new_blob,
            revised_blob_location_absolute=new_blob_location,
            revised_blob_coordinates=new_blob_coords,
        )

    def export_csvfile(self, dataframe_to_export, filename, group_index) -> None:
        # filename = eval(f'f"{filename}"', locals, globals)

        # This is just a quick fix to remove the eval, so it uses string formatting with assumptions
        # Check if the  self.inference_id starts with flaw_sizing, if so create a string without it
        short_inference_id = self.inference_id
        if self.inference_id.startswith("flaw_sizing"):
            short_inference_id = self.inference_id[12:]
        filename = "indication_table_" + str(group_index) + "_" + short_inference_id
        path = self.io_module.generate_path("flaw_sizing_results")
        if path is not None and path.exists():
            self.io_module.save_csv(
                dir_path=path, file_name=filename, data=dataframe_to_export
            )

    def add_flaw_sizing_metric_comparison(
        self, indic_table, result_metric_name, target_metric_name, objective=operator.gt
    ) -> pd.DataFrame:
        indic_table[result_metric_name + " Evaluation"] = "na"
        try:
            indic_table = indic_table.astype(
                {result_metric_name: "str", target_metric_name: "str"}
            )
            rows_idx_with_values = indic_table[
                indic_table[result_metric_name].str.replace(".", "").str.isnumeric()
                & indic_table[target_metric_name].str.replace(".", "").str.isnumeric()
            ].index
            indic_table.loc[
                rows_idx_with_values,
                result_metric_name + " Evaluation",
            ] = objective(
                indic_table.loc[rows_idx_with_values, result_metric_name],
                indic_table.loc[rows_idx_with_values, target_metric_name],
            )
        except Exception as e:
            print(e)
        return indic_table

    def add_target_area_in2(
        self, indic_table: pd.DataFrame, factor=0.75
    ) -> pd.DataFrame:
        indic_table["target_area"] = "na"
        indic_table = indic_table.astype({"theoretical_area": "str"})
        rows_idx_with_value = indic_table[
            indic_table["theoretical_area"].str.replace(".", "").str.isnumeric()
        ].index
        indic_table.loc[rows_idx_with_value, "target_area"] = round(
            indic_table["theoretical_area"].loc[rows_idx_with_value].astype(float)
            * factor,
            3,
        )
        return indic_table

    def add_flaw_sizing_evaluation(self, indic_table) -> pd.DataFrame:
        # Calculation of the target Area (corresponding to 0.75 x Theoretical area)
        indic_table = self.add_target_area_in2(indic_table)
        # Evaluation of Area when calculated from pixel qty on C-Scan (must be greater than target)
        indic_table = self.add_flaw_sizing_metric_comparison(
            indic_table, "Area from px qty (in2)", "target_area", operator.gt
        )
        # Evaluation of Area when calculated with bounding box of connected components (must be greater than target)
        indic_table = self.add_flaw_sizing_metric_comparison(
            indic_table, "Area (in2)", "target_area", operator.gt
        )
        # Evaluation of length of the flaw (must be greater (?) than theoretical length)
        indic_table = self.add_flaw_sizing_metric_comparison(
            indic_table, "Lengthwise Size (in)", "theoretical_u_size", operator.gt
        )
        indic_table.rename(
            columns={
                "target_area": "Target Area (in2)",
                "theoretical_area": "Theoretical Area (in2)",
                "theoretical_u_size": "Theoretical u size (in)",
                "theoretical_v_size": "Theoretical v size (in)",
                "SNR_analysis_mean_ref_area": "SNR Analysis Mean Ref Area",
                "SNR_analysis_std_ref_area": "SNR Analysis Std Ref Area",
                "SNR_analysis_median_ref_area": "SNR Analysis Median Ref Area",
                "SNR_analysis_mad_ref_area": "SNR Analysis Mad Ref Area",
                "SNR_analysis_threshold": "SNR Analysis Threshold",
                "SNR_analysis_mean_ref_area%": "SNR Analysis Mean Ref Area (%)",
                "SNR_analysis_std_ref_area%": "SNR Analysis Std Ref Area (%)",
                "SNR_analysis_median_ref_area%": "SNR Analysis Median Ref Area (%)",
                "SNR_analysis_mad_ref_area%": "SNR Analysis Mad Ref Area (%)",
                "SNR_analysis_threshold%": "SNR Analysis Threshold (%)",
            },
            inplace=True,
        )

        return indic_table

    def _validate_calibration(self, calibration_df: pd.DataFrame) -> bool:
        """Checks if the columns required are present in the input csv
        Args:
            calibration_df (pd.DataFrame): The calibration dataframe"""

        columns = ["theoretical_area", "theoretical_u_size"]

        for column in columns:
            if column not in calibration_df.columns:
                logger.warning(
                    f"Column {column} is missing in the calibration csv, could not perform flaw sizing."
                )
                return False

        return True
