import numpy as np
import pandas as pd
from scipy import spatial

from sentinelai.models.Swin_UNETR.data_processing.flaw_sizing_methods.maskSizing_v3_cropOnV import (
    adjust_mask,
)
from sentinelai.models.Swin_UNETR.swin_unetr_util import (
    generate_indication_table_and_labeled_array,
)

from ..postprocessing_plugin_manager import (
    PostprocessInput,
    PostprocessOutput,
    PostprocessPlugin,
    postprocess_plugin_registry,
    register_postprocess_plugin,
)


@register_postprocess_plugin(postprocess_plugin_registry)
class FlawSizingv3_SNR2_5_cropOnV(PostprocessPlugin):
    def process_inference_result(
        self, post_process_input: PostprocessInput
    ) -> PostprocessOutput:
        """Process the inference result and return a modified inference array and indication table."""
        self.inference_id = "flaw_sizing_v3 SNR=2.5 CropOnV"
        base_data_array = post_process_input.base_data_array
        base_inference_array = post_process_input.base_inference_array
        base_indication_table = post_process_input.base_indication_table
        status_info = post_process_input.status_info
        group_index = post_process_input.group_index
        metadata = post_process_input.metadata

        adjusted_mask = self.generate_adjusted_mask(
            base_data_array=base_data_array,
            base_inference_array=base_inference_array,
            base_indication_table=base_indication_table,
        )
        adjusted_indications_table = self.generate_adjusted_indications(
            base_data_array=base_data_array,
            base_inference_array=base_inference_array,
            base_indication_table=base_indication_table,
            status_info=status_info,
            group_index=group_index,
            adjusted_mask=adjusted_mask,
        )
        adjusted_indications_table.sort_values(
            by=["Lengthwise Start (mm)", "Lengthwise End (mm)"],
            inplace=True,
            ignore_index=True,
        )
        # If the adjusted mask has more than 3 dimensions, remove the first dimension
        # Hacky solution to the the fact that the adjust_mask function takes in a 4D array

        if len(adjusted_mask.shape) > 3:
            adjusted_mask = adjusted_mask[0]

        postprocess_output = PostprocessOutput(
            group_index=group_index,
            modified_array=adjusted_mask,
            modified_table=adjusted_indications_table,
            inference_id=self.inference_id,
            metadata=metadata,
            dataset_id=post_process_input.dataset_id,
        )
        return postprocess_output

    def generate_adjusted_mask(
        self, base_data_array, base_inference_array, base_indication_table
    ):
        """Generate an adjusted mask."""
        # Add a dimension to the prediction array, so that the current array [x,y,z] becomes [1,x,y,z]
        base_inference_array = np.expand_dims(base_inference_array, axis=0)

        adjusted_mask, self.sizing_stats = adjust_mask(
            data_array=base_data_array,
            pred_array=base_inference_array,
            indication_table=base_indication_table,
            keep_largest_blob=False,
            u_window_half_length=15,
            dilation_max_iteration=10,
            return_stats=True,
            K=2.5,
        )
        # Check if adjusted mask is not a tuple
        if isinstance(adjusted_mask, tuple):
            adjusted_mask = adjusted_mask[0]
        return adjusted_mask

    def generate_adjusted_indications(
        self,
        base_data_array,
        base_inference_array,
        base_indication_table,
        status_info,
        group_index,
        adjusted_mask,
        classes_names="flaw_label1.0",
    ):
        """Process the inputs and return a modified inference array and indication table."""

        adjusted_indication_table, _ = generate_indication_table_and_labeled_array(
            status_info=status_info,
            one_hot_cleanedMask=adjusted_mask,
            data_array=base_data_array,
            group_index=group_index,
            classes_names=classes_names,
            add_stats=True,
        )

        if self.sizing_stats:
            adjusted_indication_table = self.add_analysis_stats(
                indication_table=adjusted_indication_table,
                analysis_stats=self.sizing_stats,
            )

        if adjusted_indication_table is None:
            adjusted_indication_table = base_indication_table

        adjusted_indication_table.to_csv(
            f"./indication_table_{group_index}_SNR=2_5_CropOnV.csv"
        )

        return adjusted_indication_table

    def add_analysis_stats(self, indication_table, analysis_stats):
        # Complex merging of two tables based on the distance between old blob centroid and new blob centroid
        new_adjusted_indication_table = indication_table.copy()

        analysis_stats_to_add = [
            "SNR_analysis_mean_ref_area",
            "SNR_analysis_std_ref_area",
            "SNR_analysis_threshold",
            "SNR_analysis_mean_ref_area%",
            "SNR_analysis_std_ref_area%",
            "SNR_analysis_threshold%",
        ]
        for stat_name in analysis_stats_to_add:
            new_adjusted_indication_table[stat_name] = np.nan

        df_analysis_stats = pd.DataFrame.from_dict(analysis_stats)
        all_centroids = df_analysis_stats["centroids"].to_list()
        tree = spatial.KDTree(all_centroids)

        for new_blob_id in new_adjusted_indication_table.index:
            blob_to_identify = (
                new_adjusted_indication_table.at[new_blob_id, "center_x"],
                new_adjusted_indication_table.at[new_blob_id, "center_y"],
                new_adjusted_indication_table.at[new_blob_id, "center_z"],
            )
            dist_old_and_new_blob, old_blob_id = tree.query(blob_to_identify)
            for stat_name in analysis_stats_to_add:
                new_adjusted_indication_table.at[new_blob_id, stat_name] = round(
                    df_analysis_stats.at[old_blob_id, stat_name], 1
                )

        return new_adjusted_indication_table
