from pathlib import Path


from sentinelai.inference.anomaly_detection.inference_devnet import InferenceDevNet
from sentinelai.utils.data_array_preprocessing import (
    shift_values_dynamically_in_chunks,
)
import time
import numpy as np
from ..anomaly_detection_plugin_manager import (
    AnomalyInferenceInput,
    AnomalyInferenceOutput,
    AnomalyInferencePlugin,
    default_anomaly_plugin_registry,
    register_anomaly_inference_plugin,
)
import logging

logger = logging.getLogger(__name__)


# @register_anomaly_inference_plugin(anomaly_inference_plugin_registry, "DevNet")
class DevNetPlugin(AnomalyInferencePlugin):
    _model_id = "devnet"

    def __init__(
        self,
        checkpoint_path: Path,
        inference_id: str | None = None,
        axis="z",
    ):
        if inference_id is None:
            inference_id = checkpoint_path.stem
        self._inference_id = inference_id
        self.inference_type = "anomaly_detection"
        self.axis = axis
        self.checkpoint_path = checkpoint_path

        self.inference_model = InferenceDevNet(axis=axis, model_path=checkpoint_path)

        self._inference_id = self.inference_model.model_name

    @property
    def model_id(self) -> str:
        """Identifier for the model used in the inference."""
        return self._model_id

    @property
    def inference_id(self) -> str:
        return self._inference_id

    @inference_id.setter
    def inference_id(self, value: str):
        self._inference_id = value

    def process_inference(
        self, input_data: AnomalyInferenceInput
    ) -> AnomalyInferenceOutput:
        """Process the inference and return binary results and scores."""
        group_index = input_data.group_index
        data_array = input_data.raw_data_array.copy()
        # data_array = input_data.data_array.copy()

        if np.min(data_array) < 0:
            bipolar_min = np.min(data_array)
            bipolar_max = np.max(data_array)
            logger.info(
                f"Data array is bipolar, min: {bipolar_min}, max: {bipolar_max}, normalizing to 0-255 after abs"
            )
            abs_min = np.min(np.abs(data_array))
            abs_max = np.max(np.abs(data_array))
            logger.info(f"Absolute min: {abs_min}, Absolute max: {abs_max}")
            if abs_max == 0:
                data_array = np.zeros_like(data_array, dtype=np.uint8)
            else:
                data_array = np.abs(data_array)

                data_array = (data_array / abs_max) * 255.0
                data_array = data_array.astype(np.uint8)
                real_min = np.min(data_array)
                real_max = np.max(data_array)
                logger.info(
                    f"Data array normalized to 0-255, new min: {real_min},  new max: {real_max}"
                )
        elif np.max(data_array) > 255:
            # if its unipolar, we just scale it to 0-255
            data_min = np.min(data_array)
            data_max = np.max(data_array)
            if data_max - data_min == 0:
                data_array = np.zeros_like(data_array, dtype=np.uint8)
            else:
                data_array = ((data_array - data_min) / (data_max - data_min)) * 255.0
                data_array = data_array.astype(np.uint8)
        # if input_data.raw_data_array.min() < 0:
        #     # This is a bit of  a hack, since we want to use the actually
        #     # processed data in almost every case except for when the data is negative
        #     # as we want to use the older way of shifting the dta from 0-255 but with
        #     # a normalization step. Wheraas now,the shift is done differently,
        #     # without normalizaton
        #     # data_array = input_data.raw_data_array.astype(np.uint8)
        #     # clip all values to 0-255, dropping everything below 0
        #     data_array = np.clip(input_data.raw_data_array.copy(), 0, 255).astype(
        #         np.uint8
        #     )

        shape = data_array.shape

        binary_results, score_array = self.inference_model.start_inference(data_array)

        ckpt = self.checkpoint_path
        logger.info(
            f"DevNetPlugin: Inference completed for group {group_index} with checkpoint {ckpt}"
        )

        array_mapping = {
            "binary_results": binary_results,
            "score_array": score_array,
        }

        anomaly_output = AnomalyInferenceOutput(
            group_index=group_index,
            dataset_id=input_data.dataset_id,
            model_class="devnet",
            original_data_array=data_array,
            result_arrays=array_mapping,
            anomaly_inference_input=input_data,
            inferences_settings=input_data.inference_context,  # type: ignore
            inference_type=self.inference_type,
            grouped_data=input_data.grouped_data,
            model_checkpoint=self.checkpoint_path,  # type: ignore
            inference_id=self.inference_id,
        )

        return anomaly_output
