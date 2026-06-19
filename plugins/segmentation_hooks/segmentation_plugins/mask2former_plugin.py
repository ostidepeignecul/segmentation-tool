"""
Mask2Former workflow steps
─────────────────────────────────────────────────────────────────────
Three `StepPlugin` classes that plug directly into the PipelinePluginManager.

They run inference on local Mask2Former models using Hugging Face's transformers library.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from plugins.hook_types import HookType
from ..segmentation_plugin_manager import (
    PipelineInput,
    PreprocessOutput,
    InferenceOutput,
    nnUNetPostprocessOutput,
)
from ..step_registration import register_step, StepPlugin

logger = logging.getLogger(__name__)


# ──────────────────────────── 1. Pre-process ─────────────────────

@register_step(
    HookType.SEGMENTATION_PREPROCESS,
    "mask2former_preprocess",
    bucket_id="mask2former",
    override=True,
)
class Mask2FormerPreprocessor(StepPlugin[PipelineInput, PreprocessOutput]):
    """
    Normalizes inputs and prepares metadata for the Mask2Former inference.
    """

    optional = False

    def process(self, input_data: PipelineInput) -> PreprocessOutput:
        model_path = input_data.pipeline_context.get("exported_model_folder", None)
        if model_path is None:
            raise ValueError("No model folder path provided in the pipeline context.")

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Dossier de modèle Mask2Former introuvable: {model_path}")

        # The input data is typically grayscale (0-255 or 0-1)
        raw = input_data.data_array.astype(np.float32, copy=True)
        if raw.max() <= 1.01:
            raw *= 255.0
        raw = np.clip(raw, 0, 255)

        meta = {
            "original_shape": raw.shape,  # (Z, Y, X)
            "model_path": str(model_path),
        }

        return PreprocessOutput(
            clean_data=raw,
            meta=meta,
            pipeline_input=input_data,
        )


# ───────────────────────────── 2. Inference ───────────────────────

@register_step(
    HookType.SEGMENTATION_INFERENCE,
    "mask2former_inference",
    bucket_id="mask2former",
    override=True,
)
class Mask2FormerInference(StepPlugin[PreprocessOutput, InferenceOutput]):
    """
    Performs inference slice-by-slice using a local Mask2Former model.
    """

    optional = False

    def process(self, input_data: PreprocessOutput) -> InferenceOutput:
        # Import transformers dynamically at call time
        from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

        sinp = input_data.pipeline_input
        model_path = input_data.meta["model_path"]

        logger.info(f"Loading local Mask2Former model from: {model_path}")
        processor = Mask2FormerImageProcessor.from_pretrained(model_path)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        vol = input_data.clean_data  # Shape (Z, Y, X)
        n_slices = vol.shape[0]
        slice_height, slice_width = vol.shape[1], vol.shape[2]

        segmentations = []
        batch_size = int(sinp.config.get("batch_size", 4) or 4)

        logger.info(f"Mask2Former inference starting. Slices: {n_slices}, Batch size: {batch_size}, Device: {device}")

        with torch.no_grad():
            for i in range(0, n_slices, batch_size):
                batch_slices = []
                target_sizes = []
                for j in range(i, min(i + batch_size, n_slices)):
                    slice_img = vol[j]  # Shape (Y, X)
                    # Convert to RGB by duplicating the channel
                    rgb_slice = np.stack([slice_img] * 3, axis=-1).astype(np.uint8)
                    batch_slices.append(rgb_slice)
                    target_sizes.append((slice_height, slice_width))

                if not batch_slices:
                    continue

                # Process batch
                inputs = processor(images=batch_slices, return_tensors="pt").to(device)
                outputs = model(**inputs)

                # Post-process panoptic segmentation to extract target maps
                preds = processor.post_process_panoptic_segmentation(outputs, threshold=0.3, target_sizes=target_sizes, label_ids_to_fuse=set())
                for pred in preds:
                    segmentation_tensor = pred["segmentation"].cpu().numpy()
                    segments_info = pred["segments_info"]
                    
                    # Create empty mask for this slice (default is Background=0)
                    slice_mask = np.zeros_like(segmentation_tensor, dtype=np.uint8)
                    
                    # Map each segment (instance) to the editor's semantic labels
                    for segment in segments_info:
                        seg_id = segment["id"]
                        model_class_id = segment["label_id"]
                        
                        # Map model class ID to editor label ID
                        if model_class_id == 0:    # frontwall -> FW
                            editor_label = 3
                        elif model_class_id == 1:  # backwall -> BW
                            editor_label = 4
                        elif model_class_id == 2:  # bw_echo1 -> BW echo 1
                            editor_label = 5
                        else:                      # background / other -> Background
                            editor_label = 0
                            
                        slice_mask[segmentation_tensor == seg_id] = editor_label
                        
                    segmentations.append(slice_mask)

        segmentation_array = np.stack(segmentations, axis=0)  # Shape (Z, Y, X)
        segmentations = [segmentation_array[idx] for idx in range(segmentation_array.shape[0])]

        labels_dict = {
            "Background": 0,
            "FW": 3,
            "BW": 4,
            "BW echo 1": 5,
        }

        labels_mapping = {
            "labels": labels_dict,
            "channel_names": {"0": "ultrasound"}
        }

        return InferenceOutput(
            group_index=sinp.group_index,
            dataset_id=sinp.dataset_id,
            inference_id="mask2former",
            config={"checkpoint_path": model_path},
            segmentation_mask={
                "segmentation_list": segmentations,
                "mask": segmentation_array,
            },
            pipeline_input=sinp,
            labels_mapping=labels_mapping,
            probabilities_array=None,
        )


# ─────────────────────────── 3. Post-process ─────────────────────

@register_step(
    HookType.SEGMENTATION_POSTPROCESS,
    "mask2former_postprocess",
    bucket_id="mask2former",
    override=True,
)
class Mask2FormerPostprocessor(StepPlugin[InferenceOutput, nnUNetPostprocessOutput]):
    """
    Ensures mask is contiguous and computes bounding box coordinates.
    """

    optional = False

    def process(self, input_data: InferenceOutput) -> nnUNetPostprocessOutput:
        mask = input_data.segmentation_mask["mask"]
        mask = np.ascontiguousarray(mask)

        coords = self._bbox_from_mask(mask)
        report = {
            "voxels_foreground": int(mask.sum()),
            "bbox": coords,
        }

        logger.info(f"Final Mask2Former Mask shape: {mask.shape}")
        return nnUNetPostprocessOutput(
            group_index=input_data.group_index,
            dataset_id=input_data.dataset_id,
            inference_id=input_data.inference_id,
            segmentation_mask={"mask": mask},
            segmentation_coordinates=coords,
            report=report,
            labels_mapping=input_data.labels_mapping,
            probabilities_array=input_data.probabilities_array,
            pipeline_input=input_data.pipeline_input,
            metadata=input_data.pipeline_input.metadata,
        )

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> dict[str, int]:
        if mask.max() == 0:
            return dict(x=0, X=0, y=0, Y=0, z=0, Z=0)
        idx = np.where(mask)
        return dict(
            z=int(idx[0].min()),
            Z=int(idx[0].max()),
            y=int(idx[1].min()),
            Y=int(idx[1].max()),
            x=int(idx[2].min()),
            X=int(idx[2].max()),
        )
