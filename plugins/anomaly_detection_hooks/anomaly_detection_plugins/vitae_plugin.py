# import logger and set itup
import logging
import os
import time
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.ndimage import zoom

from sentinelai.training.model.AE.AE_models.ViTAE_Monai import ViTAE_Monai as ViTAE
from sentinelai.training.model_parameters import ViTAEmonai_PARAMS

import torch.nn.functional as F
from ..anomaly_detection_plugin_manager import (
    AnomalyInferenceInput,
    AnomalyInferenceOutput,
    AnomalyInferencePlugin,
)

from torchvision.transforms.functional import crop

logger = logging.getLogger(__name__)
# Assuming SimplifiedViTAE and other necessary classes are already defined


def remove_noise_array(images):
    if images.ndim == 4:
        resized_images = np.empty(
            (images.shape[0], images.shape[1], images.shape[2], images.shape[3]),
            dtype=np.uint8,
        )
    else:
        resized_images = np.empty(
            (images.shape[0], images.shape[1], images.shape[2]), dtype=np.uint8
        )

    for i in range(images.shape[0]):
        if images.ndim == 4:
            for c in range(images.shape[3]):
                resized_images[i, :, :, c] = zoom(
                    images[i, :, :, c], zoom=(1, 1), order=3
                )
        else:
            resized_images[i, :, :] = zoom(images[i, :, :], zoom=(1, 1), order=3)
    return resized_images.astype(np.uint8)


def _build_preprocess_pipeline(
    initial_resize: tuple[int, int],
    final_resize: tuple[int, int],
    roi: tuple[int, int, int, int] | None,
    mean: float,
    std: float,
) -> transforms.Compose:
    """
    ToTensor (0-1) → Normalize(mean, std) → Resize → ROI → Resize
    NOTE: `mean` and `std` should be 0.5, 0.5 for [-1, 1] range.
    """
    tfms: list = [
        transforms.ToTensor(),  # uint8 [0, 255] → float32 [0, 1]
        transforms.Resize(initial_resize, antialias=True),
        transforms.Normalize(mean=(mean,), std=(std,)),  # [0, 1] → [-1, 1]
    ]

    if roi is not None:
        x, y, w, h = roi
        tfms.append(
            transforms.Lambda(lambda img: crop(img, top=y, left=x, height=h, width=w))
        )

    if roi is not None or initial_resize != final_resize:
        tfms.append(transforms.Resize(final_resize, antialias=True))

    return transforms.Compose(tfms)


class InferenceViTAE:
    def __init__(self, model_path=None, axis="z", batch_size=32):
        self.model_path = model_path
        self.axis = axis
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "vitae"
        try:
            self.model = self.load_model_from_checkpoint().to(self.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
        self.classes_names = ["anomaly"]

    def load_model_from_checkpoint(self):
        model_path = self.model_path
        if self.model_path is None:
            logger.info("No checkpoint path provided, using default checkpoint")

            self.model_name = "vitae"
            model_path = files("sentinelai.models.ViTAE").joinpath("model_ViTAE.ckpt")
            self.model_path = model_path
        else:
            self.model_name = Path(model_path).stem  # type: ignore ,traversable is a valid Path like object

        logger.info(
            f"Loading ViTAE model from {model_path} (Direct input to pytorch. This is guaranteed to be the model chekpoint used)"
        )

        try:
            checkpoint_dict = torch.load(
                model_path,  # type: ignore
                map_location=self.device,
                weights_only=False,  # type: ignore
            )  # type: ignore

        except Exception as e:
            logger.error(f"Error loading model from checkpoint: {e}")
            raise e
        # We check if the checkpoint has a 'hyperparameters' key, if it does, we are using a pytorch lightning checkpoint
        hyperparameters = None

        if "hyperparameters" in checkpoint_dict.keys():
            hyperparameters = checkpoint_dict["hyperparameters"]
        elif "hyperparameters" in checkpoint_dict["state_dict"].keys():
            hyperparameters = checkpoint_dict["state_dict"]["hyperparameters"]
        elif "hyper_parameters" in checkpoint_dict.keys():
            hyperparameters = checkpoint_dict["hyper_parameters"]
        elif "hyper_parameters" in checkpoint_dict["state_dict"].keys():
            hyperparameters = checkpoint_dict["state_dict"]["hyper_parameters"]

        if hyperparameters is not None:
            args = hyperparameters.copy()
            print("Loading model with hyperparameters: ")
            # remove dataet key from the hyperparameters dict before printing
            if "dataset" in args.keys():
                del args["dataset"]
            # pprint(args)
            model = ViTAE(args)
            logger.debug("Loaded model with hyperparameters from checkpoint")
        else:
            model = ViTAE(**ViTAEmonai_PARAMS)
            logger.debug("Loaded model with default hyperparameters")

        state_dict = checkpoint_dict["state_dict"]  # type: ignore
        model.load_state_dict(state_dict=state_dict)  # type: ignore
        model.eval()

        return model

        # self.save_model_to_checkpoint(model_path, pth, f"{self.model_name}-cleaned")

    def save_model_to_checkpoint(
        self, input_path: Path, output_path: Path, model_name: str
    ):
        if input_path is None:
            raise ValueError("Path to save the model is not provided")
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        output_model_path = output_path / model_name

        # Before saving, we check the state dict for any WindowsPath objects and convert them to strings
        checkpoint_dict = torch.load(
            input_path,  # type: ignore
            map_location=self.device,
            weights_only=False,  # type: ignore
        )
        state_dict = checkpoint_dict["state_dict"]  # type: ignore
        hyperparams = checkpoint_dict.get("hyper_parameters")
        if hyperparams is not None:
            for key, value in hyperparams.items():
                if isinstance(value, Path):
                    hyperparams[key] = value.as_posix()
                # if its  nested dict, we need to convert all the values to strings
            if "dataset" in hyperparams.keys():
                # just remove the dataset key from the dict
                print("Removing dataset key from hyperparameters")
                del hyperparams["dataset"]

        model = ViTAE(hyperparams)
        model.load_state_dict(state_dict=state_dict)  # type: ignore
        model.eval()
        # Save the model
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyperparameters": hyperparams,
                "pytorch_lightning_version": checkpoint_dict.get(
                    "pytorch_lightning_version", None
                ),
            },
            output_model_path,
        )
        print(f"Model saved to {output_model_path}")

    def infer_dataset(
        self,
        numpy_data: np.ndarray,
        *,
        roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            disp_imgs  - uint8  (N,H_orig,W_orig[,C])
            disp_mse   - float32(N,) - MSE in original data space
            raw_recons - float32(N,H_orig,W_orig[,C]) 0-255
            raw_mse    - float32(N,) - MSE in normalized space
        """
        t0 = time.time()
        logger.debug("Starting ViTAE inference")

        initial_resize = self.model.model.img_size
        final_resize = initial_resize  # For now, they are the same

        # ── 0. Ensure consistent data types and format -------------------------
        if numpy_data.ndim == 4 and numpy_data.shape[1] in (1, 3, 4):
            numpy_data = np.transpose(numpy_data, (0, 2, 3, 1))  # NCHW→NHWC

        # Ensure float32 for consistent processing
        # if numpy_data.dtype != np.float32:
        #     numpy_data = numpy_data.astype(np.float32)

        n_samples = len(numpy_data)
        sample_shape = numpy_data.shape[1:]  # (H, W) or (H, W, C)

        # ── 1. Pre-allocate output arrays for efficiency --------------------
        disp_imgs = np.zeros((n_samples, *sample_shape[:2]), dtype=np.uint8)
        raw_recons = np.zeros((n_samples, *sample_shape[:2]), dtype=np.float32)
        disp_mse = np.zeros(n_samples, dtype=np.float32)
        raw_mse = np.zeros(n_samples, dtype=np.float32)

        # ── 2. Fix normalization values for proper [-1, 1] range -----------
        norm_mean = 0.5  # For [0, 1] input to get [-1, 1] output
        norm_std = 0.5

        preprocess = _build_preprocess_pipeline(
            initial_resize,
            final_resize,
            roi,
            mean=norm_mean,
            std=norm_std,
        )

        mean_t = torch.tensor(norm_mean, device=self.device).view(1, 1, 1, 1)
        std_t = torch.tensor(norm_std, device=self.device).view(1, 1, 1, 1)

        # ── 3. Batch processing with proper indexing -----------------------
        processed_count = 0

        with torch.no_grad():
            logger.info(
                f"Starting inference on {n_samples} samples with batch size {self.batch_size}"
            )
            for b0 in range(0, n_samples, self.batch_size):
                b1 = min(b0 + self.batch_size, n_samples)
                batch_np = numpy_data[b0:b1]
                logger.debug(f"Processing batch {b0 // self.batch_size + 1}")
                # ToTensor is weird. If its not a np.uint8, it wont do a [0, 255] to [0, 1] conversion
                # so we need to do it but only if the data isnt a np.uint8
                # as the transform will do it for us
                if batch_np.dtype != np.uint8:
                    batch_np = batch_np.astype(np.float32) / 255.0

                batch_size_current = b1 - b0

                # Remember the *true* (H,W) of every slice
                orig_sizes = [(img.shape[0], img.shape[1]) for img in batch_np]

                # ── 4. Preprocessing pipeline (identical to training) -----------
                batch = torch.stack([preprocess(img) for img in batch_np]).to(
                    self.device
                )  # (B,1,H,W)

                # ── 5. Forward pass ---------------------------------------------
                recon_norm = self.model(batch)  # normalized output [-1, 1]

                # ── 6. MSE in normalized space (for model evaluation) ----------
                mse_norm = (
                    torch.mean((batch - recon_norm) ** 2, dim=(1, 2, 3)).cpu().numpy()
                )
                raw_mse[b0:b1] = mse_norm

                # ── 7. Un-normalize to original scale [0, 255] -----------------
                recon = recon_norm * std_t + mean_t  # Back to [0, 255]
                batch_un = batch * std_t + mean_t  # Reference in [0, 255]

                # ── 8. Restore original size ------------------------------------
                resized_recon = []
                resized_batch = []

                for idx, (h, w) in enumerate(orig_sizes):
                    # Handle reconstruction
                    if recon.shape[2] == h and recon.shape[3] == w:
                        resized_recon.append(recon[idx])
                    else:
                        r = F.interpolate(
                            recon[idx : idx + 1],  # Keep single sample with batch dim
                            size=(h, w),  # target size
                            mode="bilinear",
                            align_corners=False,
                        )
                        resized_recon.append(r[0])  # Extract the single sample

                    # Handle reference (for MSE calculation in original space)
                    if batch_un.shape[2] == h and batch_un.shape[3] == w:
                        resized_batch.append(batch_un[idx])
                    else:
                        b = F.interpolate(
                            batch_un[
                                idx : idx + 1
                            ],  # Keep single sample with batch dim
                            size=(h, w),  # target size
                            mode="bilinear",
                            align_corners=False,
                        )
                        resized_batch.append(b[0])  # Extract the single sample
                recon = torch.stack(resized_recon)  # (B, 1, H_orig, W_orig)
                batch_un = torch.stack(resized_batch)  # (B, 1, H_orig, W_orig)

                # ── 9. Convert to numpy and process individual images ----------
                recon_np = recon.cpu().numpy()[:, 0, ...]  # (B, H, W)
                batch_np_un = batch_un.cpu().numpy()[:, 0, ...]  # (B, H, W)

                # ── 10. Calculate MSE in original space ------------------------
                mse_original = np.mean((recon_np - batch_np_un) ** 2, axis=(1, 2))
                disp_mse[b0:b1] = mse_original

                # ── 11. Process each image individually ------------------------
                for i, (rec, ref) in enumerate(zip(recon_np, batch_np_un)):
                    global_idx = b0 + i

                    # Store raw reconstruction (without clamping for anomaly detection)
                    raw_recons[global_idx] = rec.copy()

                    # Create display image with proper normalization
                    # Use reference range for consistent display
                    ref_min, ref_max = float(ref.min()), float(ref.max())

                    # Avoid division by zero
                    if ref_max - ref_min > 1e-8:
                        # Normalize to [0, 1] then scale to [0, 255]
                        disp = (rec - ref_min) / (ref_max - ref_min) * 255.0
                        disp = np.clip(disp, 0, 255).astype(np.uint8)
                    else:
                        # Handle constant images
                        disp = np.full_like(rec, 0, dtype=np.uint8)

                    disp_imgs[global_idx] = disp

                processed_count += batch_size_current

            logger.debug(f"Processed {processed_count} samples")

        logger.debug(f"Inference done in {time.time() - t0:.2f}s")

        return (
            disp_imgs,  # uint8 (N,H_orig,W_orig) - display images
            disp_mse,  # float32 (N,) - MSE in original space
            raw_recons,  # float32 (N,H_orig,W_orig) - raw reconstructions
            raw_mse,  # float32 (N,) - MSE in normalized space
        )

    def start_inference(self, numpy_data, axis="z", *args, **kwargs):
        self.axis = axis
        if self.axis == "x":
            numpy_data = numpy_data.transpose(1, 2, 0)
        elif self.axis == "y":
            numpy_data = numpy_data.transpose(0, 2, 1)

        display_imgs, display_mse, raw_recons, raw_mse = self.infer_dataset(numpy_data)
        return display_imgs, display_mse, raw_recons, raw_mse

    def set_current_nde_file_name(self, nde_file_name):
        pass


logger = logging.getLogger(__name__)


# @register_anomaly_inference_plugin(anomaly_inference_plugin_registry, "vitae")
class ViTAEPlugin(AnomalyInferencePlugin):
    _model_id = "vitae"

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

        self.inference_model = InferenceViTAE(axis=axis, model_path=checkpoint_path)

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
        data_array = input_data.data_array
        # data_array = input_data.raw_data_array

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

        display_imgs, display_mse, raw_recons, raw_mse = (
            self.inference_model.start_inference(data_array)
        )

        ckpt = self.checkpoint_path
        logger.info(
            f"ViTAEPlugin: Inference completed for group {group_index} with checkpoint {ckpt}"
        )
        array_mapping = {
            "display_imgs": display_imgs,
            "display_mse": display_mse,
            "raw_recons": raw_recons,
            "raw_mse": raw_mse,
        }

        anomaly_output = AnomalyInferenceOutput(
            group_index=group_index,
            dataset_id=input_data.dataset_id,
            original_data_array=data_array,
            result_arrays=array_mapping,
            anomaly_inference_input=input_data,
            inferences_settings=input_data.inference_context,  # type: ignore
            inference_type=self.inference_type,
            grouped_data=input_data.grouped_data,
            model_checkpoint=self.checkpoint_path,  # type: ignore
            inference_id=self.inference_id,
            model_class="vitae",
        )

        return anomaly_output
