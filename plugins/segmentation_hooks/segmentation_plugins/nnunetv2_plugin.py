"""
nnUNet-v2 workflow steps
─────────────────────────────────────────────────────────────────────
Three `StepPlugin` classes that plug directly into the generic
PipelinePluginManager you already have.

They call **nnUNet v2** via its *Python SDK* (no shelling-out) and
implement the canonical workflow:

    1.  Pre-processing   →  convert ndarray ⇒ float32, collect
                            per-slice meta ("properties") expected by
                            nnUNet.
    2.  Inference        →  instantiate `nnUNetPredictor`, load the
                            *explicit* checkpoint given in
                            `SegInferenceInput.inference_context`, run
                            `predict_single_npy_array`.
    3.  Post-processing  →  convert softmax/label map to mask,
                            extract a bounding-box coordinate dict,
                            build a small report.

All three classes are registered in the **default** registry.  If you
want them isolated (eg. `"nnunet"` registry), just pass
`registry_id="nnunet"` in each `@register_step` line.

Dependencies
------------
* `pip install nnunetv2`
* torch ≥ 1.13 (CUDA optional but recommended)

References
----------
nnUNet Python API examples and predictor class definition :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING

from collections.abc import Iterator
import pprint
import numpy as np
import torch
import os

import torch._dynamo


from plugins.hook_types import HookType, nnUNetModelTypes
from ..segmentation_plugin_manager import (
    PipelineInput,
    InferenceOutput,
    nnUNetPostprocessOutput,
)
from ..step_registration import register_step, StepPlugin
from ._nnunetv2_utils import parse_folder_structure, log_std, ProgressIterator
from PIL import Image

if TYPE_CHECKING:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

logger = logging.getLogger(__name__)

# to avoid triton related errors:
torch._dynamo.config.suppress_errors = True  # type: ignore


def build_nnunet_data_iterator_respecting_plans(
    predictor: nnUNetPredictor,
    input_images: list[np.ndarray],
    input_props: list[dict],
    prev_preds: list[np.ndarray] | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Custom iterator that preprocesses input images using nnU-Net's own
    PreprocessorForInference, exactly like the CLI does.

    Each item yielded is a dict with keys:
    - 'data': torch tensor (preprocessed image)
    - 'data_properties': dict with spacing and shape
    - 'ofile': None (indicating to return, not save)
    """
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
        DefaultPreprocessor,
    )

    preprocessor_cls = predictor.configuration_manager.preprocessor_class
    preprocessor: DefaultPreprocessor = preprocessor_cls(verbose=predictor.verbose)

    config_manager = predictor.configuration_manager
    plans_manager = predictor.plans_manager
    dataset_json = predictor.dataset_json

    if config_manager is None or plans_manager is None or dataset_json is None:
        raise ValueError(
            "Configuration manager, plans manager, and dataset JSON must be set in the predictor."
        )

    for i, (img, prop) in enumerate(zip(input_images, input_props)):
        seg_prev_stage = prev_preds[i] if prev_preds else None
        data, seg, props = preprocessor.run_case_npy(
            img,
            seg_prev_stage,
            prop,
            plans_manager,
            config_manager,
            dataset_json,
        )
        yield {
            "data": torch.from_numpy(data).contiguous().pin_memory(),
            "data_properties": props,
            "ofile": None,
        }


def legacy_png_roundtrip(
    img_f32: np.ndarray,
    *,
    clip_percentiles: tuple[float, float] | None = (0.5, 99.5),
    to_uint8: bool = True,
) -> np.ndarray:
    """
    Replicates the *exact* data path that happened during training:

    1.  Float/uint16 ultrasound slice  ──►
    2.  **saved** as 8-bit PNG (RGB)     ──►
    3.  `Image.open(...).convert("L")`   ──►
    4.  NumPy float32 fed into nnU-Net.

    Parameters
    ----------
    img_f32
        2-D array in **row-column** order (H, W) or **channel-row-col**
        (C, H, W) where C=1.  The function always returns a (1, H, W) array.
    clip_percentiles
        Optional tuple *(lo, hi)*.  Values below the *lo*-th percentile and
        above the *hi*-th percentile are clipped before scaling.
        Set to `None` to disable.
    to_uint8
        If *True* we really quantise to 8-bit before the PIL round-trip.
        Set to *False* if you merely want the clipping & rescaling but
        keep the original dynamic range.

    Returns
    -------
    np.ndarray
        Float32 array, channel-first, scaled back to **[0, 1]** - ready to be
        stacked into the batch list of nnU-Net.
    """
    exp_ndim = img_f32.ndim
    if img_f32.ndim == 4 and img_f32.shape[0] == 1:
        img_f32 = img_f32[0]

    if img_f32.ndim == 3 and img_f32.shape[0] == 1:
        img_f32 = img_f32[0]  # strip the dummy channel first

    if img_f32.ndim != 2:
        raise ValueError("`img_f32` must be HxW or 1xHxW")

    img = img_f32.astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    # 1) optional percentile clipping to remove hot speckles / dropouts  #
    # ------------------------------------------------------------------ #
    if clip_percentiles is not None:
        lo, hi = np.percentile(img, clip_percentiles)
        img = np.clip(img, lo, hi)

    # ------------------------------------------------------------------ #
    # 2) scale to [0, 1] – identical to what you did before saving PNGs  #
    # ------------------------------------------------------------------ #
    img_min, img_max = float(img.min()), float(img.max())
    img = (img - img_min) / (img_max - img_min + 1e-8)

    # ------------------------------------------------------------------ #
    # 3) fake the “PNG → RGB → L” round-trip                              #
    # ------------------------------------------------------------------ #
    if to_uint8:
        img_u8 = (img * 255.0 + 0.5).astype(np.uint8)

        # create an *RGB* PNG even though channels are identical;
        # Pillow will apply the Y = 0.299 R + 0.587 G + 0.114 B weights
        # when we convert back to "L"
        pil_rgb = Image.fromarray(np.stack([img_u8] * 3, axis=-1), mode="RGB")
        pil_gray = pil_rgb.convert("L")

        # back to float32 in [0, 1]
        img = np.asarray(pil_gray, dtype=np.float32) / 255.0

    # ------------------------------------------------------------------ #
    # 4) channel-first tensor expected by nnU-Net                         #
    # ------------------------------------------------------------------ #
    # dependingon expected input dims, we add a channel dimension
    # if its 3, we add once, if its 4 we add twice
    if exp_ndim == 2:
        return img
    if exp_ndim >= 3:
        # we add a channel dimension, so we get 1, H, W
        img = img[None]
    if exp_ndim >= 4:
        # we add another channel dimension, so we get 1, 1, H, W
        img = img[None]
    return img


# ─────────────────────────── extended dataclass ──────────────────
@dataclass
class nnUNetPreprocessOutput:
    """
    Adds `slice_properties` to the generic PreprocessOutput so the
    predictor can access per-slice statistics if desired.
    """

    clean_data: np.ndarray
    # Since the model can take a 3d image as a "single" image and does not necessarily know about
    # slices, we use this to indicate to the inference step if the entire array or specific slice ranges
    inference_input_array_list: list[np.ndarray]
    meta: dict
    slice_properties: list[dict]
    seg_inference_input: PipelineInput


# ──────────────────────────── 1. Pre-process ─────────────────────


@register_step(
    HookType.SEGMENTATION_PREPROCESS,
    "nnunetv2_preprocess",
    bucket_id="nnunetv2",
    override=True,
)
class nnUNetv2Preprocessor(StepPlugin[PipelineInput, nnUNetPreprocessOutput]):
    """
    Converts raw ndarray → float32, normalises to [0, 1] and collects
    simple statistics per axial slice (mean, std) as an example of
    "properties".  You can extend this to whatever nnUNet trainer you
    use needs (spacing, orientation, etc.).
    """

    optional = False

    def process(self, input_data: PipelineInput) -> nnUNetPreprocessOutput:
        # defer to avoid very slow import (nnunet brings a lot of deps)
        from nnunetv2.inference.predict_from_raw_data import load_json

        exported_model_folder = input_data.pipeline_context.get(
            "exported_model_folder", None
        )
        if exported_model_folder is None:
            raise ValueError(
                "No model folder export path provided in the input config. "
            )

        exported_model_folder = Path(exported_model_folder).expanduser().resolve()
        # iF no model type is provicded (which would force the retrieval logic to take a specific model type
        # even if the model checkpoint has multiple model types in it) the code falls back on taking whatever model
        # type is present in the folder structure
        user_provided_model_type: nnUNetModelTypes | None = input_data.config.get(
            "nnunet_model_type", None
        )

        model_paths, model_type = parse_folder_structure(
            exported_model_folder, user_provided_model_type
        )

        if model_type not in [
            "2d",
            "3d_lowres",
            "3d_fullres",
            "3d_cascade_fullres",
        ]:
            raise ValueError(
                f"nnunet_model_type must be one of ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], got {model_type}"
            )

        # Tell the type checker that the nnunet_model_type is a valid type (not just a string, but a str that fits the nnUNetModelTypes literal type
        # we cant do isinstance since we cant check  a literal type at runtime

        model_type = cast(nnUNetModelTypes, model_type)

        # Set the model type in the input data config so it can be used later
        input_data.config["nnunet_model_type"] = model_type

        # for now, we take the first model path, which is the one with the most epochs
        if model_paths is None or len(model_paths) == 0:
            raise ValueError(
                f"No valid nnUNet model paths found in {exported_model_folder}."
            )
        model_path, epochs = model_paths[0]

        plans_json_path = model_path / "plans.json"
        if not plans_json_path.is_file():
            raise FileNotFoundError(
                f"Plans JSON not found at {plans_json_path}. "
                "Make sure the model folder contains a valid plans.json file."
            )
        # parse the plans to get some data we need such as the model type
        plans_dict = load_json(str(plans_json_path))

        logger.info(
            f"Using model path {model_path} with {epochs} epochs for nnUNet preprocessing."
        )

        input_data.config["model_path"] = str(model_path)
        input_data.pipeline_context["model_folder"] = str(model_path)

        inference_input_array_list = []
        slice_props = []

        # ————————— raw → float32 in [0, 255] ————————— #
        raw = input_data.data_array.astype(np.float32, copy=True)
        if raw.max() <= 1.01:  # scanner gives 0-1 floats
            raw *= 255.0
        raw = np.clip(raw, 0, 255)
        # raw = np.flip(raw, axis=1)  # (

        # nnU-Net needs channel first.  For grayscale we add C=1
        vol = raw[None, ...]  # (1, Z, Y, X)
        logger.info(f"Input data shape (C,Z,Y,X): {vol.shape}")
        # ————————— spacing ————————— #
        spacing = (
            input_data.config.get("spacing")
            or plans_dict.get("original_median_spacing_after_transp")
            or (1.0, 1.0, 1.0)  # fallback if nothing was provided
        )

        # ————————— decide slice strategy ————————— #
        is_3d = model_type.startswith("3d")
        flip = False
        if flip is True:
            vol = np.flip(
                vol, axis=1
            )  # (N, Z, Y, X) → (N, Z, Y, X) with flipped X axis
        if is_3d:
            # single element list → iterator will treat as one “case”
            logger.info(f"Using 3D model type: {model_type}.")
            inference_list: list[np.ndarray] = [vol]
            slice_props = [
                {
                    "spacing": spacing,
                    "original_shape": raw.shape,
                }
            ]
        else:
            # 2-D model: run inference slice by slice over the full volume (no chunking).
            n_slices = vol.shape[1]  # number of slices in Z
            inference_list = [
                np.expand_dims(vol[:, z, ...], 1)  # (1,1,Y,X)
                for z in range(n_slices)
            ]
            slice_props = [
                {
                    "slice_index": z,
                    "spacing": spacing,
                    "original_shape": raw.shape,
                }
                for z in range(n_slices)
            ]
        meta = dict(
            original_shape=raw.shape,
            spacing=spacing,
            model_type=model_type,
            total_slices=len(inference_list),
        )
        logger.debug("Input data props\n%s", pprint.pformat(meta, indent=2))

        return nnUNetPreprocessOutput(
            clean_data=vol,
            inference_input_array_list=inference_list,
            meta=meta,
            slice_properties=slice_props,
            seg_inference_input=input_data,
        )


# ───────────────────────────── 2. Inference ───────────────────────


# ────────────────────────── utilities ────────────────────────────
def _as_path(obj: str | os.PathLike) -> Path:
    p = Path(obj).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _load_plans(obj: Any) -> dict[str, Any] | None:
    from nnunetv2.inference.predict_from_raw_data import load_json

    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (str, os.PathLike)):
        return load_json(str(_as_path(obj)))
    raise TypeError("'plans' must be dict | str | Path")


def _ensure_list(array_or_list: Any, name: str) -> list[Any]:
    if isinstance(array_or_list, list):
        return array_or_list
    if isinstance(array_or_list, np.ndarray):
        return [array_or_list]
    raise TypeError(f"'{name}' must be ndarray or list")


def _load_inference_labels(model_folder: Path) -> dict[str, Any]:
    from nnunetv2.inference.predict_from_raw_data import load_json

    """Uses the model folder to access the dataset.json file, reads it, and parses the labels and channel names, with
        their corresponding ids.  The structure of the dataset.json file is similar to this:
       {
        "channel_names": {
            "0": "ultrasound"
        },
        "labels": {
            "background": 0,
            "frontwall": 1,
            "backwall": 2
        },
        "numTraining": 97,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO"
    },"""

    dataset_json_path = model_folder / "dataset.json"
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"Dataset JSON not found at {dataset_json_path}")

    dataset_json = load_json(str(dataset_json_path))
    if "labels" not in dataset_json or "channel_names" not in dataset_json:
        raise ValueError(
            f"Dataset JSON at {dataset_json_path} does not contain 'labels' or 'channel_names'"
        )

    return {
        "labels": dataset_json["labels"],
        "channel_names": dataset_json["channel_names"],
    }


# ───────────────────── helper: previous preds ────────────────────
def _collect_prev_preds(
    inp: PipelineInput,
    n_imgs: int,
) -> list[np.ndarray] | None:
    """
    Returns a list (length == n_imgs) of previous-stage segmentations or
    a list of Nones if not provided.
    """
    cand = (
        getattr(inp, "prev_predictions", None)
        or getattr(inp, "previous_stage_predictions", None)
        or getattr(inp, "previous_stage_output", None)
    )

    if cand is None:
        return None

    if isinstance(cand, np.ndarray):
        cand_list = [cand]
    elif isinstance(cand, list):
        cand_list = cand
    else:
        raise TypeError("'previous stage predictions' must be ndarray or list[ndarray]")

    if len(cand_list) != n_imgs:
        raise ValueError(
            "Number of previous-stage predictions "
            f"({len(cand_list)}) does not match number of images ({n_imgs})"
        )

    return cand_list


# We also create a custom iterator that also logs the progress of the inference, since nnunetv2 does not have a built-in progress bar


# ────────────────────────── plugin class ─────────────────────────
@register_step(
    HookType.SEGMENTATION_INFERENCE,
    plugin_id="nnunetv2_inference_iter",
    bucket_id="nnunetv2",
    override=True,
)
class nnUNetv2IteratorInference(StepPlugin[nnUNetPreprocessOutput, InferenceOutput]):
    """
    Iterator-based nnUNet inference (no direct logits function).
    """

    optional = False

    # ——————————————————— main entry ———————————————————
    def process(self, input_data: nnUNetPreprocessOutput) -> InferenceOutput:
        sinp: PipelineInput = input_data.seg_inference_input
        ckpt_path = sinp.pipeline_context.get("model_folder")
        if ckpt_path is None:
            raise ValueError(
                "nnUNet inference requires 'model_folder' in "
                "'SegInferenceInput.inference_context'"
            )

        # TODO: We need to have flexibilot in how we infer onnteh imagee.s
        # Dependingo n  the plans file, and if its a 3d nnunet,
        # we might want to  pass it a single 3d image (meaning, using the clean_data attribute)
        # # or a list of 3d images (meaning, using the inference_input_array_list attribute)
        ckpt_dir = _as_path(ckpt_path)
        ckpt_name = sinp.pipeline_context.get("checkpoint_name", "checkpoint_final.pth")

        # load the model labels and channel names from the dataset.json file
        labels_mapping = _load_inference_labels(ckpt_dir)

        # Optional override of plans.json. This is for advances use cases where we dont want to use the exact config that
        # an model has been set with
        plans_override = _load_plans(sinp.config.get("plans"))

        predictor = _build_predictor(ckpt_dir, ckpt_name, plans_override)
        predictor.verbose = False  # Suppress verbose output, as the livrary uses print statements instead of logging, which polutes the csl

        # build iterator -------------------------------------------------
        imgs = _ensure_list(
            input_data.inference_input_array_list, "inference_input_array_list"
        )
        props = _ensure_list(input_data.slice_properties, "slice_properties")
        prev_preds = _collect_prev_preds(sinp, len(imgs))  # NEW

        # iterator = predictor.get_data_iterator_from_raw_npy_data(
        #     imgs,
        #     prev_preds,
        #     props,
        #     truncated_ofname=None,
        #     num_processes=1,
        # )
        total_slices = len(imgs)  # progress bar = #images
        iterator = build_nnunet_data_iterator_respecting_plans(
            predictor, imgs, props, prev_preds
        )

        wrapped_iterator = ProgressIterator(
            iterator,  # dummy iterable; we update manually
            total=total_slices,
            name="nnUNet-inference-iter",
            log_every=1,
        )

        # ------------------------------------------------------------------
        # run nnU-Net inference (one call per “image”)
        # ------------------------------------------------------------------
        get_probabilities = sinp.config.get("get_probabilities", True)
        segmentations: list[np.ndarray] = []
        probabilities: list[np.ndarray] | None = [] if get_probabilities else None

        log_nn = logging.getLogger("nnunet")

        with log_std(log_nn):
            # use the iterator inference
            seg_result = predictor.predict_from_data_iterator(
                data_iterator=wrapped_iterator,
                save_probabilities=get_probabilities,
                num_processes_segmentation_export=1,
            )

            if get_probabilities:
                for i, (seg, prob) in enumerate(seg_result):
                    segmentations.append(seg)
                    probabilities.append(prob)
            else:
                segmentations.append(seg_result)  # type: ignore[arg-type]

            # wrapped_iterator.update(1)  # manual tick for the progress bar

        # Check if there is a inhomogenous part in the segmentation, if so, we
        # resize that last part, before then cropping back to the original total shape
        if len(segmentations) > 1:
            if segmentations[-1].shape != segmentations[0].shape:
                logger.info(
                    f"Last part of segmentation has a different shape ({segmentations[-1].shape}) than the first part ({segmentations[0].shape}). Resizing last part to match the first part."
                )

                # simply pad the last part to match the first part but only in
                # the z axis
                pad_z = segmentations[0].shape[0] - segmentations[-1].shape[0]
                # we dont pad the other axes, we just accept that theres something wrong if that is the case
                segmentations[-1] = np.pad(
                    segmentations[-1], ((0, pad_z), (0, 0), (0, 0)), mode="edge"
                )

                segmentation_array = np.concatenate(segmentations, axis=0)
                segmentation_array = segmentation_array[
                    : input_data.meta["original_shape"][0], :, :
                ]
            else:
                segmentation_array = np.asarray(segmentations)
        else:
            segmentation_array = np.asarray(segmentations)

        # flatten potential (N, 1, H, W) → (N, H, W)
        if segmentation_array.ndim == 4 and segmentation_array.shape[1] == 1:
            segmentation_array = segmentation_array.squeeze(axis=1)

        # handle stacked batches (e.g. 2-D model on 3-D volume)
        if segmentation_array.ndim == 4 and segmentation_array.shape[1] > 1:
            n_slices = segmentation_array.shape[0] * segmentation_array.shape[1]
            segmentation_array = segmentation_array.reshape(
                n_slices, *segmentation_array.shape[2:]
            )
            if segmentation_array.shape[0] != sinp.data_array.shape[0]:
                segmentation_array = segmentation_array[: sinp.data_array.shape[0]]

        probabilities_array = None
        # if probabilities is not None:
        #     probabilities_array = np.concatenate(probabilities, axis=0)
        #     if probabilities_array.ndim == 5 and probabilities_array.shape[2] == 1:
        #         probabilities_array = probabilities_array.squeeze(axis=2)

        #     if probabilities_array.ndim == 5 and probabilities_array.shape[2] > 1:
        #         n_slices = probabilities_array.shape[0] * probabilities_array.shape[2]
        #         probabilities_array = probabilities_array.reshape(
        #             n_slices,
        #             probabilities_array.shape[1],
        #             *probabilities_array.shape[3:],
        #         )
        #         if probabilities_array.shape[0] != sinp.data_array.shape[0]:
        #             probabilities_array = probabilities_array[
        #                 : sinp.data_array.shape[0]
        #             ]

        return InferenceOutput(
            group_index=sinp.group_index,
            dataset_id=sinp.dataset_id,
            inference_id="nnunetv2_iter",
            config={"checkpoint_path": str(ckpt_dir)},
            segmentation_mask={
                "segmentation_list": segmentations,
                "mask": segmentation_array,
            },
            pipeline_input=sinp,
            labels_mapping=labels_mapping,
            probabilities_array=probabilities_array,
        )


# ───────────────────── predictor factory ─────────────────────────
def _build_predictor(
    ckpt_dir: Path,
    ckpt_name: str,
    plans_override: dict[str, Any] | None,
) -> nnUNetPredictor:
    """
    Build nnUNetPredictor. If *plans_override* is provided, use manual
    initialization; otherwise fall back to
    `initialize_from_trained_model_folder`.
    """
    # deferred import to avoid slow app load at init
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.inference.predict_from_raw_data import load_json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = nnUNetPredictor(
        tile_step_size=0.9,
        device=device,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        verbose_preprocessing=True,
        verbose=True,
        allow_tqdm=False,
    )

    if plans_override is None:

        def get_available_folds() -> tuple[int | str]:
            folds = []
            for d in ckpt_dir.glob("fold_*"):
                if (d / ckpt_name).is_file():
                    fold_str = d.name.split("_")[-1]
                    if fold_str.isdigit():
                        folds.append(int(fold_str))
                    else:
                        folds.append(fold_str)
            return tuple(sorted(folds))

        available_folds = get_available_folds()
        # we dont want to high jack the os.environ to set torch compile to true (nnunet checks
        # the nnUNet_compile if it should compile the model or not, which is set via an env variable
        # so we set it to true, but then set it back to whatever it was before
        original_torch_compile = os.environ.get("nnUNet_compile", "0")
        # os.environ["nnUNet_compile"] = "1"

        pred.initialize_from_trained_model_folder(
            str(ckpt_dir),
            use_folds=available_folds,
            checkpoint_name=ckpt_name,
        )
        os.environ["nnUNet_compile"] = original_torch_compile
        return pred

    # manual init with custom plans
    from nnunetv2.utilities.label_handling.label_handling import (
        determine_num_input_channels,
    )
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    import nnunetv2

    dataset_json = load_json(str(ckpt_dir / "dataset.json"))
    pm = PlansManager(plans_override)

    # assume first fold for meta
    fold_dir = next(
        (d for d in ckpt_dir.glob("fold_*") if (d / ckpt_name).is_file()), None
    )
    if fold_dir is None:
        raise RuntimeError(f"No fold dirs with {ckpt_name} in {ckpt_dir}")

    ckpt = torch.load(fold_dir / ckpt_name, map_location="cpu")
    trainer_name = ckpt["trainer_name"]
    cfg_name = ckpt["init_args"]["configuration"]

    cfg_mgr = pm.get_configuration(cfg_name)

    channels = determine_num_input_channels(pm, cfg_mgr, dataset_json)
    trainer_cls = recursive_find_python_class(
        (nnunetv2.__path__[0] / "training" / "nnUNetTrainer").as_posix(),  # type: ignore[union-attr] #this is very specific to nnunetv2
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )
    if trainer_cls is None:
        raise RuntimeError(f"Trainer {trainer_name} not found")

    net = trainer_cls.build_network_architecture(
        cfg_mgr.network_arch_class_name,
        cfg_mgr.network_arch_init_kwargs,
        cfg_mgr.network_arch_init_kwargs_req_import,
        channels,
        pm.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False,
    )
    net.load_state_dict(ckpt["network_weights"])

    # gather weights for all folds
    weights = [
        torch.load(f / ckpt_name, map_location="cpu")["network_weights"]
        for f in ckpt_dir.glob("fold_*")
        if (f / ckpt_name).is_file()
    ]
    pred.manual_initialization(
        network=net,
        plans_manager=pm,
        configuration_manager=cfg_mgr,
        parameters=weights,
        dataset_json=dataset_json,
        trainer_name=trainer_name,
        inference_allowed_mirroring_axes=ckpt.get("inference_allowed_mirroring_axes"),
    )
    return pred


# ─────────────────────────── 3. Post-process ─────────────────────
@register_step(
    HookType.SEGMENTATION_POSTPROCESS,
    "nnunetv2_postprocess",
    bucket_id="nnunetv2",
    override=True,
)
class nnUNetv2Postprocessor(StepPlugin[InferenceOutput, nnUNetPostprocessOutput]):
    """
    Converts nnUNet label map → mask mask & extracts a simple bounding
    box coordinate dict (you can swap for connected-component filtering,
    morphological ops, etc.).
    """

    optional = False

    def process(self, input_data: InferenceOutput) -> nnUNetPostprocessOutput:
        mask = input_data.segmentation_mask["mask"]
        # Keep the original (Z, Y, X) orientation returned by nnUNet; ensure contiguous for downstream use.
        mask = np.ascontiguousarray(mask)

        # # fix the transposition back for 2d
        # if input_data.pipeline_input.config["nnunet_model_type"] == "2d":
        #     mask = np.transpose(mask, (0, 2, 1))  # (N, Z, X, Y) → (N, Z, Y, X)

        # # If the model is 3d, we also need to transpose the mask back to (N, Z, Y, X)
        # if input_data.pipeline_input.config["nnunet_model_type"] == "3d_fullres":

        labels = input_data.labels_mapping.get("labels", {})

        # create a mask for each label, where the mask is 1 for the label and 0 for the rest
        # we want to use the smallest possible dtype (bool or uint8) to save memory
        # for label, label_id in labels.items():
        #     if label_id == 0:
        #         continue

        # bounding box in numpy slice coords
        coords = self._bbox_from_mask(mask)

        report = {
            "voxels_foreground": int(mask.sum()),
            "bbox": coords,
        }

        logger.info(f"Final nnUnet Mask shape: {mask.shape}")
        return nnUNetPostprocessOutput(
            group_index=input_data.group_index,
            dataset_id=input_data.dataset_id,
            inference_id=input_data.inference_id,
            segmentation_mask={"mask": mask},
            # for now the coords arent really calculated, but in the future we want to return a calculated bounding box dict
            segmentation_coordinates=coords,
            report=report,
            labels_mapping=input_data.labels_mapping,
            probabilities_array=input_data.probabilities_array,
            pipeline_input=input_data.pipeline_input,
            metadata=input_data.pipeline_input.metadata,
        )
    # -------------------- helpers -------------------------------
    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> dict[str, int]:
        if mask.max() == 0:
            return dict(x=0, X=0, y=0, Y=0, z=0, Z=0)  # empty
        idx = np.where(mask)
        return dict(
            z=int(idx[0].min()),
            Z=int(idx[0].max()),
            y=int(idx[1].min()),
            Y=int(idx[1].max()),
            x=int(idx[2].min()),
            X=int(idx[2].max()),
        )
