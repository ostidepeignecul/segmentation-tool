from __future__ import annotations

import hashlib
import logging
import zipfile
from pathlib import Path
from typing import Tuple

from collections.abc import Iterable
import shutil
from plugins.hook_types import nnUNetModelTypes

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO


import logging, time
from typing import Any, Dict

from collections.abc import Iterator
import numpy as np

logger = logging.getLogger(__name__)


# only function that needs to be called for parsing the folder structure
def parse_folder_structure(
    exported_model_folder: Path,
    model_type: nnUNetModelTypes | None = None,
) -> tuple[list[tuple[Path, int]] | None, str | None]:
    """
    Locate nnUNet trainer directories inside *exported_model_folder* (or inside the
    extracted contents when *exported_model_folder* is a ``.zip``) and return them
    ordered by the number of epochs (descending).

    If *exported_model_folder* is a ZIP archive, the archive is extracted **once**
    to ``exported_model_folder.parent / exported_model_folder.stem`` unless an
    existing extraction with an identical ``plans.json`` already exists.
    """
    model_root = _ensure_directory(exported_model_folder)
    return _parse_trainer_dirs(model_root, model_type)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_directory(path: Path) -> Path:
    """Return a directory path, extracting *path* if it is a ZIP."""
    if path.is_dir():
        return path

    if path.suffix != ".zip":
        raise ValueError(f"{path} is neither a directory nor a .zip archive.")

    if not path.is_file():
        raise ValueError(f"ZIP archive {path} does not exist.")

    dest = path.parent / path.stem

    if dest.exists() and _same_model_as_zip(path, dest):
        logger.info(f"Re-using previously extracted model at {dest}")
        return dest

    logger.info(f"Extracting {path.name} -> {dest}")
    # If the destination exists, remove it completely, so that we can extract a fresh copy
    # otherwise, we might end up with a mix of old and new files as zopfile does not overwrite existing files
    if dest.exists():
        shutil.rmtree(dest)  # wipe the folder completely
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest)

    return dest


def _same_model_as_zip(zip_path: Path, extracted_dir: Path) -> bool:
    """
    Return *True* if the ``plans.json`` inside *zip_path* matches the one inside
    *extracted_dir* (SHA-256 digest comparison).  Reads **only** those two files.
    """
    # ── read plans.json from ZIP ───────────────────────────────────────────
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zip_member = _first_match(zf.namelist(), lambda n: n.endswith("plans.json"))
            if zip_member is None:
                logger.warning(
                    f"No plans.json inside {zip_path.name} - cannot validate."
                )
                return False
            zip_digest = _sha256_bytes(zf.read(zip_member))
    except Exception as e:  # corrupt zip etc.
        logger.warning(f"Failed reading {zip_path.name}: {e}")
        return False

    # ── read plans.json from extracted dir ────────────────────────────────
    try:
        fs_plans_path = next(extracted_dir.rglob("plans.json"))
    except StopIteration:
        logger.warning(f"No plans.json inside extracted dir {extracted_dir}")
        return False
    except Exception as e:
        logger.warning(f"Error searching plans.json in {extracted_dir}: {e}")
        return False

    try:
        fs_digest = _sha256_file(fs_plans_path)
    except Exception as e:
        logger.warning(f"Failed reading {fs_plans_path}: {e}")
        return False

    same = zip_digest == fs_digest
    if not same:
        logger.info("plans.json differs - will re-extract.")
    return same


def _parse_trainer_dirs(
    root: Path, model_type: str | None
) -> tuple[list[tuple[Path, int]] | None, str | None]:
    """
    Collect ``nnUNetTrainer_*__*__{model_type}`` directories directly under *root*
    **or one level deeper** and return ``[(path, epochs), …]`` sorted by epochs
    (descending).  Returns *None* if none are found. Returns also the
    *model_type* if it was not provided, or *None* if it could not be determined.
    """
    trainer_dirs = list(_trainer_candidates(root))
    if not trainer_dirs:
        logger.warning(f"No nnUNetTrainer directories found in {root}")
        return None, model_type

    parsed: list[tuple[Path, int]] = []
    for d in trainer_dirs:
        parts = d.name.split("__")
        if len(parts) < 2 or not parts[0].startswith("nnUNetTrainer"):
            continue

        try:
            epochs = int(parts[0].split("_")[-1].removesuffix("epochs"))
        except ValueError:
            logger.warning(f"Cannot parse epoch count from {d.name} - skipping.")
            continue

        if model_type is None:
            model_type = _get_model_type_from_trainer_dir(d)
            if model_type is None:
                logger.warning(f"Cannot determine model type from {d.name} - skipping.")
                continue
        if model_type not in d.name:
            logger.debug(f"Model type {model_type} not in {d.name} - skipping.")
            continue

        parsed.append((d, epochs))

    parsed.sort(key=lambda t: t[1], reverse=True)
    return (parsed or None), model_type


def _get_model_type_from_trainer_dir(trainer_dir: Path) -> str | None:
    """
    Return the model type from a trainer directory name, e.g. ``nnUNetTrainerV2__2d__nnUNetPlans``.
    Returns *None* if the model type cannot be determined.
    """
    parts = trainer_dir.name.split("__")
    if len(parts) < 2 or not parts[0].startswith("nnUNetTrainer"):
        # Try to glob for any trainer directory inside and parse that instead
        candidate = _first_match(
            (p for p in trainer_dir.iterdir() if p.is_dir()),
            lambda d: d.name.startswith("nnUNetTrainer") and "__" in d.name,
        )
        if candidate is not None:
            return _get_model_type_from_trainer_dir(Path(candidate))
        return None

    # the third part is the model type
    # assuming: 'nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres' for example
    model_type = parts[2]
    if model_type not in ("2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"):
        return None

    return model_type


def _trainer_candidates(root: Path) -> Iterable[Path]:
    """
    Yield candidate trainer directories in *root* or, if none found, one level
    deeper.
    """
    first_level = [
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("nnUNetTrainer")
    ]
    if first_level:
        yield from first_level
        return

    # look one level deeper
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        matches = [
            d
            for d in sub.iterdir()
            if d.is_dir() and d.name.startswith("nnUNetTrainer")
        ]
        if matches:
            yield from matches
            break
    # otherwise, we want to glob the entire tree for any trainer dirs
    for d in root.rglob("nnUNetTrainer*__*__*"):
        if d.is_dir() and d.name.startswith("nnUNetTrainer"):
            yield d


# ──────────────────────────────────────────────────────────────────────────────
# Mini-utilities
# ──────────────────────────────────────────────────────────────────────────────
def _first_match(items: Iterable[str], predicate) -> str | None:
    """Return the first element in *items* satisfying *predicate*, else *None*."""
    return next((it for it in items if predicate(it)), None)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, buf_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


# ---------- colour helpers --------------------------------------------------
try:
    from colorama import Fore, Style, init as _cinit

    _cinit(autoreset=True)
    C, Y, G, M, R = Fore.CYAN, Fore.YELLOW, Fore.GREEN, Fore.MAGENTA, Style.RESET_ALL
except ModuleNotFoundError:  # colourless fallback
    C = Y = G = M = R = ""

try:  # memory (optional)
    import psutil

    _PROC = psutil.Process()
except ModuleNotFoundError:
    _PROC = None

log = logging.getLogger(__name__)


class ProgressIterator(Iterator[dict[str, Any]]):  # each item is *at least* a dict
    """
    Wrap *any* iterable that nnU-Net can consume and emit rich progress
    without altering the underlying data.

    Works with:
      • the built-in tuple iterator from `get_data_iterator_from_raw_npy_data`
      • a DIY dict iterator per the “Predicting with a custom iterator” section
    """

    def __init__(
        self,
        base_iter: Iterable[Any],
        total: int,
        *,
        name: str = "nnUNet",
        log_every: int | None = None,
    ) -> None:
        self._inner = iter(base_iter)
        self._total = max(total, 1)
        self._name = name
        self._log_every = log_every
        self._n = 0
        self._t0 = self._t_prev = time.perf_counter()

    def __iter__(self) -> ProgressIterator:
        return self

    def __next__(self):
        item = next(self._inner)
        self._n += 1
        if self._log_every is None or self._n % self._log_every == 0:
            self._log(item)
        return item

    def __len__(self) -> int:  # nnU-Net queues rely on this
        return self._total

    def update(self, total: int) -> None:
        """
        Update the iteration cycle total. Useful to manually call for a "tick" when
        the total number of items is not known in advance.

        """
        if total < 1:
            raise ValueError("Total must be a positive integer.")
        self._total = total
        self._t0 = time.perf_counter()

    # ----------------------------------------------------------- internals ---
    def _extract_ndarray(self, item: Any) -> np.ndarray:
        """
        Grab the first ndarray / tensor in `item` for shape reporting.
        Accepts tuples, dicts or custom objects with a `.data` attribute.
        """
        if isinstance(item, dict):
            return np.asarray(item["data"])
        if isinstance(item, tuple):
            return np.asarray(item[0])
        # fall back to attr lookup
        return np.asarray(getattr(item, "data"))

    def _log(self, item: Any) -> None:
        now = time.perf_counter()
        dt = now - self._t_prev
        elapsed = now - self._t0
        avg = elapsed / self._n
        eta = max(self._total - self._n, 0) * avg

        arr = self._extract_ndarray(item)
        mem = ""
        if _PROC is not None:
            mem = f"{M}{_PROC.memory_info().rss / 1_048_576:6.1f} MiB{R}"

        width = len(str(self._total))
        log.info(
            f"[{self._n:>{width}}/{self._total}] "
            f"{Y}deltat={dt:5.2f}s{R} "
            f"{G}avg={avg:4.2f}s ETA={eta:5.1f}s{R} "
            f"shape={tuple(arr.shape)} {arr.dtype} "
            f"{R}Mem usage: {mem} {R} "
            f"({self._n / elapsed:5.2f} it/s)",
        )
        self._t_prev = now


# Currently this is useful to suppress noisy prints from nnUNet
# but could be extracted to be used by other plugins if needed


@contextmanager
def log_std(logger: logging.Logger, level=logging.INFO):
    """
    Capture *all* text written to sys.stdout / sys.stderr and emit it
    via `logger.log(level, ...)` instead.  No global side-effects.
    used like this:
        log = logging.getLogger("nnunet")

        with log_std(log, logging.DEBUG):
            segs = predictor.predict_from_data_iterator(
                iterator,
                save_probabilities=False,
                num_processes_segmentation_export=3,
            )
    """

    class _Stream(StringIO):
        def write(self, s):  #  type: ignore[override]
            if s.strip():
                logger.log(level, s.rstrip())

        # flush() inherited from StringIO is fine

    buf = _Stream()
    with redirect_stdout(buf), redirect_stderr(buf):
        yield


# from monai.data import Dataset, DataLoader, decollate_batch
# from nnunetv2.training.ne
# from monai.transforms import (
#     Compose,
#     EnsureTyped,
#     Invertd,
#     Spacingd,
#     LoadImaged,
#     NormalizeIntensityd,
#     EnsureChannelFirstd,
#     DivisiblePadd,
#     Orientationd,
#     ResizeWithPadOrCropd,
#     ThresholdIntensityd,
# )

# NNUNET global params
INIT_FILTERS = 32
ENABLE_DS = True

nnunet_plans = {
    "UNet_class_name": "PlainConvUNet",
    "UNet_base_num_features": INIT_FILTERS,
    "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2],
    "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
    "pool_op_kernel_sizes": [
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [1, 2, 2],
    ],
    "conv_kernel_sizes": [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    # NOTE: starting from contrast-agnostic v2.5, the monai-based nnunet model has more features at
    # the deeper layers of the network, hence update the max features in the `plans` dict
    "unet_max_num_features": 384,
}


# def create_nnunet_from_plans(path_model, device: torch.device):
#     """
#     Adapted from nnUNet's source code:
#     https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/get_network_from_plans.py#L9

#     """
#     plans = nnunet_plans
#     num_input_channels = 1
#     num_classes = 1
#     deep_supervision = ENABLE_DS

#     num_stages = len(plans["conv_kernel_sizes"])

#     dim = len(plans["conv_kernel_sizes"][0])
#     conv_op = convert_dim_to_conv_op(dim)

#     segmentation_network_class_name = plans["UNet_class_name"]
#     mapping = {
#         "PlainConvUNet": PlainConvUNet,
#         "ResidualEncoderUNet": ResidualEncoderUNet,
#     }
#     kwargs = {
#         "PlainConvUNet": {
#             "conv_bias": True,
#             "norm_op": get_matching_instancenorm(conv_op),
#             "norm_op_kwargs": {"eps": 1e-5, "affine": True},
#             "dropout_op": None,
#             "dropout_op_kwargs": None,
#             "nonlin": nn.LeakyReLU,
#             "nonlin_kwargs": {"inplace": True},
#         },
#         "ResidualEncoderUNet": {
#             "conv_bias": True,
#             "norm_op": get_matching_instancenorm(conv_op),
#             "norm_op_kwargs": {"eps": 1e-5, "affine": True},
#             "dropout_op": None,
#             "dropout_op_kwargs": None,
#             "nonlin": nn.LeakyReLU,
#             "nonlin_kwargs": {"inplace": True},
#         },
#     }
#     assert segmentation_network_class_name in mapping.keys(), (
#         "The network architecture specified by the plans file "
#         "is non-standard (maybe your own?). Yo'll have to dive "
#         "into either this "
#         "function (get_network_from_plans) or "
#         "the init of your nnUNetModule to accomodate that."
#     )
#     network_class = mapping[segmentation_network_class_name]

#     conv_or_blocks_per_stage = {
#         "n_conv_per_stage"
#         if network_class != ResidualEncoderUNet
#         else "n_blocks_per_stage": plans["n_conv_per_stage_encoder"],
#         "n_conv_per_stage_decoder": plans["n_conv_per_stage_decoder"],
#     }

#     # network class name!!
#     model = network_class(
#         input_channels=num_input_channels,
#         n_stages=num_stages,
#         features_per_stage=[
#             min(plans["UNet_base_num_features"] * 2**i, plans["unet_max_num_features"])
#             for i in range(num_stages)
#         ],
#         conv_op=conv_op,
#         kernel_sizes=plans["conv_kernel_sizes"],
#         strides=plans["pool_op_kernel_sizes"],
#         num_classes=num_classes,
#         deep_supervision=deep_supervision,
#         **conv_or_blocks_per_stage,
#         **kwargs[segmentation_network_class_name],
#     )
#     model.apply(InitWeights_He(1e-2))
#     if network_class == ResidualEncoderUNet:
#         model.apply(init_last_bn_before_add_to_0)

#     # this loop only takes about 0.2s on average on a CPU
#     chkp_paths = glob.glob(os.path.join(path_model, "**", "*.ckpt"), recursive=True)
#     if not chkp_paths:
#         raise FileNotFoundError(
#             f"Could not find .ckpt (i.e. model checkpoint) file in {path_model}"
#         )
#     chkp_path = chkp_paths[0]
#     checkpoint = torch.load(chkp_path, map_location=torch.device(device))["state_dict"]
#     # NOTE: remove the 'net.' prefix from the keys because of how the model was initialized in lightning
#     # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
#     for key in list(checkpoint.keys()):
#         if "net." in key:
#             checkpoint[key.replace("net.", "")] = checkpoint[key]
#             del checkpoint[key]

#     # load the trained model weights
#     model.load_state_dict(checkpoint)
#     model.to(device)
#     model.eval()

#     return model
