from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol, Tuple, Type, List, Literal

from collections.abc import Callable

import numpy as np
import pandas as pd


# Shared types that multiple plugins and hooks can use go here
nnUNetModelTypes = Literal["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"]


class HookType(Enum):
    DATAINIT = "datainit"

    ANOMALY_PREPROCESS = "anomaly_preprocess"
    ANOMALY_INFERENCE = "anomaly_inference"
    ANOMALY_POSTPROCESS = "anomaly_postprocess"

    FLAW_DETECTION_PREPROCESS = "flaw_detection_preprocess"
    FLAW_DETECTION_INFERENCE = "flaw_detection_inference"
    FLAW_DETECTION_POSTPROCESS = "flaw_detection_postprocess"

    SEGMENTATION_PREPROCESS = "segmentation_preprocess"
    SEGMENTATION_INFERENCE = "segmentation_inference"
    SEGMENTATION_POSTPROCESS = "segmentation_postprocess"

    PREDRAW = "predraw"
    DRAW = "draw"
    POSTDRAW = "postdraw"
    DATA_ANALYSIS = "analyze"


@dataclass
class PostprocessInput:
    """
    Input data for postprocessing plugins.
    base_inference_array: np.ndarray
        The inference array from the base inference.
    base_indication_table: pd.DataFrame
        The indication table from the base inference.
    """

    group_index: int
    status_info: dict[str, Any]
    base_data_array: np.ndarray
    base_inference_array: np.ndarray
    base_indication_table: pd.DataFrame


@dataclass
class PostprocessOutput:
    modified_array: np.ndarray
    modified_table: pd.DataFrame
    inference_id: str
    modified_labeled_array_lst: list[np.ndarray]


class PluginWorker(Protocol):
    def __init__(self, name: str): ...

    def enqueue_task(
        self,
        task_function: Callable,
        callback: Callable | None = None,
        args: tuple[Any] | Any = (),
        kwargs: dict = {},
        persistent_plugin: type | None = None,
    ): ...

    def start(self): ...

    def stop(self): ...
