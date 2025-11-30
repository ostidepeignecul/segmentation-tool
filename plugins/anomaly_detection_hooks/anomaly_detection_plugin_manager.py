from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Type, TYPE_CHECKING

from collections.abc import Callable, Mapping

import numpy as np
from typing_extensions import TypedDict

from sentinelai.inference.inferences_manager import InferencesManager
from sentinelai.event_handling.processing_event_bus import ProcessingEventBus
from sentinelai.inference import AnomalyDetectionWorker
from sentinelai.plugins.hook_types import HookType
from sentinelai.plugins.plugin_executor import IPluginManager
import logging

if TYPE_CHECKING:
    from sentinelai.datasets.data_input_types import GroupedDataset

logger = logging.getLogger(__name__)


@dataclass
class AnomalyInferenceInput:
    """
    Input data for anomaly inference plugins.
    """

    group_index: int

    data_array: np.ndarray
    raw_data_array: np.ndarray
    inference_context: dict | None
    dataset_id: str
    grouped_data: GroupedDataset


@dataclass
class AnomalyInferenceOutput:
    """
    Output data from anomaly inference plugins.
    group_index: int
        The group index of the inference.
    result_arrays: Mapping[str, np.ndarray]
    inference_id: str
        The identifier of the inference.
    inference_type: str
        The type of the inference.
    """

    group_index: int
    dataset_id: str
    inference_id: str
    model_class: str  # Eg. devnet, vitae
    original_data_array: np.ndarray
    result_arrays: Mapping[str, np.ndarray]

    # reference to the input data that was used to generate the output
    anomaly_inference_input: AnomalyInferenceInput
    inferences_settings: dict

    grouped_data: GroupedDataset
    inference_type: str = "anomaly_detection"
    model_checkpoint: Path | None = None


@dataclass
class AnomalyInferenceImageOutput:
    """ """

    group_index: int
    inference_id: str
    image_array: np.ndarray


class AnomalyInferencePlugin(ABC):
    """Base class for all anomaly inference plugins."""

    # For  access before instantiation
    _model_id = ""

    @abstractmethod
    def __init__(self, checkpoint_path: Path | None = None):
        pass

    @abstractmethod
    def process_inference(
        self, input_data: AnomalyInferenceInput
    ) -> AnomalyInferenceOutput:
        """Process the inference and return binary results and scores."""
        pass

    @property
    @abstractmethod
    def inference_id(self) -> str:
        """Unique identifier for the inference model."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifier for the model used in the inference."""
        pass


class AnomalyInferencePluginConfig(TypedDict):
    plugin_registry: dict[str, type[AnomalyInferencePlugin]]
    event_bus: ProcessingEventBus
    registry_id: str
    result_callback: Callable
    context_uid: str


default_anomaly_plugin_registry: dict[str, type[AnomalyInferencePlugin]] = {}
anomaly_inference_plugin_toggle: dict[str, bool] = {}

anomaly_inference_plugin_registry_mapping = {
    "default": default_anomaly_plugin_registry,
}


def register_anomaly_inference_plugin(
    registry: dict[str, type[AnomalyInferencePlugin]],
    inference_id: str,
    auto_enable: bool = True,
    model_path: Path | None = None,
):
    def decorator(cls: type[AnomalyInferencePlugin]) -> type[AnomalyInferencePlugin]:
        if not issubclass(cls, AnomalyInferencePlugin):
            raise TypeError(
                f"Plugin {cls.__name__} must inherit from AnomalyInferencePlugin"
            )
        # for now, overwrite the class name with the inference_id
        # if inference_id in registry:
        #     # If it is, do nothing further
        #     return cls

        registry[inference_id] = cls
        anomaly_inference_plugin_toggle[inference_id] = auto_enable
        return cls

    return decorator


def remove_from_anomaly_inference_plugin_registry(
    registry: dict[str, type[AnomalyInferencePlugin]], inference_id: str
):
    """Remove a plugin from the registry."""
    if inference_id in registry:
        del registry[inference_id]
        anomaly_inference_plugin_toggle[inference_id] = False
    else:
        logger.warning(
            f"Plugin {inference_id} not found in the registry. Cannot remove."
        )


class AnomalyInferencePluginManager(
    IPluginManager[AnomalyInferenceInput, AnomalyInferenceOutput]
):
    def __init__(
        self,
        config: AnomalyInferencePluginConfig,
    ):
        self._hook_type = HookType.ANOMALY_INFERENCE
        self._config = config

        self.event_bus = config["event_bus"]
        self.anomaly_inference_plugin_registry = config["plugin_registry"]
        self.task_count = 0
        self.task_count_lock = Lock()
        self.completion_callback = None
        self.worker = AnomalyDetectionWorker()

    @property
    def hook_type(self):
        return self._hook_type

    def set_plugin_registry(
        self, plugin_registry: dict[str, type[AnomalyInferencePlugin]], registry_id
    ):
        self.anomaly_inference_plugin_registry = plugin_registry
        self.registry_id = registry_id

    def update_plugin_manager_config(self, new_config: AnomalyInferencePluginConfig):
        self._config = new_config

        self.event_bus = new_config["event_bus"]
        self.anomaly_inference_plugin_registry = new_config["plugin_registry"]

    def get_registered_plugins(self) -> dict[str, type[AnomalyInferencePlugin]]:
        return self.anomaly_inference_plugin_registry

    def execute(
        self, input_data: AnomalyInferenceInput, metadata, completion_callback: Callable
    ):
        self.set_completion_callback(callback=completion_callback)
        # plugin_count = len(self.get_registered_plugins())
        if not self.worker.is_started:
            self.initialize_worker()

        self.start_task()  # Increment task counter
        for plugin_name, plugin_class in self.get_registered_plugins().items():
            if anomaly_inference_plugin_toggle[plugin_name] is False:
                continue

            plugin_instance = plugin_class(
                checkpoint_path=metadata[plugin_name]["checkpoint_path"]
            )
            self.worker.enqueue_inference(plugin_instance, input_data, self.on_result)
        self.finish_task()  # Decrement task counter once all tasks are enqueued

    def on_result(self, result: AnomalyInferenceOutput, worker_callback: Callable):
        # Process the result asynchronously
        # worker_callback()

        inference_result_callback = self._config["result_callback"]
        inference_result_callback(
            group_index=result.group_index,
            dataset_id=result.dataset_id,
            original_data_array=result.original_data_array,
            model_class=result.model_class,
            result_arrays=result.result_arrays,
            inference_id=result.inference_id,
            inference_type=result.inference_type,
            inference_settings=result.inferences_settings,
            grouped_dataset=result.grouped_data,
        )
        self.finish_task()
        if self.task_count == 0 and self.completion_callback:
            # self.cleanup()
            self.completion_callback(self.hook_type)

    def stop_worker(self):
        self.worker.stop()
        # self.processing_thread.join()  # Ensure the processing thread is properly stopped

    def start_worker(self):
        self.worker.start()
        # self.processing_thread.start()

    def initialize_worker(self):
        return self.worker.start()

    def cleanup(self):
        return self.worker.stop()
