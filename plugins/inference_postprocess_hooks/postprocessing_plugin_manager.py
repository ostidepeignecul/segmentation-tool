import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from multiprocessing import Lock
from queue import Queue as ThreadQueue
from typing import Any, Dict, List, Optional, Type

from collections.abc import Callable

import numpy as np
import pandas as pd
from typing_extensions import TypedDict

from sentinelai.inference.inferences_manager import InferencesManager
from sentinelai.data_containers.inference_contracts import FlawSizingData
from sentinelai.event_handling.processing_event_bus import ProcessingEventBus
from sentinelai.plugins.hook_types import HookType
from sentinelai.plugins.plugin_executor import IPluginManager
from sentinelai.utils.async_worker import ThreadedAsyncWorker
from sentinelai.utils.mp_async_worker import MultiProcessingAsyncWorker

logger = logging.getLogger(__name__)


# Contract for postprocessing plugins, with the input and output expected from flaw sizing plugins.
@dataclass()
class PostprocessInput:
    """
    Input data for postprocessing plugins.
    base_inference_array: np.ndarray
        The inference array from the base inference.
    base_indication_table: pd.DataFrame
        The indication table from the base inference.
    """

    group_index: int
    dataset_id: str
    status_info: dict[str, Any]
    metadata: dict[str, Any]
    base_data_array: np.ndarray
    base_inference_array: np.ndarray
    base_indication_table: pd.DataFrame
    classes_names: list[str]
    calibration_flaw_ids: pd.DataFrame | None = None


@dataclass()
class PostprocessOutput:
    modified_array: np.ndarray
    modified_table: pd.DataFrame
    modified_labeled_array_lst: list[np.ndarray]
    dataset_id: str
    metadata: dict[str, Any]
    inference_id: str
    group_index: int
    classes_names: list[str]
    flaw_sizing_data: dict[str, FlawSizingData] = field(default_factory=dict)
    flaw_sizing_log: str = field(default_factory=str)


class PostprocessPlugin:
    """Base class for all inference plugins."""

    @abstractmethod
    def process_inference_result(
        self, post_process_input: PostprocessInput
    ) -> PostprocessOutput:
        """Process the inference result and return a modified inference array and indication table."""
        pass


postprocess_plugin_registry: dict[str, type[PostprocessPlugin]] = {}
postprocess_plugin_toggle: dict[str, bool] = {}


def register_postprocess_plugin(
    postprocess_plugin_registry: dict[str, type[PostprocessPlugin]],
    auto_enable: bool = True,
):
    def decorator(cls: type[PostprocessPlugin]) -> type[PostprocessPlugin]:
        if not issubclass(cls, PostprocessPlugin):
            raise TypeError(
                f"Plugin {cls.__name__} must inherit from PostprocessPlugin"
            )
        postprocess_plugin_registry[cls.__name__] = cls
        postprocess_plugin_toggle[cls.__name__] = auto_enable
        return cls

    return decorator


def process_inference_result_wrapper(input_data, plugin_class):
    plugin_instance = plugin_class()
    return plugin_instance.process_inference_result(input_data)


def on_result_static(result_queue, result):
    # This function should only interact with result_queue
    result_queue.put(result)


class PostprocessPluginManagerConfig(TypedDict):
    result_callback: Callable
    plugin_registry: dict[str, type[PostprocessPlugin]]
    event_bus: ProcessingEventBus
    context_uid: str


class PostprocessPluginManager(IPluginManager[PostprocessInput, PostprocessOutput]):
    def __init__(
        self,
        config: PostprocessPluginManagerConfig,
    ):
        self._hook_type = HookType.FLAW_DETECTION_POSTPROCESS
        self._config = config

        self.postprocess_plugin_registry = config["plugin_registry"]
        self.event_bus = self._config["event_bus"]
        self.task_count = 0
        self.task_count_lock = Lock()
        # Callback that tindicates the processing is done, but does not return any data
        self.completion_callback = None
        self.on_result_queue = ThreadQueue()
        self.worker = MultiProcessingAsyncWorker(
            name=self._hook_type.value
        )  # Reusing the same worker logic

        # self.worker = ThreadedAsyncWorker(name=self._hook_type.value)

        # self.processing_thread = threading.Thread(target=self.process_queue)
        # self.processing_thread.daemon = (
        #     True  # Ensure the thread will exit when the main thread does
        # )
        # self.processing_thread.start()
        if isinstance(self.worker, ThreadedAsyncWorker):
            logger.debug("Using Threaded Worker instead of Multiprocess Worker")
        self.initialize_worker()

    @property
    def hook_type(self):
        return self._hook_type

    def update_plugin_manager_config(self, new_config: PostprocessPluginManagerConfig):
        self._config = new_config

        self.event_bus = new_config["event_bus"]
        self.postprocess_plugin_registry = new_config["plugin_registry"]
        if isinstance(self.worker, MultiProcessingAsyncWorker):
            for plugin_class in self.get_registered_plugins().values():
                if getattr(plugin_class, "persistent", True):
                    self.worker._create_persistent_process(
                        plugin_classes=[plugin_class]
                    )

    def get_registered_plugins(self) -> dict[str, type[PostprocessPlugin]]:
        return self.postprocess_plugin_registry

    def execute(self, input_data, metadata, completion_callback):
        self.set_completion_callback(completion_callback)
        if not self.worker._started:
            self.initialize_worker()

        self.start_task()

        for plugin_class in self.get_registered_plugins().values():
            if not postprocess_plugin_toggle[plugin_class.__name__]:
                continue
            plugin_instance = plugin_class()
            task_function = plugin_instance.process_inference_result
            persistent_plugin = None

            # Check if the plugin should be persistent
            if getattr(plugin_class, "persistent", True):
                persistent_plugin = plugin_class
            logger.debug(f"Enqueuing task for plugin: {plugin_class.__name__}")
            self.worker.enqueue_task(
                task_function,
                args=(input_data,),
                kwargs={},
                callback=self.on_result,
                persistent_plugin=persistent_plugin,
            )

        # self.finish_task()  # Decrement task counter once all tasks are enqueued

    def on_result(self, result: PostprocessOutput):
        self.on_result_queue.put(result)
        self.process_queue()

        self.finish_task()

        # self.completion_callback(self.hook_type)

    def process_queue(self):
        while not self.on_result_queue.empty():
            result: PostprocessOutput = self.on_result_queue.get()
            logging.debug(f"Received postprocess result: {result.inference_id}")
            self.add_postprocess(result)
            self.on_result_queue.task_done()

    def add_postprocess(self, result: PostprocessOutput):
        if result is None:
            logger.error(
                "Received None result from postprocess plugin. This is probably due to an error in the plugin."
            )
        if isinstance(result, tuple):
            if result[0] is None:
                logger.error(
                    "Received None result from postprocess plugin. This is probably due to an error in the plugin."
                )
                try:
                    str(result[1])
                    logger.error(f"Error message: {result[1]}")
                except Exception:
                    logger.error("Error message: None")

        logging.debug(
            f"Adding postprocess result to inferences manager: {result.inference_id}"
        )

        result_callback = self._config["result_callback"]
        result_callback(
            group_index=result.group_index,
            inference_array=result.modified_array,
            indication_table=result.modified_table,
            labeled_array_lst=result.modified_labeled_array_lst,
            dataset_id=result.dataset_id,
            classes_names=result.classes_names,
            inference_id=result.inference_id,
            inference_type="flaw_detection",
            flaw_sizing_data=result.flaw_sizing_data,
            flaw_sizing_log=result.flaw_sizing_log,
            metadata=result.metadata,
        )

    def stop_worker(self):
        # self.processing_thread.join()  # Ensure the processing thread is properly stopped
        self.worker.stop()

    #
    def start_worker(self):
        # self.processing_thread.start()
        self.worker.start()

    def initialize_worker(self):
        return self.worker.start()

    def cleanup(self):
        return self.worker.stop()
