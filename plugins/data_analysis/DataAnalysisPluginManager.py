import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue as ThreadQueue
from threading import Lock
from typing import Any, Dict, Optional, Type

from collections.abc import Callable

import numpy as np
from sentinelai.plugins.plugin_executor import IPluginManager
from sentinelai.plugins.hook_types import HookType
from sentinelai.utils.mp_async_worker_old import MultiProcessingAsyncWorker

from sentinelai.event_handling.processing_event_bus import ProcessingEventBus
from sentinelai.qt_files.main_window_ui.data_assessment_ui.progress_callback import (
    ProgressCallback,
)


class AnalysisType(Enum):
    PCA = auto()
    FAISS = auto()
    # before implementing a new analysis type, add it here


@dataclass
class DataAnalysisOutput:
    analysis_type: AnalysisType
    analysis_display_name: str
    results: Any  # needs to be any since we can have any type of results,we need to have better type definitions for each step
    metadata: dict | None = None  # Additional metadata from analysis


@dataclass
class DataAnalysisInput:
    analysis_type: AnalysisType
    data: np.ndarray  # This is the group array
    additional_params: dict | None = None  # Additional parameters for analysis
    status_info: dict | None = None  # Status information for analysis
    parent_analysis_output: DataAnalysisOutput | None = (
        None  # Parent analysis output
    )


class DataAnalysisPlugin(ABC):
    """Base class for all data assessment plugins."""

    _plugin_type: AnalysisType

    @abstractmethod
    def perform_analysis(self, input_data: DataAnalysisInput) -> DataAnalysisOutput:
        """Performs the specified analysis on the provided input data and returns the results."""
        pass

    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Unique identifier for the analysis plugin."""
        pass

    @property
    @abstractmethod
    def plugin_type(self) -> AnalysisType:
        """Type of analysis performed by the plugin."""
        return self._plugin_type

    @abstractmethod
    def plugin_display_name(self) -> str:
        """Human-readable name of the analysis plugin."""
        pass


data_analysis_plugin_registry: dict[AnalysisType, type[DataAnalysisPlugin]] = {}


def register_data_analysis_plugin(
    registry: dict[AnalysisType, type[DataAnalysisPlugin]], plugin_type: AnalysisType
):
    def decorator(cls: type[DataAnalysisPlugin]) -> type[DataAnalysisPlugin]:
        if not issubclass(cls, DataAnalysisPlugin):
            raise TypeError(
                f"Plugin {cls.__name__} must inherit from DataAnalysisPlugin"
            )
        if plugin_type in registry:
            return cls  # Plugin already registered

        registry[plugin_type] = cls
        return cls

    return decorator


@dataclass
class DataAnalysisPluginManagerConfig:
    plugin_registry: dict[AnalysisType, type[DataAnalysisPlugin]]
    event_bus: ProcessingEventBus
    callbacks: ProgressCallback


class DataAnalysisPluginManager(IPluginManager[DataAnalysisInput, DataAnalysisOutput]):
    def __init__(
        self,
        config: DataAnalysisPluginManagerConfig ,
    ):
        super().__init__()
        self._config = config
        self._hook_type = HookType.DATA_ANALYSIS
        self.data_analysis_plugin_registry = config.plugin_registry
        self.event_bus = config.event_bus
        self.callbacks = config.callbacks
        self.task_count = 0
        self.task_count_lock = Lock()
        self.completion_callback = None
        self.on_result_queue = ThreadQueue()
        self.worker = MultiProcessingAsyncWorker(
            name=self._hook_type.value
        )  # Reusing the same worker logic
        self.processing_thread = threading.Thread(target=self.process_queue)
        self.processing_thread.daemon = (
            True  # Ensure the thread will exit when the main thread does
        )
        self.processing_thread.start()

    @property
    def hook_type(self):
        return self._hook_type

    def update_plugin_manager_config(self, new_config: dict[str, Any]) -> None:
        pass  # Update the plugin manager configuration if necessary

    def execute(
        self,
        input_data: DataAnalysisInput,
        metadata: Any,
        completion_callback: Callable,
    ):
        self.set_completion_callback(completion_callback)

        # if not self.worker._started:
        if not self.worker.is_started:
            self.initialize_worker()

        self.outputs = []

        # Based on the analysis type, find and execute the appropriate plugin
        for plugin_id, plugin_class in self.data_analysis_plugin_registry.items():
            if plugin_id == input_data.analysis_type:
                self.start_task()  # Increment task counter
                plugin_instance = plugin_class()
                if getattr(plugin_class, "persistent", True):
                    persistent_plugin = plugin_class
                self.callbacks.update_status(
                    f"Starting {plugin_instance.plugin_display_name}...", True
                )
                task_function = plugin_instance.perform_analysis
                self.worker.enqueue_task(
                    # task_function, args=(input_data,), callback=self.on_result, persistent_plugin=persistent_plugin
                    task_function,
                    args=(input_data,),
                    callback=self.on_result,
                )
                break  # Assuming one plugin per analysis type, break after finding

    def on_result(self, result: DataAnalysisOutput):
        self.on_result_queue.put(result)
        self.process_queue()
        self.finish_task()
        if self.task_count == 0:
            self.callbacks.update_status("Data analysis done!", True)
            self.callbacks.analysis_done()

    def process_queue(self):
        while not self.on_result_queue.empty():
            result = self.on_result_queue.get()
            self.outputs.append(result)
            self.callbacks.update_status(
                f"Finished {result.analysis_display_name}", True
            )
            self.callbacks.add_output(result, result.analysis_type.name)
            self.on_result_queue.task_done()

    def stop_worker(self):
        # self.processing_thread.join()  # Ensure the processing thread is properly stopped
        self.worker.stop()

    def start_worker(self):
        # self.processing_thread.start()
        self.worker.start()

    def initialize_worker(self):
        return self.worker.start()

    def cleanup(self):
        return self.worker.stop()
