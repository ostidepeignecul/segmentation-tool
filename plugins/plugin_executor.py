from abc import ABC, abstractmethod
from queue import Queue
from threading import Lock
from typing import Any, Generic, TypeVar

from collections.abc import Callable
import threading
import time

from .hook_types import HookType

# Define type variables for input and output types
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
# PluginManagerConfigType = TypeVar("PluginManagerConfigType", bound=TypedDict)
import logging

logger = logging.getLogger(__name__)


class IPluginManager(ABC, Generic[InputType, OutputType]):
    def __init__(self):
        self.task_count = 0
        self.task_count_lock = Lock()
        self.completion_callback = None

    @property
    @abstractmethod
    def hook_type(self) -> HookType:
        pass

    @abstractmethod
    def execute(
        self, input_data: InputType, metadata: Any, completion_callback: Callable
    ) -> OutputType | list[OutputType] | None:
        pass

    @abstractmethod
    def update_plugin_manager_config(self, new_config) -> None:
        pass

    def start_task(self):
        with self.task_count_lock:
            self.task_count += 1

    def finish_task(self):
        with self.task_count_lock:
            self.task_count -= 1
            if self.task_count == 0 and self.completion_callback:
                self.completion_callback(self.hook_type)

    def set_completion_callback(self, callback: Callable):
        self.completion_callback = callback

    @abstractmethod
    def cleanup(self):
        pass

    @abstractmethod
    def initialize_worker(self):
        pass


class ThreadManager:
    """
    Manages a pool of worker threads:
      - Spawns threads when the queue grows.
      - Retires threads that have been idle for too long.
      - Runs in its own manager thread so it doesn't block your main thread.
    """

    def __init__(
        self,
        task_queue: Queue,
        max_workers: int,
        check_interval: float = 1.0,
        idle_timeout: float = 5.0,
    ):
        """
        :param task_queue: The shared execution queue for tasks.
        :param max_workers: The maximum number of active worker threads.
        :param check_interval: How often (secs) the manager thread checks for queue backlog & idle workers.
        :param idle_timeout: Workers idle for this many seconds can be retired.
        """
        self.task_queue = task_queue
        self.max_workers = max_workers
        self.check_interval = check_interval
        self.idle_timeout = idle_timeout

        # Protect shared state
        self._lock = threading.Lock()

        # All live worker threads: maps thread object to its associated state
        self._workers = {}  # type: Dict[threading.Thread, Dict[str, Any]]

        # Shutdown signal for the manager thread
        self._shutdown_event = threading.Event()
        self._manager_thread = threading.Thread(
            target=self._manager_loop, daemon=True, name="ThreadPoolManager"
        )
        self._manager_thread.start()

    def _manager_loop(self):
        """
        Runs in a dedicated manager thread. Periodically checks:
          - how many tasks are waiting,
          - how many workers are idle/busy,
          - how many workers are active,
          and then spawns or retires threads as necessary.
        """
        while not self._shutdown_event.is_set():
            time.sleep(self.check_interval)
            self._adjust_workers()

    def _adjust_workers(self):
        with self._lock:
            # Remove any threads that are no longer alive
            dead_threads = []
            for t, info in self._workers.items():
                if not t.is_alive():
                    dead_threads.append(t)
            for t in dead_threads:
                del self._workers[t]

            # Get the current number of waiting tasks in the queue
            queue_size = self.task_queue.qsize()

            # Count active (busy) versus idle workers; mark idle workers beyond idle_timeout for retirement.
            now = time.time()
            active_workers = 0
            idle_workers = 0
            for t, info in self._workers.items():
                if info["busy"]:
                    active_workers += 1
                else:
                    idle_for = now - info["last_idle_time"]
                    if idle_for >= self.idle_timeout:
                        info["should_stop"] = True
                    idle_workers += 1

            total_workers = len(self._workers)

            # If there are pending tasks and we haven't reached max_workers, spawn additional workers
            if total_workers < self.max_workers and queue_size > 0:
                needed = min(queue_size, self.max_workers - total_workers)
                self._spawn_workers(needed)

    def _spawn_workers(self, count: int):
        for i in range(count):
            t = threading.Thread(
                target=self._worker_loop, daemon=True, name=f"PluginManagerWorker-{i}"
            )
            self._workers[t] = {
                "busy": False,  # Indicates if currently processing a task
                "should_stop": False,  # A signal to indicate when the manager wants this thread to shut down
                "last_idle_time": time.time(),
            }
            t.start()

    def _worker_loop(self):
        """
        Each worker polls the task queue for up to 1 second.
        It uses queue.empty() and then get_nowait() to retrieve a task without relying on exception handling.
        """
        t = threading.current_thread()
        while True:
            with self._lock:
                info = self._workers.get(t)
                if not info or info["should_stop"]:
                    break

            # Poll the queue for a task for up to 1 second
            deadline = time.time() + 1.0
            task = None
            while time.time() < deadline:
                if not self.task_queue.empty():
                    # If the queue is not empty, immediately call get_nowait() to retrieve the task.
                    task = self.task_queue.get_nowait()
                    break
                time.sleep(0.1)  # Sleep briefly before checking again

            if task is None:
                with self._lock:
                    info = self._workers.get(t)
                    if not info or info["should_stop"]:
                        break
                    # Mark thread as idle if no task was found within 1 second
                    info["busy"] = False
                    if info["last_idle_time"] == 0:
                        info["last_idle_time"] = time.time()
                continue

            # Got a task; update state to busy and reset idle time
            with self._lock:
                info = self._workers[t]
                info["busy"] = True
                info["last_idle_time"] = 0

            self._execute_task(task)
            self.task_queue.task_done()

            # After processing, mark thread as idle again
            with self._lock:
                info = self._workers[t]
                info["busy"] = False
                info["last_idle_time"] = time.time()

        # Exiting the loop: the manager will remove this thread from _workers on the next check.

    def _execute_task(self, task):
        """
        Executes the plugin's task. Each task is a tuple of the form:
          (manager, input_data, metadata, callback, hook_type)
        """
        manager, input_data, metadata, callback, hook_type = task
        try:
            manager.execute(input_data, metadata, callback)
        except Exception as e:
            print(f"Error in plugin execution: {e}")

    def stop(self):
        """Graceful shutdown: signal the manager thread & all worker threads to exit, then join them."""
        self._shutdown_event.set()
        self._manager_thread.join(timeout=3.0)

        # Signal all worker threads to stop
        with self._lock:
            for t, info in self._workers.items():
                info["should_stop"] = True

        # Wait for all worker threads to finish
        for t in list(self._workers.keys()):
            t.join(timeout=2.0)

        self._workers.clear()


class PluginExecutor:
    def __init__(self, event_bus: Any, max_workers: int = 1):
        self.plugin_manager_configs: dict[HookType, Any] = {}
        self.plugin_managers: dict[HookType, Any] = {}
        self.event_bus = event_bus
        self.results_queue = Queue()
        self.execution_queue = Queue()

        # The manager starts with 0 threads but can scale up to max_workers
        self.thread_manager = ThreadManager(
            task_queue=self.execution_queue,
            max_workers=max_workers,
            check_interval=1.0,  # manager wakes up every second
            idle_timeout=5.0,  # if a worker is idle for 5 seconds, retire it
        )

    def register_plugin_manager_config(
        self, hook_type: HookType, manager_class: type[IPluginManager], config
    ) -> None:
        self.plugin_manager_configs[hook_type] = PluginManagerConfig(
            manager_class, config, event_bus=self.event_bus
        )

    def update_plugin_manager_config(
        self, hook_type: HookType, manager_class: type[IPluginManager], config
    ) -> None:
        if hook_type in self.plugin_manager_configs:
            # We update the config
            self.plugin_manager_configs[hook_type] = PluginManagerConfig(
                manager_class, config, event_bus=self.event_bus
            )
            # we then delete any existing manager instance to ensure it's re-initialized with new config
            if hook_type in self.plugin_managers:
                del self.plugin_managers[hook_type]
        else:
            raise ValueError(
                f"No configuration registered for hook type {hook_type.value}."
            )

    def get_plugin_manager(self, hook_type: HookType) -> IPluginManager:
        if hook_type not in self.plugin_managers:
            if hook_type in self.plugin_manager_configs:
                config_obj = self.plugin_manager_configs[hook_type]
                manager_instance = config_obj.manager_class(config_obj.config)
                manager_instance.update_plugin_manager_config(config_obj.config)
                self.plugin_managers[hook_type] = manager_instance
            else:
                raise ValueError(
                    f"No plugin manager registered for hook type {hook_type.value}."
                )
        else:
            # Re-apply config if changed
            config_obj = self.plugin_manager_configs[hook_type]
            self.plugin_managers[hook_type].update_plugin_manager_config(
                config_obj.config
            )
        return self.plugin_managers[hook_type]

    def execute_step(self, hook_type: HookType, input_data: Any, metadata: Any) -> None:
        try:
            manager = self.get_plugin_manager(hook_type)
            # If desired, do a deepcopy for thread safety
            # input_data = copy.deepcopy(input_data)
            task = (manager, input_data, metadata, self._on_task_complete, hook_type)
            self.execution_queue.put(task)
        except Exception as e:
            print(f"Error scheduling plugin execution: {e}")

    def initialize_plugin_managers(self):
        for hook_type in self.plugin_manager_configs:
            self.get_plugin_manager(hook_type)

    def _on_task_complete(self, hook_type: HookType):
        hook_type_str = hook_type.value
        self.event_bus.publish("done", hook_type_str, "plugin_executor")

    def get_results(self):
        while not self.results_queue.empty():
            yield self.results_queue.get()

    def stop(self):
        """
        Cleanly shut down all workers and the manager.
        """
        self.thread_manager.stop()


# # Define a generic class for plugin manager configuration
class PluginManagerConfig:
    def __init__(
        self,
        manager_class: type[IPluginManager],
        config: dict,
        event_bus: Any,
    ):
        self.manager_class = manager_class
        self.config = config
        self.event_bus = event_bus


# class PluginExecutor:
#     def __init__(self, event_bus: ProcessingEventBus):
#         self.plugin_manager_configs: Dict[HookType, PluginManagerConfig] = {}
#         self.plugin_managers: Dict[HookType, IPluginManager] = {}
#         self.event_bus = event_bus
#         self.results_queue = Queue()
#         self.execution_queue = Queue()

#     def register_plugin_manager_config(
#         self,
#         hook_type: HookType,
#         manager_class: Type[IPluginManager],
#         config,  # Generic type variable, not TypedDict
#     ) -> None:
#         self.plugin_manager_configs[hook_type] = PluginManagerConfig(
#             manager_class, config, event_bus=self.event_bus
#         )

#     def update_plugin_manager_config(
#         self, hook_type: HookType, new_config: Dict
#     ) -> None:
#         if hook_type in self.plugin_manager_configs:
#             # Update the configuration part of PluginManagerConfig
#             self.plugin_manager_configs[hook_type].config = new_config
#             # Invalidate the existing manager instance to ensure it's re-initialized with new config
#             if hook_type in self.plugin_managers:
#                 del self.plugin_managers[hook_type]
#         else:
#             raise ValueError(
#                 f"No configuration registered for hook type {hook_type.value}."
#             )

#     def get_plugin_manager(self, hook_type: HookType) -> IPluginManager:
#         if hook_type not in self.plugin_managers:
#             if hook_type in self.plugin_manager_configs:
#                 config_obj = self.plugin_manager_configs[hook_type]
#                 # Create a new instance of the manager using the current configuration
#                 self.plugin_managers[hook_type] = config_obj.manager_class(
#                     **config_obj.config
#                 )
#                 # for now, to ensure correct initialization of the worker,
#                 # we have trigger the update even if the config is the same
#                 self.plugin_managers[hook_type].update_plugin_manager_config(
#                     config_obj.config
#                 )
#             else:
#                 raise ValueError(
#                     f"No plugin manager registered for hook type {hook_type.value}."
#                 )
#         else:
#             # Update the plugin manager config if it has changed
#             config_obj = self.plugin_manager_configs[hook_type]
#             self.plugin_managers[hook_type].update_plugin_manager_config(
#                 config_obj.config
#             )
#         return self.plugin_managers[hook_type]

#     def execute_step(self, hook_type: HookType, input_data: Any, metadata: Any) -> None:
#         try:
#             manager = self.get_plugin_manager(hook_type)
#             # Ensure input data is not modified by the plugin, and that it's thread-safe
#             try:
#                 input_data = input_data
#             except Exception as e:
#                 logger.info(f"Cannot deepcopy input data: {e}, using original data")
#             # Pass the _on_task_complete method as a callback
#             thread = Thread(
#                 target=self._execute_manager,
#                 args=(manager, input_data, metadata, self._on_task_complete, hook_type),
#             )
#             thread.start()
#         except Exception as e:
#             print(f"Error in getting or executing plugin manager: {e}")

#     def initialize_plugin_managers(self):
#         for hook_type in self.plugin_manager_configs:
#             self.get_plugin_manager(hook_type)

#     def _on_task_complete(self, hook_type: HookType):
#         hook_type_str = hook_type.value
#         self.event_bus.publish("done", hook_type_str, "plugin_executor")

#     def _execute_manager(
#         self, manager: IPluginManager, input_data: object, metadata, callback, hook_type
#     ):
#         try:
#             # Now the manager.execute also accepts a callback
#             manager.execute(input_data, metadata, callback)
#         except Exception as e:
#             print(f"Error in plugin execution: {e}")

#     def get_results(self):
#         while not self.results_queue.empty():
#             yield self.results_queue.get()
