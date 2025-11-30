import threading
import time
from queue import Queue

from collections.abc import Callable


class ThreadedAsyncWorker:
    """
    Minimal async worker used by the plugin pipeline.
    Runs tasks in a dedicated thread and invokes an optional callback with the result.
    """

    def __init__(self, name: str):
        self.task_queue: Queue = Queue()
        self.callback_lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._run, daemon=True, name=name)
        self._started = False

    def _run(self) -> None:
        while True:
            task = self.task_queue.get()
            if task is None:
                time.sleep(0.2)
                break

            task_function, args, kwargs, callback = task
            try:
                result = _process(task_function, args, kwargs)
            except Exception as exc:  # propagate errors via callback
                result = exc

            if callback:
                with self.callback_lock:
                    callback(result)

    def enqueue_task(
        self,
        task_function: Callable,
        callback: Callable | None = None,
        args=(),
        kwargs=None,
        *,
        persistent_plugin=None,  # kept for API compatibility, unused here
    ) -> None:
        if kwargs is None:
            kwargs = {}
        self.task_queue.put((task_function, args, kwargs, callback))

    def stop(self) -> None:
        self.task_queue.put(None)
        self.worker_thread.join()
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self.worker_thread.start()

    @property
    def is_started(self) -> bool:
        return self._started

    @is_started.setter
    def is_started(self, value: bool) -> None:
        self._started = value


def _process(task_function: Callable, args, kwargs):
    if isinstance(args, tuple):
        return task_function(*args, **kwargs)
    return task_function(args, **kwargs)
