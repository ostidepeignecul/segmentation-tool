"""
───────────────────────────────────────────────────────────────────────────────
Multi-Step Plugin Execution Framework
───────────────────────────────────────────────────────────────────────────────

Overview
========
The framework lets you compose *typed* plugin pipelines (“workflows”)
and execute **many independent workflows concurrently** through a single
`PluginExecutor`.  A workflow is an *ordered list* of **HookType**
categories (pre-process → inference → post-process …).  For every
HookType you can register one or more *plugins*; the **first enabled
plugin** that matches the step is instantiated and its
``process(input) -> output`` result is fed into the next step.

Key building blocks
===================

              ┌────────────────────────┐
              │      StepPlugin        │  (one unit of work)
              └────────────────────────┘
                        ▲  ▲
                        │  └─ registered via
                        │     @register_step(HookType, plugin_id)
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              _STEP_REGISTRIES[HookType]                                  │
│    {plugin_id: StepPluginSubclass, …}  +  _STEP_ENABLED toggle flags     │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ PipelineConfig                                                           │
│   • workflow  : List[HookType]           order of steps                  │
│   • overrides : {HookType: plugin_id?}   force a particular plugin       │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ PipelinePluginManager(IPluginManager)                                    │
│   • owns ONE PipelineConfig                                              │
│   • iterate workflow, resolve plugin, run .process(), chain output       │
│   • signature  execute(input, metadata, cb) ← keeps old API intact       │
└──────────────────────────────────────────────────────────────────────────┘

           (zero-or-more)
                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ PluginExecutor                                                           │
│   • holds configs   {HookType -> PluginManagerConfig}                    │
│   • lazily instantiates managers on first use                            │
│   • enqueue tasks in ThreadManager                                       │
│   • publishes "done" event to EventBus when manager says it is finished  │
└──────────────────────────────────────────────────────────────────────────┘

Typical data-flow
=================

User code
─────────
        + register plugin classes
        + build one or many PipelineConfig objects
        + executor.register_plugin_manager_config(hook_type, PipelinePluginManager, cfg)
        + executor.execute_step(hook_type, input, meta)

Runtime
───────
                (1) Executor enqueues task ────────────────────────────┐
                (2) ThreadManager pulls task, calls manager.execute()  │
                (3) Manager walks its workflow:                        │
                    ┌─Step 1 (HookType.A)──┐                           │
      input ───>    │  plugin_A.process()  │                           │
                    └──────────────────────┘                           │
                                   │ output                            │
                    ┌─Step 2 (HookType.B)──┐                           │
                    │  plugin_B.process()  │                           │
                    └──────────────────────┘                           │
                                   │ …                                 │
                (4) On last step, manager invokes completion_callback  │
                (5) Manager.finish_task() → Executor publishes "done"  │
                                                                       │
               └───────────────────────────────────────────────────────┘

ASCII sequence diagram
──────────────────────
Caller            Executor          ThreadManager         Manager            Plugin
│ execute_step()  │                │                     │                   │
│───────────────->│ queue task     │                     │                   │
│                │────────────────->│ pulls task         │                   │
│                │                 │────────────────────>│ start_task()      │
│                │                 │                     │─┐                 │
│                │                 │                     │ │process step 1   │
│                │                 │                     │ │process step 2   │
│                │                 │                     │ │…                │
│                │                 │                     │─┘                 │
│                │                 │                     │ completion_cb()   │
│                │                 │                     │ finish_task()     │
│                │                 │<────────────────────│                   │
│ done event <───│                 │                                         │
│                                                                       …    │

Defining plugins
================
```python
@register_step(HookType.SEGMENTATION_PREPROCESS, "basic_norm")
class BasicNormalisation(StepPlugin[SegInferenceInput, PreprocessOutput]):
    optional = True                    # safe to disable
    def process(self, data): …
Switching plugins on/off at runtime
───────────────────────────────
from pipeline_plugins import enable_plugin
enable_plugin("basic_norm", False)     # disable globally


Multiple workflows
───────────────────────────────
Create a separate PipelinePluginManager (or MultiWorkflowManager)
per workflow and register each with the executor under a root HookType.


seg_cfg  = PipelineConfig(workflow=[PRE, INF, POST])
alt_cfg  = PipelineConfig(workflow=[INF, POST],
                          overrides={INF: "unet_alt"})

executor.register_plugin_manager_config(PRE, PipelinePluginManager, seg_cfg)
executor.register_plugin_manager_config(INF, PipelinePluginManager, alt_cfg)
Both pipelines share the same plugin codebase but run different step
sequences or models.

Thread-safety & task counting


When a manager enqueues sub-tasks internally (e.g. via its own worker),
call start_task() before enqueue and finish_task() in the
sub-tasks callback so the executor knows when the high-level task is
really finished.



"""

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
)

from collections.abc import Callable

import numpy as np


from plugins.hook_types import HookType
from plugins.plugin_executor import IPluginManager
import logging
from plugins.utils.async_worker import ThreadedAsyncWorker
from plugins.segmentation_hooks.step_registration import (
    StepPlugin,
    DEFAULT_REGISTRY,
)

logger = logging.getLogger(__name__)


# ────────────────────────── I/O contracts ────────────────────────
@dataclass
class PipelineInput:
    config: dict
    pipeline_id: str
    group_index: int
    data_array: np.ndarray
    raw_data_array: np.ndarray
    # for specific context state of the pipeline. For example,training pipelines wont have teh same context as inference pipelines.
    dataset_id: str
    pipeline_context: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class PreprocessOutput:
    clean_data: np.ndarray
    meta: dict
    pipeline_input: PipelineInput


@dataclass
class InferenceOutput:
    group_index: int
    dataset_id: str
    inference_id: str
    config: dict
    segmentation_mask: dict
    labels_mapping: dict
    probabilities_array: np.ndarray | None
    pipeline_input: PipelineInput


@dataclass
class nnUNetPostprocessOutput:
    group_index: int
    dataset_id: str
    inference_id: str
    segmentation_mask: dict[str, np.ndarray]
    # stores the segmentation in a coordinate format
    probabilities_array: np.ndarray | None
    segmentation_coordinates: dict
    labels_mapping: dict
    report: dict
    pipeline_input: PipelineInput
    metadata: dict = field(default_factory=dict)


@dataclass
class EmptyPipelineOutput:
    """Used to map the empty output to the input, even if the result of the pipelines operations
    is empty. This is useful since we want to keep track of each pipeline run and its results,
    whereas with a simple None return we would lose the information about the run when multiple pipelines are run in parallel."""

    group_index: int
    dataset_id: str
    inference_id: str
    pipeline_input: PipelineInput


# ───────────────────────── config object ──────────────────────────
@dataclass
class PipelineConfig:
    workflow: list[HookType] = field(
        default_factory=lambda: [
            HookType.SEGMENTATION_PREPROCESS,
            HookType.SEGMENTATION_INFERENCE,
            HookType.SEGMENTATION_POSTPROCESS,
        ]
    )
    registry_id: str = "default"
    overrides: dict[HookType, str | None] = field(default_factory=dict)
    context_uid: str = ""


R_Final = TypeVar("R_Final")


class PipelinePluginManager(IPluginManager[PipelineInput, R_Final]):
    """
    Non-blocking, multi-registry plugin manager.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        super().__init__()
        self._registry = DEFAULT_REGISTRY
        self._cfg = config or PipelineConfig()
        self._lock = Lock()
        self._worker = ThreadedAsyncWorker("pipeline_worker")
        self._worker.start()
        self._validate_config(self._cfg)

    # -------- IPluginManager --------------------------------------
    @property
    def hook_type(self) -> HookType:
        return self._cfg.workflow[0]

    def update_plugin_manager_config(self, new_config: PipelineConfig) -> None:
        with self._lock:
            self._validate_config(new_config)
            self._cfg = new_config

    def execute(
        self,
        input_data: PipelineInput,
        metadata: Any,
        completion_callback: Callable[[R_Final], None],
    ) -> None:  # non-blocking
        self.start_task()

        def _pipeline():
            return self._run_sync_pipeline(input_data)

        def _done(res):
            try:
                completion_callback(res)
            finally:
                self.finish_task()

        self._worker.enqueue_task(
            task_function=_pipeline, callback=_done, args=(), kwargs={}
        )

    def cleanup(self) -> None:
        self._worker.stop()

    def initialize_worker(self) -> None: ...

    # -------- internal helpers ------------------------------------
    def _run_sync_pipeline(
        self, pipeline_input: PipelineInput
    ) -> Any | EmptyPipelineOutput:
        pipeline_input = pipeline_input
        total_steps = len(self._cfg.workflow)
        current_step = 0

        # If we are at step 0, the step_output is the input itself (since the step_output is whats fed to the current step)
        step_input = pipeline_input
        # Set a default empty pipeline output to be used in case the pipeline is empty or no steps are run.
        pipeline_output = EmptyPipelineOutput(
            group_index=step_input.group_index,
            dataset_id=step_input.dataset_id,
            inference_id=pipeline_input.pipeline_id,
            pipeline_input=pipeline_input,
        )
        for step in self._cfg.workflow:
            plugin_cls = self._resolve_unique_plugin(step)
            if plugin_cls is None:
                logger.info("Skipping optional step %s", step)
                continue

            logger.debug(
                f"Running {step} via {plugin_cls.plugin_id} - Registry: {self._cfg.registry_id}"
            )
            step_output = plugin_cls().process(step_input)

            step_input = step_output
            current_step += 1
            if current_step == total_steps:
                # If we are at the last step, we need to set the pipeline_output to the step_output
                pipeline_output = step_output
        return pipeline_output

    # -- validation ------------------------------------------------
    def _validate_config(self, cfg: PipelineConfig) -> None:
        if len(cfg.workflow) != len(set(cfg.workflow)):
            raise ValueError("Workflow contains duplicated HookType entries")

        for ht, pid in cfg.overrides.items():
            if pid not in self._registry.get_step_bucket(cfg.registry_id, ht):
                raise ValueError(
                    f"Override '{pid}' not registered for {ht} in registry "
                    f"'{cfg.registry_id}'"
                )

        for ht in cfg.workflow:
            cls = self._resolve_unique_plugin(ht)
            if cls is None and not self._step_optional(cfg.registry_id, ht):
                raise RuntimeError(
                    f"No enabled plugin for mandatory step {ht} "
                    f"in registry '{cfg.registry_id}'"
                )

    def _resolve_unique_plugin(
        self,
        step: HookType,
    ) -> type[StepPlugin] | None:
        reg_id = self._cfg.registry_id
        override = self._cfg.overrides.get(step)

        bucket = self._registry.get_step_bucket(reg_id, step)

        if override:
            return bucket.get(override)

        enabled_cls = [
            cls for pid, cls in bucket.items() if self._registry.is_enabled(reg_id, pid)
        ]
        if not enabled_cls:
            return None
        if len(enabled_cls) > 1:
            raise RuntimeError(
                f"Ambiguous: multiple enabled plugins for {step} in '{reg_id}'"
            )
        return enabled_cls[0]

    def _step_optional(self, reg_id: str, step: HookType) -> bool:
        bucket = self._registry.get_step_bucket(reg_id, step)
        return all(cls.optional for cls in bucket.values())


if __name__ == "__main__":
    """
    Run a tiny end-to-end pipeline to verify that:
      • the registry wiring works
      • each step is discovered and executed
      • callbacks fire and results propagate

    NOTE
    ----
    * A real nnUNet checkpoint folder is required for the inference step
      to succeed.  Set `CHECKPOINT_DIR` to an existing model directory
      (for example one created by `nnUNetv2_train ...`).
    * With an invalid path the pipeline will raise at runtime, proving
      early-validation works.
    """
    import threading
    import time
    import numpy as np

    # init the plugins by importing them
    from plugins.segmentation_hooks.segmentation_plugins import (  # noqa: F401
        nnUNetv2IteratorInference,
        nnUNetv2Preprocessor,
        nnUNetv2Postprocessor,
    )

    CHECKPOINT_DIR = "/Users/AnassElimrani/Downloads/model_2d_foldall.zip"  # Path to a valid but nested nnUNetv2 export folder

    cfg = PipelineConfig(
        workflow=[
            HookType.SEGMENTATION_PREPROCESS,
            HookType.SEGMENTATION_INFERENCE,
            HookType.SEGMENTATION_POSTPROCESS,
        ],
        registry_id="nnunetv2",
    )

    # Instantiate the manager directly (bypassing PluginExecutor)
    manager = PipelinePluginManager(cfg)

    # load array at "C:\Users\AnassElimrani\Documents\inference_gui\oldoldfiles\input\group1_demo_500endviews.npy"
    saved_volume_path = Path(
        "/Users/AnassElimrani/Documents/inference_gui/oldoldfiles/input/group1_demo_500endviews.npy"
    )
    volume = np.load(saved_volume_path, allow_pickle=True)
    seg_inp = PipelineInput(
        config={},  # no override plans for smoke test
        pipeline_id="demo-pipeline",
        group_index=0,
        data_array=volume,
        raw_data_array=volume,
        pipeline_context={"exported_model_folder": CHECKPOINT_DIR},
        dataset_id="demo-case-001",
    )

    done_evt = threading.Event()

    def _on_finish(result: nnUNetPostprocessOutput) -> None:  # type: ignore[arg-type]
        print("\n── pipeline finished ──")
        print("dataset_id :", result.dataset_id)
        print("bbox       :", result.segmentation_coordinates)

        # save the segmentation mask = segmentation_mask={"mask": binary}
        sg_mask = result.segmentation_mask.get("inverted_mask", None)
        if sg_mask is not None:
            np.save("segmentation_mask.npy", sg_mask)
            print("Segmentation mask saved to 'segmentation_mask.npy'")
        done_evt.set()

    manager.execute(seg_inp, metadata=None, completion_callback=_on_finish)

    # wait for the pipeline to finish
    if not done_evt.wait(timeout=300):
        print("Pipeline did not finish within 5 minutes - aborting.")
        manager.cleanup()
    else:
        # give the worker thread a moment to shut down cleanly
        time.sleep(0.5)
        manager.cleanup()
