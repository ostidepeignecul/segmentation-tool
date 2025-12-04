from abc import ABC, abstractmethod
from dataclasses import field
from collections import defaultdict
from typing import (
    Dict,
    Generic,
    Type,
    TypeVar,
)

from collections.abc import Sequence


from plugins.hook_types import HookType
import logging

logger = logging.getLogger(__name__)


# ────────────────────────── generic step type ────────────────────
T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")


class StepPlugin(ABC, Generic[T_In, T_Out]):
    """
    One pluggable unit (normalisation, UNet inference, PNG export…)
    Subclasses *must* override:
        • step_type      (HookType enum member)
        • plugin_id      (unique string)
        • process()
    Optionally:
        • optional       (skip if no plugin enabled for that step)
        • requires       (list[str] – other plugin_ids that *must* run first)
    """

    # populated by decorator
    step_type: HookType
    plugin_id: str

    optional: bool = False
    requires: Sequence[str] | tuple = field(default_factory=tuple)

    # ---- lifecycle ----------------
    def __init__(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def process(self, input_data: T_In) -> T_Out: ...


# ─────────────────────── registry machinery ───────────────────────
StepCls = type[StepPlugin]
RegistryBucket = dict[str, StepCls]  # {plugin_id: class}


class _PluginRegistry:
    """
    Singleton container.  Holds:

        self._impl[registry_id][hook_type] -> {plugin_id: cls}
        self._toggles[registry_id][plugin_id] -> bool
    """

    def __init__(self) -> None:
        self._impl: dict[str, dict[HookType, RegistryBucket]] = defaultdict(dict)
        self._toggles: dict[str, dict[str, bool]] = defaultdict(dict)

    # ───────── public API ─────────────────────────────────────────
    def register(
        self,
        *,
        bucket_id: str,
        hook_type: HookType,
        plugin_id: str,
        cls: StepCls,
        enable: bool,
        override: bool,
    ) -> None:
        bucket = self._impl[bucket_id].setdefault(hook_type, {})
        if override and plugin_id in bucket:
            logger.warning(
                f"'{plugin_id}' already registered for {hook_type} "
                f"in registry '{bucket_id}'. Overriding with {cls.__name__} (overriding has been enabled)."
            )
            return
        if plugin_id in bucket:
            raise KeyError(
                f"'{plugin_id}' already registered for {hook_type} "
                f"in registry '{bucket_id}'"
            )
        bucket[plugin_id] = cls
        self._toggles[bucket_id].setdefault(plugin_id, enable)

    def get_step_bucket(self, registry_id: str, hook_type: HookType) -> RegistryBucket:
        return self._impl[registry_id].setdefault(hook_type, {})

    def is_enabled(self, registry_id: str, plugin_id: str) -> bool:
        return self._toggles[registry_id].get(plugin_id, False)

    def set_enabled(self, registry_id: str, plugin_id: str, value: bool) -> None:
        if plugin_id not in self._toggles[registry_id]:
            raise KeyError(
                f"Unknown plugin_id '{plugin_id}' in registry '{registry_id}'"
            )
        self._toggles[registry_id][plugin_id] = value

    def remove_plugin(self, registry_id: str, plugin_id: str) -> None:
        for hook_type in self._impl[registry_id].keys():
            if plugin_id in self._impl[registry_id][hook_type]:
                del self._impl[registry_id][hook_type][plugin_id]
                break
        else:
            raise KeyError(
                f"Plugin '{plugin_id}' not found in registry '{registry_id}'"
            )


# single instance used everywhere. This is to ensure that all plugins are registered in the same registry.
DEFAULT_REGISTRY = _PluginRegistry()


def register_step(
    step_type: HookType,
    plugin_id: str,
    override: bool = False,
    *,
    bucket_id: str = "default",
    auto_enable: bool = True,
    registry: _PluginRegistry = DEFAULT_REGISTRY,  # default registry
):
    def decorator(cls: type[StepPlugin]) -> type[StepPlugin]:
        if not issubclass(cls, StepPlugin):
            raise TypeError("Plugin must extend StepPlugin")

        cls.plugin_id = plugin_id
        cls.step_type = step_type

        registry.register(
            bucket_id=bucket_id,
            hook_type=step_type,
            plugin_id=plugin_id,
            cls=cls,
            enable=auto_enable,
            override=override,
        )
        return cls

    return decorator


def enable_plugin(registry_id: str, plugin_id: str, enable: bool = True) -> None:
    DEFAULT_REGISTRY.set_enabled(registry_id, plugin_id, enable)


def disable_plugin(registry_id: str, plugin_id: str) -> None:
    DEFAULT_REGISTRY.set_enabled(registry_id, plugin_id, False)
