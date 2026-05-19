from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import uuid

import numpy as np

from models.overlay_data import OverlayData


@dataclass
class CorrosionLayerState:
    """Corrosion workflow payload persisted per layer."""

    stage: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrosionRuntimeCache:
    """Runtime-only corrosion data that can be rebuilt from layer sources."""

    projection: Optional[Tuple[Any, Tuple[float, float]]] = None
    interpolated_projection: Optional[Tuple[Any, Tuple[float, float]]] = None
    raw_distance_map: Optional[Any] = None
    piece_volume_raw: Optional[Any] = None
    piece_volume_interpolated: Optional[Any] = None
    piece_volume_legacy_raw: Optional[Any] = None
    piece_volume_legacy_interpolated: Optional[Any] = None
    piece_anchor: Optional[Tuple[float, float, float]] = None
    piece_show_interpolated: bool = True
    piece_view_enabled: bool = False


@dataclass
class LayerState:
    """Single annotation layer stored inside a session document."""

    id: str
    name: str
    mask_volume: Optional[np.ndarray]
    label_palette: Dict[int, Tuple[int, int, int, int]]
    label_visibility: Dict[int, bool]
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    overlay_cache: Optional[OverlayData] = None
    layer_kind: str = "annotation"
    corrosion_state: Optional[CorrosionLayerState] = None
    corrosion_runtime_cache: Optional[CorrosionRuntimeCache] = None

    @classmethod
    def create(
        cls,
        *,
        name: str = "Layer 1",
        mask_volume: Optional[np.ndarray],
        label_palette: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
        label_visibility: Optional[Dict[int, bool]] = None,
        visible: bool = True,
        locked: bool = False,
        opacity: float = 1.0,
        overlay_cache: Optional[OverlayData] = None,
        layer_kind: str = "annotation",
        corrosion_state: Optional[CorrosionLayerState] = None,
        corrosion_runtime_cache: Optional[CorrosionRuntimeCache] = None,
        layer_id: Optional[str] = None,
    ) -> "LayerState":
        """Build a layer while normalizing runtime metadata."""
        normalized_kind = str(layer_kind or "annotation").strip().casefold()
        if normalized_kind != "corrosion":
            normalized_kind = "annotation"
            corrosion_state = None
            corrosion_runtime_cache = None
        return cls(
            id=str(layer_id or uuid.uuid4().hex),
            name=str(name or "Layer 1").strip() or "Layer 1",
            mask_volume=mask_volume,
            label_palette=dict(label_palette or {}),
            label_visibility=dict(label_visibility or {}),
            visible=bool(visible),
            locked=bool(locked),
            opacity=max(0.0, min(1.0, float(opacity))),
            overlay_cache=overlay_cache,
            layer_kind=normalized_kind,
            corrosion_state=corrosion_state,
            corrosion_runtime_cache=corrosion_runtime_cache,
        )


@dataclass
class LayerStackModel:
    """Ordered layer collection for one annotation session."""

    layers: list[LayerState] = field(default_factory=list)
    active_layer_id: Optional[str] = None

    def get_layer(self, layer_id: Optional[str]) -> Optional[LayerState]:
        target_id = str(layer_id or "").strip()
        if not target_id:
            return None
        for layer in self.layers:
            if str(layer.id) == target_id:
                return layer
        return None

    def get_active_layer(self) -> Optional[LayerState]:
        self.ensure_active_layer()
        return self.get_layer(self.active_layer_id)

    def list_visible_layers(self) -> list[LayerState]:
        """Return visible layers while preserving stack order."""
        return [layer for layer in self.layers if bool(layer.visible)]

    def set_active_layer(self, layer_id: Optional[str]) -> bool:
        """Select one existing layer as the active editable layer."""
        target = self.get_layer(layer_id)
        if target is None:
            return False
        self.active_layer_id = str(target.id)
        return True

    def ensure_active_layer(self) -> Optional[str]:
        """Keep `active_layer_id` consistent with the current layer list."""
        if self.get_layer(self.active_layer_id) is not None:
            return self.active_layer_id
        if not self.layers:
            self.active_layer_id = None
            return None
        self.active_layer_id = str(self.layers[0].id)
        return self.active_layer_id
