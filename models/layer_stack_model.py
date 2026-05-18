from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import uuid

import numpy as np

from models.overlay_data import OverlayData


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
        layer_id: Optional[str] = None,
    ) -> "LayerState":
        """Build a layer while normalizing runtime metadata."""
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

    def ensure_active_layer(self) -> Optional[str]:
        """Keep `active_layer_id` consistent with the current layer list."""
        if self.get_layer(self.active_layer_id) is not None:
            return self.active_layer_id
        if not self.layers:
            self.active_layer_id = None
            return None
        self.active_layer_id = str(self.layers[0].id)
        return self.active_layer_id

