from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from models.annotation_model import AnnotationModel
from models.layer_stack_model import CorrosionLayerState, LayerStackModel, LayerState
from models.overlay_data import OverlayData, OverlayLayerData, OverlayStackData
from models.roi_model import ROI, RoiModel
from models.temp_mask_model import TempMaskModel
from models.view_state_model import ViewStateModel


CORROSION_LAYER_VIEW_STATE_KEYS = (
    "corrosion_projection",
    "corrosion_interpolated_projection",
    "corrosion_overlay_volume",
    "corrosion_overlay_palette",
    "corrosion_overlay_label_ids",
    "corrosion_peak_index_map_a",
    "corrosion_peak_index_map_b",
    "corrosion_raw_peak_index_map_a",
    "corrosion_raw_peak_index_map_b",
    "corrosion_raw_distance_map",
    "corrosion_ascan_support_map",
    "corrosion_interpolation_algo",
    "corrosion_peak_selection_mode",
    "corrosion_peak_selection_mode_a",
    "corrosion_peak_selection_mode_b",
    "corrosion_label_a",
    "corrosion_label_b",
    "corrosion_session_stage",
    "corrosion_piece_volume_raw",
    "corrosion_piece_volume_interpolated",
    "corrosion_piece_volume_legacy_raw",
    "corrosion_piece_volume_legacy_interpolated",
    "corrosion_piece_anchor",
    "corrosion_piece_show_interpolated",
    "corrosion_piece_view_enabled",
)


@dataclass
class AnnotationSessionState:
    """Complete annotation session snapshot."""

    name: str
    layer_stack: LayerStackModel
    mask_volume: Optional[np.ndarray]
    label_palette: Dict[int, tuple[int, int, int, int]]
    label_visibility: Dict[int, bool]
    temp_mask_volume: Optional[np.ndarray]
    temp_coverage_volume: Optional[np.ndarray]
    temp_palette: Dict[int, tuple[int, int, int, int]]
    temp_visibility: Dict[int, bool]
    rois: List[ROI]
    next_roi_id: int
    view_state: dict
    overlay_cache: Optional[OverlayData]


class AnnotationSessionManager:
    """Manage multiple annotation sessions in memory."""

    def __init__(self) -> None:
        self._sessions: Dict[str, AnnotationSessionState] = {}
        self._active_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ensure_default(
        self,
        *,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
    ) -> str:
        """Create a default session if none exists."""
        if self._active_id is not None:
            return self._active_id
        return self.create_from_models(
            name="New session",
            annotation_model=annotation_model,
            temp_mask_model=temp_mask_model,
            roi_model=roi_model,
            view_state_model=view_state_model,
            set_active=True,
        )

    def reset_for_new_dataset(
        self,
        *,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
    ) -> str:
        """Reset all sessions when loading a new dataset."""
        self._sessions.clear()
        self._active_id = None
        return self.ensure_default(
            annotation_model=annotation_model,
            temp_mask_model=temp_mask_model,
            roi_model=roi_model,
            view_state_model=view_state_model,
        )

    def list_sessions(self) -> List[tuple[str, str, bool]]:
        """Return session list as (id, name, is_active)."""
        return [
            (sid, state.name, sid == self._active_id)
            for sid, state in self._sessions.items()
        ]

    def has_session(self, session_id: str) -> bool:
        """Return whether a session exists in memory."""
        return session_id in self._sessions

    def get_active_session_id(self) -> Optional[str]:
        """Expose the active session id."""
        return self._active_id

    def get_session_name(self, session_id: str) -> Optional[str]:
        """Return a session name when it exists."""
        state = self._sessions.get(session_id)
        if state is None:
            return None
        return state.name

    def get_active_session_name(self) -> Optional[str]:
        """Expose the active session name."""
        if self._active_id is None:
            return None
        return self.get_session_name(self._active_id)

    def get_session_layer_stack(self, session_id: str) -> Optional[LayerStackModel]:
        """Expose one session layer stack after normalizing legacy state."""
        state = self._sessions.get(session_id)
        if state is None:
            return None
        return self._normalize_session_state(state).layer_stack

    def get_active_layer_stack(self) -> Optional[LayerStackModel]:
        """Expose the active session layer stack."""
        active_id = self._active_id
        if active_id is None:
            return None
        return self.get_session_layer_stack(active_id)

    def get_active_layer(self, session_id: Optional[str] = None) -> Optional[LayerState]:
        """Expose the active editable layer for one session."""
        target_id = str(session_id or self._active_id or "").strip()
        if not target_id:
            return None
        stack = self.get_session_layer_stack(target_id)
        if stack is None:
            return None
        return stack.get_active_layer()

    def sync_active_layer_from_model(
        self,
        *,
        annotation_model: AnnotationModel,
        view_state_model: Optional[ViewStateModel] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Rebind the active session layer to the current live annotation model payload."""
        target_id = str(session_id or self._active_id or "").strip()
        if not target_id or target_id not in self._sessions:
            return False
        state = self._normalize_session_state(self._sessions[target_id])
        active_layer = state.layer_stack.get_active_layer()
        if active_layer is None:
            return False
        active_layer.mask_volume = annotation_model.get_mask_volume()
        active_layer.label_palette = annotation_model.label_palette
        active_layer.label_visibility = annotation_model.label_visibility
        active_layer.overlay_cache = annotation_model.overlay_cache
        if view_state_model is not None:
            active_layer.layer_kind = self._layer_kind_from_view_state(view_state_model)
            active_layer.corrosion_state = self._build_corrosion_layer_state_from_view_state(
                view_state_model
            )
        self._sync_session_legacy_fields_from_active_layer(state, active_layer)
        return True

    def list_active_layers(self) -> List[tuple[str, str, bool, bool]]:
        """Expose active-session layers as `(id, name, visible, is_active)`."""
        active_id = self._active_id
        if active_id is None or active_id not in self._sessions:
            return []
        state = self._normalize_session_state(self._sessions[active_id])
        active_layer_id = state.layer_stack.ensure_active_layer()
        return [
            (
                str(layer.id),
                str(layer.name),
                bool(layer.visible),
                str(layer.id) == str(active_layer_id),
            )
            for layer in state.layer_stack.layers
        ]

    def switch_active_layer(
        self,
        layer_id: str,
        *,
        annotation_model: AnnotationModel,
        view_state_model: Optional[ViewStateModel] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Select a different active layer and rebind the live annotation model."""
        target_id = str(session_id or self._active_id or "").strip()
        if not target_id or target_id not in self._sessions:
            return False
        self.sync_active_layer_from_model(
            annotation_model=annotation_model,
            view_state_model=view_state_model,
            session_id=target_id,
        )
        state = self._normalize_session_state(self._sessions[target_id])
        if not state.layer_stack.set_active_layer(layer_id):
            return False
        active_layer = state.layer_stack.get_active_layer()
        self._sync_session_legacy_fields_from_active_layer(state, active_layer)
        if target_id == self._active_id:
            self._apply_annotation_layer_to_model(annotation_model, active_layer)
            if view_state_model is not None:
                self._apply_layer_corrosion_state_to_view_state(active_layer, view_state_model)
        return True

    def set_layer_visibility(
        self,
        layer_id: str,
        visible: bool,
        *,
        session_id: Optional[str] = None,
    ) -> bool:
        """Toggle one layer visibility inside a session stack."""
        target_id = str(session_id or self._active_id or "").strip()
        if not target_id or target_id not in self._sessions:
            return False
        state = self._normalize_session_state(self._sessions[target_id])
        layer = state.layer_stack.get_layer(layer_id)
        if layer is None:
            return False
        layer.visible = bool(visible)
        return True

    def create_empty_layer(
        self,
        *,
        annotation_model: AnnotationModel,
        view_state_model: Optional[ViewStateModel] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Append a new empty layer to one session and make it active."""
        target_id = str(session_id or self._active_id or "").strip()
        if not target_id or target_id not in self._sessions:
            return None
        self.sync_active_layer_from_model(
            annotation_model=annotation_model,
            view_state_model=view_state_model,
            session_id=target_id,
        )
        state = self._normalize_session_state(self._sessions[target_id])
        active_layer = state.layer_stack.get_active_layer()
        source_mask = (
            active_layer.mask_volume
            if active_layer is not None
            else annotation_model.get_mask_volume()
        )
        palette = copy.deepcopy(
            active_layer.label_palette
            if active_layer is not None
            else annotation_model.label_palette
        )
        visibility = copy.deepcopy(
            active_layer.label_visibility
            if active_layer is not None
            else annotation_model.label_visibility
        )
        new_layer = LayerState.create(
            name=self._next_layer_name(state.layer_stack),
            mask_volume=self._empty_mask_like(source_mask),
            label_palette=palette,
            label_visibility=visibility,
            overlay_cache=None,
            layer_kind="annotation",
        )
        state.layer_stack.layers.append(new_layer)
        state.layer_stack.set_active_layer(new_layer.id)
        self._sync_session_legacy_fields_from_active_layer(state, new_layer)
        if target_id == self._active_id:
            self._apply_annotation_layer_to_model(annotation_model, new_layer)
            if view_state_model is not None:
                self._apply_layer_corrosion_state_to_view_state(new_layer, view_state_model)
        return str(new_layer.id)

    def duplicate_active_layer(
        self,
        *,
        annotation_model: AnnotationModel,
        view_state_model: Optional[ViewStateModel] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Duplicate the active layer inside one session and make the copy active."""
        target_id = str(session_id or self._active_id or "").strip()
        if not target_id or target_id not in self._sessions:
            return None
        self.sync_active_layer_from_model(
            annotation_model=annotation_model,
            view_state_model=view_state_model,
            session_id=target_id,
        )
        state = self._normalize_session_state(self._sessions[target_id])
        active_layer = state.layer_stack.get_active_layer()
        if active_layer is None:
            return None
        duplicated_layer = LayerState.create(
            name=self._next_layer_name(state.layer_stack),
            mask_volume=self._copy_mask_volume(active_layer.mask_volume),
            label_palette=copy.deepcopy(active_layer.label_palette),
            label_visibility=copy.deepcopy(active_layer.label_visibility),
            visible=active_layer.visible,
            locked=active_layer.locked,
            opacity=active_layer.opacity,
            overlay_cache=None,
            layer_kind=active_layer.layer_kind,
            corrosion_state=self._copy_corrosion_layer_state(active_layer.corrosion_state),
        )
        state.layer_stack.layers.append(duplicated_layer)
        state.layer_stack.set_active_layer(duplicated_layer.id)
        self._sync_session_legacy_fields_from_active_layer(state, duplicated_layer)
        if target_id == self._active_id:
            self._apply_annotation_layer_to_model(annotation_model, duplicated_layer)
            if view_state_model is not None:
                self._apply_layer_corrosion_state_to_view_state(
                    duplicated_layer,
                    view_state_model,
                )
        return str(duplicated_layer.id)

    def delete_layer(
        self,
        layer_id: str,
        *,
        annotation_model: AnnotationModel,
        view_state_model: Optional[ViewStateModel] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Delete one layer while keeping at least one editable layer in the session."""
        target_id = str(session_id or self._active_id or "").strip()
        if not target_id or target_id not in self._sessions:
            return False
        self.sync_active_layer_from_model(
            annotation_model=annotation_model,
            view_state_model=view_state_model,
            session_id=target_id,
        )
        state = self._normalize_session_state(self._sessions[target_id])
        if len(state.layer_stack.layers) <= 1:
            return False
        normalized_layer_id = str(layer_id or "").strip()
        if not normalized_layer_id:
            return False
        delete_index = next(
            (
                index
                for index, layer in enumerate(state.layer_stack.layers)
                if str(layer.id) == normalized_layer_id
            ),
            -1,
        )
        if delete_index < 0:
            return False

        was_active = str(state.layer_stack.active_layer_id) == normalized_layer_id
        state.layer_stack.layers.pop(delete_index)
        if was_active:
            replacement_index = min(delete_index, len(state.layer_stack.layers) - 1)
            replacement_layer = state.layer_stack.layers[replacement_index]
            state.layer_stack.active_layer_id = str(replacement_layer.id)
        else:
            state.layer_stack.ensure_active_layer()

        active_layer = state.layer_stack.get_active_layer()
        self._sync_session_legacy_fields_from_active_layer(state, active_layer)
        if target_id == self._active_id:
            self._apply_annotation_layer_to_model(annotation_model, active_layer)
            if view_state_model is not None:
                self._apply_layer_corrosion_state_to_view_state(active_layer, view_state_model)
        return True

    def create_layer_from_model_state(
        self,
        *,
        name: str,
        annotation_model: AnnotationModel,
        view_state_model: ViewStateModel,
        session_id: Optional[str] = None,
        set_active: bool = True,
        save_current: bool = True,
        layer_kind: Optional[str] = None,
    ) -> Optional[str]:
        """Create a new layer from the current live model payload."""
        target_id = str(session_id or self._active_id or "").strip()
        if not target_id or target_id not in self._sessions:
            return None
        if save_current:
            self.sync_active_layer_from_model(
                annotation_model=annotation_model,
                view_state_model=view_state_model,
                session_id=target_id,
            )

        state = self._normalize_session_state(self._sessions[target_id])
        normalized_kind = self._normalize_layer_kind(
            layer_kind or self._layer_kind_from_view_state(view_state_model)
        )
        normalized_name = self._unique_layer_name(state.layer_stack, name)
        new_layer = LayerState.create(
            name=normalized_name,
            mask_volume=self._copy_mask_volume(annotation_model.get_mask_volume()),
            label_palette=copy.deepcopy(annotation_model.label_palette),
            label_visibility=copy.deepcopy(annotation_model.label_visibility),
            overlay_cache=None,
            layer_kind=normalized_kind,
            corrosion_state=(
                self._build_corrosion_layer_state_from_view_state(view_state_model)
                if normalized_kind == "corrosion"
                else None
            ),
        )
        state.layer_stack.layers.append(new_layer)
        if set_active:
            state.layer_stack.set_active_layer(new_layer.id)
            self._sync_session_legacy_fields_from_active_layer(state, new_layer)
            if target_id == self._active_id:
                self._apply_annotation_layer_to_model(annotation_model, new_layer)
                self._apply_layer_corrosion_state_to_view_state(new_layer, view_state_model)
        return str(new_layer.id)

    def compose_active_layers(
        self,
    ) -> tuple[
        Optional[np.ndarray],
        Dict[int, tuple[int, int, int, int]],
        Optional[set[int]],
        int,
    ]:
        """Compose visible layers of the active session into one render payload."""
        active_id = self._active_id
        if active_id is None or active_id not in self._sessions:
            return None, {}, None, 0
        state = self._normalize_session_state(self._sessions[active_id])
        return self._compose_layer_stack(state.layer_stack)

    def build_active_overlay_stack(self) -> Optional[OverlayStackData]:
        """Build the visible active-session layers as a render stack payload."""
        active_id = self._active_id
        if active_id is None or active_id not in self._sessions:
            return None
        state = self._normalize_session_state(self._sessions[active_id])
        stack_layers: list[OverlayLayerData] = []
        for layer in state.layer_stack.list_visible_layers():
            overlay = self._build_overlay_data_for_layer(layer)
            if overlay is None:
                continue
            visible_labels = self._visible_label_ids_for_layer(layer)
            stack_layers.append(
                OverlayLayerData(
                    layer_id=str(layer.id),
                    name=str(layer.name),
                    overlay=overlay,
                    visible_labels=(
                        frozenset(int(label_id) for label_id in visible_labels)
                        if visible_labels is not None
                        else None
                    ),
                    opacity=max(0.0, min(1.0, float(layer.opacity))),
                )
            )
        if not stack_layers:
            return None
        return OverlayStackData(
            layers=tuple(stack_layers),
            active_layer_id=state.layer_stack.active_layer_id,
        )

    def rename_session(self, session_id: str, name: str) -> None:
        """Update a session name without touching its contents."""
        state = self._sessions.get(session_id)
        if state is None:
            return
        state.name = self._normalize_session_name(name)

    def create_from_models(
        self,
        *,
        name: str,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
        set_active: bool = True,
        save_active: bool = True,
    ) -> str:
        """Capture current models into a new session."""
        source_state: Optional[AnnotationSessionState] = None
        if save_active and self._active_id is not None:
            source_state = self._sessions[self._active_id]
            self._sessions[self._active_id] = self._snapshot(
                name=self._sessions[self._active_id].name,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
                existing_state=source_state,
            )
            source_state = self._sessions[self._active_id]
        elif self._active_id is not None:
            source_state = self._sessions.get(self._active_id)

        session_id = uuid.uuid4().hex
        state = self._snapshot(
            name=name,
            annotation_model=annotation_model,
            temp_mask_model=temp_mask_model,
            roi_model=roi_model,
            view_state_model=view_state_model,
            existing_state=source_state,
        )
        self._sessions[session_id] = state
        if set_active:
            self._active_id = session_id
        return session_id

    def create_empty_session(
        self,
        *,
        name: str,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
        set_active: bool = True,
        save_active: bool = True,
    ) -> str:
        """Create an empty session on the current dataset."""
        if save_active and self._active_id is not None:
            self._sessions[self._active_id] = self._snapshot(
                name=self._sessions[self._active_id].name,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
                existing_state=self._sessions[self._active_id],
            )

        session_id = uuid.uuid4().hex
        state = self._empty_state(
            name=name,
            annotation_model=annotation_model,
            temp_mask_model=temp_mask_model,
            view_state_model=view_state_model,
        )
        self._sessions[session_id] = state
        if set_active:
            self._apply(
                state,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
            )
            self._active_id = session_id
        return session_id

    def delete_session(
        self,
        session_id: str,
        *,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
    ) -> Optional[str]:
        """
        Delete a session. If the active one is removed, switch to another one.
        Returns the new active session id.
        """
        if session_id not in self._sessions or len(self._sessions) <= 1:
            return self._active_id
        del self._sessions[session_id]
        if self._active_id == session_id:
            new_id = next(iter(self._sessions.keys()))
            self.switch_session(
                new_id,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
                save_current=False,
            )
        return self._active_id

    def switch_session(
        self,
        session_id: str,
        *,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
        save_current: bool = True,
    ) -> None:
        """Save the active session and load the target into live models."""
        if session_id not in self._sessions:
            return
        if save_current and self._active_id is not None:
            self._sessions[self._active_id] = self._snapshot(
                name=self._sessions[self._active_id].name,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
                existing_state=self._sessions[self._active_id],
            )
        target = self._sessions[session_id]
        self._apply(
            target,
            annotation_model=annotation_model,
            temp_mask_model=temp_mask_model,
            roi_model=roi_model,
            view_state_model=view_state_model,
        )
        self._active_id = session_id

    def build_dump(
        self,
        *,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
    ) -> dict[str, Any]:
        """Capture current state and return a serializable manager dump."""
        if self._active_id is not None and self._active_id in self._sessions:
            self._sessions[self._active_id] = self._snapshot(
                name=self._sessions[self._active_id].name,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
                existing_state=self._sessions[self._active_id],
            )
        return {
            "sessions": self._sessions,
            "active_id": self._active_id,
        }

    def build_active_dump(
        self,
        *,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
    ) -> dict[str, Any]:
        """Capture and return only the active session."""
        active_id = self._active_id
        if not isinstance(active_id, str):
            raise ValueError("Aucune session active a sauvegarder.")
        return self.build_session_dump(
            active_id,
            annotation_model=annotation_model,
            temp_mask_model=temp_mask_model,
            roi_model=roi_model,
            view_state_model=view_state_model,
        )

    @classmethod
    def clone_view_state_model(cls, source: ViewStateModel) -> ViewStateModel:
        """Clone a ViewStateModel without duplicating large NumPy payloads."""
        cloned = ViewStateModel()
        for key, value in vars(source).items():
            setattr(cloned, key, cls._copy_view_state_value(value))
        return cloned

    def build_session_dump(
        self,
        session_id: str,
        *,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
    ) -> dict[str, Any]:
        """Capture and return one session by id."""
        target_id = str(session_id).strip()
        if self._active_id is not None and self._active_id in self._sessions:
            self._sessions[self._active_id] = self._snapshot(
                name=self._sessions[self._active_id].name,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
                existing_state=self._sessions[self._active_id],
            )
        if not target_id or target_id not in self._sessions:
            raise ValueError("Session introuvable pour la sauvegarde.")
        return {
            "sessions": {
                target_id: self._build_persistable_session_state(self._sessions[target_id]),
            },
            "active_id": target_id,
        }

    def restore_dump(self, payload: dict[str, Any]) -> Optional[str]:
        """Replace in-memory sessions with a previously saved dump."""
        sessions = payload.get("sessions")
        if not isinstance(sessions, dict) or not sessions:
            raise ValueError("Dump de sessions invalide ou vide.")

        restored: Dict[str, AnnotationSessionState] = {}
        for session_id, state in sessions.items():
            if not isinstance(session_id, str) or not isinstance(state, AnnotationSessionState):
                raise ValueError("Dump de sessions corrompu.")
            restored[session_id] = self._normalize_session_state(state)

        active_id = payload.get("active_id")
        if not isinstance(active_id, str) or active_id not in restored:
            active_id = next(iter(restored.keys()))

        self._sessions = restored
        self._active_id = active_id
        return self._active_id

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _snapshot(
        self,
        *,
        name: str,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
        existing_state: Optional[AnnotationSessionState] = None,
    ) -> AnnotationSessionState:
        if existing_state is None and self._active_id is not None:
            existing_state = self._sessions.get(self._active_id)

        mask_volume = annotation_model.mask_volume
        label_palette = copy.deepcopy(annotation_model.label_palette)
        label_visibility = copy.deepcopy(annotation_model.label_visibility)
        overlay_cache = annotation_model.get_overlay_cache()
        layer_stack = self._snapshot_layer_stack(
            mask_volume=mask_volume,
            label_palette=label_palette,
            label_visibility=label_visibility,
            overlay_cache=overlay_cache,
            view_state_model=view_state_model,
            existing_state=existing_state,
        )
        active_layer = layer_stack.get_active_layer()

        temp_mask_volume = temp_mask_model.mask_volume
        temp_coverage_volume = temp_mask_model.coverage_volume
        temp_palette = copy.deepcopy(temp_mask_model.label_palette)
        temp_visibility = copy.deepcopy(temp_mask_model.label_visibility)

        rois = copy.deepcopy(roi_model._rois)  # noqa: SLF001
        next_roi_id = roi_model._next_id  # noqa: SLF001
        view_state = self._copy_view_state_mapping(vars(view_state_model))

        return AnnotationSessionState(
            name=self._normalize_session_name(name),
            layer_stack=layer_stack,
            mask_volume=active_layer.mask_volume if active_layer is not None else mask_volume,
            label_palette=copy.deepcopy(active_layer.label_palette)
            if active_layer is not None
            else label_palette,
            label_visibility=copy.deepcopy(active_layer.label_visibility)
            if active_layer is not None
            else label_visibility,
            temp_mask_volume=temp_mask_volume,
            temp_coverage_volume=temp_coverage_volume,
            temp_palette=temp_palette,
            temp_visibility=temp_visibility,
            rois=rois,
            next_roi_id=next_roi_id,
            view_state=view_state,
            overlay_cache=active_layer.overlay_cache if active_layer is not None else overlay_cache,
        )

    def _empty_state(
        self,
        *,
        name: str,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        view_state_model: ViewStateModel,
    ) -> AnnotationSessionState:
        shape = self._resolve_volume_shape(
            annotation_model=annotation_model,
            temp_mask_model=temp_mask_model,
        )
        mask_volume = np.zeros(shape, dtype=np.uint8) if shape is not None else None
        temp_mask_volume = np.zeros(shape, dtype=np.uint8) if shape is not None else None
        temp_coverage_volume = np.zeros(shape, dtype=bool) if shape is not None else None
        view_state = self._copy_view_state_mapping(vars(view_state_model))
        view_state["corrosion_active"] = False
        view_state["corrosion_projection"] = None
        view_state["corrosion_interpolated_projection"] = None
        view_state["corrosion_overlay_volume"] = None
        view_state["corrosion_overlay_palette"] = None
        view_state["corrosion_overlay_label_ids"] = None
        view_state["corrosion_peak_index_map_a"] = None
        view_state["corrosion_peak_index_map_b"] = None
        view_state["corrosion_raw_peak_index_map_a"] = None
        view_state["corrosion_raw_peak_index_map_b"] = None
        view_state["corrosion_raw_distance_map"] = None
        view_state["corrosion_ascan_support_map"] = None
        view_state["corrosion_session_stage"] = "base"
        view_state["corrosion_piece_volume_raw"] = None
        view_state["corrosion_piece_volume_interpolated"] = None
        view_state["corrosion_piece_volume_legacy_raw"] = None
        view_state["corrosion_piece_volume_legacy_interpolated"] = None
        view_state["corrosion_piece_anchor"] = None
        view_state["corrosion_piece_show_interpolated"] = True
        view_state["corrosion_piece_view_enabled"] = False

        layer_stack = self._build_single_layer_stack(
            mask_volume=mask_volume,
            label_palette=copy.deepcopy(annotation_model.label_palette),
            label_visibility=copy.deepcopy(annotation_model.label_visibility),
            overlay_cache=None,
        )
        active_layer = layer_stack.get_active_layer()

        return AnnotationSessionState(
            name=self._normalize_session_name(name),
            layer_stack=layer_stack,
            mask_volume=active_layer.mask_volume if active_layer is not None else mask_volume,
            label_palette=copy.deepcopy(active_layer.label_palette)
            if active_layer is not None
            else copy.deepcopy(annotation_model.label_palette),
            label_visibility=copy.deepcopy(active_layer.label_visibility)
            if active_layer is not None
            else copy.deepcopy(annotation_model.label_visibility),
            temp_mask_volume=temp_mask_volume,
            temp_coverage_volume=temp_coverage_volume,
            temp_palette=copy.deepcopy(temp_mask_model.label_palette),
            temp_visibility=copy.deepcopy(temp_mask_model.label_visibility),
            rois=[],
            next_roi_id=1,
            view_state=view_state,
            overlay_cache=None,
        )

    def _apply(
        self,
        state: AnnotationSessionState,
        *,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
    ) -> None:
        state = self._normalize_session_state(state)
        active_layer = state.layer_stack.get_active_layer()

        self._sync_session_legacy_fields_from_active_layer(state, active_layer)
        self._apply_annotation_layer_to_model(annotation_model, active_layer)

        temp_mask_model.clear()
        if state.temp_mask_volume is not None:
            temp_mask_model.initialize(state.temp_mask_volume.shape)
            temp_mask_model.mask_volume = state.temp_mask_volume  # type: ignore[assignment]
        if state.temp_coverage_volume is not None:
            temp_mask_model.coverage_volume = state.temp_coverage_volume  # type: ignore[assignment]
        temp_mask_model.label_palette = copy.deepcopy(state.temp_palette)
        temp_mask_model.label_visibility = copy.deepcopy(state.temp_visibility)
        temp_mask_model.ensure_persistent_labels()

        roi_model._rois = copy.deepcopy(state.rois)  # noqa: SLF001
        roi_model._next_id = state.next_roi_id  # noqa: SLF001

        default_view_state = vars(ViewStateModel())
        for key, val in default_view_state.items():
            if key not in state.view_state:
                setattr(view_state_model, key, self._copy_view_state_value(val))
        for key, val in state.view_state.items():
            setattr(view_state_model, key, self._copy_view_state_value(val))
        self._apply_layer_corrosion_state_to_view_state(active_layer, view_state_model)

    def _build_persistable_session_state(
        self,
        state: AnnotationSessionState,
    ) -> AnnotationSessionState:
        """Return a disk-oriented clone without runtime overlay caches."""
        normalized_state = self._normalize_session_state(state)
        persistable_stack = self._build_persistable_layer_stack(normalized_state.layer_stack)
        active_layer = persistable_stack.get_active_layer()
        return AnnotationSessionState(
            name=self._normalize_session_name(normalized_state.name),
            layer_stack=persistable_stack,
            mask_volume=active_layer.mask_volume if active_layer is not None else normalized_state.mask_volume,
            label_palette=copy.deepcopy(active_layer.label_palette)
            if active_layer is not None
            else copy.deepcopy(normalized_state.label_palette),
            label_visibility=copy.deepcopy(active_layer.label_visibility)
            if active_layer is not None
            else copy.deepcopy(normalized_state.label_visibility),
            temp_mask_volume=normalized_state.temp_mask_volume,
            temp_coverage_volume=normalized_state.temp_coverage_volume,
            temp_palette=copy.deepcopy(normalized_state.temp_palette),
            temp_visibility=copy.deepcopy(normalized_state.temp_visibility),
            rois=copy.deepcopy(normalized_state.rois),
            next_roi_id=normalized_state.next_roi_id,
            view_state=self._copy_view_state_mapping(normalized_state.view_state),
            overlay_cache=None,
        )

    def _normalize_session_state(
        self,
        state: AnnotationSessionState,
    ) -> AnnotationSessionState:
        """Promote legacy single-overlay state to a stack-backed session."""
        state.name = self._normalize_session_name(getattr(state, "name", ""))

        layer_stack = self._normalize_layer_stack(getattr(state, "layer_stack", None))
        if layer_stack is None or not layer_stack.layers:
            layer_stack = self._build_single_layer_stack(
                mask_volume=getattr(state, "mask_volume", None),
                label_palette=copy.deepcopy(getattr(state, "label_palette", {})),
                label_visibility=copy.deepcopy(getattr(state, "label_visibility", {})),
                overlay_cache=getattr(state, "overlay_cache", None),
            )
        else:
            layer_stack.ensure_active_layer()

        active_layer = layer_stack.get_active_layer()
        if active_layer is None:
            layer_stack = self._build_single_layer_stack(
                mask_volume=getattr(state, "mask_volume", None),
                label_palette=copy.deepcopy(getattr(state, "label_palette", {})),
                label_visibility=copy.deepcopy(getattr(state, "label_visibility", {})),
                overlay_cache=getattr(state, "overlay_cache", None),
            )
            active_layer = layer_stack.get_active_layer()

        state.layer_stack = layer_stack
        state.mask_volume = active_layer.mask_volume if active_layer is not None else None
        state.label_palette = active_layer.label_palette if active_layer is not None else {}
        state.label_visibility = active_layer.label_visibility if active_layer is not None else {}
        state.overlay_cache = active_layer.overlay_cache if active_layer is not None else None
        return state

    def _snapshot_layer_stack(
        self,
        *,
        mask_volume: Optional[np.ndarray],
        label_palette: Dict[int, tuple[int, int, int, int]],
        label_visibility: Dict[int, bool],
        overlay_cache: Optional[OverlayData],
        view_state_model: ViewStateModel,
        existing_state: Optional[AnnotationSessionState],
    ) -> LayerStackModel:
        normalized_existing = (
            self._normalize_session_state(existing_state)
            if existing_state is not None
            else None
        )
        source_stack = normalized_existing.layer_stack if normalized_existing is not None else None
        if source_stack is None or not source_stack.layers:
            return self._build_single_layer_stack(
                mask_volume=mask_volume,
                label_palette=label_palette,
                label_visibility=label_visibility,
                overlay_cache=overlay_cache,
            )

        active_layer_id = source_stack.ensure_active_layer()
        cloned_layers: list[LayerState] = []
        for source_layer in source_stack.layers:
            if str(source_layer.id) == str(active_layer_id):
                cloned_layers.append(
                    LayerState.create(
                        layer_id=source_layer.id,
                        name=source_layer.name,
                        mask_volume=mask_volume,
                        label_palette=label_palette,
                        label_visibility=label_visibility,
                        visible=source_layer.visible,
                        locked=source_layer.locked,
                        opacity=source_layer.opacity,
                        overlay_cache=overlay_cache,
                        layer_kind=self._layer_kind_from_view_state(view_state_model),
                        corrosion_state=self._build_corrosion_layer_state_from_view_state(
                            view_state_model
                        ),
                    )
                )
            else:
                cloned_layers.append(
                    LayerState.create(
                        layer_id=source_layer.id,
                        name=source_layer.name,
                        mask_volume=source_layer.mask_volume,
                        label_palette=copy.deepcopy(source_layer.label_palette),
                        label_visibility=copy.deepcopy(source_layer.label_visibility),
                        visible=source_layer.visible,
                        locked=source_layer.locked,
                        opacity=source_layer.opacity,
                        overlay_cache=source_layer.overlay_cache,
                        layer_kind=source_layer.layer_kind,
                        corrosion_state=self._copy_corrosion_layer_state(
                            source_layer.corrosion_state
                        ),
                    )
                )
        return LayerStackModel(layers=cloned_layers, active_layer_id=active_layer_id)

    def _build_single_layer_stack(
        self,
        *,
        mask_volume: Optional[np.ndarray],
        label_palette: Dict[int, tuple[int, int, int, int]],
        label_visibility: Dict[int, bool],
        overlay_cache: Optional[OverlayData],
    ) -> LayerStackModel:
        """Wrap the current annotation payload in a one-layer stack."""
        layer = LayerState.create(
            name="Layer 1",
            mask_volume=mask_volume,
            label_palette=label_palette,
            label_visibility=label_visibility,
            overlay_cache=overlay_cache,
        )
        return LayerStackModel(layers=[layer], active_layer_id=layer.id)

    def _build_persistable_layer_stack(
        self,
        layer_stack: LayerStackModel,
    ) -> LayerStackModel:
        """Clone one layer stack while dropping runtime caches."""
        layer_stack.ensure_active_layer()
        layers = [
            LayerState.create(
                layer_id=layer.id,
                name=layer.name,
                mask_volume=layer.mask_volume,
                label_palette=copy.deepcopy(layer.label_palette),
                label_visibility=copy.deepcopy(layer.label_visibility),
                visible=layer.visible,
                locked=layer.locked,
                opacity=layer.opacity,
                overlay_cache=None,
                layer_kind=layer.layer_kind,
                corrosion_state=self._copy_corrosion_layer_state(layer.corrosion_state),
            )
            for layer in layer_stack.layers
        ]
        persistable = LayerStackModel(
            layers=layers,
            active_layer_id=layer_stack.active_layer_id,
        )
        persistable.ensure_active_layer()
        return persistable

    @staticmethod
    def _sync_session_legacy_fields_from_active_layer(
        state: AnnotationSessionState,
        active_layer: Optional[LayerState],
    ) -> None:
        """Keep legacy single-layer fields aligned with the selected active layer."""
        if active_layer is None:
            state.mask_volume = None
            state.label_palette = {}
            state.label_visibility = {}
            state.overlay_cache = None
            return
        state.mask_volume = active_layer.mask_volume
        state.label_palette = active_layer.label_palette
        state.label_visibility = active_layer.label_visibility
        state.overlay_cache = active_layer.overlay_cache

    @staticmethod
    def _apply_annotation_layer_to_model(
        annotation_model: AnnotationModel,
        active_layer: Optional[LayerState],
    ) -> None:
        """Rebind the live annotation model to the current editable layer."""
        if active_layer is None:
            annotation_model.clear()
            return
        annotation_model.mask_volume = active_layer.mask_volume  # type: ignore[assignment]
        annotation_model.label_palette = active_layer.label_palette
        annotation_model.label_visibility = active_layer.label_visibility
        annotation_model.overlay_cache = active_layer.overlay_cache
        if active_layer.mask_volume is None:
            annotation_model.detected_label_ids = (0,)
        else:
            annotation_model.detected_label_ids = tuple(
                sorted(
                    {
                        0,
                        *(
                            int(label_id)
                            for label_id in active_layer.label_palette.keys()
                            if int(label_id) >= 0
                        ),
                    }
                )
            )
        annotation_model.ensure_persistent_labels()

    def _compose_layer_stack(
        self,
        layer_stack: LayerStackModel,
    ) -> tuple[
        Optional[np.ndarray],
        Dict[int, tuple[int, int, int, int]],
        Optional[set[int]],
        int,
    ]:
        """Flatten visible layers into one mask/palette payload for legacy views."""
        visible_layers = [
            layer
            for layer in layer_stack.list_visible_layers()
            if layer.mask_volume is not None
        ]
        if not visible_layers:
            return None, {}, None, 0

        base_shape = tuple(int(dim) for dim in np.asarray(visible_layers[0].mask_volume).shape)
        composite_mask = np.zeros(base_shape, dtype=np.uint8)
        composite_palette: Dict[int, tuple[int, int, int, int]] = {}

        for layer in visible_layers:
            layer_mask = np.asarray(layer.mask_volume, dtype=np.uint8)
            if tuple(int(dim) for dim in layer_mask.shape) != base_shape:
                continue
            visible_label_ids = self._visible_label_ids_for_layer(layer)
            if visible_label_ids is None:
                draw_mask = layer_mask > 0
                palette_source = layer.label_palette.items()
            else:
                visible_array = np.fromiter(sorted(visible_label_ids), dtype=np.uint8)
                if visible_array.size == 0:
                    continue
                draw_mask = np.isin(layer_mask, visible_array)
                palette_source = (
                    (label, color)
                    for label, color in layer.label_palette.items()
                    if int(label) in visible_label_ids
                )
            if np.any(draw_mask):
                composite_mask[draw_mask] = layer_mask[draw_mask]
            for label, color in palette_source:
                label_int = int(label)
                if label_int == 0:
                    continue
                composite_palette[label_int] = tuple(int(channel) for channel in color)

        return composite_mask, composite_palette, None, len(visible_layers)

    @staticmethod
    def _visible_label_ids_for_layer(layer: LayerState) -> Optional[set[int]]:
        """Resolve per-layer visible label ids for composition."""
        if not layer.label_visibility:
            return None
        if all(layer.label_visibility.values()):
            return None
        return {
            int(label_id)
            for label_id, is_visible in layer.label_visibility.items()
            if bool(is_visible)
        }

    @staticmethod
    def _build_overlay_data_for_layer(layer: LayerState) -> Optional[OverlayData]:
        """Build one overlay payload from a layer state, reusing cached volumes when available."""
        if layer.mask_volume is None:
            return None
        cached = layer.overlay_cache
        normalized_mask = np.asarray(layer.mask_volume, dtype=np.uint8)
        normalized_palette = {
            int(label_id): tuple(int(channel) for channel in color)
            for label_id, color in layer.label_palette.items()
        }
        if (
            cached is not None
            and cached.mask_volume is normalized_mask
            and cached.palette == normalized_palette
        ):
            return cached
        label_volumes = cached.label_volumes if cached is not None else {}
        overlay_data = OverlayData(
            mask_volume=normalized_mask,
            palette=normalized_palette,
            label_volumes=label_volumes,
        )
        layer.overlay_cache = overlay_data
        return overlay_data

    @classmethod
    def _normalize_layer_kind(cls, value: Optional[str]) -> str:
        normalized = str(value or "annotation").strip().casefold()
        return "corrosion" if normalized == "corrosion" else "annotation"

    @classmethod
    def _layer_kind_from_view_state(cls, view_state_model: ViewStateModel) -> str:
        if bool(getattr(view_state_model, "corrosion_active", False)):
            return "corrosion"
        return "annotation"

    @classmethod
    def _build_corrosion_layer_state_from_view_state(
        cls,
        view_state_model: ViewStateModel,
    ) -> Optional[CorrosionLayerState]:
        if not bool(getattr(view_state_model, "corrosion_active", False)):
            return None
        payload = {
            key: cls._copy_view_state_value(getattr(view_state_model, key, None))
            for key in CORROSION_LAYER_VIEW_STATE_KEYS
        }
        stage = view_state_model.get_corrosion_session_stage()
        payload["corrosion_session_stage"] = stage
        return CorrosionLayerState(stage=stage, payload=payload)

    @classmethod
    def _normalize_corrosion_layer_state(cls, payload: Any) -> Optional[CorrosionLayerState]:
        if payload is None:
            return None
        if isinstance(payload, CorrosionLayerState):
            raw_stage = payload.stage
            raw_mapping = payload.payload
        elif isinstance(payload, dict):
            raw_stage = payload.get("stage") or payload.get("corrosion_session_stage")
            raw_mapping = payload.get("payload", payload)
        else:
            return None
        if not isinstance(raw_mapping, dict):
            raw_mapping = {}
        normalized_payload = {
            str(key): cls._copy_view_state_value(value)
            for key, value in raw_mapping.items()
        }
        stage = ViewStateModel.normalize_corrosion_session_stage(raw_stage)
        normalized_payload["corrosion_session_stage"] = stage
        return CorrosionLayerState(stage=stage, payload=normalized_payload)

    @classmethod
    def _copy_corrosion_layer_state(
        cls,
        corrosion_state: Optional[CorrosionLayerState],
    ) -> Optional[CorrosionLayerState]:
        return cls._normalize_corrosion_layer_state(corrosion_state)

    @classmethod
    def _apply_layer_corrosion_state_to_view_state(
        cls,
        layer: Optional[LayerState],
        view_state_model: ViewStateModel,
    ) -> None:
        view_state_model.deactivate_corrosion()
        if layer is None:
            return
        if cls._normalize_layer_kind(getattr(layer, "layer_kind", None)) != "corrosion":
            return
        corrosion_state = cls._normalize_corrosion_layer_state(
            getattr(layer, "corrosion_state", None)
        )
        if corrosion_state is None:
            return
        for key, value in corrosion_state.payload.items():
            if key == "corrosion_active":
                continue
            setattr(view_state_model, key, cls._copy_view_state_value(value))
        view_state_model.corrosion_active = True
        view_state_model.set_corrosion_session_stage(corrosion_state.stage)

    @classmethod
    def _normalize_layer_state(cls, layer: Any) -> Optional[LayerState]:
        if layer is None:
            return None
        return LayerState.create(
            layer_id=getattr(layer, "id", None),
            name=getattr(layer, "name", "Layer 1"),
            mask_volume=getattr(layer, "mask_volume", None),
            label_palette=copy.deepcopy(getattr(layer, "label_palette", {})),
            label_visibility=copy.deepcopy(getattr(layer, "label_visibility", {})),
            visible=getattr(layer, "visible", True),
            locked=getattr(layer, "locked", False),
            opacity=getattr(layer, "opacity", 1.0),
            overlay_cache=getattr(layer, "overlay_cache", None),
            layer_kind=cls._normalize_layer_kind(getattr(layer, "layer_kind", "annotation")),
            corrosion_state=cls._normalize_corrosion_layer_state(
                getattr(layer, "corrosion_state", None)
            ),
        )

    @classmethod
    def _normalize_layer_stack(cls, layer_stack: Any) -> Optional[LayerStackModel]:
        if not isinstance(layer_stack, LayerStackModel):
            return None
        raw_layers = list(getattr(layer_stack, "layers", []))
        if raw_layers and all(isinstance(layer, LayerState) for layer in raw_layers):
            layer_stack.ensure_active_layer()
            return layer_stack
        normalized_layers = [
            normalized_layer
            for normalized_layer in (
                cls._normalize_layer_state(layer) for layer in getattr(layer_stack, "layers", [])
            )
            if normalized_layer is not None
        ]
        if not normalized_layers:
            return None
        normalized_stack = LayerStackModel(
            layers=normalized_layers,
            active_layer_id=str(getattr(layer_stack, "active_layer_id", "") or "") or None,
        )
        normalized_stack.ensure_active_layer()
        return normalized_stack

    @classmethod
    def _copy_view_state_mapping(cls, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            str(key): cls._copy_view_state_value(value)
            for key, value in payload.items()
        }

    @classmethod
    def _copy_view_state_value(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, tuple):
            return tuple(cls._copy_view_state_value(item) for item in value)
        if isinstance(value, list):
            return [cls._copy_view_state_value(item) for item in value]
        if isinstance(value, dict):
            return {
                copy.deepcopy(key): cls._copy_view_state_value(item)
                for key, item in value.items()
            }
        if isinstance(value, set):
            return {cls._copy_view_state_value(item) for item in value}
        return copy.deepcopy(value)

    @staticmethod
    def _normalize_session_name(name: str, *, fallback: str = "New session") -> str:
        normalized = str(name or "").strip()
        return normalized or fallback

    @staticmethod
    def _copy_mask_volume(mask_volume: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask_volume is None:
            return None
        return np.asarray(mask_volume, dtype=np.uint8).copy()

    @classmethod
    def _empty_mask_like(cls, mask_volume: Optional[np.ndarray]) -> Optional[np.ndarray]:
        copied = cls._copy_mask_volume(mask_volume)
        if copied is None:
            return None
        copied.fill(0)
        return copied

    @staticmethod
    def _next_layer_name(layer_stack: LayerStackModel) -> str:
        existing_names = {str(layer.name or "").strip() for layer in layer_stack.layers}
        index = 1
        while True:
            candidate = f"Layer {index}"
            if candidate not in existing_names:
                return candidate
            index += 1

    @staticmethod
    def _unique_layer_name(layer_stack: LayerStackModel, requested_name: str) -> str:
        normalized = str(requested_name or "").strip()
        if not normalized:
            return AnnotationSessionManager._next_layer_name(layer_stack)
        existing_names = {str(layer.name or "").strip() for layer in layer_stack.layers}
        if normalized not in existing_names:
            return normalized
        index = 2
        while True:
            candidate = f"{normalized} ({index})"
            if candidate not in existing_names:
                return candidate
            index += 1

    @staticmethod
    def _resolve_volume_shape(
        *,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
    ) -> Optional[tuple[int, int, int]]:
        for volume in (
            annotation_model.mask_volume,
            temp_mask_model.mask_volume,
            temp_mask_model.coverage_volume,
        ):
            if volume is None:
                continue
            if getattr(volume, "ndim", 0) != 3:
                continue
            return tuple(int(dim) for dim in volume.shape)
        return None
