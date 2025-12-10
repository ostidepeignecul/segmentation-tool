from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from models.annotation_model import AnnotationModel
from models.roi_model import ROI, RoiModel
from models.temp_mask_model import TempMaskModel
from models.view_state_model import ViewStateModel


@dataclass
class AnnotationSessionState:
    """Snapshot complet d'une session d'annotation."""

    name: str
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


class AnnotationSessionManager:
    """Gère plusieurs sessions d'annotation en mémoire et permet de basculer entre elles."""

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
        """Crée une session par défaut si aucune n'existe."""
        if self._active_id is not None:
            return self._active_id
        return self.create_from_models(
            name="Session 1",
            annotation_model=annotation_model,
            temp_mask_model=temp_mask_model,
            roi_model=roi_model,
            view_state_model=view_state_model,
            set_active=True,
        )

    def list_sessions(self) -> List[tuple[str, str, bool]]:
        """Retourne la liste des sessions (id, nom, is_active)."""
        return [
            (sid, state.name, sid == self._active_id)
            for sid, state in self._sessions.items()
        ]

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
        """Capture les modèles courants dans une nouvelle session."""
        if save_active and self._active_id is not None:
            # Sauvegarde l'état actuel avant de créer une nouvelle session
            self._sessions[self._active_id] = self._snapshot(
                name=self._sessions[self._active_id].name,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
            )
        session_id = uuid.uuid4().hex
        state = self._snapshot(
            name=name,
            annotation_model=annotation_model,
            temp_mask_model=temp_mask_model,
            roi_model=roi_model,
            view_state_model=view_state_model,
        )
        self._sessions[session_id] = state
        if set_active:
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
        Supprime une session. Si on supprime l'actuelle, bascule sur une autre.
        Retourne la nouvelle session active.
        """
        if session_id not in self._sessions or len(self._sessions) <= 1:
            return self._active_id
        del self._sessions[session_id]
        if self._active_id == session_id:
            # Choisir arbitrairement la première restante
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
        """Sauvegarde la session active, charge la session cible dans les modèles."""
        if session_id not in self._sessions:
            return
        if save_current and self._active_id is not None:
            self._sessions[self._active_id] = self._snapshot(
                name=self._sessions[self._active_id].name,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
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
    ) -> AnnotationSessionState:
        mask_volume = (
            None if annotation_model.mask_volume is None else np.array(annotation_model.mask_volume, copy=True)
        )
        label_palette = copy.deepcopy(annotation_model.label_palette)
        label_visibility = copy.deepcopy(annotation_model.label_visibility)

        temp_mask_volume = (
            None if temp_mask_model.mask_volume is None else np.array(temp_mask_model.mask_volume, copy=True)
        )
        temp_coverage_volume = (
            None if temp_mask_model.coverage_volume is None else np.array(temp_mask_model.coverage_volume, copy=True)
        )
        temp_palette = copy.deepcopy(temp_mask_model.label_palette)
        temp_visibility = copy.deepcopy(temp_mask_model.label_visibility)

        rois = copy.deepcopy(roi_model._rois)  # noqa: SLF001
        next_roi_id = roi_model._next_id  # noqa: SLF001

        view_state = copy.deepcopy(vars(view_state_model))

        return AnnotationSessionState(
            name=name,
            mask_volume=mask_volume,
            label_palette=label_palette,
            label_visibility=label_visibility,
            temp_mask_volume=temp_mask_volume,
            temp_coverage_volume=temp_coverage_volume,
            temp_palette=temp_palette,
            temp_visibility=temp_visibility,
            rois=rois,
            next_roi_id=next_roi_id,
            view_state=view_state,
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
        # Annotation model
        annotation_model.clear()
        if state.mask_volume is not None:
            annotation_model.set_mask_volume(np.array(state.mask_volume, copy=True))
        annotation_model.label_palette = copy.deepcopy(state.label_palette)
        annotation_model.label_visibility = copy.deepcopy(state.label_visibility)
        annotation_model.overlay_cache = None

        # Temp masks
        temp_mask_model.clear()
        if state.temp_mask_volume is not None:
            temp_mask_model.initialize(state.temp_mask_volume.shape)
            temp_mask_model.mask_volume = np.array(state.temp_mask_volume, copy=True)
        if state.temp_coverage_volume is not None:
            temp_mask_model.coverage_volume = np.array(state.temp_coverage_volume, copy=True)
        temp_mask_model.label_palette = copy.deepcopy(state.temp_palette)
        temp_mask_model.label_visibility = copy.deepcopy(state.temp_visibility)

        # ROIs
        roi_model._rois = copy.deepcopy(state.rois)  # noqa: SLF001
        roi_model._next_id = state.next_roi_id  # noqa: SLF001

        # View state
        for key, val in state.view_state.items():
            setattr(view_state_model, key, copy.deepcopy(val))
