from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from models.annotation_model import AnnotationModel
from models.overlay_data import OverlayData
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
    overlay_cache: Optional[OverlayData]


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
        """Réinitialise toutes les sessions lors du chargement d'un nouveau NDE."""
        self._sessions.clear()
        self._active_id = None
        return self.ensure_default(
            annotation_model=annotation_model,
            temp_mask_model=temp_mask_model,
            roi_model=roi_model,
            view_state_model=view_state_model,
        )

    def list_sessions(self) -> List[tuple[str, str, bool]]:
        """Retourne la liste des sessions (id, nom, is_active)."""
        return [
            (sid, state.name, sid == self._active_id)
            for sid, state in self._sessions.items()
        ]

    def has_session(self, session_id: str) -> bool:
        """Indique si une session existe en memoire."""
        return session_id in self._sessions

    def get_active_session_id(self) -> Optional[str]:
        """Expose l'id de la session active."""
        return self._active_id

    def get_session_name(self, session_id: str) -> Optional[str]:
        """Retourne le nom d'une session si elle existe."""
        state = self._sessions.get(session_id)
        if state is None:
            return None
        return state.name

    def get_active_session_name(self) -> Optional[str]:
        """Expose le nom de la session active."""
        if self._active_id is None:
            return None
        return self.get_session_name(self._active_id)

    def rename_session(self, session_id: str, name: str) -> None:
        """Met a jour le nom d'une session sans modifier son contenu."""
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
        """Cree une session vide sur le dataset courant sans dupliquer les annotations."""
        if save_active and self._active_id is not None:
            self._sessions[self._active_id] = self._snapshot(
                name=self._sessions[self._active_id].name,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
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

    def build_dump(
        self,
        *,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
    ) -> dict[str, Any]:
        """Capture l'etat courant et retourne un dump serialisable du gestionnaire."""
        if self._active_id is not None and self._active_id in self._sessions:
            self._sessions[self._active_id] = self._snapshot(
                name=self._sessions[self._active_id].name,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
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
        """Capture et retourne uniquement la session active."""
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
        """Capture et retourne une session unique par son identifiant."""
        target_id = str(session_id).strip()
        if self._active_id is not None and self._active_id in self._sessions:
            self._sessions[self._active_id] = self._snapshot(
                name=self._sessions[self._active_id].name,
                annotation_model=annotation_model,
                temp_mask_model=temp_mask_model,
                roi_model=roi_model,
                view_state_model=view_state_model,
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
        """Replace les sessions en memoire par un dump precedemment sauvegarde."""
        sessions = payload.get("sessions")
        if not isinstance(sessions, dict) or not sessions:
            raise ValueError("Dump de sessions invalide ou vide.")

        restored: Dict[str, AnnotationSessionState] = {}
        for session_id, state in sessions.items():
            if not isinstance(session_id, str) or not isinstance(state, AnnotationSessionState):
                raise ValueError("Dump de sessions corrompu.")
            restored[session_id] = state

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
    ) -> AnnotationSessionState:
        mask_volume = annotation_model.mask_volume
        label_palette = copy.deepcopy(annotation_model.label_palette)
        label_visibility = copy.deepcopy(annotation_model.label_visibility)
        overlay_cache = annotation_model.get_overlay_cache()

        temp_mask_volume = temp_mask_model.mask_volume
        temp_coverage_volume = temp_mask_model.coverage_volume
        temp_palette = copy.deepcopy(temp_mask_model.label_palette)
        temp_visibility = copy.deepcopy(temp_mask_model.label_visibility)

        rois = copy.deepcopy(roi_model._rois)  # noqa: SLF001
        next_roi_id = roi_model._next_id  # noqa: SLF001

        view_state = self._copy_view_state_mapping(vars(view_state_model))

        return AnnotationSessionState(
            name=self._normalize_session_name(name),
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
            overlay_cache=overlay_cache,
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

        return AnnotationSessionState(
            name=self._normalize_session_name(name),
            mask_volume=mask_volume,
            label_palette=copy.deepcopy(annotation_model.label_palette),
            label_visibility=copy.deepcopy(annotation_model.label_visibility),
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
        # Annotation model
        annotation_model.clear()
        if state.mask_volume is not None:
            annotation_model.mask_volume = state.mask_volume  # type: ignore[assignment]
        annotation_model.label_palette = copy.deepcopy(state.label_palette)
        annotation_model.label_visibility = copy.deepcopy(state.label_visibility)
        annotation_model.overlay_cache = state.overlay_cache
        annotation_model.ensure_persistent_labels()

        # Temp masks
        temp_mask_model.clear()
        if state.temp_mask_volume is not None:
            temp_mask_model.initialize(state.temp_mask_volume.shape)
            temp_mask_model.mask_volume = state.temp_mask_volume  # type: ignore[assignment]
        if state.temp_coverage_volume is not None:
            temp_mask_model.coverage_volume = state.temp_coverage_volume  # type: ignore[assignment]
        temp_mask_model.label_palette = copy.deepcopy(state.temp_palette)
        temp_mask_model.label_visibility = copy.deepcopy(state.temp_visibility)
        temp_mask_model.ensure_persistent_labels()

        # ROIs
        roi_model._rois = copy.deepcopy(state.rois)  # noqa: SLF001
        roi_model._next_id = state.next_roi_id  # noqa: SLF001

        # View state
        default_view_state = vars(ViewStateModel())
        for key, val in default_view_state.items():
            if key not in state.view_state:
                setattr(view_state_model, key, self._copy_view_state_value(val))
        for key, val in state.view_state.items():
            setattr(view_state_model, key, self._copy_view_state_value(val))

    def _build_persistable_session_state(
        self,
        state: AnnotationSessionState,
    ) -> AnnotationSessionState:
        """Return a disk-oriented clone without derived runtime caches."""
        return AnnotationSessionState(
            name=self._normalize_session_name(state.name),
            mask_volume=state.mask_volume,
            label_palette=copy.deepcopy(state.label_palette),
            label_visibility=copy.deepcopy(state.label_visibility),
            temp_mask_volume=state.temp_mask_volume,
            temp_coverage_volume=state.temp_coverage_volume,
            temp_palette=copy.deepcopy(state.temp_palette),
            temp_visibility=copy.deepcopy(state.temp_visibility),
            rois=copy.deepcopy(state.rois),
            next_roi_id=state.next_roi_id,
            view_state=self._copy_view_state_mapping(state.view_state),
            overlay_cache=None,
        )

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
