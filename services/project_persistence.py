from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ProjectSessionPayload:
    """Persisted project session payload stored inside a `.session` file."""

    version: int
    nde_path: str
    annotation_axis_mode: str
    signal_processing_selection: dict[str, bool]
    session_manager_dump: dict[str, Any]


class ProjectPersistence:
    """Read/write `.session` files for the full annotation session manager dump."""

    VERSION = 1
    COMPRESS_LEVEL = 1

    def save_session(
        self,
        destination: str,
        *,
        nde_path: str,
        annotation_axis_mode: str,
        signal_processing_selection: Mapping[str, Any] | None,
        session_manager_dump: dict[str, Any],
    ) -> str:
        """Persist the current project session to disk."""
        if not nde_path:
            raise ValueError("Chemin NDE manquant pour sauvegarder la session.")
        if not isinstance(session_manager_dump, dict) or not session_manager_dump.get("sessions"):
            raise ValueError("Dump de sessions invalide.")

        file_path = Path(destination)
        if file_path.suffix.lower() != ".session":
            file_path = file_path.with_suffix(".session")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = ProjectSessionPayload(
            version=self.VERSION,
            nde_path=str(nde_path),
            annotation_axis_mode=str(annotation_axis_mode or "Auto"),
            signal_processing_selection=self._normalize_processing_selection(
                signal_processing_selection
            ),
            session_manager_dump=session_manager_dump,
        )
        with gzip.open(file_path, "wb", compresslevel=self.COMPRESS_LEVEL) as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return str(file_path)

    def load_session(self, source: str) -> ProjectSessionPayload:
        """Load and validate a persisted `.session` file."""
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier session introuvable: {file_path}")

        with gzip.open(file_path, "rb") as stream:
            payload = pickle.load(stream)

        if isinstance(payload, ProjectSessionPayload):
            validated = payload
        elif isinstance(payload, Mapping):
            validated = ProjectSessionPayload(
                version=int(payload.get("version", 0)),
                nde_path=str(payload.get("nde_path") or ""),
                annotation_axis_mode=str(payload.get("annotation_axis_mode") or "Auto"),
                signal_processing_selection=self._normalize_processing_selection(
                    payload.get("signal_processing_selection")
                ),
                session_manager_dump=dict(payload.get("session_manager_dump") or {}),
            )
        else:
            raise ValueError("Format de session non supporte.")

        if validated.version != self.VERSION:
            raise ValueError(
                f"Version de session non supportee: {validated.version} (attendu {self.VERSION})."
            )
        if not validated.nde_path:
            raise ValueError("Le fichier session ne contient pas de chemin NDE.")
        if not isinstance(validated.session_manager_dump, dict) or not validated.session_manager_dump.get(
            "sessions"
        ):
            raise ValueError("Le fichier session ne contient aucun dump de sessions valide.")
        return validated

    @staticmethod
    def _normalize_processing_selection(payload: Mapping[str, Any] | None) -> dict[str, bool]:
        """Keep only the persisted processing flags used at NDE open time."""
        source = payload if isinstance(payload, Mapping) else {}
        return {
            "apply_hilbert": bool(source.get("apply_hilbert", False)),
            "apply_smoothing": bool(source.get("apply_smoothing", False)),
        }
