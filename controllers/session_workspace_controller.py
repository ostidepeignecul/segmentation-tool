from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox


class SessionSaveSignals(QObject):
    finished = pyqtSignal(str, str)  # session_id, path
    error = pyqtSignal(str, object)  # session_id, exception


class SessionSaveWorker(QRunnable):
    def __init__(
        self,
        session_id: str,
        destination: str,
        persistence: ProjectPersistence,
        payload_args: dict[str, Any],
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.destination = destination
        self.persistence = persistence
        self.payload_args = payload_args
        self.signals = SessionSaveSignals()

    def run(self) -> None:
        try:
            saved_path = self.persistence.save_session(self.destination, **self.payload_args)
            self.signals.finished.emit(self.session_id, saved_path)
        except Exception as exc:
            self.signals.error.emit(self.session_id, exc)

from models.annotation_model import AnnotationModel
from models.roi_model import RoiModel
from models.temp_mask_model import TempMaskModel
from models.view_state_model import ViewStateModel
from services.annotation_session_manager import AnnotationSessionManager
from services.nde_signal_processing_service import NdeSignalProcessingOptions
from services.project_persistence import ProjectPersistence
from views.session_manager_dialog import SessionManagerDialog


class SessionWorkspaceController:
    """Coordinate session persistence, dirty tracking and session UI workflows."""

    def __init__(
        self,
        *,
        main_window: QMainWindow,
        logger: logging.Logger,
        session_manager: AnnotationSessionManager,
        project_persistence: ProjectPersistence,
        annotation_model: AnnotationModel,
        temp_mask_model: TempMaskModel,
        roi_model: RoiModel,
        view_state_model: ViewStateModel,
        current_nde_path: Callable[[], Optional[str]],
        require_nde_path: Callable[[], str],
        get_annotation_axis_mode: Callable[[], str],
        get_signal_processing_selection: Callable[[], dict[str, bool]],
        load_nde_file: Callable[..., bool],
        after_session_switch: Callable[[], None],
        status_message: Callable[..., None],
        has_pending_mask_edits: Callable[[], bool],
        commit_pending_mask_edits: Callable[[], None],
        has_pending_corrosion_edits: Callable[[], bool],
        commit_pending_corrosion_edits: Callable[[], bool],
    ) -> None:
        self.main_window = main_window
        self.logger = logger
        self.session_manager = session_manager
        self.project_persistence = project_persistence
        self.annotation_model = annotation_model
        self.temp_mask_model = temp_mask_model
        self.roi_model = roi_model
        self.view_state_model = view_state_model
        self._current_nde_path = current_nde_path
        self._require_nde_path = require_nde_path
        self._get_annotation_axis_mode = get_annotation_axis_mode
        self._get_signal_processing_selection = get_signal_processing_selection
        self._load_nde_file = load_nde_file
        self._after_session_switch = after_session_switch
        self._status_message = status_message
        self._has_pending_mask_edits = has_pending_mask_edits
        self._commit_pending_mask_edits = commit_pending_mask_edits
        self._has_pending_corrosion_edits = has_pending_corrosion_edits
        self._commit_pending_corrosion_edits = commit_pending_corrosion_edits

        self._session_dialog: Optional[SessionManagerDialog] = None
        self._session_file_paths: dict[str, str] = {}
        self._session_dirty_flags: dict[str, bool] = {}
        self._session_autosave_paths: dict[str, str] = {}
        self._pending_autosave_session_ids: set[str] = set()
        self._active_save_workers: set[SessionSaveWorker] = set()
        self._autosave_inflight_session_ids: set[str] = set()
        self._autosave_cleanup_pending_ids: set[str] = set()
        self._autosave_cleanup_retry_counts: dict[str, int] = {}
        self._session_autosave_timer = QTimer(self.main_window)
        self._session_autosave_timer.setSingleShot(True)
        self._session_autosave_timer.timeout.connect(self._flush_session_autosaves)

    def open_session_via_dialog(self) -> None:
        """Open a persisted `.session` file and restore its single session."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Ouvrir une session",
            "",
            "Session Files (*.session);;All Files (*)",
        )
        if not file_path:
            return
        if not self.confirm_unsaved_sessions_before_reset("ouvrir une autre session"):
            return

        try:
            payload = self.project_persistence.load_session(file_path)
            nde_path = str(payload.nde_path)
            if not Path(nde_path).exists():
                raise FileNotFoundError(f"Fichier NDE introuvable: {nde_path}")

            processing_options = NdeSignalProcessingOptions(
                apply_hilbert=bool(
                    payload.signal_processing_selection.get("apply_hilbert", False)
                ),
                apply_smoothing=bool(
                    payload.signal_processing_selection.get("apply_smoothing", False)
                ),
            )
            if not self._load_nde_file(
                nde_path,
                prompt_open_options=False,
                axis_mode=payload.annotation_axis_mode,
                processing_options=processing_options,
            ):
                return

            session_dump = self._extract_single_session_dump(payload.session_manager_dump)
            active_id = self.session_manager.restore_dump(session_dump)
            if active_id is None:
                raise ValueError("Aucune session active a restaurer.")

            loaded_session_path = str(Path(file_path).with_suffix(".session"))
            loaded_session_name = self._session_name_from_path(
                loaded_session_path,
                fallback=self.session_manager.get_active_session_name(),
            )
            self.session_manager.rename_session(active_id, loaded_session_name)
            self.session_manager.switch_session(
                active_id,
                annotation_model=self.annotation_model,
                temp_mask_model=self.temp_mask_model,
                roi_model=self.roi_model,
                view_state_model=self.view_state_model,
                save_current=False,
            )
            self._session_file_paths = {active_id: loaded_session_path}
            self._set_session_dirty(active_id, False, schedule_autosave=False)
            self.refresh_session_dialog()
            self._after_session_switch()
            self._status_message(f"Session chargee: {file_path}", timeout_ms=5000)
        except Exception as exc:
            QMessageBox.critical(self.main_window, "Erreur session", str(exc))

    def save_active_session_via_ui(self, *, force_dialog: bool) -> None:
        """Save the active session and surface UI feedback/errors."""
        try:
            self._require_nde_path()
        except Exception:
            QMessageBox.warning(
                self.main_window,
                "Session",
                "Chargez un NDE avant de sauvegarder une session.",
            )
            return

        try:
            self.save_active_session(force_dialog=force_dialog, sync=False)
        except Exception as exc:
            QMessageBox.critical(self.main_window, "Erreur sauvegarde session", str(exc))

    def save_active_session(self, *, force_dialog: bool, sync: bool = False) -> Optional[str]:
        """Save the active session, optionally forcing the save dialog."""
        active_id = self.session_manager.get_active_session_id()
        if active_id is None:
            raise ValueError("Aucune session active a sauvegarder.")
        return self.save_session(active_id, force_dialog=force_dialog, clean_after_save=True, sync=sync)

    def save_session(
        self,
        session_id: str,
        *,
        force_dialog: bool,
        clean_after_save: bool,
        sync: bool = False,
    ) -> Optional[str]:
        """Save one session by id, optionally forcing the save dialog."""
        target_id = str(session_id).strip()
        if not target_id or not self.session_manager.has_session(target_id):
            raise ValueError("Session introuvable pour la sauvegarde.")

        if target_id == self.session_manager.get_active_session_id():
            if not self._prepare_active_session_for_persistence():
                return None

        current_name = self._normalize_session_name(self.session_manager.get_session_name(target_id))
        current_path = self._session_file_paths.get(target_id)
        destination = current_path

        if force_dialog or not current_path:
            suggested_path = self._suggest_session_save_path(
                session_name=current_name,
                current_path=current_path,
            )
            selected_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Enregistrer la session",
                suggested_path,
                "Session Files (*.session);;All Files (*)",
            )
            if not selected_path:
                return None
            destination = str(Path(selected_path).with_suffix(".session"))
            chosen_name = self._session_name_from_path(destination, fallback=current_name)
            self.session_manager.rename_session(target_id, chosen_name)
            self.refresh_session_dialog()

        if not destination:
            raise ValueError("Chemin de sauvegarde de session introuvable.")

        return self._persist_session_to_destination(
            target_id,
            destination,
            sync=sync,
            clean_after_save=clean_after_save,
            is_autosave=False,
        )

    def confirm_unsaved_sessions_before_reset(self, action_label: str) -> bool:
        """Prompt the user about unsaved sessions before replacing the workspace."""
        return self._confirm_unsaved_sessions(
            self._ordered_unsaved_session_ids(),
            action_label=action_label,
        )

    def on_main_window_close_event(self, event: Any) -> None:
        """Intercept window close to protect unsaved sessions."""
        if self._confirm_unsaved_sessions(
            self._ordered_unsaved_session_ids(),
            action_label="quitter l'application",
        ):
            event.accept()
            return
        event.ignore()

    def mark_active_session_dirty(self) -> None:
        """Flag the active session as modified and schedule autosave."""
        active_id = self.session_manager.get_active_session_id()
        if active_id is None:
            return
        self._set_session_dirty(active_id, True)

    def clear_runtime_state(self, *, remove_autosaves: bool) -> None:
        """Reset runtime-only session bookkeeping."""
        self._session_autosave_timer.stop()
        self._pending_autosave_session_ids.clear()
        if remove_autosaves:
            for session_id in list(self._session_autosave_paths.keys()):
                self._cleanup_session_autosave(session_id)
        else:
            self._session_autosave_paths.clear()
        self._session_file_paths.clear()
        self._session_dirty_flags.clear()

    def open_session_dialog(self) -> None:
        """Show the session selector dialog."""
        if self._session_dialog is None:
            self._session_dialog = SessionManagerDialog(parent=self.main_window)
            self._session_dialog.session_selected.connect(self.on_session_selected)
            self._session_dialog.session_created.connect(self.on_session_created)
            self._session_dialog.session_duplicated.connect(self.on_session_duplicated)
            self._session_dialog.session_deleted.connect(self.on_session_deleted)
        self.refresh_session_dialog()
        self._session_dialog.show()
        self._session_dialog.raise_()
        self._session_dialog.activateWindow()

    def refresh_session_dialog(self) -> None:
        """Refresh the session list shown in the selector dialog."""
        if self._session_dialog is None:
            return
        self._session_dialog.set_sessions(self.session_manager.list_sessions())

    def on_session_created(self, name: str) -> None:
        """Create a new empty session on the current dataset."""
        new_session_id = self.session_manager.create_empty_session(
            name=self._normalize_session_name(name),
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
            set_active=True,
            save_active=True,
        )
        self.register_unsaved_session(new_session_id, dirty=False)
        self._after_session_switch()

    def on_session_duplicated(self, name: str) -> None:
        """Duplicate the active session as a new unsaved session."""
        source_name = self.session_manager.get_active_session_name()
        duplicate_name = self._normalize_session_name(
            name,
            fallback=f"{self._normalize_session_name(source_name)} copy",
        )
        new_session_id = self.session_manager.create_from_models(
            name=duplicate_name,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
            set_active=True,
            save_active=True,
        )
        self.register_unsaved_session(new_session_id, dirty=True)
        self._after_session_switch()

    def on_session_selected(self, session_id: str) -> None:
        """Switch the active session."""
        self.session_manager.switch_session(
            session_id,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
            save_current=True,
        )
        self._after_session_switch()

    def on_session_deleted(self, session_id: str) -> None:
        """Delete a session after confirming unsaved changes."""
        had_session = self.session_manager.has_session(session_id)
        if had_session and not self._confirm_unsaved_sessions(
            [session_id],
            action_label="supprimer cette session",
        ):
            return

        self.session_manager.delete_session(
            session_id,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
        )
        if had_session and not self.session_manager.has_session(session_id):
            self._session_file_paths.pop(session_id, None)
            self._session_dirty_flags.pop(session_id, None)
            self._pending_autosave_session_ids.discard(session_id)
            self._cleanup_session_autosave(session_id)
        self.refresh_session_dialog()
        self._after_session_switch()

    def register_unsaved_session(self, session_id: str, *, dirty: bool) -> None:
        """Detach a session from any persisted path and update its dirty state."""
        target_id = str(session_id).strip()
        if not target_id:
            return
        self._session_file_paths.pop(target_id, None)
        self._set_session_dirty(target_id, dirty, schedule_autosave=dirty)
        self.refresh_session_dialog()

    def _prepare_active_session_for_persistence(self) -> bool:
        """Apply pending live edits so the saved session reflects the visible document."""
        if self._has_pending_mask_edits():
            self._commit_pending_mask_edits()
        if self._has_pending_corrosion_edits():
            if not self._commit_pending_corrosion_edits():
                QMessageBox.warning(
                    self.main_window,
                    "Session",
                    "Impossible d'appliquer les modifications corrosion en cours avant la sauvegarde.",
                )
                return False
            self.mark_active_session_dirty()
        return True

    def _persist_session_to_destination(
        self, 
        session_id: str, 
        destination: str,
        *,
        sync: bool = False,
        clean_after_save: bool = False,
        is_autosave: bool = False,
    ) -> Optional[str]:
        """Serialize one session to the provided destination path in background."""
        normalized_session_id = str(session_id).strip()
        session_dump = self.session_manager.build_session_dump(
            normalized_session_id,
            annotation_model=self.annotation_model,
            temp_mask_model=self.temp_mask_model,
            roi_model=self.roi_model,
            view_state_model=self.view_state_model,
        )
        payload_args = {
            "nde_path": self._require_nde_path(),
            "annotation_axis_mode": self._get_annotation_axis_mode(),
            "signal_processing_selection": self._get_signal_processing_selection(),
            "session_manager_dump": session_dump,
        }

        if sync:
            saved_path = self.project_persistence.save_session(destination, **payload_args)
            if not is_autosave:
                self._session_file_paths[session_id] = saved_path
            if clean_after_save:
                self._set_session_dirty(session_id, False, schedule_autosave=False)
            return saved_path

        worker = SessionSaveWorker(
            session_id=normalized_session_id,
            destination=destination,
            persistence=self.project_persistence,
            payload_args=payload_args,
        )
        worker.setAutoDelete(False)
        self._active_save_workers.add(worker)
        if is_autosave:
            self._autosave_inflight_session_ids.add(normalized_session_id)

        def finalize_worker(sid: str) -> None:
            target_id = str(sid).strip()
            self._active_save_workers.discard(worker)
            if not is_autosave:
                return
            self._autosave_inflight_session_ids.discard(target_id)
            if target_id in self._autosave_cleanup_pending_ids:
                self._cleanup_session_autosave(target_id)

        def on_finished(sid: str, path: str) -> None:
            if not is_autosave:
                self._session_file_paths[sid] = path
                self._status_message(f"Session sauvegardee: {path}", timeout_ms=5000)
            if clean_after_save:
                self._set_session_dirty(sid, False, schedule_autosave=False)
            finalize_worker(sid)

        def on_error(sid: str, exc: object) -> None:
            err_msg = str(exc)
            self.logger.exception(f"Erreur Async Save Session {sid}: {err_msg}")
            if not is_autosave:
                QMessageBox.critical(self.main_window, "Erreur sauvegarde", err_msg)
            finalize_worker(sid)

        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)

        QThreadPool.globalInstance().start(worker)
        if not is_autosave:
            self._status_message("Sauvegarde de session en cours...", timeout_ms=3000)
        return None

    def _set_session_dirty(
        self,
        session_id: str,
        dirty: bool,
        *,
        schedule_autosave: bool = True,
    ) -> None:
        target_id = str(session_id).strip()
        if not target_id:
            return
        self._session_dirty_flags[target_id] = bool(dirty)
        if dirty:
            if schedule_autosave:
                self._schedule_session_autosave(target_id)
            return
        self._pending_autosave_session_ids.discard(target_id)
        self._cleanup_session_autosave(target_id)

    def _is_session_dirty(self, session_id: str) -> bool:
        return bool(self._session_dirty_flags.get(str(session_id).strip(), False))

    def _active_session_has_pending_runtime_edits(self) -> bool:
        return bool(
            self._has_pending_mask_edits()
            or self._has_pending_corrosion_edits()
        )

    def _session_has_unsaved_changes(self, session_id: str) -> bool:
        target_id = str(session_id).strip()
        if not target_id:
            return False
        if self._is_session_dirty(target_id):
            return True
        return (
            target_id == self.session_manager.get_active_session_id()
            and self._active_session_has_pending_runtime_edits()
        )

    def _ordered_unsaved_session_ids(self) -> list[str]:
        active_id = self.session_manager.get_active_session_id()
        ordered: list[str] = []
        if isinstance(active_id, str) and self._session_has_unsaved_changes(active_id):
            ordered.append(active_id)
        for session_id, _name, _is_active in self.session_manager.list_sessions():
            if session_id == active_id:
                continue
            if self._is_session_dirty(session_id):
                ordered.append(session_id)
        return ordered

    def _confirm_unsaved_sessions(self, session_ids: list[str], *, action_label: str) -> bool:
        for session_id in session_ids:
            if not self._session_has_unsaved_changes(session_id):
                continue
            if not self._confirm_unsaved_session(session_id, action_label=action_label):
                return False
        return True

    def _confirm_unsaved_session(self, session_id: str, *, action_label: str) -> bool:
        session_name = self._normalize_session_name(self.session_manager.get_session_name(session_id))
        box = QMessageBox(self.main_window)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Session non enregistree")
        box.setText(f"La session '{session_name}' contient des modifications non enregistrees.")
        details = f"Voulez-vous l'enregistrer avant de {action_label} ?"
        if (
            session_id == self.session_manager.get_active_session_id()
            and self._active_session_has_pending_runtime_edits()
        ):
            details += "\n\nLes modifications en cours seront consolidees avant la sauvegarde."
        box.setInformativeText(details)
        save_button = box.addButton("Enregistrer", QMessageBox.ButtonRole.AcceptRole)
        discard_button = box.addButton(
            "Ne pas enregistrer",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_button = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(save_button)
        box.exec()

        clicked = box.clickedButton()
        if clicked is save_button:
            saved_path = self.save_session(session_id, force_dialog=False, clean_after_save=True, sync=True)
            return bool(saved_path)
        if clicked is discard_button:
            return True
        return clicked is not cancel_button

    def _schedule_session_autosave(self, session_id: str) -> None:
        self._pending_autosave_session_ids.add(session_id)
        self._session_autosave_timer.start(30000)

    def _flush_session_autosaves(self) -> None:
        pending_ids = sorted(self._pending_autosave_session_ids)
        self._pending_autosave_session_ids.clear()
        nde_path = self._current_nde_path()
        for session_id in pending_ids:
            if not self.session_manager.has_session(session_id):
                continue
            if not self._is_session_dirty(session_id):
                continue
            if not nde_path:
                continue
            if session_id in self._autosave_inflight_session_ids:
                self._pending_autosave_session_ids.add(session_id)
                continue
            destination = self._session_autosave_paths.get(session_id)
            if not destination:
                destination = self._autosave_session_path(session_id, nde_path=nde_path)
                self._session_autosave_paths[session_id] = destination
            try:
                self._persist_session_to_destination(
                    session_id,
                    destination,
                    sync=False,
                    clean_after_save=False,
                    is_autosave=True,
                )
            except Exception:
                self.logger.exception("Unable to autosave temporary session %s", session_id)
        if self._pending_autosave_session_ids:
            self._session_autosave_timer.start(1000)

    @staticmethod
    def _autosave_session_path(session_id: str, *, nde_path: str) -> str:
        dataset_name = Path(nde_path).stem or "unsaved"
        base_dir = (
            Path(tempfile.gettempdir())
            / "segmentation-tool"
            / "session-autosave"
            / dataset_name
        )
        return str(base_dir / f"{session_id}.session")

    def _cleanup_session_autosave(self, session_id: str) -> None:
        target_id = str(session_id).strip()
        if not target_id:
            return
        path_str = self._session_autosave_paths.get(target_id)
        if not path_str:
            self._autosave_cleanup_pending_ids.discard(target_id)
            self._autosave_cleanup_retry_counts.pop(target_id, None)
            return
        if target_id in self._autosave_inflight_session_ids:
            self._autosave_cleanup_pending_ids.add(target_id)
            return
        try:
            path = Path(path_str)
            if path.exists():
                path.unlink()
            self._session_autosave_paths.pop(target_id, None)
            self._autosave_cleanup_pending_ids.discard(target_id)
            self._autosave_cleanup_retry_counts.pop(target_id, None)
        except PermissionError:
            retry_count = self._autosave_cleanup_retry_counts.get(target_id, 0) + 1
            self._autosave_cleanup_retry_counts[target_id] = retry_count
            self._autosave_cleanup_pending_ids.add(target_id)
            if retry_count <= 5:
                QTimer.singleShot(
                    250 * retry_count,
                    lambda sid=target_id: self._cleanup_session_autosave(sid),
                )
                return
            self.logger.warning(
                "Temporary session autosave still locked after %d retries: %s",
                retry_count,
                path_str,
            )
        except Exception:
            self.logger.exception("Unable to remove temporary session autosave: %s", path_str)

    def _suggest_session_save_path(
        self,
        *,
        session_name: Optional[str],
        current_path: Optional[str] = None,
    ) -> str:
        """Return the default destination proposed by the save dialog."""
        if current_path:
            return str(Path(current_path))
        nde_path = Path(self._require_nde_path())
        normalized_name = self._normalize_session_name(session_name)
        return str(nde_path.with_name(f"{normalized_name}.session"))

    @staticmethod
    def _extract_single_session_dump(payload: dict[str, Any]) -> dict[str, Any]:
        """Reduce any persisted dump to a single active session payload."""
        sessions = payload.get("sessions")
        if not isinstance(sessions, dict) or not sessions:
            raise ValueError("Dump de sessions invalide ou vide.")

        active_id = payload.get("active_id")
        if not isinstance(active_id, str) or active_id not in sessions:
            active_id = next(iter(sessions.keys()))
        return {
            "sessions": {active_id: sessions[active_id]},
            "active_id": active_id,
        }

    @staticmethod
    def _normalize_session_name(session_name: Optional[str], *, fallback: str = "New session") -> str:
        raw_name = str(session_name or "").strip()
        return raw_name or fallback

    @staticmethod
    def _session_name_from_path(file_path: str, *, fallback: Optional[str] = None) -> str:
        stem = Path(file_path).stem.strip()
        if stem:
            return stem
        return SessionWorkspaceController._normalize_session_name(fallback)
