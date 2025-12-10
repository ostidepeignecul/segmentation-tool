from __future__ import annotations

from typing import Iterable, Optional, Tuple

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


class SessionManagerDialog(QDialog):
    """Dialogue léger pour lister, créer, supprimer et activer des sessions."""

    session_selected = pyqtSignal(str)
    session_created = pyqtSignal(str)
    session_deleted = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sessions d'annotation")
        self.setModal(False)
        self.setMinimumWidth(360)

        self._list = QListWidget(self)
        self._name_edit = QLineEdit(self)
        self._name_edit.setPlaceholderText("Nom de la nouvelle session")

        create_btn = QPushButton("Créer (dupliquer l'actuelle)", self)
        delete_btn = QPushButton("Supprimer", self)
        close_btn = QPushButton("Fermer", self)

        create_btn.clicked.connect(self._on_create)
        delete_btn.clicked.connect(self._on_delete)
        close_btn.clicked.connect(self.close)
        self._list.currentItemChanged.connect(self._on_selection_changed)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Sessions disponibles :", self))
        layout.addWidget(self._list, 1)
        layout.addWidget(self._name_edit, 0)

        buttons = QHBoxLayout()
        buttons.addWidget(create_btn)
        buttons.addWidget(delete_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_sessions(self, sessions: Iterable[Tuple[str, str, bool]]) -> None:
        """Alimente la liste avec (id, nom, is_active)."""
        current_id = self._current_session_id()
        self._list.blockSignals(True)
        self._list.clear()
        for sid, name, is_active in sessions:
            item = QListWidgetItem(f"{name} ({sid[:8]})")
            item.setData(256, sid)  # Qt.UserRole
            if is_active:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._list.addItem(item)
            if is_active or sid == current_id:
                self._list.setCurrentItem(item)
        self._list.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _current_session_id(self) -> Optional[str]:
        item = self._list.currentItem()
        return item.data(256) if item else None

    def _on_selection_changed(self, current: QListWidgetItem, _prev: QListWidgetItem) -> None:
        if current is None:
            return
        sid = current.data(256)
        if sid:
            self.session_selected.emit(sid)

    def _on_create(self) -> None:
        name = self._name_edit.text().strip() or "Nouvelle session"
        self.session_created.emit(name)
        self._name_edit.clear()

    def _on_delete(self) -> None:
        sid = self._current_session_id()
        if sid:
            self.session_deleted.emit(sid)

