"""Dock Session : enregistrements / canaux, puis zone Mapping MEA en dessous."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.channel_panel import ChannelPanel
from gui.widgets.recordings_panel import RecordingsPanel


class SessionPanel(QWidget):
    """Enregistrements / Canaux en haut ; Mapping MEA toujours visible en dessous."""

    probePathChanged = Signal(object)  # Path | None

    def __init__(
        self,
        recordings: RecordingsPanel,
        channels: ChannelPanel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.recordings = recordings
        self.channels = channels
        self._updating_path = False

        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(recordings, "Enregistrements")
        self.tabs.addTab(channels, "Canaux")
        self.tabs.setTabToolTip(
            0, "Ajouter des .rhs, traiter, comparer couleurs / légendes."
        )
        self.tabs.setTabToolTip(
            1, "Sélectionner un canal (aperçu) ; cases = visibilité en revue montage."
        )

        self.map_box = QGroupBox("Mapping MEA", self)
        self.map_box.setObjectName("meaMapBox")
        self.map_box.setMinimumHeight(280)
        self.map_box.setToolTip(
            "Carte des électrodes : clic = sélectionner, double-clic = inspecter."
        )

        self._help = QLabel(
            "JSON de géométrie (optionnel) — sans lui la carte reste vide, "
            "le traitement .rhs fonctionne quand même.",
            self.map_box,
        )
        self._help.setObjectName("hintLabel")
        self._help.setWordWrap(True)

        self._path_edit = QLineEdit(self.map_box)
        self._path_edit.setPlaceholderText("Chemin du JSON de sonde MEA…")
        self._path_edit.setClearButtonEnabled(True)
        self._path_edit.setToolTip(
            "Formats acceptés : probeinterface ou mea_editor (.json).\n"
            "Exemple : un fichier exporté depuis l’éditeur de sonde / probeinterface."
        )
        self._path_edit.editingFinished.connect(self._on_path_edited)

        self._btn_load = QPushButton("Charger…", self.map_box)
        self._btn_load.setToolTip("Choisir un fichier JSON de mapping MEA")
        self._btn_load.clicked.connect(self._browse_probe)
        self._btn_clear = QPushButton("Effacer", self.map_box)
        self._btn_clear.setToolTip("Retirer le mapping (la carte redevient vide)")
        self._btn_clear.clicked.connect(self._clear_probe)

        path_row = QHBoxLayout()
        path_row.setSpacing(4)
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(self._btn_load)
        path_row.addWidget(self._btn_clear)

        map_layout = QVBoxLayout(self.map_box)
        map_layout.setContentsMargins(6, 12, 6, 6)
        map_layout.setSpacing(6)
        map_layout.addWidget(self._help)
        map_layout.addLayout(path_row)
        map_layout.addWidget(channels.map_section, 1)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setObjectName("sessionSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.map_box)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([280, 420])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter)

    # ------------------------------------------------------------------- API

    def show_recordings(self) -> None:
        self.tabs.setCurrentWidget(self.recordings)

    def show_channels(self) -> None:
        self.tabs.setCurrentWidget(self.channels)

    def set_probe_path(self, path: Path | str | None) -> None:
        text = str(path) if path else ""
        if self._path_edit.text() == text:
            return
        self._updating_path = True
        self._path_edit.setText(text)
        self._updating_path = False

    def probe_path(self) -> Path | None:
        text = self._path_edit.text().strip()
        return Path(text) if text else None

    # ---------------------------------------------------------------- actions

    def _browse_probe(self) -> None:
        start = self._path_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Charger un mapping MEA (JSON)",
            start,
            "JSON sonde MEA (*.json);;Tous les fichiers (*)",
        )
        if not path:
            return
        self._updating_path = True
        self._path_edit.setText(path)
        self._updating_path = False
        self.probePathChanged.emit(Path(path))

    def _clear_probe(self) -> None:
        if not self._path_edit.text() and not self._updating_path:
            self.probePathChanged.emit(None)
            return
        self._updating_path = True
        self._path_edit.clear()
        self._updating_path = False
        self.probePathChanged.emit(None)

    def _on_path_edited(self) -> None:
        if self._updating_path:
            return
        self.probePathChanged.emit(self.probe_path())
