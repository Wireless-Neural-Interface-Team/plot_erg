"""Dock Session : enregistrements et canaux dans un seul panneau à onglets."""

from __future__ import annotations

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from gui.widgets.channel_panel import ChannelPanel
from gui.widgets.recordings_panel import RecordingsPanel


class SessionPanel(QWidget):
    """Navigation de session : fichiers d’abord, puis canaux."""

    def __init__(
        self,
        recordings: RecordingsPanel,
        channels: ChannelPanel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.recordings = recordings
        self.channels = channels

        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(recordings, "Recordings")
        self.tabs.addTab(channels, "Channels")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs)

    def show_recordings(self) -> None:
        self.tabs.setCurrentWidget(self.recordings)

    def show_channels(self) -> None:
        self.tabs.setCurrentWidget(self.channels)
