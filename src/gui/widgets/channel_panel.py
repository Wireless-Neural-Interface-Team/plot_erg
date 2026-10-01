"""Sélection de canal : carte MEA + liste filtrable, synchronisées."""

from __future__ import annotations

from typing import Any, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .mea_map import MeaMapWidget

_METRIC_NONE = "Sans couleur"
_METRIC_RMS = "RMS moyen"
_METRIC_SPIKES = "Spikes"


class ChannelPanel(QWidget):
    """Choisir le canal affiché, depuis la liste ou la carte."""

    channelChanged = Signal(str)
    openGraphRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._channels: list[str] = []
        self._current: str | None = None
        self._updating = False
        self._recording: Any | None = None

        self.list = QListWidget(self)
        self.list.setAlternatingRowColors(True)
        self.list.setUniformItemSizes(True)
        self.list.setToolTip("Double-clic : ouvrir un graphique pour ce canal.")
        self.list.currentItemChanged.connect(self._on_list_changed)
        self.list.itemDoubleClicked.connect(self._on_list_double_clicked)

        self.map = MeaMapWidget(self)
        self.map.channelSelected.connect(self._on_map_clicked)

        self._filter = QLineEdit(self)
        self._filter.setPlaceholderText("Filtrer…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)

        self._metric_box = QComboBox(self)
        self._metric_box.addItems([_METRIC_NONE, _METRIC_RMS, _METRIC_SPIKES])
        self._metric_box.setCurrentIndex(1)
        self._metric_box.setToolTip("Colorer la carte selon une métrique par canal.")
        self._metric_box.currentIndexChanged.connect(lambda _i: self._refresh_metric())

        self._labels_box = QCheckBox("Noms", self)
        self._labels_box.setChecked(True)
        self._labels_box.setToolTip("Afficher le nom des contacts sur la carte.")
        self._labels_box.toggled.connect(self.map.set_labels_visible)

        prev_button = QToolButton(self)
        prev_button.setText("◀")
        prev_button.setToolTip("Canal précédent (Ctrl+←)")
        prev_button.setAutoRaise(True)
        prev_button.clicked.connect(lambda: self.step(-1))
        next_button = QToolButton(self)
        next_button.setText("▶")
        next_button.setToolTip("Canal suivant (Ctrl+→)")
        next_button.setAutoRaise(True)
        next_button.clicked.connect(lambda: self.step(1))

        self._title = QLabel("Aucun canal")
        self._title.setObjectName("sectionTitle")

        header = QHBoxLayout()
        header.setSpacing(4)
        header.addWidget(self._title, 1)
        header.addWidget(prev_button)
        header.addWidget(next_button)

        list_side = QWidget(self)
        list_layout = QVBoxLayout(list_side)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(4)
        list_layout.addWidget(self._filter)
        list_layout.addWidget(self.list, 1)

        map_side = QWidget(self)
        map_layout = QVBoxLayout(map_side)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(4)
        map_controls = QHBoxLayout()
        map_controls.setSpacing(4)
        map_controls.addWidget(self._metric_box, 1)
        map_controls.addWidget(self._labels_box)
        map_layout.addLayout(map_controls)
        map_layout.addWidget(self.map, 1)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(map_side)
        splitter.addWidget(list_side)
        splitter.setSizes([360, 200])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------- inputs

    def set_channels(self, names: Sequence[str], *, keep_selection: bool = True) -> None:
        self._channels = [str(name) for name in names]
        previous = self._current if keep_selection else None
        self._updating = True
        self.list.clear()
        for name in self._channels:
            self.list.addItem(QListWidgetItem(name))
        self._updating = False
        self._apply_filter(self._filter.text())
        target = previous if previous in self._channels else (self._channels[0] if self._channels else None)
        if target is None:
            self._current = None
            self._title.setText("Aucun canal")
            self.map.set_selected_channel(None)
        else:
            self.select(target, emit=target != previous)

    def set_probe(self, layout: Any | None) -> None:
        self.map.set_probe(layout, self._channels)
        self.map.set_selected_channel(self._current)

    def set_reference_recording(self, recording: Any | None) -> None:
        """Enregistrement utilisé pour colorer la carte (premier tracé)."""
        self._recording = recording
        self._refresh_metric()

    # ---------------------------------------------------------------- selection

    @property
    def current_channel(self) -> str | None:
        return self._current

    @property
    def current_index(self) -> int:
        if self._current is None:
            return 0
        try:
            return self._channels.index(self._current)
        except ValueError:
            return 0

    @property
    def channels(self) -> list[str]:
        return list(self._channels)

    def select(self, channel: str, *, emit: bool = True) -> None:
        if channel not in self._channels:
            return
        changed = channel != self._current
        self._current = channel
        self._title.setText(channel)
        self._updating = True
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.text() == channel:
                self.list.setCurrentItem(item)
                self.list.scrollToItem(item)
                break
        self._updating = False
        self.map.set_selected_channel(channel)
        if emit and changed:
            self.channelChanged.emit(channel)

    def step(self, delta: int) -> None:
        if not self._channels:
            return
        visible = [
            self.list.item(row).text()
            for row in range(self.list.count())
            if not self.list.item(row).isHidden()
        ] or self._channels
        try:
            index = visible.index(self._current) if self._current in visible else 0
        except ValueError:
            index = 0
        self.select(visible[(index + delta) % len(visible)])

    # ------------------------------------------------------------------- slots

    def _on_list_changed(self, current: QListWidgetItem | None, _previous: Any) -> None:
        if self._updating or current is None:
            return
        self.select(current.text())

    def _on_list_double_clicked(self, item: QListWidgetItem) -> None:
        self.select(item.text())
        self._emit_open_graph()

    def _on_map_clicked(self, channel: str) -> None:
        self.select(str(channel))

    def _emit_open_graph(self) -> None:
        if self._current:
            self.openGraphRequested.emit(self._current)

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().casefold()
        for row in range(self.list.count()):
            item = self.list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().casefold())

    def _refresh_metric(self) -> None:
        choice = self._metric_box.currentText()
        recording = self._recording
        if recording is None or choice == _METRIC_NONE:
            self.map.set_metric(None, "")
            return
        values: dict[str, float] = {}
        names = list(getattr(recording, "channel_names", []))
        if choice == _METRIC_RMS:
            for index, name in enumerate(names):
                try:
                    value = float(recording.channel_rms_uv(index))
                except Exception:
                    continue
                if value == value:
                    values[str(name)] = value
            self.map.set_metric(values, "RMS moyen µV")
            return
        spikes = getattr(recording, "spikes", None)
        if spikes is None and not hasattr(recording, "spike_count"):
            self.map.set_metric(None, "")
            return
        for index, name in enumerate(names):
            try:
                if hasattr(recording, "spike_count"):
                    values[str(name)] = float(recording.spike_count(index))
                else:
                    values[str(name)] = float(spikes.total_for_channel(index))
            except Exception:
                continue
        self.map.set_metric(values, "spikes")
