"""Sélection de canal : carte MEA + liste filtrable, synchronisées."""

from __future__ import annotations

from typing import Any, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .mea_map import MeaMapWidget

_METRIC_NONE = "Sans couleur"
_METRIC_RMS = "RMS moyen"
_METRIC_SPIKES = "Spikes"
_HIDDEN_FG = QColor("#94a3b8")


class ChannelPanel(QWidget):
    """Choisir le canal affiché, depuis la liste ou la carte."""

    channelChanged = Signal(str)
    inspectChannelRequested = Signal(str)
    visibilityChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._channels: list[str] = []
        self._hidden: set[str] = set()
        self._current: str | None = None
        self._updating = False
        self._recording: Any | None = None

        self.list = QListWidget(self)
        self.list.setAlternatingRowColors(True)
        self.list.setUniformItemSizes(True)
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.setToolTip(
            "Coche : visible dans le montage.\n"
            "Clic : sélectionner (aperçu central).\n"
            "Double-clic : inspecter le canal (barres de plage).\n"
            "Clic droit : masquer / afficher."
        )
        self.list.currentItemChanged.connect(self._on_list_changed)
        self.list.itemDoubleClicked.connect(self._on_list_double_clicked)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.customContextMenuRequested.connect(self._on_context_menu)

        self.map = MeaMapWidget(self)
        self.map.setToolTip(
            "Clic : sélectionner le canal.\n"
            "Double-clic : inspecter (traces + barres de plage)."
        )
        self.map.channelSelected.connect(self._on_map_clicked)
        self.map.channelActivated.connect(self._on_map_activated)

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

        self._btn_show_all = QPushButton("Tout", self)
        self._btn_show_all.setToolTip("Afficher tous les canaux dans le montage")
        self._btn_show_all.clicked.connect(self.show_all)
        self._btn_hide_selected = QPushButton("Masquer", self)
        self._btn_hide_selected.setToolTip("Masquer les canaux sélectionnés dans le montage")
        self._btn_hide_selected.clicked.connect(self.hide_selected)
        self._btn_solo = QPushButton("Seul", self)
        self._btn_solo.setToolTip("N’afficher que les canaux sélectionnés dans le montage")
        self._btn_solo.clicked.connect(self.solo_selected)

        self._title = QLabel("Aucun canal")
        self._title.setObjectName("sectionTitle")
        self._visibility_label = QLabel("", self)
        self._visibility_label.setObjectName("hintLabel")

        header = QHBoxLayout()
        header.setSpacing(4)
        header.addWidget(self._title, 1)
        header.addWidget(prev_button)
        header.addWidget(next_button)

        visibility_row = QHBoxLayout()
        visibility_row.setSpacing(4)
        visibility_row.addWidget(self._btn_show_all)
        visibility_row.addWidget(self._btn_hide_selected)
        visibility_row.addWidget(self._btn_solo)
        visibility_row.addWidget(self._visibility_label, 1)

        # Zone carte exposée à SessionPanel (sous Recordings / Channels).
        self.map_section = QWidget(self)
        self.map_section.setObjectName("meaMapSection")
        map_layout = QVBoxLayout(self.map_section)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(4)
        map_controls = QHBoxLayout()
        map_controls.setSpacing(4)
        color_label = QLabel("Couleur :", self.map_section)
        color_label.setToolTip(
            "Teinte optionnelle des contacts selon une métrique par canal "
            "(RMS moyen ou nombre de spikes). N’affecte pas le traitement."
        )
        self._metric_box.setToolTip(
            "Colorer les électrodes selon une métrique (aide visuelle seulement)."
        )
        map_controls.addWidget(color_label)
        map_controls.addWidget(self._metric_box, 1)
        map_controls.addWidget(self._labels_box)
        map_layout.addLayout(map_controls)
        map_layout.addWidget(self.map, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(self._filter)
        layout.addLayout(visibility_row)
        layout.addWidget(self.list, 1)

    # ------------------------------------------------------------------- inputs

    def set_channels(self, names: Sequence[str], *, keep_selection: bool = True) -> None:
        self._channels = [str(name) for name in names]
        previous = self._current if keep_selection else None
        # Conserver le masquage pour les canaux encore présents.
        self._hidden = {name for name in self._hidden if name in self._channels}
        self._updating = True
        self.list.clear()
        for name in self._channels:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(
                Qt.CheckState.Unchecked if name in self._hidden else Qt.CheckState.Checked
            )
            self.list.addItem(item)
        self._updating = False
        self._apply_filter(self._filter.text())
        self._refresh_ready_badges()
        self._sync_map_hidden()
        self._update_visibility_label()
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
        self._sync_map_hidden()

    def set_reference_recording(self, recording: Any | None) -> None:
        """Enregistrement utilisé pour colorer la carte (premier tracé)."""
        self._recording = recording
        self._refresh_ready_badges()
        self._refresh_metric()

    def _refresh_ready_badges(self) -> None:
        """Préfixe ✓ / ○ selon que le canal a déjà été calculé."""
        recording = self._recording
        self._updating = True
        for row in range(self.list.count()):
            item = self.list.item(row)
            name = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
            if not name:
                name = str(item.text()).lstrip("✓○ ").strip()
            ready = False
            if recording is not None and hasattr(recording, "channel_index"):
                index = recording.channel_index(name)
                if index is not None and hasattr(recording, "is_channel_ready"):
                    ready = bool(recording.is_channel_ready(int(index)))
            mark = "✓ " if ready else "○ "
            item.setText(f"{mark}{name}")
            tip = "Canal calculé" if ready else "Canal non calculé (F6 / Inspecter)"
            if name in self._hidden:
                tip += " — masqué dans le montage"
            item.setToolTip(tip)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setForeground(QBrush(_HIDDEN_FG) if name in self._hidden else QBrush())
            item.setCheckState(
                Qt.CheckState.Unchecked if name in self._hidden else Qt.CheckState.Checked
            )
        self._updating = False

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

    @property
    def hidden_channels(self) -> tuple[str, ...]:
        return tuple(name for name in self._channels if name in self._hidden)

    @property
    def visible_channels(self) -> list[str]:
        return [name for name in self._channels if name not in self._hidden]

    def select(self, channel: str, *, emit: bool = True) -> None:
        channel = str(channel).lstrip("✓○ ").strip()
        if channel not in self._channels:
            return
        changed = channel != self._current
        self._current = channel
        self._title.setText(channel)
        self._updating = True
        for row in range(self.list.count()):
            item = self.list.item(row)
            name = str(item.data(Qt.ItemDataRole.UserRole) or item.text()).lstrip("✓○ ")
            if name == channel:
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
            str(self.list.item(row).data(Qt.ItemDataRole.UserRole) or self.list.item(row).text()).lstrip("✓○ ")
            for row in range(self.list.count())
            if not self.list.item(row).isHidden()
        ] or self._channels
        try:
            index = visible.index(self._current) if self._current in visible else 0
        except ValueError:
            index = 0
        self.select(visible[(index + delta) % len(visible)])

    # -------------------------------------------------------------- visibility

    def show_all(self) -> None:
        if not self._hidden:
            return
        self._hidden.clear()
        self._apply_hidden_to_items()
        self._emit_visibility()

    def hide_selected(self) -> None:
        names = self._selected_names()
        if not names:
            if self._current:
                names = [self._current]
            else:
                return
        before = set(self._hidden)
        self._hidden.update(names)
        # Toujours garder au moins un canal visible.
        if len(self._hidden) >= len(self._channels) and self._channels:
            keep = names[0] if names[0] in self._channels else self._channels[0]
            self._hidden.discard(keep)
        if self._hidden == before:
            return
        self._apply_hidden_to_items()
        self._emit_visibility()

    def solo_selected(self) -> None:
        names = self._selected_names()
        if not names and self._current:
            names = [self._current]
        if not names:
            return
        keep = {name for name in names if name in self._channels}
        if not keep:
            return
        self._hidden = {name for name in self._channels if name not in keep}
        self._apply_hidden_to_items()
        self._emit_visibility()

    def set_channel_hidden(self, channel: str, hidden: bool) -> None:
        channel = str(channel).lstrip("✓○ ").strip()
        if channel not in self._channels:
            return
        if hidden:
            if len(self._hidden) + 1 >= len(self._channels):
                return
            self._hidden.add(channel)
        else:
            self._hidden.discard(channel)
        self._apply_hidden_to_items()
        self._emit_visibility()

    def _selected_names(self) -> list[str]:
        names: list[str] = []
        for item in self.list.selectedItems():
            name = str(item.data(Qt.ItemDataRole.UserRole) or item.text()).lstrip("✓○ ").strip()
            if name and name not in names:
                names.append(name)
        return names

    def _apply_hidden_to_items(self) -> None:
        self._updating = True
        for row in range(self.list.count()):
            item = self.list.item(row)
            name = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
            item.setCheckState(
                Qt.CheckState.Unchecked if name in self._hidden else Qt.CheckState.Checked
            )
            item.setForeground(QBrush(_HIDDEN_FG) if name in self._hidden else QBrush())
            tip = item.toolTip().split(" — masqué")[0]
            if name in self._hidden and "masqué" not in tip:
                item.setToolTip(f"{tip} — masqué dans le montage")
            elif name not in self._hidden:
                item.setToolTip(tip.replace(" — masqué dans le montage", ""))
        self._updating = False
        self._sync_map_hidden()
        self._update_visibility_label()

    def _sync_map_hidden(self) -> None:
        if hasattr(self.map, "set_hidden_channels"):
            self.map.set_hidden_channels(self._hidden)

    def _update_visibility_label(self) -> None:
        total = len(self._channels)
        hidden = len(self._hidden)
        if total <= 0:
            self._visibility_label.setText("")
        elif hidden <= 0:
            self._visibility_label.setText(f"{total} visibles")
        else:
            self._visibility_label.setText(f"{total - hidden}/{total} visibles")

    def _emit_visibility(self) -> None:
        self.visibilityChanged.emit()

    # ------------------------------------------------------------------- slots

    def _on_list_changed(self, current: QListWidgetItem | None, _previous: Any) -> None:
        if self._updating or current is None:
            return
        name = str(current.data(Qt.ItemDataRole.UserRole) or current.text()).lstrip("✓○ ")
        self.select(name)

    def _on_list_double_clicked(self, item: QListWidgetItem) -> None:
        name = str(item.data(Qt.ItemDataRole.UserRole) or item.text()).lstrip("✓○ ")
        self.select(name)
        self._emit_inspect()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        name = str(item.data(Qt.ItemDataRole.UserRole) or item.text()).lstrip("✓○ ").strip()
        if not name or name not in self._channels:
            return
        want_hidden = item.checkState() == Qt.CheckState.Unchecked
        is_hidden = name in self._hidden
        if want_hidden == is_hidden:
            return
        if want_hidden and len(self._hidden) + 1 >= len(self._channels):
            # Empêcher de tout masquer.
            self._updating = True
            item.setCheckState(Qt.CheckState.Checked)
            self._updating = False
            return
        if want_hidden:
            self._hidden.add(name)
        else:
            self._hidden.discard(name)
        # Mise à jour locale légère uniquement — le montage est redessiné via debounce.
        item.setForeground(QBrush(_HIDDEN_FG) if name in self._hidden else QBrush())
        tip = item.toolTip().split(" — masqué")[0]
        if name in self._hidden:
            item.setToolTip(f"{tip} — masqué dans le montage")
        else:
            item.setToolTip(tip)
        self._sync_map_hidden()
        self._update_visibility_label()
        self._emit_visibility()

    def _on_context_menu(self, pos: Any) -> None:
        item = self.list.itemAt(pos)
        if item is not None and item not in self.list.selectedItems():
            self.list.setCurrentItem(item)
        menu = QMenu(self)
        act_hide = QAction("Masquer la sélection", self)
        act_hide.triggered.connect(self.hide_selected)
        act_solo = QAction("Afficher uniquement la sélection", self)
        act_solo.triggered.connect(self.solo_selected)
        act_show = QAction("Tout afficher", self)
        act_show.triggered.connect(self.show_all)
        menu.addAction(act_hide)
        menu.addAction(act_solo)
        menu.addSeparator()
        menu.addAction(act_show)
        menu.exec(self.list.mapToGlobal(pos))

    def _on_map_clicked(self, channel: str) -> None:
        self.select(str(channel))

    def _on_map_activated(self, channel: str) -> None:
        self.select(str(channel))
        self._emit_inspect()

    def _emit_inspect(self) -> None:
        if self._current:
            self.inspectChannelRequested.emit(self._current)

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().casefold()
        for row in range(self.list.count()):
            item = self.list.item(row)
            name = str(item.data(Qt.ItemDataRole.UserRole) or item.text()).lstrip("✓○ ")
            item.setHidden(bool(needle) and needle not in name.casefold())

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
                if hasattr(recording, "is_channel_ready") and not recording.is_channel_ready(index):
                    continue
                try:
                    value = float(recording.channel_rms_uv(index))
                except Exception:
                    continue
                if value == value and value > 0:
                    values[str(name)] = value
            self.map.set_metric(values if values else None, "RMS moyen µV" if values else "")
            return
        spikes = getattr(recording, "spikes", None)
        if spikes is None and not hasattr(recording, "spike_count"):
            self.map.set_metric(None, "")
            return
        for index, name in enumerate(names):
            if hasattr(recording, "is_channel_ready") and not recording.is_channel_ready(index):
                continue
            try:
                if hasattr(recording, "spike_count"):
                    values[str(name)] = float(recording.spike_count(index))
                else:
                    values[str(name)] = float(spikes.total_for_channel(index))
            except Exception:
                continue
        self.map.set_metric(values if values else None, "spikes" if values else "")
