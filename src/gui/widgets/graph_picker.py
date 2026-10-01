"""Dialogue arborescent : choisir un graphique figé pour un canal."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from view_config import (
    SECTION_KEYS,
    SECTION_LABELS,
    PanelPlacement,
    panel_label,
)

_KEY_ROLE = int(Qt.ItemDataRole.UserRole)
_SECTION_ROLE = int(Qt.ItemDataRole.UserRole) + 1


@dataclass(frozen=True)
class GraphChoice:
    placement: PanelPlacement
    label: str


# Catalogue : vue complète seulement — les zooms se définissent dans la fenêtre ouverte.
_GRAPH_TREE: tuple[tuple[str, tuple[tuple[str, str | None], ...]], ...] = (
    (
        "Enregistrement (continu)",
        (("full_recording", None),),
    ),
    (
        "Analyse",
        (
            ("analysis_raw", "full"),
            ("analysis_hp", "full"),
            ("analysis_lp", "full"),
            ("analysis_rms", "full"),
        ),
    ),
    (
        "Spikes",
        (
            ("analysis_raster", "full"),
            ("analysis_psth", "full"),
            ("analysis_isi", "full"),
            ("analysis_overlay", "full"),
        ),
    ),
    (
        "Contexte",
        (
            ("mea_layout", None),
            ("impedance", None),
        ),
    ),
    (
        "Moyennes classiques",
        (
            ("mean_raw", "full"),
            ("mean_hp", "full"),
            ("mean_lp", "full"),
            ("rms", "full"),
        ),
    ),
    (
        "Stimulations classiques",
        (
            ("first_trigger_raw", "full"),
            ("first_trigger_hp", "full"),
            ("first_trigger_lp", "full"),
            ("second_trigger_raw", "full"),
            ("second_trigger_hp", "full"),
            ("second_trigger_lp", "full"),
        ),
    ),
)


class GraphTreeDialog(QDialog):
    """Choisir un graphique dans l’arbre pour le canal donné."""

    def __init__(self, channel_name: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Ouvrir un graphique — {channel_name}")
        self.resize(520, 560)
        self._choice: GraphChoice | None = None

        hint = QLabel(
            f"Canal : <b>{channel_name}</b><br>"
            "Choisissez un graphique. Il s’ouvrira dans une fenêtre figée "
            "avec ses propres paramètres.",
            self,
        )
        hint.setObjectName("workflowHint")
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)

        self._tree = QTreeWidget(self)
        self._tree.setHeaderLabels(["Graphiques"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setExpandsOnDoubleClick(False)
        self._populate()
        self._tree.itemDoubleClicked.connect(self._accept_item)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_current)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self._tree, 1)
        layout.addWidget(buttons)

    def _populate(self) -> None:
        for group, entries in _GRAPH_TREE:
            parent = QTreeWidgetItem([group])
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._tree.addTopLevelItem(parent)
            for panel, section in entries:
                if section is None:
                    label = panel_label(panel)
                    sec = "full"
                else:
                    label = f"{panel_label(panel)} — {SECTION_LABELS.get(section, section)}"
                    sec = section
                child = QTreeWidgetItem([label])
                child.setData(0, _KEY_ROLE, panel)
                child.setData(0, _SECTION_ROLE, sec)
                parent.addChild(child)
            parent.setExpanded(True)

    def _choice_from_item(self, item: QTreeWidgetItem | None) -> GraphChoice | None:
        if item is None:
            return None
        panel = item.data(0, _KEY_ROLE)
        if not panel:
            return None
        section = str(item.data(0, _SECTION_ROLE) or "full")
        if section not in SECTION_KEYS:
            section = "full"
        placement = PanelPlacement(panel=str(panel), section=section)  # type: ignore[arg-type]
        return GraphChoice(placement=placement, label=item.text(0))

    def _accept_item(self, item: QTreeWidgetItem, _column: int) -> None:
        choice = self._choice_from_item(item)
        if choice is None:
            return
        self._choice = choice
        self.accept()

    def _accept_current(self) -> None:
        choice = self._choice_from_item(self._tree.currentItem())
        if choice is None:
            return
        self._choice = choice
        self.accept()

    def choice(self) -> GraphChoice | None:
        return self._choice


def pick_graph_for_channel(channel_name: str, parent=None) -> GraphChoice | None:
    dialog = GraphTreeDialog(channel_name, parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.choice()
