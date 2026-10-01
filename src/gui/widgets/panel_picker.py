"""Dialog to choose which graphs a view tab shows, and in which order."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from panel_registry import PANEL_CATALOG, panel_info
from view_config import (
    SECTION_LABELS,
    PanelPlacement,
    ViewTab,
    is_section_independent,
)

_PLACEMENT_ROLE = int(Qt.ItemDataRole.UserRole) + 1


class PanelPickerDialog(QDialog):
    """Pick panels (left tree), order them (right list), set the grid geometry."""

    def __init__(self, tab: ViewTab, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Configurer les panneaux — {tab.name}")
        self.resize(900, 620)
        self._tab = tab

        self._name_edit = QLineEdit(tab.name)
        self._columns_spin = QSpinBox()
        self._columns_spin.setRange(1, 6)
        self._columns_spin.setValue(int(tab.columns))
        self._height_spin = QSpinBox()
        self._height_spin.setRange(160, 1200)
        self._height_spin.setSingleStep(20)
        self._height_spin.setSuffix(" px")
        self._height_spin.setValue(int(tab.panel_height_px))

        geometry = QFormLayout()
        geometry.addRow("Nom de la vue :", self._name_edit)
        geometry.addRow("Colonnes :", self._columns_spin)
        geometry.addRow("Hauteur minimale des panneaux :", self._height_spin)

        self._section_boxes: dict[str, QCheckBox] = {}
        sections_row = QHBoxLayout()
        sections_row.addWidget(QLabel("Section (montage principal) :"))
        box = QCheckBox(SECTION_LABELS["full"])
        box.setChecked(True)
        box.setEnabled(False)
        box.setToolTip("Les zooms personnalisés s’ajoutent dans les fenêtres d’analyse ouvertes.")
        self._section_boxes["full"] = box
        sections_row.addWidget(box)
        sections_row.addStretch(1)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Graphiques disponibles"])
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tree.setAlternatingRowColors(True)
        self._populate_tree()
        self._tree.itemDoubleClicked.connect(lambda *_: self._add_selected())

        self._selected = QListWidget()
        self._selected.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._selected.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._selected.setAlternatingRowColors(True)
        for placement in tab.panels:
            self._append_placement(placement)

        add_button = QPushButton("Ajouter →")
        add_button.clicked.connect(self._add_selected)
        remove_button = QPushButton("← Retirer")
        remove_button.setObjectName("dangerButton")
        remove_button.clicked.connect(self._remove_selected)
        up_button = QPushButton("Monter")
        up_button.clicked.connect(lambda: self._move_selected(-1))
        down_button = QPushButton("Descendre")
        down_button.clicked.connect(lambda: self._move_selected(1))
        clear_button = QPushButton("Tout effacer")
        clear_button.clicked.connect(self._selected.clear)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(sections_row)
        left_layout.addWidget(self._tree, 1)
        left_layout.addWidget(add_button)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("Affichés dans cette vue (glisser pour réordonner) :"))
        right_layout.addWidget(self._selected, 1)
        buttons_row = QHBoxLayout()
        for widget in (up_button, down_button, remove_button, clear_button):
            buttons_row.addWidget(widget)
        right_layout.addLayout(buttons_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([430, 430])

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(geometry)
        layout.addWidget(splitter, 1)
        layout.addWidget(box)

    # ------------------------------------------------------------------ helpers

    def _populate_tree(self) -> None:
        groups: dict[str, QTreeWidgetItem] = {}
        for info in PANEL_CATALOG:
            parent = groups.get(info.group)
            if parent is None:
                parent = QTreeWidgetItem(self._tree, [info.group])
                parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
                parent.setExpanded(True)
                groups[info.group] = parent
            item = QTreeWidgetItem(parent, [info.label])
            item.setData(0, _PLACEMENT_ROLE, info.key)
            hints: list[str] = []
            if info.is_global:
                hints.append("indépendant du canal")
            if info.needs_spikes:
                hints.append("nécessite la détection de spikes")
            if info.needs_impedance:
                hints.append("nécessite un CSV d’impédance")
            if info.needs_streams:
                hints.append("nécessite les flux bruts")
            item.setToolTip(0, ", ".join(hints) if hints else info.label)
        self._tree.expandAll()

    def _append_placement(self, placement: PanelPlacement) -> None:
        for row in range(self._selected.count()):
            existing = self._selected.item(row).data(_PLACEMENT_ROLE)
            if existing == (placement.panel, placement.section):
                return
        item = QListWidgetItem(placement.title())
        item.setData(_PLACEMENT_ROLE, (placement.panel, placement.section))
        self._selected.addItem(item)

    def _checked_sections(self) -> list[str]:
        checked = [key for key, box in self._section_boxes.items() if box.isChecked()]
        return checked or ["full"]

    def _add_selected(self) -> None:
        for item in self._tree.selectedItems():
            key = item.data(0, _PLACEMENT_ROLE)
            if not key:
                continue
            info = panel_info(str(key))
            if is_section_independent(info.key):
                self._append_placement(PanelPlacement(panel=info.key, section="full"))
                continue
            for section in self._checked_sections():
                self._append_placement(
                    PanelPlacement(panel=info.key, section=section)  # type: ignore[arg-type]
                )

    def _remove_selected(self) -> None:
        for item in self._selected.selectedItems():
            self._selected.takeItem(self._selected.row(item))

    def _move_selected(self, delta: int) -> None:
        rows = sorted(self._selected.row(item) for item in self._selected.selectedItems())
        if not rows:
            return
        if delta < 0 and rows[0] == 0:
            return
        if delta > 0 and rows[-1] == self._selected.count() - 1:
            return
        for row in rows if delta < 0 else reversed(rows):
            item = self._selected.takeItem(row)
            self._selected.insertItem(row + delta, item)
            item.setSelected(True)

    # ------------------------------------------------------------------- result

    def result_tab(self) -> ViewTab:
        placements: list[PanelPlacement] = []
        for row in range(self._selected.count()):
            data = self._selected.item(row).data(_PLACEMENT_ROLE)
            if not data:
                continue
            panel, section = data
            placements.append(PanelPlacement(panel=str(panel), section=str(section)))  # type: ignore[arg-type]
        name = self._name_edit.text().strip() or self._tab.name
        return ViewTab(
            name=name,
            panels=tuple(placements),
            columns=int(self._columns_spin.value()),
            panel_height_px=int(self._height_spin.value()),
        )


def pick_panels(tab: ViewTab, parent: QWidget | None = None) -> ViewTab | None:
    dialog = PanelPickerDialog(tab, parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.result_tab()


def quick_add_entries() -> Sequence[tuple[str, str, str]]:
    """``(group, key, label)`` triples for a compact “add panel” menu."""
    return tuple((info.group, info.key, info.label) for info in PANEL_CATALOG)
