"""Liste des enregistrements à comparer : fichiers, légende, couleur, statut."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from display_config import RECORDING_COLOR_PRESETS, RecordingStyle
from erg_cache import DATASET_SUFFIX

_COL_NAME = 0
_COL_LABEL = 1
_COL_COLOR = 2
_COL_PLOT = 3
_COL_STATUS = 4

_STATUS_LABELS = {
    "queued": "en attente",
    "building": "en cours",
    "ready": "prêt",
    "failed": "échec",
}

_STATUS_COLORS = {
    "queued": "#64748b",
    "building": "#2563eb",
    "ready": "#16a34a",
    "failed": "#dc2626",
}


@dataclass
class RecordingEntry:
    """Une ligne : fichier source (``.rhs`` ou dataset traité) et son style."""

    row_id: int
    path: Path
    label: str = ""
    style: RecordingStyle = RecordingStyle()
    status: str = "queued"
    detail: str = ""
    recording: Any | None = None
    report: Any | None = None

    @property
    def is_processed(self) -> bool:
        suffix = self.path.suffix.lower()
        return suffix in {DATASET_SUFFIX, ".zip"} or self.path.is_dir()

    @property
    def display_label(self) -> str:
        return self.label.strip() or self.path.stem

    @property
    def is_ready(self) -> bool:
        return self.status == "ready" and self.recording is not None


class RecordingsPanel(QWidget):
    """Liste des enregistrements : ajouter / traiter / légende / statut."""

    entriesChanged = Signal()
    styleChanged = Signal()
    processRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[RecordingEntry] = []
        self._next_id = 1
        self._updating = False
        self._last_dir = str(Path.home())

        self.table = QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(
            ["Fichier", "Légende", "Couleur", "Afficher", "Statut"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setToolTip(
            "Chaque ligne est un enregistrement. Couleur et légende s’appliquent "
            "immédiatement ; le statut indique si le traitement est prêt."
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(_COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_LABEL, QHeaderView.ResizeMode.Stretch)
        for column in (_COL_COLOR, _COL_PLOT, _COL_STATUS):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

        self._btn_add = QPushButton("Ajouter Intan .rhs…", self)
        self._btn_add.setToolTip("Ajouter un ou plusieurs fichiers Intan .rhs (Ctrl+O).")
        self._btn_add.clicked.connect(self.browse_rhs)
        self._btn_open_processed = QPushButton("Ouvrir traité…", self)
        self._btn_open_processed.setToolTip(
            "Ouvrir un dataset déjà exporté (.ergdataset / .zip) pour le comparer."
        )
        self._btn_open_processed.clicked.connect(self.browse_processed)
        self._btn_process = QPushButton("Traiter", self)
        self._btn_process.setObjectName("primaryButton")
        self._btn_process.setToolTip("Traiter les enregistrements en attente (F5).")
        self._btn_process.clicked.connect(self.processRequested.emit)

        remove = QPushButton("Retirer", self)
        remove.setObjectName("dangerButton")
        remove.setToolTip("Retirer les lignes sélectionnées.")
        remove.clicked.connect(self.remove_selected)

        self._summary = QLabel("Aucun enregistrement.")
        self._summary.setObjectName("hintLabel")
        self._summary.setWordWrap(True)

        top = QHBoxLayout()
        top.setSpacing(4)
        top.addWidget(self._btn_add)
        top.addWidget(self._btn_open_processed)
        top.addWidget(self._btn_process)
        top.addStretch(1)
        top.addWidget(remove)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addWidget(self.table, 1)
        layout.addWidget(self._summary)

    # ------------------------------------------------------------------- access

    @property
    def entries(self) -> list[RecordingEntry]:
        return list(self._entries)

    def ready_entries(self) -> list[RecordingEntry]:
        return [e for e in self._entries if e.is_ready]

    def plotted_entries(self) -> list[RecordingEntry]:
        return [e for e in self.ready_entries() if e.style.plot_visible]

    def pending_entries(self) -> list[RecordingEntry]:
        return [e for e in self._entries if e.status in {"queued", "failed"}]

    def entry(self, row_id: int) -> RecordingEntry | None:
        for item in self._entries:
            if item.row_id == row_id:
                return item
        return None

    def first_ready(self) -> RecordingEntry | None:
        ready = self.ready_entries()
        return ready[0] if ready else None

    # -------------------------------------------------------------- mutation

    def add_paths(self, paths: Iterable[Path]) -> int:
        existing = {entry.path.resolve() for entry in self._entries}
        added = 0
        for path in paths:
            resolved = Path(path).resolve()
            if resolved in existing:
                continue
            existing.add(resolved)
            self._entries.append(RecordingEntry(row_id=self._next_id, path=Path(path)))
            self._next_id += 1
            added += 1
        if added:
            self._rebuild_table()
            self.entriesChanged.emit()
        return added

    def browse_rhs(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Sélectionner des enregistrements Intan .rhs",
            self._last_dir,
            "Intan RHS (*.rhs);;Tous les fichiers (*)",
        )
        if paths:
            self._last_dir = str(Path(paths[0]).parent)
            self.add_paths(Path(p) for p in paths)

    def browse_processed(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir une archive de dataset traité",
            self._last_dir,
            "Dataset traité (*.zip);;Tous les fichiers (*)",
        )
        if not path:
            directory = QFileDialog.getExistingDirectory(
                self, "Ouvrir un dossier de dataset traité", self._last_dir
            )
            path = directory
        if path:
            self._last_dir = str(Path(path).parent)
            self.add_paths([Path(path)])

    def remove_selected(self) -> None:
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        if not rows:
            return
        keep: list[RecordingEntry] = []
        for index, entry in enumerate(self._entries):
            if index in rows:
                recording = entry.recording
                if recording is not None:
                    try:
                        recording.close()
                    except Exception:
                        pass
            else:
                keep.append(entry)
        self._entries = keep
        self._rebuild_table()
        self.entriesChanged.emit()

    def clear(self) -> None:
        for entry in self._entries:
            if entry.recording is not None:
                try:
                    entry.recording.close()
                except Exception:
                    pass
        self._entries.clear()
        self._rebuild_table()
        self.entriesChanged.emit()

    def set_status(self, row_id: int, status: str, detail: str = "") -> None:
        entry = self.entry(row_id)
        if entry is None:
            return
        entry.status = status
        entry.detail = detail
        self._refresh_status_cell(entry)
        self._refresh_summary()

    def set_result(self, row_id: int, recording: Any, report: Any) -> None:
        entry = self.entry(row_id)
        if entry is None:
            return
        if entry.recording is not None and entry.recording is not recording:
            try:
                entry.recording.close()
            except Exception:
                pass
        entry.recording = recording
        entry.report = report
        entry.status = "ready"
        total = getattr(report, "total_s", None)
        entry.detail = f"{total:.1f} s" if total is not None else "prêt"
        if not entry.label:
            label = getattr(recording, "label", "")
            if label and label != entry.path.stem:
                entry.label = str(label)
                self._refresh_label_cell(entry)
        self._refresh_status_cell(entry)
        self._refresh_summary()

    def invalidate_results(self) -> None:
        """Marquer chaque ligne comme à retraiter (paramètre de calcul modifié)."""
        for entry in self._entries:
            if entry.recording is not None:
                try:
                    entry.recording.close()
                except Exception:
                    pass
            entry.recording = None
            entry.report = None
            entry.status = "queued"
            entry.detail = "paramètres modifiés"
            self._refresh_status_cell(entry)
        self._refresh_summary()

    def set_busy(self, busy: bool) -> None:
        """Désactiver Ajouter / Traiter pendant un build."""
        enabled = not busy
        self._btn_add.setEnabled(enabled)
        self._btn_open_processed.setEnabled(enabled)
        self._btn_process.setEnabled(enabled)

    # ----------------------------------------------------------------- rendering

    def _rebuild_table(self) -> None:
        self._updating = True
        self.table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            name_item = QTableWidgetItem(entry.path.name)
            name_item.setToolTip(str(entry.path))
            if entry.is_processed:
                name_item.setText(f"{entry.path.name}  (traité)")
            self.table.setItem(row, _COL_NAME, name_item)

            label_edit = QLineEdit(entry.label)
            label_edit.setPlaceholderText(entry.path.stem)
            label_edit.setToolTip("Texte de légende pour cet enregistrement.")
            label_edit.textChanged.connect(
                lambda text, rid=entry.row_id: self._on_label_changed(rid, text)
            )
            self.table.setCellWidget(row, _COL_LABEL, label_edit)

            color_box = QComboBox()
            for name, value in RECORDING_COLOR_PRESETS:
                color_box.addItem(name, value)
            current = (entry.style.color or "").strip()
            index = color_box.findData(current)
            color_box.setCurrentIndex(index if index >= 0 else 0)
            color_box.currentIndexChanged.connect(
                lambda _i, rid=entry.row_id, box=color_box: self._on_color_changed(
                    rid, str(box.currentData() or "")
                )
            )
            self.table.setCellWidget(row, _COL_COLOR, color_box)

            # Afficher = tracer + légende (même bascule pour alléger la table).
            plot_box = self._centered_check(
                entry.style.plot_visible,
                "Afficher / masquer les courbes de cet enregistrement.",
            )
            plot_box.toggled.connect(
                lambda checked, rid=entry.row_id: self._on_plot_toggled(rid, checked)
            )
            self.table.setCellWidget(row, _COL_PLOT, self._wrap_check(plot_box))

            self.table.setItem(row, _COL_STATUS, QTableWidgetItem(""))
            self._refresh_status_cell(entry)
        self._updating = False
        self._refresh_summary()

    @staticmethod
    def _centered_check(checked: bool, tooltip: str) -> QCheckBox:
        box = QCheckBox()
        box.setChecked(bool(checked))
        box.setToolTip(tooltip)
        return box

    @staticmethod
    def _wrap_check(box: QCheckBox) -> QWidget:
        """Centrer la case ; un clic dans la marge de la cellule bascule aussi."""
        holder = QWidget()
        holder.setToolTip(box.toolTip())
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)
        layout.addWidget(box)
        layout.addStretch(1)

        def _on_press(event: Any) -> None:
            # Les clics sur la case vont au QCheckBox ; ici = marge autour.
            box.toggle()
            QWidget.mousePressEvent(holder, event)

        holder.mousePressEvent = _on_press  # type: ignore[method-assign]
        return holder

    def _row_of(self, entry: RecordingEntry) -> int:
        try:
            return self._entries.index(entry)
        except ValueError:
            return -1

    def _refresh_status_cell(self, entry: RecordingEntry) -> None:
        row = self._row_of(entry)
        if row < 0:
            return
        item = self.table.item(row, _COL_STATUS)
        if item is None:
            return
        status_fr = _STATUS_LABELS.get(entry.status, entry.status)
        text = status_fr if not entry.detail else f"{status_fr} — {entry.detail}"
        item.setText(text)
        item.setToolTip(text)
        from PySide6.QtGui import QColor

        item.setForeground(QColor(_STATUS_COLORS.get(entry.status, "#334155")))

    def _refresh_label_cell(self, entry: RecordingEntry) -> None:
        row = self._row_of(entry)
        if row < 0:
            return
        widget = self.table.cellWidget(row, _COL_LABEL)
        if isinstance(widget, QLineEdit) and widget.text() != entry.label:
            self._updating = True
            widget.setText(entry.label)
            self._updating = False

    def _refresh_summary(self) -> None:
        total = len(self._entries)
        ready = len(self.ready_entries())
        pending = len(self.pending_entries())
        if total == 0:
            self._summary.setText("Ajoutez un .rhs, puis Traiter (F5).")
            return
        parts = [f"{total} fichier(s)", f"{ready} prêt(s)"]
        if pending:
            parts.append(f"{pending} à traiter")
        self._summary.setText(" · ".join(parts))

    # ------------------------------------------------------------------- slots

    def _on_label_changed(self, row_id: int, text: str) -> None:
        if self._updating:
            return
        entry = self.entry(row_id)
        if entry is None:
            return
        entry.label = text
        if entry.recording is not None:
            try:
                entry.recording.label = entry.display_label
            except Exception:
                pass
        self.styleChanged.emit()

    def _on_color_changed(self, row_id: int, value: str) -> None:
        if self._updating:
            return
        entry = self.entry(row_id)
        if entry is None:
            return
        entry.style = replace(entry.style, color=value or None)
        self.styleChanged.emit()

    def _on_plot_toggled(self, row_id: int, checked: bool) -> None:
        if self._updating:
            return
        entry = self.entry(row_id)
        if entry is None:
            return
        # Une seule bascule : tracer + légende ensemble.
        entry.style = replace(
            entry.style, plot_visible=bool(checked), legend_visible=bool(checked)
        )
        self.styleChanged.emit()

    def _on_legend_toggled(self, row_id: int, checked: bool) -> None:
        if self._updating:
            return
        entry = self.entry(row_id)
        if entry is None:
            return
        entry.style = replace(entry.style, legend_visible=bool(checked))
        self.styleChanged.emit()

    # --------------------------------------------------------------- utilities

    def common_channel_names(self) -> list[str]:
        """Noms de canaux partagés par tous les enregistrements tracés."""
        plotted = self.plotted_entries()
        if not plotted:
            return []
        names: Sequence[str] = plotted[0].recording.channel_names
        shared = list(names)
        for entry in plotted[1:]:
            other = set(entry.recording.channel_names)
            shared = [name for name in shared if name in other]
        return shared
