"""Petits dialogues : export de dataset traité et gestion du cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from erg_cache import (
    DEFAULT_CACHE_MAX_BYTES,
    cache_size_bytes,
    clear_cache,
    describe_cache,
    human_bytes,
    prune_cache,
)


@dataclass(frozen=True)
class ExportOptions:
    """Contenu à écrire dans un dataset traité exporté."""

    directory: Path
    include_trigger_windows: bool = True
    include_overlay: bool = True
    include_streams: bool = True
    make_archive: bool = False


class ExportDatasetDialog(QDialog):
    """Choisir où et comment écrire le dataset traité réutilisable."""

    def __init__(self, default_directory: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Exporter un dataset traité")
        self.setMinimumWidth(560)

        self._directory = QLineEdit(str(default_directory))
        browse = QPushButton("Parcourir…")
        browse.clicked.connect(self._browse)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        row_layout.addWidget(self._directory, 1)
        row_layout.addWidget(browse)

        self._trigger_windows = QCheckBox(
            "Fenêtres par stimulation (nécessaires aux panneaux 1re / 2e stim)"
        )
        self._trigger_windows.setChecked(True)
        self._overlay = QCheckBox(
            "Snippets de formes d’onde de spikes (nécessaires à la superposition)"
        )
        self._overlay.setChecked(True)
        self._streams = QCheckBox(
            "Flux complets brut / HP / LP (requis pour export PDF sans le .rhs ; volumineux)"
        )
        self._streams.setChecked(True)
        self._streams.setToolTip(
            "Inclure les memmaps ampli / passe-haut / passe-bas. "
            "Nécessaire pour générer un rapport PDF depuis le dataset traité seul."
        )
        self._archive = QCheckBox("Créer aussi une archive .zip pour le transport")
        self._archive.setChecked(False)

        form = QFormLayout()
        form.addRow("Dossier de destination :", row)

        hint = QLabel(
            "Le dataset exporté se rouvre sans le fichier .rhs d’origine : moyennes "
            "d’essais, profils RMS, trains de spikes, seuils et métadonnées sont inclus. "
            "Cochez les flux complets si vous voulez aussi exporter un PDF hors machine source."
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._trigger_windows)
        layout.addWidget(self._overlay)
        layout.addWidget(self._streams)
        layout.addWidget(self._archive)
        layout.addWidget(hint)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        start = self._directory.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, "Choisir le dossier de destination", start
        )
        if path:
            self._directory.setText(path)

    def options(self) -> ExportOptions:
        return ExportOptions(
            directory=Path(self._directory.text().strip() or str(Path.home())),
            include_trigger_windows=self._trigger_windows.isChecked(),
            include_overlay=self._overlay.isChecked(),
            include_streams=self._streams.isChecked(),
            make_archive=self._archive.isChecked(),
        )


class CacheDialog(QDialog):
    """Inspecter le cache de données traitées et libérer de l’espace disque."""

    def __init__(self, cache_root: Path, protected: set[Path], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cache des données traitées")
        self.resize(780, 480)
        self._root = Path(cache_root)
        self._protected = {Path(p) for p in protected}

        self._summary = QLabel("")
        self._summary.setWordWrap(True)

        self.table = QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(
            ["Enregistrement", "Créé", "Taille", "Canaux", "Stimulations"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

        self._limit_gb = QDoubleSpinBox(self)
        self._limit_gb.setRange(0.5, 500.0)
        self._limit_gb.setDecimals(1)
        self._limit_gb.setSingleStep(1.0)
        self._limit_gb.setSuffix(" Go")
        self._limit_gb.setValue(DEFAULT_CACHE_MAX_BYTES / (1024**3))
        self._limit_gb.setToolTip(
            "Budget disque cible. L’élagage retire d’abord les bundles / filtres / "
            "flux bruts les plus anciens (datasets ouverts protégés)."
        )

        prune = QPushButton("Élaguer au budget")
        prune.setToolTip(
            "Supprimer les entrées les plus anciennes (bundles, filtres HP/LP, flux bruts) "
            "jusqu’à tenir dans le budget choisi."
        )
        prune.clicked.connect(self._prune)
        clear = QPushButton("Vider le cache")
        clear.setObjectName("dangerButton")
        clear.setToolTip(
            "Supprimer tous les datasets en cache sauf ceux actuellement ouverts."
        )
        clear.clicked.connect(self._clear)
        close = QPushButton("Fermer")
        close.clicked.connect(self.accept)

        budget_row = QHBoxLayout()
        budget_row.addWidget(QLabel("Budget :"))
        budget_row.addWidget(self._limit_gb)
        budget_row.addStretch(1)

        buttons = QHBoxLayout()
        buttons.addWidget(prune)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        buttons.addWidget(close)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Dossier de cache : {self._root}"))
        layout.addWidget(self.table, 1)
        layout.addWidget(self._summary)
        layout.addLayout(budget_row)
        layout.addLayout(buttons)

        self.refresh()

    def _budget_bytes(self) -> int:
        return int(float(self._limit_gb.value()) * (1024**3))

    def refresh(self) -> None:
        entries = describe_cache(self._root)
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            cells = (
                entry.source_name,
                entry.created_at.replace("T", " ")[:19],
                human_bytes(entry.size_bytes),
                str(entry.n_channels or "—"),
                str(entry.n_trials or "—"),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setToolTip(str(entry.root))
                self.table.setItem(row, column, item)
        total = cache_size_bytes(self._root)
        limit = self._budget_bytes()
        self._summary.setText(
            f"{len(entries)} dataset(s) en cache · {human_bytes(total)} sur disque "
            f"(budget {human_bytes(limit)} ; bundles + filtres + flux bruts)."
        )

    def _prune(self) -> None:
        total = cache_size_bytes(self._root)
        limit = self._budget_bytes()
        if total <= limit:
            QMessageBox.information(
                self,
                "Cache",
                f"Le cache fait déjà {human_bytes(total)} "
                f"(≤ {human_bytes(limit)}). Rien à supprimer.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Élaguer le cache",
            f"Cache actuel : {human_bytes(total)}.\n"
            f"Supprimer les entrées les plus anciennes jusqu’à ≈ {human_bytes(limit)} ?\n"
            f"(Les datasets ouverts restent protégés ; les flux bruts orphelins sont inclus.)",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        removed = prune_cache(
            self._root, max_bytes=limit, keep=self._protected, include_raw=True
        )
        self._summary.setText(f"{removed} entrée(s) de cache supprimée(s).")
        self.refresh()

    def _clear(self) -> None:
        removed = clear_cache(self._root, keep=self._protected, include_raw=True)
        self._summary.setText(f"{removed} dataset(s) supprimé(s).")
        self.refresh()
