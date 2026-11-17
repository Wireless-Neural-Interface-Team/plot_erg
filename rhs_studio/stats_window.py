"""Statistiques des blocs tracées avec seaborn (matplotlib intégré à Qt)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")

import matplotlib

matplotlib.use("QtAgg")
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .config import SEABORN_PALETTES

# (clé, libellé, table, colonne, nature, unité de l'axe)
ANALYSES = [
    ("rms", "RMS post-trigger par essai", "trials", "rms", "cat", "RMS"),
    ("min", "Pic minimum (onde a)", "trials", "min", "cat", "Amplitude"),
    ("max", "Pic maximum (onde b)", "trials", "max", "cat", "Amplitude"),
    ("p2p", "Amplitude crête-à-crête (b − a)", "trials", "crête_à_crête", "cat", "Amplitude"),
    ("t_min", "Latence du minimum", "trials", "t_min", "cat", "Latence"),
    ("t_max", "Latence du maximum", "trials", "t_max", "cat", "Latence"),
    ("evo", "Évolution du pic max au fil des essais", "trials", "max", "line", "Amplitude"),
    ("spk", "Spikes par essai", "spike_counts", "spikes", "cat", "Spikes"),
    ("isi", "Intervalles inter-spikes (ISI)", "isi", "isi_ms", "dist", "ISI (ms)"),
    ("amp", "Amplitude des spikes", "spike_amp", "amplitude", "dist", "Amplitude (µV)"),
    ("dur", "Durée des triggers", "trig_dur", "durée_ms", "dist", "Durée (ms)"),
]
CAT_KINDS = [
    ("box", "Boîte + points"), ("violin", "Violon"), ("strip", "Points"),
    ("swarm", "Essaim"), ("bar", "Barres (moyenne ± IC)"), ("point", "Moyenne ± IC"),
]
DIST_KINDS = [("hist", "Histogramme"), ("kde", "Densité (KDE)"), ("ecdf", "ECDF")]


class StatsDialog(QDialog):
    def __init__(self, get_blocks, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Statistiques (seaborn)")
        self.resize(1000, 680)
        self._get_blocks = get_blocks

        self.block_combo = QComboBox()
        self.analysis_combo = QComboBox()
        for key, label, *_ in ANALYSES:
            self.analysis_combo.addItem(label, key)
        self.kind_combo = QComboBox()
        self.style_combo = QComboBox()
        self.style_combo.addItems(["whitegrid", "darkgrid", "white", "ticks", "dark"])
        self.context_combo = QComboBox()
        self.context_combo.addItems(["notebook", "paper", "talk", "poster"])
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(SEABORN_PALETTES)
        self.w_spin = QDoubleSpinBox()
        self.w_spin.setRange(2, 30)
        self.w_spin.setValue(8)
        self.w_spin.setSuffix(" in")
        self.h_spin = QDoubleSpinBox()
        self.h_spin.setRange(2, 30)
        self.h_spin.setValue(5)
        self.h_spin.setSuffix(" in")
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 1200)
        self.dpi_spin.setValue(150)

        form = QFormLayout()
        form.addRow("Bloc", self.block_combo)
        form.addRow("Analyse", self.analysis_combo)
        form.addRow("Représentation", self.kind_combo)
        form.addRow("Style seaborn", self.style_combo)
        form.addRow("Contexte", self.context_combo)
        form.addRow("Palette", self.palette_combo)
        form.addRow("Largeur", self.w_spin)
        form.addRow("Hauteur", self.h_spin)
        form.addRow("Résolution export", self.dpi_spin)
        self.message = QLabel()
        self.message.setWordWrap(True)
        refresh = QPushButton("Actualiser les blocs")
        export = QPushButton("Exporter la figure…")
        csv = QPushButton("Exporter les données (CSV)…")
        refresh.clicked.connect(self.refresh_blocks)
        export.clicked.connect(self._export)
        csv.clicked.connect(self._export_csv)

        side = QVBoxLayout()
        side.addLayout(form)
        side.addWidget(refresh)
        side.addWidget(export)
        side.addWidget(csv)
        side.addWidget(self.message)
        side.addStretch(1)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(300)

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        right = QVBoxLayout()
        right.addWidget(NavigationToolbar2QT(self.canvas, self))
        right.addWidget(self.canvas, 1)

        root = QHBoxLayout(self)
        root.addWidget(side_w)
        root.addLayout(right, 1)

        self.analysis_combo.currentIndexChanged.connect(self._analysis_changed)
        for w in (self.block_combo, self.kind_combo, self.style_combo, self.context_combo, self.palette_combo):
            w.currentIndexChanged.connect(self.redraw)
        self.refresh_blocks()
        self._analysis_changed()

    def refresh_blocks(self) -> None:
        current = self.block_combo.currentData()
        self.block_combo.blockSignals(True)
        self.block_combo.clear()
        for cfg, _data in self._get_blocks():
            self.block_combo.addItem(cfg.title, cfg.id)
        i = self.block_combo.findData(current)
        self.block_combo.setCurrentIndex(max(0, i))
        self.block_combo.blockSignals(False)
        self.redraw()

    def select_block(self, block_id: str) -> None:
        i = self.block_combo.findData(block_id)
        if i >= 0:
            self.block_combo.setCurrentIndex(i)

    def _analysis(self):
        key = self.analysis_combo.currentData()
        return next(a for a in ANALYSES if a[0] == key)

    def _analysis_changed(self) -> None:
        nature = self._analysis()[4]
        kinds = DIST_KINDS if nature == "dist" else CAT_KINDS if nature == "cat" else [("line", "Ligne")]
        self.kind_combo.blockSignals(True)
        self.kind_combo.clear()
        for value, label in kinds:
            self.kind_combo.addItem(label, value)
        self.kind_combo.blockSignals(False)
        self.redraw()

    def _frame(self) -> tuple[pd.DataFrame | None, str]:
        block_id = self.block_combo.currentData()
        entry = next(((c, d) for c, d in self._get_blocks() if c.id == block_id), None)
        if entry is None or entry[1] is None:
            return None, "Ce bloc n'a pas encore de données calculées."
        cfg, data = entry
        _, label, table, col, *_ = self._analysis()
        rows = data.stats.get(table)
        if not rows:
            hint = {
                "trials": "un bloc WIDE / LOW / HIGH / RMS aligné sur trigger",
                "spike_counts": "un bloc SPIKES ou SPIKE SCOPE aligné sur trigger",
                "isi": "un bloc SPIKES ou SPIKE SCOPE",
                "spike_amp": "un bloc SPIKES ou SPIKE SCOPE",
                "trig_dur": "un trigger actif",
            }[table]
            return None, f"Pas de données pour « {label} » : il faut {hint}."
        return pd.DataFrame(rows), ""

    def redraw(self) -> None:
        sns.set_theme(
            style=self.style_combo.currentText(),
            context=self.context_combo.currentText(),
            palette=self.palette_combo.currentText(),
        )
        self.figure.clear()
        self.figure.set_size_inches(self.w_spin.value(), self.h_spin.value())
        ax = self.figure.add_subplot(111)
        df, msg = self._frame()
        self.message.setText(msg)
        if df is None:
            ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True, transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        _, label, _, col, nature, unit = self._analysis()
        kind = self.kind_combo.currentData()
        pal = self.palette_combo.currentText()
        has_canal = "canal" in df.columns
        n_hue = df["canal"].nunique() if has_canal else 1
        try:
            if nature == "dist":
                hue = "canal" if has_canal and n_hue > 1 else None
                if kind == "kde":
                    sns.kdeplot(df, x=col, hue=hue, fill=True, common_norm=False, palette=pal if hue else None, ax=ax)
                elif kind == "ecdf":
                    sns.ecdfplot(df, x=col, hue=hue, palette=pal if hue else None, ax=ax)
                else:
                    sns.histplot(df, x=col, hue=hue, kde=True, element="step", palette=pal if hue else None, ax=ax)
                ax.set_xlabel(unit)
            elif nature == "line":
                sns.lineplot(df, x="essai", y=col, hue="canal", marker="o", palette=pal, ax=ax)
                ax.set_xlabel("Essai")
                ax.set_ylabel(unit)
            else:
                common = dict(data=df, x="canal", y=col, hue="canal", palette=pal, ax=ax, legend=False)
                if kind == "violin":
                    sns.violinplot(**common, inner="quart")
                elif kind == "strip":
                    sns.stripplot(**common, alpha=0.7)
                elif kind == "swarm":
                    sns.swarmplot(**common, size=3)
                elif kind == "bar":
                    sns.barplot(**common, errorbar=("ci", 95))
                elif kind == "point":
                    sns.pointplot(**common, errorbar=("ci", 95))
                else:
                    sns.boxplot(**common, fliersize=0)
                    sns.stripplot(data=df, x="canal", y=col, color="black", size=2.5, alpha=0.5, ax=ax)
                ax.set_xlabel("Canal")
                ax.set_ylabel(unit)
                if n_hue > 6:
                    ax.tick_params(axis="x", rotation=45)
            ax.set_title(f"{self.block_combo.currentText()} — {label}")
            self.figure.tight_layout()
        except Exception as exc:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Erreur seaborn : {exc}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        self.canvas.draw_idle()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter la figure", "statistiques.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if path:
            self.figure.savefig(path, dpi=self.dpi_spin.value(), bbox_inches="tight")

    def _export_csv(self) -> None:
        df, msg = self._frame()
        if df is None:
            self.message.setText(msg)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporter les données", "statistiques.csv", "CSV (*.csv)")
        if path:
            df.to_csv(path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
