"""Panneau de contrôle bas — disposition type Intan RHX Control Panel."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from view_config import AnalysisSettings


class ControlPanel(QFrame):
    """Filtres WIDE/LOW/HIGH, canal actif, ouverture des outils d’analyse."""

    filterChanged = Signal()
    showAnalysisRequested = Signal()
    openSpikesRequested = Signal()
    openGraphRequested = Signal()
    processRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("controlPanel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._n_trials = 0
        self._channel = ""
        self._updating = False

        self._channel_label = QLabel("—", self)
        self._channel_label.setObjectName("controlChannel")
        self._channel_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._channel_label.setToolTip("Canal actuellement sélectionné")

        self._btn_wide = QPushButton("WIDE", self)
        self._btn_wide.setObjectName("filterWide")
        self._btn_wide.setCheckable(True)
        self._btn_wide.setChecked(True)
        self._btn_wide.setToolTip("Brut (wideband) — signal complet")

        self._btn_low = QPushButton("LOW", self)
        self._btn_low.setObjectName("filterLow")
        self._btn_low.setCheckable(True)
        self._btn_low.setToolTip("Passe-bas — LFP / composantes lentes")

        self._btn_high = QPushButton("HIGH", self)
        self._btn_high.setObjectName("filterHigh")
        self._btn_high.setCheckable(True)
        self._btn_high.setToolTip("Passe-haut — spikes / composantes rapides")

        self._filter_group = QButtonGroup(self)
        self._filter_group.setExclusive(True)
        for button in (self._btn_wide, self._btn_low, self._btn_high):
            self._filter_group.addButton(button)

        self._mark_stims = QCheckBox("Stim markers", self)
        self._mark_stims.setChecked(True)
        self._mark_stims.setToolTip("Marquer les stimulations sur le montage")

        self._show_raw = QCheckBox("WIDE", self)
        self._show_hp = QCheckBox("HIGH", self)
        self._show_lp = QCheckBox("LOW", self)
        self._show_rms = QCheckBox("RMS", self)
        for box in (self._show_raw, self._show_hp, self._show_lp, self._show_rms):
            box.setChecked(True)

        self._btn_analysis = QPushButton("Analyse", self)
        self._btn_analysis.setObjectName("primaryButton")
        self._btn_analysis.setToolTip("Fenêtre d’analyse (traces cochées)")
        self._btn_spikes = QPushButton("Spike Scope", self)
        self._btn_spikes.setObjectName("secondaryButton")
        self._btn_spikes.setToolTip("Raster · PSTH · ISI · superposition")
        self._btn_graph = QPushButton("Graph…", self)
        self._btn_graph.setToolTip("Choisir un graphique pour le canal (Ctrl+G)")
        self._btn_process = QPushButton("Traiter", self)
        self._btn_process.setObjectName("primaryButton")
        self._btn_process.setToolTip("Traiter / actualiser (F5)")

        for button in (self._btn_analysis, self._btn_spikes, self._btn_graph):
            button.setEnabled(False)

        # Rangée 1 : identité + filtre d’affichage montage
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("Channel", self))
        row1.addWidget(self._channel_label)
        row1.addSpacing(12)
        row1.addWidget(QLabel("Filter display", self))
        for button in (self._btn_wide, self._btn_low, self._btn_high):
            row1.addWidget(button)
        row1.addSpacing(8)
        row1.addWidget(self._mark_stims)
        row1.addStretch(1)
        row1.addWidget(self._btn_process)

        # Rangée 2 : outils d’analyse
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(QLabel("Open", self))
        for box in (self._show_raw, self._show_hp, self._show_lp, self._show_rms):
            row2.addWidget(box)
        row2.addSpacing(10)
        row2.addWidget(self._btn_analysis)
        row2.addWidget(self._btn_spikes)
        row2.addWidget(self._btn_graph)
        row2.addStretch(1)
        self._hint = QLabel("Ajoutez un .rhs puis Traiter.", self)
        self._hint.setObjectName("hintLabel")
        row2.addWidget(self._hint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(row1)
        layout.addLayout(row2)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(86)

        self._filter_group.buttonClicked.connect(self._emit_filter)
        self._mark_stims.toggled.connect(self._emit_filter)
        for box in (self._show_raw, self._show_hp, self._show_lp, self._show_rms):
            box.toggled.connect(self._emit_filter)
        self._btn_analysis.clicked.connect(self.showAnalysisRequested.emit)
        self._btn_spikes.clicked.connect(self.openSpikesRequested.emit)
        self._btn_graph.clicked.connect(self.openGraphRequested.emit)
        self._btn_process.clicked.connect(self.processRequested.emit)

    # ---------------------------------------------------------------- values

    def continuous_stream(self) -> str:
        if self._btn_high.isChecked():
            return "hp"
        if self._btn_low.isChecked():
            return "lp"
        return "raw"

    def mark_stimulations(self) -> bool:
        return bool(self._mark_stims.isChecked())

    def analysis_settings(self) -> AnalysisSettings:
        return AnalysisSettings(
            mode="average",
            stim_index=0,
            show_raw=self._show_raw.isChecked(),
            show_hp=self._show_hp.isChecked(),
            show_lp=self._show_lp.isChecked(),
            show_rms=self._show_rms.isChecked(),
        )

    def set_channel(self, channel: str | None) -> None:
        self._channel = (channel or "").strip()
        self._channel_label.setText(self._channel or "—")
        self._refresh_ready()

    def set_trial_count(self, n_trials: int) -> None:
        self._n_trials = max(0, int(n_trials))
        self._refresh_ready()

    def set_busy(self, busy: bool) -> None:
        self._btn_process.setEnabled(not busy)
        self._btn_process.setText("…" if busy else "Traiter")

    def _refresh_ready(self) -> None:
        has_channel = bool(self._channel)
        ready = self._n_trials > 0
        self._btn_analysis.setEnabled(ready and has_channel)
        self._btn_spikes.setEnabled(ready and has_channel)
        self._btn_graph.setEnabled(has_channel)
        if ready and has_channel:
            self._hint.setText(f"{self._n_trials} stim(s) · {self._channel}")
        elif has_channel:
            self._hint.setText(f"Canal {self._channel}")
        else:
            self._hint.setText("Ajoutez un .rhs puis Traiter.")

    def _emit_filter(self, *_args: Any) -> None:
        if self._updating:
            return
        self.filterChanged.emit()
