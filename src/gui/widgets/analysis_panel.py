"""Explorer : ouvrir des vues ; les paramètres fins vivent dans chaque fenêtre."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from view_config import AnalysisSettings


class AnalysisPanel(QWidget):
    """Lancer des fenêtres d’analyse / spikes ; réglages du montage central."""

    analysisChanged = Signal()
    showAnalysisRequested = Signal()
    openSpikesRequested = Signal()
    openGraphRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._n_trials = 0
        self._channel: str = ""
        self._updating = False

        self._continuous_stream = QComboBox(self)
        self._continuous_stream.addItem("Brut", "raw")
        self._continuous_stream.addItem("Passe-haut", "hp")
        self._continuous_stream.addItem("Passe-bas", "lp")
        self._mark_stims = QCheckBox("Marquer les stimulations", self)
        self._mark_stims.setChecked(True)

        self._show_raw = QCheckBox("Brut", self)
        self._show_hp = QCheckBox("HP", self)
        self._show_lp = QCheckBox("LP", self)
        self._show_rms = QCheckBox("RMS", self)
        for box in (self._show_raw, self._show_hp, self._show_lp, self._show_rms):
            box.setChecked(True)

        self._btn_analysis = QPushButton("Analyse…", self)
        self._btn_analysis.setObjectName("primaryButton")
        self._btn_analysis.setToolTip(
            "Ouvre une fenêtre avec les graphiques cochés. "
            "Mode, stimulation et zooms se règlent dans la fenêtre."
        )
        self._btn_spikes = QPushButton("Spikes…", self)
        self._btn_spikes.setObjectName("secondaryButton")
        self._btn_spikes.setToolTip(
            "Raster, PSTH, ISI et superposition — paramètres dans la fenêtre."
        )
        self._btn_graph = QPushButton("Graphique canal…", self)
        self._btn_graph.setToolTip("Choisir un type de graphique pour le canal actif (Ctrl+G).")

        for button in (self._btn_analysis, self._btn_spikes, self._btn_graph):
            button.setEnabled(False)

        self._channel_label = QLabel("Canal : —", self)
        self._channel_label.setObjectName("sectionTitle")
        self._ready_hint = QLabel("Traitez un enregistrement pour ouvrir des vues.", self)
        self._ready_hint.setObjectName("hintLabel")
        self._ready_hint.setWordWrap(True)

        montage_box = QGroupBox("Montage central", self)
        montage_form = QFormLayout(montage_box)
        montage_form.setContentsMargins(8, 12, 8, 8)
        montage_form.addRow("Flux", self._continuous_stream)
        montage_form.addRow(self._mark_stims)

        open_box = QGroupBox("Ouvrir", self)
        open_layout = QVBoxLayout(open_box)
        open_layout.setContentsMargins(8, 12, 8, 8)
        open_layout.setSpacing(8)
        open_layout.addWidget(self._channel_label)
        open_layout.addWidget(self._ready_hint)

        streams_row = QHBoxLayout()
        streams_row.setSpacing(6)
        for box in (self._show_raw, self._show_hp, self._show_lp, self._show_rms):
            streams_row.addWidget(box)
        streams_row.addStretch(1)
        open_layout.addLayout(streams_row)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(6)
        buttons_row.addWidget(self._btn_analysis, 1)
        buttons_row.addWidget(self._btn_spikes, 1)
        open_layout.addLayout(buttons_row)
        open_layout.addWidget(self._btn_graph)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(montage_box)
        layout.addWidget(open_box)
        layout.addStretch(1)

        self._continuous_stream.currentIndexChanged.connect(self._emit_changed)
        self._mark_stims.toggled.connect(self._emit_changed)
        for box in (self._show_raw, self._show_hp, self._show_lp, self._show_rms):
            box.toggled.connect(self._emit_changed)

        self._btn_analysis.clicked.connect(self.showAnalysisRequested.emit)
        self._btn_spikes.clicked.connect(self.openSpikesRequested.emit)
        self._btn_graph.clicked.connect(self.openGraphRequested.emit)

    def analysis_settings(self) -> AnalysisSettings:
        """Valeurs initiales pour une nouvelle fenêtre (moyenne d’essais)."""
        return AnalysisSettings(
            mode="average",
            stim_index=0,
            show_raw=self._show_raw.isChecked(),
            show_hp=self._show_hp.isChecked(),
            show_lp=self._show_lp.isChecked(),
            show_rms=self._show_rms.isChecked(),
        )

    def continuous_stream(self) -> str:
        data = self._continuous_stream.currentData()
        return str(data) if data else "raw"

    def mark_stimulations(self) -> bool:
        return bool(self._mark_stims.isChecked())

    def set_trial_count(self, n_trials: int) -> None:
        self._n_trials = max(0, int(n_trials))
        self._refresh_ready_state()

    def set_channel(self, channel: str | None) -> None:
        self._channel = (channel or "").strip()
        self._channel_label.setText(
            f"Canal : {self._channel}" if self._channel else "Canal : —"
        )
        self._refresh_ready_state()

    def set_enabled_actions(self, enabled: bool) -> None:
        ready = enabled and self._n_trials > 0
        self._btn_analysis.setEnabled(ready)
        self._btn_spikes.setEnabled(ready)
        self._btn_graph.setEnabled(enabled and bool(self._channel))

    def _refresh_ready_state(self) -> None:
        has_channel = bool(self._channel)
        ready = self._n_trials > 0
        self._btn_analysis.setEnabled(ready and has_channel)
        self._btn_spikes.setEnabled(ready and has_channel)
        self._btn_graph.setEnabled(has_channel)
        if ready and has_channel:
            self._ready_hint.setText(
                f"{self._n_trials} stimulation(s). "
                "Cochez les traces, puis ouvrez une fenêtre."
            )
        elif has_channel:
            self._ready_hint.setText(
                "Enregistrement prêt. Ouvrez un graphique, "
                "ou attendez les stimulations pour l’analyse."
            )
        else:
            self._ready_hint.setText("Traitez un enregistrement pour ouvrir des vues.")

    def _emit_changed(self, *_args: Any) -> None:
        if self._updating:
            return
        self.analysisChanged.emit()
