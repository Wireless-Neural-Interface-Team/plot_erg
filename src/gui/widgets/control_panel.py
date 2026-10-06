"""Panneau de contrôle bas — canal actif et filtres d’aperçu."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from view_config import AnalysisStream


class ControlPanel(QFrame):
    """Filtres WIDE/LOW/HIGH (multi) et canal actif.

    L’inspection se lance via la barre d’outils / menus / double-clic MEA.
    """

    filterChanged = Signal()

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

        self._show_wide = QCheckBox("WIDE", self)
        self._show_wide.setChecked(True)
        self._show_wide.setToolTip("Aperçu canal / revue montage : signal brut (wideband)")
        self._show_low = QCheckBox("LOW", self)
        self._show_low.setToolTip("Aperçu canal / revue montage : passe-bas (LFP)")
        self._show_high = QCheckBox("HIGH", self)
        self._show_high.setToolTip("Aperçu canal / revue montage : passe-haut (spikes)")

        self._mark_stims = QCheckBox("Stim markers", self)
        self._mark_stims.setChecked(True)
        self._mark_stims.setToolTip("Marquer les stimulations sur l’aperçu / le montage")

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Channel", self))
        row.addWidget(self._channel_label)
        row.addSpacing(12)
        row.addWidget(QLabel("Flux", self))
        for box in (self._show_wide, self._show_high, self._show_low):
            row.addWidget(box)
        row.addSpacing(8)
        row.addWidget(self._mark_stims)
        row.addStretch(1)
        self._hint = QLabel("Ajoutez un .rhs puis Traiter (F5).", self)
        self._hint.setObjectName("hintLabel")
        row.addWidget(self._hint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(row)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(48)

        for box in (self._show_wide, self._show_high, self._show_low, self._mark_stims):
            box.toggled.connect(self._on_stream_toggled)

    # ---------------------------------------------------------------- values

    def continuous_streams(self) -> tuple[AnalysisStream, ...]:
        streams: list[AnalysisStream] = []
        if self._show_wide.isChecked():
            streams.append("raw")
        if self._show_high.isChecked():
            streams.append("hp")
        if self._show_low.isChecked():
            streams.append("lp")
        return tuple(streams) or ("raw",)

    def continuous_stream(self) -> str:
        streams = self.continuous_streams()
        return streams[0]

    def mark_stimulations(self) -> bool:
        return bool(self._mark_stims.isChecked())

    def set_channel(self, channel: str | None) -> None:
        self._channel = (channel or "").strip()
        self._channel_label.setText(self._channel or "—")
        self._refresh_ready()

    def set_trial_count(self, n_trials: int) -> None:
        self._n_trials = max(0, int(n_trials))
        self._refresh_ready()

    def set_busy(self, busy: bool) -> None:
        """Pendant un build, l’inspection reste disponible si un canal est prêt."""
        del busy
        self._refresh_ready()

    def _refresh_ready(self) -> None:
        has_channel = bool(self._channel)
        ready = self._n_trials > 0
        if ready and has_channel:
            self._hint.setText(
                f"{self._n_trials} stim(s) · {self._channel} — Inspecter ou double-clic"
            )
        elif has_channel:
            self._hint.setText(f"Canal {self._channel} — Traiter (F5) puis inspecter")
        else:
            self._hint.setText("Ajoutez un .rhs, Traiter (F5), puis Inspecter.")

    def _on_stream_toggled(self, *_args: Any) -> None:
        if self._updating:
            return
        # Toujours au moins un flux affiché.
        if not any(
            box.isChecked() for box in (self._show_wide, self._show_high, self._show_low)
        ):
            self._updating = True
            self._show_wide.setChecked(True)
            self._updating = False
        self.filterChanged.emit()
