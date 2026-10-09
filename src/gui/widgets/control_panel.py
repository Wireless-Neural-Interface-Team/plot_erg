"""Panneau de contrôle bas — canal actif, mode de vue et raccourci Analyse."""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

ViewMode = Literal["preview", "montage"]


class ControlPanel(QFrame):
    """Indicateur de mode / canal et bouton Analyse.

    Les flux WIDE / HIGH / LOW et les marqueurs stim se règlent dans
    Paramètres → Affichage (Traces continues). Mode continuous / moyenne /
    stimulation : Paramètres → Canal.
    """

    analysisRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("controlPanel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._n_trials = 0
        self._channel = ""
        self._view_mode: ViewMode = "preview"

        self._mode_badge = QLabel("Aperçu", self)
        self._mode_badge.setObjectName("viewModeBadge")
        self._mode_badge.setProperty("mode", "preview")
        self._mode_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._mode_badge.setToolTip("Mode de la vue centrale")

        self._channel_label = QLabel("—", self)
        self._channel_label.setObjectName("controlChannel")
        self._channel_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._channel_label.setToolTip("Canal actuellement sélectionné")

        self._analyse = QPushButton("Analyse", self)
        self._analyse.setObjectName("filterAnalyse")
        self._analyse.setToolTip(
            "Afficher la moyenne d’essais dans l’aperçu "
            "(Paramètres → Canal pour continuous / moyenne / stimulation). "
            "Ctrl+I · aussi double-clic MEA / liste."
        )

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._mode_badge)
        row.addSpacing(4)
        row.addWidget(QLabel("Canal", self))
        row.addWidget(self._channel_label)
        row.addSpacing(12)
        row.addWidget(self._analyse)
        row.addStretch(1)

        self._hint = QLabel("Ajoutez un .rhs puis Traiter (F5).", self)
        self._hint.setObjectName("hintLabel")
        row.addWidget(self._hint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(row)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(52)

        self._analyse.clicked.connect(self.analysisRequested.emit)

    def set_channel(self, channel: str | None) -> None:
        self._channel = (channel or "").strip()
        self._channel_label.setText(self._channel or "—")
        self._refresh_ready()

    def set_view_mode(self, mode: ViewMode) -> None:
        self._view_mode = mode if mode in ("preview", "montage") else "preview"
        if self._view_mode == "montage":
            self._mode_badge.setText("Montage")
            self._mode_badge.setProperty("mode", "montage")
        else:
            self._mode_badge.setText("Aperçu")
            self._mode_badge.setProperty("mode", "preview")
        # Forcer le recalcul du style Qt après changement de propriété.
        self._mode_badge.style().unpolish(self._mode_badge)
        self._mode_badge.style().polish(self._mode_badge)
        self._refresh_ready()

    def view_mode(self) -> ViewMode:
        return self._view_mode

    def set_trial_count(self, n_trials: int) -> None:
        self._n_trials = max(0, int(n_trials))
        self._refresh_ready()

    def set_busy(self, busy: bool) -> None:
        """Pendant un build, l’analyse reste disponible si un canal est prêt."""
        del busy
        self._refresh_ready()

    def _refresh_ready(self) -> None:
        has_channel = bool(self._channel)
        ready = self._n_trials > 0
        if self._view_mode == "montage":
            self._hint.setText(
                "Revue montage — cases Canaux = visibilité · Ctrl+Shift+M = retour aperçu"
            )
        elif ready and has_channel:
            self._hint.setText(
                f"{self._n_trials} stim(s) · {self._channel} — "
                "mode : Paramètres → Canal"
            )
        elif has_channel:
            self._hint.setText(f"Canal {self._channel} — Traiter (F5)")
        else:
            self._hint.setText("Ajoutez un .rhs puis Traiter (F5).")
