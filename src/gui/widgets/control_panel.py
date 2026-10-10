"""Panneau de contrôle bas — canal actif et mode de vue."""



from __future__ import annotations



from typing import Literal



from PySide6.QtCore import QSize, Qt, Signal

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

_MONTAGE_HINT_TOOLTIP = (
    "Revue montage — Mode/Pipeline = graphs · Canaux = visibilité · "
    "Paramètres = canaux/page · Ctrl+Shift+M = aperçu"
)





class ControlPanel(QFrame):

    """Indicateur de mode / canal et pagination montage.



    Les flux WIDE / HIGH / LOW se règlent dans Paramètres → Canal → Traitement.

    Marqueurs stim : Affichage. Mode continuous / moyenne / stimulation : Canal.

    """



    montagePrevPage = Signal()

    montageNextPage = Signal()



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



        self._montage_prev = QPushButton("◀ Préc.", self)

        self._montage_prev.setObjectName("montagePagePrev")

        self._montage_prev.setToolTip("Page de canaux précédente (revue montage)")

        self._montage_next = QPushButton("Suiv. ▶", self)

        self._montage_next.setObjectName("montagePageNext")

        self._montage_next.setToolTip("Page de canaux suivante (revue montage)")

        self._montage_page_label = QLabel("", self)

        self._montage_page_label.setObjectName("montagePageLabel")

        self._montage_page_label.setToolTip(

            "Tranche de canaux affichée — réglez le nombre dans Paramètres → Affichage"

        )

        for widget in (self._montage_prev, self._montage_next, self._montage_page_label):

            widget.setVisible(False)
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
            )

        self._montage_page_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )

        row = QHBoxLayout()

        row.setSpacing(8)

        row.addWidget(self._mode_badge)

        row.addSpacing(4)

        row.addWidget(QLabel("Canal", self))

        row.addWidget(self._channel_label)

        row.addSpacing(12)

        row.addWidget(self._montage_prev)

        row.addWidget(self._montage_page_label)

        row.addWidget(self._montage_next)

        self._hint = QLabel("Ajoutez un .rhs puis Traiter (F5).", self)

        self._hint.setObjectName("hintLabel")
        self._hint.setMinimumWidth(0)
        self._hint.setWordWrap(True)
        self._hint.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )

        row.addWidget(self._hint, 1)



        layout = QVBoxLayout(self)

        layout.setContentsMargins(10, 8, 10, 8)

        layout.setSpacing(6)

        layout.addLayout(row)



        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.setMinimumHeight(52)



        self._montage_prev.clicked.connect(self.montagePrevPage.emit)

        self._montage_next.clicked.connect(self.montageNextPage.emit)

    def minimumSizeHint(self) -> QSize:  # noqa: D102
        # Ne pas imposer la largeur du libellé d’aide (texte montage très long).
        return QSize(0, super().minimumSizeHint().height())

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

        show_pager = self._view_mode == "montage"

        for widget in (self._montage_prev, self._montage_next, self._montage_page_label):

            widget.setVisible(show_pager)

        self._refresh_ready()



    def view_mode(self) -> ViewMode:

        return self._view_mode



    def set_montage_page_info(

        self,

        *,

        page: int,

        n_pages: int,

        start: int,

        end: int,

        n_visible: int,

    ) -> None:

        """Mettre à jour le libellé et l’état des boutons de pagination montage."""

        page = max(0, int(page))

        n_pages = max(1, int(n_pages))

        start = max(0, int(start))

        end = max(start, int(end))

        n_visible = max(0, int(n_visible))

        if n_visible <= 0:

            self._montage_page_label.setText("Aucun canal")

        else:

            self._montage_page_label.setText(

                f"Canaux {start}–{end} / {n_visible} · p. {page + 1}/{n_pages}"

            )

        self._montage_prev.setEnabled(page > 0)

        self._montage_next.setEnabled(page < n_pages - 1)



    def set_trial_count(self, n_trials: int) -> None:

        self._n_trials = max(0, int(n_trials))

        self._refresh_ready()



    def set_busy(self, busy: bool) -> None:

        """Pendant un build, rafraîchir les indications (canal / mode)."""

        del busy

        self._refresh_ready()



    def _refresh_ready(self) -> None:

        has_channel = bool(self._channel)

        ready = self._n_trials > 0

        if self._view_mode == "montage":
            self._hint.setText("Revue montage — Ctrl+Shift+M = aperçu")
            self._hint.setToolTip(_MONTAGE_HINT_TOOLTIP)
        elif ready and has_channel:
            self._hint.setToolTip("")
            self._hint.setText(
                f"{self._n_trials} stim(s) · {self._channel} — "
                "mode : Paramètres → Canal"
            )
        elif has_channel:
            self._hint.setToolTip("")
            self._hint.setText(f"Canal {self._channel} — Traiter (F5)")
        else:
            self._hint.setToolTip("")
            self._hint.setText("Ajoutez un .rhs puis Traiter (F5).")


