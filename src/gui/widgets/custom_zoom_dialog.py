"""Dialogue pour définir un zoom temporel personnalisé."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class CustomZoomSpec:
    t0_s: float
    t1_s: float
    label: str = ""


class CustomZoomDialog(QDialog):
    """Demande une fenêtre [t0, t1] relative à la stimulation."""

    def __init__(
        self,
        *,
        default_t0: float = -0.1,
        default_t1: float = 0.2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ajouter un zoom")
        self.setMinimumWidth(360)

        hint = QLabel(
            "Fenêtre temporelle relative au début de stimulation (t = 0 s). "
            "Exemple : t₀ = −0,1 s et t₁ = 0,4 s. Les graphs d’analyse "
            "(moyenne / une stim) utiliseront cette plage telle quelle."
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)

        self._label = QLineEdit("")
        self._label.setPlaceholderText("Optionnel — ex. début, late, 50–150 ms…")
        self._t0 = QDoubleSpinBox()
        self._t0.setRange(-1000.0, 1000.0)
        self._t0.setDecimals(3)
        self._t0.setSingleStep(0.01)
        self._t0.setSuffix(" s")
        self._t0.setValue(float(default_t0))
        self._t1 = QDoubleSpinBox()
        self._t1.setRange(-1000.0, 1000.0)
        self._t1.setDecimals(3)
        self._t1.setSingleStep(0.01)
        self._t1.setSuffix(" s")
        self._t1.setValue(float(default_t1))

        form = QFormLayout()
        form.addRow("Nom :", self._label)
        form.addRow("Début (t₀) :", self._t0)
        form.addRow("Fin (t₁) :", self._t1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self._spec: CustomZoomSpec | None = None

    def _accept(self) -> None:
        t0 = float(self._t0.value())
        t1 = float(self._t1.value())
        if t1 <= t0:
            self._t1.setFocus()
            return
        self._spec = CustomZoomSpec(
            t0_s=t0, t1_s=t1, label=self._label.text().strip()
        )
        self.accept()

    def spec(self) -> CustomZoomSpec | None:
        return self._spec


def ask_custom_zoom(
    parent: QWidget | None = None,
    *,
    default_t0: float = -0.1,
    default_t1: float = 0.2,
) -> CustomZoomSpec | None:
    dialog = CustomZoomDialog(default_t0=default_t0, default_t1=default_t1, parent=parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.spec()
