"""Panneau de paramètres locaux pour une fenêtre de vue.

Initialisé avec les valeurs par défaut ; l’utilisateur les modifie ici uniquement.
Les zooms personnalisés s’ajoutent via le bouton « Ajouter un zoom… » de la fenêtre.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from view_config import AnalysisSettings, ViewerSettings


def _spin(
    minimum: float,
    maximum: float,
    value: float,
    *,
    decimals: int = 3,
    step: float = 0.01,
    suffix: str = "",
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(float(value))
    box.setKeyboardTracking(False)
    if suffix:
        box.setSuffix(suffix)
    return box


class LocalViewParams(QWidget):
    """Contrôles d’affichage / analyse propres à une fenêtre de vue."""

    changed = Signal()

    def __init__(
        self,
        settings: ViewerSettings,
        *,
        show_analysis: bool = True,
        show_continuous: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = replace(settings)
        self._updating = True

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        form_layout = QVBoxLayout(inner)
        form_layout.setContentsMargins(4, 4, 4, 4)
        form_layout.setSpacing(8)

        display = QGroupBox("Affichage", inner)
        display_form = QFormLayout(display)
        self._line_width = _spin(0.2, 8.0, settings.style.line_width, decimals=2, step=0.1)
        self._grid = QCheckBox("Grille")
        self._grid.setChecked(bool(settings.style.grid))
        self._psth_bin = _spin(0.001, 10.0, settings.psth_bin_window_s, suffix=" s")
        self._sampling = QSpinBox()
        self._sampling.setRange(1, 100)
        self._sampling.setValue(int(settings.sampling_percent))
        self._sampling.setSuffix(" %")
        self._sampling.setKeyboardTracking(False)
        display_form.addRow("Épaisseur de trait", self._line_width)
        display_form.addRow(self._grid)
        display_form.addRow("Bin PSTH", self._psth_bin)
        display_form.addRow("Spikes dessinés", self._sampling)
        form_layout.addWidget(display)

        if show_continuous:
            continuous = QGroupBox("Enregistrement continu", inner)
            continuous_form = QFormLayout(continuous)
            self._stream = QComboBox()
            self._stream.addItem("WIDE", "raw")
            self._stream.addItem("HIGH", "hp")
            self._stream.addItem("LOW", "lp")
            idx = self._stream.findData(settings.continuous_stream)
            self._stream.setCurrentIndex(idx if idx >= 0 else 0)
            self._mark_stims = QCheckBox("Marquer les stimulations")
            self._mark_stims.setChecked(bool(settings.continuous_mark_stims))
            continuous_form.addRow("Flux", self._stream)
            continuous_form.addRow(self._mark_stims)
            form_layout.addWidget(continuous)
        else:
            self._stream = None
            self._mark_stims = None

        if show_analysis:
            analysis = QGroupBox("Analyse", inner)
            analysis_form = QFormLayout(analysis)
            self._mode = QComboBox()
            self._mode.addItem("Moyenne d’essais", "average")
            self._mode.addItem("Une stimulation", "stimulation")
            mode_idx = self._mode.findData(settings.analysis.mode)
            self._mode.setCurrentIndex(mode_idx if mode_idx >= 0 else 0)
            self._stim_index = QSpinBox()
            self._stim_index.setMinimum(1)
            self._stim_index.setMaximum(max(1, int(settings.analysis.stim_index) + 1))
            self._stim_index.setValue(int(settings.analysis.stim_index) + 1)
            self._stim_index.setKeyboardTracking(False)
            analysis_form.addRow("Mode", self._mode)
            analysis_form.addRow("Stimulation n°", self._stim_index)
            form_layout.addWidget(analysis)
        else:
            self._mode = None
            self._stim_index = None

        form_layout.addStretch(1)
        scroll.setWidget(inner)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

        for widget in self._watch_widgets():
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._emit_changed)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._emit_changed)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self._emit_changed)

        self._updating = False

    def _watch_widgets(self) -> list[Any]:
        widgets: list[Any] = [self._line_width, self._grid, self._psth_bin, self._sampling]
        for optional in (self._stream, self._mark_stims, self._mode, self._stim_index):
            if optional is not None:
                widgets.append(optional)
        return widgets

    def _emit_changed(self, *_args: Any) -> None:
        if self._updating:
            return
        self.changed.emit()

    def set_trial_count(self, n_trials: int) -> None:
        if self._stim_index is None:
            return
        n = max(1, int(n_trials))
        self._updating = True
        self._stim_index.setMaximum(n)
        if self._stim_index.value() > n:
            self._stim_index.setValue(n)
        self._updating = False

    def settings(self) -> ViewerSettings:
        style = replace(
            self._settings.style,
            line_width=float(self._line_width.value()),
            grid=bool(self._grid.isChecked()),
        )
        if self._mode is not None and self._stim_index is not None:
            analysis = AnalysisSettings(
                mode=str(self._mode.currentData() or "average"),  # type: ignore[arg-type]
                stim_index=max(0, int(self._stim_index.value()) - 1),
                show_raw=self._settings.analysis.show_raw,
                show_hp=self._settings.analysis.show_hp,
                show_lp=self._settings.analysis.show_lp,
                show_rms=self._settings.analysis.show_rms,
            )
        else:
            analysis = self._settings.analysis

        kwargs: dict[str, Any] = {
            "style": style,
            "analysis": analysis,
            "psth_bin_window_s": float(self._psth_bin.value()),
            "sampling_percent": int(self._sampling.value()),
        }
        if self._stream is not None and self._mark_stims is not None:
            kwargs["continuous_stream"] = str(self._stream.currentData() or "raw")
            kwargs["continuous_mark_stims"] = bool(self._mark_stims.isChecked())
        return replace(self._settings, **kwargs)
