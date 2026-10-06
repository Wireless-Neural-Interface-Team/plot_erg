"""Panneau de paramètres d’affichage locaux pour une fenêtre de vue.

Chaque fenêtre ouverte possède sa propre instance : tous ses graphs partagent
ces réglages ; les autres fenêtres restent indépendantes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.jobs import Debouncer
from view_config import (
    LEGEND_LOCATIONS,
    AxisLimits,
    LegendSettings,
    PanelStyle,
    ViewerSettings,
)


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


def _int_spin(minimum: int, maximum: int, value: int, *, step: int = 1) -> QSpinBox:
    box = QSpinBox()
    box.setRange(int(minimum), int(maximum))
    box.setSingleStep(int(step))
    box.setValue(int(value))
    box.setKeyboardTracking(False)
    return box


class _CompactDoubleSpinBox(QDoubleSpinBox):
    """Spinbox dont le sizeHint ignore la plage ±1e6 (sinon trop large pour un dock)."""

    def __init__(self, *, hint_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hint_text = hint_text
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # noqa: D102
        height = super().sizeHint().height()
        # Boutons ± + marge : largeur basée sur une valeur typique, pas sur ±1e6.
        width = self.fontMetrics().horizontalAdvance(self._hint_text) + 28
        return QSize(max(72, width), height)

    def minimumSizeHint(self) -> QSize:  # noqa: D102
        height = super().minimumSizeHint().height()
        return QSize(56, height)


def _axis_spin(
    value: float,
    *,
    decimals: int,
    step: float,
    suffix: str,
) -> QDoubleSpinBox:
    hint = f"-000.{'0' * decimals}{suffix}"
    box = _CompactDoubleSpinBox(hint_text=hint)
    box.setDecimals(decimals)
    box.setRange(-1e6, 1e6)
    box.setSingleStep(step)
    box.setValue(float(value))
    box.setKeyboardTracking(False)
    if suffix:
        box.setSuffix(suffix)
    return box


class AxisLimitRow(QWidget):
    """Case « Manuel » + spinboxes min/max pour une échelle d’axe."""

    changed = Signal()

    def __init__(
        self,
        limits: AxisLimits,
        *,
        unit: str = " µV",
        decimals: int = 2,
        step: float = 10.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._enabled = QCheckBox("Manuel")
        self._enabled.setChecked(bool(limits.enabled))
        self._min = _axis_spin(
            limits.minimum, decimals=decimals, step=step, suffix=unit
        )
        self._max = _axis_spin(
            limits.maximum, decimals=decimals, step=step, suffix=unit
        )
        # Ligne 1 : case à cocher — ligne 2 : min/max (reste visible si le dock est étroit).
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.addWidget(self._enabled)
        range_row = QHBoxLayout()
        range_row.setContentsMargins(0, 0, 0, 0)
        range_row.setSpacing(4)
        range_row.addWidget(self._min, 1)
        range_row.addWidget(QLabel("à"))
        range_row.addWidget(self._max, 1)
        root.addLayout(range_row)
        self._enabled.toggled.connect(self._on_enabled)
        self._min.valueChanged.connect(lambda _v: self.changed.emit())
        self._max.valueChanged.connect(lambda _v: self.changed.emit())
        self._on_enabled(self._enabled.isChecked(), emit=False)

    def _on_enabled(self, checked: bool, *, emit: bool = True) -> None:
        self._min.setEnabled(bool(checked))
        self._max.setEnabled(bool(checked))
        if emit:
            self.changed.emit()

    def value(self) -> AxisLimits:
        return AxisLimits(
            enabled=self._enabled.isChecked(),
            minimum=float(self._min.value()),
            maximum=float(self._max.value()),
        )

    def set_value(self, limits: AxisLimits) -> None:
        self._enabled.setChecked(bool(limits.enabled))
        self._min.setValue(float(limits.minimum))
        self._max.setValue(float(limits.maximum))


class LocalViewParams(QWidget):
    """Contrôles légende / style (/ analyse) propres à une fenêtre."""

    changed = Signal()

    def __init__(
        self,
        settings: ViewerSettings,
        *,
        show_analysis: bool = True,
        show_continuous: bool = True,
        show_display_extras: bool = True,
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

        legend = settings.legend
        style = settings.style

        axis_group = QGroupBox("Échelles des axes", inner)
        axis_form = QFormLayout(axis_group)
        axis_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        axis_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._x_limits = AxisLimitRow(
            settings.x_limits,
            unit=" s",
            decimals=3,
            step=0.01,
        )
        self._x_limits.setToolTip(
            "Limites X manuelles (temps en secondes), appliquées à tous les "
            "graphs temporels de cette fenêtre. Relatif à la stim pour les "
            "traces d’analyse ; temps absolu pour les enregistrements continus."
        )
        self._trace_ylim = AxisLimitRow(settings.trace_ylim, unit=" µV")
        self._rms_ylim = AxisLimitRow(settings.rms_ylim, unit=" µV")
        self._stim_hp_ylim = AxisLimitRow(settings.stim_hp_ylim, unit=" µV")
        axis_form.addRow("Axe X (temps) :", self._x_limits)
        axis_form.addRow("Axe Y — traces :", self._trace_ylim)
        axis_form.addRow("Axe Y — RMS :", self._rms_ylim)
        axis_form.addRow("Axe Y — passe-haut stim :", self._stim_hp_ylim)
        form_layout.addWidget(axis_group)

        legend_group = QGroupBox("Légendes", inner)
        legend_form = QFormLayout(legend_group)
        self._legend_visible = QCheckBox("Afficher les légendes")
        self._legend_visible.setChecked(bool(legend.visible))
        self._legend_location = QComboBox()
        for location in LEGEND_LOCATIONS:
            self._legend_location.addItem(
                "Sous le panneau" if location == "below" else location.capitalize(),
                location,
            )
        loc_idx = self._legend_location.findData(legend.location)
        self._legend_location.setCurrentIndex(loc_idx if loc_idx >= 0 else 0)
        self._legend_font = _spin(
            4.0, 24.0, legend.font_size, decimals=1, step=0.5, suffix=" pt"
        )
        self._legend_columns = _int_spin(1, 6, legend.columns)
        self._legend_frame = QCheckBox("Cadre autour des légendes")
        self._legend_frame.setChecked(bool(legend.frame))
        self._legend_filters = QCheckBox("Détails du filtre")
        self._legend_filters.setChecked(bool(legend.show_filter_details))
        self._legend_counts = QCheckBox("Compteurs d’échantillons / spikes")
        self._legend_counts.setChecked(bool(legend.show_sample_counts))
        legend_form.addRow(self._legend_visible)
        legend_form.addRow("Position :", self._legend_location)
        legend_form.addRow("Taille de police :", self._legend_font)
        legend_form.addRow("Colonnes :", self._legend_columns)
        legend_form.addRow(self._legend_frame)
        legend_form.addRow(self._legend_filters)
        legend_form.addRow(self._legend_counts)
        form_layout.addWidget(legend_group)

        style_group = QGroupBox("Style des panneaux", inner)
        style_form = QFormLayout(style_group)
        self._title_font = _spin(
            5.0, 24.0, style.title_font_size, decimals=1, step=0.5, suffix=" pt"
        )
        self._label_font = _spin(
            5.0, 24.0, style.label_font_size, decimals=1, step=0.5, suffix=" pt"
        )
        self._tick_font = _spin(
            4.0, 20.0, style.tick_font_size, decimals=1, step=0.5, suffix=" pt"
        )
        self._line_width = _spin(
            0.2, 8.0, style.line_width, decimals=2, step=0.1
        )
        self._grid = QCheckBox("Afficher la grille")
        self._grid.setChecked(bool(style.grid))
        self._grid_alpha = _spin(
            0.0, 1.0, style.grid_alpha, decimals=2, step=0.05
        )
        self._show_borders = QCheckBox("Afficher les bordures du graphique")
        self._show_borders.setChecked(bool(style.show_borders))
        self._show_borders.setToolTip(
            "Cadre avec graduations autour de la zone de tracé de chaque panneau."
        )
        self._ticks_inside = QCheckBox("Graduations à l’intérieur du cadre")
        self._ticks_inside.setChecked(bool(style.ticks_inside))
        self._ticks_inside.setToolTip(
            "Oriente les graduations (ticks) vers l’intérieur du carré du graphique."
        )
        self._stim_markers = QCheckBox("Afficher les pointillés de stimulation")
        self._stim_markers.setChecked(bool(legend.show_reference_markers))
        self._stim_markers.setToolTip(
            "Lignes en pointillés au début (et à la fin) de stimulation sur les graphiques."
        )
        self._max_points = _int_spin(
            500, 200000, style.max_points_per_curve, step=500
        )
        self._max_points.setToolTip(
            "Points dessinés par courbe. Au-delà, une enveloppe min/max conserve "
            "les pics tout en restant rapide."
        )
        style_form.addRow("Police du titre :", self._title_font)
        style_form.addRow("Police des axes :", self._label_font)
        style_form.addRow("Police des ticks :", self._tick_font)
        style_form.addRow("Épaisseur de trait :", self._line_width)
        style_form.addRow(self._grid)
        style_form.addRow("Opacité de la grille :", self._grid_alpha)
        style_form.addRow(self._show_borders)
        style_form.addRow(self._ticks_inside)
        style_form.addRow(self._stim_markers)
        style_form.addRow("Points max / courbe :", self._max_points)
        form_layout.addWidget(style_group)

        if show_display_extras:
            display = QGroupBox("Affichage spikes", inner)
            display_form = QFormLayout(display)
            self._psth_bin = _spin(
                0.001, 10.0, settings.psth_bin_window_s, suffix=" s"
            )
            self._sampling = _int_spin(1, 100, settings.sampling_percent)
            self._sampling.setSuffix(" %")
            display_form.addRow("Bin PSTH", self._psth_bin)
            display_form.addRow("Spikes dessinés", self._sampling)
            form_layout.addWidget(display)
        else:
            self._psth_bin = None
            self._sampling = None

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

        self._debouncer = Debouncer(120, self)
        self._debouncer.triggered.connect(self.changed.emit)

        for widget in self._watch_widgets():
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._emit_changed)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._emit_changed)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self._emit_changed)
            elif hasattr(widget, "changed"):
                widget.changed.connect(self._emit_changed)

        self._updating = False

    def _watch_widgets(self) -> list[Any]:
        widgets: list[Any] = [
            self._x_limits,
            self._trace_ylim,
            self._rms_ylim,
            self._stim_hp_ylim,
            self._legend_visible,
            self._legend_location,
            self._legend_font,
            self._legend_columns,
            self._legend_frame,
            self._legend_filters,
            self._legend_counts,
            self._title_font,
            self._label_font,
            self._tick_font,
            self._line_width,
            self._grid,
            self._grid_alpha,
            self._show_borders,
            self._ticks_inside,
            self._stim_markers,
            self._max_points,
        ]
        for optional in (
            self._psth_bin,
            self._sampling,
            self._stream,
            self._mark_stims,
            self._mode,
            self._stim_index,
        ):
            if optional is not None:
                widgets.append(optional)
        return widgets

    def _emit_changed(self, *_args: Any) -> None:
        if self._updating:
            return
        self._debouncer.request()

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
        legend = LegendSettings(
            visible=bool(self._legend_visible.isChecked()),
            location=str(self._legend_location.currentData() or "below"),  # type: ignore[arg-type]
            font_size=float(self._legend_font.value()),
            columns=int(self._legend_columns.value()),
            frame=bool(self._legend_frame.isChecked()),
            show_filter_details=bool(self._legend_filters.isChecked()),
            show_sample_counts=bool(self._legend_counts.isChecked()),
            show_reference_markers=bool(self._stim_markers.isChecked()),
        )
        style = PanelStyle(
            title_font_size=float(self._title_font.value()),
            label_font_size=float(self._label_font.value()),
            tick_font_size=float(self._tick_font.value()),
            line_width=float(self._line_width.value()),
            grid=bool(self._grid.isChecked()),
            grid_alpha=float(self._grid_alpha.value()),
            show_borders=bool(self._show_borders.isChecked()),
            ticks_inside=bool(self._ticks_inside.isChecked()),
            max_points_per_curve=int(self._max_points.value()),
            tight_layout=bool(self._settings.style.tight_layout),
        )
        if self._mode is not None and self._stim_index is not None:
            analysis = replace(
                self._settings.analysis,
                mode=str(self._mode.currentData() or "average"),  # type: ignore[arg-type]
                stim_index=max(0, int(self._stim_index.value()) - 1),
            )
        else:
            analysis = self._settings.analysis

        kwargs: dict[str, Any] = {
            "legend": legend,
            "style": style,
            "analysis": analysis,
            "x_limits": self._x_limits.value(),
            "trace_ylim": self._trace_ylim.value(),
            "rms_ylim": self._rms_ylim.value(),
            "stim_hp_ylim": self._stim_hp_ylim.value(),
        }
        if self._psth_bin is not None and self._sampling is not None:
            kwargs["psth_bin_window_s"] = float(self._psth_bin.value())
            kwargs["sampling_percent"] = int(self._sampling.value())
        if self._stream is not None and self._mark_stims is not None:
            stream = str(self._stream.currentData() or "raw")
            kwargs["continuous_stream"] = stream
            kwargs["continuous_streams"] = (stream,)
            kwargs["continuous_mark_stims"] = bool(self._mark_stims.isChecked())
        return replace(self._settings, **kwargs)

    def sync_base_settings(self, settings: ViewerSettings) -> None:
        """Mettre à jour la base (analyse / plages) sans écraser légende & style UI."""
        # Conserver les réglages UI courants, n’actualiser que le socle.
        current = self.settings()
        self._settings = replace(
            settings,
            legend=current.legend,
            style=current.style,
            psth_bin_window_s=current.psth_bin_window_s,
            sampling_percent=current.sampling_percent,
            x_limits=current.x_limits,
            trace_ylim=current.trace_ylim,
            rms_ylim=current.rms_ylim,
            stim_hp_ylim=current.stim_hp_ylim,
        )
