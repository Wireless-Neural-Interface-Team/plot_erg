"""Panneau de paramètres d’affichage locaux pour une fenêtre de vue.

Chaque fenêtre ouverte possède sa propre instance : tous ses graphs partagent
ces réglages ; les autres fenêtres restent indépendantes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.form_widgets import (
    AxisLimitRow,
    FitWidthScrollArea,
    configure_narrow_form,
    make_double_spin as _spin,
    make_int_spin as _int_spin,
)
from gui.jobs import Debouncer
from view_config import (
    LEGEND_LOCATIONS,
    TIME_SYNC_LABELS,
    LegendSettings,
    PanelStyle,
    ViewerSettings,
)

# Réexport pour params_panel et autres imports historiques.
__all__ = ["AxisLimitRow", "LocalViewParams"]


def _scrollable(inner: QWidget) -> FitWidthScrollArea:
    area = FitWidthScrollArea()
    area.setWidget(inner)
    return area


class LocalViewParams(QWidget):
    """Contrôles légende / style / axes propres à une fenêtre.

    Si ``host_tabs`` est fourni, les pages Affichage / Style sont ajoutées
    à ce ``QTabWidget`` (comme le dock Paramètres de la fenêtre principale).
    Sinon, tout est empilé dans un scroll unique.

    Mode / courbes / spikes → Paramètres → Canal (aperçu).
    Flux continuous → Paramètres Affichage (embarqué) ou coches Traces continues (flottant).
    Sync / trigger → fenêtre principale (ou flottant continuous).
    """

    changed = Signal()

    def __init__(
        self,
        settings: ViewerSettings,
        *,
        show_sync: bool = True,
        show_display_extras: bool = True,
        host_tabs: QTabWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = replace(settings)
        self._updating = True
        self._show_sync = bool(show_sync)
        self._show_display_extras = bool(show_display_extras)

        display_page = self._build_display_page(settings)
        style_page = self._build_style_page(settings)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if host_tabs is not None:
            host_tabs.addTab(_scrollable(display_page), "Affichage")
            host_tabs.addTab(_scrollable(style_page), "Style")
            host_tabs.setTabToolTip(
                host_tabs.count() - 2,
                "Échelles et spikes — redessin immédiat (cette fenêtre)",
            )
            host_tabs.setTabToolTip(
                host_tabs.count() - 1,
                "Légendes et style des panneaux — redessin immédiat",
            )
        else:
            scroll = FitWidthScrollArea(self)
            inner = QWidget()
            form_layout = QVBoxLayout(inner)
            form_layout.setContentsMargins(4, 4, 4, 4)
            form_layout.setSpacing(8)
            form_layout.addWidget(display_page)
            form_layout.addWidget(style_page)
            form_layout.addStretch(1)
            scroll.setWidget(inner)
            root.addWidget(scroll)

        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

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

    def _build_display_page(self, settings: ViewerSettings) -> QWidget:
        page = QWidget()
        form_layout = QVBoxLayout(page)
        form_layout.setContentsMargins(8, 8, 8, 8)
        form_layout.setSpacing(8)

        axis_group = QGroupBox("Échelles des axes", page)
        axis_form = configure_narrow_form(QFormLayout(axis_group))
        self._x_limits = AxisLimitRow(
            settings.x_limits,
            unit=" s",
            decimals=3,
            step=0.01,
        )
        self._x_limits.setToolTip(
            "Limites X manuelles (temps en secondes), appliquées à tous les "
            "graphs temporels de cette fenêtre."
        )
        self._trace_ylim = AxisLimitRow(settings.trace_ylim, unit=" µV")
        # Échelle Y RMS : sous la case RMS (onglet Analyse / Canal).
        self._stim_hp_ylim = AxisLimitRow(settings.stim_hp_ylim, unit=" µV")
        axis_form.addRow("Axe X (temps) :", self._x_limits)
        axis_form.addRow("Axe Y — traces :", self._trace_ylim)
        axis_form.addRow("Axe Y — passe-haut stim :", self._stim_hp_ylim)
        form_layout.addWidget(axis_group)

        if self._show_display_extras:
            display = QGroupBox("Affichage spikes", page)
            display_form = configure_narrow_form(QFormLayout(display))
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

        if self._show_sync:
            sync_group = QGroupBox("Synchronisation / déclencheur", page)
            sync_form = configure_narrow_form(QFormLayout(sync_group))
            self._time_sync = QComboBox()
            for key, label in TIME_SYNC_LABELS.items():
                self._time_sync.addItem(label, key)
            sync_idx = self._time_sync.findData(settings.time_sync)
            self._time_sync.setCurrentIndex(sync_idx if sync_idx >= 0 else 0)
            self._time_sync.setToolTip(
                "Origine de l’axe temps des traces continues : début du fichier "
                "ou premier trigger détecté."
            )
            self._trigger_polarity = QComboBox()
            self._trigger_polarity.addItem("Low (front descendant)", "low")
            self._trigger_polarity.addItem("High (front montant)", "high")
            pol_idx = self._trigger_polarity.findData(settings.trigger_polarity)
            self._trigger_polarity.setCurrentIndex(pol_idx if pol_idx >= 0 else 0)
            self._trigger_threshold = _spin(
                -100.0, 100.0, settings.trigger_threshold, decimals=3, suffix=" V"
            )
            self._trigger_threshold.setToolTip(
                "Seuil de détection du trigger (V). Pour recalculer la segmentation, "
                "utilisez aussi Traiter (F5) dans la fenêtre principale."
            )
            sync_form.addRow("Synchroniser sur :", self._time_sync)
            sync_form.addRow("Trigger :", self._trigger_polarity)
            sync_form.addRow("Limite de détection :", self._trigger_threshold)
            form_layout.addWidget(sync_group)
        else:
            self._time_sync = None
            self._trigger_polarity = None
            self._trigger_threshold = None

        form_layout.addStretch(1)
        return page

    def _build_style_page(self, settings: ViewerSettings) -> QWidget:
        page = QWidget()
        form_layout = QVBoxLayout(page)
        form_layout.setContentsMargins(8, 8, 8, 8)
        form_layout.setSpacing(8)

        legend = settings.legend
        style = settings.style

        legend_group = QGroupBox("Légendes", page)
        legend_form = configure_narrow_form(QFormLayout(legend_group))
        self._legend_visible = QCheckBox("Afficher les légendes")
        self._legend_visible.setChecked(bool(legend.visible))
        self._legend_location = QComboBox()
        self._legend_location.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._legend_location.setMinimumContentsLength(10)
        self._legend_location.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
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
        self._legend_gap = _spin(0.0, 1.0, legend.gap, decimals=2, step=0.02)
        self._legend_gap.setToolTip(
            "Écart entre la légende et le graphique (fraction de la hauteur des axes)."
        )
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
        legend_form.addRow("Distance au graphique :", self._legend_gap)
        legend_form.addRow(self._legend_frame)
        legend_form.addRow(self._legend_filters)
        legend_form.addRow(self._legend_counts)
        form_layout.addWidget(legend_group)

        style_group = QGroupBox("Style des panneaux", page)
        style_form = configure_narrow_form(QFormLayout(style_group))
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
        style_form.addRow(self._stim_markers)
        style_form.addRow("Points max / courbe :", self._max_points)
        form_layout.addWidget(style_group)
        form_layout.addStretch(1)
        return page

    def _watch_widgets(self) -> list[Any]:
        widgets: list[Any] = [
            self._x_limits,
            self._trace_ylim,
            self._stim_hp_ylim,
            self._legend_visible,
            self._legend_location,
            self._legend_font,
            self._legend_columns,
            self._legend_gap,
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
            self._stim_markers,
            self._max_points,
        ]
        for optional in (
            self._psth_bin,
            self._sampling,
            self._time_sync,
            self._trigger_polarity,
            self._trigger_threshold,
        ):
            if optional is not None:
                widgets.append(optional)
        return widgets

    def _emit_changed(self, *_args: Any) -> None:
        if self._updating:
            return
        self._debouncer.request()

    def settings(self) -> ViewerSettings:
        legend = LegendSettings(
            visible=bool(self._legend_visible.isChecked()),
            location=str(self._legend_location.currentData() or "below"),  # type: ignore[arg-type]
            font_size=float(self._legend_font.value()),
            columns=int(self._legend_columns.value()),
            gap=float(self._legend_gap.value()),
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
            max_points_per_curve=int(self._max_points.value()),
            tight_layout=bool(self._settings.style.tight_layout),
        )
        kwargs: dict[str, Any] = {
            "legend": legend,
            "style": style,
            "analysis": self._settings.analysis,
            "x_limits": self._x_limits.value(),
            "trace_ylim": self._trace_ylim.value(),
            "rms_ylim": self._settings.rms_ylim,
            "stim_hp_ylim": self._stim_hp_ylim.value(),
        }
        if self._psth_bin is not None and self._sampling is not None:
            kwargs["psth_bin_window_s"] = float(self._psth_bin.value())
            kwargs["sampling_percent"] = int(self._sampling.value())
        if (
            self._time_sync is not None
            and self._trigger_polarity is not None
            and self._trigger_threshold is not None
        ):
            kwargs["time_sync"] = str(
                self._time_sync.currentData() or "recording_start"
            )
            kwargs["trigger_polarity"] = str(
                self._trigger_polarity.currentData() or "low"
            )
            kwargs["trigger_threshold"] = float(self._trigger_threshold.value())
        return replace(self._settings, **kwargs)

    def sync_base_settings(self, settings: ViewerSettings) -> None:
        """Mettre à jour la base sans écraser légende, style et axes UI."""
        current = self.settings()
        keep: dict[str, Any] = {
            "legend": current.legend,
            "style": current.style,
            "psth_bin_window_s": current.psth_bin_window_s,
            "sampling_percent": current.sampling_percent,
            "x_limits": current.x_limits,
            "trace_ylim": current.trace_ylim,
            # RMS : contrôlé sous la case RMS ; ne pas écraser depuis le parent.
            "rms_ylim": current.rms_ylim,
            "stim_hp_ylim": current.stim_hp_ylim,
        }
        if self._show_sync:
            keep["time_sync"] = current.time_sync
            keep["trigger_polarity"] = current.trigger_polarity
            keep["trigger_threshold"] = current.trigger_threshold
        self._settings = replace(settings, **keep)
