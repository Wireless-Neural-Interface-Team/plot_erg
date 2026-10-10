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
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.form_widgets import (
    AxisLimitRow,
    FitWidthScrollArea,
    FontSizeFaceRow,
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
    TextOverrides,
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

    Si ``host_tabs`` est fourni, la page Affichage (échelles + style) est
    ajoutée à ce ``QTabWidget`` (comme le dock Paramètres de la fenêtre
    principale). Sinon, tout est empilé dans un scroll unique.

    Mode / courbes / spikes → Paramètres → Canal → Traitement
    (mêmes cases pour continuous / moyenne / stimulation).
    Sync / style → Affichage ; détection trigger → Canal (fenêtre principale).
    """

    changed = Signal()

    def __init__(
        self,
        settings: ViewerSettings,
        *,
        show_sync: bool = True,
        show_display_extras: bool = True,
        show_graph_height: bool = True,
        host_tabs: QTabWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = replace(settings)
        self._updating = True
        self._show_sync = bool(show_sync)
        self._show_display_extras = bool(show_display_extras)
        self._show_graph_height = bool(show_graph_height)

        display_page = self._build_display_page(settings)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if host_tabs is not None:
            host_tabs.addTab(_scrollable(display_page), "Affichage")
            host_tabs.setTabToolTip(
                host_tabs.count() - 1,
                "Échelles, sync, disposition, légendes et apparence — "
                "redessin immédiat (cette fenêtre)",
            )
        else:
            scroll = FitWidthScrollArea(self)
            inner = QWidget()
            form_layout = QVBoxLayout(inner)
            form_layout.setContentsMargins(4, 4, 4, 4)
            form_layout.setSpacing(8)
            form_layout.addWidget(display_page)
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
            elif hasattr(widget, "textChanged"):
                widget.textChanged.connect(self._emit_changed)

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
        self._raw_ylim = AxisLimitRow(settings.raw_ylim, unit=" µV")
        self._raw_ylim.setToolTip("Échelle Y des panneaux WIDE.")
        self._hp_ylim = AxisLimitRow(settings.hp_ylim, unit=" µV")
        self._hp_ylim.setToolTip("Échelle Y des panneaux HIGH.")
        self._lp_ylim = AxisLimitRow(settings.lp_ylim, unit=" µV")
        self._lp_ylim.setToolTip("Échelle Y des panneaux LOW.")
        self._rms_ylim = AxisLimitRow(settings.rms_ylim, unit=" µV", step=1.0)
        self._rms_ylim.setToolTip("Échelle Y des panneaux et résumés RMS.")
        axis_form.addRow("Axe X (temps) :", self._x_limits)
        axis_form.addRow("WIDE (µV) :", self._raw_ylim)
        axis_form.addRow("HIGH (µV) :", self._hp_ylim)
        axis_form.addRow("LOW (µV) :", self._lp_ylim)
        axis_form.addRow("RMS (µV) :", self._rms_ylim)
        form_layout.addWidget(axis_group)

        layout_group = QGroupBox("Disposition", page)
        self._layout_form = configure_narrow_form(QFormLayout(layout_group))
        layout_form = self._layout_form
        self._graph_height = _int_spin(
            160, 1200, int(settings.graph_height_px), step=20, suffix=" px"
        )
        self._graph_height.setToolTip(
            "Hauteur fixe de chaque graphique de cette fenêtre."
        )
        self._show_scale_bars = QCheckBox("Échelle flottante (barres temps / amplitude)")
        self._show_scale_bars.setChecked(bool(getattr(settings.style, "show_scale_bars", False)))
        self._show_scale_bars.setToolTip(
            "Masque les graduations numériques sur les bords et affiche une barre "
            "d’échelle flottante (ex. 100 ms · 200 µV)."
        )
        style = settings.style
        self._scale_bar_amp_manual = QCheckBox("Manuel")
        self._scale_bar_amp_manual.setChecked(
            bool(getattr(style, "scale_bar_amplitude_manual", False))
        )
        self._scale_bar_amp = _spin(
            0.01,
            1_000_000.0,
            float(getattr(style, "scale_bar_amplitude", 100.0) or 100.0),
            decimals=2,
            step=10.0,
            suffix=" µV",
        )
        self._scale_bar_amp.setToolTip(
            "Amplitude représentée par le bâton vertical. La longueur pixel "
            "s’adapte automatiquement au zoom / à l’échelle Y."
        )
        self._scale_bar_time_manual = QCheckBox("Manuel")
        self._scale_bar_time_manual.setChecked(
            bool(getattr(style, "scale_bar_time_manual", False))
        )
        # UI en millisecondes (stockage interne en secondes).
        time_s = float(getattr(style, "scale_bar_time_s", 0.1) or 0.1)
        self._scale_bar_time_ms = _spin(
            0.01,
            1_000_000.0,
            time_s * 1000.0,
            decimals=2,
            step=10.0,
            suffix=" ms",
        )
        self._scale_bar_time_ms.setToolTip(
            "Durée représentée par le bâton horizontal. La longueur pixel "
            "s’adapte automatiquement au zoom / à l’échelle X."
        )
        amp_row = QWidget(page)
        amp_lay = QHBoxLayout(amp_row)
        amp_lay.setContentsMargins(0, 0, 0, 0)
        amp_lay.setSpacing(6)
        amp_lay.addWidget(self._scale_bar_amp_manual)
        amp_lay.addWidget(self._scale_bar_amp, 1)
        time_row = QWidget(page)
        time_lay = QHBoxLayout(time_row)
        time_lay.setContentsMargins(0, 0, 0, 0)
        time_lay.setSpacing(6)
        time_lay.addWidget(self._scale_bar_time_manual)
        time_lay.addWidget(self._scale_bar_time_ms, 1)
        self._scale_bar_amp_manual.toggled.connect(self._sync_scale_bar_inputs)
        self._scale_bar_time_manual.toggled.connect(self._sync_scale_bar_inputs)
        self._show_scale_bars.toggled.connect(self._sync_scale_bar_inputs)
        self._sync_scale_bar_inputs()
        self._max_points = _int_spin(
            500, 200000, settings.style.max_points_per_curve, step=500
        )
        self._max_points.setToolTip(
            "Points dessinés par courbe. Au-delà, une enveloppe min/max conserve "
            "les pics tout en restant rapide."
        )
        if self._show_graph_height:
            layout_form.addRow("Hauteur des graphs :", self._graph_height)
        else:
            self._graph_height.hide()
        layout_form.addRow(self._show_scale_bars)
        layout_form.addRow("Amplitude barre :", amp_row)
        layout_form.addRow("Temps barre :", time_row)
        layout_form.addRow("Points max / courbe :", self._max_points)
        form_layout.addWidget(layout_group)

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
            sync_group = QGroupBox("Temps & déclencheur", page)
            sync_form = configure_narrow_form(QFormLayout(sync_group))
            self._time_sync = QComboBox()
            for key, label in TIME_SYNC_LABELS.items():
                self._time_sync.addItem(label, key)
            sync_idx = self._time_sync.findData(settings.time_sync)
            self._time_sync.setCurrentIndex(sync_idx if sync_idx >= 0 else 0)
            self._time_sync.setToolTip(
                "Origine de l’axe temps des traces continues : début du fichier "
                "ou premier trigger détecté. Polarité / seuil ci-dessous "
                "modifient aussi le traitement (F5) de la fenêtre principale."
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
            sync_form.addRow("Origine du temps :", self._time_sync)
            sync_form.addRow("Trigger :", self._trigger_polarity)
            sync_form.addRow("Limite de détection :", self._trigger_threshold)
            form_layout.addWidget(sync_group)
        else:
            self._time_sync = None
            self._trigger_polarity = None
            self._trigger_threshold = None

        self._add_style_groups(form_layout, settings)
        form_layout.addStretch(1)
        return page

    def _add_style_groups(
        self, form_layout: QVBoxLayout, settings: ViewerSettings
    ) -> None:
        legend = settings.legend
        style = settings.style

        legend_group = QGroupBox("Légendes")
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
        self._legend_font = FontSizeFaceRow(
            4.0, 24.0, legend.font_size, legend.face, decimals=1, step=0.5
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
        legend_form.addRow("Police de la légende :", self._legend_font)
        legend_form.addRow("Colonnes :", self._legend_columns)
        legend_form.addRow("Distance au graphique :", self._legend_gap)
        legend_form.addRow(self._legend_frame)
        legend_form.addRow(self._legend_filters)
        legend_form.addRow(self._legend_counts)
        form_layout.addWidget(legend_group)

        texts = settings.texts
        style_group = QGroupBox("Textes et apparence")
        style_form = configure_narrow_form(QFormLayout(style_group))
        self._text_title = QLineEdit(str(texts.title or ""))
        self._text_title.setPlaceholderText("Automatique")
        self._text_title.setToolTip("Laissez vide pour le titre généré automatiquement.")
        self._text_xlabel = QLineEdit(str(texts.xlabel or ""))
        self._text_xlabel.setPlaceholderText("Automatique")
        self._text_ylabel = QLineEdit(str(texts.ylabel or ""))
        self._text_ylabel.setPlaceholderText("Automatique")
        self._text_series_suffix = QCheckBox(
            "Suffixe de type de courbe (moyenne, RMS…)"
        )
        self._text_series_suffix.setChecked(bool(texts.show_series_suffix))
        self._text_series_suffix.setToolTip(
            "Décochez pour n’afficher que le nom d’enregistrement dans la légende."
        )
        self._text_legend_labels = QPlainTextEdit()
        if texts.legend_labels:
            self._text_legend_labels.setPlainText("\n".join(texts.legend_labels))
        self._text_legend_labels.setPlaceholderText(
            "Automatique — une entrée de légende par ligne,\n"
            "dans l’ordre des courbes. Ligne vide = garder l’auto."
        )
        self._text_legend_labels.setMaximumHeight(90)
        self._text_legend_labels.setToolTip(
            "Remplace les libellés de légende. Les noms de base restent "
            "aussi éditables dans le dock Enregistrements → Légende."
        )
        self._title_font = FontSizeFaceRow(
            5.0, 24.0, style.title_font_size, style.title_face, decimals=1, step=0.5
        )
        self._label_font = FontSizeFaceRow(
            5.0, 24.0, style.label_font_size, style.label_face, decimals=1, step=0.5
        )
        self._tick_font = FontSizeFaceRow(
            4.0, 20.0, style.tick_font_size, style.tick_face, decimals=1, step=0.5
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
        self._ticks_inside = QCheckBox("Graduations vers l’intérieur")
        self._ticks_inside.setChecked(bool(style.ticks_inside))
        self._ticks_inside.setToolTip(
            "Oriente les graduations (ticks) vers l’intérieur du cadre de tracé."
        )
        self._stim_markers = QCheckBox("Pointillés de début / fin de stimulation")
        self._stim_markers.setChecked(bool(legend.show_reference_markers))
        self._stim_markers.setToolTip(
            "Lignes en pointillés au début (et à la fin) de stimulation sur les graphiques."
        )
        style_form.addRow("Titre :", self._text_title)
        style_form.addRow("Axe X :", self._text_xlabel)
        style_form.addRow("Axe Y :", self._text_ylabel)
        style_form.addRow(self._text_series_suffix)
        style_form.addRow("Légendes :", self._text_legend_labels)
        style_form.addRow("Police du titre :", self._title_font)
        style_form.addRow("Police des axes :", self._label_font)
        style_form.addRow("Police des ticks :", self._tick_font)
        style_form.addRow("Épaisseur de trait :", self._line_width)
        style_form.addRow(self._grid)
        style_form.addRow("Opacité de la grille :", self._grid_alpha)
        style_form.addRow(self._show_borders)
        style_form.addRow(self._ticks_inside)
        style_form.addRow(self._stim_markers)
        form_layout.addWidget(style_group)

    def _sync_scale_bar_inputs(self, *_args: Any) -> None:
        enabled = bool(self._show_scale_bars.isChecked())
        self._scale_bar_amp_manual.setEnabled(enabled)
        self._scale_bar_time_manual.setEnabled(enabled)
        self._scale_bar_amp.setEnabled(
            enabled and bool(self._scale_bar_amp_manual.isChecked())
        )
        self._scale_bar_time_ms.setEnabled(
            enabled and bool(self._scale_bar_time_manual.isChecked())
        )

    def _watch_widgets(self) -> list[Any]:
        widgets: list[Any] = [
            self._x_limits,
            self._raw_ylim,
            self._hp_ylim,
            self._lp_ylim,
            self._rms_ylim,
            self._legend_visible,
            self._legend_location,
            self._legend_font,
            self._legend_columns,
            self._legend_gap,
            self._legend_frame,
            self._legend_filters,
            self._legend_counts,
            self._text_title,
            self._text_xlabel,
            self._text_ylabel,
            self._text_series_suffix,
            self._text_legend_labels,
            self._title_font,
            self._label_font,
            self._tick_font,
            self._line_width,
            self._grid,
            self._grid_alpha,
            self._show_borders,
            self._ticks_inside,
            self._show_scale_bars,
            self._scale_bar_amp_manual,
            self._scale_bar_amp,
            self._scale_bar_time_manual,
            self._scale_bar_time_ms,
            self._stim_markers,
            self._max_points,
        ]
        if self._show_graph_height:
            widgets.append(self._graph_height)
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

    def apply_texts(self, texts: TextOverrides) -> None:
        """Mettre à jour les champs texte (édition in-place sur le graphique)."""
        self._updating = True
        try:
            self._text_title.setText(str(texts.title or ""))
            self._text_xlabel.setText(str(texts.xlabel or ""))
            self._text_ylabel.setText(str(texts.ylabel or ""))
            self._text_series_suffix.setChecked(bool(texts.show_series_suffix))
            if texts.legend_labels:
                self._text_legend_labels.setPlainText("\n".join(texts.legend_labels))
            else:
                self._text_legend_labels.setPlainText("")
        finally:
            self._updating = False
        self._emit_changed()

    def settings(self) -> ViewerSettings:
        legend = LegendSettings(
            visible=bool(self._legend_visible.isChecked()),
            location=str(self._legend_location.currentData() or "below"),  # type: ignore[arg-type]
            font_size=self._legend_font.size_value(),
            face=self._legend_font.face_value(),
            columns=int(self._legend_columns.value()),
            gap=float(self._legend_gap.value()),
            frame=bool(self._legend_frame.isChecked()),
            show_filter_details=bool(self._legend_filters.isChecked()),
            show_sample_counts=bool(self._legend_counts.isChecked()),
            show_reference_markers=bool(self._stim_markers.isChecked()),
        )
        style = PanelStyle(
            title_font_size=self._title_font.size_value(),
            label_font_size=self._label_font.size_value(),
            tick_font_size=self._tick_font.size_value(),
            title_face=self._title_font.face_value(),
            label_face=self._label_font.face_value(),
            tick_face=self._tick_font.face_value(),
            line_width=float(self._line_width.value()),
            grid=bool(self._grid.isChecked()),
            grid_alpha=float(self._grid_alpha.value()),
            show_borders=bool(self._show_borders.isChecked()),
            show_scale_bars=bool(self._show_scale_bars.isChecked()),
            scale_bar_time_manual=bool(self._scale_bar_time_manual.isChecked()),
            scale_bar_time_s=float(self._scale_bar_time_ms.value()) / 1000.0,
            scale_bar_amplitude_manual=bool(self._scale_bar_amp_manual.isChecked()),
            scale_bar_amplitude=float(self._scale_bar_amp.value()),
            ticks_inside=bool(self._ticks_inside.isChecked()),
            max_points_per_curve=int(self._max_points.value()),
            tight_layout=bool(self._settings.style.tight_layout),
        )
        legend_lines = tuple(
            line.rstrip("\r")
            for line in self._text_legend_labels.toPlainText().split("\n")
        )
        if legend_lines == ("",):
            legend_lines = ()
        texts = TextOverrides(
            title=self._text_title.text().strip(),
            xlabel=self._text_xlabel.text().strip(),
            ylabel=self._text_ylabel.text().strip(),
            legend_labels=legend_lines,
            show_series_suffix=bool(self._text_series_suffix.isChecked()),
        )
        kwargs: dict[str, Any] = {
            "legend": legend,
            "style": style,
            "texts": texts,
            "analysis": self._settings.analysis,
            "x_limits": self._x_limits.value(),
            "raw_ylim": self._raw_ylim.value(),
            "hp_ylim": self._hp_ylim.value(),
            "lp_ylim": self._lp_ylim.value(),
            "rms_ylim": self._rms_ylim.value(),
            "graph_height_px": int(self._graph_height.value()),
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
            "texts": current.texts,
            "psth_bin_window_s": current.psth_bin_window_s,
            "sampling_percent": current.sampling_percent,
            "x_limits": current.x_limits,
            "raw_ylim": current.raw_ylim,
            "hp_ylim": current.hp_ylim,
            "lp_ylim": current.lp_ylim,
            "rms_ylim": current.rms_ylim,
            "graph_height_px": current.graph_height_px,
        }
        if self._show_sync:
            keep["time_sync"] = current.time_sync
            keep["trigger_polarity"] = current.trigger_polarity
            keep["trigger_threshold"] = current.trigger_threshold
        self._settings = replace(settings, **keep)
