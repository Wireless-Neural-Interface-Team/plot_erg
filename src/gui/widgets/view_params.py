"""Panneau de paramètres d’affichage locaux pour une fenêtre de vue.

Chaque fenêtre ouverte possède sa propre instance : tous ses graphs partagent
ces réglages ; les autres fenêtres restent indépendantes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from display_config import STIM_OFFSET_COLOR, STIM_ONSET_COLOR
from gui.form_widgets import (
    AxisLimitRow,
    ColorSwatchButton,
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
    AnalysisSettings,
    LegendSettings,
    PanelStyle,
    TextOverrides,
    ViewerSettings,
    resolved_scale_bar_time_offset_px,
)

# Réexport pour params_panel et autres imports historiques.
__all__ = ["AxisLimitRow", "LocalViewParams"]


def _scrollable(inner: QWidget) -> FitWidthScrollArea:
    area = FitWidthScrollArea()
    area.setWidget(inner)
    return area


def _collapsible_group(title: str, *, expanded: bool = False) -> tuple[QGroupBox, QWidget]:
    """Groupe à cocher : décoché = contenu masqué."""
    box = QGroupBox(title)
    box.setCheckable(True)
    box.setChecked(expanded)
    box.setObjectName("collapsibleGroup")
    content = QWidget(box)
    outer = QVBoxLayout(box)
    outer.setContentsMargins(6, 14, 6, 6)
    outer.setSpacing(4)
    outer.addWidget(content)

    def _on_toggled(checked: bool) -> None:
        content.setVisible(checked)

    box.toggled.connect(_on_toggled)
    content.setVisible(expanded)
    return box, content


class LocalViewParams(QWidget):
    """Contrôles légende / style / axes propres à une fenêtre.

    Si ``host_tabs`` est fourni, les pages sont ajoutées à ce ``QTabWidget``
    (comme le dock Paramètres de la fenêtre principale). Sinon, tout est
    empilé dans un scroll unique — ou dans des onglets locaux si
    ``show_canal_tab`` est actif.

    Avec ``show_canal_tab`` (fenêtre zoom) :
    - *Canal* — courbes + échelles Y (+ spikes d’affichage)
    - *Affichage* — axe X, sync, disposition, légendes, apparence

    Sans onglet Canal : Y restent dans Affichage (fenêtres détachées simples).
    """

    changed = Signal()

    def __init__(
        self,
        settings: ViewerSettings,
        *,
        show_sync: bool = True,
        show_display_extras: bool = True,
        show_graph_height: bool = True,
        show_canal_tab: bool = False,
        host_tabs: QTabWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = replace(settings)
        self._updating = True
        self._show_sync = bool(show_sync)
        self._show_display_extras = bool(show_display_extras)
        self._show_graph_height = bool(show_graph_height)
        self._show_canal_tab = bool(show_canal_tab)

        # Widgets Canal (créés seulement si show_canal_tab).
        self._curve_wide_box: QGroupBox | None = None
        self._curve_high_box: QGroupBox | None = None
        self._curve_low_box: QGroupBox | None = None
        self._curve_rms_box: QGroupBox | None = None
        self._curve_spikes_box: QGroupBox | None = None
        self._cb_psth: QCheckBox | None = None
        self._cb_trial_rate: QCheckBox | None = None
        self._cb_raster: QCheckBox | None = None
        self._cb_raster_trials: QCheckBox | None = None
        self._cb_isi: QCheckBox | None = None
        self._cb_overlay: QCheckBox | None = None
        self._cb_mark_stims: QCheckBox | None = None

        canal_page = (
            self._build_canal_page(settings) if self._show_canal_tab else None
        )
        display_page = self._build_display_page(settings)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if host_tabs is not None:
            if canal_page is not None:
                host_tabs.addTab(_scrollable(canal_page), "Canal")
                host_tabs.setTabToolTip(
                    host_tabs.count() - 1,
                    "Courbes et échelles Y de cette fenêtre — "
                    "redessin immédiat (indépendant de l’aperçu)",
                )
            host_tabs.addTab(_scrollable(display_page), "Affichage")
            host_tabs.setTabToolTip(
                host_tabs.count() - 1,
                "Axe X, sync, disposition, légendes et apparence — "
                "redessin immédiat (cette fenêtre)",
            )
        elif canal_page is not None:
            tabs = QTabWidget(self)
            tabs.setObjectName("paramsTabs")
            tabs.setDocumentMode(True)
            tabs.setUsesScrollButtons(False)
            tab_bar = tabs.tabBar()
            tab_bar.setElideMode(Qt.TextElideMode.ElideRight)
            tab_bar.setExpanding(True)
            tabs.addTab(_scrollable(canal_page), "Canal")
            tabs.setTabToolTip(
                0,
                "Courbes et échelles Y de cette fenêtre — "
                "redessin immédiat (indépendant de l’aperçu)",
            )
            tabs.addTab(_scrollable(display_page), "Affichage")
            tabs.setTabToolTip(
                1,
                "Axe X, sync, disposition, légendes et apparence — "
                "redessin immédiat (cette fenêtre)",
            )
            root.addWidget(tabs, 1)
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
            elif hasattr(widget, "changed"):
                widget.changed.connect(self._emit_changed)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self._emit_changed)
            elif hasattr(widget, "textChanged"):
                widget.textChanged.connect(self._emit_changed)

        self._updating = False

    def _build_canal_page(self, settings: ViewerSettings) -> QWidget:
        """Courbes + échelles Y (et spikes d’affichage) — scope fenêtre locale."""
        analysis = settings.analysis
        page = QWidget()
        form_layout = QVBoxLayout(page)
        form_layout.setContentsMargins(8, 8, 8, 8)
        form_layout.setSpacing(8)

        hint = QLabel(
            "Cochez les courbes à afficher dans cette fenêtre. "
            "Les échelles Y se règlent dans chaque section. "
            "Indépendant de l’aperçu / du dock Paramètres."
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        form_layout.addWidget(hint)

        self._curve_wide_box, wide_inner = _collapsible_group(
            "WIDE (brut)", expanded=bool(analysis.show_raw)
        )
        wide_form = configure_narrow_form(QFormLayout(wide_inner))
        wide_form.setContentsMargins(0, 0, 0, 0)
        self._raw_ylim = AxisLimitRow(settings.raw_ylim, unit=" µV")
        self._raw_ylim.setToolTip("Plage Y affichée (µV) — panneaux WIDE.")
        wide_form.addRow("Plage Y :", self._raw_ylim)
        self._curve_wide_box.setToolTip(
            "Afficher WIDE dans cette fenêtre (continuous / moyenne / stim)."
        )

        self._curve_high_box, high_inner = _collapsible_group(
            "HIGH (passe-haut)", expanded=bool(analysis.show_hp)
        )
        high_form = configure_narrow_form(QFormLayout(high_inner))
        high_form.setContentsMargins(0, 0, 0, 0)
        self._hp_ylim = AxisLimitRow(settings.hp_ylim, unit=" µV")
        self._hp_ylim.setToolTip("Plage Y affichée (µV) — panneaux HIGH.")
        high_form.addRow("Plage Y :", self._hp_ylim)

        self._curve_low_box, low_inner = _collapsible_group(
            "LOW (passe-bas)", expanded=bool(analysis.show_lp)
        )
        low_form = configure_narrow_form(QFormLayout(low_inner))
        low_form.setContentsMargins(0, 0, 0, 0)
        self._lp_ylim = AxisLimitRow(settings.lp_ylim, unit=" µV")
        self._lp_ylim.setToolTip("Plage Y affichée (µV) — panneaux LOW.")
        low_form.addRow("Plage Y :", self._lp_ylim)

        self._curve_rms_box, rms_inner = _collapsible_group(
            "RMS", expanded=bool(analysis.show_rms)
        )
        rms_form = configure_narrow_form(QFormLayout(rms_inner))
        rms_form.setContentsMargins(0, 0, 0, 0)
        self._rms_ylim = AxisLimitRow(settings.rms_ylim, unit=" µV", step=1.0)
        self._rms_ylim.setToolTip("Plage Y affichée (µV) — panneaux RMS.")
        rms_form.addRow("Plage Y :", self._rms_ylim)

        self._curve_spikes_box, spikes_inner = _collapsible_group(
            "Spikes",
            expanded=bool(
                analysis.show_psth
                or analysis.show_trial_rate
                or analysis.show_raster
                or analysis.show_raster_trials
                or analysis.show_isi
                or analysis.show_overlay
            ),
        )
        spikes_layout = QVBoxLayout(spikes_inner)
        spikes_layout.setContentsMargins(0, 0, 0, 0)
        spikes_layout.setSpacing(6)
        self._cb_psth = QCheckBox("PSTH")
        self._cb_trial_rate = QCheckBox("Firing rate / essai")
        self._cb_raster = QCheckBox("Raster")
        self._cb_raster_trials = QCheckBox("Raster / essai")
        self._cb_isi = QCheckBox("ISI")
        self._cb_overlay = QCheckBox("Spike scope")
        self._cb_psth.setChecked(bool(analysis.show_psth))
        self._cb_trial_rate.setChecked(bool(analysis.show_trial_rate))
        self._cb_raster.setChecked(bool(analysis.show_raster))
        self._cb_raster_trials.setChecked(bool(analysis.show_raster_trials))
        self._cb_isi.setChecked(bool(analysis.show_isi))
        self._cb_overlay.setChecked(bool(analysis.show_overlay))
        for box in (
            self._cb_psth,
            self._cb_trial_rate,
            self._cb_raster,
            self._cb_raster_trials,
            self._cb_isi,
            self._cb_overlay,
        ):
            spikes_layout.addWidget(box)

        if self._show_display_extras:
            spike_form_host = QWidget(spikes_inner)
            spike_form = configure_narrow_form(QFormLayout(spike_form_host))
            spike_form.setContentsMargins(0, 0, 0, 0)
            self._psth_bin = _spin(
                0.001, 10.0, settings.psth_bin_window_s, suffix=" s"
            )
            self._sampling = _int_spin(1, 100, settings.sampling_percent)
            self._sampling.setSuffix(" %")
            spike_form.addRow("Bin PSTH :", self._psth_bin)
            spike_form.addRow("Spikes dessinés :", self._sampling)
            spikes_layout.addWidget(spike_form_host)
        else:
            self._psth_bin = None
            self._sampling = None

        if not any(
            box.isChecked()
            for box in (
                self._curve_wide_box,
                self._curve_high_box,
                self._curve_low_box,
            )
        ):
            self._curve_wide_box.setChecked(True)

        for group in (
            self._curve_wide_box,
            self._curve_high_box,
            self._curve_low_box,
            self._curve_rms_box,
            self._curve_spikes_box,
        ):
            form_layout.addWidget(group)

        foot = QLabel(
            "Filtres / détection → dock Paramètres → Canal → Traitement (F5). "
            "Ici : affichage local de la fenêtre uniquement."
        )
        foot.setObjectName("hintLabel")
        foot.setWordWrap(True)
        form_layout.addWidget(foot)
        form_layout.addStretch(1)
        return page

    def _build_display_page(self, settings: ViewerSettings) -> QWidget:
        page = QWidget()
        form_layout = QVBoxLayout(page)
        form_layout.setContentsMargins(8, 8, 8, 8)
        form_layout.setSpacing(8)

        axis_group = QGroupBox("Plages d’affichage", page)
        axis_form = configure_narrow_form(QFormLayout(axis_group))
        self._x_limits = AxisLimitRow(
            settings.x_limits,
            unit=" s",
            decimals=3,
            step=0.01,
        )
        self._x_limits.setToolTip(
            "Plage de temps visible (fenêtre X, en secondes). Le gain temporel "
            "(s/px) se règle avec Temps barre + Taille temps."
        )
        axis_form.addRow("Plage X (temps) :", self._x_limits)

        if self._show_canal_tab:
            # Y déjà dans Canal — comme le dock Paramètres principal.
            axis_hint = QLabel(
                "Plages Y (WIDE / HIGH / LOW / RMS) → Canal. Recadrent l’affichage ; "
                "le gain µV/px vient d’Amplitude + Taille amp."
            )
            axis_hint.setObjectName("hintLabel")
            axis_hint.setWordWrap(True)
            axis_form.addRow(axis_hint)
        else:
            self._raw_ylim = AxisLimitRow(settings.raw_ylim, unit=" µV")
            self._raw_ylim.setToolTip("Plage Y affichée (µV) — panneaux WIDE.")
            self._hp_ylim = AxisLimitRow(settings.hp_ylim, unit=" µV")
            self._hp_ylim.setToolTip("Plage Y affichée (µV) — panneaux HIGH.")
            self._lp_ylim = AxisLimitRow(settings.lp_ylim, unit=" µV")
            self._lp_ylim.setToolTip("Plage Y affichée (µV) — panneaux LOW.")
            self._rms_ylim = AxisLimitRow(settings.rms_ylim, unit=" µV", step=1.0)
            self._rms_ylim.setToolTip("Plage Y affichée (µV) — panneaux RMS.")
            axis_form.addRow("Plage Y WIDE :", self._raw_ylim)
            axis_form.addRow("Plage Y HIGH :", self._hp_ylim)
            axis_form.addRow("Plage Y LOW :", self._lp_ylim)
            axis_form.addRow("Plage Y RMS :", self._rms_ylim)
        form_layout.addWidget(axis_group)

        layout_group = QGroupBox("Disposition", page)
        self._layout_form = configure_narrow_form(QFormLayout(layout_group))
        layout_form = self._layout_form
        self._graph_height = _int_spin(
            160, 1200, int(settings.graph_height_px), step=20, suffix=" px"
        )
        self._graph_height.setToolTip(
            "Hauteur du panneau si l’amplitude n’est pas en manuel. "
            "Sinon la hauteur suit la valeur et la taille d’amplitude."
        )
        self._show_scale_bars = QCheckBox(
            "Barres flottantes (sinon graduations sur les axes)"
        )
        self._show_scale_bars.setChecked(bool(getattr(settings.style, "show_scale_bars", False)))
        self._show_scale_bars.setToolTip(
            "Temps / Amplitude définissent toujours l’échelle. Coché : barres "
            "flottantes ; décoché : graduations sur les axes."
        )
        style = settings.style
        self._scale_bar_amp_manual = QCheckBox("Manuel")
        self._scale_bar_amp_manual.setChecked(
            bool(getattr(style, "scale_bar_amplitude_manual", False))
            or bool(getattr(style, "scale_bar_amplitude_length_manual", False))
        )
        self._scale_bar_amp_manual.setToolTip(
            "Échelle d’amplitude manuelle : valeur (µV) et taille (px) ensemble. "
            "Coche : la hauteur du graphique suit ces deux valeurs "
            "(ex. 50 µV = 40 px)."
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
            "Amplitude du bâton vertical (ex. 50 µV). "
            "Avec la taille en pixels, fixe la hauteur du graphique."
        )
        _amp_len0 = int(
            round(float(getattr(style, "scale_bar_amplitude_length_px", 40.0) or 40.0))
        )
        self._scale_bar_amp_len_px = _int_spin(
            4, 800, max(4, min(800, _amp_len0)), step=5, suffix=" px"
        )
        self._scale_bar_amp_len_px.setToolTip(
            "Hauteur du bâton en pixels. "
            "Activée avec la case Amplitude. "
            "Avec la valeur, fixe la hauteur du graphique (ex. 50 µV = 40 px)."
        )
        self._scale_bar_time_manual = QCheckBox("Manuel")
        self._scale_bar_time_manual.setChecked(
            bool(getattr(style, "scale_bar_time_manual", False))
            or bool(getattr(style, "scale_bar_time_length_manual", False))
        )
        self._scale_bar_time_manual.setToolTip(
            "Échelle de temps manuelle : durée (ms) et taille (px) ensemble. "
            "Coche : la largeur du graphique suit ces deux valeurs "
            "(ex. 200 ms = 40 px)."
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
            "Durée du bâton horizontal (ex. 200 ms). "
            "Avec la taille en pixels, fixe la largeur du graphique."
        )
        self._scale_bar_time_len_px = _int_spin(
            4,
            800,
            int(round(float(getattr(style, "scale_bar_time_length_px", 40.0) or 40.0))),
            suffix=" px",
        )
        self._scale_bar_time_len_px.setToolTip(
            "Longueur du bâton de temps en pixels. "
            "Activée avec la case Temps. "
            "Avec la durée, fixe la largeur du graphique (ex. 200 ms = 40 px)."
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
        bar_lw = float(getattr(style, "scale_bar_linewidth", 1.4) or 1.4)
        self._scale_bar_linewidth = _spin(
            0.2, 20.0, bar_lw, decimals=2, step=0.1, suffix=" pt"
        )
        self._scale_bar_linewidth.setToolTip(
            "Épaisseur du trait noir des barres d’échelle flottantes "
            "(temps et amplitude)."
        )
        self._scale_bar_dx = _int_spin(
            -400,
            400,
            int(round(float(getattr(style, "scale_bar_dx_px", 0.0) or 0.0))),
            suffix=" px",
        )
        self._scale_bar_dx.setToolTip(
            "Décalage horizontal de l’échelle amplitude (axe vertical). "
            "0 = position par défaut. "
            "Positif décale vers la droite, négatif vers la gauche."
        )
        self._scale_bar_dy = _int_spin(
            -400,
            400,
            int(round(float(getattr(style, "scale_bar_dy_px", 0.0) or 0.0))),
            suffix=" px",
        )
        self._scale_bar_dy.setToolTip(
            "Décalage vertical de l’échelle amplitude (axe vertical). "
            "0 = position par défaut. "
            "Positif monte la barre, négatif la descend."
        )
        _time_dx0, _time_dy0 = resolved_scale_bar_time_offset_px(style)
        self._scale_bar_time_dx = _int_spin(
            -400, 400, int(round(float(_time_dx0))), suffix=" px"
        )
        self._scale_bar_time_dx.setToolTip(
            "Décalage horizontal de l’échelle temps (axe horizontal). "
            "0 = position par défaut. "
            "Positif décale vers la droite, négatif vers la gauche."
        )
        self._scale_bar_time_dy = _int_spin(
            -400, 400, int(round(float(_time_dy0))), suffix=" px"
        )
        self._scale_bar_time_dy.setToolTip(
            "Décalage vertical de l’échelle temps (axe horizontal). "
            "0 = position par défaut. "
            "Positif monte la barre, négatif la descend."
        )
        self._scale_bar_amp_manual.toggled.connect(self._sync_scale_bar_inputs)
        self._scale_bar_time_manual.toggled.connect(self._sync_scale_bar_inputs)
        self._show_scale_bars.toggled.connect(self._sync_scale_bar_inputs)
        self._sync_scale_bar_inputs()
        _max_pts_lo, _max_pts_hi = 500, 200_000
        _max_pts_mid = (_max_pts_lo + _max_pts_hi) // 2
        _max_pts_default = int(settings.style.max_points_per_curve) or _max_pts_mid
        _max_pts_default = max(_max_pts_lo, min(_max_pts_hi, _max_pts_default))
        self._max_points = _int_spin(
            _max_pts_lo, _max_pts_hi, _max_pts_default, step=500
        )
        self._max_points.setToolTip(
            "Points dessinés par courbe (aperçu, montage et fenêtres zoom). "
            "Au-delà, une enveloppe min/max conserve les pics tout en restant rapide."
        )
        max_pts_row = QWidget(page)
        max_pts_lay = QHBoxLayout(max_pts_row)
        max_pts_lay.setContentsMargins(0, 0, 0, 0)
        max_pts_lay.setSpacing(6)
        max_pts_min_btn = QPushButton("Min", max_pts_row)
        max_pts_max_btn = QPushButton("Max", max_pts_row)
        max_pts_min_btn.setToolTip(f"Mettre le minimum ({_max_pts_lo})")
        max_pts_max_btn.setToolTip(f"Mettre le maximum ({_max_pts_hi})")
        max_pts_min_btn.clicked.connect(
            lambda: self._max_points.setValue(_max_pts_lo)
        )
        max_pts_max_btn.clicked.connect(
            lambda: self._max_points.setValue(_max_pts_hi)
        )
        max_pts_lay.addWidget(self._max_points, 1)
        max_pts_lay.addWidget(max_pts_min_btn)
        max_pts_lay.addWidget(max_pts_max_btn)
        if self._show_graph_height:
            layout_form.addRow("Hauteur des graphs :", self._graph_height)
        else:
            self._graph_height.hide()
        layout_form.addRow(self._show_scale_bars)
        layout_form.addRow("Amplitude barre :", amp_row)
        layout_form.addRow("Taille amp. :", self._scale_bar_amp_len_px)
        layout_form.addRow("Temps barre :", time_row)
        layout_form.addRow("Taille temps :", self._scale_bar_time_len_px)
        layout_form.addRow("Épaisseur du trait :", self._scale_bar_linewidth)
        layout_form.addRow("Temps — horizontal :", self._scale_bar_time_dx)
        layout_form.addRow("Temps — vertical :", self._scale_bar_time_dy)
        layout_form.addRow("Amplitude — horizontal :", self._scale_bar_dx)
        layout_form.addRow("Amplitude — vertical :", self._scale_bar_dy)
        layout_form.addRow("Points max / courbe :", max_pts_row)
        form_layout.addWidget(layout_group)

        if self._show_display_extras and not self._show_canal_tab:
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
        elif not self._show_canal_tab:
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
                "ou premier trigger détecté (redessin immédiat)."
            )
            if self._show_canal_tab:
                self._cb_mark_stims = QCheckBox(
                    "Marqueurs de stimulation sur les traces"
                )
                self._cb_mark_stims.setChecked(bool(settings.continuous_mark_stims))
                self._cb_mark_stims.setToolTip(
                    "Marquer les stimulations sur les zooms de cette fenêtre."
                )
                sync_form.addRow("Origine du temps :", self._time_sync)
                sync_form.addRow(self._cb_mark_stims)
                self._trigger_polarity = None
                self._trigger_threshold = None
            else:
                self._trigger_polarity = QComboBox()
                self._trigger_polarity.addItem("Low (front descendant)", "low")
                self._trigger_polarity.addItem("High (front montant)", "high")
                pol_idx = self._trigger_polarity.findData(settings.trigger_polarity)
                self._trigger_polarity.setCurrentIndex(pol_idx if pol_idx >= 0 else 0)
                self._trigger_polarity.setToolTip(
                    "Mémorisé avec le style local. Le front réel de détection se règle "
                    "dans Paramètres → Canal → Traitement (puis Traiter / F5)."
                )
                self._trigger_threshold = _spin(
                    -100.0, 100.0, settings.trigger_threshold, decimals=3, suffix=" V"
                )
                self._trigger_threshold.setToolTip(
                    "Mémorisé avec le style local. Le seuil réel de détection se règle "
                    "dans Paramètres → Canal → Traitement (puis Traiter / F5)."
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
            4.0, 50.0, legend.font_size, legend.face, decimals=1, step=0.5
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
            5.0, 50.0, style.title_font_size, style.title_face, decimals=1, step=0.5
        )
        self._title_dx = _int_spin(
            -400,
            400,
            int(round(float(getattr(style, "title_dx_px", 0.0) or 0.0))),
            suffix=" px",
        )
        self._title_dx.setToolTip(
            "Décalage horizontal du titre de graphique. Positif = droite."
        )
        self._title_dy = _int_spin(
            -400,
            400,
            int(round(float(getattr(style, "title_dy_px", 0.0) or 0.0))),
            suffix=" px",
        )
        self._title_dy.setToolTip(
            "Décalage vertical du titre de graphique. Positif = haut."
        )
        self._label_font = FontSizeFaceRow(
            5.0, 50.0, style.label_font_size, style.label_face, decimals=1, step=0.5
        )
        self._tick_font = FontSizeFaceRow(
            4.0, 50.0, style.tick_font_size, style.tick_face, decimals=1, step=0.5
        )
        self._line_width = _spin(
            0.2, 20.0, style.line_width, decimals=2, step=0.1
        )
        stim_lw = float(getattr(style, "stim_linewidth", 1.0) or 1.0)
        self._stim_linewidth = _spin(0.2, 20.0, stim_lw, decimals=2, step=0.1)
        self._stim_linewidth.setToolTip(
            "Épaisseur (points) des pointillés de début et de fin de stimulation."
        )
        self._stim_onset_color = ColorSwatchButton(
            str(getattr(style, "stim_onset_color", None) or STIM_ONSET_COLOR),
            dialog_title="Couleur du début de stimulation",
        )
        self._stim_onset_color.setToolTip(
            "Couleur des pointillés au début de chaque stimulation."
        )
        self._stim_offset_color = ColorSwatchButton(
            str(getattr(style, "stim_offset_color", None) or STIM_OFFSET_COLOR),
            dialog_title="Couleur de la fin de stimulation",
        )
        self._stim_offset_color.setToolTip(
            "Couleur des pointillés à la fin de chaque stimulation."
        )
        self._text_stim_on = QLineEdit(str(texts.stim_on if texts.stim_on is not None else "on"))
        self._text_stim_on.setPlaceholderText("on")
        self._text_stim_on.setToolTip(
            "Texte au-dessus du trait de début de stimulation. Vide = masquer."
        )
        self._text_stim_off = QLineEdit(
            str(texts.stim_off if texts.stim_off is not None else "off")
        )
        self._text_stim_off.setPlaceholderText("off")
        self._text_stim_off.setToolTip(
            "Texte au-dessus du trait de fin de stimulation. Vide = masquer."
        )
        self._stim_on_font = FontSizeFaceRow(
            4.0,
            50.0,
            float(getattr(style, "stim_on_font_size", 8.0) or 8.0),
            getattr(style, "stim_on_face", None),
            decimals=1,
            step=0.5,
        )
        self._stim_off_font = FontSizeFaceRow(
            4.0,
            50.0,
            float(getattr(style, "stim_off_font_size", 8.0) or 8.0),
            getattr(style, "stim_off_face", None),
            decimals=1,
            step=0.5,
        )
        self._stim_on_dx = _int_spin(
            -400,
            400,
            int(round(float(getattr(style, "stim_on_dx_px", 0.0) or 0.0))),
            suffix=" px",
        )
        self._stim_on_dx.setToolTip(
            "Décalage horizontal du libellé « on ». Positif = droite."
        )
        self._stim_on_dy = _int_spin(
            -400,
            400,
            int(round(float(getattr(style, "stim_on_dy_px", 0.0) or 0.0))),
            suffix=" px",
        )
        self._stim_on_dy.setToolTip(
            "Décalage vertical du libellé « on ». Positif = haut."
        )
        self._stim_off_dx = _int_spin(
            -400,
            400,
            int(round(float(getattr(style, "stim_off_dx_px", 0.0) or 0.0))),
            suffix=" px",
        )
        self._stim_off_dx.setToolTip(
            "Décalage horizontal du libellé « off ». Positif = droite."
        )
        self._stim_off_dy = _int_spin(
            -400,
            400,
            int(round(float(getattr(style, "stim_off_dy_px", 0.0) or 0.0))),
            suffix=" px",
        )
        self._stim_off_dy.setToolTip(
            "Décalage vertical du libellé « off ». Positif = haut."
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
        self._show_stim_labels = QCheckBox("Libellés on / off sur les traits")
        self._show_stim_labels.setChecked(
            bool(getattr(style, "show_stim_labels", True))
        )
        self._show_stim_labels.setToolTip(
            "Afficher les textes « on » et « off » juste sous le haut des traits "
            "de début et de fin de stimulation."
        )
        style_form.addRow("Titre :", self._text_title)
        style_form.addRow("Axe X :", self._text_xlabel)
        style_form.addRow("Axe Y :", self._text_ylabel)
        style_form.addRow(self._text_series_suffix)
        style_form.addRow("Légendes :", self._text_legend_labels)
        style_form.addRow("Police du titre :", self._title_font)
        style_form.addRow("Titre — horizontal :", self._title_dx)
        style_form.addRow("Titre — vertical :", self._title_dy)
        style_form.addRow("Police des axes :", self._label_font)
        style_form.addRow("Police des ticks :", self._tick_font)
        style_form.addRow("Épaisseur de trait :", self._line_width)
        style_form.addRow(self._grid)
        style_form.addRow("Opacité de la grille :", self._grid_alpha)
        style_form.addRow(self._show_borders)
        style_form.addRow(self._ticks_inside)
        style_form.addRow(self._stim_markers)
        style_form.addRow("Épaisseur du trait de stimulation :", self._stim_linewidth)
        style_form.addRow("Couleur début de stimulation :", self._stim_onset_color)
        style_form.addRow("Couleur fin de stimulation :", self._stim_offset_color)
        style_form.addRow(self._show_stim_labels)
        style_form.addRow("Texte début (on) :", self._text_stim_on)
        style_form.addRow("Police début (on) :", self._stim_on_font)
        style_form.addRow("Début — horizontal :", self._stim_on_dx)
        style_form.addRow("Début — vertical :", self._stim_on_dy)
        style_form.addRow("Texte fin (off) :", self._text_stim_off)
        style_form.addRow("Police fin (off) :", self._stim_off_font)
        style_form.addRow("Fin — horizontal :", self._stim_off_dx)
        style_form.addRow("Fin — vertical :", self._stim_off_dy)
        self._show_zone_labels = QCheckBox("Libellés sous les zones de zoom")
        self._show_zone_labels.setChecked(
            bool(getattr(style, "show_zone_labels", True))
        )
        self._show_zone_labels.setToolTip(
            "Afficher « Zone N » (ou le nom édité dans Plages / zooms) "
            "sous chaque bande de zoom, en bas du graphique."
        )
        self._zone_font = FontSizeFaceRow(
            4.0,
            50.0,
            float(getattr(style, "zone_font_size", 8.0) or 8.0),
            getattr(style, "zone_face", None),
            decimals=1,
            step=0.5,
        )
        self._zone_dx = _int_spin(
            -400,
            400,
            int(round(float(getattr(style, "zone_dx_px", 0.0) or 0.0))),
            suffix=" px",
        )
        self._zone_dx.setToolTip(
            "Décalage horizontal du libellé de zone. Positif = droite."
        )
        self._zone_dy = _int_spin(
            -400,
            400,
            int(round(float(getattr(style, "zone_dy_px", 0.0) or 0.0))),
            suffix=" px",
        )
        self._zone_dy.setToolTip(
            "Décalage vertical du libellé de zone. Positif = haut."
        )
        style_form.addRow(self._show_zone_labels)
        style_form.addRow("Police des zones :", self._zone_font)
        style_form.addRow("Zone — horizontal :", self._zone_dx)
        style_form.addRow("Zone — vertical :", self._zone_dy)
        self._show_stim_labels.toggled.connect(self._sync_stim_label_inputs)
        self._sync_stim_label_inputs()
        self._show_zone_labels.toggled.connect(self._sync_zone_label_inputs)
        self._sync_zone_label_inputs()
        form_layout.addWidget(style_group)

    def _sync_scale_bar_inputs(self, *_args: Any) -> None:
        floating = bool(self._show_scale_bars.isChecked())
        # Valeur / taille px : barres flottantes ou graduations sur les axes.
        amp_on = bool(self._scale_bar_amp_manual.isChecked())
        time_on = bool(self._scale_bar_time_manual.isChecked())
        self._scale_bar_amp_manual.setEnabled(True)
        self._scale_bar_time_manual.setEnabled(True)
        self._scale_bar_amp.setEnabled(amp_on)
        self._scale_bar_amp_len_px.setEnabled(amp_on)
        self._scale_bar_time_ms.setEnabled(time_on)
        self._scale_bar_time_len_px.setEnabled(time_on)
        self._scale_bar_linewidth.setEnabled(floating)
        self._scale_bar_dx.setEnabled(floating)
        self._scale_bar_dy.setEnabled(floating)
        self._scale_bar_time_dx.setEnabled(floating)
        self._scale_bar_time_dy.setEnabled(floating)
        amp_len_on = amp_on
        if hasattr(self, "_graph_height") and bool(
            getattr(self, "_show_graph_height", True)
        ):
            show_h = not amp_len_on
            self._graph_height.setVisible(show_h)
            if hasattr(self, "_layout_form"):
                label = self._layout_form.labelForField(self._graph_height)
                if label is not None:
                    label.setVisible(show_h)

    def _sync_stim_label_inputs(self, *_args: Any) -> None:
        enabled = bool(self._show_stim_labels.isChecked())
        for widget in (
            self._text_stim_on,
            self._text_stim_off,
            self._stim_on_font,
            self._stim_off_font,
            self._stim_on_dx,
            self._stim_on_dy,
            self._stim_off_dx,
            self._stim_off_dy,
        ):
            widget.setEnabled(enabled)

    def _sync_zone_label_inputs(self, *_args: Any) -> None:
        enabled = bool(self._show_zone_labels.isChecked())
        for widget in (self._zone_font, self._zone_dx, self._zone_dy):
            widget.setEnabled(enabled)

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
            self._text_stim_on,
            self._text_stim_off,
            self._title_font,
            self._title_dx,
            self._title_dy,
            self._label_font,
            self._tick_font,
            self._stim_on_font,
            self._stim_off_font,
            self._stim_on_dx,
            self._stim_on_dy,
            self._stim_off_dx,
            self._stim_off_dy,
            self._show_zone_labels,
            self._zone_font,
            self._zone_dx,
            self._zone_dy,
            self._line_width,
            self._stim_linewidth,
            self._stim_onset_color,
            self._stim_offset_color,
            self._show_stim_labels,
            self._grid,
            self._grid_alpha,
            self._show_borders,
            self._ticks_inside,
            self._show_scale_bars,
            self._scale_bar_amp_manual,
            self._scale_bar_amp,
            self._scale_bar_amp_len_px,
            self._scale_bar_time_manual,
            self._scale_bar_time_ms,
            self._scale_bar_time_len_px,
            self._scale_bar_linewidth,
            self._scale_bar_dx,
            self._scale_bar_dy,
            self._scale_bar_time_dx,
            self._scale_bar_time_dy,
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
            self._cb_mark_stims,
            self._curve_wide_box,
            self._curve_high_box,
            self._curve_low_box,
            self._curve_rms_box,
            self._curve_spikes_box,
            self._cb_psth,
            self._cb_trial_rate,
            self._cb_raster,
            self._cb_raster_trials,
            self._cb_isi,
            self._cb_overlay,
        ):
            if optional is not None:
                widgets.append(optional)
        return widgets

    def _emit_changed(self, *_args: Any) -> None:
        if self._updating:
            return
        if self._show_canal_tab:
            self._ensure_stream_checked()
            self._ensure_spike_child_checked()
        self._debouncer.request()

    def _ensure_stream_checked(self) -> None:
        """Au moins un flux WIDE/HIGH/LOW doit rester coché."""
        boxes = (
            self._curve_wide_box,
            self._curve_high_box,
            self._curve_low_box,
        )
        if any(b is not None and b.isChecked() for b in boxes):
            return
        if self._curve_wide_box is not None:
            self._updating = True
            try:
                self._curve_wide_box.setChecked(True)
            finally:
                self._updating = False

    def _ensure_spike_child_checked(self) -> None:
        """Si Spikes est coché sans sous-case, activer PSTH (sinon aucun panneau)."""
        if self._curve_spikes_box is None or not self._curve_spikes_box.isChecked():
            return
        children = (
            self._cb_psth,
            self._cb_trial_rate,
            self._cb_raster,
            self._cb_raster_trials,
            self._cb_isi,
            self._cb_overlay,
        )
        if any(c is not None and c.isChecked() for c in children):
            return
        if self._cb_psth is not None:
            self._updating = True
            try:
                self._cb_psth.setChecked(True)
            finally:
                self._updating = False

    def _analysis_from_ui(self) -> AnalysisSettings:
        if not self._show_canal_tab or self._curve_wide_box is None:
            return self._settings.analysis
        show_spikes = bool(
            self._curve_spikes_box.isChecked() if self._curve_spikes_box else False
        )
        return replace(
            self._settings.analysis,
            show_raw=bool(self._curve_wide_box.isChecked()),
            show_hp=bool(
                self._curve_high_box.isChecked() if self._curve_high_box else False
            ),
            show_lp=bool(
                self._curve_low_box.isChecked() if self._curve_low_box else False
            ),
            show_rms=bool(
                self._curve_rms_box.isChecked() if self._curve_rms_box else False
            ),
            show_psth=bool(
                show_spikes and self._cb_psth is not None and self._cb_psth.isChecked()
            ),
            show_trial_rate=bool(
                show_spikes
                and self._cb_trial_rate is not None
                and self._cb_trial_rate.isChecked()
            ),
            show_raster=bool(
                show_spikes
                and self._cb_raster is not None
                and self._cb_raster.isChecked()
            ),
            show_raster_trials=bool(
                show_spikes
                and self._cb_raster_trials is not None
                and self._cb_raster_trials.isChecked()
            ),
            show_isi=bool(
                show_spikes and self._cb_isi is not None and self._cb_isi.isChecked()
            ),
            show_overlay=bool(
                show_spikes
                and self._cb_overlay is not None
                and self._cb_overlay.isChecked()
            ),
        )

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
            self._text_stim_on.setText(str(getattr(texts, "stim_on", "on") or ""))
            self._text_stim_off.setText(str(getattr(texts, "stim_off", "off") or ""))
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
            title_dx_px=float(self._title_dx.value()),
            title_dy_px=float(self._title_dy.value()),
            line_width=float(self._line_width.value()),
            stim_linewidth=float(self._stim_linewidth.value()),
            stim_onset_color=self._stim_onset_color.color_value(),
            stim_offset_color=self._stim_offset_color.color_value(),
            show_stim_labels=bool(self._show_stim_labels.isChecked()),
            stim_on_font_size=self._stim_on_font.size_value(),
            stim_off_font_size=self._stim_off_font.size_value(),
            stim_on_face=self._stim_on_font.face_value(),
            stim_off_face=self._stim_off_font.face_value(),
            stim_on_dx_px=float(self._stim_on_dx.value()),
            stim_on_dy_px=float(self._stim_on_dy.value()),
            stim_off_dx_px=float(self._stim_off_dx.value()),
            stim_off_dy_px=float(self._stim_off_dy.value()),
            show_zone_labels=bool(self._show_zone_labels.isChecked()),
            zone_font_size=self._zone_font.size_value(),
            zone_face=self._zone_font.face_value(),
            zone_dx_px=float(self._zone_dx.value()),
            zone_dy_px=float(self._zone_dy.value()),
            grid=bool(self._grid.isChecked()),
            grid_alpha=float(self._grid_alpha.value()),
            show_borders=bool(self._show_borders.isChecked()),
            show_scale_bars=bool(self._show_scale_bars.isChecked()),
            scale_bar_time_manual=bool(self._scale_bar_time_manual.isChecked()),
            scale_bar_time_s=float(self._scale_bar_time_ms.value()) / 1000.0,
            scale_bar_amplitude_manual=bool(self._scale_bar_amp_manual.isChecked()),
            scale_bar_amplitude=float(self._scale_bar_amp.value()),
            scale_bar_amplitude_length_manual=bool(
                self._scale_bar_amp_manual.isChecked()
            ),
            scale_bar_amplitude_length_px=float(self._scale_bar_amp_len_px.value()),
            scale_bar_time_length_manual=bool(
                self._scale_bar_time_manual.isChecked()
            ),
            scale_bar_time_length_px=float(self._scale_bar_time_len_px.value()),
            scale_bar_linewidth=float(self._scale_bar_linewidth.value()),
            scale_bar_dx_px=float(self._scale_bar_dx.value()),
            scale_bar_dy_px=float(self._scale_bar_dy.value()),
            scale_bar_time_dx_px=float(self._scale_bar_time_dx.value()),
            scale_bar_time_dy_px=float(self._scale_bar_time_dy.value()),
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
            stim_on=self._text_stim_on.text(),
            stim_off=self._text_stim_off.text(),
        )
        analysis = self._analysis_from_ui()
        kwargs: dict[str, Any] = {
            "legend": legend,
            "style": style,
            "texts": texts,
            "analysis": analysis,
            "x_limits": self._x_limits.value(),
            "raw_ylim": self._raw_ylim.value(),
            "hp_ylim": self._hp_ylim.value(),
            "lp_ylim": self._lp_ylim.value(),
            "rms_ylim": self._rms_ylim.value(),
            "graph_height_px": int(self._graph_height.value()),
        }
        if self._show_canal_tab:
            streams: list[str] = []
            if analysis.show_raw:
                streams.append("raw")
            if analysis.show_hp:
                streams.append("hp")
            if analysis.show_lp:
                streams.append("lp")
            if not streams:
                streams = ["raw"]
            kwargs["continuous_streams"] = tuple(streams)
            kwargs["continuous_stream"] = streams[0]
            if self._cb_mark_stims is not None:
                kwargs["continuous_mark_stims"] = bool(
                    self._cb_mark_stims.isChecked()
                )
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
        elif self._time_sync is not None:
            kwargs["time_sync"] = str(
                self._time_sync.currentData() or "recording_start"
            )
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
        if self._show_canal_tab:
            keep["analysis"] = current.analysis
            keep["continuous_streams"] = current.continuous_streams
            keep["continuous_stream"] = current.continuous_stream
            keep["continuous_mark_stims"] = current.continuous_mark_stims
        if self._show_sync:
            keep["time_sync"] = current.time_sync
            if not self._show_canal_tab:
                keep["trigger_polarity"] = current.trigger_polarity
                keep["trigger_threshold"] = current.trigger_threshold
        self._settings = replace(settings, **keep)
