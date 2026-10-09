"""Paramètres d’affichage et de traitement, selon le coût d’un changement.

- *Affichage* / *Légende & style* → :attr:`viewChanged` (redessin immédiat).
- *Canal* (section Pipeline) → :attr:`configChanged` (recalcul F5 ; cache par étapes).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Mapping

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

ViewMode = Literal["preview", "montage"]

from config import AnalysisConfig
from gui.form_widgets import (
    AxisLimitRow,
    FitWidthScrollArea,
    configure_narrow_form,
    make_double_spin as _spin,
    make_int_spin as _int_spin,
)
from gui.jobs import Debouncer
from view_config import (
    EDGE_TO_TRIGGER_POLARITY,
    LEGEND_LOCATIONS,
    TIME_SYNC_LABELS,
    TRIGGER_POLARITY_TO_EDGE,
    AnalysisStream,
    AxisLimits,
    LegendSettings,
    PanelStyle,
    ViewerSettings,
)

_EDGE_CHOICES = (
    ("Trigger Low — front descendant (ANALOG-IN-0)", "falling"),
    ("Trigger High — front montant (ANALOG-IN-0)", "rising"),
    ("Sans déclencheur (sections fixes)", "none"),
)
_TIME_SYNC_CHOICES = (
    (TIME_SYNC_LABELS["recording_start"], "recording_start"),
    (TIME_SYNC_LABELS["trigger"], "trigger"),
)
_TRIGGER_POLARITY_CHOICES = (
    ("Low (front descendant)", "low"),
    ("High (front montant)", "high"),
)
_FILTER_TYPES = (("Bessel", "bessel"), ("Butterworth", "butterworth"))
_NOTCH_CHOICES = (
    ("Désactivé", 0),
    ("50 Hz", 50),
    ("60 Hz", 60),
)
_THRESHOLD_MODES = (("Fixe (µV)", "fixed"), ("Multiple du RMS", "rms_multiple"))
_POLARITIES = (("Négative", "negative"), ("Positive", "positive"))
_SECTION_SPECS = (("Nombre de sections", "count"), ("Durée de section (s)", "duration"))


def _choice(options: tuple[tuple[str, Any], ...], value: Any) -> QComboBox:
    box = QComboBox()
    # Ne pas forcer la largeur min du dock sur le libellé le plus long.
    box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(12)
    box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    for label, key in options:
        box.addItem(label, key)
    index = box.findData(value)
    box.setCurrentIndex(index if index >= 0 else 0)
    return box


def _scrollable(inner: QWidget) -> FitWidthScrollArea:
    """Onglet Paramètres : contenu calé sur la largeur visible (pas de débordement droit)."""
    area = FitWidthScrollArea()
    area.setWidget(inner)
    return area


def _collapsible_group(title: str, *, expanded: bool = False) -> tuple[QGroupBox, QWidget]:
    """Groupe à cocher : décoché = contenu masqué (disclosure progressive)."""
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


def _set_form_row_visible(form: QFormLayout, field: QWidget, visible: bool) -> None:
    """Affiche / masque une ligne QFormLayout (libellé + champ)."""
    field.setVisible(visible)
    label = form.labelForField(field)
    if label is not None:
        label.setVisible(visible)


class ParamsPanel(QWidget):
    """Réglages d’affichage et de traitement dans le dock Paramètres."""

    viewChanged = Signal()
    configChanged = Signal()
    processRequested = Signal()
    # Flux WIDE/HIGH/LOW cochés ou décochés (pour déplier le Pipeline dans Canal).
    streamsChanged = Signal(object)

    def __init__(self, defaults: Mapping[str, Any] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._defaults = dict(defaults or {})
        self._loading = True
        self._view_mode: ViewMode = "preview"
        # Ignored horizontal : le dock impose la largeur. Preferred/Expanding
        # faisait « aspirer » le layout vers sizeHint et écrasait le panneau au drag.
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("paramsTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(False)
        tab_bar = self.tabs.tabBar()
        # Partage équitable + élision : 3 onglets lisibles dès ~220 px.
        tab_bar.setElideMode(Qt.TextElideMode.ElideRight)
        tab_bar.setExpanding(True)
        self._channel_tab_index: int | None = None
        self._channel_side: QWidget | None = None
        # Contenu extrait du side_panel (évite scroll imbriqué dans Canal).
        self._channel_side_content: QWidget | None = None
        self._channel_placeholder = QLabel(
            "Sélectionnez un canal traité pour afficher "
            "mode, courbes, plages, contexte et résumés ici."
        )
        self._channel_placeholder.setObjectName("hintLabel")
        self._channel_placeholder.setWordWrap(True)
        self._channel_placeholder.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._channel_placeholder.setContentsMargins(0, 0, 0, 4)

        # Onglet Canal = aperçu (mode/courbes/plages) + Pipeline F5.
        self._channel_host = QWidget()
        self._channel_host_layout = QVBoxLayout(self._channel_host)
        self._channel_host_layout.setContentsMargins(0, 0, 0, 0)
        self._channel_host_layout.setSpacing(0)
        self._channel_host_layout.addWidget(self._channel_placeholder)

        self._pipeline_header = QLabel("Pipeline (Traiter F5)")
        self._pipeline_header.setObjectName("sectionLabel")
        self._pipeline_header.setWordWrap(True)
        self._pipeline_sep = QFrame()
        self._pipeline_sep.setFrameShape(QFrame.Shape.HLine)
        self._pipeline_sep.setObjectName("sectionSeparator")

        # Affichage avant Pipeline : _sync_section_rows lie trigger Affichage ↔ edge.
        display_page = self._build_display_tab()
        legend_page = self._build_legend_tab()
        self._processing_content = self._build_processing_tab()
        canal_inner = QWidget()
        canal_layout = QVBoxLayout(canal_inner)
        canal_layout.setContentsMargins(8, 8, 8, 8)
        canal_layout.setSpacing(8)
        canal_layout.addWidget(self._channel_host)
        canal_layout.addWidget(self._pipeline_sep)
        canal_layout.addWidget(self._pipeline_header)
        canal_layout.addWidget(self._processing_content)
        canal_layout.addStretch(1)
        self._canal_scroll = _scrollable(canal_inner)

        self.tabs.addTab(self._canal_scroll, "Canal")
        self._channel_tab_index = 0
        self.tabs.addTab(_scrollable(display_page), "Affichage")
        self.tabs.addTab(_scrollable(legend_page), "Style")
        self.tabs.setTabToolTip(
            0,
            "Mode aperçu, courbes, plages — et Pipeline (filtres, spikes, RMS) → Traiter (F5)",
        )
        self.tabs.setTabToolTip(
            1, "Flux WIDE/HIGH/LOW, échelles, sync, montages — redessin immédiat"
        )
        self.tabs.setTabToolTip(2, "Légendes et style des panneaux — redessin immédiat")

        dirty_row = QWidget(self)
        dirty_layout = QHBoxLayout(dirty_row)
        dirty_layout.setContentsMargins(6, 4, 6, 6)
        dirty_layout.setSpacing(6)
        self._dirty_label = QLabel("")
        self._dirty_label.setObjectName("warningLabel")
        self._dirty_label.setWordWrap(True)
        self._btn_apply_process = QPushButton("Traiter (F5)", dirty_row)
        self._btn_apply_process.setObjectName("primaryButton")
        self._btn_apply_process.setToolTip("Recalculer avec les nouveaux paramètres de traitement")
        self._btn_apply_process.clicked.connect(self.processRequested.emit)
        dirty_layout.addWidget(self._dirty_label, 1)
        dirty_layout.addWidget(self._btn_apply_process, 0)
        dirty_row.setVisible(False)
        self._dirty_row = dirty_row

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self._dirty_row)

        self._view_debouncer = Debouncer(120, self)
        self._view_debouncer.triggered.connect(self.viewChanged.emit)
        self._apply_view_mode_visibility()
        self._loading = False

    def sizeHint(self) -> QSize:  # noqa: D102
        # Suivre la largeur courante : évite le snap du dock vers 320/431 px.
        if self.isVisible() and self.width() > 0:
            return QSize(self.width(), max(200, self.height()))
        return QSize(320, 640)

    def minimumSizeHint(self) -> QSize:  # noqa: D102
        return QSize(200, 200)

    # ----------------------------------------------------------- display tab

    def _build_display_tab(self) -> QWidget:
        defaults = self._defaults
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(8)

        # Flux visibles sur l’aperçu continuous et la revue montage.
        streams_group = QGroupBox("Traces continues")
        streams_layout = QVBoxLayout(streams_group)
        streams_layout.setContentsMargins(8, 8, 8, 8)
        streams_layout.setSpacing(4)
        self._cb_stream_wide = QCheckBox("WIDE (brut)")
        self._cb_stream_high = QCheckBox("HIGH (passe-haut)")
        self._cb_stream_low = QCheckBox("LOW (passe-bas)")
        self._cb_stream_wide.setChecked(True)
        self._cb_stream_wide.setToolTip(
            "Signal brut (wideband) dans l’aperçu continuous. "
            "Coche → Canal / Pipeline : notch / artefacts."
        )
        self._cb_stream_high.setToolTip(
            "Passe-haut dans l’aperçu continuous. "
            "Coche → Canal / Pipeline : type / ordre / coupure (propre au HIGH)."
        )
        self._cb_stream_low.setToolTip(
            "Passe-bas dans l’aperçu continuous. "
            "Coche → Canal / Pipeline : type / ordre / coupure (propre au LOW)."
        )
        self._cb_mark_stims = QCheckBox("Marqueurs de stimulation")
        self._cb_mark_stims.setChecked(True)
        self._cb_mark_stims.setToolTip(
            "Marquer les stimulations sur l’aperçu continuous et la revue montage."
        )
        for box in (
            self._cb_stream_wide,
            self._cb_stream_high,
            self._cb_stream_low,
            self._cb_mark_stims,
        ):
            streams_layout.addWidget(box)
        for box in (self._cb_stream_wide, self._cb_stream_high, self._cb_stream_low):
            box.toggled.connect(self._on_streams_toggled)
        self._cb_mark_stims.toggled.connect(self._emit_view)

        sync_group = QGroupBox("Synchronisation / déclencheur")
        sync_form = configure_narrow_form(QFormLayout(sync_group))
        default_edge = str(defaults.get("default_edge", "falling"))
        default_polarity = EDGE_TO_TRIGGER_POLARITY.get(default_edge, "low")
        self._time_sync = _choice(
            _TIME_SYNC_CHOICES,
            str(defaults.get("default_time_sync", "recording_start")),
        )
        self._time_sync.setToolTip(
            "Origine de l’axe temps des traces continues : début du fichier "
            "ou premier trigger détecté (t=0)."
        )
        self._trigger_polarity = _choice(_TRIGGER_POLARITY_CHOICES, default_polarity)
        self._trigger_polarity.setToolTip(
            "Polarité TTL sur ANALOG-IN-0. Low = front descendant, "
            "High = front montant. Modifie aussi le traitement (F5)."
        )
        self._display_trigger_threshold = _spin(
            -100.0,
            100.0,
            defaults.get("default_threshold", 1.0),
            decimals=3,
            suffix=" V",
        )
        self._display_trigger_threshold.setToolTip(
            "Seuil de détection du trigger sur ANALOG-IN-0 (V). "
            "Modifie aussi le traitement (F5)."
        )
        sync_form.addRow("Synchroniser sur :", self._time_sync)
        sync_form.addRow("Trigger :", self._trigger_polarity)
        sync_form.addRow("Limite de détection :", self._display_trigger_threshold)

        # Bin PSTH / échantillonnage spikes : défauts (réglables dans Analyse).
        self._psth_bin_s = float(defaults.get("default_psth_bin_window_s", 0.025))
        self._sampling_percent = int(defaults.get("default_sampling_percent", 100) or 100)

        axis_group = QGroupBox("Échelles des axes")
        self._axis_form = configure_narrow_form(QFormLayout(axis_group))
        self._x_limits = AxisLimitRow(
            AxisLimits(enabled=False, minimum=-0.1, maximum=0.4),
            unit=" s",
            decimals=3,
            step=0.01,
        )
        self._x_limits.setToolTip(
            "Limites X manuelles (temps, s) pour tous les graphs temporels. "
            "Prioritaire sur la section / le zoom par défaut."
        )
        self._stim_hp_ylim = AxisLimitRow(
            AxisLimits(
                enabled=bool(defaults.get("default_first_trigger_hp_ylim_enabled", False)),
                minimum=float(defaults.get("default_first_trigger_hp_ylim_min_uv", -200.0)),
                maximum=float(defaults.get("default_first_trigger_hp_ylim_max_uv", 200.0)),
            )
        )
        self._stim_hp_ylim.setToolTip(
            "Échelle Y des panneaux passe-haut autour d’une stimulation "
            "(aperçu / Analyse). Sans effet sur la revue montage continue."
        )
        # Échelle Y RMS : sous la case RMS (onglet Canal), pas ici.
        self._rms_ylim = AxisLimits(enabled=True, minimum=0.0, maximum=20.0)
        self._trace_ylim = AxisLimitRow(AxisLimits())
        self._trace_ylim.setToolTip(
            "Échelle Y des traces continues (aperçu canal et revue montage)."
        )
        self._axis_form.addRow("Axe X (temps) :", self._x_limits)
        self._axis_form.addRow("Panneaux de traces :", self._trace_ylim)
        self._axis_form.addRow("Passe-haut stimulation :", self._stim_hp_ylim)

        # Revue montage continue uniquement.
        self._montage_review_group = QGroupBox("Revue montage")
        review_form = configure_narrow_form(QFormLayout(self._montage_review_group))
        self._montage_row_height = _int_spin(
            36,
            200,
            int(defaults.get("default_montage_row_min_height_px", 52) or 52),
            suffix=" px",
        )
        self._montage_row_height.setToolTip(
            "Hauteur minimale d’une ligne canal×flux dans la revue montage. "
            "Plus bas = ouverture plus rapide, moins de détail vertical."
        )
        review_form.addRow("Hauteur de ligne :", self._montage_row_height)
        review_hint = QLabel(
            "Visibilité des canaux : Session → Channels (cases). "
            "Retour aperçu : Ctrl+Shift+M."
        )
        review_hint.setObjectName("hintLabel")
        review_hint.setWordWrap(True)
        review_form.addRow(review_hint)

        # Montages PDF (moyenne / 2e stim) — hors revue continue.
        montage_group, montage_inner = _collapsible_group(
            "Montages PDF (moyenne / 2e stim)", expanded=False
        )
        self._montage_pdf_group = montage_group
        montage_form = configure_narrow_form(QFormLayout(montage_inner))
        montage_form.setContentsMargins(0, 0, 0, 0)
        self._montage_channels = _int_spin(2, 1024, 12)
        self._montage_channels.setToolTip(
            "Canaux empilés par page pour les montages PDF moyenne / 2e stim. "
            "La revue continue affiche toujours tous les canaux visibles."
        )
        self._montage_page = _int_spin(0, 64, 0)
        self._montage_page.setToolTip(
            "Bloc de canaux pour les montages PDF (0 = première page). "
            "Sans effet sur la revue montage continue."
        )
        montage_form.addRow("Canaux par montage :", self._montage_channels)
        montage_form.addRow("Page de montage :", self._montage_page)

        for group in (
            streams_group,
            sync_group,
            axis_group,
            self._montage_review_group,
            montage_group,
        ):
            page_layout.addWidget(group)
        page_layout.addStretch(1)

        for int_widget in (
            self._montage_channels,
            self._montage_page,
            self._montage_row_height,
        ):
            int_widget.valueChanged.connect(lambda _v: self._emit_view())
        for row in (self._x_limits, self._stim_hp_ylim, self._trace_ylim):
            row.changed.connect(self._emit_view)
        self._time_sync.currentIndexChanged.connect(lambda _i: self._emit_view())
        self._trigger_polarity.currentIndexChanged.connect(
            lambda _i: self._on_display_trigger_changed()
        )
        self._display_trigger_threshold.valueChanged.connect(
            lambda _v: self._on_display_trigger_changed()
        )
        return page

    # ------------------------------------------------------------ legend tab

    def _build_legend_tab(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(8)

        legend_group = QGroupBox("Légendes")
        legend_form = configure_narrow_form(QFormLayout(legend_group))
        self._legend_visible = QCheckBox("Afficher les légendes")
        self._legend_visible.setChecked(True)
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
                "Sous le panneau" if location == "below" else location.capitalize(), location
            )
        self._legend_font = _spin(4.0, 24.0, 9.0, decimals=1, step=0.5, suffix=" pt")
        self._legend_columns = _int_spin(1, 6, 1)
        self._legend_gap = _spin(0.0, 1.0, 0.22, decimals=2, step=0.02)
        self._legend_gap.setToolTip(
            "Écart entre la légende et le graphique (fraction de la hauteur des axes)."
        )
        self._legend_frame = QCheckBox("Cadre autour des légendes")
        self._legend_frame.setChecked(True)
        self._legend_filters = QCheckBox("Détails du filtre (type, ordre, coupure)")
        self._legend_filters.setChecked(True)
        self._legend_counts = QCheckBox("Compteurs d’échantillons / spikes")
        self._legend_counts.setChecked(True)
        legend_form.addRow(self._legend_visible)
        legend_form.addRow("Position :", self._legend_location)
        legend_form.addRow("Taille de police :", self._legend_font)
        legend_form.addRow("Colonnes :", self._legend_columns)
        legend_form.addRow("Distance au graphique :", self._legend_gap)
        legend_form.addRow(self._legend_frame)
        legend_form.addRow(self._legend_filters)
        legend_form.addRow(self._legend_counts)

        style_group = QGroupBox("Style des panneaux")
        style_form = configure_narrow_form(QFormLayout(style_group))
        self._title_font = _spin(5.0, 24.0, 10.0, decimals=1, step=0.5, suffix=" pt")
        self._label_font = _spin(5.0, 24.0, 9.0, decimals=1, step=0.5, suffix=" pt")
        self._tick_font = _spin(4.0, 20.0, 8.0, decimals=1, step=0.5, suffix=" pt")
        self._line_width = _spin(0.2, 5.0, 1.2, decimals=2, step=0.1)
        self._grid = QCheckBox("Afficher la grille")
        self._grid.setChecked(True)
        self._grid_alpha = _spin(0.0, 1.0, 0.3, decimals=2, step=0.05)
        self._show_borders = QCheckBox("Afficher les bordures du graphique")
        self._show_borders.setChecked(True)
        self._show_borders.setToolTip(
            "Cadre avec graduations autour de la zone de tracé de chaque panneau."
        )
        self._stim_markers = QCheckBox("Afficher les pointillés de stimulation")
        self._stim_markers.setChecked(True)
        self._stim_markers.setToolTip(
            "Lignes en pointillés au début (et à la fin) de stimulation sur les graphiques."
        )
        self._max_points = _int_spin(500, 200000, 6000, step=500)
        self._max_points.setToolTip(
            "Points dessinés par courbe. Au-delà, une enveloppe min/max conserve "
            "tous les pics tout en restant rapide."
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

        page_layout.addWidget(legend_group)
        page_layout.addWidget(style_group)
        page_layout.addStretch(1)

        for box in (
            self._legend_visible,
            self._legend_frame,
            self._legend_filters,
            self._legend_counts,
            self._grid,
            self._show_borders,
            self._stim_markers,
        ):
            box.toggled.connect(lambda _c: self._emit_view())
        self._legend_location.currentIndexChanged.connect(lambda _i: self._emit_view())
        for spin in (
            self._legend_font,
            self._legend_gap,
            self._title_font,
            self._label_font,
            self._tick_font,
            self._line_width,
            self._grid_alpha,
        ):
            spin.valueChanged.connect(lambda _v: self._emit_view())
        for int_spin in (self._legend_columns, self._max_points):
            int_spin.valueChanged.connect(lambda _v: self._emit_view())
        return page

    # -------------------------------------------------------- processing tab

    def _build_processing_tab(self) -> QWidget:
        """Contenu Pipeline (embarqué sous l’onglet Canal)."""
        defaults = self._defaults
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)

        trigger_group = QGroupBox("Détection des stimulations")
        trigger_form = configure_narrow_form(QFormLayout(trigger_group))
        self._edge = _choice(_EDGE_CHOICES, str(defaults.get("default_edge", "falling")))
        self._threshold = _spin(
            -100.0, 100.0, defaults.get("default_threshold", 1.0), decimals=3, suffix=" V"
        )
        self._pre_s = _spin(0.0, 600.0, defaults.get("default_pre_s", 2.0), suffix=" s")
        self._post_s = _spin(0.001, 3600.0, defaults.get("default_post_s", 10.0), suffix=" s")
        self._section_spec = _choice(
            _SECTION_SPECS, str(defaults.get("default_section_spec", "count"))
        )
        self._section_count = _int_spin(
            1, 10000, int(defaults.get("default_section_count", 10) or 10)
        )
        self._section_duration = _spin(
            0.001, 3600.0, defaults.get("default_section_duration_s") or 10.0, suffix=" s"
        )
        self._section_start = _spin(
            0.0, 3600.0, defaults.get("default_section_trigger_start_s", 1.0), suffix=" s"
        )
        self._section_end = _spin(
            0.0, 3600.0, defaults.get("default_section_trigger_end_s", 4.0), suffix=" s"
        )
        trigger_form.addRow("Front de déclenchement :", self._edge)
        trigger_form.addRow("Seuil de déclenchement :", self._threshold)
        trigger_form.addRow("Fenêtre avant :", self._pre_s)
        trigger_form.addRow("Fenêtre après :", self._post_s)
        self._section_spec.currentIndexChanged.connect(lambda _i: self._sync_section_rows())
        self._edge.currentIndexChanged.connect(lambda _i: self._on_processing_edge_changed())

        # Mode sans déclencheur : rarement utilisé → replié par défaut.
        self._sections_box, sections_inner = _collapsible_group(
            "Mode sans déclencheur (avancé)", expanded=False
        )
        sections_form = configure_narrow_form(QFormLayout(sections_inner))
        sections_form.setContentsMargins(0, 0, 0, 0)
        sections_form.addRow("Découpage :", self._section_spec)
        sections_form.addRow("Nombre de sections :", self._section_count)
        sections_form.addRow("Durée de section :", self._section_duration)
        sections_form.addRow("Début virtuel :", self._section_start)
        sections_form.addRow("Fin virtuelle :", self._section_end)

        curves_hint = QLabel(
            "Cochez une courbe pour afficher ses paramètres. "
            "Les cases n’activent que le panneau (le pipeline F5 reste complet)."
        )
        curves_hint.setObjectName("hintLabel")
        curves_hint.setWordWrap(True)

        # ---- WIDE : prétraitement wideband (notch / artefacts) ----
        self._curve_wide_box, wide_inner = _collapsible_group(
            "WIDE (brut)", expanded=True
        )
        wide_form = configure_narrow_form(QFormLayout(wide_inner))
        wide_form.setContentsMargins(0, 0, 0, 0)
        notch_default = int(defaults.get("default_software_notch_hz", 0) or 0)
        self._software_notch = _choice(_NOTCH_CHOICES, notch_default)
        self._software_notch.setToolTip(
            "Notch logiciel appliqué sur le wideband avant HP/LP (bruit secteur). "
            "Indépendant du notch éventuellement déjà enregistré dans le .rhs."
        )
        self._artifact_enabled = QCheckBox("Supprimer les artefacts de stimulation")
        self._artifact_enabled.setChecked(True)
        self._artifact_threshold = _spin(
            1.0, 100000.0, 2500.0, decimals=1, step=50.0, suffix=" µV"
        )
        wide_form.addRow("Notch logiciel :", self._software_notch)
        wide_form.addRow(self._artifact_enabled)
        wide_form.addRow("Seuil d’artefact :", self._artifact_threshold)

        # ---- HIGH : filtre passe-haut (indépendant de LOW) ----
        self._curve_high_box, high_inner = _collapsible_group(
            "HIGH (passe-haut)", expanded=False
        )
        high_form = configure_narrow_form(QFormLayout(high_inner))
        high_form.setContentsMargins(0, 0, 0, 0)
        self._hp_filter_type = _choice(
            _FILTER_TYPES, str(defaults.get("default_intan_hp_filter_type", "bessel"))
        )
        self._hp_filter_order = _int_spin(
            1, 8, int(defaults.get("default_intan_hp_filter_order", 2) or 2)
        )
        self._hp_filter_cutoff = _spin(
            0.1,
            20000.0,
            defaults.get("default_intan_hp_filter_cutoff_hz", 250.0),
            decimals=2,
            step=10.0,
            suffix=" Hz",
        )
        self._hp_filter_cutoff.setToolTip(
            "Fréquence de coupure du passe-haut (HIGH). Indépendante du LOW."
        )
        high_form.addRow("Type de filtre :", self._hp_filter_type)
        high_form.addRow("Ordre du filtre :", self._hp_filter_order)
        high_form.addRow("Coupure :", self._hp_filter_cutoff)

        # ---- LOW : filtre passe-bas (indépendant de HIGH) ----
        self._curve_low_box, low_inner = _collapsible_group(
            "LOW (passe-bas)", expanded=False
        )
        low_form = configure_narrow_form(QFormLayout(low_inner))
        low_form.setContentsMargins(0, 0, 0, 0)
        self._lp_filter_type = _choice(
            _FILTER_TYPES, str(defaults.get("default_intan_lp_filter_type", "bessel"))
        )
        self._lp_filter_order = _int_spin(
            1, 8, int(defaults.get("default_intan_lp_filter_order", 2) or 2)
        )
        self._lp_filter_cutoff = _spin(
            0.1,
            20000.0,
            defaults.get("default_intan_lp_filter_cutoff_hz", 250.0),
            decimals=2,
            step=10.0,
            suffix=" Hz",
        )
        self._lp_filter_cutoff.setToolTip(
            "Fréquence de coupure du passe-bas (LOW). Indépendante du HIGH."
        )
        low_form.addRow("Type de filtre :", self._lp_filter_type)
        low_form.addRow("Ordre du filtre :", self._lp_filter_order)
        low_form.addRow("Coupure :", self._lp_filter_cutoff)

        # ---- Spikes : détection sur le flux HIGH ----
        self._curve_spikes_box, spikes_inner = _collapsible_group(
            "Spikes", expanded=False
        )
        spike_form = configure_narrow_form(QFormLayout(spikes_inner))
        spike_form.setContentsMargins(0, 0, 0, 0)
        self._threshold_mode = _choice(
            _THRESHOLD_MODES, str(defaults.get("default_spike_threshold_mode", "fixed"))
        )
        self._spike_threshold = _spin(
            0.1,
            100000.0,
            abs(float(defaults.get("default_spike_threshold_uv", 70.0))),
            decimals=2,
            step=5.0,
            suffix=" µV",
        )
        self._polarity = _choice(
            _POLARITIES, str(defaults.get("default_spike_threshold_polarity", "negative"))
        )
        self._rms_multiplier = _spin(
            0.1,
            100.0,
            defaults.get("default_spike_threshold_rms_multiplier", 4.0),
            decimals=2,
            step=0.5,
        )
        self._overlay_pre = _spin(
            0.0, 500.0, defaults.get("default_spike_overlay_pre_ms", 2.0), decimals=2, suffix=" ms"
        )
        self._overlay_post = _spin(
            0.01,
            500.0,
            defaults.get("default_spike_overlay_post_ms", 4.0),
            decimals=2,
            suffix=" ms",
        )
        for widget in (self._overlay_pre, self._overlay_post):
            widget.setToolTip(
                "Fenêtre de forme d’onde autour de chaque détection. La modifier "
                "ré-extrait les snippets (rapide : les flux filtrés restent en cache)."
            )
        spike_form.addRow("Mode de seuil :", self._threshold_mode)
        spike_form.addRow("Seuil fixe :", self._spike_threshold)
        spike_form.addRow("Polarité :", self._polarity)
        spike_form.addRow("Multiplicateur RMS :", self._rms_multiplier)
        spike_form.addRow("Avant le spike :", self._overlay_pre)
        spike_form.addRow("Après le spike :", self._overlay_post)
        self._threshold_mode.currentIndexChanged.connect(lambda _i: self._sync_threshold_rows())

        # ---- RMS ----
        self._curve_rms_box, rms_inner = _collapsible_group("RMS", expanded=False)
        rms_form = configure_narrow_form(QFormLayout(rms_inner))
        rms_form.setContentsMargins(0, 0, 0, 0)
        self._rms_window = _spin(
            0.001, 60.0, defaults.get("default_rms_window_s", 1.0), suffix=" s"
        )
        self._rms_window.setToolTip("Fenêtre glissante utilisée pour les profils RMS.")
        rms_form.addRow("Fenêtre RMS :", self._rms_window)

        # Mapping MEA : uniquement dans Session → Mapping (évite la double saisie).
        # Champ masqué pour sync API / build_config.
        self._probe_edit = QLineEdit(str(defaults.get("default_probe_layout_json") or ""))
        self._probe_edit.hide()

        resources_box, resources_inner = _collapsible_group("Ressources", expanded=False)
        resources_form = configure_narrow_form(QFormLayout(resources_inner))
        resources_form.setContentsMargins(0, 0, 0, 0)
        map_hint = QLabel("Mapping MEA : Session → Mapping MEA (pas ici).")
        map_hint.setObjectName("hintLabel")
        map_hint.setWordWrap(True)
        self._work_edit = QLineEdit("")
        self._work_edit.setPlaceholderText("Par défaut : à côté du fichier .rhs")
        work_button = QPushButton("Parcourir…")
        work_button.clicked.connect(self._browse_work_dir)
        work_row = QWidget()
        work_layout = QHBoxLayout(work_row)
        work_layout.setContentsMargins(0, 0, 0, 0)
        work_layout.setSpacing(4)
        work_layout.addWidget(self._work_edit, 1)
        work_layout.addWidget(work_button)
        self._channel_workers = _int_spin(
            0, 64, int(defaults.get("default_channel_workers") or 0)
        )
        self._channel_workers.setSpecialValueText("Auto")
        self._channel_workers.setToolTip(
            "Travailleurs parallèles pendant le filtrage (0 = automatique)."
        )
        resources_form.addRow(map_hint)
        resources_form.addRow("Dossier cache :", work_row)
        resources_form.addRow("Travailleurs canaux :", self._channel_workers)

        page_layout.addWidget(trigger_group)
        page_layout.addWidget(self._sections_box)
        page_layout.addWidget(curves_hint)
        for group in (
            self._curve_wide_box,
            self._curve_high_box,
            self._curve_low_box,
            self._curve_spikes_box,
            self._curve_rms_box,
            resources_box,
        ):
            page_layout.addWidget(group)
        hint = QLabel(
            "Tout changement ici exige Traiter (F5). "
            "Le cache ne recalcule que les étapes touchées."
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        page_layout.addWidget(hint)
        page_layout.addStretch(1)

        for spin in (
            self._pre_s,
            self._post_s,
            self._section_duration,
            self._section_start,
            self._section_end,
            self._hp_filter_cutoff,
            self._lp_filter_cutoff,
            self._artifact_threshold,
            self._rms_window,
            self._spike_threshold,
            self._rms_multiplier,
            self._overlay_pre,
            self._overlay_post,
        ):
            spin.valueChanged.connect(lambda _v: self._emit_config())
        self._threshold.valueChanged.connect(lambda _v: self._on_processing_threshold_changed())
        for int_spin in (
            self._section_count,
            self._hp_filter_order,
            self._lp_filter_order,
            self._channel_workers,
        ):
            int_spin.valueChanged.connect(lambda _v: self._emit_config())
        for box in (
            self._section_spec,
            self._hp_filter_type,
            self._lp_filter_type,
            self._software_notch,
            self._threshold_mode,
            self._polarity,
        ):
            box.currentIndexChanged.connect(lambda _i: self._emit_config())
        self._artifact_enabled.toggled.connect(lambda _c: self._emit_config())
        self._probe_edit.textChanged.connect(lambda _t: self._emit_config())
        self._work_edit.textChanged.connect(lambda _t: self._emit_config())

        self._sync_section_rows()
        self._sync_threshold_rows()
        return page

    def reveal_curve_params(self, *curves: str, focus_tab: bool = True) -> None:
        """Déplie les panneaux de traitement associés aux courbes demandées.

        ``curves`` accepte ``raw``/``wide``, ``hp``/``high``, ``lp``/``low``,
        ``spikes``, ``rms``. Les cases ne pilotent que l’UI (pas le pipeline).
        """
        mapping = {
            "raw": "_curve_wide_box",
            "wide": "_curve_wide_box",
            "hp": "_curve_high_box",
            "high": "_curve_high_box",
            "lp": "_curve_low_box",
            "low": "_curve_low_box",
            "spikes": "_curve_spikes_box",
            "rms": "_curve_rms_box",
        }
        opened = False
        for key in curves:
            attr = mapping.get(str(key).strip().lower())
            if not attr:
                continue
            box = getattr(self, attr, None)
            if box is None:
                continue
            if not box.isChecked():
                box.setChecked(True)
            opened = True
        if opened and focus_tab:
            self.focus_channel_tab()

    # -------------------------------------------------------------- behaviour

    def _sync_section_rows(self) -> None:
        no_trigger = str(self._edge.currentData()) == "none"
        by_count = str(self._section_spec.currentData()) == "count"
        self._threshold.setEnabled(not no_trigger)
        self._display_trigger_threshold.setEnabled(not no_trigger)
        self._trigger_polarity.setEnabled(not no_trigger)
        self._section_spec.setEnabled(no_trigger)
        self._section_count.setEnabled(no_trigger and by_count)
        self._section_duration.setEnabled(no_trigger and not by_count)
        self._section_start.setEnabled(no_trigger)
        self._section_end.setEnabled(no_trigger)
        # Ouvrir le groupe avancé si l’utilisateur choisit « sans déclencheur ».
        if no_trigger and hasattr(self, "_sections_box") and not self._sections_box.isChecked():
            self._sections_box.setChecked(True)

    def _sync_threshold_rows(self) -> None:
        fixed = str(self._threshold_mode.currentData()) == "fixed"
        self._spike_threshold.setEnabled(fixed)
        self._rms_multiplier.setEnabled(not fixed)

    def _on_display_trigger_changed(self) -> None:
        """Affichage → Pipeline : polarité / seuil, puis dirty + redessin."""
        if self._loading:
            return
        polarity = str(self._trigger_polarity.currentData() or "low")
        edge = TRIGGER_POLARITY_TO_EDGE.get(polarity, "falling")
        if str(self._edge.currentData()) != "none":
            idx = self._edge.findData(edge)
            if idx >= 0 and self._edge.currentIndex() != idx:
                self._edge.blockSignals(True)
                self._edge.setCurrentIndex(idx)
                self._edge.blockSignals(False)
        thr = float(self._display_trigger_threshold.value())
        if abs(float(self._threshold.value()) - thr) > 1e-12:
            self._threshold.blockSignals(True)
            self._threshold.setValue(thr)
            self._threshold.blockSignals(False)
        self._emit_config()
        self._emit_view()

    def _on_processing_edge_changed(self) -> None:
        if self._loading:
            return
        edge = str(self._edge.currentData() or "falling")
        polarity = EDGE_TO_TRIGGER_POLARITY.get(edge)
        if polarity is not None:
            idx = self._trigger_polarity.findData(polarity)
            if idx >= 0 and self._trigger_polarity.currentIndex() != idx:
                self._trigger_polarity.blockSignals(True)
                self._trigger_polarity.setCurrentIndex(idx)
                self._trigger_polarity.blockSignals(False)
        self._sync_section_rows()
        self._emit_config()
        self._emit_view()

    def _on_processing_threshold_changed(self) -> None:
        if self._loading:
            return
        thr = float(self._threshold.value())
        if abs(float(self._display_trigger_threshold.value()) - thr) > 1e-12:
            self._display_trigger_threshold.blockSignals(True)
            self._display_trigger_threshold.setValue(thr)
            self._display_trigger_threshold.blockSignals(False)
        self._emit_config()
        self._emit_view()

    def _browse_work_dir(self) -> None:
        start = self._work_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Choisir le dossier de cache", start)
        if path:
            self._work_edit.setText(path)

    def _emit_view(self) -> None:
        if self._loading:
            return
        self._view_debouncer.request()

    def _emit_config(self) -> None:
        if self._loading:
            return
        self.configChanged.emit()

    def _on_streams_toggled(self, *_args: Any) -> None:
        if self._loading:
            return
        if not any(
            box.isChecked()
            for box in (self._cb_stream_wide, self._cb_stream_high, self._cb_stream_low)
        ):
            self._loading = True
            self._cb_stream_wide.setChecked(True)
            self._loading = False
        self.streamsChanged.emit(self.continuous_streams())

    def continuous_streams(self) -> tuple[AnalysisStream, ...]:
        """Flux WIDE/HIGH/LOW cochés (aperçu continuous + montage)."""
        streams: list[AnalysisStream] = []
        if self._cb_stream_wide.isChecked():
            streams.append("raw")
        if self._cb_stream_high.isChecked():
            streams.append("hp")
        if self._cb_stream_low.isChecked():
            streams.append("lp")
        return tuple(streams) or ("raw",)

    def mark_stimulations(self) -> bool:
        return bool(self._cb_mark_stims.isChecked())

    def set_dirty_message(self, message: str) -> None:
        self._dirty_label.setText(message)
        self._dirty_row.setVisible(bool(message))

    def set_view_mode(self, mode: ViewMode) -> None:
        """Adapter Affichage / onglet Canal au mode aperçu ou revue montage."""
        resolved: ViewMode = mode if mode in ("preview", "montage") else "preview"
        if resolved == self._view_mode:
            return
        self._view_mode = resolved
        self._apply_view_mode_visibility()

    def view_mode(self) -> ViewMode:
        return self._view_mode

    def _apply_view_mode_visibility(self) -> None:
        """Masquer les contrôles qui n’agissent que sur l’autre vue."""
        montage = self._view_mode == "montage"
        preview = not montage

        # Canal reste visible (Pipeline F5) ; seule la zone aperçu est masquée en montage.
        if hasattr(self, "_channel_host"):
            self._channel_host.setVisible(preview)
            if hasattr(self, "_pipeline_sep"):
                self._pipeline_sep.setVisible(preview)
            if hasattr(self, "_pipeline_header"):
                self._pipeline_header.setVisible(True)

        if self._channel_tab_index is not None:
            self.tabs.setTabVisible(self._channel_tab_index, True)
            tip = (
                "Pipeline (filtres, spikes, RMS) → Traiter (F5)"
                if montage
                else "Mode aperçu, courbes, plages — et Pipeline → Traiter (F5)"
            )
            self.tabs.setTabToolTip(self._channel_tab_index, tip)

        # Échelles : HP stim = aperçu (Analyse) ; pas le montage continu.
        # RMS est réglé sous la case RMS (onglet Canal).
        if hasattr(self, "_axis_form"):
            _set_form_row_visible(self._axis_form, self._stim_hp_ylim, preview)

        if hasattr(self, "_montage_review_group"):
            self._montage_review_group.setVisible(montage)
        if hasattr(self, "_montage_pdf_group"):
            # Pagination PDF utile surtout quand on travaille en multi-canaux.
            self._montage_pdf_group.setVisible(montage)

        # Tooltips Affichage selon le mode.
        if montage:
            self.tabs.setTabToolTip(
                1,
                "Flux, sync, échelles traces, hauteur de ligne — redessin immédiat",
            )
        else:
            self.tabs.setTabToolTip(
                1, "Flux, échelles, sync — redessin immédiat (aperçu canal)"
            )

    def focus_processing_tab(self) -> None:
        """Compat : le Pipeline vit dans l’onglet Canal."""
        self.focus_channel_tab()

    def focus_channel_tab(self) -> None:
        if self._channel_tab_index is None:
            return
        if not self.tabs.isTabVisible(self._channel_tab_index):
            return
        self.tabs.setCurrentIndex(self._channel_tab_index)

    def _clear_channel_host(self) -> None:
        """Vide le slot aperçu sans détruire les widgets (ré-attache possible)."""
        while self._channel_host_layout.count():
            item = self._channel_host_layout.takeAt(0)
            child = item.widget()
            if child is None:
                continue
            if child is self._channel_placeholder:
                child.setParent(self)
            elif (
                self._channel_side is not None
                and self._channel_side_content is child
                and isinstance(self._channel_side, QScrollArea)
            ):
                # Remettre le contenu dans le scroll d’origine (ChannelAnalysisWindow).
                self._channel_side.setWidget(child)
                self._channel_side_content = None
            else:
                child.setParent(None)

    def set_channel_side_panel(
        self, widget: QWidget | None, *, focus: bool = False
    ) -> None:
        """Héberger le panneau latéral de l’aperçu canal dans l’onglet Canal.

        ``focus`` n’active l’onglet Canal que sur une vraie attache (pas à
        chaque redraw / re-sync — sinon l’utilisateur est arraché d’Affichage).
        """
        if widget is self._channel_side:
            # Déjà en place (ou déjà vidé) : no-op strict.
            return

        self._clear_channel_host()
        self._channel_side = widget
        self._channel_side_content = None

        if widget is None:
            self._channel_host_layout.addWidget(self._channel_placeholder)
        else:
            widget.setMinimumWidth(0)
            content: QWidget | None
            if isinstance(widget, QScrollArea):
                content = widget.takeWidget()
                self._channel_side_content = content
            else:
                content = widget
                widget.setParent(None)
            if content is not None:
                content.setMinimumWidth(0)
                self._channel_host_layout.addWidget(content)
            if focus and self._view_mode == "preview":
                self.focus_channel_tab()

        self._apply_view_mode_visibility()

    # ----------------------------------------------------------------- values

    def viewer_settings(self) -> ViewerSettings:
        legend = LegendSettings(
            visible=self._legend_visible.isChecked(),
            location=str(self._legend_location.currentData()),  # type: ignore[arg-type]
            font_size=float(self._legend_font.value()),
            columns=int(self._legend_columns.value()),
            gap=float(self._legend_gap.value()),
            frame=self._legend_frame.isChecked(),
            show_filter_details=self._legend_filters.isChecked(),
            show_sample_counts=self._legend_counts.isChecked(),
            show_reference_markers=self._stim_markers.isChecked(),
        )
        style = PanelStyle(
            title_font_size=float(self._title_font.value()),
            label_font_size=float(self._label_font.value()),
            tick_font_size=float(self._tick_font.value()),
            line_width=float(self._line_width.value()),
            grid=self._grid.isChecked(),
            grid_alpha=float(self._grid_alpha.value()),
            show_borders=self._show_borders.isChecked(),
            max_points_per_curve=int(self._max_points.value()),
        )
        d = self._defaults
        return ViewerSettings(
            zoom_onset_t0_s=float(d.get("default_zoom_onset_t0_s", -0.1)),
            zoom_onset_t1_s=float(d.get("default_zoom_onset_t1_s", 0.2)),
            zoom_end_t0_s=float(d.get("default_zoom_end_t0_s", -0.1)),
            zoom_end_t1_s=float(d.get("default_zoom_end_t1_s", 0.2)),
            psth_bin_window_s=float(self._psth_bin_s),
            sampling_percent=int(self._sampling_percent),
            spike_overlay_pre_ms=float(self._overlay_pre.value()),
            spike_overlay_post_ms=float(self._overlay_post.value()),
            x_limits=self._x_limits.value(),
            stim_hp_ylim=self._stim_hp_ylim.value(),
            rms_ylim=self._rms_ylim,
            trace_ylim=self._trace_ylim.value(),
            legend=legend,
            style=style,
            montage_channels=int(self._montage_channels.value()),
            montage_page=int(self._montage_page.value()),
            montage_row_min_height_px=int(self._montage_row_height.value()),
            time_sync=str(self._time_sync.currentData() or "recording_start"),  # type: ignore[arg-type]
            trigger_polarity=str(self._trigger_polarity.currentData() or "low"),  # type: ignore[arg-type]
            trigger_threshold=float(self._display_trigger_threshold.value()),
            continuous_stream=self.continuous_streams()[0],
            continuous_streams=self.continuous_streams(),
            continuous_mark_stims=self.mark_stimulations(),
        )

    def probe_layout_path(self) -> Path | None:
        text = self._probe_edit.text().strip()
        return Path(text) if text else None

    def set_probe_path(self, path: Path | str | None) -> None:
        """Met à jour le champ sans boucler sur configChanged si inchangé."""
        text = str(path) if path else ""
        if self._probe_edit.text() == text:
            return
        self._probe_edit.blockSignals(True)
        self._probe_edit.setText(text)
        self._probe_edit.blockSignals(False)

    def build_config(self, rhs_file: Path, *, base: AnalysisConfig | None = None) -> AnalysisConfig:
        """Processing configuration for one recording."""
        work_text = self._work_edit.text().strip()
        polarity = str(self._polarity.currentData())
        magnitude = abs(float(self._spike_threshold.value()))
        signed = -magnitude if polarity == "negative" else magnitude
        section_duration = (
            float(self._section_duration.value())
            if str(self._section_spec.currentData()) == "duration"
            else None
        )
        # Polarité / seuil : Affichage prime si le mode n’est pas « sans déclencheur ».
        edge = str(self._edge.currentData())
        threshold = float(self._threshold.value())
        if edge != "none":
            polarity = str(self._trigger_polarity.currentData() or "low")
            edge = TRIGGER_POLARITY_TO_EDGE.get(polarity, edge)
            threshold = float(self._display_trigger_threshold.value())
        config = AnalysisConfig(
            rhs_file=Path(rhs_file),
            threshold=threshold,
            edge=edge,  # type: ignore[arg-type]
            pre_s=float(self._pre_s.value()),
            post_s=float(self._post_s.value()),
            section_count=int(self._section_count.value()),
            section_duration_s=section_duration,
            section_spec=str(self._section_spec.currentData()),  # type: ignore[arg-type]
            section_trigger_start_s=float(self._section_start.value()),
            section_trigger_end_s=float(self._section_end.value()),
            spike_threshold_uv=signed,
            spike_threshold_polarity=polarity,  # type: ignore[arg-type]
            spike_threshold_mode=str(self._threshold_mode.currentData()),  # type: ignore[arg-type]
            spike_threshold_rms_multiplier=float(self._rms_multiplier.value()),
            psth_bin_window_s=float(self._psth_bin_s),
            spike_overlay_pre_ms=float(self._overlay_pre.value()),
            spike_overlay_post_ms=float(self._overlay_post.value()),
            zoom_onset_t0_s=float(self._defaults.get("default_zoom_onset_t0_s", -0.1)),
            zoom_onset_t1_s=float(self._defaults.get("default_zoom_onset_t1_s", 0.2)),
            zoom_end_t0_s=float(self._defaults.get("default_zoom_end_t0_s", -0.1)),
            zoom_end_t1_s=float(self._defaults.get("default_zoom_end_t1_s", 0.2)),
            first_trigger_hp_ylim_enabled=self._stim_hp_ylim.value().enabled,
            first_trigger_hp_ylim_min_uv=self._stim_hp_ylim.value().minimum,
            first_trigger_hp_ylim_max_uv=self._stim_hp_ylim.value().maximum,
            rms_window_s=float(self._rms_window.value()),
            intan_hp_filter_order=int(self._hp_filter_order.value()),
            intan_hp_filter_type=str(self._hp_filter_type.currentData()),  # type: ignore[arg-type]
            intan_hp_filter_cutoff_hz=float(self._hp_filter_cutoff.value()),
            intan_lp_filter_order=int(self._lp_filter_order.value()),
            intan_lp_filter_type=str(self._lp_filter_type.currentData()),  # type: ignore[arg-type]
            intan_lp_filter_cutoff_hz=float(self._lp_filter_cutoff.value()),
            software_notch_hz=int(self._software_notch.currentData() or 0),  # type: ignore[arg-type]
            intan_artifact_threshold_uv=float(self._artifact_threshold.value()),
            intan_artifact_suppression_enabled=self._artifact_enabled.isChecked(),
            work_dir=Path(work_text) if work_text else None,
            channel_workers=int(self._channel_workers.value()) or None,
            sampling_percent=int(self._sampling_percent),
            probe_layout_json=self.probe_layout_path(),
        )
        if base is not None:
            config = replace(
                config,
                save_dir=base.save_dir,
                pdf_title=base.pdf_title,
                comparison_workers=base.comparison_workers,
            )
        return config

    def validate(self) -> list[str]:
        problems = self.viewer_settings().validate()
        if self._post_s.value() <= 0:
            problems.append("La fenêtre après la stimulation doit être > 0 s.")
        if str(self._edge.currentData()) == "none" and self._section_end.value() <= self._section_start.value():
            problems.append(
                "Mode sans déclencheur : la fin virtuelle doit être après le début virtuel."
            )
        if self._overlay_post.value() <= 0:
            problems.append("Superposition de spikes : le temps après détection doit être > 0 ms.")
        return problems
