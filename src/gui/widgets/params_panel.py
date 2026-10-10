"""Paramètres d’affichage et de traitement, selon le coût d’un changement.

Répartition des onglets (modèle mental) :

- *Canal* — contenu du canal (mode, plages, contexte) + Pipeline de traitement
  (filtres, spikes, RMS, détection) → :attr:`configChanged` / Traiter (F5).
- *Affichage* — axe X, sync, disposition, légendes, textes, apparence →
  :attr:`viewChanged` (redessin immédiat, pas de recalcul).
  Les échelles Y vivent dans Canal → section de chaque courbe cochée.
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
    QPlainTextEdit,
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
    FontSizeFaceRow,
    configure_narrow_form,
    make_double_spin as _spin,
    make_int_spin as _int_spin,
)
from gui.defaults import DEFAULT_PSTH_BIN_S, DEFAULT_ZOOM_T0_S, DEFAULT_ZOOM_T1_S
from gui.jobs import Debouncer
from view_config import (
    EDGE_TO_TRIGGER_POLARITY,
    LEGEND_LOCATIONS,
    TIME_SYNC_LABELS,
    AnalysisSettings,
    AnalysisStream,
    AxisLimits,
    TextOverrides,
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


def _set_form_row_enabled(form: QFormLayout, field: QWidget, enabled: bool) -> None:
    """Active / grise une ligne QFormLayout (libellé + champ)."""
    field.setEnabled(enabled)
    label = form.labelForField(field)
    if label is not None:
        label.setEnabled(enabled)


class ParamsPanel(QWidget):
    """Réglages d’affichage et de traitement dans le dock Paramètres."""

    viewChanged = Signal()
    configChanged = Signal()
    processRequested = Signal()
    # Visibilité Pipeline (WIDE/HIGH/LOW/RMS/Spikes) — même signal pour
    # continuous, moyenne et stimulation.
    pipelineVisibilityChanged = Signal()

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
        # Partage équitable + élision : 2 onglets lisibles dès ~220 px.
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

        self._pipeline_header = QLabel("Traitement (Traiter F5)")
        self._pipeline_header.setObjectName("sectionLabel")
        self._pipeline_header.setWordWrap(True)
        self._pipeline_header.setToolTip(
            "Paramètres qui recalculent les données. "
            "Cochez WIDE / HIGH / LOW / RMS / Spikes pour afficher les courbes."
        )
        self._pipeline_sep = QFrame()
        self._pipeline_sep.setFrameShape(QFrame.Shape.HLine)
        self._pipeline_sep.setObjectName("sectionSeparator")

        display_page = self._build_display_tab()
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
        self.tabs.setTabToolTip(
            0,
            "Contenu du canal (mode, plages, résumés) et traitement "
            "(filtres, spikes, RMS) → Traiter (F5)",
        )
        self.tabs.setTabToolTip(
            1,
            "Axe X, sync, disposition, légendes et apparence — "
            "redessin immédiat (échelles Y → Canal, par type de courbe)",
        )

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
        """Échelles, sync, disposition, légendes, apparence — redessin immédiat."""
        defaults = self._defaults
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(8)

        sync_group = QGroupBox("Temps & marqueurs")
        sync_form = configure_narrow_form(QFormLayout(sync_group))
        self._time_sync = _choice(
            _TIME_SYNC_CHOICES,
            str(defaults.get("default_time_sync", "recording_start")),
        )
        self._time_sync.setToolTip(
            "Origine de l’axe temps des traces continues : début du fichier "
            "ou premier trigger détecté (t=0). La détection du trigger se règle "
            "dans Canal → Traitement."
        )
        self._cb_mark_stims = QCheckBox("Marqueurs de stimulation sur les traces")
        self._cb_mark_stims.setChecked(True)
        self._cb_mark_stims.setToolTip(
            "Marquer les stimulations sur l’aperçu continuous et la revue montage."
        )
        sync_form.addRow("Origine du temps :", self._time_sync)
        sync_form.addRow(self._cb_mark_stims)
        self._cb_mark_stims.toggled.connect(self._emit_view)

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
            "Prioritaire sur la section / le zoom par défaut. "
            "Les échelles Y se règlent dans Canal → section de chaque courbe."
        )
        self._axis_form.addRow("Axe X (temps) :", self._x_limits)
        axis_hint = QLabel(
            "Échelles verticales (WIDE / HIGH / LOW / RMS) → Canal, "
            "dans la section de la courbe cochée."
        )
        axis_hint.setObjectName("hintLabel")
        axis_hint.setWordWrap(True)
        self._axis_form.addRow(axis_hint)

        zoom_group, zoom_inner = _collapsible_group(
            "Fenêtres de zoom (PDF / sections)", expanded=False
        )
        zoom_form = configure_narrow_form(QFormLayout(zoom_inner))
        zoom_form.setContentsMargins(0, 0, 0, 0)
        self._zoom_onset_t0 = _spin(
            -60.0,
            60.0,
            float(defaults.get("default_zoom_onset_t0_s", DEFAULT_ZOOM_T0_S)),
            decimals=3,
            step=0.01,
            suffix=" s",
        )
        self._zoom_onset_t1 = _spin(
            -60.0,
            60.0,
            float(defaults.get("default_zoom_onset_t1_s", DEFAULT_ZOOM_T1_S)),
            decimals=3,
            step=0.01,
            suffix=" s",
        )
        self._zoom_end_t0 = _spin(
            -60.0,
            60.0,
            float(defaults.get("default_zoom_end_t0_s", DEFAULT_ZOOM_T0_S)),
            decimals=3,
            step=0.01,
            suffix=" s",
        )
        self._zoom_end_t1 = _spin(
            -60.0,
            60.0,
            float(defaults.get("default_zoom_end_t1_s", DEFAULT_ZOOM_T1_S)),
            decimals=3,
            step=0.01,
            suffix=" s",
        )
        for widget, tip in (
            (
                self._zoom_onset_t0,
                "Début de la fenêtre relative au début de stimulation (PDF / panneaux zoom début).",
            ),
            (
                self._zoom_onset_t1,
                "Fin de la fenêtre relative au début de stimulation.",
            ),
            (
                self._zoom_end_t0,
                "Début de la fenêtre relative à la fin de stimulation (PDF / panneaux zoom fin).",
            ),
            (
                self._zoom_end_t1,
                "Fin de la fenêtre relative à la fin de stimulation.",
            ),
        ):
            widget.setToolTip(tip)
        zoom_hint = QLabel(
            "Temps relatifs à la stim (t=0). Utilisés pour les sections PDF "
            "« zoom début / fin » et comme valeurs par défaut des plages Rel. stim."
        )
        zoom_hint.setObjectName("hintLabel")
        zoom_hint.setWordWrap(True)
        zoom_form.addRow("Zoom début — de :", self._zoom_onset_t0)
        zoom_form.addRow("Zoom début — à :", self._zoom_onset_t1)
        zoom_form.addRow("Zoom fin — de :", self._zoom_end_t0)
        zoom_form.addRow("Zoom fin — à :", self._zoom_end_t1)
        zoom_form.addRow(zoom_hint)

        layout_group = QGroupBox("Disposition")
        self._layout_form = configure_narrow_form(QFormLayout(layout_group))
        self._graph_height = _int_spin(
            160,
            1200,
            int(defaults.get("default_graph_height_px", 400) or 400),
            step=20,
            suffix=" px",
        )
        self._graph_height_tooltip = (
            "Hauteur fixe de chaque graphique (aperçu, analyse, onglets)."
        )
        self._graph_height.setToolTip(self._graph_height_tooltip)
        self._show_scale_bars = QCheckBox("Échelle flottante (barres temps / amplitude)")
        self._show_scale_bars.setChecked(False)
        self._show_scale_bars.setToolTip(
            "Masque les graduations numériques sur les bords et affiche une barre "
            "d’échelle flottante (ex. 100 ms · 200 µV)."
        )
        self._scale_bar_amp_manual = QCheckBox("Manuel")
        self._scale_bar_amp_manual.setChecked(False)
        self._scale_bar_amp = _spin(
            0.01, 1_000_000.0, 100.0, decimals=2, step=10.0, suffix=" µV"
        )
        self._scale_bar_amp.setToolTip(
            "Amplitude représentée par le bâton vertical. La longueur pixel "
            "s’adapte automatiquement au zoom / à l’échelle Y."
        )
        self._scale_bar_time_manual = QCheckBox("Manuel")
        self._scale_bar_time_manual.setChecked(False)
        self._scale_bar_time_ms = _spin(
            0.01, 1_000_000.0, 100.0, decimals=2, step=10.0, suffix=" ms"
        )
        self._scale_bar_time_ms.setToolTip(
            "Durée représentée par le bâton horizontal. La longueur pixel "
            "s’adapte automatiquement au zoom / à l’échelle X."
        )
        amp_row = QWidget()
        amp_lay = QHBoxLayout(amp_row)
        amp_lay.setContentsMargins(0, 0, 0, 0)
        amp_lay.setSpacing(6)
        amp_lay.addWidget(self._scale_bar_amp_manual)
        amp_lay.addWidget(self._scale_bar_amp, 1)
        time_row = QWidget()
        time_lay = QHBoxLayout(time_row)
        time_lay.setContentsMargins(0, 0, 0, 0)
        time_lay.setSpacing(6)
        time_lay.addWidget(self._scale_bar_time_manual)
        time_lay.addWidget(self._scale_bar_time_ms, 1)
        self._scale_bar_amp_manual.toggled.connect(self._sync_scale_bar_inputs)
        self._scale_bar_time_manual.toggled.connect(self._sync_scale_bar_inputs)
        self._show_scale_bars.toggled.connect(self._sync_scale_bar_inputs)
        self._sync_scale_bar_inputs()
        self._max_points = _int_spin(500, 200000, 6000, step=500)
        self._max_points.setToolTip(
            "Points dessinés par courbe. Au-delà, une enveloppe min/max conserve "
            "tous les pics tout en restant rapide."
        )
        self._layout_form.addRow("Hauteur des graphs :", self._graph_height)
        self._layout_form.addRow(self._show_scale_bars)
        self._layout_form.addRow("Amplitude barre :", amp_row)
        self._layout_form.addRow("Temps barre :", time_row)
        self._layout_form.addRow("Points max / courbe :", self._max_points)

        # Revue montage continue uniquement.
        self._montage_review_group = QGroupBox("Revue montage")
        review_form = configure_narrow_form(QFormLayout(self._montage_review_group))
        self._montage_review_channels = _int_spin(1, 128, 10)
        self._montage_review_channels_tooltip = (
            "Nombre de canaux empilés par page dans la revue montage. "
            "Utilisez les boutons Suivant / Précédent du bandeau pour changer de page."
        )
        self._montage_review_channels.setToolTip(self._montage_review_channels_tooltip)
        review_form.addRow("Canaux visibles :", self._montage_review_channels)
        self._montage_review_page = 0
        self._montage_row_height = _int_spin(
            36,
            200,
            int(defaults.get("default_montage_row_min_height_px", 52) or 52),
            suffix=" px",
        )
        self._montage_row_tooltip = (
            "Hauteur minimale d’une ligne canal×flux dans la revue montage. "
            "Plus bas = ouverture plus rapide, moins de détail vertical."
        )
        self._montage_row_height.setToolTip(self._montage_row_tooltip)
        review_form.addRow("Hauteur de ligne :", self._montage_row_height)
        review_hint = QLabel(
            "Les graphs Pipeline (WIDE/HIGH/LOW/RMS/Spikes) sont empilés "
            "par canal, page par page. Mode = Canal. "
            "Visibilité canaux : Session → Channels. Retour : Ctrl+Shift+M."
        )
        review_hint.setObjectName("hintLabel")
        review_hint.setWordWrap(True)
        review_form.addRow(review_hint)
        self._montage_review_tooltip = (
            "Réglages propres à la revue montage (Ctrl+M)."
        )
        self._montage_review_group.setToolTip(self._montage_review_tooltip)

        montage_group, montage_inner = _collapsible_group(
            "Montages PDF (moyenne / 2e stim)", expanded=False
        )
        self._montage_pdf_group = montage_group
        self._montage_pdf_tooltip = (
            "Pagination des montages PDF moyenne / 2e stim (export). "
            "Sans effet sur la revue montage continue."
        )
        self._montage_pdf_group.setToolTip(self._montage_pdf_tooltip)
        montage_form = configure_narrow_form(QFormLayout(montage_inner))
        montage_form.setContentsMargins(0, 0, 0, 0)
        self._montage_channels = _int_spin(2, 1024, 12)
        self._montage_channels_tooltip = (
            "Canaux empilés par page pour les montages PDF moyenne / 2e stim. "
            "La revue continue utilise « Canaux visibles » ci-dessus."
        )
        self._montage_channels.setToolTip(self._montage_channels_tooltip)
        self._montage_page = _int_spin(0, 64, 0)
        self._montage_page_tooltip = (
            "Bloc de canaux pour les montages PDF (0 = première page). "
            "Sans effet sur la revue montage continue."
        )
        self._montage_page.setToolTip(self._montage_page_tooltip)
        montage_form.addRow("Canaux par montage :", self._montage_channels)
        montage_form.addRow("Page de montage :", self._montage_page)

        for group in (
            sync_group,
            axis_group,
            zoom_group,
            layout_group,
            self._montage_review_group,
            montage_group,
        ):
            page_layout.addWidget(group)

        for int_widget in (
            self._montage_channels,
            self._montage_page,
            self._montage_review_channels,
            self._montage_row_height,
            self._graph_height,
            self._max_points,
        ):
            int_widget.valueChanged.connect(lambda _v: self._emit_view())
        for zoom_spin in (
            self._zoom_onset_t0,
            self._zoom_onset_t1,
            self._zoom_end_t0,
            self._zoom_end_t1,
        ):
            zoom_spin.valueChanged.connect(lambda _v: self._emit_view())
        self._x_limits.changed.connect(self._emit_view)
        self._time_sync.currentIndexChanged.connect(lambda _i: self._emit_view())
        self._show_scale_bars.toggled.connect(lambda _c: self._emit_view())
        self._scale_bar_amp_manual.toggled.connect(lambda _c: self._emit_view())
        self._scale_bar_time_manual.toggled.connect(lambda _c: self._emit_view())
        self._scale_bar_amp.valueChanged.connect(lambda _v: self._emit_view())
        self._scale_bar_time_ms.valueChanged.connect(lambda _v: self._emit_view())

        self._add_style_groups(page_layout)
        page_layout.addStretch(1)
        return page

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

    # ---------------------------------------------------- style (dans Affichage)

    def _add_style_groups(self, page_layout: QVBoxLayout) -> None:
        """Légendes, textes et apparence — redessin immédiat (jamais F5)."""
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
        self._legend_font = FontSizeFaceRow(4.0, 24.0, 9.0, decimals=1, step=0.5)
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
        legend_form.addRow("Police de la légende :", self._legend_font)
        legend_form.addRow("Colonnes :", self._legend_columns)
        legend_form.addRow("Distance au graphique :", self._legend_gap)
        legend_form.addRow(self._legend_frame)
        legend_form.addRow(self._legend_filters)
        legend_form.addRow(self._legend_counts)

        style_group = QGroupBox("Textes et apparence")
        style_form = configure_narrow_form(QFormLayout(style_group))
        self._text_title = QLineEdit()
        self._text_title.setPlaceholderText("Automatique")
        self._text_title.setToolTip("Laissez vide pour le titre généré automatiquement.")
        self._text_xlabel = QLineEdit()
        self._text_xlabel.setPlaceholderText("Automatique")
        self._text_ylabel = QLineEdit()
        self._text_ylabel.setPlaceholderText("Automatique")
        self._text_series_suffix = QCheckBox(
            "Suffixe de type de courbe (moyenne, RMS…)"
        )
        self._text_series_suffix.setChecked(True)
        self._text_series_suffix.setToolTip(
            "Décochez pour n’afficher que le nom d’enregistrement dans la légende."
        )
        self._text_legend_labels = QPlainTextEdit()
        self._text_legend_labels.setPlaceholderText(
            "Automatique — une entrée de légende par ligne,\n"
            "dans l’ordre des courbes. Ligne vide = garder l’auto."
        )
        self._text_legend_labels.setMaximumHeight(90)
        self._text_legend_labels.setToolTip(
            "Remplace les libellés de légende. Les noms de base restent "
            "aussi éditables dans le dock Enregistrements → Légende."
        )
        self._title_font = FontSizeFaceRow(5.0, 24.0, 10.0, decimals=1, step=0.5)
        self._label_font = FontSizeFaceRow(5.0, 24.0, 9.0, decimals=1, step=0.5)
        self._tick_font = FontSizeFaceRow(4.0, 20.0, 8.0, decimals=1, step=0.5)
        self._line_width = _spin(0.2, 5.0, 1.2, decimals=2, step=0.1)
        self._grid = QCheckBox("Afficher la grille")
        self._grid.setChecked(True)
        self._grid_alpha = _spin(0.0, 1.0, 0.3, decimals=2, step=0.05)
        self._show_borders = QCheckBox("Afficher les bordures du graphique")
        self._show_borders.setChecked(True)
        self._show_borders.setToolTip(
            "Cadre avec graduations autour de la zone de tracé de chaque panneau."
        )
        self._ticks_inside = QCheckBox("Graduations vers l’intérieur")
        self._ticks_inside.setChecked(False)
        self._ticks_inside.setToolTip(
            "Oriente les graduations (ticks) vers l’intérieur du cadre de tracé."
        )
        self._stim_markers = QCheckBox("Pointillés de début / fin de stimulation")
        self._stim_markers.setChecked(True)
        self._stim_markers.setToolTip(
            "Lignes en pointillés au début (et à la fin) de stimulation sur les graphiques. "
            "Distinct des marqueurs sur les traces continues (Temps & marqueurs)."
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

        page_layout.addWidget(legend_group)
        page_layout.addWidget(style_group)

        for box in (
            self._legend_visible,
            self._legend_frame,
            self._legend_filters,
            self._legend_counts,
            self._text_series_suffix,
            self._grid,
            self._show_borders,
            self._ticks_inside,
            self._stim_markers,
        ):
            box.toggled.connect(lambda _c: self._emit_view())
        self._legend_location.currentIndexChanged.connect(lambda _i: self._emit_view())
        for spin in (
            self._legend_gap,
            self._line_width,
            self._grid_alpha,
        ):
            spin.valueChanged.connect(lambda _v: self._emit_view())
        for font_row in (
            self._legend_font,
            self._title_font,
            self._label_font,
            self._tick_font,
        ):
            font_row.changed.connect(self._emit_view)
        self._legend_columns.valueChanged.connect(lambda _v: self._emit_view())
        for line in (self._text_title, self._text_xlabel, self._text_ylabel):
            line.textChanged.connect(lambda _t: self._emit_view())
        self._text_legend_labels.textChanged.connect(self._emit_view)

    # -------------------------------------------------------- processing tab

    def _build_processing_tab(self) -> QWidget:
        """Traitement F5 (embarqué sous l’onglet Canal)."""
        defaults = self._defaults
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)

        trigger_group = QGroupBox("Détection des stimulations")
        trigger_form = configure_narrow_form(QFormLayout(trigger_group))
        self._edge = _choice(_EDGE_CHOICES, str(defaults.get("default_edge", "falling")))
        self._edge.setToolTip(
            "Polarité TTL sur ANALOG-IN-0. Unique réglage de détection "
            "(alimente aussi la sync « sur trigger » de l’onglet Affichage)."
        )
        self._threshold = _spin(
            -100.0, 100.0, defaults.get("default_threshold", 1.0), decimals=3, suffix=" V"
        )
        self._threshold.setToolTip(
            "Seuil de détection du trigger sur ANALOG-IN-0 (V). Exige Traiter (F5)."
        )
        self._pre_s = _spin(0.0, 600.0, defaults.get("default_pre_s", 1.0), suffix=" s")
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
            "Cochez un groupe pour afficher ses courbes (tous modes) et régler "
            "ses paramètres. Le traitement F5 reste complet même si une case est décochée."
        )
        curves_hint.setObjectName("hintLabel")
        curves_hint.setWordWrap(True)

        # ---- WIDE : prétraitement wideband (notch / artefacts) ----
        self._curve_wide_box, wide_inner = _collapsible_group(
            "WIDE (brut)", expanded=True
        )
        self._curve_wide_box.setToolTip(
            "Afficher WIDE (continuous / moyenne / stim) et ses paramètres (notch / artefacts)."
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
        self._artifact_enabled.setChecked(
            bool(defaults.get("default_intan_artifact_suppression_enabled", True))
        )
        self._artifact_threshold = _spin(
            1.0,
            100000.0,
            float(defaults.get("default_intan_artifact_threshold_uv", 2500.0)),
            decimals=1,
            step=50.0,
            suffix=" µV",
        )
        wide_form.addRow("Notch logiciel :", self._software_notch)
        wide_form.addRow(self._artifact_enabled)
        wide_form.addRow("Seuil d’artefact :", self._artifact_threshold)
        self._raw_ylim = AxisLimitRow(AxisLimits(), unit=" µV")
        self._raw_ylim.setToolTip(
            "Échelle Y des panneaux WIDE (continuous, moyenne, stimulation, montage). "
            "Redessin immédiat — pas de Traiter (F5)."
        )
        wide_form.addRow("Échelle Y :", self._raw_ylim)

        # ---- HIGH : filtre passe-haut (indépendant de LOW) ----
        self._curve_high_box, high_inner = _collapsible_group(
            "HIGH (passe-haut)", expanded=False
        )
        self._curve_high_box.setToolTip(
            "Afficher HIGH (continuous / moyenne / stim) et ses paramètres de filtre."
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
        self._hp_ylim = AxisLimitRow(
            AxisLimits(
                enabled=bool(defaults.get("default_first_trigger_hp_ylim_enabled", False)),
                minimum=float(defaults.get("default_first_trigger_hp_ylim_min_uv", -200.0)),
                maximum=float(defaults.get("default_first_trigger_hp_ylim_max_uv", 200.0)),
            ),
            unit=" µV",
        )
        self._hp_ylim.setToolTip(
            "Échelle Y des panneaux HIGH (continuous, moyenne, stimulation, montage). "
            "Redessin immédiat — pas de Traiter (F5)."
        )
        high_form.addRow("Échelle Y :", self._hp_ylim)

        # ---- LOW : filtre passe-bas (indépendant de HIGH) ----
        self._curve_low_box, low_inner = _collapsible_group(
            "LOW (passe-bas)", expanded=False
        )
        self._curve_low_box.setToolTip(
            "Afficher LOW (continuous / moyenne / stim) et ses paramètres de filtre."
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
        self._lp_ylim = AxisLimitRow(AxisLimits(), unit=" µV")
        self._lp_ylim.setToolTip(
            "Échelle Y des panneaux LOW (continuous, moyenne, stimulation, montage). "
            "Redessin immédiat — pas de Traiter (F5)."
        )
        low_form.addRow("Échelle Y :", self._lp_ylim)

        # ---- Spikes : détection sur le flux HIGH + graphs analyse ----
        self._curve_spikes_box, spikes_inner = _collapsible_group(
            "Spikes", expanded=False
        )
        self._curve_spikes_box.setToolTip(
            "Afficher les graphs spikes et les paramètres de détection "
            "(même cases pour continuous / moyenne / stimulation)."
        )
        spikes_layout = QVBoxLayout(spikes_inner)
        spikes_layout.setContentsMargins(0, 0, 0, 0)
        spikes_layout.setSpacing(6)
        spike_form_host = QWidget(spikes_inner)
        spike_form = configure_narrow_form(QFormLayout(spike_form_host))
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
        self._psth_bin = _spin(
            0.001,
            10.0,
            float(defaults.get("default_psth_bin_window_s", DEFAULT_PSTH_BIN_S)),
            decimals=3,
            suffix=" s",
        )
        self._psth_bin.setToolTip(
            "Largeur des bins du PSTH / firing rate. Modifier → Traiter (F5)."
        )
        self._sampling = _int_spin(
            1, 100, int(defaults.get("default_sampling_percent", 100) or 100)
        )
        self._sampling.setToolTip(
            "Pourcentage d’essais utilisés pour les graphs spikes (sous-échantillon)."
        )
        spike_form.addRow("Avant le spike :", self._overlay_pre)
        spike_form.addRow("Après le spike :", self._overlay_post)
        spike_form.addRow("Bin PSTH :", self._psth_bin)
        spike_form.addRow("Échantillonnage :", self._sampling)
        self._threshold_mode.currentIndexChanged.connect(lambda _i: self._sync_threshold_rows())
        spikes_layout.addWidget(spike_form_host)

        spike_graphs_label = QLabel("Graphs (tous modes)")
        spike_graphs_label.setObjectName("hintLabel")
        spikes_layout.addWidget(spike_graphs_label)
        self._cb_pipe_psth = QCheckBox("PSTH")
        self._cb_pipe_trial_rate = QCheckBox("Firing rate / essai")
        self._cb_pipe_raster = QCheckBox("Raster")
        self._cb_pipe_raster.setToolTip(
            "Aperçu canal : raster compact (comme en montage) + raster tous essais.\n"
            "Revue montage : une ligne RST par canal."
        )
        self._cb_pipe_isi = QCheckBox("ISI")
        self._cb_pipe_overlay = QCheckBox("Spike scope")
        for box in (
            self._cb_pipe_psth,
            self._cb_pipe_trial_rate,
            self._cb_pipe_raster,
            self._cb_pipe_isi,
            self._cb_pipe_overlay,
        ):
            spikes_layout.addWidget(box)

        # ---- RMS ----
        self._curve_rms_box, rms_inner = _collapsible_group("RMS", expanded=False)
        self._curve_rms_box.setToolTip(
            "Afficher le profil RMS et ses paramètres "
            "(même case pour continuous / moyenne / stimulation)."
        )
        rms_form = configure_narrow_form(QFormLayout(rms_inner))
        rms_form.setContentsMargins(0, 0, 0, 0)
        self._rms_window = _spin(
            0.001, 60.0, defaults.get("default_rms_window_s", 1.0), suffix=" s"
        )
        self._rms_window.setToolTip("Fenêtre glissante utilisée pour les profils RMS.")
        self._rms_ylim_row = AxisLimitRow(
            AxisLimits(enabled=True, minimum=0.0, maximum=20.0),
            unit=" µV",
            step=1.0,
        )
        self._rms_ylim_row.setToolTip(
            "Échelle Y des panneaux et résumés RMS. "
            "Redessin immédiat — pas de Traiter (F5)."
        )
        rms_form.addRow("Fenêtre RMS :", self._rms_window)
        rms_form.addRow("Échelle Y :", self._rms_ylim_row)

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
            "Filtres / spikes / fenêtre RMS → Traiter (F5). "
            "Échelles Y dans chaque section → redessin immédiat. "
            "Axe X / sync / apparence → Affichage."
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
            self._psth_bin,
        ):
            spin.valueChanged.connect(lambda _v: self._emit_config())
        self._threshold.valueChanged.connect(lambda _v: self._on_processing_threshold_changed())
        for int_spin in (
            self._section_count,
            self._hp_filter_order,
            self._lp_filter_order,
            self._channel_workers,
            self._sampling,
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

        # Cases Pipeline → même handler pour continuous / moyenne / stimulation.
        for box in (
            self._curve_wide_box,
            self._curve_high_box,
            self._curve_low_box,
            self._curve_rms_box,
            self._curve_spikes_box,
            self._cb_pipe_psth,
            self._cb_pipe_trial_rate,
            self._cb_pipe_raster,
            self._cb_pipe_isi,
            self._cb_pipe_overlay,
        ):
            box.toggled.connect(self._on_pipeline_visibility_toggled)
        self._curve_spikes_box.toggled.connect(self._on_spikes_group_toggled)

        for ylim_row in (
            self._raw_ylim,
            self._hp_ylim,
            self._lp_ylim,
            self._rms_ylim_row,
        ):
            ylim_row.changed.connect(self._emit_view)

        self._sync_section_rows()
        self._sync_threshold_rows()
        return page

    def reveal_curve_params(
        self,
        *curves: str,
        focus_tab: bool = True,
        emit_streams: bool = True,
    ) -> None:
        """Coche / déplie les groupes Pipeline associés aux courbes demandées.

        ``curves`` accepte ``raw``/``wide``, ``hp``/``high``, ``lp``/``low``,
        ``spikes``, ``rms``. WIDE/HIGH/LOW pilotent aussi l’affichage continuous.
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
        stream_attrs = {"_curve_wide_box", "_curve_high_box", "_curve_low_box"}
        opened = False
        touched_streams = False
        self._loading = True
        try:
            for key in curves:
                attr = mapping.get(str(key).strip().lower())
                if not attr:
                    continue
                box = getattr(self, attr, None)
                if box is None:
                    continue
                if not box.isChecked():
                    box.setChecked(True)
                    if attr in stream_attrs:
                        touched_streams = True
                opened = True
        finally:
            self._loading = False
        if opened and focus_tab:
            self.focus_channel_tab()
        if (touched_streams or opened) and emit_streams:
            self.pipelineVisibilityChanged.emit()

    # -------------------------------------------------------------- behaviour

    def _sync_section_rows(self) -> None:
        no_trigger = str(self._edge.currentData()) == "none"
        by_count = str(self._section_spec.currentData()) == "count"
        self._threshold.setEnabled(not no_trigger)
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

    def _trigger_polarity_value(self) -> str:
        """Polarité d’affichage dérivée du front de détection (Canal)."""
        edge = str(self._edge.currentData() or "falling")
        return EDGE_TO_TRIGGER_POLARITY.get(edge, "low")

    def _on_processing_edge_changed(self) -> None:
        if self._loading:
            return
        self._sync_section_rows()
        self._emit_config()
        self._emit_view()

    def _on_processing_threshold_changed(self) -> None:
        if self._loading:
            return
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

    def montage_review_channels(self) -> int:
        """Canaux par page dans la revue montage."""
        return max(1, int(self._montage_review_channels.value()))

    def montage_review_page(self) -> int:
        """Index de page courant (0-based) de la revue montage."""
        return max(0, int(self._montage_review_page))

    def set_montage_review_page(self, page: int, *, emit: bool = True) -> None:
        """Fixer la page de revue montage (boutons Suivant / Précédent)."""
        page = max(0, int(page))
        if page == int(self._montage_review_page):
            return
        self._montage_review_page = page
        if emit:
            self._emit_view()

    def clamp_montage_review_page(
        self, *, n_visible: int | None = None, emit: bool = True
    ) -> None:
        """Ramener la page dans les bornes quand le nombre de canaux change."""
        per_page = self.montage_review_channels()
        if n_visible is None:
            if self._montage_review_page < 0:
                self._montage_review_page = 0
                if emit:
                    self._emit_view()
            return
        n_pages = max(1, (max(0, int(n_visible)) + per_page - 1) // per_page)
        page = max(0, min(n_pages - 1, int(self._montage_review_page)))
        if page != int(self._montage_review_page):
            self._montage_review_page = page
            if emit:
                self._emit_view()

    def _emit_config(self) -> None:
        if self._loading:
            return
        self.configChanged.emit()

    def _on_pipeline_visibility_toggled(self, *_args: Any) -> None:
        """Cases Pipeline → continuous / moyenne / stimulation (même chemin)."""
        if self._loading:
            return
        if not any(
            box.isChecked()
            for box in (
                self._curve_wide_box,
                self._curve_high_box,
                self._curve_low_box,
                self._curve_rms_box,
                self._curve_spikes_box,
            )
        ):
            self._loading = True
            self._curve_wide_box.setChecked(True)
            self._loading = False
        self.pipelineVisibilityChanged.emit()

    def _on_spikes_group_toggled(self, checked: bool) -> None:
        """À l’ouverture de Spikes : cocher PSTH si aucun graph n’est sélectionné."""
        if self._loading or not checked:
            return
        if not any(
            box.isChecked()
            for box in (
                self._cb_pipe_psth,
                self._cb_pipe_trial_rate,
                self._cb_pipe_raster,
                self._cb_pipe_isi,
                self._cb_pipe_overlay,
            )
        ):
            self._loading = True
            self._cb_pipe_psth.setChecked(True)
            self._loading = False
            # Réémettre : le toggle Spikes a déjà émis sans graph sélectionné.
            self.pipelineVisibilityChanged.emit()

    def continuous_streams(self) -> tuple[AnalysisStream, ...]:
        """Flux WIDE/HIGH/LOW cochés — dérivés des mêmes cases que l’analyse."""
        streams: list[AnalysisStream] = []
        if self._curve_wide_box.isChecked():
            streams.append("raw")
        if self._curve_high_box.isChecked():
            streams.append("hp")
        if self._curve_low_box.isChecked():
            streams.append("lp")
        return tuple(streams) or ("raw",)

    def pipeline_visibility_flags(self) -> dict[str, bool]:
        """Visibilité des courbes (tous modes) depuis les cases Pipeline."""
        spikes_on = bool(self._curve_spikes_box.isChecked())
        return {
            "show_raw": bool(self._curve_wide_box.isChecked()),
            "show_hp": bool(self._curve_high_box.isChecked()),
            "show_lp": bool(self._curve_low_box.isChecked()),
            "show_rms": bool(self._curve_rms_box.isChecked()),
            "show_psth": spikes_on and bool(self._cb_pipe_psth.isChecked()),
            "show_trial_rate": spikes_on and bool(self._cb_pipe_trial_rate.isChecked()),
            "show_raster": spikes_on and bool(self._cb_pipe_raster.isChecked()),
            "show_isi": spikes_on and bool(self._cb_pipe_isi.isChecked()),
            "show_overlay": spikes_on and bool(self._cb_pipe_overlay.isChecked()),
        }

    # Alias historique.
    def pipeline_analysis_flags(self) -> dict[str, bool]:
        return self.pipeline_visibility_flags()

    def sync_analysis_checks_from(self, analysis: AnalysisSettings | object) -> None:
        """Aligne les cases Pipeline sur un ``AnalysisSettings`` (sans réémettre)."""
        self._loading = True
        try:
            self._curve_wide_box.setChecked(bool(getattr(analysis, "show_raw", False)))
            self._curve_high_box.setChecked(bool(getattr(analysis, "show_hp", False)))
            self._curve_low_box.setChecked(bool(getattr(analysis, "show_lp", False)))
            self._curve_rms_box.setChecked(bool(getattr(analysis, "show_rms", False)))
            spike_flags = (
                bool(getattr(analysis, "show_psth", False)),
                bool(getattr(analysis, "show_trial_rate", False)),
                bool(getattr(analysis, "show_raster", False)),
                bool(getattr(analysis, "show_isi", False)),
                bool(getattr(analysis, "show_overlay", False)),
            )
            any_spike = any(spike_flags)
            self._curve_spikes_box.setChecked(any_spike)
            self._cb_pipe_psth.setChecked(spike_flags[0])
            self._cb_pipe_trial_rate.setChecked(spike_flags[1])
            self._cb_pipe_raster.setChecked(spike_flags[2])
            self._cb_pipe_isi.setChecked(spike_flags[3])
            self._cb_pipe_overlay.setChecked(spike_flags[4])
            if not any(
                box.isChecked()
                for box in (
                    self._curve_wide_box,
                    self._curve_high_box,
                    self._curve_low_box,
                    self._curve_rms_box,
                )
            ) and not any_spike:
                self._curve_wide_box.setChecked(True)
        finally:
            self._loading = False

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
        """Tous les paramètres restent visibles ; ceux de l’autre vue sont grisés."""
        montage = self._view_mode == "montage"
        preview_only_tip = "Sans effet en revue montage — basculez en aperçu (Ctrl+Shift+M)."
        montage_only_tip = "Sans effet en aperçu canal — basculez en revue montage (Ctrl+M)."

        # Mode + Pipeline restent disponibles dans les deux vues (mêmes courbes).
        if hasattr(self, "_channel_host"):
            self._channel_host.setVisible(True)
            if hasattr(self, "_pipeline_sep"):
                self._pipeline_sep.setVisible(True)
            if hasattr(self, "_pipeline_header"):
                self._pipeline_header.setVisible(True)

        if self._channel_tab_index is not None:
            self.tabs.setTabVisible(self._channel_tab_index, True)
            tip = (
                "Mode + Pipeline : mêmes courbes que l’aperçu pour la revue montage. "
                "Traitement → Traiter (F5)"
                if montage
                else "Contenu du canal (mode, plages) et traitement → Traiter (F5)"
            )
            self.tabs.setTabToolTip(self._channel_tab_index, tip)

        # Toujours visibles ; actifs seulement dans la vue concernée.
        # Groupe revue (non repliable) : griser le bloc entier.
        # Groupe PDF (repliable) : garder le titre cliquable, griser seulement les champs
        # pour pouvoir l’ouvrir et voir les paramètres.
        if hasattr(self, "_montage_review_group"):
            self._montage_review_group.setVisible(True)
            self._montage_review_group.setEnabled(montage)
            base = getattr(self, "_montage_review_tooltip", "")
            self._montage_review_group.setToolTip(
                base if montage else f"{base} {montage_only_tip}".strip()
            )
        if hasattr(self, "_montage_pdf_group"):
            self._montage_pdf_group.setVisible(True)
            self._montage_pdf_group.setEnabled(True)
            base = getattr(self, "_montage_pdf_tooltip", "")
            self._montage_pdf_group.setToolTip(
                base if montage else f"{base} {montage_only_tip}".strip()
            )
        if hasattr(self, "_montage_row_height"):
            base = getattr(self, "_montage_row_tooltip", "")
            self._montage_row_height.setToolTip(
                base if montage else f"{base} {montage_only_tip}".strip()
            )
        if hasattr(self, "_montage_review_channels"):
            base = getattr(self, "_montage_review_channels_tooltip", "")
            self._montage_review_channels.setToolTip(
                base if montage else f"{base} {montage_only_tip}".strip()
            )
        if hasattr(self, "_montage_channels"):
            self._montage_channels.setEnabled(montage)
            base = getattr(self, "_montage_channels_tooltip", "")
            self._montage_channels.setToolTip(
                base if montage else f"{base} {montage_only_tip}".strip()
            )
        if hasattr(self, "_montage_page"):
            self._montage_page.setEnabled(montage)
            base = getattr(self, "_montage_page_tooltip", "")
            self._montage_page.setToolTip(
                base if montage else f"{base} {montage_only_tip}".strip()
            )

        # Hauteur des graphs : aperçu uniquement (le montage utilise la hauteur de ligne).
        if hasattr(self, "_layout_form") and hasattr(self, "_graph_height"):
            _set_form_row_visible(self._layout_form, self._graph_height, True)
            _set_form_row_enabled(self._layout_form, self._graph_height, not montage)
            base = getattr(self, "_graph_height_tooltip", "")
            self._graph_height.setToolTip(
                base if not montage else f"{base} {preview_only_tip}".strip()
            )

        if montage:
            self.tabs.setTabToolTip(
                1,
                "Temps, axe X, hauteur de ligne, légendes et apparence — "
                "redessin immédiat (échelles Y → Canal, par type de courbe). "
                "Hauteur des graphs grisée : sans effet ici.",
            )
        else:
            self.tabs.setTabToolTip(
                1,
                "Axe X, sync, disposition, légendes et apparence — "
                "redessin immédiat (échelles Y → Canal, par type de courbe). "
                "Revue montage / PDF grisés : sans effet en aperçu.",
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
            font_size=self._legend_font.size_value(),
            face=self._legend_font.face_value(),
            columns=int(self._legend_columns.value()),
            gap=float(self._legend_gap.value()),
            frame=self._legend_frame.isChecked(),
            show_filter_details=self._legend_filters.isChecked(),
            show_sample_counts=self._legend_counts.isChecked(),
            show_reference_markers=self._stim_markers.isChecked(),
        )
        style = PanelStyle(
            title_font_size=self._title_font.size_value(),
            label_font_size=self._label_font.size_value(),
            tick_font_size=self._tick_font.size_value(),
            title_face=self._title_font.face_value(),
            label_face=self._label_font.face_value(),
            tick_face=self._tick_font.face_value(),
            line_width=float(self._line_width.value()),
            grid=self._grid.isChecked(),
            grid_alpha=float(self._grid_alpha.value()),
            show_borders=self._show_borders.isChecked(),
            ticks_inside=self._ticks_inside.isChecked(),
            show_scale_bars=self._show_scale_bars.isChecked(),
            scale_bar_time_manual=bool(self._scale_bar_time_manual.isChecked()),
            scale_bar_time_s=float(self._scale_bar_time_ms.value()) / 1000.0,
            scale_bar_amplitude_manual=bool(self._scale_bar_amp_manual.isChecked()),
            scale_bar_amplitude=float(self._scale_bar_amp.value()),
            max_points_per_curve=int(self._max_points.value()),
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
        flags = self.pipeline_visibility_flags()
        analysis = AnalysisSettings(
            show_raw=bool(flags["show_raw"]),
            show_hp=bool(flags["show_hp"]),
            show_lp=bool(flags["show_lp"]),
            show_rms=bool(flags["show_rms"]),
            show_psth=bool(flags["show_psth"]),
            show_trial_rate=bool(flags["show_trial_rate"]),
            show_raster=bool(flags["show_raster"]),
            show_isi=bool(flags["show_isi"]),
            show_overlay=bool(flags["show_overlay"]),
        )
        return ViewerSettings(
            zoom_onset_t0_s=float(self._zoom_onset_t0.value()),
            zoom_onset_t1_s=float(self._zoom_onset_t1.value()),
            zoom_end_t0_s=float(self._zoom_end_t0.value()),
            zoom_end_t1_s=float(self._zoom_end_t1.value()),
            psth_bin_window_s=float(self._psth_bin.value()),
            sampling_percent=int(self._sampling.value()),
            spike_overlay_pre_ms=float(self._overlay_pre.value()),
            spike_overlay_post_ms=float(self._overlay_post.value()),
            x_limits=self._x_limits.value(),
            raw_ylim=self._raw_ylim.value(),
            hp_ylim=self._hp_ylim.value(),
            lp_ylim=self._lp_ylim.value(),
            rms_ylim=self._rms_ylim_row.value(),
            legend=legend,
            style=style,
            texts=texts,
            analysis=analysis,
            montage_channels=int(self._montage_channels.value()),
            montage_page=int(self._montage_page.value()),
            montage_review_channels=int(self._montage_review_channels.value()),
            montage_review_page=int(self._montage_review_page),
            graph_height_px=int(self._graph_height.value()),
            montage_row_min_height_px=int(self._montage_row_height.value()),
            time_sync=str(self._time_sync.currentData() or "recording_start"),  # type: ignore[arg-type]
            trigger_polarity=self._trigger_polarity_value(),  # type: ignore[arg-type]
            trigger_threshold=float(self._threshold.value()),
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
        # Front / seuil : source unique = Canal → Traitement.
        edge = str(self._edge.currentData())
        threshold = float(self._threshold.value())
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
            psth_bin_window_s=float(self._psth_bin.value()),
            spike_overlay_pre_ms=float(self._overlay_pre.value()),
            spike_overlay_post_ms=float(self._overlay_post.value()),
            zoom_onset_t0_s=float(self._zoom_onset_t0.value()),
            zoom_onset_t1_s=float(self._zoom_onset_t1.value()),
            zoom_end_t0_s=float(self._zoom_end_t0.value()),
            zoom_end_t1_s=float(self._zoom_end_t1.value()),
            first_trigger_hp_ylim_enabled=self._hp_ylim.value().enabled,
            first_trigger_hp_ylim_min_uv=self._hp_ylim.value().minimum,
            first_trigger_hp_ylim_max_uv=self._hp_ylim.value().maximum,
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
            sampling_percent=int(self._sampling.value()),
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
