"""Paramètres d’affichage et de traitement, selon le coût d’un changement.

Deux profils dock indépendants (mêmes noms, valeurs séparées) :

- **preview** — aperçu canal (grille centrale)
- **montage** — revue montage

Les fenêtres zoom ont leur propre scope (``LocalViewParams`` / ``settings_for_zoom_window``).
Les plages / zooms (``TimeRangeBar``) vivent sur l’aperçu canal, pas dans ce dock.

Le dock montre le profil de la vue active.

- *Canal* — courbes et échelles Y du profil actif, plus le traitement commun
  (filtres, spikes, RMS, détection) → :attr:`configChanged` / Traiter (F5).
- *Affichage* — axe X, sync, disposition, légendes, textes, apparence du
  profil actif → :attr:`viewChanged` (redessin immédiat, pas de recalcul).
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
from gui.defaults import DEFAULT_PSTH_BIN_S, DEFAULT_ZOOM_T0_S, DEFAULT_ZOOM_T1_S
from gui.jobs import Debouncer
from view_config import (
    EDGE_TO_TRIGGER_POLARITY,
    LEGEND_LOCATIONS,
    TIME_SYNC_LABELS,
    AnalysisSettings,
    AnalysisStream,
    AxisLimits,
    TextFace,
    TextOverrides,
    LegendSettings,
    PanelStyle,
    ViewerSettings,
    resolved_scale_bar_time_offset_px,
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
        # Contenu (continuous / moyenne / stimulation) du profil affiché.
        self._content_mode = "continuous"
        self._content_stim = 0
        self._profiles: dict[str, ViewerSettings] = {}
        # Ignored horizontal : le dock impose la largeur. Preferred/Expanding
        # faisait « aspirer » le layout vers sizeHint et écrasait le panneau au drag.
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        self._mode_hint = QLabel(
            "Canal + Pipeline → Traiter (F5) · Affichage → redessin immédiat"
        )
        self._mode_hint.setObjectName("hintLabel")
        self._mode_hint.setWordWrap(True)
        self._mode_hint.setToolTip(
            "Onglet Canal : contenu du canal et paramètres de calcul (filtres, spikes, RMS). "
            "Onglet Affichage : axe X, sync, légendes et apparence — appliqués sans F5."
        )

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
        layout.addWidget(self._mode_hint)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self._dirty_row)

        self._view_debouncer = Debouncer(120, self)
        self._view_debouncer.triggered.connect(self.viewChanged.emit)
        snap = self.viewer_settings()
        self._profiles = {"preview": snap, "montage": snap}
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
            "Hauteur fixe de chaque graphique (aperçu, analyse, onglets). "
            "Grisée si « Taille amp. » est en manuel "
            "(le bâton fixe alors le gain Y)."
        )
        self._graph_height.setToolTip(self._graph_height_tooltip)
        self._preview_title_gap = _int_spin(-40, 240, 0, suffix=" px")
        self._preview_title_gap_tooltip = (
            "Espace vertical entre le titre du graphique et la courbe, "
            "en aperçu canal. 0 = écart actuel. "
            "Positif agrandit l’espace, négatif le réduit jusqu’à coller le titre à la courbe."
        )
        self._preview_title_gap.setToolTip(self._preview_title_gap_tooltip)
        self._show_scale_bars = QCheckBox("Échelle flottante (barres temps / amplitude)")
        self._show_scale_bars.setChecked(False)
        self._show_scale_bars.setToolTip(
            "Masque les graduations numériques sur les bords (les titres d’axes "
            "restent) et affiche une barre d’échelle flottante (ex. 100 ms · 200 µV)."
        )
        self._scale_bar_amp_manual = QCheckBox("Manuel")
        self._scale_bar_amp_manual.setChecked(False)
        self._scale_bar_amp = _spin(
            0.01, 1_000_000.0, 100.0, decimals=2, step=10.0, suffix=" µV"
        )
        self._scale_bar_amp.setToolTip(
            "Amplitude du bâton vertical (ex. 50 µV). "
            "Seule : le bâton suit le zoom. "
            "Avec « Taille amp. » : fixe la hauteur du graph "
            "(pas de recalcul du libellé)."
        )
        self._scale_bar_amp_len_manual = QCheckBox("Manuel")
        self._scale_bar_amp_len_manual.setChecked(False)
        self._scale_bar_amp_len_px = _int_spin(4, 800, 40, step=5, suffix=" px")
        self._scale_bar_amp_len_px.setToolTip(
            "Hauteur du bâton en pixels. "
            "Seule : le libellé est calculé depuis le zoom. "
            "Avec Amplitude : fixe la hauteur du graph "
            "(ex. 50 µV = 40 px)."
        )
        self._scale_bar_time_manual = QCheckBox("Manuel")
        self._scale_bar_time_manual.setChecked(False)
        self._scale_bar_time_ms = _spin(
            0.01, 1_000_000.0, 100.0, decimals=2, step=10.0, suffix=" ms"
        )
        self._scale_bar_time_ms.setToolTip(
            "Durée du bâton horizontal (ex. 200 ms). "
            "Seule : le bâton suit le zoom X, libellé = cette durée. "
            "Avec « Taille temps » : fixe la largeur du graph "
            "(pas de recalcul du libellé)."
        )
        self._scale_bar_time_len_manual = QCheckBox("Manuel")
        self._scale_bar_time_len_manual.setChecked(False)
        self._scale_bar_time_len_px = _int_spin(4, 800, 40, suffix=" px")
        self._scale_bar_time_len_px.setToolTip(
            "Longueur du bâton de temps en pixels. "
            "Seule : le libellé est calculé depuis le zoom X. "
            "Avec Durée : fixe la largeur du graph "
            "(ex. 200 ms = 40 px)."
        )
        amp_row = QWidget()
        amp_lay = QHBoxLayout(amp_row)
        amp_lay.setContentsMargins(0, 0, 0, 0)
        amp_lay.setSpacing(6)
        amp_lay.addWidget(self._scale_bar_amp_manual)
        amp_lay.addWidget(self._scale_bar_amp, 1)
        amp_len_row = QWidget()
        amp_len_lay = QHBoxLayout(amp_len_row)
        amp_len_lay.setContentsMargins(0, 0, 0, 0)
        amp_len_lay.setSpacing(6)
        amp_len_lay.addWidget(self._scale_bar_amp_len_manual)
        amp_len_lay.addWidget(self._scale_bar_amp_len_px, 1)
        time_row = QWidget()
        time_lay = QHBoxLayout(time_row)
        time_lay.setContentsMargins(0, 0, 0, 0)
        time_lay.setSpacing(6)
        time_lay.addWidget(self._scale_bar_time_manual)
        time_lay.addWidget(self._scale_bar_time_ms, 1)
        time_len_row = QWidget()
        time_len_lay = QHBoxLayout(time_len_row)
        time_len_lay.setContentsMargins(0, 0, 0, 0)
        time_len_lay.setSpacing(6)
        time_len_lay.addWidget(self._scale_bar_time_len_manual)
        time_len_lay.addWidget(self._scale_bar_time_len_px, 1)
        self._scale_bar_linewidth = _spin(
            0.2, 20.0, 1.4, decimals=2, step=0.1, suffix=" pt"
        )
        self._scale_bar_linewidth.setToolTip(
            "Épaisseur du trait noir des barres d’échelle flottantes "
            "(temps et amplitude)."
        )
        self._scale_bar_dx = _int_spin(-400, 400, 0, suffix=" px")
        self._scale_bar_dx.setToolTip(
            "Décalage horizontal de l’échelle amplitude (axe vertical). "
            "0 = position par défaut. "
            "Positif décale vers la droite, négatif vers la gauche."
        )
        self._scale_bar_dy = _int_spin(-400, 400, 0, suffix=" px")
        self._scale_bar_dy.setToolTip(
            "Décalage vertical de l’échelle amplitude (axe vertical). "
            "0 = position par défaut. "
            "Positif monte la barre, négatif la descend."
        )
        self._scale_bar_time_dx = _int_spin(-400, 400, 0, suffix=" px")
        self._scale_bar_time_dx.setToolTip(
            "Décalage horizontal de l’échelle temps (axe horizontal). "
            "0 = position par défaut. "
            "Positif décale vers la droite, négatif vers la gauche."
        )
        self._scale_bar_time_dy = _int_spin(-400, 400, 0, suffix=" px")
        self._scale_bar_time_dy.setToolTip(
            "Décalage vertical de l’échelle temps (axe horizontal). "
            "0 = position par défaut. "
            "Positif monte la barre, négatif la descend."
        )
        self._scale_bar_amp_manual.toggled.connect(self._sync_scale_bar_inputs)
        self._scale_bar_time_manual.toggled.connect(self._sync_scale_bar_inputs)
        self._scale_bar_amp_len_manual.toggled.connect(self._sync_scale_bar_inputs)
        self._scale_bar_time_len_manual.toggled.connect(self._sync_scale_bar_inputs)
        self._show_scale_bars.toggled.connect(self._sync_scale_bar_inputs)
        self._sync_scale_bar_inputs()
        _max_pts_lo, _max_pts_hi = 500, 200_000
        _max_pts_mid = (_max_pts_lo + _max_pts_hi) // 2
        self._max_points = _int_spin(
            _max_pts_lo, _max_pts_hi, _max_pts_mid, step=500
        )
        self._max_points.setToolTip(
            "Points dessinés par courbe (aperçu, montage et fenêtres zoom). "
            "Au-delà, une enveloppe min/max conserve tous les pics tout en restant rapide."
        )
        max_pts_row = QWidget()
        max_pts_lay = QHBoxLayout(max_pts_row)
        max_pts_lay.setContentsMargins(0, 0, 0, 0)
        max_pts_lay.setSpacing(6)
        max_pts_min_btn = QPushButton("Min")
        max_pts_max_btn = QPushButton("Max")
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
        self._layout_form.addRow("Hauteur des graphs :", self._graph_height)
        self._layout_form.addRow("Écart titre / courbe :", self._preview_title_gap)
        self._layout_form.addRow(self._show_scale_bars)
        self._layout_form.addRow("Amplitude barre :", amp_row)
        self._layout_form.addRow("Taille amp. :", amp_len_row)
        self._layout_form.addRow("Temps barre :", time_row)
        self._layout_form.addRow("Taille temps :", time_len_row)
        self._layout_form.addRow("Épaisseur du trait :", self._scale_bar_linewidth)
        self._layout_form.addRow("Temps — horizontal :", self._scale_bar_time_dx)
        self._layout_form.addRow("Temps — vertical :", self._scale_bar_time_dy)
        self._layout_form.addRow("Amplitude — horizontal :", self._scale_bar_dx)
        self._layout_form.addRow("Amplitude — vertical :", self._scale_bar_dy)
        self._layout_form.addRow("Points max / courbe :", max_pts_row)

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
        self._montage_shared_ylim = QCheckBox("Même échelle pour toutes les courbes")
        self._montage_shared_ylim.setChecked(False)
        self._montage_shared_ylim_tooltip = (
            "Les courbes WIDE, HIGH et LOW de la page partagent la même échelle Y, "
            "calculée sur l’amplitude la plus grande des canaux visibles. "
            "Une petite courbe reste petite. "
            "Sans cette option, chaque canal est ajusté sur sa propre amplitude. "
            "Une échelle Y manuelle (Canal) reste prioritaire pour son type de courbe. "
            "RMS, PSTH et spikes partagent une échelle entre canaux du même type."
        )
        self._montage_shared_ylim.setToolTip(self._montage_shared_ylim_tooltip)
        review_form.addRow(self._montage_shared_ylim)
        self._montage_show_kind_label = QCheckBox(
            "Type à côté du canal (WIDE, LOW, HIGH…)"
        )
        self._montage_show_kind_label.setChecked(True)
        self._montage_show_kind_tooltip = (
            "Affiche ou masque le nom du flux à côté du canal "
            "(WIDE, LOW, HIGH, RMS, PSTH…). "
            "Décochez pour ne garder que le nom du canal. "
            "Les libellés eux-mêmes ne changent pas."
        )
        self._montage_show_kind_label.setToolTip(self._montage_show_kind_tooltip)
        review_form.addRow(self._montage_show_kind_label)
        self._montage_label_dx = _int_spin(-300, 300, 0, suffix=" px")
        self._montage_label_dx_tooltip = (
            "Décalage horizontal des noms de canal, à côté des courbes. "
            "0 = position par défaut. Positif rapproche le titre de la courbe, "
            "négatif l’éloigne vers la gauche."
        )
        self._montage_label_dx.setToolTip(self._montage_label_dx_tooltip)
        self._montage_label_dy = _int_spin(-120, 120, 0, suffix=" px")
        self._montage_label_dy_tooltip = (
            "Décalage vertical des noms de canal par rapport au centre de la ligne. "
            "0 = centré sur la courbe. Positif monte le titre, négatif le descend."
        )
        self._montage_label_dy.setToolTip(self._montage_label_dy_tooltip)
        review_form.addRow("Titres — horizontal :", self._montage_label_dx)
        review_form.addRow("Titres — vertical :", self._montage_label_dy)
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
            self._montage_label_dx,
            self._montage_label_dy,
            self._preview_title_gap,
            self._max_points,
        ):
            int_widget.valueChanged.connect(lambda _v: self._emit_view())
        self._graph_height.valueChanged.connect(lambda _v: self._emit_view())
        self._scale_bar_amp_len_px.valueChanged.connect(lambda _v: self._emit_view())
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
        self._montage_show_kind_label.toggled.connect(lambda _c: self._emit_view())
        self._montage_shared_ylim.toggled.connect(lambda _c: self._emit_view())
        self._scale_bar_amp_manual.toggled.connect(lambda _c: self._emit_view())
        self._scale_bar_time_manual.toggled.connect(lambda _c: self._emit_view())
        self._scale_bar_amp.valueChanged.connect(lambda _v: self._emit_view())
        self._scale_bar_time_ms.valueChanged.connect(lambda _v: self._emit_view())
        self._scale_bar_amp_len_manual.toggled.connect(lambda _c: self._emit_view())
        self._scale_bar_time_len_manual.toggled.connect(lambda _c: self._emit_view())
        self._scale_bar_time_len_px.valueChanged.connect(lambda _v: self._emit_view())
        self._scale_bar_linewidth.valueChanged.connect(lambda _v: self._emit_view())
        self._scale_bar_dx.valueChanged.connect(lambda _v: self._emit_view())
        self._scale_bar_dy.valueChanged.connect(lambda _v: self._emit_view())
        self._scale_bar_time_dx.valueChanged.connect(lambda _v: self._emit_view())
        self._scale_bar_time_dy.valueChanged.connect(lambda _v: self._emit_view())

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
        self._scale_bar_amp_len_manual.setEnabled(enabled)
        self._scale_bar_time_len_manual.setEnabled(enabled)
        self._scale_bar_amp_len_px.setEnabled(
            enabled and bool(self._scale_bar_amp_len_manual.isChecked())
        )
        self._scale_bar_time_len_px.setEnabled(
            enabled and bool(self._scale_bar_time_len_manual.isChecked())
        )
        self._scale_bar_linewidth.setEnabled(enabled)
        self._scale_bar_dx.setEnabled(enabled)
        self._scale_bar_dy.setEnabled(enabled)
        self._scale_bar_time_dx.setEnabled(enabled)
        self._scale_bar_time_dy.setEnabled(enabled)
        # Taille amp. manuelle → griser « Hauteur des graphs » (même formulaire).
        amp_len_on = enabled and bool(self._scale_bar_amp_len_manual.isChecked())
        if hasattr(self, "_graph_height"):
            self._graph_height.setEnabled(not amp_len_on)
            if hasattr(self, "_layout_form"):
                label = self._layout_form.labelForField(self._graph_height)
                if label is not None:
                    label.setEnabled(not amp_len_on)

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
        self._legend_font = FontSizeFaceRow(4.0, 50.0, 9.0, decimals=1, step=0.5)
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
        self._title_font = FontSizeFaceRow(5.0, 50.0, 10.0, decimals=1, step=0.5)
        self._title_dx = _int_spin(-400, 400, 0, suffix=" px")
        self._title_dx.setToolTip(
            "Décalage horizontal du titre de graphique. Positif = droite."
        )
        self._title_dy = _int_spin(-400, 400, 0, suffix=" px")
        self._title_dy.setToolTip(
            "Décalage vertical du titre de graphique. Positif = haut."
        )
        self._label_font = FontSizeFaceRow(5.0, 50.0, 9.0, decimals=1, step=0.5)
        self._tick_font = FontSizeFaceRow(4.0, 50.0, 8.0, decimals=1, step=0.5)
        self._line_width = _spin(0.2, 20.0, 1.2, decimals=2, step=0.1)
        self._stim_linewidth = _spin(0.2, 20.0, 1.0, decimals=2, step=0.1)
        self._stim_linewidth.setToolTip(
            "Épaisseur (points) des pointillés de début et de fin de stimulation."
        )
        self._stim_onset_color = ColorSwatchButton(
            STIM_ONSET_COLOR, dialog_title="Couleur du début de stimulation"
        )
        self._stim_onset_color.setToolTip(
            "Couleur des pointillés au début de chaque stimulation."
        )
        self._stim_offset_color = ColorSwatchButton(
            STIM_OFFSET_COLOR, dialog_title="Couleur de la fin de stimulation"
        )
        self._stim_offset_color.setToolTip(
            "Couleur des pointillés à la fin de chaque stimulation."
        )
        self._text_stim_on = QLineEdit("on")
        self._text_stim_on.setPlaceholderText("on")
        self._text_stim_on.setToolTip(
            "Texte au-dessus du trait de début de stimulation. Vide = masquer."
        )
        self._text_stim_off = QLineEdit("off")
        self._text_stim_off.setPlaceholderText("off")
        self._text_stim_off.setToolTip(
            "Texte au-dessus du trait de fin de stimulation. Vide = masquer."
        )
        self._stim_on_font = FontSizeFaceRow(4.0, 50.0, 8.0, decimals=1, step=0.5)
        self._stim_off_font = FontSizeFaceRow(4.0, 50.0, 8.0, decimals=1, step=0.5)
        self._stim_on_dx = _int_spin(-400, 400, 0, suffix=" px")
        self._stim_on_dx.setToolTip(
            "Décalage horizontal du libellé « on ». Positif = droite."
        )
        self._stim_on_dy = _int_spin(-400, 400, 0, suffix=" px")
        self._stim_on_dy.setToolTip(
            "Décalage vertical du libellé « on ». Positif = haut."
        )
        self._stim_off_dx = _int_spin(-400, 400, 0, suffix=" px")
        self._stim_off_dx.setToolTip(
            "Décalage horizontal du libellé « off ». Positif = droite."
        )
        self._stim_off_dy = _int_spin(-400, 400, 0, suffix=" px")
        self._stim_off_dy.setToolTip(
            "Décalage vertical du libellé « off ». Positif = haut."
        )
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
            "Lignes en pointillés au début et à la fin de stimulation "
            "(aperçu, revue montage et graphiques d’analyse). "
            "Sur les traces continues, décochez aussi "
            "« Marqueurs de stimulation sur les traces » pour les masquer."
        )
        self._show_stim_labels = QCheckBox("Libellés on / off sur les traits")
        self._show_stim_labels.setChecked(True)
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
        self._show_zone_labels.setChecked(True)
        self._show_zone_labels.setToolTip(
            "Afficher « Zone N » (ou le nom édité dans Plages / zooms) "
            "sous chaque bande de zoom, en bas du graphique."
        )
        self._zone_font = FontSizeFaceRow(4.0, 50.0, 8.0, decimals=1, step=0.5)
        self._zone_dx = _int_spin(-400, 400, 0, suffix=" px")
        self._zone_dx.setToolTip(
            "Décalage horizontal du libellé de zone. Positif = droite."
        )
        self._zone_dy = _int_spin(-400, 400, 0, suffix=" px")
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
            self._show_stim_labels,
            self._show_zone_labels,
        ):
            box.toggled.connect(lambda _c: self._emit_view())
        self._legend_location.currentIndexChanged.connect(lambda _i: self._emit_view())
        for spin in (
            self._legend_gap,
            self._line_width,
            self._stim_linewidth,
            self._grid_alpha,
            self._title_dx,
            self._title_dy,
            self._stim_on_dx,
            self._stim_on_dy,
            self._stim_off_dx,
            self._stim_off_dy,
            self._zone_dx,
            self._zone_dy,
        ):
            spin.valueChanged.connect(lambda _v: self._emit_view())
        self._stim_onset_color.changed.connect(self._emit_view)
        self._stim_offset_color.changed.connect(self._emit_view)
        for font_row in (
            self._legend_font,
            self._title_font,
            self._label_font,
            self._tick_font,
            self._stim_on_font,
            self._stim_off_font,
            self._zone_font,
        ):
            font_row.changed.connect(self._emit_view)
        self._legend_columns.valueChanged.connect(lambda _v: self._emit_view())
        for line in (
            self._text_title,
            self._text_xlabel,
            self._text_ylabel,
            self._text_stim_on,
            self._text_stim_off,
        ):
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
        self._pre_s.setToolTip(
            "Fenêtre avant t=0 (s) pour moyennes, RMS, 1re/2e stim, etc. "
            "Identique avec déclencheur ANALOG-IN ou sections sans stimulation."
        )
        self._post_s = _spin(0.001, 3600.0, defaults.get("default_post_s", 10.0), suffix=" s")
        self._post_s.setToolTip(
            "Fenêtre après t=0 (s). Même système d’affichage avec ou sans stimulation réelle."
        )
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
        self._section_start.setToolTip(
            "Position du début virtuel de stimulation dans chaque section (t=0). "
            "Les courbes affichent [-avant, +après] autour de ce point."
        )
        self._section_end = _spin(
            0.0, 3600.0, defaults.get("default_section_trigger_end_s", 4.0), suffix=" s"
        )
        self._section_end.setToolTip(
            "Position de la fin virtuelle de stimulation dans chaque section "
            "(marqueurs on/off), relative au début de section."
        )
        trigger_form.addRow("Front de déclenchement :", self._edge)
        trigger_form.addRow("Seuil de déclenchement :", self._threshold)
        trigger_form.addRow("Fenêtre avant :", self._pre_s)
        trigger_form.addRow("Fenêtre après :", self._post_s)
        self._section_spec.currentIndexChanged.connect(lambda _i: self._sync_section_rows())
        self._edge.currentIndexChanged.connect(lambda _i: self._on_processing_edge_changed())

        # Mode sans déclencheur : découpage en sections + stims virtuelles ;
        # les fenêtres [-avant,+après] restent celles du formulaire ci-dessus.
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
            "Cochez un groupe pour afficher ses courbes dans la vue active "
            "(aperçu ou montage) et régler ses paramètres. "
            "Le traitement F5 reste commun, même si une case est décochée."
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
                "Fenêtre de forme d’onde autour de chaque détection. "
                "Le spike scope s’y adapte tout de suite (les flux filtrés restent en cache)."
            )
        self._overlay_alpha = _spin(0.0, 1.0, 0.4, decimals=2, step=0.05)
        self._overlay_alpha.setToolTip(
            "Opacité des formes d’onde individuelles dans le spike scope "
            "(0 = invisible, 1 = opaque). La courbe moyenne reste opaque. "
            "Redessin immédiat — pas de Traiter (F5)."
        )
        self._show_spike_threshold = QCheckBox("Afficher le seuil sur les courbes")
        self._show_spike_threshold.setChecked(False)
        self._show_spike_threshold.setToolTip(
            "Ligne horizontale du seuil de détection (µV) sur les courbes WIDE, HIGH, "
            "LOW et le spike scope. Redessin immédiat — pas de Traiter (F5)."
        )
        spike_form.addRow("Mode de seuil :", self._threshold_mode)
        spike_form.addRow("Seuil fixe :", self._spike_threshold)
        spike_form.addRow("Polarité :", self._polarity)
        spike_form.addRow(self._show_spike_threshold)
        spike_form.addRow("Multiplicateur RMS :", self._rms_multiplier)
        self._psth_bin = _spin(
            0.001,
            10.0,
            float(defaults.get("default_psth_bin_window_s", DEFAULT_PSTH_BIN_S)),
            decimals=3,
            suffix=" s",
        )
        self._psth_bin.setToolTip(
            "Largeur des bins du PSTH / firing rate à l’écran (redessin immédiat). "
            "L’export PDF / le cache utilisent aussi cette valeur au prochain Traiter (F5)."
        )
        self._sampling = _int_spin(
            1, 100, int(defaults.get("default_sampling_percent", 100) or 100)
        )
        self._sampling.setToolTip(
            "Pourcentage d’essais / spikes dessinés (sous-échantillon). "
            "Redessin immédiat à l’écran ; pris en compte aussi au prochain Traiter (F5)."
        )
        spike_form.addRow("Avant le spike :", self._overlay_pre)
        spike_form.addRow("Après le spike :", self._overlay_post)
        spike_form.addRow("Opacité des traces :", self._overlay_alpha)
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
            "Raster compact (tous essais sur une ligne).\n"
            "Revue montage : une ligne RST par canal."
        )
        self._cb_pipe_raster_trials = QCheckBox("Raster / essai")
        self._cb_pipe_raster_trials.setToolTip(
            "Raster avec une ligne par essai (aperçu canal uniquement)."
        )
        self._cb_pipe_isi = QCheckBox("ISI")
        self._cb_pipe_overlay = QCheckBox("Spike scope")
        for box in (
            self._cb_pipe_psth,
            self._cb_pipe_trial_rate,
            self._cb_pipe_raster,
            self._cb_pipe_raster_trials,
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
            AxisLimits(enabled=False, minimum=0.0, maximum=20.0),
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
        ):
            spin.valueChanged.connect(lambda _v: self._emit_config())
        # Fenêtre du spike scope : redessin immédiat, sans invalider le cache filtres.
        self._overlay_pre.valueChanged.connect(lambda _v: self._emit_view())
        self._overlay_post.valueChanged.connect(lambda _v: self._emit_view())
        self._overlay_alpha.valueChanged.connect(lambda _v: self._emit_view())
        self._show_spike_threshold.toggled.connect(lambda _c: self._emit_view())
        # Bin PSTH / sampling : display (redraw) + config (PDF / F5).
        self._psth_bin.valueChanged.connect(lambda _v: self._emit_display_and_config())
        self._threshold.valueChanged.connect(lambda _v: self._on_processing_threshold_changed())
        for int_spin in (
            self._section_count,
            self._hp_filter_order,
            self._lp_filter_order,
            self._channel_workers,
        ):
            int_spin.valueChanged.connect(lambda _v: self._emit_config())
        self._sampling.valueChanged.connect(lambda _v: self._emit_display_and_config())
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
            self._cb_pipe_raster_trials,
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

    def _emit_display_and_config(self) -> None:
        """Réglages mixtes : redessin immédiat + marquage dirty pour F5 / PDF."""
        if self._loading:
            return
        self._emit_config()
        self._emit_view()

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
                self._cb_pipe_raster_trials,
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
            "show_raster_trials": spikes_on
            and bool(self._cb_pipe_raster_trials.isChecked()),
            "show_isi": spikes_on and bool(self._cb_pipe_isi.isChecked()),
            "show_overlay": spikes_on and bool(self._cb_pipe_overlay.isChecked()),
        }

    # Alias historique.

    def sync_analysis_checks_from(self, analysis: AnalysisSettings | object) -> None:
        """Aligne les cases Pipeline sur un ``AnalysisSettings`` (sans réémettre)."""
        previous = self._loading
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
                bool(getattr(analysis, "show_raster_trials", False)),
                bool(getattr(analysis, "show_isi", False)),
                bool(getattr(analysis, "show_overlay", False)),
            )
            any_spike = any(spike_flags)
            self._curve_spikes_box.setChecked(any_spike)
            self._cb_pipe_psth.setChecked(spike_flags[0])
            self._cb_pipe_trial_rate.setChecked(spike_flags[1])
            self._cb_pipe_raster.setChecked(spike_flags[2])
            self._cb_pipe_raster_trials.setChecked(spike_flags[3])
            self._cb_pipe_isi.setChecked(spike_flags[4])
            self._cb_pipe_overlay.setChecked(spike_flags[5])
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
            self._loading = previous

    def mark_stimulations(self) -> bool:
        return bool(self._cb_mark_stims.isChecked())

    def set_dirty_message(self, message: str) -> None:
        self._dirty_label.setText(message)
        self._dirty_row.setVisible(bool(message))

    def set_content_mode(
        self, mode: str, stim_index: int = 0, *, emit: bool = True
    ) -> None:
        """Mémoriser le mode d’affichage du profil en cours (aperçu ou montage)."""
        raw = str(mode or "continuous")
        if raw not in ("continuous", "average", "stimulation"):
            raw = "continuous"
        stim = max(0, int(stim_index))
        if raw == self._content_mode and stim == self._content_stim:
            return
        self._content_mode = raw
        self._content_stim = stim
        if emit:
            self._emit_view()

    def settings_for(self, mode: ViewMode) -> ViewerSettings:
        """Réglages d’une vue, y compris si elle n’est pas affichée."""
        resolved: ViewMode = mode if mode in ("preview", "montage") else "preview"
        if resolved == self._view_mode:
            return self.viewer_settings()
        stored = self._profiles.get(resolved)
        return stored if stored is not None else self.viewer_settings()

    def set_view_mode(self, mode: ViewMode, *, emit: bool = True) -> None:
        """Afficher le profil de l’aperçu ou celui de la revue montage."""
        resolved: ViewMode = mode if mode in ("preview", "montage") else "preview"
        if resolved != self._view_mode:
            self._profiles[self._view_mode] = self.viewer_settings()
            self._view_mode = resolved
            stored = self._profiles.get(resolved)
            if stored is not None:
                self._load_profile(stored)
        self._apply_view_mode_visibility()
        if emit and not self._loading:
            self.viewChanged.emit()

    def _load_profile(self, settings: ViewerSettings) -> None:
        """Recharger les widgets d’affichage depuis un profil (sans signal)."""
        self._loading = True
        try:
            self._content_mode = str(getattr(settings, "preview_content", None) or "continuous")
            if self._content_mode not in ("continuous", "average", "stimulation"):
                self._content_mode = "continuous"
            self._content_stim = max(0, int(settings.analysis.stim_index))
            self._x_limits.set_value(settings.x_limits)
            self._raw_ylim.set_value(settings.raw_ylim)
            self._hp_ylim.set_value(settings.hp_ylim)
            self._lp_ylim.set_value(settings.lp_ylim)
            self._rms_ylim_row.set_value(settings.rms_ylim)
            self._zoom_onset_t0.setValue(float(settings.zoom_onset_t0_s))
            self._zoom_onset_t1.setValue(float(settings.zoom_onset_t1_s))
            self._zoom_end_t0.setValue(float(settings.zoom_end_t0_s))
            self._zoom_end_t1.setValue(float(settings.zoom_end_t1_s))
            self._set_combo_data(self._time_sync, settings.time_sync)
            self._cb_mark_stims.setChecked(bool(settings.continuous_mark_stims))
            self._graph_height.setValue(
                max(160, min(1200, int(settings.graph_height_px)))
            )
            self._preview_title_gap.setValue(int(settings.preview_title_gap_px))
            style = settings.style
            self._show_scale_bars.setChecked(bool(style.show_scale_bars))
            self._scale_bar_amp_manual.setChecked(bool(style.scale_bar_amplitude_manual))
            self._scale_bar_amp.setValue(float(style.scale_bar_amplitude))
            self._scale_bar_time_manual.setChecked(bool(style.scale_bar_time_manual))
            self._scale_bar_time_ms.setValue(float(style.scale_bar_time_s) * 1000.0)
            self._scale_bar_amp_len_manual.setChecked(
                bool(style.scale_bar_amplitude_length_manual)
            )
            self._scale_bar_amp_len_px.setValue(
                max(
                    4,
                    min(
                        800,
                        int(round(float(style.scale_bar_amplitude_length_px or 40.0))),
                    ),
                )
            )
            self._scale_bar_time_len_manual.setChecked(
                bool(style.scale_bar_time_length_manual)
            )
            self._scale_bar_time_len_px.setValue(
                int(round(float(style.scale_bar_time_length_px)))
            )
            self._scale_bar_linewidth.setValue(float(style.scale_bar_linewidth))
            self._scale_bar_dx.setValue(int(round(float(style.scale_bar_dx_px))))
            self._scale_bar_dy.setValue(int(round(float(style.scale_bar_dy_px))))
            time_dx, time_dy = resolved_scale_bar_time_offset_px(style)
            self._scale_bar_time_dx.setValue(int(round(float(time_dx))))
            self._scale_bar_time_dy.setValue(int(round(float(time_dy))))
            self._sync_scale_bar_inputs()
            self._max_points.setValue(int(style.max_points_per_curve))
            self._montage_review_channels.setValue(int(settings.montage_review_channels))
            self._montage_review_page = max(0, int(settings.montage_review_page))
            self._montage_row_height.setValue(int(settings.montage_row_min_height_px))
            self._montage_show_kind_label.setChecked(bool(settings.montage_show_kind_label))
            self._montage_shared_ylim.setChecked(bool(settings.montage_shared_ylim))
            self._montage_label_dx.setValue(int(round(float(settings.montage_label_dx_px))))
            self._montage_label_dy.setValue(int(round(float(settings.montage_label_dy_px))))
            self._montage_channels.setValue(int(settings.montage_channels))
            self._montage_page.setValue(int(settings.montage_page))
            legend = settings.legend
            self._legend_visible.setChecked(bool(legend.visible))
            self._set_combo_data(self._legend_location, legend.location)
            self._legend_font.set_size_value(float(legend.font_size))
            self._legend_font.set_face_value(legend.face)
            self._legend_columns.setValue(int(legend.columns))
            self._legend_gap.setValue(float(legend.gap))
            self._legend_frame.setChecked(bool(legend.frame))
            self._legend_filters.setChecked(bool(legend.show_filter_details))
            self._legend_counts.setChecked(bool(legend.show_sample_counts))
            texts = settings.texts
            self._text_title.setText(str(texts.title or ""))
            self._text_xlabel.setText(str(texts.xlabel or ""))
            self._text_ylabel.setText(str(texts.ylabel or ""))
            self._text_series_suffix.setChecked(bool(texts.show_series_suffix))
            self._text_legend_labels.setPlainText(
                "\n".join(texts.legend_labels) if texts.legend_labels else ""
            )
            self._text_stim_on.setText(str(getattr(texts, "stim_on", "on") or ""))
            self._text_stim_off.setText(str(getattr(texts, "stim_off", "off") or ""))
            self._title_font.set_size_value(float(style.title_font_size))
            self._title_font.set_face_value(style.title_face)
            self._title_dx.setValue(
                int(round(float(getattr(style, "title_dx_px", 0.0) or 0.0)))
            )
            self._title_dy.setValue(
                int(round(float(getattr(style, "title_dy_px", 0.0) or 0.0)))
            )
            self._label_font.set_size_value(float(style.label_font_size))
            self._label_font.set_face_value(style.label_face)
            self._tick_font.set_size_value(float(style.tick_font_size))
            self._tick_font.set_face_value(style.tick_face)
            self._stim_on_font.set_size_value(
                float(getattr(style, "stim_on_font_size", 8.0) or 8.0)
            )
            self._stim_on_font.set_face_value(
                getattr(style, "stim_on_face", None) or TextFace()
            )
            self._stim_off_font.set_size_value(
                float(getattr(style, "stim_off_font_size", 8.0) or 8.0)
            )
            self._stim_off_font.set_face_value(
                getattr(style, "stim_off_face", None) or TextFace()
            )
            self._stim_on_dx.setValue(
                int(round(float(getattr(style, "stim_on_dx_px", 0.0) or 0.0)))
            )
            self._stim_on_dy.setValue(
                int(round(float(getattr(style, "stim_on_dy_px", 0.0) or 0.0)))
            )
            self._stim_off_dx.setValue(
                int(round(float(getattr(style, "stim_off_dx_px", 0.0) or 0.0)))
            )
            self._stim_off_dy.setValue(
                int(round(float(getattr(style, "stim_off_dy_px", 0.0) or 0.0)))
            )
            self._line_width.setValue(float(style.line_width))
            self._stim_linewidth.setValue(float(style.stim_linewidth))
            self._stim_onset_color.set_color(str(style.stim_onset_color))
            self._stim_offset_color.set_color(str(style.stim_offset_color))
            self._show_stim_labels.setChecked(
                bool(getattr(style, "show_stim_labels", True))
            )
            self._sync_stim_label_inputs()
            self._show_zone_labels.setChecked(
                bool(getattr(style, "show_zone_labels", True))
            )
            self._zone_font.set_size_value(
                float(getattr(style, "zone_font_size", 8.0) or 8.0)
            )
            self._zone_font.set_face_value(
                getattr(style, "zone_face", None) or TextFace()
            )
            self._zone_dx.setValue(
                int(round(float(getattr(style, "zone_dx_px", 0.0) or 0.0)))
            )
            self._zone_dy.setValue(
                int(round(float(getattr(style, "zone_dy_px", 0.0) or 0.0)))
            )
            self._sync_zone_label_inputs()
            self._grid.setChecked(bool(style.grid))
            self._grid_alpha.setValue(float(style.grid_alpha))
            self._show_borders.setChecked(bool(style.show_borders))
            self._ticks_inside.setChecked(bool(style.ticks_inside))
            self._stim_markers.setChecked(bool(legend.show_reference_markers))
            self._overlay_pre.setValue(float(settings.spike_overlay_pre_ms))
            self._overlay_post.setValue(float(settings.spike_overlay_post_ms))
            self._overlay_alpha.setValue(float(settings.spike_overlay_alpha))
            self._show_spike_threshold.setChecked(bool(settings.show_spike_threshold))
            self._psth_bin.setValue(float(settings.psth_bin_window_s))
            self._sampling.setValue(int(settings.sampling_percent))
            self.sync_analysis_checks_from(settings.analysis)
        finally:
            self._loading = False

    @staticmethod
    def _set_combo_data(box: QComboBox, value: object) -> None:
        index = box.findData(value)
        box.setCurrentIndex(index if index >= 0 else 0)

    def _apply_view_mode_visibility(self) -> None:
        """Montrer les paramètres de la vue active ; masquer ceux de l’autre."""
        montage = self._view_mode == "montage"

        # Courbes, échelles et apparence : mêmes noms, valeurs du profil actif.
        if hasattr(self, "_channel_host"):
            self._channel_host.setVisible(True)
            if hasattr(self, "_pipeline_sep"):
                self._pipeline_sep.setVisible(True)
            if hasattr(self, "_pipeline_header"):
                self._pipeline_header.setVisible(True)

        view_name = "revue montage" if montage else "aperçu canal"
        if self._channel_tab_index is not None:
            self.tabs.setTabVisible(self._channel_tab_index, True)
            self.tabs.setTabToolTip(
                self._channel_tab_index,
                f"Courbes et échelles Y de la {view_name}. "
                "Le traitement (filtres, détection) reste commun → Traiter (F5).",
            )

        if hasattr(self, "_montage_review_group"):
            self._montage_review_group.setVisible(montage)
            self._montage_review_group.setEnabled(True)
        if hasattr(self, "_montage_pdf_group"):
            self._montage_pdf_group.setVisible(montage)
            self._montage_pdf_group.setEnabled(True)
        if hasattr(self, "_layout_form") and hasattr(self, "_graph_height"):
            _set_form_row_visible(self._layout_form, self._graph_height, not montage)
            # Respecter le grisé « Taille amp. manuelle » (ne pas forcer True).
            self._sync_scale_bar_inputs()
        if hasattr(self, "_layout_form") and hasattr(self, "_preview_title_gap"):
            _set_form_row_visible(self._layout_form, self._preview_title_gap, not montage)
            _set_form_row_enabled(self._layout_form, self._preview_title_gap, True)

        self.tabs.setTabToolTip(
            1,
            f"Affichage de la {view_name} — indépendant de l’autre vue. "
            "Redessin immédiat.",
        )


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
            grid=self._grid.isChecked(),
            grid_alpha=float(self._grid_alpha.value()),
            show_borders=self._show_borders.isChecked(),
            ticks_inside=self._ticks_inside.isChecked(),
            show_scale_bars=self._show_scale_bars.isChecked(),
            scale_bar_time_manual=bool(self._scale_bar_time_manual.isChecked()),
            scale_bar_time_s=float(self._scale_bar_time_ms.value()) / 1000.0,
            scale_bar_amplitude_manual=bool(self._scale_bar_amp_manual.isChecked()),
            scale_bar_amplitude=float(self._scale_bar_amp.value()),
            scale_bar_amplitude_length_manual=bool(
                self._scale_bar_amp_len_manual.isChecked()
            ),
            scale_bar_amplitude_length_px=float(self._scale_bar_amp_len_px.value()),
            scale_bar_time_length_manual=bool(
                self._scale_bar_time_len_manual.isChecked()
            ),
            scale_bar_time_length_px=float(self._scale_bar_time_len_px.value()),
            scale_bar_linewidth=float(self._scale_bar_linewidth.value()),
            scale_bar_dx_px=float(self._scale_bar_dx.value()),
            scale_bar_dy_px=float(self._scale_bar_dy.value()),
            scale_bar_time_dx_px=float(self._scale_bar_time_dx.value()),
            scale_bar_time_dy_px=float(self._scale_bar_time_dy.value()),
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
            stim_on=self._text_stim_on.text(),
            stim_off=self._text_stim_off.text(),
        )
        flags = self.pipeline_visibility_flags()
        content = str(self._content_mode or "continuous")
        if content not in ("continuous", "average", "stimulation"):
            content = "continuous"
        analysis_mode = "stimulation" if content == "stimulation" else "average"
        analysis = AnalysisSettings(
            mode=analysis_mode,  # type: ignore[arg-type]
            stim_index=max(0, int(self._content_stim)),
            show_raw=bool(flags["show_raw"]),
            show_hp=bool(flags["show_hp"]),
            show_lp=bool(flags["show_lp"]),
            show_rms=bool(flags["show_rms"]),
            show_psth=bool(flags["show_psth"]),
            show_trial_rate=bool(flags["show_trial_rate"]),
            show_raster=bool(flags["show_raster"]),
            show_raster_trials=bool(flags["show_raster_trials"]),
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
            spike_overlay_alpha=float(self._overlay_alpha.value()),
            show_spike_threshold=bool(self._show_spike_threshold.isChecked()),
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
            graph_height_px=max(160, min(1200, int(self._graph_height.value()))),
            preview_title_gap_px=int(self._preview_title_gap.value()),
            montage_row_min_height_px=int(self._montage_row_height.value()),
            montage_show_kind_label=bool(self._montage_show_kind_label.isChecked()),
            montage_shared_ylim=bool(self._montage_shared_ylim.isChecked()),
            montage_label_dx_px=float(self._montage_label_dx.value()),
            montage_label_dy_px=float(self._montage_label_dy.value()),
            time_sync=str(self._time_sync.currentData() or "recording_start"),  # type: ignore[arg-type]
            trigger_polarity=self._trigger_polarity_value(),  # type: ignore[arg-type]
            trigger_threshold=float(self._threshold.value()),
            continuous_stream=self.continuous_streams()[0],
            continuous_streams=self.continuous_streams(),
            continuous_mark_stims=self.mark_stimulations(),
            preview_content=content,  # type: ignore[arg-type]
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
