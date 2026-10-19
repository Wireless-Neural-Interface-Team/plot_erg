"""Live display settings for the interactive viewer.

Separates two kinds of settings:

- :class:`ViewerSettings` — everything that only changes how cached data is
  drawn. Changing it never invalidates the processed dataset, so panels can be
  redrawn immediately.
- :class:`WorkspaceLayout` — which panels are shown, in which tab, in which
  order.

Recompute-level parameters (filters, segmentation, spike threshold) stay in
:class:`config.AnalysisConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Sequence

from display_config import (
    STIM_OFFSET_COLOR,
    STIM_ONSET_COLOR,
    ZOOM_RANGE_COLORS,
    PlotDisplaySettings,
    SectionPanels,
    ZoomMode,
)
from panel_catalog import (
    EXTRA_CHANNEL_PANEL_FIELD_NAMES,
    GLOBAL_PANEL_FIELD_NAMES,
    SECTION_PANEL_FIELD_NAMES as PANEL_FIELD_NAMES,
    is_global_panel,
    is_section_independent,
    merge_product_needs,
    panel_label,
    panel_needs_spikes,
    panel_product_needs,
)

SectionKey = Literal["full", "zoom_onset", "zoom_trigger_end"]

# Flux tension affichés (WIDE / HIGH / LOW).
_ANALYSIS_STREAMS = ("raw", "hp", "lp")


def preview_needs_means(preview_content: str) -> bool:
    """True seulement pour la moyenne d’essais.

    Le mode ``stimulation`` lit des fenêtres memmap (``trigger_window``) :
    pas besoin de précalculer les moyennes.
    """
    return str(preview_content or "").strip() == "average"


def analysis_display_streams(analysis: object | None) -> tuple[str, ...]:
    """Flux WIDE/HIGH/LOW cochés pour l’affichage courant."""
    if analysis is None:
        return ("raw",)
    streams: list[str] = []
    if bool(getattr(analysis, "show_raw", False)):
        streams.append("raw")
    if bool(getattr(analysis, "show_hp", False)):
        streams.append("hp")
    if bool(getattr(analysis, "show_lp", False)):
        streams.append("lp")
    return tuple(streams) if streams else ("raw",)


def display_filter_streams(
    *,
    preview_content: str | None = None,
    analysis: object | None = None,
    continuous_streams: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Flux filtrés (hp/lp) à chauffer pour la vue affichée — jamais le raw."""
    mode = str(preview_content or "").strip()
    if mode in {"average", "stimulation"}:
        wanted = analysis_display_streams(analysis)
    elif continuous_streams is not None:
        wanted = tuple(
            str(s) for s in continuous_streams if str(s) in _ANALYSIS_STREAMS
        )
    else:
        wanted = analysis_display_streams(analysis)
    return tuple(s for s in wanted if s in {"hp", "lp"})


def display_mean_streams(analysis: object | None) -> tuple[str, ...]:
    """Flux pour lesquels calculer la moyenne d’essais (mode average)."""
    return analysis_display_streams(analysis)


def display_rms_kinds(
    *,
    preview_content: str | None = None,
    analysis: object | None = None,
) -> tuple[str, ...]:
    """Kinds RMS réellement lus par la vue (mean / first / second)."""
    mode = str(preview_content or "").strip()
    if mode == "stimulation" and analysis is not None:
        idx = int(getattr(analysis, "stim_index", 0) or 0)
        if idx == 0:
            return ("first",)
        if idx == 1:
            return ("second",)
        # Indices > 1 : le panneau retombe sur le profil « mean » stocké.
        return ("mean",)
    return ("mean",)


def included_channel_indices(
    n_channels: int,
    names: Sequence[str],
    hidden: Sequence[str] = (),
) -> list[int]:
    """Indices des canaux cochés (hors noms masqués)."""
    hidden_set = {str(name) for name in hidden}
    count = max(0, int(n_channels))
    return [
        index
        for index in range(count)
        if (names[index] if index < len(names) else f"CH{index}") not in hidden_set
    ]


SECTION_KEYS: tuple[SectionKey, ...] = ("full", "zoom_onset", "zoom_trigger_end")

SECTION_LABELS: dict[str, str] = {
    "full": "Vue complète",
    "zoom_onset": "Zoom début",
    "zoom_trigger_end": "Zoom fin",
}

LegendLocation = Literal[
    "below",
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center right",
    "center left",
]

LEGEND_LOCATIONS: tuple[LegendLocation, ...] = (
    "below",
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center right",
    "center left",
)

AnalysisMode = Literal["average", "stimulation"]
# Contenu de l’aperçu canal (fenêtre principale) : continuous ou graphs d’analyse.
PreviewContentMode = Literal["continuous", "average", "stimulation"]
AnalysisStream = Literal["raw", "hp", "lp"]
TimeSyncMode = Literal["recording_start", "trigger"]
TriggerPolarity = Literal["low", "high"]  # low = falling, high = rising

STREAM_SHORT_LABELS: dict[str, str] = {
    "raw": "WIDE",
    "hp": "HIGH",
    "lp": "LOW",
}

# Suffixes des lignes montage (flux + graphs Pipeline). Les noms affichés
# restent ceux-ci ; seule leur présence à côté du canal est optionnelle.
MONTAGE_KIND_SHORT_LABELS: dict[str, str] = {
    **STREAM_SHORT_LABELS,
    "rms": "RMS",
    "psth": "PSTH",
    "trial_rate": "FR",
    "raster": "RST",
    "isi": "ISI",
    "overlay": "WV",
}

TIME_SYNC_LABELS: dict[TimeSyncMode, str] = {
    "recording_start": "Début de l’enregistrement",
    "trigger": "Trigger détecté",
}

TRIGGER_POLARITY_TO_EDGE: dict[TriggerPolarity, str] = {
    "low": "falling",
    "high": "rising",
}

EDGE_TO_TRIGGER_POLARITY: dict[str, TriggerPolarity] = {
    "falling": "low",
    "rising": "high",
}


def montage_kind_short(kind: str) -> str:
    """Libellé court d’un flux ou graph de montage (WIDE, LOW, RMS, …)."""
    key = str(kind or "").strip().lower()
    if key in MONTAGE_KIND_SHORT_LABELS:
        return MONTAGE_KIND_SHORT_LABELS[key]
    return key.upper()


def montage_row_axis_label(channel_name: str, kind: str, *, show_kind: bool) -> str:
    """Nom de canal, avec ou sans le type (WIDE, LOW, HIGH, RMS, …)."""
    name = str(channel_name or "").strip()
    if not show_kind:
        return name
    short = montage_kind_short(kind)
    return f"{name} {short}".strip() if short else name


def continuous_sync_offset_s(
    stim_times_s: Sequence[float],
    mode: TimeSyncMode,
) -> float:
    """Décalage à soustraire du temps absolu pour l’affichage continu.

    - ``recording_start`` → 0 (t=0 au début du fichier)
    - ``trigger`` → premier onset détecté (t=0 sur le trigger)
    """
    if mode != "trigger":
        return 0.0
    times = [float(t) for t in stim_times_s]
    if not times:
        return 0.0
    return float(times[0])


def continuous_anchor_offset_s(
    stim_times_s: Sequence[float],
    mode: TimeSyncMode,
    *,
    stim_index: int | None = None,
    anchor_to_stim: bool = False,
) -> float:
    """Offset continuous pour lecture memmap / axe affiché.

    Quand ``anchor_to_stim`` est vrai (zoom / axe X relatif à un essai), ancre
    sur ``stim_index`` comme les courbes stimulation / moyenne. Sinon délègue à
    ``continuous_sync_offset_s``.
    """
    if anchor_to_stim and stim_index is not None:
        times = [float(t) for t in stim_times_s]
        idx = int(stim_index)
        if 0 <= idx < len(times):
            return times[idx]
    return continuous_sync_offset_s(stim_times_s, mode)


@dataclass(frozen=True)
class TimeRangeBar:
    """Paire de barres temporelles [t0, t1] pour zoom / plage de traitement.

    Si ``relative_to_stim`` est True, ``t0_s`` / ``t1_s`` sont relatifs au début
    de stimulation (t=0). Sinon ce sont des temps absolus sur l’enregistrement.
    ``color`` vide → palette ``ZOOM_RANGE_COLORS`` selon l’index d’affichage.
    """

    t0_s: float
    t1_s: float
    label: str = ""
    bar_id: str = ""
    relative_to_stim: bool = False
    color: str = ""

    def ordered(self) -> tuple[float, float]:
        a, b = float(self.t0_s), float(self.t1_s)
        return (a, b) if a <= b else (b, a)

    def with_bounds(self, t0_s: float, t1_s: float) -> TimeRangeBar:
        return replace(self, t0_s=float(t0_s), t1_s=float(t1_s))

    def with_color(self, color: str) -> TimeRangeBar:
        return replace(self, color=str(color or "").strip())


@dataclass(frozen=True)
class AnalysisSettings:
    """Courbes / mode d’aperçu — mêmes flags pour continuous, moyenne et stimulation.

    ``show_raw`` / ``show_hp`` / ``show_lp`` pilotent aussi les flux continuous.
    ``mode`` / ``stim_index`` ne changent que la source des traces (brut, mean, trial).
    """

    mode: AnalysisMode = "average"
    stim_index: int = 0  # 0-based stimulation index when mode == stimulation
    # Courbes : coches Paramètres → Canal → Pipeline (tous modes).
    show_raw: bool = True
    show_hp: bool = False
    show_lp: bool = False
    show_rms: bool = False
    show_isi: bool = False
    show_overlay: bool = False  # spike scope
    # Spikes : mêmes coches Pipeline.
    show_psth: bool = False
    show_trial_rate: bool = False
    show_raster: bool = False  # raster compact (une ligne, style montage)
    show_raster_trials: bool = False  # raster / essai (une ligne par essai)
    # Résumés : aperçu canal (tous modes).
    show_summary_rms: bool = False  # mean RMS par enregistrement (profil)
    show_summary_rms_table: bool = False  # mean RMS par canal (table)

    def trigger_index(self) -> int | None:
        """``None`` means average across trials; otherwise one stimulation index."""
        if self.mode == "stimulation":
            return max(0, int(self.stim_index))
        return None

    def selected_curve_panels(self) -> tuple[str, ...]:
        """Traces / ISI / spike scope — tous modes d’aperçu."""
        panels: list[str] = []
        if self.show_raw:
            panels.append("analysis_raw")
        if self.show_hp:
            panels.append("analysis_hp")
        if self.show_lp:
            panels.append("analysis_lp")
        if self.show_rms:
            panels.append("analysis_rms")
        if self.show_isi:
            panels.append("analysis_isi")
        if self.show_overlay:
            panels.append("analysis_overlay")
        return tuple(panels)


    def selected_spike_panels(self) -> tuple[str, ...]:
        """PSTH / FR par essai / rasters — tous modes d’aperçu."""
        panels: list[str] = []
        if self.show_psth:
            panels.append("analysis_psth")
        if self.show_trial_rate:
            panels.append("analysis_trial_rate")
        if self.show_raster:
            panels.append("analysis_raster_channel")
        if self.show_raster_trials:
            panels.append("analysis_raster")
        return tuple(panels)

    def selected_summary_panels(self) -> tuple[str, ...]:
        """Résumés RMS — aperçu canal uniquement."""
        panels: list[str] = []
        if self.show_summary_rms:
            panels.append("summary_rms")
        if self.show_summary_rms_table:
            panels.append("summary_rms_table")
        return tuple(panels)

    def selected_analysis_panels(self) -> tuple[str, ...]:
        """Courbes + spikes (sans résumés)."""
        return self.selected_curve_panels() + self.selected_spike_panels()

    def selected_continuous_extra_panels(self) -> tuple[str, ...]:
        """RMS / ISI / spike scope / spikes — sous les panneaux ``full_recording``.

        WIDE/HIGH/LOW = un ``full_recording`` par flux (pas de doublon
        ``analysis_raw`` / ``analysis_hp`` / ``analysis_lp``).
        """
        stream_panels = {"analysis_raw", "analysis_hp", "analysis_lp"}
        return tuple(
            key
            for key in self.selected_analysis_panels()
            if key not in stream_panels
        )

    def selected_global_panels(self) -> tuple[str, ...]:
        """Panneaux additionnels d’aperçu : résumés seulement."""
        return self.selected_summary_panels()

    def describe(self) -> str:
        if self.mode == "stimulation":
            return f"stimulation n°{int(self.stim_index) + 1}"
        return "moyenne d’essais"


@dataclass(frozen=True)
class TextFace:
    """Gras / italique / souligné pour un rôle typographique (titre, légende, …)."""

    bold: bool = False
    italic: bool = False
    underline: bool = False


@dataclass(frozen=True)
class LegendSettings:
    """Legend appearance, applied to every panel at draw time."""

    visible: bool = True
    location: LegendLocation = "below"
    font_size: float = 9.0
    face: TextFace = field(default_factory=TextFace)
    columns: int = 1
    # Distance légende ↔ graphique (fraction de la hauteur des axes si « below »,
    # sinon padding borderaxespad matplotlib).
    gap: float = 0.22
    frame: bool = True
    show_filter_details: bool = True
    show_sample_counts: bool = True
    show_reference_markers: bool = True


@dataclass(frozen=True)
class PanelStyle:
    """Typography, line weights, and axis behaviour shared by all panels."""

    title_font_size: float = 10.0
    label_font_size: float = 9.0
    tick_font_size: float = 8.0
    title_face: TextFace = field(default_factory=TextFace)
    label_face: TextFace = field(default_factory=TextFace)
    tick_face: TextFace = field(default_factory=TextFace)
    # Décalage du titre de graphique (px) : +dx droite, +dy haut.
    title_dx_px: float = 0.0
    title_dy_px: float = 0.0
    line_width: float = 1.2
    # Épaisseur (points) des pointillés de début / fin de stimulation.
    stim_linewidth: float = 1.0
    # Couleur des pointillés de début (onset) et de fin (offset) de stimulation.
    stim_onset_color: str = STIM_ONSET_COLOR
    stim_offset_color: str = STIM_OFFSET_COLOR
    # Libellés « on » / « off » au-dessus des traits de stimulation.
    show_stim_labels: bool = True
    stim_on_font_size: float = 8.0
    stim_off_font_size: float = 8.0
    stim_on_face: TextFace = field(default_factory=TextFace)
    stim_off_face: TextFace = field(default_factory=TextFace)
    # Décalage des libellés (px) : +dx droite, +dy haut.
    stim_on_dx_px: float = 0.0
    stim_on_dy_px: float = 0.0
    stim_off_dx_px: float = 0.0
    stim_off_dy_px: float = 0.0
    # Libellés « Zone N » sous les plages de zoom (bas du graphique).
    show_zone_labels: bool = True
    zone_font_size: float = 8.0
    zone_face: TextFace = field(default_factory=TextFace)
    zone_dx_px: float = 0.0
    zone_dy_px: float = 0.0
    grid: bool = True
    grid_alpha: float = 0.3
    # Cadre (spines) autour de la zone de tracé.
    show_borders: bool = True
    # Graduations (ticks) orientées vers l’intérieur du cadre.
    ticks_inside: bool = False
    # Barres d’échelle flottantes (temps / amplitude) à la place des graduations
    # numériques sur les bords — style montage EEG / traces.
    show_scale_bars: bool = False
    # Longueurs manuelles (sinon auto ~18 % du span visible). Par défaut la
    # taille pixel des bâtons suit le zoom (valeur fixe en unités de données).
    scale_bar_time_manual: bool = False
    scale_bar_time_s: float = 0.1
    scale_bar_amplitude_manual: bool = False
    scale_bar_amplitude: float = 100.0
    # Longueur pixel du bâton d’amplitude. Avec amplitude manuelle, fixe le
    # gain Y : ``span = amplitude × hauteur_plot / taille_px`` (ex. 50 µV = 80 px).
    scale_bar_amplitude_length_manual: bool = False
    scale_bar_amplitude_length_px: float = 40.0
    scale_bar_time_length_manual: bool = False
    scale_bar_time_length_px: float = 40.0
    # Épaisseur (points) du trait noir des barres d’échelle flottantes.
    scale_bar_linewidth: float = 1.4
    # Décalage de l’échelle amplitude / axe vertical (px). +dx droite, +dy haut.
    scale_bar_dx_px: float = 0.0
    scale_bar_dy_px: float = 0.0
    # Décalage de l’échelle temps / axe horizontal (px). None = même décalage
    # que l’amplitude (compatibilité avec les réglages qui n’avaient qu’une paire).
    scale_bar_time_dx_px: float | None = None
    scale_bar_time_dy_px: float | None = None
    # Échelle propre du spike scope (ms / µV). Indépendante du gain des traces.
    # La case « Manuel » lie la valeur et la taille en pixels, comme les traces.
    spike_scale_amplitude_manual: bool = False
    spike_scale_amplitude: float = 50.0
    spike_scale_amplitude_length_px: float = 40.0
    spike_scale_time_manual: bool = False
    spike_scale_time_ms: float = 1.0
    spike_scale_time_length_px: float = 40.0
    # Max points drawn per curve (min/max envelope decimation above this).
    # UI range is 500…200000 ; défaut = milieu de la plage.
    max_points_per_curve: int = 100_250
    tight_layout: bool = True


@dataclass(frozen=True)
class TextOverrides:
    """Remplacements manuels des textes de graphique (vide = automatique)."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    # Une entrée par courbe, dans l’ordre de la légende. Entrée vide = garder l’auto.
    legend_labels: tuple[str, ...] = ()
    # Si False, masque les suffixes du type « — trial-averaged », « — RMS », …
    show_series_suffix: bool = True
    # Libellés au-dessus des traits de début / fin de stimulation (vides = masquer).
    stim_on: str = "on"
    stim_off: str = "off"

    def resolved_legend_labels(self) -> tuple[str, ...]:
        return tuple(str(label) for label in self.legend_labels)

    def resolved_stim_on(self) -> str:
        return str(self.stim_on)

    def resolved_stim_off(self) -> str:
        return str(self.stim_off)


@dataclass(frozen=True)
class AxisLimits:
    """Optional manual axis bounds (X or Y) for panels that use them."""

    enabled: bool = False
    minimum: float = -200.0
    maximum: float = 200.0

    def as_tuple(self) -> tuple[float, float] | None:
        if not self.enabled:
            return None
        if self.maximum <= self.minimum:
            return None
        return (float(self.minimum), float(self.maximum))


@dataclass(frozen=True)
class ViewerSettings:
    """Display-only parameters: changing these never invalidates cached data."""

    zoom_onset_t0_s: float = -0.1
    zoom_onset_t1_s: float = 0.4
    zoom_end_t0_s: float = -0.1
    zoom_end_t1_s: float = 0.4
    psth_bin_window_s: float = 0.050
    sampling_percent: int = 100
    spike_overlay_pre_ms: float = 2.0
    spike_overlay_post_ms: float = 4.0
    # Opacité des formes d’onde individuelles du spike scope (0–1). La moyenne reste opaque.
    spike_overlay_alpha: float = 0.4
    # Ligne horizontale du seuil de détection sur les courbes en µV et le spike scope.
    show_spike_threshold: bool = False
    # Échelle X manuelle (temps, s) — graphs *sans* zoom placement uniquement.
    # Un ``PanelPlacement`` avec custom zoom ignore toujours ``x_limits``.
    x_limits: AxisLimits = field(
        default_factory=lambda: AxisLimits(enabled=False, minimum=-0.1, maximum=0.4)
    )
    # Échelles Y par type de courbe (réglables dans Canal → section cochée).
    raw_ylim: AxisLimits = field(default_factory=AxisLimits)
    hp_ylim: AxisLimits = field(default_factory=AxisLimits)
    lp_ylim: AxisLimits = field(default_factory=AxisLimits)
    # Désactivé par défaut : autoscale sur les données (évite de tronquer > 20 µV).
    rms_ylim: AxisLimits = field(
        default_factory=lambda: AxisLimits(enabled=False, minimum=0.0, maximum=20.0)
    )
    legend: LegendSettings = field(default_factory=LegendSettings)
    style: PanelStyle = field(default_factory=PanelStyle)
    texts: TextOverrides = field(default_factory=TextOverrides)
    # Montage panels: how many channels to stack in one figure.
    # Kept modest — the default UX is channel-first, not a global montage.
    montage_channels: int = 12
    montage_page: int = 0
    # Revue montage continue (GUI) : canaux par page + page courante.
    montage_review_channels: int = 10
    montage_review_page: int = 0
    # Noms de canaux exclus du traitement et du montage (cases décochées).
    hidden_channels: tuple[str, ...] = ()
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    # Contenu de l’aperçu canal : continuous | moyenne | une stimulation.
    preview_content: PreviewContentMode = "continuous"
    # Continuous recording view: primary stream (compat) + multi-stream montage.
    continuous_stream: AnalysisStream = "raw"
    continuous_streams: tuple[AnalysisStream, ...] = ("raw",)
    continuous_mark_stims: bool = True
    # Origine temporelle des traces continues (immédiat, sans retraitement).
    time_sync: TimeSyncMode = "recording_start"
    # Polarité / seuil ANALOG-IN-0 (affichage + alimente le traitement F5).
    trigger_polarity: TriggerPolarity = "low"
    trigger_threshold: float = 1.0
    # Hauteur fixe de chaque graphique (px) — aperçu, analyse, onglets.
    # Pas d’ajustement automatique au viewport.
    graph_height_px: int = 400
    # Écart vertical titre ↔ courbe en aperçu canal (px).
    # 0 = bandeau pyqtgraph (30 px, texte centré). Positif agrandit l’espace
    # sous le titre, négatif le réduit.
    preview_title_gap_px: int = 0
    # Hauteur minimale d’une ligne canal×flux dans le montage continu (px).
    # Plus bas = moins de pixels Agg (ouverture plus rapide).
    montage_row_min_height_px: int = 52
    # Titres d’axes du montage : « A-002 LOW » si True, « A-002 » si False.
    montage_show_kind_label: bool = True
    # Revue montage : une seule échelle Y (µV) pour toutes les courbes WIDE / HIGH / LOW
    # de la page. False : chaque ligne suit l’amplitude de son canal.
    montage_shared_ylim: bool = False
    # Décalage des titres de canal à côté des courbes (px).
    # +x rapproche le titre de la courbe, +y le monte.
    montage_label_dx_px: float = 0.0
    montage_label_dy_px: float = 0.0
    # Barres de plage (zoom / traitement). Vide = aucune plage (défaut).
    range_bars: tuple[TimeRangeBar, ...] = ()
    active_range_index: int = 0

    def trigger_edge(self) -> str:
        """Edge de détection correspondant à la polarité d’affichage."""
        return TRIGGER_POLARITY_TO_EDGE.get(self.trigger_polarity, "falling")

    def resolved_continuous_streams(self) -> tuple[AnalysisStream, ...]:
        streams = tuple(s for s in self.continuous_streams if s in {"raw", "hp", "lp"})
        if streams:
            return streams  # type: ignore[return-value]
        stream = self.continuous_stream if self.continuous_stream in {"raw", "hp", "lp"} else "raw"
        return (stream,)  # type: ignore[return-value]

    def active_range_bar(self) -> TimeRangeBar | None:
        bars = self.range_bars
        if not bars:
            return None
        index = int(self.active_range_index)
        if index < 0:
            return None
        index = max(0, min(len(bars) - 1, index))
        return bars[index]

    def zoom_onset_window(self) -> tuple[float, float]:
        return (float(self.zoom_onset_t0_s), float(self.zoom_onset_t1_s))

    def zoom_end_window(self) -> tuple[float, float]:
        return (float(self.zoom_end_t0_s), float(self.zoom_end_t1_s))

    def ylim_for_stream(self, stream: str) -> AxisLimits:
        """Échelle Y manuelle pour un flux WIDE / HIGH / LOW."""
        key = str(stream).strip().lower()
        if key in {"hp", "high"}:
            return self.hp_ylim
        if key in {"lp", "low"}:
            return self.lp_ylim
        return self.raw_ylim

    def validate(self) -> list[str]:
        """Liste de problèmes lisibles (vide si tout est valide)."""
        problems: list[str] = []
        if self.zoom_onset_t1_s <= self.zoom_onset_t0_s:
            problems.append("Zoom début : la fin doit être strictement après le début.")
        if self.zoom_end_t1_s <= self.zoom_end_t0_s:
            problems.append("Zoom fin : la fin doit être strictement après le début.")
        if self.psth_bin_window_s <= 0:
            problems.append("La fenêtre PSTH doit être > 0 s.")
        if not (1 <= int(self.sampling_percent) <= 100):
            problems.append("L’échantillonnage d’affichage des spikes doit être entre 1 et 100 %.")
        if self.spike_overlay_pre_ms < 0:
            problems.append("Superposition : le temps avant détection doit être ≥ 0 ms.")
        if self.spike_overlay_post_ms <= 0:
            problems.append("Superposition : le temps après détection doit être > 0 ms.")
        if not (0.0 <= float(self.spike_overlay_alpha) <= 1.0):
            problems.append("Spike scope : l’opacité des traces doit être entre 0 et 1.")
        if self.x_limits.enabled and self.x_limits.as_tuple() is None:
            problems.append("Axe X : le maximum doit être > minimum.")
        for label, limits in (
            ("WIDE", self.raw_ylim),
            ("HIGH", self.hp_ylim),
            ("LOW", self.lp_ylim),
            ("RMS", self.rms_ylim),
        ):
            if limits.enabled and limits.as_tuple() is None:
                problems.append(f"Axe Y {label} : le maximum doit être > minimum.")
        if self.analysis.stim_index < 0:
            problems.append("L’indice de stimulation doit être ≥ 0.")
        return problems


def copy_scale_bar_style(target: PanelStyle, source: PanelStyle) -> PanelStyle:
    """Recopier uniquement les champs d’échelle flottante de ``source`` vers ``target``."""
    amp_dx, amp_dy = resolved_scale_bar_offset_px(source)
    time_dx, time_dy = resolved_scale_bar_time_offset_px(source)
    return replace(
        target,
        show_scale_bars=bool(source.show_scale_bars),
        scale_bar_time_manual=bool(source.scale_bar_time_manual),
        scale_bar_time_s=float(source.scale_bar_time_s),
        scale_bar_amplitude_manual=bool(source.scale_bar_amplitude_manual),
        scale_bar_amplitude=float(source.scale_bar_amplitude),
        scale_bar_amplitude_length_manual=bool(
            getattr(source, "scale_bar_amplitude_length_manual", False)
        ),
        scale_bar_amplitude_length_px=float(
            getattr(source, "scale_bar_amplitude_length_px", 40.0) or 40.0
        ),
        scale_bar_time_length_manual=bool(
            getattr(source, "scale_bar_time_length_manual", False)
        ),
        scale_bar_time_length_px=float(
            getattr(source, "scale_bar_time_length_px", 40.0) or 40.0
        ),
        scale_bar_linewidth=resolved_scale_bar_linewidth(source),
        scale_bar_dx_px=amp_dx,
        scale_bar_dy_px=amp_dy,
        scale_bar_time_dx_px=time_dx,
        scale_bar_time_dy_px=time_dy,
        spike_scale_amplitude_manual=bool(
            getattr(source, "spike_scale_amplitude_manual", False)
        ),
        spike_scale_amplitude=float(
            getattr(source, "spike_scale_amplitude", 50.0) or 50.0
        ),
        spike_scale_amplitude_length_px=float(
            getattr(source, "spike_scale_amplitude_length_px", 40.0) or 40.0
        ),
        spike_scale_time_manual=bool(getattr(source, "spike_scale_time_manual", False)),
        spike_scale_time_ms=float(getattr(source, "spike_scale_time_ms", 1.0) or 1.0),
        spike_scale_time_length_px=float(
            getattr(source, "spike_scale_time_length_px", 40.0) or 40.0
        ),
    )


def scale_bar_ref_height_px(settings: object, *, montage: bool = False) -> float:
    """Hauteur de plot de référence (px) pour le gain lié à l’échelle."""
    if settings is None:
        return 0.0
    if montage:
        return float(
            max(36, int(getattr(settings, "montage_row_min_height_px", 36) or 36))
        )
    return float(max(1, int(getattr(settings, "graph_height_px", 400) or 400)))


def scale_bar_drives_widget_height(settings: object) -> bool:
    """True si amplitude + taille px manuelles : le widget doit suivre le gain."""
    if settings is None:
        return False
    style = getattr(settings, "style", None)
    if style is None:
        style = settings
    if not bool(getattr(style, "show_scale_bars", False)):
        return False
    if not bool(getattr(style, "scale_bar_amplitude_manual", False)):
        return False
    if not bool(getattr(style, "scale_bar_amplitude_length_manual", False)):
        return False
    try:
        amp = float(getattr(style, "scale_bar_amplitude", 0.0) or 0.0)
        length = float(getattr(style, "scale_bar_amplitude_length_px", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return bool(amp > 0.0 and length > 0.0 and amp == amp and length == length)


def scale_bar_view_height_for_span(
    data_span: float,
    settings: object,
) -> int | None:
    """Hauteur de zone de tracé (px) pour que ``amplitude`` occupe ``amp_length_px``.

    ``height = data_span × amp_length_px / amplitude``.
    Le ylim reste celui des données (autoscale) ; le widget grandit pour tout voir.
    """
    if not scale_bar_drives_widget_height(settings):
        return None
    style = getattr(settings, "style", None)
    if style is None:
        style = settings
    try:
        amp = float(getattr(style, "scale_bar_amplitude", 0.0) or 0.0)
        length = float(getattr(style, "scale_bar_amplitude_length_px", 0.0) or 0.0)
        span = float(data_span)
    except (TypeError, ValueError):
        return None
    if not (amp > 0.0 and length > 0.0 and span > 0.0):
        return None
    if amp != amp or length != length or span != span:
        return None
    height = span * (length / amp)
    if not (height > 0.0) or height != height:
        return None
    return max(40, int(round(height)))


def linked_scale_bar_y_limits(
    settings: object,
    *,
    height_px: float | None = None,
    montage: bool = False,
) -> tuple[float, float] | None:
    """Ylim symétrique si on fixe la hauteur (export / fallback).

    Préférer ``scale_bar_view_height_for_span`` à l’écran : le widget s’adapte
    et le ylim suit les données (pas de rognage).
    """
    if not scale_bar_drives_widget_height(settings):
        return None
    style = getattr(settings, "style", None)
    if style is None:
        style = settings
    try:
        amp = float(getattr(style, "scale_bar_amplitude", 0.0) or 0.0)
        length = float(getattr(style, "scale_bar_amplitude_length_px", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if height_px is None:
        height_px = scale_bar_ref_height_px(settings, montage=montage)
    try:
        height = float(height_px)
    except (TypeError, ValueError):
        return None
    if not (amp > 0.0 and length > 0.0 and height > 0.0):
        return None
    if amp != amp or length != length or height != height:
        return None
    half = 0.5 * amp * (height / length)
    if not (half > 0.0) or half != half:
        return None
    return (-half, half)


def copy_stim_marker_style(target: PanelStyle, source: PanelStyle) -> PanelStyle:
    """Recopier épaisseur / couleurs / libellés des pointillés de stimulation."""
    return replace(
        target,
        stim_linewidth=resolved_stim_linewidth(source),
        stim_onset_color=resolved_stim_onset_color(source),
        stim_offset_color=resolved_stim_offset_color(source),
        show_stim_labels=bool(getattr(source, "show_stim_labels", True)),
        stim_on_font_size=float(getattr(source, "stim_on_font_size", 8.0) or 8.0),
        stim_off_font_size=float(getattr(source, "stim_off_font_size", 8.0) or 8.0),
        stim_on_face=getattr(source, "stim_on_face", TextFace()) or TextFace(),
        stim_off_face=getattr(source, "stim_off_face", TextFace()) or TextFace(),
        stim_on_dx_px=_scale_bar_offset_one(source, "stim_on_dx_px"),
        stim_on_dy_px=_scale_bar_offset_one(source, "stim_on_dy_px"),
        stim_off_dx_px=_scale_bar_offset_one(source, "stim_off_dx_px"),
        stim_off_dy_px=_scale_bar_offset_one(source, "stim_off_dy_px"),
        show_zone_labels=bool(getattr(source, "show_zone_labels", True)),
        zone_font_size=float(getattr(source, "zone_font_size", 8.0) or 8.0),
        zone_face=getattr(source, "zone_face", TextFace()) or TextFace(),
        zone_dx_px=_scale_bar_offset_one(source, "zone_dx_px"),
        zone_dy_px=_scale_bar_offset_one(source, "zone_dy_px"),
    )


def copy_global_display_style(target: PanelStyle, source: PanelStyle) -> PanelStyle:
    """Réglages Affichage globaux : échelle flottante + marqueurs de stim."""
    return copy_stim_marker_style(copy_scale_bar_style(target, source), source)


def _scale_bar_offset_one(style: object, name: str, *, default: float = 0.0) -> float:
    raw = getattr(style, name, default)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if value != value:
        return float(default)
    return value


def resolved_scale_bar_offset_px(style: object) -> tuple[float, float]:
    """Décalage (dx, dy) de l’échelle amplitude (axe vertical), en pixels.

    ``dx`` positif vers la droite, ``dy`` positif vers le haut.
    """
    return (
        _scale_bar_offset_one(style, "scale_bar_dx_px"),
        _scale_bar_offset_one(style, "scale_bar_dy_px"),
    )


def resolved_scale_bar_time_offset_px(style: object) -> tuple[float, float]:
    """Décalage (dx, dy) de l’échelle temps (axe horizontal), en pixels.

    Si ``scale_bar_time_dx_px`` / ``scale_bar_time_dy_px`` sont ``None``,
    reprend le décalage amplitude (anciens réglages à une seule paire).
    """
    amp_dx, amp_dy = resolved_scale_bar_offset_px(style)
    raw_dx = getattr(style, "scale_bar_time_dx_px", None)
    raw_dy = getattr(style, "scale_bar_time_dy_px", None)
    dx = amp_dx if raw_dx is None else _scale_bar_offset_one(style, "scale_bar_time_dx_px")
    dy = amp_dy if raw_dy is None else _scale_bar_offset_one(style, "scale_bar_time_dy_px")
    return (dx, dy)


def resolved_scale_bar_linewidth(style: object) -> float:
    """Épaisseur (points) du trait noir des barres d’échelle flottantes."""
    raw = getattr(style, "scale_bar_linewidth", 1.4)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.4
    if value != value or value <= 0.0:
        return 1.4
    return value


def resolved_stim_linewidth(style: object) -> float:
    """Épaisseur (points) des pointillés de début / fin de stimulation."""
    raw = getattr(style, "stim_linewidth", 1.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if value != value or value <= 0.0:
        return 1.0
    return value


def _resolved_hex_color(raw: object, fallback: str) -> str:
    """``#rgb`` / ``#rrggbb`` valide, sinon la couleur par défaut."""
    text = str(raw or "").strip()
    if len(text) == 4 and text.startswith("#"):
        try:
            int(text[1:], 16)
        except ValueError:
            return fallback
        red, green, blue = text[1], text[2], text[3]
        return f"#{red}{red}{green}{green}{blue}{blue}".lower()
    if len(text) == 7 and text.startswith("#"):
        try:
            int(text[1:], 16)
        except ValueError:
            return fallback
        return text.lower()
    return fallback


def resolved_stim_onset_color(style: object) -> str:
    """Couleur des pointillés de début de stimulation."""
    return _resolved_hex_color(
        getattr(style, "stim_onset_color", None), STIM_ONSET_COLOR
    )


def resolved_stim_offset_color(style: object) -> str:
    """Couleur des pointillés de fin de stimulation."""
    return _resolved_hex_color(
        getattr(style, "stim_offset_color", None), STIM_OFFSET_COLOR
    )


def default_zoom_range_color(index: int) -> str:
    """Couleur par défaut de la n-ième plage / zone de zoom."""
    palette = ZOOM_RANGE_COLORS or ("#16a34a",)
    return palette[int(index) % len(palette)]


def default_zone_label(index: int) -> str:
    """Libellé par défaut de la n-ième plage : « Zone » + numéro (1-based)."""
    return f"Zone {int(index) + 1}"


def resolved_zone_label(bar: object, index: int = 0) -> str:
    """Texte affiché sous une zone de zoom (nom éditable, sinon Zone N)."""
    raw = str(getattr(bar, "label", "") or "").strip()
    return raw or default_zone_label(index)


def resolved_range_bar_color(bar: object, index: int = 0) -> str:
    """Couleur d’une plage : champ ``color`` valide, sinon palette par index."""
    fallback = default_zoom_range_color(index)
    return _resolved_hex_color(getattr(bar, "color", None), fallback)


def apply_local_display_settings(
    base: ViewerSettings,
    local: ViewerSettings,
    *,
    include_analysis: bool = False,
    inherit_scale_bars: bool = False,
) -> ViewerSettings:
    """Appliquer les réglages d’affichage d’*une* fenêtre (ou d’un panneau) sur une base.

    Chaque fenêtre avec panneau local (détachée, Analyse, …) ou chaque graphique
    avec overrides dédiés contrôle légende, style, axes et sync pour *ses*
    graphs. Sans ceci, un redraw réinjecte les axes / la sync de la fenêtre
    parente et ignore le panneau local.

    ``inherit_scale_bars=True`` : échelle flottante, pointillés de stim, légende
    et hauteur de graph viennent de ``base`` (dock Affichage), pas du snapshot
    local figé — évite qu’une édition de texte / un override partiel bloque
    ces réglages globaux sur certains graphs.
    """
    style = local.style
    legend = local.legend
    graph_height_px = int(local.graph_height_px)
    if inherit_scale_bars:
        style = copy_global_display_style(style, base.style)
        legend = base.legend
        graph_height_px = int(base.graph_height_px)
    kwargs: dict[str, object] = {
        "legend": legend,
        "style": style,
        "texts": local.texts,
        "psth_bin_window_s": local.psth_bin_window_s,
        "sampling_percent": local.sampling_percent,
        "x_limits": local.x_limits,
        "raw_ylim": local.raw_ylim,
        "hp_ylim": local.hp_ylim,
        "lp_ylim": local.lp_ylim,
        "rms_ylim": local.rms_ylim,
        "time_sync": local.time_sync,
        "trigger_polarity": local.trigger_polarity,
        "trigger_threshold": local.trigger_threshold,
        "graph_height_px": graph_height_px,
    }
    if include_analysis:
        kwargs["analysis"] = local.analysis
        kwargs["continuous_streams"] = local.continuous_streams
        kwargs["continuous_stream"] = local.continuous_stream
        kwargs["continuous_mark_stims"] = local.continuous_mark_stims
    return replace(base, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PanelPlacement:
    """Une instance de panneau (éventuellement avec une fenêtre de zoom personnalisée)."""

    panel: str
    section: SectionKey = "full"
    # Si définis, la fenêtre X est [zoom_t0_s, zoom_t1_s].
    # Par défaut relative à la stimulation ; ``zoom_absolute=True`` = temps absolu
    # (barres sur l’enregistrement continu). Indépendant de ``section``.
    zoom_t0_s: float | None = None
    zoom_t1_s: float | None = None
    zoom_label: str = ""
    zoom_absolute: bool = False
    # Identifiant d’instance (ex. bar_id d’une plage) — clé stable, pas affiché.
    instance_id: str = ""
    # Flux unique pour ``full_recording`` (raw / hp / lp) — un panneau par courbe,
    # comme analysis_raw / analysis_hp / analysis_lp en moyenne / stimulation.
    stream: str = ""

    @property
    def has_custom_zoom(self) -> bool:
        return self.zoom_t0_s is not None and self.zoom_t1_s is not None

    @property
    def key(self) -> str:
        stream_part = f"/{self.stream}" if self.stream else ""
        suffix = f"#{self.instance_id}" if self.instance_id else ""
        if self.has_custom_zoom:
            label = self.zoom_label.strip() or f"{self.zoom_t0_s}:{self.zoom_t1_s}"
            return f"{self.panel}{stream_part}@custom:{label}{suffix}"
        if is_section_independent(self.panel):
            base = f"{self.panel}{stream_part}"
            return f"{base}{suffix}" if suffix else base
        return f"{self.panel}{stream_part}@{self.section}{suffix}"

    def title(self) -> str:
        label = panel_label(self.panel)
        if self.stream:
            short = STREAM_SHORT_LABELS.get(self.stream, self.stream.upper())
            label = f"{label} — {short}"
        if self.has_custom_zoom:
            name = self.zoom_label.strip() or "zoom"
            unit = "s abs." if self.zoom_absolute else "s rel. stim"
            return f"{label} — {name} [{self.zoom_t0_s:g} … {self.zoom_t1_s:g} {unit}]"
        if is_section_independent(self.panel):
            return label
        return f"{label} — {SECTION_LABELS.get(self.section, self.section)}"

    def with_custom_zoom(
        self,
        t0_s: float,
        t1_s: float,
        *,
        label: str = "",
        instance_id: str = "",
        absolute: bool = False,
    ) -> PanelPlacement:
        return replace(
            self,
            section="full",
            zoom_t0_s=float(t0_s),
            zoom_t1_s=float(t1_s),
            zoom_label=str(label or ""),
            zoom_absolute=bool(absolute),
            instance_id=str(instance_id or self.instance_id or ""),
        )


@dataclass(frozen=True)
class ViewTab:
    """A named, independently configurable grid of panels."""

    name: str
    panels: tuple[PanelPlacement, ...] = ()
    columns: int = 2
    panel_height_px: int = 400

    def with_panels(self, panels: Sequence[PanelPlacement]) -> ViewTab:
        return replace(self, panels=tuple(panels))


def _placements(section: SectionKey, panels: Sequence[str]) -> tuple[PanelPlacement, ...]:
    return tuple(PanelPlacement(panel=p, section=section) for p in panels)


def default_workspace_tabs() -> tuple[ViewTab, ...]:
    """Onglets de démarrage (vides) : l’aperçu canal est embarqué hors onglets.

    Le montage multi-canaux reste disponible via *Revue montage*.
    Les barres de plage vivent dans l’aperçu central ; les graphs
    d’analyse s’ouvrent via le bouton Analyse.
    """
    return ()


@dataclass(frozen=True)
class WorkspaceLayout:
    """All view tabs plus the index of the active one."""

    tabs: tuple[ViewTab, ...] = field(default_factory=default_workspace_tabs)
    active_index: int = 0

    def with_tabs(self, tabs: Sequence[ViewTab]) -> WorkspaceLayout:
        return replace(self, tabs=tuple(tabs))

    def required_sections(self) -> set[str]:
        """Sections referenced by at least one visible panel."""
        out: set[str] = set()
        for tab in self.tabs:
            for placement in tab.panels:
                if not is_global_panel(placement.panel):
                    out.add(placement.section)
        return out

    def needs_spikes(self) -> bool:
        return any(
            panel_needs_spikes(placement.panel)
            for tab in self.tabs
            for placement in tab.panels
        )

    def product_needs(self) -> tuple[bool, bool, bool, bool]:
        """``(means, rms, spikes, overlay)`` required by any laid-out panel."""
        needs = [
            panel_product_needs(placement.panel)
            for tab in self.tabs
            for placement in tab.panels
        ]
        if not needs:
            return False, False, False, False
        return merge_product_needs(*needs)

    def needs_all_channels(self) -> bool:
        """True when at least one panel reads every channel (montage / résumé)."""
        for tab in self.tabs:
            for placement in tab.panels:
                panel = str(placement.panel or "")
                if panel.startswith("montage_") and panel != "montage_continuous_raw":
                    return True
                if panel in {"summary_rms", "summary_rms_table"}:
                    return True
                means, rms, spikes, overlay = panel_product_needs(panel)
                if is_global_panel(panel) and (means or rms or spikes or overlay):
                    return True
        return False

    def to_plot_display(self, *, base: PlotDisplaySettings | None = None) -> PlotDisplaySettings:
        """Derive classic PDF settings from the panels currently laid out.

        When the workspace has no panels (channel-preview-only GUI), fall back to
        a full PDF layout so export is still useful.
        """
        has_any_panel = any(tab.panels for tab in self.tabs)
        if not has_any_panel:
            return base or PlotDisplaySettings.all_on()
        per_section: dict[str, set[str]] = {key: set() for key in SECTION_KEYS}
        globals_seen: set[str] = set()
        extras: set[str] = set()
        for tab in self.tabs:
            for placement in tab.panels:
                if is_global_panel(placement.panel):
                    globals_seen.add(placement.panel)
                elif placement.panel in EXTRA_CHANNEL_PANEL_FIELD_NAMES:
                    extras.add(placement.panel)
                else:
                    per_section.setdefault(placement.section, set()).add(placement.panel)
        reference = base or PlotDisplaySettings.all_on()
        sections = {
            key: SectionPanels(
                **{name: (name in per_section.get(key, set())) for name in PANEL_FIELD_NAMES}
            )
            for key in SECTION_KEYS
        }
        # Le bundle PDF « 2e stim / 2e→3e » ne se déclenche pas pour la revue
        # continue seule : elle n’a pas les mêmes pages.
        stim_montages = {
            "montage_mean_raw",
            "montage_mean_hp",
            "montage_mean_lp",
            "montage_second_raw",
            "montage_second_hp",
            "montage_second_lp",
            "montage_second_to_third_lp",
        }
        montage = bool(globals_seen & stim_montages)
        return PlotDisplaySettings(
            mea_layout="mea_layout" in extras or reference.mea_layout,
            impedance="impedance" in extras or reference.impedance,
            summary_rms_page="summary_rms" in globals_seen,
            summary_rms_table_page="summary_rms_table" in globals_seen,
            summary_impedance_page="summary_impedance" in globals_seen,
            summary_second_stim_montage_page=montage,
            full_view=sections["full"],
            zoom_onset=sections["zoom_onset"],
            zoom_trigger_end=sections["zoom_trigger_end"],
        )

    def zoom_mode(self) -> ZoomMode:
        sections = self.required_sections()
        onset = "zoom_onset" in sections
        end = "zoom_trigger_end" in sections
        if onset and end:
            return "both"
        if onset:
            return "onset"
        if end:
            return "trigger_end"
        return "none"
