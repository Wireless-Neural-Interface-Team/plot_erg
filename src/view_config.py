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
    PANEL_FIELD_NAMES,
    PANEL_LABELS,
    PlotDisplaySettings,
    SectionPanels,
    ZoomMode,
)

SectionKey = Literal["full", "zoom_onset", "zoom_trigger_end"]

SECTION_KEYS: tuple[SectionKey, ...] = ("full", "zoom_onset", "zoom_trigger_end")

SECTION_LABELS: dict[str, str] = {
    "full": "Vue complète",
    "zoom_onset": "Zoom début",
    "zoom_trigger_end": "Zoom fin",
}

SECTION_SHORT_LABELS: dict[str, str] = {
    "full": "complet",
    "zoom_onset": "début",
    "zoom_trigger_end": "fin",
}

# Panels that do not depend on the selected channel.
GLOBAL_PANEL_FIELD_NAMES: tuple[str, ...] = (
    "montage_continuous_raw",
    "summary_rms",
    "summary_rms_table",
    "summary_impedance",
    "montage_mean_raw",
    "montage_mean_hp",
    "montage_mean_lp",
    "montage_second_raw",
    "montage_second_hp",
    "montage_second_lp",
    "montage_second_to_third_lp",
)

GLOBAL_PANEL_LABELS: dict[str, str] = {
    "montage_continuous_raw": "Montage — tous canaux, brut continu",
    "summary_rms": "Résumé — RMS moyen par canal",
    "summary_rms_table": "Résumé — table RMS moyen par canal",
    "summary_impedance": "Résumé — impédance moyenne |Z| @ 1 kHz",
    "montage_mean_raw": "Montage — tous canaux, moyenne brut",
    "montage_mean_hp": "Montage — tous canaux, moyenne passe-haut",
    "montage_mean_lp": "Montage — tous canaux, moyenne passe-bas",
    "montage_second_raw": "Montage — tous canaux, 2e stim brut",
    "montage_second_hp": "Montage — tous canaux, 2e stim passe-haut",
    "montage_second_lp": "Montage — tous canaux, 2e stim passe-bas",
    "montage_second_to_third_lp": "Montage — tous canaux, 2e→3e stim passe-bas",
}

# Channel-aware panels that are not part of the 21 togglable section panels.
EXTRA_CHANNEL_PANEL_FIELD_NAMES: tuple[str, ...] = ("mea_layout", "impedance")

EXTRA_CHANNEL_PANEL_LABELS: dict[str, str] = {
    "mea_layout": "Carte MEA (canal sélectionné)",
    "impedance": "Impédance |Z| @ 1 kHz (canal sélectionné)",
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
AnalysisStream = Literal["raw", "hp", "lp"]

STREAM_SHORT_LABELS: dict[str, str] = {
    "raw": "WIDE",
    "hp": "HIGH",
    "lp": "LOW",
}


@dataclass(frozen=True)
class TimeRangeBar:
    """Paire de barres temporelles [t0, t1] pour zoom / plage de traitement."""

    t0_s: float
    t1_s: float
    label: str = ""
    bar_id: str = ""

    def ordered(self) -> tuple[float, float]:
        a, b = float(self.t0_s), float(self.t1_s)
        return (a, b) if a <= b else (b, a)

    def with_bounds(self, t0_s: float, t1_s: float) -> TimeRangeBar:
        return replace(self, t0_s=float(t0_s), t1_s=float(t1_s))


# Progressive-analysis panels driven by :class:`AnalysisSettings`.
ANALYSIS_PANEL_FIELD_NAMES: tuple[str, ...] = (
    "full_recording",
    "analysis_raw",
    "analysis_hp",
    "analysis_lp",
    "analysis_rms",
    "analysis_psth",
    "analysis_isi",
    "analysis_raster",
    "analysis_overlay",
)

ANALYSIS_PANEL_LABELS: dict[str, str] = {
    "full_recording": "Enregistrement complet (continu)",
    "analysis_raw": "Analyse — brut",
    "analysis_hp": "Analyse — passe-haut",
    "analysis_lp": "Analyse — passe-bas",
    "analysis_rms": "Analyse — RMS",
    "analysis_psth": "Analyse — PSTH / FR",
    "analysis_isi": "Analyse — ISI",
    "analysis_raster": "Analyse — raster",
    "analysis_overlay": "Analyse — superposition de spikes",
}


def panel_label(field_name: str) -> str:
    """Human-readable label for any panel key (section, extra, or global)."""
    if field_name in PANEL_LABELS:
        return PANEL_LABELS[field_name]
    if field_name in ANALYSIS_PANEL_LABELS:
        return ANALYSIS_PANEL_LABELS[field_name]
    if field_name in EXTRA_CHANNEL_PANEL_LABELS:
        return EXTRA_CHANNEL_PANEL_LABELS[field_name]
    return GLOBAL_PANEL_LABELS.get(field_name, field_name)


def is_global_panel(field_name: str) -> bool:
    return field_name in GLOBAL_PANEL_FIELD_NAMES


def is_section_panel(field_name: str) -> bool:
    return field_name in PANEL_FIELD_NAMES or field_name in {
        key for key in ANALYSIS_PANEL_FIELD_NAMES if key != "full_recording"
    }


def is_section_independent(field_name: str) -> bool:
    """True for panels whose content does not depend on the temporal section."""
    return (
        field_name in GLOBAL_PANEL_FIELD_NAMES
        or field_name in EXTRA_CHANNEL_PANEL_FIELD_NAMES
        or field_name == "full_recording"
    )


def panel_needs_spikes(field_name: str) -> bool:
    return field_name in {
        "psth",
        "first_psth",
        "second_psth",
        "isi",
        "first_isi",
        "second_isi",
        "trial_rate",
        "raster",
        "spike_overlay",
        "analysis_psth",
        "analysis_isi",
        "analysis_raster",
        "analysis_overlay",
    }


@dataclass(frozen=True)
class AnalysisSettings:
    """What the progressive Analysis dock is currently configured to show."""

    mode: AnalysisMode = "average"
    stim_index: int = 0  # 0-based stimulation index when mode == stimulation
    show_raw: bool = True
    show_hp: bool = True
    show_lp: bool = True
    show_rms: bool = True
    show_psth: bool = False
    show_isi: bool = False
    show_raster: bool = False
    show_overlay: bool = False

    def trigger_index(self) -> int | None:
        """``None`` means average across trials; otherwise one stimulation index."""
        if self.mode == "stimulation":
            return max(0, int(self.stim_index))
        return None

    def selected_trace_panels(self) -> tuple[str, ...]:
        panels: list[str] = []
        if self.show_raw:
            panels.append("analysis_raw")
        if self.show_hp:
            panels.append("analysis_hp")
        if self.show_lp:
            panels.append("analysis_lp")
        if self.show_rms:
            panels.append("analysis_rms")
        return tuple(panels)

    def selected_spike_panels(self) -> tuple[str, ...]:
        panels: list[str] = []
        if self.show_raster:
            panels.append("analysis_raster")
        if self.show_psth:
            panels.append("analysis_psth")
        if self.show_isi:
            panels.append("analysis_isi")
        if self.show_overlay:
            panels.append("analysis_overlay")
        return tuple(panels)

    def selected_analysis_panels(self) -> tuple[str, ...]:
        return self.selected_trace_panels() + self.selected_spike_panels()

    def describe(self) -> str:
        if self.mode == "stimulation":
            return f"stimulation n°{int(self.stim_index) + 1}"
        return "moyenne d’essais"


@dataclass(frozen=True)
class LegendSettings:
    """Legend appearance, applied to every panel at draw time."""

    visible: bool = True
    location: LegendLocation = "below"
    font_size: float = 9.0
    columns: int = 1
    frame: bool = True
    show_filter_details: bool = True
    show_sample_counts: bool = True
    show_reference_markers: bool = True

    def with_location(self, location: str) -> LegendSettings:
        loc = location if location in LEGEND_LOCATIONS else "below"
        return replace(self, location=loc)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PanelStyle:
    """Typography, line weights, and axis behaviour shared by all panels."""

    title_font_size: float = 10.0
    label_font_size: float = 9.0
    tick_font_size: float = 8.0
    line_width: float = 1.2
    grid: bool = True
    grid_alpha: float = 0.3
    # Cadre (spines) autour de la zone de tracé.
    show_borders: bool = True
    # Max points drawn per curve (min/max envelope decimation above this).
    max_points_per_curve: int = 6000
    tight_layout: bool = True


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
    # Échelle X manuelle (temps, s) — prioritaire sur la section / le zoom placement.
    x_limits: AxisLimits = field(
        default_factory=lambda: AxisLimits(enabled=False, minimum=-0.1, maximum=0.4)
    )
    stim_hp_ylim: AxisLimits = field(default_factory=AxisLimits)
    rms_ylim: AxisLimits = field(
        default_factory=lambda: AxisLimits(enabled=True, minimum=0.0, maximum=20.0)
    )
    trace_ylim: AxisLimits = field(default_factory=AxisLimits)
    legend: LegendSettings = field(default_factory=LegendSettings)
    style: PanelStyle = field(default_factory=PanelStyle)
    # Montage panels: how many channels to stack in one figure.
    # Kept modest — the default UX is channel-first, not a global montage.
    montage_channels: int = 12
    montage_page: int = 0
    # Noms de canaux exclus du montage (cases décochées dans Session → Channels).
    hidden_channels: tuple[str, ...] = ()
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    # Continuous recording view: primary stream (compat) + multi-stream montage.
    continuous_stream: AnalysisStream = "raw"
    continuous_streams: tuple[AnalysisStream, ...] = ("raw",)
    continuous_mark_stims: bool = True
    # Hauteur minimale d’une ligne canal×flux dans le montage continu (px).
    montage_row_min_height_px: int = 72
    # Barres de plage (zoom / traitement). Vide = extrémités au premier rendu.
    range_bars: tuple[TimeRangeBar, ...] = ()
    active_range_index: int = 0

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
        index = max(0, min(len(bars) - 1, int(self.active_range_index)))
        return bars[index]

    def zoom_onset_window(self) -> tuple[float, float]:
        return (float(self.zoom_onset_t0_s), float(self.zoom_onset_t1_s))

    def zoom_end_window(self) -> tuple[float, float]:
        return (float(self.zoom_end_t0_s), float(self.zoom_end_t1_s))

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
        if self.x_limits.enabled and self.x_limits.as_tuple() is None:
            problems.append("Axe X : le maximum doit être > minimum.")
        if self.stim_hp_ylim.enabled and self.stim_hp_ylim.as_tuple() is None:
            problems.append("Axe Y passe-haut stim : le maximum doit être > minimum.")
        if self.rms_ylim.enabled and self.rms_ylim.as_tuple() is None:
            problems.append("Axe Y RMS : le maximum doit être > minimum.")
        if self.trace_ylim.enabled and self.trace_ylim.as_tuple() is None:
            problems.append("Axe Y traces : le maximum doit être > minimum.")
        if self.analysis.stim_index < 0:
            problems.append("L’indice de stimulation doit être ≥ 0.")
        return problems


@dataclass(frozen=True)
class PanelPlacement:
    """Une instance de panneau (éventuellement avec une fenêtre de zoom personnalisée)."""

    panel: str
    section: SectionKey = "full"
    # Si définis, la fenêtre X est [zoom_t0_s, zoom_t1_s] relative à la stimulation
    # (indépendamment de section). Utilisé pour les zooms ajoutés par l’utilisateur.
    zoom_t0_s: float | None = None
    zoom_t1_s: float | None = None
    zoom_label: str = ""
    # Identifiant d’instance (ex. bar_id d’une plage) — clé stable, pas affiché.
    instance_id: str = ""

    @property
    def has_custom_zoom(self) -> bool:
        return self.zoom_t0_s is not None and self.zoom_t1_s is not None

    @property
    def key(self) -> str:
        suffix = f"#{self.instance_id}" if self.instance_id else ""
        if self.has_custom_zoom:
            label = self.zoom_label.strip() or f"{self.zoom_t0_s}:{self.zoom_t1_s}"
            return f"{self.panel}@custom:{label}{suffix}"
        if is_section_independent(self.panel):
            return f"{self.panel}{suffix}" if suffix else self.panel
        return f"{self.panel}@{self.section}{suffix}"

    def title(self) -> str:
        label = panel_label(self.panel)
        if self.has_custom_zoom:
            name = self.zoom_label.strip() or "zoom"
            return f"{label} — {name} [{self.zoom_t0_s:g} … {self.zoom_t1_s:g} s]"
        if is_section_independent(self.panel):
            return label
        return f"{label} — {SECTION_LABELS.get(self.section, self.section)}"

    def with_custom_zoom(
        self, t0_s: float, t1_s: float, *, label: str = "", instance_id: str = ""
    ) -> PanelPlacement:
        return replace(
            self,
            section="full",
            zoom_t0_s=float(t0_s),
            zoom_t1_s=float(t1_s),
            zoom_label=str(label or ""),
            instance_id=str(instance_id or self.instance_id or ""),
        )


@dataclass(frozen=True)
class ViewTab:
    """A named, independently configurable grid of panels."""

    name: str
    panels: tuple[PanelPlacement, ...] = ()
    columns: int = 2
    panel_height_px: int = 300

    def with_panels(self, panels: Sequence[PanelPlacement]) -> ViewTab:
        return replace(self, panels=tuple(panels))


def _placements(section: SectionKey, panels: Sequence[str]) -> tuple[PanelPlacement, ...]:
    return tuple(PanelPlacement(panel=p, section=section) for p in panels)


def default_workspace_tabs() -> tuple[ViewTab, ...]:
    """Vue de démarrage légère : aperçu du canal sélectionné uniquement.

    Le montage multi-canaux reste disponible via *Revue montage*.
    Les barres de plage et graphs d’analyse vivent dans Inspecter
    (double-clic MEA / liste), pas sur la vue centrale.
    """
    return (
        ViewTab(
            name="Canal",
            columns=1,
            panel_height_px=520,
            panels=(PanelPlacement("full_recording"),),
        ),
    )


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

    def to_plot_display(self, *, base: PlotDisplaySettings | None = None) -> PlotDisplaySettings:
        """Derive classic PDF settings from the panels currently laid out."""
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
        montage = any(name.startswith("montage_") for name in globals_seen)
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
