"""Paramètres du programme : filtres, trigger, mise en page et blocs d'affichage."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, fields

SEABORN_PALETTES = [
    "deep", "muted", "bright", "pastel", "dark", "colorblind",
    "tab10", "Set1", "Set2", "Dark2", "Paired", "husl", "hls",
    "viridis", "rocket", "mako", "flare", "crest", "magma", "icefire",
    "Spectral", "coolwarm",
]

SIGNALS = [
    ("wide", "WIDE (large bande)"),
    ("low", "LOW (passe-bas)"),
    ("high", "HIGH (passe-haut)"),
    ("spikes", "SPIKES"),
    ("scope", "SPIKE SCOPE"),
    ("rms", "RMS glissant"),
]
MODES = [("triggered", "Aligné sur trigger"), ("continuous", "Continu")]
TRIAL_DISPLAYS = [
    ("mean", "Moyenne"),
    ("overlay", "Essais superposés"),
    ("mean+overlay", "Moyenne + essais"),
    ("mean+sd", "Moyenne ± écart-type"),
    ("mean+sem", "Moyenne ± SEM"),
]
BASELINES = [("none", "Aucune"), ("pre", "Moyenne pré-trigger"), ("mean", "Moyenne du segment")]
SPIKE_VIEWS = [("trace", "Trace HIGH + spikes"), ("raster", "Raster"), ("psth", "PSTH (fréquence)")]
TIME_UNITS = [("ms", "ms"), ("s", "s")]
FILTER_TYPES = [("butterworth", "Butterworth"), ("bessel", "Bessel"), ("chebyshev", "Chebyshev I")]
LINE_STYLES = [
    ("dot", "Pointillés"),
    ("dash", "Tirets"),
    ("dashdot", "Tiret-point"),
    ("solid", "Plein"),
]
LEGEND_POSITIONS = [
    ("top-right", "Haut droite"),
    ("top-left", "Haut gauche"),
    ("bottom-right", "Bas droite"),
    ("bottom-left", "Bas gauche"),
]
ALIGNMENTS = [("center", "Centré"), ("left", "Gauche"), ("right", "Droite")]
ZOOM_SIDES = [("right", "À droite"), ("bottom", "En dessous")]

# Champs de BlockConfig dont la modification impose un nouveau calcul (le reste = style).
DATA_FIELDS = {
    "stream", "channels", "signal", "mode", "t_start_s", "t_end_s", "pre_ms", "post_ms",
    "trial_display", "baseline", "spikes_view", "time_unit", "max_overlay",
}
# Champs qui ne sont pas recopiés par « Appliquer le style à tous ».
NON_STYLE_FIELDS = DATA_FIELDS | {"id", "title", "colors", "zoom_x0", "zoom_x1"}


@dataclass
class FilterSettings:
    notch_hz: float = 0.0
    phase: str = "zero"  # zero | causal
    low_type: str = "butterworth"
    low_order: int = 4
    low_cutoff_hz: float = 300.0
    high_type: str = "butterworth"
    high_order: int = 2
    high_cutoff_hz: float = 300.0
    spike_threshold_mode: str = "rms"  # uv | rms | mad
    spike_threshold: float = 5.0
    spike_polarity: str = "negative"  # negative | positive | both
    spike_refractory_ms: float = 1.0
    spike_artifact_uv: float = 1000.0
    scope_pre_ms: float = 1.0
    scope_post_ms: float = 2.0
    scope_max_waveforms: int = 300
    scope_align: str = "threshold"  # threshold | peak
    rms_window_ms: float = 20.0
    rms_source: str = "high"  # wide | low | high
    psth_bin_ms: float = 10.0


@dataclass
class TriggerSettings:
    enabled: bool = True
    source_stream: str = "adc"
    source_index: int = 0
    threshold: float = 1.5
    edge: str = "rising"  # rising | falling
    use_abs: bool = False
    min_duration_ms: float = 0.5
    min_interval_ms: float = 50.0
    search_start_s: float = 0.0
    search_end_s: float = 0.0
    first_trial: int = 1
    max_trials: int = 0
    step: int = 1
    exclude: str = ""

    def detection_key(self) -> tuple:
        return (
            self.source_stream, self.source_index, self.threshold, self.edge, self.use_abs,
            self.min_duration_ms, self.min_interval_ms, self.search_start_s, self.search_end_s,
        )


@dataclass
class LayoutSettings:
    columns: int = 3
    spacing: int = 10
    canvas_bg: str = "#e9ecef"
    antialias: bool = True
    export_scale: float = 2.0


def new_id() -> str:
    return uuid.uuid4().hex[:10]


@dataclass
class BlockConfig:
    id: str = field(default_factory=new_id)
    title: str = "Bloc"

    # Données
    stream: str = "amp"
    channels: list = field(default_factory=lambda: [0])
    colors: list = field(default_factory=list)
    signal: str = "wide"
    mode: str = "triggered"
    t_start_s: float = 0.0
    t_end_s: float = 5.0
    pre_ms: float = 50.0
    post_ms: float = 500.0
    trial_display: str = "mean+sd"
    baseline: str = "pre"
    spikes_view: str = "raster"
    time_unit: str = "ms"
    max_overlay: int = 200

    # Dimensions, titre, cadre
    width: int = 520
    height: int = 340
    title_show: bool = True
    title_size: int = 12
    title_color: str = "#1f2328"
    title_bold: bool = True
    title_align: str = "center"
    bg_color: str = "#ffffff"
    frame_color: str = "#c9ced6"
    frame_width: int = 1
    info_show: bool = True

    # Courbes
    palette: str = "deep"
    line_width: float = 1.5
    mean_width: float = 2.2
    trial_width: float = 0.8
    trial_alpha: float = 0.3
    trial_use_channel_color: bool = True
    trial_color: str = "#8a8f98"
    band_alpha: float = 0.25
    spike_color: str = "#d62728"
    spike_size: int = 7
    threshold_show: bool = True
    threshold_color: str = "#ff7f0e"
    threshold_width: float = 1.2
    threshold_style: str = "dash"

    # Axes
    x_label: str = ""
    y_label: str = ""
    axis_label_size: int = 10
    axis_color: str = "#30363d"
    axis_width: float = 1.0
    tick_size: int = 9
    show_x_axis: bool = True
    show_y_axis: bool = True
    grid: bool = False
    grid_alpha: float = 0.3

    # Échelles
    x_auto: bool = True
    x_min: float = 0.0
    x_max: float = 100.0
    y_auto: bool = True
    y_min: float = -100.0
    y_max: float = 100.0

    # Échelle flottante (barre d'échelle déplaçable)
    sb_show: bool = False
    sb_show_x: bool = True
    sb_show_y: bool = True
    sb_x_len: float = 50.0
    sb_y_len: float = 50.0
    sb_x_unit: str = ""
    sb_y_unit: str = ""
    sb_color: str = "#000000"
    sb_width: float = 2.5
    sb_font_size: int = 9
    sb_fx: float = 0.9
    sb_fy: float = 0.12

    # Légende
    legend_show: bool = True
    legend_pos: str = "top-right"
    legend_size: int = 9
    legend_text_color: str = "#1f2328"
    legend_bg: str = "#ffffffd0"
    legend_frame: str = "#c9ced6"

    # Zone de zoom
    zoom_show: bool = False
    zoom_x0: float = 0.0
    zoom_x1: float = 0.0
    zoom_side: str = "right"
    zoom_ratio: float = 0.4
    zoom_color: str = "#1f77b4"
    zoom_alpha: float = 0.15
    zoom_y_auto: bool = True

    # Lignes de trigger ON / OFF
    trig_on_show: bool = True
    trig_on_color: str = "#2ca02c"
    trig_on_width: float = 1.5
    trig_on_style: str = "dot"
    trig_off_show: bool = True
    trig_off_color: str = "#d62728"
    trig_off_width: float = 1.5
    trig_off_style: str = "dot"
    trig_label_show: bool = True
    trig_max_lines: int = 300


def to_dict(obj) -> dict:
    return asdict(obj)


def from_dict(cls, raw: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (raw or {}).items() if k in names})


def copy_style(src: BlockConfig, dst: BlockConfig) -> None:
    for f in fields(BlockConfig):
        if f.name not in NON_STYLE_FIELDS:
            setattr(dst, f.name, getattr(src, f.name))
