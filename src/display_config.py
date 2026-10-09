"""Réglages d’affichage PDF et styles d’enregistrement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from panel_catalog import SECTION_PANEL_FIELD_NAMES, SECTION_PANEL_LABELS

# Compat : anciens noms importés par plotting / config.
PANEL_FIELD_NAMES = SECTION_PANEL_FIELD_NAMES
PANEL_LABELS = SECTION_PANEL_LABELS

ZoomMode = Literal["none", "onset", "trigger_end", "both"]


@dataclass(frozen=True)
class SectionPanels:
    """Visibilité des panneaux pour une section temporelle (vue complète / zooms)."""

    mean_raw: bool = True
    first_trigger_raw: bool = True
    second_trigger_raw: bool = False
    mean_hp: bool = True
    first_trigger_hp: bool = True
    second_trigger_hp: bool = False
    mean_lp: bool = True
    first_trigger_lp: bool = True
    second_trigger_lp: bool = False
    rms: bool = True
    first_rms: bool = True
    second_rms: bool = False
    psth: bool = True
    first_psth: bool = True
    second_psth: bool = False
    isi: bool = True
    first_isi: bool = True
    second_isi: bool = False
    trial_rate: bool = True
    raster: bool = True
    spike_overlay: bool = True

    def any_enabled(self) -> bool:
        return any(getattr(self, name) for name in PANEL_FIELD_NAMES)


@dataclass(frozen=True)
class PlotDisplaySettings:
    """Mise en page PDF et visibilité par section."""

    mea_layout: bool = True
    impedance: bool = True
    summary_rms_page: bool = True
    summary_rms_table_page: bool = True
    summary_impedance_page: bool = True
    summary_second_stim_montage_page: bool = False
    full_view: SectionPanels = SectionPanels()
    zoom_onset: SectionPanels = SectionPanels()
    zoom_trigger_end: SectionPanels = SectionPanels()

    @classmethod
    def all_on(cls) -> PlotDisplaySettings:
        return cls()

    def section_panels(self, section: str) -> SectionPanels:
        if section == "full":
            return self.full_view
        if section == "zoom_onset":
            return self.zoom_onset
        if section == "zoom_trigger_end":
            return self.zoom_trigger_end
        raise ValueError(f"Unknown section: {section}")


@dataclass(frozen=True)
class RecordingStyle:
    """Légende, visibilité et couleur d’un enregistrement."""

    plot_visible: bool = True
    legend_visible: bool = True
    color: str | None = None

    @classmethod
    def visible(cls) -> RecordingStyle:
        return cls(plot_visible=True, legend_visible=True)


RECORDING_COLOR_PRESETS: tuple[tuple[str, str], ...] = (
    ("Auto", ""),
    ("Blue", "#2563eb"),
    ("Red", "#dc2626"),
    ("Green", "#16a34a"),
    ("Orange", "#ea580c"),
    ("Purple", "#7c3aed"),
    ("Cyan", "#0891b2"),
    ("Yellow", "#ca8a04"),
    ("Gray", "#4b5563"),
)

DEFAULT_TRACE_COLORS: tuple[str, ...] = tuple(
    hex_ for _, hex_ in RECORDING_COLOR_PRESETS[1:] if hex_
)

# Couleurs / constantes de tracé partagées (GUI + PDF).
STIM_ONSET_COLOR = "#dc2626"
STIM_OFFSET_COLOR = "#1d4ed8"
STREAM_PLOT_COLORS: dict[str, str] = {
    "raw": "#334155",
    "hp": "#15803d",
    "lp": "#1e40af",
}
MUTED_AXIS_TEXT = "#64748b"
CHANNEL_HIGHLIGHT_FACE = "#fff7ed"
ZERO_LINE_COLOR = "#6b7280"
ARTIFACT_LINE_COLOR = "#2563eb"


def resolve_recording_plot_colors(
    styles: Sequence[RecordingStyle],
    indices: Sequence[int],
    *,
    fallback: Sequence[str] | None = None,
) -> list[str]:
    """Une couleur par enregistrement visible (personnalisée ou cycle)."""
    cycle = [str(c) for c in (fallback or DEFAULT_TRACE_COLORS)]
    if not cycle:
        cycle = ["#2563eb", "#dc2626", "#16a34a", "#ea580c"]
    out: list[str] = []
    for k, idx in enumerate(indices):
        style = styles[int(idx)] if 0 <= int(idx) < len(styles) else None
        custom = (style.color if style is not None else None) or ""
        custom = str(custom).strip()
        out.append(custom if custom else cycle[k % len(cycle)])
    return out


def resolve_display_label(stem: str, custom: str | None) -> str:
    text = (custom or "").strip()
    return text if text else stem
