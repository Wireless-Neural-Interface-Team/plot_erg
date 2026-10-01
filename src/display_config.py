"""Display and legend settings for PDF channel pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

ZoomMode = Literal["none", "onset", "trigger_end", "both"]

PANEL_FIELD_NAMES: tuple[str, ...] = (
    "mean_raw",
    "first_trigger_raw",
    "second_trigger_raw",
    "mean_hp",
    "first_trigger_hp",
    "second_trigger_hp",
    "mean_lp",
    "first_trigger_lp",
    "second_trigger_lp",
    "rms",
    "first_rms",
    "second_rms",
    "psth",
    "first_psth",
    "second_psth",
    "isi",
    "first_isi",
    "second_isi",
    "trial_rate",
    "raster",
    "spike_overlay",
)

PANEL_LABELS: dict[str, str] = {
    "mean_raw": "Raw trial-averaged",
    "mean_hp": "High-pass trial-averaged",
    "mean_lp": "Low-pass trial-averaged",
    "first_trigger_raw": "First stimulation (raw)",
    "first_trigger_hp": "First stimulation (high-pass)",
    "first_trigger_lp": "First stimulation (low-pass)",
    "second_trigger_raw": "Second stimulation (raw)",
    "second_trigger_hp": "Second stimulation (high-pass)",
    "second_trigger_lp": "Second stimulation (low-pass)",
    "rms": "RMS (trial-averaged)",
    "first_rms": "RMS (first stimulation)",
    "second_rms": "RMS (second stimulation)",
    "psth": "PSTH / trial-averaged FR",
    "first_psth": "PSTH / FR (first stimulation)",
    "second_psth": "PSTH / FR (second stimulation)",
    "isi": "ISI (all stimulations)",
    "first_isi": "ISI (first stimulation)",
    "second_isi": "ISI (second stimulation)",
    "trial_rate": "Rate per trial (not averaged)",
    "raster": "Raster (all stimulations)",
    "spike_overlay": "Spike overlay (all spikes, end of section)",
}

# Panels that stay unchecked by default in the Display table.
PANEL_DEFAULT_OFF: frozenset[str] = frozenset(
    {
        "second_trigger_raw",
        "second_trigger_hp",
        "second_trigger_lp",
        "second_rms",
        "second_psth",
        "second_isi",
    }
)


@dataclass(frozen=True)
class SectionPanels:
    """Visibility toggles for one temporal section (full view, zoom onset, zoom end)."""

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
    """Global PDF layout and per-section panel visibility."""

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
    """Per-recording legend, visibility, and curve color."""

    plot_visible: bool = True
    legend_visible: bool = True
    # Matplotlib color (name or #RRGGBB). Empty/None → default cycle.
    color: str | None = None

    @classmethod
    def visible(cls) -> RecordingStyle:
        return cls(plot_visible=True, legend_visible=True)


# Named presets offered in the GUI (matplotlib tab10 + Auto).
RECORDING_COLOR_PRESETS: tuple[tuple[str, str], ...] = (
    ("Auto", ""),
    ("Blue", "#1f77b4"),
    ("Orange", "#ff7f0e"),
    ("Green", "#2ca02c"),
    ("Red", "#d62728"),
    ("Purple", "#9467bd"),
    ("Brown", "#8c564b"),
    ("Pink", "#e377c2"),
    ("Gray", "#7f7f7f"),
    ("Olive", "#bcbd22"),
    ("Cyan", "#17becf"),
)


def resolve_recording_plot_colors(
    styles: Sequence[RecordingStyle],
    indices: Sequence[int],
    *,
    fallback: Sequence[str] | None = None,
) -> list[str]:
    """Resolve one plot color per visible recording (custom or fallback cycle)."""
    cycle = [str(c) for c in (fallback or [hex_ for _, hex_ in RECORDING_COLOR_PRESETS[1:] if hex_])]
    if not cycle:
        cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
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
