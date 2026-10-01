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
    "mean_raw": "Moyenne d’essais — brut",
    "mean_hp": "Moyenne d’essais — passe-haut",
    "mean_lp": "Moyenne d’essais — passe-bas",
    "first_trigger_raw": "1re stimulation (brut)",
    "first_trigger_hp": "1re stimulation (passe-haut)",
    "first_trigger_lp": "1re stimulation (passe-bas)",
    "second_trigger_raw": "2e stimulation (brut)",
    "second_trigger_hp": "2e stimulation (passe-haut)",
    "second_trigger_lp": "2e stimulation (passe-bas)",
    "rms": "RMS (moyenne d’essais)",
    "first_rms": "RMS (1re stimulation)",
    "second_rms": "RMS (2e stimulation)",
    "psth": "PSTH / FR moyenne",
    "first_psth": "PSTH / FR (1re stimulation)",
    "second_psth": "PSTH / FR (2e stimulation)",
    "isi": "ISI (toutes stimulations)",
    "first_isi": "ISI (1re stimulation)",
    "second_isi": "ISI (2e stimulation)",
    "trial_rate": "Taux par essai (non moyenné)",
    "raster": "Raster (toutes stimulations)",
    "spike_overlay": "Superposition de spikes",
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


# Named presets for GUI recordings.
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


def resolve_recording_plot_colors(
    styles: Sequence[RecordingStyle],
    indices: Sequence[int],
    *,
    fallback: Sequence[str] | None = None,
) -> list[str]:
    """Resolve one plot color per visible recording (custom or fallback cycle)."""
    cycle = [str(c) for c in (fallback or [hex_ for _, hex_ in RECORDING_COLOR_PRESETS[1:] if hex_])]
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
