"""Display and legend settings for PDF channel pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ZoomMode = Literal["none", "onset", "trigger_end", "both"]

PANEL_FIELD_NAMES: tuple[str, ...] = (
    "mean_raw",
    "mean_filtered",
    "first_trigger_raw",
    "first_trigger_hp",
    "rms",
    "raster",
    "psth",
    "trial_rate",
    "isi",
)

PANEL_LABELS: dict[str, str] = {
    "mean_raw": "Raw mean",
    "mean_filtered": "Filtered mean",
    "first_trigger_raw": "First stimulation (raw)",
    "first_trigger_hp": "First stimulation (filtered)",
    "rms": "RMS evolution",
    "raster": "Raster",
    "psth": "PSTH / firing rate",
    "trial_rate": "Rate per trial",
    "isi": "ISI",
}


@dataclass(frozen=True)
class SectionPanels:
    """Visibility toggles for one temporal section (full view, zoom onset, zoom end)."""

    mean_raw: bool = True
    mean_filtered: bool = True
    first_trigger_raw: bool = True
    first_trigger_hp: bool = True
    rms: bool = True
    raster: bool = True
    psth: bool = True
    trial_rate: bool = True
    isi: bool = True

    def any_enabled(self) -> bool:
        return any(getattr(self, name) for name in PANEL_FIELD_NAMES)


@dataclass(frozen=True)
class PlotDisplaySettings:
    """Global PDF layout and per-section panel visibility."""

    mea_layout: bool = True
    impedance: bool = True
    summary_rms_page: bool = True
    summary_impedance_page: bool = True
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
    """Per-recording legend and visibility (independent of AnalysisConfig.rhs_file)."""

    plot_visible: bool = True
    legend_visible: bool = True

    @classmethod
    def visible(cls) -> RecordingStyle:
        return cls(plot_visible=True, legend_visible=True)


def resolve_display_label(stem: str, custom: str | None) -> str:
    text = (custom or "").strip()
    return text if text else stem
