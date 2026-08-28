from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from display_config import PlotDisplaySettings, RecordingStyle, ZoomMode

EdgeKind = Literal["falling", "rising", "none"]
SpikeThresholdMode = Literal["fixed", "rms_multiple"]
SpikeThresholdPolarity = Literal["negative", "positive"]
SpikeFilterKind = Literal["highpass", "lowpass"]
IntanFilterType = Literal["bessel", "butterworth"]
SectionSpecKind = Literal["count", "duration"]


@dataclass(frozen=True)
class AnalysisConfig:
    rhs_file: Path
    threshold: float = 1.0
    edge: EdgeKind = "falling"
    pre_s: float = 1.0
    post_s: float = 10.0
    section_count: int = 10
    section_duration_s: float | None = None
    section_spec: SectionSpecKind = "count"
    section_trigger_start_s: float = 1.0
    section_trigger_end_s: float = 4.0
    save_dir: Path | None = None
    pdf_title: str | None = None
    spike_threshold_uv: float = 70.0
    spike_threshold_polarity: SpikeThresholdPolarity = "negative"
    spike_threshold_mode: SpikeThresholdMode = "fixed"
    spike_threshold_rms_multiplier: float = 4.0
    psth_bin_window_s: float = 0.050
    # Intan Spike Scope time scale T (ms): display window is [-T/2, +T] around detection.
    spike_scope_tscale_ms: float = 4.0
    # Zoom: mode selects which temporal sections are rendered in the PDF.
    zoom_mode: ZoomMode = "both"
    zoom_onset_t0_s: float = -0.1
    zoom_onset_t1_s: float = 0.4
    zoom_end_t0_s: float = -0.1
    zoom_end_t1_s: float = 0.4
    first_trigger_hp_ylim_enabled: bool = False
    first_trigger_hp_ylim_min_uv: float = -200.0
    first_trigger_hp_ylim_max_uv: float = 200.0
    rms_window_s: float = 1.0
    intan_spike_filter_kind: SpikeFilterKind = "highpass"
    intan_filter_order: int = 2
    intan_filter_type: IntanFilterType = "bessel"
    intan_filter_cutoff_hz: float = 250.0
    intan_artifact_threshold_uv: float = 2500.0
    intan_artifact_suppression_enabled: bool = True
    work_dir: Path | None = None
    comparison_workers: int = 32
    channel_workers: int | None = None
    sampling_percent: int = 100
    probe_layout_json: Path | None = None
    # Custom legend label (falls back to .rhs stem when empty).
    recording_label: str | None = None
    recording_style: RecordingStyle = field(default_factory=RecordingStyle.visible)
    plot_display: PlotDisplaySettings = field(default_factory=PlotDisplaySettings.all_on)
