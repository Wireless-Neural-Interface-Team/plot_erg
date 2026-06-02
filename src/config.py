from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

EdgeKind = Literal["falling", "rising", "none"]
CurveFilterKind = Literal["highpass", "lowpass", "bandpass", "no filter"]
SpikeThresholdMode = Literal["fixed", "rms_multiple"]
SpikeFilterKind = Literal["highpass", "lowpass"]
IntanFilterType = Literal["bessel", "butterworth"]
SectionSpecKind = Literal["count", "duration"]


@dataclass(frozen=True)
class AnalysisConfig:
    rhs_file: Path
    threshold: float = 1.0
    edge: EdgeKind = "falling"  # falling / rising on ANALOG_IN 0, or none (fixed sections)
    pre_s: float = 1.0
    post_s: float = 10.0
    # No-trigger mode: split each recording into equal sections for averaging.
    section_count: int = 10
    section_duration_s: float | None = None
    section_spec: SectionSpecKind = "count"  # which GUI field drives the other
    # No-trigger mode: imaginary trigger window inside each section (seconds from segment start).
    section_trigger_start_s: float = 1.0
    section_trigger_end_s: float = 4.0
    # Legacy: Butterworth low-pass on amplifier_data (None = disabled).
    # Kept for CLI backward compatibility.
    lowpass_cutoff_hz: Optional[float] = None
    # Curve filter for amplifier mean traces in PDF.
    curve_filter: CurveFilterKind = "no filter"
    curve_filter_low_hz: Optional[float] = None
    curve_filter_high_hz: Optional[float] = None
    save_dir: Path | None = None
    # Output PDF title / basename (.pdf added if no extension)
    pdf_title: str | None = None
    # Spike threshold (µV) on Intan HIGH: >=0 rising, <0 falling (RHX default -70)
    spike_threshold_uv: float = -70.0
    # Spike threshold mode:
    # - fixed: same threshold (µV) for all channels.
    # - rms_multiple: threshold = spike_threshold_rms_multiplier * mean RMS per channel.
    spike_threshold_mode: SpikeThresholdMode = "fixed"
    # Multiplier used when spike_threshold_mode == "rms_multiple"
    spike_threshold_rms_multiplier: float = 4.0
    # PSTH time window (s) used for each PSTH point
    psth_bin_window_s: float = 0.050
    # PDF zoom-panel window (s, time relative to trigger)
    zoom_t0_s: float = -0.1
    zoom_t1_s: float = 0.4
    # RMS window (s): fixed to 1.0 in Intan Spike Scope (kept for CLI/GUI compatibility)
    rms_window_s: float = 1.0
    # Intan RHX-style software filter for raster / PSTH / ISI / RMS (default = HIGH)
    intan_spike_filter_kind: SpikeFilterKind = "highpass"
    intan_filter_order: int = 2
    intan_filter_type: IntanFilterType = "bessel"
    intan_filter_cutoff_hz: float = 250.0
    intan_artifact_threshold_uv: float = 2500.0
    intan_artifact_suppression_enabled: bool = True
    # Deprecated: ignored; spikes/RMS use Intan HIGH above
    spike_bandpass_low_hz: Optional[float] = None
    spike_bandpass_high_hz: Optional[float] = None
    # None = auto: (save_dir or .rhs folder) / ".plot_erg" / <stem>
    work_dir: Path | None = None
    # Process workers for A/B comparison (>=1)
    comparison_workers: int = 32
    # Channel worker threads (None = all CPU cores up to 16; explicit value capped at 16)
    channel_workers: int | None = None
    # Fraction of spike-plot points to keep (1..100)
    sampling_percent: int = 100
    # probeinterface JSON (MEA map inset in PDF when channel maps)
    probe_layout_json: Path | None = None
