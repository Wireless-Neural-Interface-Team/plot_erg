from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

EdgeKind = Literal["falling", "rising"]


@dataclass(frozen=True)
class AnalysisConfig:
    rhs_file: Path
    threshold: float = 1.0
    edge: EdgeKind = "falling"  # falling / rising threshold crossing on ANALOG_IN 0
    pre_s: float = 1.0
    post_s: float = 10.0
    # Butterworth low-pass on amplifier_data (None = disabled)
    lowpass_cutoff_hz: Optional[float] = 250.0
    save_dir: Path | None = None
    # Output PDF title / basename (.pdf added if no extension)
    pdf_title: str | None = None
    # Spike threshold (µV): >=0 = upward crossing; <0 = downward crossing
    spike_threshold_uv: float = -15.0
    # PSTH firing-rate Gaussian smoothing width (s)
    firing_rate_window_s: float = 0.010
    # PDF zoom-panel window (s, time relative to trigger)
    zoom_t0_s: float = -0.1
    zoom_t1_s: float = 0.4
    # RMS computation window (s): temporal window used for moving-RMS calculation
    rms_window_s: float = 0.010
    # Butterworth band-pass on amplifier for raster / PSTH / ISI (both None = raw)
    spike_bandpass_low_hz: Optional[float] = 250.0
    spike_bandpass_high_hz: Optional[float] = 7500.0
    # None = auto: (save_dir or .rhs folder) / ".plot_erg" / <stem>
    work_dir: Path | None = None
    # Do not delete work_dir after run (keeps amplifier .npy, etc.)
    keep_intermediate_files: bool = False
    # Process workers for A/B comparison (>=1)
    comparison_workers: int = 32
    # Max channel worker threads (None = auto, cap 16)
    channel_workers: int | None = None
    # Lightweight PDF mode (raster/ISI downsample + lower DPI)
    lightweight_plot: bool = False
    # Fraction of spike-plot points to keep (1..100)
    sampling_percent: int = 100
    # probeinterface JSON (MEA map inset in PDF when channel maps)
    probe_layout_json: Path | None = None
