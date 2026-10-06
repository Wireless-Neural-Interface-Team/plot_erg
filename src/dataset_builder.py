"""Builds :class:`processed_dataset.ProcessedRecording` objects on demand.

``build_recording`` only reads the wideband stream and detects stimulations.
Per-channel products (filter, averages, RMS, spikes, overlays) are computed
later by :func:`ensure_channels` for the channels the GUI actually asks for.

====================  ==============================================
stage                 invalidated by
====================  ==============================================
``read``              the ``.rhs`` file
``segment``           edge mode, threshold, pre/post, sections
``channel``           filter + segmentation + spike / RMS / overlay
====================  ==============================================

Each stage reports progress and its own duration, which the GUI shows as
loading times.
"""

from __future__ import annotations

import datetime as _dt
import gc
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from config import AnalysisConfig
from core import (
    AmplifierSpikeSource,
    build_intan_dsp_settings,
    check_analysis_cancelled,
    get_analog_in0_signal,
    get_channel_names,
    get_sampling_rate,
    load_rhs_file,
    resolve_channel_workers,
    resolve_recording_windows,
    uses_analog_trigger,
)
from display_config import RecordingStyle
from erg_cache import (
    BundleLayout,
    CacheKeys,
    bundle_for,
    default_cache_root,
    raw_layout_for,
)
from impedance_tracking import collect_impedance_sessions
from intan_rhx_dsp import IntanDspSettings, build_intan_filter_sos
from memmap_io import load_readonly_memmap, open_writable_memmap
from processed_dataset import (
    MAX_OVERLAY_SNIPPETS_PER_CHANNEL,
    ChannelData,
    DerivedArrays,
    OverlaySnippets,
    ProcessedRecording,
    RecordingMeta,
    SegmentationInfo,
    SpikeDetectionInfo,
    SpikeTrains,
    STREAM_NAMES,
    read_bundle,
    recording_label_for,
)

# Weights for ``build_recording`` only. Channel work is a separate job and
# reports its own 0→1 overall fraction via ``ensure_channels``.
BUILD_STAGES: tuple[tuple[str, str, float], ...] = (
    ("read", "Reading recording", 3.0),
    ("segment", "Detecting stimulations", 0.4),
    ("finalize", "Preparing viewer", 0.2),
)
STAGES: tuple[tuple[str, str, float], ...] = BUILD_STAGES + (
    ("channel", "Computing selected channels", 2.0),
)

_STAGE_LABELS = {key: label for key, label, _w in STAGES}
_STAGE_WEIGHTS = {key: weight for key, _l, weight in BUILD_STAGES}
_TOTAL_WEIGHT = sum(_STAGE_WEIGHTS.values())


@dataclass(frozen=True)
class ProgressEvent:
    """One progress notification emitted while building a dataset."""

    recording: str
    stage: str
    stage_label: str
    stage_fraction: float
    overall_fraction: float
    message: str
    elapsed_s: float
    cached: bool = False


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class StageTiming:
    stage: str
    label: str
    seconds: float
    cached: bool

    def describe(self) -> str:
        suffix = " (cached)" if self.cached else ""
        return f"{self.label}: {self.seconds:.2f} s{suffix}"


@dataclass
class BuildReport:
    """Per-stage durations and cache hits for one built recording."""

    recording: str
    timings: list[StageTiming] = field(default_factory=list)
    total_s: float = 0.0
    bundle_root: Path | None = None
    reused_bundle: bool = False

    @property
    def cached_stages(self) -> list[str]:
        return [timing.label for timing in self.timings if timing.cached]

    def as_pairs(self) -> list[tuple[str, float]]:
        return [(timing.label, timing.seconds) for timing in self.timings]

    def summary(self) -> str:
        cached = len(self.cached_stages)
        detail = ", ".join(timing.describe() for timing in self.timings)
        head = f"{self.recording}: {self.total_s:.2f} s total"
        if cached:
            head += f" ({cached}/{len(self.timings)} stage(s) from cache)"
        return f"{head} — {detail}" if detail else head


class _Progress:
    """Tracks stage weights so the overall fraction advances monotonically."""

    def __init__(self, recording: str, callback: ProgressCallback | None) -> None:
        self._recording = recording
        self._callback = callback
        self._done_weight = 0.0
        self._stage = "read"
        self._started = time.perf_counter()
        self._stage_started = self._started
        self.report = BuildReport(recording=recording)

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started

    def emit(self, fraction: float, message: str, *, cached: bool = False) -> None:
        if self._callback is None:
            return
        weight = _STAGE_WEIGHTS.get(self._stage, 1.0)
        clamped = max(0.0, min(1.0, float(fraction)))
        overall = (self._done_weight + clamped * weight) / _TOTAL_WEIGHT
        self._callback(
            ProgressEvent(
                recording=self._recording,
                stage=self._stage,
                stage_label=_STAGE_LABELS.get(self._stage, self._stage),
                stage_fraction=clamped,
                overall_fraction=max(0.0, min(1.0, overall)),
                message=message,
                elapsed_s=self.elapsed_s,
                cached=cached,
            )
        )

    def start_stage(self, stage: str, message: str | None = None) -> None:
        self._stage = stage
        self._stage_started = time.perf_counter()
        self.emit(0.0, message or _STAGE_LABELS.get(stage, stage))

    def finish_stage(self, *, cached: bool = False, message: str | None = None) -> None:
        seconds = time.perf_counter() - self._stage_started
        self.report.timings.append(
            StageTiming(
                stage=self._stage,
                label=_STAGE_LABELS.get(self._stage, self._stage),
                seconds=seconds,
                cached=cached,
            )
        )
        self.emit(1.0, message or f"{_STAGE_LABELS.get(self._stage, self._stage)} done", cached=cached)
        self._done_weight += _STAGE_WEIGHTS.get(self._stage, 1.0)

    def finish(self) -> BuildReport:
        self.report.total_s = self.elapsed_s
        return self.report


def _chunked(total: int, parts: int) -> Iterator[tuple[int, int]]:
    if total <= 0:
        return
    step = max(1, total // max(1, parts))
    start = 0
    while start < total:
        end = min(total, start + step)
        yield start, end
        start = end


# ------------------------------------------------------------------ raw stream


def _ensure_raw_stream(
    config: AnalysisConfig,
    keys: CacheKeys,
    cache_root: Path,
    progress: _Progress,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Return ``(raw_meta, amplifier_memmap, analog_in0)`` from cache or the file."""
    layout = raw_layout_for(cache_root, keys)
    progress.start_stage("read")
    if layout.is_complete():
        meta = layout.read_meta()
        amplifier = load_readonly_memmap(layout.amplifier_path)
        analog = (
            np.asarray(load_readonly_memmap(layout.analog_in0_path), dtype=np.float64)
            if layout.analog_in0_path.exists()
            else np.empty(0, dtype=np.float64)
        )
        progress.finish_stage(
            cached=True,
            message=f"Reusing cached wideband stream ({meta.get('n_channels', '?')} channels)",
        )
        return meta, amplifier, analog

    if not config.rhs_file.exists():
        raise FileNotFoundError(f"File not found: {config.rhs_file}")
    progress.emit(0.05, f"Reading {config.rhs_file.name}...")
    data = load_rhs_file(config.rhs_file)
    check_analysis_cancelled()
    fs = get_sampling_rate(data)
    amplifier_raw = np.asarray(data.get("amplifier_data"))
    if amplifier_raw.size == 0:
        raise RuntimeError("RHS file does not contain amplifier_data.")
    n_channels, n_samples = int(amplifier_raw.shape[0]), int(amplifier_raw.shape[1])
    analog = (
        np.asarray(get_analog_in0_signal(data), dtype=np.float64)
        if uses_analog_trigger(config)
        else np.empty(0, dtype=np.float64)
    )
    version = data.get("version") or {}
    frequency = data.get("frequency_parameters") or {}
    meta = {
        "source": str(config.rhs_file),
        "fs": float(fs),
        "n_channels": n_channels,
        "n_samples": n_samples,
        "channel_names": get_channel_names(data, n_channels),
        "rhs_version_major": int(version.get("major", 3)),
        "notch_filter_frequency_hz": float(frequency.get("notch_filter_frequency") or 0.0),
    }

    progress.emit(0.6, f"Caching wideband stream ({n_channels} channels)...")
    layout.ensure()
    amp_mm = open_writable_memmap(
        layout.amplifier_path, (n_channels, n_samples), np.dtype(np.float32)
    )
    try:
        for index, (start, end) in enumerate(_chunked(n_channels, 10)):
            check_analysis_cancelled()
            amp_mm[start:end] = np.asarray(amplifier_raw[start:end], dtype=np.float32)
            progress.emit(
                0.6 + 0.35 * (end / max(1, n_channels)),
                f"Caching wideband stream: channel {end}/{n_channels}",
            )
            del index
        amp_mm.flush()
    finally:
        del amp_mm
    if analog.size:
        np.save(layout.analog_in0_path, np.asarray(analog, dtype=np.float32))
    layout.write_meta(meta)

    del amplifier_raw
    if isinstance(data, dict):
        data.pop("amplifier_data", None)
        data.pop("board_adc_data", None)
    del data
    gc.collect()

    amplifier = load_readonly_memmap(layout.amplifier_path)
    progress.finish_stage(message=f"Recording read ({n_channels} channels, {n_samples} samples)")
    return meta, amplifier, analog


def _dsp_settings_from_raw_meta(raw_meta: dict[str, Any], config: AnalysisConfig) -> IntanDspSettings:
    """Rebuild DSP settings without the RHS dict (cache-only path)."""
    synthetic = {
        "frequency_parameters": {
            "amplifier_sample_rate": float(raw_meta["fs"]),
            "notch_filter_frequency": float(raw_meta.get("notch_filter_frequency_hz", 0.0)),
        },
        "version": {"major": int(raw_meta.get("rhs_version_major", 3))},
    }
    return build_intan_dsp_settings(synthetic, config)


# ------------------------------------------------------------- filtered stream

# Cap in-RAM filtered rows per bank (HP and LP each). Beyond this, oldest rows
# are evicted; they are recomputed on next access.
LAZY_FILTER_CACHE_MAX_CHANNELS = 8


class LazyFilterBank:
    """Filters individual amplifier channels on first access and caches the row."""

    def __init__(
        self,
        amplifier: np.ndarray,
        dsp: IntanDspSettings,
        kind: str,
        *,
        max_cached_channels: int = LAZY_FILTER_CACHE_MAX_CHANNELS,
    ) -> None:
        if kind not in ("hp", "lp"):
            raise ValueError(f"Unsupported filter bank kind: {kind}")
        self._amplifier = amplifier
        self._dsp = dsp
        self._kind = kind
        self._max_cached = max(1, int(max_cached_channels))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.shape = (int(amplifier.shape[0]), int(amplifier.shape[1]))
        notch_sos, hp_sos, lp_sos = build_intan_filter_sos(dsp)
        self._notch_sos = notch_sos
        self._sos = hp_sos if kind == "hp" else lp_sos

    def __getitem__(self, index: int) -> np.ndarray:
        ch = int(index)
        cached = self._cache.get(ch)
        if cached is not None:
            self._cache.move_to_end(ch)
            return cached
        from scipy.signal import sosfilt

        check_analysis_cancelled()
        signal = np.asarray(self._amplifier[ch], dtype=np.float64)
        if self._notch_sos is not None:
            signal = sosfilt(self._notch_sos, signal)
        filtered = np.asarray(sosfilt(self._sos, signal), dtype=np.float32)
        self._cache[ch] = filtered
        while len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return filtered

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------- segmentation


def _segment(
    config: AnalysisConfig,
    raw_meta: dict[str, Any],
    analog_in0: np.ndarray,
    progress: _Progress,
) -> tuple[np.ndarray, np.ndarray, SegmentationInfo]:
    progress.start_stage("segment")
    fs = float(raw_meta["fs"])
    n_samples = int(raw_meta["n_samples"])
    triggers, t_rel, pre_n, post_n, n_valid, n_total, end_rising_s = resolve_recording_windows(
        config=config,
        n_samples=n_samples,
        fs=fs,
        analog_in0=analog_in0,
    )
    info = SegmentationInfo(
        mode=str(config.edge),
        n_trials=int(n_valid),
        n_triggers_detected=int(n_total),
        pre_n=int(pre_n),
        post_n=int(post_n),
        pre_s=float(pre_n) / fs,
        post_s=float(post_n) / fs,
        end_rising_s=end_rising_s,
        section_count=int(config.section_count) if config.edge == "none" else None,
        section_duration_s=(
            float(pre_n + post_n) / fs if config.edge == "none" else None
        ),
    )
    progress.finish_stage(message=info.describe())
    return np.asarray(triggers, dtype=np.int64), np.asarray(t_rel, dtype=np.float64), info


# -------------------------------------------------------------------- averages


def _trigger_windows(
    streams: dict[str, np.ndarray],
    triggers: np.ndarray,
    info: SegmentationInfo,
    window: int,
    *,
    trigger_indices: Sequence[int] = (0, 1),
) -> dict[tuple[int, str], np.ndarray]:
    """Single-stimulation windows kept in RAM only when streams are unavailable."""
    out: dict[tuple[int, str], np.ndarray] = {}
    n_channels = int(streams["raw"].shape[0])
    n_samples = int(streams["raw"].shape[1])
    for trigger_index in trigger_indices:
        if trigger_index >= int(triggers.size):
            continue
        start = int(triggers[trigger_index]) - int(info.pre_n)
        end = int(triggers[trigger_index]) + int(info.post_n)
        if start < 0 or end > n_samples:
            continue
        for stream in STREAM_NAMES:
            block = np.asarray(streams[stream][:, start:end], dtype=np.float32)
            out[(trigger_index, stream)] = block[:, :window]
        del n_channels
    return out


# ------------------------------------------------------------------ public API


def build_recording(
    config: AnalysisConfig,
    *,
    cache_root: Path | None = None,
    progress: ProgressCallback | None = None,
    label: str | None = None,
    style: RecordingStyle | None = None,
    keep_trigger_windows_in_ram: bool = False,
) -> tuple[ProcessedRecording, BuildReport]:
    """Prepare one recording: read + segment. Channels are filled on demand."""
    del keep_trigger_windows_in_ram  # Trigger windows are sliced from streams when needed.
    display_label = label or recording_label_for(config.rhs_file, config.recording_label)
    tracker = _Progress(display_label, progress)
    keys = CacheKeys.from_config(config)
    root = Path(cache_root) if cache_root is not None else default_cache_root(config)
    bundle = bundle_for(root, keys, config.rhs_file.stem)

    raw_meta, amplifier, analog_in0 = _ensure_raw_stream(config, keys, root, tracker)
    dsp = _dsp_settings_from_raw_meta(raw_meta, config)
    triggers, t_rel, info = _segment(config, raw_meta, analog_in0, tracker)
    n_channels = int(raw_meta["n_channels"])

    high = LazyFilterBank(amplifier, dsp, "hp")
    low = LazyFilterBank(amplifier, dsp, "lp")
    source = AmplifierSpikeSource(
        amplifier=amplifier,
        highpass=high,  # type: ignore[arg-type]
        lowpass=low,  # type: ignore[arg-type]
        valid_triggers=triggers,
        pre_n=int(info.pre_n),
        post_n=int(info.post_n),
        work_dir=None,
        intan_dsp=dsp,
    )

    tracker.start_stage("finalize")
    derived = DerivedArrays(
        t_rel=np.asarray(t_rel, dtype=np.float64),
        triggers=np.asarray(triggers, dtype=np.int64),
    )
    meta = RecordingMeta(
        source_path=config.rhs_file,
        source_name=config.rhs_file.name,
        fs=float(raw_meta["fs"]),
        n_channels=n_channels,
        n_samples=int(raw_meta["n_samples"]),
        channel_names=tuple(str(name) for name in raw_meta["channel_names"]),
        segmentation=info,
        spike_detection=SpikeDetectionInfo(
            mode=str(config.spike_threshold_mode),
            threshold_uv=float(config.spike_threshold_uv),
            polarity=str(config.spike_threshold_polarity),
            rms_multiplier=float(config.spike_threshold_rms_multiplier),
            artifact_threshold_uv=float(config.intan_artifact_threshold_uv),
            artifact_suppression=bool(config.intan_artifact_suppression_enabled),
        ),
        dsp=dsp,
        cache_keys=keys.as_dict(),
        created_at=_dt.datetime.now().isoformat(timespec="seconds"),
        overlay_pre_ms=float(config.spike_overlay_pre_ms),
        overlay_post_ms=float(config.spike_overlay_post_ms),
    )
    impedance_sessions = collect_impedance_sessions([config.rhs_file])
    recording = ProcessedRecording(
        meta=meta,
        derived=derived,
        spikes=SpikeTrains.empty(n_channels, int(info.n_trials)),
        overlay=OverlaySnippets.empty(n_channels, 0),
        label=display_label,
        style=style or config.recording_style,
        impedance_sessions=impedance_sessions,
        source=source,
        bundle_root=bundle.root,
        threshold_captions=[""] * n_channels,
    )
    tracker.finish_stage(
        message=(
            f"Ready — {n_channels} channels available; "
            "curves are computed when the GUI selects them"
        )
    )

    report = tracker.finish()
    report.bundle_root = bundle.root
    report.reused_bundle = False
    recording.build_timings = report.as_pairs()
    return recording, report


def ensure_channels(
    recording: ProcessedRecording,
    channels: Sequence[int],
    config: AnalysisConfig,
    *,
    progress: ProgressCallback | None = None,
    force: bool = False,
    need_means: bool = True,
    need_rms: bool = True,
    need_spikes: bool = True,
    need_overlay: bool = True,
) -> list[int]:
    """Compute derived products for the requested channel indices.

    Returns the list of channels that were (re)computed.
    Products can be selected so opening a simple average view does not pay
    for spike overlays, and vice versa.
    """
    source = recording.source
    if source is None:
        raise RuntimeError("Recording has no live streams; cannot compute channels on demand.")

    wanted = sorted({int(ch) for ch in channels if 0 <= int(ch) < recording.n_channels})
    if not force:
        still: list[int] = []
        for ch in wanted:
            if not recording.is_channel_ready(ch):
                still.append(ch)
                continue
            data = recording._channel_data.get(ch)  # noqa: SLF001 — intentional
            if data is None:
                still.append(ch)
                continue
            if need_means and not getattr(data, "means_ready", True):
                still.append(ch)
            elif need_rms and not getattr(data, "rms_ready", True):
                still.append(ch)
            elif need_spikes and not getattr(data, "spikes_ready", True):
                still.append(ch)
            elif need_overlay and not getattr(data, "overlay_ready", True):
                still.append(ch)
        wanted = still
    if not wanted:
        return []

    display_label = recording.label
    started = time.perf_counter()
    t_rel = np.asarray(recording.derived.t_rel, dtype=np.float64)
    triggers = np.asarray(recording.derived.triggers, dtype=np.int64)
    pre_n = int(recording.meta.segmentation.pre_n)
    post_n = int(recording.meta.segmentation.post_n)
    workers = resolve_channel_workers(config.channel_workers, len(wanted))
    # Cap concurrent workers to limit memmap / float64 peak RAM.
    workers = min(workers, 4) if workers > 1 else workers
    computed: list[int] = []

    def _emit(fraction: float, message: str) -> None:
        if progress is None:
            return
        clamped = max(0.0, min(1.0, float(fraction)))
        progress(
            ProgressEvent(
                recording=display_label,
                stage="channel",
                stage_label="Computing selected channels",
                stage_fraction=clamped,
                overall_fraction=clamped,
                message=message,
                elapsed_s=time.perf_counter() - started,
            )
        )

    def _one(ch: int) -> tuple[int, ChannelData]:
        check_analysis_cancelled()
        existing = recording._channel_data.get(ch)  # noqa: SLF001
        return ch, _compute_channel_data(
            recording,
            source,
            config,
            ch,
            t_rel,
            triggers,
            pre_n,
            post_n,
            need_means=need_means,
            need_rms=need_rms,
            need_spikes=need_spikes,
            need_overlay=need_overlay,
            existing=existing,
        )

    total = max(1, len(wanted))
    done = 0
    _emit(0.0, f"Computing {len(wanted)} channel(s)…")
    if workers <= 1 or len(wanted) == 1:
        for ch in wanted:
            index, data = _one(ch)
            recording.store_channel(index, data)
            computed.append(index)
            done += 1
            _emit(done / total, f"Channel {done}/{total}: {recording.channel_names[index]}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, data in pool.map(_one, wanted):
                recording.store_channel(index, data)
                computed.append(index)
                done += 1
                _emit(done / total, f"Channel {done}/{total}: {recording.channel_names[index]}")

    _emit(1.0, f"Computed {len(computed)} channel(s)")
    return computed


def _mean_one_channel(
    row: np.ndarray, triggers: np.ndarray, pre_n: int, post_n: int
) -> np.ndarray:
    win = int(pre_n + post_n)
    if triggers.size == 0 or win <= 0:
        return np.zeros(max(win, 0), dtype=np.float32)
    acc = np.zeros(win, dtype=np.float64)
    for trig in triggers:
        start = int(trig) - pre_n
        end = int(trig) + post_n
        acc += np.asarray(row[start:end], dtype=np.float64)
    return np.asarray(acc / float(triggers.size), dtype=np.float32)


def _compute_channel_data(
    recording: ProcessedRecording,
    source: AmplifierSpikeSource,
    config: AnalysisConfig,
    ch: int,
    t_rel: np.ndarray,
    triggers: np.ndarray,
    pre_n: int,
    post_n: int,
    *,
    need_means: bool = True,
    need_rms: bool = True,
    need_spikes: bool = True,
    need_overlay: bool = True,
    existing: ChannelData | None = None,
) -> ChannelData:
    from draw_primitives import _extract_spike_waveforms
    from plotting import (
        _mean_rms_profile_from_source_window,
        _resolve_channel_spike_threshold,
    )

    # Filter HP/LP once — shared by means, spikes and overlay.
    need_hp = need_means or need_spikes or need_overlay or need_rms
    need_lp = need_means
    raw_row = np.asarray(source.amplifier[ch], dtype=np.float32) if need_means else None
    hp_row = np.asarray(source.highpass[ch], dtype=np.float32) if need_hp else None
    lp_row = np.asarray(source.lowpass[ch], dtype=np.float32) if need_lp else None

    if need_means and raw_row is not None and hp_row is not None and lp_row is not None:
        means = {
            "raw": _mean_one_channel(raw_row, triggers, pre_n, post_n),
            "hp": _mean_one_channel(hp_row, triggers, pre_n, post_n),
            "lp": _mean_one_channel(lp_row, triggers, pre_n, post_n),
        }
    elif existing is not None and existing.means:
        means = dict(existing.means)
    else:
        means = {}

    rms_profiles: dict[str, np.ndarray] = {}
    rms_time = np.empty(0, dtype=np.float64)
    if need_rms:
        t0 = float(t_rel[0]) if t_rel.size else 0.0
        t1 = float(t_rel[-1]) if t_rel.size else 0.0
        rms_kinds = {"mean": None, "first": 0, "second": 1}
        for kind, trigger_index in rms_kinds.items():
            axis, values = _mean_rms_profile_from_source_window(
                source,
                t0,
                t1,
                float(config.rms_window_s),
                channel_index=ch,
                trigger_index=trigger_index,
            )
            if axis.size and rms_time.size == 0:
                rms_time = np.asarray(axis, dtype=np.float64)
            rms_profiles[kind] = np.asarray(values, dtype=np.float32)
    elif existing is not None:
        rms_profiles = dict(existing.rms_profiles)
        rms_time = np.asarray(existing.rms_time, dtype=np.float64)

    if need_spikes or need_overlay or need_rms:
        channel_rms = float(source.mean_rms_for_channel(ch))
        threshold, caption = _resolve_channel_spike_threshold(
            mode=config.spike_threshold_mode,
            fixed_threshold_uv=config.spike_threshold_uv,
            spike_threshold_polarity=config.spike_threshold_polarity,
            rms_multiplier=config.spike_threshold_rms_multiplier,
            source=source,
            channel_index=ch,
            mean_rms_uv=channel_rms,
        )
    elif existing is not None:
        channel_rms = float(existing.channel_rms_uv)
        threshold = float(existing.threshold_uv)
        caption = existing.threshold_caption
    else:
        channel_rms = 0.0
        threshold = float(config.spike_threshold_uv)
        caption = ""

    if need_spikes or need_overlay:
        trains = source.spike_times_per_trial_for_channel(ch, t_rel, float(threshold))
    elif existing is not None:
        trains = list(existing.spike_trains)
    else:
        trains = []

    if need_overlay:
        t_ms, waves, times = _extract_spike_waveforms(
            source,
            ch,
            trains,
            pre_ms=float(config.spike_overlay_pre_ms),
            post_ms=float(config.spike_overlay_post_ms),
        )
        waves = np.asarray(waves, dtype=np.float32)
        times = np.asarray(times, dtype=np.float32)
        total = int(waves.shape[0])
        overlay_mean = (
            np.mean(np.asarray(waves, dtype=np.float64), axis=0).astype(np.float32)
            if total
            else None
        )
        if total > MAX_OVERLAY_SNIPPETS_PER_CHANNEL:
            picks = np.linspace(0, total - 1, MAX_OVERLAY_SNIPPETS_PER_CHANNEL).astype(np.int64)
            waves = waves[picks]
            times = times[picks]
        overlay_t_ms = np.asarray(t_ms, dtype=np.float64)
    elif existing is not None:
        overlay_t_ms = np.asarray(existing.overlay_t_ms, dtype=np.float64)
        waves = np.asarray(existing.overlay_waves, dtype=np.float32)
        times = np.asarray(existing.overlay_times, dtype=np.float32)
        total = int(existing.overlay_total)
        overlay_mean = existing.overlay_mean
    else:
        overlay_t_ms = np.empty(0, dtype=np.float64)
        waves = np.empty((0, 0), dtype=np.float32)
        times = np.empty(0, dtype=np.float32)
        total = 0
        overlay_mean = None

    del recording  # only used for typing / future hooks
    return ChannelData(
        means=means,
        rms_time=rms_time if rms_time.size else np.asarray(t_rel, dtype=np.float64),
        rms_profiles=rms_profiles,
        channel_rms_uv=channel_rms,
        threshold_uv=float(threshold),
        threshold_caption=caption,
        spike_trains=[np.asarray(trial, dtype=np.float64) for trial in trains],
        overlay_t_ms=overlay_t_ms,
        overlay_waves=waves,
        overlay_times=times,
        overlay_total=total,
        overlay_mean=overlay_mean,
        overlay_ready=bool(need_overlay or (existing is not None and existing.overlay_ready)),
        rms_ready=bool(need_rms or (existing is not None and existing.rms_ready)),
        spikes_ready=bool(need_spikes or need_overlay or (existing is not None and existing.spikes_ready)),
        means_ready=bool(need_means or (existing is not None and existing.means_ready)),
    )


def open_dataset(
    path: Path,
    *,
    label: str | None = None,
    style: RecordingStyle | None = None,
) -> ProcessedRecording:
    """Open a processed dataset bundle (directory or ``.zip`` archive)."""
    target = Path(path)
    if target.is_file() and target.suffix.lower() == ".zip":
        from processed_dataset import extract_bundle

        destination = target.with_suffix("")
        target = extract_bundle(target, destination)
    if target.is_file() and target.name == "manifest.json":
        target = target.parent
    bundle = BundleLayout(root=target)
    if not bundle.exists():
        raise FileNotFoundError(f"No manifest.json in {target}")
    return read_bundle(bundle, label=label, style=style)


def export_dataset(
    recording: ProcessedRecording,
    destination: Path,
    *,
    include_streams: bool = False,
    include_trigger_windows: bool = True,
    include_overlay: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Write a self-contained processed dataset that reopens without the RHS."""
    from processed_dataset import write_bundle

    bundle = BundleLayout(root=Path(destination))
    if progress is not None:
        progress(f"Exporting processed dataset to {bundle.root}...")
    trigger_windows = recording.derived.trigger_windows
    if include_trigger_windows and not trigger_windows and recording.source is not None:
        hp = getattr(recording.source, "highpass", None)
        if isinstance(hp, LazyFilterBank):
            if progress is not None:
                progress(
                    "Skipping stimulation-window export — channels are computed on demand."
                )
        else:
            if progress is not None:
                progress("Extracting first / second stimulation windows...")
            recording.derived.trigger_windows = _trigger_windows(
                {
                    "raw": recording.source.amplifier,
                    "hp": recording.source.highpass,
                    "lp": recording.source.lowpass,
                },
                np.asarray(recording.derived.triggers, dtype=np.int64),
                recording.meta.segmentation,
                int(recording.derived.t_rel.size),
            )
    stream_paths: dict[str, Path] = {}
    if include_streams and recording.source is not None and recording.bundle_root is not None:
        source_bundle = BundleLayout(root=recording.bundle_root)
        manifest = source_bundle.read_manifest() if source_bundle.exists() else {}
        declared = manifest.get("streams") or {}
        for name in STREAM_NAMES:
            raw_path = declared.get(name)
            if not raw_path:
                continue
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = source_bundle.root / candidate
            if candidate.exists():
                stream_paths[name] = candidate
    root = write_bundle(
        recording,
        bundle,
        include_streams=bool(stream_paths),
        include_trigger_windows=include_trigger_windows,
        include_overlay=include_overlay,
        stream_paths=stream_paths or None,
    )
    if progress is not None:
        progress(f"Processed dataset written: {root}")
    return root


def describe_spike_settings(config: AnalysisConfig) -> str:
    params = spike_params(config)
    overlay = overlay_params(config)
    return (
        f"{params['mode']} / {params['polarity']} / {params['threshold_uv']:g} µV, "
        f"overlay [-{overlay['pre_ms']:g}, +{overlay['post_ms']:g}] ms"
    )
