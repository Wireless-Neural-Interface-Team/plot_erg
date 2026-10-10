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
import threading
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
    resolve_channel_workers,
    resolve_recording_windows,
    uses_analog_trigger,
)
from display_config import RecordingStyle
from erg_cache import (
    BundleLayout,
    CacheKeys,
    FilteredStreamLayout,
    bundle_for,
    default_cache_root,
    filtered_layout_for,
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
        # Legacy caches written with edge="none" may lack analog_in0 — fall through
        # to a full RHS read when a trigger mode needs it.
        if analog.size > 0 or not uses_analog_trigger(config):
            progress.finish_stage(
                cached=True,
                message=f"Reusing cached wideband stream ({meta.get('n_channels', '?')} channels)",
            )
            return meta, amplifier, analog
        del amplifier
        progress.emit(0.02, "Cached wideband lacks ANALOG_IN 0 — re-reading RHS…")

    if not config.rhs_file.exists():
        raise FileNotFoundError(f"File not found: {config.rhs_file}")
    progress.emit(0.05, f"Reading {config.rhs_file.name} (stream → memmap)...")
    layout.ensure()

    # Peek header for dimensions, then stream blocks straight into the cache file.
    from intanutil.header import read_header
    from intanutil.data import calculate_data_size
    from load_intan_rhs_format import read_data_to_amplifier_memmap

    with open(config.rhs_file, "rb") as fid:
        header = read_header(fid)
        data_present, _filesize, _num_blocks, num_samples = calculate_data_size(
            header, str(config.rhs_file), fid
        )
    if not data_present:
        raise RuntimeError(f"RHS file has no data blocks: {config.rhs_file}")
    n_channels = int(header["num_amplifier_channels"])
    n_samples = int(num_samples)
    if n_channels <= 0 or n_samples <= 0:
        raise RuntimeError("RHS file does not contain amplifier_data.")

    amp_mm = open_writable_memmap(
        layout.amplifier_path, (n_channels, n_samples), np.dtype(np.float32)
    )
    try:

        def _on_read_progress(fraction: float) -> None:
            check_analysis_cancelled()
            progress.emit(
                0.05 + 0.85 * float(fraction),
                f"Streaming RHS → cache ({int(fraction * 100)}%)",
            )

        data = read_data_to_amplifier_memmap(
            str(config.rhs_file),
            amp_mm,
            progress_callback=_on_read_progress,
        )
        amp_mm.flush()
    finally:
        del amp_mm

    check_analysis_cancelled()
    fs = get_sampling_rate(data)
    try:
        analog = np.asarray(get_analog_in0_signal(data), dtype=np.float64)
    except RuntimeError:
        analog = np.empty(0, dtype=np.float64)
        if uses_analog_trigger(config):
            raise
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
    if analog.size:
        np.save(layout.analog_in0_path, np.asarray(analog, dtype=np.float32))
    layout.write_meta(meta)

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
# are evicted; disk memmap (when configured) remains the source of truth.
LAZY_FILTER_CACHE_MAX_CHANNELS = 32


class _FilteredDiskStore:
    """Shared HP/LP/notch memmaps keyed by the filter content hash."""

    def __init__(
        self,
        layout: FilteredStreamLayout,
        n_channels: int,
        n_samples: int,
        *,
        with_notch: bool,
    ) -> None:
        self.layout = layout
        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.with_notch = bool(with_notch)
        layout.ensure()
        shape = (self.n_channels, self.n_samples)
        mask_shape = (self.n_channels,)
        if not layout.is_initialized():
            for path in (layout.hp_path, layout.lp_path):
                mm = open_writable_memmap(path, shape, np.dtype(np.float32))
                del mm
            np.save(layout.ready_hp_path, np.zeros(mask_shape, dtype=np.bool_))
            np.save(layout.ready_lp_path, np.zeros(mask_shape, dtype=np.bool_))
            if with_notch:
                mm = open_writable_memmap(layout.notch_path, shape, np.dtype(np.float32))
                del mm
                np.save(layout.ready_notch_path, np.zeros(mask_shape, dtype=np.bool_))
            layout.write_meta(
                {
                    "n_channels": self.n_channels,
                    "n_samples": self.n_samples,
                    "with_notch": self.with_notch,
                }
            )
        # r+ so we can fill individual channels as they are computed.
        self.hp = np.load(layout.hp_path, mmap_mode="r+")
        self.lp = np.load(layout.lp_path, mmap_mode="r+")
        self.ready_hp = np.load(layout.ready_hp_path, mmap_mode="r+")
        self.ready_lp = np.load(layout.ready_lp_path, mmap_mode="r+")
        if with_notch and layout.notch_path.exists():
            self.notch = np.load(layout.notch_path, mmap_mode="r+")
            self.ready_notch = np.load(layout.ready_notch_path, mmap_mode="r+")
        else:
            self.notch = None
            self.ready_notch = None
        # Protect concurrent channel fills from ensure_channels workers.
        self._lock = threading.Lock()
        self._closed = False

    def close(self) -> None:
        """Release memmap handles so the cache directory can be deleted on Windows."""
        if self._closed:
            return
        self._closed = True
        for attr in ("hp", "lp", "ready_hp", "ready_lp", "notch", "ready_notch"):
            arr = getattr(self, attr, None)
            setattr(self, attr, None)
            if arr is None:
                continue
            try:
                mmap = getattr(arr, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            except Exception:
                pass

    def is_ready(self, kind: str, ch: int) -> bool:
        with self._lock:
            if kind == "hp":
                return bool(self.ready_hp[ch])
            if kind == "lp":
                return bool(self.ready_lp[ch])
            if kind == "notch":
                if self.notch is None or self.ready_notch is None:
                    return False
                return bool(self.ready_notch[ch])
            raise ValueError(kind)

    def read_ready(self, kind: str, ch: int, *, copy: bool = False) -> np.ndarray | None:
        """Return a ready row, or None.

        By default returns a memmap view (float32 for hp/lp). Pass ``copy=True``
        when the caller will retain the array outside the store lock / LRU.
        """
        with self._lock:
            if kind == "hp":
                if not bool(self.ready_hp[ch]):
                    return None
                row = self.hp[ch]
                return np.array(row, dtype=np.float32, copy=True) if copy else row
            if kind == "lp":
                if not bool(self.ready_lp[ch]):
                    return None
                row = self.lp[ch]
                return np.array(row, dtype=np.float32, copy=True) if copy else row
            if kind == "notch":
                if self.notch is None or self.ready_notch is None:
                    return None
                if not bool(self.ready_notch[ch]):
                    return None
                row = self.notch[ch]
                return np.asarray(row, dtype=np.float64) if copy else row
            raise ValueError(kind)

    def write(self, kind: str, ch: int, row: np.ndarray, *, flush: bool = False) -> None:
        with self._lock:
            if kind == "hp":
                self.hp[ch] = np.asarray(row, dtype=np.float32)
                self.ready_hp[ch] = True
                if flush:
                    self.hp.flush()
                    self.ready_hp.flush()
            elif kind == "lp":
                self.lp[ch] = np.asarray(row, dtype=np.float32)
                self.ready_lp[ch] = True
                if flush:
                    self.lp.flush()
                    self.ready_lp.flush()
            elif kind == "notch":
                if self.notch is None or self.ready_notch is None:
                    return
                self.notch[ch] = np.asarray(row, dtype=np.float32)
                self.ready_notch[ch] = True
                if flush:
                    self.notch.flush()
                    self.ready_notch.flush()
            else:
                raise ValueError(kind)

    def flush(self) -> None:
        """Flush all memmaps (call after a batch of writes)."""
        with self._lock:
            for arr in (self.hp, self.lp, self.ready_hp, self.ready_lp, self.notch, self.ready_notch):
                if arr is None:
                    continue
                try:
                    arr.flush()
                except Exception:
                    pass


class LazyFilterBank:
    """Filters individual amplifier channels on first access and caches the row.

    Lookup order: RAM LRU → disk memmap (if configured) → ``sosfilt``.
    When ``notch_cache`` is shared between the HP and LP banks, the notch stage
    runs once per channel and both filters reuse the notched buffer.
    """

    def __init__(
        self,
        amplifier: np.ndarray,
        dsp: IntanDspSettings,
        kind: str,
        *,
        max_cached_channels: int = LAZY_FILTER_CACHE_MAX_CHANNELS,
        notch_cache: OrderedDict[int, np.ndarray] | None = None,
        filter_sos: np.ndarray | None = None,
        notch_sos: np.ndarray | None = None,
        disk_store: _FilteredDiskStore | None = None,
        shared_lock: threading.Lock | None = None,
    ) -> None:
        if kind not in ("hp", "lp"):
            raise ValueError(f"Unsupported filter bank kind: {kind}")
        self._amplifier = amplifier
        self._dsp = dsp
        self._kind = kind
        self._max_cached = max(1, int(max_cached_channels))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._notch_cache = notch_cache
        self._disk = disk_store
        self.shape = (int(amplifier.shape[0]), int(amplifier.shape[1]))
        if notch_sos is None or filter_sos is None:
            built_notch, hp_sos, lp_sos = build_intan_filter_sos(dsp)
            notch_sos = built_notch if notch_sos is None else notch_sos
            filter_sos = (hp_sos if kind == "hp" else lp_sos) if filter_sos is None else filter_sos
        self._notch_sos = notch_sos
        self._sos = filter_sos
        # Shared with the sibling HP/LP bank when they share ``notch_cache``.
        self._lock = shared_lock if shared_lock is not None else threading.Lock()

    def close(self) -> None:
        """Drop RAM rows and release the shared disk store (idempotent)."""
        self._cache.clear()
        if self._notch_cache is not None:
            self._notch_cache.clear()
        store = self._disk
        self._disk = None
        if store is not None:
            store.close()

    def __getitem__(self, index: int | tuple[Any, ...]) -> np.ndarray:
        # Support ``bank[ch, start:end]`` like a 2D ndarray (filters the row, then slices).
        if isinstance(index, tuple):
            if not index:
                raise IndexError("Empty index")
            row = self._row(int(index[0]), flush_disk=True)
            return row[index[1:]] if len(index) > 1 else row
        return self._row(int(index), flush_disk=True)

    def _row(self, ch: int, *, flush_disk: bool) -> np.ndarray:
        with self._lock:
            cached = self._cache.get(ch)
            if cached is not None:
                self._cache.move_to_end(ch)
                return cached

        if self._disk is not None:
            # Copy into LRU so callers keep a stable buffer after later writes.
            from_disk = self._disk.read_ready(self._kind, ch, copy=True)
            if from_disk is not None:
                with self._lock:
                    self._cache[ch] = from_disk
                    while len(self._cache) > self._max_cached:
                        self._cache.popitem(last=False)
                return from_disk

        from scipy.signal import sosfilt

        check_analysis_cancelled()
        signal = self._notched_signal(ch, sosfilt)
        filtered = np.asarray(sosfilt(self._sos, signal), dtype=np.float32)
        if self._disk is not None:
            self._disk.write(self._kind, ch, filtered, flush=False)
            if flush_disk:
                self._disk.flush()
        with self._lock:
            self._cache[ch] = filtered
            while len(self._cache) > self._max_cached:
                self._cache.popitem(last=False)
        return filtered

    def _notched_signal(self, ch: int, sosfilt: Any) -> np.ndarray:
        if self._notch_sos is None:
            return np.asarray(self._amplifier[ch], dtype=np.float64)

        if self._notch_cache is not None:
            with self._lock:
                signal = self._notch_cache.get(ch)
                if signal is not None:
                    self._notch_cache.move_to_end(ch)
                    return signal

        if self._disk is not None:
            from_disk = self._disk.read_ready("notch", ch, copy=True)
            if from_disk is not None:
                if self._notch_cache is not None:
                    with self._lock:
                        self._notch_cache[ch] = from_disk
                        while len(self._notch_cache) > self._max_cached:
                            self._notch_cache.popitem(last=False)
                return from_disk

        wide = np.asarray(self._amplifier[ch], dtype=np.float64)
        signal = sosfilt(self._notch_sos, wide)
        if self._disk is not None:
            self._disk.write("notch", ch, signal, flush=False)
        if self._notch_cache is not None:
            with self._lock:
                self._notch_cache[ch] = signal
                while len(self._notch_cache) > self._max_cached:
                    self._notch_cache.popitem(last=False)
        return signal

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def is_ready(self, ch: int) -> bool:
        """True when the row is already in RAM or on disk (no sosfilt)."""
        ch = int(ch)
        with self._lock:
            if ch in self._cache:
                return True
        if self._disk is not None:
            return self._disk.is_ready(self._kind, ch)
        return False

    def prefetch(self, channels: Sequence[int], *, flush_batch: bool = True) -> None:
        """Warm RAM (and disk) cache for the given channel indices."""
        for ch in channels:
            self._row(int(ch), flush_disk=False)
        if flush_batch and self._disk is not None:
            self._disk.flush()


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

    notch_sos, hp_sos, lp_sos = build_intan_filter_sos(dsp)
    notch_cache: OrderedDict[int, np.ndarray] | None = (
        OrderedDict() if notch_sos is not None else None
    )
    filter_layout = filtered_layout_for(root, keys)
    disk_store = _FilteredDiskStore(
        filter_layout,
        n_channels,
        int(raw_meta["n_samples"]),
        with_notch=notch_sos is not None,
    )
    filter_lock = threading.Lock()
    high = LazyFilterBank(
        amplifier,
        dsp,
        "hp",
        notch_cache=notch_cache,
        filter_sos=hp_sos,
        notch_sos=notch_sos,
        disk_store=disk_store,
        shared_lock=filter_lock,
    )
    low = LazyFilterBank(
        amplifier,
        dsp,
        "lp",
        notch_cache=notch_cache,
        filter_sos=lp_sos,
        notch_sos=notch_sos,
        disk_store=disk_store,
        shared_lock=filter_lock,
    )
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
    from channel_metrics import mean_trial_windows

    return mean_trial_windows(row, triggers, pre_n, post_n)


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
    from channel_metrics import (
        mean_rms_profile_from_source_window,
        resolve_channel_spike_threshold,
    )
    from draw_primitives import _extract_spike_waveforms
    from intan_rhx_dsp import mean_rms_intan_channel

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
    if need_rms and hp_row is not None:
        t0 = float(t_rel[0]) if t_rel.size else 0.0
        t1 = float(t_rel[-1]) if t_rel.size else 0.0
        rms_kinds = {"mean": None, "first": 0, "second": 1}
        for kind, trigger_index in rms_kinds.items():
            axis, values = mean_rms_profile_from_source_window(
                source,
                t0,
                t1,
                float(config.rms_window_s),
                channel_index=ch,
                trigger_index=trigger_index,
                hp_row=hp_row,
            )
            if axis.size and rms_time.size == 0:
                rms_time = np.asarray(axis, dtype=np.float64)
            rms_profiles[kind] = np.asarray(values, dtype=np.float32)
    elif existing is not None:
        rms_profiles = dict(existing.rms_profiles)
        rms_time = np.asarray(existing.rms_time, dtype=np.float64)

    if need_spikes or need_overlay or need_rms:
        if hp_row is not None:
            channel_rms = float(mean_rms_intan_channel(hp_row, source.intan_dsp))
        else:
            channel_rms = float(source.mean_rms_for_channel(ch))
        threshold, caption = resolve_channel_spike_threshold(
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
    if include_trigger_windows and not recording.derived.trigger_windows:
        if progress is not None:
            progress("Extracting first / second stimulation windows...")
        # Prefer bulk extract when full stacks are available; otherwise
        # ``write_bundle`` materializes per-channel slices from the live source.
        hp = getattr(recording.source, "highpass", None) if recording.source else None
        if recording.source is not None and not isinstance(hp, LazyFilterBank):
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

