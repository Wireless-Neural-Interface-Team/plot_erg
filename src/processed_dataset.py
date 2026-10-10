"""Recording datasets for the interactive viewer.

A :class:`ProcessedRecording` starts as a light skeleton (wideband stream +
segmentation). Per-channel products are filled on demand via
:class:`ChannelData` when the GUI requests one or more channels:

- trial-averaged traces (raw / high-pass / low-pass),
- sliding-RMS profiles,
- spike trains, thresholds and waveform snippets.

Exported ``.ergproc`` bundles remain supported for fully materialised datasets.
"""

from __future__ import annotations

import datetime as _dt
import gc
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from display_config import RecordingStyle, resolve_display_label
from erg_cache import (
    BundleLayout,
    CACHE_FORMAT_VERSION,
    CacheKeys,
    DATASET_SUFFIX,
)
from impedance_tracking import ImpedanceSession
from intan_rhx_dsp import IntanDspSettings

STREAM_NAMES: tuple[str, ...] = ("raw", "hp", "lp")
RMS_KINDS: tuple[str, ...] = ("mean", "first", "second")

# Spike snippets kept per channel for the overlay panel (the exact mean waveform
# over *all* detections is stored separately, so the overlay stays faithful).
MAX_OVERLAY_SNIPPETS_PER_CHANNEL = 3000


def _load_memmap(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return None


def _load_array(path: Path) -> np.ndarray | None:
    """Read a small array fully into RAM.

    Used for the 1-D arrays (time bases, per-channel scalars, spike times): they
    cost little memory and, unlike a memmap, they keep no file handle on the
    bundle, so the cache stays deletable while a recording is open.
    """
    if not path.exists():
        return None
    try:
        return np.load(path)
    except (OSError, ValueError):
        return None


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array))


def _row(array: np.ndarray | None, index: int) -> np.ndarray | None:
    """One channel row, always copied out of the memmap (never a live view)."""
    if array is None:
        return None
    if index < 0 or index >= int(array.shape[0]):
        return None
    return np.array(array[index], dtype=np.float64)


@dataclass(frozen=True)
class SegmentationInfo:
    """How the recording was cut into trials."""

    mode: str  # "falling" | "rising" | "none"
    n_trials: int
    n_triggers_detected: int
    pre_n: int
    post_n: int
    pre_s: float
    post_s: float
    end_rising_s: float | None
    section_count: int | None = None
    section_duration_s: float | None = None

    def describe(self) -> str:
        if self.mode == "none":
            return (
                f"{self.n_trials} fixed section(s)"
                + (
                    f" of {self.section_duration_s:.3f} s"
                    if self.section_duration_s
                    else ""
                )
            )
        excluded = max(0, int(self.n_triggers_detected) - int(self.n_trials))
        text = (
            f"{self.n_trials} stimulation(s) on {self.mode} edge "
            f"(window [-{self.pre_s:g}, +{self.post_s:g}] s)"
        )
        if excluded:
            text += f", {excluded} excluded (window outside signal)"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_trials": int(self.n_trials),
            "n_triggers_detected": int(self.n_triggers_detected),
            "pre_n": int(self.pre_n),
            "post_n": int(self.post_n),
            "pre_s": float(self.pre_s),
            "post_s": float(self.post_s),
            "end_rising_s": None if self.end_rising_s is None else float(self.end_rising_s),
            "section_count": self.section_count,
            "section_duration_s": self.section_duration_s,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SegmentationInfo:
        return cls(
            mode=str(raw.get("mode", "falling")),
            n_trials=int(raw.get("n_trials", 0)),
            n_triggers_detected=int(raw.get("n_triggers_detected", 0)),
            pre_n=int(raw.get("pre_n", 0)),
            post_n=int(raw.get("post_n", 0)),
            pre_s=float(raw.get("pre_s", 0.0)),
            post_s=float(raw.get("post_s", 0.0)),
            end_rising_s=(
                None if raw.get("end_rising_s") is None else float(raw["end_rising_s"])
            ),
            section_count=raw.get("section_count"),
            section_duration_s=raw.get("section_duration_s"),
        )


@dataclass(frozen=True)
class SpikeDetectionInfo:
    """Spike detection settings actually used to build the stored spike trains."""

    mode: str
    threshold_uv: float
    polarity: str
    rms_multiplier: float
    artifact_threshold_uv: float
    artifact_suppression: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "threshold_uv": float(self.threshold_uv),
            "polarity": self.polarity,
            "rms_multiplier": float(self.rms_multiplier),
            "artifact_threshold_uv": float(self.artifact_threshold_uv),
            "artifact_suppression": bool(self.artifact_suppression),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SpikeDetectionInfo:
        return cls(
            mode=str(raw.get("mode", "fixed")),
            threshold_uv=float(raw.get("threshold_uv", 70.0)),
            polarity=str(raw.get("polarity", "negative")),
            rms_multiplier=float(raw.get("rms_multiplier", 4.0)),
            artifact_threshold_uv=float(raw.get("artifact_threshold_uv", 2500.0)),
            artifact_suppression=bool(raw.get("artifact_suppression", True)),
        )


@dataclass(frozen=True)
class RecordingMeta:
    """Identity, geometry and processing parameters of one processed recording."""

    source_path: Path
    source_name: str
    fs: float
    n_channels: int
    n_samples: int
    channel_names: tuple[str, ...]
    segmentation: SegmentationInfo
    spike_detection: SpikeDetectionInfo
    dsp: IntanDspSettings
    cache_keys: dict[str, str]
    created_at: str
    overlay_pre_ms: float
    overlay_post_ms: float

    @property
    def window_samples(self) -> int:
        return int(self.segmentation.pre_n + self.segmentation.post_n)

    @property
    def duration_s(self) -> float:
        return float(self.n_samples) / float(self.fs) if self.fs > 0 else 0.0

    def filter_short_label(self, kind: str = "highpass") -> str:
        return self.dsp.filter_short_label(kind)  # type: ignore[arg-type]

    def filter_title_label(self, kind: str = "highpass") -> str:
        return self.dsp.filter_title_label(kind)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "source_path": str(self.source_path),
            "source_name": self.source_name,
            "fs": float(self.fs),
            "n_channels": int(self.n_channels),
            "n_samples": int(self.n_samples),
            "n_trials": int(self.segmentation.n_trials),
            "channel_names": list(self.channel_names),
            "segmentation": self.segmentation.to_dict(),
            "spike_detection": self.spike_detection.to_dict(),
            "dsp": asdict(self.dsp),
            "cache_keys": dict(self.cache_keys),
            "created_at": self.created_at,
            "overlay_pre_ms": float(self.overlay_pre_ms),
            "overlay_post_ms": float(self.overlay_post_ms),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RecordingMeta:
        dsp = IntanDspSettings.from_dict(dict(raw.get("dsp", {})))
        return cls(
            source_path=Path(str(raw.get("source_path", ""))),
            source_name=str(raw.get("source_name", "")),
            fs=float(raw.get("fs", 0.0)),
            n_channels=int(raw.get("n_channels", 0)),
            n_samples=int(raw.get("n_samples", 0)),
            channel_names=tuple(str(c) for c in raw.get("channel_names", ())),
            segmentation=SegmentationInfo.from_dict(raw.get("segmentation", {})),
            spike_detection=SpikeDetectionInfo.from_dict(raw.get("spike_detection", {})),
            dsp=dsp,
            cache_keys=dict(raw.get("cache_keys", {})),
            created_at=str(raw.get("created_at", "")),
            overlay_pre_ms=float(raw.get("overlay_pre_ms", 2.0)),
            overlay_post_ms=float(raw.get("overlay_post_ms", 4.0)),
        )


class SpikeTrains:
    """Ragged per-(channel, trial) spike times, stored as three flat arrays."""

    def __init__(self, counts: np.ndarray, times: np.ndarray) -> None:
        self.counts = np.asarray(counts, dtype=np.int32)
        self.times = np.asarray(times, dtype=np.float32)
        flat = self.counts.reshape(-1).astype(np.int64)
        self._offsets = np.concatenate(([0], np.cumsum(flat)))

    @property
    def n_channels(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_trials(self) -> int:
        return int(self.counts.shape[1]) if self.counts.ndim == 2 else 0

    @classmethod
    def from_lists(cls, per_channel: Sequence[Sequence[np.ndarray]], n_trials: int) -> SpikeTrains:
        n_ch = len(per_channel)
        counts = np.zeros((n_ch, max(int(n_trials), 0)), dtype=np.int32)
        chunks: list[np.ndarray] = []
        for ch, trials in enumerate(per_channel):
            for trial in range(counts.shape[1]):
                arr = (
                    np.asarray(trials[trial], dtype=np.float32).ravel()
                    if trial < len(trials)
                    else np.empty(0, dtype=np.float32)
                )
                counts[ch, trial] = int(arr.size)
                if arr.size:
                    chunks.append(arr)
        times = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
        return cls(counts, times)

    @classmethod
    def empty(cls, n_channels: int, n_trials: int) -> SpikeTrains:
        return cls(
            np.zeros((max(n_channels, 0), max(n_trials, 0)), dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    def per_trial(self, ch: int) -> list[np.ndarray]:
        """Spike times (s, relative to stimulation) for every trial of ``ch``."""
        if ch < 0 or ch >= self.n_channels:
            return []
        out: list[np.ndarray] = []
        base = ch * self.n_trials
        for trial in range(self.n_trials):
            start = int(self._offsets[base + trial])
            end = int(self._offsets[base + trial + 1])
            out.append(np.asarray(self.times[start:end], dtype=np.float64))
        return out

    def total_for_channel(self, ch: int) -> int:
        if ch < 0 or ch >= self.n_channels:
            return 0
        return int(np.sum(self.counts[ch]))

    def save(self, directory: Path) -> None:
        _save_array(directory / "counts.npy", self.counts)
        _save_array(directory / "times.npy", self.times)

    @classmethod
    def load(cls, directory: Path) -> SpikeTrains | None:
        counts = _load_array(directory / "counts.npy")
        times = _load_array(directory / "times.npy")
        if counts is None or times is None:
            return None
        return cls(counts, times)


class OverlaySnippets:
    """Spike waveform snippets per channel, capped for display, plus exact means."""

    def __init__(
        self,
        t_ms: np.ndarray,
        counts: np.ndarray,
        totals: np.ndarray,
        waves: np.ndarray,
        t_rel: np.ndarray,
        means: np.ndarray,
    ) -> None:
        self.t_ms = np.asarray(t_ms, dtype=np.float64)
        self.counts = np.asarray(counts, dtype=np.int32)
        self.totals = np.asarray(totals, dtype=np.int32)
        self.waves = waves
        self.t_rel = t_rel
        self.means = means
        offsets = np.concatenate(([0], np.cumsum(self.counts.astype(np.int64))))
        self._offsets = offsets

    @property
    def n_channels(self) -> int:
        return int(self.counts.shape[0])

    @classmethod
    def empty(cls, n_channels: int, window: int) -> OverlaySnippets:
        return cls(
            np.zeros(max(window, 0), dtype=np.float64),
            np.zeros(max(n_channels, 0), dtype=np.int32),
            np.zeros(max(n_channels, 0), dtype=np.int32),
            np.empty((0, max(window, 0)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.zeros((max(n_channels, 0), max(window, 0)), dtype=np.float32),
        )

    def for_channel(self, ch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """``(t_ms, waveforms, spike_times_s, total_detections)`` for one channel."""
        if ch < 0 or ch >= self.n_channels:
            empty = np.empty((0, self.t_ms.size), dtype=np.float32)
            return self.t_ms, empty, np.empty(0, dtype=np.float64), 0
        start = int(self._offsets[ch])
        end = int(self._offsets[ch + 1])
        waves = np.array(self.waves[start:end])
        times = np.array(self.t_rel[start:end], dtype=np.float64)
        return self.t_ms, waves, times, int(self.totals[ch])

    def mean_for_channel(self, ch: int) -> np.ndarray | None:
        if ch < 0 or ch >= self.n_channels or int(self.totals[ch]) == 0:
            return None
        return np.array(self.means[ch], dtype=np.float64)

    def save(self, directory: Path) -> None:
        _save_array(directory / "t_ms.npy", self.t_ms)
        _save_array(directory / "counts.npy", self.counts)
        _save_array(directory / "totals.npy", self.totals)
        _save_array(directory / "waves.npy", np.asarray(self.waves, dtype=np.float32))
        _save_array(directory / "t_rel.npy", np.asarray(self.t_rel, dtype=np.float32))
        _save_array(directory / "means.npy", np.asarray(self.means, dtype=np.float32))

    @classmethod
    def load(cls, directory: Path) -> OverlaySnippets | None:
        parts: dict[str, np.ndarray | None] = {
            name: _load_array(directory / f"{name}.npy")
            for name in ("t_ms", "counts", "totals", "t_rel", "means")
        }
        # Only the snippet matrix is large enough to be worth memmapping.
        parts["waves"] = _load_memmap(directory / "waves.npy")
        if any(value is None for value in parts.values()):
            return None
        return cls(
            parts["t_ms"],  # type: ignore[arg-type]
            parts["counts"],  # type: ignore[arg-type]
            parts["totals"],  # type: ignore[arg-type]
            parts["waves"],
            parts["t_rel"],
            parts["means"],
        )


@dataclass
class DerivedArrays:
    """Memmapped per-channel products, all shaped ``(n_channels, window)``."""

    t_rel: np.ndarray
    triggers: np.ndarray
    means: dict[str, np.ndarray] = field(default_factory=dict)
    trigger_windows: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)
    rms_time: np.ndarray | None = None
    rms_profiles: dict[str, np.ndarray] = field(default_factory=dict)
    channel_rms_uv: np.ndarray | None = None
    thresholds_uv: np.ndarray | None = None

    @staticmethod
    def trigger_key(trigger_index: int, stream: str) -> str:
        return f"trig{int(trigger_index)}_{stream}"


@dataclass
class ChannelData:
    """Products computed on demand for a single channel."""

    means: dict[str, np.ndarray]
    rms_time: np.ndarray
    rms_profiles: dict[str, np.ndarray]
    channel_rms_uv: float
    threshold_uv: float
    threshold_caption: str
    spike_trains: list[np.ndarray]
    overlay_t_ms: np.ndarray
    overlay_waves: np.ndarray
    overlay_times: np.ndarray
    overlay_total: int
    overlay_mean: np.ndarray | None
    # False when spike overlay was skipped (lazy); True after a real extract pass.
    overlay_ready: bool = True
    rms_ready: bool = True
    spikes_ready: bool = True
    means_ready: bool = True


class ProcessedRecording:
    """One recording prepared for viewing; channels are filled on demand."""

    def __init__(
        self,
        meta: RecordingMeta,
        derived: DerivedArrays,
        spikes: SpikeTrains,
        overlay: OverlaySnippets,
        *,
        label: str,
        style: RecordingStyle | None = None,
        impedance_sessions: Sequence[ImpedanceSession] = (),
        source: Any | None = None,
        bundle_root: Path | None = None,
        threshold_captions: Sequence[str] = (),
        build_timings: Sequence[tuple[str, float]] = (),
    ) -> None:
        self.meta = meta
        self.derived = derived
        self.spikes = spikes
        self.overlay = overlay
        self.label = label
        self.style = style or RecordingStyle.visible()
        self.impedance_sessions = list(impedance_sessions)
        self.source = source
        self.bundle_root = bundle_root
        self.threshold_captions = list(threshold_captions)
        self.build_timings = list(build_timings)
        self._channel_data: dict[int, ChannelData] = {}
        self._data_generation = 0
        # Cache des enveloppes continues (stream, ch, max_points) → (t, y).
        self._continuous_cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}

    # ---------------------------------------------------------------- identity

    @property
    def n_channels(self) -> int:
        return int(self.meta.n_channels)

    @property
    def n_trials(self) -> int:
        return int(self.meta.segmentation.n_trials)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self.meta.channel_names

    @property
    def t_rel(self) -> np.ndarray:
        return np.asarray(self.derived.t_rel, dtype=np.float64)

    @property
    def end_marker_s(self) -> float | None:
        return self.meta.segmentation.end_rising_s

    @property
    def has_streams(self) -> bool:
        """True when the raw / filtered streams are still available."""
        return self.source is not None

    def channel_index(self, channel_name: str) -> int | None:
        for index, name in enumerate(self.channel_names):
            if name == channel_name:
                return index
        return None

    # --------------------------------------------------------- on-demand cache

    @property
    def ready_channels(self) -> set[int]:
        """Channel indices already computed for viewing."""
        if self._channel_data:
            return set(self._channel_data)
        # Fully precomputed datasets (export / synthetic) expose every channel.
        if self.derived.means:
            return set(range(self.n_channels))
        return set()

    def is_channel_ready(self, ch: int) -> bool:
        ch = int(ch)
        if ch in self._channel_data:
            return True
        if self.derived.means and 0 <= ch < self.n_channels:
            row = _row(next(iter(self.derived.means.values()), None), ch)
            return row is not None and np.any(np.isfinite(row))
        return False

    def store_channel(self, ch: int, data: ChannelData) -> None:
        self._channel_data[int(ch)] = data
        self._data_generation = int(getattr(self, "_data_generation", 0)) + 1
        while len(self.threshold_captions) <= int(ch):
            self.threshold_captions.append("")
        self.threshold_captions[int(ch)] = data.threshold_caption

    @property
    def data_generation(self) -> int:
        """Monotonic counter bumped when channel products change (redraw fingerprint)."""
        return int(getattr(self, "_data_generation", 0))

    def clear_channel_cache(self) -> None:
        self._channel_data.clear()
        self._data_generation = int(getattr(self, "_data_generation", 0)) + 1

    def spike_count(self, ch: int) -> int:
        data = self._channel_data.get(int(ch))
        if data is not None:
            return int(sum(int(np.asarray(trial).size) for trial in data.spike_trains))
        return int(self.spikes.total_for_channel(ch))

    def stream_ready(self, stream: str, ch: int) -> bool:
        """True when the continuous stream row can be read without filtering."""
        stream = str(stream)
        ch = int(ch)
        if stream == "raw":
            source = self.source
            amp = getattr(source, "amplifier", None) if source is not None else None
            return amp is not None and 0 <= ch < int(getattr(amp, "shape", (0,))[0])
        source = self.source
        if source is None:
            return False
        bank = {
            "hp": getattr(source, "highpass", None),
            "lp": getattr(source, "lowpass", None),
        }.get(stream)
        if bank is None:
            return False
        is_ready = getattr(bank, "is_ready", None)
        if callable(is_ready):
            return bool(is_ready(ch))
        # Non-lazy ndarray (e.g. precomputed stacks).
        return 0 <= ch < int(getattr(bank, "shape", (0,))[0])

    def continuous_trace(
        self,
        stream: str,
        ch: int,
        *,
        max_points: int | None = None,
        require_ready: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Absolute-time continuous trace ``(t_s, values_uv)`` for one channel.

        When ``max_points`` is set and the row is denser, returns a min/max
        envelope so callers avoid building a full-rate time axis.

        When ``require_ready`` is True and the HP/LP row is not cached yet,
        returns empty arrays instead of running ``sosfilt`` (UI-safe).
        """
        empty = np.empty(0, dtype=np.float64)
        limit = int(max_points) if max_points is not None else 0
        if limit <= 0 and max_points is None:
            # Guardrail: never build a full-rate axis unless explicitly requested
            # with max_points=0 from a worker / offline path.
            limit = 0
        cache_key = (str(stream), int(ch), limit, bool(require_ready))
        cached = self._continuous_cache.get(cache_key)
        if cached is not None:
            return cached
        source = self.source
        if source is None:
            return empty, empty
        if require_ready and str(stream) in {"hp", "lp"} and not self.stream_ready(stream, ch):
            return empty, empty
        array = {
            "raw": getattr(source, "amplifier", None),
            "hp": getattr(source, "highpass", None),
            "lp": getattr(source, "lowpass", None),
        }.get(stream)
        if array is None or ch < 0 or ch >= int(array.shape[0]):
            return empty, empty
        # Vue memmap (pas de copie tant qu’on reste en enveloppe float32).
        row = array[ch]
        n_samples = int(getattr(row, "shape", (0,))[0]) if row is not None else 0
        if n_samples <= 0:
            return empty, empty
        fs = float(self.meta.fs)
        if fs <= 0:
            return empty, empty
        if limit > 16 and n_samples > limit:
            # Enveloppe min/max vectorisée (pas de boucle Python par bin).
            n_bins = max(1, limit // 2)
            bin_size = max(1, n_samples // n_bins)
            usable = bin_size * n_bins
            reshaped = np.asarray(row[:usable], dtype=np.float32).reshape(n_bins, bin_size)
            lows = reshaped.min(axis=1)
            highs = reshaped.max(axis=1)
            mids = (np.arange(n_bins, dtype=np.float64) + 0.5) * (bin_size / fs)
            out_t = np.repeat(mids, 2)
            out_y = np.empty(n_bins * 2, dtype=np.float64)
            out_y[0::2] = lows
            out_y[1::2] = highs
            result = (out_t, out_y)
        else:
            data = np.asarray(row, dtype=np.float64)
            t = np.arange(data.size, dtype=np.float64) / fs
            result = (t, data)
        # Cap mémoire : garder les enveloppes les plus récentes.
        if len(self._continuous_cache) >= 512:
            self._continuous_cache.clear()
        self._continuous_cache[cache_key] = result
        return result

    def stimulation_times_s(self) -> np.ndarray:
        """Absolute stimulation onset times in seconds."""
        triggers = np.asarray(self.derived.triggers, dtype=np.float64)
        fs = float(self.meta.fs)
        if triggers.size == 0 or fs <= 0:
            return np.empty(0, dtype=np.float64)
        return triggers / fs

    # -------------------------------------------------------------- trace data

    def mean(self, stream: str, ch: int) -> np.ndarray | None:
        """Trial-averaged trace for one channel (µV)."""
        data = self._channel_data.get(int(ch))
        if data is not None:
            curve = data.means.get(stream)
            return None if curve is None else np.asarray(curve, dtype=np.float64)
        return _row(self.derived.means.get(stream), ch)

    def trigger_window(self, trigger_index: int, stream: str, ch: int) -> np.ndarray | None:
        """Single-stimulation window, from the cache or sliced from the streams."""
        key = (int(trigger_index), stream)
        cached = _row(self.derived.trigger_windows.get(key), ch)
        if cached is not None:
            return cached
        return self._slice_trigger_window(trigger_index, stream, ch)

    def _slice_trigger_window(
        self, trigger_index: int, stream: str, ch: int
    ) -> np.ndarray | None:
        source = self.source
        if source is None:
            return None
        triggers = np.asarray(self.derived.triggers, dtype=np.int64)
        if trigger_index < 0 or trigger_index >= int(triggers.size):
            return None
        array = {
            "raw": getattr(source, "amplifier", None),
            "hp": getattr(source, "highpass", None),
            "lp": getattr(source, "lowpass", None),
        }.get(stream)
        if array is None or ch < 0 or ch >= int(array.shape[0]):
            return None
        pre_n = int(self.meta.segmentation.pre_n)
        post_n = int(self.meta.segmentation.post_n)
        start = int(triggers[trigger_index]) - pre_n
        end = int(triggers[trigger_index]) + post_n
        n_samples = int(array.shape[1])
        if start < 0 or end > n_samples:
            return None
        # Prefer a direct 2D slice on memmaps/ndarrays. LazyFilterBank only
        # supports integer channel indexing, so fall back to row-then-slice.
        try:
            window = array[ch, start:end]
        except (IndexError, TypeError, ValueError):
            window = array[ch][start:end]
        return np.asarray(window, dtype=np.float64)

    def rms_profile(self, kind: str, ch: int) -> tuple[np.ndarray, np.ndarray]:
        """``(time_s, rms_uv)`` for ``kind`` in ``mean`` / ``first`` / ``second``."""
        empty = np.empty(0, dtype=np.float64)
        data = self._channel_data.get(int(ch))
        if data is not None:
            values = data.rms_profiles.get(kind)
            if values is None or data.rms_time.size == 0:
                return empty, empty
            time_axis = np.asarray(data.rms_time, dtype=np.float64)
            values_arr = np.asarray(values, dtype=np.float64)
            n = min(time_axis.size, values_arr.size)
            if n == 0:
                return empty, empty
            return time_axis[:n], values_arr[:n]
        values = _row(self.derived.rms_profiles.get(kind), ch)
        if values is None or self.derived.rms_time is None:
            return empty, empty
        time_axis = np.asarray(self.derived.rms_time, dtype=np.float64)
        n = min(time_axis.size, values.size)
        if n == 0:
            return empty, empty
        return time_axis[:n], values[:n]

    def mean_rms_across_channels(self, kind: str = "mean") -> tuple[np.ndarray, np.ndarray]:
        """RMS profile averaged over ready channels (summary panel)."""
        empty = np.empty(0, dtype=np.float64)
        if self._channel_data:
            rows: list[np.ndarray] = []
            time_axis: np.ndarray | None = None
            for data in self._channel_data.values():
                values = data.rms_profiles.get(kind)
                if values is None or values.size == 0:
                    continue
                if time_axis is None:
                    time_axis = np.asarray(data.rms_time, dtype=np.float64)
                rows.append(np.asarray(values, dtype=np.float64))
            if not rows or time_axis is None:
                return empty, empty
            width = min(time_axis.size, min(row.size for row in rows))
            stacked = np.vstack([row[:width] for row in rows])
            with np.errstate(invalid="ignore"):
                profile = np.nanmean(stacked, axis=0)
            return time_axis[:width], profile
        array = self.derived.rms_profiles.get(kind)
        if array is None or self.derived.rms_time is None:
            return empty, empty
        values = np.asarray(array, dtype=np.float64)
        if values.size == 0:
            return empty, empty
        with np.errstate(invalid="ignore"):
            profile = np.nanmean(values, axis=0)
        time_axis = np.asarray(self.derived.rms_time, dtype=np.float64)
        n = min(time_axis.size, profile.size)
        return time_axis[:n], profile[:n]

    def channel_rms_uv(self, ch: int) -> float:
        data = self._channel_data.get(int(ch))
        if data is not None:
            return float(data.channel_rms_uv)
        array = self.derived.channel_rms_uv
        if array is None or ch < 0 or ch >= int(array.shape[0]):
            return 0.0
        return float(array[ch])

    # -------------------------------------------------------------- spike data

    def threshold_uv(self, ch: int) -> float:
        data = self._channel_data.get(int(ch))
        if data is not None:
            return float(data.threshold_uv)
        array = self.derived.thresholds_uv
        if array is None or ch < 0 or ch >= int(array.shape[0]):
            return float(self.meta.spike_detection.threshold_uv)
        return float(array[ch])

    def threshold_caption(self, ch: int) -> str:
        data = self._channel_data.get(int(ch))
        if data is not None:
            return data.threshold_caption
        if 0 <= ch < len(self.threshold_captions) and self.threshold_captions[ch]:
            return self.threshold_captions[ch]
        info = self.meta.spike_detection
        sign = "−" if info.polarity == "negative" else "+"
        return f"threshold {sign}{abs(info.threshold_uv):g} µV"

    def spike_times(
        self,
        ch: int,
        *,
        trigger_index: int | None = None,
        t_range_s: tuple[float, float] | None = None,
    ) -> list[np.ndarray]:
        """Spike times per trial, optionally restricted to one trial / window.

        Detection runs once over the full trial window when the channel is
        ensured; narrowing to a zoom window filters those detections instead of
        re-detecting, which keeps zooming instantaneous and consistent.
        """
        data = self._channel_data.get(int(ch))
        trains = (
            [np.asarray(trial, dtype=np.float64) for trial in data.spike_trains]
            if data is not None
            else self.spikes.per_trial(ch)
        )
        if trigger_index is not None:
            trains = (
                [trains[int(trigger_index)]]
                if 0 <= int(trigger_index) < len(trains)
                else []
            )
        if t_range_s is None:
            return trains
        lo, hi = float(t_range_s[0]), float(t_range_s[1])
        return [arr[(arr >= lo) & (arr <= hi)] for arr in trains]

    def overlay_for_channel(
        self, ch: int, *, t_range_s: tuple[float, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray | None]:
        """``(t_ms, waveforms, spike_times, total, exact_mean_or_None)``."""
        data = self._channel_data.get(int(ch))
        if data is not None:
            t_ms = np.asarray(data.overlay_t_ms, dtype=np.float64)
            waves = np.asarray(data.overlay_waves)
            times = np.asarray(data.overlay_times, dtype=np.float64)
            total = int(data.overlay_total)
            mean = None if data.overlay_mean is None else np.asarray(data.overlay_mean, dtype=np.float64)
        else:
            t_ms, waves, times, total = self.overlay.for_channel(ch)
            mean = self.overlay.mean_for_channel(ch)
        if t_range_s is None:
            return t_ms, waves, times, total, mean
        lo, hi = float(t_range_s[0]), float(t_range_s[1])
        keep = (times >= lo) & (times <= hi)
        return t_ms, np.asarray(waves)[keep], times[keep], int(np.count_nonzero(keep)), None

    # ---------------------------------------------------------------- lifetime

    def close(self) -> None:
        """Release the streams and drop every memmap.

        Dropping the arrays matters on Windows, where an open memmap keeps a file
        handle on the bundle and prevents it from being deleted or overwritten.
        """
        self._channel_data.clear()
        source = self.source
        self.source = None
        if source is not None and hasattr(source, "close"):
            try:
                source.close()
            except Exception:
                pass
        empty = np.empty(0, dtype=np.float64)
        self.derived.means.clear()
        self.derived.trigger_windows.clear()
        self.derived.rms_profiles.clear()
        self.derived.t_rel = empty
        self.derived.triggers = empty
        self.derived.rms_time = None
        self.derived.channel_rms_uv = None
        self.derived.thresholds_uv = None
        self.spikes = SpikeTrains.empty(0, 0)
        self.overlay = OverlaySnippets.empty(0, 0)
        gc.collect()

    def describe(self) -> str:
        seg = self.meta.segmentation
        return (
            f"{self.label} — {self.n_channels} channels, {seg.describe()}, "
            f"fs = {self.meta.fs:.0f} Hz"
        )


# --------------------------------------------------------------------- bundles


def _serialize_impedance(sessions: Iterable[ImpedanceSession]) -> list[dict[str, Any]]:
    return [
        {
            "when": session.when.isoformat(),
            "rhs_path": str(session.rhs_path),
            "rhs_label": session.rhs_label,
            "source_csv": str(session.source_csv),
            "magnitudes_ohm": {k: float(v) for k, v in session.magnitudes_ohm.items()},
        }
        for session in sessions
    ]


def _deserialize_impedance(raw: Sequence[dict[str, Any]]) -> list[ImpedanceSession]:
    sessions: list[ImpedanceSession] = []
    for item in raw:
        try:
            when = _dt.datetime.fromisoformat(str(item["when"]))
        except (KeyError, ValueError):
            continue
        sessions.append(
            ImpedanceSession(
                when=when,
                magnitudes_ohm={str(k): float(v) for k, v in item.get("magnitudes_ohm", {}).items()},
                rhs_path=Path(str(item.get("rhs_path", ""))),
                rhs_label=str(item.get("rhs_label", "")),
                source_csv=Path(str(item.get("source_csv", ""))),
            )
        )
    return sorted(sessions, key=lambda s: s.when)


def materialize_channel_cache(recording: ProcessedRecording) -> bool:
    """Fold on-demand ``_channel_data`` into the arrays ``write_bundle`` persists.

    The live GUI keeps trial averages / spikes / overlays only in RAM per channel.
    Without this step, export would write an empty shell (``t_rel`` + triggers).
    Returns True when at least one channel product was packed.
    """
    cache = getattr(recording, "_channel_data", None) or {}
    if not cache:
        return False

    n_ch = int(recording.n_channels)
    n_trials = int(recording.n_trials)
    window = max(int(recording.meta.window_samples), 1)
    derived = recording.derived
    packed = False

    for stream in STREAM_NAMES:
        if not any(
            stream in data.means and data.means.get(stream) is not None
            for data in cache.values()
        ):
            continue
        arr = np.full((n_ch, window), np.nan, dtype=np.float32)
        for ch, data in cache.items():
            curve = data.means.get(stream)
            if curve is None:
                continue
            row = np.asarray(curve, dtype=np.float32).ravel()
            n = min(window, int(row.size))
            if n > 0:
                arr[int(ch), :n] = row[:n]
                packed = True
        derived.means[stream] = arr

    rms_time: np.ndarray | None = None
    for data in cache.values():
        if data.rms_time is not None and np.asarray(data.rms_time).size:
            rms_time = np.asarray(data.rms_time, dtype=np.float64).ravel()
            break
    if rms_time is not None and rms_time.size:
        derived.rms_time = rms_time
        rms_width = int(rms_time.size)
        for kind in RMS_KINDS:
            if not any(kind in data.rms_profiles for data in cache.values()):
                continue
            arr = np.full((n_ch, rms_width), np.nan, dtype=np.float32)
            for ch, data in cache.items():
                values = data.rms_profiles.get(kind)
                if values is None:
                    continue
                row = np.asarray(values, dtype=np.float32).ravel()
                n = min(rms_width, int(row.size))
                if n > 0:
                    arr[int(ch), :n] = row[:n]
                    packed = True
            derived.rms_profiles[kind] = arr

    channel_rms = (
        np.asarray(derived.channel_rms_uv, dtype=np.float32).copy()
        if derived.channel_rms_uv is not None and int(np.asarray(derived.channel_rms_uv).shape[0]) == n_ch
        else np.full(n_ch, np.nan, dtype=np.float32)
    )
    thresholds = (
        np.asarray(derived.thresholds_uv, dtype=np.float32).copy()
        if derived.thresholds_uv is not None and int(np.asarray(derived.thresholds_uv).shape[0]) == n_ch
        else np.full(n_ch, np.nan, dtype=np.float32)
    )
    captions = list(recording.threshold_captions)
    while len(captions) < n_ch:
        captions.append("")
    for ch, data in cache.items():
        idx = int(ch)
        if 0 <= idx < n_ch:
            channel_rms[idx] = float(data.channel_rms_uv)
            thresholds[idx] = float(data.threshold_uv)
            captions[idx] = str(data.threshold_caption or captions[idx])
            packed = True
    if np.any(np.isfinite(channel_rms)):
        derived.channel_rms_uv = channel_rms
    if np.any(np.isfinite(thresholds)):
        derived.thresholds_uv = thresholds
    recording.threshold_captions = captions

    per_channel: list[list[np.ndarray]] = []
    for ch in range(n_ch):
        data = cache.get(ch)
        if data is not None:
            trains = [np.asarray(trial, dtype=np.float64) for trial in data.spike_trains]
            while len(trains) < n_trials:
                trains.append(np.empty(0, dtype=np.float64))
            per_channel.append(trains[:n_trials])
            packed = True
        else:
            per_channel.append(recording.spikes.per_trial(ch))
    recording.spikes = SpikeTrains.from_lists(per_channel, n_trials)

    t_ms: np.ndarray | None = None
    for data in cache.values():
        if data.overlay_t_ms is not None and np.asarray(data.overlay_t_ms).size:
            t_ms = np.asarray(data.overlay_t_ms, dtype=np.float64).ravel()
            break
    if t_ms is None and recording.overlay.t_ms.size:
        t_ms = np.asarray(recording.overlay.t_ms, dtype=np.float64).ravel()
    if t_ms is not None:
        ov_window = max(int(t_ms.size), 1)
        counts = np.zeros(n_ch, dtype=np.int32)
        totals = np.zeros(n_ch, dtype=np.int32)
        means = np.zeros((n_ch, ov_window), dtype=np.float32)
        wave_chunks: list[np.ndarray] = []
        time_chunks: list[np.ndarray] = []
        for ch in range(n_ch):
            data = cache.get(ch)
            if data is not None:
                waves = np.asarray(data.overlay_waves, dtype=np.float32)
                times = np.asarray(data.overlay_times, dtype=np.float32).ravel()
                totals[ch] = int(data.overlay_total)
                if data.overlay_mean is not None:
                    mean = np.asarray(data.overlay_mean, dtype=np.float32).ravel()
                    n = min(ov_window, int(mean.size))
                    if n:
                        means[ch, :n] = mean[:n]
                packed = True
            else:
                _t, waves, times, total = recording.overlay.for_channel(ch)
                waves = np.asarray(waves, dtype=np.float32)
                times = np.asarray(times, dtype=np.float32).ravel()
                totals[ch] = int(total)
                mean = recording.overlay.mean_for_channel(ch)
                if mean is not None:
                    mean_arr = np.asarray(mean, dtype=np.float32).ravel()
                    n = min(ov_window, int(mean_arr.size))
                    if n:
                        means[ch, :n] = mean_arr[:n]
            if waves.ndim != 2 or waves.shape[0] == 0:
                continue
            if int(waves.shape[1]) != ov_window:
                fixed = np.zeros((waves.shape[0], ov_window), dtype=np.float32)
                n = min(ov_window, int(waves.shape[1]))
                fixed[:, :n] = waves[:, :n]
                waves = fixed
            counts[ch] = int(waves.shape[0])
            wave_chunks.append(waves)
            time_chunks.append(times[: waves.shape[0]])
        waves_all = (
            np.concatenate(wave_chunks, axis=0)
            if wave_chunks
            else np.empty((0, ov_window), dtype=np.float32)
        )
        times_all = (
            np.concatenate(time_chunks)
            if time_chunks
            else np.empty(0, dtype=np.float32)
        )
        recording.overlay = OverlaySnippets(
            t_ms, counts, totals, waves_all, times_all, means
        )

    return packed


def materialize_trigger_windows(recording: ProcessedRecording) -> int:
    """Slice first/second stimulation windows from the live source into ``derived``.

    Returns the number of ``(trigger, stream)`` arrays written.
    """
    if recording.source is None:
        return 0
    n_ch = int(recording.n_channels)
    window = max(int(recording.meta.window_samples), 1)
    written = 0
    for trigger_index in (0, 1):
        for stream in STREAM_NAMES:
            key = (trigger_index, stream)
            if key in recording.derived.trigger_windows:
                continue
            arr = np.full((n_ch, window), np.nan, dtype=np.float32)
            any_ok = False
            for ch in range(n_ch):
                row = recording.trigger_window(trigger_index, stream, ch)
                if row is None:
                    continue
                values = np.asarray(row, dtype=np.float32).ravel()
                n = min(window, int(values.size))
                if n <= 0:
                    continue
                arr[ch, :n] = values[:n]
                any_ok = True
            if any_ok:
                recording.derived.trigger_windows[key] = arr
                written += 1
    return written


def write_bundle(
    recording: ProcessedRecording,
    bundle: BundleLayout,
    *,
    include_streams: bool = False,
    include_trigger_windows: bool = True,
    include_overlay: bool = True,
    stream_paths: dict[str, Path] | None = None,
) -> Path:
    """Persist a recording as a reusable bundle directory. Returns its root."""
    materialize_channel_cache(recording)
    if include_trigger_windows:
        materialize_trigger_windows(recording)

    bundle.ensure()
    derived = recording.derived
    if not derived.means and not getattr(recording, "_channel_data", None):
        raise ValueError(
            "Nothing to export: no trial averages are available. "
            "Compute channels (F6 / F7) before exporting a processed dataset."
        )
    _save_array(bundle.derived_path("t_rel"), np.asarray(derived.t_rel, dtype=np.float64))
    _save_array(bundle.derived_path("triggers"), np.asarray(derived.triggers, dtype=np.int64))
    for stream, array in derived.means.items():
        _save_array(bundle.derived_path(f"mean_{stream}"), np.asarray(array, dtype=np.float32))
    if include_trigger_windows:
        for (trigger_index, stream), array in derived.trigger_windows.items():
            _save_array(
                bundle.derived_path(DerivedArrays.trigger_key(trigger_index, stream)),
                np.asarray(array, dtype=np.float32),
            )
    if derived.rms_time is not None:
        _save_array(bundle.derived_path("rms_time"), np.asarray(derived.rms_time, dtype=np.float64))
    for kind, array in derived.rms_profiles.items():
        _save_array(bundle.derived_path(f"rms_{kind}"), np.asarray(array, dtype=np.float32))
    if derived.channel_rms_uv is not None:
        _save_array(
            bundle.derived_path("channel_rms"), np.asarray(derived.channel_rms_uv, dtype=np.float32)
        )
    if derived.thresholds_uv is not None:
        _save_array(
            bundle.derived_path("thresholds"), np.asarray(derived.thresholds_uv, dtype=np.float32)
        )
    recording.spikes.save(bundle.spikes_dir)
    if include_overlay:
        recording.overlay.save(bundle.overlay_dir)
    elif bundle.overlay_dir.exists():
        shutil.rmtree(bundle.overlay_dir, ignore_errors=True)
        bundle.overlay_dir.mkdir(parents=True, exist_ok=True)

    if include_streams and stream_paths:
        bundle.streams_dir.mkdir(parents=True, exist_ok=True)
        for name, path in stream_paths.items():
            target = bundle.stream_path(name)
            if Path(path).resolve() != target.resolve() and Path(path).exists():
                shutil.copy2(path, target)

    stored_streams = include_streams and bool(stream_paths)
    bundle.write_manifest(
        {
            "format": "plot_erg.processed_dataset",
            "version": CACHE_FORMAT_VERSION,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "label": recording.label,
            "streams": (
                {name: f"streams/{name}.npy" for name in stream_paths} if stored_streams else {}
            ),
            "has_streams": bool(stored_streams),
            "has_trigger_windows": bool(include_trigger_windows and derived.trigger_windows),
            "has_overlay": bool(include_overlay),
            "meta": recording.meta.to_dict(),
            "threshold_captions": list(recording.threshold_captions),
            "impedance_sessions": _serialize_impedance(recording.impedance_sessions),
            "build_timings": [[name, float(seconds)] for name, seconds in recording.build_timings],
        }
    )
    return bundle.root


def read_bundle(
    bundle: BundleLayout,
    *,
    label: str | None = None,
    style: RecordingStyle | None = None,
) -> ProcessedRecording:
    """Load a bundle written by :func:`write_bundle` (memmapped, no RHS needed)."""
    manifest = bundle.read_manifest()
    if manifest.get("format") != "plot_erg.processed_dataset":
        raise ValueError(f"Not a processed dataset bundle: {bundle.root}")
    meta = RecordingMeta.from_dict(manifest.get("meta", {}))
    t_rel = _load_array(bundle.derived_path("t_rel"))
    if t_rel is None:
        raise ValueError(f"Bundle is missing derived/t_rel.npy: {bundle.root}")
    triggers = _load_array(bundle.derived_path("triggers"))
    derived = DerivedArrays(
        t_rel=np.asarray(t_rel, dtype=np.float64),
        triggers=(
            np.asarray(triggers, dtype=np.int64)
            if triggers is not None
            else np.empty(0, dtype=np.int64)
        ),
    )
    for stream in STREAM_NAMES:
        array = _load_memmap(bundle.derived_path(f"mean_{stream}"))
        if array is not None:
            derived.means[stream] = array
        for trigger_index in (0, 1):
            window = _load_memmap(
                bundle.derived_path(DerivedArrays.trigger_key(trigger_index, stream))
            )
            if window is not None:
                derived.trigger_windows[(trigger_index, stream)] = window
    derived.rms_time = _load_array(bundle.derived_path("rms_time"))
    for kind in RMS_KINDS:
        array = _load_memmap(bundle.derived_path(f"rms_{kind}"))
        if array is not None:
            derived.rms_profiles[kind] = array
    derived.channel_rms_uv = _load_array(bundle.derived_path("channel_rms"))
    derived.thresholds_uv = _load_array(bundle.derived_path("thresholds"))

    window = int(meta.window_samples)
    spikes = SpikeTrains.load(bundle.spikes_dir) or SpikeTrains.empty(
        meta.n_channels, meta.segmentation.n_trials
    )
    overlay = OverlaySnippets.load(bundle.overlay_dir) or OverlaySnippets.empty(
        meta.n_channels, 0
    )
    del window

    source = _open_streams(bundle, meta, derived, manifest.get("streams") or {})
    return ProcessedRecording(
        meta=meta,
        derived=derived,
        spikes=spikes,
        overlay=overlay,
        label=label or str(manifest.get("label") or meta.source_name),
        style=style,
        impedance_sessions=_deserialize_impedance(manifest.get("impedance_sessions", [])),
        source=source,
        bundle_root=bundle.root,
        threshold_captions=[str(c) for c in manifest.get("threshold_captions", [])],
        build_timings=[
            (str(name), float(seconds))
            for name, seconds in manifest.get("build_timings", [])
        ],
    )


def _resolve_stream_path(bundle: BundleLayout, declared: Any, name: str) -> Path:
    if declared:
        candidate = Path(str(declared))
        return candidate if candidate.is_absolute() else bundle.root / candidate
    return bundle.stream_path(name)


def _open_streams(
    bundle: BundleLayout,
    meta: RecordingMeta,
    derived: DerivedArrays,
    declared: dict[str, Any],
) -> Any | None:
    """Reattach an :class:`core.AmplifierSpikeSource` when streams are present."""
    amp = _load_memmap(_resolve_stream_path(bundle, declared.get("raw"), "raw"))
    high = _load_memmap(_resolve_stream_path(bundle, declared.get("hp"), "hp"))
    low = _load_memmap(_resolve_stream_path(bundle, declared.get("lp"), "lp"))
    if amp is None or high is None or low is None:
        return None
    from core import AmplifierSpikeSource

    return AmplifierSpikeSource(
        amplifier=amp,
        highpass=high,
        lowpass=low,
        valid_triggers=np.asarray(derived.triggers, dtype=np.int64),
        pre_n=int(meta.segmentation.pre_n),
        post_n=int(meta.segmentation.post_n),
        work_dir=None,
        intan_dsp=meta.dsp,
    )


def dataset_target_path(directory: Path, stem: str) -> Path:
    """Default export path for a processed dataset bundle."""
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in stem).strip() or "dataset"
    return Path(directory) / f"{safe}{DATASET_SUFFIX}"


def archive_bundle(bundle_root: Path, archive_path: Path) -> Path:
    """Zip a bundle directory for transport. Returns the archive path."""
    archive_path = Path(archive_path)
    base = archive_path.with_suffix("") if archive_path.suffix == ".zip" else archive_path
    created = shutil.make_archive(str(base), "zip", root_dir=str(bundle_root))
    return Path(created)


def extract_bundle(archive_path: Path, destination: Path) -> Path:
    """Unpack a zipped bundle and return the directory containing the manifest."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), str(destination))
    if (destination / "manifest.json").exists():
        return destination
    for child in destination.rglob("manifest.json"):
        return child.parent
    raise ValueError(f"No manifest.json inside archive: {archive_path}")


def recording_label_for(path: Path, custom: str | None) -> str:
    return resolve_display_label(Path(path).stem, custom)


def keys_match(meta: RecordingMeta, keys: CacheKeys) -> bool:
    """True when a loaded bundle was produced with the given cache keys."""
    stored = dict(meta.cache_keys)
    return all(stored.get(name) == value for name, value in keys.as_dict().items())
