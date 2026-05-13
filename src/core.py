"""Core analysis: Intan RHS loading, Butterworth filters, triggers, spike windows, mmap amplifier."""

from __future__ import annotations

import contextlib
import contextvars
import gc
import importlib.util
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

from config import AnalysisConfig

_analysis_cancel_event: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "analysis_cancel_event",
    default=None,
)
MAX_PARALLEL_CHANNELS = 16


@contextlib.contextmanager
def analysis_cancel_scope(event: threading.Event | None) -> Iterator[None]:
    """Used by the GUI thread: if ``event`` is set, ``check_analysis_cancelled`` raises."""
    token = _analysis_cancel_event.set(event)
    try:
        yield
    finally:
        _analysis_cancel_event.reset(token)


def check_analysis_cancelled() -> None:
    """Raise InterruptedError if the user requested stop (GUI)."""
    ev = _analysis_cancel_event.get()
    if ev is not None and ev.is_set():
        raise InterruptedError("Analysis interrupted.")


def apply_butterworth_lowpass(
    data: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth low-pass, zero-phase (filtfilt), per channel (axis=1)."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise RuntimeError("apply_butterworth_lowpass expects [n_channels, n_samples].")
    nyq = 0.5 * fs
    if cutoff_hz <= 0:
        raise ValueError("Cutoff frequency must be > 0.")
    if cutoff_hz >= nyq:
        raise ValueError(
            f"Cutoff ({cutoff_hz} Hz) must be below Nyquist frequency ({nyq:.1f} Hz)."
        )
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, data, axis=1)


def apply_butterworth_bandpass(
    data: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth band-pass, zero-phase (filtfilt), per channel (axis=1)."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise RuntimeError("apply_butterworth_bandpass expects [n_channels, n_samples].")
    nyq = 0.5 * fs
    if low_hz <= 0 or high_hz <= 0:
        raise ValueError("Band-pass corner frequencies must be > 0.")
    if low_hz >= high_hz:
        raise ValueError("Band-pass: low frequency must be < high frequency.")
    if high_hz >= nyq:
        raise ValueError(
            f"High corner ({high_hz} Hz) must be below Nyquist ({nyq:.1f} Hz)."
        )
    wn = (low_hz / nyq, high_hz / nyq)
    b, a = butter(order, wn, btype="band")
    return filtfilt(b, a, data, axis=1)


def _import_intan_loader():
    src_dir = Path(__file__).resolve().parent
    loader_path = src_dir / "load_intan_rhs_format.py"
    if not loader_path.exists():
        raise RuntimeError(f"File not found: {loader_path}")

    spec = importlib.util.spec_from_file_location("load_intan_rhs_format", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot build Python loader for load_intan_rhs_format.py.")

    # Resolve reader internal imports (e.g. intanutil.*) from src/.
    inserted_path = False
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        inserted_path = True

    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        read_data = getattr(module, "read_data", None)
        if not callable(read_data):
            raise RuntimeError("read_data is missing from load_intan_rhs_format.py.")
        return read_data
    except Exception as exc:
        raise RuntimeError(
            "Cannot import 'load_intan_rhs_format.py'. "
            "Ensure the 'intanutil' package is present under src/ (full Intan zip). "
            f"Detail: {exc}"
        ) from exc
    finally:
        if inserted_path:
            try:
                sys.path.remove(str(src_dir))
            except ValueError:
                pass


def load_rhs_file(rhs_path: Path) -> dict[str, Any]:
    read_data = _import_intan_loader()
    data = read_data(str(rhs_path))
    if not isinstance(data, dict):
        raise RuntimeError("Intan reader did not return a dictionary.")
    return data


def get_sampling_rate(data: dict[str, Any]) -> float:
    freq = data.get("frequency_parameters", {})
    sample_rate = freq.get("amplifier_sample_rate")
    if sample_rate is None:
        raise RuntimeError("Sample rate not found.")
    return float(sample_rate)


def get_analog_in0_signal(data: dict[str, Any]) -> np.ndarray:
    board_adc_data = np.asarray(data.get("board_adc_data"))
    if board_adc_data.size == 0:
        raise RuntimeError("RHS file has no board_adc_data.")
    if board_adc_data.ndim != 2:
        raise RuntimeError("Unexpected board_adc_data shape.")
    if board_adc_data.shape[0] < 1:
        raise RuntimeError("No ADC channel found.")
    return board_adc_data[0]


def detect_edges(signal: np.ndarray, threshold: float, edge: str) -> np.ndarray:
    """Sample indices at threshold crossing: 'falling' or 'rising'."""
    above = signal > threshold
    if edge == "rising":
        return np.where((~above[:-1]) & (above[1:]))[0] + 1
    if edge == "falling":
        return np.where((above[:-1]) & (~above[1:]))[0] + 1
    raise ValueError("edge must be 'falling' or 'rising'.")


def mean_time_to_next_rising_edge_s(
    signal: np.ndarray,
    trigger_indices: np.ndarray,
    threshold: float,
    fs: float,
) -> float | None:
    """Mean time (s) from each trigger to the next rising-edge crossing (typical pulse end)."""
    trigger_indices = np.asarray(trigger_indices, dtype=np.int64)
    rising_idx = detect_edges(signal, threshold, "rising")
    if rising_idx.size == 0:
        return None
    deltas_s: list[float] = []
    for tr in trigger_indices:
        after = rising_idx[rising_idx > tr]
        if after.size == 0:
            continue
        deltas_s.append(float(after[0] - tr) / fs)
    if not deltas_s:
        return None
    return float(np.mean(deltas_s))


def detect_spikes_at_threshold(
    trace: np.ndarray,
    fs: float,
    threshold: float,
    refractory_s: float = 0.001,
) -> np.ndarray:
    """Sample indices of threshold crossings with refractory period.

    - threshold >= 0: rising crossing (signal goes above threshold).
    - threshold < 0: falling crossing (signal goes below threshold), for negative spikes.
    """
    trace = np.asarray(trace, dtype=np.float64).ravel()
    if trace.size < 2:
        return np.array([], dtype=np.int64)
    if threshold >= 0:
        above = trace > threshold
        cross = np.where((~above[:-1]) & (above[1:]))[0] + 1
    else:
        below = trace < threshold
        cross = np.where((~below[:-1]) & (below[1:]))[0] + 1
    if cross.size == 0:
        return cross.astype(np.int64)
    min_dist = max(1, int(round(refractory_s * fs)))
    kept: list[int] = [int(cross[0])]
    for c in cross[1:]:
        if c - kept[-1] >= min_dist:
            kept.append(int(c))
    return np.asarray(kept, dtype=np.int64)


def detect_spikes_threshold_rising(
    trace: np.ndarray,
    fs: float,
    threshold: float,
    refractory_s: float = 0.001,
) -> np.ndarray:
    """Backward compatibility: delegates to detect_spikes_at_threshold (negative threshold supported)."""
    return detect_spikes_at_threshold(trace, fs, threshold, refractory_s=refractory_s)


def resolve_work_dir(config: AnalysisConfig) -> Path:
    """Directory for amplifier_raw.npy and intermediate files."""
    if config.work_dir is not None:
        return config.work_dir
    root = config.save_dir if config.save_dir is not None else config.rhs_file.parent
    return root / ".plot_erg" / f"{config.rhs_file.stem}_work"


def cleanup_plot_erg_root_if_empty(work_dir: Path | None) -> None:
    """Remove the root .plot_erg folder only if empty."""
    if work_dir is None:
        return
    root = work_dir.parent
    if root.name != ".plot_erg":
        return
    if not root.exists():
        return
    try:
        if not any(root.iterdir()):
            root.rmdir()
    except OSError:
        # Non-empty, locked, or permission error: ignore.
        pass


class AmplifierSpikeSource:
    """Per-trigger windows, one channel at a time (no full 3D tensor)."""

    def __init__(
        self,
        amplifier: np.ndarray,
        valid_triggers: np.ndarray,
        pre_n: int,
        post_n: int,
        work_dir: Path | None,
        keep_intermediate_files: bool,
        fs: float,
        bandpass_low_hz: float | None = None,
        bandpass_high_hz: float | None = None,
    ) -> None:
        self.amplifier = amplifier
        self.valid_triggers = np.asarray(valid_triggers, dtype=np.int64)
        self.pre_n = pre_n
        self.post_n = post_n
        self._offsets = np.arange(-pre_n, post_n, dtype=np.int64)
        self.work_dir = work_dir
        self.keep_intermediate_files = keep_intermediate_files
        self.fs = float(fs)
        self.bandpass_low_hz = bandpass_low_hz
        self.bandpass_high_hz = bandpass_high_hz
        self._closed = False

    def windows_2d_for_channel(self, ch: int) -> np.ndarray:
        channel_trace = np.asarray(self.amplifier[ch], dtype=np.float64)
        if self.bandpass_low_hz is not None and self.bandpass_high_hz is not None:
            channel_trace_2d = channel_trace.reshape(1, -1)
            channel_trace = apply_butterworth_bandpass(
                channel_trace_2d, self.fs, self.bandpass_low_hz, self.bandpass_high_hz
            )[0]
        trial_count = int(self.valid_triggers.size)
        window_length = int(self._offsets.size)
        # Guard: avoid huge spike-window allocations.
        if trial_count * window_length > 25_000_000:
            raise RuntimeError(
                "Spike window too large for 2D RAM extraction. "
                "Reduce pre/post, sampling %, or enable lightweight mode."
            )
        trial_windows = np.empty((trial_count, window_length), dtype=np.float64)
        for trial_index, trigger_index in enumerate(self.valid_triggers):
            sample_start = int(trigger_index - self.pre_n)
            sample_end = int(trigger_index + self.post_n)
            trial_windows[trial_index] = channel_trace[sample_start:sample_end]
        return trial_windows

    def spike_times_per_trial_for_channel(
        self,
        ch: int,
        t_rel: np.ndarray,
        threshold: float,
        refractory_s: float = 0.001,
    ) -> list[np.ndarray]:
        """Detect spikes trial by trial without building a giant 2D matrix in RAM."""
        channel_trace = np.asarray(self.amplifier[ch], dtype=np.float64)
        if self.bandpass_low_hz is not None and self.bandpass_high_hz is not None:
            channel_trace_2d = channel_trace.reshape(1, -1)
            channel_trace = apply_butterworth_bandpass(
                channel_trace_2d, self.fs, self.bandpass_low_hz, self.bandpass_high_hz
            )[0]
        spike_times_by_trial: list[np.ndarray] = []
        for trigger_index in self.valid_triggers:
            sample_start = int(trigger_index - self.pre_n)
            sample_end = int(trigger_index + self.post_n)
            trial_trace = channel_trace[sample_start:sample_end]
            spike_sample_indices = detect_spikes_at_threshold(
                trial_trace, self.fs, threshold, refractory_s=refractory_s
            )
            spike_times_by_trial.append(np.asarray(t_rel[spike_sample_indices], dtype=np.float64))
        return spike_times_by_trial

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        amplifier_array = self.amplifier
        try:
            if isinstance(amplifier_array, np.memmap):
                amplifier_array._mmap.close()
        except Exception:
            pass
        self.amplifier = np.empty((0,))  # drop reference
        if self.work_dir is not None and self.work_dir.exists() and not self.keep_intermediate_files:
            shutil.rmtree(self.work_dir, ignore_errors=True)
            cleanup_plot_erg_root_if_empty(self.work_dir)


def valid_triggers_and_timebase(
    n_samples: int,
    trigger_indices: np.ndarray,
    fs: float,
    pre_s: float,
    post_s: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Valid triggers and relative time axis without extracting windows."""
    trigger_indices = np.asarray(trigger_indices, dtype=np.int64)
    pre_n = int(round(pre_s * fs))
    post_n = int(round(post_s * fs))
    win_len = pre_n + post_n
    if win_len <= 1:
        raise RuntimeError("Invalid time window.")
    start_idx = trigger_indices - pre_n
    end_idx = trigger_indices + post_n
    valid_mask = (start_idx >= 0) & (end_idx <= n_samples)
    valid_triggers = trigger_indices[valid_mask]
    if valid_triggers.size == 0:
        raise RuntimeError("No valid window around triggers.")
    t_rel = np.arange(-pre_n, post_n, dtype=np.float64) / fs
    return valid_triggers, t_rel, pre_n, post_n


def resolve_channel_workers(channel_workers: int | None, n_channels: int) -> int:
    """Channel worker count with safety cap."""
    if n_channels <= 0:
        return 1
    if channel_workers is not None:
        return max(1, min(int(channel_workers), int(MAX_PARALLEL_CHANNELS), int(n_channels)))
    cpu_half = max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(cpu_half, int(MAX_PARALLEL_CHANNELS), int(n_channels)))


def mean_filtered_channelwise(
    amplifier_2d: np.ndarray,
    valid_triggers: np.ndarray,
    fs: float,
    pre_n: int,
    post_n: int,
    cutoff_hz: float,
    channel_workers: int | None = None,
) -> np.ndarray:
    """Low-pass per channel, parallelized up to MAX_PARALLEL_CHANNELS."""
    channel_count, _ = amplifier_2d.shape
    window_length = pre_n + post_n
    mean_windows = np.zeros((channel_count, window_length), dtype=np.float64)

    def _compute_one_channel(c: int) -> tuple[int, np.ndarray]:
        check_analysis_cancelled()
        filtered_channel = apply_butterworth_lowpass(amplifier_2d[c : c + 1], fs, cutoff_hz)[0]
        summed_windows = np.zeros(window_length, dtype=np.float64)
        for trigger_index in valid_triggers:
            sample_start = int(trigger_index - pre_n)
            sample_end = int(trigger_index + post_n)
            summed_windows += filtered_channel[sample_start:sample_end]
        return c, summed_windows / float(valid_triggers.size)

    worker_count = resolve_channel_workers(channel_workers, channel_count)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for channel_index, channel_mean in pool.map(_compute_one_channel, range(channel_count)):
            mean_windows[channel_index] = channel_mean
    return mean_windows


def persist_amplifier_float32(amplifier_2d: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(amplifier_2d, dtype=np.float32))


def extract_triggered_windows(
    amplifier_data: np.ndarray,
    trigger_indices: np.ndarray,
    fs: float,
    pre_s: float,
    post_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy 3D window extraction.

    Implemented without a giant 2D sample index matrix to avoid huge int64 allocations.
    """
    amplifier_data = np.asarray(amplifier_data)
    if amplifier_data.ndim != 2:
        raise RuntimeError("Unexpected amplifier_data shape (expected [n_channels, n_samples]).")

    _, n_samples = amplifier_data.shape
    pre_n = int(round(pre_s * fs))
    post_n = int(round(post_s * fs))
    win_len = pre_n + post_n
    if win_len <= 1:
        raise RuntimeError("Invalid time window.")

    trigger_indices = np.asarray(trigger_indices, dtype=np.int64)
    start_idx = trigger_indices - pre_n
    end_idx = trigger_indices + post_n
    valid_mask = (start_idx >= 0) & (end_idx <= n_samples)
    valid_triggers = trigger_indices[valid_mask]

    if valid_triggers.size == 0:
        raise RuntimeError("No valid window around triggers.")

    n_trials = int(valid_triggers.size)
    n_channels = int(amplifier_data.shape[0])
    # Memory guard for excessive 3D allocation.
    est_values = n_trials * n_channels * win_len
    if est_values > 200_000_000:
        raise RuntimeError(
            "3D windows too large for RAM extraction. Use the per-channel streaming pipeline."
        )

    windows = np.empty((n_trials, n_channels, win_len), dtype=np.asarray(amplifier_data).dtype)
    for i, trig in enumerate(valid_triggers):
        start = int(trig - pre_n)
        end = int(trig + post_n)
        windows[i] = amplifier_data[:, start:end]
    t_rel = np.arange(-pre_n, post_n, dtype=np.float64) / fs
    return windows, t_rel


def mean_triggered_windows_channelwise(
    amplifier_data: np.ndarray,
    valid_triggers: np.ndarray,
    pre_n: int,
    post_n: int,
    channel_workers: int | None = None,
) -> np.ndarray:
    """Raw mean per channel, parallelized up to MAX_PARALLEL_CHANNELS."""
    amplifier_data = np.asarray(amplifier_data)
    if amplifier_data.ndim != 2:
        raise RuntimeError("Unexpected amplifier_data shape (expected [n_channels, n_samples]).")

    n_ch, _ = amplifier_data.shape
    win_len = pre_n + post_n
    if win_len <= 1:
        raise RuntimeError("Invalid time window.")
    valid_triggers = np.asarray(valid_triggers, dtype=np.int64)
    if valid_triggers.size == 0:
        raise RuntimeError("No valid window around triggers.")

    # float32 is enough for PDF display and cuts RAM a lot.
    out = np.zeros((n_ch, win_len), dtype=np.float32)

    def _compute_one_channel(c: int) -> tuple[int, np.ndarray]:
        check_analysis_cancelled()
        row = np.asarray(amplifier_data[c], dtype=np.float32)
        acc = np.zeros(win_len, dtype=np.float32)
        for trig in valid_triggers:
            start = int(trig - pre_n)
            end = int(trig + post_n)
            acc += row[start:end]
        return c, acc / float(valid_triggers.size)

    n_workers = resolve_channel_workers(channel_workers, n_ch)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for c, m in pool.map(_compute_one_channel, range(n_ch)):
            out[c] = m
    return out


def get_channel_names(data: dict[str, Any], n_channels: int) -> list[str]:
    channels = data.get("amplifier_channels", [])
    if isinstance(channels, list) and len(channels) == n_channels:
        names = []
        for i, ch in enumerate(channels):
            if isinstance(ch, dict):
                name = ch.get("native_channel_name") or ch.get("custom_channel_name") or f"CH{i}"
            else:
                name = f"CH{i}"
            names.append(str(name))
        return names
    return [f"CH{i}" for i in range(n_channels)]


def compute_average_per_channel(
    config: AnalysisConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    int,
    int,
    float,
    float | None,
    AmplifierSpikeSource,
    np.ndarray,
]:
    check_analysis_cancelled()
    if not config.rhs_file.exists():
        raise FileNotFoundError(f"File not found: {config.rhs_file}")

    data = load_rhs_file(config.rhs_file)
    check_analysis_cancelled()
    fs = get_sampling_rate(data)
    analog_in0 = get_analog_in0_signal(data)
    trigger_indices = detect_edges(analog_in0, threshold=config.threshold, edge=config.edge)
    if trigger_indices.size == 0:
        edge_fr = "falling" if config.edge == "falling" else "rising"
        raise RuntimeError(f"No {edge_fr} edge detected on ANALOG_IN 0.")

    amplifier_raw = np.asarray(data.get("amplifier_data"))
    if amplifier_raw.size == 0:
        raise RuntimeError("RHS file has no amplifier_data.")

    _, n_samples = amplifier_raw.shape
    valid_triggers, t_rel, pre_n, post_n = valid_triggers_and_timebase(
        n_samples=n_samples,
        trigger_indices=trigger_indices,
        fs=fs,
        pre_s=config.pre_s,
        post_s=config.post_s,
    )
    check_analysis_cancelled()

    channel_names = get_channel_names(data, amplifier_raw.shape[0])
    n_valid = int(valid_triggers.shape[0])
    n_total = int(trigger_indices.size)

    # Raw mean in streaming mode (avoids large RAM allocations).
    mean_per_channel_raw = mean_triggered_windows_channelwise(
        amplifier_data=amplifier_raw,
        valid_triggers=valid_triggers,
        pre_n=pre_n,
        post_n=post_n,
        channel_workers=config.channel_workers,
    )

    if config.lowpass_cutoff_hz is not None:
        mean_per_channel = mean_filtered_channelwise(
            amplifier_raw,
            valid_triggers,
            fs,
            pre_n,
            post_n,
            config.lowpass_cutoff_hz,
            config.channel_workers,
        )
    else:
        mean_per_channel = mean_per_channel_raw

    end_rising_rel_s = mean_time_to_next_rising_edge_s(
        analog_in0, trigger_indices, config.threshold, fs
    )

    bp_lo: float | None = None
    bp_hi: float | None = None
    if config.spike_bandpass_low_hz is not None or config.spike_bandpass_high_hz is not None:
        if config.spike_bandpass_low_hz is None or config.spike_bandpass_high_hz is None:
            raise ValueError(
                "Spike band-pass: set both spike_bandpass_low_hz and spike_bandpass_high_hz, "
                "or leave both unset (raw signal)."
            )
        bp_lo = float(config.spike_bandpass_low_hz)
        bp_hi = float(config.spike_bandpass_high_hz)
        nyq = 0.5 * fs
        if bp_lo <= 0 or bp_hi <= 0:
            raise ValueError("Spike band-pass: frequencies must be > 0 Hz.")
        if bp_lo >= bp_hi:
            raise ValueError("Spike band-pass: low frequency must be < high frequency.")
        if bp_hi >= nyq:
            raise ValueError(
                f"Spike band-pass: high frequency ({bp_hi:g} Hz) must be < Nyquist ({nyq:.1f} Hz)."
            )

    # mmap file: drop RHS dict and in-RAM amplifier copy before PDF.
    work_dir = resolve_work_dir(config)
    amp_path = work_dir / "amplifier_raw.npy"
    persist_amplifier_float32(amplifier_raw, amp_path)
    del amplifier_raw
    if isinstance(data, dict):
        data.pop("amplifier_data", None)
    del data
    gc.collect()

    amp_mm = np.load(amp_path, mmap_mode="r")
    spike_source = AmplifierSpikeSource(
        amplifier=amp_mm,
        valid_triggers=valid_triggers,
        pre_n=pre_n,
        post_n=post_n,
        work_dir=work_dir,
        keep_intermediate_files=config.keep_intermediate_files,
        fs=fs,
        bandpass_low_hz=bp_lo,
        bandpass_high_hz=bp_hi,
    )

    return (
        mean_per_channel,
        t_rel,
        channel_names,
        n_valid,
        n_total,
        fs,
        end_rising_rel_s,
        spike_source,
        mean_per_channel_raw,
    )
