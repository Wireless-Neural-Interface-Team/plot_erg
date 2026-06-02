"""Core analysis: Intan RHS loading, Butterworth filters, triggers, spike windows, mmap amplifier."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import gc
import importlib.util
import io
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

from config import AnalysisConfig, CurveFilterKind, SectionSpecKind
from intan_rhx_dsp import (
    IntanDspSettings,
    compute_high_channel_stack,
    detect_spikes_intan,
    mean_rms_intan_channel,
)

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


@functools.lru_cache(maxsize=128)
def _butter_lowpass_coeffs(fs: float, cutoff_hz: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    if cutoff_hz <= 0:
        raise ValueError("Cutoff frequency must be > 0.")
    if cutoff_hz >= nyq:
        raise ValueError(
            f"Cutoff ({cutoff_hz} Hz) must be below Nyquist frequency ({nyq:.1f} Hz)."
        )
    wn = cutoff_hz / nyq
    return butter(order, wn, btype="low")


@functools.lru_cache(maxsize=128)
def _butter_highpass_coeffs(fs: float, cutoff_hz: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    if cutoff_hz <= 0:
        raise ValueError("Cutoff frequency must be > 0.")
    if cutoff_hz >= nyq:
        raise ValueError(
            f"Cutoff ({cutoff_hz} Hz) must be below Nyquist frequency ({nyq:.1f} Hz)."
        )
    wn = cutoff_hz / nyq
    return butter(order, wn, btype="high")


@functools.lru_cache(maxsize=128)
def _butter_bandpass_coeffs(
    fs: float, low_hz: float, high_hz: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
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
    return butter(order, wn, btype="band")


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
    b, a = _butter_lowpass_coeffs(float(fs), float(cutoff_hz), int(order))
    return filtfilt(b, a, data, axis=1)


def apply_butterworth_highpass(
    data: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth high-pass, zero-phase (filtfilt), per channel (axis=1)."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise RuntimeError("apply_butterworth_highpass expects [n_channels, n_samples].")
    b, a = _butter_highpass_coeffs(float(fs), float(cutoff_hz), int(order))
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
    b, a = _butter_bandpass_coeffs(float(fs), float(low_hz), float(high_hz), int(order))
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


def uses_analog_trigger(config: AnalysisConfig) -> bool:
    """True when segmentation uses ANALOG_IN 0 edges (not fixed sections)."""
    return config.edge != "none"


def peek_rhs_recording_info(rhs_path: Path) -> tuple[int, float]:
    """Read RHS header only: return (num_samples, sample_rate_hz)."""
    if not rhs_path.exists():
        raise FileNotFoundError(f"File not found: {rhs_path}")
    from intanutil.data import calculate_num_samples, get_bytes_per_data_block
    from intanutil.header import read_header

    with open(rhs_path, "rb") as fid:
        with contextlib.redirect_stdout(io.StringIO()):
            header = read_header(fid)
        fs = float(header["sample_rate"])
        bytes_per_block = get_bytes_per_data_block(header)
        bytes_remaining = os.path.getsize(rhs_path) - fid.tell()
        if bytes_remaining <= 0:
            raise RuntimeError(f"RHS file contains no data: {rhs_path}")
        if bytes_remaining % bytes_per_block != 0:
            raise RuntimeError(f"Invalid RHS file size: {rhs_path}")
        num_blocks = int(bytes_remaining / bytes_per_block)
        num_samples = int(calculate_num_samples(header, num_blocks))
    if num_samples < 2:
        raise RuntimeError(f"Recording too short: {rhs_path}")
    return num_samples, fs


def validate_section_trigger_window(
    section_duration_s: float,
    trigger_start_s: float,
    trigger_end_s: float,
) -> None:
    """Raise if the imaginary trigger window falls outside a segment."""
    if trigger_start_s < 0:
        raise ValueError(
            f"Imaginary trigger start ({trigger_start_s:g} s) must be >= 0 within each segment."
        )
    if trigger_end_s > section_duration_s:
        raise ValueError(
            f"Imaginary trigger end ({trigger_end_s:g} s) exceeds segment duration "
            f"({section_duration_s:g} s)."
        )
    if trigger_start_s >= trigger_end_s:
        raise ValueError(
            f"Imaginary trigger start ({trigger_start_s:g} s) must be strictly before "
            f"end ({trigger_end_s:g} s)."
        )


def no_trigger_sections_and_timebase(
    n_samples: int,
    fs: float,
    section_count: int,
    section_duration_s: float | None,
    section_spec: SectionSpecKind,
    section_trigger_start_s: float,
    section_trigger_end_s: float,
) -> tuple[np.ndarray, np.ndarray, int, int, int, float]:
    """Split recording into sections; average an imaginary trigger window in each."""
    if n_samples < 2:
        raise RuntimeError("Recording too short for section averaging.")
    total_s = float(n_samples) / float(fs)
    if section_spec == "duration":
        if section_duration_s is None or float(section_duration_s) <= 0:
            raise ValueError("Section duration (s) must be > 0.")
        n_sections = max(1, int(total_s / float(section_duration_s)))
    else:
        if int(section_count) < 1:
            raise ValueError("Section count must be >= 1.")
        n_sections = int(section_count)
    section_len = int(n_samples // n_sections)
    if section_len < 2:
        raise RuntimeError(
            f"Too many sections ({n_sections}) for recording length ({total_s:.3f} s)."
        )
    n_sections = int(n_samples // section_len)
    section_duration_resolved = section_len / float(fs)
    validate_section_trigger_window(
        section_duration_resolved,
        float(section_trigger_start_s),
        float(section_trigger_end_s),
    )
    start_n = int(round(float(section_trigger_start_s) * fs))
    end_n = int(round(float(section_trigger_end_s) * fs))
    start_n = max(0, min(start_n, section_len - 1))
    end_n = max(start_n + 1, min(end_n, section_len))
    window_n = end_n - start_n
    if window_n < 2:
        raise RuntimeError("Imaginary trigger window too short (< 2 samples).")
    starts = np.arange(n_sections, dtype=np.int64) * section_len
    triggers = starts + start_n
    if np.any(triggers + window_n > n_samples):
        raise RuntimeError(
            "Imaginary trigger window extends beyond the recording for at least one section."
        )
    pre_n = 0
    post_n = window_n
    t_rel = np.arange(0, window_n, dtype=np.float64) / float(fs)
    return triggers, t_rel, pre_n, post_n, n_sections, section_duration_resolved


def resolve_recording_windows(
    config: AnalysisConfig,
    n_samples: int,
    fs: float,
    analog_in0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, float | None]:
    """Return segmentation windows: triggers or fixed sections."""
    if config.edge == "none":
        starts, t_rel, pre_n, post_n, n_sections, _section_duration_s = no_trigger_sections_and_timebase(
            n_samples=n_samples,
            fs=fs,
            section_count=config.section_count,
            section_duration_s=config.section_duration_s,
            section_spec=config.section_spec,
            section_trigger_start_s=config.section_trigger_start_s,
            section_trigger_end_s=config.section_trigger_end_s,
        )
        return starts, t_rel, pre_n, post_n, n_sections, n_sections, None

    trigger_indices = detect_edges(analog_in0, threshold=config.threshold, edge=config.edge)
    if trigger_indices.size == 0:
        edge_fr = "falling" if config.edge == "falling" else "rising"
        raise RuntimeError(f"No {edge_fr} edge detected on ANALOG_IN 0.")
    valid_triggers, t_rel, pre_n, post_n = valid_triggers_and_timebase(
        n_samples=n_samples,
        trigger_indices=trigger_indices,
        fs=fs,
        pre_s=config.pre_s,
        post_s=config.post_s,
    )
    end_rising_rel_s = mean_time_to_next_rising_edge_s(
        analog_in0, trigger_indices, config.threshold, fs
    )
    return (
        valid_triggers,
        t_rel,
        pre_n,
        post_n,
        int(valid_triggers.size),
        int(trigger_indices.size),
        end_rising_rel_s,
    )


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
    """Per-trigger windows on Intan RHX HIGH waveforms (spikeplot / cpuinterface)."""

    def __init__(
        self,
        amplifier: np.ndarray,
        highpass: np.ndarray,
        valid_triggers: np.ndarray,
        pre_n: int,
        post_n: int,
        work_dir: Path | None,
        intan_dsp: IntanDspSettings,
    ) -> None:
        self.amplifier = amplifier
        self.highpass = highpass
        self.valid_triggers = np.asarray(valid_triggers, dtype=np.int64)
        self.pre_n = pre_n
        self.post_n = post_n
        self._offsets = np.arange(-pre_n, post_n, dtype=np.int64)
        self.work_dir = work_dir
        self.fs = float(intan_dsp.fs)
        self.intan_dsp = intan_dsp
        self._closed = False

    def high_trace_for_channel(self, ch: int) -> np.ndarray:
        return np.asarray(self.highpass[ch], dtype=np.float64)

    def windows_2d_for_channel(self, ch: int) -> np.ndarray:
        channel_trace = self.high_trace_for_channel(ch)
        trial_count = int(self.valid_triggers.size)
        window_length = int(self._offsets.size)
        if trial_count * window_length > 25_000_000:
            raise RuntimeError(
                "Spike window too large for 2D RAM extraction. "
                "Reduce pre/post window or spike display sampling (%)."
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
        """Detect spikes per trial on HIGH (Intan cpuinterface, no hoops)."""
        del refractory_s  # Intan uses snippet_size refractory instead.
        from dataclasses import replace

        channel_trace = self.high_trace_for_channel(ch)
        dsp = replace(self.intan_dsp, spike_threshold_uv=float(threshold))
        spike_times_by_trial: list[np.ndarray] = []
        for trigger_index in self.valid_triggers:
            sample_start = int(trigger_index - self.pre_n)
            sample_end = int(trigger_index + self.post_n)
            spike_sample_indices = detect_spikes_intan(
                channel_trace,
                dsp,
                start_sample=sample_start,
                end_sample=sample_end,
            )
            rel = spike_sample_indices - int(trigger_index) + int(self.pre_n)
            rel = rel[(rel >= 0) & (rel < int(t_rel.size))]
            spike_times_by_trial.append(np.asarray(t_rel[rel], dtype=np.float64))
        return spike_times_by_trial

    def mean_rms_for_channel(self, ch: int) -> float:
        """RMS over the last 1 s of HIGH (Spike Scope, spikeplot.cpp)."""
        return mean_rms_intan_channel(self.high_trace_for_channel(ch), self.intan_dsp)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for arr in (self.amplifier, self.highpass):
            try:
                if isinstance(arr, np.memmap):
                    arr._mmap.close()
            except Exception:
                pass
        self.amplifier = np.empty((0,))
        self.highpass = np.empty((0,))
        if self.work_dir is not None and self.work_dir.exists():
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


def resolve_curve_filter(config: AnalysisConfig, fs: float) -> tuple[CurveFilterKind, float | None, float | None]:
    """Resolve and validate curve filter settings, with legacy low-pass fallback."""
    kind = str(config.curve_filter).strip().lower()
    if kind not in {"highpass", "lowpass", "bandpass", "no filter"}:
        raise ValueError(
            "Curve filter: expected one of highpass, lowpass, bandpass, no filter."
        )
    filter_kind: CurveFilterKind = kind  # type: ignore[assignment]
    low_hz = config.curve_filter_low_hz
    high_hz = config.curve_filter_high_hz

    # Backward compatibility for CLI / old configs.
    if filter_kind == "no filter" and config.lowpass_cutoff_hz is not None:
        filter_kind = "lowpass"
        low_hz = float(config.lowpass_cutoff_hz)
        high_hz = None

    nyq = 0.5 * float(fs)
    if filter_kind == "no filter":
        return filter_kind, None, None
    if filter_kind in {"highpass", "lowpass"}:
        if low_hz is None:
            raise ValueError(f"Curve filter '{filter_kind}': provide one cutoff frequency (Hz).")
        fc = float(low_hz)
        if fc <= 0:
            raise ValueError("Curve filter cutoff must be > 0 Hz.")
        if fc >= nyq:
            raise ValueError(f"Curve filter cutoff ({fc:g} Hz) must be < Nyquist ({nyq:.1f} Hz).")
        return filter_kind, fc, None
    if low_hz is None or high_hz is None:
        raise ValueError("Curve filter 'bandpass': provide both low and high cutoffs (Hz).")
    flo = float(low_hz)
    fhi = float(high_hz)
    if flo <= 0 or fhi <= 0:
        raise ValueError("Curve filter bandpass: both frequencies must be > 0 Hz.")
    if flo >= fhi:
        raise ValueError("Curve filter bandpass: low cutoff must be < high cutoff.")
    if fhi >= nyq:
        raise ValueError(f"Curve filter bandpass high cutoff ({fhi:g} Hz) must be < Nyquist ({nyq:.1f} Hz).")
    return filter_kind, flo, fhi


def mean_filtered_channelwise(
    amplifier_2d: np.ndarray,
    valid_triggers: np.ndarray,
    fs: float,
    pre_n: int,
    post_n: int,
    filter_kind: CurveFilterKind,
    low_hz: float | None,
    high_hz: float | None,
    channel_workers: int | None = None,
) -> np.ndarray:
    """Butterworth filter per channel, parallelized up to MAX_PARALLEL_CHANNELS."""
    channel_count, _ = amplifier_2d.shape
    window_length = pre_n + post_n
    mean_windows = np.zeros((channel_count, window_length), dtype=np.float64)

    def _compute_one_channel(c: int) -> tuple[int, np.ndarray]:
        check_analysis_cancelled()
        row_2d = amplifier_2d[c : c + 1]
        if filter_kind == "lowpass":
            if low_hz is None:
                raise ValueError("Curve filter low-pass requires cutoff frequency.")
            filtered_channel = apply_butterworth_lowpass(row_2d, fs, low_hz)[0]
        elif filter_kind == "highpass":
            if low_hz is None:
                raise ValueError("Curve filter high-pass requires cutoff frequency.")
            filtered_channel = apply_butterworth_highpass(row_2d, fs, low_hz)[0]
        elif filter_kind == "bandpass":
            if low_hz is None or high_hz is None:
                raise ValueError("Curve filter band-pass requires low/high frequencies.")
            filtered_channel = apply_butterworth_bandpass(row_2d, fs, low_hz, high_hz)[0]
        else:
            filtered_channel = np.asarray(row_2d[0], dtype=np.float64)
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


def build_intan_dsp_settings(data: dict[str, Any], config: AnalysisConfig) -> IntanDspSettings:
    """RHX-compatible DSP settings from RHS metadata + analysis config."""
    return IntanDspSettings.from_rhs_data(
        data,
        spike_threshold_uv=float(config.spike_threshold_uv),
        spike_filter_kind=config.intan_spike_filter_kind,  # type: ignore[arg-type]
        filter_order=int(config.intan_filter_order),
        filter_type=config.intan_filter_type,  # type: ignore[arg-type]
        filter_cutoff_hz=float(config.intan_filter_cutoff_hz),
        artifact_threshold_uv=float(config.intan_artifact_threshold_uv),
        artifact_suppression_enabled=bool(config.intan_artifact_suppression_enabled),
    )


def persist_intan_high_stack(
    amplifier_2d: np.ndarray,
    data: dict[str, Any],
    config: AnalysisConfig,
    work_dir: Path,
) -> tuple[Path, Path, IntanDspSettings]:
    """Write wideband + Intan HIGH mmaps under work_dir."""
    intan_dsp = build_intan_dsp_settings(data, config)
    amp_path = work_dir / "amplifier_raw.npy"
    high_path = work_dir / "high_intan.npy"
    persist_amplifier_float32(amplifier_2d, amp_path)
    intan_dsp.save_json(work_dir / "intan_dsp.json")
    check_analysis_cancelled()
    high_stack = compute_high_channel_stack(
        amplifier_2d,
        intan_dsp,
        config.channel_workers,
        cancel_check=check_analysis_cancelled,
    )
    np.save(high_path, high_stack)
    return amp_path, high_path, intan_dsp


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
    analog_in0 = get_analog_in0_signal(data) if uses_analog_trigger(config) else np.array([], dtype=np.float64)

    amplifier_raw = np.asarray(data.get("amplifier_data"))
    if amplifier_raw.size == 0:
        raise RuntimeError("RHS file has no amplifier_data.")

    _, n_samples = amplifier_raw.shape
    valid_triggers, t_rel, pre_n, post_n, n_valid, n_total, end_rising_rel_s = resolve_recording_windows(
        config=config,
        n_samples=n_samples,
        fs=fs,
        analog_in0=analog_in0,
    )
    check_analysis_cancelled()

    channel_names = get_channel_names(data, amplifier_raw.shape[0])

    # Raw mean in streaming mode (avoids large RAM allocations).
    mean_per_channel_raw = mean_triggered_windows_channelwise(
        amplifier_data=amplifier_raw,
        valid_triggers=valid_triggers,
        pre_n=pre_n,
        post_n=post_n,
        channel_workers=config.channel_workers,
    )

    curve_filter_kind, curve_filter_low_hz, curve_filter_high_hz = resolve_curve_filter(config, fs)
    if curve_filter_kind != "no filter":
        mean_per_channel = mean_filtered_channelwise(
            amplifier_raw,
            valid_triggers,
            fs,
            pre_n,
            post_n,
            curve_filter_kind,
            curve_filter_low_hz,
            curve_filter_high_hz,
            config.channel_workers,
        )
    else:
        mean_per_channel = mean_per_channel_raw

    work_dir = resolve_work_dir(config)
    amp_path, high_path, intan_dsp = persist_intan_high_stack(
        amplifier_raw, data, config, work_dir
    )

    del amplifier_raw
    if isinstance(data, dict):
        data.pop("amplifier_data", None)
    del data
    gc.collect()

    amp_mm = np.load(amp_path, mmap_mode="r")
    high_mm = np.load(high_path, mmap_mode="r")
    spike_source = AmplifierSpikeSource(
        amplifier=amp_mm,
        highpass=high_mm,
        valid_triggers=valid_triggers,
        pre_n=pre_n,
        post_n=post_n,
        work_dir=work_dir,
        intan_dsp=intan_dsp,
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
