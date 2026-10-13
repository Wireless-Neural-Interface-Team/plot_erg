"""Core analysis: Intan RHS loading, triggers, spike windows, mmap amplifier."""

from __future__ import annotations

import contextlib
import contextvars
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from collections.abc import Iterator
from typing import Any, Sequence

import numpy as np

from config import AnalysisConfig, SectionSpecKind
from intan_rhx_dsp import (
    effective_spike_threshold_uv,
    IntanDspSettings,
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


def validate_section_trigger_window(
    section_duration_s: float,
    trigger_start_s: float,
    trigger_end_s: float,
) -> None:
    """Raise if the imaginary trigger window falls outside a segment."""
    if trigger_start_s < 0:
        raise ValueError(
            f"Imaginary stimulation start ({trigger_start_s:g} s) must be >= 0 within each segment."
        )
    if trigger_end_s > section_duration_s:
        raise ValueError(
            f"Imaginary stimulation end ({trigger_end_s:g} s) exceeds segment duration "
            f"({section_duration_s:g} s)."
        )
    if trigger_start_s >= trigger_end_s:
        raise ValueError(
            f"Imaginary stimulation start ({trigger_start_s:g} s) must be strictly before "
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
    """Split recording into equal sections; t=0 is the imaginary stimulation onset.

    Each section is extracted in full. ``section_trigger_start_s`` / ``_end_s``
    place t=0 and the stimulation-end marker inside the section; they do not
    crop the plotted window.
    """
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
    # Garder le nombre demandé. Le reste (n_samples % section_len) ne forme pas
    # une section complète : le recalculer ici ajoutait une section en trop
    # dès que n_samples // section_len > section_count.
    section_duration_resolved = section_len / float(fs)
    validate_section_trigger_window(
        section_duration_resolved,
        float(section_trigger_start_s),
        float(section_trigger_end_s),
    )
    start_n = int(round(float(section_trigger_start_s) * fs))
    start_n = max(0, min(start_n, section_len - 1))
    starts = np.arange(n_sections, dtype=np.int64) * section_len
    triggers = starts + start_n
    pre_n = int(start_n)
    post_n = int(section_len - start_n)
    if pre_n + post_n < 2:
        raise RuntimeError("Section window too short (< 2 samples).")
    if np.any(triggers - pre_n < 0) or np.any(triggers + post_n > n_samples):
        raise RuntimeError(
            "Section window extends beyond the recording for at least one section."
        )
    t_rel = np.arange(-pre_n, post_n, dtype=np.float64) / float(fs)
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
        end_rel_s = float(config.section_trigger_end_s) - float(config.section_trigger_start_s)
        if end_rel_s <= 0:
            end_rel_s = None
        return starts, t_rel, pre_n, post_n, n_sections, n_sections, end_rel_s

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
    end_rising_rel_s = mean_time_to_pulse_end_s(
        analog_in0, valid_triggers, config.threshold, fs, trigger_edge=config.edge
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


def mean_time_to_pulse_end_s(
    signal: np.ndarray,
    trigger_indices: np.ndarray,
    threshold: float,
    fs: float,
    *,
    trigger_edge: str = "falling",
) -> float | None:
    """Mean time (s) from each trigger to the opposite edge (typical pulse end).

    Falling trigger → next rising (active-high pulse end).
    Rising trigger → next falling (active-low / rising-onset pulse end).
    """
    trigger_indices = np.asarray(trigger_indices, dtype=np.int64)
    end_edge = "rising" if trigger_edge == "falling" else "falling"
    end_idx = detect_edges(signal, threshold, end_edge)
    if end_idx.size == 0 or trigger_indices.size == 0:
        return None
    positions = np.searchsorted(end_idx, trigger_indices + 1, side="left")
    valid = positions < end_idx.size
    if not np.any(valid):
        return None
    deltas = (end_idx[positions[valid]] - trigger_indices[valid]).astype(np.float64) / float(fs)
    return float(np.mean(deltas))


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


def resolve_work_dir(config: AnalysisConfig) -> Path:
    """Directory for amplifier_raw.npy and intermediate files."""
    if config.work_dir is not None:
        return config.work_dir
    root = config.save_dir if config.save_dir is not None else config.rhs_file.parent
    return root / ".plot_erg" / f"{config.rhs_file.stem}_work"


class AmplifierSpikeSource:
    """Per-trigger windows on Intan RHX waveforms (spikes on HIGH / high-pass)."""

    def __init__(
        self,
        amplifier: np.ndarray,
        highpass: np.ndarray,
        lowpass: np.ndarray,
        valid_triggers: np.ndarray,
        pre_n: int,
        post_n: int,
        work_dir: Path | None,
        intan_dsp: IntanDspSettings,
    ) -> None:
        self.amplifier = amplifier
        self.highpass = highpass
        self.lowpass = lowpass
        self.valid_triggers = np.asarray(valid_triggers, dtype=np.int64)
        self.pre_n = pre_n
        self.post_n = post_n
        self._offsets = np.arange(-pre_n, post_n, dtype=np.int64)
        self.work_dir = work_dir
        self.fs = float(intan_dsp.fs)
        self.intan_dsp = intan_dsp
        self._closed = False

    def _high_row(self, ch: int) -> np.ndarray:
        """1D mmap/view for one high-pass channel (no full-row copy)."""
        return self.highpass[ch]

    def _low_row(self, ch: int) -> np.ndarray:
        """1D mmap/view for one low-pass channel (no full-row copy)."""
        return self.lowpass[ch]

    def _amp_row(self, ch: int) -> np.ndarray:
        """1D mmap/view for one amplifier channel (no full-row copy)."""
        return self.amplifier[ch]

    def high_trace_for_channel(self, ch: int) -> np.ndarray:
        return self._high_row(ch)


    def spike_times_per_trial_for_channel(
        self,
        ch: int,
        t_rel: np.ndarray,
        threshold: float,
        refractory_s: float = 0.001,
        t_range_s: tuple[float, float] | None = None,
        trigger_index: int | None = None,
    ) -> list[np.ndarray]:
        """Detect spikes per trial on high-pass trace, optionally limited to ``t_range_s``.

        ``t_range_s`` is (t0, t1) in seconds relative to stimulation. When omitted,
        detection uses the full pre/post trial window.
        ``trigger_index`` restricts detection to one stimulation (0 = first, 1 = second).
        """
        del refractory_s  # Intan uses snippet_size refractory instead.
        from dataclasses import replace

        row = self._high_row(ch)
        dsp = replace(self.intan_dsp, spike_threshold_uv=float(threshold))
        fs = float(self.fs)
        if t_range_s is not None:
            win_t0, win_t1 = float(t_range_s[0]), float(t_range_s[1])
        else:
            win_t0, win_t1 = None, None
        all_triggers = np.asarray(self.valid_triggers, dtype=np.int64)
        n_trig = int(all_triggers.size)
        empty = np.empty(0, dtype=np.float64)
        if n_trig == 0:
            return []
        if trigger_index is not None:
            sel = int(trigger_index)
            if sel < 0 or sel >= n_trig:
                return [empty.copy() for _ in range(n_trig)]
            trial_indices = [sel]
        else:
            trial_indices = list(range(n_trig))
        spike_times_by_trial: list[np.ndarray] = [empty.copy() for _ in range(n_trig)]
        for trial_i in trial_indices:
            trigger_index_abs = int(all_triggers[trial_i])
            if win_t0 is None or win_t1 is None or win_t1 <= win_t0:
                sample_start = int(trigger_index_abs - self.pre_n)
                sample_end = int(trigger_index_abs + self.post_n)
            else:
                sample_start = int(trigger_index_abs + int(round(win_t0 * fs)))
                sample_end = int(trigger_index_abs + int(round(win_t1 * fs)))
                sample_start = max(sample_start, int(trigger_index_abs - self.pre_n))
                sample_end = min(sample_end, int(trigger_index_abs + self.post_n))
            if sample_end - sample_start < 2:
                continue
            spike_sample_indices = detect_spikes_intan(
                row,
                dsp,
                start_sample=sample_start,
                end_sample=sample_end,
            )
            rel = spike_sample_indices - int(trigger_index_abs) + int(self.pre_n)
            rel = rel[(rel >= 0) & (rel < int(t_rel.size))]
            times = np.asarray(t_rel[rel], dtype=np.float64)
            if win_t0 is not None and win_t1 is not None:
                times = times[(times >= win_t0) & (times <= win_t1)]
            spike_times_by_trial[trial_i] = times
        return spike_times_by_trial

    def mean_rms_for_channel(self, ch: int) -> float:
        """RMS over the last 1 s of high-pass trace (Spike Scope, spikeplot.cpp)."""
        return mean_rms_intan_channel(self._high_row(ch), self.intan_dsp)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name in ("amplifier", "highpass", "lowpass"):
            arr = getattr(self, name)
            setattr(self, name, np.empty((0,)))
            if isinstance(arr, np.memmap):
                try:
                    arr._mmap.close()
                except Exception:
                    pass
            elif hasattr(arr, "close"):
                # LazyFilterBank (and similar) hold their own disk memmaps.
                try:
                    arr.close()
                except Exception:
                    pass
        # Keep ``work_dir`` on disk so the next CLI run can reuse HP/LP memmaps.
        # Callers that want a throwaway temp dir should delete it themselves.


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
        raise RuntimeError("No valid window around stimulations.")
    t_rel = np.arange(-pre_n, post_n, dtype=np.float64) / fs
    return valid_triggers, t_rel, pre_n, post_n


def resolve_channel_workers(channel_workers: int | None, n_channels: int) -> int:
    """Auto: use all CPU cores up to MAX_PARALLEL_CHANNELS (16)."""
    if n_channels <= 0:
        return 1
    if channel_workers is not None:
        return max(1, min(int(channel_workers), int(MAX_PARALLEL_CHANNELS), int(n_channels)))
    n_cpu = int(os.cpu_count() or 2)
    return max(1, min(int(n_channels), int(MAX_PARALLEL_CHANNELS), n_cpu))


def gui_channel_workers(channel_workers: int | None, n_channels: int = MAX_PARALLEL_CHANNELS) -> int:
    """Cap parallelism so the Qt event loop keeps a free core.

    Heavy ThreadPoolExecutor work otherwise saturates every CPU and starves the
    GUI thread via the GIL, even when computation runs in a QThread.
    """
    n_cpu = int(os.cpu_count() or 2)
    budget = max(1, min(4, n_cpu - 1))
    resolved = resolve_channel_workers(channel_workers, max(1, int(n_channels)))
    return max(1, min(resolved, budget))


def build_intan_dsp_settings(data: dict[str, Any], config: AnalysisConfig) -> IntanDspSettings:
    """RHX-compatible DSP settings from RHS metadata + analysis config."""
    return IntanDspSettings.from_rhs_data(
        data,
        spike_threshold_uv=effective_spike_threshold_uv(
            config.spike_threshold_uv,
            config.spike_threshold_polarity,
        ),
        hp_filter_order=int(config.intan_hp_filter_order),
        hp_filter_type=config.intan_hp_filter_type,  # type: ignore[arg-type]
        hp_filter_cutoff_hz=float(config.intan_hp_filter_cutoff_hz),
        lp_filter_order=int(config.intan_lp_filter_order),
        lp_filter_type=config.intan_lp_filter_type,  # type: ignore[arg-type]
        lp_filter_cutoff_hz=float(config.intan_lp_filter_cutoff_hz),
        artifact_threshold_uv=float(config.intan_artifact_threshold_uv),
        artifact_suppression_enabled=bool(config.intan_artifact_suppression_enabled),
        rms_window_s=float(config.rms_window_s),
        software_notch_hz=float(getattr(config, "software_notch_hz", 0) or 0),
    )


def mean_triggered_windows_channelwise(
    amplifier_data: np.ndarray,
    valid_triggers: np.ndarray,
    pre_n: int,
    post_n: int,
    channel_workers: int | None = None,
    channels: Sequence[int] | None = None,
) -> np.ndarray:
    """Raw mean per channel (parallel, mmap-friendly row reads).

    ``channels`` restricts the pass to those indices. Other rows stay zero.
    """
    amplifier_data = np.asarray(amplifier_data)
    if amplifier_data.ndim != 2:
        raise RuntimeError("Unexpected amplifier_data shape (expected [n_channels, n_samples]).")

    n_ch, _ = amplifier_data.shape
    win_len = pre_n + post_n
    if win_len <= 1:
        raise RuntimeError("Invalid time window.")
    valid_triggers = np.asarray(valid_triggers, dtype=np.int64)
    if valid_triggers.size == 0:
        raise RuntimeError("No valid window around stimulations.")

    if channels is None:
        wanted = list(range(n_ch))
    else:
        wanted = []
        seen: set[int] = set()
        for raw in channels:
            ch = int(raw)
            if 0 <= ch < n_ch and ch not in seen:
                seen.add(ch)
                wanted.append(ch)

    out = np.zeros((n_ch, win_len), dtype=np.float32)
    if not wanted:
        return out
    n_trig = float(valid_triggers.size)

    def _compute_one_channel(c: int) -> tuple[int, np.ndarray]:
        check_analysis_cancelled()
        row = np.asarray(amplifier_data[c], dtype=np.float32)
        acc = np.zeros(win_len, dtype=np.float32)
        for trig in valid_triggers:
            start = int(trig - pre_n)
            end = int(trig + post_n)
            acc += row[start:end]
        return c, acc / n_trig

    worker_count = resolve_channel_workers(channel_workers, len(wanted))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for c, mean_row in pool.map(_compute_one_channel, wanted):
            out[c] = mean_row
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
