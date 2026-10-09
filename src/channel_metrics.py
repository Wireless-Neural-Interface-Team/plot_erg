"""Channel-level metrics (RMS profiles, spike thresholds) without matplotlib.

Kept separate from ``plotting`` so GUI ``ensure_channels`` never imports Agg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from intan_rhx_dsp import (
    effective_spike_threshold_uv,
    normalize_spike_threshold,
    sliding_rms_intan_profile_range,
)

if TYPE_CHECKING:
    from core import AmplifierSpikeSource


def spike_threshold_caption(
    threshold_uv: float,
    polarity: str | None = None,
) -> str:
    mag, pol = normalize_spike_threshold(threshold_uv, polarity)  # type: ignore[arg-type]
    if pol == "positive":
        return f"threshold +{mag:g} µV (above, rising edge)"
    return f"threshold −{mag:g} µV (below, falling edge)"


def resolve_channel_spike_threshold(
    *,
    mode: str,
    fixed_threshold_uv: float,
    spike_threshold_polarity: str = "negative",
    rms_multiplier: float,
    source: AmplifierSpikeSource | None,
    channel_index: int,
    mean_rms_uv: float | None = None,
) -> tuple[float, str]:
    """Resolve spike threshold value and caption for one channel."""
    pol = spike_threshold_polarity
    if str(mode).strip().lower() != "rms_multiple":
        eff = effective_spike_threshold_uv(fixed_threshold_uv, pol)  # type: ignore[arg-type]
        return eff, spike_threshold_caption(fixed_threshold_uv, pol)
    if mean_rms_uv is None:
        if source is None:
            eff = effective_spike_threshold_uv(fixed_threshold_uv, pol)  # type: ignore[arg-type]
            return eff, spike_threshold_caption(fixed_threshold_uv, pol)
        mean_rms_uv = source.mean_rms_for_channel(int(channel_index))
    magnitude_uv = float(rms_multiplier) * float(mean_rms_uv)
    eff = effective_spike_threshold_uv(magnitude_uv, pol)  # type: ignore[arg-type]
    direction = "below" if pol == "negative" else "above"
    return eff, (
        f"{rms_multiplier:g}x RMS mean/channel "
        f"({magnitude_uv:g} µV, {direction})"
    )


def mean_trial_windows(
    row: np.ndarray, triggers: np.ndarray, pre_n: int, post_n: int
) -> np.ndarray:
    """Mean of peri-stimulus windows (vectorized over trials)."""
    win = int(pre_n + post_n)
    if triggers.size == 0 or win <= 0:
        return np.zeros(max(win, 0), dtype=np.float32)
    trigs = np.asarray(triggers, dtype=np.int64)
    starts = trigs - int(pre_n)
    offsets = np.arange(win, dtype=np.int64)
    index = starts[:, None] + offsets[None, :]
    stacked = np.asarray(row, dtype=np.float64)[index]
    return np.asarray(stacked.mean(axis=0), dtype=np.float32)


def mean_rms_profile_from_row(
    row: np.ndarray,
    *,
    fs: float,
    valid_triggers: np.ndarray,
    t0_s: float,
    t1_s: float,
    dsp: Any,
    trigger_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean RMS profile for one already-materialized high-pass row."""
    if valid_triggers.size == 0 or t1_s <= t0_s:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    start_off = int(round(float(t0_s) * fs))
    end_off = int(round(float(t1_s) * fs))
    if end_off <= start_off:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    n_samples = int(np.asarray(row).shape[-1])
    n_win = int(end_off - start_off)
    t_axis = np.arange(start_off, end_off, dtype=np.float64) / fs

    all_triggers = np.asarray(valid_triggers, dtype=np.int64)
    if trigger_index is not None:
        sel = int(trigger_index)
        if sel < 0 or sel >= int(all_triggers.size):
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        candidate_trigs = all_triggers[sel : sel + 1]
    else:
        candidate_trigs = all_triggers

    pad = int(dsp.rms_window_samples)
    acc = np.zeros(n_win, dtype=np.float64)
    n_ok = 0
    row_arr = np.asarray(row)
    for trig in candidate_trigs:
        trig_i = int(trig)
        seg_start = int(trig_i + start_off)
        seg_end = int(trig_i + end_off)
        if seg_start < 0 or seg_end > n_samples:
            continue
        r0 = max(0, seg_start - pad)
        rms_seg = sliding_rms_intan_profile_range(row_arr, dsp, r0, seg_end)
        acc += rms_seg[seg_start - r0 : seg_end - r0]
        n_ok += 1
    if n_ok == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return t_axis, acc / float(n_ok)


def mean_rms_profile_from_source_window(
    source: AmplifierSpikeSource,
    t0_s: float,
    t1_s: float,
    rms_window_s: float,
    channel_index: int | None = None,
    trigger_index: int | None = None,
    *,
    hp_row: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean RMS profile in [t0_s, t1_s], averaged over triggers.

    When ``trigger_index`` is set, only that stimulation is used (no averaging).
    When ``hp_row`` is provided with a single ``channel_index``, the row is used
    directly (avoids re-hitting LazyFilterBank).
    """
    del rms_window_s  # Intan uses source.intan_dsp.rms_window_s (1 s).
    if source.valid_triggers.size == 0 or t1_s <= t0_s:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    if hp_row is not None and channel_index is not None:
        return mean_rms_profile_from_row(
            hp_row,
            fs=float(source.fs),
            valid_triggers=source.valid_triggers,
            t0_s=t0_s,
            t1_s=t1_s,
            dsp=source.intan_dsp,
            trigger_index=trigger_index,
        )

    fs = float(source.fs)
    start_off = int(round(float(t0_s) * fs))
    end_off = int(round(float(t1_s) * fs))
    if end_off <= start_off:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    n_channels = int(source.highpass.shape[0])
    n_samples = int(source.highpass.shape[1])
    ch_idx = int(channel_index) if channel_index is not None else None
    if ch_idx is not None and (ch_idx < 0 or ch_idx >= n_channels):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    n_win = int(end_off - start_off)
    t_axis = np.arange(start_off, end_off, dtype=np.float64) / fs

    all_triggers = np.asarray(source.valid_triggers, dtype=np.int64)
    if trigger_index is not None:
        sel = int(trigger_index)
        if sel < 0 or sel >= int(all_triggers.size):
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        candidate_trigs = [int(all_triggers[sel])]
    else:
        candidate_trigs = [int(trig) for trig in all_triggers]

    valid_trigs: list[int] = []
    for trig in candidate_trigs:
        start = int(trig + start_off)
        end = int(trig + end_off)
        if start < 0 or end > n_samples:
            continue
        valid_trigs.append(int(trig))
    if not valid_trigs:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    channel_indices: Sequence[int] = (
        [ch_idx] if ch_idx is not None else list(range(n_channels))
    )
    pad = source.intan_dsp.rms_window_samples
    acc = np.zeros(n_win, dtype=np.float64)
    n_ok = 0
    for trig in valid_trigs:
        seg_stack = np.empty((len(channel_indices), n_win), dtype=np.float64)
        for i, ch in enumerate(channel_indices):
            row = source.highpass[ch]
            seg_start = int(trig + start_off)
            seg_end = int(trig + end_off)
            r0 = max(0, seg_start - pad)
            rms_seg = sliding_rms_intan_profile_range(
                row, source.intan_dsp, r0, seg_end
            )
            seg_stack[i, :] = rms_seg[seg_start - r0 : seg_end - r0]
        acc += np.mean(seg_stack, axis=0)
        n_ok += 1
    if n_ok == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return t_axis, acc / float(n_ok)
