"""PDF figures for triggered averaged traces, spike raster/PSTH/ISI, comparisons, and optional MEA layout inset."""

from __future__ import annotations

import functools
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from core import (
    AmplifierSpikeSource,
    apply_butterworth_bandpass,
    apply_butterworth_lowpass,
    check_analysis_cancelled,
    detect_spikes_at_threshold,
)
from impedance_tracking import ImpedanceSession
from plot_utils import downsample_points, shift_axes_down, shorten_filename_for_windows
from probe_layout import draw_probe_layout_on_axes, load_probe_layout_json, match_contact_index

# Zoom panel window (s), time relative to trigger (t=0)
ZOOM_T0 = -0.1
ZOOM_T1 = 0.2

# ISI: only spikes within [-ISI_HALF_WINDOW_S, +ISI_HALF_WINDOW_S] (s relative to trigger)
ISI_HALF_WINDOW_S = 1.0

# X-axis label for all time-relative-to-trigger plots
TIME_REL_XLABEL = "Time relative to trigger (s)"
LEGEND_FONT_SIZE = 10
RMS_INTAN_LIKE_BANDPASS_LOW_HZ = 300.0
RMS_INTAN_LIKE_BANDPASS_HIGH_HZ = 7500.0
_PROFILE_ENABLED = os.environ.get("PLOT_ERG_PROFILE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_PROFILE_STATS: dict[str, tuple[float, int]] = {}


def _profile_record(name: str, elapsed_s: float) -> None:
    if not _PROFILE_ENABLED:
        return
    total, count = _PROFILE_STATS.get(name, (0.0, 0))
    _PROFILE_STATS[name] = (total + float(elapsed_s), count + 1)


def _profile_snapshot() -> dict[str, tuple[float, int]]:
    return dict(_PROFILE_STATS)


def _profile_print_delta(title: str, before: dict[str, tuple[float, int]], total_s: float) -> None:
    if not _PROFILE_ENABLED:
        return
    rows: list[tuple[str, float, int]] = []
    for key, (after_t, after_c) in _PROFILE_STATS.items():
        before_t, before_c = before.get(key, (0.0, 0))
        dt = after_t - before_t
        dc = after_c - before_c
        if dt > 0 and dc > 0:
            rows.append((key, dt, dc))
    rows.sort(key=lambda x: x[1], reverse=True)
    print(f"[PROFILE] {title}: total={total_s:.3f}s")
    for key, dt, dc in rows[:10]:
        print(f"[PROFILE]   {key}: {dt:.3f}s ({dc} calls, {dt / dc:.4f}s/call)")


def _profiled(name: str):
    def _deco(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            if not _PROFILE_ENABLED:
                return func(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                _profile_record(name, time.perf_counter() - t0)

        return _wrapped

    return _deco


def _lightweight_pdf_dpi(lightweight_mode: bool) -> int:
    """Lower DPI aggressively in lightweight mode to speed up rendering."""
    return 60 if lightweight_mode else 120


def _scale_page_size_for_lightweight(
    width_in: float,
    height_in: float,
    lightweight_mode: bool,
) -> tuple[float, float]:
    """Scale page size down when lightweight mode is enabled."""
    if not lightweight_mode:
        return width_in, height_in
    # Clamp to a strict lightweight envelope to keep rendering fast even
    # when upstream page sizes are very large.
    scaled_width_in = width_in * 0.55
    scaled_height_in = height_in * 0.45
    return (
        max(10.0, min(14.0, scaled_width_in)),
        max(20.0, min(42.0, scaled_height_in)),
    )


def _draw_mea_layout_panel(
    ax: Any,
    probe_layout: Any,
    channel_name: str,
) -> None:
    """Draw MEA layout in a dedicated stacked panel (no inset)."""
    if probe_layout is None or match_contact_index(probe_layout, channel_name) is None:
        ax.axis("off")
        ax.text(
            0.02,
            0.5,
            "MEA layout unavailable for this channel",
            ha="left",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )
        return
    draw_probe_layout_on_axes(ax, probe_layout, channel_name)
    ax.set_title(f"MEA layout — highlighted channel: {channel_name}", fontsize=10)


def _soften_figure_linewidths(fig: Any, scale: float = 0.85, min_width: float = 0.7) -> None:
    """Reduce line thickness globally for a figure."""
    for ax in fig.axes:
        for line in ax.get_lines():
            try:
                lw = float(line.get_linewidth())
            except Exception:
                continue
            line.set_linewidth(max(min_width, lw * scale))


def _spike_times_per_trial(
    windows_ch: np.ndarray,
    t_rel: np.ndarray,
    fs: float,
    threshold: float,
) -> list[np.ndarray]:
    """For one channel: list of spike-time arrays (s rel. trigger), one per trial."""
    n_trials = int(windows_ch.shape[0])
    out: list[np.ndarray] = []
    for i in range(n_trials):
        idx = detect_spikes_at_threshold(windows_ch[i], fs, threshold)
        out.append(np.asarray(t_rel[idx], dtype=np.float64))
    return out


def _psth_mean_hz(
    spike_times_per_trial: list[np.ndarray],
    t_rel: np.ndarray,
    n_trials: int,
    bin_width_s: float,
    smooth_sigma_s: float,
    t_range_s: Optional[Tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean PSTH (Hz): spike count per bin / (n_trials * bin_width).

    t_range_s: if (t0, t1), histogram only on this interval (out-of-range spikes excluded).
    """
    if t_range_s is not None:
        t0, t1 = float(t_range_s[0]), float(t_range_s[1])
    else:
        t0, t1 = float(t_rel[0]), float(t_rel[-1])
    if t1 <= t0 or bin_width_s <= 0:
        return np.array([]), np.array([])
    edges = np.arange(t0, t1 + bin_width_s, bin_width_s)
    if edges.size < 2:
        return np.array([]), np.array([])
    counts = np.zeros(edges.size - 1, dtype=np.float64)
    for st in spike_times_per_trial:
        if st.size == 0:
            continue
        counts += np.histogram(st, bins=edges)[0]
    rate = counts / (max(n_trials, 1) * bin_width_s)
    centers = (edges[:-1] + edges[1:]) * 0.5
    sigma_bins = smooth_sigma_s / bin_width_s if bin_width_s > 0 else 1.0
    sigma_bins = max(float(sigma_bins), 0.5)
    rate_s = gaussian_filter1d(rate, sigma=sigma_bins, mode="nearest")
    return centers, rate_s


def _mean_firing_rate_in_window_hz(
    spike_times_per_trial: list[np.ndarray],
    t_window: tuple[float, float],
) -> float:
    """Average firing rate (Hz) in a given time window."""
    t0, t1 = float(t_window[0]), float(t_window[1])
    if t1 <= t0:
        return 0.0
    n_trials = max(1, len(spike_times_per_trial))
    n_spikes = 0
    for st in spike_times_per_trial:
        st_arr = np.asarray(st, dtype=np.float64)
        n_spikes += int(np.sum((st_arr >= t0) & (st_arr <= t1)))
    return float(n_spikes) / (float(n_trials) * (t1 - t0))


def _add_psth_mean_table(ax_fr: Any, rows: list[tuple[str, float]]) -> None:
    """Render a compact PSTH mean-rate table below FR axis."""
    if not rows:
        return
    cell_text = [[name, f"{rate:.2f}"] for name, rate in rows]
    tbl = ax_fr.table(
        cellText=cell_text,
        colLabels=["Signal", "Mean FR (Hz)"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, -0.90, 1.0, 0.42],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.75)


def _trial_mean_firing_rate_hz(
    spike_times_per_trial: list[np.ndarray],
    t_window: tuple[float, float],
) -> np.ndarray:
    """Per-trial mean firing rate (Hz) in a given time window."""
    t0, t1 = float(t_window[0]), float(t_window[1])
    dur = max(1e-12, t1 - t0)
    out = np.zeros(len(spike_times_per_trial), dtype=np.float64)
    for i, st in enumerate(spike_times_per_trial):
        st_arr = np.asarray(st, dtype=np.float64)
        n_spikes = int(np.sum((st_arr >= t0) & (st_arr <= t1)))
        out[i] = float(n_spikes) / dur
    return out


def _spike_pipeline_captions(
    spike_bandpass_low_hz: Optional[float],
    spike_bandpass_high_hz: Optional[float],
) -> Tuple[str, str]:
    """(short for subtitles, detailed for footer note)"""
    if spike_bandpass_low_hz is not None and spike_bandpass_high_hz is not None:
        flo = float(spike_bandpass_low_hz)
        fhi = float(spike_bandpass_high_hz)
        short = f"band-pass {flo:g}–{fhi:g} Hz"
        detail = f"Butterworth band-pass {flo:g}–{fhi:g} Hz (order 4, filtfilt)"
        return short, detail
    return "raw", "raw mmap signal (no spike band-pass)"


def _isi_time_and_values_s(
    spike_times_per_trial: list[np.ndarray],
    *,
    isi_window_s: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interval end time (s rel. trigger) and ISI (s) for each consecutive pair."""
    if isi_window_s is None:
        lo, hi = -float(ISI_HALF_WINDOW_S), float(ISI_HALF_WINDOW_S)
    else:
        lo, hi = float(isi_window_s[0]), float(isi_window_s[1])
    tx: list[np.ndarray] = []
    dy: list[np.ndarray] = []
    for st in spike_times_per_trial:
        st = np.sort(np.asarray(st, dtype=np.float64))
        st = st[(st >= lo) & (st <= hi)]
        if st.size < 2:
            continue
        d = np.diff(st)
        t_end = st[1:]
        tx.append(t_end)
        dy.append(d)
    if not tx:
        return np.array([]), np.array([])
    return np.concatenate(tx), np.concatenate(dy)


def _concat_isi_s(
    spike_times_per_trial: list[np.ndarray],
    *,
    isi_window_s: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Within-trial ISI (s), concatenated intervals."""
    _, isi = _isi_time_and_values_s(spike_times_per_trial, isi_window_s=isi_window_s)
    return isi


def _set_adaptive_x_limits(
    ax: Any,
    x_values: np.ndarray,
    *,
    fallback_limits: tuple[float, float],
    pad_ratio: float = 0.06,
) -> None:
    """Set x-limits from displayed data with a small visual padding."""
    vals = np.asarray(x_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        ax.set_xlim(float(fallback_limits[0]), float(fallback_limits[1]))
        return
    x_min = float(np.min(vals))
    x_max = float(np.max(vals))
    if np.isclose(x_min, x_max):
        pad = max(abs(x_min) * 0.05, 1e-3)
    else:
        pad = (x_max - x_min) * max(float(pad_ratio), 0.0)
    ax.set_xlim(x_min - pad, x_max + pad)


def _draw_spike_panels_single_channel(
    ax_raster: Any,
    ax_fr: Any,
    ax_trial_fr: Any,
    ax_isi: Any,
    w_ch: Optional[np.ndarray],
    t_rel: np.ndarray,
    fs: float,
    spike_threshold_uv: float,
    firing_rate_window_s: float,
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
    *,
    t_range_s: Optional[Tuple[float, float]] = None,
    section_title: str = "",
    st_per_tr: Optional[list[np.ndarray]] = None,
    sampling_percent: int = 100,
) -> None:
    """Raster, PSTH / firing rate, ISI (time rel. trigger x duration) for one channel."""
    short, _ = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    if st_per_tr is None:
        if w_ch is None:
            raise ValueError("w_ch is required when st_per_tr is not provided.")
        st_per_tr = _spike_times_per_trial(w_ch, t_rel, fs, spike_threshold_uv)
    n_tr = int(len(st_per_tr))
    if t_range_s is None:
        t_xlim_lo, t_xlim_hi = float(t_rel[0]), float(t_rel[-1])
        psth_t_range: Optional[Tuple[float, float]] = None
        isi_window: Optional[Tuple[float, float]] = None
        isi_caption = f"spikes within ±{ISI_HALF_WINDOW_S:g} s of trigger, within-trial"
        isi_empty_hint = (
            "fewer than 2 spikes within ±"
            f"{ISI_HALF_WINDOW_S:g} s of trigger, per trial"
        )
    else:
        t_xlim_lo, t_xlim_hi = float(t_range_s[0]), float(t_range_s[1])
        psth_t_range = (t_xlim_lo, t_xlim_hi)
        isi_window = (t_xlim_lo, t_xlim_hi)
        isi_caption = f"spikes in [{t_xlim_lo:g}, {t_xlim_hi:g}] s relative to trigger, within-trial"
        isi_empty_hint = (
            f"fewer than 2 spikes in [{t_xlim_lo:g}, {t_xlim_hi:g}] s, per trial"
        )

    sec = f"{section_title} — " if section_title else ""
    for tri, st in enumerate(st_per_tr):
        st_plot = st
        if t_range_s is not None:
            st_plot = st[(st >= t_xlim_lo) & (st <= t_xlim_hi)]
        if st_plot.size:
            y_pts = np.full(st_plot.shape, tri)
            st_ds, y_ds = downsample_points(st_plot, y_pts, sampling_percent)
            ax_raster.scatter(
                st_ds,
                y_ds,
                s=4,
                c="k",
                alpha=0.75,
                linewidths=0,
            )
    ax_raster.set_ylabel("Trial #")
    cap = (
        f"crossing below {spike_threshold_uv:g} µV (falling edge)"
        if spike_threshold_uv < 0
        else f"crossing above {spike_threshold_uv:g} µV (rising edge)"
    )
    ax_raster.set_title(f"{sec}Raster — {short} ({cap}, 1 ms refractory)")
    ax_raster.grid(True, alpha=0.25, axis="x")
    ax_raster.set_ylim(-0.5, max(n_tr - 0.5, 0.5))
    ax_raster.set_xlim(t_xlim_lo, t_xlim_hi)

    bin_w = max(1.0 / fs, min(0.002, max(firing_rate_window_s / 12.0, 5e-5)))
    tc, rate_hz = _psth_mean_hz(
        st_per_tr,
        t_rel,
        n_tr,
        bin_w,
        firing_rate_window_s,
        t_range_s=psth_t_range,
    )
    if tc.size:
        ax_fr.plot(tc, rate_hz, linewidth=1.3, color="darkred", label="Smoothed PSTH (Hz)")
    ax_fr.set_ylabel("Rate (Hz)")
    ax_fr.set_title(
        f"{sec}Mean firing rate — {short} (Gaussian PSTH, σ={firing_rate_window_s:g} s)"
    )
    ax_fr.grid(True, alpha=0.3)
    ax_fr.set_xlim(t_xlim_lo, t_xlim_hi)
    ax_raster.set_xlabel(TIME_REL_XLABEL)
    ax_fr.set_xlabel(TIME_REL_XLABEL)
    fr_trials = _trial_mean_firing_rate_hz(st_per_tr, (t_xlim_lo, t_xlim_hi))
    x_trials = np.arange(1, len(fr_trials) + 1)
    if fr_trials.size:
        ax_trial_fr.plot(x_trials, fr_trials, color="teal", linewidth=1.1, marker="o", markersize=2.6)
    ax_trial_fr.set_title(f"{sec}Mean firing rate per trial — shown window")
    ax_trial_fr.set_xlabel("Trial index")
    ax_trial_fr.set_ylabel("Firing rate (Hz)")
    ax_trial_fr.grid(True, alpha=0.25)

    t_isi, isi_vals_s = _isi_time_and_values_s(st_per_tr, isi_window_s=isi_window)
    if t_isi.size:
        t_isi, isi_vals_s = downsample_points(t_isi, isi_vals_s, sampling_percent)
        ax_isi.scatter(
            t_isi,
            isi_vals_s * 1e3,
            s=10,
            c="steelblue",
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        ax_isi.set_ylabel("ISI (ms)")
        ax_isi.set_title(
            f"{sec}ISI — {short} ({isi_caption} ; x-axis = time of 2nd spike in pair)"
        )
        ax_isi.grid(True, alpha=0.25)
        _set_adaptive_x_limits(ax_isi, t_isi, fallback_limits=(t_xlim_lo, t_xlim_hi))
        ax_isi.set_xlabel(TIME_REL_XLABEL)
    else:
        ax_isi.text(
            0.5,
            0.5,
            "Not enough spikes for ISI\n(" + isi_empty_hint + ")",
            ha="center",
            va="center",
            transform=ax_isi.transAxes,
        )
        ax_isi.set_axis_off()


def _draw_spike_panels_dual_channel(
    ax_raster: Any,
    ax_fr: Any,
    ax_trial_fr: Any,
    ax_isi: Any,
    w_a: Optional[np.ndarray],
    w_b: Optional[np.ndarray],
    t_rel: np.ndarray,
    fs: float,
    spike_threshold_uv: float,
    firing_rate_window_s: float,
    label_a: str,
    label_b: str,
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
    *,
    t_range_s: Optional[Tuple[float, float]] = None,
    section_title: str = "",
    sta: Optional[list[np.ndarray]] = None,
    stb: Optional[list[np.ndarray]] = None,
    sampling_percent: int = 100,
) -> None:
    """Overlaid raster / PSTH / ISI for two recordings (same channel, same time axis)."""
    short, _ = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    if sta is None:
        if w_a is None:
            raise ValueError("w_a is required when sta is not provided.")
        sta = _spike_times_per_trial(w_a, t_rel, fs, spike_threshold_uv)
    if stb is None:
        if w_b is None:
            raise ValueError("w_b is required when stb is not provided.")
        stb = _spike_times_per_trial(w_b, t_rel, fs, spike_threshold_uv)
    n_a = int(len(sta))
    n_b = int(len(stb))

    if t_range_s is None:
        t_xlim_lo, t_xlim_hi = float(t_rel[0]), float(t_rel[-1])
        psth_t_range = None
        isi_window = None
        isi_caption = f"±{ISI_HALF_WINDOW_S:g} s of trigger, within-trial"
        isi_empty_hint = f"±{ISI_HALF_WINDOW_S:g} s of trigger"
    else:
        t_xlim_lo, t_xlim_hi = float(t_range_s[0]), float(t_range_s[1])
        psth_t_range = (t_xlim_lo, t_xlim_hi)
        isi_window = (t_xlim_lo, t_xlim_hi)
        isi_caption = f"[{t_xlim_lo:g}, {t_xlim_hi:g}] s rel. trigger, within-trial"
        isi_empty_hint = f"[{t_xlim_lo:g}, {t_xlim_hi:g}] s of trigger"

    sec = f"{section_title} — " if section_title else ""
    for tri, st in enumerate(sta):
        st_plot = st
        if t_range_s is not None:
            st_plot = st[(st >= t_xlim_lo) & (st <= t_xlim_hi)]
        if st_plot.size:
            y_pts = np.full(st_plot.shape, tri)
            st_ds, y_ds = downsample_points(st_plot, y_pts, sampling_percent)
            ax_raster.scatter(
                st_ds,
                y_ds,
                s=4,
                c="C0",
                alpha=0.75,
                linewidths=0,
                label=label_a if tri == 0 else "",
            )
    offset = n_a
    for tri, st in enumerate(stb):
        st_plot = st
        if t_range_s is not None:
            st_plot = st[(st >= t_xlim_lo) & (st <= t_xlim_hi)]
        if st_plot.size:
            y_pts = np.full(st_plot.shape, offset + tri)
            st_ds, y_ds = downsample_points(st_plot, y_pts, sampling_percent)
            ax_raster.scatter(
                st_ds,
                y_ds,
                s=4,
                c="C1",
                alpha=0.75,
                linewidths=0,
                label=label_b if tri == 0 else "",
            )
    ax_raster.set_ylabel("Trial # (A then B stacked)")
    cap = (
        f"threshold {spike_threshold_uv:g} µV (falling)"
        if spike_threshold_uv < 0
        else f"threshold {spike_threshold_uv:g} µV (rising)"
    )
    ax_raster.set_title(f"{sec}Raster — {short} ({cap})")
    ax_raster.grid(True, alpha=0.25, axis="x")
    ax_raster.set_ylim(-0.5, max(offset + n_b - 0.5, 0.5))
    ax_raster.set_xlim(t_xlim_lo, t_xlim_hi)
    ax_raster.axhline(offset - 0.5, color="0.5", linestyle="--", linewidth=0.8, alpha=0.7)

    bin_w = max(1.0 / fs, min(0.002, max(firing_rate_window_s / 12.0, 5e-5)))
    tc_a, rate_a = _psth_mean_hz(
        sta, t_rel, n_a, bin_w, firing_rate_window_s, t_range_s=psth_t_range
    )
    tc_b, rate_b = _psth_mean_hz(
        stb, t_rel, n_b, bin_w, firing_rate_window_s, t_range_s=psth_t_range
    )
    if tc_a.size:
        ax_fr.plot(tc_a, rate_a, linewidth=1.3, color="C0", label=label_a)
    if tc_b.size:
        ax_fr.plot(tc_b, rate_b, linewidth=1.3, color="C1", label=label_b)
    ax_fr.set_ylabel("Rate (Hz)")
    ax_fr.set_title(
        f"{sec}Firing rate (Gaussian PSTH σ={firing_rate_window_s:g} s) — {short}"
    )
    ax_fr.grid(True, alpha=0.3)
    ax_fr.set_xlim(t_xlim_lo, t_xlim_hi)
    ax_raster.set_xlabel(TIME_REL_XLABEL)
    ax_fr.set_xlabel(TIME_REL_XLABEL)
    fr_a = _trial_mean_firing_rate_hz(sta, (t_xlim_lo, t_xlim_hi))
    fr_b = _trial_mean_firing_rate_hz(stb, (t_xlim_lo, t_xlim_hi))
    xa = np.arange(1, len(fr_a) + 1)
    xb = np.arange(1, len(fr_b) + 1)
    if fr_a.size:
        ax_trial_fr.plot(xa, fr_a, color="C0", linewidth=1.1, marker="o", markersize=2.4, label=label_a)
    if fr_b.size:
        ax_trial_fr.plot(xb, fr_b, color="C1", linewidth=1.1, marker="o", markersize=2.4, label=label_b)
    ax_trial_fr.set_title(f"{sec}Mean firing rate per trial — shown window")
    ax_trial_fr.set_xlabel("Trial index")
    ax_trial_fr.set_ylabel("Firing rate (Hz)")
    ax_trial_fr.grid(True, alpha=0.25)

    t_a, isi_a_s = _isi_time_and_values_s(sta, isi_window_s=isi_window)
    t_b, isi_b_s = _isi_time_and_values_s(stb, isi_window_s=isi_window)
    if t_a.size or t_b.size:
        if t_a.size:
            t_a, isi_a_s = downsample_points(t_a, isi_a_s, sampling_percent)
        if t_b.size:
            t_b, isi_b_s = downsample_points(t_b, isi_b_s, sampling_percent)
        if t_a.size:
            ax_isi.scatter(
                t_a,
                isi_a_s * 1e3,
                s=10,
                c="C0",
                alpha=0.35,
                linewidths=0,
                label=label_a,
                rasterized=True,
            )
        if t_b.size:
            ax_isi.scatter(
                t_b,
                isi_b_s * 1e3,
                s=10,
                c="C1",
                alpha=0.35,
                linewidths=0,
                label=label_b,
                rasterized=True,
            )
        ax_isi.set_ylabel("ISI (ms)")
        ax_isi.set_title(
            f"{sec}ISI — {short} ({isi_caption} ; x-axis = time of 2nd spike in pair)"
        )
        ax_isi.grid(True, alpha=0.25)
        isi_time_union = np.concatenate([t_a, t_b]) if (t_a.size and t_b.size) else (t_a if t_a.size else t_b)
        _set_adaptive_x_limits(ax_isi, isi_time_union, fallback_limits=(t_xlim_lo, t_xlim_hi))
        ax_isi.set_xlabel(TIME_REL_XLABEL)
    else:
        ax_isi.text(
            0.5,
            0.5,
            f"Not enough spikes for ISI\n({isi_empty_hint})",
            ha="center",
            va="center",
            transform=ax_isi.transAxes,
        )
        ax_isi.set_axis_off()


def _draw_spike_panels_multi_channel(
    ax_raster: Any,
    ax_fr: Any,
    ax_trial_fr: Any,
    ax_isi: Any,
    windows_list: Optional[Sequence[np.ndarray]],
    t_rel: np.ndarray,
    fs: float,
    spike_threshold_uv: float,
    firing_rate_window_s: float,
    labels: Sequence[str],
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
    *,
    t_range_s: Optional[Tuple[float, float]] = None,
    section_title: str = "",
    spikes_per_recording: Optional[list[list[np.ndarray]]] = None,
    sampling_percent: int = 100,
) -> None:
    """Overlaid raster / PSTH / ISI for N recordings."""
    short, _ = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    if spikes_per_recording is None:
        if windows_list is None:
            raise ValueError("windows_list is required when spikes_per_recording is not provided.")
        spikes_per_recording = [
            _spike_times_per_trial(w, t_rel, fs, spike_threshold_uv) for w in windows_list
        ]

    if t_range_s is None:
        t_xlim_lo, t_xlim_hi = float(t_rel[0]), float(t_rel[-1])
        psth_t_range = None
        isi_window = None
        isi_caption = f"±{ISI_HALF_WINDOW_S:g} s of trigger, within-trial"
        isi_empty_hint = f"±{ISI_HALF_WINDOW_S:g} s of trigger"
    else:
        t_xlim_lo, t_xlim_hi = float(t_range_s[0]), float(t_range_s[1])
        psth_t_range = (t_xlim_lo, t_xlim_hi)
        isi_window = (t_xlim_lo, t_xlim_hi)
        isi_caption = f"[{t_xlim_lo:g}, {t_xlim_hi:g}] s rel. trigger, within-trial"
        isi_empty_hint = f"[{t_xlim_lo:g}, {t_xlim_hi:g}] s of trigger"

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    sec = f"{section_title} — " if section_title else ""
    y_offset = 0
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        color = colors[rec_idx % len(colors)]
        for tri, st in enumerate(st_per_trial):
            st_plot = st
            if t_range_s is not None:
                st_plot = st[(st >= t_xlim_lo) & (st <= t_xlim_hi)]
            if st_plot.size:
                y_pts = np.full(st_plot.shape, y_offset + tri)
                st_ds, y_ds = downsample_points(st_plot, y_pts, sampling_percent)
                ax_raster.scatter(
                    st_ds,
                    y_ds,
                    s=4,
                    c=color,
                    alpha=0.75,
                    linewidths=0,
                    label=labels[rec_idx] if tri == 0 else "",
                )
        y_offset += len(st_per_trial)
        if rec_idx < len(spikes_per_recording) - 1:
            ax_raster.axhline(y_offset - 0.5, color="0.55", linestyle="--", linewidth=0.8, alpha=0.7)
    cap = (
        f"threshold {spike_threshold_uv:g} µV (falling)"
        if spike_threshold_uv < 0
        else f"threshold {spike_threshold_uv:g} µV (rising)"
    )
    ax_raster.set_ylabel("Trial # (grouped by file)")
    ax_raster.set_title(f"{sec}Raster — {short} ({cap})")
    ax_raster.grid(True, alpha=0.25, axis="x")
    ax_raster.set_ylim(-0.5, max(y_offset - 0.5, 0.5))
    ax_raster.set_xlim(t_xlim_lo, t_xlim_hi)

    bin_w = max(1.0 / fs, min(0.002, max(firing_rate_window_s / 12.0, 5e-5)))
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        tc, rate = _psth_mean_hz(
            st_per_trial,
            t_rel,
            max(len(st_per_trial), 1),
            bin_w,
            firing_rate_window_s,
            t_range_s=psth_t_range,
        )
        if tc.size:
            ax_fr.plot(tc, rate, linewidth=1.3, color=colors[rec_idx % len(colors)], label=labels[rec_idx])
    ax_fr.set_ylabel("Rate (Hz)")
    ax_fr.set_title(f"{sec}Firing rate (Gaussian PSTH σ={firing_rate_window_s:g} s) — {short}")
    ax_fr.grid(True, alpha=0.3)
    ax_fr.set_xlim(t_xlim_lo, t_xlim_hi)
    ax_raster.set_xlabel(TIME_REL_XLABEL)
    ax_fr.set_xlabel(TIME_REL_XLABEL)
    max_trials = 0
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        fr_trials = _trial_mean_firing_rate_hz(st_per_trial, (t_xlim_lo, t_xlim_hi))
        max_trials = max(max_trials, len(fr_trials))
        x = np.arange(1, len(fr_trials) + 1)
        if fr_trials.size:
            ax_trial_fr.plot(
                x,
                fr_trials,
                color=colors[rec_idx % len(colors)],
                linewidth=1.0,
                marker="o",
                markersize=2.2,
                label=labels[rec_idx],
            )
    ax_trial_fr.set_title(f"{sec}Mean firing rate per trial — shown window")
    ax_trial_fr.set_xlabel("Trial index")
    ax_trial_fr.set_ylabel("Firing rate (Hz)")
    ax_trial_fr.grid(True, alpha=0.25)
    if max_trials > 0:
        ax_trial_fr.set_xlim(1, max_trials)

    has_isi = False
    isi_time_chunks: list[np.ndarray] = []
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        tx, isi_vals_s = _isi_time_and_values_s(st_per_trial, isi_window_s=isi_window)
        if tx.size:
            has_isi = True
            tx, isi_vals_s = downsample_points(tx, isi_vals_s, sampling_percent)
            isi_time_chunks.append(np.asarray(tx, dtype=np.float64))
            ax_isi.scatter(
                tx,
                isi_vals_s * 1e3,
                s=10,
                c=colors[rec_idx % len(colors)],
                alpha=0.35,
                linewidths=0,
                label=labels[rec_idx],
                rasterized=True,
            )
    if has_isi:
        ax_isi.set_ylabel("ISI (ms)")
        ax_isi.set_title(f"{sec}ISI — {short} ({isi_caption} ; x-axis = time of 2nd spike)")
        ax_isi.grid(True, alpha=0.25)
        isi_time_union = np.concatenate(isi_time_chunks) if isi_time_chunks else np.empty(0, dtype=np.float64)
        _set_adaptive_x_limits(ax_isi, isi_time_union, fallback_limits=(t_xlim_lo, t_xlim_hi))
        ax_isi.set_xlabel(TIME_REL_XLABEL)
    else:
        ax_isi.text(
            0.5,
            0.5,
            f"Not enough spikes for ISI\n({isi_empty_hint})",
            ha="center",
            va="center",
            transform=ax_isi.transAxes,
        )
        ax_isi.set_axis_off()


def _draw_impedance_evolution_panel(
    ax_imp: Any,
    channel_name: str,
    sessions: Sequence[ImpedanceSession],
) -> None:
    """Semi-log evolution of |Z| @ 1 kHz for one channel vs session timestamps."""
    times_num = np.array([mdates.date2num(s.when) for s in sessions], dtype=np.float64)
    ys = np.array([s.magnitudes_ohm.get(channel_name, float("nan")) for s in sessions], dtype=np.float64)
    valid = np.isfinite(ys) & (ys > 0)
    if not np.any(valid):
        ax_imp.text(
            0.5,
            0.5,
            "No impedance data for this channel",
            ha="center",
            va="center",
            transform=ax_imp.transAxes,
            fontsize=10,
        )
        ax_imp.set_axis_off()
        return
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    valid_idx = np.flatnonzero(valid)
    point_colors = [default_colors[int(i) % len(default_colors)] for i in valid_idx]
    ax_imp.semilogy(
        times_num[valid],
        ys[valid],
        linestyle="None",
        marker="o",
        markersize=5,
        markeredgewidth=0.0,
        color="none",
    )
    ax_imp.scatter(
        times_num[valid],
        ys[valid],
        s=26,
        c=point_colors,
        alpha=0.95,
        edgecolors="none",
        zorder=3,
    )
    for x_val, y_val in zip(times_num[valid], ys[valid]):
        ax_imp.annotate(
            f"{y_val:.3e} Ω",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            va="bottom",
            fontsize=7,
            alpha=0.9,
            zorder=4,
        )
    ax_imp.set_ylabel("|Z| @ 1 kHz (Ω)", fontsize=8)
    ax_imp.set_xlabel("Session time (_YYMMDD_HHMMSS)", fontsize=8)
    valid_times = times_num[valid]
    day_start = np.floor(np.min(valid_times))
    day_end = np.ceil(np.max(valid_times))
    if day_end <= day_start:
        day_end = day_start + 1.0
    ax_imp.set_xlim(day_start, day_end)
    ax_imp.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_imp.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_imp.set_xticks(valid_times.tolist(), minor=True)
    ax_imp.tick_params(axis="both", labelsize=7)
    ax_imp.grid(True, which="major", alpha=0.35)
    ax_imp.grid(True, which="minor", alpha=0.12)
    for label in ax_imp.get_xticklabels():
        label.set_rotation(18)
        label.set_ha("right")


def _append_mean_impedance_summary_page(
    pdf: PdfPages,
    sessions: Sequence[ImpedanceSession],
    lightweight_mode: bool,
) -> None:
    """Final PDF page: mean |Z|@1 kHz averaged over CSV channels vs session time."""
    if not sessions:
        return
    check_analysis_cancelled()
    dpi = _lightweight_pdf_dpi(lightweight_mode)
    fig, ax = plt.subplots(figsize=(11, 6))
    times_num = np.array([mdates.date2num(s.when) for s in sessions], dtype=np.float64)
    means_z: list[float] = []
    stem_labels: list[str] = []
    for s in sessions:
        vals = np.asarray(list(s.magnitudes_ohm.values()), dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        means_z.append(float(np.mean(vals)) if vals.size > 0 else float("nan"))
        stem_labels.append(s.rhs_label[:50] + ("..." if len(s.rhs_label) > 50 else ""))

    means_arr = np.asarray(means_z, dtype=np.float64)
    valid = np.isfinite(means_arr) & (means_arr > 0)
    if not np.any(valid):
        ax.text(
            0.5,
            0.5,
            "No valid mean impedance across sessions",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_axis_off()
    else:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        valid_idx = np.flatnonzero(valid)
        point_colors = [default_colors[int(i) % len(default_colors)] for i in valid_idx]
        ax.semilogy(
            times_num[valid],
            means_arr[valid],
            linestyle="None",
            marker="o",
            markersize=6,
            markeredgewidth=0.0,
            color="none",
        )
        ax.scatter(
            times_num[valid],
            means_arr[valid],
            s=42,
            c=point_colors,
            alpha=0.95,
            edgecolors="none",
            zorder=3,
        )
        ax.set_title(
            "Recording-mean impedance |Z| @ 1 kHz\n(mean over channels in each recording)"
        )
        ax.set_ylabel("Mean |Z| (Ω), log scale")
        ax.set_xlabel("Session time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.12)
        for label in ax.get_xticklabels():
            label.set_rotation(15)
            label.set_ha("right")
        if int(np.count_nonzero(valid)) <= 12:
            for i in np.flatnonzero(valid):
                ax.annotate(
                    stem_labels[int(i)],
                    (times_num[int(i)], means_arr[int(i)]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=6,
                    alpha=0.85,
                )

    _soften_figure_linewidths(fig)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25, dpi=dpi)
    plt.close(fig)


@_profiled("rms_from_source_window")
def _mean_rms_profile_from_source_window(
    source: AmplifierSpikeSource,
    t0_s: float,
    t1_s: float,
    rms_window_s: float,
    channel_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean RMS profile in window [t0_s, t1_s], averaged over triggers.

    Intan-like preprocessing: per-channel DC removal + AP-like band-pass.
    """
    if source.valid_triggers.size == 0 or t1_s <= t0_s:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    fs = float(source.fs)
    start_off = int(round(float(t0_s) * fs))
    end_off = int(round(float(t1_s) * fs))
    if end_off <= start_off:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    bp_low = (
        float(source.bandpass_low_hz)
        if source.bandpass_low_hz is not None
        else RMS_INTAN_LIKE_BANDPASS_LOW_HZ
    )
    bp_high = (
        float(source.bandpass_high_hz)
        if source.bandpass_high_hz is not None
        else RMS_INTAN_LIKE_BANDPASS_HIGH_HZ
    )
    nyq = 0.5 * fs
    use_bandpass = bp_low > 0 and bp_high > bp_low and bp_high < nyq
    n_channels = int(source.amplifier.shape[0])
    n_samples = int(source.amplifier.shape[1])
    ch_idx = int(channel_index) if channel_index is not None else None
    if ch_idx is not None and (ch_idx < 0 or ch_idx >= n_channels):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    n_win = int(end_off - start_off)
    rms_n = max(1, int(round(float(rms_window_s) * fs)))
    t_axis = np.arange(start_off, end_off, dtype=np.float64) / fs

    valid_trigs: list[int] = []
    for trig in source.valid_triggers:
        start = int(trig + start_off)
        end = int(trig + end_off)
        if start < 0 or end > n_samples:
            continue
        valid_trigs.append(int(trig))
    if not valid_trigs:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    if ch_idx is not None:
        # Fast path: stack all trigger windows for one channel, then filter in batch.
        seg_stack = np.empty((len(valid_trigs), n_win), dtype=np.float64)
        for i, trig in enumerate(valid_trigs):
            start = int(trig + start_off)
            end = int(trig + end_off)
            seg_stack[i, :] = np.asarray(source.amplifier[ch_idx, start:end], dtype=np.float64)
        seg_stack = seg_stack - np.mean(seg_stack, axis=1, keepdims=True)
        if use_bandpass and n_win >= 32:
            try:
                seg_stack = apply_butterworth_bandpass(seg_stack, fs, bp_low, bp_high)
            except Exception:
                pass
        sq = seg_stack * seg_stack
        rms_trials = np.sqrt(uniform_filter1d(sq, size=rms_n, axis=1, mode="constant", cval=0.0))
        return t_axis, np.mean(rms_trials, axis=0)

    acc = np.zeros(n_win, dtype=np.float64)
    n_ok = 0
    for trig in valid_trigs:
        start = int(trig + start_off)
        end = int(trig + end_off)
        seg = np.asarray(source.amplifier[:, start:end], dtype=np.float64)
        if seg.size == 0 or seg.shape[1] != n_win:
            continue
        seg = seg - np.mean(seg, axis=1, keepdims=True)
        if use_bandpass and seg.shape[1] >= 32:
            try:
                seg = apply_butterworth_bandpass(seg, fs, bp_low, bp_high)
            except Exception:
                pass
        sq = seg * seg
        ch_rms = np.sqrt(uniform_filter1d(sq, size=rms_n, axis=1, mode="constant", cval=0.0))
        acc += np.mean(ch_rms, axis=0)
        n_ok += 1
    if n_ok == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return t_axis, acc / float(n_ok)


@_profiled("rms_from_windows_window")
def _mean_rms_profile_from_windows_window(
    windows: np.ndarray,
    t_rel: np.ndarray,
    t0_s: float,
    t1_s: float,
    rms_window_s: float,
    channel_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean RMS profile in window [t0_s, t1_s], averaged over trials.

    Intan-like preprocessing: per-channel DC removal + AP-like band-pass.
    """
    if windows is None or windows.ndim != 3 or windows.shape[0] == 0 or t1_s <= t0_s:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    start_idx = int(np.searchsorted(t_rel, float(t0_s), side="left"))
    end_idx = int(np.searchsorted(t_rel, float(t1_s), side="left"))
    start_idx = max(0, min(start_idx, int(windows.shape[2])))
    end_idx = max(0, min(end_idx, int(windows.shape[2])))
    if end_idx <= start_idx:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    seg = np.asarray(windows[:, :, start_idx:end_idx], dtype=np.float64)
    if seg.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    ch_idx = int(channel_index) if channel_index is not None else None
    if ch_idx is not None:
        if ch_idx < 0 or ch_idx >= seg.shape[1]:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        # Performance: in per-channel mode, process only one channel.
        seg = seg[:, ch_idx : ch_idx + 1, :]
    seg = seg - np.mean(seg, axis=2, keepdims=True)
    dt = float(np.median(np.diff(t_rel)))
    fs = 1.0 / dt if dt > 0 else 0.0
    rms_n = max(1, int(round(float(rms_window_s) * fs))) if fs > 0 else 1
    nyq = 0.5 * fs if fs > 0 else 0.0
    use_bandpass = (
        fs > 0
        and RMS_INTAN_LIKE_BANDPASS_LOW_HZ > 0
        and RMS_INTAN_LIKE_BANDPASS_HIGH_HZ > RMS_INTAN_LIKE_BANDPASS_LOW_HZ
        and RMS_INTAN_LIKE_BANDPASS_HIGH_HZ < nyq
        and seg.shape[2] >= 32
    )
    if use_bandpass:
        try:
            flat = seg.reshape(seg.shape[0] * seg.shape[1], seg.shape[2])
            flat = apply_butterworth_bandpass(
                flat, fs, RMS_INTAN_LIKE_BANDPASS_LOW_HZ, RMS_INTAN_LIKE_BANDPASS_HIGH_HZ
            )
            seg = flat.reshape(seg.shape[0], seg.shape[1], seg.shape[2])
        except Exception:
            pass
    sq = seg * seg
    rms_all = np.sqrt(uniform_filter1d(sq, size=rms_n, axis=2, mode="constant", cval=0.0))
    if ch_idx is not None:
        profile = np.mean(rms_all[:, 0, :], axis=0)
    else:
        profile = np.mean(np.mean(rms_all, axis=1), axis=0)
    return np.asarray(t_rel[start_idx:end_idx], dtype=np.float64), np.asarray(profile, dtype=np.float64)


def _slice_rms_profile_window(
    tx: np.ndarray,
    values: np.ndarray,
    t0_s: float,
    t1_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a precomputed RMS profile to [t0_s, t1_s]."""
    if tx.size == 0 or values.size == 0 or t1_s <= t0_s:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    mask = (tx >= float(t0_s)) & (tx <= float(t1_s))
    if not np.any(mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return np.asarray(tx[mask], dtype=np.float64), np.asarray(values[mask], dtype=np.float64)


def _plot_rms_series(
    ax: Any,
    rms_series: Sequence[tuple[str, np.ndarray, np.ndarray]],
    title: str,
    x_limits: tuple[float, float] | None = None,
) -> None:
    """Plot mean RMS profile (time in window) for one or many recordings."""
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    has_data = False
    for i, (label, tx, values) in enumerate(rms_series):
        if values.size == 0 or tx.size == 0:
            continue
        has_data = True
        ax.plot(
            tx,
            values,
            linewidth=1.35,
            color=colors[i % len(colors)],
            label=label,
        )
    if has_data:
        ax.set_title(title)
        ax.set_xlabel(TIME_REL_XLABEL)
        ax.set_ylabel("Mean RMS (µV)")
        ax.set_ylim(0.0, 10.0)
        if x_limits is not None:
            ax.set_xlim(float(x_limits[0]), float(x_limits[1]))
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "RMS evolution unavailable\n(no valid trigger window)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_axis_off()


def _append_mean_rms_evolution_page(
    pdf: PdfPages,
    rms_series: Sequence[tuple[str, np.ndarray, np.ndarray]],
    rms_window_s: float,
    lightweight_mode: bool,
) -> None:
    """Append one summary page: mean RMS profile on analysis timebase."""
    check_analysis_cancelled()
    fig_w, fig_h = _scale_page_size_for_lightweight(12.0, 6.0, lightweight_mode)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    has_data = False
    for i, (label, tx, values) in enumerate(rms_series):
        if values.size == 0 or tx.size == 0:
            continue
        has_data = True
        ax.plot(
            tx,
            values,
            linewidth=1.35,
            color=colors[i % len(colors)],
            label=label,
        )
    if has_data:
        ax.set_title(f"Mean RMS profile (RMS window = {rms_window_s:g} s)")
        ax.set_xlabel(TIME_REL_XLABEL)
        ax.set_ylabel("Mean RMS across channels (µV)")
        ax.set_ylim(0.0, 10.0)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "RMS evolution unavailable\n(no valid trigger window)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_axis_off()
    _soften_figure_linewidths(fig)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25, dpi=_lightweight_pdf_dpi(lightweight_mode))
    plt.close(fig)


def plot_channel_multi_comparison(
    t_rel: np.ndarray,
    means: Sequence[np.ndarray],
    channel_names: Sequence[str],
    output_dir: Path,
    labels: Sequence[str],
    pdf_title: Optional[str] = None,
    lowpass_cutoff_hz: Optional[float] = None,
    trigger_end_rising_rel_s_list: Optional[Sequence[Optional[float]]] = None,
    means_raw: Optional[Sequence[np.ndarray]] = None,
    spike_sources: Optional[Sequence[Optional[AmplifierSpikeSource]]] = None,
    fs: Optional[float] = None,
    spike_threshold_uv: float = -40.0,
    firing_rate_window_s: float = 0.025,
    rms_window_s: float = 0.050,
    zoom_t0_s: float = ZOOM_T0,
    zoom_t1_s: float = ZOOM_T1,
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
    lightweight_mode: bool = False,
    sampling_percent: int = 100,
    probe_layout_json: Optional[Path] = None,
    streaming_mode: bool = False,
    pre_n_common: Optional[int] = None,
    post_n_common: Optional[int] = None,
    impedance_sessions: Optional[Sequence[ImpedanceSession]] = None,
) -> Path:
    """Multi-page PDF: overlay of N recordings (same aligned channels)."""
    _profile_before = _profile_snapshot()
    _profile_t0 = time.perf_counter()
    n_records = len(labels)
    if n_records < 2:
        raise ValueError("plot_channel_multi_comparison requires at least 2 aligned recordings.")
    if not streaming_mode and (len(means) < 2 or len(means) != n_records):
        raise ValueError("Non-streaming mode: `means` must contain N recordings.")
    if streaming_mode and (spike_sources is None or len(spike_sources) != n_records):
        raise ValueError("Streaming mode: `spike_sources` must contain N recordings.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if pdf_title is not None and pdf_title.strip():
        safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in pdf_title.strip())
        pdf_stem = safe_title.removesuffix(".pdf")
    else:
        pdf_stem = "multi_comparison"
    pdf_name = shorten_filename_for_windows(output_dir, f"{pdf_stem}.pdf")
    pdf_path = output_dir / pdf_name

    zoom_t0, zoom_t1 = float(zoom_t0_s), float(zoom_t1_s)
    filt_note = f" — Butterworth low-pass {lowpass_cutoff_hz:g} Hz" if lowpass_cutoff_hz is not None else ""
    both_note = " — raw signal overlaid (filtered curve emphasized)" if lowpass_cutoff_hz is not None else ""
    zoom_title = f"Zoom: {zoom_t0:.1f} to {zoom_t1:.1f} s (relative to trigger){filt_note}{both_note}"
    n_channels = (
        min(src.amplifier.shape[0] for src in spike_sources if src is not None)
        if streaming_mode and spike_sources is not None
        else means[0].shape[0]
    )
    zmask = (t_rel >= zoom_t0) & (t_rel <= zoom_t1)
    end_markers = [v for v in (trigger_end_rising_rel_s_list or []) if v is not None]
    _has_spike_cmp = (
        fs is not None
        and spike_sources is not None
        and len(spike_sources) == n_records
        and all(src is not None for src in spike_sources)
    )
    _, spike_cmp_pipe = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

    probe_layout_loaded = None
    if probe_layout_json is not None:
        probe_layout_loaded = load_probe_layout_json(Path(probe_layout_json))
    with PdfPages(pdf_path) as pdf:
        for ch in range(n_channels):
            check_analysis_cancelled()
            channel_name = str(channel_names[ch])
            means_ch: list[np.ndarray] = []
            means_raw_ch: list[np.ndarray] | None = [] if lowpass_cutoff_hz is not None else None
            if streaming_mode:
                if pre_n_common is None or post_n_common is None:
                    raise ValueError("Streaming mode: pre_n_common/post_n_common are required.")
                win_len = int(pre_n_common) + int(post_n_common)
                for src in (spike_sources or []):
                    if src is None:
                        continue
                    row = np.asarray(src.amplifier[ch], dtype=np.float64)
                    acc_raw = np.zeros(win_len, dtype=np.float64)
                    for trig in src.valid_triggers:
                        start = int(trig - int(pre_n_common))
                        end = int(trig + int(post_n_common))
                        acc_raw += row[start:end]
                    y_raw = acc_raw / float(max(len(src.valid_triggers), 1))
                    if len(y_raw) != len(t_rel):
                        y_raw = np.asarray(y_raw[: len(t_rel)])
                    if lowpass_cutoff_hz is not None:
                        row_f = apply_butterworth_lowpass(row.reshape(1, -1), float(fs or src.fs), lowpass_cutoff_hz)[0]
                        acc_f = np.zeros(win_len, dtype=np.float64)
                        for trig in src.valid_triggers:
                            start = int(trig - int(pre_n_common))
                            end = int(trig + int(post_n_common))
                            acc_f += row_f[start:end]
                        y_f = acc_f / float(max(len(src.valid_triggers), 1))
                        if len(y_f) != len(t_rel):
                            y_f = np.asarray(y_f[: len(t_rel)])
                        means_ch.append(np.asarray(y_f))
                        assert means_raw_ch is not None
                        means_raw_ch.append(np.asarray(y_raw))
                    else:
                        means_ch.append(np.asarray(y_raw))
            else:
                means_ch = [np.asarray(means[i][ch]) for i in range(n_records)]
                if means_raw is not None and lowpass_cutoff_hz is not None:
                    means_raw_ch = [np.asarray(means_raw[i][ch]) for i in range(n_records)]

            page_width_in = 12.0
            mea_row_height_ratio = 1.60
            panel_height_ratios = [
                mea_row_height_ratio,
                1.70,
                1.60,
                1.30,
                1.20,
                1.15,
                1.05,
                1.15,
                0.06,
                1.70,
                1.50,
                1.20,
                1.30,
                1.15,
                1.05,
                1.15,
                0.06,
                1.70,
                1.50,
                1.20,
                1.30,
                1.15,
                1.05,
                1.15,
            ]
            if impedance_sessions:
                full_height_ratios = [*panel_height_ratios, 0.05, 0.95]
                page_height_in = 56.0
                page_width_in, page_height_in = _scale_page_size_for_lightweight(
                    page_width_in, page_height_in, lightweight_mode
                )
                fig = plt.figure(figsize=(page_width_in, page_height_in))
                gs = fig.add_gridspec(len(full_height_ratios), 1, height_ratios=full_height_ratios, hspace=0.70)
            else:
                page_height_in = 52.0
                page_width_in, page_height_in = _scale_page_size_for_lightweight(
                    page_width_in, page_height_in, lightweight_mode
                )
                fig = plt.figure(figsize=(page_width_in, page_height_in))
                gs = fig.add_gridspec(24, 1, height_ratios=panel_height_ratios, hspace=0.70)
            ax_mea = fig.add_subplot(gs[0, 0])
            _draw_mea_layout_panel(ax_mea, probe_layout_loaded, channel_name)
            ax_full = fig.add_subplot(gs[1, 0])
            ax_first_trigger = fig.add_subplot(gs[2, 0], sharex=ax_full)
            ax_full_rms = fig.add_subplot(gs[3, 0], sharex=ax_full)
            ax_raster_f = fig.add_subplot(gs[4, 0], sharex=ax_full)
            ax_fr_f = fig.add_subplot(gs[5, 0], sharex=ax_full)
            ax_trial_fr_f = fig.add_subplot(gs[6, 0])
            ax_isi_f = fig.add_subplot(gs[7, 0])
            ax_hdr2 = fig.add_subplot(gs[8, 0]); ax_hdr2.axis("off")
            ax_hdr2.text(0.02, 0.5, f"Part 2 — Zoomed view [{zoom_t0:.2f}, {zoom_t1:.2f}] s (relative to trigger)", ha="left", va="center", fontsize=11, fontweight="bold", transform=ax_hdr2.transAxes)
            ax_zoom = fig.add_subplot(gs[9, 0])
            ax_zoom_first = fig.add_subplot(gs[10, 0], sharex=ax_zoom)
            ax_zoom_rms = fig.add_subplot(gs[11, 0])
            ax_raster_z = fig.add_subplot(gs[12, 0], sharex=ax_zoom)
            ax_fr_z = fig.add_subplot(gs[13, 0], sharex=ax_zoom)
            ax_trial_fr_z = fig.add_subplot(gs[14, 0])
            ax_isi_z = fig.add_subplot(gs[15, 0])
            ax_hdr3 = fig.add_subplot(gs[16, 0]); ax_hdr3.axis("off")
            ax_hdr3.text(0.02, 0.5, "Part 3 — Trigger-end zoom (rising edge)", ha="left", va="center", fontsize=11, fontweight="bold", transform=ax_hdr3.transAxes)
            ax_zoom_end = fig.add_subplot(gs[17, 0])
            ax_zoom_end_first = fig.add_subplot(gs[18, 0], sharex=ax_zoom_end)
            ax_zoom_end_rms = fig.add_subplot(gs[19, 0])
            ax_raster_ze = fig.add_subplot(gs[20, 0], sharex=ax_zoom_end)
            ax_fr_ze = fig.add_subplot(gs[21, 0], sharex=ax_zoom_end)
            ax_trial_fr_ze = fig.add_subplot(gs[22, 0])
            ax_isi_ze = fig.add_subplot(gs[23, 0])

            if impedance_sessions:
                ax_imp_hdr = fig.add_subplot(gs[24, 0])
                ax_imp_hdr.axis("off")
                ax_imp_hdr.text(
                    0.02,
                    0.25,
                    "Part 4 — Impedance |Z| @ 1 kHz vs session",
                    ha="left",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    transform=ax_imp_hdr.transAxes,
                )
                ax_imp = fig.add_subplot(gs[25, 0])
                _draw_impedance_evolution_panel(ax_imp, channel_name, impedance_sessions)

            show_filtered_and_raw = (
                lowpass_cutoff_hz is not None
                and means_raw_ch is not None
                and len(means_raw_ch) == len(means_ch)
                and any(
                    not np.allclose(np.asarray(means_ch[i]), np.asarray(means_raw_ch[i]), rtol=0.0, atol=1e-9)
                    for i in range(len(means_ch))
                )
            )
            rms_series_full_multi: list[tuple[str, np.ndarray, np.ndarray]] = []
            rms_series_zoom_multi: list[tuple[str, np.ndarray, np.ndarray]] = []
            rms_series_zoom_end_multi: list[tuple[str, np.ndarray, np.ndarray]] = []
            if spike_sources is not None:
                t0_rms = float(t_rel[0]) if t_rel.size else 0.0
                t1_rms = float(t_rel[-1]) if t_rel.size else 0.0
                for i, src in enumerate(spike_sources):
                    if src is None:
                        continue
                    label = labels[i] if i < len(labels) else f"Recording {i + 1}"
                    tx_full, rms_full_vals = _mean_rms_profile_from_source_window(
                        src,
                        t0_rms,
                        t1_rms,
                        rms_window_s,
                        channel_index=ch,
                    )
                    rms_series_full_multi.append((label, tx_full, rms_full_vals))
                    tx_zoom, rms_zoom_vals = _slice_rms_profile_window(
                        tx_full,
                        rms_full_vals,
                        zoom_t0,
                        zoom_t1,
                    )
                    rms_series_zoom_multi.append((label, tx_zoom, rms_zoom_vals))
                    marker_i = None
                    if trigger_end_rising_rel_s_list is not None and i < len(trigger_end_rising_rel_s_list):
                        marker_i = trigger_end_rising_rel_s_list[i]
                    if marker_i is None:
                        rms_series_zoom_end_multi.append(
                            (label, np.array([], dtype=np.float64), np.array([], dtype=np.float64))
                        )
                    else:
                        tx_end, rms_zoom_end_vals = _slice_rms_profile_window(
                            tx_full,
                            rms_full_vals,
                            float(marker_i + zoom_t0),
                            float(marker_i + zoom_t1),
                        )
                        rms_series_zoom_end_multi.append((label, tx_end, rms_zoom_end_vals))
            first_trigger_curves: list[Optional[np.ndarray]] = []
            if spike_sources is not None:
                for src in spike_sources:
                    if src is None or src.valid_triggers.size == 0:
                        first_trigger_curves.append(None)
                        continue
                    first_trig = int(src.valid_triggers[0])
                    start = int(first_trig - src.pre_n)
                    end = int(first_trig + src.post_n)
                    first_curve = np.asarray(src.amplifier[ch, start:end], dtype=np.float64)
                    if first_curve.shape[0] != t_rel.shape[0]:
                        first_trigger_curves.append(None)
                        continue
                    first_trigger_curves.append(first_curve)
            for recording_index, recording_curve in enumerate(means_ch):
                line_color = colors[recording_index % len(colors)]
                if show_filtered_and_raw and means_raw_ch is not None:
                    raw_curve = np.asarray(means_raw_ch[recording_index])
                    ax_full.plot(t_rel, raw_curve, linewidth=1.0, color=line_color, alpha=0.35, label="_nolegend_")
                    ax_full.plot(t_rel, recording_curve, linewidth=1.35, color=line_color, label=f"{labels[recording_index]} (filtered)")
                    ax_zoom.plot(t_rel[zmask], raw_curve[zmask], linewidth=1.05, color=line_color, alpha=0.35)
                    ax_zoom.plot(t_rel[zmask], recording_curve[zmask], linewidth=1.35, color=line_color, label=labels[recording_index])
                else:
                    ax_full.plot(t_rel, recording_curve, linewidth=1.2, color=line_color, label=labels[recording_index])
                    ax_zoom.plot(t_rel[zmask], recording_curve[zmask], linewidth=1.3, color=line_color, label=labels[recording_index])
            ax_full.axvline(0.0, linestyle="--", linewidth=1.0, color="red", label="Trigger (onset)")
            for v in end_markers:
                ax_full.axvline(v, linestyle=":", linewidth=0.9, color="0.45")
                ax_zoom.axvline(v, linestyle=":", linewidth=0.9, color="0.45")
            ax_full.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zoom region")
            if end_markers:
                end_zoom_t0 = float(min(end_markers) + zoom_t0)
                end_zoom_t1 = float(max(end_markers) + zoom_t1)
                ax_full.axvspan(end_zoom_t0, end_zoom_t1, alpha=0.10, color="gold", label="Trigger-end zoom region")
            ax_full.set_title(f"Multi-comparison — {channel_name} (full view){filt_note}{both_note}")
            ax_full.set_ylabel("Potential (µV)")
            ax_full.set_xlabel(TIME_REL_XLABEL)
            ax_full.grid(True, alpha=0.3)
            ax_full.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=LEGEND_FONT_SIZE)

            if any(curve is not None for curve in first_trigger_curves):
                for recording_index, first_curve in enumerate(first_trigger_curves):
                    if first_curve is None:
                        continue
                    line_color = colors[recording_index % len(colors)]
                    ax_first_trigger.plot(
                        t_rel,
                        first_curve,
                        linewidth=1.1,
                        color=line_color,
                        label=f"{labels[recording_index]} first trigger raw",
                    )
                ax_first_trigger.axvline(0.0, linestyle="--", linewidth=1.0, color="red", label="Trigger (onset)")
                for v in end_markers:
                    ax_first_trigger.axvline(v, linestyle=":", linewidth=0.9, color="0.45")
                ax_first_trigger.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zoom region")
                if end_markers:
                    end_zoom_t0 = float(min(end_markers) + zoom_t0)
                    end_zoom_t1 = float(max(end_markers) + zoom_t1)
                    ax_first_trigger.axvspan(
                        end_zoom_t0,
                        end_zoom_t1,
                        alpha=0.10,
                        color="gold",
                        label="Trigger-end zoom region",
                    )
                ax_first_trigger.set_title("First trigger — raw signal (no averaging)")
                ax_first_trigger.set_ylabel("Potential (µV)")
                ax_first_trigger.set_xlabel(TIME_REL_XLABEL)
                ax_first_trigger.grid(True, alpha=0.3)
            else:
                ax_first_trigger.text(
                    0.5,
                    0.5,
                    "First trigger raw signal unavailable",
                    ha="center",
                    va="center",
                    transform=ax_first_trigger.transAxes,
                )
                ax_first_trigger.set_axis_off()
            _plot_rms_series(
                ax_full_rms,
                rms_series_full_multi,
                "Part 1 — RMS evolution (full window)",
                x_limits=(float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None,
            )

            ax_zoom.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
            ax_zoom.set_xlim(zoom_t0, zoom_t1)
            ax_zoom.set_title(zoom_title)
            ax_zoom.set_ylabel("Potential (µV)")
            ax_zoom.set_xlabel(TIME_REL_XLABEL)
            ax_zoom.grid(True, alpha=0.3)
            if any(curve is not None for curve in first_trigger_curves):
                for recording_index, first_curve in enumerate(first_trigger_curves):
                    if first_curve is None:
                        continue
                    line_color = colors[recording_index % len(colors)]
                    ax_zoom_first.plot(
                        t_rel[zmask],
                        first_curve[zmask],
                        linewidth=1.2,
                        color=line_color,
                        label=f"{labels[recording_index]} first trigger raw",
                    )
                ax_zoom_first.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
                for v in end_markers:
                    ax_zoom_first.axvline(v, linestyle=":", linewidth=0.9, color="0.45")
                ax_zoom_first.set_xlim(zoom_t0, zoom_t1)
                ax_zoom_first.set_title("Part 2 — First trigger raw (separate view)")
                ax_zoom_first.set_ylabel("Potential (µV)")
                ax_zoom_first.set_xlabel(TIME_REL_XLABEL)
                ax_zoom_first.grid(True, alpha=0.3)
            else:
                ax_zoom_first.text(0.5, 0.5, "First trigger raw signal unavailable", ha="center", va="center", transform=ax_zoom_first.transAxes)
                ax_zoom_first.set_axis_off()
            _plot_rms_series(
                ax_zoom_rms,
                rms_series_zoom_multi,
                f"Part 2 — RMS evolution (window [{zoom_t0:.3f}, {zoom_t1:.3f}] s)",
                x_limits=(zoom_t0, zoom_t1),
            )

            end_zoom_range: tuple[float, float] | None = None
            if end_markers:
                end_zoom_t0 = float(min(end_markers) + zoom_t0)
                end_zoom_t1 = float(max(end_markers) + zoom_t1)
                end_zoom_range = (end_zoom_t0, end_zoom_t1)
                end_mask = (t_rel >= end_zoom_t0) & (t_rel <= end_zoom_t1)
                for recording_index, recording_curve in enumerate(means_ch):
                    line_color = colors[recording_index % len(colors)]
                    if show_filtered_and_raw and means_raw_ch is not None:
                        raw_curve = np.asarray(means_raw_ch[recording_index])
                        ax_zoom_end.plot(t_rel[end_mask], raw_curve[end_mask], linewidth=1.05, color=line_color, alpha=0.35)
                        ax_zoom_end.plot(t_rel[end_mask], recording_curve[end_mask], linewidth=1.35, color=line_color, label=labels[recording_index])
                    else:
                        ax_zoom_end.plot(t_rel[end_mask], recording_curve[end_mask], linewidth=1.35, color=line_color, label=labels[recording_index])
                ax_zoom_end.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
                for v in end_markers:
                    ax_zoom_end.axvline(v, linestyle=":", linewidth=0.9, color="0.45")
                ax_zoom_end.set_xlim(end_zoom_t0, end_zoom_t1)
                ax_zoom_end.set_title(f"Trigger-end zoom: {end_zoom_t0:.2f} to {end_zoom_t1:.2f} s (relative to trigger){filt_note}{both_note}")
                ax_zoom_end.set_ylabel("Potential (µV)")
                ax_zoom_end.set_xlabel(TIME_REL_XLABEL)
                ax_zoom_end.grid(True, alpha=0.3)
                if any(curve is not None for curve in first_trigger_curves):
                    for recording_index, first_curve in enumerate(first_trigger_curves):
                        if first_curve is None:
                            continue
                        line_color = colors[recording_index % len(colors)]
                        ax_zoom_end_first.plot(
                            t_rel[end_mask],
                            first_curve[end_mask],
                            linewidth=1.2,
                            color=line_color,
                            label=f"{labels[recording_index]} first trigger raw",
                        )
                    for v in end_markers:
                        ax_zoom_end_first.axvline(v, linestyle=":", linewidth=0.9, color="0.45")
                    ax_zoom_end_first.set_xlim(end_zoom_t0, end_zoom_t1)
                    ax_zoom_end_first.set_title("Part 3 — First trigger raw (separate view)")
                    ax_zoom_end_first.set_ylabel("Potential (µV)")
                    ax_zoom_end_first.set_xlabel(TIME_REL_XLABEL)
                    ax_zoom_end_first.grid(True, alpha=0.3)
                else:
                    ax_zoom_end_first.text(0.5, 0.5, "First trigger raw signal unavailable", ha="center", va="center", transform=ax_zoom_end_first.transAxes)
                    ax_zoom_end_first.set_axis_off()
                _plot_rms_series(
                    ax_zoom_end_rms,
                    rms_series_zoom_end_multi,
                    f"Part 3 — RMS evolution (window [{end_zoom_t0:.3f}, {end_zoom_t1:.3f}] s)",
                    x_limits=(end_zoom_t0, end_zoom_t1),
                )
            else:
                ax_zoom_end.text(0.5, 0.5, "Trigger-end zoom unavailable\n(no rising edge after trigger)", ha="center", va="center", transform=ax_zoom_end.transAxes)
                ax_zoom_end.set_axis_off()
                ax_zoom_end_first.text(0.5, 0.5, "First trigger raw signal unavailable", ha="center", va="center", transform=ax_zoom_end_first.transAxes)
                ax_zoom_end_first.set_axis_off()
                ax_zoom_end_rms.text(0.5, 0.5, "RMS evolution unavailable", ha="center", va="center", transform=ax_zoom_end_rms.transAxes)
                ax_zoom_end_rms.set_axis_off()

            if _has_spike_cmp and spike_sources is not None:
                sources_ok = [src for src in spike_sources if src is not None]
                # Spike detection is computed once per channel/file,
                # then reused for full / zoom / trigger-end zoom.
                st_list = [
                    src.spike_times_per_trial_for_channel(ch, t_rel, spike_threshold_uv)
                    for src in sources_ok
                ]
                _draw_spike_panels_multi_channel(
                    ax_raster_f, ax_fr_f, ax_trial_fr_f, ax_isi_f, None, t_rel, float(fs), spike_threshold_uv,
                    firing_rate_window_s, labels, spike_bandpass_low_hz, spike_bandpass_high_hz,
                    t_range_s=None, spikes_per_recording=st_list, sampling_percent=sampling_percent,
                )
                _draw_spike_panels_multi_channel(
                    ax_raster_z, ax_fr_z, ax_trial_fr_z, ax_isi_z, None, t_rel, float(fs), spike_threshold_uv,
                    firing_rate_window_s, labels, spike_bandpass_low_hz, spike_bandpass_high_hz,
                    t_range_s=(zoom_t0, zoom_t1), spikes_per_recording=st_list, sampling_percent=sampling_percent,
                )
                if end_zoom_range is not None:
                    _draw_spike_panels_multi_channel(
                        ax_raster_ze, ax_fr_ze, ax_trial_fr_ze, ax_isi_ze, None, t_rel, float(fs), spike_threshold_uv,
                        firing_rate_window_s, labels, spike_bandpass_low_hz, spike_bandpass_high_hz,
                        t_range_s=end_zoom_range, section_title="Trigger-end zoom", spikes_per_recording=st_list, sampling_percent=sampling_percent,
                    )
                else:
                    for ax in (ax_raster_ze, ax_fr_ze):
                        ax.text(0.5, 0.5, "Trigger-end zoom unavailable\n(no rising edge after trigger)", ha="center", va="center", transform=ax.transAxes)
                        ax.set_axis_off()
                    ax_isi_ze.text(0.5, 0.5, "ISI unavailable", ha="center", va="center", transform=ax_isi_ze.transAxes)
                    ax_isi_ze.set_axis_off()
            else:
                for ax in (ax_raster_f, ax_fr_f, ax_raster_z, ax_fr_z, ax_raster_ze, ax_fr_ze):
                    ax.text(0.5, 0.5, "Raster / PSTH / ISI unavailable\n(missing mmap sources)", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                    ax.set_axis_off()
                for ax in (ax_isi_f, ax_isi_z, ax_isi_ze):
                    ax.text(0.5, 0.5, "ISI unavailable", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()

            _axes_tick_bottom = [
                ax_full,
                ax_first_trigger,
                ax_full_rms,
                ax_zoom,
                ax_zoom_first,
                ax_zoom_rms,
                ax_raster_f,
                ax_fr_f,
                ax_trial_fr_f,
                ax_isi_f,
                ax_raster_z,
                ax_fr_z,
                ax_trial_fr_z,
                ax_isi_z,
                ax_raster_ze,
                ax_fr_ze,
                ax_trial_fr_ze,
                ax_isi_ze,
                ax_zoom_end,
                ax_zoom_end_first,
                ax_zoom_end_rms,
            ]
            if impedance_sessions:
                _axes_tick_bottom.append(ax_imp)
            for ax in _axes_tick_bottom:
                ax.tick_params(axis="x", labelbottom=True)

            fig.tight_layout()
            shift_axes_down(
                [
                    ax_first_trigger,
                    ax_full_rms,
                    ax_raster_f,
                    ax_fr_f,
                    ax_trial_fr_f,
                    ax_isi_f,
                    ax_hdr2,
                    ax_zoom,
                    ax_zoom_first,
                    ax_zoom_rms,
                    ax_raster_z,
                    ax_fr_z,
                    ax_trial_fr_z,
                    ax_isi_z,
                    ax_hdr3,
                    ax_zoom_end,
                    ax_zoom_end_first,
                    ax_zoom_end_rms,
                    ax_raster_ze,
                    ax_fr_ze,
                    ax_trial_fr_ze,
                    ax_isi_ze,
                ],
                delta=0.015,
            )
            _soften_figure_linewidths(fig)
            page_dpi = _lightweight_pdf_dpi(lightweight_mode)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2, dpi=page_dpi)
            plt.close(fig)

        if spike_sources is not None:
            rms_series: list[tuple[str, np.ndarray, np.ndarray]] = []
            t0_rms = float(t_rel[0]) if t_rel.size else 0.0
            t1_rms = float(t_rel[-1]) if t_rel.size else 0.0
            for i, src in enumerate(spike_sources):
                if src is None:
                    continue
                tx_rms, rms_vals = _mean_rms_profile_from_source_window(src, t0_rms, t1_rms, rms_window_s)
                label = labels[i] if i < len(labels) else f"Recording {i + 1}"
                rms_series.append((label, tx_rms, rms_vals))
            if rms_series:
                _append_mean_rms_evolution_page(pdf, rms_series, rms_window_s, lightweight_mode)

        if impedance_sessions:
            _append_mean_impedance_summary_page(pdf, impedance_sessions, lightweight_mode)

    _profile_print_delta(
        "plot_channel_multi_comparison",
        _profile_before,
        time.perf_counter() - _profile_t0,
    )
    return pdf_path


def plot_channel_averages(
    t_rel: np.ndarray,
    mean_per_channel: np.ndarray,
    channel_names: Sequence[str],
    output_dir: Path,
    rhs_file: Path,
    pdf_title: Optional[str] = None,
    lowpass_cutoff_hz: Optional[float] = None,
    trigger_end_rising_rel_s: Optional[float] = None,
    windows: Optional[np.ndarray] = None,
    spike_source: Optional[AmplifierSpikeSource] = None,
    fs: Optional[float] = None,
    spike_threshold_uv: float = -40.0,
    firing_rate_window_s: float = 0.025,
    rms_window_s: float = 0.050,
    zoom_t0_s: float = ZOOM_T0,
    zoom_t1_s: float = ZOOM_T1,
    mean_per_channel_raw: Optional[np.ndarray] = None,
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
    lightweight_mode: bool = False,
    sampling_percent: int = 100,
    probe_layout_json: Optional[Path] = None,
) -> Path:
    """Single multi-page PDF (one page per channel), without GUI window.

    Intan amplifier values are displayed in microvolts (uV), as
    provided by load_intan_rhs_format (amplifier_data).
    """
    _profile_before = _profile_snapshot()
    _profile_t0 = time.perf_counter()
    n_channels = mean_per_channel.shape[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    if pdf_title is not None and pdf_title.strip():
        safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in pdf_title.strip())
        pdf_stem = safe_title.removesuffix(".pdf")
    else:
        pdf_stem = rhs_file.stem
    pdf_name = shorten_filename_for_windows(output_dir, f"{pdf_stem}.pdf")
    pdf_path = output_dir / pdf_name

    zoom_t0, zoom_t1 = float(zoom_t0_s), float(zoom_t1_s)
    filt_note = (
        f" — Butterworth low-pass {lowpass_cutoff_hz:g} Hz"
        if lowpass_cutoff_hz is not None
        else ""
    )
    both_note = (
        " — raw signal overlaid (filtered curve emphasized)"
        if lowpass_cutoff_hz is not None
        else ""
    )
    zoom_title = (
        f"Zoom: {zoom_t0:.1f} to {zoom_t1:.1f} s (relative to trigger){filt_note}{both_note}"
    )

    def _spike_threshold_caption(thr: float) -> str:
        if thr >= 0:
            return f"threshold {thr:g} µV (rising edge)"
        return f"threshold {thr:g} µV (falling edge, negative spike)"

    _has_spike_data = fs is not None and (
        spike_source is not None or (windows is not None and getattr(windows, "ndim", 0) == 3)
    )
    _, spike_pipe_detail = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    spike_note = (
        f" — spikes ({spike_pipe_detail}): {_spike_threshold_caption(spike_threshold_uv)}, "
        f"FR smoothing σ={firing_rate_window_s:g} s"
        if _has_spike_data
        else ""
    )
    probe_layout_loaded = None
    if probe_layout_json is not None:
        probe_layout_loaded = load_probe_layout_json(Path(probe_layout_json))

    with PdfPages(pdf_path) as pdf:
        for ch in range(n_channels):
            check_analysis_cancelled()
            channel_name = str(channel_names[ch])
            channel_mean = mean_per_channel[ch]
            rms_series_full: list[tuple[str, np.ndarray, np.ndarray]] = []
            rms_series_zoom: list[tuple[str, np.ndarray, np.ndarray]] = []
            rms_series_zoom_end: list[tuple[str, np.ndarray, np.ndarray]] = []
            t0_rms = float(t_rel[0]) if t_rel.size else 0.0
            t1_rms = float(t_rel[-1]) if t_rel.size else 0.0
            if spike_source is not None:
                tx_full, rms_full_vals = _mean_rms_profile_from_source_window(
                    spike_source,
                    t0_rms,
                    t1_rms,
                    rms_window_s,
                    channel_index=ch,
                )
                rms_series_full = [("RMS", tx_full, rms_full_vals)]
                tx_zoom, rms_zoom_vals = _slice_rms_profile_window(tx_full, rms_full_vals, zoom_t0, zoom_t1)
                rms_series_zoom = [("RMS", tx_zoom, rms_zoom_vals)]
                if trigger_end_rising_rel_s is not None:
                    tx_end, rms_zoom_end_vals = _slice_rms_profile_window(
                        tx_full,
                        rms_full_vals,
                        float(trigger_end_rising_rel_s + zoom_t0),
                        float(trigger_end_rising_rel_s + zoom_t1),
                    )
                    rms_series_zoom_end = [("RMS", tx_end, rms_zoom_end_vals)]
            elif windows is not None and getattr(windows, "ndim", 0) == 3:
                windows_arr = np.asarray(windows)
                tx_full, rms_full_vals = _mean_rms_profile_from_windows_window(
                    windows_arr,
                    t_rel,
                    t0_rms,
                    t1_rms,
                    rms_window_s,
                    channel_index=ch,
                )
                rms_series_full = [("RMS", tx_full, rms_full_vals)]
                tx_zoom, rms_zoom_vals = _slice_rms_profile_window(tx_full, rms_full_vals, zoom_t0, zoom_t1)
                rms_series_zoom = [("RMS", tx_zoom, rms_zoom_vals)]
                if trigger_end_rising_rel_s is not None:
                    tx_end, rms_zoom_end_vals = _slice_rms_profile_window(
                        tx_full,
                        rms_full_vals,
                        float(trigger_end_rising_rel_s + zoom_t0),
                        float(trigger_end_rising_rel_s + zoom_t1),
                    )
                    rms_series_zoom_end = [("RMS", tx_end, rms_zoom_end_vals)]
            page_width_in = 12.0
            mea_row_height_ratio = 1.60
            page_height_in = 52.0
            page_width_in, page_height_in = _scale_page_size_for_lightweight(
                page_width_in, page_height_in, lightweight_mode
            )
            fig = plt.figure(figsize=(page_width_in, page_height_in))
            gs = fig.add_gridspec(
                24,
                1,
                height_ratios=[
                    mea_row_height_ratio,
                    1.70,
                    1.60,
                    1.30,
                    1.20,
                    1.15,
                    1.05,
                    1.15,
                    0.06,
                    1.70,
                    1.50,
                    1.20,
                    1.30,
                    1.15,
                    1.05,
                    1.15,
                    0.06,
                    1.70,
                    1.50,
                    1.20,
                    1.30,
                    1.15,
                    1.05,
                    1.15,
                ],
                hspace=0.74,
            )
            ax_mea = fig.add_subplot(gs[0, 0])
            _draw_mea_layout_panel(ax_mea, probe_layout_loaded, channel_name)
            ax_full = fig.add_subplot(gs[1, 0])
            ax_first_trigger = fig.add_subplot(gs[2, 0], sharex=ax_full)
            ax_full_rms = fig.add_subplot(gs[3, 0], sharex=ax_full)
            ax_raster_f = fig.add_subplot(gs[4, 0], sharex=ax_full)
            ax_fr_f = fig.add_subplot(gs[5, 0], sharex=ax_full)
            ax_trial_fr_f = fig.add_subplot(gs[6, 0])
            ax_isi_f = fig.add_subplot(gs[7, 0])
            ax_hdr2 = fig.add_subplot(gs[8, 0])
            ax_hdr2.axis("off")
            ax_hdr2.text(
                0.02,
                0.5,
                f"Part 2 — Zoomed view [{zoom_t0:.2f}, {zoom_t1:.2f}] s (relative to trigger)",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                transform=ax_hdr2.transAxes,
            )
            ax_zoom = fig.add_subplot(gs[9, 0])
            ax_zoom_first = fig.add_subplot(gs[10, 0], sharex=ax_zoom)
            ax_zoom_rms = fig.add_subplot(gs[11, 0])
            ax_raster_z = fig.add_subplot(gs[12, 0], sharex=ax_zoom)
            ax_fr_z = fig.add_subplot(gs[13, 0], sharex=ax_zoom)
            ax_trial_fr_z = fig.add_subplot(gs[14, 0])
            ax_isi_z = fig.add_subplot(gs[15, 0])
            ax_hdr3 = fig.add_subplot(gs[16, 0])
            ax_hdr3.axis("off")
            ax_hdr3.text(
                0.02,
                0.5,
                "Part 3 — Trigger-end zoom (rising edge)",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                transform=ax_hdr3.transAxes,
            )
            ax_zoom_end = fig.add_subplot(gs[17, 0])
            ax_zoom_end_first = fig.add_subplot(gs[18, 0], sharex=ax_zoom_end)
            ax_zoom_end_rms = fig.add_subplot(gs[19, 0])
            ax_raster_ze = fig.add_subplot(gs[20, 0], sharex=ax_zoom_end)
            ax_fr_ze = fig.add_subplot(gs[21, 0], sharex=ax_zoom_end)
            ax_trial_fr_ze = fig.add_subplot(gs[22, 0])
            ax_isi_ze = fig.add_subplot(gs[23, 0])

            channel_mean_raw = (
                mean_per_channel_raw[ch]
                if mean_per_channel_raw is not None
                else None
            )
            channel_first_trigger_raw: Optional[np.ndarray] = None
            if spike_source is not None and spike_source.valid_triggers.size > 0:
                first_trigger_index = int(spike_source.valid_triggers[0])
                sample_start = int(first_trigger_index - spike_source.pre_n)
                sample_end = int(first_trigger_index + spike_source.post_n)
                channel_first_trigger_raw = np.asarray(
                    spike_source.amplifier[ch, sample_start:sample_end],
                    dtype=np.float64,
                )
                if channel_first_trigger_raw.shape[0] != t_rel.shape[0]:
                    channel_first_trigger_raw = None
            elif windows is not None and getattr(windows, "ndim", 0) == 3 and windows.shape[0] > 0:
                first_trial_window = np.asarray(windows[0, ch, :], dtype=np.float64)
                if first_trial_window.shape[0] == t_rel.shape[0]:
                    channel_first_trigger_raw = first_trial_window
            show_filtered_and_raw = (
                lowpass_cutoff_hz is not None
                and mean_per_channel_raw is not None
                and channel_mean_raw is not None
                and not np.allclose(channel_mean, channel_mean_raw, rtol=0.0, atol=1e-9)
            )
            if show_filtered_and_raw:
                ax_full.plot(
                    t_rel,
                    channel_mean_raw,
                    linewidth=1.05,
                    color="0.35",
                    alpha=0.85,
                    label="Mean (raw)",
                    zorder=1,
                )
                ax_full.plot(
                    t_rel,
                    channel_mean,
                    linewidth=1.35,
                    color="C0",
                    label=f"Mean (filtered, {lowpass_cutoff_hz:g} Hz)",
                    zorder=2,
                )
            else:
                ax_full.plot(t_rel, channel_mean, linewidth=1.2, color="C0", label="Mean")
            ax_full.axvline(0.0, linestyle="--", linewidth=1.0, color="red", label="Trigger (onset)")
            if trigger_end_rising_rel_s is not None:
                ax_full.axvline(
                    trigger_end_rising_rel_s,
                    linestyle=":",
                    linewidth=1.0,
                    color="darkorange",
                    label="Trigger end (rising)",
                )
            ax_full.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zoom region")
            if trigger_end_rising_rel_s is not None:
                end_zoom_t0 = float(trigger_end_rising_rel_s + zoom_t0)
                end_zoom_t1 = float(trigger_end_rising_rel_s + zoom_t1)
                ax_full.axvspan(end_zoom_t0, end_zoom_t1, alpha=0.10, color="gold", label="Trigger-end zoom region")
            ax_full.set_title(
                f"Trigger mean — {channel_name} (full view){filt_note}{both_note}{spike_note}"
            )
            ax_full.set_ylabel("Potential (µV)")
            ax_full.set_xlabel(TIME_REL_XLABEL)
            ax_full.grid(True, alpha=0.3)
            ax_full.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.28),
                ncol=3,
                fontsize=LEGEND_FONT_SIZE,
            )

            if channel_first_trigger_raw is not None:
                ax_first_trigger.plot(
                    t_rel,
                    channel_first_trigger_raw,
                    linewidth=1.1,
                    color="C3",
                    label="First trigger (raw, no averaging)",
                )
                ax_first_trigger.axvline(
                    0.0,
                    linestyle="--",
                    linewidth=1.0,
                    color="red",
                    label="Trigger (onset)",
                )
                if trigger_end_rising_rel_s is not None:
                    ax_first_trigger.axvline(
                        trigger_end_rising_rel_s,
                        linestyle=":",
                        linewidth=1.0,
                        color="darkorange",
                        label="Trigger end (rising)",
                    )
                ax_first_trigger.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zoom region")
                if trigger_end_rising_rel_s is not None:
                    end_zoom_t0 = float(trigger_end_rising_rel_s + zoom_t0)
                    end_zoom_t1 = float(trigger_end_rising_rel_s + zoom_t1)
                    ax_first_trigger.axvspan(
                        end_zoom_t0,
                        end_zoom_t1,
                        alpha=0.10,
                        color="gold",
                        label="Trigger-end zoom region",
                    )
                ax_first_trigger.set_title("First trigger — raw signal (no averaging)")
                ax_first_trigger.set_ylabel("Potential (µV)")
                ax_first_trigger.set_xlabel(TIME_REL_XLABEL)
                ax_first_trigger.grid(True, alpha=0.3)
            else:
                ax_first_trigger.text(
                    0.5,
                    0.5,
                    "First trigger raw signal unavailable",
                    ha="center",
                    va="center",
                    transform=ax_first_trigger.transAxes,
                )
                ax_first_trigger.set_axis_off()

            _plot_rms_series(
                ax_full_rms,
                rms_series_full,
                "Part 1 — RMS evolution (full window)",
                x_limits=(float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None,
            )

            zmask = (t_rel >= zoom_t0) & (t_rel <= zoom_t1)
            if show_filtered_and_raw and channel_mean_raw is not None:
                ax_zoom.plot(
                    t_rel[zmask],
                    channel_mean_raw[zmask],
                    linewidth=1.15,
                    color="0.35",
                    alpha=0.85,
                    label="Raw",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    channel_mean[zmask],
                    linewidth=1.45,
                    color="C0",
                    label=f"Filtered ({lowpass_cutoff_hz:g} Hz)",
                )
            else:
                ax_zoom.plot(t_rel[zmask], channel_mean[zmask], linewidth=1.4, color="C0")
            ax_zoom.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
            if trigger_end_rising_rel_s is not None:
                ax_zoom.axvline(
                    trigger_end_rising_rel_s,
                    linestyle=":",
                    linewidth=1.0,
                    color="darkorange",
                )
            ax_zoom.set_xlim(zoom_t0, zoom_t1)
            ax_zoom.set_title(zoom_title)
            ax_zoom.set_ylabel("Potential (µV)")
            ax_zoom.set_xlabel(TIME_REL_XLABEL)
            ax_zoom.grid(True, alpha=0.3)
            if show_filtered_and_raw:
                ax_zoom.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
            if channel_first_trigger_raw is not None:
                ax_zoom_first.plot(
                    t_rel[zmask],
                    channel_first_trigger_raw[zmask],
                    linewidth=1.2,
                    color="C3",
                    label="First trigger raw (no averaging)",
                )
                ax_zoom_first.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
                if trigger_end_rising_rel_s is not None:
                    ax_zoom_first.axvline(
                        trigger_end_rising_rel_s,
                        linestyle=":",
                        linewidth=1.0,
                        color="darkorange",
                    )
                ax_zoom_first.set_xlim(zoom_t0, zoom_t1)
                ax_zoom_first.set_title("Part 2 — First trigger raw (separate view)")
                ax_zoom_first.set_ylabel("Potential (µV)")
                ax_zoom_first.set_xlabel(TIME_REL_XLABEL)
                ax_zoom_first.grid(True, alpha=0.3)
            else:
                ax_zoom_first.text(
                    0.5,
                    0.5,
                    "First trigger raw signal unavailable",
                    ha="center",
                    va="center",
                    transform=ax_zoom_first.transAxes,
                )
                ax_zoom_first.set_axis_off()
            _plot_rms_series(
                ax_zoom_rms,
                rms_series_zoom,
                f"Part 2 — RMS evolution (window [{zoom_t0:.3f}, {zoom_t1:.3f}] s)",
                x_limits=(zoom_t0, zoom_t1),
            )
            end_zoom_range: tuple[float, float] | None = None
            if trigger_end_rising_rel_s is not None:
                end_zoom_t0 = float(trigger_end_rising_rel_s + zoom_t0)
                end_zoom_t1 = float(trigger_end_rising_rel_s + zoom_t1)
                end_zoom_range = (end_zoom_t0, end_zoom_t1)
                end_mask = (t_rel >= end_zoom_t0) & (t_rel <= end_zoom_t1)
                if show_filtered_and_raw and channel_mean_raw is not None:
                    ax_zoom_end.plot(t_rel[end_mask], channel_mean_raw[end_mask], linewidth=1.15, color="0.35", alpha=0.85, label="Unfiltered")
                    ax_zoom_end.plot(t_rel[end_mask], channel_mean[end_mask], linewidth=1.45, color="C0", label=f"Filtered ({lowpass_cutoff_hz:g} Hz)")
                else:
                    ax_zoom_end.plot(t_rel[end_mask], channel_mean[end_mask], linewidth=1.4, color="C0")
                ax_zoom_end.axvline(trigger_end_rising_rel_s, linestyle="--", linewidth=1.0, color="darkorange")
                ax_zoom_end.set_xlim(end_zoom_t0, end_zoom_t1)
                ax_zoom_end.set_title(f"Part 3 — Trigger-end zoom [{end_zoom_t0:.2f}, {end_zoom_t1:.2f}] s")
                ax_zoom_end.set_ylabel("Potential (µV)")
                ax_zoom_end.set_xlabel(TIME_REL_XLABEL)
                ax_zoom_end.grid(True, alpha=0.3)
                if show_filtered_and_raw:
                    ax_zoom_end.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
                if channel_first_trigger_raw is not None:
                    ax_zoom_end_first.plot(
                        t_rel[end_mask],
                        channel_first_trigger_raw[end_mask],
                        linewidth=1.2,
                        color="C3",
                        label="First trigger raw (no averaging)",
                    )
                    ax_zoom_end_first.axvline(trigger_end_rising_rel_s, linestyle="--", linewidth=1.0, color="darkorange")
                    ax_zoom_end_first.set_xlim(end_zoom_t0, end_zoom_t1)
                    ax_zoom_end_first.set_title("Part 3 — First trigger raw (separate view)")
                    ax_zoom_end_first.set_ylabel("Potential (µV)")
                    ax_zoom_end_first.set_xlabel(TIME_REL_XLABEL)
                    ax_zoom_end_first.grid(True, alpha=0.3)
                else:
                    ax_zoom_end_first.text(
                        0.5,
                        0.5,
                        "First trigger raw signal unavailable",
                        ha="center",
                        va="center",
                        transform=ax_zoom_end_first.transAxes,
                    )
                    ax_zoom_end_first.set_axis_off()
                _plot_rms_series(
                    ax_zoom_end_rms,
                    rms_series_zoom_end,
                    f"Part 3 — RMS evolution (window [{end_zoom_t0:.3f}, {end_zoom_t1:.3f}] s)",
                    x_limits=(end_zoom_t0, end_zoom_t1),
                )
            else:
                ax_zoom_end.text(0.5, 0.5, "Trigger-end zoom unavailable\n(no rising edge after trigger)", ha="center", va="center", transform=ax_zoom_end.transAxes)
                ax_zoom_end.set_axis_off()
                ax_zoom_end_first.text(0.5, 0.5, "First trigger raw signal unavailable", ha="center", va="center", transform=ax_zoom_end_first.transAxes)
                ax_zoom_end_first.set_axis_off()
                ax_zoom_end_rms.text(0.5, 0.5, "RMS evolution unavailable", ha="center", va="center", transform=ax_zoom_end_rms.transAxes)
                ax_zoom_end_rms.set_axis_off()

            if _has_spike_data and (
                spike_source is not None
                or (windows is not None and windows.ndim == 3)
            ):
                if spike_source is not None:
                    st_per_tr = spike_source.spike_times_per_trial_for_channel(
                        ch, t_rel, spike_threshold_uv
                    )
                else:
                    w_ch = np.asarray(windows[:, ch, :])
                    # Spike detection is computed once per channel,
                    # then reused for full / zoom / trigger-end zoom.
                    st_per_tr = _spike_times_per_trial(
                        w_ch, t_rel, float(fs), spike_threshold_uv
                    )
                _draw_spike_panels_single_channel(
                    ax_raster_f,
                    ax_fr_f,
                    ax_trial_fr_f,
                    ax_isi_f,
                    None if spike_source is not None else w_ch,
                    t_rel,
                    float(fs),
                    spike_threshold_uv,
                    firing_rate_window_s,
                    spike_bandpass_low_hz,
                    spike_bandpass_high_hz,
                    t_range_s=None,
                    st_per_tr=st_per_tr,
                    sampling_percent=sampling_percent,
                )
                _draw_spike_panels_single_channel(
                    ax_raster_z,
                    ax_fr_z,
                    ax_trial_fr_z,
                    ax_isi_z,
                    None if spike_source is not None else w_ch,
                    t_rel,
                    float(fs),
                    spike_threshold_uv,
                    firing_rate_window_s,
                    spike_bandpass_low_hz,
                    spike_bandpass_high_hz,
                    t_range_s=(zoom_t0, zoom_t1),
                    st_per_tr=st_per_tr,
                    sampling_percent=sampling_percent,
                )
                if end_zoom_range is not None:
                    _draw_spike_panels_single_channel(
                        ax_raster_ze,
                        ax_fr_ze,
                        ax_trial_fr_ze,
                        ax_isi_ze,
                        None if spike_source is not None else w_ch,
                        t_rel,
                        float(fs),
                        spike_threshold_uv,
                        firing_rate_window_s,
                        spike_bandpass_low_hz,
                        spike_bandpass_high_hz,
                        t_range_s=end_zoom_range,
                        st_per_tr=st_per_tr,
                        section_title="Trigger-end zoom",
                    )
                else:
                    for ax in (ax_raster_ze, ax_fr_ze, ax_trial_fr_ze):
                        ax.text(
                            0.5,
                            0.5,
                            "Trigger-end zoom unavailable\n(no rising edge after trigger)",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_axis_off()
                    ax_isi_ze.text(
                        0.5,
                        0.5,
                        "ISI unavailable",
                        ha="center",
                        va="center",
                        transform=ax_isi_ze.transAxes,
                    )
                    ax_isi_ze.set_axis_off()
            elif not _has_spike_data:
                for ax in (ax_raster_f, ax_fr_f, ax_trial_fr_f, ax_raster_z, ax_fr_z, ax_trial_fr_z, ax_raster_ze, ax_fr_ze, ax_trial_fr_ze):
                    ax.text(
                        0.5,
                        0.5,
                        "Raw trial data unavailable",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                for ax in (ax_isi_f, ax_isi_z, ax_isi_ze):
                    ax.text(
                        0.5,
                        0.5,
                        "ISI unavailable",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()

            for ax in (
                ax_full,
                ax_first_trigger,
                ax_full_rms,
                ax_zoom,
                ax_zoom_first,
                ax_zoom_rms,
                ax_raster_f,
                ax_fr_f,
                ax_trial_fr_f,
                ax_isi_f,
                ax_raster_z,
                ax_fr_z,
                ax_trial_fr_z,
                ax_isi_z,
                ax_raster_ze,
                ax_fr_ze,
                ax_trial_fr_ze,
                ax_isi_ze,
                ax_zoom_end,
                ax_zoom_end_first,
                ax_zoom_end_rms,
            ):
                ax.tick_params(axis="x", labelbottom=True)

            fig.tight_layout()
            shift_axes_down(
                [
                    ax_first_trigger,
                    ax_full_rms,
                    ax_raster_f,
                    ax_fr_f,
                    ax_trial_fr_f,
                    ax_isi_f,
                    ax_hdr2,
                    ax_zoom,
                    ax_zoom_first,
                    ax_zoom_rms,
                    ax_raster_z,
                    ax_fr_z,
                    ax_trial_fr_z,
                    ax_isi_z,
                    ax_hdr3,
                    ax_zoom_end,
                    ax_zoom_end_first,
                    ax_zoom_end_rms,
                    ax_raster_ze,
                    ax_fr_ze,
                    ax_trial_fr_ze,
                    ax_isi_ze,
                ],
                delta=0.015,
            )
            _soften_figure_linewidths(fig)
            page_dpi = _lightweight_pdf_dpi(lightweight_mode)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2, dpi=page_dpi)
            plt.close(fig)

        rms_tx = np.array([], dtype=np.float64)
        rms_values = np.array([], dtype=np.float64)
        t0_rms = float(t_rel[0]) if t_rel.size else 0.0
        t1_rms = float(t_rel[-1]) if t_rel.size else 0.0
        if spike_source is not None:
            rms_tx, rms_values = _mean_rms_profile_from_source_window(
                spike_source, t0_rms, t1_rms, rms_window_s
            )
        elif windows is not None and getattr(windows, "ndim", 0) == 3:
            rms_tx, rms_values = _mean_rms_profile_from_windows_window(
                np.asarray(windows),
                t_rel,
                t0_rms,
                t1_rms,
                rms_window_s,
            )
        _append_mean_rms_evolution_page(
            pdf,
            [("Mean RMS", rms_tx, rms_values)],
            rms_window_s,
            lightweight_mode,
        )

    _profile_print_delta(
        "plot_channel_averages",
        _profile_before,
        time.perf_counter() - _profile_t0,
    )
    return pdf_path


def plot_channel_comparison(
    t_rel: np.ndarray,
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    channel_names: Sequence[str],
    output_dir: Path,
    label_a: str,
    label_b: str,
    pdf_title: Optional[str] = None,
    lowpass_cutoff_hz: Optional[float] = None,
    trigger_end_rising_rel_s_a: Optional[float] = None,
    trigger_end_rising_rel_s_b: Optional[float] = None,
    mean_a_raw: Optional[np.ndarray] = None,
    mean_b_raw: Optional[np.ndarray] = None,
    spike_source_a: Optional[AmplifierSpikeSource] = None,
    spike_source_b: Optional[AmplifierSpikeSource] = None,
    fs: Optional[float] = None,
    spike_threshold_uv: float = -40.0,
    firing_rate_window_s: float = 0.025,
    rms_window_s: float = 0.050,
    zoom_t0_s: float = ZOOM_T0,
    zoom_t1_s: float = ZOOM_T1,
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
    lightweight_mode: bool = False,
    sampling_percent: int = 100,
) -> Path:
    """Multi-page PDF: per channel, means + zoom + raster / PSTH / ISI (two overlaid recordings)."""
    _profile_before = _profile_snapshot()
    _profile_t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_a = "".join(c if c.isalnum() or c in "._-" else "_" for c in label_a)[:80]
    safe_b = "".join(c if c.isalnum() or c in "._-" else "_" for c in label_b)[:80]
    if pdf_title is not None and pdf_title.strip():
        safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in pdf_title.strip())
        pdf_stem = safe_title.removesuffix(".pdf")
    else:
        pdf_stem = f"{safe_a}_vs_{safe_b}"
    pdf_name = shorten_filename_for_windows(output_dir, f"{pdf_stem}.pdf")
    pdf_path = output_dir / pdf_name

    zoom_t0, zoom_t1 = float(zoom_t0_s), float(zoom_t1_s)
    filt_note = (
        f" — Butterworth low-pass {lowpass_cutoff_hz:g} Hz"
        if lowpass_cutoff_hz is not None
        else ""
    )
    both_note = (
        " — raw signal overlaid (filtered curve emphasized)"
        if lowpass_cutoff_hz is not None
        else ""
    )
    zoom_title = f"Zoom: {zoom_t0:.1f} to {zoom_t1:.1f} s (relative to trigger){filt_note}{both_note}"
    n_channels = mean_a.shape[0]
    zmask = (t_rel >= zoom_t0) & (t_rel <= zoom_t1)
    _has_spike_cmp = (
        fs is not None
        and spike_source_a is not None
        and spike_source_b is not None
    )
    _, spike_cmp_pipe = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    with PdfPages(pdf_path) as pdf:
        for ch in range(n_channels):
            check_analysis_cancelled()
            channel_name = str(channel_names[ch])
            channel_mean_a = mean_a[ch]
            channel_mean_b = mean_b[ch]
            t0_rms = float(t_rel[0]) if t_rel.size else 0.0
            t1_rms = float(t_rel[-1]) if t_rel.size else 0.0
            rms_full_a = (
                _mean_rms_profile_from_source_window(
                    spike_source_a,
                    t0_rms,
                    t1_rms,
                    rms_window_s,
                    channel_index=ch,
                )
                if spike_source_a is not None
                else (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
            )
            rms_full_b = (
                _mean_rms_profile_from_source_window(
                    spike_source_b,
                    t0_rms,
                    t1_rms,
                    rms_window_s,
                    channel_index=ch,
                )
                if spike_source_b is not None
                else (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
            )
            rms_zoom_a = _slice_rms_profile_window(rms_full_a[0], rms_full_a[1], zoom_t0, zoom_t1)
            rms_zoom_b = _slice_rms_profile_window(rms_full_b[0], rms_full_b[1], zoom_t0, zoom_t1)
            rms_end_a = (
                _slice_rms_profile_window(
                    rms_full_a[0],
                    rms_full_a[1],
                    float(trigger_end_rising_rel_s_a + zoom_t0),
                    float(trigger_end_rising_rel_s_a + zoom_t1),
                )
                if trigger_end_rising_rel_s_a is not None
                else (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
            )
            rms_end_b = (
                _slice_rms_profile_window(
                    rms_full_b[0],
                    rms_full_b[1],
                    float(trigger_end_rising_rel_s_b + zoom_t0),
                    float(trigger_end_rising_rel_s_b + zoom_t1),
                )
                if trigger_end_rising_rel_s_b is not None
                else (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
            )
            channel_mean_a_raw = mean_a_raw[ch] if mean_a_raw is not None else None
            channel_mean_b_raw = mean_b_raw[ch] if mean_b_raw is not None else None
            show_filtered_and_raw = (
                lowpass_cutoff_hz is not None
                and channel_mean_a_raw is not None
                and channel_mean_b_raw is not None
                and (
                    not np.allclose(channel_mean_a, channel_mean_a_raw, rtol=0.0, atol=1e-9)
                    or not np.allclose(channel_mean_b, channel_mean_b_raw, rtol=0.0, atol=1e-9)
                )
            )
            first_trigger_a_raw: Optional[np.ndarray] = None
            first_trigger_b_raw: Optional[np.ndarray] = None
            if spike_source_a is not None and spike_source_a.valid_triggers.size > 0:
                trig_a0 = int(spike_source_a.valid_triggers[0])
                s0_a = int(trig_a0 - spike_source_a.pre_n)
                s1_a = int(trig_a0 + spike_source_a.post_n)
                first_trigger_a_raw = np.asarray(spike_source_a.amplifier[ch, s0_a:s1_a], dtype=np.float64)
                if first_trigger_a_raw.shape[0] != t_rel.shape[0]:
                    first_trigger_a_raw = None
            if spike_source_b is not None and spike_source_b.valid_triggers.size > 0:
                trig_b0 = int(spike_source_b.valid_triggers[0])
                s0_b = int(trig_b0 - spike_source_b.pre_n)
                s1_b = int(trig_b0 + spike_source_b.post_n)
                first_trigger_b_raw = np.asarray(spike_source_b.amplifier[ch, s0_b:s1_b], dtype=np.float64)
                if first_trigger_b_raw.shape[0] != t_rel.shape[0]:
                    first_trigger_b_raw = None
            page_width_in, page_height_in = _scale_page_size_for_lightweight(
                12.0, 52.0, lightweight_mode
            )
            fig = plt.figure(figsize=(page_width_in, page_height_in))
            gs = fig.add_gridspec(
                24,
                1,
                height_ratios=[0.06, 1.70, 1.60, 1.30, 1.20, 1.15, 1.05, 1.15, 0.06, 1.70, 1.50, 1.20, 1.30, 1.15, 1.05, 1.15, 0.06, 1.70, 1.50, 1.20, 1.30, 1.15, 1.05, 1.15],
                hspace=0.70,
            )
            ax_hdr1 = fig.add_subplot(gs[0, 0])
            ax_hdr1.axis("off")
            ax_hdr1.text(
                0.02,
                0.5,
                "Part 1 — Full view (entire pre/post-trigger window)",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                transform=ax_hdr1.transAxes,
            )
            ax_full = fig.add_subplot(gs[1, 0])
            ax_first_trigger = fig.add_subplot(gs[2, 0], sharex=ax_full)
            ax_full_rms = fig.add_subplot(gs[3, 0], sharex=ax_full)
            ax_raster_f = fig.add_subplot(gs[4, 0], sharex=ax_full)
            ax_fr_f = fig.add_subplot(gs[5, 0], sharex=ax_full)
            ax_trial_fr_f = fig.add_subplot(gs[6, 0])
            ax_isi_f = fig.add_subplot(gs[7, 0])
            ax_hdr2 = fig.add_subplot(gs[8, 0])
            ax_hdr2.axis("off")
            ax_hdr2.text(
                0.02,
                0.5,
                f"Part 2 — Zoomed view [{zoom_t0:.2f}, {zoom_t1:.2f}] s (relative to trigger)",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                transform=ax_hdr2.transAxes,
            )
            ax_zoom = fig.add_subplot(gs[9, 0])
            ax_zoom_first = fig.add_subplot(gs[10, 0], sharex=ax_zoom)
            ax_zoom_rms = fig.add_subplot(gs[11, 0])
            ax_raster_z = fig.add_subplot(gs[12, 0], sharex=ax_zoom)
            ax_fr_z = fig.add_subplot(gs[13, 0], sharex=ax_zoom)
            ax_trial_fr_z = fig.add_subplot(gs[14, 0])
            ax_isi_z = fig.add_subplot(gs[15, 0])
            ax_hdr3 = fig.add_subplot(gs[16, 0])
            ax_hdr3.axis("off")
            ax_hdr3.text(
                0.02,
                0.5,
                "Part 3 — Trigger-end zoom (rising edge)",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                transform=ax_hdr3.transAxes,
            )
            ax_zoom_end = fig.add_subplot(gs[17, 0])
            ax_zoom_end_first = fig.add_subplot(gs[18, 0], sharex=ax_zoom_end)
            ax_zoom_end_rms = fig.add_subplot(gs[19, 0])
            ax_raster_ze = fig.add_subplot(gs[20, 0], sharex=ax_zoom_end)
            ax_fr_ze = fig.add_subplot(gs[21, 0], sharex=ax_zoom_end)
            ax_trial_fr_ze = fig.add_subplot(gs[22, 0])
            ax_isi_ze = fig.add_subplot(gs[23, 0])
            if show_filtered_and_raw:
                ax_full.plot(
                    t_rel,
                    channel_mean_a_raw,
                    linewidth=1.0,
                    color="C0",
                    alpha=0.4,
                    label="_nolegend_",
                )
                ax_full.plot(
                    t_rel,
                    channel_mean_b_raw,
                    linewidth=1.0,
                    color="C1",
                    alpha=0.4,
                    label="_nolegend_",
                )
                ax_full.plot(
                    t_rel,
                    channel_mean_a,
                    linewidth=1.35,
                    color="C0",
                    label=f"{label_a} (filtered {lowpass_cutoff_hz:g} Hz)",
                )
                ax_full.plot(
                    t_rel,
                    channel_mean_b,
                    linewidth=1.35,
                    color="C1",
                    label=f"{label_b} (filtered {lowpass_cutoff_hz:g} Hz)",
                )
            else:
                ax_full.plot(t_rel, channel_mean_a, linewidth=1.2, color="C0", label=label_a)
                ax_full.plot(t_rel, channel_mean_b, linewidth=1.2, color="C1", label=label_b)
            ax_full.axvline(0.0, linestyle="--", linewidth=1.0, color="red", label="Trigger (onset)")
            if trigger_end_rising_rel_s_a is not None:
                ax_full.axvline(
                    trigger_end_rising_rel_s_a,
                    linestyle=":",
                    linewidth=1.0,
                    color="darkorange",
                    label=f"End (rising) {label_a}",
                )
            if trigger_end_rising_rel_s_b is not None:
                ax_full.axvline(
                    trigger_end_rising_rel_s_b,
                    linestyle=":",
                    linewidth=1.0,
                    color="purple",
                    label=f"End (rising) {label_b}",
                )
            ax_full.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zoom region")
            end_markers = [v for v in (trigger_end_rising_rel_s_a, trigger_end_rising_rel_s_b) if v is not None]
            if end_markers:
                end_zoom_t0 = float(min(end_markers) + zoom_t0)
                end_zoom_t1 = float(max(end_markers) + zoom_t1)
                ax_full.axvspan(end_zoom_t0, end_zoom_t1, alpha=0.10, color="gold", label="Trigger-end zoom region")
            spike_cmp_note = (
                f" — + raster / rate / ISI ({spike_cmp_pipe})"
                if _has_spike_cmp
                else ""
            )
            ax_full.set_title(
                f"Comparison — {channel_name} (full view){filt_note}{both_note}{spike_cmp_note}"
            )
            ax_full.set_ylabel("Potential (µV)")
            ax_full.set_xlabel(TIME_REL_XLABEL)
            ax_full.grid(True, alpha=0.3)
            ax_full.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.28),
                ncol=3,
                fontsize=LEGEND_FONT_SIZE,
            )
            if first_trigger_a_raw is not None or first_trigger_b_raw is not None:
                if first_trigger_a_raw is not None:
                    ax_first_trigger.plot(
                        t_rel,
                        first_trigger_a_raw,
                        linewidth=1.1,
                        color="C0",
                        label=f"{label_a} first trigger raw",
                    )
                if first_trigger_b_raw is not None:
                    ax_first_trigger.plot(
                        t_rel,
                        first_trigger_b_raw,
                        linewidth=1.1,
                        color="C1",
                        label=f"{label_b} first trigger raw",
                    )
                ax_first_trigger.axvline(0.0, linestyle="--", linewidth=1.0, color="red", label="Trigger (onset)")
                if trigger_end_rising_rel_s_a is not None:
                    ax_first_trigger.axvline(
                        trigger_end_rising_rel_s_a,
                        linestyle=":",
                        linewidth=1.0,
                        color="darkorange",
                        label=f"End (rising) {label_a}",
                    )
                if trigger_end_rising_rel_s_b is not None:
                    ax_first_trigger.axvline(
                        trigger_end_rising_rel_s_b,
                        linestyle=":",
                        linewidth=1.0,
                        color="purple",
                        label=f"End (rising) {label_b}",
                    )
                ax_first_trigger.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zoom region")
                end_markers_first = [
                    v for v in (trigger_end_rising_rel_s_a, trigger_end_rising_rel_s_b) if v is not None
                ]
                if end_markers_first:
                    end_zoom_t0 = float(min(end_markers_first) + zoom_t0)
                    end_zoom_t1 = float(max(end_markers_first) + zoom_t1)
                    ax_first_trigger.axvspan(
                        end_zoom_t0,
                        end_zoom_t1,
                        alpha=0.10,
                        color="gold",
                        label="Trigger-end zoom region",
                    )
                ax_first_trigger.set_title("First trigger — raw signal (no averaging)")
                ax_first_trigger.set_ylabel("Potential (µV)")
                ax_first_trigger.set_xlabel(TIME_REL_XLABEL)
                ax_first_trigger.grid(True, alpha=0.3)
            else:
                ax_first_trigger.text(
                    0.5,
                    0.5,
                    "First trigger raw signal unavailable",
                    ha="center",
                    va="center",
                    transform=ax_first_trigger.transAxes,
                )
                ax_first_trigger.set_axis_off()
            _plot_rms_series(
                ax_full_rms,
                [(label_a, rms_full_a[0], rms_full_a[1]), (label_b, rms_full_b[0], rms_full_b[1])],
                "Part 1 — RMS evolution (full window)",
                x_limits=(float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None,
            )

            if show_filtered_and_raw and channel_mean_a_raw is not None and channel_mean_b_raw is not None:
                ax_zoom.plot(
                    t_rel[zmask],
                    channel_mean_a_raw[zmask],
                    linewidth=1.1,
                    color="C0",
                    alpha=0.45,
                    label=f"{label_a} raw",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    channel_mean_b_raw[zmask],
                    linewidth=1.1,
                    color="C1",
                    alpha=0.45,
                    label=f"{label_b} raw",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    channel_mean_a[zmask],
                    linewidth=1.45,
                    color="C0",
                    label=f"{label_a} filtered",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    channel_mean_b[zmask],
                    linewidth=1.45,
                    color="C1",
                    label=f"{label_b} filtered",
                )
            else:
                ax_zoom.plot(t_rel[zmask], channel_mean_a[zmask], linewidth=1.4, color="C0", label=label_a)
                ax_zoom.plot(t_rel[zmask], channel_mean_b[zmask], linewidth=1.4, color="C1", label=label_b)
            ax_zoom.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
            if trigger_end_rising_rel_s_a is not None:
                ax_zoom.axvline(
                    trigger_end_rising_rel_s_a,
                    linestyle=":",
                    linewidth=1.0,
                    color="darkorange",
                )
            if trigger_end_rising_rel_s_b is not None:
                ax_zoom.axvline(
                    trigger_end_rising_rel_s_b,
                    linestyle=":",
                    linewidth=1.0,
                    color="purple",
                )
            ax_zoom.set_xlim(zoom_t0, zoom_t1)
            ax_zoom.set_title(zoom_title)
            ax_zoom.set_ylabel("Potential (µV)")
            ax_zoom.set_xlabel(TIME_REL_XLABEL)
            ax_zoom.grid(True, alpha=0.3)
            ax_zoom.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
            if first_trigger_a_raw is not None or first_trigger_b_raw is not None:
                if first_trigger_a_raw is not None:
                    ax_zoom_first.plot(
                        t_rel[zmask],
                        first_trigger_a_raw[zmask],
                        linewidth=1.2,
                        color="C0",
                        label=f"{label_a} first trigger raw",
                    )
                if first_trigger_b_raw is not None:
                    ax_zoom_first.plot(
                        t_rel[zmask],
                        first_trigger_b_raw[zmask],
                        linewidth=1.2,
                        color="C1",
                        label=f"{label_b} first trigger raw",
                    )
                ax_zoom_first.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
                if trigger_end_rising_rel_s_a is not None:
                    ax_zoom_first.axvline(trigger_end_rising_rel_s_a, linestyle=":", linewidth=1.0, color="darkorange")
                if trigger_end_rising_rel_s_b is not None:
                    ax_zoom_first.axvline(trigger_end_rising_rel_s_b, linestyle=":", linewidth=1.0, color="purple")
                ax_zoom_first.set_xlim(zoom_t0, zoom_t1)
                ax_zoom_first.set_title("Part 2 — First trigger raw (separate view)")
                ax_zoom_first.set_ylabel("Potential (µV)")
                ax_zoom_first.set_xlabel(TIME_REL_XLABEL)
                ax_zoom_first.grid(True, alpha=0.3)
            else:
                ax_zoom_first.text(0.5, 0.5, "First trigger raw signal unavailable", ha="center", va="center", transform=ax_zoom_first.transAxes)
                ax_zoom_first.set_axis_off()
            _plot_rms_series(
                ax_zoom_rms,
                    [(label_a, rms_zoom_a[0], rms_zoom_a[1]), (label_b, rms_zoom_b[0], rms_zoom_b[1])],
                f"Part 2 — RMS evolution (window [{zoom_t0:.3f}, {zoom_t1:.3f}] s)",
                x_limits=(zoom_t0, zoom_t1),
            )
            end_zoom_range: tuple[float, float] | None = None
            end_markers = [v for v in (trigger_end_rising_rel_s_a, trigger_end_rising_rel_s_b) if v is not None]
            if end_markers:
                end_zoom_t0 = float(min(end_markers) + zoom_t0)
                end_zoom_t1 = float(max(end_markers) + zoom_t1)
                end_zoom_range = (end_zoom_t0, end_zoom_t1)
                end_mask = (t_rel >= end_zoom_t0) & (t_rel <= end_zoom_t1)
                ax_zoom_end.plot(t_rel[end_mask], channel_mean_a[end_mask], linewidth=1.35, color="C0", label=label_a)
                ax_zoom_end.plot(t_rel[end_mask], channel_mean_b[end_mask], linewidth=1.35, color="C1", label=label_b)
                if trigger_end_rising_rel_s_a is not None:
                    ax_zoom_end.axvline(trigger_end_rising_rel_s_a, linestyle=":", linewidth=1.0, color="darkorange")
                if trigger_end_rising_rel_s_b is not None:
                    ax_zoom_end.axvline(trigger_end_rising_rel_s_b, linestyle=":", linewidth=1.0, color="purple")
                ax_zoom_end.set_xlim(end_zoom_t0, end_zoom_t1)
                ax_zoom_end.set_title(f"Part 3 — Trigger-end zoom [{end_zoom_t0:.2f}, {end_zoom_t1:.2f}] s")
                ax_zoom_end.set_ylabel("Potential (µV)")
                ax_zoom_end.set_xlabel(TIME_REL_XLABEL)
                ax_zoom_end.grid(True, alpha=0.3)
                ax_zoom_end.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
                if first_trigger_a_raw is not None or first_trigger_b_raw is not None:
                    if first_trigger_a_raw is not None:
                        ax_zoom_end_first.plot(
                            t_rel[end_mask],
                            first_trigger_a_raw[end_mask],
                            linewidth=1.2,
                            color="C0",
                            label=f"{label_a} first trigger raw",
                        )
                    if first_trigger_b_raw is not None:
                        ax_zoom_end_first.plot(
                            t_rel[end_mask],
                            first_trigger_b_raw[end_mask],
                            linewidth=1.2,
                            color="C1",
                            label=f"{label_b} first trigger raw",
                        )
                    if trigger_end_rising_rel_s_a is not None:
                        ax_zoom_end_first.axvline(trigger_end_rising_rel_s_a, linestyle=":", linewidth=1.0, color="darkorange")
                    if trigger_end_rising_rel_s_b is not None:
                        ax_zoom_end_first.axvline(trigger_end_rising_rel_s_b, linestyle=":", linewidth=1.0, color="purple")
                    ax_zoom_end_first.set_xlim(end_zoom_t0, end_zoom_t1)
                    ax_zoom_end_first.set_title("Part 3 — First trigger raw (separate view)")
                    ax_zoom_end_first.set_ylabel("Potential (µV)")
                    ax_zoom_end_first.set_xlabel(TIME_REL_XLABEL)
                    ax_zoom_end_first.grid(True, alpha=0.3)
                else:
                    ax_zoom_end_first.text(0.5, 0.5, "First trigger raw signal unavailable", ha="center", va="center", transform=ax_zoom_end_first.transAxes)
                    ax_zoom_end_first.set_axis_off()
                _plot_rms_series(
                    ax_zoom_end_rms,
                    [(label_a, rms_end_a[0], rms_end_a[1]), (label_b, rms_end_b[0], rms_end_b[1])],
                    f"Part 3 — RMS evolution (window [{end_zoom_t0:.3f}, {end_zoom_t1:.3f}] s)",
                    x_limits=(end_zoom_t0, end_zoom_t1),
                )
            else:
                ax_zoom_end.text(0.5, 0.5, "Trigger-end zoom unavailable\n(no rising edge after trigger)", ha="center", va="center", transform=ax_zoom_end.transAxes)
                ax_zoom_end.set_axis_off()
                ax_zoom_end_first.text(0.5, 0.5, "First trigger raw signal unavailable", ha="center", va="center", transform=ax_zoom_end_first.transAxes)
                ax_zoom_end_first.set_axis_off()
                ax_zoom_end_rms.text(0.5, 0.5, "RMS evolution unavailable", ha="center", va="center", transform=ax_zoom_end_rms.transAxes)
                ax_zoom_end_rms.set_axis_off()

            if _has_spike_cmp:
                # Spike detection is computed once per channel (A/B),
                # then reused for full / zoom / trigger-end zoom.
                sta = spike_source_a.spike_times_per_trial_for_channel(ch, t_rel, spike_threshold_uv)
                stb = spike_source_b.spike_times_per_trial_for_channel(ch, t_rel, spike_threshold_uv)
                _draw_spike_panels_dual_channel(
                    ax_raster_f,
                    ax_fr_f,
                    ax_trial_fr_f,
                    ax_isi_f,
                    None,
                    None,
                    t_rel,
                    float(fs),
                    spike_threshold_uv,
                    firing_rate_window_s,
                    label_a,
                    label_b,
                    spike_bandpass_low_hz,
                    spike_bandpass_high_hz,
                    t_range_s=None,
                    sta=sta,
                    stb=stb,
                    sampling_percent=sampling_percent,
                )
                _draw_spike_panels_dual_channel(
                    ax_raster_z,
                    ax_fr_z,
                    ax_trial_fr_z,
                    ax_isi_z,
                    None,
                    None,
                    t_rel,
                    float(fs),
                    spike_threshold_uv,
                    firing_rate_window_s,
                    label_a,
                    label_b,
                    spike_bandpass_low_hz,
                    spike_bandpass_high_hz,
                    t_range_s=(zoom_t0, zoom_t1),
                    sta=sta,
                    stb=stb,
                    sampling_percent=sampling_percent,
                )
                if end_zoom_range is not None:
                    _draw_spike_panels_dual_channel(
                        ax_raster_ze,
                        ax_fr_ze,
                        ax_trial_fr_ze,
                        ax_isi_ze,
                        None,
                        None,
                        t_rel,
                        float(fs),
                        spike_threshold_uv,
                        firing_rate_window_s,
                        label_a,
                        label_b,
                        spike_bandpass_low_hz,
                        spike_bandpass_high_hz,
                        t_range_s=end_zoom_range,
                        sta=sta,
                        stb=stb,
                        section_title="Trigger-end zoom",
                    )
                else:
                    for ax in (ax_raster_ze, ax_fr_ze, ax_trial_fr_ze):
                        ax.text(
                            0.5,
                            0.5,
                            "Trigger-end zoom unavailable\n(no rising edge after trigger)",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=9,
                        )
                        ax.set_axis_off()
                    ax_isi_ze.text(
                        0.5,
                        0.5,
                        "ISI unavailable",
                        ha="center",
                        va="center",
                        transform=ax_isi_ze.transAxes,
                    )
                    ax_isi_ze.set_axis_off()
            else:
                for ax in (ax_raster_f, ax_fr_f, ax_trial_fr_f, ax_raster_z, ax_fr_z, ax_trial_fr_z, ax_raster_ze, ax_fr_ze, ax_trial_fr_ze):
                    ax.text(
                        0.5,
                        0.5,
                        "Raster / PSTH / ISI unavailable\n(missing mmap sources)",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                    )
                    ax.set_axis_off()
                for ax in (ax_isi_f, ax_isi_z, ax_isi_ze):
                    ax.text(
                        0.5,
                        0.5,
                        "ISI unavailable",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()

            for ax in (
                ax_full,
                ax_first_trigger,
                ax_full_rms,
                ax_zoom,
                ax_zoom_first,
                ax_zoom_rms,
                ax_raster_f,
                ax_fr_f,
                ax_trial_fr_f,
                ax_isi_f,
                ax_raster_z,
                ax_fr_z,
                ax_trial_fr_z,
                ax_isi_z,
                ax_raster_ze,
                ax_fr_ze,
                ax_trial_fr_ze,
                ax_isi_ze,
                ax_zoom_end,
                ax_zoom_end_first,
                ax_zoom_end_rms,
            ):
                ax.tick_params(axis="x", labelbottom=True)

            fig.tight_layout()
            shift_axes_down(
                [
                    ax_first_trigger,
                    ax_full_rms,
                    ax_raster_f,
                    ax_fr_f,
                    ax_trial_fr_f,
                    ax_isi_f,
                    ax_hdr2,
                    ax_zoom,
                    ax_zoom_first,
                    ax_zoom_rms,
                    ax_raster_z,
                    ax_fr_z,
                    ax_trial_fr_z,
                    ax_isi_z,
                    ax_hdr3,
                    ax_zoom_end,
                    ax_zoom_end_first,
                    ax_zoom_end_rms,
                    ax_raster_ze,
                    ax_fr_ze,
                    ax_trial_fr_ze,
                    ax_isi_ze,
                ],
                delta=0.015,
            )
            _soften_figure_linewidths(fig)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2, dpi=_lightweight_pdf_dpi(lightweight_mode))
            plt.close(fig)

        rms_series: list[tuple[str, np.ndarray, np.ndarray]] = []
        t0_rms = float(t_rel[0]) if t_rel.size else 0.0
        t1_rms = float(t_rel[-1]) if t_rel.size else 0.0
        if spike_source_a is not None:
            rms_series.append(
                (label_a, *_mean_rms_profile_from_source_window(spike_source_a, t0_rms, t1_rms, rms_window_s))
            )
        if spike_source_b is not None:
            rms_series.append(
                (label_b, *_mean_rms_profile_from_source_window(spike_source_b, t0_rms, t1_rms, rms_window_s))
            )
        if rms_series:
            _append_mean_rms_evolution_page(pdf, rms_series, rms_window_s, lightweight_mode)

    _profile_print_delta(
        "plot_channel_comparison",
        _profile_before,
        time.perf_counter() - _profile_t0,
    )
    return pdf_path
