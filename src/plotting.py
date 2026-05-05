from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d

from core import (
    AmplifierSpikeSource,
    apply_butterworth_lowpass,
    check_analysis_cancelled,
    detect_spikes_at_threshold,
)
from impedance_tracking import ImpedanceSession

# Zoom panel window (s), time relative to trigger (t=0)
ZOOM_T0 = -0.1
ZOOM_T1 = 0.2

# ISI: only spikes within [-ISI_HALF_WINDOW_S, +ISI_HALF_WINDOW_S] (s relative to trigger)
ISI_HALF_WINDOW_S = 1.0

# X-axis (time relative to trigger, s) for ISI panels (time x ISI scatter)
ISI_ABSCISSA_T0_S = 0.0
ISI_ABSCISSA_T1_S = 2.0

# X-axis label for all time-relative-to-trigger plots
TIME_REL_XLABEL = "Time relative to trigger (s)"


def _downsample_points(x: np.ndarray, y: np.ndarray, sampling_percent: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically subsample (x, y) points."""
    pct = int(np.clip(int(sampling_percent), 1, 100))
    if pct >= 100 or x.size <= 1:
        return x, y
    keep = max(1, int(np.ceil(x.size * (pct / 100.0))))
    idx = np.linspace(0, x.size - 1, keep, dtype=np.int64)
    return x[idx], y[idx]


def _shorten_filename_for_windows(output_dir: Path, filename: str, max_full_len: int = 240) -> str:
    """Shorten a filename to keep Windows path length under control."""
    full = str(output_dir / filename)
    if len(full) <= max_full_len:
        return filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".pdf"
    overhead = len(str(output_dir / ("_" + suffix)))
    max_stem = max(8, max_full_len - overhead)
    return f"{stem[:max_stem]}{suffix}"


def _shift_axes_down(axes: Sequence[Any], delta: float) -> None:
    """Shift a group of axes downward (figure coordinates)."""
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - delta, pos.width, pos.height])


def _shorten_filename_for_windows(output_dir: Path, filename: str, max_total_len: int = 240) -> str:
    """Shorten filename if total path length may exceed Windows limits."""
    full_len = len(str(output_dir / filename))
    if full_len <= max_total_len:
        return filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".pdf"
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
    budget = max_total_len - len(str(output_dir)) - len(suffix) - len(digest) - 2
    budget = max(24, budget)
    short_stem = stem[:budget]
    return f"{short_stem}_{digest}{suffix}"


def _downsample_points(x: np.ndarray, y: np.ndarray, sampling_percent: int) -> tuple[np.ndarray, np.ndarray]:
    if sampling_percent >= 100:
        return x, y
    pct = max(1, min(100, int(sampling_percent)))
    step = max(1, int(np.ceil(100.0 / float(pct))))
    return x[::step], y[::step]


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


def _draw_spike_panels_single_channel(
    ax_raster: Any,
    ax_fr: Any,
    ax_trial_fr: Any,
    ax_isi: Any,
    w_ch: np.ndarray,
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
    lightweight_mode: bool = False,
    sampling_percent: int = 100,
) -> None:
    """Raster, PSTH / firing rate, ISI (time rel. trigger x duration) for one channel."""
    short, _ = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    n_tr = int(w_ch.shape[0])
    if st_per_tr is None:
        st_per_tr = _spike_times_per_trial(w_ch, t_rel, fs, spike_threshold_uv)
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
            st_ds, y_ds = _downsample_points(st_plot, y_pts, sampling_percent)
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
        t_isi, isi_vals_s = _downsample_points(t_isi, isi_vals_s, sampling_percent)
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
        ax_isi.set_xlim(ISI_ABSCISSA_T0_S, ISI_ABSCISSA_T1_S)
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
    w_a: np.ndarray,
    w_b: np.ndarray,
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
    lightweight_mode: bool = False,
    sampling_percent: int = 100,
) -> None:
    """Overlaid raster / PSTH / ISI for two recordings (same channel, same time axis)."""
    short, _ = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    n_a = int(w_a.shape[0])
    n_b = int(w_b.shape[0])
    if sta is None:
        sta = _spike_times_per_trial(w_a, t_rel, fs, spike_threshold_uv)
    if stb is None:
        stb = _spike_times_per_trial(w_b, t_rel, fs, spike_threshold_uv)

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
            st_ds, y_ds = _downsample_points(st_plot, y_pts, sampling_percent)
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
            st_ds, y_ds = _downsample_points(st_plot, y_pts, sampling_percent)
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
            t_a, isi_a_s = _downsample_points(t_a, isi_a_s, sampling_percent)
        if t_b.size:
            t_b, isi_b_s = _downsample_points(t_b, isi_b_s, sampling_percent)
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
        ax_isi.set_xlim(ISI_ABSCISSA_T0_S, ISI_ABSCISSA_T1_S)
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
    ax_isi: Any,
    windows_list: Sequence[np.ndarray],
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
    lightweight_mode: bool = False,
    sampling_percent: int = 100,
) -> None:
    """Overlaid raster / PSTH / ISI for N recordings."""
    short, _ = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    if spikes_per_recording is None:
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
                st_ds, y_ds = _downsample_points(st_plot, y_pts, sampling_percent)
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
    fr_rows: list[tuple[str, float]] = []
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        mean_fr = _mean_firing_rate_in_window_hz(st_per_trial, (t_xlim_lo, t_xlim_hi))
        fr_rows.append((labels[rec_idx], mean_fr))
    _add_psth_mean_table(ax_fr, fr_rows)

    has_isi = False
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        tx, isi_vals_s = _isi_time_and_values_s(st_per_trial, isi_window_s=isi_window)
        if tx.size:
            has_isi = True
            tx, isi_vals_s = _downsample_points(tx, isi_vals_s, sampling_percent)
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
        ax_isi.set_xlim(ISI_ABSCISSA_T0_S, ISI_ABSCISSA_T1_S)
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
    windows_list: Sequence[np.ndarray],
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
    lightweight_mode: bool = False,
    sampling_percent: int = 100,
) -> None:
    """Overlaid raster / PSTH / ISI for N recordings."""
    short, _ = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    if spikes_per_recording is None:
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
                st_ds, y_ds = _downsample_points(st_plot, y_pts, sampling_percent)
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
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        tx, isi_vals_s = _isi_time_and_values_s(st_per_trial, isi_window_s=isi_window)
        if tx.size:
            has_isi = True
            tx, isi_vals_s = _downsample_points(tx, isi_vals_s, sampling_percent)
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
        ax_isi.set_xlim(ISI_ABSCISSA_T0_S, ISI_ABSCISSA_T1_S)
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
    dpi = 100 if lightweight_mode else 120
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

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25, dpi=dpi)
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
    zoom_t0_s: float = ZOOM_T0,
    zoom_t1_s: float = ZOOM_T1,
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
    lightweight_mode: bool = False,
    sampling_percent: int = 100,
    streaming_mode: bool = False,
    pre_n_common: Optional[int] = None,
    post_n_common: Optional[int] = None,
    impedance_sessions: Optional[Sequence[ImpedanceSession]] = None,
) -> Path:
    """Multi-page PDF: overlay of N recordings (same aligned channels)."""
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
    pdf_name = _shorten_filename_for_windows(output_dir, f"{pdf_stem}.pdf")
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

    with PdfPages(pdf_path) as pdf:
        for ch in range(n_channels):
            check_analysis_cancelled()
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

            hr_multi = [0.06, 1.05, 1.10, 0.90, 0.85, 0.95, 0.06, 1.05, 1.10, 0.90, 0.85, 0.95, 0.06, 1.05, 1.10, 0.90, 0.85, 0.95]
            if impedance_sessions:
                hr_layout = [*hr_multi, 0.05, 0.95]
                fig = plt.figure(figsize=(12, 42))
                gs = fig.add_gridspec(len(hr_layout), 1, height_ratios=hr_layout, hspace=0.88)
            else:
                fig = plt.figure(figsize=(12, 38))
                gs = fig.add_gridspec(18, 1, height_ratios=hr_multi, hspace=0.9)
            ax_hdr1 = fig.add_subplot(gs[0, 0]); ax_hdr1.axis("off")
            ax_hdr1.text(0.02, 0.5, "Part 1 — Full view (entire pre/post-trigger window)", ha="left", va="center", fontsize=11, fontweight="bold", transform=ax_hdr1.transAxes)
            ax_full = fig.add_subplot(gs[1, 0])
            ax_raster_f = fig.add_subplot(gs[2, 0], sharex=ax_full)
            ax_fr_f = fig.add_subplot(gs[3, 0], sharex=ax_full)
            ax_trial_fr_f = fig.add_subplot(gs[4, 0])
            ax_isi_f = fig.add_subplot(gs[5, 0])
            ax_hdr2 = fig.add_subplot(gs[6, 0]); ax_hdr2.axis("off")
            ax_hdr2.text(0.02, 0.5, f"Part 2 — Zoomed view [{zoom_t0:.2f}, {zoom_t1:.2f}] s (relative to trigger)", ha="left", va="center", fontsize=11, fontweight="bold", transform=ax_hdr2.transAxes)
            ax_zoom = fig.add_subplot(gs[7, 0])
            ax_raster_z = fig.add_subplot(gs[8, 0], sharex=ax_zoom)
            ax_fr_z = fig.add_subplot(gs[9, 0], sharex=ax_zoom)
            ax_trial_fr_z = fig.add_subplot(gs[10, 0])
            ax_isi_z = fig.add_subplot(gs[11, 0])
            ax_hdr3 = fig.add_subplot(gs[12, 0]); ax_hdr3.axis("off")
            ax_hdr3.text(0.02, 0.5, "Part 3 — Trigger-end zoom (rising edge)", ha="left", va="center", fontsize=11, fontweight="bold", transform=ax_hdr3.transAxes)
            ax_zoom_end = fig.add_subplot(gs[13, 0])
            ax_raster_ze = fig.add_subplot(gs[14, 0], sharex=ax_zoom_end)
            ax_fr_ze = fig.add_subplot(gs[15, 0], sharex=ax_zoom_end)
            ax_trial_fr_ze = fig.add_subplot(gs[16, 0])
            ax_isi_ze = fig.add_subplot(gs[17, 0])

            if impedance_sessions:
                ax_imp_hdr = fig.add_subplot(gs[18, 0])
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
                ax_imp = fig.add_subplot(gs[19, 0])
                _draw_impedance_evolution_panel(ax_imp, channel_names[ch], impedance_sessions)

            show_both = (
                lowpass_cutoff_hz is not None
                and means_raw_ch is not None
                and len(means_raw_ch) == len(means_ch)
                and any(
                    not np.allclose(np.asarray(means_ch[i]), np.asarray(means_raw_ch[i]), rtol=0.0, atol=1e-9)
                    for i in range(len(means_ch))
                )
            )
            for i, y in enumerate(means_ch):
                c = colors[i % len(colors)]
                if show_both and means_raw_ch is not None:
                    y_raw = np.asarray(means_raw_ch[i])
                    ax_full.plot(t_rel, y_raw, linewidth=1.0, color=c, alpha=0.35, label="_nolegend_")
                    ax_full.plot(t_rel, y, linewidth=1.35, color=c, label=f"{labels[i]} (filtered)")
                    ax_zoom.plot(t_rel[zmask], y_raw[zmask], linewidth=1.05, color=c, alpha=0.35)
                    ax_zoom.plot(t_rel[zmask], y[zmask], linewidth=1.35, color=c, label=labels[i])
                else:
                    ax_full.plot(t_rel, y, linewidth=1.2, color=c, label=labels[i])
                    ax_zoom.plot(t_rel[zmask], y[zmask], linewidth=1.3, color=c, label=labels[i])
            ax_full.axvline(0.0, linestyle="--", linewidth=1.0, color="red", label="Trigger (onset)")
            for v in end_markers:
                ax_full.axvline(v, linestyle=":", linewidth=0.9, color="0.45")
                ax_zoom.axvline(v, linestyle=":", linewidth=0.9, color="0.45")
            ax_full.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zoom region")
            if end_markers:
                end_zoom_t0 = float(min(end_markers) + zoom_t0)
                end_zoom_t1 = float(max(end_markers) + zoom_t1)
                ax_full.axvspan(end_zoom_t0, end_zoom_t1, alpha=0.10, color="gold", label="Trigger-end zoom region")
            ax_full.set_title(f"Multi-comparison — {channel_names[ch]} (full view){filt_note}{both_note}")
            ax_full.set_ylabel("Potential (µV)")
            ax_full.set_xlabel(TIME_REL_XLABEL)
            ax_full.grid(True, alpha=0.3)
            ax_full.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=4, fontsize=6)

            ax_zoom.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
            ax_zoom.set_xlim(zoom_t0, zoom_t1)
            ax_zoom.set_title(zoom_title)
            ax_zoom.set_ylabel("Potential (µV)")
            ax_zoom.set_xlabel(TIME_REL_XLABEL)
            ax_zoom.grid(True, alpha=0.3)

            end_zoom_range: tuple[float, float] | None = None
            if end_markers:
                end_zoom_t0 = float(min(end_markers) + zoom_t0)
                end_zoom_t1 = float(max(end_markers) + zoom_t1)
                end_zoom_range = (end_zoom_t0, end_zoom_t1)
                end_mask = (t_rel >= end_zoom_t0) & (t_rel <= end_zoom_t1)
                for i, y in enumerate(means_ch):
                    c = colors[i % len(colors)]
                    if show_both and means_raw_ch is not None:
                        y_raw = np.asarray(means_raw_ch[i])
                        ax_zoom_end.plot(t_rel[end_mask], y_raw[end_mask], linewidth=1.05, color=c, alpha=0.35)
                        ax_zoom_end.plot(t_rel[end_mask], y[end_mask], linewidth=1.35, color=c, label=labels[i])
                    else:
                        ax_zoom_end.plot(t_rel[end_mask], y[end_mask], linewidth=1.35, color=c, label=labels[i])
                ax_zoom_end.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
                for v in end_markers:
                    ax_zoom_end.axvline(v, linestyle=":", linewidth=0.9, color="0.45")
                ax_zoom_end.set_xlim(end_zoom_t0, end_zoom_t1)
                ax_zoom_end.set_title(f"Trigger-end zoom: {end_zoom_t0:.2f} to {end_zoom_t1:.2f} s (relative to trigger){filt_note}{both_note}")
                ax_zoom_end.set_ylabel("Potential (µV)")
                ax_zoom_end.set_xlabel(TIME_REL_XLABEL)
                ax_zoom_end.grid(True, alpha=0.3)
            else:
                ax_zoom_end.text(0.5, 0.5, "Trigger-end zoom unavailable\n(no rising edge after trigger)", ha="center", va="center", transform=ax_zoom_end.transAxes)
                ax_zoom_end.set_axis_off()

            if _has_spike_cmp and spike_sources is not None:
                sources_ok = [src for src in spike_sources if src is not None]
                # Spike detection is computed once per channel/file,
                # then reused for full / zoom / trigger-end zoom.
                st_list = [
                    src.spike_times_per_trial_for_channel(ch, t_rel, spike_threshold_uv)
                    for src in sources_ok
                ]
                w_list = [np.empty((len(st), 1), dtype=np.float32) for st in st_list]
                _draw_spike_panels_multi_channel(
                    ax_raster_f, ax_fr_f, ax_trial_fr_f, ax_isi_f, w_list, t_rel, float(fs), spike_threshold_uv,
                    firing_rate_window_s, labels, spike_bandpass_low_hz, spike_bandpass_high_hz,
                    t_range_s=None, spikes_per_recording=st_list, lightweight_mode=lightweight_mode, sampling_percent=sampling_percent,
                )
                _draw_spike_panels_multi_channel(
                    ax_raster_z, ax_fr_z, ax_trial_fr_z, ax_isi_z, w_list, t_rel, float(fs), spike_threshold_uv,
                    firing_rate_window_s, labels, spike_bandpass_low_hz, spike_bandpass_high_hz,
                    t_range_s=(zoom_t0, zoom_t1), spikes_per_recording=st_list, lightweight_mode=lightweight_mode, sampling_percent=sampling_percent,
                )
                if end_zoom_range is not None:
                    _draw_spike_panels_multi_channel(
                        ax_raster_ze, ax_fr_ze, ax_trial_fr_ze, ax_isi_ze, w_list, t_rel, float(fs), spike_threshold_uv,
                        firing_rate_window_s, labels, spike_bandpass_low_hz, spike_bandpass_high_hz,
                        t_range_s=end_zoom_range, section_title="Trigger-end zoom", spikes_per_recording=st_list, lightweight_mode=lightweight_mode, sampling_percent=sampling_percent,
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
                ax_zoom,
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
            ]
            if impedance_sessions:
                _axes_tick_bottom.append(ax_imp)
            for ax in _axes_tick_bottom:
                ax.tick_params(axis="x", labelbottom=True)

            fig.tight_layout()
            _shift_axes_down(
                [
                    ax_raster_f,
                    ax_fr_f,
                    ax_trial_fr_f,
                    ax_isi_f,
                    ax_hdr2,
                    ax_zoom,
                    ax_raster_z,
                    ax_fr_z,
                    ax_trial_fr_z,
                    ax_isi_z,
                    ax_hdr3,
                    ax_zoom_end,
                    ax_raster_ze,
                    ax_fr_ze,
                    ax_trial_fr_ze,
                    ax_isi_ze,
                ],
                delta=0.015,
            )
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2, dpi=100 if lightweight_mode else 120)
            plt.close(fig)

        if impedance_sessions:
            _append_mean_impedance_summary_page(pdf, impedance_sessions, lightweight_mode)

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
    zoom_t0_s: float = ZOOM_T0,
    zoom_t1_s: float = ZOOM_T1,
    mean_per_channel_raw: Optional[np.ndarray] = None,
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
    lightweight_mode: bool = False,
    sampling_percent: int = 100,
) -> Path:
    """Single multi-page PDF (one page per channel), without GUI window.

    Intan amplifier values are displayed in microvolts (uV), as
    provided by load_intan_rhs_format (amplifier_data).
    """
    n_channels = mean_per_channel.shape[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    if pdf_title is not None and pdf_title.strip():
        safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in pdf_title.strip())
        pdf_stem = safe_title.removesuffix(".pdf")
    else:
        pdf_stem = rhs_file.stem
    pdf_name = _shorten_filename_for_windows(output_dir, f"{pdf_stem}.pdf")
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

    with PdfPages(pdf_path) as pdf:
        for ch in range(n_channels):
            check_analysis_cancelled()
            y = mean_per_channel[ch]
            fig = plt.figure(figsize=(12, 38))
            hr = [
                0.06,
                1.05,
                1.15,
                0.95,
                0.95,
                0.06,
                1.05,
                1.15,
                0.95,
                0.95,
                0.06,
                1.05,
                1.15,
                0.95,
                0.95,
            ]
            gs = fig.add_gridspec(18, 1, height_ratios=[0.06, 1.05, 1.10, 0.90, 0.85, 0.95, 0.06, 1.05, 1.10, 0.90, 0.85, 0.95, 0.06, 1.05, 1.10, 0.90, 0.85, 0.95], hspace=0.9)
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
            ax_raster_f = fig.add_subplot(gs[2, 0], sharex=ax_full)
            ax_fr_f = fig.add_subplot(gs[3, 0], sharex=ax_full)
            ax_trial_fr_f = fig.add_subplot(gs[4, 0])
            ax_isi_f = fig.add_subplot(gs[5, 0])
            ax_hdr2 = fig.add_subplot(gs[6, 0])
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
            ax_zoom = fig.add_subplot(gs[7, 0])
            ax_raster_z = fig.add_subplot(gs[8, 0], sharex=ax_zoom)
            ax_fr_z = fig.add_subplot(gs[9, 0], sharex=ax_zoom)
            ax_trial_fr_z = fig.add_subplot(gs[10, 0])
            ax_isi_z = fig.add_subplot(gs[11, 0])
            ax_hdr3 = fig.add_subplot(gs[12, 0])
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
            ax_zoom_end = fig.add_subplot(gs[13, 0])
            ax_raster_ze = fig.add_subplot(gs[14, 0], sharex=ax_zoom_end)
            ax_fr_ze = fig.add_subplot(gs[15, 0], sharex=ax_zoom_end)
            ax_trial_fr_ze = fig.add_subplot(gs[16, 0])
            ax_isi_ze = fig.add_subplot(gs[17, 0])

            y_raw = (
                mean_per_channel_raw[ch]
                if mean_per_channel_raw is not None
                else None
            )
            show_both = (
                lowpass_cutoff_hz is not None
                and mean_per_channel_raw is not None
                and y_raw is not None
                and not np.allclose(y, y_raw, rtol=0.0, atol=1e-9)
            )
            if show_both:
                ax_full.plot(
                    t_rel,
                    y_raw,
                    linewidth=1.05,
                    color="0.35",
                    alpha=0.85,
                    label="Mean (raw)",
                    zorder=1,
                )
                ax_full.plot(
                    t_rel,
                    y,
                    linewidth=1.35,
                    color="C0",
                    label=f"Mean (filtered, {lowpass_cutoff_hz:g} Hz)",
                    zorder=2,
                )
            else:
                ax_full.plot(t_rel, y, linewidth=1.2, color="C0", label="Mean")
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
                f"Trigger mean — {channel_names[ch]} (full view){filt_note}{both_note}{spike_note}"
            )
            ax_full.set_ylabel("Potential (µV)")
            ax_full.set_xlabel(TIME_REL_XLABEL)
            ax_full.grid(True, alpha=0.3)
            ax_full.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.36),
                ncol=3,
                fontsize=6,
            )

            zmask = (t_rel >= zoom_t0) & (t_rel <= zoom_t1)
            if show_both and y_raw is not None:
                ax_zoom.plot(
                    t_rel[zmask],
                    y_raw[zmask],
                    linewidth=1.15,
                    color="0.35",
                    alpha=0.85,
                    label="Raw",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    y[zmask],
                    linewidth=1.45,
                    color="C0",
                    label=f"Filtered ({lowpass_cutoff_hz:g} Hz)",
                )
            else:
                ax_zoom.plot(t_rel[zmask], y[zmask], linewidth=1.4, color="C0")
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
            if show_both:
                ax_zoom.legend(loc="best", fontsize=6)
            end_zoom_range: tuple[float, float] | None = None
            if trigger_end_rising_rel_s is not None:
                end_zoom_t0 = float(trigger_end_rising_rel_s + zoom_t0)
                end_zoom_t1 = float(trigger_end_rising_rel_s + zoom_t1)
                end_zoom_range = (end_zoom_t0, end_zoom_t1)
                end_mask = (t_rel >= end_zoom_t0) & (t_rel <= end_zoom_t1)
                if show_both and y_raw is not None:
                    ax_zoom_end.plot(t_rel[end_mask], y_raw[end_mask], linewidth=1.15, color="0.35", alpha=0.85, label="Unfiltered")
                    ax_zoom_end.plot(t_rel[end_mask], y[end_mask], linewidth=1.45, color="C0", label=f"Filtered ({lowpass_cutoff_hz:g} Hz)")
                else:
                    ax_zoom_end.plot(t_rel[end_mask], y[end_mask], linewidth=1.4, color="C0")
                ax_zoom_end.axvline(trigger_end_rising_rel_s, linestyle="--", linewidth=1.0, color="darkorange")
                ax_zoom_end.set_xlim(end_zoom_t0, end_zoom_t1)
                ax_zoom_end.set_title(f"Part 3 — Trigger-end zoom [{end_zoom_t0:.2f}, {end_zoom_t1:.2f}] s")
                ax_zoom_end.set_ylabel("Potential (µV)")
                ax_zoom_end.set_xlabel(TIME_REL_XLABEL)
                ax_zoom_end.grid(True, alpha=0.3)
            else:
                ax_zoom_end.text(0.5, 0.5, "Trigger-end zoom unavailable\n(no rising edge after trigger)", ha="center", va="center", transform=ax_zoom_end.transAxes)
                ax_zoom_end.set_axis_off()

            if _has_spike_data and (
                spike_source is not None
                or (windows is not None and windows.ndim == 3)
            ):
                if spike_source is not None:
                    st_per_tr = spike_source.spike_times_per_trial_for_channel(
                        ch, t_rel, spike_threshold_uv
                    )
                    w_ch = np.empty((len(st_per_tr), 1), dtype=np.float32)
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
                    w_ch,
                    t_rel,
                    float(fs),
                    spike_threshold_uv,
                    firing_rate_window_s,
                    spike_bandpass_low_hz,
                    spike_bandpass_high_hz,
                    t_range_s=None,
                    st_per_tr=st_per_tr,
                    lightweight_mode=lightweight_mode,
                    sampling_percent=sampling_percent,
                )
                _draw_spike_panels_single_channel(
                    ax_raster_z,
                    ax_fr_z,
                    ax_trial_fr_z,
                    ax_isi_z,
                    w_ch,
                    t_rel,
                    float(fs),
                    spike_threshold_uv,
                    firing_rate_window_s,
                    spike_bandpass_low_hz,
                    spike_bandpass_high_hz,
                    t_range_s=(zoom_t0, zoom_t1),
                    st_per_tr=st_per_tr,
                    lightweight_mode=lightweight_mode,
                    sampling_percent=sampling_percent,
                )
                if end_zoom_range is not None:
                    _draw_spike_panels_single_channel(
                        ax_raster_ze,
                        ax_fr_ze,
                        ax_trial_fr_ze,
                        ax_isi_ze,
                        w_ch,
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
                ax_zoom,
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
            ):
                ax.tick_params(axis="x", labelbottom=True)

            fig.tight_layout()
            _shift_axes_down(
                [
                    ax_raster_f,
                    ax_fr_f,
                    ax_trial_fr_f,
                    ax_isi_f,
                    ax_hdr2,
                    ax_zoom,
                    ax_raster_z,
                    ax_fr_z,
                    ax_trial_fr_z,
                    ax_isi_z,
                    ax_hdr3,
                    ax_zoom_end,
                    ax_raster_ze,
                    ax_fr_ze,
                    ax_trial_fr_ze,
                    ax_isi_ze,
                ],
                delta=0.015,
            )
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2, dpi=100 if lightweight_mode else 120)
            plt.close(fig)

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
    zoom_t0_s: float = ZOOM_T0,
    zoom_t1_s: float = ZOOM_T1,
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
    lightweight_mode: bool = False,
    sampling_percent: int = 100,
) -> Path:
    """Multi-page PDF: per channel, means + zoom + raster / PSTH / ISI (two overlaid recordings)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_a = "".join(c if c.isalnum() or c in "._-" else "_" for c in label_a)[:80]
    safe_b = "".join(c if c.isalnum() or c in "._-" else "_" for c in label_b)[:80]
    if pdf_title is not None and pdf_title.strip():
        safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in pdf_title.strip())
        pdf_stem = safe_title.removesuffix(".pdf")
    else:
        pdf_stem = f"{safe_a}_vs_{safe_b}"
    pdf_name = _shorten_filename_for_windows(output_dir, f"{pdf_stem}.pdf")
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
            ya = mean_a[ch]
            yb = mean_b[ch]
            ya_raw = mean_a_raw[ch] if mean_a_raw is not None else None
            yb_raw = mean_b_raw[ch] if mean_b_raw is not None else None
            show_both = (
                lowpass_cutoff_hz is not None
                and ya_raw is not None
                and yb_raw is not None
                and (
                    not np.allclose(ya, ya_raw, rtol=0.0, atol=1e-9)
                    or not np.allclose(yb, yb_raw, rtol=0.0, atol=1e-9)
                )
            )
            fig = plt.figure(figsize=(12, 38))
            hr = [
                0.06,
                1.05,
                1.15,
                0.95,
                0.95,
                0.06,
                1.05,
                1.15,
                0.95,
                0.95,
                0.06,
                1.05,
                1.15,
                0.95,
                0.95,
            ]
            gs = fig.add_gridspec(18, 1, height_ratios=[0.06, 1.05, 1.10, 0.90, 0.85, 0.95, 0.06, 1.05, 1.10, 0.90, 0.85, 0.95, 0.06, 1.05, 1.10, 0.90, 0.85, 0.95], hspace=0.9)
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
            ax_raster_f = fig.add_subplot(gs[2, 0], sharex=ax_full)
            ax_fr_f = fig.add_subplot(gs[3, 0], sharex=ax_full)
            ax_trial_fr_f = fig.add_subplot(gs[4, 0])
            ax_isi_f = fig.add_subplot(gs[5, 0])
            ax_hdr2 = fig.add_subplot(gs[6, 0])
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
            ax_zoom = fig.add_subplot(gs[7, 0])
            ax_raster_z = fig.add_subplot(gs[8, 0], sharex=ax_zoom)
            ax_fr_z = fig.add_subplot(gs[9, 0], sharex=ax_zoom)
            ax_trial_fr_z = fig.add_subplot(gs[10, 0])
            ax_isi_z = fig.add_subplot(gs[11, 0])
            ax_hdr3 = fig.add_subplot(gs[12, 0])
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
            ax_zoom_end = fig.add_subplot(gs[13, 0])
            ax_raster_ze = fig.add_subplot(gs[14, 0], sharex=ax_zoom_end)
            ax_fr_ze = fig.add_subplot(gs[15, 0], sharex=ax_zoom_end)
            ax_trial_fr_ze = fig.add_subplot(gs[16, 0])
            ax_isi_ze = fig.add_subplot(gs[17, 0])
            if show_both:
                ax_full.plot(
                    t_rel,
                    ya_raw,
                    linewidth=1.0,
                    color="C0",
                    alpha=0.4,
                    label="_nolegend_",
                )
                ax_full.plot(
                    t_rel,
                    yb_raw,
                    linewidth=1.0,
                    color="C1",
                    alpha=0.4,
                    label="_nolegend_",
                )
                ax_full.plot(
                    t_rel,
                    ya,
                    linewidth=1.35,
                    color="C0",
                    label=f"{label_a} (filtered {lowpass_cutoff_hz:g} Hz)",
                )
                ax_full.plot(
                    t_rel,
                    yb,
                    linewidth=1.35,
                    color="C1",
                    label=f"{label_b} (filtered {lowpass_cutoff_hz:g} Hz)",
                )
            else:
                ax_full.plot(t_rel, ya, linewidth=1.2, color="C0", label=label_a)
                ax_full.plot(t_rel, yb, linewidth=1.2, color="C1", label=label_b)
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
                f"Comparison — {channel_names[ch]} (full view){filt_note}{both_note}{spike_cmp_note}"
            )
            ax_full.set_ylabel("Potential (µV)")
            ax_full.set_xlabel(TIME_REL_XLABEL)
            ax_full.grid(True, alpha=0.3)
            ax_full.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.36),
                ncol=3,
                fontsize=6,
            )

            if show_both and ya_raw is not None and yb_raw is not None:
                ax_zoom.plot(
                    t_rel[zmask],
                    ya_raw[zmask],
                    linewidth=1.1,
                    color="C0",
                    alpha=0.45,
                    label=f"{label_a} raw",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    yb_raw[zmask],
                    linewidth=1.1,
                    color="C1",
                    alpha=0.45,
                    label=f"{label_b} raw",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    ya[zmask],
                    linewidth=1.45,
                    color="C0",
                    label=f"{label_a} filtered",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    yb[zmask],
                    linewidth=1.45,
                    color="C1",
                    label=f"{label_b} filtered",
                )
            else:
                ax_zoom.plot(t_rel[zmask], ya[zmask], linewidth=1.4, color="C0", label=label_a)
                ax_zoom.plot(t_rel[zmask], yb[zmask], linewidth=1.4, color="C1", label=label_b)
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
            ax_zoom.legend(loc="best", fontsize=6)
            end_zoom_range: tuple[float, float] | None = None
            end_markers = [v for v in (trigger_end_rising_rel_s_a, trigger_end_rising_rel_s_b) if v is not None]
            if end_markers:
                end_zoom_t0 = float(min(end_markers) + zoom_t0)
                end_zoom_t1 = float(max(end_markers) + zoom_t1)
                end_zoom_range = (end_zoom_t0, end_zoom_t1)
                end_mask = (t_rel >= end_zoom_t0) & (t_rel <= end_zoom_t1)
                ax_zoom_end.plot(t_rel[end_mask], ya[end_mask], linewidth=1.35, color="C0", label=label_a)
                ax_zoom_end.plot(t_rel[end_mask], yb[end_mask], linewidth=1.35, color="C1", label=label_b)
                if trigger_end_rising_rel_s_a is not None:
                    ax_zoom_end.axvline(trigger_end_rising_rel_s_a, linestyle=":", linewidth=1.0, color="darkorange")
                if trigger_end_rising_rel_s_b is not None:
                    ax_zoom_end.axvline(trigger_end_rising_rel_s_b, linestyle=":", linewidth=1.0, color="purple")
                ax_zoom_end.set_xlim(end_zoom_t0, end_zoom_t1)
                ax_zoom_end.set_title(f"Part 3 — Trigger-end zoom [{end_zoom_t0:.2f}, {end_zoom_t1:.2f}] s")
                ax_zoom_end.set_ylabel("Potential (µV)")
                ax_zoom_end.set_xlabel(TIME_REL_XLABEL)
                ax_zoom_end.grid(True, alpha=0.3)
                ax_zoom_end.legend(loc="best", fontsize=6)
            else:
                ax_zoom_end.text(0.5, 0.5, "Trigger-end zoom unavailable\n(no rising edge after trigger)", ha="center", va="center", transform=ax_zoom_end.transAxes)
                ax_zoom_end.set_axis_off()

            if _has_spike_cmp:
                # Spike detection is computed once per channel (A/B),
                # then reused for full / zoom / trigger-end zoom.
                sta = spike_source_a.spike_times_per_trial_for_channel(ch, t_rel, spike_threshold_uv)
                stb = spike_source_b.spike_times_per_trial_for_channel(ch, t_rel, spike_threshold_uv)
                w_a = np.empty((len(sta), 1), dtype=np.float32)
                w_b = np.empty((len(stb), 1), dtype=np.float32)
                _draw_spike_panels_dual_channel(
                    ax_raster_f,
                    ax_fr_f,
                    ax_trial_fr_f,
                    ax_isi_f,
                    w_a,
                    w_b,
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
                    lightweight_mode=lightweight_mode,
                    sampling_percent=sampling_percent,
                )
                _draw_spike_panels_dual_channel(
                    ax_raster_z,
                    ax_fr_z,
                    ax_trial_fr_z,
                    ax_isi_z,
                    w_a,
                    w_b,
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
                    lightweight_mode=lightweight_mode,
                    sampling_percent=sampling_percent,
                )
                if end_zoom_range is not None:
                    _draw_spike_panels_dual_channel(
                        ax_raster_ze,
                        ax_fr_ze,
                        ax_trial_fr_ze,
                        ax_isi_ze,
                        w_a,
                        w_b,
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
                ax_zoom,
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
            ):
                ax.tick_params(axis="x", labelbottom=True)

            fig.tight_layout()
            _shift_axes_down(
                [
                    ax_raster_f,
                    ax_fr_f,
                    ax_trial_fr_f,
                    ax_isi_f,
                    ax_hdr2,
                    ax_zoom,
                    ax_raster_z,
                    ax_fr_z,
                    ax_trial_fr_z,
                    ax_isi_z,
                    ax_hdr3,
                    ax_zoom_end,
                    ax_raster_ze,
                    ax_fr_ze,
                    ax_trial_fr_ze,
                    ax_isi_ze,
                ],
                delta=0.015,
            )
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2, dpi=100 if lightweight_mode else 120)
            plt.close(fig)

    return pdf_path
