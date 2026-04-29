from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d

from core import AmplifierSpikeSource, check_analysis_cancelled, detect_spikes_at_threshold

# Fenetre du panneau zoom (s), temps relatif au trigger (t=0)
ZOOM_T0 = -0.1
ZOOM_T1 = 0.2

# ISI : uniquement les spikes dans [-ISI_HALF_WINDOW_S, +ISI_HALF_WINDOW_S] (s rel. trigger)
ISI_HALF_WINDOW_S = 1.0

# Abscisse (temps rel. trigger, s) des panneaux ISI (nuage temps × ISI)
ISI_ABSCISSA_T0_S = 0.0
ISI_ABSCISSA_T1_S = 2.0

# Libellé d'abscisse pour tous les graphiques en temps rel. au trigger
TIME_REL_XLABEL = "Temps relatif au trigger (s)"


def _shift_axes_down(axes: Sequence[Any], delta: float) -> None:
    """Décale un groupe d'axes vers le bas (coordonnées figure)."""
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - delta, pos.width, pos.height])


def _spike_times_per_trial(
    windows_ch: np.ndarray,
    t_rel: np.ndarray,
    fs: float,
    threshold: float,
) -> list[np.ndarray]:
    """Pour un canal : liste de tableaux de temps (s) relatifs au trigger, un par essai."""
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
    """PSTH moyen (Hz) : nombre de spikes par bin / (n_trials * bin_width).

    t_range_s : si (t0, t1), histogramme sur cet intervalle uniquement (spikes hors plage exclus).
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


<<<<<<< Updated upstream
=======
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


>>>>>>> Stashed changes
def _spike_pipeline_captions(
    spike_bandpass_low_hz: Optional[float],
    spike_bandpass_high_hz: Optional[float],
) -> Tuple[str, str]:
    """(court pour sous-titres, détail pour note de page)"""
    if spike_bandpass_low_hz is not None and spike_bandpass_high_hz is not None:
        flo = float(spike_bandpass_low_hz)
        fhi = float(spike_bandpass_high_hz)
        short = f"passe-bande {flo:g}–{fhi:g} Hz"
        detail = f"Butterworth passe-bande {flo:g}–{fhi:g} Hz (ordre 4, filtfilt)"
        return short, detail
    return "brut", "signal brut mmap (sans passe-bande spikes)"


def _isi_time_and_values_s(
    spike_times_per_trial: list[np.ndarray],
    *,
    isi_window_s: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Temps de fin d'intervalle (s rel. trigger) et ISI (s) pour chaque paire consécutive."""
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
    """ISI intra-essai (s), concaténation des écarts."""
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
) -> None:
    """Raster, PSTH / firing rate, ISI (temps rel. trigger × durée) pour un canal."""
    short, _ = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    n_tr = int(w_ch.shape[0])
    if st_per_tr is None:
        st_per_tr = _spike_times_per_trial(w_ch, t_rel, fs, spike_threshold_uv)
    if t_range_s is None:
        t_xlim_lo, t_xlim_hi = float(t_rel[0]), float(t_rel[-1])
        psth_t_range: Optional[Tuple[float, float]] = None
        isi_window: Optional[Tuple[float, float]] = None
        isi_caption = f"spikes dans ±{ISI_HALF_WINDOW_S:g} s du trigger, intra-essai"
        isi_empty_hint = (
            "moins de 2 spikes dans ±"
            f"{ISI_HALF_WINDOW_S:g} s du trigger, par essai"
        )
    else:
        t_xlim_lo, t_xlim_hi = float(t_range_s[0]), float(t_range_s[1])
        psth_t_range = (t_xlim_lo, t_xlim_hi)
        isi_window = (t_xlim_lo, t_xlim_hi)
        isi_caption = f"spikes dans [{t_xlim_lo:g}, {t_xlim_hi:g}] s rel. trigger, intra-essai"
        isi_empty_hint = (
            f"moins de 2 spikes dans [{t_xlim_lo:g}, {t_xlim_hi:g}] s, par essai"
        )

    sec = f"{section_title} — " if section_title else ""
    for tri, st in enumerate(st_per_tr):
        st_plot = st
        if t_range_s is not None:
            st_plot = st[(st >= t_xlim_lo) & (st <= t_xlim_hi)]
        if st_plot.size:
            ax_raster.scatter(
                st_plot,
                np.full(st_plot.shape, tri),
                s=4,
                c="k",
                alpha=0.75,
                linewidths=0,
            )
    ax_raster.set_ylabel("Essai n°")
    cap = (
        f"passage sous {spike_threshold_uv:g} µV (front descendant)"
        if spike_threshold_uv < 0
        else f"passage au-dessus de {spike_threshold_uv:g} µV (front montant)"
    )
    ax_raster.set_title(f"{sec}Raster — {short} ({cap}, réfractaire 1 ms)")
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
        ax_fr.plot(tc, rate_hz, linewidth=1.3, color="darkred", label="PSTH lissé (Hz)")
    ax_fr.set_ylabel("Taux (Hz)")
    ax_fr.set_title(
        f"{sec}Taux de décharge moyen — {short} (PSTH gaussien, σ={firing_rate_window_s:g} s)"
    )
    ax_fr.grid(True, alpha=0.3)
    ax_fr.legend(loc="best", fontsize=6)
    ax_fr.set_xlim(t_xlim_lo, t_xlim_hi)
    ax_raster.set_xlabel(TIME_REL_XLABEL)
    ax_fr.set_xlabel(TIME_REL_XLABEL)
<<<<<<< Updated upstream
=======
    fr_trials = _trial_mean_firing_rate_hz(st_per_tr, (t_xlim_lo, t_xlim_hi))
    x_trials = np.arange(1, len(fr_trials) + 1)
    if fr_trials.size:
        ax_trial_fr.plot(x_trials, fr_trials, color="teal", linewidth=1.1, marker="o", markersize=2.6)
    ax_trial_fr.set_title(f"{sec}Mean firing rate per trial — shown window")
    ax_trial_fr.set_xlabel("Trial index")
    ax_trial_fr.set_ylabel("FR (Hz)")
    ax_trial_fr.grid(True, alpha=0.25)
>>>>>>> Stashed changes

    t_isi, isi_vals_s = _isi_time_and_values_s(st_per_tr, isi_window_s=isi_window)
    if t_isi.size:
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
            f"{sec}ISI — {short} ({isi_caption} ; abscisse = temps du 2e spike de la paire)"
        )
        ax_isi.grid(True, alpha=0.25)
        ax_isi.set_xlim(ISI_ABSCISSA_T0_S, ISI_ABSCISSA_T1_S)
        ax_isi.set_xlabel(TIME_REL_XLABEL)
    else:
        ax_isi.text(
            0.5,
            0.5,
            "Pas assez de spikes pour l'ISI\n(" + isi_empty_hint + ")",
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
) -> None:
    """Raster / PSTH / ISI superposés pour deux enregistrements (même canal, même axe temps)."""
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
        isi_caption = f"±{ISI_HALF_WINDOW_S:g} s du trigger, intra-essai"
        isi_empty_hint = f"±{ISI_HALF_WINDOW_S:g} s du trigger"
    else:
        t_xlim_lo, t_xlim_hi = float(t_range_s[0]), float(t_range_s[1])
        psth_t_range = (t_xlim_lo, t_xlim_hi)
        isi_window = (t_xlim_lo, t_xlim_hi)
        isi_caption = f"[{t_xlim_lo:g}, {t_xlim_hi:g}] s rel. trigger, intra-essai"
        isi_empty_hint = f"[{t_xlim_lo:g}, {t_xlim_hi:g}] s du trigger"

    sec = f"{section_title} — " if section_title else ""
    for tri, st in enumerate(sta):
        st_plot = st
        if t_range_s is not None:
            st_plot = st[(st >= t_xlim_lo) & (st <= t_xlim_hi)]
        if st_plot.size:
            ax_raster.scatter(
                st_plot,
                np.full(st_plot.shape, tri),
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
            ax_raster.scatter(
                st_plot,
                np.full(st_plot.shape, offset + tri),
                s=4,
                c="C1",
                alpha=0.75,
                linewidths=0,
                label=label_b if tri == 0 else "",
            )
    ax_raster.set_ylabel("Essai n° (A puis B empilés)")
    cap = (
        f"seuil {spike_threshold_uv:g} µV (desc.)"
        if spike_threshold_uv < 0
        else f"seuil {spike_threshold_uv:g} µV (montant)"
    )
    ax_raster.set_title(f"{sec}Raster — {short} ({cap})")
    ax_raster.grid(True, alpha=0.25, axis="x")
    ax_raster.set_ylim(-0.5, max(offset + n_b - 0.5, 0.5))
    ax_raster.set_xlim(t_xlim_lo, t_xlim_hi)
    ax_raster.axhline(offset - 0.5, color="0.5", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_raster.legend(loc="upper right", fontsize=6)

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
    ax_fr.set_ylabel("Taux (Hz)")
    ax_fr.set_title(
        f"{sec}Taux de décharge (PSTH gaussien σ={firing_rate_window_s:g} s) — {short}"
    )
    ax_fr.grid(True, alpha=0.3)
    ax_fr.legend(loc="best", fontsize=6)
    ax_fr.set_xlim(t_xlim_lo, t_xlim_hi)
    ax_raster.set_xlabel(TIME_REL_XLABEL)
    ax_fr.set_xlabel(TIME_REL_XLABEL)
<<<<<<< Updated upstream
=======
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
    ax_trial_fr.set_ylabel("FR (Hz)")
    ax_trial_fr.grid(True, alpha=0.25)
>>>>>>> Stashed changes

    t_a, isi_a_s = _isi_time_and_values_s(sta, isi_window_s=isi_window)
    t_b, isi_b_s = _isi_time_and_values_s(stb, isi_window_s=isi_window)
    if t_a.size or t_b.size:
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
            f"{sec}ISI — {short} ({isi_caption} ; abscisse = temps du 2e spike de la paire)"
        )
        ax_isi.grid(True, alpha=0.25)
        ax_isi.legend(loc="best", fontsize=6)
        ax_isi.set_xlim(ISI_ABSCISSA_T0_S, ISI_ABSCISSA_T1_S)
        ax_isi.set_xlabel(TIME_REL_XLABEL)
    else:
        ax_isi.text(
            0.5,
            0.5,
            f"Pas assez de spikes pour l'ISI\n({isi_empty_hint})",
            ha="center",
            va="center",
            transform=ax_isi.transAxes,
        )
        ax_isi.set_axis_off()


<<<<<<< Updated upstream
=======
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
    """Raster / PSTH / ISI superposés pour N enregistrements."""
    short, _ = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    if spikes_per_recording is None:
        spikes_per_recording = [
            _spike_times_per_trial(w, t_rel, fs, spike_threshold_uv) for w in windows_list
        ]

    if t_range_s is None:
        t_xlim_lo, t_xlim_hi = float(t_rel[0]), float(t_rel[-1])
        psth_t_range = None
        isi_window = None
        isi_caption = f"±{ISI_HALF_WINDOW_S:g} s du trigger, intra-essai"
        isi_empty_hint = f"±{ISI_HALF_WINDOW_S:g} s du trigger"
    else:
        t_xlim_lo, t_xlim_hi = float(t_range_s[0]), float(t_range_s[1])
        psth_t_range = (t_xlim_lo, t_xlim_hi)
        isi_window = (t_xlim_lo, t_xlim_hi)
        isi_caption = f"[{t_xlim_lo:g}, {t_xlim_hi:g}] s rel. trigger, intra-essai"
        isi_empty_hint = f"[{t_xlim_lo:g}, {t_xlim_hi:g}] s du trigger"

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
    ax_raster.set_ylabel("Essai n° (groupés par fichier)")
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
    ax_fr.set_ylabel("Taux (Hz)")
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
    ax_trial_fr.set_ylabel("FR (Hz)")
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
        ax_isi.set_title(f"{sec}ISI — {short} ({isi_caption} ; abscisse = temps du 2e spike)")
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
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
    lightweight_mode: bool = False,
    sampling_percent: int = 100,
    streaming_mode: bool = False,
    pre_n_common: Optional[int] = None,
    post_n_common: Optional[int] = None,
) -> Path:
    """PDF multi-pages : superposition de N enregistrements (mêmes canaux alignés)."""
    n_records = len(labels)
    if n_records < 2:
        raise ValueError("plot_channel_multi_comparison attend au moins 2 enregistrements alignés.")
    if not streaming_mode and (len(means) < 2 or len(means) != n_records):
        raise ValueError("Mode non streaming: `means` doit contenir N enregistrements.")
    if streaming_mode and (spike_sources is None or len(spike_sources) != n_records):
        raise ValueError("Mode streaming: `spike_sources` doit contenir N enregistrements.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if pdf_title is not None and pdf_title.strip():
        safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in pdf_title.strip())
        pdf_stem = safe_title.removesuffix(".pdf")
    else:
        pdf_stem = "multi_comparison"
    pdf_name = _shorten_filename_for_windows(output_dir, f"{pdf_stem}.pdf")
    pdf_path = output_dir / pdf_name

    zoom_t0, zoom_t1 = ZOOM_T0, ZOOM_T1
    filt_note = f" — Butterworth passe-bas {lowpass_cutoff_hz:g} Hz" if lowpass_cutoff_hz is not None else ""
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
                    raise ValueError("Mode streaming: pre_n_common/post_n_common requis.")
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

            fig = plt.figure(figsize=(12, 38))
            hr = [0.06, 1.05, 1.10, 0.90, 0.85, 0.95, 0.06, 1.05, 1.10, 0.90, 0.85, 0.95, 0.06, 1.05, 1.10, 0.90, 0.85, 0.95]
            gs = fig.add_gridspec(18, 1, height_ratios=hr, hspace=0.9)
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
            ax_full.axvline(0.0, linestyle="--", linewidth=1.0, color="red", label="Trigger (début)")
            for v in end_markers:
                ax_full.axvline(v, linestyle=":", linewidth=0.9, color="0.45")
                ax_zoom.axvline(v, linestyle=":", linewidth=0.9, color="0.45")
            ax_full.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zone zoom")
            if end_markers:
                end_zoom_t0 = float(min(end_markers) + zoom_t0)
                end_zoom_t1 = float(max(end_markers) + zoom_t1)
                ax_full.axvspan(end_zoom_t0, end_zoom_t1, alpha=0.10, color="gold", label="Trigger-end zoom area")
            spike_note = f" — + raster / rate / ISI ({spike_cmp_pipe})" if _has_spike_cmp else ""
            ax_full.set_title(f"Multi-comparison — {channel_names[ch]} (full view){filt_note}{both_note}{spike_note}")
            ax_full.set_ylabel("Amplitude (µV)")
            ax_full.set_xlabel(TIME_REL_XLABEL)
            ax_full.grid(True, alpha=0.3)
            ax_full.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=4, fontsize=6)

            ax_zoom.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
            ax_zoom.set_xlim(zoom_t0, zoom_t1)
            ax_zoom.set_title(zoom_title)
            ax_zoom.set_ylabel("Amplitude (µV)")
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
                ax_zoom_end.set_ylabel("Amplitude (µV)")
                ax_zoom_end.set_xlabel(TIME_REL_XLABEL)
                ax_zoom_end.grid(True, alpha=0.3)
            else:
                ax_zoom_end.text(0.5, 0.5, "Trigger-end zoom unavailable\n(no rising edge after trigger)", ha="center", va="center", transform=ax_zoom_end.transAxes)
                ax_zoom_end.set_axis_off()

            if _has_spike_cmp and spike_sources is not None:
                sources_ok = [src for src in spike_sources if src is not None]
                # Détection des spikes calculée une seule fois par canal/fichier,
                # puis réutilisée pour full / zoom / zoom fin.
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


>>>>>>> Stashed changes
def plot_channel_averages(
    t_rel: np.ndarray,
    mean_per_channel: np.ndarray,
    channel_names: Sequence[str],
    output_dir: Path,
    rhs_file: Path,
    lowpass_cutoff_hz: Optional[float] = None,
    trigger_end_rising_rel_s: Optional[float] = None,
    windows: Optional[np.ndarray] = None,
    spike_source: Optional[AmplifierSpikeSource] = None,
    fs: Optional[float] = None,
    spike_threshold_uv: float = -40.0,
    firing_rate_window_s: float = 0.025,
    mean_per_channel_raw: Optional[np.ndarray] = None,
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
) -> Path:
    """Un seul PDF multi-pages (une page par canal), sans fenêtre graphique.

    Les valeurs amplificateur Intan sont affichées en microvolts (µV), comme
    fournies par load_intan_rhs_format (amplifier_data).
    """
    n_channels = mean_per_channel.shape[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{rhs_file.stem}.pdf"

    zoom_t0, zoom_t1 = ZOOM_T0, ZOOM_T1
    filt_note = (
        f" — Butterworth passe-bas {lowpass_cutoff_hz:g} Hz"
        if lowpass_cutoff_hz is not None
        else ""
    )
    both_note = (
        " — moyennes brute + filtrée superposées"
        if lowpass_cutoff_hz is not None
        else ""
    )
    zoom_title = (
        f"Zoom : {zoom_t0:.1f} à {zoom_t1:.1f} s (relatif au trigger){filt_note}{both_note}"
    )

    def _spike_threshold_caption(thr: float) -> str:
        if thr >= 0:
            return f"seuil {thr:g} µV (front montant)"
        return f"seuil {thr:g} µV (front descendant, pic négatif)"

    _has_spike_data = fs is not None and (
        spike_source is not None or (windows is not None and getattr(windows, "ndim", 0) == 3)
    )
    _, spike_pipe_detail = _spike_pipeline_captions(spike_bandpass_low_hz, spike_bandpass_high_hz)
    spike_note = (
        f" — spikes ({spike_pipe_detail}): {_spike_threshold_caption(spike_threshold_uv)}, "
        f"lissage FR σ={firing_rate_window_s:g} s"
        if _has_spike_data
        else ""
    )

    with PdfPages(pdf_path) as pdf:
        for ch in range(n_channels):
            check_analysis_cancelled()
            y = mean_per_channel[ch]
<<<<<<< Updated upstream
            fig = plt.figure(figsize=(12, 26))
            hr = [0.06, 1.05, 1.15, 0.95, 0.95, 0.06, 1.05, 1.15, 0.95, 0.95]
            gs = fig.add_gridspec(10, 1, height_ratios=hr, hspace=0.65)
=======
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
>>>>>>> Stashed changes
            ax_hdr1 = fig.add_subplot(gs[0, 0])
            ax_hdr1.axis("off")
            ax_hdr1.text(
                0.02,
                0.5,
                "Partie 1 — Vue complète (toute la fenêtre pré/post-trigger)",
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
                f"Partie 2 — Vue zoomée [{zoom_t0:.2f}, {zoom_t1:.2f}] s (relatif au trigger)",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                transform=ax_hdr2.transAxes,
            )
<<<<<<< Updated upstream
            ax_zoom = fig.add_subplot(gs[6, 0])
            ax_raster_z = fig.add_subplot(gs[7, 0], sharex=ax_zoom)
            ax_fr_z = fig.add_subplot(gs[8, 0], sharex=ax_zoom)
            ax_isi_z = fig.add_subplot(gs[9, 0])
=======
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
>>>>>>> Stashed changes

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
                    label="Moyenne (non filtrée)",
                    zorder=1,
                )
                ax_full.plot(
                    t_rel,
                    y,
                    linewidth=1.35,
                    color="C0",
                    label=f"Moyenne filtrée ({lowpass_cutoff_hz:g} Hz)",
                    zorder=2,
                )
            else:
                ax_full.plot(t_rel, y, linewidth=1.2, color="C0", label="Moyenne")
            ax_full.axvline(0.0, linestyle="--", linewidth=1.0, color="red", label="Trigger (début)")
            if trigger_end_rising_rel_s is not None:
                ax_full.axvline(
                    trigger_end_rising_rel_s,
                    linestyle=":",
                    linewidth=1.0,
                    color="darkorange",
                    label="Fin trigger (↗)",
                )
            ax_full.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zone zoom")
            ax_full.set_title(
                f"Moyenne trigger — {channel_names[ch]} (vue complète){filt_note}{both_note}{spike_note}"
            )
            ax_full.set_ylabel("Amplitude (µV)")
            ax_full.set_xlabel(TIME_REL_XLABEL)
            ax_full.grid(True, alpha=0.3)
            ax_full.legend(
                loc="upper center",
<<<<<<< Updated upstream
                bbox_to_anchor=(0.5, -0.2),
=======
                bbox_to_anchor=(0.5, -0.36),
>>>>>>> Stashed changes
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
                    label="Non filtrée",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    y[zmask],
                    linewidth=1.45,
                    color="C0",
                    label=f"Filtrée ({lowpass_cutoff_hz:g} Hz)",
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
            ax_zoom.set_ylabel("Amplitude (µV)")
            ax_zoom.set_xlabel(TIME_REL_XLABEL)
            ax_zoom.grid(True, alpha=0.3)
            if show_both:
                ax_zoom.legend(loc="best", fontsize=6)

            if _has_spike_data and (
                spike_source is not None
                or (windows is not None and windows.ndim == 3)
            ):
                if spike_source is not None:
                    w_ch = spike_source.windows_2d_for_channel(ch)
                else:
                    w_ch = np.asarray(windows[:, ch, :])
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
                )
<<<<<<< Updated upstream
            elif not _has_spike_data:
                for ax in (ax_raster_f, ax_fr_f, ax_raster_z, ax_fr_z):
=======
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
                        lightweight_mode=lightweight_mode,
                        sampling_percent=sampling_percent,
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
>>>>>>> Stashed changes
                    ax.text(
                        0.5,
                        0.5,
                        "Données brutes par essai indisponibles",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                for ax in (ax_isi_f, ax_isi_z):
                    ax.text(
                        0.5,
                        0.5,
                        "ISI indisponible",
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
<<<<<<< Updated upstream
=======
                ax_raster_ze,
                ax_fr_ze,
                ax_trial_fr_ze,
                ax_isi_ze,
                ax_zoom_end,
>>>>>>> Stashed changes
            ):
                ax.tick_params(axis="x", labelbottom=True)

            fig.tight_layout()
            _shift_axes_down(
<<<<<<< Updated upstream
                [ax_raster_f, ax_fr_f, ax_isi_f, ax_hdr2, ax_zoom, ax_raster_z, ax_fr_z, ax_isi_z],
=======
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
>>>>>>> Stashed changes
                delta=0.015,
            )
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2, dpi=120)
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
    spike_bandpass_low_hz: Optional[float] = None,
    spike_bandpass_high_hz: Optional[float] = None,
) -> Path:
    """PDF multi-pages : par canal, moyennes + zoom + raster / PSTH / ISI (deux enregistrements superposés)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_a = "".join(c if c.isalnum() or c in "._-" else "_" for c in label_a)[:80]
    safe_b = "".join(c if c.isalnum() or c in "._-" else "_" for c in label_b)[:80]
    pdf_path = output_dir / f"{safe_a}_vs_{safe_b}.pdf"

    zoom_t0, zoom_t1 = ZOOM_T0, ZOOM_T1
    filt_note = (
        f" — Butterworth passe-bas {lowpass_cutoff_hz:g} Hz"
        if lowpass_cutoff_hz is not None
        else ""
    )
    both_note = (
        " — moyennes brute + filtrée superposées"
        if lowpass_cutoff_hz is not None
        else ""
    )
    zoom_title = f"Zoom : {zoom_t0:.1f} à {zoom_t1:.1f} s (relatif au trigger){filt_note}{both_note}"
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
<<<<<<< Updated upstream
            fig = plt.figure(figsize=(12, 26))
            hr = [0.06, 1.05, 1.15, 0.95, 0.95, 0.06, 1.05, 1.15, 0.95, 0.95]
            gs = fig.add_gridspec(10, 1, height_ratios=hr, hspace=0.65)
=======
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
>>>>>>> Stashed changes
            ax_hdr1 = fig.add_subplot(gs[0, 0])
            ax_hdr1.axis("off")
            ax_hdr1.text(
                0.02,
                0.5,
                "Partie 1 — Vue complète (toute la fenêtre pré/post-trigger)",
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
                f"Partie 2 — Vue zoomée [{zoom_t0:.2f}, {zoom_t1:.2f}] s (relatif au trigger)",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                transform=ax_hdr2.transAxes,
            )
<<<<<<< Updated upstream
            ax_zoom = fig.add_subplot(gs[6, 0])
            ax_raster_z = fig.add_subplot(gs[7, 0], sharex=ax_zoom)
            ax_fr_z = fig.add_subplot(gs[8, 0], sharex=ax_zoom)
            ax_isi_z = fig.add_subplot(gs[9, 0])
=======
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
>>>>>>> Stashed changes
            if show_both:
                ax_full.plot(
                    t_rel,
                    ya_raw,
                    linewidth=1.0,
                    color="C0",
                    alpha=0.4,
                    label=f"{label_a} (non filtré)",
                )
                ax_full.plot(
                    t_rel,
                    yb_raw,
                    linewidth=1.0,
                    color="C1",
                    alpha=0.4,
                    label=f"{label_b} (non filtré)",
                )
                ax_full.plot(
                    t_rel,
                    ya,
                    linewidth=1.35,
                    color="C0",
                    label=f"{label_a} (filtré {lowpass_cutoff_hz:g} Hz)",
                )
                ax_full.plot(
                    t_rel,
                    yb,
                    linewidth=1.35,
                    color="C1",
                    label=f"{label_b} (filtré {lowpass_cutoff_hz:g} Hz)",
                )
            else:
                ax_full.plot(t_rel, ya, linewidth=1.2, color="C0", label=label_a)
                ax_full.plot(t_rel, yb, linewidth=1.2, color="C1", label=label_b)
            ax_full.axvline(0.0, linestyle="--", linewidth=1.0, color="red", label="Trigger (début)")
            if trigger_end_rising_rel_s_a is not None:
                ax_full.axvline(
                    trigger_end_rising_rel_s_a,
                    linestyle=":",
                    linewidth=1.0,
                    color="darkorange",
                    label=f"Fin (↗) {label_a}",
                )
            if trigger_end_rising_rel_s_b is not None:
                ax_full.axvline(
                    trigger_end_rising_rel_s_b,
                    linestyle=":",
                    linewidth=1.0,
                    color="purple",
                    label=f"Fin (↗) {label_b}",
                )
            ax_full.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zone zoom")
            spike_cmp_note = (
                f" — + raster / taux / ISI ({spike_cmp_pipe})"
                if _has_spike_cmp
                else ""
            )
            ax_full.set_title(
                f"Comparaison — {channel_names[ch]} (vue complète){filt_note}{both_note}{spike_cmp_note}"
            )
            ax_full.set_ylabel("Amplitude (µV)")
            ax_full.set_xlabel(TIME_REL_XLABEL)
            ax_full.grid(True, alpha=0.3)
            ax_full.legend(
                loc="upper center",
<<<<<<< Updated upstream
                bbox_to_anchor=(0.5, -0.2),
=======
                bbox_to_anchor=(0.5, -0.36),
>>>>>>> Stashed changes
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
                    label=f"{label_a} brut",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    yb_raw[zmask],
                    linewidth=1.1,
                    color="C1",
                    alpha=0.45,
                    label=f"{label_b} brut",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    ya[zmask],
                    linewidth=1.45,
                    color="C0",
                    label=f"{label_a} filtré",
                )
                ax_zoom.plot(
                    t_rel[zmask],
                    yb[zmask],
                    linewidth=1.45,
                    color="C1",
                    label=f"{label_b} filtré",
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
            ax_zoom.set_ylabel("Amplitude (µV)")
            ax_zoom.set_xlabel(TIME_REL_XLABEL)
            ax_zoom.grid(True, alpha=0.3)
            ax_zoom.legend(loc="best", fontsize=6)

            if _has_spike_cmp:
                w_a = spike_source_a.windows_2d_for_channel(ch)
                w_b = spike_source_b.windows_2d_for_channel(ch)
                sta = _spike_times_per_trial(w_a, t_rel, float(fs), spike_threshold_uv)
                stb = _spike_times_per_trial(w_b, t_rel, float(fs), spike_threshold_uv)
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
                )
<<<<<<< Updated upstream
            else:
                for ax in (ax_raster_f, ax_fr_f, ax_raster_z, ax_fr_z):
=======
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
                        lightweight_mode=lightweight_mode,
                        sampling_percent=sampling_percent,
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
>>>>>>> Stashed changes
                    ax.text(
                        0.5,
                        0.5,
                        "Raster / PSTH / ISI indisponibles\n(sources mmap manquantes)",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                    )
                    ax.set_axis_off()
                for ax in (ax_isi_f, ax_isi_z):
                    ax.text(
                        0.5,
                        0.5,
                        "ISI indisponible",
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
<<<<<<< Updated upstream
=======
                ax_raster_ze,
                ax_fr_ze,
                ax_trial_fr_ze,
                ax_isi_ze,
                ax_zoom_end,
>>>>>>> Stashed changes
            ):
                ax.tick_params(axis="x", labelbottom=True)

            fig.tight_layout()
            _shift_axes_down(
<<<<<<< Updated upstream
                [ax_raster_f, ax_fr_f, ax_isi_f, ax_hdr2, ax_zoom, ax_raster_z, ax_fr_z, ax_isi_z],
=======
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
>>>>>>> Stashed changes
                delta=0.015,
            )
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2, dpi=120)
            plt.close(fig)

    return pdf_path
