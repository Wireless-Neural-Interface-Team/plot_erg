"""Shared matplotlib draw primitives for screen (panel_registry) and PDF (plotting).

Screen and PDF keep their own orchestration; this module holds the common
spike / impedance / marker rendering used by both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from core import AmplifierSpikeSource, detect_spikes_at_threshold
from display_config import (
    ARTIFACT_LINE_COLOR,
    MUTED_AXIS_TEXT,
    STIM_OFFSET_COLOR,
    STIM_ONSET_COLOR,
    ZERO_LINE_COLOR,
)
from impedance_tracking import ImpedanceSession
from intan_rhx_dsp import IntanDspSettings, detect_spikes_intan
from pdf_layout import LayoutFonts, place_legend_below
from plot_utils import downsample_points

# ISI: only spikes within [-ISI_HALF_WINDOW_S, +ISI_HALF_WINDOW_S] (s relative to stimulation)
ISI_HALF_WINDOW_S = 1.0

# Superimposed spike waveforms around threshold crossing.
SPIKE_OVERLAY_MAX_TRACES = 3000
SPIKE_OVERLAY_DEFAULT_PRE_MS = 2.0
SPIKE_OVERLAY_DEFAULT_POST_MS = 4.0

TIME_REL_XLABEL = "Time relative to stimulation (s)"
# Defaults PDF (grand format). La GUI passe un DrawAppearance plus compact.
AXIS_LABEL_FONT_SIZE = 15
TICK_LABEL_FONT_SIZE = 15
ANNOTATION_FONT_SIZE = 13
UNAVAILABLE_FONT_SIZE = 15
LEGEND_FONT_SIZE = 15
DEFAULT_STIM_LINEWIDTH = 1.15
DEFAULT_GRID_ALPHA = 0.3


@dataclass(frozen=True)
class DrawAppearance:
    """Typo / traits / grille — PDF garde les défauts module ; GUI injecte PanelStyle."""

    axis_label_font_size: float = AXIS_LABEL_FONT_SIZE
    tick_font_size: float = TICK_LABEL_FONT_SIZE
    annotation_font_size: float = ANNOTATION_FONT_SIZE
    unavailable_font_size: float = UNAVAILABLE_FONT_SIZE
    legend_font_size: float = LEGEND_FONT_SIZE
    line_width: float = 1.2
    stim_linewidth: float = DEFAULT_STIM_LINEWIDTH
    grid: bool = True
    grid_alpha: float = DEFAULT_GRID_ALPHA
    stim_onset_color: str = STIM_ONSET_COLOR
    stim_offset_color: str = STIM_OFFSET_COLOR

    @classmethod
    def for_pdf(cls) -> DrawAppearance:
        return cls()

    @classmethod
    def from_panel_style(
        cls,
        style: Any,
        *,
        legend_font_size: float | None = None,
    ) -> DrawAppearance:
        """Construire depuis ``view_config.PanelStyle`` (+ taille légende optionnelle)."""
        lw = float(getattr(style, "line_width", 1.2))
        return cls(
            axis_label_font_size=float(getattr(style, "label_font_size", 9.0)),
            tick_font_size=float(getattr(style, "tick_font_size", 8.0)),
            annotation_font_size=float(getattr(style, "tick_font_size", 8.0)),
            unavailable_font_size=float(getattr(style, "label_font_size", 9.0)),
            legend_font_size=float(
                legend_font_size
                if legend_font_size is not None
                else getattr(style, "label_font_size", 9.0)
            ),
            line_width=lw,
            stim_linewidth=max(0.7, lw * 0.95),
            grid=bool(getattr(style, "grid", True)),
            grid_alpha=float(getattr(style, "grid_alpha", DEFAULT_GRID_ALPHA)),
        )

    def layout_fonts(self) -> LayoutFonts:
        return LayoutFonts(
            legend=self.legend_font_size,
            axis_title=self.axis_label_font_size + 1.0,
            axis_label=self.axis_label_font_size,
            tick=self.tick_font_size,
            section_header=self.axis_label_font_size + 5.0,
            mea_title=self.axis_label_font_size + 1.0,
            table=self.tick_font_size,
            unavailable=self.unavailable_font_size,
        )

    def apply_grid(self, ax: Any, *, axis: str | None = None) -> None:
        if self.grid:
            kwargs: dict[str, Any] = {"alpha": self.grid_alpha}
            if axis is not None:
                kwargs["axis"] = axis
            ax.grid(True, **kwargs)
        else:
            ax.grid(False)


def _legend_fonts(appearance: DrawAppearance | None = None) -> LayoutFonts:
    app = appearance or DrawAppearance.for_pdf()
    return app.layout_fonts()


def _draw_onset_offset_lines(
    ax: Any,
    *,
    end_markers: Sequence[float],
    end_line_specs: Optional[Sequence[tuple[float, str]]] = None,
    label_in_legend: bool = False,
    appearance: DrawAppearance | None = None,
) -> None:
    """Draw onset (t=0) and trigger-offset markers without zoom spans."""
    app = appearance or DrawAppearance.for_pdf()
    lw = float(app.stim_linewidth)
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=lw,
        color=app.stim_onset_color,
        label=("Stimulation (onset)" if label_in_legend else "_nolegend_"),
    )
    if end_line_specs:
        for idx, (value, label) in enumerate(end_line_specs):
            ax.axvline(
                value,
                linestyle="-.",
                linewidth=lw,
                color=app.stim_offset_color,
                label=(label if label_in_legend and idx == 0 else "_nolegend_"),
            )
        return
    labeled = False
    for value in end_markers:
        ax.axvline(
            float(value),
            linestyle="-.",
            linewidth=lw,
            color=app.stim_offset_color,
            label=(
                "Stimulation (offset)"
                if label_in_legend and not labeled
                else "_nolegend_"
            ),
        )
        labeled = True


def mark_stim_times(
    ax: Any,
    times: Sequence[float],
    *,
    appearance: DrawAppearance | None = None,
    alpha: float = 0.55,
    linewidth: float | None = None,
    linestyle: str = "--",
) -> None:
    """Marqueurs d’onset (liste de temps) — continuous / montage."""
    xs = [float(t) for t in times]
    if not xs:
        return
    app = appearance or DrawAppearance.for_pdf()
    lw = float(linewidth) if linewidth is not None else max(0.55, float(app.stim_linewidth) * 0.7)
    if len(xs) == 1:
        ax.axvline(
            xs[0],
            color=app.stim_onset_color,
            linestyle=linestyle,
            linewidth=lw,
            alpha=alpha,
        )
        return
    ax.vlines(
        xs,
        ymin=0.0,
        ymax=1.0,
        transform=ax.get_xaxis_transform(),
        colors=app.stim_onset_color,
        linestyles=linestyle,
        linewidths=lw,
        alpha=alpha,
        zorder=1,
    )


def _spike_times_per_trial(
    windows_ch: np.ndarray,
    t_rel: np.ndarray,
    fs: float,
    threshold: float,
    intan_dsp: IntanDspSettings | None = None,
) -> list[np.ndarray]:
    """For one channel: list of spike-time arrays (s rel. stimulation), one per trial."""
    from dataclasses import replace

    n_trials = int(windows_ch.shape[0])
    out: list[np.ndarray] = []
    for i in range(n_trials):
        if intan_dsp is not None:
            dsp = replace(intan_dsp, spike_threshold_uv=float(threshold))
            idx = detect_spikes_intan(
                windows_ch[i],
                dsp,
                start_sample=0,
                end_sample=int(windows_ch.shape[1]),
            )
        else:
            idx = detect_spikes_at_threshold(windows_ch[i], fs, threshold)
        out.append(np.asarray(t_rel[idx], dtype=np.float64))
    return out


def _psth_mean_hz(
    spike_times_per_trial: list[np.ndarray],
    t_rel: np.ndarray,
    n_trials: int,
    bin_width_s: float,
    t_range_s: Optional[Tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean sliding PSTH (Hz): spike count in moving window / (n_trials * window_width).

    The PSTH is evaluated at each sample step (derived from t_rel).
    t_range_s: if (t0, t1), process only this interval (out-of-range spikes excluded).
    """
    if t_range_s is not None:
        t0, t1 = float(t_range_s[0]), float(t_range_s[1])
    else:
        t0, t1 = float(t_rel[0]), float(t_rel[-1])
    if t1 <= t0 or bin_width_s <= 0 or t_rel.size < 2:
        return np.array([]), np.array([])
    dt = float(np.median(np.diff(t_rel)))
    if dt <= 0:
        return np.array([]), np.array([])
    # Evaluate PSTH at sampling cadence.
    edges = np.arange(t0, t1 + dt, dt)
    if edges.size < 2:
        return np.array([]), np.array([])
    if spike_times_per_trial:
        all_spikes = [
            np.asarray(st, dtype=np.float64)
            for st in spike_times_per_trial
            if getattr(st, "size", 0)
        ]
        if all_spikes:
            counts = np.histogram(np.concatenate(all_spikes), bins=edges)[0].astype(
                np.float64, copy=False
            )
        else:
            counts = np.zeros(edges.size - 1, dtype=np.float64)
    else:
        counts = np.zeros(edges.size - 1, dtype=np.float64)
    window_bins = max(1, int(round(float(bin_width_s) / dt)))
    kernel = np.ones(window_bins, dtype=np.float64)
    sliding_counts = np.convolve(counts, kernel, mode="same")
    effective_window_s = float(window_bins) * dt
    rate = sliding_counts / (max(n_trials, 1) * effective_window_s)
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers, rate


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
    intan_dsp: IntanDspSettings | None = None,
) -> Tuple[str, str]:
    """(short for subtitles, detailed for footer note) — spikes use high-pass."""
    if intan_dsp is None:
        short = "bessel high-pass order 2 @ 250 Hz"
        return short, (
            "bessel high-pass @ 250 Hz, order 2; "
            "RMS window 1 s; spike detection on high-pass signal"
        )
    short = intan_dsp.filter_short_label("highpass")
    detail = (
        f"{intan_dsp.filter_title_label('highpass')}, order {intan_dsp.hp_filter_order}; "
        f"RMS window {intan_dsp.rms_window_s:g} s; "
        f"spike thr. {intan_dsp.spike_threshold_uv:g} µV"
    )
    if intan_dsp.artifact_suppression_enabled:
        detail += f"; artifact {intan_dsp.artifact_threshold_uv:g} µV"
    return short, detail


def _isi_time_and_values_s(
    spike_times_per_trial: list[np.ndarray],
    *,
    isi_window_s: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interval end time (s rel. stimulation) and ISI (s) for each consecutive pair."""
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


def _add_raster_threshold_legend(
    ax_raster: Any,
    threshold_caption: str,
    threshold_entries: Sequence[tuple[str, str]] | None = None,
    *,
    appearance: DrawAppearance | None = None,
) -> None:
    """Place raster legend below the plot with threshold information."""
    app = appearance or DrawAppearance.for_pdf()
    if threshold_entries:
        handles, labels = ax_raster.get_legend_handles_labels()
        color_by_label: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            if not label or label == "_nolegend_" or label in color_by_label:
                continue
            color_val: Any = "0.25"
            try:
                face_colors = handle.get_facecolor()
                if face_colors is not None and len(face_colors) > 0:
                    color_val = face_colors[0]
            except Exception:
                pass
            color_by_label[label] = color_val
        unique_handles: list[Any] = []
        unique_labels: list[str] = []
        for rec_label, thr_text in threshold_entries:
            marker_color = color_by_label.get(rec_label, "0.25")
            unique_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markersize=5,
                    markerfacecolor=marker_color,
                    markeredgewidth=0.0,
                )
            )
            unique_labels.append(f"{rec_label}: {thr_text}")
    else:
        unique_handles = [
            Line2D([0], [0], color="0.25", linestyle="--", linewidth=app.line_width)
        ]
        unique_labels = [f"Threshold: {threshold_caption}"]
    legend = place_legend_below(
        ax_raster,
        _legend_fonts(app),
        ncol=1,
        handles=unique_handles,
        labels=unique_labels,
    )
    if legend is not None:
        legend.set_in_layout(False)


def _draw_spike_panels_multi_channel(
    ax_raster: Any,
    ax_fr: Any,
    ax_trial_fr: Any,
    ax_isi: Any,
    windows_list: Optional[Sequence[np.ndarray]],
    t_rel: np.ndarray,
    fs: float,
    spike_threshold_uv: float,
    psth_bin_window_s: float,
    labels: Sequence[str],
    *,
    intan_dsp: IntanDspSettings | None = None,
    t_range_s: Optional[Tuple[float, float]] = None,
    section_title: str = "",
    spikes_per_recording: Optional[list[list[np.ndarray]]] = None,
    sampling_percent: int = 100,
    threshold_caption: str | None = None,
    threshold_entries: Sequence[tuple[str, str]] | None = None,
    legend_visible: Sequence[bool] | None = None,
    show_raster: bool = True,
    show_psth: bool = True,
    show_trial_rate: bool = True,
    show_isi: bool = True,
    across_trials: bool = True,
    colors: Sequence[Any] | None = None,
    appearance: DrawAppearance | None = None,
) -> None:
    """Overlaid raster / PSTH / ISI for N recordings.

    ``across_trials=True``: spikes pooled/averaged over all stimulations.
    ``across_trials=False``: single stimulation (first or second); section_title
    should already name that stimulation.
    """
    app = appearance or DrawAppearance.for_pdf()
    short, _ = _spike_pipeline_captions(intan_dsp=intan_dsp)
    show_raster = bool(show_raster and ax_raster is not None)
    show_psth = bool(show_psth and ax_fr is not None)
    show_trial_rate = bool(show_trial_rate and ax_trial_fr is not None)
    show_isi = bool(show_isi and ax_isi is not None)
    if not (show_raster or show_psth or show_trial_rate or show_isi):
        return
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
        isi_caption = f"±{ISI_HALF_WINDOW_S:g} s of stimulation, within-trial"
        isi_empty_hint = f"±{ISI_HALF_WINDOW_S:g} s of stimulation"
    else:
        t_xlim_lo, t_xlim_hi = float(t_range_s[0]), float(t_range_s[1])
        psth_t_range = (t_xlim_lo, t_xlim_hi)
        isi_window = (t_xlim_lo, t_xlim_hi)
        isi_caption = f"[{t_xlim_lo:g}, {t_xlim_hi:g}] s rel. stimulation, within-trial"
        isi_empty_hint = f"[{t_xlim_lo:g}, {t_xlim_hi:g}] s of stimulation"

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    colors = list(colors) if colors is not None else list(default_colors)
    if not colors:
        colors = list(default_colors)
    n_rec = len(spikes_per_recording)
    dense_overlay = n_rec > 2
    base_lw = float(app.line_width)
    raster_alpha = 0.75 if not dense_overlay else 0.55
    psth_lw = (base_lw * 1.08) if not dense_overlay else (base_lw * 0.8)
    trial_lw = max(0.8, base_lw * 0.85)
    trial_marker = "o" if n_rec <= 3 else "None"
    trial_markersize = 2.2 if n_rec <= 3 else 0.0
    isi_alpha = 0.35 if not dense_overlay else 0.25
    sep_lw = max(0.6, base_lw * 0.65)
    sec = f"{section_title} — " if section_title else ""
    y_offset = 0
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        color = colors[rec_idx % len(colors)]
        show_leg = True if legend_visible is None else bool(legend_visible[rec_idx])
        if show_raster:
            xs: list[np.ndarray] = []
            ys: list[np.ndarray] = []
            for tri, st in enumerate(st_per_trial):
                st_plot = st
                if t_range_s is not None:
                    st_plot = st[(st >= t_xlim_lo) & (st <= t_xlim_hi)]
                if not st_plot.size:
                    continue
                xs.append(np.asarray(st_plot, dtype=np.float64))
                ys.append(np.full(st_plot.shape, y_offset + tri, dtype=np.float64))
            if xs:
                st_all = np.concatenate(xs)
                y_all = np.concatenate(ys)
                st_ds, y_ds = downsample_points(st_all, y_all, sampling_percent)
                ax_raster.scatter(
                    st_ds,
                    y_ds,
                    s=4,
                    c=color,
                    alpha=raster_alpha,
                    linewidths=0,
                    label=labels[rec_idx] if show_leg else "_nolegend_",
                    rasterized=True,
                )
        y_offset += len(st_per_trial)
        if rec_idx < len(spikes_per_recording) - 1 and show_raster:
            ax_raster.axhline(
                y_offset - 0.5, color="0.55", linestyle="--", linewidth=sep_lw, alpha=0.7
            )
    if threshold_caption is not None:
        cap = threshold_caption
    else:
        cap = (
            f"threshold {spike_threshold_uv:g} µV (falling)"
            if spike_threshold_uv < 0
            else f"threshold {spike_threshold_uv:g} µV (rising)"
        )
    if show_raster:
        ax_raster.set_ylabel("Trial # (grouped by file)")
        if across_trials:
            ax_raster.set_title(f"{sec}Raster — all stimulations — {short}")
        else:
            ax_raster.set_title(f"{sec}Raster — {short}")
        app.apply_grid(ax_raster, axis="x")
        ax_raster.set_ylim(-0.5, max(y_offset - 0.5, 0.5))
        ax_raster.set_xlim(t_xlim_lo, t_xlim_hi)
        ax_raster.set_xlabel(TIME_REL_XLABEL)
        _add_raster_threshold_legend(
            ax_raster,
            cap,
            threshold_entries=threshold_entries,
            appearance=app,
        )

    bin_w = max(float(psth_bin_window_s), 1.0 / fs)
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        show_leg = True if legend_visible is None else bool(legend_visible[rec_idx])
        tc, rate = _psth_mean_hz(
            st_per_trial,
            t_rel,
            max(len(st_per_trial), 1),
            bin_w,
            t_range_s=psth_t_range,
        )
        if tc.size and show_psth:
            ax_fr.plot(
                tc,
                rate,
                linewidth=psth_lw,
                color=colors[rec_idx % len(colors)],
                label=labels[rec_idx] if show_leg else "_nolegend_",
            )
    if show_psth:
        if across_trials:
            ax_fr.set_ylabel("Trial-averaged rate (Hz)")
            ax_fr.set_title(
                f"{sec}PSTH — trial-averaged firing rate "
                f"(window = {bin_w:g} s) — {short}"
            )
        else:
            ax_fr.set_ylabel("Firing rate (Hz)")
            ax_fr.set_title(
                f"{sec}PSTH — firing rate "
                f"(window = {bin_w:g} s) — {short}"
            )
        app.apply_grid(ax_fr)
        ax_fr.set_xlim(t_xlim_lo, t_xlim_hi)
        ax_fr.set_xlabel(TIME_REL_XLABEL)
    max_trials = 0
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        fr_trials = _trial_mean_firing_rate_hz(st_per_trial, (t_xlim_lo, t_xlim_hi))
        max_trials = max(max_trials, len(fr_trials))
        x = np.arange(1, len(fr_trials) + 1)
        show_leg = True if legend_visible is None else bool(legend_visible[rec_idx])
        if fr_trials.size and show_trial_rate:
            ax_trial_fr.plot(
                x,
                fr_trials,
                color=colors[rec_idx % len(colors)],
                linewidth=trial_lw,
                marker=trial_marker,
                markersize=trial_markersize,
                label=labels[rec_idx] if show_leg else "_nolegend_",
            )
    if show_trial_rate:
        ax_trial_fr.set_title(
            f"{sec}Firing rate per trial — not averaged — displayed window"
        )
        ax_trial_fr.set_xlabel("Trial index")
        ax_trial_fr.set_ylabel("Rate per trial (Hz, not averaged)")
        app.apply_grid(ax_trial_fr)
        if max_trials > 0:
            ax_trial_fr.set_xlim(1, max_trials)

    has_isi = False
    isi_time_chunks: list[np.ndarray] = []
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        tx, isi_vals_s = _isi_time_and_values_s(st_per_trial, isi_window_s=isi_window)
        show_leg = True if legend_visible is None else bool(legend_visible[rec_idx])
        if tx.size and show_isi:
            has_isi = True
            tx, isi_vals_s = downsample_points(tx, isi_vals_s, sampling_percent)
            isi_time_chunks.append(np.asarray(tx, dtype=np.float64))
            ax_isi.scatter(
                tx,
                isi_vals_s * 1e3,
                s=10,
                c=colors[rec_idx % len(colors)],
                alpha=isi_alpha,
                linewidths=0,
                label=labels[rec_idx] if show_leg else "_nolegend_",
                rasterized=True,
            )
    if show_isi:
        if has_isi:
            ax_isi.set_ylabel("ISI (ms)")
            if across_trials:
                ax_isi.set_title(
                    f"{sec}ISI — all stimulations — {short} "
                    f"({isi_caption} ; x-axis = time of 2nd spike)"
                )
            else:
                ax_isi.set_title(
                    f"{sec}ISI — {short} "
                    f"({isi_caption} ; x-axis = time of 2nd spike)"
                )
            app.apply_grid(ax_isi)
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
                fontsize=app.unavailable_font_size,
                color=MUTED_AXIS_TEXT,
            )
            ax_isi.set_axis_off()


def _extract_spike_waveforms(
    source: AmplifierSpikeSource,
    ch: int,
    spike_times_per_trial: list[np.ndarray],
    *,
    pre_ms: float = SPIKE_OVERLAY_DEFAULT_PRE_MS,
    post_ms: float = SPIKE_OVERLAY_DEFAULT_POST_MS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HIGH snippets aligned on detection for the spike overlay panel.

    Buffer is ``[t - pre_ms, t + post_ms)``. Returns
    ``(t_ms, waveforms, t_rel_s)`` with t=0 at the threshold-crossing sample.
    """
    fs = float(source.fs)
    pre_n = max(0, int(math.ceil(max(0.0, float(pre_ms)) * fs / 1000.0)))
    post_n = max(1, int(math.ceil(max(0.0, float(post_ms)) * fs / 1000.0)))
    win_len = pre_n + post_n
    t_ms = (np.arange(-pre_n, post_n, dtype=np.float64) / fs) * 1e3
    empty = (
        t_ms,
        np.empty((0, win_len), dtype=np.float32),
        np.empty(0, dtype=np.float64),
    )
    row = np.asarray(source.high_trace_for_channel(ch), dtype=np.float32).ravel()
    n_samples = int(row.shape[0])
    triggers = np.asarray(source.valid_triggers, dtype=np.int64)
    if triggers.size == 0 or n_samples < win_len:
        return empty

    centers_list: list[np.ndarray] = []
    t_rel_list: list[np.ndarray] = []
    n_trials = min(len(spike_times_per_trial), int(triggers.size))
    for trial_i in range(n_trials):
        st_arr = np.asarray(spike_times_per_trial[trial_i], dtype=np.float64).ravel()
        if st_arr.size == 0:
            continue
        trig = int(triggers[trial_i])
        centers = trig + np.rint(st_arr * fs).astype(np.int64)
        valid = (centers - pre_n >= 0) & (centers + post_n <= n_samples)
        if not np.any(valid):
            continue
        centers_list.append(centers[valid])
        t_rel_list.append(st_arr[valid])
    if not centers_list:
        return empty

    centers_all = np.concatenate(centers_list)
    t_rel_keep = np.concatenate(t_rel_list)
    # Cap early to avoid allocating huge overlays.
    from processed_dataset import MAX_OVERLAY_SNIPPETS_PER_CHANNEL as _MAX_OV

    max_keep = int(_MAX_OV)
    if centers_all.size > max_keep:
        picks = np.linspace(0, centers_all.size - 1, max_keep).astype(np.int64)
        centers_all = centers_all[picks]
        t_rel_keep = t_rel_keep[picks]

    starts = centers_all - pre_n
    # Build a (n_spikes, win_len) view via advanced indexing.
    offsets = np.arange(win_len, dtype=np.int64)
    index = starts[:, None] + offsets[None, :]
    waves = np.asarray(row[index], dtype=np.float32)
    return t_ms, waves, np.asarray(t_rel_keep, dtype=np.float64)


def _spike_overlay_xticks(t_min_ms: float, t_max_ms: float) -> list[float]:
    """Nice tick marks for an arbitrary overlay time window (ms)."""
    t0 = float(t_min_ms)
    t1 = float(t_max_ms)
    span = t1 - t0
    if span <= 0:
        return [0.0]
    candidates = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
    target = span / 6.0
    step = min(candidates, key=lambda s: abs(float(s) - target))
    start = math.ceil(t0 / step - 1e-12) * step
    ticks = [float(x) for x in np.arange(start, t1 + 0.5 * step, step)]
    if not ticks or abs(ticks[0] - t0) > 1e-9:
        ticks.insert(0, t0)
    if abs(ticks[-1] - t1) > 1e-9:
        ticks.append(t1)
    # Drop near-duplicates from float noise
    cleaned: list[float] = []
    for t in ticks:
        if not cleaned or abs(t - cleaned[-1]) > step * 1e-6:
            cleaned.append(t)
    return cleaned


def _spike_overlay_ylim_uv(abs_peak: float) -> float:
    """Symmetric auto-scale half-range (µV) with a small margin."""
    peak = max(1.0, float(abs_peak))
    return peak * 1.05


def _spike_overlay_line_alpha(n_spikes: int) -> float:
    """Line opacity from spike count: 1.0 at ≤10 spikes, 0.35 at ≥1000."""
    n = max(0, int(n_spikes))
    n_lo, n_hi = 10.0, 1000.0
    a_lo, a_hi = 1.0, 0.35
    if n <= n_lo:
        return a_lo
    if n >= n_hi:
        return a_hi
    t = (n - n_lo) / (n_hi - n_lo)
    return float(a_lo + t * (a_hi - a_lo))


def _subsample_overlay_rows(
    waveforms: np.ndarray,
    sampling_percent: int,
    max_traces: int = SPIKE_OVERLAY_MAX_TRACES,
) -> np.ndarray:
    """Evenly keep a subset of spike snippets for display."""
    n = int(waveforms.shape[0])
    if n <= 1:
        return waveforms
    pct = max(1, min(100, int(sampling_percent)))
    keep = n if pct >= 100 else max(1, int(np.ceil(n * pct / 100.0)))
    keep = min(keep, int(max_traces), n)
    if keep >= n:
        return waveforms
    idx = np.linspace(0, n - 1, keep).astype(np.int64)
    return waveforms[idx]


def _draw_spike_overlay_panel(
    ax: Any,
    overlay_per_recording: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    labels: Sequence[str],
    *,
    t_range_s: Optional[Tuple[float, float]] = None,
    section_title: str = "",
    sampling_percent: int = 100,
    legend_visible: Sequence[bool] | None = None,
    intan_dsp: IntanDspSettings | None = None,
    pre_ms: float = SPIKE_OVERLAY_DEFAULT_PRE_MS,
    post_ms: float = SPIKE_OVERLAY_DEFAULT_POST_MS,
    thresholds_uv: Sequence[float] | None = None,
    colors: Sequence[Any] | None = None,
    appearance: DrawAppearance | None = None,
) -> None:
    """Overlay HIGH snippets aligned on threshold crossing."""
    if ax is None:
        return
    app = appearance or DrawAppearance.for_pdf()
    short, _ = _spike_pipeline_captions(intan_dsp=intan_dsp)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    colors = list(colors) if colors is not None else list(default_colors)
    if not colors:
        colors = list(default_colors)
    n_rec = len(overlay_per_recording)
    dense = n_rec > 2
    base_lw = float(app.line_width)
    line_width = (base_lw * 0.38) if not dense else (base_lw * 0.29)
    mean_lw = max(1.0, base_lw * 1.3)
    guide_lw = max(0.65, base_lw * 0.7)
    sec = f"{section_title} — " if section_title else ""
    t_min_ms = -max(0.0, float(pre_ms))
    t_max_ms = max(1e-6, float(post_ms))
    abs_peak = 0.0
    n_all_total = 0
    has_any = False
    for rec_idx, (t_ms, waves, t_rel_spk) in enumerate(overlay_per_recording):
        t_ms = np.asarray(t_ms, dtype=np.float64)
        waves = np.asarray(waves)
        t_rel_spk = np.asarray(t_rel_spk, dtype=np.float64)
        if t_range_s is not None and waves.shape[0] > 0 and t_rel_spk.size == waves.shape[0]:
            t0, t1 = float(t_range_s[0]), float(t_range_s[1])
            keep = (t_rel_spk >= t0) & (t_rel_spk <= t1)
            waves = waves[keep]
        n_all = int(waves.shape[0])
        n_all_total += n_all
        if n_all == 0 or t_ms.size == 0:
            continue
        time_mask = (t_ms >= t_min_ms - 1e-9) & (t_ms <= t_max_ms + 1e-9)
        if not np.any(time_mask):
            time_mask = np.ones(t_ms.size, dtype=bool)
        t_disp = t_ms[time_mask]
        waves_disp = waves[:, time_mask]
        has_any = True
        color = colors[rec_idx % len(colors)]
        shown = _subsample_overlay_rows(waves_disp, sampling_percent)
        n_shown = int(shown.shape[0])
        shown_f = np.asarray(shown, dtype=np.float64)
        # Mean over all detections (not just the displayed subsample).
        mean_w = np.asarray(waves_disp.mean(axis=0), dtype=np.float64)
        if shown_f.size:
            abs_peak = max(abs_peak, float(np.nanmax(np.abs(shown_f))))
        abs_peak = max(abs_peak, float(np.nanmax(np.abs(mean_w))))
        line_alpha = _spike_overlay_line_alpha(n_all)
        if n_shown > 0:
            segs = np.empty((n_shown, t_disp.size, 2), dtype=np.float64)
            segs[:, :, 0] = t_disp
            segs[:, :, 1] = shown_f
            ax.add_collection(
                LineCollection(
                    segs,
                    colors=(to_rgba(color, alpha=line_alpha),),
                    linewidths=line_width,
                    rasterized=True,
                    zorder=1,
                )
            )
        show_leg = True if legend_visible is None else bool(legend_visible[rec_idx])
        label = labels[rec_idx] if rec_idx < len(labels) else f"Recording {rec_idx + 1}"
        if n_shown < n_all:
            mean_label = f"{label} (mean, n={n_all}, {n_shown} shown)"
        else:
            mean_label = f"{label} (mean, n={n_all})"
        ax.plot(
            t_disp,
            mean_w,
            color=color,
            linewidth=mean_lw,
            zorder=3,
            label=mean_label if show_leg else "_nolegend_",
        )
        if thresholds_uv is not None and rec_idx < len(thresholds_uv):
            thr = float(thresholds_uv[rec_idx])
            thr_label = f"{label} threshold ({thr:g} µV)" if show_leg else "_nolegend_"
            if n_rec == 1:
                thr_label = f"Threshold ({thr:g} µV)" if show_leg else "_nolegend_"
            ax.axhline(
                thr,
                color=app.stim_onset_color,
                linestyle="--",
                linewidth=max(0.7, guide_lw),
                zorder=2,
                alpha=0.85,
                label=thr_label,
            )
            abs_peak = max(abs_peak, abs(thr))
    if not has_any:
        ax.text(
            0.5,
            0.5,
            "No spikes to overlay in this window",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=app.unavailable_font_size,
            color=MUTED_AXIS_TEXT,
        )
        ax.set_axis_off()
        return
    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=guide_lw, zorder=2)
    ax.axvline(0.0, color=ZERO_LINE_COLOR, linestyle=":", linewidth=guide_lw, zorder=2)
    if intan_dsp is not None and intan_dsp.artifact_suppression_enabled:
        art = float(intan_dsp.artifact_threshold_uv)
        ax.axhline(
            art, color=ARTIFACT_LINE_COLOR, linestyle=":", linewidth=guide_lw, zorder=2, alpha=0.7
        )
        ax.axhline(
            -art, color=ARTIFACT_LINE_COLOR, linestyle=":", linewidth=guide_lw, zorder=2, alpha=0.7
        )
    ax.set_xlabel("Time relative to detection (ms)")
    ax.set_ylabel("Potential (µV)")
    ax.set_title(
        f"{sec}Spike overlay — all detections [{t_min_ms:g}, {t_max_ms:g}] ms "
        f"(n={n_all_total}) — {short}"
    )
    ax.set_xlim(t_min_ms, t_max_ms)
    y_lim = _spike_overlay_ylim_uv(abs_peak)
    ax.set_ylim(-y_lim, y_lim)
    ticks = _spike_overlay_xticks(t_min_ms, t_max_ms)
    if ticks:
        ax.set_xticks(ticks)
    app.apply_grid(ax)


def _draw_impedance_evolution_panel(
    ax_imp: Any,
    channel_name: str,
    sessions: Sequence[ImpedanceSession],
    *,
    appearance: DrawAppearance | None = None,
) -> None:
    """Semi-log evolution of |Z| @ 1 kHz for one channel vs session timestamps."""
    app = appearance or DrawAppearance.for_pdf()
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
            fontsize=app.unavailable_font_size,
            color=MUTED_AXIS_TEXT,
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
            fontsize=app.annotation_font_size,
            alpha=0.9,
            zorder=4,
        )
    ax_imp.set_ylabel("|Z| @ 1 kHz (Ω)", fontsize=app.axis_label_font_size)
    ax_imp.set_xlabel("Session time (_YYMMDD_HHMMSS)", fontsize=app.axis_label_font_size)
    ax_imp.margins(x=0.08)
    date_locator = mdates.AutoDateLocator()
    ax_imp.xaxis.set_major_locator(date_locator)
    ax_imp.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
    ax_imp.tick_params(axis="both", labelsize=app.tick_font_size)
    app.apply_grid(ax_imp)
    for label in ax_imp.get_xticklabels():
        label.set_rotation(18)
        label.set_ha("right")

