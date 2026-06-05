"""PDF figures for triggered averaged traces, spike raster/PSTH/ISI, comparisons, and optional MEA layout inset."""

from __future__ import annotations

import functools
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from concurrent.futures import ThreadPoolExecutor

from core import (
    AmplifierSpikeSource,
    check_analysis_cancelled,
    detect_spikes_at_threshold,
    mean_triggered_windows_channelwise,
    resolve_channel_workers,
)
from intan_rhx_dsp import (
    IntanDspSettings,
    detect_spikes_intan,
    sliding_rms_intan_profile,
    sliding_rms_intan_profile_range,
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
LEGEND_FONT_SIZE = 8
AXIS_TITLE_FONT_SIZE = 9
AXIS_LABEL_FONT_SIZE = 8
TICK_LABEL_FONT_SIZE = 7
# Three-part PDF layout (inches). Panel height in the PDF is controlled by:
#   1. THREE_PART_PAGE_HEIGHT_*  — total page height
#   2. THREE_PART_PANEL_HEIGHT_SCALE — global multiplier on panel height
#   3. THREE_PART_GRID_HSPACE — lower = more height for plots, less for row gaps
# Default layout targets screen/PDF review (~38 in page height at 72 DPI).
# Set env PLOT_ERG_HIGH_QUALITY_PDF=1 to restore the older very tall export (128 in @ 120 DPI).
_HIGH_QUALITY_PDF = os.environ.get("PLOT_ERG_HIGH_QUALITY_PDF", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
THREE_PART_PAGE_WIDTH_IN = 12.0
if _HIGH_QUALITY_PDF:
    THREE_PART_PAGE_HEIGHT_NO_IMP = (128.0, 2.2)
    THREE_PART_PAGE_HEIGHT_IMP = (138.0, 2.7)
    PDF_DPI = 120
else:
    THREE_PART_PAGE_HEIGHT_NO_IMP = (100.0, 0.9)
    THREE_PART_PAGE_HEIGHT_IMP = (110.0, 1.0)
    PDF_DPI = 72
THREE_PART_PANEL_HEIGHT_SCALE = 1.0  # e.g. 1.25 for 25% taller panels (same width)
THREE_PART_PAGE_HEIGHT_REF = 120.0
THREE_PART_GRID_HSPACE = 0.52
# Leave room for y-axis tick labels + vertical ylabels (e.g. "Trial # (grouped by file)").
THREE_PART_SUBPLOT_LEFT = 0.10
THREE_PART_SUBPLOT_RIGHT = 0.99
THREE_PART_SUBPLOT_BOTTOM = 0.01
THREE_PART_SUBPLOT_TOP = 0.99
SUMMARY_PAGE_WIDTH_IN = 16.0
SUMMARY_PAGE_HEIGHT_IN = 9.0
TRACE_PANEL_LEGEND_KWARGS = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, -0.15),
    "fontsize": LEGEND_FONT_SIZE,
    "framealpha": None,
}
THREE_PART_ROW_HEIGHTS = [
    # Part 1 — full view (9 panels)
    1.70,
    1.60,
    1.35,
    1.25,
    1.30,
    1.20,
    1.45,
    1.05,
    1.15,
    0.08,  # Part 2 title band (gs[10])
    # Part 2 — zoom (9 panels)
    1.70,
    1.60,
    1.35,
    1.25,
    1.30,
    1.20,
    1.45,
    1.05,
    1.15,
    0.08,  # Part 3 title band (gs[20])
    # Part 3 — trigger-end zoom (9 panels)
    1.70,
    1.60,
    1.35,
    1.25,
    1.30,
    1.20,
    1.45,
    1.05,
    1.15,
]
THREE_PART1_PANEL_KEYS = [
    "ax_full",
    "ax_full_filt",
    "ax_first_trigger",
    "ax_first_trigger_hp",
    "ax_full_rms",
    "ax_raster_f",
    "ax_fr_f",
    "ax_trial_fr_f",
    "ax_isi_f",
]
THREE_PART2_PANEL_KEYS = [
    "ax_hdr2",
    "ax_zoom",
    "ax_zoom_filt",
    "ax_zoom_first",
    "ax_zoom_first_hp",
    "ax_zoom_rms",
    "ax_raster_z",
    "ax_fr_z",
    "ax_trial_fr_z",
    "ax_isi_z",
]
THREE_PART2_CURVE_KEYS = [k for k in THREE_PART2_PANEL_KEYS if not k.startswith("ax_hdr")]
THREE_PART3_PANEL_KEYS = [
    "ax_hdr3",
    "ax_zoom_end",
    "ax_zoom_end_filt",
    "ax_zoom_end_first",
    "ax_zoom_end_first_hp",
    "ax_zoom_end_rms",
    "ax_raster_ze",
    "ax_fr_ze",
    "ax_trial_fr_ze",
    "ax_isi_ze",
]
THREE_PART3_CURVE_KEYS = [k for k in THREE_PART3_PANEL_KEYS if not k.startswith("ax_hdr")]
THREE_PART_AXIS_ORDER = (
    THREE_PART1_PANEL_KEYS + THREE_PART2_PANEL_KEYS + THREE_PART3_PANEL_KEYS
)
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
    draw_probe_layout_on_axes(ax, probe_layout, channel_name, set_mea_title=False)
    ax.set_title(f"MEA layout — highlighted channel: {channel_name}", fontsize=10, pad=4)


def _soften_figure_linewidths(
    fig: Any,
    scale: float = 0.3,
    min_width: float = 0.3,
    marker_scale: float = 0.3,
    scatter_scale: float = 0.3,
) -> None:
    """Reduce line/marker thickness globally for a figure."""
    for ax in fig.axes:
        for line in ax.get_lines():
            try:
                lw = float(line.get_linewidth())
            except Exception:
                continue
            line.set_linewidth(max(min_width, lw * scale))
            try:
                ms = float(line.get_markersize())
                line.set_markersize(max(1.0, ms * marker_scale))
            except Exception:
                pass
        for coll in ax.collections:
            try:
                sizes = coll.get_sizes()
                if sizes is not None and len(sizes) > 0:
                    coll.set_sizes(np.maximum(1.0, np.asarray(sizes, dtype=np.float64) * scatter_scale))
            except Exception:
                pass
            try:
                lws = coll.get_linewidths()
                if lws is not None and len(lws) > 0:
                    coll.set_linewidths(np.maximum(min_width, np.asarray(lws, dtype=np.float64) * scale))
            except Exception:
                pass


def _three_part_page_height(recording_count: int, *, include_imp: bool) -> float:
    base, per_recording = (
        THREE_PART_PAGE_HEIGHT_IMP if include_imp else THREE_PART_PAGE_HEIGHT_NO_IMP
    )
    height_in = base + per_recording * float(max(1, recording_count) - 1)
    return height_in * max(0.5, float(THREE_PART_PANEL_HEIGHT_SCALE))


def _apply_compact_axis_fonts(fig: Any) -> None:
    """Reduce subplot titles and axis label/tick font sizes on a figure."""
    for ax in fig.axes:
        if ax.get_title():
            ax.title.set_fontsize(AXIS_TITLE_FONT_SIZE)
        if ax.get_xlabel():
            ax.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
        if ax.get_ylabel():
            ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
            # Keep vertical titles inside the widened left margin.
            ax.yaxis.label.set_clip_on(False)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)


def _build_three_part_page_axes(
    *,
    zoom_t0: float,
    zoom_t1: float,
    n_recordings: int,
    first_row_height_ratio: float,
    first_row_text: Optional[str],
    first_row_mea_channel_name: Optional[str],
    probe_layout: Any,
    include_impedance_panel: bool,
) -> tuple[Any, dict[str, Any]]:
    recording_count = max(1, int(n_recordings))
    page_height_in = _three_part_page_height(recording_count, include_imp=include_impedance_panel)
    page_width_in = THREE_PART_PAGE_WIDTH_IN
    height_ratios = [first_row_height_ratio, *THREE_PART_ROW_HEIGHTS]
    if include_impedance_panel:
        height_ratios = [*height_ratios, 0.08, 0.92]
    fig = plt.figure(figsize=(page_width_in, page_height_in))
    gs = fig.add_gridspec(
        len(height_ratios),
        1,
        height_ratios=height_ratios,
        hspace=THREE_PART_GRID_HSPACE,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    if first_row_mea_channel_name is not None:
        _draw_mea_layout_panel(ax_top, probe_layout, first_row_mea_channel_name)
    else:
        ax_top.axis("off")
        if first_row_text:
            ax_top.text(
                0.02,
                0.5,
                first_row_text,
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                transform=ax_top.transAxes,
            )
    ax_full = fig.add_subplot(gs[1, 0])
    ax_full_filt = fig.add_subplot(gs[2, 0], sharex=ax_full)
    ax_first_trigger = fig.add_subplot(gs[3, 0], sharex=ax_full)
    ax_first_trigger_hp = fig.add_subplot(gs[4, 0], sharex=ax_full)
    ax_full_rms = fig.add_subplot(gs[5, 0], sharex=ax_full)
    ax_raster_f = fig.add_subplot(gs[6, 0], sharex=ax_full)
    ax_fr_f = fig.add_subplot(gs[7, 0], sharex=ax_full)
    ax_trial_fr_f = fig.add_subplot(gs[8, 0])
    ax_isi_f = fig.add_subplot(gs[9, 0])
    ax_hdr2 = fig.add_subplot(gs[10, 0])
    ax_hdr2.axis("off")
    ax_hdr2.text(
        0.02,
        0.04,
        f"Part 2 — Zoomed view [{zoom_t0:.2f}, {zoom_t1:.2f}] s (relative to trigger)",
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        transform=ax_hdr2.transAxes,
    )
    ax_zoom = fig.add_subplot(gs[11, 0])
    ax_zoom_filt = fig.add_subplot(gs[12, 0], sharex=ax_zoom)
    ax_zoom_first = fig.add_subplot(gs[13, 0], sharex=ax_zoom)
    ax_zoom_first_hp = fig.add_subplot(gs[14, 0], sharex=ax_zoom)
    ax_zoom_rms = fig.add_subplot(gs[15, 0])
    ax_raster_z = fig.add_subplot(gs[16, 0], sharex=ax_zoom)
    ax_fr_z = fig.add_subplot(gs[17, 0], sharex=ax_zoom)
    ax_trial_fr_z = fig.add_subplot(gs[18, 0])
    ax_isi_z = fig.add_subplot(gs[19, 0])
    ax_hdr3 = fig.add_subplot(gs[20, 0])
    ax_hdr3.axis("off")
    ax_hdr3.text(
        0.02,
        0.04,
        "Part 3 — Trigger-end zoom (rising edge)",
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        transform=ax_hdr3.transAxes,
    )
    ax_zoom_end = fig.add_subplot(gs[21, 0])
    ax_zoom_end_filt = fig.add_subplot(gs[22, 0], sharex=ax_zoom_end)
    ax_zoom_end_first = fig.add_subplot(gs[23, 0], sharex=ax_zoom_end)
    ax_zoom_end_first_hp = fig.add_subplot(gs[24, 0], sharex=ax_zoom_end)
    ax_zoom_end_rms = fig.add_subplot(gs[25, 0])
    ax_raster_ze = fig.add_subplot(gs[26, 0], sharex=ax_zoom_end)
    ax_fr_ze = fig.add_subplot(gs[27, 0], sharex=ax_zoom_end)
    ax_trial_fr_ze = fig.add_subplot(gs[28, 0])
    ax_isi_ze = fig.add_subplot(gs[29, 0])
    axes: dict[str, Any] = {
        "ax_top": ax_top,
        "ax_full": ax_full,
        "ax_full_filt": ax_full_filt,
        "ax_first_trigger": ax_first_trigger,
        "ax_first_trigger_hp": ax_first_trigger_hp,
        "ax_full_rms": ax_full_rms,
        "ax_raster_f": ax_raster_f,
        "ax_fr_f": ax_fr_f,
        "ax_trial_fr_f": ax_trial_fr_f,
        "ax_isi_f": ax_isi_f,
        "ax_hdr2": ax_hdr2,
        "ax_zoom": ax_zoom,
        "ax_zoom_filt": ax_zoom_filt,
        "ax_zoom_first": ax_zoom_first,
        "ax_zoom_first_hp": ax_zoom_first_hp,
        "ax_zoom_rms": ax_zoom_rms,
        "ax_raster_z": ax_raster_z,
        "ax_fr_z": ax_fr_z,
        "ax_trial_fr_z": ax_trial_fr_z,
        "ax_isi_z": ax_isi_z,
        "ax_hdr3": ax_hdr3,
        "ax_zoom_end": ax_zoom_end,
        "ax_zoom_end_filt": ax_zoom_end_filt,
        "ax_zoom_end_first": ax_zoom_end_first,
        "ax_zoom_end_first_hp": ax_zoom_end_first_hp,
        "ax_zoom_end_rms": ax_zoom_end_rms,
        "ax_raster_ze": ax_raster_ze,
        "ax_fr_ze": ax_fr_ze,
        "ax_trial_fr_ze": ax_trial_fr_ze,
        "ax_isi_ze": ax_isi_ze,
    }
    if include_impedance_panel:
        ax_imp_hdr = fig.add_subplot(gs[30, 0])
        ax_imp_hdr.axis("off")
        ax_imp_hdr.text(
            0.02,
            0.04,
            "Part 4 — Impedance |Z| @ 1 kHz vs session",
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            transform=ax_imp_hdr.transAxes,
        )
        axes["ax_imp_hdr"] = ax_imp_hdr
        axes["ax_imp"] = fig.add_subplot(gs[31, 0])
    return fig, axes


def _shift_panel_group_down(
    axes: dict[str, Any],
    panel_keys: Sequence[str],
    start_key: str,
    delta: float,
) -> None:
    """Shift a contiguous panel group downward (figure coordinates)."""
    if start_key not in panel_keys or start_key not in axes:
        return
    start_idx = panel_keys.index(start_key)
    shift_axes_down(
        [axes[key] for key in panel_keys[start_idx:] if key in axes],
        float(delta),
    )


def _fit_axes_vertical_to_figure(
    axes: dict[str, Any],
    *,
    bottom: float = THREE_PART_SUBPLOT_BOTTOM,
    top: float = THREE_PART_SUBPLOT_TOP,
) -> None:
    """Remap stacked axes into [bottom, top] when legend/section shifts push content off-page."""
    if not axes:
        return
    y_min = min(float(ax.get_position().y0) for ax in axes.values())
    y_max = max(float(ax.get_position().y1) for ax in axes.values())
    content_h = y_max - y_min
    avail_h = float(top) - float(bottom)
    if content_h <= 1e-9 or avail_h <= 1e-9:
        return
    if y_min >= bottom - 1e-6 and y_max <= top + 1e-6:
        return
    for ax in axes.values():
        pos = ax.get_position()
        rel_bottom = (float(pos.y0) - y_min) / content_h
        rel_height = float(pos.height) / content_h
        ax.set_position(
            [pos.x0, bottom + rel_bottom * avail_h, pos.width, rel_height * avail_h]
        )


def _enforce_min_vertical_gap(
    axes: dict[str, Any],
    axis_order: Sequence[str],
    below_start_key: str,
    min_gap: float,
) -> None:
    """Ensure a minimum figure-coordinate gap between the row above a section header and that header."""
    if below_start_key not in axes or below_start_key not in axis_order:
        return
    start_idx = axis_order.index(below_start_key)
    if start_idx <= 0:
        return
    above_key = axis_order[start_idx - 1]
    if above_key not in axes:
        return
    below_axes = [axes[key] for key in axis_order[start_idx:] if key in axes]
    if not below_axes:
        return
    pos_above = axes[above_key].get_position()
    pos_below = below_axes[0].get_position()
    delta = float(min_gap) - float(pos_above.y0 - pos_below.y1)
    if delta > 0:
        shift_axes_down(below_axes, delta)


@_profiled("pdf_savefig_channel_page")
def _finalize_and_save_three_part_page(
    *,
    fig: Any,
    pdf: PdfPages,
    axes: dict[str, Any],
    n_recordings: int,
) -> None:
    tick_keys = [
        "ax_full",
        "ax_full_filt",
        "ax_first_trigger_hp",
        "ax_first_trigger",
        "ax_full_rms",
        "ax_zoom",
        "ax_zoom_filt",
        "ax_zoom_first_hp",
        "ax_zoom_first",
        "ax_zoom_rms",
        "ax_raster_f",
        "ax_fr_f",
        "ax_trial_fr_f",
        "ax_isi_f",
        "ax_raster_z",
        "ax_fr_z",
        "ax_trial_fr_z",
        "ax_isi_z",
        "ax_raster_ze",
        "ax_fr_ze",
        "ax_trial_fr_ze",
        "ax_isi_ze",
        "ax_zoom_end",
        "ax_zoom_end_filt",
        "ax_zoom_end_first_hp",
        "ax_zoom_end_first",
        "ax_zoom_end_rms",
    ]
    if "ax_imp" in axes:
        tick_keys.append("ax_imp")
    for key in tick_keys:
        axes[key].tick_params(axis="x", labelbottom=True)
    fig.subplots_adjust(
        left=THREE_PART_SUBPLOT_LEFT,
        right=THREE_PART_SUBPLOT_RIGHT,
        top=THREE_PART_SUBPLOT_TOP,
        bottom=THREE_PART_SUBPLOT_BOTTOM,
        hspace=THREE_PART_GRID_HSPACE,
    )
    recording_count = max(1, int(n_recordings))
    include_imp = "ax_imp" in axes
    page_height_scale = max(
        1.0,
        _three_part_page_height(recording_count, include_imp=include_imp) / THREE_PART_PAGE_HEIGHT_REF,
    )
    gap_legend = max(
        0.0025,
        (0.006 + 0.003 * float(recording_count - 1)) * max(1.0, page_height_scale),
    )
    gap_legend_hp = gap_legend * 0.55
    gap_4_5 = max(
        0.006,
        (0.010 + 0.003 * float(recording_count - 1)) * max(1.0, page_height_scale),
    )
    # Fixed minimum gaps between Part 1/2/3 (do not shrink on compact PDF pages).
    gap_section = 0.026 + 0.006 * float(recording_count - 1)

    axis_order = list(THREE_PART_AXIS_ORDER)
    if include_imp:
        if "ax_imp_hdr" in axes:
            axis_order.append("ax_imp_hdr")
        if "ax_imp" in axes:
            axis_order.append("ax_imp")

    # First-trigger raw is above HP on the page; match legend-shift order to panel stack.
    part1_legend_shifts = (
        ("ax_full", gap_legend),
        ("ax_full_filt", gap_legend),
        ("ax_first_trigger", gap_legend),
        ("ax_first_trigger_hp", gap_legend_hp),
        ("ax_fr_f", gap_4_5),
    )
    part2_legend_shifts = (
        ("ax_zoom", gap_legend),
        ("ax_zoom_filt", gap_legend),
        ("ax_zoom_first", gap_legend),
        ("ax_zoom_first_hp", gap_legend_hp),
        ("ax_fr_z", gap_4_5),
    )
    part3_legend_shifts = (
        ("ax_zoom_end", gap_legend),
        ("ax_zoom_end_filt", gap_legend),
        ("ax_zoom_end_first", gap_legend),
        ("ax_zoom_end_first_hp", gap_legend_hp),
        ("ax_fr_ze", gap_4_5),
    )
    for start_key, gap in part1_legend_shifts:
        _shift_panel_group_down(axes, THREE_PART1_PANEL_KEYS, start_key, gap)
    for start_key, gap in part2_legend_shifts:
        # Keep Part 2 section title glued to the first curve panel (legend clearance).
        if start_key == "ax_zoom":
            _shift_panel_group_down(axes, THREE_PART2_PANEL_KEYS, "ax_hdr2", gap)
        else:
            _shift_panel_group_down(axes, THREE_PART2_CURVE_KEYS, start_key, gap)
    for start_key, gap in part3_legend_shifts:
        if start_key == "ax_zoom_end":
            _shift_panel_group_down(axes, THREE_PART3_PANEL_KEYS, "ax_hdr3", gap)
        else:
            _shift_panel_group_down(axes, THREE_PART3_CURVE_KEYS, start_key, gap)

    _enforce_min_vertical_gap(axes, axis_order, "ax_hdr2", gap_section)
    _enforce_min_vertical_gap(axes, axis_order, "ax_hdr3", gap_section)
    if include_imp and "ax_imp_hdr" in axes:
        _enforce_min_vertical_gap(axes, axis_order, "ax_imp_hdr", gap_section)

    panel_width = THREE_PART_SUBPLOT_RIGHT - THREE_PART_SUBPLOT_LEFT
    for ax in axes.values():
        pos = ax.get_position()
        ax.set_position([THREE_PART_SUBPLOT_LEFT, pos.y0, panel_width, pos.height])
    _fit_axes_vertical_to_figure(axes)
    _soften_figure_linewidths(fig)
    _apply_compact_axis_fonts(fig)
    # Avoid bbox_inches="tight" on huge multi-panel figures (very slow).
    pdf.savefig(fig, dpi=PDF_DPI, pad_inches=0.15)
    plt.close(fig)


def _spike_times_per_trial(
    windows_ch: np.ndarray,
    t_rel: np.ndarray,
    fs: float,
    threshold: float,
    intan_dsp: IntanDspSettings | None = None,
) -> list[np.ndarray]:
    """For one channel: list of spike-time arrays (s rel. trigger), one per trial."""
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
    counts = np.zeros(edges.size - 1, dtype=np.float64)
    for st in spike_times_per_trial:
        if st.size == 0:
            continue
        counts += np.histogram(st, bins=edges)[0]
    window_bins = max(1, int(round(float(bin_width_s) / dt)))
    kernel = np.ones(window_bins, dtype=np.float64)
    sliding_counts = np.convolve(counts, kernel, mode="same")
    effective_window_s = float(window_bins) * dt
    rate = sliding_counts / (max(n_trials, 1) * effective_window_s)
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers, rate


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


def _default_filter_short_label() -> str:
    return "bessel high-pass ord.2 @ 250 Hz"


def _default_filter_title_label() -> str:
    return "bessel high-pass @ 250 Hz"


def _spike_pipeline_captions(
    intan_dsp: IntanDspSettings | None = None,
) -> Tuple[str, str]:
    """(short for subtitles, detailed for footer note)"""
    if intan_dsp is None:
        short = _default_filter_short_label()
        return short, (
            f"{_default_filter_title_label()}, order 2; "
            "RMS window 1 s; spike detection on filtered signal"
        )
    short = intan_dsp.filter_short_label()
    detail = (
        f"{intan_dsp.filter_title_label()}, order {intan_dsp.filter_order}; "
        f"RMS window {intan_dsp.rms_window_s:g} s; "
        f"spike thr. {intan_dsp.spike_threshold_uv:g} µV"
    )
    if intan_dsp.artifact_suppression_enabled:
        detail += f"; artifact {intan_dsp.artifact_threshold_uv:g} µV"
    return short, detail


def _intan_hp_mean_filter_captions(intan_dsp: IntanDspSettings | None) -> tuple[str, str]:
    """Return (title suffix, compact legend spec) for software-filtered mean traces."""
    if intan_dsp is None:
        short = _default_filter_short_label()
        return f" — mean trace: {short}", short
    short = intan_dsp.filter_short_label()
    return f" — mean trace: {short}", short


def _intan_hp_mean_trace_label(name: str, intan_hp_legend: str) -> str:
    return f"{name} mean ({intan_hp_legend})"


def _mean_triggered_average_row(
    row: np.ndarray,
    valid_triggers: np.ndarray,
    pre_n: int,
    post_n: int,
    n_expected: int,
) -> np.ndarray:
    """Average triggered windows from one 1D channel row (mmap view: only window slices are read)."""
    win_len = int(pre_n) + int(post_n)
    acc = np.zeros(win_len, dtype=np.float64)
    triggers = np.asarray(valid_triggers, dtype=np.int64)
    for trig in triggers:
        start = int(trig - pre_n)
        end = int(trig + post_n)
        acc += np.asarray(row[start:end], dtype=np.float64)
    y = acc / float(max(triggers.size, 1))
    if y.shape[0] != n_expected:
        y = np.asarray(y[:n_expected], dtype=np.float64)
    return y


def _first_trigger_window(
    source: AmplifierSpikeSource,
    ch: int,
    n_expected: int,
    *,
    highpass: bool = False,
) -> Optional[np.ndarray]:
    """Extract the first-trigger window (raw amplifier or Intan high-pass)."""
    if source.valid_triggers.size == 0:
        return None
    first_trig = int(source.valid_triggers[0])
    start = int(first_trig - source.pre_n)
    end = int(first_trig + source.post_n)
    data = source.highpass if highpass else source.amplifier
    curve = np.asarray(data[ch, start:end], dtype=np.float64)
    if curve.shape[0] != n_expected:
        return None
    return curve


def _collect_first_trigger_windows(
    spike_sources: Sequence[AmplifierSpikeSource],
    ch: int,
    n_expected: int,
) -> tuple[list[Optional[np.ndarray]], list[Optional[np.ndarray]]]:
    raw_curves: list[Optional[np.ndarray]] = []
    hp_curves: list[Optional[np.ndarray]] = []
    for src in spike_sources:
        raw_curves.append(_first_trigger_window(src, ch, n_expected, highpass=False))
        hp_curves.append(_first_trigger_window(src, ch, n_expected, highpass=True))
    return raw_curves, hp_curves


def _trigger_end_zoom_bounds(
    end_markers: Sequence[float],
    zoom_t0: float,
    zoom_t1: float,
) -> tuple[float, float] | None:
    if not end_markers:
        return None
    return float(min(end_markers) + zoom_t0), float(max(end_markers) + zoom_t1)


def _mark_unavailable_axis(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _add_trace_reference_overlays(
    ax: Any,
    *,
    zoom_t0: float,
    zoom_t1: float,
    end_markers: Sequence[float],
    onset_label: str = "Trigger (onset)",
    end_line_specs: Optional[Sequence[tuple[float, str]]] = None,
    show_zoom_span: bool = True,
    show_end_zoom_span: bool = True,
    aux_legend_label: Optional[str] = None,
) -> None:
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="red",
        label=onset_label if aux_legend_label is None else aux_legend_label,
    )
    if end_line_specs:
        for value, label in end_line_specs:
            ax.axvline(value, linestyle=":", linewidth=0.9, color="0.45", label=label)
    else:
        for value in end_markers:
            ax.axvline(value, linestyle=":", linewidth=0.9, color="0.45")
    if show_zoom_span:
        ax.axvspan(
            zoom_t0,
            zoom_t1,
            alpha=0.12,
            color="green",
            label=("Zoom region" if aux_legend_label is None else aux_legend_label),
        )
    if show_end_zoom_span:
        end_bounds = _trigger_end_zoom_bounds(end_markers, zoom_t0, zoom_t1)
        if end_bounds is not None:
            ax.axvspan(
                end_bounds[0],
                end_bounds[1],
                alpha=0.10,
                color="gold",
                label=(
                    "Trigger-end zoom region"
                    if aux_legend_label is None
                    else aux_legend_label
                ),
            )


def _plot_mean_section_trace_panels(
    *,
    ax_raw: Any,
    ax_filt: Any,
    ax_first_hp: Any,
    ax_first_raw: Any,
    t_rel: np.ndarray,
    time_mask: Optional[np.ndarray],
    x_limits: Optional[tuple[float, float]],
    labels: Sequence[str],
    mean_filtered: Sequence[np.ndarray],
    mean_raw: Optional[Sequence[np.ndarray]],
    first_trigger_raw: Sequence[Optional[np.ndarray]],
    first_trigger_hp: Sequence[Optional[np.ndarray]],
    colors: Sequence[Any],
    zoom_t0: float,
    zoom_t1: float,
    end_markers: Sequence[float],
    intan_hp_legends: Sequence[str],
    title_raw: str,
    title_filt: str,
    title_first_hp: str,
    title_first_raw: str,
    legend_cols: int,
    aux_legend_label: Optional[str] = None,
    end_line_specs: Optional[Sequence[tuple[float, str]]] = None,
    show_reference_on_first_raw: bool = True,
    show_reference_on_first_hp: bool = False,
    first_trigger_hp_ylim: tuple[float, float] | None = None,
    base_lw: float = 1.2,
    main_lw: float = 1.35,
    first_lw: float = 1.1,
) -> None:
    """Plot the four mean-trace panels shared by each three-part section."""
    if time_mask is not None:
        t_plot = t_rel[time_mask]
    else:
        t_plot = t_rel

    def _slice(y: np.ndarray) -> np.ndarray:
        return y[time_mask] if time_mask is not None else y

    for i, y_filt in enumerate(mean_filtered):
        line_color = colors[i % len(colors)]
        y_raw = (
            np.asarray(mean_raw[i], dtype=np.float64)
            if mean_raw is not None and i < len(mean_raw)
            else np.asarray(y_filt, dtype=np.float64)
        )
        raw_label = f"{labels[i]} mean (raw)" if len(labels) > 1 else "Mean (raw)"
        ax_raw.plot(t_plot, _slice(y_raw), linewidth=base_lw, color=line_color, label=raw_label)
        hp_legend = (
            intan_hp_legends[i]
            if i < len(intan_hp_legends)
            else _default_filter_short_label()
        )
        filt_label = _intan_hp_mean_trace_label(labels[i], hp_legend)
        ax_filt.plot(t_plot, _slice(y_filt), linewidth=main_lw, color=line_color, label=filt_label)

    _add_trace_reference_overlays(
        ax_raw,
        zoom_t0=zoom_t0,
        zoom_t1=zoom_t1,
        end_markers=end_markers,
        end_line_specs=end_line_specs,
        aux_legend_label=aux_legend_label,
    )
    ax_raw.set_title(title_raw)
    ax_raw.set_ylabel("Potential (µV)")
    ax_raw.set_xlabel(TIME_REL_XLABEL)
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(ncol=legend_cols, **TRACE_PANEL_LEGEND_KWARGS)

    _add_trace_reference_overlays(
        ax_filt,
        zoom_t0=zoom_t0,
        zoom_t1=zoom_t1,
        end_markers=end_markers,
        end_line_specs=end_line_specs,
        aux_legend_label=aux_legend_label,
    )
    ax_filt.set_title(title_filt)
    ax_filt.set_ylabel("Potential (µV)")
    ax_filt.set_xlabel(TIME_REL_XLABEL)
    ax_filt.grid(True, alpha=0.3)
    ax_filt.legend(ncol=legend_cols, **TRACE_PANEL_LEGEND_KWARGS)

    if any(curve is not None for curve in first_trigger_hp):
        for i, curve in enumerate(first_trigger_hp):
            if curve is None:
                continue
            line_color = colors[i % len(colors)]
            hp_legend = (
                intan_hp_legends[i]
                if i < len(intan_hp_legends)
                else _default_filter_short_label()
            )
            hp_label = (
                f"{labels[i]} first trigger ({hp_legend})"
                if len(labels) > 1
                else f"First trigger ({hp_legend})"
            )
            ax_first_hp.plot(t_plot, _slice(curve), linewidth=first_lw, color=line_color, label=hp_label)
        if show_reference_on_first_hp:
            _add_trace_reference_overlays(
                ax_first_hp,
                zoom_t0=zoom_t0,
                zoom_t1=zoom_t1,
                end_markers=end_markers,
                end_line_specs=end_line_specs,
                aux_legend_label=aux_legend_label,
            )
        else:
            ax_first_hp.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
            if end_line_specs:
                for value, _label in end_line_specs:
                    ax_first_hp.axvline(value, linestyle=":", linewidth=0.9, color="0.45")
            else:
                for value in end_markers:
                    ax_first_hp.axvline(value, linestyle=":", linewidth=0.9, color="0.45")
        ax_first_hp.set_title(title_first_hp)
        ax_first_hp.set_ylabel("Potential (µV)")
        ax_first_hp.set_xlabel(TIME_REL_XLABEL)
        ax_first_hp.grid(True, alpha=0.3)
        if first_trigger_hp_ylim is not None:
            ax_first_hp.set_ylim(
                float(first_trigger_hp_ylim[0]),
                float(first_trigger_hp_ylim[1]),
            )
    else:
        _mark_unavailable_axis(ax_first_hp, "First trigger filtered signal unavailable")

    if any(curve is not None for curve in first_trigger_raw):
        for i, curve in enumerate(first_trigger_raw):
            if curve is None:
                continue
            line_color = colors[i % len(colors)]
            raw_label = (
                f"{labels[i]} first trigger raw"
                if len(labels) > 1
                else "First trigger (raw, no averaging)"
            )
            ax_first_raw.plot(t_plot, _slice(curve), linewidth=first_lw, color=line_color, label=raw_label)
        if show_reference_on_first_raw:
            _add_trace_reference_overlays(
                ax_first_raw,
                zoom_t0=zoom_t0,
                zoom_t1=zoom_t1,
                end_markers=end_markers,
                end_line_specs=end_line_specs,
                aux_legend_label=aux_legend_label,
            )
        else:
            ax_first_raw.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
            if end_line_specs:
                for value, _label in end_line_specs:
                    ax_first_raw.axvline(value, linestyle=":", linewidth=0.9, color="0.45")
            else:
                for value in end_markers:
                    ax_first_raw.axvline(value, linestyle=":", linewidth=0.9, color="0.45")
        ax_first_raw.set_title(title_first_raw)
        ax_first_raw.set_ylabel("Potential (µV)")
        ax_first_raw.set_xlabel(TIME_REL_XLABEL)
        ax_first_raw.grid(True, alpha=0.3)
    else:
        _mark_unavailable_axis(ax_first_raw, "First trigger raw signal unavailable")

    if x_limits is not None:
        for ax in (ax_raw, ax_filt, ax_first_hp, ax_first_raw):
            ax.set_xlim(float(x_limits[0]), float(x_limits[1]))


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


def _add_raster_threshold_legend(
    ax_raster: Any,
    threshold_caption: str,
    threshold_entries: Sequence[tuple[str, str]] | None = None,
) -> None:
    """Place raster legend below the plot with threshold information."""
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
        unique_handles = [Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.0)]
        unique_labels = [f"Threshold: {threshold_caption}"]
    entry_count = len(unique_labels)
    ncol = 1
    rows = max(1, entry_count)
    # Extra vertical offset when many stacked legend rows (single column).
    legend_y = -0.22 - 0.035 * max(0, rows - 1)
    legend = ax_raster.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=ncol,
        fontsize=8,
        framealpha=0.95,
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.2,
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
) -> None:
    """Overlaid raster / PSTH / ISI for N recordings."""
    short, _ = _spike_pipeline_captions(intan_dsp=intan_dsp)
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
    n_rec = len(spikes_per_recording)
    dense_overlay = n_rec > 2
    raster_alpha = 0.75 if not dense_overlay else 0.55
    psth_lw = 1.3 if not dense_overlay else 0.95
    trial_marker = "o" if n_rec <= 3 else "None"
    trial_markersize = 2.2 if n_rec <= 3 else 0.0
    isi_alpha = 0.35 if not dense_overlay else 0.25
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
                    alpha=raster_alpha,
                    linewidths=0,
                    label=labels[rec_idx] if tri == 0 else "",
                )
        y_offset += len(st_per_trial)
        if rec_idx < len(spikes_per_recording) - 1:
            ax_raster.axhline(y_offset - 0.5, color="0.55", linestyle="--", linewidth=0.8, alpha=0.7)
    if threshold_caption is not None:
        cap = threshold_caption
    else:
        cap = (
            f"threshold {spike_threshold_uv:g} µV (falling)"
            if spike_threshold_uv < 0
            else f"threshold {spike_threshold_uv:g} µV (rising)"
        )
    ax_raster.set_ylabel("Trial # (grouped by file)")
    ax_raster.set_title(f"{sec}Raster — {short}")
    ax_raster.grid(True, alpha=0.25, axis="x")
    ax_raster.set_ylim(-0.5, max(y_offset - 0.5, 0.5))
    ax_raster.set_xlim(t_xlim_lo, t_xlim_hi)
    _add_raster_threshold_legend(
        ax_raster,
        cap,
        threshold_entries=threshold_entries,
    )

    # Sliding PSTH time window; values are evaluated at sampling cadence in _psth_mean_hz().
    bin_w = max(float(psth_bin_window_s), 1.0 / fs)
    for rec_idx, st_per_trial in enumerate(spikes_per_recording):
        tc, rate = _psth_mean_hz(
            st_per_trial,
            t_rel,
            max(len(st_per_trial), 1),
            bin_w,
            t_range_s=psth_t_range,
        )
        if tc.size:
            ax_fr.plot(
                tc,
                rate,
                linewidth=psth_lw,
                color=colors[rec_idx % len(colors)],
                label=labels[rec_idx],
            )
    ax_fr.set_ylabel("Rate (Hz)")
    ax_fr.set_title(f"{sec}Firing rate (PSTH time window = {bin_w:g} s) — {short}")
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
                marker=trial_marker,
                markersize=trial_markersize,
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
                alpha=isi_alpha,
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


def _spike_threshold_caption(
    threshold_uv: float,
    polarity: str | None = None,
) -> str:
    from intan_rhx_dsp import normalize_spike_threshold

    mag, pol = normalize_spike_threshold(threshold_uv, polarity)  # type: ignore[arg-type]
    if pol == "positive":
        return f"threshold +{mag:g} µV (above, rising edge)"
    return f"threshold −{mag:g} µV (below, falling edge)"


def _resolve_channel_spike_threshold(
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
    from intan_rhx_dsp import effective_spike_threshold_uv

    pol = spike_threshold_polarity
    if str(mode).strip().lower() != "rms_multiple":
        eff = effective_spike_threshold_uv(fixed_threshold_uv, pol)  # type: ignore[arg-type]
        return eff, _spike_threshold_caption(fixed_threshold_uv, pol)
    if mean_rms_uv is None:
        if source is None:
            eff = effective_spike_threshold_uv(fixed_threshold_uv, pol)  # type: ignore[arg-type]
            return eff, _spike_threshold_caption(fixed_threshold_uv, pol)
        mean_rms_uv = source.mean_rms_for_channel(int(channel_index))
    magnitude_uv = float(rms_multiplier) * float(mean_rms_uv)
    eff = effective_spike_threshold_uv(magnitude_uv, pol)  # type: ignore[arg-type]
    direction = "below" if pol == "negative" else "above"
    return eff, (
        f"{rms_multiplier:g}x RMS mean/channel "
        f"({magnitude_uv:g} µV, {direction})"
    )


class _PlotRenderCache:
    """Caches per-channel means, RMS profiles, spike times and thresholds during PDF render."""

    def __init__(
        self,
        t_rel: np.ndarray,
        pre_n: int,
        post_n: int,
        t0_rms: float,
        t1_rms: float,
        rms_window_s: float,
    ) -> None:
        self.t_rel = t_rel
        self.pre_n = int(pre_n)
        self.post_n = int(post_n)
        self.t0_rms = float(t0_rms)
        self.t1_rms = float(t1_rms)
        self.rms_window_s = float(rms_window_s)
        self.n_expected = int(t_rel.shape[0])
        self._mean_raw: dict[tuple[int, int], np.ndarray] = {}
        self._mean_hp: dict[tuple[int, int], np.ndarray] = {}
        self._rms: dict[tuple[int, int | None], tuple[np.ndarray, np.ndarray]] = {}
        self._spikes: dict[tuple[int, int, float], list[np.ndarray]] = {}
        self._channel_rms_uv: dict[tuple[int, int], float] = {}
        self._threshold: dict[tuple[int, int], tuple[float, str]] = {}

    @staticmethod
    def _normalize_mean_row(row: np.ndarray, n_expected: int) -> np.ndarray:
        y = np.asarray(row, dtype=np.float64)
        if y.shape[0] != n_expected:
            y = np.asarray(y[:n_expected], dtype=np.float64)
        return y

    def prefill_means(
        self,
        spike_sources: Sequence[AmplifierSpikeSource],
        n_channels: int,
        channel_workers: int | None = None,
    ) -> None:
        """Pre-compute all channel means in parallel (mmap-friendly, one pass per stack)."""
        for src_idx, src in enumerate(spike_sources):
            n_ch = min(int(n_channels), int(src.amplifier.shape[0]))
            if n_ch <= 0:
                continue
            workers = resolve_channel_workers(channel_workers, n_ch)
            check_analysis_cancelled()
            raw_all = mean_triggered_windows_channelwise(
                src.amplifier,
                src.valid_triggers,
                self.pre_n,
                self.post_n,
                channel_workers=workers,
            )
            check_analysis_cancelled()
            hp_all = mean_triggered_windows_channelwise(
                src.highpass,
                src.valid_triggers,
                self.pre_n,
                self.post_n,
                channel_workers=workers,
            )
            for ch in range(n_ch):
                self._mean_raw[(src_idx, ch)] = self._normalize_mean_row(
                    raw_all[ch], self.n_expected
                )
                self._mean_hp[(src_idx, ch)] = self._normalize_mean_row(
                    hp_all[ch], self.n_expected
                )

    def mean_raw(self, src_idx: int, source: AmplifierSpikeSource, ch: int) -> np.ndarray:
        key = (src_idx, ch)
        cached = self._mean_raw.get(key)
        if cached is not None:
            return cached
        mean = _mean_triggered_average_row(
            source._amp_row(ch),
            source.valid_triggers,
            self.pre_n,
            self.post_n,
            self.n_expected,
        )
        self._mean_raw[key] = mean
        return mean

    def mean_hp(self, src_idx: int, source: AmplifierSpikeSource, ch: int) -> np.ndarray:
        key = (src_idx, ch)
        cached = self._mean_hp.get(key)
        if cached is not None:
            return cached
        mean = _mean_triggered_average_row(
            source._high_row(ch),
            source.valid_triggers,
            self.pre_n,
            self.post_n,
            self.n_expected,
        )
        self._mean_hp[key] = mean
        return mean

    def rms_profile(
        self,
        src_idx: int,
        source: AmplifierSpikeSource,
        channel_index: int | None = None,
        *,
        n_channels: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if channel_index is None:
            if n_channels is None:
                n_channels = int(source.highpass.shape[0])
            return self.rms_profile_recording_mean(src_idx, int(n_channels))
        key = (src_idx, channel_index)
        cached = self._rms.get(key)
        if cached is not None:
            return cached
        profile = _mean_rms_profile_from_source_window(
            source,
            self.t0_rms,
            self.t1_rms,
            self.rms_window_s,
            channel_index=channel_index,
        )
        self._rms[key] = profile
        return profile

    def rms_profile_recording_mean(
        self,
        src_idx: int,
        n_channels: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean RMS across channels from already-cached per-channel profiles."""
        key = (src_idx, None)
        cached = self._rms.get(key)
        if cached is not None:
            return cached
        tx_ref: np.ndarray | None = None
        acc: np.ndarray | None = None
        n_ok = 0
        for ch in range(int(n_channels)):
            per_ch = self._rms.get((src_idx, ch))
            if per_ch is None:
                continue
            tx, vals = per_ch
            if vals.size == 0 or tx.size == 0:
                continue
            if acc is None:
                tx_ref = np.asarray(tx, dtype=np.float64)
                acc = np.asarray(vals, dtype=np.float64)
                n_ok = 1
            elif tx_ref is not None and tx.shape == tx_ref.shape and np.array_equal(tx, tx_ref):
                acc += np.asarray(vals, dtype=np.float64)
                n_ok += 1
        if acc is None or tx_ref is None or n_ok == 0:
            empty = np.array([], dtype=np.float64)
            profile = (empty, empty)
        else:
            profile = (tx_ref, acc / float(n_ok))
        self._rms[key] = profile
        return profile

    def prefill_channel_metrics(
        self,
        spike_sources: Sequence[AmplifierSpikeSource],
        n_channels: int,
        channel_workers: int | None,
        *,
        spike_threshold_mode: str,
        spike_threshold_uv: float,
        spike_threshold_polarity: str,
        spike_threshold_rms_multiplier: float,
    ) -> None:
        """Pre-compute per-channel RMS and spike times in parallel before PDF pages."""
        n_src = len(spike_sources)
        n_ch = int(n_channels)
        jobs = [(si, ch) for si in range(n_src) for ch in range(n_ch)]

        def _one(job: tuple[int, int]) -> None:
            si, ch = job
            src = spike_sources[si]
            check_analysis_cancelled()
            self.rms_profile(si, src, ch)
            thr_uv, _ = self.resolve_threshold(
                si,
                mode=spike_threshold_mode,
                fixed_threshold_uv=spike_threshold_uv,
                spike_threshold_polarity=spike_threshold_polarity,
                rms_multiplier=spike_threshold_rms_multiplier,
                source=src,
                channel_index=ch,
            )
            self.spike_times_per_trial(si, src, ch, thr_uv)

        workers = resolve_channel_workers(channel_workers, len(jobs))
        if workers <= 1 or len(jobs) <= 1:
            for job in jobs:
                _one(job)
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_one, jobs, chunksize=max(1, len(jobs) // (workers * 4))))

    def channel_rms_uv(self, src_idx: int, source: AmplifierSpikeSource, ch: int) -> float:
        key = (src_idx, ch)
        cached = self._channel_rms_uv.get(key)
        if cached is not None:
            return cached
        value = float(source.mean_rms_for_channel(ch))
        self._channel_rms_uv[key] = value
        return value

    def resolve_threshold(
        self,
        src_idx: int,
        *,
        mode: str,
        fixed_threshold_uv: float,
        spike_threshold_polarity: str,
        rms_multiplier: float,
        source: AmplifierSpikeSource,
        channel_index: int,
    ) -> tuple[float, str]:
        key = (src_idx, channel_index)
        cached = self._threshold.get(key)
        if cached is not None:
            return cached
        if str(mode).strip().lower() != "rms_multiple":
            resolved = _resolve_channel_spike_threshold(
                mode=mode,
                fixed_threshold_uv=fixed_threshold_uv,
                spike_threshold_polarity=spike_threshold_polarity,
                rms_multiplier=rms_multiplier,
                source=source,
                channel_index=channel_index,
            )
        else:
            resolved = _resolve_channel_spike_threshold(
                mode=mode,
                fixed_threshold_uv=fixed_threshold_uv,
                spike_threshold_polarity=spike_threshold_polarity,
                rms_multiplier=rms_multiplier,
                source=source,
                channel_index=channel_index,
                mean_rms_uv=self.channel_rms_uv(src_idx, source, channel_index),
            )
        self._threshold[key] = resolved
        return resolved

    def spike_times_per_trial(
        self,
        src_idx: int,
        source: AmplifierSpikeSource,
        ch: int,
        threshold_uv: float,
    ) -> list[np.ndarray]:
        key = (src_idx, ch, float(threshold_uv))
        cached = self._spikes.get(key)
        if cached is not None:
            return cached
        spikes = source.spike_times_per_trial_for_channel(ch, self.t_rel, threshold_uv)
        self._spikes[key] = spikes
        return spikes


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
            fontsize=6,
            alpha=0.9,
            zorder=4,
        )
    ax_imp.set_ylabel("|Z| @ 1 kHz (Ω)", fontsize=8)
    ax_imp.set_xlabel("Session time (_YYMMDD_HHMMSS)", fontsize=8)
    ax_imp.margins(x=0.08)
    date_locator = mdates.AutoDateLocator()
    ax_imp.xaxis.set_major_locator(date_locator)
    ax_imp.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
    ax_imp.tick_params(axis="both", labelsize=7)
    ax_imp.grid(True, which="major", alpha=0.35)
    for label in ax_imp.get_xticklabels():
        label.set_rotation(18)
        label.set_ha("right")


def _append_mean_impedance_summary_page(
    pdf: PdfPages,
    sessions: Sequence[ImpedanceSession],
) -> None:
    """Final PDF page: mean |Z|@1 kHz averaged over CSV channels vs session time."""
    if not sessions:
        return
    check_analysis_cancelled()
    fig, ax = plt.subplots(figsize=(SUMMARY_PAGE_WIDTH_IN, SUMMARY_PAGE_HEIGHT_IN))
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
    _apply_compact_axis_fonts(fig)
    pdf.savefig(fig, dpi=PDF_DPI)
    plt.close(fig)


@_profiled("rms_from_source_window")
def _mean_rms_profile_from_source_window(
    source: AmplifierSpikeSource,
    t0_s: float,
    t1_s: float,
    rms_window_s: float,
    channel_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean RMS profile in [t0_s, t1_s], averaged over triggers (Intan Spike Scope, 1 s on HIGH)."""
    del rms_window_s  # Intan uses source.intan_dsp.rms_window_s (1 s).
    if source.valid_triggers.size == 0 or t1_s <= t0_s:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
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

    valid_trigs: list[int] = []
    for trig in source.valid_triggers:
        start = int(trig + start_off)
        end = int(trig + end_off)
        if start < 0 or end > n_samples:
            continue
        valid_trigs.append(int(trig))
    if not valid_trigs:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    channel_indices = [ch_idx] if ch_idx is not None else list(range(n_channels))
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
        ax.set_ylim(0.0, 20.0)
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
    *,
    filter_title: str | None = None,
) -> None:
    """Append one summary page: mean RMS profile on analysis timebase."""
    title_filter = filter_title or _default_filter_title_label()
    check_analysis_cancelled()
    fig, ax = plt.subplots(figsize=(SUMMARY_PAGE_WIDTH_IN, SUMMARY_PAGE_HEIGHT_IN))
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
        ax.set_title(f"Mean RMS profile ({title_filter}, RMS window = 1 s)")
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
    _apply_compact_axis_fonts(fig)
    pdf.savefig(fig, dpi=PDF_DPI)
    plt.close(fig)


def plot_channel_multi_comparison(
    t_rel: np.ndarray,
    channel_names: Sequence[str],
    output_dir: Path,
    labels: Sequence[str],
    spike_sources: Sequence[AmplifierSpikeSource],
    pre_n_common: int,
    post_n_common: int,
    pdf_title: Optional[str] = None,
    trigger_end_rising_rel_s_list: Optional[Sequence[Optional[float]]] = None,
    fs: Optional[float] = None,
    spike_threshold_uv: float = 70.0,
    spike_threshold_polarity: str = "negative",
    spike_threshold_mode: str = "fixed",
    spike_threshold_rms_multiplier: float = 4.0,
    psth_bin_window_s: float = 0.025,
    rms_window_s: float = 0.050,
    zoom_t0_s: float = ZOOM_T0,
    zoom_t1_s: float = ZOOM_T1,
    sampling_percent: int = 100,
    probe_layout_json: Optional[Path] = None,
    impedance_sessions: Optional[Sequence[ImpedanceSession]] = None,
    channel_workers: int | None = None,
    first_trigger_hp_ylim_enabled: bool = False,
    first_trigger_hp_ylim_min_uv: float = -200.0,
    first_trigger_hp_ylim_max_uv: float = 200.0,
) -> Path:
    """Multi-page PDF: overlay of N recordings (same aligned channels)."""
    _profile_before = _profile_snapshot()
    _profile_t0 = time.perf_counter()
    n_records = len(labels)
    if n_records < 1:
        raise ValueError("plot_channel_multi_comparison requires at least 1 aligned recording.")
    if len(spike_sources) != n_records:
        raise ValueError("`spike_sources` must contain N recordings.")
    if any(src is None for src in spike_sources):
        raise ValueError("All entries in `spike_sources` must be non-null.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if pdf_title is not None and pdf_title.strip():
        safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in pdf_title.strip())
        pdf_stem = safe_title.removesuffix(".pdf")
    else:
        pdf_stem = "multi_comparison"
    pdf_name = shorten_filename_for_windows(output_dir, f"{pdf_stem}.pdf")
    pdf_path = output_dir / pdf_name

    zoom_t0, zoom_t1 = float(zoom_t0_s), float(zoom_t1_s)
    first_trigger_hp_ylim: tuple[float, float] | None = None
    if first_trigger_hp_ylim_enabled:
        y_lo = float(first_trigger_hp_ylim_min_uv)
        y_hi = float(first_trigger_hp_ylim_max_uv)
        if y_hi <= y_lo:
            raise ValueError(
                "First-trigger HP y-axis: maximum (µV) must be strictly greater than minimum."
            )
        first_trigger_hp_ylim = (y_lo, y_hi)
    main_intan_dsp = spike_sources[0].intan_dsp if spike_sources else None
    filter_title = (
        main_intan_dsp.filter_title_label()
        if main_intan_dsp is not None
        else _default_filter_title_label()
    )
    intan_hp_filt_note, _ = _intan_hp_mean_filter_captions(main_intan_dsp)
    both_note = " — raw and filtered means on separate panels"
    zoom_title = f"Zoom: {zoom_t0:.1f} to {zoom_t1:.1f} s (relative to trigger){intan_hp_filt_note}{both_note}"
    n_channels = min(src.amplifier.shape[0] for src in spike_sources)
    zmask = (t_rel >= zoom_t0) & (t_rel <= zoom_t1)
    end_markers = [v for v in (trigger_end_rising_rel_s_list or []) if v is not None]
    _has_spike_cmp = (
        fs is not None
        and len(spike_sources) == n_records
    )
    _, spike_cmp_pipe = _spike_pipeline_captions(intan_dsp=spike_sources[0].intan_dsp)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

    probe_layout_loaded = None
    if probe_layout_json is not None:
        probe_layout_loaded = load_probe_layout_json(Path(probe_layout_json))
    t0_rms = float(t_rel[0]) if t_rel.size else 0.0
    t1_rms = float(t_rel[-1]) if t_rel.size else 0.0
    render_cache = _PlotRenderCache(
        t_rel,
        int(pre_n_common),
        int(post_n_common),
        t0_rms,
        t1_rms,
        rms_window_s,
    )
    render_cache.prefill_means(spike_sources, n_channels, channel_workers=channel_workers)
    print(
        f"Pre-computing RMS and spikes for {n_channels} channel(s) × {n_records} recording(s)..."
    )
    render_cache.prefill_channel_metrics(
        spike_sources,
        n_channels,
        channel_workers,
        spike_threshold_mode=spike_threshold_mode,
        spike_threshold_uv=spike_threshold_uv,
        spike_threshold_polarity=spike_threshold_polarity,
        spike_threshold_rms_multiplier=spike_threshold_rms_multiplier,
    )
    with PdfPages(pdf_path) as pdf:
        for ch in range(n_channels):
            check_analysis_cancelled()
            channel_name = str(channel_names[ch])
            means_ch: list[np.ndarray] = []
            means_raw_ch: list[np.ndarray] = []
            intan_hp_legends: list[str] = []
            for src_idx, src in enumerate(spike_sources):
                means_raw_ch.append(render_cache.mean_raw(src_idx, src, ch))
                means_ch.append(render_cache.mean_hp(src_idx, src, ch))
                _hp_note, hp_legend = _intan_hp_mean_filter_captions(src.intan_dsp)
                intan_hp_legends.append(hp_legend)

            fig, _axes = _build_three_part_page_axes(
                zoom_t0=zoom_t0,
                zoom_t1=zoom_t1,
                n_recordings=n_records,
                first_row_height_ratio=2.2,
                first_row_text=None,
                first_row_mea_channel_name=channel_name,
                probe_layout=probe_layout_loaded,
                include_impedance_panel=bool(impedance_sessions),
            )
            ax_full = _axes["ax_full"]
            ax_full_filt = _axes["ax_full_filt"]
            ax_first_trigger_hp = _axes["ax_first_trigger_hp"]
            ax_first_trigger = _axes["ax_first_trigger"]
            ax_full_rms = _axes["ax_full_rms"]
            ax_raster_f = _axes["ax_raster_f"]
            ax_fr_f = _axes["ax_fr_f"]
            ax_trial_fr_f = _axes["ax_trial_fr_f"]
            ax_isi_f = _axes["ax_isi_f"]
            ax_zoom = _axes["ax_zoom"]
            ax_zoom_filt = _axes["ax_zoom_filt"]
            ax_zoom_first_hp = _axes["ax_zoom_first_hp"]
            ax_zoom_first = _axes["ax_zoom_first"]
            ax_zoom_rms = _axes["ax_zoom_rms"]
            ax_raster_z = _axes["ax_raster_z"]
            ax_fr_z = _axes["ax_fr_z"]
            ax_trial_fr_z = _axes["ax_trial_fr_z"]
            ax_isi_z = _axes["ax_isi_z"]
            ax_zoom_end = _axes["ax_zoom_end"]
            ax_zoom_end_filt = _axes["ax_zoom_end_filt"]
            ax_zoom_end_first_hp = _axes["ax_zoom_end_first_hp"]
            ax_zoom_end_first = _axes["ax_zoom_end_first"]
            ax_zoom_end_rms = _axes["ax_zoom_end_rms"]
            ax_raster_ze = _axes["ax_raster_ze"]
            ax_fr_ze = _axes["ax_fr_ze"]
            ax_trial_fr_ze = _axes["ax_trial_fr_ze"]
            ax_isi_ze = _axes["ax_isi_ze"]

            if impedance_sessions:
                ax_imp = _axes["ax_imp"]
                _draw_impedance_evolution_panel(ax_imp, channel_name, impedance_sessions)

            legend_cols = 1
            rms_series_full_multi: list[tuple[str, np.ndarray, np.ndarray]] = []
            rms_series_zoom_multi: list[tuple[str, np.ndarray, np.ndarray]] = []
            rms_series_zoom_end_multi: list[tuple[str, np.ndarray, np.ndarray]] = []
            for i, src in enumerate(spike_sources):
                label = labels[i] if i < len(labels) else f"Recording {i + 1}"
                tx_full, rms_full_vals = render_cache.rms_profile(i, src, channel_index=ch)
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
            first_trigger_raw, first_trigger_hp = _collect_first_trigger_windows(
                spike_sources,
                ch,
                int(t_rel.shape[0]),
            )
            record_labels = [
                labels[i] if i < len(labels) else f"Recording {i + 1}"
                for i in range(len(means_ch))
            ]
            _plot_mean_section_trace_panels(
                ax_raw=ax_full,
                ax_filt=ax_full_filt,
                ax_first_hp=ax_first_trigger_hp,
                ax_first_raw=ax_first_trigger,
                t_rel=t_rel,
                time_mask=None,
                x_limits=(float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None,
                labels=record_labels,
                mean_filtered=means_ch,
                mean_raw=means_raw_ch,
                first_trigger_raw=first_trigger_raw,
                first_trigger_hp=first_trigger_hp,
                colors=colors,
                zoom_t0=zoom_t0,
                zoom_t1=zoom_t1,
                end_markers=end_markers,
                intan_hp_legends=intan_hp_legends,
                title_raw=f"Multi-comparison — {channel_name} — mean (raw, averaged){both_note}",
                title_filt=f"Multi-comparison — {channel_name} (full view){intan_hp_filt_note}{both_note}",
                title_first_hp=f"Part 1 — First trigger ({filter_title})",
                title_first_raw="Part 1 — First trigger raw (no averaging)",
                legend_cols=legend_cols,
                first_trigger_hp_ylim=first_trigger_hp_ylim,
            )
            _plot_rms_series(
                ax_full_rms,
                rms_series_full_multi,
                f"Part 1 — RMS evolution ({filter_title}, RMS window = 1 s)",
                x_limits=(float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None,
            )

            _plot_mean_section_trace_panels(
                ax_raw=ax_zoom,
                ax_filt=ax_zoom_filt,
                ax_first_hp=ax_zoom_first_hp,
                ax_first_raw=ax_zoom_first,
                t_rel=t_rel,
                time_mask=zmask,
                x_limits=(zoom_t0, zoom_t1),
                labels=record_labels,
                mean_filtered=means_ch,
                mean_raw=means_raw_ch,
                first_trigger_raw=first_trigger_raw,
                first_trigger_hp=first_trigger_hp,
                colors=colors,
                zoom_t0=zoom_t0,
                zoom_t1=zoom_t1,
                end_markers=end_markers,
                intan_hp_legends=intan_hp_legends,
                title_raw=f"Part 2 — Zoom mean (raw, averaged){both_note}",
                title_filt=zoom_title,
                title_first_hp=f"Part 2 — First trigger ({filter_title})",
                title_first_raw="Part 2 — First trigger raw (separate view)",
                legend_cols=legend_cols,
                show_reference_on_first_raw=False,
                show_reference_on_first_hp=False,
                first_trigger_hp_ylim=first_trigger_hp_ylim,
            )
            _plot_rms_series(
                ax_zoom_rms,
                rms_series_zoom_multi,
                f"Part 2 — RMS evolution ({filter_title}, RMS window = 1 s)",
                x_limits=(zoom_t0, zoom_t1),
            )

            end_zoom_range: tuple[float, float] | None = None
            if end_markers:
                end_zoom_t0 = float(min(end_markers) + zoom_t0)
                end_zoom_t1 = float(max(end_markers) + zoom_t1)
                end_zoom_range = (end_zoom_t0, end_zoom_t1)
                end_mask = (t_rel >= end_zoom_t0) & (t_rel <= end_zoom_t1)
                _plot_mean_section_trace_panels(
                    ax_raw=ax_zoom_end,
                    ax_filt=ax_zoom_end_filt,
                    ax_first_hp=ax_zoom_end_first_hp,
                    ax_first_raw=ax_zoom_end_first,
                    t_rel=t_rel,
                    time_mask=end_mask,
                    x_limits=(end_zoom_t0, end_zoom_t1),
                    labels=record_labels,
                    mean_filtered=means_ch,
                    mean_raw=means_raw_ch,
                    first_trigger_raw=first_trigger_raw,
                    first_trigger_hp=first_trigger_hp,
                    colors=colors,
                    zoom_t0=zoom_t0,
                    zoom_t1=zoom_t1,
                    end_markers=end_markers,
                    intan_hp_legends=intan_hp_legends,
                    title_raw=f"Part 3 — Trigger-end zoom mean (raw, averaged){both_note}",
                    title_filt=(
                        f"Trigger-end zoom: {end_zoom_t0:.2f} to {end_zoom_t1:.2f} s "
                        f"(relative to trigger){intan_hp_filt_note}{both_note}"
                    ),
                    title_first_hp=f"Part 3 — First trigger ({filter_title})",
                    title_first_raw="Part 3 — First trigger raw (separate view)",
                    legend_cols=legend_cols,
                    show_reference_on_first_raw=False,
                    show_reference_on_first_hp=False,
                    first_trigger_hp_ylim=first_trigger_hp_ylim,
                )
                _plot_rms_series(
                    ax_zoom_end_rms,
                    rms_series_zoom_end_multi,
                    f"Part 3 — RMS evolution ({filter_title}, RMS window = 1 s)",
                    x_limits=(end_zoom_t0, end_zoom_t1),
                )
            else:
                for ax, msg in (
                    (ax_zoom_end, "Trigger-end zoom unavailable\n(no rising edge after trigger)"),
                    (ax_zoom_end_filt, "Trigger-end zoom unavailable\n(no rising edge after trigger)"),
                    (ax_zoom_end_first_hp, "First trigger filtered signal unavailable"),
                    (ax_zoom_end_first, "First trigger raw signal unavailable"),
                    (ax_zoom_end_rms, "RMS evolution unavailable"),
                ):
                    _mark_unavailable_axis(ax, msg)

            if _has_spike_cmp:
                thresholds_and_captions = [
                    render_cache.resolve_threshold(
                        src_idx,
                        mode=spike_threshold_mode,
                        fixed_threshold_uv=spike_threshold_uv,
                        spike_threshold_polarity=spike_threshold_polarity,
                        rms_multiplier=spike_threshold_rms_multiplier,
                        source=src,
                        channel_index=ch,
                    )
                    for src_idx, src in enumerate(spike_sources)
                ]
                st_list = [
                    render_cache.spike_times_per_trial(src_idx, src, ch, thr_uv)
                    for src_idx, (src, (thr_uv, _caption)) in enumerate(
                        zip(spike_sources, thresholds_and_captions)
                    )
                ]
                if str(spike_threshold_mode).strip().lower() == "rms_multiple":
                    threshold_labels = [
                        f"{labels[i]}: {caption}"
                        for i, (_thr_uv, caption) in enumerate(thresholds_and_captions)
                    ]
                    threshold_caption = " | ".join(threshold_labels)
                else:
                    threshold_caption = _spike_threshold_caption(
                        spike_threshold_uv, spike_threshold_polarity
                    )
                threshold_entries = [
                    (labels[i], caption)
                    for i, (_thr_uv, caption) in enumerate(thresholds_and_captions)
                ]
                _draw_spike_panels_multi_channel(
                    ax_raster_f, ax_fr_f, ax_trial_fr_f, ax_isi_f, None, t_rel, float(fs), spike_threshold_uv,
                    psth_bin_window_s, labels,
                    intan_dsp=spike_sources[0].intan_dsp,
                    t_range_s=None, spikes_per_recording=st_list, sampling_percent=sampling_percent,
                    threshold_caption=threshold_caption,
                    threshold_entries=threshold_entries,
                )
                _draw_spike_panels_multi_channel(
                    ax_raster_z, ax_fr_z, ax_trial_fr_z, ax_isi_z, None, t_rel, float(fs), spike_threshold_uv,
                    psth_bin_window_s, labels,
                    intan_dsp=spike_sources[0].intan_dsp,
                    t_range_s=(zoom_t0, zoom_t1), spikes_per_recording=st_list, sampling_percent=sampling_percent,
                    threshold_caption=threshold_caption,
                    threshold_entries=threshold_entries,
                )
                if end_zoom_range is not None:
                    _draw_spike_panels_multi_channel(
                        ax_raster_ze, ax_fr_ze, ax_trial_fr_ze, ax_isi_ze, None, t_rel, float(fs), spike_threshold_uv,
                        psth_bin_window_s, labels,
                        intan_dsp=spike_sources[0].intan_dsp,
                        t_range_s=end_zoom_range, section_title="Trigger-end zoom", spikes_per_recording=st_list, sampling_percent=sampling_percent,
                        threshold_caption=threshold_caption,
                        threshold_entries=threshold_entries,
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

            _finalize_and_save_three_part_page(
                fig=fig,
                pdf=pdf,
                axes=_axes,
                n_recordings=n_records,
            )

        rms_series: list[tuple[str, np.ndarray, np.ndarray]] = []
        for i, src in enumerate(spike_sources):
            tx_rms, rms_vals = render_cache.rms_profile_recording_mean(i, n_channels)
            label = labels[i] if i < len(labels) else f"Recording {i + 1}"
            rms_series.append((label, tx_rms, rms_vals))
        if rms_series:
            _append_mean_rms_evolution_page(
                pdf, rms_series, rms_window_s, filter_title=filter_title
            )

        if impedance_sessions:
            _append_mean_impedance_summary_page(pdf, impedance_sessions)

    _profile_print_delta(
        "plot_channel_multi_comparison",
        _profile_before,
        time.perf_counter() - _profile_t0,
    )
    return pdf_path


