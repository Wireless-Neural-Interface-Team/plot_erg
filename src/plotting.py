"""PDF figures for triggered averaged traces, spike raster/PSTH/ISI, comparisons, and optional MEA layout inset."""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from concurrent.futures import ThreadPoolExecutor

from core import (
    AmplifierSpikeSource,
    check_analysis_cancelled,
    mean_triggered_windows_channelwise,
    resolve_channel_workers,
)
from intan_rhx_dsp import IntanDspSettings
from impedance_tracking import ImpedanceSession
from display_config import (
    PlotDisplaySettings,
    RecordingStyle,
    SectionPanels,
    ZoomMode,
    resolve_recording_plot_colors,
)
from pdf_layout import (
    LayoutFonts,
    Slot,
    build_stacked_pages,
    estimate_legend_rows,
    place_legend_below,
    save_figure_to_pdf,
)
from plot_utils import decimate_envelope, downsample_points, shorten_filename_for_windows

from draw_primitives import (  # noqa: F401 — re-export for PDF + legacy callers
    SPIKE_OVERLAY_DEFAULT_POST_MS,
    SPIKE_OVERLAY_DEFAULT_PRE_MS,
    TIME_REL_XLABEL,
    _draw_impedance_evolution_panel,
    _draw_onset_offset_lines,
    _draw_spike_overlay_panel,
    _draw_spike_panels_multi_channel,
    _extract_spike_waveforms,
)

from probe_layout import (
    draw_probe_layout_on_axes,
    load_probe_layout_json,
    match_contact_index,
    mea_panel_size_in,
)

# Zoom panel window (s), time relative to trigger (t=0)
ZOOM_T0 = -0.1
ZOOM_T1 = 0.4

LEGEND_FONT_SIZE = 15
AXIS_TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 15
TICK_LABEL_FONT_SIZE = 15
SECTION_HEADER_FONT_SIZE = 20
UNAVAILABLE_FONT_SIZE = 15
ANNOTATION_FONT_SIZE = 13
TABLE_FONT_SIZE = 15

# MEA map text (independent of the PDF fonts above)
MEA_TITLE_FONT_SIZE = 16
MEA_CONTACT_LABEL_FONT_MIN = 4.0
MEA_CONTACT_LABEL_FONT_MAX = 8.0
MEA_CONTACT_LABEL_FONT_SCALE = 60.0

plt.rcParams.update(
    {
        "font.size": AXIS_LABEL_FONT_SIZE,
        "axes.titlesize": AXIS_TITLE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_LABEL_FONT_SIZE,
        "ytick.labelsize": TICK_LABEL_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "pdf.compression": 6,
        "pdf.fonttype": 42,
        "path.simplify": True,
        "path.simplify_threshold": 0.35,
        "agg.path.chunksize": 20000,
    }
)
_HIGH_QUALITY_PDF = os.environ.get("PLOT_ERG_HIGH_QUALITY_PDF", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
THREE_PART_PAGE_WIDTH_IN = 12.0
PDF_DPI = 120 if _HIGH_QUALITY_PDF else 100
THREE_PART_PANEL_HEIGHT_SCALE = 1.15 if _HIGH_QUALITY_PDF else 1.0
MAX_CHANNEL_PAGE_HEIGHT_IN = 36.0
SUMMARY_PAGE_WIDTH_IN = 16.0
SUMMARY_PAGE_HEIGHT_IN = 9.0


@dataclass(frozen=True)
class PanelSpec:
    """One togglable graph type. Add a new PDF graph by appending a spec here."""

    field: str
    plot_height_in: float
    has_legend: bool = False
    extra_below: str = "none"


# Data-axes heights in inches (title / xlabel / legend are added by pdf_layout).
# Order: for each filter type (raw → HP → LP), trial-averaged / first stim / second stim;
# then RMS / PSTH / ISI (trial-averaged → first → second); then trial rate + raster.
# Zoom sections reuse the same order.
PANEL_SPECS: tuple[PanelSpec, ...] = (
    PanelSpec("mean_raw", 2.20, has_legend=True),
    PanelSpec("first_trigger_raw", 1.90, has_legend=True),
    PanelSpec("second_trigger_raw", 1.90, has_legend=True),
    PanelSpec("mean_hp", 2.10, has_legend=True),
    PanelSpec("first_trigger_hp", 1.80, has_legend=True),
    PanelSpec("second_trigger_hp", 1.80, has_legend=True),
    PanelSpec("mean_lp", 2.10, has_legend=True),
    PanelSpec("first_trigger_lp", 1.80, has_legend=True),
    PanelSpec("second_trigger_lp", 1.80, has_legend=True),
    PanelSpec("rms", 1.70, has_legend=True),
    PanelSpec("first_rms", 1.70, has_legend=True),
    PanelSpec("second_rms", 1.70, has_legend=True),
    PanelSpec("psth", 1.90, has_legend=True),
    PanelSpec("first_psth", 1.90, has_legend=True),
    PanelSpec("second_psth", 1.90, has_legend=True),
    PanelSpec("isi", 1.55, has_legend=True),
    PanelSpec("first_isi", 1.55, has_legend=True),
    PanelSpec("second_isi", 1.55, has_legend=True),
    PanelSpec("trial_rate", 1.45, has_legend=True),
    PanelSpec("raster", 1.85, has_legend=True),
    PanelSpec("spike_overlay", 2.00, has_legend=True),
)
MEA_PLOT_HEIGHT_IN = 4.80
IMPEDANCE_PLOT_HEIGHT_IN = 2.40

_FULL_PANEL_TO_AXIS: dict[str, str] = {
    "mean_raw": "ax_full",
    "mean_hp": "ax_full_filt_hp",
    "mean_lp": "ax_full_filt_lp",
    "first_trigger_raw": "ax_first_trigger",
    "first_trigger_hp": "ax_first_trigger_hp",
    "first_trigger_lp": "ax_first_trigger_lp",
    "second_trigger_raw": "ax_second_trigger",
    "second_trigger_hp": "ax_second_trigger_hp",
    "second_trigger_lp": "ax_second_trigger_lp",
    "rms": "ax_full_rms",
    "first_rms": "ax_full_rms_first",
    "second_rms": "ax_full_rms_second",
    "psth": "ax_fr_f",
    "first_psth": "ax_fr_first_f",
    "second_psth": "ax_fr_second_f",
    "isi": "ax_isi_f",
    "first_isi": "ax_isi_first_f",
    "second_isi": "ax_isi_second_f",
    "trial_rate": "ax_trial_fr_f",
    "raster": "ax_raster_f",
    "spike_overlay": "ax_overlay_f",
}

_ZOOM_ONSET_PANEL_TO_AXIS: dict[str, str] = {
    "mean_raw": "ax_zoom",
    "mean_hp": "ax_zoom_filt_hp",
    "mean_lp": "ax_zoom_filt_lp",
    "first_trigger_raw": "ax_zoom_first",
    "first_trigger_hp": "ax_zoom_first_hp",
    "first_trigger_lp": "ax_zoom_first_lp",
    "second_trigger_raw": "ax_zoom_second",
    "second_trigger_hp": "ax_zoom_second_hp",
    "second_trigger_lp": "ax_zoom_second_lp",
    "rms": "ax_zoom_rms",
    "first_rms": "ax_zoom_rms_first",
    "second_rms": "ax_zoom_rms_second",
    "psth": "ax_fr_z",
    "first_psth": "ax_fr_first_z",
    "second_psth": "ax_fr_second_z",
    "isi": "ax_isi_z",
    "first_isi": "ax_isi_first_z",
    "second_isi": "ax_isi_second_z",
    "trial_rate": "ax_trial_fr_z",
    "raster": "ax_raster_z",
    "spike_overlay": "ax_overlay_z",
}

_ZOOM_END_PANEL_TO_AXIS: dict[str, str] = {
    "mean_raw": "ax_zoom_end",
    "mean_hp": "ax_zoom_end_filt_hp",
    "mean_lp": "ax_zoom_end_filt_lp",
    "first_trigger_raw": "ax_zoom_end_first",
    "first_trigger_hp": "ax_zoom_end_first_hp",
    "first_trigger_lp": "ax_zoom_end_first_lp",
    "second_trigger_raw": "ax_zoom_end_second",
    "second_trigger_hp": "ax_zoom_end_second_hp",
    "second_trigger_lp": "ax_zoom_end_second_lp",
    "rms": "ax_zoom_end_rms",
    "first_rms": "ax_zoom_end_rms_first",
    "second_rms": "ax_zoom_end_rms_second",
    "psth": "ax_fr_ze",
    "first_psth": "ax_fr_first_ze",
    "second_psth": "ax_fr_second_ze",
    "isi": "ax_isi_ze",
    "first_isi": "ax_isi_first_ze",
    "second_isi": "ax_isi_second_ze",
    "trial_rate": "ax_trial_fr_ze",
    "raster": "ax_raster_ze",
    "spike_overlay": "ax_overlay_ze",
}

# Sub-part titles inserted before the first enabled panel of each group.
_PANEL_PART_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Voltage — raw (trial-averaged / first / second stim)",
        ("mean_raw", "first_trigger_raw", "second_trigger_raw"),
    ),
    (
        "Voltage — high-pass (trial-averaged / first / second stim)",
        ("mean_hp", "first_trigger_hp", "second_trigger_hp"),
    ),
    (
        "Voltage — low-pass (trial-averaged / first / second stim)",
        ("mean_lp", "first_trigger_lp", "second_trigger_lp"),
    ),
    (
        "RMS (trial-averaged / first / second stim)",
        ("rms", "first_rms", "second_rms"),
    ),
    (
        "PSTH / firing rate (trial-averaged / first / second stim)",
        ("psth", "first_psth", "second_psth"),
    ),
    (
        "ISI (all stimulations / first / second stim)",
        ("isi", "first_isi", "second_isi"),
    ),
    (
        "Rate per trial & raster (all stimulations)",
        ("trial_rate", "raster"),
    ),
    (
        "Spike overlay — all threshold detections",
        ("spike_overlay",),
    ),
)
_FIELD_TO_PART_GROUP: dict[str, str] = {
    field: title for title, fields in _PANEL_PART_GROUPS for field in fields
}
_OVERLAY_AXIS_KEYS: tuple[str, ...] = ("ax_overlay_f", "ax_overlay_z", "ax_overlay_ze")
_PROFILE_ENABLED = os.environ.get("PLOT_ERG_PROFILE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_PDF_TRACE_MAX_POINTS = 8000
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


def _trace_panels_enabled(panels: SectionPanels) -> bool:
    return any(
        getattr(panels, key)
        for key in (
            "mean_raw",
            "mean_hp",
            "mean_lp",
            "first_trigger_raw",
            "first_trigger_hp",
            "first_trigger_lp",
            "second_trigger_raw",
            "second_trigger_hp",
            "second_trigger_lp",
        )
    )


def _section_included(zoom_mode: ZoomMode, section: str) -> bool:
    if section == "full":
        return True
    if section == "zoom_onset":
        return zoom_mode in ("onset", "both")
    if section == "zoom_trigger_end":
        return zoom_mode in ("trigger_end", "both")
    return False


def _active_section_panels(
    display: PlotDisplaySettings, zoom_mode: ZoomMode
) -> list[SectionPanels]:
    sections: list[SectionPanels] = []
    if _section_included(zoom_mode, "full"):
        sections.append(display.full_view)
    if _section_included(zoom_mode, "zoom_onset"):
        sections.append(display.zoom_onset)
    if _section_included(zoom_mode, "zoom_trigger_end"):
        sections.append(display.zoom_trigger_end)
    return sections


def _display_stream_needs(
    display: PlotDisplaySettings, zoom_mode: ZoomMode
) -> tuple[bool, bool, bool, bool, bool]:
    """Return (need_raw, need_hp, need_lp, need_rms, need_spikes)."""
    sections = _active_section_panels(display, zoom_mode)
    if not sections:
        return False, False, False, False, False
    need_raw = any(
        s.mean_raw or s.first_trigger_raw or s.second_trigger_raw for s in sections
    )
    need_hp = any(
        s.mean_hp or s.first_trigger_hp or s.second_trigger_hp for s in sections
    )
    need_lp = any(
        s.mean_lp or s.first_trigger_lp or s.second_trigger_lp for s in sections
    )
    need_rms = any(s.rms or s.first_rms or s.second_rms for s in sections)
    need_spikes = any(
        s.raster
        or s.psth
        or s.first_psth
        or s.second_psth
        or s.isi
        or s.first_isi
        or s.second_isi
        or s.trial_rate
        or s.spike_overlay
        for s in sections
    )
    return need_raw, need_hp, need_lp, need_rms, need_spikes


def _legend_label(
    base: str,
    suffix: str,
    *,
    multi: bool,
    show_legend: bool,
) -> str | None:
    if not show_legend:
        return "_nolegend_"
    if multi:
        return f"{base} {suffix}"
    return suffix.strip() or base


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
            fontsize=UNAVAILABLE_FONT_SIZE,
            transform=ax.transAxes,
        )
        return
    draw_probe_layout_on_axes(
        ax,
        probe_layout,
        channel_name,
        set_mea_title=False,
        title_fontsize=MEA_TITLE_FONT_SIZE,
        contact_label_font_min=MEA_CONTACT_LABEL_FONT_MIN,
        contact_label_font_max=MEA_CONTACT_LABEL_FONT_MAX,
        contact_label_font_scale=MEA_CONTACT_LABEL_FONT_SCALE,
    )
    ax.set_title(f"MEA map — channel: {channel_name}", fontsize=MEA_TITLE_FONT_SIZE, pad=4)


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


def _pdf_fonts() -> LayoutFonts:
    return LayoutFonts(
        legend=LEGEND_FONT_SIZE,
        axis_title=AXIS_TITLE_FONT_SIZE,
        axis_label=AXIS_LABEL_FONT_SIZE,
        tick=TICK_LABEL_FONT_SIZE,
        section_header=SECTION_HEADER_FONT_SIZE,
        mea_title=MEA_TITLE_FONT_SIZE,
        table=TABLE_FONT_SIZE,
        unavailable=UNAVAILABLE_FONT_SIZE,
    )


def _scaled_plot_height_in(height_in: float) -> float:
    return float(height_in) * max(0.5, float(THREE_PART_PANEL_HEIGHT_SCALE))


def _channel_page_sections(display: PlotDisplaySettings, zoom_mode: ZoomMode) -> list[str]:
    """Enabled temporal sections, in PDF order. One PDF page is created per section."""
    sections: list[str] = []
    if _section_included(zoom_mode, "full") and display.full_view.any_enabled():
        sections.append("full")
    if _section_included(zoom_mode, "zoom_onset") and display.zoom_onset.any_enabled():
        sections.append("zoom_onset")
    if _section_included(zoom_mode, "zoom_trigger_end") and display.zoom_trigger_end.any_enabled():
        sections.append("zoom_trigger_end")
    return sections


def _layout_slots_for_page(
    display: PlotDisplaySettings,
    zoom_mode: ZoomMode,
    *,
    page_sections: Sequence[str],
    include_mea: bool,
    include_impedance: bool,
    zoom_onset_t0: float,
    zoom_onset_t1: float,
    n_legend_rows: int,
    probe_layout: Any = None,
) -> list[Slot]:
    """Build the ordered slot list for one page (only enabled panels)."""
    slots: list[Slot] = []
    wanted = set(page_sections)
    if include_mea:
        max_w = THREE_PART_PAGE_WIDTH_IN - 1.28 - 0.28
        if probe_layout is not None:
            mea_w, mea_h = mea_panel_size_in(
                probe_layout,
                max_width_in=max_w,
                max_height_in=10.5,
                font_min=MEA_CONTACT_LABEL_FONT_MIN,
                font_max=MEA_CONTACT_LABEL_FONT_MAX,
                font_scale=MEA_CONTACT_LABEL_FONT_SCALE,
            )
        else:
            mea_w, mea_h = max_w, _scaled_plot_height_in(MEA_PLOT_HEIGHT_IN)
        slots.append(
            Slot(
                key="ax_top",
                kind="mea",
                plot_height_in=mea_h,
                width_in=mea_w,
            )
        )

    def _append_section(
        section: str,
        panels: SectionPanels,
        axis_map: dict[str, str],
        header_key: str | None,
        header_text: str | None,
    ) -> None:
        if section not in wanted or not _section_included(zoom_mode, section):
            return
        enabled = [
            spec
            for spec in PANEL_SPECS
            if getattr(panels, spec.field)
        ]
        if not enabled:
            return
        if header_key is not None and header_text is not None:
            slots.append(
                Slot(
                    key=header_key,
                    kind="header",
                    plot_height_in=0.36,
                    header_text=header_text,
                )
            )
        seen_groups: set[str] = set()
        group_idx = 0
        for spec in enabled:
            group_title = _FIELD_TO_PART_GROUP.get(spec.field)
            if group_title is not None and group_title not in seen_groups:
                seen_groups.add(group_title)
                group_idx += 1
                slots.append(
                    Slot(
                        key=f"ax_hdr_{section}_g{group_idx}",
                        kind="header",
                        plot_height_in=0.30,
                        header_text=group_title,
                    )
                )
            extra = spec.extra_below if spec.extra_below in {"none", "psth_table"} else "none"
            slots.append(
                Slot(
                    key=axis_map[spec.field],
                    kind="plot",
                    plot_height_in=_scaled_plot_height_in(spec.plot_height_in),
                    has_legend=spec.has_legend,
                    extra_below=extra,  # type: ignore[arg-type]
                    table_rows=n_legend_rows if extra == "psth_table" else 0,
                )
            )
    _append_section(
        "full",
        display.full_view,
        _FULL_PANEL_TO_AXIS,
        "ax_hdr1",
        "Part 1 — Full view (trial-averaged / first / second stim)",
    )
    _append_section(
        "zoom_onset",
        display.zoom_onset,
        _ZOOM_ONSET_PANEL_TO_AXIS,
        "ax_hdr2",
        f"Part 2 — Stimulation-onset zoom [{zoom_onset_t0:.2f}, {zoom_onset_t1:.2f}] s rel. stimulation",
    )
    _append_section(
        "zoom_trigger_end",
        display.zoom_trigger_end,
        _ZOOM_END_PANEL_TO_AXIS,
        "ax_hdr3",
        "Part 3 — Stimulation-end zoom (next rising edge)",
    )
    if include_impedance:
        slots.append(
            Slot(
                key="ax_imp_hdr",
                kind="header",
                plot_height_in=0.32,
                header_text="Part 4 — Impedance |Z| @ 1 kHz vs session",
            )
        )
        slots.append(
            Slot(
                key="ax_imp",
                kind="plot",
                plot_height_in=_scaled_plot_height_in(IMPEDANCE_PLOT_HEIGHT_IN),
                has_legend=False,
            )
        )
    return slots


_SHAREX_FIELDS = frozenset(
    {
        "mean_raw",
        "mean_hp",
        "mean_lp",
        "first_trigger_raw",
        "first_trigger_hp",
        "first_trigger_lp",
        "second_trigger_raw",
        "second_trigger_hp",
        "second_trigger_lp",
        "rms",
        "first_rms",
        "second_rms",
        "raster",
        "psth",
        "first_psth",
        "second_psth",
    }
)


def _sharex_groups_for_slots() -> dict[str, str]:
    groups: dict[str, str] = {}
    for field, key in _FULL_PANEL_TO_AXIS.items():
        if field in _SHAREX_FIELDS:
            groups[key] = "full"
    for field, key in _ZOOM_ONSET_PANEL_TO_AXIS.items():
        if field in _SHAREX_FIELDS:
            groups[key] = "zoom_onset"
    for field, key in _ZOOM_END_PANEL_TO_AXIS.items():
        if field in _SHAREX_FIELDS:
            groups[key] = "zoom_trigger_end"
    return groups


def _legend_below(
    ax: Any,
    ncol: int = 1,
    handles: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
) -> Any:
    if ax is None:
        return None
    return place_legend_below(
        ax,
        _pdf_fonts(),
        ncol=ncol,
        handles=handles,
        labels=labels,
    )


def _attach_missing_legends(axes: dict[str, Any]) -> None:
    """Add a below-graph legend on plot axes that have labels but no legend yet."""
    for ax in axes.values():
        if ax is None or not ax.get_visible():
            continue
        if ax.get_legend() is not None:
            continue
        handles, labels = ax.get_legend_handles_labels()
        keep_handles: list[Any] = []
        keep_labels: list[str] = []
        for handle, lab in zip(handles, labels):
            if lab and lab != "_nolegend_":
                keep_handles.append(handle)
                keep_labels.append(lab)
        if not keep_labels:
            continue
        _legend_below(ax, handles=keep_handles, labels=keep_labels)


def _apply_compact_axis_fonts(fig: Any) -> None:
    """Apply subplot title, axis label, and tick font sizes on a figure."""
    for ax in fig.axes:
        title = ax.get_title()
        if title:
            if title.startswith("MEA"):
                ax.title.set_fontsize(MEA_TITLE_FONT_SIZE)
            else:
                ax.title.set_fontsize(AXIS_TITLE_FONT_SIZE)
        if ax.get_xlabel():
            ax.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
        if ax.get_ylabel():
            ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
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
    display: PlotDisplaySettings | None = None,
    zoom_mode: ZoomMode = "both",
    page_sections: Sequence[str] | None = None,
    n_legend_rows: int = 2,
    legend_labels: Sequence[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create page axes for one channel section using the stacked inch layout."""
    del first_row_height_ratio
    fonts = _pdf_fonts()
    display_settings = display or PlotDisplaySettings.all_on()
    sections = list(page_sections) if page_sections else _channel_page_sections(display_settings, zoom_mode)
    include_mea = bool(first_row_mea_channel_name) and display_settings.mea_layout
    labels = [str(lab) for lab in (legend_labels or []) if str(lab).strip()]
    axes_width_in = THREE_PART_PAGE_WIDTH_IN - 1.28 - 0.28
    rows_from_labels = estimate_legend_rows(labels, fonts, axes_width_in) if labels else 1
    legend_rows = max(int(n_legend_rows), rows_from_labels, 1)
    slots = _layout_slots_for_page(
        display_settings,
        zoom_mode,
        page_sections=sections,
        include_mea=include_mea,
        include_impedance=include_impedance_panel,
        zoom_onset_t0=zoom_t0,
        zoom_onset_t1=zoom_t1,
        n_legend_rows=legend_rows,
        probe_layout=probe_layout,
    )
    if not slots:
        slots = [Slot(key="ax_full", kind="plot", plot_height_in=_scaled_plot_height_in(2.0), has_legend=True)]

    pages = build_stacked_pages(
        slots,
        fonts=fonts,
        n_legend_rows=legend_rows,
        width_in=THREE_PART_PAGE_WIDTH_IN,
        sharex_groups=_sharex_groups_for_slots(),
        max_height_in=MAX_CHANNEL_PAGE_HEIGHT_IN,
    )
    if not pages:
        pages = build_stacked_pages(
            [Slot(key="ax_full", kind="plot", plot_height_in=_scaled_plot_height_in(2.0), has_legend=True)],
            fonts=fonts,
            n_legend_rows=legend_rows,
            width_in=THREE_PART_PAGE_WIDTH_IN,
            max_height_in=MAX_CHANNEL_PAGE_HEIGHT_IN,
        )
    axes: dict[str, Any] = {}
    figs: list[Any] = []
    for page in pages:
        axes.update(page.axes)
        figs.append(page.fig)
    if "ax_top" in axes:
        if first_row_mea_channel_name is not None:
            _draw_mea_layout_panel(axes["ax_top"], probe_layout, first_row_mea_channel_name)
        else:
            axes["ax_top"].axis("off")
            if first_row_text:
                axes["ax_top"].text(
                    0.02,
                    0.5,
                    first_row_text,
                    ha="left",
                    va="center",
                    fontsize=SECTION_HEADER_FONT_SIZE,
                    fontweight="bold",
                    transform=axes["ax_top"].transAxes,
                )
    return figs, axes


@_profiled("pdf_savefig_channel_page")
def _finalize_and_save_three_part_page(
    *,
    fig: Any,
    pdf: PdfPages,
    axes: dict[str, Any],
    n_recordings: int,
) -> None:
    del n_recordings
    for ax in axes.values():
        if ax is None or not ax.get_visible():
            continue
        if ax.axison and ax.get_xlabel():
            ax.tick_params(axis="x", labelbottom=True)
    _attach_missing_legends(axes)
    figures = fig if isinstance(fig, (list, tuple)) else (fig,)
    for one_fig in figures:
        if one_fig is None:
            continue
        save_figure_to_pdf(
            pdf,
            one_fig,
            dpi=PDF_DPI,
            soften_linewidths=_soften_figure_linewidths,
            apply_fonts=_apply_compact_axis_fonts,
        )

def _default_filter_short_label(kind: str = "highpass") -> str:
    pass_label = "high-pass" if kind == "highpass" else "low-pass"
    return f"bessel {pass_label} order 2 @ 250 Hz"


def _default_filter_title_label(kind: str = "highpass") -> str:
    pass_label = "high-pass" if kind == "highpass" else "low-pass"
    return f"bessel {pass_label} @ 250 Hz"

def _intan_filter_mean_captions(
    intan_dsp: IntanDspSettings | None,
    kind: str = "highpass",
) -> tuple[str, str]:
    """Return (title suffix, compact legend spec) for software-filtered mean traces."""
    if intan_dsp is None:
        short = _default_filter_short_label(kind)
        return f" — mean trace: {short}", short
    short = intan_dsp.filter_short_label(kind)  # type: ignore[arg-type]
    return f" — mean trace: {short}", short


def _intan_hp_mean_filter_captions(intan_dsp: IntanDspSettings | None) -> tuple[str, str]:
    return _intan_filter_mean_captions(intan_dsp, "highpass")


def _intan_lp_mean_filter_captions(intan_dsp: IntanDspSettings | None) -> tuple[str, str]:
    return _intan_filter_mean_captions(intan_dsp, "lowpass")


def _mean_n_samples_title_suffix(n_samples_per_recording: Sequence[int]) -> str:
    """Suffix for mean-trace titles: number of trials/sections used in the average."""
    counts = [int(n) for n in n_samples_per_recording]
    if not counts:
        return ""
    if len(counts) == 1 or len(set(counts)) == 1:
        return f" (n={counts[0]})"
    return f" (n={', '.join(str(n) for n in counts)})"


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


def _nth_trigger_window(
    source: AmplifierSpikeSource,
    ch: int,
    n_expected: int,
    *,
    trigger_index: int,
    stream: str = "amplifier",
) -> Optional[np.ndarray]:
    """Extract the Nth valid-stimulation window (raw / high-pass / low-pass)."""
    triggers = np.asarray(source.valid_triggers, dtype=np.int64)
    if triggers.size <= int(trigger_index):
        return None
    trig = int(triggers[int(trigger_index)])
    start = int(trig - source.pre_n)
    end = int(trig + source.post_n)
    if stream == "highpass":
        data = source.highpass
    elif stream == "lowpass":
        data = source.lowpass
    else:
        data = source.amplifier
    curve = np.asarray(data[ch, start:end], dtype=np.float64)
    if curve.shape[0] != n_expected:
        return None
    return curve


def _collect_nth_trigger_windows(
    spike_sources: Sequence[AmplifierSpikeSource],
    ch: int,
    n_expected: int,
    *,
    trigger_index: int,
) -> tuple[list[Optional[np.ndarray]], list[Optional[np.ndarray]], list[Optional[np.ndarray]]]:
    raw_curves: list[Optional[np.ndarray]] = []
    hp_curves: list[Optional[np.ndarray]] = []
    lp_curves: list[Optional[np.ndarray]] = []
    for src in spike_sources:
        raw_curves.append(
            _nth_trigger_window(
                src, ch, n_expected, trigger_index=trigger_index, stream="amplifier"
            )
        )
        hp_curves.append(
            _nth_trigger_window(
                src, ch, n_expected, trigger_index=trigger_index, stream="highpass"
            )
        )
        lp_curves.append(
            _nth_trigger_window(
                src, ch, n_expected, trigger_index=trigger_index, stream="lowpass"
            )
        )
    return raw_curves, hp_curves, lp_curves


def _trigger_end_zoom_bounds(
    end_markers: Sequence[float],
    zoom_t0: float,
    zoom_t1: float,
) -> tuple[float, float] | None:
    if not end_markers:
        return None
    return float(min(end_markers) + zoom_t0), float(max(end_markers) + zoom_t1)


def _mark_unavailable_axis(ax: Any, message: str) -> None:
    from plot_utils import mark_unavailable_axis

    mark_unavailable_axis(ax, message, fontsize=UNAVAILABLE_FONT_SIZE)


def _add_trace_reference_overlays(
    ax: Any,
    *,
    zoom_t0: float,
    zoom_t1: float,
    end_markers: Sequence[float],
    onset_label: str = "Stimulation (onset)",
    offset_label: str = "Stimulation (offset)",
    end_line_specs: Optional[Sequence[tuple[float, str]]] = None,
    show_zoom_span: bool = False,
    show_end_zoom_span: bool = False,
    end_zoom_t0: float | None = None,
    end_zoom_t1: float | None = None,
    aux_legend_label: Optional[str] = None,
) -> None:
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.15,
        color="red",
        label=onset_label if aux_legend_label is None else aux_legend_label,
    )
    if end_line_specs:
        for idx, (value, label) in enumerate(end_line_specs):
            ax.axvline(
                value,
                linestyle="-.",
                linewidth=1.15,
                color="#1d4ed8",
                label=(label if aux_legend_label is None else (aux_legend_label if idx == 0 else "_nolegend_")),
            )
    else:
        labeled = False
        for value in end_markers:
            if aux_legend_label is not None:
                leg = aux_legend_label if not labeled else "_nolegend_"
            else:
                leg = offset_label if not labeled else "_nolegend_"
            ax.axvline(
                float(value),
                linestyle="-.",
                linewidth=1.15,
                color="#1d4ed8",
                label=leg,
            )
            labeled = True
    if show_zoom_span:
        ax.axvspan(
            zoom_t0,
            zoom_t1,
            alpha=0.12,
            color="green",
            label=("Onset zoom region" if aux_legend_label is None else aux_legend_label),
        )
    if show_end_zoom_span:
        ez0 = float(end_zoom_t0 if end_zoom_t0 is not None else zoom_t0)
        ez1 = float(end_zoom_t1 if end_zoom_t1 is not None else zoom_t1)
        end_bounds = _trigger_end_zoom_bounds(end_markers, ez0, ez1)
        if end_bounds is not None:
            ax.axvspan(
                end_bounds[0],
                end_bounds[1],
                alpha=0.10,
                color="gold",
                label=(
                    "End zoom region"
                    if aux_legend_label is None
                    else aux_legend_label
                ),
            )

def _plot_stim_event_panel(
    ax: Any,
    *,
    curves: Sequence[Optional[np.ndarray]],
    t_plot: np.ndarray,
    slice_fn,
    labels: Sequence[str],
    colors: Sequence[Any],
    intan_hp_legends: Sequence[str],
    legend_visible: Sequence[bool] | None,
    title: str,
    legend_suffix: str,
    single_legend: str,
    unavailable_message: str,
    first_lw: float,
    overlay_kwargs: dict[str, Any],
    end_markers: Sequence[float],
    end_line_specs: Optional[Sequence[tuple[float, str]]],
    show_reference: bool,
    ylim: tuple[float, float] | None = None,
    filtered: bool = False,
) -> None:
    if ax is None:
        return
    if not any(curve is not None for curve in curves):
        _mark_unavailable_axis(ax, unavailable_message)
        return
    for i, curve in enumerate(curves):
        if curve is None:
            continue
        line_color = colors[i % len(colors)]
        show_leg = True if legend_visible is None else bool(legend_visible[i])
        if filtered:
            hp_legend = (
                intan_hp_legends[i]
                if i < len(intan_hp_legends)
                else _default_filter_short_label()
            )
            label = _legend_label(
                labels[i],
                f"{legend_suffix} ({hp_legend})",
                multi=len(labels) > 1,
                show_legend=show_leg,
            )
            if len(labels) <= 1 and show_leg:
                label = f"{single_legend} ({hp_legend})"
        else:
            label = _legend_label(
                labels[i],
                legend_suffix,
                multi=len(labels) > 1,
                show_legend=show_leg,
            )
            if len(labels) <= 1 and show_leg:
                label = single_legend
        y_plot = np.asarray(slice_fn(curve), dtype=np.float64)
        t_ds, y_ds = decimate_envelope(
            np.asarray(t_plot, dtype=np.float64), y_plot, _PDF_TRACE_MAX_POINTS
        )
        ax.plot(t_ds, y_ds, linewidth=first_lw, color=line_color, label=label)
    if show_reference:
        _add_trace_reference_overlays(ax, **overlay_kwargs)
    else:
        _draw_onset_offset_lines(
            ax,
            end_markers=end_markers,
            end_line_specs=end_line_specs,
        )
    ax.set_title(title)
    ax.set_ylabel("Potential (µV)")
    ax.set_xlabel(TIME_REL_XLABEL)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))


def _plot_mean_section_trace_panels(
    *,
    ax_raw: Any,
    ax_filt_hp: Any,
    ax_filt_lp: Any,
    ax_first_hp: Any,
    ax_first_lp: Any,
    ax_first_raw: Any,
    t_rel: np.ndarray,
    time_mask: Optional[np.ndarray],
    x_limits: Optional[tuple[float, float]],
    labels: Sequence[str],
    mean_hp: Sequence[np.ndarray],
    mean_lp: Sequence[np.ndarray],
    mean_raw: Optional[Sequence[np.ndarray]],
    first_trigger_raw: Sequence[Optional[np.ndarray]],
    first_trigger_hp: Sequence[Optional[np.ndarray]],
    first_trigger_lp: Sequence[Optional[np.ndarray]],
    colors: Sequence[Any],
    zoom_t0: float,
    zoom_t1: float,
    end_markers: Sequence[float],
    intan_hp_legends: Sequence[str],
    intan_lp_legends: Sequence[str],
    title_raw: str,
    title_filt_hp: str,
    title_filt_lp: str,
    title_first_hp: str,
    title_first_lp: str,
    title_first_raw: str,
    legend_cols: int,
    legend_visible: Sequence[bool] | None = None,
    aux_legend_label: Optional[str] = None,
    end_line_specs: Optional[Sequence[tuple[float, str]]] = None,
    show_reference_on_first_raw: bool = True,
    show_reference_on_first_hp: bool = False,
    show_reference_on_first_lp: bool = False,
    show_zoom_span: bool = False,
    show_end_zoom_span: bool = False,
    end_zoom_t0: float | None = None,
    end_zoom_t1: float | None = None,
    first_trigger_hp_ylim: tuple[float, float] | None = None,
    ax_second_hp: Any = None,
    ax_second_lp: Any = None,
    ax_second_raw: Any = None,
    second_trigger_raw: Sequence[Optional[np.ndarray]] | None = None,
    second_trigger_hp: Sequence[Optional[np.ndarray]] | None = None,
    second_trigger_lp: Sequence[Optional[np.ndarray]] | None = None,
    title_second_hp: str = "Second stimulation — high-pass",
    title_second_lp: str = "Second stimulation — low-pass",
    title_second_raw: str = "Second stimulation — raw",
    show_reference_on_second_raw: bool = True,
    show_reference_on_second_hp: bool = False,
    show_reference_on_second_lp: bool = False,
    base_lw: float = 1.2,
    main_lw: float = 1.35,
    first_lw: float = 1.1,
) -> None:
    """Plot mean-trace and per-stimulation panels (raw + separate HP/LP)."""
    del legend_cols  # Reserved for callers that still pass layout ncols.
    if time_mask is not None:
        t_plot = t_rel[time_mask]
    else:
        t_plot = t_rel

    def _slice(y: np.ndarray) -> np.ndarray:
        return y[time_mask] if time_mask is not None else y

    overlay_kwargs = dict(
        zoom_t0=zoom_t0,
        zoom_t1=zoom_t1,
        end_markers=end_markers,
        end_line_specs=end_line_specs,
        show_zoom_span=show_zoom_span,
        show_end_zoom_span=show_end_zoom_span,
        end_zoom_t0=end_zoom_t0,
        end_zoom_t1=end_zoom_t1,
        aux_legend_label=aux_legend_label,
    )

    n_series = max(len(mean_hp), len(mean_lp), len(mean_raw or []))
    for i in range(n_series):
        line_color = colors[i % len(colors)]
        y_hp = np.asarray(mean_hp[i], dtype=np.float64) if i < len(mean_hp) else None
        y_lp = np.asarray(mean_lp[i], dtype=np.float64) if i < len(mean_lp) else None
        y_raw = (
            np.asarray(mean_raw[i], dtype=np.float64)
            if mean_raw is not None and i < len(mean_raw)
            else (y_hp if y_hp is not None else y_lp)
        )
        show_leg = True if legend_visible is None else bool(legend_visible[i])
        multi = len(labels) > 1
        label_i = labels[i] if i < len(labels) else f"Recording {i + 1}"
        raw_label = _legend_label(label_i, "raw trial-averaged", multi=multi, show_legend=show_leg)
        if ax_raw is not None and y_raw is not None:
            t_ds, y_ds = decimate_envelope(
                np.asarray(t_plot, dtype=np.float64),
                np.asarray(_slice(y_raw), dtype=np.float64),
                _PDF_TRACE_MAX_POINTS,
            )
            ax_raw.plot(
                t_ds,
                y_ds,
                linewidth=base_lw,
                color=line_color,
                label=raw_label,
            )
        hp_legend = (
            intan_hp_legends[i]
            if i < len(intan_hp_legends)
            else _default_filter_short_label("highpass")
        )
        lp_legend = (
            intan_lp_legends[i]
            if i < len(intan_lp_legends)
            else _default_filter_short_label("lowpass")
        )
        if ax_filt_hp is not None and y_hp is not None:
            hp_label = _legend_label(
                label_i, f"high-pass trial-averaged ({hp_legend})", multi=multi, show_legend=show_leg
            )
            if not multi and show_leg:
                hp_label = f"High-pass trial-averaged ({hp_legend})"
            t_ds, y_ds = decimate_envelope(
                np.asarray(t_plot, dtype=np.float64),
                np.asarray(_slice(y_hp), dtype=np.float64),
                _PDF_TRACE_MAX_POINTS,
            )
            ax_filt_hp.plot(
                t_ds, y_ds, linewidth=main_lw, color=line_color, label=hp_label
            )
        if ax_filt_lp is not None and y_lp is not None:
            lp_label = _legend_label(
                label_i, f"low-pass trial-averaged ({lp_legend})", multi=multi, show_legend=show_leg
            )
            if not multi and show_leg:
                lp_label = f"Low-pass trial-averaged ({lp_legend})"
            t_ds, y_ds = decimate_envelope(
                np.asarray(t_plot, dtype=np.float64),
                np.asarray(_slice(y_lp), dtype=np.float64),
                _PDF_TRACE_MAX_POINTS,
            )
            ax_filt_lp.plot(
                t_ds, y_ds, linewidth=main_lw, color=line_color, label=lp_label
            )

    if ax_raw is not None:
        _add_trace_reference_overlays(ax_raw, **overlay_kwargs)
        ax_raw.set_title(title_raw)
        ax_raw.set_ylabel("Potential (µV)")
        ax_raw.set_xlabel(TIME_REL_XLABEL)
        ax_raw.grid(True, alpha=0.3)

    for ax, title in ((ax_filt_hp, title_filt_hp), (ax_filt_lp, title_filt_lp)):
        if ax is not None:
            _add_trace_reference_overlays(ax, **overlay_kwargs)
            ax.set_title(title)
            ax.set_ylabel("Potential (µV)")
            ax.set_xlabel(TIME_REL_XLABEL)
            ax.grid(True, alpha=0.3)

    _plot_stim_event_panel(
        ax_first_raw,
        curves=first_trigger_raw,
        t_plot=t_plot,
        slice_fn=_slice,
        labels=labels,
        colors=colors,
        intan_hp_legends=intan_hp_legends,
        legend_visible=legend_visible,
        title=title_first_raw,
        legend_suffix="first stimulation raw",
        single_legend="First stimulation raw",
        unavailable_message="First stimulation raw signal unavailable",
        first_lw=first_lw,
        overlay_kwargs=overlay_kwargs,
        end_markers=end_markers,
        end_line_specs=end_line_specs,
        show_reference=show_reference_on_first_raw,
    )
    _plot_stim_event_panel(
        ax_first_hp,
        curves=first_trigger_hp,
        t_plot=t_plot,
        slice_fn=_slice,
        labels=labels,
        colors=colors,
        intan_hp_legends=intan_hp_legends,
        legend_visible=legend_visible,
        title=title_first_hp,
        legend_suffix="first stimulation HP",
        single_legend="First stimulation (high-pass)",
        unavailable_message="First stimulation high-pass signal unavailable",
        first_lw=first_lw,
        overlay_kwargs=overlay_kwargs,
        end_markers=end_markers,
        end_line_specs=end_line_specs,
        show_reference=show_reference_on_first_hp,
        ylim=first_trigger_hp_ylim,
        filtered=True,
    )
    _plot_stim_event_panel(
        ax_first_lp,
        curves=first_trigger_lp,
        t_plot=t_plot,
        slice_fn=_slice,
        labels=labels,
        colors=colors,
        intan_hp_legends=intan_lp_legends,
        legend_visible=legend_visible,
        title=title_first_lp,
        legend_suffix="first stimulation LP",
        single_legend="First stimulation (low-pass)",
        unavailable_message="First stimulation low-pass signal unavailable",
        first_lw=first_lw,
        overlay_kwargs=overlay_kwargs,
        end_markers=end_markers,
        end_line_specs=end_line_specs,
        show_reference=show_reference_on_first_lp,
        filtered=True,
    )
    second_raw = second_trigger_raw or []
    second_hp = second_trigger_hp or []
    second_lp = second_trigger_lp or []
    _plot_stim_event_panel(
        ax_second_raw,
        curves=second_raw,
        t_plot=t_plot,
        slice_fn=_slice,
        labels=labels,
        colors=colors,
        intan_hp_legends=intan_hp_legends,
        legend_visible=legend_visible,
        title=title_second_raw,
        legend_suffix="second stimulation raw",
        single_legend="Second stimulation raw",
        unavailable_message="Second stimulation raw signal unavailable",
        first_lw=first_lw,
        overlay_kwargs=overlay_kwargs,
        end_markers=end_markers,
        end_line_specs=end_line_specs,
        show_reference=show_reference_on_second_raw,
    )
    _plot_stim_event_panel(
        ax_second_hp,
        curves=second_hp,
        t_plot=t_plot,
        slice_fn=_slice,
        labels=labels,
        colors=colors,
        intan_hp_legends=intan_hp_legends,
        legend_visible=legend_visible,
        title=title_second_hp,
        legend_suffix="second stimulation HP",
        single_legend="Second stimulation (high-pass)",
        unavailable_message="Second stimulation high-pass signal unavailable",
        first_lw=first_lw,
        overlay_kwargs=overlay_kwargs,
        end_markers=end_markers,
        end_line_specs=end_line_specs,
        show_reference=show_reference_on_second_hp,
        ylim=first_trigger_hp_ylim,
        filtered=True,
    )
    _plot_stim_event_panel(
        ax_second_lp,
        curves=second_lp,
        t_plot=t_plot,
        slice_fn=_slice,
        labels=labels,
        colors=colors,
        intan_hp_legends=intan_lp_legends,
        legend_visible=legend_visible,
        title=title_second_lp,
        legend_suffix="second stimulation LP",
        single_legend="Second stimulation (low-pass)",
        unavailable_message="Second stimulation low-pass signal unavailable",
        first_lw=first_lw,
        overlay_kwargs=overlay_kwargs,
        end_markers=end_markers,
        end_line_specs=end_line_specs,
        show_reference=show_reference_on_second_lp,
        filtered=True,
    )

    if x_limits is not None:
        for ax in (
            ax_raw,
            ax_filt_hp,
            ax_filt_lp,
            ax_first_hp,
            ax_first_lp,
            ax_first_raw,
            ax_second_hp,
            ax_second_lp,
            ax_second_raw,
        ):
            if ax is not None:
                ax.set_xlim(float(x_limits[0]), float(x_limits[1]))

# Re-exports — implementations live in channel_metrics (no matplotlib).
from channel_metrics import (  # noqa: E402
    mean_rms_profile_from_source_window as _mean_rms_profile_from_source_window,
    resolve_channel_spike_threshold as _resolve_channel_spike_threshold,
    spike_threshold_caption as _spike_threshold_caption,
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
        self._mean_lp: dict[tuple[int, int], np.ndarray] = {}
        self._rms: dict[
            tuple[int, int | None, int | None], tuple[np.ndarray, np.ndarray]
        ] = {}
        self._spikes: dict[
            tuple[int, int, float, tuple[float, float] | None, int | None],
            list[np.ndarray],
        ] = {}
        self._waveforms: dict[
            tuple[int, int, float, tuple[float, float] | None, int | None, float, float],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
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
        *,
        need_raw: bool = True,
        need_hp: bool = True,
        need_lp: bool = True,
    ) -> None:
        """Pre-compute channel means in parallel for the streams that are displayed."""
        if not (need_raw or need_hp or need_lp):
            return
        for src_idx, src in enumerate(spike_sources):
            n_ch = min(int(n_channels), int(src.amplifier.shape[0]))
            if n_ch <= 0:
                continue
            workers = resolve_channel_workers(channel_workers, n_ch)
            raw_all = hp_all = lp_all = None
            if need_raw:
                check_analysis_cancelled()
                raw_all = mean_triggered_windows_channelwise(
                    src.amplifier,
                    src.valid_triggers,
                    self.pre_n,
                    self.post_n,
                    channel_workers=workers,
                )
            if need_hp:
                check_analysis_cancelled()
                hp_all = mean_triggered_windows_channelwise(
                    src.highpass,
                    src.valid_triggers,
                    self.pre_n,
                    self.post_n,
                    channel_workers=workers,
                )
            if need_lp:
                check_analysis_cancelled()
                lp_all = mean_triggered_windows_channelwise(
                    src.lowpass,
                    src.valid_triggers,
                    self.pre_n,
                    self.post_n,
                    channel_workers=workers,
                )
            for ch in range(n_ch):
                if raw_all is not None:
                    self._mean_raw[(src_idx, ch)] = self._normalize_mean_row(
                        raw_all[ch], self.n_expected
                    )
                if hp_all is not None:
                    self._mean_hp[(src_idx, ch)] = self._normalize_mean_row(
                        hp_all[ch], self.n_expected
                    )
                if lp_all is not None:
                    self._mean_lp[(src_idx, ch)] = self._normalize_mean_row(
                        lp_all[ch], self.n_expected
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

    def mean_lp(self, src_idx: int, source: AmplifierSpikeSource, ch: int) -> np.ndarray:
        key = (src_idx, ch)
        cached = self._mean_lp.get(key)
        if cached is not None:
            return cached
        mean = _mean_triggered_average_row(
            source._low_row(ch),
            source.valid_triggers,
            self.pre_n,
            self.post_n,
            self.n_expected,
        )
        self._mean_lp[key] = mean
        return mean

    def rms_profile(
        self,
        src_idx: int,
        source: AmplifierSpikeSource,
        channel_index: int | None = None,
        *,
        n_channels: int | None = None,
        trigger_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if channel_index is None:
            if n_channels is None:
                n_channels = int(source.highpass.shape[0])
            return self.rms_profile_recording_mean(
                src_idx, int(n_channels), trigger_index=trigger_index
            )
        key = (src_idx, channel_index, trigger_index)
        cached = self._rms.get(key)
        if cached is not None:
            return cached
        profile = _mean_rms_profile_from_source_window(
            source,
            self.t0_rms,
            self.t1_rms,
            self.rms_window_s,
            channel_index=channel_index,
            trigger_index=trigger_index,
        )
        self._rms[key] = profile
        return profile

    def rms_profile_recording_mean(
        self,
        src_idx: int,
        n_channels: int,
        *,
        trigger_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean RMS across channels from already-cached per-channel profiles."""
        key = (src_idx, None, trigger_index)
        cached = self._rms.get(key)
        if cached is not None:
            return cached
        tx_ref: np.ndarray | None = None
        acc: np.ndarray | None = None
        n_ok = 0
        for ch in range(int(n_channels)):
            per_ch = self._rms.get((src_idx, ch, trigger_index))
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

    @staticmethod
    def _t_range_key(
        t_range_s: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if t_range_s is None:
            return None
        t0, t1 = float(t_range_s[0]), float(t_range_s[1])
        return (round(t0, 9), round(t1, 9))

    def spike_times_per_trial(
        self,
        src_idx: int,
        source: AmplifierSpikeSource,
        ch: int,
        threshold_uv: float,
        t_range_s: tuple[float, float] | None = None,
        trigger_index: int | None = None,
    ) -> list[np.ndarray]:
        key = (
            src_idx,
            ch,
            float(threshold_uv),
            self._t_range_key(t_range_s),
            trigger_index,
        )
        cached = self._spikes.get(key)
        if cached is not None:
            return cached
        spikes = source.spike_times_per_trial_for_channel(
            ch,
            self.t_rel,
            threshold_uv,
            t_range_s=t_range_s,
            trigger_index=trigger_index,
        )
        self._spikes[key] = spikes
        return spikes

    def spike_waveforms(
        self,
        src_idx: int,
        source: AmplifierSpikeSource,
        ch: int,
        threshold_uv: float,
        t_range_s: tuple[float, float] | None = None,
        trigger_index: int | None = None,
        pre_ms: float = SPIKE_OVERLAY_DEFAULT_PRE_MS,
        post_ms: float = SPIKE_OVERLAY_DEFAULT_POST_MS,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (
            src_idx,
            ch,
            float(threshold_uv),
            self._t_range_key(t_range_s),
            trigger_index,
            float(pre_ms),
            float(post_ms),
        )
        cached = self._waveforms.get(key)
        if cached is not None:
            return cached
        times = self.spike_times_per_trial(
            src_idx,
            source,
            ch,
            threshold_uv,
            t_range_s=t_range_s,
            trigger_index=trigger_index,
        )
        extracted = _extract_spike_waveforms(
            source, ch, times, pre_ms=pre_ms, post_ms=post_ms
        )
        self._waveforms[key] = extracted
        return extracted

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
            fontsize=UNAVAILABLE_FONT_SIZE,
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
        ax.tick_params(axis="x", labelsize=TICK_LABEL_FONT_SIZE)
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
                    fontsize=ANNOTATION_FONT_SIZE,
                    alpha=0.85,
                )

    save_figure_to_pdf(
        pdf,
        fig,
        dpi=PDF_DPI,
        soften_linewidths=_soften_figure_linewidths,
        apply_fonts=_apply_compact_axis_fonts,
    )
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
    ylabel: str = "RMS (µV)",
    colors: Sequence[Any] | None = None,
) -> None:
    """Plot RMS profile (time in window) for one or many recordings."""
    if ax is None:
        return
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    palette = list(colors) if colors is not None else list(default_colors)
    if not palette:
        palette = list(default_colors)
    has_data = False
    for i, (label, tx, values) in enumerate(rms_series):
        if values.size == 0 or tx.size == 0:
            continue
        has_data = True
        ax.plot(
            tx,
            values,
            linewidth=1.35,
            color=palette[i % len(palette)],
            label=label,
        )
    if has_data:
        ax.set_title(title)
        ax.set_xlabel(TIME_REL_XLABEL)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 20.0)
        if x_limits is not None:
            ax.set_xlim(float(x_limits[0]), float(x_limits[1]))
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "RMS evolution unavailable\n(no valid stimulation window)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=UNAVAILABLE_FONT_SIZE,
        )
        ax.set_axis_off()


def _append_mean_rms_evolution_page(
    pdf: PdfPages,
    rms_series: Sequence[tuple[str, np.ndarray, np.ndarray]],
    rms_window_s: float,
    *,
    filter_title: str | None = None,
    colors: Sequence[Any] | None = None,
) -> None:
    """Append one summary page: mean RMS profile on analysis timebase."""
    title_filter = filter_title or _default_filter_title_label()
    check_analysis_cancelled()
    fig, ax = plt.subplots(figsize=(SUMMARY_PAGE_WIDTH_IN, SUMMARY_PAGE_HEIGHT_IN))
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    palette = list(colors) if colors is not None else list(default_colors)
    if not palette:
        palette = list(default_colors)
    has_data = False
    for i, (label, tx, values) in enumerate(rms_series):
        if values.size == 0 or tx.size == 0:
            continue
        has_data = True
        ax.plot(
            tx,
            values,
            linewidth=1.35,
            color=palette[i % len(palette)],
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
            "RMS evolution unavailable\n(no valid stimulation window)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=UNAVAILABLE_FONT_SIZE,
        )
        ax.set_axis_off()
    save_figure_to_pdf(
        pdf,
        fig,
        dpi=PDF_DPI,
        soften_linewidths=_soften_figure_linewidths,
        apply_fonts=_apply_compact_axis_fonts,
    )


_RMS_TABLE_ROWS_PER_PAGE = 36

_MONTAGE_CHANNELS_PER_PAGE = 12
_MONTAGE_LP_COLOR = "#1e40af"
_MONTAGE_HP_COLOR = "#15803d"
_MONTAGE_STIM3_MARKER_COLOR = "#c2410c"
_MONTAGE_CHANNEL_HEIGHT_IN = 0.72
_SECOND_STIM_TRIGGER_INDEX = 1
_THIRD_STIM_TRIGGER_INDEX = 2


def _montage_amplitude_uv(curve: np.ndarray) -> float:
    """Peak-to-peak amplitude used to bin similar magnitude scales."""
    vals = np.asarray(curve, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.ptp(vals))


def _montage_magnitude_bin(amplitude_uv: float) -> int | None:
    """Order-of-magnitude bin (log10 decade). Same bin → shared Y scale."""
    if not math.isfinite(amplitude_uv) or amplitude_uv <= 0.0:
        return None
    return int(math.floor(math.log10(amplitude_uv)))


def _montage_padded_ylim(y_lo: float, y_hi: float) -> tuple[float, float]:
    if np.isclose(y_lo, y_hi):
        pad = max(abs(y_lo) * 0.05, 1.0)
        return (y_lo - pad, y_hi + pad)
    pad = (y_hi - y_lo) * 0.05
    return (y_lo - pad, y_hi + pad)


def _montage_accumulate_ylim(
    ylim_by_bin: dict[int, tuple[float, float]],
    curve: np.ndarray,
) -> int | None:
    vals = np.asarray(curve, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    mag_bin = _montage_magnitude_bin(_montage_amplitude_uv(finite))
    if mag_bin is None or finite.size == 0:
        return mag_bin
    y_lo = float(np.min(finite))
    y_hi = float(np.max(finite))
    if mag_bin in ylim_by_bin:
        prev_lo, prev_hi = ylim_by_bin[mag_bin]
        ylim_by_bin[mag_bin] = (min(prev_lo, y_lo), max(prev_hi, y_hi))
    else:
        ylim_by_bin[mag_bin] = (y_lo, y_hi)
    return mag_bin


def _montage_draw_reference_lines(
    ax: Any,
    *,
    end_marker: float | None,
    zoom_onset_t0: float,
    zoom_onset_t1: float,
    show_zoom_span: bool,
) -> None:
    ax.axvline(0.0, linestyle="--", linewidth=0.9, color="red", zorder=1)
    if end_marker is not None:
        ax.axvline(
            end_marker,
            linestyle="-.",
            linewidth=0.9,
            color="#1d4ed8",
            zorder=1,
        )
    if show_zoom_span:
        ax.axvline(
            float(zoom_onset_t0),
            linestyle=":",
            linewidth=0.85,
            color="#60a5fa",
            zorder=1,
        )
        ax.axvline(
            float(zoom_onset_t1),
            linestyle=":",
            linewidth=0.85,
            color="#60a5fa",
            zorder=1,
        )


def _append_single_curve_stream_montage_pages(
    pdf: PdfPages,
    *,
    curves: Sequence[Optional[np.ndarray]],
    channel_names: Sequence[str],
    t_plot: np.ndarray,
    page_title: str,
    unavailable_row_message: str,
    unavailable_page_message: str,
    line_color: str,
    show_rec_in_title: bool,
    sampling_percent: int,
    end_marker: float | None,
    zoom_onset_t0: float,
    zoom_onset_t1: float,
    show_zoom_span: bool,
) -> None:
    """Stack one curve per channel (shared Y scale within magnitude bins)."""
    n_channels = len(curves)
    if n_channels < 1:
        return
    ylim_by_bin: dict[int, tuple[float, float]] = {}
    bin_by_channel: list[int | None] = []
    for curve in curves:
        if curve is None:
            bin_by_channel.append(None)
            continue
        bin_by_channel.append(_montage_accumulate_ylim(ylim_by_bin, curve))

    shared_ylim_by_bin = {
        mag_bin: _montage_padded_ylim(y_lo, y_hi)
        for mag_bin, (y_lo, y_hi) in ylim_by_bin.items()
    }

    channels_per_page = max(1, int(_MONTAGE_CHANNELS_PER_PAGE))
    n_ch_pages = max(1, int(math.ceil(n_channels / float(channels_per_page))))
    for page_i in range(n_ch_pages):
        check_analysis_cancelled()
        ch0 = page_i * channels_per_page
        ch1 = min(n_channels, ch0 + channels_per_page)
        page_channels = list(range(ch0, ch1))
        n_rows = len(page_channels)
        fig_h = max(
            4.5,
            0.55 + n_rows * (_MONTAGE_CHANNEL_HEIGHT_IN + 0.08) + 0.55,
        )
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(SUMMARY_PAGE_WIDTH_IN, fig_h),
            sharex=True,
            squeeze=False,
        )
        axes_flat = list(axes[:, 0])
        page_suffix = f" ({page_i + 1}/{n_ch_pages})" if n_ch_pages > 1 else ""
        fig.suptitle(
            f"{page_title}{page_suffix}",
            fontsize=AXIS_TITLE_FONT_SIZE,
            y=0.995,
        )

        any_curve = False
        for row_i, ch in enumerate(page_channels):
            ax = axes_flat[row_i]
            curve = curves[ch]
            ch_name = str(channel_names[ch]) if ch < len(channel_names) else f"Ch {ch}"
            if curve is None:
                ax.text(
                    0.5,
                    0.5,
                    unavailable_row_message,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=max(8.0, UNAVAILABLE_FONT_SIZE - 4),
                    color="0.45",
                )
            else:
                any_curve = True
                tx, yy = downsample_points(
                    t_plot, np.asarray(curve, dtype=np.float64), sampling_percent
                )
                ax.plot(tx, yy, color=line_color, linewidth=1.15, zorder=2)
                mag_bin = bin_by_channel[ch]
                if mag_bin is not None and mag_bin in shared_ylim_by_bin:
                    y0, y1 = shared_ylim_by_bin[mag_bin]
                    ax.set_ylim(y0, y1)
            _montage_draw_reference_lines(
                ax,
                end_marker=end_marker,
                zoom_onset_t0=zoom_onset_t0,
                zoom_onset_t1=zoom_onset_t1,
                show_zoom_span=show_zoom_span,
            )
            ax.set_ylabel(
                ch_name,
                fontsize=max(8.0, TICK_LABEL_FONT_SIZE - 3),
                rotation=0,
                ha="right",
                va="center",
                labelpad=10,
            )
            ax.tick_params(axis="y", labelsize=max(7.0, TICK_LABEL_FONT_SIZE - 5))
            ax.tick_params(axis="x", labelbottom=(row_i == n_rows - 1))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_i < n_rows - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", bottom=False)

        if not any_curve:
            axes_flat[0].text(
                0.5,
                0.5,
                unavailable_page_message,
                ha="center",
                va="center",
                transform=axes_flat[0].transAxes,
                fontsize=UNAVAILABLE_FONT_SIZE,
            )
        axes_flat[-1].set_xlabel(TIME_REL_XLABEL)
        axes_flat[-1].set_xlim(float(t_plot[0]), float(t_plot[-1]))
        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            top=0.90 if n_ch_pages > 1 or show_rec_in_title else 0.88,
            bottom=0.08,
            hspace=0.08,
        )
        save_figure_to_pdf(
            pdf,
            fig,
            dpi=PDF_DPI,
            soften_linewidths=_soften_figure_linewidths,
            apply_fonts=_apply_compact_axis_fonts,
        )


def _append_second_stim_stream_montage_pages(
    pdf: PdfPages,
    *,
    source: AmplifierSpikeSource,
    channel_names: Sequence[str],
    n_channels: int,
    t_plot: np.ndarray,
    n_expected: int,
    stream: str,
    stream_title: str,
    filter_short: str,
    line_color: str,
    rec_label: str,
    show_rec_in_title: bool,
    sampling_percent: int,
    end_marker: float | None,
    zoom_onset_t0: float,
    zoom_onset_t1: float,
    show_zoom_span: bool,
) -> None:
    """One filter stream: all channels stacked for the second stimulation."""
    curves: list[Optional[np.ndarray]] = []
    for ch in range(n_channels):
        check_analysis_cancelled()
        curves.append(
            _nth_trigger_window(
                source,
                ch,
                n_expected,
                trigger_index=_SECOND_STIM_TRIGGER_INDEX,
                stream=stream,
            )
        )
    rec_suffix = f" — {rec_label}" if show_rec_in_title else ""
    _append_single_curve_stream_montage_pages(
        pdf,
        curves=curves,
        channel_names=channel_names,
        t_plot=t_plot,
        page_title=(
            f"All-channels montage — 2nd stimulation — {stream_title}"
            f" ({filter_short}){rec_suffix}"
        ),
        unavailable_row_message="2nd stim. unavailable",
        unavailable_page_message=(
            "Second stimulation unavailable\n(need at least 2 valid stimulations)"
        ),
        line_color=line_color,
        show_rec_in_title=show_rec_in_title,
        sampling_percent=sampling_percent,
        end_marker=end_marker,
        zoom_onset_t0=zoom_onset_t0,
        zoom_onset_t1=zoom_onset_t1,
        show_zoom_span=show_zoom_span,
    )


def _append_mean_stim_stream_montage_pages(
    pdf: PdfPages,
    *,
    source: AmplifierSpikeSource,
    src_idx: int,
    channel_names: Sequence[str],
    n_channels: int,
    t_plot: np.ndarray,
    n_expected: int,
    stream: str,
    stream_title: str,
    filter_short: str,
    line_color: str,
    rec_label: str,
    show_rec_in_title: bool,
    sampling_percent: int,
    end_marker: float | None,
    zoom_onset_t0: float,
    zoom_onset_t1: float,
    show_zoom_span: bool,
    render_cache: Optional[Any] = None,
) -> None:
    """One filter stream: all channels stacked for trial-averaged stimulations."""
    triggers = np.asarray(source.valid_triggers, dtype=np.int64)
    n_stim = int(triggers.size)
    curves: list[Optional[np.ndarray]] = []
    for ch in range(n_channels):
        check_analysis_cancelled()
        if n_stim < 1:
            curves.append(None)
            continue
        if render_cache is not None:
            if stream == "highpass":
                curve = render_cache.mean_hp(src_idx, source, ch)
            elif stream == "lowpass":
                curve = render_cache.mean_lp(src_idx, source, ch)
            else:
                curve = render_cache.mean_raw(src_idx, source, ch)
        else:
            if stream == "highpass":
                row = source._high_row(ch)
            elif stream == "lowpass":
                row = source._low_row(ch)
            else:
                row = source._amp_row(ch)
            curve = _mean_triggered_average_row(
                row,
                triggers,
                int(source.pre_n),
                int(source.post_n),
                n_expected,
            )
        curves.append(np.asarray(curve, dtype=np.float64))

    rec_suffix = f" — {rec_label}" if show_rec_in_title else ""
    n_suffix = f", n={n_stim}" if n_stim > 0 else ""
    _append_single_curve_stream_montage_pages(
        pdf,
        curves=curves,
        channel_names=channel_names,
        t_plot=t_plot,
        page_title=(
            f"All-channels montage — trial-averaged — {stream_title}"
            f" ({filter_short}{n_suffix}){rec_suffix}"
        ),
        unavailable_row_message="Mean unavailable",
        unavailable_page_message=(
            "Trial-averaged montage unavailable\n(no valid stimulations)"
        ),
        line_color=line_color,
        show_rec_in_title=show_rec_in_title,
        sampling_percent=sampling_percent,
        end_marker=end_marker,
        zoom_onset_t0=zoom_onset_t0,
        zoom_onset_t1=zoom_onset_t1,
        show_zoom_span=show_zoom_span,
    )


def _append_second_to_third_stim_span_lp_montage_pages(
    pdf: PdfPages,
    *,
    source: AmplifierSpikeSource,
    channel_names: Sequence[str],
    n_channels: int,
    filter_short: str,
    rec_label: str,
    show_rec_in_title: bool,
    sampling_percent: int,
    end_marker: float | None,
) -> None:
    """Continuous low-pass montage from 2nd stim window start through 3rd stim window end."""
    triggers = np.asarray(source.valid_triggers, dtype=np.int64)
    if triggers.size <= _THIRD_STIM_TRIGGER_INDEX:
        curves: list[Optional[np.ndarray]] = [None] * n_channels
        t_plot = np.asarray([0.0, 1.0], dtype=np.float64)
        third_onset_s: float | None = None
        unavailable = True
    else:
        unavailable = False
        trig2 = int(triggers[_SECOND_STIM_TRIGGER_INDEX])
        trig3 = int(triggers[_THIRD_STIM_TRIGGER_INDEX])
        start = int(trig2 - source.pre_n)
        end = int(trig3 + source.post_n)
        n_samples = int(source.lowpass.shape[1])
        if start < 0 or end > n_samples or end <= start:
            curves = [None] * n_channels
            t_plot = np.asarray([0.0, 1.0], dtype=np.float64)
            third_onset_s = None
            unavailable = True
        else:
            fs = float(source.fs)
            t_plot = (np.arange(start, end, dtype=np.float64) - float(trig2)) / fs
            third_onset_s = float(trig3 - trig2) / fs
            curves = []
            for ch in range(n_channels):
                check_analysis_cancelled()
                curves.append(
                    np.asarray(source.lowpass[ch, start:end], dtype=np.float64)
                )

    rec_suffix = f" — {rec_label}" if show_rec_in_title else ""
    page_title = (
        f"All-channels montage — 2nd→3rd stim span — Low-pass"
        f" ({filter_short}){rec_suffix}"
    )

    ylim_by_bin: dict[int, tuple[float, float]] = {}
    bin_by_channel: list[int | None] = []
    for curve in curves:
        if curve is None:
            bin_by_channel.append(None)
            continue
        bin_by_channel.append(_montage_accumulate_ylim(ylim_by_bin, curve))
    shared_ylim_by_bin = {
        mag_bin: _montage_padded_ylim(y_lo, y_hi)
        for mag_bin, (y_lo, y_hi) in ylim_by_bin.items()
    }

    channels_per_page = max(1, int(_MONTAGE_CHANNELS_PER_PAGE))
    n_ch_pages = max(1, int(math.ceil(n_channels / float(channels_per_page))))
    for page_i in range(n_ch_pages):
        check_analysis_cancelled()
        ch0 = page_i * channels_per_page
        ch1 = min(n_channels, ch0 + channels_per_page)
        page_channels = list(range(ch0, ch1))
        n_rows = len(page_channels)
        fig_h = max(
            4.5,
            0.55 + n_rows * (_MONTAGE_CHANNEL_HEIGHT_IN + 0.08) + 0.70,
        )
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(SUMMARY_PAGE_WIDTH_IN, fig_h),
            sharex=True,
            squeeze=False,
        )
        axes_flat = list(axes[:, 0])
        page_suffix = f" ({page_i + 1}/{n_ch_pages})" if n_ch_pages > 1 else ""
        fig.suptitle(
            f"{page_title}{page_suffix}",
            fontsize=AXIS_TITLE_FONT_SIZE,
            y=0.995,
        )

        any_curve = False
        for row_i, ch in enumerate(page_channels):
            ax = axes_flat[row_i]
            curve = curves[ch]
            ch_name = str(channel_names[ch]) if ch < len(channel_names) else f"Ch {ch}"
            if curve is None:
                ax.text(
                    0.5,
                    0.5,
                    "2nd→3rd span unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=max(8.0, UNAVAILABLE_FONT_SIZE - 4),
                    color="0.45",
                )
            else:
                any_curve = True
                tx, yy = downsample_points(
                    t_plot, np.asarray(curve, dtype=np.float64), sampling_percent
                )
                ax.plot(tx, yy, color=_MONTAGE_LP_COLOR, linewidth=1.15, zorder=2)
                mag_bin = bin_by_channel[ch]
                if mag_bin is not None and mag_bin in shared_ylim_by_bin:
                    y0, y1 = shared_ylim_by_bin[mag_bin]
                    ax.set_ylim(y0, y1)

            # t=0 = 2nd stim onset; mark 3rd stim onset on the continuous axis.
            ax.axvline(
                0.0,
                linestyle="--",
                linewidth=0.9,
                color="red",
                zorder=1,
                label="2nd stim." if row_i == 0 else "_nolegend_",
            )
            if third_onset_s is not None:
                ax.axvline(
                    third_onset_s,
                    linestyle="--",
                    linewidth=0.9,
                    color=_MONTAGE_STIM3_MARKER_COLOR,
                    zorder=1,
                    label="3rd stim." if row_i == 0 else "_nolegend_",
                )
            if end_marker is not None:
                ax.axvline(
                    float(end_marker),
                    linestyle="-.",
                    linewidth=0.9,
                    color="#1d4ed8",
                    zorder=1,
                )
                if third_onset_s is not None:
                    ax.axvline(
                        float(third_onset_s) + float(end_marker),
                        linestyle="-.",
                        linewidth=0.9,
                        color="#1d4ed8",
                        zorder=1,
                    )

            ax.set_ylabel(
                ch_name,
                fontsize=max(8.0, TICK_LABEL_FONT_SIZE - 3),
                rotation=0,
                ha="right",
                va="center",
                labelpad=10,
            )
            ax.tick_params(axis="y", labelsize=max(7.0, TICK_LABEL_FONT_SIZE - 5))
            ax.tick_params(axis="x", labelbottom=(row_i == n_rows - 1))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_i < n_rows - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", bottom=False)

        if any_curve and not unavailable:
            axes_flat[0].legend(
                loc="upper right",
                fontsize=max(8.0, LEGEND_FONT_SIZE - 4),
                framealpha=0.9,
            )
        elif unavailable or not any_curve:
            axes_flat[0].text(
                0.5,
                0.5,
                "2nd→3rd stim span unavailable\n(need at least 3 valid stimulations)",
                ha="center",
                va="center",
                transform=axes_flat[0].transAxes,
                fontsize=UNAVAILABLE_FONT_SIZE,
            )
        axes_flat[-1].set_xlabel("Time relative to 2nd stimulation (s)")
        axes_flat[-1].set_xlim(float(t_plot[0]), float(t_plot[-1]))
        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            top=0.88 if n_ch_pages > 1 or show_rec_in_title else 0.86,
            bottom=0.08,
            hspace=0.08,
        )
        save_figure_to_pdf(
            pdf,
            fig,
            dpi=PDF_DPI,
            soften_linewidths=_soften_figure_linewidths,
            apply_fonts=_apply_compact_axis_fonts,
        )


def _append_second_stim_channel_montage_pages(
    pdf: PdfPages,
    *,
    spike_sources: Sequence[AmplifierSpikeSource],
    plot_indices: Sequence[int],
    labels: Sequence[str],
    channel_names: Sequence[str],
    t_rel: np.ndarray,
    sampling_percent: int = 100,
    trigger_end_rising_rel_s_list: Optional[Sequence[Optional[float]]] = None,
    zoom_onset_t0: float = ZOOM_T0,
    zoom_onset_t1: float = ZOOM_T1,
    show_zoom_span: bool = False,
    filter_short_lp: str | None = None,
    filter_short_hp: str | None = None,
    render_cache: Optional[Any] = None,
) -> None:
    """Summary: trial-averaged LP/HP, 2nd stim LP/HP, then continuous 2nd→3rd LP span."""
    n_expected = int(t_rel.shape[0])
    if n_expected < 1 or not plot_indices:
        return
    n_channels = min(len(channel_names), min(src.amplifier.shape[0] for src in spike_sources))
    if n_channels < 1:
        return
    lp_label = filter_short_lp or _default_filter_short_label("lowpass")
    hp_label = filter_short_hp or _default_filter_short_label("highpass")
    t_plot = np.asarray(t_rel, dtype=np.float64)
    show_rec = len(plot_indices) > 1

    for src_idx in plot_indices:
        check_analysis_cancelled()
        source = spike_sources[int(src_idx)]
        rec_label = (
            labels[int(src_idx)]
            if int(src_idx) < len(labels)
            else f"Recording {int(src_idx) + 1}"
        )
        end_marker: float | None = None
        if trigger_end_rising_rel_s_list is not None and int(src_idx) < len(
            trigger_end_rising_rel_s_list
        ):
            raw_marker = trigger_end_rising_rel_s_list[int(src_idx)]
            if raw_marker is not None:
                end_marker = float(raw_marker)

        common = dict(
            source=source,
            channel_names=channel_names,
            n_channels=n_channels,
            t_plot=t_plot,
            n_expected=n_expected,
            rec_label=rec_label,
            show_rec_in_title=show_rec,
            sampling_percent=sampling_percent,
            end_marker=end_marker,
            zoom_onset_t0=zoom_onset_t0,
            zoom_onset_t1=zoom_onset_t1,
            show_zoom_span=show_zoom_span,
        )
        _append_mean_stim_stream_montage_pages(
            pdf,
            src_idx=int(src_idx),
            stream="lowpass",
            stream_title="Low-pass",
            filter_short=lp_label,
            line_color=_MONTAGE_LP_COLOR,
            render_cache=render_cache,
            **common,
        )
        _append_mean_stim_stream_montage_pages(
            pdf,
            src_idx=int(src_idx),
            stream="highpass",
            stream_title="High-pass",
            filter_short=hp_label,
            line_color=_MONTAGE_HP_COLOR,
            render_cache=render_cache,
            **common,
        )
        _append_second_stim_stream_montage_pages(
            pdf,
            stream="lowpass",
            stream_title="Low-pass",
            filter_short=lp_label,
            line_color=_MONTAGE_LP_COLOR,
            **common,
        )
        _append_second_stim_stream_montage_pages(
            pdf,
            stream="highpass",
            stream_title="High-pass",
            filter_short=hp_label,
            line_color=_MONTAGE_HP_COLOR,
            **common,
        )
        _append_second_to_third_stim_span_lp_montage_pages(
            pdf,
            source=source,
            channel_names=channel_names,
            n_channels=n_channels,
            filter_short=lp_label,
            rec_label=rec_label,
            show_rec_in_title=show_rec,
            sampling_percent=sampling_percent,
            end_marker=end_marker,
        )


def _append_mean_rms_per_channel_table_page(
    pdf: PdfPages,
    channel_names: Sequence[str],
    rms_by_recording: Sequence[tuple[str, Sequence[float]]],
    *,
    filter_title: str | None = None,
    rms_window_s: float = 1.0,
) -> None:
    """Append summary page(s): mean RMS (µV) per channel, one column per recording."""
    if not channel_names or not rms_by_recording:
        return
    title_filter = filter_title or _default_filter_title_label()
    n_ch = len(channel_names)

    def _short_label(text: str, max_len: int = 28) -> str:
        s = str(text).strip() or "Recording"
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    col_labels = ["Channel"] + [
        f"{_short_label(label)} (µV)" for label, _ in rms_by_recording
    ]
    n_pages = max(1, int(math.ceil(n_ch / float(_RMS_TABLE_ROWS_PER_PAGE))))
    for page_i in range(n_pages):
        check_analysis_cancelled()
        start = page_i * _RMS_TABLE_ROWS_PER_PAGE
        end = min(n_ch, start + _RMS_TABLE_ROWS_PER_PAGE)
        cell_text: list[list[str]] = []
        for ch in range(start, end):
            row = [str(channel_names[ch])]
            for _, values in rms_by_recording:
                if ch < len(values):
                    val = float(values[ch])
                    row.append(f"{val:.2f}" if math.isfinite(val) else "—")
                else:
                    row.append("—")
            cell_text.append(row)

        fig, ax = plt.subplots(figsize=(SUMMARY_PAGE_WIDTH_IN, SUMMARY_PAGE_HEIGHT_IN))
        ax.set_axis_off()
        page_suffix = f" ({page_i + 1}/{n_pages})" if n_pages > 1 else ""
        ax.set_title(
            f"Mean RMS per channel{page_suffix}\n"
            f"({title_filter}, last {rms_window_s:g} s of recording)",
            fontsize=AXIS_LABEL_FONT_SIZE + 2,
            pad=12,
        )
        if not cell_text:
            ax.text(
                0.5,
                0.5,
                "RMS per-channel table unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=UNAVAILABLE_FONT_SIZE,
            )
        else:
            tbl = ax.table(
                cellText=cell_text,
                colLabels=col_labels,
                cellLoc="center",
                colLoc="center",
                loc="upper center",
                bbox=[0.02, 0.02, 0.96, 0.88],
            )
            tbl.auto_set_font_size(False)
            n_rows_page = len(cell_text)
            font_size = max(7.0, min(12.0, 320.0 / max(n_rows_page + 1, 1)))
            tbl.set_fontsize(font_size)
            for (row, col), cell in tbl.get_celld().items():
                cell.set_edgecolor("#b0b0b0")
                cell.set_linewidth(0.4)
                props: dict[str, Any] = {}
                if row == 0:
                    cell.set_facecolor("#e8e8e8")
                    props["weight"] = "bold"
                elif row % 2 == 0:
                    cell.set_facecolor("#f7f7f7")
                if col == 0:
                    props["ha"] = "left"
                    cell.PAD = 0.02
                if props:
                    cell.set_text_props(**props)

        save_figure_to_pdf(
            pdf,
            fig,
            dpi=PDF_DPI,
            soften_linewidths=_soften_figure_linewidths,
            apply_fonts=_apply_compact_axis_fonts,
        )


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
    spike_overlay_pre_ms: float = SPIKE_OVERLAY_DEFAULT_PRE_MS,
    spike_overlay_post_ms: float = SPIKE_OVERLAY_DEFAULT_POST_MS,
    psth_bin_window_s: float = 0.025,
    rms_window_s: float = 0.050,
    zoom_mode: ZoomMode = "both",
    zoom_onset_t0_s: float = ZOOM_T0,
    zoom_onset_t1_s: float = ZOOM_T1,
    zoom_end_t0_s: float = ZOOM_T0,
    zoom_end_t1_s: float = ZOOM_T1,
    sampling_percent: int = 100,
    probe_layout_json: Optional[Path] = None,
    impedance_sessions: Optional[Sequence[ImpedanceSession]] = None,
    channel_workers: int | None = None,
    first_trigger_hp_ylim_enabled: bool = False,
    first_trigger_hp_ylim_min_uv: float = -200.0,
    first_trigger_hp_ylim_max_uv: float = 200.0,
    plot_display: PlotDisplaySettings | None = None,
    recording_styles: Sequence[RecordingStyle] | None = None,
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
    elif n_records == 1 and labels:
        safe_title = "".join(
            c if c.isalnum() or c in "._- " else "_" for c in str(labels[0]).strip()
        )
        pdf_stem = safe_title or "analysis"
    elif n_records >= 2 and labels:
        safe_title = "".join(
            c if c.isalnum() or c in "._- " else "_"
            for c in f"{labels[0]}_vs_{n_records - 1}_others"
        )
        pdf_stem = safe_title or "multi_comparison"
    else:
        pdf_stem = "multi_comparison"
    pdf_name = shorten_filename_for_windows(output_dir, f"{pdf_stem}.pdf")
    pdf_path = output_dir / pdf_name

    display = plot_display or PlotDisplaySettings.all_on()
    styles: list[RecordingStyle] = (
        list(recording_styles)
        if recording_styles is not None
        else [RecordingStyle.visible()] * n_records
    )
    if len(styles) < n_records:
        styles.extend([RecordingStyle.visible()] * (n_records - len(styles)))

    plot_indices = [i for i in range(n_records) if styles[i].plot_visible]
    if not plot_indices:
        plot_indices = list(range(n_records))
    legend_flags = [styles[i].legend_visible for i in plot_indices]

    zoom_onset_t0 = float(zoom_onset_t0_s)
    zoom_onset_t1 = float(zoom_onset_t1_s)
    zoom_end_t0 = float(zoom_end_t0_s)
    zoom_end_t1 = float(zoom_end_t1_s)
    first_trigger_hp_ylim: tuple[float, float] | None = None
    if first_trigger_hp_ylim_enabled:
        y_lo = float(first_trigger_hp_ylim_min_uv)
        y_hi = float(first_trigger_hp_ylim_max_uv)
        if y_hi <= y_lo:
            raise ValueError(
                "First-stimulation HP y-axis: maximum (µV) must be strictly greater than minimum."
            )
        first_trigger_hp_ylim = (y_lo, y_hi)
    main_intan_dsp = spike_sources[0].intan_dsp if spike_sources else None
    filter_title_hp = (
        main_intan_dsp.filter_title_label("highpass")
        if main_intan_dsp is not None
        else _default_filter_title_label("highpass")
    )
    filter_short_hp = (
        main_intan_dsp.filter_short_label("highpass")
        if main_intan_dsp is not None
        else _default_filter_short_label("highpass")
    )
    filter_title_lp = (
        main_intan_dsp.filter_title_label("lowpass")
        if main_intan_dsp is not None
        else _default_filter_title_label("lowpass")
    )
    filter_short_lp = (
        main_intan_dsp.filter_short_label("lowpass")
        if main_intan_dsp is not None
        else _default_filter_short_label("lowpass")
    )
    # Spikes / RMS stay on high-pass (Intan Spike Scope).
    filter_title = filter_title_hp
    filter_short = filter_short_hp
    rms_note = f"window {rms_window_s:g} s"
    n_channels = min(src.amplifier.shape[0] for src in spike_sources)
    end_markers = [v for v in (trigger_end_rising_rel_s_list or []) if v is not None]
    _has_spike_cmp = (
        fs is not None
        and len(spike_sources) == n_records
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    plot_colors = resolve_recording_plot_colors(
        styles,
        plot_indices,
        fallback=colors,
    )

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
    need_raw, need_hp, need_lp, need_rms, need_spikes = _display_stream_needs(
        display, zoom_mode
    )
    render_cache.prefill_means(
        spike_sources,
        n_channels,
        channel_workers=channel_workers,
        need_raw=need_raw,
        need_hp=need_hp,
        need_lp=need_lp,
    )
    if need_rms or need_spikes:
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
            means_ch_hp: list[np.ndarray] = []
            means_ch_lp: list[np.ndarray] = []
            means_raw_ch: list[np.ndarray] = []
            intan_hp_legends: list[str] = []
            intan_lp_legends: list[str] = []
            for src_idx in plot_indices:
                src = spike_sources[src_idx]
                if need_raw:
                    means_raw_ch.append(render_cache.mean_raw(src_idx, src, ch))
                if need_hp:
                    means_ch_hp.append(render_cache.mean_hp(src_idx, src, ch))
                if need_lp:
                    means_ch_lp.append(render_cache.mean_lp(src_idx, src, ch))
                _hp_note, hp_legend = _intan_hp_mean_filter_captions(src.intan_dsp)
                _lp_note, lp_legend = _intan_lp_mean_filter_captions(src.intan_dsp)
                intan_hp_legends.append(hp_legend)
                intan_lp_legends.append(lp_legend)

            figs, _axes = _build_three_part_page_axes(
                zoom_t0=zoom_onset_t0,
                zoom_t1=zoom_onset_t1,
                n_recordings=len(plot_indices),
                first_row_height_ratio=2.2,
                first_row_text=None,
                first_row_mea_channel_name=channel_name if display.mea_layout else None,
                probe_layout=probe_layout_loaded if display.mea_layout else None,
                include_impedance_panel=bool(impedance_sessions) and display.impedance,
                display=display,
                zoom_mode=zoom_mode,
                n_legend_rows=max(1, sum(1 for flag in legend_flags if flag)) + 3,
                legend_labels=[
                    labels[i] if i < len(labels) else f"Recording {i + 1}"
                    for i, flag in zip(plot_indices, legend_flags)
                    if flag
                ],
            )
            ax_full = _axes.get("ax_full")
            ax_full_filt_hp = _axes.get("ax_full_filt_hp")
            ax_full_filt_lp = _axes.get("ax_full_filt_lp")
            ax_first_trigger_hp = _axes.get("ax_first_trigger_hp")
            ax_first_trigger_lp = _axes.get("ax_first_trigger_lp")
            ax_first_trigger = _axes.get("ax_first_trigger")
            ax_second_trigger_hp = _axes.get("ax_second_trigger_hp")
            ax_second_trigger_lp = _axes.get("ax_second_trigger_lp")
            ax_second_trigger = _axes.get("ax_second_trigger")
            ax_full_rms = _axes.get("ax_full_rms")
            ax_raster_f = _axes.get("ax_raster_f")
            ax_fr_f = _axes.get("ax_fr_f")
            ax_trial_fr_f = _axes.get("ax_trial_fr_f")
            ax_isi_f = _axes.get("ax_isi_f")
            ax_zoom = _axes.get("ax_zoom")
            ax_zoom_filt_hp = _axes.get("ax_zoom_filt_hp")
            ax_zoom_filt_lp = _axes.get("ax_zoom_filt_lp")
            ax_zoom_first_hp = _axes.get("ax_zoom_first_hp")
            ax_zoom_first_lp = _axes.get("ax_zoom_first_lp")
            ax_zoom_first = _axes.get("ax_zoom_first")
            ax_zoom_second_hp = _axes.get("ax_zoom_second_hp")
            ax_zoom_second_lp = _axes.get("ax_zoom_second_lp")
            ax_zoom_second = _axes.get("ax_zoom_second")
            ax_zoom_rms = _axes.get("ax_zoom_rms")
            ax_raster_z = _axes.get("ax_raster_z")
            ax_fr_z = _axes.get("ax_fr_z")
            ax_trial_fr_z = _axes.get("ax_trial_fr_z")
            ax_isi_z = _axes.get("ax_isi_z")
            ax_zoom_end = _axes.get("ax_zoom_end")
            ax_zoom_end_filt_hp = _axes.get("ax_zoom_end_filt_hp")
            ax_zoom_end_filt_lp = _axes.get("ax_zoom_end_filt_lp")
            ax_zoom_end_first_hp = _axes.get("ax_zoom_end_first_hp")
            ax_zoom_end_first_lp = _axes.get("ax_zoom_end_first_lp")
            ax_zoom_end_first = _axes.get("ax_zoom_end_first")
            ax_zoom_end_second_hp = _axes.get("ax_zoom_end_second_hp")
            ax_zoom_end_second_lp = _axes.get("ax_zoom_end_second_lp")
            ax_zoom_end_second = _axes.get("ax_zoom_end_second")
            ax_zoom_end_rms = _axes.get("ax_zoom_end_rms")
            ax_raster_ze = _axes.get("ax_raster_ze")
            ax_fr_ze = _axes.get("ax_fr_ze")
            ax_trial_fr_ze = _axes.get("ax_trial_fr_ze")
            ax_isi_ze = _axes.get("ax_isi_ze")

            if impedance_sessions and "ax_imp" in _axes:
                _draw_impedance_evolution_panel(
                    _axes["ax_imp"], channel_name, impedance_sessions
                )

            show_onset_span = _section_included(zoom_mode, "zoom_onset")
            show_end_span = _section_included(zoom_mode, "zoom_trigger_end")
            legend_cols = 1

            def _collect_rms_series(
                trigger_index: int | None,
            ) -> tuple[
                list[tuple[str, np.ndarray, np.ndarray]],
                list[tuple[str, np.ndarray, np.ndarray]],
                list[tuple[str, np.ndarray, np.ndarray]],
            ]:
                full_series: list[tuple[str, np.ndarray, np.ndarray]] = []
                zoom_series: list[tuple[str, np.ndarray, np.ndarray]] = []
                end_series: list[tuple[str, np.ndarray, np.ndarray]] = []
                for i in plot_indices:
                    src = spike_sources[i]
                    label = labels[i] if i < len(labels) else f"Recording {i + 1}"
                    tx_full, rms_full_vals = render_cache.rms_profile(
                        i, src, channel_index=ch, trigger_index=trigger_index
                    )
                    full_series.append((label, tx_full, rms_full_vals))
                    tx_zoom, rms_zoom_vals = _slice_rms_profile_window(
                        tx_full,
                        rms_full_vals,
                        zoom_onset_t0,
                        zoom_onset_t1,
                    )
                    zoom_series.append((label, tx_zoom, rms_zoom_vals))
                    marker_i = None
                    if (
                        trigger_end_rising_rel_s_list is not None
                        and i < len(trigger_end_rising_rel_s_list)
                    ):
                        marker_i = trigger_end_rising_rel_s_list[i]
                    if marker_i is None:
                        end_series.append(
                            (label, np.array([], dtype=np.float64), np.array([], dtype=np.float64))
                        )
                    else:
                        tx_end, rms_zoom_end_vals = _slice_rms_profile_window(
                            tx_full,
                            rms_full_vals,
                            float(marker_i + zoom_end_t0),
                            float(marker_i + zoom_end_t1),
                        )
                        end_series.append((label, tx_end, rms_zoom_end_vals))
                return full_series, zoom_series, end_series

            empty_rms: list[tuple[str, np.ndarray, np.ndarray]] = []
            active_panels = _active_section_panels(display, zoom_mode)
            need_mean_rms = any(s.rms for s in active_panels)
            need_first_rms = any(s.first_rms for s in active_panels)
            need_second_rms = any(s.second_rms for s in active_panels)
            need_first_trig = any(
                s.first_trigger_raw or s.first_trigger_hp or s.first_trigger_lp
                for s in active_panels
            )
            need_second_trig = any(
                s.second_trigger_raw or s.second_trigger_hp or s.second_trigger_lp
                for s in active_panels
            )
            if need_mean_rms:
                rms_series_full_multi, rms_series_zoom_multi, rms_series_zoom_end_multi = (
                    _collect_rms_series(None)
                )
            else:
                rms_series_full_multi = rms_series_zoom_multi = rms_series_zoom_end_multi = (
                    empty_rms
                )
            if need_first_rms:
                rms_first_full, rms_first_zoom, rms_first_end = _collect_rms_series(0)
            else:
                rms_first_full = rms_first_zoom = rms_first_end = empty_rms
            if need_second_rms:
                rms_second_full, rms_second_zoom, rms_second_end = _collect_rms_series(1)
            else:
                rms_second_full = rms_second_zoom = rms_second_end = empty_rms
            if need_first_trig:
                first_trigger_raw_all, first_trigger_hp_all, first_trigger_lp_all = (
                    _collect_nth_trigger_windows(
                        spike_sources,
                        ch,
                        int(t_rel.shape[0]),
                        trigger_index=0,
                    )
                )
                first_trigger_raw = [first_trigger_raw_all[i] for i in plot_indices]
                first_trigger_hp = [first_trigger_hp_all[i] for i in plot_indices]
                first_trigger_lp = [first_trigger_lp_all[i] for i in plot_indices]
            else:
                first_trigger_raw = first_trigger_hp = first_trigger_lp = [
                    None for _ in plot_indices
                ]
            if need_second_trig:
                second_trigger_raw_all, second_trigger_hp_all, second_trigger_lp_all = (
                    _collect_nth_trigger_windows(
                        spike_sources,
                        ch,
                        int(t_rel.shape[0]),
                        trigger_index=1,
                    )
                )
                second_trigger_raw = [second_trigger_raw_all[i] for i in plot_indices]
                second_trigger_hp = [second_trigger_hp_all[i] for i in plot_indices]
                second_trigger_lp = [second_trigger_lp_all[i] for i in plot_indices]
            else:
                second_trigger_raw = second_trigger_hp = second_trigger_lp = [
                    None for _ in plot_indices
                ]
            record_labels = [
                labels[i] if i < len(labels) else f"Recording {i + 1}"
                for i in plot_indices
            ]
            mean_n_suffix = _mean_n_samples_title_suffix(
                [int(spike_sources[i].valid_triggers.size) for i in plot_indices]
            )
            full_panels = display.full_view
            onset_panels = display.zoom_onset
            end_panels = display.zoom_trigger_end

            if _section_included(zoom_mode, "full") and full_panels.any_enabled():
                if _trace_panels_enabled(full_panels):
                    _plot_mean_section_trace_panels(
                    ax_raw=ax_full,
                    ax_filt_hp=ax_full_filt_hp,
                    ax_filt_lp=ax_full_filt_lp,
                    ax_first_hp=ax_first_trigger_hp,
                    ax_first_lp=ax_first_trigger_lp,
                    ax_first_raw=ax_first_trigger,
                    ax_second_hp=ax_second_trigger_hp,
                    ax_second_lp=ax_second_trigger_lp,
                    ax_second_raw=ax_second_trigger,
                    t_rel=t_rel,
                    time_mask=None,
                    x_limits=(float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None,
                    labels=record_labels,
                    mean_hp=means_ch_hp,
                    mean_lp=means_ch_lp,
                    mean_raw=means_raw_ch,
                    first_trigger_raw=first_trigger_raw,
                    first_trigger_hp=first_trigger_hp,
                    first_trigger_lp=first_trigger_lp,
                    second_trigger_raw=second_trigger_raw,
                    second_trigger_hp=second_trigger_hp,
                    second_trigger_lp=second_trigger_lp,
                    colors=plot_colors,
                    zoom_t0=zoom_onset_t0,
                    zoom_t1=zoom_onset_t1,
                    end_markers=end_markers,
                    intan_hp_legends=intan_hp_legends,
                    intan_lp_legends=intan_lp_legends,
                    title_raw=f"{channel_name} — Raw trial-averaged (full view){mean_n_suffix}",
                    title_filt_hp=f"{channel_name} — High-pass trial-averaged ({filter_short_hp}){mean_n_suffix}",
                    title_filt_lp=f"{channel_name} — Low-pass trial-averaged ({filter_short_lp}){mean_n_suffix}",
                    title_first_hp=f"First stimulation — high-pass ({filter_title_hp})",
                    title_first_lp=f"First stimulation — low-pass ({filter_title_lp})",
                    title_first_raw="First stimulation — raw",
                    title_second_hp=f"Second stimulation — high-pass ({filter_title_hp})",
                    title_second_lp=f"Second stimulation — low-pass ({filter_title_lp})",
                    title_second_raw="Second stimulation — raw",
                    legend_cols=legend_cols,
                    legend_visible=legend_flags,
                    first_trigger_hp_ylim=first_trigger_hp_ylim,
                    show_zoom_span=show_onset_span,
                    show_end_zoom_span=show_end_span,
                    end_zoom_t0=zoom_end_t0,
                    end_zoom_t1=zoom_end_t1,
                    )
                if full_panels.rms:
                    _plot_rms_series(
                        ax_full_rms,
                        rms_series_full_multi,
                        f"RMS (trial-averaged) — {filter_short}, {rms_note}",
                        x_limits=(float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None,
                        ylabel="Trial-averaged RMS (µV)",
                        colors=plot_colors,
                    )
                if full_panels.first_rms:
                    _plot_rms_series(
                        _axes.get("ax_full_rms_first"),
                        rms_first_full,
                        f"RMS (first stimulation) — {filter_short}, {rms_note}",
                        x_limits=(float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None,
                        ylabel="RMS (µV)",
                        colors=plot_colors,
                    )
                if full_panels.second_rms:
                    _plot_rms_series(
                        _axes.get("ax_full_rms_second"),
                        rms_second_full,
                        f"RMS (second stimulation) — {filter_short}, {rms_note}",
                        x_limits=(float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None,
                        ylabel="RMS (µV)",
                        colors=plot_colors,
                    )

            zmask = (t_rel >= zoom_onset_t0) & (t_rel <= zoom_onset_t1)
            if _section_included(zoom_mode, "zoom_onset") and onset_panels.any_enabled():
                if _trace_panels_enabled(onset_panels):
                    _plot_mean_section_trace_panels(
                    ax_raw=ax_zoom,
                    ax_filt_hp=ax_zoom_filt_hp,
                    ax_filt_lp=ax_zoom_filt_lp,
                    ax_first_hp=ax_zoom_first_hp,
                    ax_first_lp=ax_zoom_first_lp,
                    ax_first_raw=ax_zoom_first,
                    ax_second_hp=ax_zoom_second_hp,
                    ax_second_lp=ax_zoom_second_lp,
                    ax_second_raw=ax_zoom_second,
                    t_rel=t_rel,
                    time_mask=zmask,
                    x_limits=(zoom_onset_t0, zoom_onset_t1),
                    labels=record_labels,
                    mean_hp=means_ch_hp,
                    mean_lp=means_ch_lp,
                    mean_raw=means_raw_ch,
                    first_trigger_raw=first_trigger_raw,
                    first_trigger_hp=first_trigger_hp,
                    first_trigger_lp=first_trigger_lp,
                    second_trigger_raw=second_trigger_raw,
                    second_trigger_hp=second_trigger_hp,
                    second_trigger_lp=second_trigger_lp,
                    colors=plot_colors,
                    zoom_t0=zoom_onset_t0,
                    zoom_t1=zoom_onset_t1,
                    end_markers=end_markers,
                    intan_hp_legends=intan_hp_legends,
                    intan_lp_legends=intan_lp_legends,
                    title_raw=f"{channel_name} — Raw trial-averaged (onset zoom [{zoom_onset_t0:g}, {zoom_onset_t1:g}] s){mean_n_suffix}",
                    title_filt_hp=f"{channel_name} — High-pass trial-averaged — onset zoom ({filter_short_hp}){mean_n_suffix}",
                    title_filt_lp=f"{channel_name} — Low-pass trial-averaged — onset zoom ({filter_short_lp}){mean_n_suffix}",
                    title_first_hp=f"First stimulation — high-pass, onset zoom ({filter_title_hp})",
                    title_first_lp=f"First stimulation — low-pass, onset zoom ({filter_title_lp})",
                    title_first_raw="First stimulation — raw (onset zoom)",
                    title_second_hp=f"Second stimulation — high-pass, onset zoom ({filter_title_hp})",
                    title_second_lp=f"Second stimulation — low-pass, onset zoom ({filter_title_lp})",
                    title_second_raw="Second stimulation — raw (onset zoom)",
                    legend_cols=legend_cols,
                    legend_visible=legend_flags,
                    show_reference_on_first_raw=False,
                    show_reference_on_first_hp=False,
                    show_reference_on_first_lp=False,
                    show_reference_on_second_raw=False,
                    show_reference_on_second_hp=False,
                    show_reference_on_second_lp=False,
                    first_trigger_hp_ylim=first_trigger_hp_ylim,
                    show_zoom_span=False,
                    show_end_zoom_span=False,
                    )
                if onset_panels.rms:
                    _plot_rms_series(
                        ax_zoom_rms,
                        rms_series_zoom_multi,
                        f"RMS (trial-averaged) — onset zoom, {filter_short}, {rms_note}",
                        x_limits=(zoom_onset_t0, zoom_onset_t1),
                        ylabel="Trial-averaged RMS (µV)",
                        colors=plot_colors,
                    )
                if onset_panels.first_rms:
                    _plot_rms_series(
                        _axes.get("ax_zoom_rms_first"),
                        rms_first_zoom,
                        f"RMS (first stimulation) — onset zoom, {filter_short}, {rms_note}",
                        x_limits=(zoom_onset_t0, zoom_onset_t1),
                        ylabel="RMS (µV)",
                        colors=plot_colors,
                    )
                if onset_panels.second_rms:
                    _plot_rms_series(
                        _axes.get("ax_zoom_rms_second"),
                        rms_second_zoom,
                        f"RMS (second stimulation) — onset zoom, {filter_short}, {rms_note}",
                        x_limits=(zoom_onset_t0, zoom_onset_t1),
                        ylabel="RMS (µV)",
                        colors=plot_colors,
                    )

            end_zoom_range: tuple[float, float] | None = None
            if end_markers and _section_included(zoom_mode, "zoom_trigger_end"):
                end_zoom_t0 = float(min(end_markers) + zoom_end_t0)
                end_zoom_t1 = float(max(end_markers) + zoom_end_t1)
                end_zoom_range = (end_zoom_t0, end_zoom_t1)
                end_mask = (t_rel >= end_zoom_t0) & (t_rel <= end_zoom_t1)
                if end_panels.any_enabled():
                    if _trace_panels_enabled(end_panels):
                        _plot_mean_section_trace_panels(
                        ax_raw=ax_zoom_end,
                        ax_filt_hp=ax_zoom_end_filt_hp,
                        ax_filt_lp=ax_zoom_end_filt_lp,
                        ax_first_hp=ax_zoom_end_first_hp,
                        ax_first_lp=ax_zoom_end_first_lp,
                        ax_first_raw=ax_zoom_end_first,
                        ax_second_hp=ax_zoom_end_second_hp,
                        ax_second_lp=ax_zoom_end_second_lp,
                        ax_second_raw=ax_zoom_end_second,
                        t_rel=t_rel,
                        time_mask=end_mask,
                        x_limits=(end_zoom_t0, end_zoom_t1),
                        labels=record_labels,
                        mean_hp=means_ch_hp,
                        mean_lp=means_ch_lp,
                        mean_raw=means_raw_ch,
                        first_trigger_raw=first_trigger_raw,
                        first_trigger_hp=first_trigger_hp,
                        first_trigger_lp=first_trigger_lp,
                        second_trigger_raw=second_trigger_raw,
                        second_trigger_hp=second_trigger_hp,
                        second_trigger_lp=second_trigger_lp,
                        colors=plot_colors,
                        zoom_t0=zoom_end_t0,
                        zoom_t1=zoom_end_t1,
                        end_markers=end_markers,
                        intan_hp_legends=intan_hp_legends,
                        intan_lp_legends=intan_lp_legends,
                        title_raw=f"{channel_name} — Raw trial-averaged (end zoom [{end_zoom_t0:.2f}, {end_zoom_t1:.2f}] s){mean_n_suffix}",
                        title_filt_hp=f"{channel_name} — High-pass trial-averaged — end zoom ({filter_short_hp}){mean_n_suffix}",
                        title_filt_lp=f"{channel_name} — Low-pass trial-averaged — end zoom ({filter_short_lp}){mean_n_suffix}",
                        title_first_hp=f"First stimulation — high-pass, end zoom ({filter_title_hp})",
                        title_first_lp=f"First stimulation — low-pass, end zoom ({filter_title_lp})",
                        title_first_raw="First stimulation — raw (end zoom)",
                        title_second_hp=f"Second stimulation — high-pass, end zoom ({filter_title_hp})",
                        title_second_lp=f"Second stimulation — low-pass, end zoom ({filter_title_lp})",
                        title_second_raw="Second stimulation — raw (end zoom)",
                        legend_cols=legend_cols,
                        legend_visible=legend_flags,
                        show_reference_on_first_raw=False,
                        show_reference_on_first_hp=False,
                        show_reference_on_first_lp=False,
                        show_reference_on_second_raw=False,
                        show_reference_on_second_hp=False,
                        show_reference_on_second_lp=False,
                        first_trigger_hp_ylim=first_trigger_hp_ylim,
                        show_zoom_span=False,
                        show_end_zoom_span=False,
                        )
                    if end_panels.rms:
                        _plot_rms_series(
                            ax_zoom_end_rms,
                            rms_series_zoom_end_multi,
                            f"RMS (trial-averaged) — end zoom, {filter_short}, {rms_note}",
                            x_limits=(end_zoom_t0, end_zoom_t1),
                            ylabel="Trial-averaged RMS (µV)",
                            colors=plot_colors,
                        )
                    if end_panels.first_rms:
                        _plot_rms_series(
                            _axes.get("ax_zoom_end_rms_first"),
                            rms_first_end,
                            f"RMS (first stimulation) — end zoom, {filter_short}, {rms_note}",
                            x_limits=(end_zoom_t0, end_zoom_t1),
                            ylabel="RMS (µV)",
                            colors=plot_colors,
                        )
                    if end_panels.second_rms:
                        _plot_rms_series(
                            _axes.get("ax_zoom_end_rms_second"),
                            rms_second_end,
                            f"RMS (second stimulation) — end zoom, {filter_short}, {rms_note}",
                            x_limits=(end_zoom_t0, end_zoom_t1),
                            ylabel="RMS (µV)",
                            colors=plot_colors,
                        )
            elif _section_included(zoom_mode, "zoom_trigger_end"):
                for ax, msg in (
                    (ax_zoom_end, "End zoom unavailable\n(no rising edge after stimulation)"),
                    (ax_zoom_end_filt_hp, "End zoom unavailable\n(no rising edge after stimulation)"),
                    (ax_zoom_end_filt_lp, "End zoom unavailable\n(no rising edge after stimulation)"),
                    (ax_zoom_end_first_hp, "First high-pass stimulation unavailable"),
                    (ax_zoom_end_first_lp, "First low-pass stimulation unavailable"),
                    (ax_zoom_end_first, "First raw stimulation unavailable"),
                    (ax_zoom_end_second_hp, "Second high-pass stimulation unavailable"),
                    (ax_zoom_end_second_lp, "Second low-pass stimulation unavailable"),
                    (ax_zoom_end_second, "Second raw stimulation unavailable"),
                    (ax_zoom_end_rms, "RMS unavailable"),
                    (_axes.get("ax_zoom_end_rms_first"), "RMS (first stim) unavailable"),
                    (_axes.get("ax_zoom_end_rms_second"), "RMS (second stim) unavailable"),
                    (
                        _axes.get("ax_overlay_ze"),
                        "Spike overlay unavailable\n(no rising edge after stimulation)",
                    ),
                    (_axes.get("ax_fr_first_ze"), "PSTH (first stim) unavailable"),
                    (_axes.get("ax_fr_second_ze"), "PSTH (second stim) unavailable"),
                    (_axes.get("ax_isi_first_ze"), "ISI (first stim) unavailable"),
                    (_axes.get("ax_isi_second_ze"), "ISI (second stim) unavailable"),
                ):
                    if ax is not None and ax.get_visible():
                        _mark_unavailable_axis(ax, msg)

            if _has_spike_cmp:
                thresholds_and_captions = [
                    render_cache.resolve_threshold(
                        src_idx,
                        mode=spike_threshold_mode,
                        fixed_threshold_uv=spike_threshold_uv,
                        spike_threshold_polarity=spike_threshold_polarity,
                        rms_multiplier=spike_threshold_rms_multiplier,
                        source=spike_sources[src_idx],
                        channel_index=ch,
                    )
                    for src_idx in plot_indices
                ]
                st_list = [
                    render_cache.spike_times_per_trial(
                        src_idx, spike_sources[src_idx], ch, thr_uv
                    )
                    for src_idx, (thr_uv, _caption) in zip(plot_indices, thresholds_and_captions)
                ]
                if str(spike_threshold_mode).strip().lower() == "rms_multiple":
                    threshold_labels = [
                        f"{record_labels[i]}: {caption}"
                        for i, (_thr_uv, caption) in enumerate(thresholds_and_captions)
                    ]
                    threshold_caption = " | ".join(threshold_labels)
                else:
                    threshold_caption = _spike_threshold_caption(
                        spike_threshold_uv, spike_threshold_polarity
                    )
                threshold_entries = [
                    (record_labels[i], caption)
                    for i, (_thr_uv, caption) in enumerate(thresholds_and_captions)
                ]

                def _end_spike_range(src_idx: int) -> tuple[float, float] | None:
                    if trigger_end_rising_rel_s_list is None or src_idx >= len(
                        trigger_end_rising_rel_s_list
                    ):
                        return None
                    marker = trigger_end_rising_rel_s_list[src_idx]
                    if marker is None:
                        return None
                    return (
                        float(marker) + float(zoom_end_t0),
                        float(marker) + float(zoom_end_t1),
                    )

                def _draw_spikes_for_section(
                    section: SectionPanels,
                    ax_raster: Any,
                    ax_fr: Any,
                    ax_trial: Any,
                    ax_isi: Any,
                    t_range: tuple[float, float] | None,
                    sec_title: str = "",
                    ranges_per_recording: Sequence[tuple[float, float] | None] | None = None,
                    *,
                    trigger_index: int | None = None,
                    show_raster: bool | None = None,
                    show_psth: bool | None = None,
                    show_trial_rate: bool | None = None,
                    show_isi: bool | None = None,
                ) -> None:
                    if not section.any_enabled():
                        return
                    want_raster = section.raster if show_raster is None else show_raster
                    want_psth = section.psth if show_psth is None else show_psth
                    want_trial = section.trial_rate if show_trial_rate is None else show_trial_rate
                    want_isi = section.isi if show_isi is None else show_isi
                    if not (want_raster or want_psth or want_trial or want_isi):
                        return
                    st_section: list[list[np.ndarray]] = []
                    for rec_i, (src_idx, (thr_uv, _caption)) in enumerate(
                        zip(plot_indices, thresholds_and_captions)
                    ):
                        if ranges_per_recording is not None:
                            rng = ranges_per_recording[rec_i]
                            if rng is None:
                                rng = (0.0, 0.0)
                        else:
                            rng = t_range
                        spikes = render_cache.spike_times_per_trial(
                            src_idx,
                            spike_sources[src_idx],
                            ch,
                            thr_uv,
                            t_range_s=rng,
                            trigger_index=trigger_index,
                        )
                        if trigger_index is None:
                            st_section.append(spikes)
                        elif 0 <= int(trigger_index) < len(spikes):
                            st_section.append([spikes[int(trigger_index)]])
                        else:
                            st_section.append([])
                    _draw_spike_panels_multi_channel(
                        ax_raster,
                        ax_fr,
                        ax_trial,
                        ax_isi,
                        None,
                        t_rel,
                        float(fs),
                        spike_threshold_uv,
                        psth_bin_window_s,
                        record_labels,
                        intan_dsp=spike_sources[0].intan_dsp,
                        t_range_s=t_range,
                        section_title=sec_title,
                        spikes_per_recording=st_section,
                        sampling_percent=sampling_percent,
                        threshold_caption=threshold_caption,
                        threshold_entries=threshold_entries,
                        legend_visible=legend_flags,
                        show_raster=want_raster,
                        show_psth=want_psth,
                        show_trial_rate=want_trial,
                        show_isi=want_isi,
                        across_trials=trigger_index is None,
                        colors=plot_colors,
                    )

                def _draw_psth_isi_for_trigger(
                    section: SectionPanels,
                    which: str,
                    ax_fr: Any,
                    ax_isi: Any,
                    t_range: tuple[float, float] | None,
                    sec_title: str = "",
                    ranges_per_recording: Sequence[tuple[float, float] | None] | None = None,
                ) -> None:
                    trigger_index = 0 if which == "first" else 1
                    stim_label = "First stimulation" if which == "first" else "Second stimulation"
                    want_psth = (
                        section.first_psth if which == "first" else section.second_psth
                    )
                    want_isi = section.first_isi if which == "first" else section.second_isi
                    title = f"{sec_title} — {stim_label}" if sec_title else stim_label
                    _draw_spikes_for_section(
                        section,
                        None,
                        ax_fr,
                        None,
                        ax_isi,
                        t_range,
                        title,
                        ranges_per_recording=ranges_per_recording,
                        trigger_index=trigger_index,
                        show_raster=False,
                        show_psth=want_psth,
                        show_trial_rate=False,
                        show_isi=want_isi,
                    )

                def _draw_all_spikes_overlay(
                    section: SectionPanels,
                    ax: Any,
                    t_range: tuple[float, float] | None,
                    sec_title: str = "",
                    ranges_per_recording: Sequence[tuple[float, float] | None] | None = None,
                ) -> None:
                    if ax is None or not section.spike_overlay:
                        return
                    overlay_section: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
                    for rec_i, (src_idx, (thr_uv, _caption)) in enumerate(
                        zip(plot_indices, thresholds_and_captions)
                    ):
                        if ranges_per_recording is not None:
                            rng = ranges_per_recording[rec_i]
                            if rng is None:
                                rng = (0.0, 0.0)
                        else:
                            rng = t_range
                        overlay_section.append(
                            render_cache.spike_waveforms(
                                src_idx,
                                spike_sources[src_idx],
                                ch,
                                thr_uv,
                                t_range_s=rng,
                                trigger_index=None,
                                pre_ms=spike_overlay_pre_ms,
                                post_ms=spike_overlay_post_ms,
                            )
                        )
                    title = (
                        f"{sec_title} — all stimulations"
                        if sec_title
                        else "All stimulations"
                    )
                    _draw_spike_overlay_panel(
                        ax,
                        overlay_section,
                        record_labels,
                        t_range_s=None,
                        section_title=title,
                        sampling_percent=sampling_percent,
                        legend_visible=legend_flags,
                        intan_dsp=spike_sources[0].intan_dsp,
                        pre_ms=spike_overlay_pre_ms,
                        post_ms=spike_overlay_post_ms,
                        thresholds_uv=[thr_uv for thr_uv, _cap in thresholds_and_captions],
                        colors=plot_colors,
                    )

                if _section_included(zoom_mode, "full"):
                    _draw_spikes_for_section(
                        full_panels,
                        ax_raster_f,
                        ax_fr_f,
                        ax_trial_fr_f,
                        ax_isi_f,
                        None,
                    )
                    _draw_psth_isi_for_trigger(
                        full_panels,
                        "first",
                        _axes.get("ax_fr_first_f"),
                        _axes.get("ax_isi_first_f"),
                        None,
                    )
                    _draw_psth_isi_for_trigger(
                        full_panels,
                        "second",
                        _axes.get("ax_fr_second_f"),
                        _axes.get("ax_isi_second_f"),
                        None,
                    )
                    _draw_all_spikes_overlay(
                        full_panels,
                        _axes.get("ax_overlay_f"),
                        None,
                    )
                if _section_included(zoom_mode, "zoom_onset"):
                    _draw_spikes_for_section(
                        onset_panels,
                        ax_raster_z,
                        ax_fr_z,
                        ax_trial_fr_z,
                        ax_isi_z,
                        (zoom_onset_t0, zoom_onset_t1),
                        "Onset zoom",
                    )
                    _draw_psth_isi_for_trigger(
                        onset_panels,
                        "first",
                        _axes.get("ax_fr_first_z"),
                        _axes.get("ax_isi_first_z"),
                        (zoom_onset_t0, zoom_onset_t1),
                        "Onset zoom",
                    )
                    _draw_psth_isi_for_trigger(
                        onset_panels,
                        "second",
                        _axes.get("ax_fr_second_z"),
                        _axes.get("ax_isi_second_z"),
                        (zoom_onset_t0, zoom_onset_t1),
                        "Onset zoom",
                    )
                    _draw_all_spikes_overlay(
                        onset_panels,
                        _axes.get("ax_overlay_z"),
                        (zoom_onset_t0, zoom_onset_t1),
                        "Onset zoom",
                    )
                if _section_included(zoom_mode, "zoom_trigger_end") and end_zoom_range is not None:
                    end_ranges = [_end_spike_range(src_idx) for src_idx in plot_indices]
                    _draw_spikes_for_section(
                        end_panels,
                        ax_raster_ze,
                        ax_fr_ze,
                        ax_trial_fr_ze,
                        ax_isi_ze,
                        end_zoom_range,
                        "End zoom",
                        ranges_per_recording=end_ranges,
                    )
                    _draw_psth_isi_for_trigger(
                        end_panels,
                        "first",
                        _axes.get("ax_fr_first_ze"),
                        _axes.get("ax_isi_first_ze"),
                        end_zoom_range,
                        "End zoom",
                        ranges_per_recording=end_ranges,
                    )
                    _draw_psth_isi_for_trigger(
                        end_panels,
                        "second",
                        _axes.get("ax_fr_second_ze"),
                        _axes.get("ax_isi_second_ze"),
                        end_zoom_range,
                        "End zoom",
                        ranges_per_recording=end_ranges,
                    )
                    _draw_all_spikes_overlay(
                        end_panels,
                        _axes.get("ax_overlay_ze"),
                        end_zoom_range,
                        "End zoom",
                        ranges_per_recording=end_ranges,
                    )
            else:
                spike_axes = (
                    ax_raster_f,
                    ax_fr_f,
                    ax_raster_z,
                    ax_fr_z,
                    ax_raster_ze,
                    ax_fr_ze,
                    _axes.get("ax_fr_first_f"),
                    _axes.get("ax_fr_second_f"),
                    _axes.get("ax_fr_first_z"),
                    _axes.get("ax_fr_second_z"),
                    _axes.get("ax_fr_first_ze"),
                    _axes.get("ax_fr_second_ze"),
                )
                for ax in spike_axes:
                    if ax is not None and ax.get_visible():
                        ax.text(
                            0.5,
                            0.5,
                            "Raster / PSTH / ISI unavailable\n(missing mmap source)",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=UNAVAILABLE_FONT_SIZE,
                        )
                        ax.set_axis_off()
                isi_axes = (
                    ax_isi_f,
                    ax_isi_z,
                    ax_isi_ze,
                    _axes.get("ax_isi_first_f"),
                    _axes.get("ax_isi_second_f"),
                    _axes.get("ax_isi_first_z"),
                    _axes.get("ax_isi_second_z"),
                    _axes.get("ax_isi_first_ze"),
                    _axes.get("ax_isi_second_ze"),
                )
                for ax in isi_axes:
                    if ax is not None and ax.get_visible():
                        ax.text(0.5, 0.5, "ISI unavailable", ha="center", va="center", transform=ax.transAxes, fontsize=UNAVAILABLE_FONT_SIZE)
                        ax.set_axis_off()
                for key in _OVERLAY_AXIS_KEYS:
                    ax = _axes.get(key)
                    if ax is not None and ax.get_visible():
                        _mark_unavailable_axis(ax, "Spike overlay unavailable\n(missing mmap source)")

            _finalize_and_save_three_part_page(
                fig=figs,
                pdf=pdf,
                axes=_axes,
                n_recordings=len(plot_indices),
            )

        rms_series: list[tuple[str, np.ndarray, np.ndarray]] = []
        for i in plot_indices:
            tx_rms, rms_vals = render_cache.rms_profile_recording_mean(i, n_channels)
            label = labels[i] if i < len(labels) else f"Recording {i + 1}"
            rms_series.append((label, tx_rms, rms_vals))
        if rms_series and display.summary_rms_page:
            _append_mean_rms_evolution_page(
                pdf,
                rms_series,
                rms_window_s,
                filter_title=filter_title,
                colors=plot_colors,
            )

        if display.summary_rms_table_page:
            rms_table_window_s = (
                float(main_intan_dsp.rms_window_s)
                if main_intan_dsp is not None
                else float(rms_window_s)
            )
            rms_by_recording: list[tuple[str, list[float]]] = []
            for i in plot_indices:
                src = spike_sources[i]
                label = labels[i] if i < len(labels) else f"Recording {i + 1}"
                values = [
                    render_cache.channel_rms_uv(i, src, ch) for ch in range(n_channels)
                ]
                rms_by_recording.append((label, values))
            _append_mean_rms_per_channel_table_page(
                pdf,
                channel_names[:n_channels],
                rms_by_recording,
                filter_title=filter_title,
                rms_window_s=rms_table_window_s,
            )

        if impedance_sessions and display.summary_impedance_page:
            _append_mean_impedance_summary_page(pdf, impedance_sessions)

        if display.summary_second_stim_montage_page:
            _append_second_stim_channel_montage_pages(
                pdf,
                spike_sources=spike_sources,
                plot_indices=plot_indices,
                labels=labels,
                channel_names=channel_names[:n_channels],
                t_rel=t_rel,
                sampling_percent=sampling_percent,
                trigger_end_rising_rel_s_list=trigger_end_rising_rel_s_list,
                zoom_onset_t0=zoom_onset_t0,
                zoom_onset_t1=zoom_onset_t1,
                show_zoom_span=_section_included(zoom_mode, "zoom_onset"),
                filter_short_lp=filter_short_lp,
                filter_short_hp=filter_short_hp,
                render_cache=render_cache,
            )

    _profile_print_delta(
        "plot_channel_multi_comparison",
        _profile_before,
        time.perf_counter() - _profile_t0,
    )
    return pdf_path


