"""PDF figures for triggered averaged traces, spike raster/PSTH/ISI, comparisons, and optional MEA layout inset."""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
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
    INTAN_SPIKE_POST_DETECT_SAMPLES,
    INTAN_SPIKE_PRE_DETECT_SAMPLES,
    INTAN_SPIKE_SCOPE_TSCALES_MS,
    INTAN_SPIKE_SCOPE_YSCALES_UV,
    IntanDspSettings,
    detect_spikes_intan,
    sliding_rms_intan_profile,
    sliding_rms_intan_profile_range,
)
from impedance_tracking import ImpedanceSession
from display_config import (
    PANEL_FIELD_NAMES,
    PlotDisplaySettings,
    RecordingStyle,
    SectionPanels,
    ZoomMode,
    resolve_display_label,
)
from pdf_layout import (
    LayoutFonts,
    Slot,
    build_stacked_pages,
    estimate_legend_rows,
    place_legend_below,
    psth_table_bbox,
    save_figure_to_pdf,
)
from plot_utils import downsample_points, shorten_filename_for_windows
from probe_layout import (
    draw_probe_layout_on_axes,
    load_probe_layout_json,
    match_contact_index,
    mea_panel_size_in,
)

# Zoom panel window (s), time relative to trigger (t=0)
ZOOM_T0 = -0.1
ZOOM_T1 = 0.2

# ISI: only spikes within [-ISI_HALF_WINDOW_S, +ISI_HALF_WINDOW_S] (s relative to stimulation)
ISI_HALF_WINDOW_S = 1.0

# Superimposed spike waveforms: Intan RHX Spike Scope (spikeplot.cpp).
# Display window is [-T/2, +T] ms around detection; snippets are 300 pre + 600 post samples.
SPIKE_OVERLAY_MAX_TRACES = 3000

# X-axis label for all time-relative-to-trigger plots
TIME_REL_XLABEL = "Time relative to stimulation (s)"
LEGEND_FONT_SIZE = 16
AXIS_TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 16
SECTION_HEADER_FONT_SIZE = 20
UNAVAILABLE_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 14
TABLE_FONT_SIZE = 16

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
PANEL_SPECS: tuple[PanelSpec, ...] = (
    PanelSpec("mean_raw", 2.20, has_legend=True),
    PanelSpec("mean_filtered", 2.10, has_legend=True),
    PanelSpec("first_trigger_raw", 1.90, has_legend=True),
    PanelSpec("first_trigger_hp", 1.80, has_legend=True),
    PanelSpec("second_trigger_raw", 1.90, has_legend=True),
    PanelSpec("second_trigger_hp", 1.80, has_legend=True),
    PanelSpec("rms", 1.70, has_legend=True),
    PanelSpec("raster", 1.85, has_legend=True),
    PanelSpec("psth", 1.90, has_legend=True),
    PanelSpec("trial_rate", 1.45, has_legend=True),
    PanelSpec("isi", 1.55, has_legend=True),
    PanelSpec("spike_overlay", 2.00, has_legend=True),
)
MEA_PLOT_HEIGHT_IN = 4.80
IMPEDANCE_PLOT_HEIGHT_IN = 2.40

_FULL_PANEL_TO_AXIS: dict[str, str] = {
    "mean_raw": "ax_full",
    "mean_filtered": "ax_full_filt",
    "first_trigger_raw": "ax_first_trigger",
    "first_trigger_hp": "ax_first_trigger_hp",
    "second_trigger_raw": "ax_second_trigger",
    "second_trigger_hp": "ax_second_trigger_hp",
    "rms": "ax_full_rms",
    "raster": "ax_raster_f",
    "psth": "ax_fr_f",
    "trial_rate": "ax_trial_fr_f",
    "isi": "ax_isi_f",
}

_ZOOM_ONSET_PANEL_TO_AXIS: dict[str, str] = {
    "mean_raw": "ax_zoom",
    "mean_filtered": "ax_zoom_filt",
    "first_trigger_raw": "ax_zoom_first",
    "first_trigger_hp": "ax_zoom_first_hp",
    "second_trigger_raw": "ax_zoom_second",
    "second_trigger_hp": "ax_zoom_second_hp",
    "rms": "ax_zoom_rms",
    "raster": "ax_raster_z",
    "psth": "ax_fr_z",
    "trial_rate": "ax_trial_fr_z",
    "isi": "ax_isi_z",
}

_ZOOM_END_PANEL_TO_AXIS: dict[str, str] = {
    "mean_raw": "ax_zoom_end",
    "mean_filtered": "ax_zoom_end_filt",
    "first_trigger_raw": "ax_zoom_end_first",
    "first_trigger_hp": "ax_zoom_end_first_hp",
    "second_trigger_raw": "ax_zoom_end_second",
    "second_trigger_hp": "ax_zoom_end_second_hp",
    "rms": "ax_zoom_end_rms",
    "raster": "ax_raster_ze",
    "psth": "ax_fr_ze",
    "trial_rate": "ax_trial_fr_ze",
    "isi": "ax_isi_ze",
}

# Overlay is injected just after first/second raw, not as a standalone Display row.
_TRIGGER_OVERLAY_AXIS: dict[str, dict[str, str]] = {
    "full": {"first": "ax_overlay_first_f", "second": "ax_overlay_second_f"},
    "zoom_onset": {"first": "ax_overlay_first_z", "second": "ax_overlay_second_z"},
    "zoom_trigger_end": {"first": "ax_overlay_first_ze", "second": "ax_overlay_second_ze"},
}
_SPIKE_OVERLAY_SPEC = next(spec for spec in PANEL_SPECS if spec.field == "spike_overlay")
_OVERLAY_AXIS_KEYS: tuple[str, ...] = tuple(
    key for mapping in _TRIGGER_OVERLAY_AXIS.values() for key in mapping.values()
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


def _trace_panels_enabled(panels: SectionPanels) -> bool:
    return any(
        getattr(panels, key)
        for key in (
            "mean_raw",
            "mean_filtered",
            "first_trigger_raw",
            "first_trigger_hp",
            "second_trigger_raw",
            "second_trigger_hp",
        )
    )


def _hide_axis(ax: Any) -> None:
    if ax is None:
        return
    ax.set_visible(False)
    ax.set_axis_off()


def _section_included(zoom_mode: ZoomMode, section: str) -> bool:
    if section == "full":
        return True
    if section == "zoom_onset":
        return zoom_mode in ("onset", "both")
    if section == "zoom_trigger_end":
        return zoom_mode in ("trigger_end", "both")
    return False


def _want_trigger_spike_overlay(panels: SectionPanels, which: str) -> bool:
    """Overlay exists only if the matching first/second raw panel is enabled."""
    raw_on = panels.first_trigger_raw if which == "first" else panels.second_trigger_raw
    return bool(raw_on and panels.spike_overlay)


def _overlay_slot_for(section: str, which: str) -> Slot:
    spec = _SPIKE_OVERLAY_SPEC
    return Slot(
        key=_TRIGGER_OVERLAY_AXIS[section][which],
        kind="plot",
        plot_height_in=_scaled_plot_height_in(spec.plot_height_in),
        has_legend=spec.has_legend,
        extra_below="none",
        table_rows=0,
    )


def _panel_axis_key(section: str, panel: str) -> str | None:
    if section == "full":
        return _FULL_PANEL_TO_AXIS.get(panel)
    if section == "zoom_onset":
        return _ZOOM_ONSET_PANEL_TO_AXIS.get(panel)
    if section == "zoom_trigger_end":
        return _ZOOM_END_PANEL_TO_AXIS.get(panel)
    return None


def _apply_panel_visibility(axes: dict[str, Any], display: PlotDisplaySettings, zoom_mode: ZoomMode) -> None:
    """No-op: panel visibility is applied by building only enabled axes."""
    del axes, display, zoom_mode


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
            if spec.field != "spike_overlay" and getattr(panels, spec.field)
        ]
        if not enabled:
            return
        if header_key is not None and header_text is not None:
            slots.append(
                Slot(
                    key=header_key,
                    kind="header",
                    plot_height_in=0.32,
                    header_text=header_text,
                )
            )
        for spec in enabled:
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
            if spec.field == "first_trigger_raw" and _want_trigger_spike_overlay(panels, "first"):
                slots.append(_overlay_slot_for(section, "first"))
            elif spec.field == "second_trigger_raw" and _want_trigger_spike_overlay(
                panels, "second"
            ):
                slots.append(_overlay_slot_for(section, "second"))

    _append_section("full", display.full_view, _FULL_PANEL_TO_AXIS, None, None)
    _append_section(
        "zoom_onset",
        display.zoom_onset,
        _ZOOM_ONSET_PANEL_TO_AXIS,
        "ax_hdr2",
        f"Stimulation-onset zoom [{zoom_onset_t0:.2f}, {zoom_onset_t1:.2f}] s rel. stimulation",
    )
    _append_section(
        "zoom_trigger_end",
        display.zoom_trigger_end,
        _ZOOM_END_PANEL_TO_AXIS,
        "ax_hdr3",
        "Stimulation-end zoom (next rising edge)",
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
        "mean_filtered",
        "first_trigger_raw",
        "first_trigger_hp",
        "second_trigger_raw",
        "second_trigger_hp",
        "rms",
        "raster",
        "psth",
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
        bbox=psth_table_bbox(ax_fr, _pdf_fonts(), len(rows)),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(TABLE_FONT_SIZE)
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
    return "bessel high-pass order 2 @ 250 Hz"


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
    highpass: bool = False,
) -> Optional[np.ndarray]:
    """Extract the Nth valid-stimulation window (raw amplifier or Intan filtered)."""
    triggers = np.asarray(source.valid_triggers, dtype=np.int64)
    if triggers.size <= int(trigger_index):
        return None
    trig = int(triggers[int(trigger_index)])
    start = int(trig - source.pre_n)
    end = int(trig + source.post_n)
    data = source.highpass if highpass else source.amplifier
    curve = np.asarray(data[ch, start:end], dtype=np.float64)
    if curve.shape[0] != n_expected:
        return None
    return curve


def _first_trigger_window(
    source: AmplifierSpikeSource,
    ch: int,
    n_expected: int,
    *,
    highpass: bool = False,
) -> Optional[np.ndarray]:
    """Extract the first-stimulation window (raw amplifier or Intan filtered)."""
    return _nth_trigger_window(
        source, ch, n_expected, trigger_index=0, highpass=highpass
    )


def _collect_nth_trigger_windows(
    spike_sources: Sequence[AmplifierSpikeSource],
    ch: int,
    n_expected: int,
    *,
    trigger_index: int,
) -> tuple[list[Optional[np.ndarray]], list[Optional[np.ndarray]]]:
    raw_curves: list[Optional[np.ndarray]] = []
    hp_curves: list[Optional[np.ndarray]] = []
    for src in spike_sources:
        raw_curves.append(
            _nth_trigger_window(
                src, ch, n_expected, trigger_index=trigger_index, highpass=False
            )
        )
        hp_curves.append(
            _nth_trigger_window(
                src, ch, n_expected, trigger_index=trigger_index, highpass=True
            )
        )
    return raw_curves, hp_curves


def _collect_first_trigger_windows(
    spike_sources: Sequence[AmplifierSpikeSource],
    ch: int,
    n_expected: int,
) -> tuple[list[Optional[np.ndarray]], list[Optional[np.ndarray]]]:
    return _collect_nth_trigger_windows(
        spike_sources, ch, n_expected, trigger_index=0
    )


def _trigger_end_zoom_bounds(
    end_markers: Sequence[float],
    zoom_t0: float,
    zoom_t1: float,
) -> tuple[float, float] | None:
    if not end_markers:
        return None
    return float(min(end_markers) + zoom_t0), float(max(end_markers) + zoom_t1)


def _mark_unavailable_axis(ax: Any, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=UNAVAILABLE_FONT_SIZE,
    )
    ax.set_axis_off()


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


def _draw_onset_offset_lines(
    ax: Any,
    *,
    end_markers: Sequence[float],
    end_line_specs: Optional[Sequence[tuple[float, str]]] = None,
    label_in_legend: bool = False,
) -> None:
    """Draw onset (t=0) and trigger-offset markers without zoom spans."""
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.15,
        color="red",
        label=("Stimulation (onset)" if label_in_legend else "_nolegend_"),
    )
    if end_line_specs:
        for idx, (value, label) in enumerate(end_line_specs):
            ax.axvline(
                value,
                linestyle="-.",
                linewidth=1.15,
                color="#1d4ed8",
                label=(label if label_in_legend and idx == 0 else "_nolegend_"),
            )
        return
    labeled = False
    for value in end_markers:
        ax.axvline(
            float(value),
            linestyle="-.",
            linewidth=1.15,
            color="#1d4ed8",
            label=(
                "Stimulation (offset)"
                if label_in_legend and not labeled
                else "_nolegend_"
            ),
        )
        labeled = True


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
        ax.plot(t_plot, slice_fn(curve), linewidth=first_lw, color=line_color, label=label)
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
    legend_visible: Sequence[bool] | None = None,
    aux_legend_label: Optional[str] = None,
    end_line_specs: Optional[Sequence[tuple[float, str]]] = None,
    show_reference_on_first_raw: bool = True,
    show_reference_on_first_hp: bool = False,
    show_zoom_span: bool = False,
    show_end_zoom_span: bool = False,
    end_zoom_t0: float | None = None,
    end_zoom_t1: float | None = None,
    first_trigger_hp_ylim: tuple[float, float] | None = None,
    ax_second_hp: Any = None,
    ax_second_raw: Any = None,
    second_trigger_raw: Sequence[Optional[np.ndarray]] | None = None,
    second_trigger_hp: Sequence[Optional[np.ndarray]] | None = None,
    title_second_hp: str = "Second stimulation — filtered",
    title_second_raw: str = "Second stimulation — raw",
    show_reference_on_second_raw: bool = True,
    show_reference_on_second_hp: bool = False,
    base_lw: float = 1.2,
    main_lw: float = 1.35,
    first_lw: float = 1.1,
) -> None:
    """Plot mean-trace and per-stimulation panels shared by each section."""
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

    for i, y_filt in enumerate(mean_filtered):
        line_color = colors[i % len(colors)]
        y_raw = (
            np.asarray(mean_raw[i], dtype=np.float64)
            if mean_raw is not None and i < len(mean_raw)
            else np.asarray(y_filt, dtype=np.float64)
        )
        show_leg = True if legend_visible is None else bool(legend_visible[i])
        multi = len(labels) > 1
        raw_label = _legend_label(labels[i], "raw mean", multi=multi, show_legend=show_leg)
        if ax_raw is not None:
            ax_raw.plot(
                t_plot,
                _slice(y_raw),
                linewidth=base_lw,
                color=line_color,
                label=raw_label,
            )
        hp_legend = (
            intan_hp_legends[i]
            if i < len(intan_hp_legends)
            else _default_filter_short_label()
        )
        filt_label = _legend_label(
            labels[i],
            f"filtered mean ({hp_legend})",
            multi=multi,
            show_legend=show_leg,
        )
        if not multi and show_leg:
            filt_label = f"Filtered mean ({hp_legend})"
        if ax_filt is not None:
            ax_filt.plot(
                t_plot, _slice(y_filt), linewidth=main_lw, color=line_color, label=filt_label
            )

    if ax_raw is not None:
        _add_trace_reference_overlays(ax_raw, **overlay_kwargs)
        ax_raw.set_title(title_raw)
        ax_raw.set_ylabel("Potential (µV)")
        ax_raw.set_xlabel(TIME_REL_XLABEL)
        ax_raw.grid(True, alpha=0.3)

    if ax_filt is not None:
        _add_trace_reference_overlays(ax_filt, **overlay_kwargs)
        ax_filt.set_title(title_filt)
        ax_filt.set_ylabel("Potential (µV)")
        ax_filt.set_xlabel(TIME_REL_XLABEL)
        ax_filt.grid(True, alpha=0.3)

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
        legend_suffix="first stimulation",
        single_legend="First stimulation",
        unavailable_message="First stimulation filtered signal unavailable",
        first_lw=first_lw,
        overlay_kwargs=overlay_kwargs,
        end_markers=end_markers,
        end_line_specs=end_line_specs,
        show_reference=show_reference_on_first_hp,
        ylim=first_trigger_hp_ylim,
        filtered=True,
    )
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
    second_raw = second_trigger_raw or []
    second_hp = second_trigger_hp or []
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
        legend_suffix="second stimulation",
        single_legend="Second stimulation",
        unavailable_message="Second stimulation filtered signal unavailable",
        first_lw=first_lw,
        overlay_kwargs=overlay_kwargs,
        end_markers=end_markers,
        end_line_specs=end_line_specs,
        show_reference=show_reference_on_second_hp,
        ylim=first_trigger_hp_ylim,
        filtered=True,
    )
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

    if x_limits is not None:
        for ax in (
            ax_raw,
            ax_filt,
            ax_first_hp,
            ax_first_raw,
            ax_second_hp,
            ax_second_raw,
        ):
            if ax is not None:
                ax.set_xlim(float(x_limits[0]), float(x_limits[1]))


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
    del entry_count
    legend = _legend_below(
        ax_raster,
        ncol=ncol,
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
) -> None:
    """Overlaid raster / PSTH / ISI for N recordings."""
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
        show_leg = True if legend_visible is None else bool(legend_visible[rec_idx])
        for tri, st in enumerate(st_per_trial):
            st_plot = st
            if t_range_s is not None:
                st_plot = st[(st >= t_xlim_lo) & (st <= t_xlim_hi)]
            if st_plot.size and show_raster:
                y_pts = np.full(st_plot.shape, y_offset + tri)
                st_ds, y_ds = downsample_points(st_plot, y_pts, sampling_percent)
                leg = labels[rec_idx] if tri == 0 and show_leg else "_nolegend_"
                ax_raster.scatter(
                    st_ds,
                    y_ds,
                    s=4,
                    c=color,
                    alpha=raster_alpha,
                    linewidths=0,
                    label=leg,
                )
        y_offset += len(st_per_trial)
        if rec_idx < len(spikes_per_recording) - 1 and show_raster:
            ax_raster.axhline(y_offset - 0.5, color="0.55", linestyle="--", linewidth=0.8, alpha=0.7)
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
        ax_raster.set_title(f"{sec}Raster — {short}")
        ax_raster.grid(True, alpha=0.25, axis="x")
        ax_raster.set_ylim(-0.5, max(y_offset - 0.5, 0.5))
        ax_raster.set_xlim(t_xlim_lo, t_xlim_hi)
        ax_raster.set_xlabel(TIME_REL_XLABEL)
        _add_raster_threshold_legend(
            ax_raster,
            cap,
            threshold_entries=threshold_entries,
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
        ax_fr.set_ylabel("Rate (Hz)")
        ax_fr.set_title(f"{sec}Firing rate (PSTH window = {bin_w:g} s) — {short}")
        ax_fr.grid(True, alpha=0.3)
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
                linewidth=1.0,
                marker=trial_marker,
                markersize=trial_markersize,
                label=labels[rec_idx] if show_leg else "_nolegend_",
            )
    if show_trial_rate:
        ax_trial_fr.set_title(f"{sec}Firing rate per trial — displayed window")
        ax_trial_fr.set_xlabel("Trial index")
        ax_trial_fr.set_ylabel("Firing rate (Hz)")
        ax_trial_fr.grid(True, alpha=0.25)
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


def _extract_spike_waveforms(
    source: AmplifierSpikeSource,
    ch: int,
    spike_times_per_trial: list[np.ndarray],
    *,
    pre_samples: int = INTAN_SPIKE_PRE_DETECT_SAMPLES,
    post_samples: int = INTAN_SPIKE_POST_DETECT_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HIGH snippets aligned on detection, as in RHX Spike Scope (spikeplot.cpp).

    Buffer is ``[t - pre_samples, t + post_samples)``. Returns
    ``(t_ms, waveforms, t_rel_s)`` with t=0 at the threshold-crossing sample.
    """
    fs = float(source.fs)
    pre_n = max(0, int(pre_samples))
    post_n = max(1, int(post_samples))
    win_len = pre_n + post_n
    t_ms = (np.arange(-pre_n, post_n, dtype=np.float64) / fs) * 1e3
    empty = (
        t_ms,
        np.empty((0, win_len), dtype=np.float32),
        np.empty(0, dtype=np.float64),
    )
    row = source.high_trace_for_channel(ch)
    n_samples = int(np.asarray(row).shape[0])
    triggers = np.asarray(source.valid_triggers, dtype=np.int64)
    if triggers.size == 0 or n_samples < win_len:
        return empty
    snippets: list[np.ndarray] = []
    t_rel_keep: list[float] = []
    n_trials = min(len(spike_times_per_trial), int(triggers.size))
    for trial_i in range(n_trials):
        st_arr = np.asarray(spike_times_per_trial[trial_i], dtype=np.float64).ravel()
        if st_arr.size == 0:
            continue
        trig = int(triggers[trial_i])
        centers = trig + np.rint(st_arr * fs).astype(np.int64)
        for t_val, center in zip(st_arr, centers):
            start = int(center) - pre_n
            end = int(center) + post_n
            if start < 0 or end > n_samples:
                continue
            snippet = np.asarray(row[start:end], dtype=np.float32)
            if snippet.shape[0] != win_len:
                continue
            snippets.append(snippet)
            t_rel_keep.append(float(t_val))
    if not snippets:
        return empty
    return t_ms, np.vstack(snippets), np.asarray(t_rel_keep, dtype=np.float64)


def _spike_scope_time_window_ms(tscale_ms: float) -> tuple[float, float]:
    """Intan Spike Scope x-axis: tMin = −T/2, tMax = +T (systemstate.cpp)."""
    t = float(tscale_ms)
    if t not in INTAN_SPIKE_SCOPE_TSCALES_MS:
        nearest = min(INTAN_SPIKE_SCOPE_TSCALES_MS, key=lambda v: abs(v - t))
        t = float(nearest)
    return (-t / 2.0, t)


def _spike_scope_xtick_divisor_ms(tscale_ms: float) -> int:
    t = int(round(float(tscale_ms)))
    if t == 10:
        return 2
    if t in (16, 20):
        return 4
    return 1


def _spike_scope_ylim_uv(abs_peak: float) -> float:
    peak = max(0.0, float(abs_peak))
    for scale in INTAN_SPIKE_SCOPE_YSCALES_UV:
        if peak <= scale:
            return float(scale)
    return float(INTAN_SPIKE_SCOPE_YSCALES_UV[-1])


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
    tscale_ms: float = 4.0,
    thresholds_uv: Sequence[float] | None = None,
) -> None:
    """Overlay HIGH snippets like RHX Spike Scope (spikeplot.cpp)."""
    if ax is None:
        return
    short, _ = _spike_pipeline_captions(intan_dsp=intan_dsp)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    n_rec = len(overlay_per_recording)
    dense = n_rec > 2
    line_alpha = 0.10 if not dense else 0.06
    line_width = 0.45 if not dense else 0.35
    sec = f"{section_title} — " if section_title else ""
    t_min_ms, t_max_ms = _spike_scope_time_window_ms(tscale_ms)
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
        if shown_f.size:
            abs_peak = max(abs_peak, float(np.nanpercentile(np.abs(shown_f), 99.0)))
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
            rec_label = f"{label} (n={n_all}, {n_shown} shown)"
        else:
            rec_label = f"{label} (n={n_all})"
        ax.plot(
            [],
            [],
            color=color,
            linewidth=1.2,
            label=rec_label if show_leg else "_nolegend_",
        )
        if thresholds_uv is not None and rec_idx < len(thresholds_uv):
            thr = float(thresholds_uv[rec_idx])
            ax.axhline(
                thr,
                color=color,
                linestyle="--",
                linewidth=0.85,
                zorder=2,
                alpha=0.85,
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
            fontsize=UNAVAILABLE_FONT_SIZE,
        )
        ax.set_axis_off()
        return
    ax.axhline(0.0, color="0.35", linewidth=0.8, zorder=2)
    ax.axvline(0.0, color="0.35", linestyle=":", linewidth=0.9, zorder=2)
    if intan_dsp is not None and intan_dsp.artifact_suppression_enabled:
        art = float(intan_dsp.artifact_threshold_uv)
        ax.axhline(art, color="#2563eb", linestyle=":", linewidth=0.8, zorder=2, alpha=0.7)
        ax.axhline(-art, color="#2563eb", linestyle=":", linewidth=0.8, zorder=2, alpha=0.7)
    ax.set_xlabel("Time relative to detection (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(
        f"{sec}Spike Scope overlay — [{t_min_ms:g}, {t_max_ms:g}] ms "
        f"(n={n_all_total}) — {short}"
    )
    ax.set_xlim(t_min_ms, t_max_ms)
    y_lim = _spike_scope_ylim_uv(abs_peak)
    ax.set_ylim(-y_lim, y_lim)
    divisor = _spike_scope_xtick_divisor_ms(tscale_ms)
    ticks = [
        float(t)
        for t in range(int(np.ceil(t_min_ms)), int(np.floor(t_max_ms)) + 1)
        if t == int(t_min_ms) or t == int(t_max_ms) or t % divisor == 0
    ]
    if ticks:
        ax.set_xticks(ticks)
    ax.grid(True, alpha=0.3)


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
        self._spikes: dict[tuple[int, int, float, tuple[float, float] | None], list[np.ndarray]] = {}
        self._waveforms: dict[
            tuple[int, int, float, tuple[float, float] | None],
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (
            src_idx,
            ch,
            float(threshold_uv),
            self._t_range_key(t_range_s),
            trigger_index,
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
        extracted = _extract_spike_waveforms(source, ch, times)
        self._waveforms[key] = extracted
        return extracted


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
            fontsize=UNAVAILABLE_FONT_SIZE,
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
            fontsize=ANNOTATION_FONT_SIZE,
            alpha=0.9,
            zorder=4,
        )
    ax_imp.set_ylabel("|Z| @ 1 kHz (Ω)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax_imp.set_xlabel("Session time (_YYMMDD_HHMMSS)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax_imp.margins(x=0.08)
    date_locator = mdates.AutoDateLocator()
    ax_imp.xaxis.set_major_locator(date_locator)
    ax_imp.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
    ax_imp.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
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
    if ax is None:
        return
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
    spike_scope_tscale_ms: float = 4.0,
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
    filter_title = (
        main_intan_dsp.filter_title_label()
        if main_intan_dsp is not None
        else _default_filter_title_label()
    )
    filter_short = (
        main_intan_dsp.filter_short_label()
        if main_intan_dsp is not None
        else _default_filter_short_label()
    )
    rms_note = f"window {rms_window_s:g} s"
    n_channels = min(src.amplifier.shape[0] for src in spike_sources)
    end_markers = [v for v in (trigger_end_rising_rel_s_list or []) if v is not None]
    _has_spike_cmp = (
        fs is not None
        and len(spike_sources) == n_records
    )
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
            for src_idx in plot_indices:
                src = spike_sources[src_idx]
                means_raw_ch.append(render_cache.mean_raw(src_idx, src, ch))
                means_ch.append(render_cache.mean_hp(src_idx, src, ch))
                _hp_note, hp_legend = _intan_hp_mean_filter_captions(src.intan_dsp)
                intan_hp_legends.append(hp_legend)

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
            ax_full_filt = _axes.get("ax_full_filt")
            ax_first_trigger_hp = _axes.get("ax_first_trigger_hp")
            ax_first_trigger = _axes.get("ax_first_trigger")
            ax_second_trigger_hp = _axes.get("ax_second_trigger_hp")
            ax_second_trigger = _axes.get("ax_second_trigger")
            ax_full_rms = _axes.get("ax_full_rms")
            ax_raster_f = _axes.get("ax_raster_f")
            ax_fr_f = _axes.get("ax_fr_f")
            ax_trial_fr_f = _axes.get("ax_trial_fr_f")
            ax_isi_f = _axes.get("ax_isi_f")
            ax_zoom = _axes.get("ax_zoom")
            ax_zoom_filt = _axes.get("ax_zoom_filt")
            ax_zoom_first_hp = _axes.get("ax_zoom_first_hp")
            ax_zoom_first = _axes.get("ax_zoom_first")
            ax_zoom_second_hp = _axes.get("ax_zoom_second_hp")
            ax_zoom_second = _axes.get("ax_zoom_second")
            ax_zoom_rms = _axes.get("ax_zoom_rms")
            ax_raster_z = _axes.get("ax_raster_z")
            ax_fr_z = _axes.get("ax_fr_z")
            ax_trial_fr_z = _axes.get("ax_trial_fr_z")
            ax_isi_z = _axes.get("ax_isi_z")
            ax_zoom_end = _axes.get("ax_zoom_end")
            ax_zoom_end_filt = _axes.get("ax_zoom_end_filt")
            ax_zoom_end_first_hp = _axes.get("ax_zoom_end_first_hp")
            ax_zoom_end_first = _axes.get("ax_zoom_end_first")
            ax_zoom_end_second_hp = _axes.get("ax_zoom_end_second_hp")
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
            rms_series_full_multi: list[tuple[str, np.ndarray, np.ndarray]] = []
            rms_series_zoom_multi: list[tuple[str, np.ndarray, np.ndarray]] = []
            rms_series_zoom_end_multi: list[tuple[str, np.ndarray, np.ndarray]] = []
            for i in plot_indices:
                src = spike_sources[i]
                label = labels[i] if i < len(labels) else f"Recording {i + 1}"
                tx_full, rms_full_vals = render_cache.rms_profile(i, src, channel_index=ch)
                rms_series_full_multi.append((label, tx_full, rms_full_vals))
                tx_zoom, rms_zoom_vals = _slice_rms_profile_window(
                    tx_full,
                    rms_full_vals,
                    zoom_onset_t0,
                    zoom_onset_t1,
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
                        float(marker_i + zoom_end_t0),
                        float(marker_i + zoom_end_t1),
                    )
                    rms_series_zoom_end_multi.append((label, tx_end, rms_zoom_end_vals))
            first_trigger_raw_all, first_trigger_hp_all = _collect_nth_trigger_windows(
                spike_sources,
                ch,
                int(t_rel.shape[0]),
                trigger_index=0,
            )
            second_trigger_raw_all, second_trigger_hp_all = _collect_nth_trigger_windows(
                spike_sources,
                ch,
                int(t_rel.shape[0]),
                trigger_index=1,
            )
            first_trigger_raw = [first_trigger_raw_all[i] for i in plot_indices]
            first_trigger_hp = [first_trigger_hp_all[i] for i in plot_indices]
            second_trigger_raw = [second_trigger_raw_all[i] for i in plot_indices]
            second_trigger_hp = [second_trigger_hp_all[i] for i in plot_indices]
            record_labels = [
                labels[i] if i < len(labels) else f"Recording {i + 1}"
                for i in plot_indices
            ]
            mean_n_suffix = _mean_n_samples_title_suffix(
                [int(spike_sources[i].valid_triggers.size) for i in plot_indices]
            )
            plot_colors = [colors[k % len(colors)] for k in range(len(plot_indices))]
            full_panels = display.full_view
            onset_panels = display.zoom_onset
            end_panels = display.zoom_trigger_end

            if _section_included(zoom_mode, "full") and full_panels.any_enabled():
                if _trace_panels_enabled(full_panels):
                    _plot_mean_section_trace_panels(
                    ax_raw=ax_full,
                    ax_filt=ax_full_filt,
                    ax_first_hp=ax_first_trigger_hp,
                    ax_first_raw=ax_first_trigger,
                    ax_second_hp=ax_second_trigger_hp,
                    ax_second_raw=ax_second_trigger,
                    t_rel=t_rel,
                    time_mask=None,
                    x_limits=(float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None,
                    labels=record_labels,
                    mean_filtered=means_ch,
                    mean_raw=means_raw_ch,
                    first_trigger_raw=first_trigger_raw,
                    first_trigger_hp=first_trigger_hp,
                    second_trigger_raw=second_trigger_raw,
                    second_trigger_hp=second_trigger_hp,
                    colors=plot_colors,
                    zoom_t0=zoom_onset_t0,
                    zoom_t1=zoom_onset_t1,
                    end_markers=end_markers,
                    intan_hp_legends=intan_hp_legends,
                    title_raw=f"{channel_name} — Raw mean (full view){mean_n_suffix}",
                    title_filt=f"{channel_name} — Filtered mean ({filter_short}){mean_n_suffix}",
                    title_first_hp=f"First stimulation — filtered ({filter_title})",
                    title_first_raw="First stimulation — raw",
                    title_second_hp=f"Second stimulation — filtered ({filter_title})",
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
                        f"RMS — {filter_short}, {rms_note}",
                        x_limits=(float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None,
                    )

            zmask = (t_rel >= zoom_onset_t0) & (t_rel <= zoom_onset_t1)
            if _section_included(zoom_mode, "zoom_onset") and onset_panels.any_enabled():
                if _trace_panels_enabled(onset_panels):
                    _plot_mean_section_trace_panels(
                    ax_raw=ax_zoom,
                    ax_filt=ax_zoom_filt,
                    ax_first_hp=ax_zoom_first_hp,
                    ax_first_raw=ax_zoom_first,
                    ax_second_hp=ax_zoom_second_hp,
                    ax_second_raw=ax_zoom_second,
                    t_rel=t_rel,
                    time_mask=zmask,
                    x_limits=(zoom_onset_t0, zoom_onset_t1),
                    labels=record_labels,
                    mean_filtered=means_ch,
                    mean_raw=means_raw_ch,
                    first_trigger_raw=first_trigger_raw,
                    first_trigger_hp=first_trigger_hp,
                    second_trigger_raw=second_trigger_raw,
                    second_trigger_hp=second_trigger_hp,
                    colors=plot_colors,
                    zoom_t0=zoom_onset_t0,
                    zoom_t1=zoom_onset_t1,
                    end_markers=end_markers,
                    intan_hp_legends=intan_hp_legends,
                    title_raw=f"{channel_name} — Raw mean (onset zoom [{zoom_onset_t0:g}, {zoom_onset_t1:g}] s){mean_n_suffix}",
                    title_filt=f"{channel_name} — Filtered mean — onset zoom ({filter_short}){mean_n_suffix}",
                    title_first_hp=f"First stimulation — filtered, onset zoom ({filter_title})",
                    title_first_raw="First stimulation — raw (onset zoom)",
                    title_second_hp=f"Second stimulation — filtered, onset zoom ({filter_title})",
                    title_second_raw="Second stimulation — raw (onset zoom)",
                    legend_cols=legend_cols,
                    legend_visible=legend_flags,
                    show_reference_on_first_raw=False,
                    show_reference_on_first_hp=False,
                    show_reference_on_second_raw=False,
                    show_reference_on_second_hp=False,
                    first_trigger_hp_ylim=first_trigger_hp_ylim,
                    show_zoom_span=False,
                    show_end_zoom_span=False,
                    )
                if onset_panels.rms:
                    _plot_rms_series(
                        ax_zoom_rms,
                        rms_series_zoom_multi,
                        f"RMS — onset zoom, {filter_short}, {rms_note}",
                        x_limits=(zoom_onset_t0, zoom_onset_t1),
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
                        ax_filt=ax_zoom_end_filt,
                        ax_first_hp=ax_zoom_end_first_hp,
                        ax_first_raw=ax_zoom_end_first,
                        ax_second_hp=ax_zoom_end_second_hp,
                        ax_second_raw=ax_zoom_end_second,
                        t_rel=t_rel,
                        time_mask=end_mask,
                        x_limits=(end_zoom_t0, end_zoom_t1),
                        labels=record_labels,
                        mean_filtered=means_ch,
                        mean_raw=means_raw_ch,
                        first_trigger_raw=first_trigger_raw,
                        first_trigger_hp=first_trigger_hp,
                        second_trigger_raw=second_trigger_raw,
                        second_trigger_hp=second_trigger_hp,
                        colors=plot_colors,
                        zoom_t0=zoom_end_t0,
                        zoom_t1=zoom_end_t1,
                        end_markers=end_markers,
                        intan_hp_legends=intan_hp_legends,
                        title_raw=f"{channel_name} — Raw mean (end zoom [{end_zoom_t0:.2f}, {end_zoom_t1:.2f}] s){mean_n_suffix}",
                        title_filt=f"{channel_name} — Filtered mean — end zoom ({filter_short}){mean_n_suffix}",
                        title_first_hp=f"First stimulation — filtered, end zoom ({filter_title})",
                        title_first_raw="First stimulation — raw (end zoom)",
                        title_second_hp=f"Second stimulation — filtered, end zoom ({filter_title})",
                        title_second_raw="Second stimulation — raw (end zoom)",
                        legend_cols=legend_cols,
                        legend_visible=legend_flags,
                        show_reference_on_first_raw=False,
                        show_reference_on_first_hp=False,
                        show_reference_on_second_raw=False,
                        show_reference_on_second_hp=False,
                        first_trigger_hp_ylim=first_trigger_hp_ylim,
                        show_zoom_span=False,
                        show_end_zoom_span=False,
                        )
                    if end_panels.rms:
                        _plot_rms_series(
                            ax_zoom_end_rms,
                            rms_series_zoom_end_multi,
                            f"RMS — end zoom, {filter_short}, {rms_note}",
                            x_limits=(end_zoom_t0, end_zoom_t1),
                        )
            elif _section_included(zoom_mode, "zoom_trigger_end"):
                for ax, msg in (
                    (ax_zoom_end, "End zoom unavailable\n(no rising edge after stimulation)"),
                    (ax_zoom_end_filt, "End zoom unavailable\n(no rising edge after stimulation)"),
                    (ax_zoom_end_first_hp, "First filtered stimulation unavailable"),
                    (ax_zoom_end_first, "First raw stimulation unavailable"),
                    (ax_zoom_end_second_hp, "Second filtered stimulation unavailable"),
                    (ax_zoom_end_second, "Second raw stimulation unavailable"),
                    (ax_zoom_end_rms, "RMS unavailable"),
                    (
                        _axes.get("ax_overlay_first_ze"),
                        "Spike overlay unavailable\n(no rising edge after stimulation)",
                    ),
                    (
                        _axes.get("ax_overlay_second_ze"),
                        "Spike overlay unavailable\n(no rising edge after stimulation)",
                    ),
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
                ) -> None:
                    if not section.any_enabled():
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
                        st_section.append(
                            render_cache.spike_times_per_trial(
                                src_idx,
                                spike_sources[src_idx],
                                ch,
                                thr_uv,
                                t_range_s=rng,
                            )
                        )
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
                        show_raster=section.raster,
                        show_psth=section.psth,
                        show_trial_rate=section.trial_rate,
                        show_isi=section.isi,
                    )

                def _draw_trigger_overlay(
                    section: SectionPanels,
                    which: str,
                    ax: Any,
                    t_range: tuple[float, float] | None,
                    sec_title: str = "",
                    ranges_per_recording: Sequence[tuple[float, float] | None] | None = None,
                ) -> None:
                    if ax is None or not _want_trigger_spike_overlay(section, which):
                        return
                    trigger_index = 0 if which == "first" else 1
                    trigger_label = (
                        "First stimulation" if which == "first" else "Second stimulation"
                    )
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
                                trigger_index=trigger_index,
                            )
                        )
                    title = (
                        f"{sec_title} — {trigger_label}" if sec_title else trigger_label
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
                        tscale_ms=spike_scope_tscale_ms,
                        thresholds_uv=[thr_uv for thr_uv, _cap in thresholds_and_captions],
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
                    _draw_trigger_overlay(
                        full_panels,
                        "first",
                        _axes.get("ax_overlay_first_f"),
                        None,
                    )
                    _draw_trigger_overlay(
                        full_panels,
                        "second",
                        _axes.get("ax_overlay_second_f"),
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
                    _draw_trigger_overlay(
                        onset_panels,
                        "first",
                        _axes.get("ax_overlay_first_z"),
                        (zoom_onset_t0, zoom_onset_t1),
                        "Onset zoom",
                    )
                    _draw_trigger_overlay(
                        onset_panels,
                        "second",
                        _axes.get("ax_overlay_second_z"),
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
                    _draw_trigger_overlay(
                        end_panels,
                        "first",
                        _axes.get("ax_overlay_first_ze"),
                        end_zoom_range,
                        "End zoom",
                        ranges_per_recording=end_ranges,
                    )
                    _draw_trigger_overlay(
                        end_panels,
                        "second",
                        _axes.get("ax_overlay_second_ze"),
                        end_zoom_range,
                        "End zoom",
                        ranges_per_recording=end_ranges,
                    )
            else:
                for ax in (ax_raster_f, ax_fr_f, ax_raster_z, ax_fr_z, ax_raster_ze, ax_fr_ze):
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
                for ax in (ax_isi_f, ax_isi_z, ax_isi_ze):
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
                pdf, rms_series, rms_window_s, filter_title=filter_title
            )

        if impedance_sessions and display.summary_impedance_page:
            _append_mean_impedance_summary_page(pdf, impedance_sessions)

    _profile_print_delta(
        "plot_channel_multi_comparison",
        _profile_before,
        time.perf_counter() - _profile_t0,
    )
    return pdf_path


