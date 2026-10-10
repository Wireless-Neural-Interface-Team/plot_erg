"""Backend-neutral panel data preparation.

Builds :class:`PanelSpec` trees from a :class:`~panel_registry.RenderRequest`
without drawing. Screen (pyqtgraph) and PDF/CLI (matplotlib via
:mod:`panel_draw_mpl`) both consume the same specs for supported panels.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from display_config import STREAM_PLOT_COLORS, recording_trace_color
from panel_catalog import panel_label
from view_config import STREAM_SHORT_LABELS, continuous_sync_offset_s

# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CurveSeries:
    """Polyline series in data coordinates."""

    x: np.ndarray
    y: np.ndarray
    color: str
    label: str = ""
    linewidth: float = 1.0
    row: int = 0


@dataclass
class ScatterSeries:
    """Point cloud (raster / overlay markers)."""

    x: np.ndarray
    y: np.ndarray
    color: str
    label: str = ""
    size: float = 4.0
    symbol: str = "o"
    row: int = 0


@dataclass
class HistSeries:
    """Histogram bars from bin edges + counts."""

    edges: np.ndarray
    counts: np.ndarray
    color: str
    label: str = ""
    row: int = 0


@dataclass
class AnnotationSpec:
    """Lightweight annotation (vline / text / hline)."""

    kind: str  # "vline" | "hline" | "text" | "span"
    x: float = 0.0
    y: float = 0.0
    x2: float = 0.0
    text: str = ""
    color: str = "#dc2626"
    row: int = 0


@dataclass
class ScaleBarSpec:
    """Floating time / amplitude scale bars (screen path)."""

    enabled: bool = False
    x_unit: str | None = None
    y_unit: str | None = None
    time_manual: bool = False
    time_s: float = 0.1
    amp_manual: bool = False
    amplitude: float = 100.0
    linewidth: float = 1.8
    fontsize: float = 8.0
    color: str = "#111827"


SeriesItem = CurveSeries | ScatterSeries | HistSeries


@dataclass
class PanelSpec:
    """Neutral draw recipe for one panel (possibly multi-row)."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    series: list[SeriesItem] = field(default_factory=list)
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    annotations: list[AnnotationSpec] = field(default_factory=list)
    status: str = "ok"  # ok | empty | unavailable | pending
    layout_kind: str = "single"  # single | stacked | montage | placeholder
    # Multi-row extras (full_recording / montage).
    n_rows: int = 1
    row_labels: list[str] = field(default_factory=list)
    row_ylims: list[tuple[float, float] | None] = field(default_factory=list)
    row_xlims: list[tuple[float, float] | None] = field(default_factory=list)
    row_kinds: list[str] = field(default_factory=list)
    row_channels: list[int] = field(default_factory=list)
    row_keys: list[tuple[int, str]] = field(default_factory=list)
    row_highlight: list[bool] = field(default_factory=list)
    status_message: str = ""
    grid: bool = True
    grid_alpha: float = 0.35
    show_legend: bool = True
    scale_bars: ScaleBarSpec = field(default_factory=ScaleBarSpec)
    # Affichage (ViewerSettings → PlotHost) — polices / bordures / légende.
    title_font_size: float = 10.0
    label_font_size: float = 9.0
    tick_font_size: float = 8.0
    title_bold: bool = False
    title_italic: bool = False
    title_underline: bool = False
    label_bold: bool = False
    label_italic: bool = False
    label_underline: bool = False
    tick_bold: bool = False
    tick_italic: bool = False
    tick_underline: bool = False
    show_borders: bool = True
    ticks_inside: bool = False
    legend_location: str = "below"
    legend_font_size: float = 9.0
    legend_columns: int = 1
    legend_gap: float = 0.22
    legend_frame: bool = True
    legend_bold: bool = False
    legend_italic: bool = False
    legend_underline: bool = False


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def prepare_panel(request: Any) -> PanelSpec:
    """Extract drawable series for ``request.panel`` (never runs sosfilt)."""
    panel = str(getattr(request, "panel", "") or "")
    style = getattr(request, "style", None)
    grid = bool(getattr(style, "grid", True)) if style is not None else True
    grid_alpha = float(getattr(style, "grid_alpha", 0.35) or 0.35) if style else 0.35

    if not getattr(request, "recordings", None):
        return PanelSpec(
            title=panel_label(panel) if panel else "—",
            status="unavailable",
            status_message="Charger un enregistrement pour afficher ce panneau.",
            layout_kind="placeholder",
            grid=False,
        )

    try:
        if panel == "full_recording":
            spec = _prepare_full_recording(request)
        elif panel in _MEAN_PANELS or panel in _TRIGGER_PANELS or panel in _ANALYSIS_TRACE:
            spec = _prepare_triggered_trace(request)
        elif panel in _RMS_PANELS or panel == "analysis_rms":
            spec = _prepare_rms(request)
        elif panel in _SPIKE_PANELS or panel.startswith("analysis_"):
            spec = _prepare_spike_like(request)
        elif panel == "montage_continuous_raw":
            spec = _prepare_montage_continuous(request)
        elif panel.startswith("montage_"):
            spec = _prepare_montage_triggered(request)
        elif panel in {"mea_layout", "impedance"}:
            spec = _prepare_placeholder(
                request,
                f"{panel_label(panel)}\n(aperçu texte — export PDF via matplotlib)",
            )
        elif panel.startswith("summary_"):
            spec = _prepare_placeholder(
                request,
                f"{panel_label(panel)}\n(résumé — export PDF via matplotlib)",
            )
        else:
            spec = _prepare_placeholder(
                request,
                f"Panneau non pris en charge à l’écran : {panel}",
                status="unavailable",
            )
    except Exception as exc:  # defensive: never crash the GUI path
        return PanelSpec(
            title=str(getattr(request, "channel_name", "") or panel),
            status="unavailable",
            status_message=f"Erreur de préparation : {exc}",
            layout_kind="placeholder",
            grid=False,
        )

    spec.grid = grid
    spec.grid_alpha = grid_alpha
    legend = getattr(request, "legend", None)
    if legend is not None:
        spec.show_legend = bool(getattr(legend, "visible", True))
    spec.scale_bars = _scale_bar_spec(request, panel=panel)
    _apply_display_settings(spec, request)
    _apply_text_overrides(spec, request)
    _apply_legend_label_overrides(spec, request)
    return spec


def restyle_panel_spec(spec: PanelSpec, request: Any) -> PanelSpec:
    """Refresh display attrs on a cached spec without reloading traces.

    Reuses the same ``x`` / ``y`` arrays. Used by the screen path when the
    structure fingerprint is unchanged (channel highlight, colors, ylim, …).
    """
    out = copy(spec)
    style = getattr(request, "style", None)
    grid = bool(getattr(style, "grid", True)) if style is not None else True
    grid_alpha = float(getattr(style, "grid_alpha", 0.35) or 0.35) if style else 0.35
    out.grid = grid
    out.grid_alpha = grid_alpha
    legend = getattr(request, "legend", None)
    if legend is not None:
        out.show_legend = bool(getattr(legend, "visible", True))

    panel = str(getattr(request, "panel", "") or "")
    montage = panel == "montage_continuous_raw" or panel.startswith("montage_")
    lw = _line_width(request)
    if montage:
        lw = max(0.6, lw * 0.85)

    colors = list(getattr(request, "colors", []) or [])
    selected = int(getattr(request, "channel_index", -1) or -1)

    # Surbrillance canal (montage) + ylims par ligne.
    if out.row_channels:
        out.row_highlight = [int(ch) == selected for ch in out.row_channels]
    if out.row_kinds:
        settings = getattr(request, "settings", None)
        new_ylims: list[tuple[float, float] | None] = []
        for kind in out.row_kinds:
            kind_s = str(kind)
            if kind_s in {"raw", "hp", "lp"}:
                new_ylims.append(_ylim_tuple(settings, kind_s))
            elif kind_s == "rms":
                new_ylims.append(_ylim_tuple(settings, "rms"))
            else:
                new_ylims.append(None)
        out.row_ylims = new_ylims

    # Couleurs / épaisseur — séries regroupées par ligne (ordre = enregistrements).
    by_row: dict[int, list[int]] = {}
    for index, item in enumerate(out.series):
        row = int(getattr(item, "row", 0) or 0)
        by_row.setdefault(row, []).append(index)

    new_series: list[SeriesItem] = list(out.series)
    for row, indices in by_row.items():
        kind = str(out.row_kinds[row]) if row < len(out.row_kinds) else ""
        default_color = STREAM_PLOT_COLORS.get(kind, "#334155")
        for j, series_index in enumerate(indices):
            item = new_series[series_index]
            color = recording_trace_color(
                colors, j, fallback=str(getattr(item, "color", default_color) or default_color)
            )
            if isinstance(item, CurveSeries):
                new_series[series_index] = replace(
                    item, color=color, linewidth=float(lw)
                )
            elif isinstance(item, ScatterSeries):
                new_series[series_index] = replace(item, color=color)
            elif isinstance(item, HistSeries):
                new_series[series_index] = replace(item, color=color)
    out.series = new_series

    # Ylim panneau simple (hors montage multi-lignes).
    if not out.row_kinds:
        settings = getattr(request, "settings", None)
        stream = None
        if panel.endswith("_hp") or "_hp" in panel:
            stream = "hp"
        elif panel.endswith("_lp") or "_lp" in panel:
            stream = "lp"
        elif "rms" in panel:
            stream = "rms"
        elif panel.endswith("_raw") or "_raw" in panel or panel == "full_recording":
            placed = str(getattr(getattr(request, "placement", None), "stream", "") or "")
            stream = placed if placed in {"raw", "hp", "lp"} else "raw"
        if stream is not None:
            out.ylim = _ylim_tuple(settings, stream)

    out.scale_bars = _scale_bar_spec(request, panel=panel)
    _apply_display_settings(out, request)
    _apply_text_overrides(out, request)
    _apply_legend_label_overrides(out, request)
    return out


def screen_prepare_key(request: Any) -> tuple[Any, ...]:
    """Fingerprint for screen prepare-cache (structure + stream readiness)."""
    try:
        from panel_registry import (
            _montage_continuous_channels,
            _structure_fingerprint,
        )
    except Exception:
        return (str(getattr(request, "panel", "")),)

    base = _structure_fingerprint(request)
    recordings = list(getattr(request, "recordings", []) or [])
    # data_generation : invalide le cache écran après Traiter (même pour montage continu).
    generations = tuple(
        int(getattr(rec, "data_generation", 0) or 0) for rec in recordings
    )
    ready: list[Any] = []
    panel = str(getattr(request, "panel", "") or "")
    if not recordings:
        return base + (generations, ())

    if panel == "montage_continuous_raw":
        try:
            n_channels = min(int(rec.n_channels) for rec in recordings)
            channels = _montage_continuous_channels(request, n_channels)
            settings = getattr(request, "settings", None)
            streams = set()
            if settings is not None:
                resolved = getattr(settings, "resolved_continuous_streams", None)
                if callable(resolved):
                    streams = set(str(s) for s in resolved())
            for rec in recordings:
                for ch in channels:
                    if "hp" in streams:
                        ready.append(bool(rec.stream_ready("hp", int(ch))))
                    if "lp" in streams:
                        ready.append(bool(rec.stream_ready("lp", int(ch))))
        except Exception:
            ready.append("err")
    else:
        ch = int(getattr(request, "channel_index", 0) or 0)
        for rec in recordings:
            try:
                ready.append(bool(rec.is_channel_ready(ch)))
            except Exception:
                ready.append(False)
            for stream in ("hp", "lp"):
                try:
                    ready.append(bool(rec.stream_ready(stream, ch)))
                except Exception:
                    ready.append(False)
    return base + (generations, tuple(ready))


def _face_flags(face: Any) -> tuple[bool, bool, bool]:
    if face is None:
        return False, False, False
    return (
        bool(getattr(face, "bold", False)),
        bool(getattr(face, "italic", False)),
        bool(getattr(face, "underline", False)),
    )


def _apply_display_settings(spec: PanelSpec, request: Any) -> None:
    """Propagate ViewerSettings style / legend onto the screen spec."""
    style = getattr(request, "style", None)
    if style is not None:
        spec.title_font_size = float(getattr(style, "title_font_size", 10.0) or 10.0)
        spec.label_font_size = float(getattr(style, "label_font_size", 9.0) or 9.0)
        spec.tick_font_size = float(getattr(style, "tick_font_size", 8.0) or 8.0)
        (
            spec.title_bold,
            spec.title_italic,
            spec.title_underline,
        ) = _face_flags(getattr(style, "title_face", None))
        (
            spec.label_bold,
            spec.label_italic,
            spec.label_underline,
        ) = _face_flags(getattr(style, "label_face", None))
        (
            spec.tick_bold,
            spec.tick_italic,
            spec.tick_underline,
        ) = _face_flags(getattr(style, "tick_face", None))
        spec.show_borders = bool(getattr(style, "show_borders", True))
        spec.ticks_inside = bool(getattr(style, "ticks_inside", False))
    legend = getattr(request, "legend", None)
    if legend is not None:
        spec.legend_location = str(getattr(legend, "location", "below") or "below")
        spec.legend_font_size = float(getattr(legend, "font_size", 9.0) or 9.0)
        spec.legend_columns = max(1, int(getattr(legend, "columns", 1) or 1))
        spec.legend_gap = float(getattr(legend, "gap", 0.22) or 0.0)
        spec.legend_frame = bool(getattr(legend, "frame", True))
        (
            spec.legend_bold,
            spec.legend_italic,
            spec.legend_underline,
        ) = _face_flags(getattr(legend, "face", None))


def _apply_text_overrides(spec: PanelSpec, request: Any) -> None:
    """Apply dock Affichage title / axis overrides (empty = keep auto)."""
    texts = getattr(getattr(request, "settings", None), "texts", None)
    if texts is None:
        return
    title = str(getattr(texts, "title", "") or "").strip()
    xlabel = str(getattr(texts, "xlabel", "") or "").strip()
    ylabel = str(getattr(texts, "ylabel", "") or "").strip()
    if title:
        spec.title = title
    if xlabel:
        spec.xlabel = xlabel
    if ylabel:
        spec.ylabel = ylabel
        # Multi-row: keep per-row channel labels; only fill empty row labels.
        if spec.row_labels:
            spec.row_labels = [
                (label if str(label or "").strip() else ylabel) for label in spec.row_labels
            ]


def _apply_legend_label_overrides(spec: PanelSpec, request: Any) -> None:
    """Replace series legend labels with non-empty TextOverrides.legend_labels."""
    texts = getattr(getattr(request, "settings", None), "texts", None)
    if texts is None:
        return
    overrides = tuple(str(x) for x in getattr(texts, "resolved_legend_labels", lambda: ())())
    if not overrides:
        overrides = tuple(str(x) for x in (getattr(texts, "legend_labels", ()) or ()))
    if not overrides:
        return
    # Map unique non-empty labels in draw order → override slots.
    slot = 0
    for item in spec.series:
        label = str(getattr(item, "label", "") or "").strip()
        if not label or label == "_nolegend_":
            continue
        if slot < len(overrides) and str(overrides[slot]).strip():
            item.label = str(overrides[slot]).strip()
        slot += 1


def _scale_bar_spec(request: Any, *, panel: str) -> ScaleBarSpec:
    """Build scale-bar recipe from ``request.style`` (disabled if unsupported)."""
    style = getattr(request, "style", None)
    if style is None or not bool(getattr(style, "show_scale_bars", False)):
        return ScaleBarSpec(enabled=False)
    try:
        from panel_registry import _scale_bar_units

        x_unit, y_unit = _scale_bar_units(panel)
    except Exception:
        x_unit, y_unit = ("s", "µV")
    # Overlay spikes: abscisse en ms (prepare_panel), pas en secondes.
    if panel in {"spike_overlay", "analysis_overlay"}:
        x_unit, y_unit = ("ms", "µV")
    if x_unit is None and y_unit is None:
        return ScaleBarSpec(enabled=False)
    lw = float(getattr(style, "line_width", 1.2) or 1.2)
    return ScaleBarSpec(
        enabled=True,
        x_unit=x_unit,
        y_unit=y_unit,
        time_manual=bool(getattr(style, "scale_bar_time_manual", False)),
        time_s=float(getattr(style, "scale_bar_time_s", 0.1) or 0.1),
        amp_manual=bool(getattr(style, "scale_bar_amplitude_manual", False)),
        amplitude=float(getattr(style, "scale_bar_amplitude", 100.0) or 100.0),
        linewidth=max(1.4, lw * 1.15),
        fontsize=float(getattr(style, "tick_font_size", 8.0) or 8.0),
    )


# ---------------------------------------------------------------------------
# Panel groups
# ---------------------------------------------------------------------------

_MEAN_PANELS = frozenset({"mean_raw", "mean_hp", "mean_lp"})
_TRIGGER_PANELS = frozenset(
    {
        "first_trigger_raw",
        "first_trigger_hp",
        "first_trigger_lp",
        "second_trigger_raw",
        "second_trigger_hp",
        "second_trigger_lp",
    }
)
_ANALYSIS_TRACE = frozenset({"analysis_raw", "analysis_hp", "analysis_lp"})
_RMS_PANELS = frozenset({"rms", "first_rms", "second_rms"})
_SPIKE_PANELS = frozenset(
    {
        "psth",
        "first_psth",
        "second_psth",
        "isi",
        "first_isi",
        "second_isi",
        "trial_rate",
        "raster",
        "spike_overlay",
        "analysis_psth",
        "analysis_trial_rate",
        "analysis_isi",
        "analysis_raster",
        "analysis_raster_channel",
        "analysis_overlay",
    }
)

_STREAM_FOR_PANEL: dict[str, str] = {
    "mean_raw": "raw",
    "mean_hp": "hp",
    "mean_lp": "lp",
    "first_trigger_raw": "raw",
    "first_trigger_hp": "hp",
    "first_trigger_lp": "lp",
    "second_trigger_raw": "raw",
    "second_trigger_hp": "hp",
    "second_trigger_lp": "lp",
    "analysis_raw": "raw",
    "analysis_hp": "hp",
    "analysis_lp": "lp",
}

_MONTAGE_TRIGGERED: dict[str, tuple[str, int | None]] = {
    "montage_mean_raw": ("raw", None),
    "montage_mean_hp": ("hp", None),
    "montage_mean_lp": ("lp", None),
    "montage_second_raw": ("raw", 1),
    "montage_second_hp": ("hp", 1),
    "montage_second_lp": ("lp", 1),
    "montage_second_to_third_lp": ("lp", 1),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_window(t: np.ndarray, window: tuple[float, float] | None) -> np.ndarray | None:
    if window is None:
        return None
    return (t >= float(window[0])) & (t <= float(window[1]))


def _union_x_range(
    series: list[SeriesItem],
    *,
    rows: set[int] | None = None,
) -> tuple[float, float] | None:
    """Union des bornes X des séries (courbes / scatter / hist), sans padding."""
    x_min: float | None = None
    x_max: float | None = None
    for item in series:
        if rows is not None and int(getattr(item, "row", 0) or 0) not in rows:
            continue
        if isinstance(item, CurveSeries) or isinstance(item, ScatterSeries):
            x = np.asarray(item.x, dtype=np.float64).ravel()
            if x.size == 0:
                continue
            finite = x[np.isfinite(x)]
            if finite.size == 0:
                continue
            lo, hi = float(np.min(finite)), float(np.max(finite))
        elif isinstance(item, HistSeries):
            edges = np.asarray(item.edges, dtype=np.float64).ravel()
            if edges.size < 2:
                continue
            finite = edges[np.isfinite(edges)]
            if finite.size < 2:
                continue
            lo, hi = float(finite[0]), float(finite[-1])
        else:
            continue
        x_min = lo if x_min is None else min(x_min, lo)
        x_max = hi if x_max is None else max(x_max, hi)
    if x_min is None or x_max is None or not (x_max > x_min):
        return None
    return (x_min, x_max)


def _decimate(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    from plot_utils import decimate_envelope

    return decimate_envelope(x, y, max_points)


def _line_width(request: Any) -> float:
    style = getattr(request, "style", None)
    return float(getattr(style, "line_width", 1.0) or 1.0) if style else 1.0


def _max_points(request: Any) -> int:
    style = getattr(request, "style", None)
    return max(200, int(getattr(style, "max_points_per_curve", 100_250) or 100_250))


def _filter_suffix(request: Any, index: int, stream: str | None) -> str:
    legend = getattr(request, "legend", None)
    if legend is None or not bool(getattr(legend, "show_filter_details", True)):
        return ""
    if stream is None or stream == "raw":
        return ""
    recordings = list(getattr(request, "recordings", []) or [])
    if index < 0 or index >= len(recordings):
        return ""
    kind = "highpass" if stream == "hp" else "lowpass"
    try:
        short = recordings[index].meta.filter_short_label(kind)
    except Exception:
        return ""
    return f" ({short})" if short else ""


def _count_suffix(request: Any) -> str:
    legend = getattr(request, "legend", None)
    if legend is None or not bool(getattr(legend, "show_sample_counts", True)):
        return ""
    recordings = list(getattr(request, "recordings", []) or [])
    if not recordings:
        return ""
    counts = {int(getattr(r, "n_trials", 0) or 0) for r in recordings}
    if not counts:
        return ""
    if len(counts) == 1:
        return f" (n={counts.pop()})"
    return " (n=" + "/".join(str(int(getattr(r, "n_trials", 0) or 0)) for r in recordings) + ")"


def _series_label(
    request: Any,
    index: int,
    suffix: str = "",
    *,
    stream: str | None = None,
) -> str:
    flags = list(getattr(request, "legend_flags", []) or [])
    labels = list(getattr(request, "labels", []) or [])
    if flags and index < len(flags) and not flags[index]:
        return ""
    label = labels[index] if index < len(labels) else f"#{index + 1}"
    label = f"{label}{_filter_suffix(request, index, stream)}"
    show_suffix = True
    texts = getattr(getattr(request, "settings", None), "texts", None)
    if texts is not None:
        show_suffix = bool(getattr(texts, "show_series_suffix", True))
    if suffix and show_suffix:
        label = f"{label} — {suffix}"
    label = f"{label}{_count_suffix(request)}"
    return str(label)


def _ylim_tuple(settings: Any, stream: str) -> tuple[float, float] | None:
    if settings is None:
        return None
    if stream == "rms":
        lim = getattr(settings, "rms_ylim", None)
    else:
        fn = getattr(settings, "ylim_for_stream", None)
        lim = fn(stream) if callable(fn) else None
    if lim is None:
        return None
    as_tuple = getattr(lim, "as_tuple", None)
    if callable(as_tuple):
        return as_tuple()
    return None


def _placeholder(
    *,
    title: str,
    message: str,
    status: str = "pending",
) -> PanelSpec:
    return PanelSpec(
        title=title,
        status=status,
        status_message=message,
        layout_kind="placeholder",
        annotations=[AnnotationSpec(kind="text", text=message, color="#64748b")],
        grid=False,
    )


def _prepare_placeholder(
    request: Any, message: str, *, status: str = "pending"
) -> PanelSpec:
    return _placeholder(
        title=f"{getattr(request, 'channel_name', '')} — {panel_label(request.panel)}",
        message=message,
        status=status,
    )


# ---------------------------------------------------------------------------
# Continuous
# ---------------------------------------------------------------------------


def _prepare_full_recording(request: Any) -> PanelSpec:
    placed = str(getattr(request.placement, "stream", "") or "").strip()
    if placed in {"raw", "hp", "lp"}:
        streams = [placed]
    else:
        streams = list(request.settings.resolved_continuous_streams()) or ["raw"]
    window = request.section_window() if request.placement.has_custom_zoom else None
    if window is None:
        window = request.settings.x_limits.as_tuple()
    max_pts = _max_points(request)
    lw = _line_width(request)
    sync_mode = request.settings.time_sync
    window_is_absolute = bool(
        request.placement.has_custom_zoom and request.placement.zoom_absolute
    )
    ref_offset = 0.0
    if request.recordings:
        ref_offset = continuous_sync_offset_s(
            request.recordings[0].stimulation_times_s(), sync_mode
        )

    series: list[SeriesItem] = []
    annotations: list[AnnotationSpec] = []
    row_labels: list[str] = []
    row_ylims: list[tuple[float, float] | None] = []
    row_kinds: list[str] = []
    pending_any = False
    drawn = 0

    for row, stream in enumerate(streams):
        short = STREAM_SHORT_LABELS.get(str(stream), str(stream).upper())
        row_labels.append(f"{short} (µV)")
        row_ylims.append(_ylim_tuple(request.settings, stream))
        row_kinds.append(str(stream))
        stream_drawn = 0
        for index, recording in enumerate(request.recordings):
            if str(stream) in {"hp", "lp"} and not recording.stream_ready(
                stream, request.channel_index
            ):
                pending_any = True
                continue
            stims = recording.stimulation_times_s()
            offset = continuous_sync_offset_s(stims, sync_mode)
            t_range = None
            if window is not None:
                if window_is_absolute:
                    t_range = (float(window[0]), float(window[1]))
                else:
                    t_range = (
                        float(window[0]) + offset,
                        float(window[1]) + offset,
                    )
            t, values = recording.continuous_trace(
                stream,
                request.channel_index,
                max_points=max_pts,
                require_ready=str(stream) in {"hp", "lp"},
                t_range=t_range,
            )
            if values.size == 0:
                if str(stream) in {"hp", "lp"}:
                    pending_any = True
                continue
            t_plot = np.asarray(t, dtype=np.float64) - offset
            color = recording_trace_color(
                request.colors,
                index,
                fallback=STREAM_PLOT_COLORS.get(str(stream), "#334155"),
            )
            label = _series_label(
                request, index, "continuous", stream=str(stream)
            )
            series.append(
                CurveSeries(
                    x=np.asarray(t_plot, dtype=np.float64),
                    y=np.asarray(values, dtype=np.float64),
                    color=str(color),
                    label=label or "",
                    linewidth=lw,
                    row=row,
                )
            )
            stream_drawn += 1
            drawn += 1
            if request.settings.continuous_mark_stims:
                for stim_t in stims:
                    stim_plot = float(stim_t) - offset
                    if window is not None:
                        if window_is_absolute:
                            if not (window[0] <= float(stim_t) <= window[1]):
                                continue
                        elif not (window[0] <= stim_plot <= window[1]):
                            continue
                    annotations.append(
                        AnnotationSpec(
                            kind="vline",
                            x=stim_plot,
                            color="#dc2626",
                            row=row,
                        )
                    )
        if stream_drawn == 0:
            msg = f"{short} — calcul…" if pending_any else f"{short} indisponible"
            annotations.append(
                AnnotationSpec(kind="text", text=msg, color="#64748b", row=row)
            )

    xlim = None
    if window is not None:
        if window_is_absolute:
            xlim = (window[0] - ref_offset, window[1] - ref_offset)
        else:
            xlim = (float(window[0]), float(window[1]))

    time_label = (
        "Temps relatif au trigger (s)"
        if sync_mode == "trigger"
        else "Temps (s)"
    )
    if drawn == 0:
        status = "pending" if pending_any else "unavailable"
        return _placeholder(
            title=f"{request.channel_name} — continuous",
            message=(
                "Filtre en cours de calcul…"
                if pending_any
                else "Enregistrement continu indisponible.\nTraiter un .rhs (F5), puis sélectionner un canal."
            ),
            status=status,
        )

    # Auto X : étendue exacte des courbes (pas enableAutoRange + padding / vlines).
    if xlim is None:
        xlim = _union_x_range(series)

    return PanelSpec(
        title=f"{request.channel_name} — continuous",
        xlabel=time_label,
        ylabel="Potential (µV)",
        series=series,
        xlim=xlim,
        annotations=annotations,
        status="ok",
        layout_kind="stacked" if len(streams) > 1 else "single",
        n_rows=len(streams),
        row_labels=row_labels,
        row_ylims=row_ylims,
        row_kinds=row_kinds,
    )


# ---------------------------------------------------------------------------
# Means / triggers / analysis traces
# ---------------------------------------------------------------------------


def _trigger_index_for_panel(request: Any) -> int | None:
    panel = str(request.panel)
    if panel in _MEAN_PANELS:
        return None
    if panel.startswith("first_"):
        return 0
    if panel.startswith("second_"):
        return 1
    if panel.startswith("analysis_"):
        return request.settings.analysis.trigger_index()
    return None


def _prepare_triggered_trace(request: Any) -> PanelSpec:
    panel = str(request.panel)
    stream = _STREAM_FOR_PANEL.get(panel)
    if stream is None:
        return _prepare_placeholder(request, f"Flux inconnu pour {panel}", status="unavailable")
    trigger_index = _trigger_index_for_panel(request)
    window = request.section_window()
    if window is None:
        return _placeholder(
            title=f"{request.channel_name} — {panel_label(panel)}",
            message="Section indisponible (pas de marqueur de fin de stimulation).",
            status="unavailable",
        )
    lw = _line_width(request)
    max_pts = _max_points(request)
    series: list[SeriesItem] = []
    drawn = 0
    for index, recording in enumerate(request.recordings):
        if trigger_index is None:
            curve = recording.mean(stream, request.channel_index)
            suffix = "moyenne d’essais"
        else:
            if trigger_index >= recording.n_trials:
                continue
            curve = recording.trigger_window(
                trigger_index, stream, request.channel_index
            )
            suffix = f"stimulation #{trigger_index + 1}"
        if curve is None:
            continue
        t = np.asarray(recording.t_rel, dtype=np.float64)
        y = np.asarray(curve, dtype=np.float64)
        mask = _mask_window(t, window)
        if mask is not None:
            t, y = t[mask], y[mask]
        t, y = _decimate(t, y, max_pts)
        series.append(
            CurveSeries(
                x=t,
                y=y,
                color=str(request.colors[index % len(request.colors)]),
                label=_series_label(request, index, suffix, stream=stream),
                linewidth=lw,
            )
        )
        drawn += 1

    if drawn == 0:
        return _placeholder(
            title=f"{request.channel_name} — {panel_label(panel)}",
            message="Trace d’analyse indisponible.\nCalculer le canal (F6).",
            status="unavailable",
        )

    mode = ""
    if panel.startswith("analysis_"):
        mode = f" ({request.settings.analysis.describe()})"
    return PanelSpec(
        title=f"{request.channel_name} — {panel_label(panel)}{mode}",
        xlabel="Temps relatif à la stimulation (s)",
        ylabel="Potential (µV)",
        series=series,
        xlim=(float(window[0]), float(window[1])),
        ylim=_ylim_tuple(request.settings, stream),
        annotations=_stim_markers_rel(request),
        status="ok",
        layout_kind="single",
        row_kinds=[stream],
    )


def _stim_markers_rel(
    request: Any,
    *,
    rows: Sequence[int] | None = None,
) -> list[AnnotationSpec]:
    """Vlines onset (t=0) / fin de stim — une copie par ligne (montage multi-rangées)."""
    out: list[AnnotationSpec] = []
    if not getattr(request.legend, "show_reference_markers", True):
        return out
    row_ids = [0] if rows is None else [int(r) for r in rows]
    if not row_ids:
        return out
    for row in row_ids:
        for marker in request.end_markers():
            out.append(
                AnnotationSpec(
                    kind="vline", x=float(marker), color="#1d4ed8", row=row
                )
            )
        out.append(AnnotationSpec(kind="vline", x=0.0, color="#dc2626", row=row))
    return out


# ---------------------------------------------------------------------------
# RMS
# ---------------------------------------------------------------------------


def _prepare_rms(request: Any) -> PanelSpec:
    panel = str(request.panel)
    if panel == "analysis_rms":
        trigger_index = request.settings.analysis.trigger_index()
        kind = "mean" if trigger_index is None else ("first" if trigger_index == 0 else "second")
        if trigger_index is not None and trigger_index > 1:
            kind = "mean"
        mode = f" ({request.settings.analysis.describe()})"
    else:
        kind = {"rms": "mean", "first_rms": "first", "second_rms": "second"}[panel]
        mode = ""
    window = request.section_window()
    if window is None:
        return _placeholder(
            title=f"{request.channel_name} — {panel_label(panel)}",
            message="Section indisponible.",
            status="unavailable",
        )
    lw = _line_width(request)
    max_pts = _max_points(request)
    series: list[SeriesItem] = []
    for index, recording in enumerate(request.recordings):
        t, values = recording.rms_profile(kind, request.channel_index)
        if values.size == 0:
            continue
        mask = _mask_window(t, window)
        x = t[mask] if mask is not None else t
        y = values[mask] if mask is not None else values
        x, y = _decimate(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            max_pts,
        )
        series.append(
            CurveSeries(
                x=x,
                y=y,
                color=str(request.colors[index % len(request.colors)]),
                label=_series_label(request, index, "RMS"),
                linewidth=lw,
            )
        )
    if not series:
        return _placeholder(
            title=f"{request.channel_name} — RMS{mode}",
            message="Profil RMS indisponible pour ce canal.",
            status="unavailable",
        )
    return PanelSpec(
        title=f"{request.channel_name} — {panel_label(panel)}{mode}",
        xlabel="Temps relatif à la stimulation (s)",
        ylabel="RMS (µV)",
        series=series,
        xlim=(float(window[0]), float(window[1])),
        ylim=_ylim_tuple(request.settings, "rms"),
        status="ok",
        row_kinds=["rms"],
    )


# ---------------------------------------------------------------------------
# Spikes
# ---------------------------------------------------------------------------


def _spike_trigger_index(request: Any) -> int | None:
    panel = str(request.panel)
    if panel.startswith("first_"):
        return 0
    if panel.startswith("second_"):
        return 1
    if panel.startswith("analysis_"):
        return request.settings.analysis.trigger_index()
    return None


def _spike_window(request: Any) -> tuple[float, float] | None:
    # Prefer relative stim window helpers from panel_registry when available.
    try:
        from panel_registry import _relative_stim_window

        window = _relative_stim_window(request)
        if window is not None:
            return window
    except Exception:
        pass
    return request.section_window()


def _prepare_spike_like(request: Any) -> PanelSpec:
    panel = str(request.panel)
    if panel in {"spike_overlay", "analysis_overlay"}:
        return _prepare_spike_overlay(request)
    if panel.endswith("psth") or panel == "analysis_psth":
        return _prepare_psth(request)
    if panel.endswith("isi") or panel == "analysis_isi":
        return _prepare_isi(request)
    if "raster" in panel:
        return _prepare_raster(request)
    if "trial_rate" in panel:
        return _prepare_trial_rate(request)
    if panel in _ANALYSIS_TRACE:
        return _prepare_triggered_trace(request)
    return _prepare_placeholder(
        request, f"Panneau spikes non pris en charge : {panel}", status="unavailable"
    )


def _spike_trains(request: Any, trigger_index: int | None) -> list[list[np.ndarray]]:
    windows = request.per_recording_windows()
    trains: list[list[np.ndarray]] = []
    for index, recording in enumerate(request.recordings):
        window = windows[index] if index < len(windows) else None
        trains.append(
            recording.spike_times(
                request.channel_index,
                trigger_index=trigger_index,
                t_range_s=window,
            )
        )
    return trains


def _prepare_psth(request: Any) -> PanelSpec:
    from draw_primitives import _psth_mean_hz

    window = _spike_window(request)
    if window is None:
        return _placeholder(
            title=f"{request.channel_name} — PSTH",
            message="Section indisponible.",
            status="unavailable",
        )
    reference = request.reference
    if reference is None:
        return _placeholder(
            title=f"{request.channel_name} — PSTH",
            message="Aucun enregistrement.",
            status="unavailable",
        )
    trigger_index = _spike_trigger_index(request)
    trains = _spike_trains(request, trigger_index)
    t_rel = np.asarray(reference.t_rel, dtype=np.float64)
    if t_rel.size < 2:
        return _placeholder(
            title=f"{request.channel_name} — PSTH",
            message="Section indisponible.",
            status="unavailable",
        )
    bin_w = max(
        float(request.settings.psth_bin_window_s),
        1.0 / max(float(reference.meta.fs), 1.0),
    )
    lw = _line_width(request)
    series: list[SeriesItem] = []
    for index, st_per_trial in enumerate(trains):
        n_trials = max(1, len(st_per_trial))
        centers, rate = _psth_mean_hz(
            st_per_trial, t_rel, n_trials, bin_w, t_range_s=window
        )
        if centers.size < 2:
            continue
        series.append(
            CurveSeries(
                x=np.asarray(centers, dtype=np.float64),
                y=np.asarray(rate, dtype=np.float64),
                color=str(request.colors[index % len(request.colors)]),
                label=_series_label(request, index),
                linewidth=lw,
            )
        )
    if not series:
        return _placeholder(
            title=f"{request.channel_name} — PSTH",
            message="Aucun spike dans cette fenêtre.",
            status="empty",
        )
    return PanelSpec(
        title=f"{request.channel_name} — {panel_label(request.panel)}",
        xlabel="Temps relatif à la stimulation (s)",
        ylabel="Taux (Hz)",
        series=series,
        xlim=(float(window[0]), float(window[1])),
        annotations=_stim_markers_rel(request),
        status="ok",
        row_kinds=["psth"],
    )


def _prepare_isi(request: Any) -> PanelSpec:
    window = _spike_window(request)
    if window is None:
        return _placeholder(
            title=f"{request.channel_name} — ISI",
            message="Section indisponible.",
            status="unavailable",
        )
    trigger_index = _spike_trigger_index(request)
    trains = _spike_trains(request, trigger_index)
    series: list[SeriesItem] = []
    for index, st_per_trial in enumerate(trains):
        intervals: list[float] = []
        for trial in st_per_trial:
            arr = np.asarray(trial, dtype=np.float64).ravel()
            if arr.size >= 2:
                intervals.extend(np.diff(np.sort(arr)).tolist())
        if not intervals:
            continue
        counts, edges = np.histogram(np.asarray(intervals, dtype=np.float64), bins=40)
        series.append(
            HistSeries(
                edges=np.asarray(edges, dtype=np.float64),
                counts=np.asarray(counts, dtype=np.float64),
                color=str(request.colors[index % len(request.colors)]),
                label=_series_label(request, index),
            )
        )
    if not series:
        return _placeholder(
            title=f"{request.channel_name} — ISI",
            message="Aucun spike dans cette fenêtre.",
            status="empty",
        )
    return PanelSpec(
        title=f"{request.channel_name} — {panel_label(request.panel)}",
        xlabel="ISI (s)",
        ylabel="Nombre",
        series=series,
        status="ok",
        row_kinds=["isi"],
    )


def _prepare_raster(request: Any) -> PanelSpec:
    window = _spike_window(request)
    if window is None:
        return _placeholder(
            title=f"{request.channel_name} — Raster",
            message="Section indisponible.",
            status="unavailable",
        )
    trigger_index = _spike_trigger_index(request)
    trains = _spike_trains(request, trigger_index)
    sampling = max(1, int(getattr(request.settings, "sampling_percent", 100) or 100))
    series: list[SeriesItem] = []
    y_cursor = 0.0
    for index, st_per_trial in enumerate(trains):
        xs: list[float] = []
        ys: list[float] = []
        trial_indices = list(range(len(st_per_trial)))
        if sampling < 100 and trial_indices:
            step = max(1, int(np.ceil(100.0 / sampling)))
            trial_indices = trial_indices[::step]
        for trial_i in trial_indices:
            trial = st_per_trial[trial_i]
            arr = np.asarray(trial, dtype=np.float64).ravel()
            if arr.size == 0:
                continue
            if window is not None:
                arr = arr[(arr >= window[0]) & (arr <= window[1])]
            if arr.size == 0:
                continue
            y = y_cursor + float(trial_i)
            xs.extend(arr.tolist())
            ys.extend([y] * int(arr.size))
        if xs:
            series.append(
                ScatterSeries(
                    x=np.asarray(xs, dtype=np.float64),
                    y=np.asarray(ys, dtype=np.float64),
                    color=str(request.colors[index % len(request.colors)]),
                    label=_series_label(request, index),
                    size=3.0,
                    symbol="|",
                )
            )
        y_cursor += max(1.0, float(len(st_per_trial)))
    if not series:
        return _placeholder(
            title=f"{request.channel_name} — Raster",
            message="Aucun spike dans cette fenêtre.",
            status="empty",
        )
    return PanelSpec(
        title=f"{request.channel_name} — {panel_label(request.panel)}",
        xlabel="Temps relatif à la stimulation (s)",
        ylabel="Essai",
        series=series,
        xlim=(float(window[0]), float(window[1])),
        annotations=_stim_markers_rel(request),
        status="ok",
        row_kinds=["raster"],
    )


def _prepare_trial_rate(request: Any) -> PanelSpec:
    window = _spike_window(request)
    if window is None:
        return _placeholder(
            title=f"{request.channel_name} — Taux / essai",
            message="Section indisponible.",
            status="unavailable",
        )
    trigger_index = _spike_trigger_index(request)
    trains = _spike_trains(request, trigger_index)
    dur = max(1e-9, float(window[1]) - float(window[0]))
    lw = _line_width(request)
    series: list[SeriesItem] = []
    for index, st_per_trial in enumerate(trains):
        rates: list[float] = []
        for trial in st_per_trial:
            arr = np.asarray(trial, dtype=np.float64).ravel()
            if window is not None and arr.size:
                arr = arr[(arr >= window[0]) & (arr <= window[1])]
            rates.append(float(arr.size) / dur)
        if not rates:
            continue
        x = np.arange(1, len(rates) + 1, dtype=np.float64)
        series.append(
            CurveSeries(
                x=x,
                y=np.asarray(rates, dtype=np.float64),
                color=str(request.colors[index % len(request.colors)]),
                label=_series_label(request, index),
                linewidth=lw,
            )
        )
    if not series:
        return _placeholder(
            title=f"{request.channel_name} — Taux / essai",
            message="Aucun spike dans cette fenêtre.",
            status="empty",
        )
    return PanelSpec(
        title=f"{request.channel_name} — {panel_label(request.panel)}",
        xlabel="Essai",
        ylabel="Taux (Hz)",
        series=series,
        status="ok",
        row_kinds=["trial_rate"],
    )


def _prepare_spike_overlay(request: Any) -> PanelSpec:
    window = request.section_window()
    windows = request.per_recording_windows()
    lw = max(0.5, _line_width(request) * 0.7)
    series: list[SeriesItem] = []
    sampling = max(1, int(getattr(request.settings, "sampling_percent", 100) or 100))
    settings = getattr(request, "settings", None)
    pre_ms = float(getattr(settings, "spike_overlay_pre_ms", 2.0) or 2.0)
    post_ms = float(getattr(settings, "spike_overlay_post_ms", 4.0) or 4.0)
    t_min_ms = -max(0.0, pre_ms)
    t_max_ms = max(1e-6, post_ms)
    abs_peak = 0.0
    for index, recording in enumerate(request.recordings):
        t_range = windows[index] if index < len(windows) else window
        t_ms, waves, _times, _total, _mean = recording.overlay_for_channel(
            request.channel_index, t_range_s=t_range
        )
        if waves is None or getattr(waves, "shape", (0,))[0] == 0:
            continue
        n = int(waves.shape[0])
        step = max(1, int(np.ceil(100.0 / sampling))) if sampling < 100 else 1
        color = str(request.colors[index % len(request.colors)])
        label = _series_label(request, index)
        labeled = False
        t_arr = np.asarray(t_ms, dtype=np.float64)
        for i in range(0, n, step):
            y_arr = np.asarray(waves[i], dtype=np.float64)
            if y_arr.size:
                peak = float(np.nanmax(np.abs(y_arr)))
                if np.isfinite(peak):
                    abs_peak = max(abs_peak, peak)
            series.append(
                CurveSeries(
                    x=t_arr,
                    y=y_arr,
                    color=color,
                    label=label if not labeled else "",
                    linewidth=lw,
                )
            )
            labeled = True
        try:
            thr = float(recording.threshold_uv(request.channel_index))
            if np.isfinite(thr):
                abs_peak = max(abs_peak, abs(thr))
        except Exception:
            pass
    if not series:
        return _placeholder(
            title=f"{request.channel_name} — Spike overlay",
            message="Aucune forme d’onde de spike dans cette fenêtre.",
            status="empty",
        )
    # Même fenêtre / symétrie Y que le chemin matplotlib (évite un autoscale
    # en secondes héritées ou une échelle Y asymétrique).
    y_half = max(1.0, abs_peak) * 1.05
    return PanelSpec(
        title=f"{request.channel_name} — {panel_label('spike_overlay')}",
        xlabel="Temps (ms)",
        ylabel="Potential (µV)",
        series=series,
        xlim=(t_min_ms, t_max_ms),
        ylim=(-y_half, y_half),
        status="ok",
        row_kinds=["overlay"],
        show_legend=True,
    )


# ---------------------------------------------------------------------------
# Montage
# ---------------------------------------------------------------------------


_MONTAGE_STREAM_KINDS = frozenset({"raw", "hp", "lp"})
_MONTAGE_EXTRA_LABELS: dict[str, str] = {
    "rms": "RMS",
    "psth": "PSTH",
    "trial_rate": "FR",
    "raster": "RST",
    "isi": "ISI",
    "overlay": "WV",
}
_MONTAGE_EXTRA_COLORS: dict[str, str] = {
    "rms": "#7c3aed",
    "psth": "#b45309",
    "trial_rate": "#c2410c",
    "raster": "#0f766e",
    "isi": "#0369a1",
    "overlay": "#4d7c0f",
}


def _montage_xlabel(mode: str, sync_mode: str, last_kind: str) -> str:
    """Libellé X selon la dernière ligne (unités incompatibles entre kinds)."""
    if last_kind == "overlay":
        return "Temps relatif au spike (ms)"
    if last_kind == "isi":
        return "ISI (ms)"
    if last_kind == "trial_rate":
        return "Essai"
    if last_kind in _MONTAGE_STREAM_KINDS and mode == "continuous":
        return (
            "Temps relatif au trigger (s)"
            if sync_mode == "trigger"
            else "Temps (s)"
        )
    return "Temps relatif à la stimulation (s)"


def _prepare_montage_continuous(request: Any) -> PanelSpec:
    """Page channels only — uses request settings for visible / paged channels."""
    try:
        from panel_registry import (
            _analysis_trigger_index,
            _montage_continuous_channels,
            _montage_continuous_max_points,
            _montage_review_curve,
            _montage_review_kinds,
            _montage_review_mode,
            _montage_rms_kind,
            _relative_stim_window,
        )
    except Exception:
        return _prepare_placeholder(
            request, "Helpers montage indisponibles.", status="unavailable"
        )

    recordings = request.recordings
    if not recordings:
        return _prepare_placeholder(request, "Aucun enregistrement.", status="unavailable")
    reference = recordings[0]
    if not any(rec.has_streams for rec in recordings):
        return _prepare_placeholder(
            request, "Flux continus indisponibles.", status="unavailable"
        )

    mode = _montage_review_mode(request)
    n_channels = min(int(rec.n_channels) for rec in recordings)
    channel_indices = _montage_continuous_channels(request, n_channels)
    kinds = _montage_review_kinds(request)
    max_pts = _montage_continuous_max_points(request)
    stim_index = int(request.settings.analysis.stim_index)
    sync_mode = request.settings.time_sync
    lw = max(0.6, _line_width(request) * 0.85)
    x_limits = request.settings.x_limits.as_tuple()
    has_extras = any(kind not in _MONTAGE_STREAM_KINDS for kind in kinds)

    series: list[SeriesItem] = []
    annotations: list[AnnotationSpec] = []
    row_labels: list[str] = []
    row_ylims: list[tuple[float, float] | None] = []
    row_xlims: list[tuple[float, float] | None] = []
    row_kinds: list[str] = []
    row_channels: list[int] = []
    row_keys: list[tuple[int, str]] = []
    row_highlight: list[bool] = []
    pending_any = False
    drawn = 0
    row = 0
    data_x_min: float | None = None
    data_x_max: float | None = None
    # Spikes / RMS : toujours t_rel (comme panel_registry._render_montage_continuous).
    extra_section_x = _relative_stim_window(request)
    stream_section_x = extra_section_x if mode != "continuous" else None
    stream_x = (
        x_limits
        if (mode == "continuous" and x_limits is not None)
        else stream_section_x
    )
    trigger_index = _analysis_trigger_index(request)
    rms_kind = _montage_rms_kind(request)

    for ch in channel_indices:
        name = (
            reference.channel_names[ch]
            if ch < len(reference.channel_names)
            else f"CH{ch}"
        )
        for kind in kinds:
            if kind in _MONTAGE_STREAM_KINDS:
                short = STREAM_SHORT_LABELS.get(kind, kind.upper())
            else:
                short = _MONTAGE_EXTRA_LABELS.get(kind, kind.upper())
            row_labels.append(f"{name} {short}")
            row_kinds.append(str(kind))
            row_channels.append(int(ch))
            row_keys.append((int(ch), str(kind)))
            row_highlight.append(int(ch) == int(request.channel_index))
            if kind in _MONTAGE_STREAM_KINDS:
                row_ylims.append(_ylim_tuple(request.settings, kind))
            elif kind == "rms":
                row_ylims.append(_ylim_tuple(request.settings, "rms"))
            elif kind == "raster":
                row_ylims.append((-0.7, 0.7))
            else:
                row_ylims.append(None)

            row_drawn = 0
            row_pending = False
            row_xlim: tuple[float, float] | None = None

            if kind in _MONTAGE_STREAM_KINDS:
                for index, recording in enumerate(recordings):
                    if kind in {"hp", "lp"} and not recording.stream_ready(kind, ch):
                        pending_any = True
                        row_pending = True
                        continue
                    offset = 0.0
                    abs_range = None
                    if mode == "continuous":
                        offset = continuous_sync_offset_s(
                            recording.stimulation_times_s(), sync_mode
                        )
                        if stream_x is not None:
                            abs_range = (
                                float(stream_x[0]) + offset,
                                float(stream_x[1]) + offset,
                            )
                    t, y = _montage_review_curve(
                        recording,
                        kind,
                        ch,
                        mode=mode,
                        stim_index=stim_index,
                        max_points=max_pts,
                        t_range=abs_range,
                    )
                    if y.size == 0:
                        if kind in {"hp", "lp"} and mode == "continuous":
                            pending_any = True
                            row_pending = True
                        continue
                    if mode == "continuous":
                        t_plot = np.asarray(t, dtype=np.float64) - offset
                    else:
                        t_plot = np.asarray(t, dtype=np.float64)
                        if stream_x is not None:
                            mask = _mask_window(t_plot, stream_x)
                            if mask is not None:
                                t_plot, y = t_plot[mask], y[mask]
                    if t_plot.size < 2:
                        continue
                    t0, t1 = float(t_plot[0]), float(t_plot[-1])
                    data_x_min = t0 if data_x_min is None else min(data_x_min, t0)
                    data_x_max = t1 if data_x_max is None else max(data_x_max, t1)
                    color = recording_trace_color(
                        request.colors,
                        index,
                        fallback=STREAM_PLOT_COLORS.get(kind, "#334155"),
                    )
                    series.append(
                        CurveSeries(
                            x=np.asarray(t_plot, dtype=np.float64),
                            y=np.asarray(y, dtype=np.float64),
                            color=str(color),
                            label=(
                                _series_label(request, index, stream=str(kind))
                                if row == 0
                                else ""
                            ),
                            linewidth=lw,
                            row=row,
                        )
                    )
                    row_drawn += 1
                    drawn += 1
                if stream_x is not None:
                    row_xlim = (float(stream_x[0]), float(stream_x[1]))
            elif kind == "rms":
                color_default = _MONTAGE_EXTRA_COLORS["rms"]
                for index, recording in enumerate(recordings):
                    if ch >= int(recording.n_channels):
                        continue
                    t, y = recording.rms_profile(rms_kind, ch)
                    if y.size == 0:
                        continue
                    t_plot = np.asarray(t, dtype=np.float64)
                    y_plot = np.asarray(y, dtype=np.float64)
                    if extra_section_x is not None:
                        mask = _mask_window(t_plot, extra_section_x)
                        if mask is not None:
                            t_plot, y_plot = t_plot[mask], y_plot[mask]
                    if t_plot.size < 2:
                        continue
                    color = recording_trace_color(
                        request.colors, index, fallback=color_default
                    )
                    series.append(
                        CurveSeries(
                            x=t_plot,
                            y=y_plot,
                            color=str(color),
                            label="",
                            linewidth=lw,
                            row=row,
                        )
                    )
                    row_drawn += 1
                    drawn += 1
                if extra_section_x is not None:
                    row_xlim = (float(extra_section_x[0]), float(extra_section_x[1]))
            elif kind == "overlay":
                color_default = _MONTAGE_EXTRA_COLORS["overlay"]
                for index, recording in enumerate(recordings):
                    if ch >= int(recording.n_channels):
                        continue
                    t_ms, waves, _times, _total, mean = recording.overlay_for_channel(
                        ch, t_range_s=extra_section_x
                    )
                    color = recording_trace_color(
                        request.colors, index, fallback=color_default
                    )
                    if mean is not None and np.asarray(mean).size and np.asarray(t_ms).size:
                        series.append(
                            CurveSeries(
                                x=np.asarray(t_ms, dtype=np.float64),
                                y=np.asarray(mean, dtype=np.float64),
                                color=str(color),
                                label="",
                                linewidth=lw,
                                row=row,
                            )
                        )
                        row_drawn += 1
                        drawn += 1
                    elif waves is not None and getattr(waves, "size", 0) and np.asarray(t_ms).size:
                        arr = np.asarray(waves, dtype=np.float64)
                        t_arr = np.asarray(t_ms, dtype=np.float64)
                        n_show = min(12, int(arr.shape[0]))
                        for wave in arr[:n_show]:
                            series.append(
                                CurveSeries(
                                    x=t_arr,
                                    y=np.asarray(wave, dtype=np.float64),
                                    color=str(color),
                                    label="",
                                    linewidth=max(0.4, lw * 0.55),
                                    row=row,
                                )
                            )
                        row_drawn += 1
                        drawn += 1
            elif kind in {"raster", "psth", "trial_rate", "isi"}:
                from draw_primitives import (
                    _isi_time_and_values_s,
                    _psth_mean_hz,
                    _trial_mean_firing_rate_hz,
                )

                color_default = _MONTAGE_EXTRA_COLORS.get(kind, "#334155")
                for index, recording in enumerate(recordings):
                    if ch >= int(recording.n_channels):
                        continue
                    trains = recording.spike_times(
                        ch, trigger_index=trigger_index, t_range_s=extra_section_x
                    )
                    color = recording_trace_color(
                        request.colors, index, fallback=color_default
                    )
                    if kind == "raster":
                        if not any(getattr(tr, "size", 0) for tr in trains):
                            continue
                        xs: list[float] = []
                        for trial in trains:
                            arr = np.asarray(trial, dtype=np.float64).ravel()
                            if arr.size == 0:
                                continue
                            xs.extend(arr.tolist())
                        if not xs:
                            continue
                        series.append(
                            ScatterSeries(
                                x=np.asarray(xs, dtype=np.float64),
                                y=np.zeros(len(xs), dtype=np.float64),
                                color=str(color),
                                size=2.5,
                                symbol="|",
                                row=row,
                            )
                        )
                        row_drawn += 1
                        drawn += 1
                    elif kind == "psth":
                        t_rel = np.asarray(recording.t_rel, dtype=np.float64)
                        if t_rel.size < 2 or extra_section_x is None:
                            continue
                        n_trials = max(
                            1,
                            len(trains)
                            if trains
                            else int(getattr(recording, "n_trials", 1) or 1),
                        )
                        centers, rate = _psth_mean_hz(
                            trains,
                            t_rel,
                            n_trials,
                            float(request.settings.psth_bin_window_s),
                            t_range_s=extra_section_x,
                        )
                        if centers.size < 2:
                            continue
                        series.append(
                            CurveSeries(
                                x=np.asarray(centers, dtype=np.float64),
                                y=np.asarray(rate, dtype=np.float64),
                                color=str(color),
                                label="",
                                linewidth=lw,
                                row=row,
                            )
                        )
                        row_drawn += 1
                        drawn += 1
                    elif kind == "trial_rate":
                        if extra_section_x is None or not trains:
                            continue
                        rates = _trial_mean_firing_rate_hz(trains, extra_section_x)
                        if rates.size == 0:
                            continue
                        x = np.arange(1, rates.size + 1, dtype=np.float64)
                        series.append(
                            CurveSeries(
                                x=x,
                                y=np.asarray(rates, dtype=np.float64),
                                color=str(color),
                                label="",
                                linewidth=lw,
                                row=row,
                            )
                        )
                        row_drawn += 1
                        drawn += 1
                    elif kind == "isi":
                        _t_isi, d_isi = _isi_time_and_values_s(
                            trains, isi_window_s=extra_section_x
                        )
                        if d_isi.size == 0:
                            continue
                        ms = np.asarray(d_isi, dtype=np.float64) * 1000.0
                        ms = ms[np.isfinite(ms) & (ms > 0)]
                        if ms.size == 0:
                            continue
                        bins = min(24, max(8, int(np.sqrt(ms.size))))
                        counts, edges = np.histogram(ms, bins=bins)
                        series.append(
                            HistSeries(
                                edges=np.asarray(edges, dtype=np.float64),
                                counts=np.asarray(counts, dtype=np.float64),
                                color=str(color),
                                label="",
                                row=row,
                            )
                        )
                        row_drawn += 1
                        drawn += 1
                if kind in {"psth", "raster"} and extra_section_x is not None:
                    row_xlim = (float(extra_section_x[0]), float(extra_section_x[1]))
            else:
                annotations.append(
                    AnnotationSpec(
                        kind="text",
                        text=f"{short} (aperçu limité)",
                        color="#64748b",
                        row=row,
                    )
                )

            if row_drawn == 0:
                if kind in _MONTAGE_STREAM_KINDS:
                    msg = "calcul…" if row_pending else "indisponible"
                else:
                    msg = "indisponible"
                annotations.append(
                    AnnotationSpec(
                        kind="text",
                        text=msg,
                        color="#64748b",
                        row=row,
                    )
                )
            row_xlims.append(row_xlim)
            row += 1

    n_rows = max(1, row)
    if drawn == 0:
        status = "pending" if pending_any else "empty"
        return _placeholder(
            title=panel_label(request.panel),
            message="Filtre en cours…" if pending_any else "Aucune courbe montage.",
            status=status,
        )

    last_kind = row_kinds[-1] if row_kinds else "raw"
    xlabel = _montage_xlabel(mode, str(sync_mode), last_kind)

    # X partagé seulement si toutes les lignes ont la même famille temporelle.
    if has_extras:
        shared_xlim = None
        # Flux : fenêtre stream ou étendue des courbes ; extras gardent leur propre X.
        if (
            data_x_min is not None
            and data_x_max is not None
            and data_x_max > data_x_min
        ):
            for i, kind in enumerate(row_kinds):
                if kind in _MONTAGE_STREAM_KINDS and row_xlims[i] is None:
                    row_xlims[i] = (data_x_min, data_x_max)
    elif stream_x is not None:
        shared_xlim = (float(stream_x[0]), float(stream_x[1]))
        row_xlims = [shared_xlim] * n_rows
    elif (
        data_x_min is not None
        and data_x_max is not None
        and data_x_max > data_x_min
    ):
        shared_xlim = (data_x_min, data_x_max)
        row_xlims = [shared_xlim] * n_rows
    else:
        shared_xlim = _union_x_range(series)
        if shared_xlim is not None:
            row_xlims = [shared_xlim] * n_rows

    # Pointillés de stim — alignés sur panel_registry._render_montage_continuous.
    if mode == "continuous" and bool(
        getattr(request.settings, "continuous_mark_stims", True)
    ):
        stim_xs: list[float] = []
        remaining = 60
        for index, recording in enumerate(recordings):
            if remaining <= 0:
                break
            stims = np.asarray(recording.stimulation_times_s(), dtype=np.float64)
            offset = continuous_sync_offset_s(stims, sync_mode)
            for stim_t in stims:
                if remaining <= 0:
                    break
                stim_x = float(stim_t) - float(offset)
                if stream_x is not None and not (stream_x[0] <= stim_x <= stream_x[1]):
                    continue
                stim_xs.append(stim_x)
                remaining -= 1
        if stim_xs:
            for row_i, kind in enumerate(row_kinds):
                if kind not in _MONTAGE_STREAM_KINDS:
                    continue
                for stim_x in stim_xs:
                    annotations.append(
                        AnnotationSpec(
                            kind="vline",
                            x=float(stim_x),
                            color="#dc2626",
                            row=row_i,
                        )
                    )
    elif mode != "continuous":
        marker_rows = [
            row_i
            for row_i, kind in enumerate(row_kinds)
            if kind in {"raw", "hp", "lp", "rms", "psth", "raster"}
        ]
        annotations.extend(_stim_markers_rel(request, rows=marker_rows))

    return PanelSpec(
        title=f"{panel_label(request.panel)} — page {len(channel_indices)} canaux",
        xlabel=xlabel,
        ylabel="",
        series=series,
        xlim=shared_xlim,
        annotations=annotations,
        status="ok",
        layout_kind="montage",
        n_rows=n_rows,
        row_labels=row_labels,
        row_ylims=row_ylims,
        row_xlims=row_xlims,
        row_kinds=row_kinds,
        row_channels=row_channels,
        row_keys=row_keys,
        row_highlight=row_highlight,
    )


def _prepare_montage_triggered(request: Any) -> PanelSpec:
    panel = str(request.panel)
    if panel not in _MONTAGE_TRIGGERED:
        return _prepare_placeholder(
            request, f"Montage inconnu : {panel}", status="unavailable"
        )
    stream, trigger_index = _MONTAGE_TRIGGERED[panel]
    try:
        from panel_registry import _montage_channel_slice
    except Exception:
        return _prepare_placeholder(request, "Helpers montage indisponibles.", status="unavailable")

    recordings = request.recordings
    reference = recordings[0]
    n_channels = min(int(rec.n_channels) for rec in recordings)
    channel_indices, _n_pages = _montage_channel_slice(request, n_channels)
    window = request.section_window()
    if window is None:
        t_rel = np.asarray(reference.t_rel, dtype=np.float64)
        if t_rel.size == 0:
            return _prepare_placeholder(request, "Section indisponible.", status="unavailable")
        window = (float(t_rel[0]), float(t_rel[-1]))
    max_pts = _max_points(request)
    lw = max(0.6, _line_width(request) * 0.85)
    default_color = STREAM_PLOT_COLORS.get(stream, "#334155")

    series: list[SeriesItem] = []
    row_labels: list[str] = []
    row_ylims: list[tuple[float, float] | None] = []
    row_kinds: list[str] = []
    row_channels: list[int] = []
    row_keys: list[tuple[int, str]] = []
    row_highlight: list[bool] = []
    drawn = 0

    for row, ch in enumerate(channel_indices):
        name = (
            reference.channel_names[ch]
            if ch < len(reference.channel_names)
            else f"CH{ch}"
        )
        row_labels.append(name)
        row_ylims.append(_ylim_tuple(request.settings, stream))
        row_kinds.append(stream)
        row_channels.append(int(ch))
        row_keys.append((int(ch), stream))
        row_highlight.append(int(ch) == int(request.channel_index))
        for index, recording in enumerate(recordings):
            if ch >= int(recording.n_channels):
                continue
            if trigger_index is None:
                curve = recording.mean(stream, ch)
            else:
                if trigger_index >= recording.n_trials:
                    continue
                curve = recording.trigger_window(trigger_index, stream, ch)
            if curve is None:
                continue
            t = np.asarray(recording.t_rel, dtype=np.float64)
            y = np.asarray(curve, dtype=np.float64)
            mask = _mask_window(t, window)
            if mask is not None:
                t, y = t[mask], y[mask]
            t, y = _decimate(t, y, max_pts)
            series.append(
                CurveSeries(
                    x=t,
                    y=y,
                    color=recording_trace_color(
                        request.colors, index, fallback=default_color
                    ),
                    label=(
                        _series_label(request, index, stream=stream)
                        if row == 0
                        else ""
                    ),
                    linewidth=lw,
                    row=row,
                )
            )
            drawn += 1

    if drawn == 0:
        return _prepare_placeholder(request, "Montage vide.", status="empty")

    n_rows = max(1, len(channel_indices))
    return PanelSpec(
        title=panel_label(panel),
        xlabel="Temps relatif à la stimulation (s)",
        series=series,
        xlim=(float(window[0]), float(window[1])),
        annotations=_stim_markers_rel(request, rows=range(n_rows)),
        status="ok",
        layout_kind="montage",
        n_rows=n_rows,
        row_labels=row_labels,
        row_ylims=row_ylims,
        row_kinds=row_kinds,
        row_channels=row_channels,
        row_keys=row_keys,
        row_highlight=row_highlight,
    )
