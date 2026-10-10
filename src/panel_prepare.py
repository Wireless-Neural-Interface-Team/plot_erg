"""Backend-neutral panel data preparation (screen path).

Builds :class:`PanelSpec` trees from a :class:`~panel_registry.RenderRequest`
without drawing. Screen backends (pyqtgraph) consume the specs; matplotlib
remains the PDF path via :mod:`panel_registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from display_config import STREAM_PLOT_COLORS
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
    row_kinds: list[str] = field(default_factory=list)
    row_channels: list[int] = field(default_factory=list)
    row_keys: list[tuple[int, str]] = field(default_factory=list)
    row_highlight: list[bool] = field(default_factory=list)
    status_message: str = ""
    grid: bool = True
    grid_alpha: float = 0.35
    show_legend: bool = True
    scale_bars: ScaleBarSpec = field(default_factory=ScaleBarSpec)


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
    return spec


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


def _decimate(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    from plot_utils import decimate_envelope

    return decimate_envelope(x, y, max_points)


def _line_width(request: Any) -> float:
    style = getattr(request, "style", None)
    return float(getattr(style, "line_width", 1.0) or 1.0) if style else 1.0


def _max_points(request: Any) -> int:
    style = getattr(request, "style", None)
    return max(200, int(getattr(style, "max_points_per_curve", 6000) or 6000))


def _series_label(request: Any, index: int, suffix: str = "") -> str:
    flags = list(getattr(request, "legend_flags", []) or [])
    labels = list(getattr(request, "labels", []) or [])
    if flags and index < len(flags) and not flags[index]:
        return ""
    label = labels[index] if index < len(labels) else f"#{index + 1}"
    if suffix and getattr(getattr(request, "settings", None), "texts", None) is not None:
        if bool(getattr(request.settings.texts, "show_series_suffix", True)):
            label = f"{label} — {suffix}"
    elif suffix:
        label = f"{label} — {suffix}"
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
    multi_rec = len(request.recordings) > 1
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
            t, values = recording.continuous_trace(
                stream,
                request.channel_index,
                max_points=max_pts,
                require_ready=str(stream) in {"hp", "lp"},
            )
            if values.size == 0:
                if str(stream) in {"hp", "lp"}:
                    pending_any = True
                continue
            stims = recording.stimulation_times_s()
            offset = continuous_sync_offset_s(stims, sync_mode)
            if window is not None:
                if window_is_absolute:
                    mask = _mask_window(t, window)
                else:
                    mask = _mask_window(
                        np.asarray(t, dtype=np.float64) - offset, window
                    )
                if mask is not None:
                    t = t[mask]
                    values = values[mask]
            t_plot = np.asarray(t, dtype=np.float64) - offset
            color = (
                request.colors[index]
                if multi_rec
                else STREAM_PLOT_COLORS.get(str(stream), request.colors[index])
            )
            label = _series_label(request, index, "continuous")
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
                label=_series_label(request, index, suffix),
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


def _stim_markers_rel(request: Any) -> list[AnnotationSpec]:
    out: list[AnnotationSpec] = []
    if not getattr(request.legend, "show_reference_markers", True):
        return out
    for marker in request.end_markers():
        out.append(AnnotationSpec(kind="vline", x=float(marker), color="#1d4ed8"))
    out.append(AnnotationSpec(kind="vline", x=0.0, color="#dc2626"))
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
    series: list[SeriesItem] = []
    y_cursor = 0.0
    for index, st_per_trial in enumerate(trains):
        xs: list[float] = []
        ys: list[float] = []
        for trial_i, trial in enumerate(st_per_trial):
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
        for i in range(0, n, step):
            series.append(
                CurveSeries(
                    x=np.asarray(t_ms, dtype=np.float64),
                    y=np.asarray(waves[i], dtype=np.float64),
                    color=color,
                    label=label if not labeled else "",
                    linewidth=lw,
                )
            )
            labeled = True
    if not series:
        return _placeholder(
            title=f"{request.channel_name} — Spike overlay",
            message="Aucune forme d’onde de spike dans cette fenêtre.",
            status="empty",
        )
    return PanelSpec(
        title=f"{request.channel_name} — {panel_label('spike_overlay')}",
        xlabel="Temps (ms)",
        ylabel="Potential (µV)",
        series=series,
        status="ok",
        row_kinds=["overlay"],
        show_legend=True,
    )


# ---------------------------------------------------------------------------
# Montage
# ---------------------------------------------------------------------------


def _prepare_montage_continuous(request: Any) -> PanelSpec:
    """Page channels only — uses request settings for visible / paged channels."""
    try:
        from panel_registry import (
            _montage_continuous_channels,
            _montage_continuous_max_points,
            _montage_review_curve,
            _montage_review_kinds,
            _montage_review_mode,
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
    max_pts = _montage_continuous_max_points(request, n_channels, len(kinds) or 1)
    stim_index = int(request.settings.analysis.stim_index)
    sync_mode = request.settings.time_sync
    multi_rec = len(recordings) > 1
    lw = max(0.6, _line_width(request) * 0.85)
    x_limits = request.settings.x_limits.as_tuple()

    series: list[SeriesItem] = []
    annotations: list[AnnotationSpec] = []
    row_labels: list[str] = []
    row_ylims: list[tuple[float, float] | None] = []
    row_kinds: list[str] = []
    row_channels: list[int] = []
    row_keys: list[tuple[int, str]] = []
    row_highlight: list[bool] = []
    pending_any = False
    drawn = 0
    row = 0
    shared_xlim: tuple[float, float] | None = None

    for ch in channel_indices:
        name = (
            reference.channel_names[ch]
            if ch < len(reference.channel_names)
            else f"CH{ch}"
        )
        for kind in kinds:
            if kind in {"raw", "hp", "lp"}:
                short = STREAM_SHORT_LABELS.get(kind, kind.upper())
            else:
                short = {
                    "rms": "RMS",
                    "psth": "PSTH",
                    "trial_rate": "FR",
                    "raster": "RST",
                    "isi": "ISI",
                    "overlay": "WV",
                }.get(kind, kind.upper())
            row_labels.append(f"{name} {short}")
            row_kinds.append(str(kind))
            row_channels.append(int(ch))
            row_keys.append((int(ch), str(kind)))
            row_highlight.append(int(ch) == int(request.channel_index))
            if kind in {"raw", "hp", "lp"}:
                row_ylims.append(_ylim_tuple(request.settings, kind))
            elif kind == "rms":
                row_ylims.append(_ylim_tuple(request.settings, "rms"))
            else:
                row_ylims.append(None)

            if kind in {"raw", "hp", "lp"}:
                for index, recording in enumerate(recordings):
                    if kind in {"hp", "lp"} and not recording.stream_ready(kind, ch):
                        pending_any = True
                        continue
                    t, y = _montage_review_curve(
                        recording,
                        kind,
                        ch,
                        mode=mode,
                        stim_index=stim_index,
                        max_points=max_pts,
                    )
                    if y.size == 0:
                        if kind in {"hp", "lp"} and mode == "continuous":
                            pending_any = True
                        continue
                    if mode == "continuous":
                        offset = continuous_sync_offset_s(
                            recording.stimulation_times_s(), sync_mode
                        )
                        t_plot = np.asarray(t, dtype=np.float64) - offset
                    else:
                        t_plot = np.asarray(t, dtype=np.float64)
                    if x_limits is not None and mode == "continuous":
                        mask = _mask_window(t_plot, x_limits)
                        if mask is not None:
                            t_plot, y = t_plot[mask], y[mask]
                    color = (
                        request.colors[index]
                        if multi_rec
                        else STREAM_PLOT_COLORS.get(kind, "#334155")
                    )
                    series.append(
                        CurveSeries(
                            x=np.asarray(t_plot, dtype=np.float64),
                            y=np.asarray(y, dtype=np.float64),
                            color=str(color),
                            label=_series_label(request, index) if row == 0 else "",
                            linewidth=lw,
                            row=row,
                        )
                    )
                    drawn += 1
                    if shared_xlim is None and t_plot.size:
                        shared_xlim = (float(t_plot[0]), float(t_plot[-1]))
            elif kind == "rms":
                rms_kind = "mean" if mode != "stimulation" else (
                    "first" if stim_index == 0 else "second"
                )
                for index, recording in enumerate(recordings):
                    t, y = recording.rms_profile(rms_kind, ch)
                    if y.size == 0:
                        continue
                    series.append(
                        CurveSeries(
                            x=np.asarray(t, dtype=np.float64),
                            y=np.asarray(y, dtype=np.float64),
                            color=str(request.colors[index % len(request.colors)]),
                            label="",
                            linewidth=lw,
                            row=row,
                        )
                    )
                    drawn += 1
            elif kind == "raster":
                for index, recording in enumerate(recordings):
                    trains = recording.spike_times(ch, trigger_index=None, t_range_s=None)
                    xs: list[float] = []
                    ys: list[float] = []
                    for ti, trial in enumerate(trains):
                        arr = np.asarray(trial, dtype=np.float64).ravel()
                        if arr.size == 0:
                            continue
                        xs.extend(arr.tolist())
                        ys.extend([float(ti)] * int(arr.size))
                    if xs:
                        series.append(
                            ScatterSeries(
                                x=np.asarray(xs, dtype=np.float64),
                                y=np.asarray(ys, dtype=np.float64),
                                color=str(request.colors[index % len(request.colors)]),
                                size=2.5,
                                symbol="|",
                                row=row,
                            )
                        )
                        drawn += 1
            else:
                annotations.append(
                    AnnotationSpec(
                        kind="text",
                        text=f"{short} (aperçu limité)",
                        color="#64748b",
                        row=row,
                    )
                )
            row += 1

    n_rows = max(1, row)
    if drawn == 0:
        status = "pending" if pending_any else "empty"
        return _placeholder(
            title=panel_label(request.panel),
            message="Filtre en cours…" if pending_any else "Aucune courbe montage.",
            status=status,
        )

    xlabel = (
        "Temps relatif au trigger (s)"
        if mode == "continuous" and sync_mode == "trigger"
        else ("Temps (s)" if mode == "continuous" else "Temps relatif à la stimulation (s)")
    )
    if x_limits is not None and mode == "continuous":
        shared_xlim = (float(x_limits[0]), float(x_limits[1]))

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
    multi_rec = len(recordings) > 1
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
                    color=str(request.colors[index] if multi_rec else default_color),
                    label=_series_label(request, index) if row == 0 else "",
                    linewidth=lw,
                    row=row,
                )
            )
            drawn += 1

    if drawn == 0:
        return _prepare_placeholder(request, "Montage vide.", status="empty")

    return PanelSpec(
        title=panel_label(panel),
        xlabel="Temps relatif à la stimulation (s)",
        series=series,
        xlim=(float(window[0]), float(window[1])),
        status="ok",
        layout_kind="montage",
        n_rows=max(1, len(channel_indices)),
        row_labels=row_labels,
        row_ylims=row_ylims,
        row_kinds=row_kinds,
        row_channels=row_channels,
        row_keys=row_keys,
        row_highlight=row_highlight,
    )
