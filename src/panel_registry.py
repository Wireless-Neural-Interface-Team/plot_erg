"""Draws any single panel onto a matplotlib figure, for live display or export.

The PDF exporter in :mod:`plotting` lays out dozens of panels on tall pages.
Here each panel is rendered on its own figure so the viewer can show, resize and
rearrange them independently, while reusing the exact same drawing primitives so
on-screen and PDF output stay consistent.

Every panel kind defined by the program is available: the 21 per-section panels,
the MEA layout and impedance panels, the three summary pages, and the
all-channel montages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np

from display_config import (
    CHANNEL_HIGHLIGHT_FACE,
    MUTED_AXIS_TEXT,
    STREAM_PLOT_COLORS,
)
from impedance_tracking import ImpedanceSession
from panel_catalog import (
    ANALYSIS_PANEL_FIELD_NAMES,
    EXTRA_CHANNEL_PANEL_FIELD_NAMES,
    GLOBAL_PANEL_FIELD_NAMES,
    SECTION_PANEL_FIELD_NAMES,
    panel_group,
    panel_label,
    panel_needs_spikes,
    preferred_height_px,
)
from processed_dataset import ProcessedRecording
from view_config import (
    SECTION_LABELS,
    STREAM_SHORT_LABELS,
    LegendSettings,
    PanelPlacement,
    PanelStyle,
    ViewerSettings,
    continuous_sync_offset_s,
)

PanelScope = Literal["channel", "global"]

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
    "full_recording": "raw",
}

_MONTAGE_SPECS: dict[str, tuple[str, int | None, str]] = {
    # panel key -> (stream, trigger index or None for trial average, colour)
    "montage_mean_raw": ("raw", None, STREAM_PLOT_COLORS["raw"]),
    "montage_mean_hp": ("hp", None, STREAM_PLOT_COLORS["hp"]),
    "montage_mean_lp": ("lp", None, STREAM_PLOT_COLORS["lp"]),
    "montage_second_raw": ("raw", 1, STREAM_PLOT_COLORS["raw"]),
    "montage_second_hp": ("hp", 1, STREAM_PLOT_COLORS["hp"]),
    "montage_second_lp": ("lp", 1, STREAM_PLOT_COLORS["lp"]),
    "montage_second_to_third_lp": ("lp", 1, STREAM_PLOT_COLORS["lp"]),
}

_UNAVAILABLE_FONT = 9.0


def _montage_line_width(style: PanelStyle) -> float:
    """Épaisseur compacte pour montages multi-lignes (dérivée du style)."""
    return max(0.6, float(style.line_width) * 0.67)


@dataclass(frozen=True)
class PanelInfo:
    """Static description of one panel kind, used to build the panel picker."""

    key: str
    label: str
    group: str
    scope: PanelScope = "channel"
    needs_spikes: bool = False
    needs_impedance: bool = False
    needs_streams: bool = False
    preferred_height_px: int = 300

    @property
    def is_global(self) -> bool:
        return self.scope == "global"


def _build_catalog() -> tuple[PanelInfo, ...]:
    panels: list[PanelInfo] = []
    for key in ANALYSIS_PANEL_FIELD_NAMES:
        panels.append(
            PanelInfo(
                key=key,
                label=panel_label(key),
                group=panel_group(key),
                needs_spikes=panel_needs_spikes(key),
                needs_streams=key == "full_recording",
                preferred_height_px=preferred_height_px(key),
            )
        )
    for key in SECTION_PANEL_FIELD_NAMES:
        panels.append(
            PanelInfo(
                key=key,
                label=panel_label(key),
                group=panel_group(key),
                scope="channel",
                needs_spikes=panel_needs_spikes(key),
                preferred_height_px=preferred_height_px(key),
            )
        )
    for key in EXTRA_CHANNEL_PANEL_FIELD_NAMES:
        panels.append(
            PanelInfo(
                key=key,
                label=panel_label(key),
                group=panel_group(key),
                needs_impedance=key == "impedance",
                preferred_height_px=preferred_height_px(key),
            )
        )
    for key in GLOBAL_PANEL_FIELD_NAMES:
        montage = key.startswith("montage_")
        panels.append(
            PanelInfo(
                key=key,
                label=panel_label(key),
                group=panel_group(key),
                scope="global",
                needs_impedance=key == "summary_impedance",
                needs_streams=montage and key.startswith("montage_second"),
                preferred_height_px=preferred_height_px(key),
            )
        )
    return tuple(panels)


PANEL_CATALOG: tuple[PanelInfo, ...] = _build_catalog()
PANEL_INFO_BY_KEY: dict[str, PanelInfo] = {info.key: info for info in PANEL_CATALOG}


def panel_info(key: str) -> PanelInfo:
    return PANEL_INFO_BY_KEY.get(
        key, PanelInfo(key=key, label=panel_label(key), group="Autre")
    )


@dataclass
class RenderRequest:
    """Everything one panel needs to draw itself."""

    placement: PanelPlacement
    recordings: list[ProcessedRecording]
    labels: list[str]
    colors: list[str]
    legend_flags: list[bool]
    channel_index: int
    channel_name: str
    settings: ViewerSettings
    probe_layout: Any | None = None
    impedance_sessions: list[ImpedanceSession] = field(default_factory=list)
    # Zones de zoom à surligner sur la vue complète (définies par l’utilisateur).
    # Chaque entrée : (t0_s, t1_s, label). Vide = aucune bande de zoom.
    highlight_zooms: tuple[tuple[float, float, str], ...] = ()
    # True = garder zoom/pan utilisateur après le redessin (ex. bascule Stim).
    preserve_view: bool = False

    @property
    def panel(self) -> str:
        return self.placement.panel

    @property
    def section(self) -> str:
        return self.placement.section

    @property
    def style(self) -> PanelStyle:
        return self.settings.style

    @property
    def legend(self) -> LegendSettings:
        return self.settings.legend

    @property
    def reference(self) -> ProcessedRecording | None:
        return self.recordings[0] if self.recordings else None

    def end_markers(self) -> list[float]:
        return [
            float(recording.end_marker_s)
            for recording in self.recordings
            if recording.end_marker_s is not None
        ]

    def section_window(self) -> tuple[float, float] | None:
        """Limites X de la section demandée, ou None si indisponible."""
        reference = self.reference
        if reference is None:
            return None
        # Zoom personnalisé défini sur le placement (prioritaire sur l’échelle X globale).
        if self.placement.has_custom_zoom:
            t0 = float(self.placement.zoom_t0_s)
            t1 = float(self.placement.zoom_t1_s)
            if t1 < t0:
                t0, t1 = t1, t0
            # Barres continues = temps absolu → convertir pour les panels d’analyse (t_rel).
            # Zooms saisis « relatifs à la stim » = déjà en t_rel.
            if str(self.panel).startswith("analysis_") and self.placement.zoom_absolute:
                stim_index = self.settings.analysis.trigger_index()
                converted = _absolute_range_to_relative(
                    reference, t0, t1, stim_index=stim_index
                )
                if converted is not None:
                    return converted
                t_rel = reference.t_rel
                if t_rel.size == 0:
                    return None
                return float(t_rel[0]), float(t_rel[-1])
            return t0, t1
        # Échelle X manuelle : s’applique à tous les graphs temporels sans zoom dédié.
        manual_x = self.settings.x_limits.as_tuple()
        if manual_x is not None:
            return manual_x
        t_rel = reference.t_rel
        if t_rel.size == 0:
            return None
        if self.section == "full":
            return float(t_rel[0]), float(t_rel[-1])
        if self.section == "zoom_onset":
            return self.settings.zoom_onset_window()
        markers = self.end_markers()
        if not markers:
            return None
        t0, t1 = self.settings.zoom_end_window()
        return float(min(markers) + t0), float(max(markers) + t1)

    def per_recording_windows(self) -> list[tuple[float, float] | None]:
        """Fenêtre de section résolue par enregistrement."""
        if (
            self.placement.has_custom_zoom
            and str(self.panel).startswith("analysis_")
            and self.placement.zoom_absolute
        ):
            t0 = float(self.placement.zoom_t0_s)
            t1 = float(self.placement.zoom_t1_s)
            stim_index = self.settings.analysis.trigger_index()
            out: list[tuple[float, float] | None] = []
            for recording in self.recordings:
                converted = _absolute_range_to_relative(
                    recording, t0, t1, stim_index=stim_index
                )
                if converted is not None:
                    out.append(converted)
                    continue
                t_rel = recording.t_rel
                out.append(
                    (float(t_rel[0]), float(t_rel[-1])) if t_rel.size else None
                )
            return out
        if self.placement.has_custom_zoom or self.section != "zoom_trigger_end":
            window = self.section_window()
            return [window for _ in self.recordings]
        t0, t1 = self.settings.zoom_end_window()
        out = []
        for recording in self.recordings:
            marker = recording.end_marker_s
            out.append(None if marker is None else (float(marker) + t0, float(marker) + t1))
        return out


# --------------------------------------------------------------------- helpers


def _appearance(request: RenderRequest):
    """Apparence de tracé alignée sur le PanelStyle / légende utilisateur."""
    from draw_primitives import DrawAppearance

    return DrawAppearance.from_panel_style(
        request.style,
        legend_font_size=float(request.legend.font_size),
    )


def _absolute_range_to_relative(
    recording: ProcessedRecording,
    t0_abs: float,
    t1_abs: float,
    *,
    stim_index: int | None = None,
) -> tuple[float, float] | None:
    """Convertit une plage absolue (barres) en fenêtre relative à une stimulation.

    Si ``stim_index`` est fourni (mode « une stimulation »), cette stim est la
    référence. Sinon : première stim dans la plage, ou stim la plus proche du centre.
    """
    stims = np.asarray(recording.stimulation_times_s(), dtype=np.float64).ravel()
    if stims.size == 0:
        return None
    lo, hi = float(t0_abs), float(t1_abs)
    if hi < lo:
        lo, hi = hi, lo
    if stim_index is not None and 0 <= int(stim_index) < stims.size:
        ref = float(stims[int(stim_index)])
    else:
        inside = stims[(stims >= lo) & (stims <= hi)]
        if inside.size:
            ref = float(inside[0])
        else:
            mid = 0.5 * (lo + hi)
            ref = float(stims[int(np.argmin(np.abs(stims - mid)))])
    return (lo - ref, hi - ref)


def _decimate(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope decimation: keeps peaks while bounding the point count."""
    from plot_utils import decimate_envelope

    return decimate_envelope(x, y, max_points)


def _plot_curve(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    style: PanelStyle,
    color: Any,
    label: str | None,
    line_width: float | None = None,
    max_points: int | None = None,
) -> Any | None:
    if x.size == 0 or y.size == 0:
        return None
    n = min(int(x.size), int(y.size))
    xd, yd = _decimate(
        np.asarray(x[:n], dtype=np.float64),
        np.asarray(y[:n], dtype=np.float64),
        int(max_points) if max_points is not None else style.max_points_per_curve,
    )
    (line,) = ax.plot(
        xd,
        yd,
        linewidth=line_width if line_width is not None else style.line_width,
        color=color,
        label=label if label else "_nolegend_",
        solid_joinstyle="round",
    )
    return line


def _set_curve_data(
    line: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_points: int,
    color: Any | None = None,
    line_width: float | None = None,
) -> None:
    if x.size == 0 or y.size == 0:
        line.set_data([], [])
        return
    n = min(int(x.size), int(y.size))
    xd, yd = _decimate(
        np.asarray(x[:n], dtype=np.float64),
        np.asarray(y[:n], dtype=np.float64),
        max_points,
    )
    line.set_data(xd, yd)
    if color is not None:
        line.set_color(color)
    if line_width is not None:
        line.set_linewidth(line_width)


def _mask_window(t: np.ndarray, window: tuple[float, float] | None) -> np.ndarray | None:
    if window is None:
        return None
    return (t >= float(window[0])) & (t <= float(window[1]))


def _unavailable(ax: Any, message: str, *, fontsize: float | None = None) -> None:
    from plot_utils import mark_unavailable_axis

    mark_unavailable_axis(
        ax,
        message,
        fontsize=float(fontsize) if fontsize is not None else _UNAVAILABLE_FONT,
    )


def _apply_axis_style(ax: Any, style: PanelStyle) -> None:
    ax.title.set_fontsize(style.title_font_size)
    ax.xaxis.label.set_fontsize(style.label_font_size)
    ax.yaxis.label.set_fontsize(style.label_font_size)
    inside = bool(style.ticks_inside)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=style.tick_font_size,
        direction="in" if inside else "out",
        top=inside,
        right=inside,
    )
    for text in ax.texts:
        if text.get_fontsize() > style.label_font_size * 1.6:
            text.set_fontsize(style.label_font_size)
    for spine in ax.spines.values():
        spine.set_visible(bool(style.show_borders))


def _dedupe(handles: Sequence[Any], labels: Sequence[str]) -> tuple[list[Any], list[str]]:
    seen: set[str] = set()
    out_handles: list[Any] = []
    out_labels: list[str] = []
    for handle, label in zip(handles, labels):
        text = str(label)
        if not text or text.startswith("_") or text in seen:
            continue
        seen.add(text)
        out_handles.append(handle)
        out_labels.append(text)
    return out_handles, out_labels


def _apply_legend(ax: Any, settings: LegendSettings) -> None:
    """Rebuild the legend according to the user's legend settings."""
    existing = ax.get_legend()
    handles: list[Any] = []
    labels: list[str] = []
    if existing is not None:
        handles = list(getattr(existing, "legend_handles", []) or [])
        labels = [text.get_text() for text in existing.get_texts()]
        existing.remove()
    if not handles:
        handles, labels = ax.get_legend_handles_labels()
    handles, labels = _dedupe(handles, labels)
    if not settings.visible or not handles:
        return
    gap = max(0.0, float(settings.gap))
    kwargs: dict[str, Any] = {
        "fontsize": settings.font_size,
        "ncol": max(1, int(settings.columns)),
        "frameon": bool(settings.frame),
        "framealpha": 0.92,
        "borderaxespad": gap if settings.location != "below" else 0.0,
    }
    if settings.location == "below":
        legend = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -gap),
            **kwargs,
        )
    else:
        legend = ax.legend(handles, labels, loc=settings.location, **kwargs)
    if legend is not None:
        legend.set_in_layout(True)


_ZOOM_SPAN_COLORS = ("#22c55e", "#eab308", "#38bdf8", "#a78bfa", "#fb7185")


def highlight_zooms_from_placements(
    placements: Sequence[PanelPlacement],
) -> tuple[tuple[float, float, str], ...]:
    """Extraire les fenêtres de zoom relatives (surlignage sur graphs d’analyse)."""
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float, str]] = []
    for placement in placements:
        if not placement.has_custom_zoom:
            continue
        # Surlignage en t_rel : ignorer les zooms absolus (trace continue).
        if placement.zoom_absolute:
            continue
        if not str(placement.panel).startswith("analysis_"):
            continue
        key = (float(placement.zoom_t0_s), float(placement.zoom_t1_s))
        if key in seen:
            continue
        seen.add(key)
        label = placement.zoom_label.strip() or f"Zoom [{key[0]:g} … {key[1]:g} s]"
        out.append((key[0], key[1], label))
    return tuple(out)


def _wants_zoom_spans(request: RenderRequest) -> bool:
    """Bandes de zoom uniquement sur la vue complète, jamais sur un panneau déjà zoomé."""
    return request.section == "full" and not request.placement.has_custom_zoom


def _reference_markers(ax: Any, request: RenderRequest, *, with_spans: bool) -> None:
    """Stimulation onset / offset lines, honouring the legend settings."""
    if not request.legend.show_reference_markers:
        return
    from draw_primitives import _draw_onset_offset_lines

    markers = request.end_markers()
    _draw_onset_offset_lines(
        ax,
        end_markers=markers,
        label_in_legend=True,
        appearance=_appearance(request),
    )
    if not with_spans:
        return
    for index, (t0, t1, label) in enumerate(request.highlight_zooms):
        color = _ZOOM_SPAN_COLORS[index % len(_ZOOM_SPAN_COLORS)]
        ax.axvspan(
            float(t0),
            float(t1),
            alpha=0.14,
            color=color,
            label=str(label or f"Zoom {index + 1}"),
        )


def _filter_suffix(recording: ProcessedRecording, stream: str, request: RenderRequest) -> str:
    if not request.legend.show_filter_details or stream == "raw":
        return ""
    kind = "highpass" if stream == "hp" else "lowpass"
    return f" ({recording.meta.filter_short_label(kind)})"


def _series_label(
    request: RenderRequest, index: int, suffix: str = "", *, stream: str | None = None
) -> str | None:
    if not request.legend_flags[index]:
        return None
    label = request.labels[index]
    if stream is not None:
        label += _filter_suffix(request.recordings[index], stream, request)
    if suffix:
        label += f" — {suffix}"
    return label


def _section_suffix(request: RenderRequest) -> str:
    if request.section == "full":
        return ""
    return f" — {SECTION_LABELS.get(request.section, request.section)}"


def _count_suffix(request: RenderRequest) -> str:
    if not request.legend.show_sample_counts:
        return ""
    counts = {recording.n_trials for recording in request.recordings}
    if not counts:
        return ""
    if len(counts) == 1:
        return f" (n={counts.pop()})"
    return " (n=" + "/".join(str(recording.n_trials) for recording in request.recordings) + ")"


def _apply_ylim(ax: Any, limits: Any) -> None:
    bounds = limits.as_tuple() if limits is not None else None
    if bounds is not None:
        ax.set_ylim(bounds[0], bounds[1])


def _apply_xlim(ax: Any, limits: Any) -> None:
    bounds = limits.as_tuple() if limits is not None else None
    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[1])


# ------------------------------------------------------------- trace renderers


def _render_mean_trace(ax: Any, request: RenderRequest) -> str:
    stream = _STREAM_FOR_PANEL[request.panel]
    window = request.section_window()
    if window is None:
        return _fail(ax, request, "Section unavailable (no stimulation-end marker).")
    drawn = 0
    for index, recording in enumerate(request.recordings):
        curve = recording.mean(stream, request.channel_index)
        if curve is None:
            continue
        t = recording.t_rel
        mask = _mask_window(t, window)
        x = t[mask] if mask is not None else t
        y = curve[mask] if mask is not None else curve
        _plot_curve(
            ax,
            x,
            y,
            style=request.style,
            color=request.colors[index],
            label=_series_label(request, index, "trial-averaged", stream=stream),
        )
        drawn += 1
    if drawn == 0:
        return _fail(ax, request, "Trial average unavailable for this channel.")
    _reference_markers(ax, request, with_spans=_wants_zoom_spans(request))
    ax.set_xlim(window[0], window[1])
    ax.set_xlabel("Time relative to stimulation (s)")
    ax.set_ylabel("Potential (µV)")
    _apply_ylim(ax, request.settings.trace_ylim)
    ax.set_title(
        f"{request.channel_name} — {panel_label(request.panel)}"
        f"{_section_suffix(request)}{_count_suffix(request)}"
    )
    return "ok"


def _render_trigger_trace(ax: Any, request: RenderRequest) -> str:
    stream = _STREAM_FOR_PANEL[request.panel]
    trigger_index = 1 if request.panel.startswith("second_") else 0
    window = request.section_window()
    if window is None:
        return _fail(ax, request, "Section unavailable (no stimulation-end marker).")
    ordinal = "Second" if trigger_index else "First"
    drawn = 0
    for index, recording in enumerate(request.recordings):
        curve = recording.trigger_window(trigger_index, stream, request.channel_index)
        if curve is None:
            continue
        t = recording.t_rel
        mask = _mask_window(t, window)
        x = t[mask] if mask is not None else t
        y = curve[mask] if mask is not None else curve
        _plot_curve(
            ax,
            x,
            y,
            style=request.style,
            color=request.colors[index],
            label=_series_label(request, index, f"{ordinal.lower()} stimulation", stream=stream),
            line_width=max(0.8, request.style.line_width * 0.9),
        )
        drawn += 1
    if drawn == 0:
        return _fail(
            ax,
            request,
            f"{ordinal} stimulation unavailable\n(fewer stimulations than required, "
            "or the processed dataset was exported without stimulation windows).",
        )
    _reference_markers(ax, request, with_spans=False)
    ax.set_xlim(window[0], window[1])
    ax.set_xlabel("Time relative to stimulation (s)")
    ax.set_ylabel("Potential (µV)")
    if stream == "hp":
        _apply_ylim(ax, request.settings.stim_hp_ylim)
    else:
        _apply_ylim(ax, request.settings.trace_ylim)
    ax.set_title(
        f"{request.channel_name} — {panel_label(request.panel)}{_section_suffix(request)}"
    )
    return "ok"


def _analysis_trigger_index(request: RenderRequest) -> int | None:
    return request.settings.analysis.trigger_index()


def _analysis_mode_label(request: RenderRequest) -> str:
    return request.settings.analysis.describe()


def _render_full_recording(figure: Any, request: RenderRequest) -> str:
    """Traces continues : un sous-graphique par flux (WIDE / HIGH / LOW)."""
    streams = list(request.settings.resolved_continuous_streams()) or ["raw"]
    window = request.section_window() if request.placement.has_custom_zoom else None
    if window is None:
        window = request.settings.x_limits.as_tuple()
    n_streams = max(1, len(streams))
    axes = figure.subplots(n_streams, 1, sharex=True, squeeze=False)[:, 0]
    max_pts = int(request.style.max_points_per_curve)
    multi_rec = len(request.recordings) > 1
    drawn = 0
    sync_mode = request.settings.time_sync
    zoom_note = ""
    if request.placement.has_custom_zoom:
        zoom_note = (
            f" — zoom [{request.placement.zoom_t0_s:g} … {request.placement.zoom_t1_s:g} s]"
        )
    elif window is not None:
        zoom_note = f" — X [{window[0]:g} … {window[1]:g} s]"
    stim_note = " — stimulations marked" if request.settings.continuous_mark_stims else ""
    sync_note = " — sync trigger" if sync_mode == "trigger" else ""
    time_label = (
        "Time relative to trigger (s)"
        if sync_mode == "trigger"
        else "Time (s)"
    )

    # Zoom depuis barres continues = temps fichier ; sinon coords d’affichage.
    window_is_absolute = bool(
        request.placement.has_custom_zoom and request.placement.zoom_absolute
    )
    ref_offset = 0.0
    if request.recordings:
        ref_offset = continuous_sync_offset_s(
            request.recordings[0].stimulation_times_s(), sync_mode
        )

    for row, stream in enumerate(streams):
        ax = axes[row]
        short = STREAM_SHORT_LABELS.get(str(stream), str(stream).upper())
        stream_drawn = 0
        for index, recording in enumerate(request.recordings):
            t, values = recording.continuous_trace(
                stream,
                request.channel_index,
                max_points=max_pts if window is None else None,
            )
            if values.size == 0:
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
            _plot_curve(
                ax,
                t_plot,
                values,
                style=request.style,
                color=color,
                label=_series_label(request, index, "continuous", stream=stream),
            )
            stream_drawn += 1
            drawn += 1
            if request.settings.continuous_mark_stims:
                stim_xs = []
                for stim_t in stims:
                    stim_plot = float(stim_t) - offset
                    if window is not None:
                        if window_is_absolute:
                            if not (window[0] <= float(stim_t) <= window[1]):
                                continue
                        elif not (window[0] <= stim_plot <= window[1]):
                            continue
                    stim_xs.append(stim_plot)
                if stim_xs:
                    from draw_primitives import mark_stim_times

                    mark_stim_times(
                        ax,
                        stim_xs,
                        appearance=_appearance(request),
                        alpha=0.55,
                    )
        if stream_drawn == 0:
            ax.text(
                0.5,
                0.5,
                f"{short} unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=request.style.tick_font_size,
                color=MUTED_AXIS_TEXT,
            )
        if window is not None:
            if window_is_absolute:
                ax.set_xlim(window[0] - ref_offset, window[1] - ref_offset)
            else:
                ax.set_xlim(window[0], window[1])
        ax.set_ylabel(f"{short}\n(µV)")
        _apply_ylim(ax, request.settings.trace_ylim)
        if request.style.grid:
            ax.grid(True, alpha=request.style.grid_alpha)
        else:
            ax.grid(False)
        _apply_legend(ax, request.legend)
        _apply_axis_style(ax, request.style)
        if row == 0:
            ax.set_title(
                f"{request.channel_name} — continuous {short}"
                f"{zoom_note}{stim_note}{sync_note}"
            )
        else:
            ax.set_title(f"{request.channel_name} — continuous {short}{zoom_note}")
        if row < n_streams - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(time_label)

    if drawn == 0:
        figure.clear()
        return _fail(
            figure.add_subplot(111),
            request,
            "Continuous recording unavailable.\n"
            "Process a .rhs file first (F5), then select a channel.",
        )
    return "ok"


def _render_analysis_trace(ax: Any, request: RenderRequest) -> str:
    stream = _STREAM_FOR_PANEL[request.panel]
    trigger_index = _analysis_trigger_index(request)
    window = request.section_window()
    if window is None:
        return _fail(ax, request, "Section unavailable (no stimulation-end marker).")
    mode_label = _analysis_mode_label(request)
    drawn = 0
    for index, recording in enumerate(request.recordings):
        if trigger_index is None:
            curve = recording.mean(stream, request.channel_index)
            series = "trial-averaged"
        else:
            if trigger_index >= recording.n_trials:
                continue
            curve = recording.trigger_window(trigger_index, stream, request.channel_index)
            series = f"stimulation #{trigger_index + 1}"
        if curve is None:
            continue
        t = recording.t_rel
        mask = _mask_window(t, window)
        x = t[mask] if mask is not None else t
        y = curve[mask] if mask is not None else curve
        _plot_curve(
            ax,
            x,
            y,
            style=request.style,
            color=request.colors[index],
            label=_series_label(request, index, series, stream=stream),
        )
        drawn += 1
    if drawn == 0:
        return _fail(
            ax,
            request,
            f"Analysis trace unavailable ({mode_label}).\n"
            "Compute the selected channel (F6) after configuring the analysis.",
        )
    _reference_markers(ax, request, with_spans=_wants_zoom_spans(request))
    ax.set_xlim(window[0], window[1])
    ax.set_xlabel("Time relative to stimulation (s)")
    ax.set_ylabel("Potential (µV)")
    if stream == "hp":
        _apply_ylim(ax, request.settings.stim_hp_ylim)
    else:
        _apply_ylim(ax, request.settings.trace_ylim)
    ax.set_title(
        f"{request.channel_name} — {panel_label(request.panel)} ({mode_label})"
        f"{_section_suffix(request)}{_count_suffix(request)}"
    )
    return "ok"


def _render_analysis_rms(ax: Any, request: RenderRequest) -> str:
    trigger_index = _analysis_trigger_index(request)
    kind = "mean" if trigger_index is None else ("first" if trigger_index == 0 else "second")
    # For stim index > 1, fall back to computing from the single-stim RMS kind when possible;
    # otherwise use "mean" profile clipped — prefer first/second for 0/1 and mean otherwise.
    if trigger_index is not None and trigger_index > 1:
        kind = "mean"
    window = request.section_window()
    if window is None:
        return _fail(ax, request, "Section unavailable (no stimulation-end marker).")
    mode_label = _analysis_mode_label(request)
    drawn = 0
    for index, recording in enumerate(request.recordings):
        t, values = recording.rms_profile(kind, request.channel_index)
        if values.size == 0:
            continue
        mask = _mask_window(t, window)
        _plot_curve(
            ax,
            t[mask] if mask is not None else t,
            values[mask] if mask is not None else values,
            style=request.style,
            color=request.colors[index],
            label=_series_label(request, index, f"RMS ({mode_label})", stream="hp"),
        )
        drawn += 1
    if drawn == 0:
        return _fail(ax, request, f"RMS unavailable ({mode_label}).")
    ax.set_xlim(window[0], window[1])
    ax.set_xlabel("Time relative to stimulation (s)")
    ax.set_ylabel("RMS (µV)")
    _apply_ylim(ax, request.settings.rms_ylim)
    ax.set_title(
        f"{request.channel_name} — Analysis RMS ({mode_label}){_section_suffix(request)}"
    )
    return "ok"


def _render_rms(ax: Any, request: RenderRequest) -> str:
    kind = {"rms": "mean", "first_rms": "first", "second_rms": "second"}[request.panel]
    window = request.section_window()
    if window is None:
        return _fail(ax, request, "Section unavailable (no stimulation-end marker).")
    drawn = 0
    for index, recording in enumerate(request.recordings):
        t, values = recording.rms_profile(kind, request.channel_index)
        if values.size == 0:
            continue
        mask = _mask_window(t, window)
        _plot_curve(
            ax,
            t[mask] if mask is not None else t,
            values[mask] if mask is not None else values,
            style=request.style,
            color=request.colors[index],
            label=_series_label(request, index, "RMS", stream="hp"),
        )
        drawn += 1
    if drawn == 0:
        return _fail(ax, request, "RMS profile unavailable for this channel.")
    reference = request.reference
    note = ""
    if reference is not None:
        note = f" — {reference.meta.filter_short_label('highpass')}, window {reference.meta.dsp.rms_window_s:g} s"
    ax.set_xlim(window[0], window[1])
    ax.set_xlabel("Time relative to stimulation (s)")
    ax.set_ylabel("RMS (µV)")
    _apply_ylim(ax, request.settings.rms_ylim)
    ax.set_title(f"{request.channel_name} — {panel_label(request.panel)}{_section_suffix(request)}{note}")
    return "ok"


# ------------------------------------------------------------- spike renderers


def _spike_trains_for(request: RenderRequest, trigger_index: int | None) -> list[list[np.ndarray]]:
    windows = request.per_recording_windows()
    trains: list[list[np.ndarray]] = []
    for index, recording in enumerate(request.recordings):
        window = windows[index]
        trains.append(
            recording.spike_times(
                request.channel_index,
                trigger_index=trigger_index,
                t_range_s=window,
            )
        )
    return trains


def _threshold_entries(request: RenderRequest) -> list[tuple[str, str]]:
    return [
        (request.labels[index], recording.threshold_caption(request.channel_index))
        for index, recording in enumerate(request.recordings)
    ]


def _render_spike_panel(ax: Any, request: RenderRequest) -> str:
    from draw_primitives import _draw_spike_panels_multi_channel

    panel = request.panel
    trigger_index = None
    if panel.startswith("first_"):
        trigger_index = 0
    elif panel.startswith("second_"):
        trigger_index = 1
    elif panel.startswith("analysis_"):
        trigger_index = _analysis_trigger_index(request)
    window = request.section_window()
    if window is None:
        return _fail(ax, request, "Section unavailable (no stimulation-end marker).")
    reference = request.reference
    if reference is None:
        return _fail(ax, request, "No recording selected.")
    trains = _spike_trains_for(request, trigger_index)
    if all(not any(arr.size for arr in rec) for rec in trains):
        return _fail(ax, request, "No spike detected in this window.")

    kind = panel
    if panel.endswith("psth"):
        kind = "psth"
    elif panel.endswith("raster"):
        kind = "raster"
    elif panel.endswith("isi"):
        kind = "isi"
    elif panel.endswith("trial_rate") or panel == "trial_rate" or panel == "analysis_trial_rate":
        kind = "trial_rate"
    axes = {"raster": None, "psth": None, "trial": None, "isi": None}
    if kind == "raster":
        axes["raster"] = ax
    elif kind == "psth":
        axes["psth"] = ax
    elif kind == "trial_rate":
        axes["trial"] = ax
    else:
        axes["isi"] = ax

    mode_note = ""
    if panel.startswith("analysis_"):
        mode_note = f" ({_analysis_mode_label(request)})"

    _draw_spike_panels_multi_channel(
        axes["raster"],
        axes["psth"],
        axes["trial"],
        axes["isi"],
        None,
        reference.t_rel,
        float(reference.meta.fs),
        float(reference.meta.spike_detection.threshold_uv),
        float(request.settings.psth_bin_window_s),
        request.labels,
        intan_dsp=reference.meta.dsp,
        t_range_s=window,
        spikes_per_recording=trains,
        sampling_percent=int(request.settings.sampling_percent),
        threshold_entries=_threshold_entries(request),
        legend_visible=request.legend_flags,
        show_raster=axes["raster"] is not None,
        show_psth=axes["psth"] is not None,
        show_trial_rate=axes["trial"] is not None,
        show_isi=axes["isi"] is not None,
        across_trials=trigger_index is None,
        colors=request.colors,
        appearance=_appearance(request),
    )
    if not ax.get_visible() or not ax.axison:
        return "empty"
    ax.set_title(
        f"{request.channel_name} — {panel_label(panel)}{mode_note}{_section_suffix(request)}"
    )
    return "ok"


def _render_spike_overlay(ax: Any, request: RenderRequest) -> str:
    from draw_primitives import _draw_spike_overlay_panel

    window = request.section_window()
    if window is None:
        return _fail(ax, request, "Section unavailable (no stimulation-end marker).")
    windows = request.per_recording_windows()
    overlays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    thresholds: list[float] = []
    for index, recording in enumerate(request.recordings):
        t_ms, waves, times, _total, _mean = recording.overlay_for_channel(
            request.channel_index, t_range_s=windows[index]
        )
        overlays.append((t_ms, waves, times))
        thresholds.append(recording.threshold_uv(request.channel_index))
    if not overlays or all(arr[1].shape[0] == 0 for arr in overlays):
        return _fail(ax, request, "No spike waveform to overlay in this window.")
    reference = request.reference
    _draw_spike_overlay_panel(
        ax,
        overlays,
        request.labels,
        t_range_s=None,
        sampling_percent=int(request.settings.sampling_percent),
        legend_visible=request.legend_flags,
        intan_dsp=reference.meta.dsp if reference is not None else None,
        pre_ms=float(request.settings.spike_overlay_pre_ms),
        post_ms=float(request.settings.spike_overlay_post_ms),
        thresholds_uv=thresholds,
        colors=request.colors,
        appearance=_appearance(request),
    )
    if not ax.axison:
        return "empty"
    ax.set_title(
        f"{request.channel_name} — {panel_label('spike_overlay')}{_section_suffix(request)}"
    )
    return "ok"


# ----------------------------------------------------------- context renderers


def _render_mea_layout(ax: Any, request: RenderRequest) -> str:
    from probe_layout import draw_probe_layout_on_axes, match_contact_index

    layout = request.probe_layout
    if layout is None:
        return _fail(ax, request, "No MEA probe loaded.\nSelect a probe JSON in the parameters.")
    if match_contact_index(layout, request.channel_name) is None:
        return _fail(
            ax, request, f"Channel {request.channel_name}\nis not mapped on this MEA layout."
        )
    draw_probe_layout_on_axes(
        ax,
        layout,
        request.channel_name,
        set_mea_title=False,
        title_fontsize=request.style.title_font_size,
        contact_label_font_min=4.0,
        contact_label_font_max=9.0,
        contact_label_font_scale=120.0,
    )
    ax.set_title(f"MEA layout — {request.channel_name}")
    return "ok"


def _render_impedance(ax: Any, request: RenderRequest) -> str:
    from draw_primitives import _draw_impedance_evolution_panel

    sessions = request.impedance_sessions
    if not sessions:
        return _fail(
            ax,
            request,
            "No impedance data.\nAdd a companion CSV next to the recording "
            "(same name as the parent folder).",
        )
    _draw_impedance_evolution_panel(
        ax, request.channel_name, sessions, appearance=_appearance(request)
    )
    if not ax.axison:
        return "empty"
    ax.set_title(f"Impedance |Z| @ 1 kHz — {request.channel_name}")
    return "ok"


# ----------------------------------------------------------- summary renderers


def _render_summary_rms(ax: Any, request: RenderRequest) -> str:
    drawn = 0
    for index, recording in enumerate(request.recordings):
        t, values = recording.mean_rms_across_channels("mean")
        if values.size == 0:
            continue
        _plot_curve(
            ax,
            t,
            values,
            style=request.style,
            color=request.colors[index],
            label=_series_label(request, index, "mean across channels", stream="hp"),
        )
        drawn += 1
    if drawn == 0:
        return _fail(ax, request, "RMS summary unavailable.")
    ax.set_xlabel("Time relative to stimulation (s)")
    ax.set_ylabel("Mean RMS across channels (µV)")
    ax.set_title("Mean RMS profile across all channels")
    return "ok"


def _render_summary_rms_table(ax: Any, request: RenderRequest) -> str:
    recordings = request.recordings
    if not recordings:
        return _fail(ax, request, "No recording selected.")
    reference = recordings[0]
    n_channels = min(recording.n_channels for recording in recordings)
    if n_channels == 0:
        return _fail(ax, request, "No channel available.")
    ax.set_axis_off()
    header = ["Channel"] + [request.labels[i] for i in range(len(recordings))]
    rows: list[list[str]] = []
    for ch in range(n_channels):
        name = (
            reference.channel_names[ch]
            if ch < len(reference.channel_names)
            else f"CH{ch}"
        )
        values = []
        for recording in recordings:
            _t, profile = recording.rms_profile("mean", ch)
            values.append(
                f"{float(np.nanmean(profile)):.2f}" if profile.size else "—"
            )
        rows.append([name] + values)
    highlight = request.channel_index
    table = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(max(5.0, request.style.tick_font_size - 1.0))
    table.scale(1.0, 1.08)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold")
        elif row - 1 == highlight:
            cell.set_facecolor("#fde68a")
        cell.set_linewidth(0.3)
    ax.set_title("Mean RMS per channel (µV)")
    return "ok"


def _render_summary_impedance(ax: Any, request: RenderRequest) -> str:
    import matplotlib.dates as mdates

    sessions = request.impedance_sessions
    if not sessions:
        return _fail(ax, request, "No impedance session available.")
    times = np.array([mdates.date2num(session.when) for session in sessions], dtype=np.float64)
    means: list[float] = []
    for session in sessions:
        values = np.asarray(list(session.magnitudes_ohm.values()), dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        means.append(float(np.mean(values)) if values.size else float("nan"))
    magnitudes = np.asarray(means, dtype=np.float64)
    valid = np.isfinite(magnitudes) & (magnitudes > 0)
    if not np.any(valid):
        return _fail(ax, request, "No valid mean impedance across sessions.")
    ax.semilogy(
        times[valid],
        magnitudes[valid],
        marker="o",
        markersize=4,
        linewidth=max(0.8, float(request.style.line_width) * 0.85),
    )
    ax.set_ylabel("Mean |Z| @ 1 kHz (Ω)")
    ax.set_xlabel("Session time")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_title("Recording-mean impedance |Z| @ 1 kHz")
    return "ok"


# ----------------------------------------------------------- montage renderers


def _montage_visible_indices(request: RenderRequest, n_channels: int) -> list[int]:
    """Indices de canaux visibles (ordre d’origine, hors canaux masqués)."""
    reference = request.reference
    names = list(getattr(reference, "channel_names", ())) if reference is not None else []
    hidden = {str(name) for name in request.settings.hidden_channels}
    indices: list[int] = []
    for ch in range(max(0, int(n_channels))):
        name = names[ch] if ch < len(names) else f"CH{ch}"
        if str(name) in hidden:
            continue
        indices.append(ch)
    return indices


def _montage_page_index(request: RenderRequest) -> int:
    """Page de montage : ``instance_id=pageN`` (legacy) ou réglage global (PDF)."""
    iid = str(getattr(request.placement, "instance_id", "") or "")
    if iid.startswith("page"):
        try:
            return max(0, int(iid[4:]))
        except ValueError:
            pass
    return int(request.settings.montage_page)


def _montage_channel_slice(request: RenderRequest, n_channels: int) -> tuple[list[int], int]:
    """Canaux à dessiner + nombre de pages (PDF / legacy ``pageN``)."""
    visible = _montage_visible_indices(request, n_channels)
    per_page = max(1, int(request.settings.montage_channels))
    n_pages = max(1, (len(visible) + per_page - 1) // per_page) if visible else 1
    page = _montage_page_index(request)
    if page < 0 and visible:
        try:
            pos = visible.index(int(request.channel_index))
        except ValueError:
            pos = 0
        page = pos // per_page
    page = max(0, min(n_pages - 1, page))
    start = page * per_page
    return visible[start : start + per_page], n_pages


def _montage_continuous_channels(request: RenderRequest, n_channels: int) -> list[int]:
    """Revue GUI : tous les canaux visibles. Legacy ``pageN`` : une tranche."""
    iid = str(getattr(request.placement, "instance_id", "") or "")
    if iid.startswith("page"):
        channel_indices, _n_pages = _montage_channel_slice(request, n_channels)
        return channel_indices
    return _montage_visible_indices(request, n_channels)


def _montage_files_caption(request: RenderRequest) -> str:
    """Libellé court pour le titre : 1 fichier, ou superposition de N."""
    labels = [str(label) for label in request.labels if str(label).strip()]
    if not labels:
        reference = request.reference
        return reference.label if reference is not None else "—"
    if len(labels) == 1:
        return labels[0]
    if len(labels) <= 3:
        return " ⊕ ".join(labels)
    return f"{labels[0]} ⊕ +{len(labels) - 1} fichiers"


def _montage_triggered_geom(
    request: RenderRequest, channel_indices: Sequence[int], window: tuple[float, float]
) -> tuple[Any, ...]:
    return (
        request.panel,
        tuple(int(ch) for ch in channel_indices),
        len(request.recordings),
        window,
        int(request.style.max_points_per_curve),
    )


def _iter_montage_triggered_xy(
    request: RenderRequest,
    ch: int,
    stream: str,
    trigger_index: int | None,
    window: tuple[float, float],
) -> list[tuple[np.ndarray, np.ndarray, Any]]:
    """[(x, y, color), ...] for one montage row."""
    multi_rec = len(request.recordings) > 1
    _, _, default_color = _MONTAGE_SPECS[request.panel]
    out: list[tuple[np.ndarray, np.ndarray, Any]] = []
    for index, recording in enumerate(request.recordings):
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
        t = recording.t_rel
        mask = _mask_window(t, window)
        x = t[mask] if mask is not None else t
        y = curve[mask] if mask is not None else curve
        line_color = request.colors[index] if multi_rec else default_color
        out.append((x, y, line_color))
    return out


def _try_update_triggered_montage(figure: Any, request: RenderRequest) -> str | None:
    """Reuse axes + set_data when montage geometry is unchanged."""
    stored = getattr(figure, "_erg_montage_state", None)
    if not isinstance(stored, dict) or stored.get("kind") != "triggered":
        return None
    stream, trigger_index, _color = _MONTAGE_SPECS[request.panel]
    recordings = request.recordings
    if not recordings:
        return None
    reference = recordings[0]
    n_channels = min(int(recording.n_channels) for recording in recordings)
    channel_indices, n_pages = _montage_channel_slice(request, n_channels)
    window = request.section_window()
    if window is None:
        window = (float(reference.t_rel[0]), float(reference.t_rel[-1]))
    geom = _montage_triggered_geom(request, channel_indices, window)
    if stored.get("geom") != geom:
        return None
    lines_by_row: list[list[Any]] = stored.get("lines") or []
    axes = list(figure.axes)
    if len(axes) != len(channel_indices) or len(lines_by_row) != len(channel_indices):
        return None
    max_pts = int(request.style.max_points_per_curve)
    drawn = 0
    for row, ch in enumerate(channel_indices):
        ax = axes[row]
        series = _iter_montage_triggered_xy(request, ch, stream, trigger_index, window)
        row_lines = lines_by_row[row]
        # Rebuild this row if the number of series changed.
        if len(series) != len(row_lines):
            return None
        for line, (x, y, color) in zip(row_lines, series):
            _set_curve_data(
                line,
                x,
                y,
                max_points=max_pts,
                color=color,
                line_width=_montage_line_width(request.style),
            )
            drawn += 1
        ax.set_xlim(window[0], window[1])
        if stream == "hp":
            _apply_ylim(ax, request.settings.stim_hp_ylim)
        else:
            _apply_ylim(ax, request.settings.trace_ylim)
        ax.set_facecolor(
            CHANNEL_HIGHLIGHT_FACE if ch == request.channel_index else "white"
        )
        if request.style.grid:
            ax.grid(True, alpha=request.style.grid_alpha)
        else:
            ax.grid(False)
    page_note = f"page {int(request.settings.montage_page) + 1}/{n_pages}" if n_pages > 1 else ""
    if channel_indices:
        span = f"channels {channel_indices[0] + 1}–{channel_indices[-1] + 1}"
        if len(channel_indices) < n_channels:
            span += f" ({len(channel_indices)} visibles)"
    else:
        span = "no visible channels"
    figure.suptitle(
        f"{panel_label(request.panel)} — {_montage_files_caption(request)} — {span}"
        + (f" ({page_note})" if page_note else ""),
        fontsize=request.style.title_font_size,
    )
    del n_pages
    return "ok" if drawn else "empty"


def _render_montage(figure: Any, request: RenderRequest) -> str:
    if request.panel == "montage_continuous_raw":
        return _render_montage_continuous(figure, request)
    stream, trigger_index, color = _MONTAGE_SPECS[request.panel]
    recordings = request.recordings
    if not recordings:
        return _fail(figure.add_subplot(111), request, "No recording selected.")
    reference = recordings[0]
    multi_rec = len(recordings) > 1
    n_channels = min(int(recording.n_channels) for recording in recordings)
    channel_indices, n_pages = _montage_channel_slice(request, n_channels)
    rows = max(1, len(channel_indices))
    window = request.section_window()
    if window is None:
        window = (float(reference.t_rel[0]), float(reference.t_rel[-1]))

    try:
        figure.set_layout_engine(None)
    except Exception:
        try:
            figure.set_constrained_layout(False)
        except Exception:
            pass

    from draw_primitives import _draw_onset_offset_lines

    axes = figure.subplots(rows, 1, sharex=True, squeeze=False)[:, 0]
    drawn = 0
    lines_by_row: list[list[Any]] = []
    draw_app = _appearance(request)
    for row, ch in enumerate(channel_indices):
        ax = axes[row]
        name = reference.channel_names[ch] if ch < len(reference.channel_names) else f"CH{ch}"
        channel_drawn = 0
        row_lines: list[Any] = []
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
            t = recording.t_rel
            mask = _mask_window(t, window)
            x = t[mask] if mask is not None else t
            y = curve[mask] if mask is not None else curve
            line_color = request.colors[index] if multi_rec else color
            # Légende uniquement sur la 1re ligne (évite N_axes copies).
            series_label = (
                _series_label(request, index, stream=stream) if row == 0 else None
            )
            line = _plot_curve(
                ax,
                x,
                y,
                style=request.style,
                color=line_color,
                label=series_label,
                line_width=_montage_line_width(request.style),
            )
            if line is not None:
                row_lines.append(line)
            channel_drawn += 1
            drawn += 1
        lines_by_row.append(row_lines)
        if channel_drawn == 0:
            ax.text(
                0.5,
                0.5,
                "unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=request.style.tick_font_size,
                color=MUTED_AXIS_TEXT,
            )
        ax.set_ylabel(
            name,
            rotation=0,
            ha="right",
            va="center",
            fontsize=max(5.0, request.style.tick_font_size - 1.0),
        )
        ax.tick_params(
            axis="both",
            labelsize=max(5.0, request.style.tick_font_size - 1.5),
            direction="in" if request.style.ticks_inside else "out",
            top=bool(request.style.ticks_inside),
            right=bool(request.style.ticks_inside),
        )
        if request.style.grid:
            ax.grid(True, alpha=request.style.grid_alpha)
        else:
            ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(bool(request.style.show_borders))
        _draw_onset_offset_lines(
            ax,
            end_markers=request.end_markers(),
            label_in_legend=False,
            appearance=draw_app,
        )
        ax.set_xlim(window[0], window[1])
        if stream == "hp":
            _apply_ylim(ax, request.settings.stim_hp_ylim)
        else:
            _apply_ylim(ax, request.settings.trace_ylim)
        if ch == request.channel_index:
            ax.set_facecolor(CHANNEL_HIGHLIGHT_FACE)
        if row == 0 and multi_rec:
            _apply_legend(ax, request.legend)
        if row < rows - 1:
            ax.set_xticklabels([])
    axes[-1].set_xlabel("Time relative to stimulation (s)")
    page_note = f"page {int(request.settings.montage_page) + 1}/{n_pages}" if n_pages > 1 else ""
    if channel_indices:
        span = f"channels {channel_indices[0] + 1}–{channel_indices[-1] + 1}"
        if len(channel_indices) < n_channels:
            span += f" ({len(channel_indices)} visibles)"
    else:
        span = "no visible channels"
    figure.suptitle(
        f"{panel_label(request.panel)} — {_montage_files_caption(request)} — {span}"
        + (f" ({page_note})" if page_note else ""),
        fontsize=request.style.title_font_size,
    )
    figure._erg_montage_state = {
        "kind": "triggered",
        "geom": _montage_triggered_geom(request, channel_indices, window),
        "lines": lines_by_row,
    }
    return "ok" if drawn else "empty"


def _render_montage_continuous(figure: Any, request: RenderRequest) -> str:
    """Traces continues : un sous-graphique par canal×flux, empilés sans espace.

    Rendu type « scope » : axe Y réel, bordures jointives, surbrillance du canal
    sélectionné, superposition multi-enregistrements.
    """
    recordings = request.recordings
    if not recordings:
        return _fail(figure.add_subplot(111), request, "No recording selected.")
    reference = recordings[0]
    if not any(recording.has_streams for recording in recordings):
        return _fail(
            figure.add_subplot(111),
            request,
            "Continuous streams unavailable.\nProcess a .rhs file (F5).",
        )
    # constrained layout + N axes = freeze ; forcer un layout libre.
    try:
        figure.set_layout_engine(None)
    except Exception:
        try:
            figure.set_constrained_layout(False)
        except Exception:
            pass

    multi_rec = len(recordings) > 1
    n_channels = min(int(recording.n_channels) for recording in recordings)
    channel_indices = _montage_continuous_channels(request, n_channels)
    streams = list(request.settings.resolved_continuous_streams()) or ["raw"]
    stream_colors = dict(STREAM_PLOT_COLORS)

    row_specs: list[tuple[int, str, str]] = []
    for ch in channel_indices:
        name = reference.channel_names[ch] if ch < len(reference.channel_names) else f"CH{ch}"
        for stream in streams:
            short = STREAM_SHORT_LABELS.get(stream, stream.upper())
            row_specs.append((ch, stream, f"{name} {short}"))
    if not row_specs:
        return _fail(
            figure.add_subplot(111),
            request,
            "No visible channels.\nUnhide channels in Session → Channels.",
        )
    rows = len(row_specs)

    min_h = max(36, int(request.settings.montage_row_min_height_px))
    fig_h = max(3.0, (rows * min_h) / 96.0)
    try:
        figure.set_size_inches(max(figure.get_figwidth(), 6.0), fig_h, forward=False)
    except Exception:
        pass

    # Budget points : assez dense pour lire les stims, plafonné pour N canaux.
    style_cap = max(200, int(request.style.max_points_per_curve))
    max_pts = min(style_cap, max(240, 10000 // max(1, rows)))
    line_w = max(0.6, float(request.style.line_width) * 0.85)
    tick_fs = max(5.0, float(request.style.tick_font_size) - 1.5)
    label_fs = max(5.0, float(request.style.tick_font_size) - 1.0)
    inside = bool(request.style.ticks_inside)

    axes = figure.subplots(rows, 1, sharex=True, squeeze=False)[:, 0]
    figure.subplots_adjust(left=0.15, right=0.995, top=0.995, bottom=0.035, hspace=0.0)

    sync_mode = request.settings.time_sync
    offsets = [
        continuous_sync_offset_s(recording.stimulation_times_s(), sync_mode)
        for recording in recordings
    ]
    selected = int(request.channel_index)
    x_window = request.settings.x_limits.as_tuple()
    drawn = 0
    data_x_min: float | None = None
    data_x_max: float | None = None

    # Marqueurs stim précalculés (1×) — évite N_axes × N_stims × N_rec.
    from draw_primitives import mark_stim_times

    draw_app = _appearance(request)
    stim_xs: list[float] = []
    if request.settings.continuous_mark_stims:
        remaining = 60
        for index, recording in enumerate(recordings):
            if remaining <= 0:
                break
            stims = np.asarray(recording.stimulation_times_s(), dtype=np.float64)
            offset = float(offsets[index]) if index < len(offsets) else 0.0
            for stim_t in stims:
                if remaining <= 0:
                    break
                stim_x = float(stim_t) - offset
                if x_window is not None and not (x_window[0] <= stim_x <= x_window[1]):
                    continue
                stim_xs.append(stim_x)
                remaining -= 1

    for row_i, (ch, stream, label) in enumerate(row_specs):
        ax = axes[row_i]
        channel_drawn = 0
        for index, recording in enumerate(recordings):
            if ch >= int(recording.n_channels) or not recording.has_streams:
                continue
            t, values = recording.continuous_trace(stream, ch, max_points=max_pts)
            if values.size == 0:
                continue
            offset = float(offsets[index]) if index < len(offsets) else 0.0
            t_arr = np.asarray(t, dtype=np.float64) - offset
            y_arr = np.asarray(values, dtype=np.float64)
            if x_window is not None:
                mask = _mask_window(t_arr, x_window)
                if mask is not None:
                    t_arr = t_arr[mask]
                    y_arr = y_arr[mask]
            if t_arr.size < 2:
                continue
            t0, t1 = float(t_arr[0]), float(t_arr[-1])
            data_x_min = t0 if data_x_min is None else min(data_x_min, t0)
            data_x_max = t1 if data_x_max is None else max(data_x_max, t1)
            color = (
                request.colors[index] if multi_rec else stream_colors.get(stream, "#334155")
            )
            series_label = (
                _series_label(request, index, stream=stream) if row_i == 0 else None
            )
            _plot_curve(
                ax,
                t_arr,
                y_arr,
                style=request.style,
                color=color,
                label=series_label,
                line_width=line_w,
            )
            channel_drawn += 1
            drawn += 1

        if channel_drawn == 0:
            ax.text(
                0.5,
                0.5,
                "unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=request.style.tick_font_size,
                color=MUTED_AXIS_TEXT,
            )

        if stim_xs:
            mark_stim_times(
                ax,
                stim_xs,
                appearance=draw_app,
                alpha=0.40,
                linewidth=0.55,
            )

        ax.set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=label_fs,
            labelpad=8,
        )
        ax.tick_params(
            axis="both",
            labelsize=tick_fs,
            direction="in" if inside else "out",
            top=inside,
            right=inside,
            labelbottom=(row_i == rows - 1),
        )
        if request.style.grid:
            ax.grid(True, alpha=request.style.grid_alpha)
            ax.set_axisbelow(True)
        else:
            ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(bool(request.style.show_borders))
            spine.set_linewidth(0.8)
        _apply_ylim(ax, request.settings.trace_ylim)
        if ch == selected:
            ax.set_facecolor(CHANNEL_HIGHLIGHT_FACE)
        if row_i == 0 and multi_rec:
            _apply_legend(ax, request.legend)
        if row_i < rows - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(
                "Time relative to trigger (s)" if sync_mode == "trigger" else "Time (s)"
            )

    if x_window is not None:
        axes[-1].set_xlim(x_window[0], x_window[1])
    elif data_x_min is not None and data_x_max is not None and data_x_max > data_x_min:
        axes[-1].set_xlim(data_x_min, data_x_max)

    # Pas de suptitle : le titre du panneau Qt suffit.
    return "ok" if drawn else "empty"


# -------------------------------------------------------------------- dispatch


def _fail(ax: Any, request: RenderRequest, message: str) -> str:
    _unavailable(ax, message, fontsize=float(request.style.label_font_size))
    return "unavailable"


_AXES_RENDERERS: dict[str, Callable[[Any, RenderRequest], str]] = {
    "analysis_raw": _render_analysis_trace,
    "analysis_hp": _render_analysis_trace,
    "analysis_lp": _render_analysis_trace,
    "analysis_rms": _render_analysis_rms,
    "analysis_psth": _render_spike_panel,
    "analysis_trial_rate": _render_spike_panel,
    "analysis_isi": _render_spike_panel,
    "analysis_raster": _render_spike_panel,
    "analysis_overlay": _render_spike_overlay,
    "mean_raw": _render_mean_trace,
    "mean_hp": _render_mean_trace,
    "mean_lp": _render_mean_trace,
    "first_trigger_raw": _render_trigger_trace,
    "first_trigger_hp": _render_trigger_trace,
    "first_trigger_lp": _render_trigger_trace,
    "second_trigger_raw": _render_trigger_trace,
    "second_trigger_hp": _render_trigger_trace,
    "second_trigger_lp": _render_trigger_trace,
    "rms": _render_rms,
    "first_rms": _render_rms,
    "second_rms": _render_rms,
    "psth": _render_spike_panel,
    "first_psth": _render_spike_panel,
    "second_psth": _render_spike_panel,
    "isi": _render_spike_panel,
    "first_isi": _render_spike_panel,
    "second_isi": _render_spike_panel,
    "trial_rate": _render_spike_panel,
    "raster": _render_spike_panel,
    "spike_overlay": _render_spike_overlay,
    "mea_layout": _render_mea_layout,
    "impedance": _render_impedance,
    "summary_rms": _render_summary_rms,
    "summary_rms_table": _render_summary_rms_table,
    "summary_impedance": _render_summary_impedance,
}


def _structure_fingerprint(request: RenderRequest) -> tuple[Any, ...]:
    """Identity of the data geometry — style-only changes must not alter this."""
    rec_fp = tuple(
        (
            str(rec.meta.source_path),
            int(rec.data_generation),
            bool(rec.is_channel_ready(request.channel_index)),
        )
        for rec in request.recordings
    )
    window = request.section_window()
    return (
        request.panel,
        int(request.channel_index),
        str(request.section),
        rec_fp,
        int(request.style.max_points_per_curve),
        window,
        tuple(request.highlight_zooms),
        int(request.settings.montage_page),
        tuple(request.settings.resolved_continuous_streams())
        if request.panel == "montage_continuous_raw"
        else (),
        str(request.settings.time_sync),
        str(request.settings.analysis.mode),
        int(request.settings.analysis.trigger_index() or -1),
    )


def _refresh_panel_style(figure: Any, request: RenderRequest) -> str:
    """Update colors / fonts / grid / ylims on an existing figure (no rebuild)."""
    status = str(getattr(figure, "_erg_status", "ok") or "ok")
    colors = list(request.colors)
    lw = float(request.style.line_width)
    for ax in list(getattr(figure, "axes", []) or []):
        lines = [ln for ln in ax.get_lines() if not str(ln.get_label()).startswith("_")]
        # Also restyle unlabeled data lines (montage often uses _nolegend_).
        data_lines = [
            ln
            for ln in ax.get_lines()
            if ln.get_label() != "Stimulation (onset)" and "offset" not in str(ln.get_label()).lower()
        ]
        targets = lines if lines else data_lines
        for index, line in enumerate(targets):
            # Skip marker vlines (typically dashed/dashdot).
            ls = str(line.get_linestyle())
            if ls in {"--", "-.", ":"}:
                continue
            if index < len(colors):
                try:
                    line.set_color(colors[index])
                except Exception:
                    pass
            try:
                line.set_linewidth(
                    lw
                    if request.panel not in _MONTAGE_SPECS
                    else _montage_line_width(request.style)
                )
            except Exception:
                pass
        if request.panel not in {"mea_layout", "summary_rms_table"}:
            if request.style.grid:
                ax.grid(True, alpha=request.style.grid_alpha)
            else:
                ax.grid(False)
        _apply_axis_style(ax, request.style)
        if request.panel.startswith("mean_") or request.panel.startswith("analysis_"):
            _apply_ylim(ax, request.settings.trace_ylim)
        elif "hp" in request.panel and (
            request.panel.startswith("first_")
            or request.panel.startswith("second_")
            or request.panel.startswith("montage_")
        ):
            _apply_ylim(ax, request.settings.stim_hp_ylim)
        elif request.panel.startswith("first_") or request.panel.startswith("second_"):
            _apply_ylim(ax, request.settings.trace_ylim)
        if request.panel not in {"mea_layout", "summary_rms_table"} and ax is figure.axes[0]:
            _apply_legend(ax, request.legend)
    return status


def render_panel(figure: Any, request: RenderRequest) -> str:
    """Draw one panel. Rebuilds only when data geometry changes."""
    fingerprint = _structure_fingerprint(request)
    previous = getattr(figure, "_erg_structure_key", None)
    panel = request.panel
    has_axes = bool(list(getattr(figure, "axes", []) or []))

    if (
        previous == fingerprint
        and has_axes
        and panel not in {"mea_layout", "summary_rms_table", "summary_rms", "summary_impedance"}
    ):
        if panel in _MONTAGE_SPECS or panel == "montage_continuous_raw":
            expected = getattr(figure, "_erg_montage_rows", None)
            if expected is not None and expected == len(figure.axes):
                status = _refresh_panel_style(figure, request)
                figure._erg_status = status
                return status
        elif panel != "full_recording":
            status = _refresh_panel_style(figure, request)
            figure._erg_status = status
            return status

    # Same montage layout, new channel data → set_data without clear/subplots.
    if has_axes and panel in _MONTAGE_SPECS:
        updated = _try_update_triggered_montage(figure, request)
        if updated is not None:
            figure._erg_structure_key = fingerprint
            figure._erg_montage_rows = len(figure.axes)
            figure._erg_status = updated
            return updated

    figure.clear()
    figure._erg_structure_key = None
    figure._erg_montage_rows = None
    figure._erg_montage_state = None
    if not request.recordings:
        _unavailable(figure.add_subplot(111), "Load a recording to display this panel.")
        return "unavailable"

    if panel == "full_recording":
        status = _render_full_recording(figure, request)
        figure._erg_structure_key = fingerprint
        figure._erg_status = status
        return status

    if panel in _MONTAGE_SPECS or panel == "montage_continuous_raw":
        status = _render_montage(figure, request)
        figure._erg_structure_key = fingerprint
        figure._erg_montage_rows = len(getattr(figure, "axes", []) or [])
        figure._erg_status = status
        return status

    ax = figure.add_subplot(111)
    renderer = _AXES_RENDERERS.get(panel)
    if renderer is None:
        _unavailable(ax, f"Unknown panel: {panel}")
        return "unavailable"

    status = renderer(ax, request)
    if status == "ok":
        if panel not in {"mea_layout", "summary_rms_table"}:
            if request.style.grid:
                ax.grid(True, alpha=request.style.grid_alpha)
            else:
                ax.grid(False)
        _apply_legend(ax, request.legend)
    _apply_axis_style(ax, request.style)
    figure._erg_structure_key = fingerprint
    figure._erg_status = status
    return status


def panels_for_scope(scope: PanelScope) -> tuple[PanelInfo, ...]:
    return tuple(info for info in PANEL_CATALOG if info.scope == scope)


def all_panel_keys() -> tuple[str, ...]:
    return tuple(info.key for info in PANEL_CATALOG)


def is_known_panel(key: str) -> bool:
    return key in PANEL_INFO_BY_KEY or key in EXTRA_CHANNEL_PANEL_FIELD_NAMES
