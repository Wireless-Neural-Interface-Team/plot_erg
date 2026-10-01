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

from display_config import PANEL_FIELD_NAMES
from impedance_tracking import ImpedanceSession
from processed_dataset import ProcessedRecording
from view_config import (
    EXTRA_CHANNEL_PANEL_FIELD_NAMES,
    GLOBAL_PANEL_FIELD_NAMES,
    SECTION_LABELS,
    LegendSettings,
    PanelPlacement,
    PanelStyle,
    ViewerSettings,
    panel_label,
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
    "montage_mean_raw": ("raw", None, "#334155"),
    "montage_mean_hp": ("hp", None, "#15803d"),
    "montage_mean_lp": ("lp", None, "#1e40af"),
    "montage_second_raw": ("raw", 1, "#334155"),
    "montage_second_hp": ("hp", 1, "#15803d"),
    "montage_second_lp": ("lp", 1, "#1e40af"),
    "montage_second_to_third_lp": ("lp", 1, "#1e40af"),
}

_UNAVAILABLE_FONT = 9.0


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


_PANEL_GROUPS: dict[str, str] = {
    "mean_raw": "Tension — brut",
    "first_trigger_raw": "Tension — brut",
    "second_trigger_raw": "Tension — brut",
    "mean_hp": "Tension — passe-haut",
    "first_trigger_hp": "Tension — passe-haut",
    "second_trigger_hp": "Tension — passe-haut",
    "mean_lp": "Tension — passe-bas",
    "first_trigger_lp": "Tension — passe-bas",
    "second_trigger_lp": "Tension — passe-bas",
    "rms": "RMS",
    "first_rms": "RMS",
    "second_rms": "RMS",
    "psth": "Spikes — taux",
    "first_psth": "Spikes — taux",
    "second_psth": "Spikes — taux",
    "trial_rate": "Spikes — taux",
    "isi": "Spikes — intervalles",
    "first_isi": "Spikes — intervalles",
    "second_isi": "Spikes — intervalles",
    "raster": "Spikes — raster",
    "spike_overlay": "Spikes — formes d’onde",
}

_SPIKE_PANELS = {
    "psth",
    "first_psth",
    "second_psth",
    "isi",
    "first_isi",
    "second_isi",
    "trial_rate",
    "raster",
    "spike_overlay",
}


def _build_catalog() -> tuple[PanelInfo, ...]:
    panels: list[PanelInfo] = []
    panels.append(
        PanelInfo(
            key="full_recording",
            label=panel_label("full_recording"),
            group="Enregistrement",
            needs_streams=True,
            preferred_height_px=420,
        )
    )
    for key in (
        "analysis_raw",
        "analysis_hp",
        "analysis_lp",
        "analysis_rms",
        "analysis_psth",
        "analysis_isi",
        "analysis_raster",
        "analysis_overlay",
    ):
        panels.append(
            PanelInfo(
                key=key,
                label=panel_label(key),
                group="Analyse (configurée)",
                needs_spikes=key in {
                    "analysis_psth",
                    "analysis_isi",
                    "analysis_raster",
                    "analysis_overlay",
                },
                preferred_height_px=320 if key == "analysis_raster" else 300,
            )
        )
    for key in PANEL_FIELD_NAMES:
        panels.append(
            PanelInfo(
                key=key,
                label=panel_label(key),
                group=_PANEL_GROUPS.get(key, "Autre"),
                scope="channel",
                needs_spikes=key in _SPIKE_PANELS,
                preferred_height_px=320 if key == "raster" else 300,
            )
        )
    panels.append(
        PanelInfo(
            key="mea_layout",
            label=panel_label("mea_layout"),
            group="Contexte",
            preferred_height_px=360,
        )
    )
    panels.append(
        PanelInfo(
            key="impedance",
            label=panel_label("impedance"),
            group="Contexte",
            needs_impedance=True,
            preferred_height_px=260,
        )
    )
    for key in GLOBAL_PANEL_FIELD_NAMES:
        montage = key.startswith("montage_")
        panels.append(
            PanelInfo(
                key=key,
                label=panel_label(key),
                group="Montages (tous canaux)" if montage else "Résumés",
                scope="global",
                needs_impedance=key == "summary_impedance",
                needs_streams=montage and key.startswith("montage_second"),
                preferred_height_px=620 if montage else 360,
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
        # Zoom personnalisé défini sur le placement (prioritaire).
        if self.placement.has_custom_zoom:
            return float(self.placement.zoom_t0_s), float(self.placement.zoom_t1_s)
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
        if self.placement.has_custom_zoom or self.section != "zoom_trigger_end":
            window = self.section_window()
            return [window for _ in self.recordings]
        t0, t1 = self.settings.zoom_end_window()
        out: list[tuple[float, float] | None] = []
        for recording in self.recordings:
            marker = recording.end_marker_s
            out.append(None if marker is None else (float(marker) + t0, float(marker) + t1))
        return out


# --------------------------------------------------------------------- helpers


def _decimate(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope decimation: keeps peaks while bounding the point count."""
    n = int(np.asarray(x).size)
    limit = max(16, int(max_points))
    if n <= limit:
        return x, y
    n_bins = max(8, limit // 2)
    edges = np.linspace(0, n, n_bins + 1).astype(np.int64)
    starts = edges[:-1]
    valid = starts < n
    starts = starts[valid]
    lows = np.minimum.reduceat(y, starts)
    highs = np.maximum.reduceat(y, starts)
    mids = np.add.reduceat(x, starts) / np.diff(np.append(starts, n))
    out_x = np.repeat(mids, 2)
    out_y = np.empty(out_x.size, dtype=np.float64)
    out_y[0::2] = lows
    out_y[1::2] = highs
    return out_x, out_y


def _plot_curve(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    style: PanelStyle,
    color: Any,
    label: str | None,
    line_width: float | None = None,
) -> None:
    if x.size == 0 or y.size == 0:
        return
    n = min(int(x.size), int(y.size))
    xd, yd = _decimate(
        np.asarray(x[:n], dtype=np.float64),
        np.asarray(y[:n], dtype=np.float64),
        style.max_points_per_curve,
    )
    ax.plot(
        xd,
        yd,
        linewidth=line_width if line_width is not None else style.line_width,
        color=color,
        label=label if label else "_nolegend_",
        solid_joinstyle="round",
    )


def _mask_window(t: np.ndarray, window: tuple[float, float] | None) -> np.ndarray | None:
    if window is None:
        return None
    return (t >= float(window[0])) & (t <= float(window[1]))


def _unavailable(ax: Any, message: str) -> None:
    ax.clear()
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=_UNAVAILABLE_FONT,
        color="#64748b",
        wrap=True,
    )
    ax.set_axis_off()


def _apply_axis_style(ax: Any, style: PanelStyle) -> None:
    ax.title.set_fontsize(style.title_font_size)
    ax.xaxis.label.set_fontsize(style.label_font_size)
    ax.yaxis.label.set_fontsize(style.label_font_size)
    ax.tick_params(axis="both", labelsize=style.tick_font_size)
    for text in ax.texts:
        if text.get_fontsize() > style.label_font_size * 1.6:
            text.set_fontsize(style.label_font_size)


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
    kwargs: dict[str, Any] = {
        "fontsize": settings.font_size,
        "ncol": max(1, int(settings.columns)),
        "frameon": bool(settings.frame),
        "framealpha": 0.92,
        "borderaxespad": 0.3,
    }
    if settings.location == "below":
        legend = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
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
    """Extraire les fenêtres de zoom personnalisées uniques (pour surlignage)."""
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float, str]] = []
    for placement in placements:
        if not placement.has_custom_zoom:
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
    from plotting import _draw_onset_offset_lines

    markers = request.end_markers()
    _draw_onset_offset_lines(ax, end_markers=markers, label_in_legend=True)
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


def _render_full_recording(ax: Any, request: RenderRequest) -> str:
    stream = str(request.settings.continuous_stream or "raw")
    drawn = 0
    for index, recording in enumerate(request.recordings):
        t, values = recording.continuous_trace(stream, request.channel_index)
        if values.size == 0:
            continue
        _plot_curve(
            ax,
            t,
            values,
            style=request.style,
            color=request.colors[index],
            label=_series_label(request, index, "continuous", stream=stream),
        )
        if request.settings.continuous_mark_stims:
            for stim_t in recording.stimulation_times_s():
                ax.axvline(
                    float(stim_t),
                    color="#dc2626",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.55,
                )
        drawn += 1
    if drawn == 0:
        return _fail(
            ax,
            request,
            "Continuous recording unavailable.\n"
            "Process a .rhs file first (F5), then select a channel.",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Potential (µV)")
    _apply_ylim(ax, request.settings.trace_ylim)
    ax.set_title(
        f"{request.channel_name} — full recording ({stream})"
        + (" — stimulations marked" if request.settings.continuous_mark_stims else "")
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
    from plotting import _draw_spike_panels_multi_channel

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
    elif panel.endswith("trial_rate") or panel == "trial_rate":
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
    )
    if not ax.get_visible() or not ax.axison:
        return "empty"
    ax.set_title(
        f"{request.channel_name} — {panel_label(panel)}{mode_note}{_section_suffix(request)}"
    )
    return "ok"


def _render_spike_overlay(ax: Any, request: RenderRequest) -> str:
    from plotting import _draw_spike_overlay_panel

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
    from plotting import _draw_impedance_evolution_panel

    sessions = request.impedance_sessions
    if not sessions:
        return _fail(
            ax,
            request,
            "No impedance data.\nAdd a companion CSV next to the recording "
            "(same name as the parent folder).",
        )
    _draw_impedance_evolution_panel(ax, request.channel_name, sessions)
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
    ax.semilogy(times[valid], magnitudes[valid], marker="o", markersize=4, linewidth=1.0)
    ax.set_ylabel("Mean |Z| @ 1 kHz (Ω)")
    ax.set_xlabel("Session time")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_title("Recording-mean impedance |Z| @ 1 kHz")
    return "ok"


# ----------------------------------------------------------- montage renderers


def _montage_channel_slice(request: RenderRequest, n_channels: int) -> tuple[int, int, int]:
    """Channel range for the current montage page, centred on the selection."""
    per_page = max(1, int(request.settings.montage_channels))
    n_pages = max(1, (n_channels + per_page - 1) // per_page)
    page = int(request.settings.montage_page)
    if page < 0:
        page = max(0, min(n_pages - 1, request.channel_index // per_page))
    page = max(0, min(n_pages - 1, page))
    start = page * per_page
    return start, min(n_channels, start + per_page), n_pages


def _render_montage(figure: Any, request: RenderRequest) -> str:
    if request.panel == "montage_continuous_raw":
        return _render_montage_continuous(figure, request)
    stream, trigger_index, color = _MONTAGE_SPECS[request.panel]
    recordings = request.recordings
    if not recordings:
        return _fail(figure.add_subplot(111), request, "No recording selected.")
    reference = recordings[0]
    n_channels = reference.n_channels
    start, end, n_pages = _montage_channel_slice(request, n_channels)
    rows = max(1, end - start)
    window = request.section_window()
    if window is None:
        window = (float(reference.t_rel[0]), float(reference.t_rel[-1]))

    axes = figure.subplots(rows, 1, sharex=True, squeeze=False)[:, 0]
    t = reference.t_rel
    mask = _mask_window(t, window)
    x = t[mask] if mask is not None else t
    drawn = 0
    for row, ch in enumerate(range(start, end)):
        ax = axes[row]
        if trigger_index is None:
            curve = reference.mean(stream, ch)
        else:
            curve = reference.trigger_window(trigger_index, stream, ch)
        name = reference.channel_names[ch] if ch < len(reference.channel_names) else f"CH{ch}"
        if curve is None:
            ax.text(
                0.5,
                0.5,
                "unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=request.style.tick_font_size,
                color="#94a3b8",
            )
        else:
            y = curve[mask] if mask is not None else curve
            _plot_curve(
                ax,
                x,
                y,
                style=request.style,
                color=color,
                label=None,
                line_width=0.8,
            )
            drawn += 1
        ax.set_ylabel(
            name,
            rotation=0,
            ha="right",
            va="center",
            fontsize=max(5.0, request.style.tick_font_size - 1.0),
        )
        ax.tick_params(axis="both", labelsize=max(5.0, request.style.tick_font_size - 1.5))
        ax.grid(request.style.grid, alpha=request.style.grid_alpha)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=0.7, alpha=0.8)
        for marker in request.end_markers():
            ax.axvline(marker, color="#1d4ed8", linestyle="-.", linewidth=0.7, alpha=0.8)
        ax.set_xlim(window[0], window[1])
        if ch == request.channel_index:
            ax.set_facecolor("#fff7ed")
        if row < rows - 1:
            ax.set_xticklabels([])
    axes[-1].set_xlabel("Time relative to stimulation (s)")
    page_note = f"page {int(request.settings.montage_page) + 1}/{n_pages}" if n_pages > 1 else ""
    figure.suptitle(
        f"{panel_label(request.panel)} — {reference.label} — channels {start + 1}–{end}"
        + (f" ({page_note})" if page_note else ""),
        fontsize=request.style.title_font_size,
    )
    return "ok" if drawn else "empty"


def _render_montage_continuous(figure: Any, request: RenderRequest) -> str:
    """Stacked continuous raw traces for every channel (paginated)."""
    recordings = request.recordings
    if not recordings:
        return _fail(figure.add_subplot(111), request, "No recording selected.")
    reference = recordings[0]
    if not reference.has_streams:
        return _fail(
            figure.add_subplot(111),
            request,
            "Continuous streams unavailable.\nProcess a .rhs file (F5).",
        )
    n_channels = reference.n_channels
    start, end, n_pages = _montage_channel_slice(request, n_channels)
    rows = max(1, end - start)
    stream = str(request.settings.continuous_stream or "raw")
    color = "#334155"
    axes = figure.subplots(rows, 1, sharex=True, squeeze=False)[:, 0]
    drawn = 0
    stim_times = (
        reference.stimulation_times_s() if request.settings.continuous_mark_stims else np.empty(0)
    )
    for row, ch in enumerate(range(start, end)):
        ax = axes[row]
        t, values = reference.continuous_trace(stream, ch)
        name = reference.channel_names[ch] if ch < len(reference.channel_names) else f"CH{ch}"
        if values.size == 0:
            ax.text(
                0.5,
                0.5,
                "unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=request.style.tick_font_size,
                color="#94a3b8",
            )
        else:
            _plot_curve(
                ax,
                t,
                values,
                style=request.style,
                color=color,
                label=None,
                line_width=max(0.5, request.style.line_width * 0.7),
            )
            drawn += 1
        for stim_t in stim_times:
            ax.axvline(float(stim_t), color="#dc2626", linestyle="--", linewidth=0.6, alpha=0.45)
        ax.set_ylabel(
            name,
            rotation=0,
            ha="right",
            va="center",
            fontsize=max(5.0, request.style.tick_font_size - 1.0),
        )
        ax.tick_params(axis="both", labelsize=max(5.0, request.style.tick_font_size - 1.5))
        ax.grid(request.style.grid, alpha=request.style.grid_alpha)
        if ch == request.channel_index:
            ax.set_facecolor("#fff7ed")
        if row < rows - 1:
            ax.set_xticklabels([])
    axes[-1].set_xlabel("Time (s)")
    page_note = f"page {int(request.settings.montage_page) + 1}/{n_pages}" if n_pages > 1 else ""
    figure.suptitle(
        f"Continuous {stream} — {reference.label} — channels {start + 1}–{end}"
        + (f" ({page_note})" if page_note else ""),
        fontsize=request.style.title_font_size,
    )
    return "ok" if drawn else "empty"


# -------------------------------------------------------------------- dispatch


def _fail(ax: Any, request: RenderRequest, message: str) -> str:
    del request
    _unavailable(ax, message)
    return "unavailable"


_AXES_RENDERERS: dict[str, Callable[[Any, RenderRequest], str]] = {
    "full_recording": _render_full_recording,
    "analysis_raw": _render_analysis_trace,
    "analysis_hp": _render_analysis_trace,
    "analysis_lp": _render_analysis_trace,
    "analysis_rms": _render_analysis_rms,
    "analysis_psth": _render_spike_panel,
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


def render_panel(figure: Any, request: RenderRequest) -> str:
    """Draw one panel on a cleared figure. Returns a short status string."""
    figure.clear()
    panel = request.panel
    if not request.recordings:
        _unavailable(figure.add_subplot(111), "Load a recording to display this panel.")
        return "unavailable"

    if panel in _MONTAGE_SPECS or panel == "montage_continuous_raw":
        status = _render_montage(figure, request)
        return status

    ax = figure.add_subplot(111)
    renderer = _AXES_RENDERERS.get(panel)
    if renderer is None:
        _unavailable(ax, f"Unknown panel: {panel}")
        return "unavailable"

    status = renderer(ax, request)
    if status == "ok":
        if panel not in {"mea_layout", "summary_rms_table"}:
            ax.grid(request.style.grid, alpha=request.style.grid_alpha)
        _apply_legend(ax, request.legend)
    _apply_axis_style(ax, request.style)
    return status


def panels_for_scope(scope: PanelScope) -> tuple[PanelInfo, ...]:
    return tuple(info for info in PANEL_CATALOG if info.scope == scope)


def all_panel_keys() -> tuple[str, ...]:
    return tuple(info.key for info in PANEL_CATALOG)


def is_known_panel(key: str) -> bool:
    return key in PANEL_INFO_BY_KEY or key in EXTRA_CHANNEL_PANEL_FIELD_NAMES
