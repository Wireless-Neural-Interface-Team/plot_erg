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
from view_config import (
    STREAM_SHORT_LABELS,
    continuous_anchor_offset_s,
    continuous_sync_offset_s,
    included_channel_indices,
    montage_row_axis_label,
    resolved_scale_bar_linewidth,
    resolved_scale_bar_offset_px,
    resolved_scale_bar_time_offset_px,
    resolved_stim_linewidth,
    resolved_stim_offset_color,
    resolved_stim_onset_color,
)

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
    alpha: float = 1.0


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
    """Lightweight annotation (vline / text / hline / stim_label)."""

    kind: str  # "vline" | "hline" | "text" | "span" | "stim_label" | "zone_label"
    x: float = 0.0
    y: float = 0.0
    x2: float = 0.0
    text: str = ""
    color: str = "#dc2626"
    row: int = 0
    linestyle: str = ":"
    label: str = ""
    # ``stim_onset`` / ``stim_offset`` : recolorables sans reconstruire les courbes.
    role: str = ""
    # Style des libellés « on » / « off » (kind == "stim_label").
    fontsize: float = 8.0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    dx_px: float = 0.0
    dy_px: float = 0.0


# Courbes en µV sur lesquelles le seuil de spike a un sens visuel.
_VOLTAGE_TRACE_KINDS = frozenset({"raw", "hp", "lp"})
_SPIKE_THRESHOLD_COLOR = "#6d28d9"


def _show_spike_threshold_lines(request: Any) -> bool:
    settings = getattr(request, "settings", None)
    return bool(getattr(settings, "show_spike_threshold", False))


def _threshold_line_annotations(
    request: Any,
    channel_index: int,
    *,
    row: int = 0,
    with_legend: bool = True,
) -> list[AnnotationSpec]:
    """Lignes horizontales du seuil de détection, une par valeur distincte."""
    if not _show_spike_threshold_lines(request):
        return []
    recordings = list(getattr(request, "recordings", []) or [])
    seen: set[float] = set()
    out: list[AnnotationSpec] = []
    for recording in recordings:
        try:
            thr = float(recording.threshold_uv(int(channel_index)))
        except Exception:
            continue
        if not np.isfinite(thr):
            continue
        key = round(thr, 3)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            AnnotationSpec(
                kind="hline",
                y=float(thr),
                color=_SPIKE_THRESHOLD_COLOR,
                row=int(row),
                linestyle="--",
                label=f"Seuil ({thr:g} µV)" if with_legend else "",
            )
        )
    return out


def _stim_line_colors(request: Any) -> tuple[str, str]:
    """``(début, fin)`` des pointillés de stimulation, depuis le style d’affichage."""
    style = getattr(request, "style", None)
    return resolved_stim_onset_color(style), resolved_stim_offset_color(style)


def _show_continuous_stim_marks(request: Any) -> bool:
    """Pointillés sur les traces continues (aperçu et revue montage).

    Les deux coches comptent : « Marqueurs de stimulation sur les traces »
    et « Pointillés de début / fin de stimulation ».
    """
    settings = getattr(request, "settings", None)
    if not bool(getattr(settings, "continuous_mark_stims", True)):
        return False
    legend = getattr(request, "legend", None)
    return bool(getattr(legend, "show_reference_markers", True))


def _show_stim_labels(request: Any) -> bool:
    """Case « Libellés on / off » du style d’affichage."""
    style = getattr(request, "style", None)
    return bool(getattr(style, "show_stim_labels", True))


def _stim_label_style(
    request: Any, *, which: str
) -> tuple[str, str, float, bool, bool, bool, float, float]:
    """``(texte, couleur, fontsize, bold, italic, underline, dx_px, dy_px)``."""
    onset_color, offset_color = _stim_line_colors(request)
    texts = getattr(getattr(request, "settings", None), "texts", None)
    style = getattr(request, "style", None)
    if which == "off":
        text = str(getattr(texts, "stim_off", "off") if texts is not None else "off")
        color = offset_color
        fontsize = float(getattr(style, "stim_off_font_size", 8.0) or 8.0)
        face = getattr(style, "stim_off_face", None)
        dx = float(getattr(style, "stim_off_dx_px", 0.0) or 0.0)
        dy = float(getattr(style, "stim_off_dy_px", 0.0) or 0.0)
    else:
        text = str(getattr(texts, "stim_on", "on") if texts is not None else "on")
        color = onset_color
        fontsize = float(getattr(style, "stim_on_font_size", 8.0) or 8.0)
        face = getattr(style, "stim_on_face", None)
        dx = float(getattr(style, "stim_on_dx_px", 0.0) or 0.0)
        dy = float(getattr(style, "stim_on_dy_px", 0.0) or 0.0)
    bold, italic, underline = _face_flags(face)
    return text, color, fontsize, bold, italic, underline, dx, dy


def _stim_label_annotation(
    request: Any,
    *,
    x: float,
    row: int,
    which: str,
) -> AnnotationSpec | None:
    """Libellé au-dessus d’un trait de stim ; ``None`` si masqué / texte vide."""
    if not _show_stim_labels(request):
        return None
    text, color, fontsize, bold, italic, underline, dx, dy = _stim_label_style(
        request, which=which
    )
    if not str(text).strip():
        return None
    role = "stim_offset" if which == "off" else "stim_onset"
    return AnnotationSpec(
        kind="stim_label",
        x=float(x),
        text=str(text),
        color=color,
        row=int(row),
        role=role,
        fontsize=float(fontsize),
        bold=bool(bold),
        italic=bool(italic),
        underline=bool(underline),
        dx_px=float(dx),
        dy_px=float(dy),
    )


def _stim_label_already_present(
    annotations: Sequence[AnnotationSpec], *, x: float, which: str
) -> bool:
    """True si un libellé on/off existe déjà pour cette stimulation."""
    target = round(float(x), 6)
    want_role = "stim_offset" if which == "off" else "stim_onset"
    for ann in annotations:
        if str(getattr(ann, "kind", "") or "") != "stim_label":
            continue
        if str(getattr(ann, "role", "") or "") != want_role:
            continue
        if round(float(ann.x), 6) == target:
            return True
    return False


def _append_stim_vline_with_label(
    annotations: list[AnnotationSpec],
    request: Any,
    *,
    x: float,
    row: int,
    which: str,
    color: str,
    with_label: bool = True,
) -> None:
    """Pointillé de stim ; libellé une seule fois par stimulation (pas par canal)."""
    role = "stim_offset" if which == "off" else "stim_onset"
    annotations.append(
        AnnotationSpec(
            kind="vline",
            x=float(x),
            color=color,
            row=int(row),
            role=role,
        )
    )
    if not with_label:
        return
    if _stim_label_already_present(annotations, x=float(x), which=which):
        return
    label = _stim_label_annotation(request, x=float(x), row=int(row), which=which)
    if label is not None:
        annotations.append(label)


def _restyle_stim_annotations(
    annotations: list[AnnotationSpec], request: Any
) -> list[AnnotationSpec]:
    """Recolorier les pointillés et régénérer les libellés on/off.

    Un seul libellé par stimulation (même ``x`` / rôle), placé sur la ligne
    du haut — pas une copie par canal.
    """
    onset, offset = _stim_line_colors(request)
    restyled: list[AnnotationSpec] = []
    # (x arrondi, which) → (x réel, row min)
    unique_edges: dict[tuple[float, str], tuple[float, int]] = {}
    for ann in annotations:
        role = str(getattr(ann, "role", "") or "")
        kind = str(getattr(ann, "kind", "") or "")
        if kind == "stim_label":
            continue
        if role == "stim_onset":
            restyled.append(replace(ann, color=onset))
            which = "on"
        elif role == "stim_offset":
            restyled.append(replace(ann, color=offset))
            which = "off"
        else:
            restyled.append(ann)
            continue
        x = float(ann.x)
        row = int(ann.row or 0)
        key = (round(x, 6), which)
        prev = unique_edges.get(key)
        if prev is None or row < prev[1]:
            unique_edges[key] = (x, row)
    for which_key, (x, row) in unique_edges.items():
        which = which_key[1]
        label = _stim_label_annotation(request, x=x, row=row, which=which)
        if label is not None:
            restyled.append(label)
    return restyled


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
    # Longueur pixel manuelle des bâtons (None / non manuel → suit le zoom).
    # Avec amp_manual + amp_length_manual : fixe le gain Y
    # (span = amplitude × hauteur_plot / amp_length_px).
    amp_length_manual: bool = False
    amp_length_px: float = 40.0
    time_length_manual: bool = False
    time_length_px: float = 40.0
    linewidth: float = 1.4
    fontsize: float = 8.0
    color: str = "#111827"
    # Décalage échelle amplitude / axe vertical (+dx droite, +dy haut).
    dx_px: float = 0.0
    dy_px: float = 0.0
    # Décalage échelle temps / axe horizontal (+dx droite, +dy haut).
    time_dx_px: float = 0.0
    time_dy_px: float = 0.0


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
    # Décalage des titres de ligne (px) : +x vers la courbe, +y vers le haut.
    row_label_dx_px: float = 0.0
    row_label_dy_px: float = 0.0
    row_ylims: list[tuple[float, float] | None] = field(default_factory=list)
    row_xlims: list[tuple[float, float] | None] = field(default_factory=list)
    row_kinds: list[str] = field(default_factory=list)
    row_channels: list[int] = field(default_factory=list)
    row_keys: list[tuple[int, str]] = field(default_factory=list)
    row_highlight: list[bool] = field(default_factory=list)
    # Hauteur minimale d’une ligne montage (px). 0 = ne pas imposer.
    row_min_height_px: int = 0
    status_message: str = ""
    grid: bool = True
    grid_alpha: float = 0.35
    show_legend: bool = True
    scale_bars: ScaleBarSpec = field(default_factory=ScaleBarSpec)
    # Épaisseur des pointillés de stimulation (début / fin).
    stim_linewidth: float = 1.0
    # Écart titre ↔ courbe (aperçu, px). 0 = défaut pyqtgraph.
    title_gap_px: int = 0
    # Décalage du titre de graphique (px) : +dx droite, +dy haut.
    title_dx_px: float = 0.0
    title_dy_px: float = 0.0
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
    y_log: bool = False
    x_ticks: list[tuple[float, str]] = field(default_factory=list)
    table_headers: list[str] = field(default_factory=list)
    table_rows: list[list[str]] = field(default_factory=list)
    table_highlight_row: int = -1


def _row_xlim_key(spec: PanelSpec, row: int) -> tuple[float, float] | None:
    """Clé d’échelle X d’une ligne (propre, sinon partagée)."""
    xlims = spec.row_xlims or []
    if 0 <= row < len(xlims) and xlims[row] is not None:
        lo, hi = xlims[row]
        return (round(float(lo), 6), round(float(hi), 6))
    if spec.xlim is not None:
        return (round(float(spec.xlim[0]), 6), round(float(spec.xlim[1]), 6))
    return None


def montage_vline_spans(
    spec: PanelSpec,
) -> list[tuple[float, str, tuple[int, ...]]]:
    """Pointillés de stimulation identiques sur des lignes contiguës.

    Chaque entrée est ``(x, couleur, lignes)`` avec au moins deux lignes qui
    partagent la même échelle X. Ces traits doivent être dessinés d’un seul
    tenant, y compris dans l’intervalle entre canaux.
    """
    if str(spec.layout_kind or "") != "montage":
        return []
    groups: dict[tuple[Any, ...], list[tuple[int, float]]] = {}
    for ann in spec.annotations:
        if str(ann.kind) != "vline":
            continue
        row = int(ann.row or 0)
        x = float(ann.x)
        if not np.isfinite(x):
            continue
        key = (
            round(x, 6),
            str(ann.color or ""),
            str(getattr(ann, "linestyle", ":") or ":"),
            _row_xlim_key(spec, row),
        )
        groups.setdefault(key, []).append((row, x))
    spans: list[tuple[float, str, tuple[int, ...]]] = []
    for key, hits in groups.items():
        color = str(key[1])
        ordered = sorted(hits, key=lambda item: item[0])
        run_rows: list[int] = []
        run_x: list[float] = []

        def flush() -> None:
            if len(run_rows) >= 2:
                spans.append((float(run_x[0]), color, tuple(run_rows)))
            run_rows.clear()
            run_x.clear()

        for row, x in ordered:
            if run_rows and row != run_rows[-1] + 1:
                flush()
            if run_rows and row == run_rows[-1]:
                continue
            run_rows.append(row)
            run_x.append(x)
        flush()
    return spans


def montage_vline_span_keys(
    spans: list[tuple[float, str, tuple[int, ...]]],
) -> set[tuple[int, float, str]]:
    """``(ligne, x arrondi à 1 µs, couleur)`` déjà couverts par un trait continu."""
    covered: set[tuple[int, float, str]] = set()
    for x, color, rows in spans:
        x_key = round(float(x), 6)
        for row in rows:
            covered.add((int(row), x_key, str(color)))
    return covered


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
        elif panel == "montage_second_to_third_lp":
            spec = _prepare_montage_second_to_third(request)
        elif panel.startswith("montage_"):
            spec = _prepare_montage_triggered(request)
        elif panel == "mea_layout":
            spec = _prepare_mea_layout(request)
        elif panel == "impedance":
            spec = _prepare_impedance(request)
        elif panel == "summary_rms":
            spec = _prepare_summary_rms(request)
        elif panel == "summary_rms_table":
            spec = _prepare_summary_rms_table(request)
        elif panel == "summary_impedance":
            spec = _prepare_summary_impedance(request)
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
        is_montage = str(panel).startswith("montage_")
        new_ylims: list[tuple[float, float] | None] = []
        for kind in out.row_kinds:
            kind_s = str(kind)
            if kind_s in {"raw", "hp", "lp"}:
                new_ylims.append(
                    _ylim_tuple(settings, kind_s, montage=is_montage)
                )
            elif kind_s == "rms":
                new_ylims.append(
                    _ylim_tuple(settings, "rms", montage=is_montage)
                )
            else:
                new_ylims.append(None)
        if panel == "montage_continuous_raw":
            new_ylims = _apply_montage_shared_ylims(
                settings, list(out.row_kinds), new_ylims, out.series
            )
        out.row_ylims = new_ylims
    if panel == "montage_continuous_raw" and out.row_channels and out.row_kinds:
        refreshed = _montage_continuous_axis_labels(
            request, out.row_channels, out.row_kinds
        )
        if refreshed is not None:
            out.row_labels = refreshed

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
                new_lw = float(lw)
                extra: dict[str, Any] = {}
                if panel in {"analysis_overlay", "spike_overlay"}:
                    is_mean = bool(str(getattr(item, "label", "") or "").strip())
                    if is_mean:
                        new_lw = max(2.4, float(lw) * 2.2)
                        extra["alpha"] = 1.0
                    else:
                        new_lw = max(0.5, float(lw) * 0.7)
                        extra["alpha"] = _spike_trace_alpha(
                            getattr(request, "settings", None)
                        )
                new_series[series_index] = replace(
                    item, color=color, linewidth=new_lw, **extra
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

    out.annotations = _restyle_stim_annotations(list(out.annotations), request)
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
        spec.title_dx_px = float(getattr(style, "title_dx_px", 0.0) or 0.0)
        spec.title_dy_px = float(getattr(style, "title_dy_px", 0.0) or 0.0)
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
        spec.stim_linewidth = resolved_stim_linewidth(style)
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
    settings = getattr(request, "settings", None)
    spec.row_label_dx_px = float(getattr(settings, "montage_label_dx_px", 0.0) or 0.0)
    spec.row_label_dy_px = float(getattr(settings, "montage_label_dy_px", 0.0) or 0.0)
    spec.title_gap_px = int(getattr(settings, "preview_title_gap_px", 0) or 0)


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
        if spec.row_labels and str(spec.layout_kind or "") == "montage":
            # Garder les noms de canal ; ne remplir que les libellés vides.
            spec.row_labels = [
                (label if str(label or "").strip() else ylabel) for label in spec.row_labels
            ]
        elif spec.row_labels:
            # Aperçu : le texte visible est row_labels (« WIDE (µV) »), pas ylabel.
            spec.row_labels = [ylabel for _label in spec.row_labels]


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
    dx_px, dy_px = resolved_scale_bar_offset_px(style)
    time_dx_px, time_dy_px = resolved_scale_bar_time_offset_px(style)
    amp_len = float(getattr(style, "scale_bar_amplitude_length_px", 40.0) or 40.0)
    if amp_len <= 0:
        amp_len = 40.0
    time_len = float(getattr(style, "scale_bar_time_length_px", 40.0) or 40.0)
    if time_len <= 0:
        time_len = 40.0
    return ScaleBarSpec(
        enabled=True,
        x_unit=x_unit,
        y_unit=y_unit,
        time_manual=bool(getattr(style, "scale_bar_time_manual", False)),
        time_s=float(getattr(style, "scale_bar_time_s", 0.1) or 0.1),
        amp_manual=bool(getattr(style, "scale_bar_amplitude_manual", False)),
        amplitude=float(getattr(style, "scale_bar_amplitude", 100.0) or 100.0),
        amp_length_manual=bool(
            getattr(style, "scale_bar_amplitude_length_manual", False)
        ),
        amp_length_px=float(amp_len),
        time_length_manual=bool(
            getattr(style, "scale_bar_time_length_manual", False)
        ),
        time_length_px=float(time_len),
        linewidth=resolved_scale_bar_linewidth(style),
        fontsize=float(getattr(style, "tick_font_size", 8.0) or 8.0),
        dx_px=dx_px,
        dy_px=dy_px,
        time_dx_px=time_dx_px,
        time_dy_px=time_dy_px,
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


def _series_y_bounds(item: SeriesItem) -> tuple[float, float] | None:
    """Min / max finis de l’ordonnée d’une série, ou ``None`` si vide."""
    if isinstance(item, HistSeries):
        arr = np.asarray(item.counts, dtype=np.float64).ravel()
    else:
        arr = np.asarray(getattr(item, "y", None), dtype=np.float64).ravel()
    if arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.min(finite)), float(np.max(finite))


def _shared_ylim_bounds(
    y_lo: float, y_hi: float, *, non_negative: bool
) -> tuple[float, float]:
    """Échelle commune : étendue des données, plancher à 0 pour les taux / histogrammes."""
    lo = float(y_lo)
    hi = float(y_hi)
    if non_negative:
        lo = 0.0
        hi = max(hi, 1.0)
    if hi > lo:
        return (lo, hi)
    pad = max(abs(lo) * 0.05, 1.0)
    return (lo - pad, lo + pad)


def _apply_montage_shared_ylims(
    settings: Any,
    row_kinds: Sequence[str],
    row_ylims: Sequence[tuple[float, float] | None],
    series: Sequence[SeriesItem],
) -> list[tuple[float, float] | None]:
    """Une échelle Y commune aux courbes auto d’un même type.

    WIDE, HIGH et LOW partagent la même étendue en µV. Les autres graphs
    (RMS, PSTH, spike scope…) partagent une échelle entre canaux du même type.
    Une échelle déjà fixée (manuelle) n’est pas remplacée.
    """
    out = list(row_ylims)
    if not bool(getattr(settings, "montage_shared_ylim", False)):
        return out
    groups: dict[str, list[int]] = {}
    for row, kind in enumerate(row_kinds):
        kind_s = str(kind)
        if kind_s == "raster":
            continue
        if row < len(out) and out[row] is not None:
            continue
        key = "voltage" if kind_s in _VOLTAGE_TRACE_KINDS else kind_s
        groups.setdefault(key, []).append(row)
    if not groups:
        return out
    spans: dict[str, tuple[float, float]] = {}
    for item in series:
        row = int(getattr(item, "row", 0) or 0)
        if row < 0 or row >= len(row_kinds):
            continue
        kind_s = str(row_kinds[row])
        key = "voltage" if kind_s in _VOLTAGE_TRACE_KINDS else kind_s
        if key not in groups:
            continue
        bounds = _series_y_bounds(item)
        if bounds is None:
            continue
        prev = spans.get(key)
        if prev is None:
            spans[key] = bounds
        else:
            spans[key] = (min(prev[0], bounds[0]), max(prev[1], bounds[1]))
    nonneg = frozenset({"psth", "trial_rate", "isi", "rms"})
    for key, rows in groups.items():
        span = spans.get(key)
        if span is None:
            continue
        ylim = _shared_ylim_bounds(span[0], span[1], non_negative=key in nonneg)
        for row in rows:
            while len(out) <= row:
                out.append(None)
            out[row] = ylim
    return out


def _ylim_tuple(
    settings: Any,
    stream: str,
    *,
    montage: bool = False,
    height_px: float | None = None,
) -> tuple[float, float] | None:
    """Échelle Y Canal manuelle uniquement.

    Avec taille amp. manuelle, le gain vient de la hauteur du widget
    (``scale_bar_view_height_for_span``) et le ylim reste en autoscale données.
    """
    del montage, height_px  # conservés pour compatibilité d’appel
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
    # Custom zoom → jamais retomber sur x_limits de l’aperçu.
    # Axe X auto → toute la courbe (downsample via max_points).
    anchor_to_stim = False
    if request.placement.has_custom_zoom:
        window = request.section_window()
        if window is None:
            t0 = float(request.placement.zoom_t0_s)
            t1 = float(request.placement.zoom_t1_s)
            window = (t0, t1) if t0 <= t1 else (t1, t0)
    else:
        window = request.settings.x_limits.as_tuple()
    max_pts = _max_points(request)
    lw = _line_width(request)
    sync_mode = request.settings.time_sync
    stim_index = int(request.settings.analysis.stim_index)
    window_is_absolute = bool(
        request.placement.has_custom_zoom and request.placement.zoom_absolute
    )
    ref_offset = 0.0
    if request.recordings:
        ref_offset = continuous_anchor_offset_s(
            request.recordings[0].stimulation_times_s(),
            sync_mode,
            stim_index=stim_index,
            anchor_to_stim=anchor_to_stim,
        )

    series: list[SeriesItem] = []
    annotations: list[AnnotationSpec] = []
    onset_color, offset_color = _stim_line_colors(request)
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
            offset = continuous_anchor_offset_s(
                stims,
                sync_mode,
                stim_index=stim_index,
                anchor_to_stim=anchor_to_stim,
            )
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
            if _show_continuous_stim_marks(request) and not anchor_to_stim:
                mark_window = None
                if window is not None and not window_is_absolute:
                    mark_window = (float(window[0]), float(window[1]))
                for stim_x, role in _continuous_stim_marks(
                    [recording], sync_mode, mark_window
                ):
                    if window_is_absolute and window is not None:
                        file_t = float(stim_x) + offset
                        if not (float(window[0]) <= file_t <= float(window[1])):
                            continue
                    is_offset = role == "offset"
                    _append_stim_vline_with_label(
                        annotations,
                        request,
                        x=float(stim_x),
                        row=row,
                        which="off" if is_offset else "on",
                        color=offset_color if is_offset else onset_color,
                        # Une seule fois au-dessus du graph (1ʳᵉ ligne de flux).
                        with_label=(row == 0),
                    )
        if stream_drawn and str(stream) in _VOLTAGE_TRACE_KINDS:
            annotations.extend(
                _threshold_line_annotations(
                    request,
                    request.channel_index,
                    row=row,
                    with_legend=(row == 0),
                )
            )
        if stream_drawn == 0:
            msg = f"{short} — calcul…" if pending_any else f"{short} indisponible"
            annotations.append(
                AnnotationSpec(kind="text", text=msg, color="#64748b", row=row)
            )

    if anchor_to_stim and _show_continuous_stim_marks(request) and drawn:
        annotations.extend(
            _stim_markers_rel(request, rows=list(range(len(streams))))
        )

    xlim = None
    if window is not None:
        if window_is_absolute:
            xlim = (window[0] - ref_offset, window[1] - ref_offset)
        else:
            xlim = (float(window[0]), float(window[1]))

    if sync_mode == "trigger":
        time_label = "Temps relatif au trigger (s)"
    elif anchor_to_stim:
        time_label = "Temps relatif à la stimulation (s)"
    else:
        time_label = "Temps (s)"
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
    pending = False
    for index, recording in enumerate(request.recordings):
        # HP/LP : ne jamais lancer sosfilt sur le thread UI.
        if str(stream) in {"hp", "lp"} and not recording.stream_ready(
            stream, request.channel_index
        ):
            pending = True
            continue
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
            if trigger_index is None:
                pending = True
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
        if pending:
            short = STREAM_SHORT_LABELS.get(str(stream), str(stream).upper())
            return _placeholder(
                title=f"{request.channel_name} — {panel_label(panel)}",
                message=f"{short} — calcul…",
                status="pending",
            )
        return _placeholder(
            title=f"{request.channel_name} — {panel_label(panel)}",
            message="Trace d’analyse indisponible.\nCalculer le canal (F6).",
            status="unavailable",
        )

    mode = ""
    if panel.startswith("analysis_"):
        mode = f" ({request.settings.analysis.describe()})"
    annotations = _stim_markers_rel(request)
    if str(stream) in _VOLTAGE_TRACE_KINDS:
        annotations.extend(
            _threshold_line_annotations(request, request.channel_index, row=0)
        )
    return PanelSpec(
        title=f"{request.channel_name} — {panel_label(panel)}{mode}",
        xlabel="Temps relatif à la stimulation (s)",
        ylabel="Potential (µV)",
        series=series,
        xlim=(float(window[0]), float(window[1])),
        ylim=_ylim_tuple(request.settings, stream),
        annotations=annotations,
        status="ok",
        layout_kind="single",
        row_kinds=[stream],
    )


def _time_in_window(x: float, window: tuple[float, float] | None) -> bool:
    if window is None or not np.isfinite(x):
        return bool(np.isfinite(x))
    return float(window[0]) <= float(x) <= float(window[1])


def _continuous_stim_marks(
    recordings: Sequence[Any],
    sync_mode: str,
    window: tuple[float, float] | None,
    *,
    limit: int = 60,
) -> list[tuple[float, str]]:
    """Début et fin de chaque stimulation, en temps affiché.

    Chaque entrée est ``(temps, "onset"|"offset")``. La fin est l’onset plus
    le délai moyen jusqu’au front opposé (``end_marker_s``). Le plafond compte
    les stimulations, pas les traits.
    """
    marks: list[tuple[float, str]] = []
    remaining = max(0, int(limit))
    for recording in recordings:
        if remaining <= 0:
            break
        stims = np.asarray(recording.stimulation_times_s(), dtype=np.float64).ravel()
        offset = float(continuous_sync_offset_s(stims, sync_mode))
        end_rel = getattr(recording, "end_marker_s", None)
        try:
            end_dt = None if end_rel is None else float(end_rel)
        except (TypeError, ValueError):
            end_dt = None
        if end_dt is not None and not np.isfinite(end_dt):
            end_dt = None
        for stim_t in stims:
            if remaining <= 0:
                break
            if not np.isfinite(stim_t):
                continue
            stim_x = float(stim_t) - offset
            start_ok = _time_in_window(stim_x, window)
            end_x = None if end_dt is None else stim_x + end_dt
            end_ok = end_x is not None and _time_in_window(float(end_x), window)
            if not start_ok and not end_ok:
                continue
            if start_ok:
                marks.append((stim_x, "onset"))
            if end_ok:
                marks.append((float(end_x), "offset"))
            remaining -= 1
    return marks


def _stim_markers_rel(
    request: Any,
    *,
    rows: Sequence[int] | None = None,
) -> list[AnnotationSpec]:
    """Vlines onset (t=0) / fin de stim — une copie par ligne (montage multi-rangées).

    Les libellés on/off ne sont posés qu’une fois, sur la première ligne.
    """
    out: list[AnnotationSpec] = []
    if not getattr(request.legend, "show_reference_markers", True):
        return out
    row_ids = [0] if rows is None else [int(r) for r in rows]
    if not row_ids:
        return out
    onset_color, offset_color = _stim_line_colors(request)
    label_row = int(row_ids[0])
    for row in row_ids:
        for marker in request.end_markers():
            _append_stim_vline_with_label(
                out,
                request,
                x=float(marker),
                row=row,
                which="off",
                color=offset_color,
                with_label=(row == label_row),
            )
        _append_stim_vline_with_label(
            out,
            request,
            x=0.0,
            row=row,
            which="on",
            color=onset_color,
            with_label=(row == label_row),
        )
    return out


# ---------------------------------------------------------------------------
# RMS
# ---------------------------------------------------------------------------


def _prepare_rms(request: Any) -> PanelSpec:
    from channel_metrics import recording_rms_window_s, rms_profile_for_trigger

    panel = str(request.panel)
    if panel == "analysis_rms":
        trigger_index = request.settings.analysis.trigger_index()
        mode = f" ({request.settings.analysis.describe()})"
    else:
        trigger_index = {"rms": None, "first_rms": 0, "second_rms": 1}[panel]
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
        t, values = rms_profile_for_trigger(
            recording,
            request.channel_index,
            trigger_index,
            recording_rms_window_s(recording),
        )
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
    """Fenêtre t_rel affichée pour les graphs spikes (zoom / plage / section)."""
    try:
        from panel_registry import _relative_stim_window, spike_scope_time_range

        reference = getattr(request, "reference", None)
        recordings = list(getattr(request, "recordings", []) or [])
        if reference is not None and recordings:
            try:
                index = next(
                    i for i, rec in enumerate(recordings) if rec is reference
                )
            except StopIteration:
                index = 0
            window = spike_scope_time_range(request, reference, index)
            if window is not None:
                return window
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
    if panel in {"analysis_raster_channel"} or panel.endswith("raster_channel"):
        return _prepare_raster(request, compact=True)
    if "raster" in panel:
        return _prepare_raster(request, compact=False)
    if "trial_rate" in panel:
        return _prepare_trial_rate(request)
    if panel in _ANALYSIS_TRACE:
        return _prepare_triggered_trace(request)
    return _prepare_placeholder(
        request, f"Panneau spikes non pris en charge : {panel}", status="unavailable"
    )


def _spike_trains(request: Any, trigger_index: int | None) -> list[list[np.ndarray]]:
    """Spikes filtrés sur la fenêtre affichée (même logique que le spike scope)."""
    from panel_registry import spike_scope_time_range

    trains: list[list[np.ndarray]] = []
    for index, recording in enumerate(request.recordings):
        window = spike_scope_time_range(request, recording, index)
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
            if window is not None and arr.size:
                arr = arr[(arr >= window[0]) & (arr <= window[1])]
            if arr.size >= 2:
                # ms : même unité que le montage ISI / convention spikes.
                intervals.extend((np.diff(np.sort(arr)) * 1000.0).tolist())
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
        xlabel="ISI (ms)",
        ylabel="Nombre",
        series=series,
        status="ok",
        row_kinds=["isi"],
    )


def _prepare_raster(request: Any, *, compact: bool = False) -> PanelSpec:
    """Raster compact (tous essais sur une ligne) ou raster / essai (une ligne par essai)."""
    title_kind = "Raster" if compact else "Raster / essai"
    window = _spike_window(request)
    if window is None:
        return _placeholder(
            title=f"{request.channel_name} — {title_kind}",
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
            y = 0.0 if compact else (y_cursor + float(trial_i))
            xs.extend(arr.tolist())
            ys.extend([y] * int(arr.size))
        if xs:
            series.append(
                ScatterSeries(
                    x=np.asarray(xs, dtype=np.float64),
                    y=np.asarray(ys, dtype=np.float64),
                    color=str(request.colors[index % len(request.colors)]),
                    label=_series_label(request, index),
                    size=3.0 if not compact else 2.5,
                    symbol="|",
                )
            )
        if not compact:
            y_cursor += max(1.0, float(len(st_per_trial)))
    if not series:
        return _placeholder(
            title=f"{request.channel_name} — {title_kind}",
            message="Aucun spike dans cette fenêtre.",
            status="empty",
        )
    ylim = (-0.7, 0.7) if compact else None
    return PanelSpec(
        title=f"{request.channel_name} — {panel_label(request.panel)}",
        xlabel="Temps relatif à la stimulation (s)",
        ylabel="" if compact else "Essai",
        series=series,
        xlim=(float(window[0]), float(window[1])),
        ylim=ylim,
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


def _spike_trace_alpha(settings: Any) -> float:
    """Opacité des traces individuelles du spike scope (la moyenne reste à 1)."""
    raw = getattr(settings, "spike_overlay_alpha", 0.4)
    if raw is None:
        raw = 0.4
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.4
    if not np.isfinite(value):
        value = 0.4
    return max(0.0, min(1.0, value))


def _spike_overlay_mean(
    waves: np.ndarray, stored: np.ndarray | None
) -> np.ndarray | None:
    """Moyenne exacte stockée, sinon moyenne des snippets affichés."""
    if stored is not None:
        arr = np.asarray(stored, dtype=np.float64).ravel()
        if arr.size and np.any(np.isfinite(arr)):
            return arr
    arr = np.asarray(waves, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=0)
    if mean.size == 0 or not np.any(np.isfinite(mean)):
        return None
    return np.asarray(mean, dtype=np.float64)


def _prepare_spike_overlay(request: Any) -> PanelSpec:
    from panel_registry import spike_scope_time_range

    base_lw = _line_width(request)
    trace_lw = max(0.5, base_lw * 0.7)
    mean_lw = max(2.4, base_lw * 2.2)
    settings = getattr(request, "settings", None)
    trace_alpha = _spike_trace_alpha(settings)
    series: list[SeriesItem] = []
    means: list[CurveSeries] = []
    sampling = max(1, int(getattr(request.settings, "sampling_percent", 100) or 100))
    pre_ms = float(getattr(settings, "spike_overlay_pre_ms", 2.0) or 2.0)
    post_ms = float(getattr(settings, "spike_overlay_post_ms", 4.0) or 4.0)
    t_min_ms = -max(0.0, pre_ms)
    t_max_ms = max(1e-6, post_ms)
    abs_peak = 0.0
    for index, recording in enumerate(request.recordings):
        t_range = spike_scope_time_range(request, recording, index)
        t_ms, waves, _times, _total, stored_mean = recording.overlay_for_channel(
            request.channel_index,
            t_range_s=t_range,
            pre_ms=pre_ms,
            post_ms=post_ms,
        )
        if waves is None or getattr(waves, "shape", (0,))[0] == 0:
            continue
        if np.asarray(t_ms).size == 0:
            continue
        n = int(waves.shape[0])
        step = max(1, int(np.ceil(100.0 / sampling))) if sampling < 100 else 1
        color = str(request.colors[index % len(request.colors)])
        label = _series_label(request, index)
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
                    label="",
                    linewidth=trace_lw,
                    alpha=trace_alpha,
                )
            )
        mean_w = _spike_overlay_mean(waves, stored_mean)
        if mean_w is not None and mean_w.size != t_arr.size:
            mean_w = _spike_overlay_mean(waves, None)
        if mean_w is not None and t_arr.size and mean_w.size == t_arr.size:
            peak = float(np.nanmax(np.abs(mean_w)))
            if np.isfinite(peak):
                abs_peak = max(abs_peak, peak)
            means.append(
                CurveSeries(
                    x=t_arr,
                    y=mean_w,
                    color=color,
                    label=label,
                    linewidth=mean_lw,
                    alpha=1.0,
                )
            )
        if _show_spike_threshold_lines(request):
            try:
                thr = float(recording.threshold_uv(request.channel_index))
                if np.isfinite(thr):
                    abs_peak = max(abs_peak, abs(thr))
            except Exception:
                pass
    series.extend(means)
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
        annotations=_threshold_line_annotations(request, request.channel_index, row=0),
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


def _montage_continuous_axis_labels(
    request: Any,
    channels: Sequence[int],
    kinds: Sequence[str],
) -> list[str] | None:
    """Libellés d’axes « canal » ou « canal TYPE » selon le réglage d’affichage."""
    recordings = list(getattr(request, "recordings", []) or [])
    if not recordings or not channels:
        return None
    names = list(getattr(recordings[0], "channel_names", []) or [])
    show_kind = bool(
        getattr(getattr(request, "settings", None), "montage_show_kind_label", True)
    )
    labels: list[str] = []
    for index, ch in enumerate(channels):
        ch_i = int(ch)
        name = names[ch_i] if 0 <= ch_i < len(names) else f"CH{ch_i}"
        kind = str(kinds[index]) if index < len(kinds) else ""
        labels.append(montage_row_axis_label(name, kind, show_kind=show_kind))
    return labels


def _montage_xlabel(
    mode: str,
    sync_mode: str,
    last_kind: str,
    *,
    stim_anchored: bool = False,
) -> str:
    """Libellé X selon la dernière ligne (unités incompatibles entre kinds)."""
    if last_kind == "overlay":
        return "Temps relatif au spike (ms)"
    if last_kind == "isi":
        return "ISI (ms)"
    if last_kind == "trial_rate":
        return "Essai"
    if last_kind in _MONTAGE_STREAM_KINDS and mode == "continuous":
        if sync_mode == "trigger":
            return "Temps relatif au trigger (s)"
        if stim_anchored:
            return "Temps relatif à la stimulation (s)"
        return "Temps (s)"
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
    show_kind = bool(getattr(request.settings, "montage_show_kind_label", True))
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
    # Spikes / RMS : toujours t_rel. Continuous : axe X manuel ou toute la courbe.
    extra_section_x = _relative_stim_window(request)
    stream_x = x_limits if mode == "continuous" else extra_section_x
    anchor_to_stim = False
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
            row_labels.append(
                montage_row_axis_label(name, str(kind), show_kind=show_kind)
            )
            row_kinds.append(str(kind))
            row_channels.append(int(ch))
            row_keys.append((int(ch), str(kind)))
            row_highlight.append(int(ch) == int(request.channel_index))
            if kind in _MONTAGE_STREAM_KINDS:
                row_ylims.append(
                    _ylim_tuple(request.settings, kind, montage=True)
                )
            elif kind == "rms":
                row_ylims.append(
                    _ylim_tuple(request.settings, "rms", montage=True)
                )
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
                        offset = continuous_anchor_offset_s(
                            recording.stimulation_times_s(),
                            sync_mode,
                            stim_index=stim_index,
                            anchor_to_stim=anchor_to_stim,
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
                        # Continuous / moyenne / stim : HP/LP en attente de filtre.
                        if kind in {"hp", "lp"}:
                            pending_any = True
                            row_pending = True
                        elif mode == "average":
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
                if kind in _VOLTAGE_TRACE_KINDS and row_drawn:
                    annotations.extend(
                        _threshold_line_annotations(
                            request, int(ch), row=row, with_legend=False
                        )
                    )
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
                from panel_registry import spike_scope_time_range

                color_default = _MONTAGE_EXTRA_COLORS["overlay"]
                pre_ms = float(
                    getattr(request.settings, "spike_overlay_pre_ms", 2.0) or 2.0
                )
                post_ms = float(
                    getattr(request.settings, "spike_overlay_post_ms", 4.0) or 4.0
                )
                row_xlim = (-max(0.0, pre_ms), max(1e-6, post_ms))
                for index, recording in enumerate(recordings):
                    if ch >= int(recording.n_channels):
                        continue
                    t_ms, waves, _times, _total, mean = recording.overlay_for_channel(
                        ch,
                        t_range_s=spike_scope_time_range(request, recording, index),
                        pre_ms=pre_ms,
                        post_ms=post_ms,
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
                        wave_alpha = _spike_trace_alpha(request.settings)
                        for wave in arr[:n_show]:
                            series.append(
                                CurveSeries(
                                    x=t_arr,
                                    y=np.asarray(wave, dtype=np.float64),
                                    color=str(color),
                                    label="",
                                    linewidth=max(0.4, lw * 0.55),
                                    alpha=wave_alpha,
                                    row=row,
                                )
                            )
                        row_drawn += 1
                        drawn += 1
                if row_drawn:
                    annotations.extend(
                        _threshold_line_annotations(
                            request, int(ch), row=row, with_legend=False
                        )
                    )
            elif kind in {"raster", "psth", "trial_rate", "isi"}:
                from draw_primitives import (
                    _isi_time_and_values_s,
                    _psth_mean_hz,
                    _trial_mean_firing_rate_hz,
                )
                from panel_registry import spike_scope_time_range

                color_default = _MONTAGE_EXTRA_COLORS.get(kind, "#334155")
                for index, recording in enumerate(recordings):
                    if ch >= int(recording.n_channels):
                        continue
                    spike_win = spike_scope_time_range(request, recording, index)
                    if spike_win is None:
                        spike_win = extra_section_x
                    trains = recording.spike_times(
                        ch, trigger_index=trigger_index, t_range_s=spike_win
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
                        if t_rel.size < 2 or spike_win is None:
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
                            t_range_s=spike_win,
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
                        if spike_win is None or not trains:
                            continue
                        rates = _trial_mean_firing_rate_hz(trains, spike_win)
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
                            trains, isi_window_s=spike_win
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
    xlabel = _montage_xlabel(
        mode, str(sync_mode), last_kind, stim_anchored=anchor_to_stim
    )

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

    # Pointillés de stim : début et fin sur les flux continus.
    if mode == "continuous" and _show_continuous_stim_marks(request):
        stream_rows = [
            row_i
            for row_i, kind in enumerate(row_kinds)
            if kind in _MONTAGE_STREAM_KINDS
        ]
        if anchor_to_stim:
            # Même on/off relatifs que stimulation / moyenne.
            annotations.extend(_stim_markers_rel(request, rows=stream_rows))
        else:
            stim_marks = _continuous_stim_marks(recordings, str(sync_mode), stream_x)
            if stim_marks:
                onset_color, offset_color = _stim_line_colors(request)
                label_row = stream_rows[0] if stream_rows else None
                for row_i in stream_rows:
                    for stim_x, role in stim_marks:
                        _append_stim_vline_with_label(
                            annotations,
                            request,
                            x=float(stim_x),
                            row=row_i,
                            which="off" if role == "offset" else "on",
                            color=offset_color if role == "offset" else onset_color,
                            with_label=(row_i == label_row),
                        )
    elif mode != "continuous":
        marker_rows = [
            row_i
            for row_i, kind in enumerate(row_kinds)
            if kind in {"raw", "hp", "lp", "rms", "psth", "raster"}
        ]
        annotations.extend(_stim_markers_rel(request, rows=marker_rows))

    row_ylims = _apply_montage_shared_ylims(
        request.settings, row_kinds, row_ylims, series
    )

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
        row_min_height_px=_montage_row_min_height_px(request),
    )


def _montage_row_min_height_px(request: Any) -> int:
    """Hauteur mini d’une ligne, lue sur les réglages (défaut 36 px)."""
    settings = getattr(request, "settings", None)
    return max(36, int(getattr(settings, "montage_row_min_height_px", 36) or 36))


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
    annotations: list[AnnotationSpec] = []
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
        row_ylims.append(
            _ylim_tuple(request.settings, stream, montage=True)
        )
        row_kinds.append(stream)
        row_channels.append(int(ch))
        row_keys.append((int(ch), stream))
        row_highlight.append(int(ch) == int(request.channel_index))
        row_drawn = 0
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
            row_drawn += 1
            drawn += 1
        if str(stream) in _VOLTAGE_TRACE_KINDS and row_drawn:
            annotations.extend(
                _threshold_line_annotations(
                    request, int(ch), row=row, with_legend=False
                )
            )

    if drawn == 0:
        return _prepare_placeholder(request, "Montage vide.", status="empty")

    n_rows = max(1, len(channel_indices))
    annotations.extend(_stim_markers_rel(request, rows=range(n_rows)))
    return PanelSpec(
        title=panel_label(panel),
        xlabel="Temps relatif à la stimulation (s)",
        series=series,
        xlim=(float(window[0]), float(window[1])),
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
        row_min_height_px=_montage_row_min_height_px(request),
    )


def _prepare_montage_second_to_third(request: Any) -> PanelSpec:
    """Passe-bas continu du début de la fenêtre 2e stim à la fin de la 3e."""
    try:
        from panel_registry import _montage_channel_slice
    except Exception:
        return _prepare_placeholder(request, "Helpers montage indisponibles.", status="unavailable")

    recordings = list(request.recordings)
    reference = recordings[0]
    n_channels = min(int(rec.n_channels) for rec in recordings)
    channel_indices, _n_pages = _montage_channel_slice(request, n_channels)
    max_pts = _max_points(request)
    lw = max(0.6, _line_width(request) * 0.85)
    color = STREAM_PLOT_COLORS.get("lp", "#1e40af")
    series: list[SeriesItem] = []
    row_labels: list[str] = []
    row_ylims: list[tuple[float, float] | None] = []
    row_kinds: list[str] = []
    row_channels: list[int] = []
    row_keys: list[tuple[int, str]] = []
    row_highlight: list[bool] = []
    annotations: list[AnnotationSpec] = []
    drawn = 0
    pending = 0
    missing_span = 0
    x_min: float | None = None
    x_max: float | None = None

    for row, ch in enumerate(channel_indices):
        name = (
            reference.channel_names[ch]
            if ch < len(reference.channel_names)
            else f"CH{ch}"
        )
        row_labels.append(name)
        row_ylims.append(
            _ylim_tuple(request.settings, "lp", montage=True)
        )
        row_kinds.append("lp")
        row_channels.append(int(ch))
        row_keys.append((int(ch), "lp-span"))
        row_highlight.append(int(ch) == int(request.channel_index))
        row_drawn = 0
        for index, recording in enumerate(recordings):
            if ch >= int(recording.n_channels):
                continue
            outcome, payload = _second_to_third_lp_span(recording, int(ch))
            if outcome == "pending":
                pending += 1
                continue
            if outcome != "ok" or payload is None:
                missing_span += 1
                continue
            t, y, third_s = payload
            t, y = _decimate(t, y, max_pts)
            series.append(
                CurveSeries(
                    x=t,
                    y=y,
                    color=recording_trace_color(request.colors, index, fallback=color),
                    label=_series_label(request, index, "2e→3e", stream="lp"),
                    linewidth=lw,
                    row=row,
                )
            )
            row_drawn += 1
            drawn += 1
            lo, hi = float(np.min(t)), float(np.max(t))
            x_min = lo if x_min is None else min(x_min, lo)
            x_max = hi if x_max is None else max(x_max, hi)
            if index == 0:
                onset_color, _offset_color = _stim_line_colors(request)
                _append_stim_vline_with_label(
                    annotations,
                    request,
                    x=0.0,
                    row=row,
                    which="on",
                    color=onset_color,
                    with_label=(row == 0),
                )
                annotations.append(
                    AnnotationSpec(kind="vline", x=float(third_s), color="#c2410c", row=row)
                )
        if row_drawn:
            annotations.extend(
                _threshold_line_annotations(
                    request, int(ch), row=row, with_legend=False
                )
            )

    if drawn == 0:
        if pending and missing_span == 0:
            return _prepare_placeholder(
                request,
                "Passe-bas pas encore calculé pour cet intervalle.\nCalculer les canaux (F6).",
                status="pending",
            )
        return _prepare_placeholder(
            request,
            "Intervalle 2e→3e indisponible.\nIl faut au moins 3 stimulations valides et le flux passe-bas.",
            status="unavailable",
        )

    n_rows = max(1, len(channel_indices))
    xlim = (x_min, x_max) if x_min is not None and x_max is not None and x_max > x_min else None
    return PanelSpec(
        title=panel_label("montage_second_to_third_lp"),
        xlabel="Temps relatif à la 2e stimulation (s)",
        series=series,
        xlim=xlim,
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
        row_min_height_px=_montage_row_min_height_px(request),
    )


def _second_to_third_lp_span(
    recording: Any, ch: int
) -> tuple[str, tuple[np.ndarray, np.ndarray, float] | None]:
    """``(status, (t_s, y_uv, t_3rd_s))`` — t=0 à l’onset de la 2e stimulation."""
    source = getattr(recording, "source", None)
    if source is None:
        return "missing", None
    triggers = np.asarray(getattr(source, "valid_triggers", []), dtype=np.int64)
    if triggers.size < 3:
        return "short", None
    if not bool(recording.stream_ready("lp", int(ch))):
        return "pending", None
    seg = recording.meta.segmentation
    start = int(triggers[1]) - int(seg.pre_n)
    end = int(triggers[2]) + int(seg.post_n)
    lp = getattr(source, "lowpass", None)
    try:
        n_samples = int(lp.shape[1])
        y = np.asarray(lp[int(ch)][start:end], dtype=np.float64)
    except Exception:
        return "missing", None
    fs = float(recording.meta.fs)
    if start < 0 or end > n_samples or end <= start or fs <= 0 or y.size == 0:
        return "bounds", None
    trig2 = int(triggers[1])
    t = (np.arange(y.size, dtype=np.float64) + float(start - trig2)) / fs
    third_s = float(int(triggers[2]) - trig2) / fs
    return "ok", (t, y, third_s)


def _prepare_mea_layout(request: Any) -> PanelSpec:
    from probe_layout import match_contact_index

    layout = getattr(request, "probe_layout", None)
    if layout is None:
        return _prepare_placeholder(
            request,
            "Aucune carte MEA chargée.\nChoisir un JSON de sonde dans les paramètres.",
            status="unavailable",
        )
    positions = np.asarray(layout.positions_um, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] == 0:
        return _prepare_placeholder(request, "Carte MEA vide.", status="unavailable")
    active = match_contact_index(layout, str(request.channel_name))
    if active is None:
        return _prepare_placeholder(
            request,
            f"Le canal {request.channel_name} n’est pas sur cette carte MEA.",
            status="unavailable",
        )
    series: list[SeriesItem] = [
        ScatterSeries(
            x=positions[:, 0],
            y=positions[:, 1],
            color="#94a3b8",
            label="Contacts",
            size=8.0,
        ),
        ScatterSeries(
            x=np.asarray([positions[int(active), 0]], dtype=np.float64),
            y=np.asarray([positions[int(active), 1]], dtype=np.float64),
            color="#ea580c",
            label=str(request.channel_name),
            size=14.0,
        ),
    ]
    pad = 0.08 * max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1])), 1.0)
    return PanelSpec(
        title=f"Carte MEA — {request.channel_name}",
        xlabel="x (µm)",
        ylabel="y (µm)",
        series=series,
        xlim=(float(np.min(positions[:, 0]) - pad), float(np.max(positions[:, 0]) + pad)),
        ylim=(float(np.min(positions[:, 1]) - pad), float(np.max(positions[:, 1]) + pad)),
        annotations=[
            AnnotationSpec(
                kind="text",
                x=float(positions[int(active), 0]),
                y=float(positions[int(active), 1]),
                text=str(request.channel_name),
                color="#9a3412",
            )
        ],
        status="ok",
    )


def _session_tick(when: Any, index: int) -> str:
    if hasattr(when, "strftime"):
        try:
            return when.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    text = str(when or "").strip()
    return text or f"Session {index + 1}"


def _prepare_impedance(request: Any) -> PanelSpec:
    sessions = list(getattr(request, "impedance_sessions", []) or [])
    if not sessions:
        return _prepare_placeholder(
            request,
            "Pas de données d’impédance.\nDéposer le CSV compagnon à côté de l’enregistrement.",
            status="unavailable",
        )
    channel = str(request.channel_name)
    xs: list[float] = []
    ys: list[float] = []
    ticks: list[tuple[float, str]] = []
    for index, session in enumerate(sessions):
        magnitudes = getattr(session, "magnitudes_ohm", {}) or {}
        value = float(magnitudes.get(channel, float("nan")))
        if not np.isfinite(value) or value <= 0:
            continue
        x = float(len(xs))
        xs.append(x)
        ys.append(value)
        ticks.append((x, _session_tick(getattr(session, "when", None), index)))
    if not xs:
        return _prepare_placeholder(request, f"Pas d’impédance pour {channel}.", status="empty")
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    return PanelSpec(
        title=f"Impédance |Z| @ 1 kHz — {channel}",
        xlabel="Session",
        ylabel="|Z| @ 1 kHz (Ω)",
        series=[
            CurveSeries(x=x_arr, y=y_arr, color="#2563eb", label=channel, linewidth=1.2),
            ScatterSeries(x=x_arr, y=y_arr, color="#2563eb", label="", size=8.0),
        ],
        x_ticks=ticks,
        y_log=True,
        status="ok",
    )


def _included_channels(request: Any, recording: Any) -> list[int]:
    hidden = tuple(getattr(getattr(request, "settings", None), "hidden_channels", ()) or ())
    names = list(getattr(recording, "channel_names", ()) or ())
    n_channels = int(getattr(recording, "n_channels", 0) or 0)
    return included_channel_indices(n_channels, names, hidden)


def _prepare_summary_rms(request: Any) -> PanelSpec:
    lw = _line_width(request)
    max_pts = _max_points(request)
    series: list[SeriesItem] = []
    for index, recording in enumerate(request.recordings):
        t, values = recording.mean_rms_across_channels(
            "mean", channels=_included_channels(request, recording)
        )
        if values.size == 0:
            continue
        x, y = _decimate(
            np.asarray(t, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            max_pts,
        )
        series.append(
            CurveSeries(
                x=x,
                y=y,
                color=str(request.colors[index % len(request.colors)]),
                label=_series_label(request, index, "RMS moyen"),
                linewidth=lw,
            )
        )
    if not series:
        return _prepare_placeholder(request, "Résumé RMS indisponible.", status="unavailable")
    return PanelSpec(
        title=panel_label("summary_rms"),
        xlabel="Temps relatif à la stimulation (s)",
        ylabel="RMS moyen (µV)",
        series=series,
        ylim=_ylim_tuple(request.settings, "rms"),
        status="ok",
        row_kinds=["rms"],
    )


def _prepare_summary_rms_table(request: Any) -> PanelSpec:
    recordings = list(request.recordings)
    reference = recordings[0]
    indices = _included_channels(request, reference)
    if not indices:
        return _prepare_placeholder(request, "Aucun canal sélectionné.", status="unavailable")
    headers = ["Canal"] + [
        str(request.labels[i] if i < len(request.labels) else f"Enreg. {i + 1}")
        for i in range(len(recordings))
    ]
    rows: list[list[str]] = []
    any_value = False
    highlight_row = -1
    highlight = int(getattr(request, "channel_index", -1) or -1)
    for row_index, ch in enumerate(indices):
        if ch == highlight:
            highlight_row = row_index
        name = (
            reference.channel_names[ch]
            if ch < len(reference.channel_names)
            else f"CH{ch}"
        )
        cells = [str(name)]
        for recording in recordings:
            _t, profile = recording.rms_profile("mean", ch)
            if profile.size and np.any(np.isfinite(profile)):
                cells.append(f"{float(np.nanmean(profile)):.2f}")
                any_value = True
            else:
                cells.append("—")
        rows.append(cells)
    if not any_value:
        return _prepare_placeholder(
            request, "Table RMS indisponible (canaux non calculés).", status="unavailable"
        )
    if not 0 <= highlight_row < len(rows):
        highlight_row = -1
    return PanelSpec(
        title=panel_label("summary_rms_table"),
        status="ok",
        layout_kind="table",
        table_headers=headers,
        table_rows=rows,
        table_highlight_row=highlight_row,
    )


def _prepare_summary_impedance(request: Any) -> PanelSpec:
    sessions = list(getattr(request, "impedance_sessions", []) or [])
    if not sessions:
        return _prepare_placeholder(request, "Aucune session d’impédance.", status="unavailable")
    xs: list[float] = []
    ys: list[float] = []
    ticks: list[tuple[float, str]] = []
    for index, session in enumerate(sessions):
        magnitudes = getattr(session, "magnitudes_ohm", {}) or {}
        values = np.asarray(list(magnitudes.values()), dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            continue
        x = float(len(xs))
        xs.append(x)
        ys.append(float(np.mean(values)))
        ticks.append((x, _session_tick(getattr(session, "when", None), index)))
    if not xs:
        return _prepare_placeholder(request, "Aucune impédance moyenne valide.", status="unavailable")
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    return PanelSpec(
        title=panel_label("summary_impedance"),
        xlabel="Session",
        ylabel="|Z| moyen @ 1 kHz (Ω)",
        series=[
            CurveSeries(x=x_arr, y=y_arr, color="#0f766e", label="|Z| moyen", linewidth=1.2),
            ScatterSeries(x=x_arr, y=y_arr, color="#0f766e", size=8.0),
        ],
        x_ticks=ticks,
        y_log=True,
        status="ok",
    )
