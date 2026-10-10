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
    AxisLimits,
    LegendSettings,
    PanelPlacement,
    PanelStyle,
    TextFace,
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

# Marges d’axes partagées (fraction de figure) : même left/right ⇒ même largeur
# pixel de la zone de tracé, quel que soit le panneau (aperçu, montage, extras).
_AXES_LEFT = 0.14
_AXES_RIGHT = 0.98
_AXES_RIGHT_SCALEBARS = 0.90
_AXES_TOP_SINGLE = 0.88
_AXES_BOTTOM_SINGLE = 0.14
_AXES_BOTTOM_SINGLE_SCALEBARS = 0.18
_AXES_TOP_MONTAGE = 0.995
_AXES_BOTTOM_MONTAGE = 0.035
_AXES_BOTTOM_MONTAGE_SCALEBARS = 0.12
# Plancher de réserve légende « below » (fraction figure) si on ne connaît
# pas encore le nombre d’entrées ; la vraie hauteur est estimée dynamiquement.
_AXES_BOTTOM_LEGEND_PAD_MIN = 0.10


def _disable_layout_engine(figure: Any) -> None:
    """Couper constrained/tight layout pour que ``subplots_adjust`` tienne."""
    try:
        figure.set_layout_engine(None)
    except Exception:
        try:
            figure.set_constrained_layout(False)
        except Exception:
            pass


def _count_legend_entries(figure: Any) -> int:
    """Nombre max d’entrées parmi les légendes déjà présentes sur la figure."""
    count = 0
    for ax in list(getattr(figure, "axes", []) or []):
        legend = ax.get_legend()
        if legend is None:
            continue
        try:
            texts = legend.get_texts()
            count = max(count, len(texts))
        except Exception:
            continue
    return count


def _legend_below_pad(
    legend: LegendSettings,
    *,
    n_entries: int,
    fig_height_in: float,
) -> float:
    """Hauteur estimée (fraction de figure) d’une légende placée sous les axes.

    Un pad fixe (~0.12) est trop petit dès que la figure est basse ou que la
    légende a plusieurs lignes : le bas du cadre est alors croppé.
    """
    ncol = max(1, int(legend.columns))
    nrows = max(1, (max(1, int(n_entries)) + ncol - 1) // ncol)
    font_size = max(4.0, float(legend.font_size))
    # Unités « em » matplotlib (borderpad / labelspacing / handleheight).
    height_em = nrows * 1.55 + 1.0
    if legend.frame:
        height_em += 0.35
    height_in = (font_size * height_em) / 72.0
    fig_h = max(1.0, float(fig_height_in))
    return min(0.42, max(_AXES_BOTTOM_LEGEND_PAD_MIN, height_in / fig_h + 0.025))


def _bottom_margin_for(
    *,
    montage: bool,
    show_scale_bars: bool,
    top: float,
    legend: LegendSettings | None = None,
    n_legend_entries: int = 0,
    fig_height_in: float = 4.0,
) -> float:
    """Marge basse figure : ticks/xlabel, et place pour une légende sous le graph."""
    if montage:
        base = (
            _AXES_BOTTOM_MONTAGE_SCALEBARS if show_scale_bars else _AXES_BOTTOM_MONTAGE
        )
    else:
        base = (
            _AXES_BOTTOM_SINGLE_SCALEBARS if show_scale_bars else _AXES_BOTTOM_SINGLE
        )
    if legend is None or not legend.visible or legend.location != "below":
        return base
    gap = max(0.0, float(legend.gap))
    entries = int(n_legend_entries) if n_legend_entries > 0 else 3
    pad = _legend_below_pad(
        legend, n_entries=entries, fig_height_in=fig_height_in
    )
    # bbox_to_anchor y=-gap (coords axes) : convertir en fraction de figure.
    # bottom > (gap * top + legend_pad) / (1 + gap)
    needed = (gap * float(top) + pad) / (1.0 + gap)
    return min(0.55, max(base, needed))


def _apply_uniform_axes_box(
    figure: Any,
    *,
    show_scale_bars: bool = False,
    montage: bool = False,
    legend: LegendSettings | None = None,
) -> None:
    """Imposer la même boîte d’axes (largeur pixel identique) à toute la figure."""
    _disable_layout_engine(figure)
    right = _AXES_RIGHT_SCALEBARS if show_scale_bars else _AXES_RIGHT
    if montage:
        top = _AXES_TOP_MONTAGE
        hspace = 0.0
    else:
        top = _AXES_TOP_SINGLE
        hspace = 0.18
    try:
        fig_h = float(figure.get_figheight())
    except Exception:
        fig_h = 4.0
    bottom = _bottom_margin_for(
        montage=montage,
        show_scale_bars=show_scale_bars,
        top=top,
        legend=legend,
        n_legend_entries=_count_legend_entries(figure),
        fig_height_in=fig_h,
    )
    try:
        figure.subplots_adjust(
            left=_AXES_LEFT,
            right=right,
            top=top,
            bottom=bottom,
            hspace=hspace,
        )
    except Exception:
        pass
    _fit_below_legend(figure, legend)


def _fit_below_legend(figure: Any, legend: LegendSettings | None) -> None:
    """Si la légende below dépasse encore le bas de la figure, pousser la marge."""
    if legend is None or not legend.visible or legend.location != "below":
        return
    canvas = getattr(figure, "canvas", None)
    get_renderer = getattr(canvas, "get_renderer", None) if canvas is not None else None
    if not callable(get_renderer):
        return
    try:
        renderer = get_renderer()
    except Exception:
        return
    gap = max(0.0, float(legend.gap))
    lowest: float | None = None
    for ax in list(getattr(figure, "axes", []) or []):
        artist = ax.get_legend()
        if artist is None:
            continue
        try:
            bbox = artist.get_window_extent(renderer).transformed(
                figure.transFigure.inverted()
            )
            y0 = float(bbox.y0)
        except Exception:
            continue
        lowest = y0 if lowest is None else min(lowest, y0)
    if lowest is None or lowest >= 0.012:
        return
    # Monter la légende de ``overflow`` en augmentant ``bottom``.
    overflow = 0.012 - lowest
    delta = overflow / max(1e-6, 1.0 + gap)
    try:
        current = float(figure.subplotpars.bottom)
    except Exception:
        return
    new_bottom = min(0.55, current + delta)
    if new_bottom <= current + 1e-4:
        return
    try:
        figure.subplots_adjust(bottom=new_bottom)
    except Exception:
        pass


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
                needs_streams=montage and key != "montage_continuous_raw",
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
        # Échelle X manuelle : s’applique aux graphs temporels sans zoom dédié.
        # Panels en t_rel (analyse / spikes) : ignorer si hors plage relative
        # (souvent des bornes continuous en temps fichier).
        manual_x = self.settings.x_limits.as_tuple()
        t_rel = reference.t_rel
        if manual_x is not None:
            if not _panel_expects_relative_x(self.panel):
                return manual_x
            if t_rel.size:
                rel0, rel1 = float(t_rel[0]), float(t_rel[-1])
                lo, hi = float(manual_x[0]), float(manual_x[1])
                if hi < lo:
                    lo, hi = hi, lo
                if hi >= rel0 and lo <= rel1:
                    return lo, hi
            # Hors plage → retomber sur la fenêtre relative par défaut.
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


def _underline_path_effect(*, linewidth: float = 0.8, y_offset_frac: float = 0.12) -> Any:
    """Soulignement sous le glyphe (matplotlib n’expose pas de flag natif)."""
    from matplotlib.patheffects import AbstractPathEffect
    from matplotlib.path import Path as MplPath

    class _UnderlinePathEffect(AbstractPathEffect):
        def __init__(self) -> None:
            super().__init__()
            self._linewidth = float(linewidth)
            self._y_offset_frac = float(y_offset_frac)

        def draw_path(
            self, renderer: Any, gc: Any, tpath: Any, affine: Any, rgbFace: Any = None
        ) -> None:
            renderer.draw_path(gc, tpath, affine, rgbFace)
            extents = tpath.get_extents()
            height = max(float(extents.height), 1e-6)
            y = float(extents.y0) - height * self._y_offset_frac
            line = MplPath([(float(extents.x0), y), (float(extents.x1), y)])
            gc0 = renderer.new_gc()
            try:
                gc0.copy_properties(gc)
                gc0.set_linewidth(self._linewidth)
                renderer.draw_path(gc0, line, affine, None)
            finally:
                gc0.restore()

    return _UnderlinePathEffect()


def apply_text_face(text: Any, face: TextFace | None) -> None:
    """Appliquer gras / italique / souligné à un artiste ``Text`` matplotlib."""
    if text is None or face is None:
        return
    try:
        text.set_fontweight("bold" if face.bold else "normal")
        text.set_fontstyle("italic" if face.italic else "normal")
    except Exception:
        return
    try:
        if face.underline:
            text.set_path_effects([_underline_path_effect()])
        else:
            text.set_path_effects([])
    except Exception:
        pass


def _scale_bar_units(panel: str) -> tuple[str | None, str | None]:
    """Unités (x, y) pour les barres d’échelle ; ``None`` = barre omise."""
    if panel in {"mea_layout", "summary_rms_table"}:
        return (None, None)
    if panel in {"impedance", "summary_impedance"}:
        return (None, "Ω")
    if panel in {"analysis_overlay"}:
        return ("ms", "µV")
    if panel in {
        "analysis_psth",
        "psth",
        "first_psth",
        "second_psth",
    }:
        return ("s", "Hz")
    if panel in {"analysis_trial_rate", "trial_rate"}:
        return (None, "Hz")
    if panel in {"analysis_isi", "isi", "first_isi", "second_isi"}:
        return ("s", "ms")
    if panel in {"analysis_raster", "analysis_raster_channel", "raster"}:
        return ("s", None)
    if panel in {
        "analysis_rms",
        "rms",
        "first_rms",
        "second_rms",
        "summary_rms",
    }:
        return ("s", "µV")
    # Traces tension (aperçu, analyse, montages, continuous…).
    return ("s", "µV")


def _scale_bar_lengths(
    style: PanelStyle,
    *,
    x_unit: str | None,
    y_unit: str | None,
    ax: Any,
) -> tuple[float | None, float | None]:
    """Longueurs X/Y (unités de données) selon réglages manuels ou auto."""
    from plot_utils import resolve_scale_bar_length

    try:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    except Exception:
        return (None, None)
    x_span = abs(float(xlim[1]) - float(xlim[0]))
    y_span = abs(float(ylim[1]) - float(ylim[0]))

    x_size: float | None = None
    if x_unit is not None and x_span > 0:
        time_s = float(getattr(style, "scale_bar_time_s", 0.1) or 0.1)
        # Stocké en secondes ; convertir si le panneau est en ms.
        manual_x = time_s * 1000.0 if x_unit == "ms" else time_s
        x_size = resolve_scale_bar_length(
            x_span,
            manual=bool(getattr(style, "scale_bar_time_manual", False)),
            value=manual_x,
        )

    y_size: float | None = None
    if y_unit is not None and y_span > 0:
        y_size = resolve_scale_bar_length(
            y_span,
            manual=bool(getattr(style, "scale_bar_amplitude_manual", False)),
            value=float(getattr(style, "scale_bar_amplitude", 100.0) or 100.0),
        )
    return (x_size, y_size)


def _apply_scale_bars(
    ax: Any,
    style: PanelStyle,
    *,
    panel: str,
    keep_ylabel: bool = False,
    keep_xlabel: bool = False,
    draw_bars: bool = True,
) -> None:
    """Masquer l’échelle des bords et, si demandé, tracer les barres flottantes."""
    from plot_utils import clear_floating_scale_bars, draw_floating_scale_bars, hide_edge_scale

    if not bool(getattr(style, "show_scale_bars", False)):
        clear_floating_scale_bars(ax)
        return
    hide_edge_scale(ax, keep_ylabel=keep_ylabel, keep_xlabel=keep_xlabel)
    # Avec barres flottantes : pas de cadre (style montage EEG).
    for spine in ax.spines.values():
        spine.set_visible(False)
    if not draw_bars:
        clear_floating_scale_bars(ax)
        return
    x_unit, y_unit = _scale_bar_units(panel)
    if x_unit is None and y_unit is None:
        clear_floating_scale_bars(ax)
        return
    x_size, y_size = _scale_bar_lengths(style, x_unit=x_unit, y_unit=y_unit, ax=ax)
    draw_floating_scale_bars(
        ax,
        x_unit=x_unit,
        y_unit=y_unit,
        x_size=x_size,
        y_size=y_size,
        fontsize=float(style.tick_font_size),
        linewidth=max(1.4, float(style.line_width) * 1.15),
    )


def _apply_montage_scale_bars(figure: Any, request: RenderRequest) -> None:
    """Échelle Y à droite de chaque ligne (marge) ; barre temps sur le dernier axe.

    Chaque ligne reflète son propre ``ylim``. L’axe X est partagé → une seule
    barre horizontale en bas. Les longueurs suivent les valeurs manuelles
    (µV / s) ou l’auto.
    """
    from plot_utils import clear_floating_scale_bars, draw_floating_scale_bars, hide_edge_scale

    axes = list(getattr(figure, "axes", []) or [])
    if not axes:
        return
    style = request.style
    show_bars = bool(getattr(style, "show_scale_bars", False))
    # Une seule ligne (ex. full_recording WIDE) : marges « single », pas montage.
    # Sinon ticks / xlabel / légende below sont coupés (bottom montage ≈ 3.5 %).
    _apply_uniform_axes_box(
        figure,
        show_scale_bars=show_bars,
        montage=len(axes) > 1,
        legend=request.legend,
    )
    if not show_bars:
        for ax in axes:
            clear_floating_scale_bars(ax)
        return

    x_unit, y_unit = _scale_bar_units(request.panel)
    last = len(axes) - 1
    fontsize = float(style.tick_font_size)
    linewidth = max(1.4, float(style.line_width) * 1.15)

    for row, ax in enumerate(axes):
        hide_edge_scale(ax, keep_ylabel=True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if x_unit is None and y_unit is None:
            clear_floating_scale_bars(ax)
            continue
        x_size, y_size = _scale_bar_lengths(style, x_unit=x_unit, y_unit=y_unit, ax=ax)
        # Y dans la gouttière à droite de chaque ligne.
        if y_unit is not None:
            draw_floating_scale_bars(
                ax,
                x_unit=None,
                y_unit=y_unit,
                x_size=None,
                y_size=y_size,
                fontsize=fontsize,
                linewidth=linewidth,
                gutter=True,
            )
        else:
            clear_floating_scale_bars(ax)
        # Temps partagé : sous le dernier graph (hors zone de tracé).
        if row == last and x_unit is not None:
            draw_floating_scale_bars(
                ax,
                x_unit=x_unit,
                y_unit=None,
                x_size=x_size,
                y_size=None,
                fontsize=fontsize,
                linewidth=linewidth,
                clear=False,
                below=True,
            )


def _apply_axis_style(
    ax: Any,
    style: PanelStyle,
    *,
    panel: str | None = None,
    keep_ylabel: bool = False,
) -> None:
    ax.title.set_fontsize(style.title_font_size)
    apply_text_face(ax.title, style.title_face)
    ax.xaxis.label.set_fontsize(style.label_font_size)
    ax.yaxis.label.set_fontsize(style.label_font_size)
    apply_text_face(ax.xaxis.label, style.label_face)
    apply_text_face(ax.yaxis.label, style.label_face)
    inside = bool(style.ticks_inside)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=style.tick_font_size,
        direction="in" if inside else "out",
        top=inside,
        right=inside,
    )
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        apply_text_face(label, style.tick_face)
    for text in ax.texts:
        if text.get_fontsize() > style.label_font_size * 1.6:
            text.set_fontsize(style.label_font_size)
    for spine in ax.spines.values():
        spine.set_visible(bool(style.show_borders))
    if panel is not None and bool(getattr(style, "show_scale_bars", False)):
        _apply_scale_bars(ax, style, panel=panel, keep_ylabel=keep_ylabel)
    figure = getattr(ax, "figure", None)
    if figure is not None:
        _apply_figure_title_style(figure, style)


def _apply_figure_title_style(figure: Any, style: PanelStyle) -> None:
    """Style du ``suptitle`` (montages) aligné sur la police de titre."""
    sup = getattr(figure, "_suptitle", None)
    if sup is None:
        return
    try:
        sup.set_fontsize(style.title_font_size)
    except Exception:
        pass
    apply_text_face(sup, style.title_face)


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


def _merge_legend_labels(
    labels: Sequence[str], overrides: Sequence[str]
) -> list[str]:
    """Remplacer les libellés auto par les overrides non vides (même ordre)."""
    if not overrides:
        return list(labels)
    merged: list[str] = []
    for index, label in enumerate(labels):
        if index < len(overrides) and str(overrides[index]).strip():
            merged.append(str(overrides[index]).strip())
        else:
            merged.append(str(label))
    return merged


def _apply_legend(
    ax: Any,
    settings: LegendSettings,
    *,
    label_overrides: Sequence[str] = (),
) -> None:
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
    labels = _merge_legend_labels(labels, label_overrides)
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
        # Layout engine désactivé : la marge basse réserve la place.
        legend.set_in_layout(False)
        try:
            legend.set_clip_on(False)
        except Exception:
            pass
        for text in legend.get_texts():
            apply_text_face(text, settings.face)


def _legend_overrides(request: RenderRequest) -> tuple[str, ...]:
    return request.settings.texts.resolved_legend_labels()


def _apply_text_overrides(figure: Any, request: RenderRequest) -> None:
    """Appliquer titre / xlabel / ylabel manuels après le rendu automatique."""
    texts = request.settings.texts
    title = str(texts.title or "").strip()
    xlabel = str(texts.xlabel or "").strip()
    ylabel = str(texts.ylabel or "").strip()
    if not title and not xlabel and not ylabel:
        return
    axes = list(getattr(figure, "axes", []) or [])
    if not axes:
        return
    multi = len(axes) > 1
    if title:
        sup = getattr(figure, "_suptitle", None)
        if multi and sup is not None:
            try:
                sup.set_text(title)
            except Exception:
                figure.suptitle(title, fontsize=request.style.title_font_size)
            _apply_figure_title_style(figure, request.style)
        elif multi:
            figure.suptitle(title, fontsize=request.style.title_font_size)
            _apply_figure_title_style(figure, request.style)
        else:
            axes[0].set_title(title)
    if xlabel:
        axes[-1].set_xlabel(xlabel)
    if ylabel:
        if multi:
            for ax in axes:
                current = str(ax.get_ylabel() or "").strip()
                if current:
                    ax.set_ylabel(ylabel)
        else:
            axes[0].set_ylabel(ylabel)


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
    if suffix and request.settings.texts.show_series_suffix:
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
    """Appliquer une échelle Y manuelle, ou ré-autoscaler si « Manuel » est off."""
    bounds = limits.as_tuple() if limits is not None else None
    if bounds is not None:
        ax.set_ylim(bounds[0], bounds[1])
        return
    # Mode auto : un set_ylim précédent (manuel) désactive l’autoscaling
    # matplotlib — il faut le réactiver explicitement au refresh style.
    try:
        ax.set_autoscaley_on(True)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
    except Exception:
        pass


def _apply_montage_row_ylim(ax: Any, kind: str, settings: Any) -> None:
    """Échelle Y d’une ligne de revue montage (flux, RMS, extras)."""
    key = str(kind).strip().lower()
    if key in _MONTAGE_STREAM_KINDS:
        _apply_stream_ylim(ax, settings, key)
        return
    if key == "rms":
        _apply_ylim(ax, getattr(settings, "rms_ylim", None))
        return
    if key == "raster":
        # Fixé dans ``_montage_draw_extra_row`` ; ne pas écraser au refresh.
        return
    if key == "psth":
        _apply_ylim(ax, AxisLimits())
        _psth_ylim_nonnegative(ax)
        return
    if key in {"overlay", "trial_rate", "isi"}:
        # Ces kinds n’ont pas de réglage Y manuel : toujours ré-autoscaler,
        # sinon un ylim figé (souvent 0–1) coupe WV / FR / ISI.
        _apply_ylim(ax, AxisLimits())


def _apply_xlim(ax: Any, limits: Any) -> None:
    bounds = limits.as_tuple() if limits is not None else None
    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[1])


def _stream_for_panel(panel: str) -> str | None:
    """Flux WIDE/HIGH/LOW associé à un panneau, ou ``None`` si non applicable."""
    if panel in _STREAM_FOR_PANEL:
        return _STREAM_FOR_PANEL[panel]
    if panel in _MONTAGE_SPECS:
        return _MONTAGE_SPECS[panel][0]
    key = str(panel)
    if "rms" in key:
        return None
    if "_hp" in key or key.endswith("_hp"):
        return "hp"
    if "_lp" in key or key.endswith("_lp"):
        return "lp"
    if "_raw" in key or key.endswith("_raw"):
        return "raw"
    return None


def _panel_expects_relative_x(panel: str) -> bool:
    """True si le panneau est en temps relatif à la stimulation (pas temps fichier)."""
    key = str(panel)
    if key in {"full_recording", "montage_continuous_raw"}:
        return False
    if key.startswith("summary_") or key in {"mea_layout", "impedance"}:
        return False
    return True


def _relative_stim_window(request: RenderRequest) -> tuple[float, float] | None:
    """Fenêtre d’affichage en t_rel (spikes / RMS / analyse), jamais le temps fichier."""
    reference = request.reference
    if reference is None:
        return None
    t_rel = np.asarray(reference.t_rel, dtype=np.float64)
    if t_rel.size == 0:
        return None
    rel0, rel1 = float(t_rel[0]), float(t_rel[-1])
    if request.placement.has_custom_zoom:
        # Zooms analyse absolus déjà convertis par ``section_window``.
        if str(request.panel).startswith("analysis_"):
            return request.section_window()
        t0 = float(request.placement.zoom_t0_s)
        t1 = float(request.placement.zoom_t1_s)
        if t1 < t0:
            t0, t1 = t1, t0
        if request.placement.zoom_absolute:
            stim_index = request.settings.analysis.trigger_index()
            converted = _absolute_range_to_relative(
                reference, t0, t1, stim_index=stim_index
            )
            return converted if converted is not None else (rel0, rel1)
        return t0, t1
    manual_x = request.settings.x_limits.as_tuple()
    if manual_x is not None:
        lo, hi = float(manual_x[0]), float(manual_x[1])
        if hi < lo:
            lo, hi = hi, lo
        if hi >= rel0 and lo <= rel1:
            return lo, hi
    return rel0, rel1


def _psth_ylim_nonnegative(ax: Any) -> None:
    """Les taux de décharge ne descendent pas sous 0 Hz."""
    try:
        y0, y1 = ax.get_ylim()
    except Exception:
        return
    top = float(y1) if np.isfinite(y1) else 1.0
    if top <= 0:
        top = 1.0
    ax.set_ylim(0.0, top)


def _ylim_for_stream(settings: Any, stream: str) -> Any:
    """Échelle Y manuelle pour un flux (fallback raw)."""
    fn = getattr(settings, "ylim_for_stream", None)
    if callable(fn):
        return fn(stream)
    return getattr(settings, "raw_ylim", None)


def _apply_stream_ylim(ax: Any, settings: Any, stream: str) -> None:
    _apply_ylim(ax, _ylim_for_stream(settings, stream))


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
    _apply_stream_ylim(ax, request.settings, stream)
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
    _apply_stream_ylim(ax, request.settings, stream)
    ax.set_title(
        f"{request.channel_name} — {panel_label(request.panel)}{_section_suffix(request)}"
    )
    return "ok"


def _analysis_trigger_index(request: RenderRequest) -> int | None:
    return request.settings.analysis.trigger_index()


def _analysis_mode_label(request: RenderRequest) -> str:
    return request.settings.analysis.describe()


def _render_full_recording(figure: Any, request: RenderRequest) -> str:
    """Traces continues : un panneau (= une figure) par flux WIDE / HIGH / LOW."""
    _disable_layout_engine(figure)
    placed = str(getattr(request.placement, "stream", "") or "").strip()
    if placed in {"raw", "hp", "lp"}:
        streams = [placed]
    else:
        streams = list(request.settings.resolved_continuous_streams()) or ["raw"]
    window = request.section_window() if request.placement.has_custom_zoom else None
    if window is None:
        window = request.settings.x_limits.as_tuple()
    n_streams = max(1, len(streams))
    axes = figure.subplots(n_streams, 1, sharex=True, squeeze=False)[:, 0]
    # Une seule ligne (cas usuel aperçu) : boîte « single » pour coller aux autres graphs.
    _apply_uniform_axes_box(
        figure,
        show_scale_bars=bool(getattr(request.style, "show_scale_bars", False)),
        montage=n_streams > 1,
        legend=request.legend,
    )
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
            # Always bound points on the UI path — never full-rate arange.
            if str(stream) in {"hp", "lp"} and not recording.stream_ready(
                stream, request.channel_index
            ):
                continue
            t, values = recording.continuous_trace(
                stream,
                request.channel_index,
                max_points=max_pts,
                require_ready=str(stream) in {"hp", "lp"},
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
            pending = any(
                str(stream) in {"hp", "lp"}
                and not rec.stream_ready(stream, request.channel_index)
                for rec in request.recordings
            )
            msg = f"{short} — calcul…" if pending else f"{short} unavailable"
            ax.text(
                0.5,
                0.5,
                msg,
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
        _apply_stream_ylim(ax, request.settings, stream)
        if request.style.grid:
            ax.grid(True, alpha=request.style.grid_alpha)
        else:
            ax.grid(False)
        _apply_legend(ax, request.legend, label_overrides=_legend_overrides(request))
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
        _apply_axis_style(ax, request.style)

    if drawn == 0:
        figure.clear()
        return _fail(
            figure.add_subplot(111),
            request,
            "Continuous recording unavailable.\n"
            "Process a .rhs file first (F5), then select a channel.",
        )
    _apply_montage_scale_bars(figure, request)
    figure._erg_montage_row_streams = list(streams)
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
    _apply_stream_ylim(ax, request.settings, stream)
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
    from draw_primitives import _draw_spike_panels_multi_channel, _psth_mean_hz

    panel = request.panel
    trigger_index = None
    if panel.startswith("first_"):
        trigger_index = 0
    elif panel.startswith("second_"):
        trigger_index = 1
    elif panel.startswith("analysis_"):
        trigger_index = _analysis_trigger_index(request)
    # Toujours une fenêtre relative (évite les x_limits continuous absolus).
    window = _relative_stim_window(request)
    if window is None:
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

    mode_note = ""
    if panel.startswith("analysis_"):
        mode_note = f" ({_analysis_mode_label(request)})"

    # PSTH aperçu : même chemin que le montage (_plot_curve + décimation),
    # ylim ≥ 0, labels compacts — évite le chrome lourd de draw_primitives.
    if kind == "psth":
        t_rel = np.asarray(reference.t_rel, dtype=np.float64)
        if t_rel.size < 2:
            return _fail(ax, request, "Section unavailable (no stimulation-end marker).")
        bin_w = max(float(request.settings.psth_bin_window_s), 1.0 / max(float(reference.meta.fs), 1.0))
        drawn = 0
        for index, st_per_trial in enumerate(trains):
            n_trials = max(1, len(st_per_trial))
            centers, rate = _psth_mean_hz(
                st_per_trial,
                t_rel,
                n_trials,
                bin_w,
                t_range_s=window,
            )
            if centers.size < 2:
                continue
            flags = list(request.legend_flags or [])
            show_leg = True if not flags else bool(flags[index] if index < len(flags) else True)
            label = request.labels[index] if show_leg and index < len(request.labels) else None
            _plot_curve(
                ax,
                centers,
                rate,
                style=request.style,
                color=request.colors[index % len(request.colors)],
                label=label,
            )
            drawn += 1
        if drawn == 0:
            return _fail(ax, request, "No spike detected in this window.")
        ax.set_xlim(float(window[0]), float(window[1]))
        ax.set_xlabel("Time relative to stimulation (s)")
        ax.set_ylabel("Rate (Hz)")
        _psth_ylim_nonnegative(ax)
        if request.legend.show_reference_markers:
            from draw_primitives import _draw_onset_offset_lines

            _draw_onset_offset_lines(
                ax,
                end_markers=request.end_markers(),
                label_in_legend=False,
                appearance=_appearance(request),
            )
        ax.set_title(
            f"{request.channel_name} — {panel_label(panel)}{mode_note}"
            f"{_section_suffix(request)}"
        )
        return "ok"

    axes = {"raster": None, "psth": None, "trial": None, "isi": None}
    if kind == "raster":
        axes["raster"] = ax
    elif kind == "trial_rate":
        axes["trial"] = ax
    else:
        axes["isi"] = ax

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
        show_psth=False,
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


def _render_channel_raster_panel(ax: Any, request: RenderRequest) -> str:
    """Raster compact (style montage) : tous les essais empilés sur une ligne."""
    window = _relative_stim_window(request)
    if window is None:
        window = request.section_window()
    if window is None:
        return _fail(ax, request, "Section unavailable (no stimulation-end marker).")
    multi_rec = len(request.recordings) > 1
    line_w = max(0.6, float(request.style.line_width) * 0.85)
    drawn = _montage_draw_extra_row(
        ax,
        request,
        ch=int(request.channel_index),
        kind="raster",
        multi_rec=multi_rec,
        line_w=line_w,
        row_i=0,
        section_x=window,
    )
    if drawn == 0:
        return _fail(ax, request, "No spike detected in this window.")
    ax.set_xlim(float(window[0]), float(window[1]))
    ax.set_yticks([])
    ax.set_xlabel("Time relative to stimulation (s)")
    mode_note = f" ({_analysis_mode_label(request)})"
    ax.set_title(
        f"{request.channel_name} — {panel_label(request.panel)}"
        f"{mode_note}{_section_suffix(request)}"
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


def _short_table_label(text: str, max_len: int = 28) -> str:
    """Tronque un libellé de colonne pour éviter le chevauchement dans les tables."""
    s = str(text).strip() or "Recording"
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _render_summary_rms_table(ax: Any, request: RenderRequest) -> str:
    recordings = request.recordings
    if not recordings:
        return _fail(ax, request, "No recording selected.")
    reference = recordings[0]
    n_channels = min(recording.n_channels for recording in recordings)
    if n_channels == 0:
        return _fail(ax, request, "No channel available.")
    ax.set_axis_off()
    # Libellés courts : les noms de fichiers complets se chevauchent sinon.
    n_cols = 1 + len(recordings)
    label_max = max(12, min(28, 48 // max(1, n_cols - 1)))
    header = ["Channel"] + [
        _short_table_label(request.labels[i], label_max) for i in range(len(recordings))
    ]
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
    # bbox : force le tableau dans l’axe (sinon les lignes débordent et sont coupées).
    table = ax.table(
        cellText=rows,
        colLabels=header,
        loc="upper center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.02, 0.96, 0.88],
    )
    table.auto_set_font_size(False)
    n_rows = len(rows)
    font_size = max(5.0, min(float(request.style.tick_font_size), 320.0 / max(n_rows + 1, 1)))
    table.set_fontsize(font_size)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_edgecolor("#b0b0b0")
        props: dict[str, Any] = {}
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            props["weight"] = "bold"
        elif row - 1 == highlight:
            cell.set_facecolor("#fde68a")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
        if col == 0:
            props["ha"] = "left"
            cell.PAD = 0.02
        if props:
            cell.set_text_props(**props)
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


def _montage_review_channel_slice(
    request: RenderRequest, n_channels: int
) -> tuple[list[int], int]:
    """Canaux de la page courante + nombre de pages (revue montage GUI)."""
    visible = _montage_visible_indices(request, n_channels)
    per_page = max(1, int(getattr(request.settings, "montage_review_channels", 10) or 10))
    n_pages = max(1, (len(visible) + per_page - 1) // per_page) if visible else 1
    page = max(0, min(n_pages - 1, int(getattr(request.settings, "montage_review_page", 0) or 0)))
    start = page * per_page
    return visible[start : start + per_page], n_pages


def _montage_continuous_channels(request: RenderRequest, n_channels: int) -> list[int]:
    """Revue GUI : tranche paginée. Legacy ``pageN`` : pagination PDF."""
    iid = str(getattr(request.placement, "instance_id", "") or "")
    if iid.startswith("page"):
        channel_indices, _n_pages = _montage_channel_slice(request, n_channels)
        return channel_indices
    channel_indices, _n_pages = _montage_review_channel_slice(request, n_channels)
    return channel_indices


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
        _apply_stream_ylim(ax, request.settings, stream)
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
    _apply_figure_title_style(figure, request.style)
    _apply_montage_scale_bars(figure, request)
    figure._erg_montage_row_streams = [stream] * len(channel_indices)
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

    _disable_layout_engine(figure)

    from draw_primitives import _draw_onset_offset_lines

    axes = figure.subplots(rows, 1, sharex=True, squeeze=False)[:, 0]
    show_bars = bool(getattr(request.style, "show_scale_bars", False))
    _apply_uniform_axes_box(
        figure, show_scale_bars=show_bars, montage=True, legend=request.legend
    )
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
        if request.legend.show_reference_markers:
            _draw_onset_offset_lines(
                ax,
                end_markers=request.end_markers(),
                label_in_legend=False,
                appearance=draw_app,
            )
        ax.set_xlim(window[0], window[1])
        _apply_stream_ylim(ax, request.settings, stream)
        if ch == request.channel_index:
            ax.set_facecolor(CHANNEL_HIGHLIGHT_FACE)
        if row == 0 and multi_rec:
            _apply_legend(ax, request.legend, label_overrides=_legend_overrides(request))
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
    _apply_figure_title_style(figure, request.style)
    _apply_montage_scale_bars(figure, request)
    figure._erg_montage_row_streams = [stream] * rows
    figure._erg_montage_state = {
        "kind": "triggered",
        "geom": _montage_triggered_geom(request, channel_indices, window),
        "lines": lines_by_row,
    }
    return "ok" if drawn else "empty"


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


def _montage_review_mode(request: RenderRequest) -> str:
    """Mode d’affichage de la revue montage — aligné sur l’aperçu canal."""
    content = str(getattr(request.settings, "preview_content", "") or "").strip()
    if content in {"continuous", "average", "stimulation"}:
        return content
    mode = str(getattr(request.settings.analysis, "mode", "") or "").strip()
    if mode in {"average", "stimulation"}:
        return mode
    return "continuous"


def _montage_review_kinds(request: RenderRequest) -> list[str]:
    """Kinds empilés : flux WIDE/HIGH/LOW + extras Pipeline (RMS / spikes…)."""
    kinds: list[str] = list(request.settings.resolved_continuous_streams())
    analysis = request.settings.analysis
    if analysis.show_rms:
        kinds.append("rms")
    if analysis.show_psth:
        kinds.append("psth")
    if analysis.show_trial_rate:
        kinds.append("trial_rate")
    if analysis.show_raster:
        kinds.append("raster")
    if analysis.show_isi:
        kinds.append("isi")
    if analysis.show_overlay:
        kinds.append("overlay")
    return kinds or ["raw"]


def _montage_review_curve(
    recording: Any,
    stream: str,
    ch: int,
    *,
    mode: str,
    stim_index: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(t, y)`` pour une ligne tension selon le mode aperçu."""
    if mode == "continuous":
        if str(stream) in {"hp", "lp"} and not recording.stream_ready(stream, ch):
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        t, values = recording.continuous_trace(
            stream, ch, max_points=max_points, require_ready=str(stream) in {"hp", "lp"}
        )
        return np.asarray(t, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if mode == "stimulation":
        if stim_index >= int(getattr(recording, "n_trials", 0) or 0):
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        curve = recording.trigger_window(stim_index, stream, ch)
    else:
        curve = recording.mean(stream, ch)
    if curve is None:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    t = np.asarray(recording.t_rel, dtype=np.float64)
    y = np.asarray(curve, dtype=np.float64)
    if t.size != y.size or y.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if y.size > max_points > 0:
        step = max(1, int(np.ceil(y.size / max_points)))
        t = t[::step]
        y = y[::step]
    return t, y


def _montage_continuous_max_points(request: RenderRequest, n_channels: int, n_kinds: int) -> int:
    """Budget de points stable (indépendant des canaux masqués → cache memmap hit)."""
    style_cap = max(200, int(request.style.max_points_per_curve))
    total_rows = max(1, int(n_channels) * max(1, int(n_kinds)))
    return min(style_cap, max(240, 10000 // total_rows))


def _montage_continuous_row_specs(
    request: RenderRequest,
) -> tuple[list[tuple[int, str, str]], Any, str, list[str], int] | None:
    """``(row_specs, reference, mode, kinds, max_pts)`` ou ``None`` si indisponible."""
    recordings = request.recordings
    if not recordings:
        return None
    reference = recordings[0]
    if not any(recording.has_streams for recording in recordings):
        return None
    mode = _montage_review_mode(request)
    n_channels = min(int(recording.n_channels) for recording in recordings)
    channel_indices = _montage_continuous_channels(request, n_channels)
    kinds = _montage_review_kinds(request)
    row_specs: list[tuple[int, str, str]] = []
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
            row_specs.append((ch, kind, f"{name} {short}"))
    max_pts = _montage_continuous_max_points(request, n_channels, len(kinds))
    return row_specs, reference, mode, kinds, max_pts


def _montage_continuous_base_geom(request: RenderRequest) -> tuple[Any, ...]:
    """Géométrie hors ``hidden_channels`` — pour réutiliser axes / cache courbes."""
    mode = _montage_review_mode(request)
    kinds = tuple(_montage_review_kinds(request))
    if mode == "continuous":
        rec_fp = tuple(
            (
                str(rec.meta.source_path),
                bool(rec.has_streams),
                int(getattr(rec, "n_channels", 0) or 0),
            )
            for rec in request.recordings
        )
    else:
        rec_fp = tuple(
            (
                str(rec.meta.source_path),
                int(rec.data_generation),
                bool(rec.has_streams),
                int(getattr(rec, "n_channels", 0) or 0),
            )
            for rec in request.recordings
        )
    n_channels = 0
    if request.recordings:
        n_channels = min(int(rec.n_channels) for rec in request.recordings)
    max_pts = _montage_continuous_max_points(request, n_channels, len(kinds) or 1)
    return (
        mode,
        int(request.settings.analysis.stim_index),
        rec_fp,
        max_pts,
        kinds,
        request.settings.x_limits.as_tuple(),
        str(request.settings.time_sync),
        bool(request.settings.continuous_mark_stims),
        bool(request.legend.show_reference_markers),
        bool(getattr(request.style, "show_scale_bars", False)),
        int(request.settings.montage_row_min_height_px),
        str(getattr(request.placement, "instance_id", "") or ""),
        int(getattr(request.settings, "montage_review_channels", 10) or 10),
        int(getattr(request.settings, "montage_review_page", 0) or 0),
    )


def _try_shrink_continuous_montage(figure: Any, request: RenderRequest) -> str | None:
    """Retirer des lignes montage sans relire les traces (masquage de canaux)."""
    stored = getattr(figure, "_erg_montage_state", None)
    if not isinstance(stored, dict) or stored.get("kind") != "continuous_review":
        return None
    if stored.get("base_geom") != _montage_continuous_base_geom(request):
        return None
    built = _montage_continuous_row_specs(request)
    if built is None:
        return None
    row_specs, _reference, mode, kinds, _max_pts = built
    if not row_specs:
        return None
    old_specs: list[tuple[int, str]] = list(stored.get("row_keys") or [])
    new_keys = [(ch, kind) for ch, kind, _label in row_specs]
    if not new_keys or new_keys == old_specs:
        return None
    # Shrink = masquage de canaux uniquement. Retirer un type de courbe
    # (kind) doit toujours passer par un rebuild complet.
    old_kinds = {str(kind) for _ch, kind in old_specs}
    new_kinds = {str(kind) for _ch, kind in new_keys}
    if old_kinds != new_kinds:
        return None
    old_set = set(old_specs)
    if not set(new_keys).issubset(old_set):
        return None  # canaux ré-affichés → rebuild (avec cache courbes)
    axes = list(getattr(figure, "axes", []) or [])
    if len(axes) != len(old_specs):
        return None
    by_key = {key: ax for key, ax in zip(old_specs, axes)}
    keep = [by_key[key] for key in new_keys if key in by_key]
    if len(keep) != len(new_keys):
        return None
    for key, ax in by_key.items():
        if key not in set(new_keys):
            try:
                figure.delaxes(ax)
            except Exception:
                return None

    rows = len(keep)
    min_h = max(36, int(request.settings.montage_row_min_height_px))
    fig_h = max(3.0, (rows * min_h) / 96.0)
    try:
        # Hauteur seule — largeur gérée par le canvas / viewport.
        figure.set_figheight(fig_h, forward=False)
    except Exception:
        pass
    show_bars = bool(getattr(request.style, "show_scale_bars", False))
    right_margin = _AXES_RIGHT_SCALEBARS if show_bars else _AXES_RIGHT
    try:
        fig_h = float(figure.get_figheight())
    except Exception:
        fig_h = 4.0
    bottom_margin = _bottom_margin_for(
        montage=True,
        show_scale_bars=show_bars,
        top=_AXES_TOP_MONTAGE,
        legend=request.legend,
        n_legend_entries=_count_legend_entries(figure),
        fig_height_in=fig_h,
    )
    try:
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(
            rows,
            1,
            figure=figure,
            left=_AXES_LEFT,
            right=right_margin,
            top=_AXES_TOP_MONTAGE,
            bottom=bottom_margin,
            hspace=0.0,
        )
        for index, ax in enumerate(keep):
            ax.set_subplotspec(gs[index])
        _fit_below_legend(figure, request.legend)
    except Exception:
        _apply_uniform_axes_box(
            figure, show_scale_bars=show_bars, montage=True, legend=request.legend
        )

    selected = int(request.channel_index)
    inside = bool(request.style.ticks_inside)
    tick_fs = max(5.0, float(request.style.tick_font_size) - 1.5)
    for row_i, (ax, (ch, kind, _label)) in enumerate(zip(keep, row_specs)):
        ax.set_facecolor(CHANNEL_HIGHLIGHT_FACE if ch == selected else "#ffffff")
        ax.tick_params(
            axis="both",
            labelsize=tick_fs,
            direction="in" if inside else "out",
            top=inside,
            right=inside,
            labelbottom=(row_i == rows - 1),
        )
        if row_i < rows - 1:
            ax.set_xlabel("")
        elif kind == "overlay":
            ax.set_xlabel("Time relative to spike (ms)")
        elif kind == "isi":
            ax.set_xlabel("ISI (ms)")
        elif kind == "trial_rate":
            ax.set_xlabel("Trial")
        elif kind in _MONTAGE_STREAM_KINDS and mode == "continuous":
            sync_mode = request.settings.time_sync
            ax.set_xlabel(
                "Time relative to trigger (s)" if sync_mode == "trigger" else "Time (s)"
            )
        else:
            ax.set_xlabel("Time relative to stimulation (s)")

    _apply_montage_scale_bars(figure, request)
    figure._erg_montage_row_channels = [ch for ch, _kind, _label in row_specs]
    figure._erg_montage_row_streams = [
        kind if kind in _MONTAGE_STREAM_KINDS else "raw" for _ch, kind, _label in row_specs
    ]
    figure._erg_montage_row_kinds = [kind for _ch, kind, _label in row_specs]
    figure._erg_montage_review_mode = mode
    curves = stored.get("curves") if isinstance(stored.get("curves"), dict) else {}
    figure._erg_montage_state = {
        "kind": "continuous_review",
        "base_geom": stored.get("base_geom"),
        "row_keys": new_keys,
        "curves": curves,
        "kinds": list(kinds),
        "mode": mode,
    }
    return "ok"


def _montage_rms_kind(request: RenderRequest) -> str:
    trigger_index = _analysis_trigger_index(request)
    if trigger_index is None:
        return "mean"
    if int(trigger_index) == 0:
        return "first"
    if int(trigger_index) == 1:
        return "second"
    return "mean"


def _montage_draw_extra_row(
    ax: Any,
    request: RenderRequest,
    *,
    ch: int,
    kind: str,
    multi_rec: bool,
    line_w: float,
    row_i: int,
    section_x: tuple[float, float] | None,
) -> int:
    """Trace une ligne RMS / spike pour un canal. Retourne le nombre de séries dessinées."""
    from draw_primitives import (
        _isi_time_and_values_s,
        _psth_mean_hz,
        _trial_mean_firing_rate_hz,
    )

    recordings = request.recordings
    color_default = _MONTAGE_EXTRA_COLORS.get(kind, "#334155")
    trigger_index = _analysis_trigger_index(request)
    drawn = 0

    for index, recording in enumerate(recordings):
        if ch >= int(recording.n_channels):
            continue
        color = request.colors[index] if multi_rec else color_default
        label = _series_label(request, index) if row_i == 0 else None

        if kind == "rms":
            t, values = recording.rms_profile(_montage_rms_kind(request), ch)
            if values.size == 0:
                continue
            t_arr = np.asarray(t, dtype=np.float64)
            y_arr = np.asarray(values, dtype=np.float64)
            if section_x is not None:
                mask = _mask_window(t_arr, section_x)
                if mask is not None:
                    t_arr, y_arr = t_arr[mask], y_arr[mask]
            if t_arr.size < 2:
                continue
            _plot_curve(
                ax, t_arr, y_arr, style=request.style, color=color, label=label, line_width=line_w
            )
            drawn += 1
            continue

        if kind == "overlay":
            t_ms, waves, _times, _total, mean = recording.overlay_for_channel(
                ch, t_range_s=section_x
            )
            if mean is not None and np.asarray(mean).size and np.asarray(t_ms).size:
                _plot_curve(
                    ax,
                    np.asarray(t_ms, dtype=np.float64),
                    np.asarray(mean, dtype=np.float64),
                    style=request.style,
                    color=color,
                    label=label,
                    line_width=line_w,
                )
                drawn += 1
            elif waves is not None and getattr(waves, "size", 0) and np.asarray(t_ms).size:
                arr = np.asarray(waves, dtype=np.float64)
                t_arr = np.asarray(t_ms, dtype=np.float64)
                n_show = min(12, int(arr.shape[0]))
                for wave in arr[:n_show]:
                    ax.plot(t_arr, wave, color=color, lw=max(0.4, line_w * 0.55), alpha=0.35)
                drawn += 1
            continue

        trains = recording.spike_times(
            ch, trigger_index=trigger_index, t_range_s=section_x
        )
        if kind == "raster":
            if not any(getattr(tr, "size", 0) for tr in trains):
                continue
            positions = [np.asarray(tr, dtype=np.float64) for tr in trains if getattr(tr, "size", 0)]
            if not positions:
                continue
            ax.eventplot(
                positions,
                colors=[color],
                lineoffsets=0.0,
                linelengths=0.85,
                linewidths=max(0.5, line_w * 0.7),
            )
            ax.set_ylim(-0.7, 0.7)
            drawn += 1
            continue

        if kind == "psth":
            t_rel = np.asarray(recording.t_rel, dtype=np.float64)
            if t_rel.size < 2 or section_x is None:
                continue
            n_trials = max(1, len(trains) if trains else int(getattr(recording, "n_trials", 1) or 1))
            centers, rate = _psth_mean_hz(
                trains,
                t_rel,
                n_trials,
                float(request.settings.psth_bin_window_s),
                t_range_s=section_x,
            )
            if centers.size < 2:
                continue
            _plot_curve(
                ax, centers, rate, style=request.style, color=color, label=label, line_width=line_w
            )
            drawn += 1
            continue

        if kind == "trial_rate":
            if section_x is None or not trains:
                continue
            rates = _trial_mean_firing_rate_hz(trains, section_x)
            if rates.size == 0:
                continue
            x = np.arange(1, rates.size + 1, dtype=np.float64)
            ax.bar(x, rates, color=color, width=0.8, alpha=0.85, label=label)
            drawn += 1
            continue

        if kind == "isi":
            t_isi, d_isi = _isi_time_and_values_s(trains, isi_window_s=section_x)
            if d_isi.size == 0:
                continue
            # Histogramme compact (ms) pour rester lisible en ligne montage.
            ms = np.asarray(d_isi, dtype=np.float64) * 1000.0
            ms = ms[np.isfinite(ms) & (ms > 0)]
            if ms.size == 0:
                continue
            bins = min(24, max(8, int(np.sqrt(ms.size))))
            ax.hist(ms, bins=bins, color=color, alpha=0.8, label=label)
            drawn += 1
            continue

    return drawn


def _render_montage_continuous(figure: Any, request: RenderRequest) -> str:
    """Revue montage : un sous-graphique par canal×graph, empilés sans espace.

    Tous les graphs Pipeline (WIDE/HIGH/LOW/RMS/Spikes) sont intégrés au montage
    multi-canaux, avec le même mode que l’aperçu (continuous / moyenne / stim).
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
    _disable_layout_engine(figure)

    mode = _montage_review_mode(request)
    stim_index = max(0, int(request.settings.analysis.stim_index))
    multi_rec = len(recordings) > 1
    n_channels = min(int(recording.n_channels) for recording in recordings)
    channel_indices = _montage_continuous_channels(request, n_channels)
    kinds = _montage_review_kinds(request)
    stream_colors = dict(STREAM_PLOT_COLORS)
    has_extras = any(kind not in _MONTAGE_STREAM_KINDS for kind in kinds)

    row_specs: list[tuple[int, str, str]] = []
    for ch in channel_indices:
        name = reference.channel_names[ch] if ch < len(reference.channel_names) else f"CH{ch}"
        for kind in kinds:
            if kind in _MONTAGE_STREAM_KINDS:
                short = STREAM_SHORT_LABELS.get(kind, kind.upper())
            else:
                short = _MONTAGE_EXTRA_LABELS.get(kind, kind.upper())
            row_specs.append((ch, kind, f"{name} {short}"))
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
        # Hauteur seule : la largeur suit le widget (pas de plancher 6″).
        figure.set_figheight(fig_h, forward=False)
    except Exception:
        pass

    # Budget stable sur n_channels totaux (pas rows visibles) → cache memmap hit
    # quand on coche/décoche des canaux.
    max_pts = _montage_continuous_max_points(request, n_channels, len(kinds))
    curve_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    prior_cache = getattr(figure, "_erg_montage_curve_cache", None)
    if isinstance(prior_cache, dict):
        curve_cache.update(prior_cache)
    figure._erg_montage_curve_cache = None
    line_w = max(0.6, float(request.style.line_width) * 0.85)
    tick_fs = max(5.0, float(request.style.tick_font_size) - 1.5)
    label_fs = max(5.0, float(request.style.tick_font_size) - 1.0)
    inside = bool(request.style.ticks_inside)

    # sharex seulement si toutes les lignes partagent la même famille temporelle.
    share_x = not has_extras and not (
        mode == "continuous" and any(k not in _MONTAGE_STREAM_KINDS for k in kinds)
    )
    axes = figure.subplots(rows, 1, sharex=share_x, squeeze=False)[:, 0]
    show_bars = bool(getattr(request.style, "show_scale_bars", False))
    _apply_uniform_axes_box(
        figure, show_scale_bars=show_bars, montage=True, legend=request.legend
    )

    sync_mode = request.settings.time_sync
    offsets = [
        continuous_sync_offset_s(recording.stimulation_times_s(), sync_mode)
        for recording in recordings
    ]
    selected = int(request.channel_index)
    global_x = request.settings.x_limits.as_tuple()
    # Spikes / RMS : toujours t_rel (pas les x_limits continuous absolus).
    section_x = _relative_stim_window(request)
    stream_x = global_x if global_x is not None else (
        section_x if mode != "continuous" else None
    )
    drawn = 0
    data_x_min: float | None = None
    data_x_max: float | None = None

    from draw_primitives import _draw_onset_offset_lines, mark_stim_times

    draw_app = _appearance(request)
    stim_xs: list[float] = []
    if mode == "continuous" and request.settings.continuous_mark_stims:
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
                if stream_x is not None and not (stream_x[0] <= stim_x <= stream_x[1]):
                    continue
                stim_xs.append(stim_x)
                remaining -= 1

    for row_i, (ch, kind, label) in enumerate(row_specs):
        ax = axes[row_i]
        channel_drawn = 0

        if kind in _MONTAGE_STREAM_KINDS:
            for index, recording in enumerate(recordings):
                if ch >= int(recording.n_channels) or not recording.has_streams:
                    continue
                cache_key = (index, int(ch), str(kind), mode, int(stim_index), int(max_pts))
                cached = curve_cache.get(cache_key)
                if cached is not None:
                    t_arr, y_arr = cached
                else:
                    t_arr, y_arr = _montage_review_curve(
                        recording,
                        kind,
                        ch,
                        mode=mode,
                        stim_index=stim_index,
                        max_points=max_pts,
                    )
                    curve_cache[cache_key] = (t_arr, y_arr)
                if y_arr.size == 0:
                    continue
                if mode == "continuous":
                    offset = float(offsets[index]) if index < len(offsets) else 0.0
                    t_arr = t_arr - offset
                if stream_x is not None:
                    mask = _mask_window(t_arr, stream_x)
                    if mask is not None:
                        t_arr = t_arr[mask]
                        y_arr = y_arr[mask]
                if t_arr.size < 2:
                    continue
                t0, t1 = float(t_arr[0]), float(t_arr[-1])
                data_x_min = t0 if data_x_min is None else min(data_x_min, t0)
                data_x_max = t1 if data_x_max is None else max(data_x_max, t1)
                color = (
                    request.colors[index]
                    if multi_rec
                    else stream_colors.get(kind, "#334155")
                )
                series_label = (
                    _series_label(request, index, stream=kind) if row_i == 0 else None
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
            if mode == "continuous" and stim_xs:
                mark_stim_times(
                    ax, stim_xs, appearance=draw_app, alpha=0.40, linewidth=0.55
                )
            elif mode != "continuous" and request.legend.show_reference_markers:
                _draw_onset_offset_lines(
                    ax,
                    end_markers=request.end_markers(),
                    label_in_legend=False,
                    appearance=draw_app,
                )
            _apply_montage_row_ylim(ax, kind, request.settings)
            if stream_x is not None:
                ax.set_xlim(stream_x[0], stream_x[1])
            elif (
                share_x
                and data_x_min is not None
                and data_x_max is not None
                and data_x_max > data_x_min
            ):
                ax.set_xlim(data_x_min, data_x_max)
        else:
            channel_drawn = _montage_draw_extra_row(
                ax,
                request,
                ch=ch,
                kind=kind,
                multi_rec=multi_rec,
                line_w=line_w,
                row_i=row_i,
                section_x=section_x,
            )
            drawn += channel_drawn
            _apply_montage_row_ylim(ax, kind, request.settings)
            if kind == "rms" and section_x is not None:
                ax.set_xlim(section_x[0], section_x[1])
            elif kind in {"psth", "raster"} and section_x is not None:
                ax.set_xlim(section_x[0], section_x[1])
            elif kind == "overlay":
                ax.set_xlabel("")  # ms — label global en bas si dernière ligne
            if kind in {"rms", "psth", "raster"} and request.legend.show_reference_markers:
                _draw_onset_offset_lines(
                    ax,
                    end_markers=request.end_markers(),
                    label_in_legend=False,
                    appearance=draw_app,
                )

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
        if ch == selected:
            ax.set_facecolor(CHANNEL_HIGHLIGHT_FACE)
        if row_i == 0 and multi_rec:
            _apply_legend(ax, request.legend, label_overrides=_legend_overrides(request))
        if row_i < rows - 1:
            ax.set_xlabel("")
        elif kind == "overlay":
            ax.set_xlabel("Time relative to spike (ms)")
        elif kind == "isi":
            ax.set_xlabel("ISI (ms)")
        elif kind == "trial_rate":
            ax.set_xlabel("Trial")
        elif kind in _MONTAGE_STREAM_KINDS and mode == "continuous":
            ax.set_xlabel(
                "Time relative to trigger (s)" if sync_mode == "trigger" else "Time (s)"
            )
        else:
            ax.set_xlabel("Time relative to stimulation (s)")

    if share_x:
        if stream_x is not None:
            axes[-1].set_xlim(stream_x[0], stream_x[1])
        elif data_x_min is not None and data_x_max is not None and data_x_max > data_x_min:
            axes[-1].set_xlim(data_x_min, data_x_max)

    _apply_montage_scale_bars(figure, request)
    figure._erg_montage_row_channels = [ch for ch, _kind, _label in row_specs]
    figure._erg_montage_row_streams = [
        kind if kind in _MONTAGE_STREAM_KINDS else "raw" for _ch, kind, _label in row_specs
    ]
    figure._erg_montage_row_kinds = [kind for _ch, kind, _label in row_specs]
    figure._erg_montage_review_mode = mode
    figure._erg_montage_state = {
        "kind": "continuous_review",
        "base_geom": _montage_continuous_base_geom(request),
        "row_keys": [(ch, kind) for ch, kind, _label in row_specs],
        "curves": curve_cache,
        "kinds": list(kinds),
        "mode": mode,
    }
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
    "analysis_raster_channel": _render_channel_raster_panel,
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
    """Identity of the data geometry — style-only changes must not alter this.

    Exception : ``show_scale_bars`` force un rebuild (masque/restaure ticks + labels).
    """
    # Montage continu : traces memmap — ignorer data_generation / sélection canal
    # (sinon chaque prefetch / clic reconstruit N axes).
    if request.panel == "montage_continuous_raw":
        review_mode = _montage_review_mode(request)
        kinds = tuple(_montage_review_kinds(request))
        needs_derived = review_mode != "continuous" or any(
            kind not in _MONTAGE_STREAM_KINDS for kind in kinds
        )
        # Memmap-only : ignorer data_generation. Moyennes / RMS / spikes : oui.
        if needs_derived:
            rec_fp = tuple(
                (
                    str(rec.meta.source_path),
                    int(rec.data_generation),
                    bool(rec.has_streams),
                    int(getattr(rec, "n_channels", 0) or 0),
                )
                for rec in request.recordings
            )
        else:
            rec_fp = tuple(
                (
                    str(rec.meta.source_path),
                    bool(rec.has_streams),
                    int(getattr(rec, "n_channels", 0) or 0),
                )
                for rec in request.recordings
            )
        x_lim = request.settings.x_limits.as_tuple()
        return (
            request.panel,
            review_mode,
            int(request.settings.analysis.stim_index),
            kinds,
            rec_fp,
            int(request.style.max_points_per_curve),
            tuple(request.settings.resolved_continuous_streams()),
            tuple(str(name) for name in request.settings.hidden_channels),
            int(request.settings.montage_row_min_height_px),
            int(getattr(request.settings, "montage_review_channels", 10) or 10),
            int(getattr(request.settings, "montage_review_page", 0) or 0),
            x_lim,
            str(request.settings.time_sync),
            bool(request.settings.continuous_mark_stims),
            bool(request.legend.show_reference_markers),
            bool(getattr(request.style, "show_scale_bars", False)),
            float(request.settings.psth_bin_window_s),
            int(request.settings.sampling_percent),
            str(request.settings.texts.title or "").strip(),
            str(request.settings.texts.xlabel or "").strip(),
            str(request.settings.texts.ylabel or "").strip(),
            tuple(str(label) for label in request.settings.texts.legend_labels),
            bool(request.settings.texts.show_series_suffix),
        )

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
        str(getattr(request.placement, "stream", "") or "")
        if request.panel == "full_recording"
        else "",
        str(request.settings.time_sync),
        str(request.settings.analysis.mode),
        int(request.settings.analysis.trigger_index() or -1),
        bool(request.legend.show_reference_markers),
        bool(getattr(request.style, "show_scale_bars", False)),
        # Geometry that affects histograms / rasters / overlays / downsampling.
        float(request.settings.psth_bin_window_s),
        int(request.settings.sampling_percent),
        float(request.settings.spike_overlay_pre_ms),
        float(request.settings.spike_overlay_post_ms),
        # Textes manuels : rebuild pour restaurer les libellés auto si vidés.
        str(request.settings.texts.title or "").strip(),
        str(request.settings.texts.xlabel or "").strip(),
        str(request.settings.texts.ylabel or "").strip(),
        tuple(str(label) for label in request.settings.texts.legend_labels),
        bool(request.settings.texts.show_series_suffix),
    )


def _refresh_panel_style(figure: Any, request: RenderRequest) -> str:
    """Update colors / fonts / ylims / surbrillance on an existing figure (no rebuild)."""
    status = str(getattr(figure, "_erg_status", "ok") or "ok")
    colors = list(request.colors)
    lw = float(request.style.line_width)
    axes = list(getattr(figure, "axes", []) or [])
    row_channels = getattr(figure, "_erg_montage_row_channels", None)
    selected = int(request.channel_index)
    for row, ax in enumerate(axes):
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
        multi_axis = request.panel in _MONTAGE_SPECS or request.panel in {
            "montage_continuous_raw",
            "full_recording",
        }
        # Style sans barres ici : ylim ensuite, puis barres (longueurs à jour).
        _apply_axis_style(ax, request.style, keep_ylabel=multi_axis)
        row_streams = getattr(figure, "_erg_montage_row_streams", None)
        row_kinds = getattr(figure, "_erg_montage_row_kinds", None)
        if request.panel in {"analysis_rms", "rms", "first_rms", "second_rms", "summary_rms"}:
            _apply_ylim(ax, request.settings.rms_ylim)
        elif (
            request.panel == "montage_continuous_raw"
            and isinstance(row_kinds, (list, tuple))
            and row < len(row_kinds)
        ):
            _apply_montage_row_ylim(ax, str(row_kinds[row]), request.settings)
        elif isinstance(row_streams, (list, tuple)) and row < len(row_streams):
            _apply_stream_ylim(ax, request.settings, str(row_streams[row]))
        else:
            stream = _stream_for_panel(request.panel)
            if request.panel == "full_recording":
                placed = str(getattr(request.placement, "stream", "") or "").strip()
                if placed in {"raw", "hp", "lp"}:
                    stream = placed
            if stream is not None:
                _apply_stream_ylim(ax, request.settings, stream)
        if request.panel == "montage_continuous_raw":
            if isinstance(row_channels, (list, tuple)) and row < len(row_channels):
                ax.set_facecolor(
                    CHANNEL_HIGHLIGHT_FACE
                    if int(row_channels[row]) == selected
                    else "#ffffff"
                )
        if request.panel not in {"mea_layout", "summary_rms_table"} and ax is axes[0]:
            _apply_legend(ax, request.legend, label_overrides=_legend_overrides(request))
        if (
            not multi_axis
            and bool(getattr(request.style, "show_scale_bars", False))
            and request.panel not in {"mea_layout", "summary_rms_table"}
        ):
            _apply_scale_bars(ax, request.style, panel=request.panel)
    if request.panel in _MONTAGE_SPECS or request.panel in {
        "montage_continuous_raw",
        "full_recording",
    }:
        _apply_montage_scale_bars(figure, request)
    elif request.panel not in {"mea_layout", "summary_rms_table"}:
        _apply_uniform_axes_box(
            figure,
            show_scale_bars=bool(getattr(request.style, "show_scale_bars", False)),
            montage=False,
            legend=request.legend,
        )
    _apply_text_overrides(figure, request)
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
        else:
            # Inclut full_recording : échelle Y / style sans rebuild.
            status = _refresh_panel_style(figure, request)
            figure._erg_status = status
            return status

    # Same montage layout, new channel data → set_data without clear/subplots.
    if has_axes and panel in _MONTAGE_SPECS:
        updated = _try_update_triggered_montage(figure, request)
        if updated is not None:
            _apply_text_overrides(figure, request)
            figure._erg_structure_key = fingerprint
            figure._erg_montage_rows = len(figure.axes)
            figure._erg_status = updated
            return updated

    # Masquer des canaux : retirer les axes sans relire les traces restantes.
    if has_axes and panel == "montage_continuous_raw":
        shrunk = _try_shrink_continuous_montage(figure, request)
        if shrunk is not None:
            _apply_text_overrides(figure, request)
            figure._erg_structure_key = fingerprint
            figure._erg_montage_rows = len(getattr(figure, "axes", []) or [])
            figure._erg_status = shrunk
            return shrunk

    # Préserver le cache courbes avant clear (ré-affichage de canaux).
    prior_state = getattr(figure, "_erg_montage_state", None)
    curve_cache = None
    if (
        panel == "montage_continuous_raw"
        and isinstance(prior_state, dict)
        and prior_state.get("kind") == "continuous_review"
        and prior_state.get("base_geom") == _montage_continuous_base_geom(request)
    ):
        curves = prior_state.get("curves")
        if isinstance(curves, dict):
            curve_cache = curves

    figure.clear()
    figure._erg_structure_key = None
    figure._erg_montage_rows = None
    figure._erg_montage_state = None
    figure._erg_montage_row_channels = None
    figure._erg_montage_row_streams = None
    figure._erg_montage_curve_cache = curve_cache
    if not request.recordings:
        _unavailable(figure.add_subplot(111), "Load a recording to display this panel.")
        return "unavailable"

    if panel == "full_recording":
        status = _render_full_recording(figure, request)
        _apply_text_overrides(figure, request)
        figure._erg_structure_key = fingerprint
        figure._erg_status = status
        return status

    if panel in _MONTAGE_SPECS or panel == "montage_continuous_raw":
        status = _render_montage(figure, request)
        _apply_text_overrides(figure, request)
        figure._erg_structure_key = fingerprint
        figure._erg_montage_rows = len(getattr(figure, "axes", []) or [])
        figure._erg_status = status
        return status

    _disable_layout_engine(figure)
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
        _apply_legend(ax, request.legend, label_overrides=_legend_overrides(request))
    _apply_axis_style(ax, request.style, panel=panel)
    if panel not in {"mea_layout", "summary_rms_table"}:
        _apply_uniform_axes_box(
            figure,
            show_scale_bars=bool(getattr(request.style, "show_scale_bars", False)),
            montage=False,
            legend=request.legend,
        )
    _apply_text_overrides(figure, request)
    figure._erg_structure_key = fingerprint
    figure._erg_status = status
    return status


def panels_for_scope(scope: PanelScope) -> tuple[PanelInfo, ...]:
    return tuple(info for info in PANEL_CATALOG if info.scope == scope)


def all_panel_keys() -> tuple[str, ...]:
    return tuple(info.key for info in PANEL_CATALOG)


def is_known_panel(key: str) -> bool:
    return key in PANEL_INFO_BY_KEY or key in EXTRA_CHANNEL_PANEL_FIELD_NAMES
