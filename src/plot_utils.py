"""Helpers de tracé partagés (PDF + GUI)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

# Marqueur sur les artistes de barre d’échelle (retrait / refresh).
_SCALE_BAR_ATTR = "_erg_scale_bar"


def mark_unavailable_axis(ax: Any, message: str, *, fontsize: float = 9.0) -> None:
    """Remplace un axe par un message centré (données indisponibles)."""
    from display_config import MUTED_AXIS_TEXT

    ax.clear()
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=fontsize,
        color=MUTED_AXIS_TEXT,
        wrap=True,
    )
    ax.set_axis_off()


def shorten_filename_for_windows(output_dir: Path, filename: str, max_total_len: int = 240) -> str:
    """Shorten filename if total path length may exceed Windows limits."""
    full_len = len(str(output_dir / filename))
    if full_len <= max_total_len:
        return filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".pdf"
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
    budget = max_total_len - len(str(output_dir)) - len(suffix) - len(digest) - 2
    budget = max(24, budget)
    short_stem = stem[:budget]
    return f"{short_stem}_{digest}{suffix}"


def downsample_points(x: np.ndarray, y: np.ndarray, sampling_percent: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically downsample points based on a percentage in [1, 100]."""
    if sampling_percent >= 100:
        return x, y
    pct = max(1, min(100, int(sampling_percent)))
    step = max(1, int(np.ceil(100.0 / float(pct))))
    return x[::step], y[::step]


def decimate_envelope(
    x: np.ndarray, y: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope decimation: keeps peaks while bounding the point count."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    n = int(x_arr.size)
    limit = max(16, int(max_points))
    if n <= limit:
        return x_arr, y_arr
    n_bins = max(8, limit // 2)
    edges = np.linspace(0, n, n_bins + 1).astype(np.int64)
    starts = edges[:-1]
    valid = starts < n
    starts = starts[valid]
    lows = np.minimum.reduceat(y_arr, starts)
    highs = np.maximum.reduceat(y_arr, starts)
    mids = np.add.reduceat(x_arr, starts) / np.diff(np.append(starts, n))
    out_x = np.repeat(mids, 2)
    out_y = np.empty(out_x.size, dtype=np.float64)
    out_y[0::2] = lows
    out_y[1::2] = highs
    return out_x, out_y


def nice_scale_length(span: float, *, target_frac: float = 0.18) -> float:
    """Longueur « ronde » (~1 / 2 / 5 × 10ⁿ) proche de ``target_frac`` du span visible."""
    span = abs(float(span))
    if not math.isfinite(span) or span <= 0:
        return 1.0
    target = span * float(target_frac)
    if target <= 0:
        return span * target_frac
    exponent = math.floor(math.log10(target))
    fraction = target / (10.0**exponent)
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if fraction <= candidate * 1.05:
            return candidate * (10.0**exponent)
    return 10.0 * (10.0**exponent)


def format_scale_label(value: float, unit: str) -> str:
    """Libellé compact pour une barre d’échelle (ex. ``100ms``, ``200µV``)."""
    unit = (unit or "").strip()
    magnitude = abs(float(value))
    if unit in {"s", "sec", "seconds"}:
        if magnitude >= 1.0:
            text = f"{magnitude:g}"
            return f"{text}s"
        ms = magnitude * 1000.0
        if ms >= 1.0:
            return f"{ms:g}ms"
        us = ms * 1000.0
        return f"{us:g}µs"
    if unit in {"ms"}:
        return f"{magnitude:g}ms"
    if not unit:
        return f"{magnitude:g}"
    # µV, Hz, Ω, …
    if magnitude >= 100 or abs(magnitude - round(magnitude)) < 1e-9:
        return f"{magnitude:g}{unit}"
    return f"{magnitude:g}{unit}"


def clear_floating_scale_bars(ax: Any) -> None:
    """Retirer les barres d’échelle précédemment ajoutées sur ``ax``."""
    for collection in (
        list(getattr(ax, "artists", []) or []),
        list(getattr(ax, "lines", []) or []),
        list(getattr(ax, "texts", []) or []),
        list(getattr(ax, "patches", []) or []),
    ):
        for artist in collection:
            if getattr(artist, _SCALE_BAR_ATTR, False):
                try:
                    artist.remove()
                except Exception:
                    pass


def hide_edge_scale(
    ax: Any,
    *,
    keep_ylabel: bool = False,
    keep_xlabel: bool = False,
) -> None:
    """Masquer graduations / libellés d’unités sur les bords (pas le cadre)."""
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )
    if not keep_xlabel:
        try:
            ax.set_xlabel("")
        except Exception:
            pass
    if not keep_ylabel:
        try:
            ax.set_ylabel("")
        except Exception:
            pass


def _nice_bar_length(span: float) -> float:
    """Longueur ronde bornée à ~45 % du span visible."""
    size = nice_scale_length(span)
    while size > span * 0.45 and size > 0:
        size *= 0.5
    return size


def resolve_scale_bar_length(
    span: float,
    *,
    manual: bool,
    value: float,
) -> float:
    """Longueur de barre : valeur manuelle si demandée, sinon auto."""
    if manual and math.isfinite(float(value)) and float(value) > 0:
        return float(value)
    return _nice_bar_length(span)


def _mark_scale_artist(artist: Any) -> Any:
    setattr(artist, _SCALE_BAR_ATTR, True)
    try:
        artist.set_clip_on(False)
    except Exception:
        pass
    return artist


def draw_floating_scale_bars(
    ax: Any,
    *,
    x_unit: str | None = "s",
    y_unit: str | None = "µV",
    x_size: float | None = None,
    y_size: float | None = None,
    fontsize: float = 8.0,
    color: str = "#111827",
    linewidth: float = 1.8,
    loc: str = "lower right",
    clear: bool = True,
    gutter: bool = False,
    below: bool = True,
) -> None:
    """Barres d’échelle flottantes (temps + amplitude).

    Longueurs en unités de données : une valeur fixe (manuelle ou auto) donne un
    bâton dont la taille pixel suit automatiquement le zoom.

    Position : ancrée sur le bord des axes (fraction d’axes), pas en coordonnées
    de données — ainsi les barres restent hors des courbes.
    ``gutter=True`` place l’échelle Y à droite de la zone de tracé.
    ``below=True`` (défaut) place l’échelle X sous le graphique.
    """
    if clear:
        clear_floating_scale_bars(ax)

    show_x = x_unit is not None
    show_y = y_unit is not None
    if not show_x and not show_y:
        return

    try:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    except Exception:
        return
    x_span = abs(float(xlim[1]) - float(xlim[0]))
    y_span = abs(float(ylim[1]) - float(ylim[0]))
    if (show_x and x_span <= 0) or (show_y and y_span <= 0):
        return

    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    x_len = (
        float(x_size)
        if (show_x and x_size is not None and float(x_size) > 0)
        else (_nice_bar_length(x_span) if show_x else 0.0)
    )
    y_len = (
        float(y_size)
        if (show_y and y_size is not None and float(y_size) > 0)
        else (_nice_bar_length(y_span) if show_y else 0.0)
    )

    # Fraction d’axes : à droite des courbes (dans la marge si gutter).
    x_axes = 1.02 if gutter else 0.985
    line_kw = {
        "color": color,
        "linewidth": linewidth,
        "solid_capstyle": "butt",
        "solid_joinstyle": "miter",
        "clip_on": False,
        "zorder": 20,
    }

    if show_y:
        # X fixe (axes) · Y en données → bâton vertical, longueur = y_len.
        trans_y = blended_transform_factory(ax.transAxes, ax.transData)
        y_mid = 0.5 * (float(ylim[0]) + float(ylim[1]))
        y0 = y_mid - 0.5 * y_len
        y1 = y0 + y_len
        vline = Line2D([x_axes, x_axes], [y0, y1], transform=trans_y, **line_kw)
        _mark_scale_artist(vline)
        ax.add_line(vline)
        label = format_scale_label(y_len, str(y_unit))
        text = ax.text(
            x_axes + (0.012 if gutter else 0.008),
            y_mid,
            label,
            transform=trans_y,
            color=color,
            fontsize=fontsize,
            ha="left",
            va="center",
            clip_on=False,
            zorder=20,
        )
        _mark_scale_artist(text)

    if show_x:
        # X en données · Y = bord bas des axes + décalage fixe (points).
        fig = ax.figure
        trans_base = blended_transform_factory(ax.transData, ax.transAxes)
        if below:
            # Sous le graph, hors zone de tracé (décalage fixe en points).
            trans_line = trans_base + ScaledTranslation(
                0, -14 / 72, fig.dpi_scale_trans
            )
            trans_label = trans_base + ScaledTranslation(
                0, -28 / 72, fig.dpi_scale_trans
            )
            y_ref = 0.0
        else:
            trans_line = trans_base
            trans_label = trans_base
            y_ref = 0.08
        x_right = float(xlim[1]) - 0.02 * x_span
        x_left = x_right - x_len
        if x_left < float(xlim[0]):
            x_left = float(xlim[0]) + 0.02 * x_span
            x_right = x_left + x_len
        hline = Line2D(
            [x_left, x_right], [y_ref, y_ref], transform=trans_line, **line_kw
        )
        _mark_scale_artist(hline)
        ax.add_line(hline)
        label = format_scale_label(x_len, str(x_unit))
        text = ax.text(
            0.5 * (x_left + x_right),
            y_ref if below else (y_ref - 0.04),
            label,
            transform=trans_label,
            color=color,
            fontsize=fontsize,
            ha="center",
            va="top",
            clip_on=False,
            zorder=20,
        )
        _mark_scale_artist(text)

    # ``loc`` conservé pour compatibilité d’appel ; placement géré ci-dessus.
    _ = loc
