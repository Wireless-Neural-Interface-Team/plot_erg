"""Helpers de tracé partagés (PDF + GUI)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

# Marqueur sur les artistes de barre d’échelle (retrait / refresh).
_SCALE_BAR_ATTR = "_erg_scale_bar"


def preview_title_row_layout(
    text_height_px: float,
    gap_px: int,
    *,
    title_dx_px: float = 0.0,
    title_dy_px: float = 0.0,
) -> tuple[int, int]:
    """Hauteur de la ligne de titre et espace sous cette ligne (aperçu canal).

    Pyqtgraph fige le titre dans une bande de 30 px, texte centré. ``gap_px``
    vaut 0 pour garder cet écart. Une valeur positive ajoute des pixels sous
    la bande. Une valeur négative réduit l’espace sous le texte, pixel pour
    pixel (la bande rétrécit de deux pixels par unité, car le texte est centré).

    ``title_dy_px`` négatif (titre plus bas) augmente l’écart sous la bande
    pour éviter le chevauchement avec la courbe. Le pad pour un ``dy`` positif
    (titre plus haut) et pour ``dx`` est géré via les marges du layout /
    de la figure, pas ici.
    """
    del title_dx_px  # pad horizontal hors bandeau (marges plot / figure)
    delta = int(gap_px)
    if delta >= 0:
        row_h, spacing = 30, delta
    else:
        text_px = max(1, int(math.ceil(float(text_height_px))))
        row_h, spacing = max(text_px, 30 + 2 * delta), 0
    dy = int(round(float(title_dy_px or 0.0)))
    if dy < 0:
        # Titre descendu vers la courbe : pousser le graphe pour libérer la place.
        spacing += -dy
    return row_h, spacing


def title_offset_margin_fractions(
    dx_px: float,
    dy_px: float,
    *,
    fig_width_in: float,
    fig_height_in: float,
    dpi: float = 100.0,
) -> tuple[float, float, float]:
    """Réserve de marge figure pour un titre décalé.

    Retourne ``(top_shrink, left_grow, right_shrink)`` en fraction de figure.
    Un ``dy`` positif (titre plus haut) réduit ``top`` ; un ``dx`` positif
    réduit ``right`` ; un ``dx`` négatif augmente ``left``.
    """
    dpi = float(dpi) if float(dpi) > 0 else 100.0
    fig_w_px = max(1.0, float(fig_width_in) * dpi)
    fig_h_px = max(1.0, float(fig_height_in) * dpi)
    top_px = max(0.0, float(dy_px or 0.0))
    left_px = max(0.0, -float(dx_px or 0.0))
    right_px = max(0.0, float(dx_px or 0.0))
    if top_px > 0.0:
        top_px += 4.0
    if left_px > 0.0:
        left_px += 4.0
    if right_px > 0.0:
        right_px += 4.0
    return (
        min(0.28, top_px / fig_h_px),
        min(0.22, left_px / fig_w_px),
        min(0.22, right_px / fig_w_px),
    )


def apply_title_pixel_offset(
    text: Any,
    figure: Any,
    *,
    dx_px: float = 0.0,
    dy_px: float = 0.0,
) -> None:
    """Décaler un titre matplotlib (``Axes.title`` / ``suptitle``) en pixels.

    ``dx_px`` positif vers la droite, ``dy_px`` positif vers le haut.
    Réapplique depuis la transform de base du texte (pas d’empilement).
    """
    if text is None or figure is None:
        return
    dx = float(dx_px or 0.0)
    dy = float(dy_px or 0.0)
    base = getattr(text, "_erg_title_base_transform", None)
    if base is None:
        try:
            base = text.get_transform()
        except Exception:
            return
        try:
            text._erg_title_base_transform = base  # noqa: SLF001
        except Exception:
            pass
    if not dx and not dy:
        try:
            text.set_transform(base)
        except Exception:
            pass
        return
    from matplotlib.transforms import ScaledTranslation

    dpi = float(getattr(figure, "dpi", 100.0) or 100.0)
    try:
        text.set_transform(
            base
            + ScaledTranslation(dx / dpi, dy / dpi, figure.dpi_scale_trans)
        )
    except Exception:
        pass


def place_montage_row_label(
    ax: Any,
    text: str,
    *,
    fontsize: float,
    labelpad: float = 8.0,
    dx_px: float = 0.0,
    dy_px: float = 0.0,
) -> None:
    """Titre de ligne montage, à gauche de la courbe.

    ``dx_px`` positif rapproche le titre de la courbe.
    ``dy_px`` positif le déplace vers le haut.
    """
    ax.set_ylabel(
        text,
        rotation=0,
        ha="right",
        va="center",
        fontsize=fontsize,
        labelpad=labelpad,
    )
    ax.tick_params(axis="y", which="both", labelleft=False)
    x, y = -0.012, 0.5
    fig = getattr(ax, "figure", None)
    if fig is not None and (dx_px or dy_px):
        from matplotlib.transforms import ScaledTranslation

        dpi = float(getattr(fig, "dpi", 100.0) or 100.0)
        trans = ax.transAxes + ScaledTranslation(
            float(dx_px) / dpi,
            float(dy_px) / dpi,
            fig.dpi_scale_trans,
        )
        ax.yaxis.set_label_coords(x, y, transform=trans)
    else:
        ax.yaxis.set_label_coords(x, y)


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
    """Sous-échantillon stride (scatter / points indépendants).

    Pour une courbe tension continue, préférer ``downsample_curve``
    (enveloppe min/max) afin de ne pas perdre les pics.
    """
    if sampling_percent >= 100:
        return x, y
    pct = max(1, min(100, int(sampling_percent)))
    step = max(1, int(np.ceil(100.0 / float(pct))))
    return x[::step], y[::step]


def downsample_curve(
    x: np.ndarray, y: np.ndarray, sampling_percent: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sous-échantillon d’une courbe continue en préservant min/max par bin."""
    if sampling_percent >= 100:
        return x, y
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    n = int(x_arr.size)
    if n <= 0:
        return x_arr, y_arr
    pct = max(1, min(100, int(sampling_percent)))
    max_points = max(16, int(np.ceil(n * pct / 100.0)))
    return decimate_envelope(x_arr, y_arr, max_points)


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


def envelope_minmax_row(
    row: Any,
    n_samples: int,
    max_points: int,
    *,
    t_origin: float = 0.0,
    fs: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope of ``row[0:n_samples]`` without a full-row RAM copy.

    Bin-sized reads keep memmap / disk-backed channels responsive on the UI
    thread even for multi-million-sample recordings (WIDE / HIGH / LOW).
    Contiguous in-RAM float buffers still use a single ``reduceat`` pass.
    """
    empty = np.empty(0, dtype=np.float64)
    n_samples = int(n_samples)
    limit = max(16, int(max_points))
    fs = float(fs)
    if n_samples <= 0 or fs <= 0:
        return empty, empty
    if n_samples <= limit:
        data = np.asarray(row[:n_samples], dtype=np.float64)
        t = float(t_origin) + np.arange(data.size, dtype=np.float64) / fs
        return t, data

    n_bins = max(1, limit // 2)
    edges = np.linspace(0, n_samples, n_bins + 1)
    edges = np.rint(edges).astype(np.int64)
    edges[0] = 0
    edges[-1] = n_samples
    edges = np.unique(edges)
    starts = edges[:-1]
    widths = np.diff(edges)
    n_out = int(starts.size)

    def _is_in_ram_float_row(buffer: Any) -> bool:
        if not isinstance(buffer, np.ndarray) or int(getattr(buffer, "ndim", 1)) != 1:
            return False
        if isinstance(buffer, np.memmap):
            return False
        base = getattr(buffer, "base", None)
        while base is not None:
            if isinstance(base, np.memmap):
                return False
            base = getattr(base, "base", None)
        return bool(buffer.flags.c_contiguous) and str(buffer.dtype.kind) == "f"

    if _is_in_ram_float_row(row):
        values = row[:n_samples]
        if values.dtype not in (np.float32, np.float64):
            values = np.asarray(values, dtype=np.float32)
        lows = np.minimum.reduceat(values, starts)
        highs = np.maximum.reduceat(values, starts)
    else:
        # Memmap / disk view: batched reduceat — avoids a full-row RAM copy
        # and the cost of one Python min/max per bin (~50k bins by default).
        lows = np.empty(n_out, dtype=np.float64)
        highs = np.empty(n_out, dtype=np.float64)
        bin_batch = 256
        for batch_start in range(0, n_out, bin_batch):
            batch_end = min(n_out, batch_start + bin_batch)
            s0 = int(starts[batch_start])
            s1 = int(edges[batch_end])
            block = np.asarray(row[s0:s1], dtype=np.float32)
            local = starts[batch_start:batch_end] - s0
            lows[batch_start:batch_end] = np.minimum.reduceat(block, local)
            highs[batch_start:batch_end] = np.maximum.reduceat(block, local)

    centers = starts.astype(np.float64) + (widths.astype(np.float64) - 1.0) / 2.0
    mids = float(t_origin) + centers / fs
    out_t = np.repeat(mids, 2)
    out_y = np.empty(mids.size * 2, dtype=np.float64)
    out_y[0::2] = lows
    out_y[1::2] = highs
    return out_t, out_y


def nice_scale_length(span: float, *, target_frac: float = 0.18) -> float:
    """Longueur « ronde » (~1 / 2 / 5 × 10ⁿ) proche de ``target_frac`` du span visible."""
    from scale_bar_model import nice_scale_length as _nice

    return _nice(span, target_frac=target_frac)


def format_scale_label(value: float, unit: str) -> str:
    """Libellé compact pour une barre d’échelle (ex. ``100ms``, ``200µV``)."""
    from scale_bar_model import format_scale_label as _fmt

    return _fmt(value, unit)


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
    keep_ylabel: bool = True,
    keep_xlabel: bool = True,
) -> None:
    """Masquer les graduations numériques sur les bords (titres d’axes conservés)."""
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


def resolve_scale_bar_length(
    span: float,
    *,
    manual: bool,
    value: float,
    clamp_to_span: bool = False,
) -> float:
    """Longueur de barre cible (unités de données) : manuelle ou auto.

    Shim vers :func:`scale_bar_model.resolve_axis_bar`.
    """
    from scale_bar_model import resolve_axis_bar

    span_abs = abs(float(span))
    if manual and math.isfinite(float(value)) and float(value) > 0:
        if not clamp_to_span:
            return float(value)
        if span_abs > 0 and float(value) <= span_abs * 0.9:
            return float(value)
    bar = resolve_axis_bar(span=span_abs, axis_px=100.0, value=None, length_px=None)
    return float(bar.value)


def scale_bar_pixel_length(
    *,
    data_value: float,
    span: float,
    axis_px: float,
    length_px: float | None = None,
    value_manual: bool = False,
) -> float:
    """Longueur du bâton en pixels (shim → ``resolve_axis_bar``)."""
    from scale_bar_model import resolve_axis_bar

    length_ok = length_px is not None and float(length_px) > 0
    if length_ok and not bool(value_manual):
        return float(
            resolve_axis_bar(
                span=span, axis_px=axis_px, value=None, length_px=float(length_px)
            ).bar_px
        )
    # Valeur manuelle, les deux, ou cible auto déjà résolue dans data_value.
    val = float(data_value) if float(data_value) > 0 else None
    return float(
        resolve_axis_bar(
            span=span,
            axis_px=axis_px,
            value=val,
            length_px=float(length_px) if (length_ok and bool(value_manual)) else None,
        ).bar_px
    )


def scale_bar_true_value(
    bar_px: float,
    *,
    span: float,
    axis_px: float,
) -> float:
    """Valeur réelle (unités de données) : ``bar_px / axis_px × span``."""
    try:
        bar = float(bar_px)
        data_span = abs(float(span))
        axis = float(axis_px)
    except (TypeError, ValueError):
        return 0.0
    if not (bar > 0.0 and data_span > 0.0 and axis > 0.0):
        return 0.0
    if bar != bar or data_span != data_span or axis != axis:
        return 0.0
    return (bar / axis) * data_span


def scale_bar_label_value(
    *,
    data_value: float,
    span: float,
    axis_px: float,
    bar_px: float,
    value_manual: bool,
    length_manual: bool,
) -> float:
    """Valeur affichée (shim → ``resolve_axis_bar``)."""
    from scale_bar_model import resolve_axis_bar

    if bool(length_manual) and not bool(value_manual):
        # Géométrie du bâton réellement dessiné.
        return float(
            resolve_axis_bar(
                span=span, axis_px=axis_px, value=None, length_px=float(bar_px)
            ).label
        )
    if bool(value_manual) and float(data_value) > 0:
        return float(
            resolve_axis_bar(
                span=span,
                axis_px=axis_px,
                value=float(data_value),
                length_px=float(bar_px) if bool(length_manual) else None,
            ).label
        )
    try:
        target = float(data_value)
    except (TypeError, ValueError):
        target = 0.0
    return target if target > 0.0 and target == target else 0.0


def _mark_scale_artist(artist: Any) -> Any:
    setattr(artist, _SCALE_BAR_ATTR, True)
    try:
        artist.set_clip_on(False)
    except Exception:
        pass
    return artist


def _apply_scale_label_face(text: Any, face: Any) -> None:
    """Gras / italique / souligné des libellés d'échelle (même rôle que les ticks)."""
    if text is None or face is None:
        return
    try:
        from panel_registry import apply_text_face

        apply_text_face(text, face)
    except Exception:
        try:
            text.set_fontweight("bold" if getattr(face, "bold", False) else "normal")
            text.set_fontstyle("italic" if getattr(face, "italic", False) else "normal")
        except Exception:
            pass


def draw_floating_scale_bars(
    ax: Any,
    *,
    x_unit: str | None = "s",
    y_unit: str | None = "µV",
    x_size: float | None = None,
    y_size: float | None = None,
    fontsize: float = 8.0,
    color: str = "#111827",
    linewidth: float = 1.4,
    loc: str = "lower right",
    clear: bool = True,
    gutter: bool = False,
    below: bool = True,
    face: Any = None,
    dx_px: float = 0.0,
    dy_px: float = 0.0,
    x_dx_px: float | None = None,
    x_dy_px: float | None = None,
    y_length_px: float | None = None,
    x_length_px: float | None = None,
    y_value_manual: bool = False,
    x_value_manual: bool = False,
) -> None:
    """Barres d’échelle flottantes via :func:`scale_bar_model.resolve_axis_bar`.

    Longueur en données = ``bar_px / axis_px × span`` (alignée pixel ↔ données).
    """
    from scale_bar_model import format_scale_label as _fmt
    from scale_bar_model import resolve_axis_bar

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
    if show_x and x_span <= 0:
        show_x = False
    if show_y and y_span <= 0:
        show_y = False
    if not show_x and not show_y:
        return

    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    fig = getattr(ax, "figure", None)

    def _px(value: Any, *, fallback: float = 0.0) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if out != out:
            return float(fallback)
        return out

    amp_dx = _px(dx_px)
    amp_dy = _px(dy_px)
    time_dx = amp_dx if x_dx_px is None else _px(x_dx_px)
    time_dy = amp_dy if x_dy_px is None else _px(x_dy_px)

    def _with_px_shift(
        transform: Any,
        *,
        shift_dx: float,
        shift_dy: float,
        extra_dy_in: float = 0.0,
    ) -> Any:
        if fig is None or (shift_dx == 0.0 and shift_dy == 0.0 and extra_dy_in == 0.0):
            return transform
        dpi = float(getattr(fig, "dpi", 100.0) or 100.0)
        if dpi <= 0.0:
            dpi = 100.0
        return transform + ScaledTranslation(
            shift_dx / dpi,
            extra_dy_in + shift_dy / dpi,
            fig.dpi_scale_trans,
        )

    def _axes_size_px() -> tuple[float, float]:
        try:
            bbox = ax.get_window_extent()
            return max(1.0, float(bbox.width)), max(1.0, float(bbox.height))
        except Exception:
            return 400.0, 300.0

    ax_w_px, ax_h_px = _axes_size_px()

    def _axis_from_caller(
        *,
        span: float,
        axis_px: float,
        size: float | None,
        length_px: float | None,
        value_manual: bool,
    ):
        """Mapper les flags appelant → ``resolve_axis_bar``.

        - taille seule : value=None, length_px=L
        - valeur (+ taille) : value=size, length_px=L|None
        - auto : value=size (nice déjà résolu) ou None
        """
        length = _px(length_px) if length_px is not None and _px(length_px) > 0 else None
        size_ok = size is not None and float(size) > 0
        if length is not None and not bool(value_manual):
            return resolve_axis_bar(
                span=span, axis_px=axis_px, value=None, length_px=length
            )
        if size_ok:
            return resolve_axis_bar(
                span=span,
                axis_px=axis_px,
                value=float(size),
                length_px=length if bool(value_manual) else None,
            )
        return resolve_axis_bar(span=span, axis_px=axis_px)

    y_bar = (
        _axis_from_caller(
            span=y_span,
            axis_px=ax_h_px,
            size=y_size,
            length_px=y_length_px,
            value_manual=bool(y_value_manual),
        )
        if show_y
        else None
    )
    x_bar = (
        _axis_from_caller(
            span=x_span,
            axis_px=ax_w_px,
            size=x_size,
            length_px=x_length_px,
            value_manual=bool(x_value_manual),
        )
        if show_x
        else None
    )

    x_axes = 1.02 if gutter else 0.985
    line_kw = {
        "color": color,
        "linewidth": linewidth,
        "solid_capstyle": "butt",
        "solid_joinstyle": "miter",
        "clip_on": False,
        "zorder": 20,
    }

    if show_y and y_bar is not None and y_bar.data_length > 0:
        trans_y = _with_px_shift(
            blended_transform_factory(ax.transAxes, ax.transData),
            shift_dx=amp_dx,
            shift_dy=amp_dy,
        )
        y_mid = 0.5 * (float(ylim[0]) + float(ylim[1]))
        half = 0.5 * float(y_bar.data_length)
        vline = Line2D(
            [x_axes, x_axes],
            [y_mid - half, y_mid + half],
            transform=trans_y,
            **line_kw,
        )
        _mark_scale_artist(vline)
        ax.add_line(vline)
        text = ax.text(
            x_axes + (0.012 if gutter else 0.008),
            y_mid,
            _fmt(y_bar.label, str(y_unit)),
            transform=trans_y,
            color=color,
            fontsize=fontsize,
            ha="left",
            va="center",
            clip_on=False,
            zorder=20,
        )
        _mark_scale_artist(text)
        _apply_scale_label_face(text, face)

    if show_x and x_bar is not None and x_bar.data_length > 0:
        trans_base = blended_transform_factory(ax.transData, ax.transAxes)
        if below:
            trans_line = _with_px_shift(
                trans_base, shift_dx=time_dx, shift_dy=time_dy, extra_dy_in=-14 / 72
            )
            trans_label = _with_px_shift(
                trans_base, shift_dx=time_dx, shift_dy=time_dy, extra_dy_in=-28 / 72
            )
            y_ref = 0.0
        else:
            trans_line = _with_px_shift(
                trans_base, shift_dx=time_dx, shift_dy=time_dy
            )
            trans_label = _with_px_shift(
                trans_base, shift_dx=time_dx, shift_dy=time_dy
            )
            y_ref = 0.08
        x_draw = float(x_bar.data_length)
        x_right = float(xlim[1]) - 0.02 * x_span
        x_left = x_right - x_draw
        if x_left < float(xlim[0]):
            x_left = float(xlim[0]) + 0.02 * x_span
            x_right = x_left + x_draw
        hline = Line2D(
            [x_left, x_right], [y_ref, y_ref], transform=trans_line, **line_kw
        )
        _mark_scale_artist(hline)
        ax.add_line(hline)
        text = ax.text(
            0.5 * (x_left + x_right),
            y_ref if below else (y_ref - 0.04),
            _fmt(x_bar.label, str(x_unit)),
            transform=trans_label,
            color=color,
            fontsize=fontsize,
            ha="center",
            va="top",
            clip_on=False,
            zorder=20,
        )
        _mark_scale_artist(text)
        _apply_scale_label_face(text, face)

    _ = loc
