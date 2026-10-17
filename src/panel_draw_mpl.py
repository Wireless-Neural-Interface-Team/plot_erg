"""Matplotlib (Agg) drawer for :class:`panel_prepare.PanelSpec`.

Used by the PDF / CLI path so screen and export share the same prepared data.
``montage_continuous_raw`` uses prepare (full Pipeline extras) like the screen.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from panel_prepare import (
    AnnotationSpec,
    CurveSeries,
    HistSeries,
    PanelSpec,
    ScatterSeries,
    _MONTAGE_TRIGGERED,
    montage_vline_span_keys,
    montage_vline_spans,
)
from plot_utils import place_montage_row_label

_CHANNEL_HIGHLIGHT = "#fff7ed"

# Panels drawn via prepare_panel → this module.
PREPARE_MPL_PANELS: frozenset[str] = frozenset(
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
        "analysis_raw",
        "analysis_hp",
        "analysis_lp",
        "analysis_rms",
        "rms",
        "first_rms",
        "second_rms",
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
        "full_recording",
        "montage_continuous_raw",
        "montage_second_to_third_lp",
        "mea_layout",
        "impedance",
        "summary_rms",
        "summary_rms_table",
        "summary_impedance",
        *(_MONTAGE_TRIGGERED.keys()),
    }
)


def uses_prepare_mpl(panel: str) -> bool:
    return str(panel or "") in PREPARE_MPL_PANELS


def _draw_table_spec(figure: Any, spec: PanelSpec, status: str) -> str:
    """Render ``layout_kind=table`` panels (summary RMS table, etc.)."""
    ax = figure.add_subplot(111)
    ax.set_axis_off()
    headers = [str(h) for h in (spec.table_headers or [])]
    rows = [[str(cell) for cell in row] for row in (spec.table_rows or [])]
    if not headers and not rows:
        message = spec.status_message or spec.title or status
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#64748b",
            wrap=True,
        )
        figure._erg_status = status
        return status
    col_labels = headers or None
    cell_text = rows or [[""] * max(1, len(headers))]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(max(6.0, float(spec.tick_font_size or 8.0)))
    table.scale(1.0, 1.35)
    highlight = int(getattr(spec, "table_highlight_row", -1) or -1)
    header_offset = 1 if col_labels else 0
    if 0 <= highlight < len(rows):
        for col in range(len(cell_text[highlight])):
            cell = table[(highlight + header_offset, col)]
            cell.set_facecolor(_CHANNEL_HIGHLIGHT)
    if spec.title:
        figure.suptitle(
            spec.title,
            fontsize=float(spec.title_font_size or 10.0),
            fontweight="bold" if spec.title_bold else "normal",
            fontstyle="italic" if spec.title_italic else "normal",
        )
    figure._erg_status = status
    return status


def _apply_spec_text_face(
    text: Any, spec: PanelSpec, *, role: str = "label"
) -> None:
    """Gras / italique / souligné depuis les flags PanelSpec (export MPL)."""
    if text is None:
        return
    if role == "title":
        bold, italic, underline = (
            bool(spec.title_bold),
            bool(spec.title_italic),
            bool(getattr(spec, "title_underline", False)),
        )
    elif role == "tick":
        bold, italic, underline = (
            bool(spec.tick_bold),
            bool(spec.tick_italic),
            bool(getattr(spec, "tick_underline", False)),
        )
    elif role == "legend":
        bold, italic, underline = (
            bool(spec.legend_bold),
            bool(spec.legend_italic),
            bool(getattr(spec, "legend_underline", False)),
        )
    else:
        bold, italic, underline = (
            bool(spec.label_bold),
            bool(spec.label_italic),
            bool(getattr(spec, "label_underline", False)),
        )
    try:
        from panel_registry import apply_text_face
        from view_config import TextFace

        apply_text_face(
            text, TextFace(bold=bold, italic=italic, underline=underline)
        )
    except Exception:
        try:
            text.set_fontweight("bold" if bold else "normal")
            text.set_fontstyle("italic" if italic else "normal")
        except Exception:
            pass


def draw_panel_spec(figure: Any, spec: PanelSpec, request: Any | None = None) -> str:
    """Clear ``figure`` and draw ``spec``. Returns status string."""
    figure.clear()
    figure._erg_from_prepare = True
    figure._erg_structure_key = None
    figure._erg_montage_rows = None
    figure._erg_montage_state = None
    figure._erg_montage_row_channels = None
    figure._erg_montage_row_streams = None
    figure._erg_montage_row_kinds = None

    status = str(spec.status or "ok")
    if spec.layout_kind == "placeholder" or (
        status in {"unavailable", "pending", "empty", "error"}
        and not spec.series
        and str(spec.layout_kind) != "table"
    ):
        ax = figure.add_subplot(111)
        message = spec.status_message or spec.title or status
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#64748b",
            wrap=True,
        )
        ax.set_axis_off()
        figure._erg_status = status
        return status

    if str(spec.layout_kind) == "table":
        return _draw_table_spec(figure, spec, status)

    is_montage = str(spec.layout_kind) == "montage"
    n_rows = max(1, int(spec.n_rows or 1))
    _disable_layout_engine(figure)
    if is_montage:
        _size_montage_figure(figure, n_rows, request)

    # sharex seulement si toutes les lignes partagent la même famille temporelle.
    stream_kinds = {"raw", "hp", "lp"}
    share_x = not (
        is_montage
        and any(str(k) not in stream_kinds for k in (spec.row_kinds or ()))
    )
    axes = figure.subplots(n_rows, 1, sharex=share_x, squeeze=False)
    axes_flat = [axes[row][0] for row in range(n_rows)]

    legend_labels: list[tuple[str, Any]] = []
    seen: set[str] = set()

    for item in spec.series:
        row = int(getattr(item, "row", 0) or 0)
        if row < 0 or row >= n_rows:
            row = 0
        ax = axes_flat[row]
        label = str(getattr(item, "label", "") or "").strip()
        show_label = bool(label and label != "_nolegend_" and label not in seen)
        if show_label:
            seen.add(label)

        if isinstance(item, CurveSeries):
            if item.x.size == 0 or item.y.size == 0:
                continue
            raw_alpha = getattr(item, "alpha", 1.0)
            alpha = 1.0 if raw_alpha is None else float(raw_alpha)
            alpha = max(0.0, min(1.0, alpha))
            (line,) = ax.plot(
                np.asarray(item.x, dtype=np.float64),
                np.asarray(item.y, dtype=np.float64),
                color=item.color,
                linewidth=max(0.5, float(item.linewidth)),
                alpha=alpha,
                zorder=3 if alpha >= 0.999 else 1,
                label=label if show_label else None,
            )
            if show_label:
                legend_labels.append((label, line))
        elif isinstance(item, ScatterSeries):
            if item.x.size == 0:
                continue
            xs = np.asarray(item.x, dtype=np.float64)
            ys = np.asarray(item.y, dtype=np.float64)
            if item.symbol == "|":
                if xs.size > 8000:
                    step = max(1, xs.size // 8000)
                    xs, ys = xs[::step], ys[::step]
                for x, y in zip(xs, ys):
                    ax.plot(
                        [x, x],
                        [y - 0.35, y + 0.35],
                        color=item.color,
                        linewidth=1.0,
                        solid_capstyle="round",
                    )
                if show_label:
                    legend_labels.append(
                        (
                            label,
                            ax.plot([], [], color=item.color, linewidth=1.0)[0],
                        )
                    )
            else:
                marker = {"o": "o", "t": "^", "s": "s", "d": "D", "+": "+"}.get(
                    item.symbol, "o"
                )
                artist = ax.scatter(
                    xs,
                    ys,
                    s=max(4.0, float(item.size) ** 2),
                    c=item.color,
                    marker=marker,
                    linewidths=0.0,
                    label=label if show_label else None,
                )
                if show_label:
                    legend_labels.append((label, artist))
        elif isinstance(item, HistSeries):
            edges = np.asarray(item.edges, dtype=np.float64)
            counts = np.asarray(item.counts, dtype=np.float64)
            if edges.size < 2 or counts.size == 0:
                continue
            artist = ax.stairs(
                counts,
                edges,
                fill=True,
                color=item.color,
                alpha=0.55,
                label=label if show_label else None,
            )
            if show_label:
                legend_labels.append((label, artist))

    stim_spans = montage_vline_spans(spec) if is_montage else []
    stim_span_keys = montage_vline_span_keys(stim_spans)
    stim_lw = max(0.2, float(getattr(spec, "stim_linewidth", 1.0) or 1.0))
    for ann in spec.annotations:
        row = int(ann.row or 0)
        if row < 0 or row >= n_rows:
            row = 0
        if ann.kind == "vline" and (
            row,
            round(float(ann.x), 6),
            str(ann.color or ""),
        ) in stim_span_keys:
            continue
        artist = _draw_annotation(axes_flat[row], ann, stim_linewidth=stim_lw)
        ann_label = str(getattr(ann, "label", "") or "").strip()
        if artist is not None and ann_label and ann_label not in seen:
            seen.add(ann_label)
            legend_labels.append((ann_label, artist))

    label_fs = float(spec.label_font_size)
    tick_fs = float(spec.tick_font_size)
    if is_montage:
        label_fs = max(5.0, label_fs - 1.0)
        tick_fs = max(5.0, tick_fs - 1.5)
    inside = bool(spec.ticks_inside)
    show_borders = bool(spec.show_borders)

    for row, ax in enumerate(axes_flat):
        if spec.grid:
            ax.grid(True, alpha=float(spec.grid_alpha))
        else:
            ax.grid(False)
        if row < len(spec.row_highlight) and spec.row_highlight[row]:
            ax.set_facecolor(_CHANNEL_HIGHLIGHT)
        ylim = None
        if row < len(spec.row_ylims) and spec.row_ylims[row] is not None:
            ylim = spec.row_ylims[row]
        elif row == 0 and spec.ylim is not None and not is_montage:
            ylim = spec.ylim
        if bool(getattr(spec, "y_log", False)) and not is_montage:
            try:
                ax.set_yscale("log")
            except Exception:
                pass
        if ylim is not None:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        else:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            kind = (
                str(spec.row_kinds[row])
                if row < len(spec.row_kinds)
                else ""
            )
            if kind in {"psth", "trial_rate", "isi", "rms"}:
                y0, y1 = ax.get_ylim()
                if y0 < 0:
                    ax.set_ylim(0.0, max(y1, 1e-9))
        row_xlim = None
        if row < len(spec.row_xlims) and spec.row_xlims[row] is not None:
            row_xlim = spec.row_xlims[row]
        elif spec.xlim is not None and share_x:
            row_xlim = spec.xlim
        if row_xlim is not None:
            ax.set_xlim(float(row_xlim[0]), float(row_xlim[1]))
        elif not share_x:
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=False)
        if row == (n_rows - 1 if is_montage else 0) and getattr(spec, "x_ticks", None):
            ticks = list(spec.x_ticks or [])
            if ticks:
                try:
                    ax.set_xticks([float(x) for x, _label in ticks])
                    ax.set_xticklabels([str(label) for _x, label in ticks])
                except Exception:
                    pass
        label_text = ""
        if row < len(spec.row_labels) and spec.row_labels[row]:
            label_text = str(spec.row_labels[row])
        elif row == 0 and spec.ylabel:
            label_text = str(spec.ylabel)
        if is_montage and label_text:
            place_montage_row_label(
                ax,
                label_text,
                fontsize=label_fs,
                dx_px=float(getattr(spec, "row_label_dx_px", 0.0) or 0.0),
                dy_px=float(getattr(spec, "row_label_dy_px", 0.0) or 0.0),
            )
        elif label_text:
            ax.set_ylabel(label_text, fontsize=label_fs)
        if row == n_rows - 1 and spec.xlabel:
            ax.set_xlabel(spec.xlabel, fontsize=label_fs)
        else:
            ax.set_xlabel("")
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=tick_fs,
            labelbottom=(row == n_rows - 1) if is_montage else True,
            direction="in" if inside else "out",
            top=inside and show_borders,
            right=inside and show_borders,
        )
        for spine in ax.spines.values():
            spine.set_visible(show_borders)
            if is_montage and show_borders:
                spine.set_linewidth(0.8)
        _apply_spec_text_face(ax.xaxis.label, spec, role="label")
        _apply_spec_text_face(ax.yaxis.label, spec, role="label")
        for tick_label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            _apply_spec_text_face(tick_label, spec, role="tick")

    _reserve_space_above_for_stim_labels(figure, axes_flat, spec)
    _reserve_space_below_for_zone_labels(figure, axes_flat, spec)

    if spec.title:
        from plot_utils import apply_title_pixel_offset, title_offset_margin_fractions

        dx = float(getattr(spec, "title_dx_px", 0.0) or 0.0)
        dy = float(getattr(spec, "title_dy_px", 0.0) or 0.0)
        if dx or dy:
            try:
                fig_w = float(figure.get_figwidth())
                fig_h = float(figure.get_figheight())
                dpi = float(getattr(figure, "dpi", 100.0) or 100.0)
            except Exception:
                fig_w, fig_h, dpi = 6.0, 4.0, 100.0
            top_s, left_g, right_s = title_offset_margin_fractions(
                dx, dy, fig_width_in=fig_w, fig_height_in=fig_h, dpi=dpi
            )
            try:
                subs = figure.subplotpars
                figure.subplots_adjust(
                    left=min(subs.right - 0.35, float(subs.left) + left_g),
                    right=max(subs.left + 0.35, float(subs.right) - right_s),
                    top=max(float(subs.bottom) + 0.20, float(subs.top) - top_s),
                )
            except Exception:
                pass
        figure.suptitle(
            spec.title,
            fontsize=float(spec.title_font_size),
            fontweight="bold" if spec.title_bold else "normal",
            fontstyle="italic" if spec.title_italic else "normal",
        )
        sup = getattr(figure, "_suptitle", None)
        _apply_spec_text_face(sup, spec, role="title")
        apply_title_pixel_offset(sup, figure, dx_px=dx, dy_px=dy)

    if spec.show_legend and legend_labels:
        handles = [artist for _label, artist in legend_labels]
        labels = [label for label, _artist in legend_labels]
        leg = axes_flat[0 if is_montage else -1].legend(
            handles,
            labels,
            loc="upper right",
            fontsize=float(spec.legend_font_size),
            frameon=bool(spec.legend_frame),
            ncol=max(1, int(spec.legend_columns)),
        )
        if leg is not None:
            for text in leg.get_texts():
                _apply_spec_text_face(text, spec, role="legend")

    if stim_spans:
        _draw_montage_stim_spans(figure, axes_flat, stim_spans, linewidth=stim_lw)
    _attach_montage_metadata(figure, spec)
    figure._erg_status = status
    return status


def _draw_montage_stim_spans(
    figure: Any,
    axes: list[Any],
    spans: list[tuple[float, str, tuple[int, ...]]],
    *,
    linewidth: float = 1.0,
) -> None:
    """Un seul pointillé par stimulation, du premier canal concerné au dernier."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory

    for x, color, rows in spans:
        if not rows or rows[0] >= len(axes) or rows[-1] >= len(axes):
            continue
        ax_top = axes[rows[0]]
        ax_bot = axes[rows[-1]]
        y_top = float(ax_top.get_position().y1)
        y_bot = float(ax_bot.get_position().y0)
        if y_top < y_bot:
            y_top, y_bot = y_bot, y_top
        box = ax_top.get_position()
        clip = Rectangle(
            (float(box.x0), y_bot),
            float(box.width),
            max(0.0, y_top - y_bot),
            transform=figure.transFigure,
            linewidth=0,
            fill=False,
        )
        line = Line2D(
            [float(x), float(x)],
            [y_bot, y_top],
            transform=blended_transform_factory(ax_top.transData, figure.transFigure),
            color=color,
            linestyle="--",
            linewidth=float(linewidth),
            clip_on=True,
            zorder=4,
            solid_capstyle="butt",
        )
        line.set_clip_path(clip)
        figure.add_artist(line)


def _attach_montage_metadata(figure: Any, spec: PanelSpec) -> None:
    """Metadata consumed by smokes / preserve_view (row_keys, kinds, …)."""
    if not (spec.row_keys or spec.row_kinds or str(spec.layout_kind) == "montage"):
        return
    kinds = list(spec.row_kinds) if spec.row_kinds else []
    channels = list(spec.row_channels) if spec.row_channels else []
    keys = list(spec.row_keys) if spec.row_keys else []
    figure._erg_montage_rows = max(1, int(spec.n_rows or 1))
    figure._erg_montage_row_kinds = kinds or None
    figure._erg_montage_row_channels = channels or None
    figure._erg_montage_row_streams = (
        [k if k in {"raw", "hp", "lp"} else "raw" for k in kinds] if kinds else None
    )
    if keys:
        figure._erg_montage_state = {
            "kind": "continuous_review" if str(spec.layout_kind) == "montage" else "stacked",
            "row_keys": keys,
            "kinds": kinds,
        }


def _size_montage_figure(figure: Any, n_rows: int, request: Any | None) -> None:
    settings = getattr(request, "settings", None) if request is not None else None
    min_h = max(36, int(getattr(settings, "montage_row_min_height_px", 36) or 36))
    fig_h = max(3.0, (max(1, n_rows) * min_h) / 96.0)
    try:
        figure.set_figheight(fig_h, forward=False)
    except Exception:
        pass


def _disable_layout_engine(figure: Any) -> None:
    try:
        figure.set_layout_engine(None)
    except Exception:
        try:
            figure.set_constrained_layout(False)
        except Exception:
            pass


def _draw_annotation(
    ax: Any, ann: AnnotationSpec, *, stim_linewidth: float = 1.0
) -> Any | None:
    if ann.kind == "vline":
        ax.axvline(
            float(ann.x),
            color=ann.color,
            linestyle="--",
            linewidth=float(stim_linewidth),
        )
        return None
    if ann.kind == "hline":
        return ax.axhline(
            float(ann.y),
            color=ann.color,
            linestyle=str(getattr(ann, "linestyle", ":") or ":"),
            linewidth=1.0,
        )
    elif ann.kind == "span":
        ax.axvspan(float(ann.x), float(ann.x2), color=ann.color, alpha=0.15)
    elif ann.kind == "stim_label" and ann.text:
        _draw_stim_label(ax, ann)
    elif ann.kind == "zone_label" and ann.text:
        _draw_zone_label(ax, ann)
    elif ann.kind == "text" and ann.text:
        ax.text(
            0.5,
            0.5,
            ann.text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=ann.color,
            fontsize=9,
        )


def _stim_label_pad_px(spec: PanelSpec) -> float:
    pad = 0.0
    for ann in spec.annotations:
        if str(getattr(ann, "kind", "") or "") != "stim_label":
            continue
        if not str(getattr(ann, "text", "") or "").strip():
            continue
        fs = max(4.0, float(getattr(ann, "fontsize", 8.0) or 8.0))
        need = fs * 1.35 + 6.0 + max(0.0, float(getattr(ann, "dy_px", 0.0) or 0.0))
        pad = max(pad, need)
    return pad


def _reserve_space_above_for_stim_labels(
    figure: Any, axes: list[Any], spec: PanelSpec
) -> None:
    """Marge figure au-dessus des axes — sans toucher aux ylim (ni aux pointillés)."""
    pad_px = _stim_label_pad_px(spec)
    if pad_px <= 0.0 or not axes:
        return
    try:
        fig_h = float(figure.get_figheight())
        dpi = float(getattr(figure, "dpi", 100.0) or 100.0)
    except Exception:
        return
    fig_h_px = max(1.0, fig_h * dpi)
    pad_frac = min(0.25, pad_px / fig_h_px)
    try:
        subs = figure.subplotpars
        new_top = max(float(subs.bottom) + 0.25, float(subs.top) - pad_frac)
        figure.subplots_adjust(top=new_top)
    except Exception:
        pass


def _zone_label_pad_px(spec: PanelSpec) -> float:
    pad = 0.0
    for ann in spec.annotations:
        if str(getattr(ann, "kind", "") or "") != "zone_label":
            continue
        if not str(getattr(ann, "text", "") or "").strip():
            continue
        fs = max(4.0, float(getattr(ann, "fontsize", 8.0) or 8.0))
        dy = float(getattr(ann, "dy_px", 0.0) or 0.0)
        need = fs * 1.35 + 6.0 + max(0.0, -dy)
        pad = max(pad, need)
    return pad


def _reserve_space_below_for_zone_labels(
    figure: Any, axes: list[Any], spec: PanelSpec
) -> None:
    """Marge figure sous les axes pour les libellés Zone (miroir on/off)."""
    pad_px = _zone_label_pad_px(spec)
    if pad_px <= 0.0 or not axes:
        return
    try:
        fig_h = float(figure.get_figheight())
        dpi = float(getattr(figure, "dpi", 100.0) or 100.0)
    except Exception:
        return
    fig_h_px = max(1.0, fig_h * dpi)
    pad_frac = min(0.25, pad_px / fig_h_px)
    try:
        subs = figure.subplotpars
        new_bottom = min(float(subs.top) - 0.25, float(subs.bottom) + pad_frac)
        figure.subplots_adjust(bottom=new_bottom)
    except Exception:
        pass


def _draw_stim_label(ax: Any, ann: AnnotationSpec) -> None:
    """Libellé « on » / « off » juste au-dessus du cadre des axes (hors courbes)."""
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    from panel_registry import apply_text_face
    from view_config import TextFace

    fig = ax.figure
    dpi = float(getattr(fig, "dpi", 100.0) or 100.0)
    dx = float(getattr(ann, "dx_px", 0.0) or 0.0)
    dy = float(getattr(ann, "dy_px", 0.0) or 0.0)
    # y=1 + va=bottom : au-dessus des axes ; les vlines restent dans le cadre.
    base = blended_transform_factory(ax.transData, ax.transAxes)
    offset = ScaledTranslation(dx / dpi, dy / dpi, fig.dpi_scale_trans)
    text = ax.text(
        float(ann.x),
        1.0,
        str(ann.text),
        transform=base + offset,
        ha="center",
        va="bottom",
        color=ann.color,
        fontsize=max(4.0, float(getattr(ann, "fontsize", 8.0) or 8.0)),
        clip_on=False,
        zorder=20,
    )
    apply_text_face(
        text,
        TextFace(
            bold=bool(getattr(ann, "bold", False)),
            italic=bool(getattr(ann, "italic", False)),
            underline=bool(getattr(ann, "underline", False)),
        ),
    )


def _draw_zone_label(ax: Any, ann: AnnotationSpec) -> None:
    """Libellé « Zone N » juste sous le cadre des axes (miroir de on/off)."""
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    from panel_registry import apply_text_face
    from view_config import TextFace

    fig = ax.figure
    dpi = float(getattr(fig, "dpi", 100.0) or 100.0)
    dx = float(getattr(ann, "dx_px", 0.0) or 0.0)
    dy = float(getattr(ann, "dy_px", 0.0) or 0.0)
    # y=0 + va=top : sous les axes (comme on/off au-dessus).
    base = blended_transform_factory(ax.transData, ax.transAxes)
    offset = ScaledTranslation(dx / dpi, dy / dpi, fig.dpi_scale_trans)
    text = ax.text(
        float(ann.x),
        0.0,
        str(ann.text),
        transform=base + offset,
        ha="center",
        va="top",
        color=ann.color,
        fontsize=max(4.0, float(getattr(ann, "fontsize", 8.0) or 8.0)),
        clip_on=False,
        zorder=20,
    )
    apply_text_face(
        text,
        TextFace(
            bold=bool(getattr(ann, "bold", False)),
            italic=bool(getattr(ann, "italic", False)),
            underline=bool(getattr(ann, "underline", False)),
        ),
    )
