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
)

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
        *(_MONTAGE_TRIGGERED.keys()),
    }
)


def uses_prepare_mpl(panel: str) -> bool:
    return str(panel or "") in PREPARE_MPL_PANELS


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
        status in {"unavailable", "pending", "empty"} and not spec.series
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

    for ann in spec.annotations:
        row = int(ann.row or 0)
        if row < 0 or row >= n_rows:
            row = 0
        artist = _draw_annotation(axes_flat[row], ann)
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
        if row < len(spec.row_labels) and spec.row_labels[row]:
            ax.set_ylabel(str(spec.row_labels[row]), fontsize=label_fs)
        elif row == 0 and spec.ylabel:
            ax.set_ylabel(spec.ylabel, fontsize=label_fs)
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

    if spec.title and not is_montage:
        figure.suptitle(
            spec.title,
            fontsize=float(spec.title_font_size),
            fontweight="bold" if spec.title_bold else "normal",
            fontstyle="italic" if spec.title_italic else "normal",
        )
        _apply_spec_text_face(getattr(figure, "_suptitle", None), spec, role="title")

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

    _attach_montage_metadata(figure, spec)
    figure._erg_status = status
    return status


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


def _draw_annotation(ax: Any, ann: AnnotationSpec) -> Any | None:
    if ann.kind == "vline":
        ax.axvline(float(ann.x), color=ann.color, linestyle="--", linewidth=1.0)
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
