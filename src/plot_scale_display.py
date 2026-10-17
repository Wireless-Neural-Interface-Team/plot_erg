"""Affichage écran / PDF : échelle unique (barres flottantes ou graduations)."""

from __future__ import annotations

from typing import Any

from scale_bar_model import (
    amp_manual_applies,
    major_ticks_for_axis,
    time_value_in_axis_units,
    units_for_kind,
    uses_shared_manual_scale,
)


def panel_scale_units(panel: str) -> tuple[str | None, str | None]:
    from panel_registry import _scale_bar_units

    return _scale_bar_units(panel)


def style_scale_lengths(
    style: object,
    *,
    x_unit: str | None,
    y_unit: str | None,
    ax: Any,
    shared: bool = True,
) -> tuple[float | None, float | None]:
    from panel_registry import _scale_bar_lengths

    return _scale_bar_lengths(
        style, x_unit=x_unit, y_unit=y_unit, ax=ax, shared=shared
    )


def apply_mpl_scale_ticks(
    ax: Any,
    style: object,
    *,
    x_unit: str | None,
    y_unit: str | None,
    x_size: float | None,
    y_size: float | None,
    shared: bool = True,
) -> None:
    """Graduations matplotlib calées sur le modèle d’échelle (pas les barres flottantes)."""
    try:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    except Exception:
        return
    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    x_span = abs(x1 - x0)
    y_span = abs(y1 - y0)
    try:
        bbox = ax.get_window_extent()
        w_px = max(1.0, float(bbox.width))
        h_px = max(1.0, float(bbox.height))
    except Exception:
        w_px, h_px = 400.0, 300.0

    amp_manual = shared and bool(getattr(style, "scale_bar_amplitude_manual", False))
    amp_length_manual = shared and bool(
        getattr(style, "scale_bar_amplitude_length_manual", False)
    )
    if y_unit is not None and y_span > 0 and amp_manual_applies(y_unit):
        y_ticks = major_ticks_for_axis(
            lo=y0,
            hi=y1,
            span=y_span,
            axis_px=h_px,
            unit=str(y_unit),
            manual_value=float(getattr(style, "scale_bar_amplitude", 0.0) or 0.0)
            if amp_manual
            else None,
            value_manual=amp_manual,
            length_px=float(getattr(style, "scale_bar_amplitude_length_px", 0.0) or 0.0)
            if amp_length_manual
            else None,
            length_manual=amp_length_manual,
        )
        if y_ticks:
            ax.set_yticks([p for p, _ in y_ticks])
            ax.set_yticklabels([lab for _, lab in y_ticks])

    if x_unit is not None and x_span > 0:
        time_manual = shared and bool(getattr(style, "scale_bar_time_manual", False))
        time_length_manual = shared and bool(
            getattr(style, "scale_bar_time_length_manual", False)
        )
        x_ticks = major_ticks_for_axis(
            lo=x0,
            hi=x1,
            span=x_span,
            axis_px=w_px,
            unit=str(x_unit),
            manual_value=time_value_in_axis_units(
                float(getattr(style, "scale_bar_time_s", 0.1) or 0.1), x_unit
            )
            if time_manual
            else None,
            value_manual=time_manual,
            length_px=float(getattr(style, "scale_bar_time_length_px", 0.0) or 0.0)
            if time_length_manual
            else None,
            length_manual=time_length_manual,
        )
        if x_ticks:
            ax.set_xticks([p for p, _ in x_ticks])
            ax.set_xticklabels([lab for _, lab in x_ticks])

    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True,
    )
    show_borders = bool(getattr(style, "show_borders", True))
    for spine in ax.spines.values():
        spine.set_visible(show_borders)


def apply_panel_scale_display(
    ax: Any,
    style: object,
    *,
    panel: str,
    keep_ylabel: bool = True,
    keep_xlabel: bool = True,
    draw_bars: bool = True,
    x_unit: str | None = None,
    y_unit: str | None = None,
) -> None:
    """Échelle temps/amplitude : barres flottantes ou graduations (même paramètres)."""
    from plot_utils import clear_floating_scale_bars, draw_floating_scale_bars, hide_edge_scale
    from view_config import resolved_scale_bar_linewidth, resolved_scale_bar_offset_px
    from view_config import resolved_scale_bar_time_offset_px

    if x_unit is None and y_unit is None:
        x_unit, y_unit = panel_scale_units(panel)
    if x_unit is None and y_unit is None:
        clear_floating_scale_bars(ax)
        return

    floating = bool(getattr(style, "show_scale_bars", False))
    shared = uses_shared_manual_scale(panel=panel)
    x_size, y_size = style_scale_lengths(
        style, x_unit=x_unit, y_unit=y_unit, ax=ax, shared=shared
    )

    if floating:
        hide_edge_scale(ax, keep_ylabel=keep_ylabel, keep_xlabel=keep_xlabel)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if not draw_bars:
            clear_floating_scale_bars(ax)
            return
        dx_px, dy_px = resolved_scale_bar_offset_px(style)
        time_dx_px, time_dy_px = resolved_scale_bar_time_offset_px(style)
        y_length_px = None
        if shared and bool(getattr(style, "scale_bar_amplitude_length_manual", False)):
            y_px = float(getattr(style, "scale_bar_amplitude_length_px", 0.0) or 0.0)
            if y_px > 0:
                y_length_px = y_px
        x_length_px = None
        if shared and bool(getattr(style, "scale_bar_time_length_manual", False)):
            x_px = float(getattr(style, "scale_bar_time_length_px", 0.0) or 0.0)
            if x_px > 0:
                x_length_px = x_px
        draw_floating_scale_bars(
            ax,
            x_unit=x_unit,
            y_unit=y_unit,
            x_size=x_size,
            y_size=y_size,
            fontsize=float(style.tick_font_size),
            linewidth=resolved_scale_bar_linewidth(style),
            gutter=True,
            face=style.tick_face,
            dx_px=dx_px,
            dy_px=dy_px,
            x_dx_px=time_dx_px,
            x_dy_px=time_dy_px,
            y_length_px=y_length_px,
            x_length_px=x_length_px,
            y_value_manual=bool(
                shared
                and getattr(style, "scale_bar_amplitude_manual", False)
                and y_unit == "µV"
            ),
            x_value_manual=bool(
                shared and getattr(style, "scale_bar_time_manual", False)
            ),
        )
        return

    clear_floating_scale_bars(ax)
    apply_mpl_scale_ticks(
        ax,
        style,
        x_unit=x_unit,
        y_unit=y_unit,
        x_size=x_size,
        y_size=y_size,
        shared=shared,
    )


def units_for_montage_row(
    row: int,
    *,
    panel: str,
    row_kinds: list[str],
) -> tuple[str | None, str | None]:
    panel_x, panel_y = panel_scale_units(panel)
    if row < len(row_kinds):
        return units_for_kind(str(row_kinds[row]), default_x=panel_x, default_y=panel_y)
    return (panel_x, panel_y)
