"""Rendu matplotlib fond clair pour l’écran GUI."""

from __future__ import annotations

from typing import Any

from gui.theme import (
    SCOPE_ACCENT,
    SCOPE_AXES,
    SCOPE_BG,
    SCOPE_FG,
    SCOPE_GRID,
    SCOPE_SPINE,
    SCOPE_TITLE,
    SCOPE_ZERO_LINE,
)


def _axis_grid_visible(ax: Any) -> bool:
    """True si la grille est déjà active sur l’axe (réglage panneau / style)."""
    try:
        return any(
            line.get_visible()
            for line in list(ax.get_xgridlines()) + list(ax.get_ygridlines())
        )
    except Exception:
        return False


def apply_intan_scope_style(
    figure: Any,
    *,
    grid: bool | None = None,
    grid_alpha: float | None = None,
    show_borders: bool | None = None,
    ticks_inside: bool | None = None,
) -> None:
    """Appliquer un fond clair lisible (pas de thème sombre).

    ``grid`` / ``grid_alpha`` / ``show_borders`` / ``ticks_inside`` : si fournis,
    respectent le style utilisateur. Sinon, la visibilité déjà posée par le
    rendu du panneau est conservée.
    """
    try:
        figure.patch.set_facecolor(SCOPE_BG)
    except Exception:
        return

    for ax in list(getattr(figure, "axes", []) or []):
        try:
            ax.set_facecolor(SCOPE_AXES)
        except Exception:
            continue

        for spine in ax.spines.values():
            spine.set_color(SCOPE_SPINE)
            spine.set_linewidth(0.8)
            if show_borders is not None:
                spine.set_visible(bool(show_borders))

        tick_kwargs: dict[str, Any] = {"colors": SCOPE_FG, "which": "both"}
        if ticks_inside is not None:
            tick_kwargs["direction"] = "in" if ticks_inside else "out"
        ax.tick_params(**tick_kwargs)
        try:
            ax.xaxis.label.set_color(SCOPE_FG)
            ax.yaxis.label.set_color(SCOPE_FG)
        except Exception:
            pass

        title = ax.title
        if title is not None and title.get_text():
            title.set_color(SCOPE_TITLE)

        try:
            show_grid = bool(grid) if grid is not None else _axis_grid_visible(ax)
            if show_grid:
                alpha = 1.0 if grid_alpha is None else float(grid_alpha)
                ax.grid(True, color=SCOPE_GRID, linewidth=0.7, alpha=alpha)
                ax.set_axisbelow(True)
            else:
                ax.grid(False)
        except Exception:
            pass

        legend = ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor("#ffffff")
            frame.set_edgecolor(SCOPE_SPINE)
            frame.set_alpha(0.95)
            for text in legend.get_texts():
                text.set_color(SCOPE_FG)

    try:
        sup = figure._suptitle  # noqa: SLF001
        if sup is not None:
            sup.set_color(SCOPE_ACCENT)
    except Exception:
        pass


def style_empty_message(ax: Any, message: str) -> None:
    ax.set_facecolor(SCOPE_AXES)
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=SCOPE_FG,
        fontsize=10,
        wrap=True,
    )


def zero_line_color() -> str:
    return SCOPE_ZERO_LINE
