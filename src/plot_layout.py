"""Modèle de layout écran : zone de tracé XOR + blocs périphériques indépendants.

Chaque rectangle (titre, stim, légende, graph, échelles, xlabel) a une taille
propre. Les blocs périphériques s’étendent uniquement vers l’extérieur :

    [ titre ]                 ↑ haut
    [ on / off stim ]         ↑ haut
    [légende][ GRAPH ][µV]    ← gauche / → droite
    [ échelle temps ]         ↓ bas
    [ Temps (s) ]             ↓ bas

Hauteur du GRAPH : ``graph_height_px`` **ou** échelle amplitude (valeur + px),
jamais les deux. Largeur du GRAPH : ``graph_width_px`` **ou** échelle temps
(valeur + px), jamais les deux.

La taille du GRAPH ne doit jamais être réduite pour faire de la place aux
autres blocs : le widget total grandit (chrome additif).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from view_config import (
    scale_bar_drives_widget_height,
    scale_bar_drives_widget_width,
)

HeightMode = Literal["fixed", "scale"]
WidthMode = Literal["fixed", "scale", "fill"]


@dataclass(frozen=True)
class GraphAreaSize:
    """Taille cible de la zone de tracé (ViewBox), hors chrome."""

    height_px: int | None
    """Somme des hauteurs ViewBox ; ``None`` = ne pas verrouiller."""

    width_px: int | None
    """Largeur ViewBox ; ``None`` = largeur fluide (remplit le conteneur)."""

    height_mode: HeightMode
    width_mode: WidthMode


@dataclass(frozen=True)
class PeripheralBlocks:
    """Blocs hors GRAPH — tailles indépendantes, expansion vers l’extérieur."""

    title: str = "up"
    stim_labels: str = "up"
    vertical_legend: str = "left"
    floating_amp_scale: str = "right"
    floating_time_scale: str = "down"
    xlabel: str = "down"


PERIPHERAL_EXPAND = PeripheralBlocks()


def height_mode_for(settings: object, *, montage: bool = False) -> HeightMode:
    """XOR hauteur : échelle amplitude manuelle prime sur la hauteur fixe."""
    if scale_bar_drives_widget_height(settings):
        return "scale"
    return "fixed"


def width_mode_for(settings: object) -> WidthMode:
    """XOR largeur : échelle temps manuelle prime ; sinon largeur fixe ou fill."""
    if scale_bar_drives_widget_width(settings):
        return "scale"
    try:
        width = int(getattr(settings, "graph_width_px", 0) or 0)
    except (TypeError, ValueError):
        width = 0
    if width > 0:
        return "fixed"
    return "fill"


def fixed_graph_height_px(settings: object, *, montage: bool = False) -> int:
    """Hauteur de zone de tracé demandée en mode fixe (une ligne montage ou panneau)."""
    if montage:
        return max(36, int(getattr(settings, "montage_row_min_height_px", 36) or 36))
    return max(40, int(getattr(settings, "graph_height_px", 400) or 400))


def fixed_graph_width_px(settings: object) -> int | None:
    """Largeur de zone de tracé en mode fixe, ou ``None`` si fill."""
    try:
        width = int(getattr(settings, "graph_width_px", 0) or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0:
        return None
    return max(40, width)


def resolve_graph_area_size(
    settings: object,
    *,
    montage: bool = False,
    n_rows: int = 1,
    scale_needed_height_px: int | None = None,
    scale_needed_width_px: int | None = None,
) -> GraphAreaSize:
    """Résoudre la taille ViewBox cible selon le XOR taille / échelle."""
    h_mode = height_mode_for(settings, montage=montage)
    w_mode = width_mode_for(settings)

    if h_mode == "scale":
        height = scale_needed_height_px
    else:
        row_h = fixed_graph_height_px(settings, montage=montage)
        rows = max(1, int(n_rows))
        height = row_h * rows if montage else row_h

    if w_mode == "scale":
        width = scale_needed_width_px
    elif w_mode == "fixed":
        width = fixed_graph_width_px(settings)
    else:
        width = None

    return GraphAreaSize(
        height_px=height,
        width_px=width,
        height_mode=h_mode,
        width_mode=w_mode,
    )


def widget_size_for_plot_area(
    *,
    plot_area_h: float,
    plot_area_w: float,
    widget_h: float,
    widget_w: float,
    target_h: float | None,
    target_w: float | None,
) -> tuple[int | None, int | None]:
    """Taille de widget pour préserver ``target_*`` ViewBox (chrome additif).

    ``required = widget - plot_area + target`` : le chrome mesuré est conservé,
    la zone de tracé est ramenée à la cible. Aucun chrome ne « mange » le graph.
    """
    out_h: int | None = None
    out_w: int | None = None
    if target_h is not None and float(target_h) > 0 and float(widget_h) >= 1.0:
        chrome_h = max(0.0, float(widget_h) - max(0.0, float(plot_area_h)))
        out_h = max(1, int(round(chrome_h + float(target_h))))
    if target_w is not None and float(target_w) > 0 and float(widget_w) >= 1.0:
        chrome_w = max(0.0, float(widget_w) - max(0.0, float(plot_area_w)))
        out_w = max(1, int(round(chrome_w + float(target_w))))
    return out_h, out_w
