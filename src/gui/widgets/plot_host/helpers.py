"""Helpers purs (barres d’échelle, montage X partagé, libellés)."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QGraphicsItem

from panel_prepare import PanelSpec, ScaleBarSpec

from .constants import _MONTAGE_STREAM_KINDS


def scale_bar_pair_px(bars: ScaleBarSpec, dx_name: str, dy_name: str) -> tuple[float, float]:
    """Lire une paire de décalages (dx, dy) en pixels. +dx droite, +dy haut."""
    try:
        dx = float(getattr(bars, dx_name, 0.0) or 0.0)
    except (TypeError, ValueError):
        dx = 0.0
    try:
        dy = float(getattr(bars, dy_name, 0.0) or 0.0)
    except (TypeError, ValueError):
        dy = 0.0
    if dx != dx:
        dx = 0.0
    if dy != dy:
        dy = 0.0
    return (dx, dy)


def scale_bar_offset_px(bars: ScaleBarSpec) -> tuple[float, float]:
    """Décalage de l’échelle amplitude (axe vertical)."""
    return scale_bar_pair_px(bars, "dx_px", "dy_px")


def scale_bar_time_offset_px(bars: ScaleBarSpec) -> tuple[float, float]:
    """Décalage de l’échelle temps (axe horizontal)."""
    return scale_bar_pair_px(bars, "time_dx_px", "time_dy_px")


def uncrop_scale_axis(axis: Any) -> None:
    """Laisser les barres dépasser la gouttière quand on les décale."""
    try:
        axis.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, False)
    except Exception:
        pass


def montage_shares_x(spec: PanelSpec) -> bool:
    """True si toutes les lignes montage partagent la même famille temporelle."""
    kinds = list(spec.row_kinds or ())
    if not kinds:
        return True
    return all(str(k) in _MONTAGE_STREAM_KINDS for k in kinds)


def _montage_label_shift(spec: Any) -> tuple[float, float]:
    return (
        float(getattr(spec, "row_label_dx_px", 0.0) or 0.0),
        float(getattr(spec, "row_label_dy_px", 0.0) or 0.0),
    )


def _montage_label_column_extra(spec: Any) -> int:
    """Largeur supplémentaire à gauche si le titre est décalé vers la gauche."""
    dx, _dy = _montage_label_shift(spec)
    if dx >= 0:
        return 0
    return int(round(-dx))
