"""Constantes du host pyqtgraph."""

from __future__ import annotations

INTERACTION_HINT = (
    "Clic texte = éditer · Ctrl+molette = zoom · clic milieu/droit = pan · "
    "double-clic = reset"
)
INTERACTION_HINT_NO_ZOOM = (
    "Clic texte = éditer · clic milieu/droit = pan (zoom désactivé sur cet aperçu)"
)

_COALESCE_MS = 16
_CHANNEL_HIGHLIGHT = "#fff7ed"
_MONTAGE_AXIS_WIDTH_MIN = 36
_MONTAGE_AXIS_WIDTH_MAX = 96
_MONTAGE_DATA_ROW_SPACING_PX = 0
_MONTAGE_X_AXIS_HEIGHT_PX = 30
_MONTAGE_DATA_BOTTOM_AXIS_PX = 0
_SCALE_BAR_PAD_PX = 6
_SCALE_BAR_RIGHT_GUTTER_PX = 58
_SCALE_BAR_BOTTOM_GUTTER_PX = 48
_MONTAGE_STREAM_KINDS = frozenset({"raw", "hp", "lp"})
