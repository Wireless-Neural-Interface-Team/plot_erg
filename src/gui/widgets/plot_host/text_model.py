"""Modèle texte éditable / pins / libellés stim-zone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pyqtgraph as pg
from PySide6.QtWidgets import QGraphicsSimpleTextItem

from .text_style import _axis_owning_label, _render_label_item

TextRole = Literal[
    "title",
    "xlabel",
    "ylabel",
    "legend",
    "annotation",
    "stim_on",
    "stim_off",
    "zone",
]


@dataclass(frozen=True)
class _TextPinKey:
    role: TextRole
    axes_index: int
    item_index: int = 0


@dataclass
class _StimLabelItem:
    """Libellé on/off en overlay scène, au-dessus de toutes les courbes."""

    item: QGraphicsSimpleTextItem
    plot: pg.PlotItem
    x: float
    dx_px: float
    dy_px: float
    fontsize: float = 8.0
    color: str = "#111827"


@dataclass
class _ZoneLabelItem:
    """Libellé Zone N en overlay scène, sous toutes les courbes."""

    item: QGraphicsSimpleTextItem
    plot: pg.PlotItem
    x: float
    dx_px: float
    dy_px: float
    fontsize: float = 8.0
    color: str = "#111827"


@dataclass
class _EditableText:
    item: Any
    role: TextRole
    axes_index: int
    item_index: int = 0

    @property
    def key(self) -> _TextPinKey:
        return _TextPinKey(self.role, int(self.axes_index), int(self.item_index))

    def get_text(self) -> str:
        item = self.item
        axis = _axis_owning_label(item)
        if axis is not None:
            return str(getattr(axis, "labelText", "") or "")
        if isinstance(item, pg.LabelItem):
            return str(item.text or "")
        if isinstance(item, QGraphicsSimpleTextItem):
            return str(item.text())
        if isinstance(item, pg.TextItem):
            inner = getattr(item, "textItem", None)
            if inner is not None:
                to_plain = getattr(inner, "toPlainText", None)
                if callable(to_plain):
                    return str(to_plain())
        to_plain = getattr(item, "toPlainText", None)
        if callable(to_plain):
            return str(to_plain())
        return str(getattr(item, "text", "") or "")

    def set_text(self, text: str) -> None:
        """Remplacer uniquement la chaîne, en gardant police et style."""
        item = self.item
        if isinstance(item, pg.LabelItem):
            _render_label_item(item, str(text))
            return
        axis = _axis_owning_label(item)
        if axis is not None:
            axis.labelText = str(text)
            try:
                axis._updateLabel()
            except Exception:
                pass
            return
        if isinstance(item, QGraphicsSimpleTextItem):
            try:
                item.setText(str(text))
            except Exception:
                pass
            return
        if isinstance(item, pg.TextItem):
            # Garder couleur / police déjà appliquées via setFont.
            try:
                item.setText(str(text))
            except Exception:
                pass
            return
        set_plain = getattr(item, "setPlainText", None)
        if callable(set_plain):
            set_plain(str(text))
            return
        set_text = getattr(item, "setText", None)
        if callable(set_text):
            set_text(str(text))

