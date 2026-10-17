"""Items montage."""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem

class _MontageRowLabel(pg.LabelItem):
    """Libellé de ligne montage : droite + centre vertical du ViewBox.

    ``LabelItem(justify='right')`` de pyqtgraph aligne le texte en haut de la
    cellule — d’où le décalage croissant par rapport aux graphes (centrés).

    ``_dx`` positif rapproche le titre de la courbe, ``_dy`` positif le monte.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._anchor_plot: pg.PlotItem | None = None
        self._dx = 0.0
        self._dy = 0.0
        super().__init__(*args, **kwargs)
        try:
            self.setFlag(self.GraphicsItemFlag.ItemClipsChildrenToShape, False)
        except Exception:
            pass

    def boundingRect(self) -> Any:  # noqa: N802
        rect = super().boundingRect()
        if self._dx == 0.0 and self._dy == 0.0:
            return rect
        pad_x = abs(self._dx) + 4.0
        pad_y = abs(self._dy) + 4.0
        return rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)

    def set_anchor_plot(self, plot: pg.PlotItem | None) -> None:
        self._anchor_plot = plot
        try:
            self.resizeEvent(None)
        except Exception:
            pass

    def set_pixel_offset(self, dx: float, dy: float) -> None:
        try:
            self.prepareGeometryChange()
        except Exception:
            pass
        self._dx = float(dx)
        self._dy = float(dy)
        try:
            self.setZValue(20)
        except Exception:
            pass
        try:
            self.resizeEvent(None)
        except Exception:
            pass

    def resizeEvent(self, ev: Any) -> None:  # noqa: N802
        self.item.setPos(0, 0)
        bounds = self.itemRect()
        rect = self.rect()
        cy = rect.center().y()
        plot = self._anchor_plot
        if plot is not None:
            try:
                vb_center = plot.getViewBox().sceneBoundingRect().center()
                cy = float(self.mapFromScene(vb_center).y())
            except Exception:
                cy = rect.center().y()
        bounds.moveCenter(QPointF(rect.center().x(), cy - self._dy))
        bounds.moveRight(rect.right() + self._dx)
        self.item.setPos(bounds.topLeft() - self.itemRect().topLeft())
        self.updateMin()

class _MontageStimSpanItem(QGraphicsItem):
    """Pointillés de stimulation d’un seul tenant, d’un canal à l’autre.

    Les ``InfiniteLine`` sont rognées par chaque ViewBox : le trait s’arrête
    au bord du graphe. Cet item vit dans la scène et traverse aussi l’intervalle
    entre les lignes.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lines: list[tuple[float, float, float, QPen]] = []
        self._bounds = QRectF()
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setZValue(1000)
        # Forme vide : le trait ne doit pas intercepter le zoom / le pan.
        try:
            self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        except Exception:
            pass

    def set_lines(self, lines: list[tuple[float, float, float, QPen]]) -> None:
        self.prepareGeometryChange()
        self._lines = lines
        if not lines:
            self._bounds = QRectF()
            self.hide()
            self.update()
            return
        xs = [item[0] for item in lines]
        ys = [item[1] for item in lines] + [item[2] for item in lines]
        pad = 3.0
        self._bounds = QRectF(
            min(xs) - pad,
            min(ys) - pad,
            max(xs) - min(xs) + 2.0 * pad,
            max(ys) - min(ys) + 2.0 * pad,
        )
        self.show()
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._bounds

    def shape(self) -> QPainterPath:  # noqa: N802
        return QPainterPath()

    def paint(self, painter: Any, option: Any, widget: Any = None) -> None:  # noqa: N802
        del option, widget
        for x, y0, y1, pen in self._lines:
            painter.setPen(pen)
            painter.drawLine(QPointF(x, y0), QPointF(x, y1))
