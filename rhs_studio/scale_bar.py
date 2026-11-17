"""Échelle flottante : barre en L déplaçable, ancrée en fraction de la vue."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFont, QFontMetricsF, QPen


class FloatingScaleBar(pg.GraphicsObject):
    """Enfant du ViewBox (coordonnées pixels) : la longueur suit l'échelle des données.

    Le coin du L est placé en (fx, fy) : fractions de la largeur depuis la gauche et
    de la hauteur depuis le bas. ``on_moved(fx, fy)`` est appelé à la fin d'un glisser.
    """

    def __init__(self, vb: pg.ViewBox, on_moved=None) -> None:
        super().__init__()
        self._vb = vb
        self._on_moved = on_moved
        self.x_len = 10.0
        self.y_len = 10.0
        self.x_text = ""
        self.y_text = ""
        self.show_x = True
        self.show_y = True
        self.fx = 0.9
        self.fy = 0.12
        self.pen = QPen(Qt.black)
        self.font = QFont()
        self._px = 0.0
        self._py = 0.0
        self._y_left = False
        self._drag_offset = QPointF()
        self.setParentItem(vb)
        self.setZValue(1e6)
        self.setCursor(Qt.OpenHandCursor)
        vb.sigRangeChanged.connect(self.update_geometry)
        vb.sigResized.connect(self.update_geometry)

    def configure(self, *, x_len, y_len, x_text, y_text, show_x, show_y, color, width, font_size, fx, fy):
        self.x_len, self.y_len = float(x_len), float(y_len)
        self.x_text, self.y_text = x_text, y_text
        self.show_x, self.show_y = bool(show_x), bool(show_y)
        self.pen = pg.mkPen(color, width=float(width))
        self.pen.setCapStyle(Qt.FlatCap)
        self.font = QFont()
        self.font.setPointSize(int(font_size))
        self.fx, self.fy = float(fx), float(fy)
        self.update_geometry()

    def update_geometry(self, *_):
        rect = self._vb.viewRect()
        w, h = self._vb.width(), self._vb.height()
        if rect.width() == 0 or rect.height() == 0 or w <= 0 or h <= 0:
            return
        self.prepareGeometryChange()
        self._px = abs(self.x_len / rect.width() * w) if self.show_x else 0.0
        self._py = abs(self.y_len / rect.height() * h) if self.show_y else 0.0
        self.setPos(self.fx * w, (1.0 - self.fy) * h)
        y_width = QFontMetricsF(self.font).boundingRect(self.y_text).width() + 10
        self._y_left = self.fx * w + y_width > w
        self.update()

    def _text_rects(self) -> tuple[QRectF, QRectF]:
        fm = QFontMetricsF(self.font)
        gap = 4.0
        xr = fm.boundingRect(self.x_text)
        yr = fm.boundingRect(self.y_text)
        x_rect = QRectF(-self._px / 2 - xr.width() / 2, gap, xr.width() + 2, fm.height())
        y_x = -(gap + 2 + yr.width() + 4) if self._y_left else gap + 2
        y_rect = QRectF(y_x, -self._py / 2 - fm.height() / 2, yr.width() + 4, fm.height())
        return x_rect, y_rect

    def boundingRect(self) -> QRectF:
        m = 6.0 + self.pen.widthF()
        r = QRectF(-self._px - m, -self._py - m, self._px + 2 * m, self._py + 2 * m)
        x_rect, y_rect = self._text_rects()
        if self.show_x and self.x_text:
            r = r.united(x_rect)
        if self.show_y and self.y_text:
            r = r.united(y_rect)
        return r.adjusted(-2, -2, 2, 2)

    def paint(self, p, *_):
        p.setRenderHint(p.RenderHint.Antialiasing, True)
        p.setPen(self.pen)
        if self.show_x and self._px > 0:
            p.drawLine(QPointF(-self._px, 0), QPointF(0, 0))
        if self.show_y and self._py > 0:
            p.drawLine(QPointF(0, 0), QPointF(0, -self._py))
        p.setFont(self.font)
        text_pen = QPen(self.pen.color())
        p.setPen(text_pen)
        x_rect, y_rect = self._text_rects()
        if self.show_x and self.x_text:
            p.drawText(x_rect, Qt.AlignHCenter | Qt.AlignTop, self.x_text)
        if self.show_y and self.y_text:
            align = Qt.AlignRight if self._y_left else Qt.AlignLeft
            p.drawText(y_rect, align | Qt.AlignVCenter, self.y_text)

    def hoverEvent(self, ev):
        if not ev.isExit():
            ev.acceptDrags(Qt.LeftButton)

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            self._drag_offset = self.pos() - self.mapToParent(ev.buttonDownPos())
        new_pos = self.mapToParent(ev.pos()) + self._drag_offset
        w, h = self._vb.width(), self._vb.height()
        if w > 0 and h > 0:
            self.fx = min(1.0, max(0.0, new_pos.x() / w))
            self.fy = min(1.0, max(0.0, 1.0 - new_pos.y() / h))
            self.update_geometry()
        if ev.isFinish() and self._on_moved is not None:
            self._on_moved(self.fx, self.fy)
