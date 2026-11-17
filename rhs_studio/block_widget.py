"""Bloc d'affichage : titre + graphique pyqtgraph + zone de zoom + infos."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPoint, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QFrame, QLabel, QMenu, QVBoxLayout

from .compute import BlockData, channel_colors
from .config import BlockConfig
from .scale_bar import FloatingScaleBar

PEN_STYLES = {
    "solid": Qt.SolidLine,
    "dash": Qt.DashLine,
    "dot": Qt.DotLine,
    "dashdot": Qt.DashDotLine,
}
LEGEND_ANCHORS = {
    "top-right": ((1, 0), (1, 0), (-8, 8)),
    "top-left": ((0, 0), (0, 0), (8, 8)),
    "bottom-right": ((1, 1), (1, 1), (-8, -8)),
    "bottom-left": ((0, 1), (0, 1), (8, -8)),
}
SELECT_COLOR = "#2f81f7"


def with_alpha(color: str, alpha: float) -> QColor:
    c = QColor(color)
    c.setAlphaF(max(0.0, min(1.0, float(alpha))))
    return c


def _pen(color, width, style="solid"):
    pen = pg.mkPen(color, width=float(width))
    pen.setStyle(PEN_STYLES.get(style, Qt.SolidLine))
    pen.setCosmetic(True)
    return pen


class BlockWidget(QFrame):
    clicked = Signal(str)
    zoomRegionChanged = Signal(str, float, float)
    scaleBarMoved = Signal(str, float, float)
    contextAction = Signal(str, str)

    def __init__(self, cfg: BlockConfig, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.data: BlockData | None = None
        self.selected = False
        self._busy = False
        self._error = ""
        self._new_data = False
        self._syncing = False
        self.main_plot: pg.PlotItem | None = None
        self.zoom_plot: pg.PlotItem | None = None
        self.region: pg.LinearRegionItem | None = None
        self.scale_bar: FloatingScaleBar | None = None
        self.legend: pg.LegendItem | None = None

        self.setObjectName("block")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)
        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setContentsMargins(2, 2, 2, 2)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        lay.addWidget(self.title_label)
        lay.addWidget(self.glw, 1)
        lay.addWidget(self.info_label)
        self.glw.viewport().installEventFilter(self)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self.redraw()

    # ------------------------------------------------------------------ état
    def set_selected(self, on: bool) -> None:
        self.selected = on
        self._apply_frame()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._error = "" if busy else self._error
        self._update_info()

    def set_error(self, text: str) -> None:
        self._busy = False
        self._error = text
        self._update_info()

    def set_data(self, data: BlockData) -> None:
        self.data = data
        self._busy = False
        self._error = ""
        self._new_data = True
        self.redraw()

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.MouseButtonPress:
            self.clicked.emit(self.cfg.id)
        return False

    def mousePressEvent(self, ev):
        self.clicked.emit(self.cfg.id)
        super().mousePressEvent(ev)

    def sizeHint(self) -> QSize:
        return QSize(self.cfg.width, self.cfg.height)

    # ---------------------------------------------------------------- dessin
    def _apply_frame(self) -> None:
        c = self.cfg
        border = SELECT_COLOR if self.selected else c.frame_color
        width = max(2, c.frame_width) if self.selected else c.frame_width
        self.setStyleSheet(
            f"QFrame#block {{ background: {c.bg_color}; border: {width}px solid {border}; }}"
        )

    def _update_info(self) -> None:
        c = self.cfg
        if self._error:
            self.info_label.setText(f"<span style='color:#cf222e'>Erreur : {self._error}</span>")
            self.info_label.setVisible(True)
            return
        if self._busy:
            self.info_label.setText("<span style='color:#6e7781'>Calcul en cours…</span>")
            self.info_label.setVisible(True)
            return
        text = self.data.info if self.data else ""
        self.info_label.setText(f"<span style='color:#57606a; font-size:8pt'>{text}</span>")
        self.info_label.setVisible(bool(c.info_show and text))

    def redraw(self) -> None:
        c = self.cfg
        self.setFixedSize(int(c.width), int(c.height))
        self._apply_frame()

        self.title_label.setVisible(c.title_show)
        font = QFont()
        font.setPointSize(int(c.title_size))
        font.setBold(bool(c.title_bold))
        self.title_label.setFont(font)
        self.title_label.setText(c.title)
        self.title_label.setStyleSheet(f"color: {c.title_color}; background: transparent; border: none;")
        align = {"left": Qt.AlignLeft, "right": Qt.AlignRight}.get(c.title_align, Qt.AlignHCenter)
        self.title_label.setAlignment(align | Qt.AlignVCenter)
        self.info_label.setStyleSheet("background: transparent; border: none;")
        self._update_info()

        prev_range = None
        if self.main_plot is not None and not self._new_data:
            vb = self.main_plot.getViewBox()
            prev_range = (vb.viewRange(), vb.autoRangeEnabled())
        self._new_data = False

        self.glw.clear()
        self.glw.setBackground(c.bg_color)
        self.legend = None
        self.region = None
        self.scale_bar = None
        self.zoom_plot = None

        self.main_plot = self.glw.addPlot(row=0, col=0)
        if c.zoom_show:
            if c.zoom_side == "bottom":
                self.zoom_plot = self.glw.addPlot(row=1, col=0)
                lay = self.glw.ci.layout
                lay.setRowStretchFactor(0, int(round((1 - c.zoom_ratio) * 100)))
                lay.setRowStretchFactor(1, int(round(c.zoom_ratio * 100)))
            else:
                self.zoom_plot = self.glw.addPlot(row=0, col=1)
                lay = self.glw.ci.layout
                lay.setColumnStretchFactor(0, int(round((1 - c.zoom_ratio) * 100)))
                lay.setColumnStretchFactor(1, int(round(c.zoom_ratio * 100)))

        data = self.data or BlockData()
        x_label = c.x_label or data.x_label
        y_label = c.y_label or data.y_label
        for plot in (self.main_plot, self.zoom_plot):
            if plot is None:
                continue
            plot.setDownsampling(auto=True, mode="peak")
            plot.setClipToView(True)
            if plot is self.main_plot:
                self._style_axes(plot, x_label, y_label)
            else:
                self._style_axes(plot, "", "")
            plot.getViewBox().setAutoVisible(y=True)

        if c.legend_show:
            anchor = LEGEND_ANCHORS.get(c.legend_pos, LEGEND_ANCHORS["top-right"])
            self.legend = pg.LegendItem(
                offset=anchor[2],
                labelTextSize=f"{c.legend_size}pt",
                labelTextColor=c.legend_text_color,
                brush=pg.mkBrush(c.legend_bg),
                pen=pg.mkPen(c.legend_frame),
                verSpacing=-4,
            )
            self.legend.setParentItem(self.main_plot.getViewBox())
            self.legend.anchor(anchor[0], anchor[1], anchor[2])

        self._populate(self.main_plot, data, self.legend)
        if self.zoom_plot is not None:
            self._populate(self.zoom_plot, data, None)
            self.zoom_plot.getViewBox().setBorder(_pen(c.zoom_color, 1.5))
        if self.legend is not None and not self.legend.items:
            self.legend.setVisible(False)

        self._apply_ranges(prev_range)
        if self.zoom_plot is not None:
            self._setup_zoom(data)
        if c.sb_show:
            self.scale_bar = FloatingScaleBar(self.main_plot.getViewBox(), self._scale_bar_moved)
            x_unit = c.sb_x_unit or data.x_unit
            y_unit = c.sb_y_unit or data.y_unit
            self.scale_bar.configure(
                x_len=c.sb_x_len, y_len=c.sb_y_len,
                x_text=f"{c.sb_x_len:g} {x_unit}".strip(),
                y_text=f"{c.sb_y_len:g} {y_unit}".strip(),
                show_x=c.sb_show_x, show_y=c.sb_show_y,
                color=c.sb_color, width=c.sb_width, font_size=c.sb_font_size,
                fx=c.sb_fx, fy=c.sb_fy,
            )

    def _style_axes(self, plot: pg.PlotItem, x_label: str, y_label: str) -> None:
        c = self.cfg
        tick_font = QFont()
        tick_font.setPointSize(int(c.tick_size))
        style = {"color": c.axis_color, "font-size": f"{c.axis_label_size}pt"}
        for name, label, show in (("bottom", x_label, c.show_x_axis), ("left", y_label, c.show_y_axis)):
            ax = plot.getAxis(name)
            ax.setPen(pg.mkPen(c.axis_color, width=float(c.axis_width)))
            ax.setTextPen(pg.mkPen(c.axis_color))
            ax.setTickFont(tick_font)
            if label:
                plot.setLabel(name, label, **style)
            else:
                ax.showLabel(False)
            if show:
                plot.showAxis(name)
            else:
                plot.hideAxis(name)
        plot.showGrid(x=c.grid, y=c.grid, alpha=float(c.grid_alpha))

    def _populate(self, plot: pg.PlotItem, data: BlockData, legend: pg.LegendItem | None) -> None:
        c = self.cfg
        colors = channel_colors(c)
        for s in data.series:
            color = colors[s.k] if 0 <= s.k < len(colors) else "#808080"
            if s.role == "band":
                lo = pg.PlotDataItem(s.x, s.y, pen=None)
                hi = pg.PlotDataItem(s.x, s.y2, pen=None)
                plot.addItem(lo)
                plot.addItem(hi)
                plot.addItem(pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(with_alpha(color, c.band_alpha))))
                continue
            if s.role in ("main", "mean"):
                width = c.line_width if s.role == "main" else c.mean_width
                item = pg.PlotDataItem(s.x, s.y, pen=_pen(color, width))
            elif s.role == "trial":
                color = color if c.trial_use_channel_color else c.trial_color
                item = pg.PlotDataItem(
                    s.x, s.y, pen=_pen(with_alpha(color, c.trial_alpha), c.trial_width), connect="finite"
                )
            elif s.role == "markers":
                item = pg.PlotDataItem(
                    s.x, s.y, pen=None, symbol="o", symbolSize=c.spike_size,
                    symbolBrush=pg.mkBrush(c.spike_color), symbolPen=None,
                )
            elif s.role == "raster":
                item = pg.PlotDataItem(
                    s.x, s.y, pen=None, symbol="|", symbolSize=c.spike_size + 4,
                    symbolPen=None, symbolBrush=pg.mkBrush(color),
                )
            elif s.role == "psth":
                item = pg.PlotDataItem(
                    s.x, s.y, stepMode="center", fillLevel=0,
                    brush=pg.mkBrush(with_alpha(color, 0.45)), pen=_pen(color, c.line_width),
                )
            elif s.role == "thr":
                if not c.threshold_show:
                    continue
                pen = _pen(c.threshold_color, c.threshold_width, c.threshold_style)
                plot.addItem(pg.InfiniteLine(pos=float(s.y[0]), angle=0, pen=pen))
                if legend is not None:
                    legend.addItem(pg.PlotDataItem([], [], pen=pen), s.name)
                continue
            else:
                continue
            plot.addItem(item)
            if s.role in ("markers", "raster", "psth"):
                item.setDownsampling(ds=1, auto=False)
                item.setClipToView(False)
            if legend is not None and s.role != "markers":
                legend.addItem(item, s.name)
        self._trigger_lines(plot, data, legend)

    def _trigger_lines(self, plot, data: BlockData, legend) -> None:
        c = self.cfg
        specs = (
            ("ON", data.trig_on, c.trig_on_show, c.trig_on_color, c.trig_on_width, c.trig_on_style),
            ("OFF", data.trig_off, c.trig_off_show, c.trig_off_color, c.trig_off_width, c.trig_off_style),
        )
        for label, xs, show, color, width, style in specs:
            if not show or xs is None or len(xs) == 0:
                continue
            pen = _pen(color, width, style)
            xs = np.asarray(xs)[: max(1, int(c.trig_max_lines))]
            text = label if c.trig_label_show and len(xs) <= 20 else None
            for x in xs.tolist():
                line = pg.InfiniteLine(
                    pos=x, angle=90, pen=pen, movable=False, label=text,
                    labelOpts={"position": 0.96, "color": color, "anchors": [(0, 0), (0, 0)]},
                )
                plot.addItem(line, ignoreBounds=True)
            if legend is not None:
                legend.addItem(pg.PlotDataItem([], [], pen=pen), f"Trigger {label}")

    def _apply_ranges(self, prev_range) -> None:
        c = self.cfg
        plot = self.main_plot
        vb = plot.getViewBox()
        extent = self.data.x_extent if self.data is not None else None
        if c.x_auto:
            if prev_range is not None and not prev_range[1][0]:
                plot.setXRange(*prev_range[0][0], padding=0)
            elif extent is not None:
                plot.setXRange(*extent, padding=0.02)
            else:
                plot.enableAutoRange(axis="x")
        else:
            plot.setXRange(c.x_min, c.x_max, padding=0)
        if c.y_auto:
            if prev_range is not None and not prev_range[1][1]:
                plot.setYRange(*prev_range[0][1], padding=0)
            else:
                plot.enableAutoRange(axis="y")
        else:
            plot.setYRange(c.y_min, c.y_max, padding=0)
        vb.updateAutoRange()

    def _setup_zoom(self, data: BlockData) -> None:
        c = self.cfg
        x0, x1 = c.zoom_x0, c.zoom_x1
        if x1 <= x0 and data.x_extent is not None:
            a, b = data.x_extent
            span = b - a
            x0, x1 = a + 0.4 * span, a + 0.6 * span
            c.zoom_x0, c.zoom_x1 = x0, x1
        self.region = pg.LinearRegionItem(
            values=(x0, x1),
            brush=pg.mkBrush(with_alpha(c.zoom_color, c.zoom_alpha)),
            pen=_pen(c.zoom_color, 1.2),
        )
        self.region.setZValue(10)
        self.main_plot.addItem(self.region, ignoreBounds=True)
        zp = self.zoom_plot
        zp.setXRange(x0, x1, padding=0)
        if c.zoom_y_auto:
            zp.enableAutoRange(axis="y")
        else:
            zp.setYRange(c.y_min, c.y_max, padding=0)

        def region_moved():
            if self._syncing:
                return
            self._syncing = True
            zp.setXRange(*self.region.getRegion(), padding=0)
            self._syncing = False

        def zoom_moved(_vb, rng):
            if self._syncing:
                return
            self._syncing = True
            self.region.setRegion(rng)
            self._syncing = False
            self._region_done()

        self.region.sigRegionChanged.connect(region_moved)
        self.region.sigRegionChangeFinished.connect(self._region_done)
        zp.sigXRangeChanged.connect(zoom_moved)

    def _region_done(self, *_):
        if self.region is None:
            return
        x0, x1 = self.region.getRegion()
        self.cfg.zoom_x0, self.cfg.zoom_x1 = float(x0), float(x1)
        self.zoomRegionChanged.emit(self.cfg.id, float(x0), float(x1))

    def _scale_bar_moved(self, fx: float, fy: float) -> None:
        self.cfg.sb_fx, self.cfg.sb_fy = fx, fy
        self.scaleBarMoved.emit(self.cfg.id, fx, fy)

    # ----------------------------------------------------------- interactions
    def current_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        (x0, x1), (y0, y1) = self.main_plot.getViewBox().viewRange()
        return (x0, x1), (y0, y1)

    def _context_menu(self, pos) -> None:
        menu = QMenu(self)
        actions = [
            ("freeze", "Figer les échelles actuelles"),
            ("auto", "Revenir aux échelles automatiques"),
            (None, None),
            ("export_png", "Exporter ce bloc en PNG…"),
            ("export_svg", "Exporter ce bloc en SVG…"),
            (None, None),
            ("duplicate", "Dupliquer"),
            ("delete", "Supprimer"),
        ]
        for key, text in actions:
            if key is None:
                menu.addSeparator()
                continue
            act = menu.addAction(text)
            act.setData(key)
        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen is not None:
            self.contextAction.emit(self.cfg.id, chosen.data())

    def render_image(self, scale: float = 2.0) -> QImage:
        was = self.selected
        self.set_selected(False)
        img = QImage(self.size() * scale, QImage.Format_ARGB32)
        img.setDevicePixelRatio(scale)
        img.fill(Qt.white)
        painter = QPainter(img)
        self.render(painter, QPoint())
        painter.end()
        self.set_selected(was)
        return img

    def export_svg(self, path: str) -> None:
        from PySide6.QtSvg import QSvgGenerator

        was = self.selected
        self.set_selected(False)
        gen = QSvgGenerator()
        gen.setFileName(path)
        gen.setSize(self.size())
        gen.setViewBox(QRectF(0, 0, self.width(), self.height()))
        gen.setTitle(self.cfg.title)
        painter = QPainter(gen)
        self.render(painter, QPoint())
        painter.end()
        self.set_selected(was)
