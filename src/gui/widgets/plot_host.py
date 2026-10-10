"""Pyqtgraph host for screen panels (Phase 3).

Replaces matplotlib InteractiveCanvas for on-screen rendering. Matplotlib
remains the PDF / export drawing path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from panel_prepare import (
    AnnotationSpec,
    CurveSeries,
    HistSeries,
    PanelSpec,
    ScaleBarSpec,
    ScatterSeries,
    prepare_panel,
)

INTERACTION_HINT = (
    "Clic texte = éditer · Ctrl+molette = zoom · clic milieu/droit = pan · "
    "double-clic = reset"
)
INTERACTION_HINT_NO_ZOOM = (
    "Clic texte = éditer · clic milieu/droit = pan (zoom désactivé sur cet aperçu)"
)

_COALESCE_MS = 16
_CHANNEL_HIGHLIGHT = "#fff7ed"
_MONTAGE_AXIS_WIDTH = 44  # px — ticks Y alignés (échelle horizontale uniforme)
_MONTAGE_LABEL_SIZE = "8pt"

TextRole = Literal["title", "xlabel", "ylabel", "legend", "annotation"]


@dataclass(frozen=True)
class _TextPinKey:
    role: TextRole
    axes_index: int
    item_index: int = 0


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
        if isinstance(item, pg.LabelItem):
            return str(item.text or "")
        to_plain = getattr(item, "toPlainText", None)
        if callable(to_plain):
            return str(to_plain())
        return str(getattr(item, "text", "") or "")

    def set_text(self, text: str) -> None:
        item = self.item
        if isinstance(item, pg.LabelItem):
            item.setText(str(text))
            return
        set_plain = getattr(item, "setPlainText", None)
        if callable(set_plain):
            set_plain(str(text))
            return
        set_text = getattr(item, "setText", None)
        if callable(set_text):
            set_text(str(text))


def _scope_colors() -> dict[str, str]:
    try:
        from gui.theme import (
            SCOPE_ACCENT,
            SCOPE_AXES,
            SCOPE_BG,
            SCOPE_FG,
            SCOPE_GRID,
            SCOPE_SPINE,
            SCOPE_TITLE,
        )

        return {
            "bg": SCOPE_BG,
            "axes": SCOPE_AXES,
            "fg": SCOPE_FG,
            "grid": SCOPE_GRID,
            "spine": SCOPE_SPINE,
            "title": SCOPE_TITLE,
            "accent": SCOPE_ACCENT,
        }
    except Exception:
        return {
            "bg": "#ffffff",
            "axes": "#ffffff",
            "fg": "#1f2937",
            "grid": "#e5e7eb",
            "spine": "#9ca3af",
            "title": "#111827",
            "accent": "#2563eb",
        }


class _AxisShim:
    """Minimal matplotlib-like axis for capture/restore and range-bar attach."""

    def __init__(self, plot_item: pg.PlotItem) -> None:
        self._plot = plot_item

    def get_xlim(self) -> tuple[float, float]:
        x0, x1 = self._plot.viewRange()[0]
        return float(x0), float(x1)

    def get_ylim(self) -> tuple[float, float]:
        y0, y1 = self._plot.viewRange()[1]
        return float(y0), float(y1)

    def set_xlim(self, a: float, b: float) -> None:
        self._plot.setXRange(float(a), float(b), padding=0)

    def set_ylim(self, a: float, b: float) -> None:
        self._plot.setYRange(float(a), float(b), padding=0)


class _FigureShim:
    """Compatibility shim so callers can read ``figure.axes`` / montage attrs."""

    def __init__(self, host: "PlotHost") -> None:
        self._host = host
        self.axes: list[_AxisShim] = []
        self._erg_montage_state: dict[str, Any] | None = None
        self._erg_montage_row_kinds: list[str] | None = None
        self._erg_montage_row_streams: list[str] | None = None
        self._erg_montage_row_channels: list[int] | None = None
        self._erg_status: str = "pending"

    def clear(self) -> None:
        self.axes.clear()
        self._erg_montage_state = None
        self._erg_montage_row_kinds = None
        self._erg_montage_row_streams = None
        self._erg_montage_row_channels = None


class PlotHost(QWidget):
    """QWidget wrapping ``GraphicsLayoutWidget`` with coalesce + view API."""

    viewChanged = Signal()
    textEdited = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        show_toolbar: bool = True,
        show_cursor: bool = True,
        min_height: int = 80,
        allow_zoom: bool = True,
        figsize: tuple[float, float] | None = None,
        layout: Any = None,
        **_kwargs: Any,
    ) -> None:
        del figsize, layout  # accepted for InteractiveCanvas API compatibility
        super().__init__(parent)
        self._allow_zoom = bool(allow_zoom)
        self._colors = _scope_colors()
        self._spec: PanelSpec | None = None
        self._plots: list[pg.PlotItem] = []
        self._text_items: list[pg.TextItem] = []
        self._row_labels: list[pg.LabelItem] = []
        self._editable: list[_EditableText] = []
        self._text_pins: dict[_TextPinKey, str] = {}
        self._text_edit: QLineEdit | None = None
        self._text_hit: _EditableText | None = None
        self._text_closing = False
        self._plot_col = 0
        self._scale_bar_items: list[tuple[pg.PlotItem, Any]] = []
        self._scale_bar_range_connected: list[Any] = []
        self._scale_bars_refreshing = False
        # Longueurs figées au rendu (pixel size suit le zoom ; label stable).
        self._scale_bar_lengths: list[tuple[float | None, float | None]] = []

        pg.setConfigOptions(antialias=True, foreground=self._colors["fg"])

        self.figure = _FigureShim(self)
        self.canvas = self  # grab() / attach callers expect a QWidget
        self.toolbar: QWidget | None = None

        self._glw = pg.GraphicsLayoutWidget(self)
        self._glw.setBackground(self._colors["bg"])
        self._glw.setMinimumHeight(int(min_height))
        self._glw.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        # Left button free for range bars; pan via middle/right.
        self._glw.viewport().setMouseTracking(True)

        hint = INTERACTION_HINT if self._allow_zoom else INTERACTION_HINT_NO_ZOOM
        self._cursor_label = QLabel(hint, self)
        self._cursor_label.setObjectName("panelStatus")
        self._cursor_label.setWordWrap(True)
        self._cursor_label.setVisible(bool(show_cursor))

        bar = QWidget(self)
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(2, 0, 2, 0)
        bar_layout.setSpacing(4)
        home_btn = QToolButton(bar)
        home_btn.setText("⌂")
        home_btn.setToolTip("Réinitialiser la vue")
        home_btn.setAutoRaise(True)
        home_btn.clicked.connect(self.reset_view)
        bar_layout.addWidget(home_btn)
        bar_layout.addStretch(1)
        bar.setMaximumHeight(28)
        bar.setVisible(bool(show_toolbar))
        self.toolbar = bar

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(bar)
        root.addWidget(self._glw, 1)
        root.addWidget(self._cursor_label)

        self._draw_pending = False
        self._draw_timer = QTimer(self)
        self._draw_timer.setSingleShot(True)
        self._draw_timer.setInterval(_COALESCE_MS)
        self._draw_timer.timeout.connect(self._flush_draw)

        self._install_interactions()

    # ---------------------------------------------------------------- API

    def clear_text_pins(self) -> None:
        """Oublier les éditions in-place (ex. réinitialisation style)."""
        self._cancel_text_edit()
        self._text_pins.clear()

    def draw_idle(self) -> None:
        self._draw_pending = True
        if not self._draw_timer.isActive():
            self._draw_timer.start()

    def _flush_draw(self) -> None:
        self._draw_pending = False
        try:
            self._glw.update()
        except Exception:
            pass

    def render_request(self, request: Any) -> str:
        spec = prepare_panel(request)
        return self.render_spec(spec)

    def render_spec(self, spec: PanelSpec) -> str:
        self._cancel_text_edit()
        self._spec = spec
        self._clear_plots()
        if spec.layout_kind == "placeholder" or spec.status in {
            "unavailable",
            "pending",
            "empty",
        } and not spec.series:
            self._draw_placeholder(spec)
            self.figure._erg_status = spec.status
            self.draw_idle()
            return spec.status

        is_montage = spec.layout_kind == "montage"
        n_rows = max(1, int(spec.n_rows or 1))
        label_col = 0 if is_montage else -1
        plot_col = 1 if is_montage else 0
        self._plot_col = plot_col
        self._plots = []
        self._row_labels = []
        self.figure.axes = []

        if is_montage:
            self._add_montage_row_labels(spec, n_rows=n_rows, label_col=label_col)

        for row in range(n_rows):
            plot = self._glw.addPlot(row=row, col=plot_col)
            self._style_plot(plot, montage=is_montage)
            self._configure_mouse(plot)
            if is_montage:
                # Largeur fixe des ticks Y → boîtes de tracé alignées (échelle X uniforme).
                plot.getAxis("left").setWidth(_MONTAGE_AXIS_WIDTH)
            self._plots.append(plot)
            self.figure.axes.append(_AxisShim(plot))
            if row > 0:
                plot.setXLink(self._plots[0])
            try:
                self._glw.ci.layout.setRowStretchFactor(row, 1)
            except Exception:
                pass

        legend_entries = self._draw_series(spec)
        self._draw_annotations(spec)
        self._apply_limits(spec)
        self._apply_labels(spec, is_montage=is_montage)
        self._apply_scale_bars(spec, is_montage=is_montage)
        if spec.show_legend and legend_entries:
            self._place_legend_below(legend_entries, n_rows=n_rows, plot_col=plot_col)
        self._reapply_text_pins()

        # Montage metadata for view restore / range bars.
        if spec.row_keys:
            self.figure._erg_montage_state = {
                "kind": "continuous_review",
                "row_keys": list(spec.row_keys),
            }
        self.figure._erg_montage_row_kinds = list(spec.row_kinds) if spec.row_kinds else None
        self.figure._erg_montage_row_streams = (
            list(spec.row_kinds) if spec.row_kinds else None
        )
        self.figure._erg_montage_row_channels = (
            list(spec.row_channels) if spec.row_channels else None
        )
        self.figure._erg_status = spec.status
        self.draw_idle()
        return spec.status

    def plot_items(self) -> list[pg.PlotItem]:
        return list(self._plots)

    def capture_view(self) -> list[tuple[tuple[float, float], tuple[float, float]]] | None:
        return self.capture_view_limits()

    def restore_view(
        self,
        limits: list[tuple[tuple[float, float], tuple[float, float]]] | None,
        *,
        restore_x: bool = True,
        restore_y: bool | list[bool] | tuple[bool, ...] = True,
    ) -> None:
        self.restore_view_limits(limits, restore_x=restore_x, restore_y=restore_y)

    def capture_view_limits(
        self,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]] | None:
        if not self._plots:
            return None
        out: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for plot in self._plots:
            try:
                (x0, x1), (y0, y1) = plot.viewRange()
            except Exception:
                return None
            if not np.isfinite([x0, x1, y0, y1]).all() or x0 == x1:
                return None
            out.append(((float(x0), float(x1)), (float(y0), float(y1))))
        return out

    def capture_montage_view_limits(
        self,
    ) -> dict[tuple[int, str], tuple[tuple[float, float], tuple[float, float]]] | None:
        stored = self.figure._erg_montage_state
        if not isinstance(stored, dict):
            return None
        row_keys = list(stored.get("row_keys") or [])
        if not row_keys or len(row_keys) != len(self._plots):
            return None
        limits: dict[tuple[int, str], tuple[tuple[float, float], tuple[float, float]]] = {}
        for key, plot in zip(row_keys, self._plots):
            if not (isinstance(key, (tuple, list)) and len(key) >= 2):
                return None
            try:
                (x0, x1), (y0, y1) = plot.viewRange()
            except Exception:
                return None
            if not np.isfinite([x0, x1, y0, y1]).all() or x0 == x1:
                return None
            limits[(int(key[0]), str(key[1]))] = (
                (float(x0), float(x1)),
                (float(y0), float(y1)),
            )
        return limits

    def restore_view_limits(
        self,
        limits: list[tuple[tuple[float, float], tuple[float, float]]] | None,
        *,
        restore_x: bool = True,
        restore_y: bool | list[bool] | tuple[bool, ...] = True,
    ) -> None:
        if not limits or not self._plots:
            return
        n = min(len(limits), len(self._plots))
        if isinstance(restore_y, (list, tuple)):
            y_mask: list[bool] | None = [bool(v) for v in restore_y]
        else:
            y_mask = None
            restore_y_all = bool(restore_y)
        for index in range(n):
            xlim, ylim = limits[index]
            plot = self._plots[index]
            try:
                if restore_x:
                    plot.setXRange(xlim[0], xlim[1], padding=0)
                do_y = (
                    y_mask[index]
                    if y_mask is not None and index < len(y_mask)
                    else (y_mask is None and restore_y_all)
                )
                if do_y and np.isfinite(ylim).all() and ylim[0] != ylim[1]:
                    plot.setYRange(ylim[0], ylim[1], padding=0)
            except Exception:
                pass
        if restore_x and len(self._plots) > n and limits:
            x0, x1 = limits[0][0]
            for plot in self._plots[n:]:
                try:
                    plot.setXRange(x0, x1, padding=0)
                except Exception:
                    pass
        self._refresh_scale_bars()

    def restore_montage_view_limits(
        self,
        limits: dict[tuple[int, str], tuple[tuple[float, float], tuple[float, float]]] | None,
        *,
        restore_x: bool = True,
        restore_y: bool | list[bool] | tuple[bool, ...] = True,
    ) -> None:
        if not limits:
            return
        stored = self.figure._erg_montage_state
        if not isinstance(stored, dict):
            return
        row_keys = list(stored.get("row_keys") or [])
        if not row_keys or len(row_keys) != len(self._plots):
            return
        if isinstance(restore_y, (list, tuple)):
            y_mask: list[bool] | None = [bool(v) for v in restore_y]
        else:
            y_mask = None
            restore_y_all = bool(restore_y)
        for index, (key, plot) in enumerate(zip(row_keys, self._plots)):
            if not (isinstance(key, (tuple, list)) and len(key) >= 2):
                continue
            saved = limits.get((int(key[0]), str(key[1])))
            if saved is None:
                continue
            xlim, ylim = saved
            try:
                if restore_x:
                    plot.setXRange(xlim[0], xlim[1], padding=0)
                do_y = (
                    y_mask[index]
                    if y_mask is not None and index < len(y_mask)
                    else (y_mask is None and restore_y_all)
                )
                if do_y and np.isfinite(ylim).all() and ylim[0] != ylim[1]:
                    plot.setYRange(ylim[0], ylim[1], padding=0)
            except Exception:
                pass
        self._refresh_scale_bars()

    def reset_view(self) -> None:
        if self._spec is not None:
            self._apply_limits(self._spec)
            for plot in self._plots:
                if self._spec.xlim is None:
                    plot.enableAutoRange(axis="x")
                # Y: re-enable auto when no manual ylim for that row.
            self._refresh_scale_bars()
            self.draw_idle()
            return
        for plot in self._plots:
            plot.enableAutoRange()
        self._refresh_scale_bars()
        self.draw_idle()

    # ---------------------------------------------------------------- draw

    def _clear_plots(self) -> None:
        self._disconnect_scale_bar_range()
        self._clear_scale_bar_items()
        self._scale_bar_lengths = []
        self._glw.clear()
        self._plots.clear()
        self._text_items.clear()
        self._row_labels.clear()
        self._editable.clear()
        self.figure.clear()

    def _style_plot(self, plot: pg.PlotItem, *, montage: bool = False) -> None:
        c = self._colors
        plot.setMenuEnabled(False)
        plot.showGrid(x=False, y=False)
        plot.getViewBox().setBackgroundColor(c["axes"])
        for axis_name in ("left", "bottom", "right", "top"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(c["spine"], width=0.8))
            axis.setTextPen(pg.mkPen(c["fg"]))
        font = QFont()
        font.setPointSize(8 if montage else 9)
        plot.getAxis("left").setStyle(tickFont=font)
        plot.getAxis("bottom").setStyle(tickFont=font)
        if montage:
            plot.getAxis("bottom").setStyle(showValues=False)
            # Pas de ylabel vertical pyqtgraph — libellés horizontaux en colonne dédiée.
            plot.getAxis("left").setStyle(showValues=True)
            try:
                plot.getAxis("left").showLabel(False)
            except Exception:
                pass

    def _configure_mouse(self, plot: pg.PlotItem) -> None:
        vb = plot.getViewBox()
        # Left button free for range bars; pan/zoom handled in eventFilter.
        vb.setMouseEnabled(x=False, y=False)
        vb.setMenuEnabled(False)

    def _install_interactions(self) -> None:
        self._pan_last = None
        self._pan_vb: pg.ViewBox | None = None
        self._glw.viewport().installEventFilter(self)

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QMouseEvent, QWheelEvent

        if obj is not self._glw.viewport():
            return super().eventFilter(obj, event)

        et = event.type()
        if et == QEvent.Type.Wheel and isinstance(event, QWheelEvent):
            if not self._allow_zoom:
                return False
            if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                return False  # let parent scroll
            plot = self._plot_at(event.position().toPoint())
            if plot is None:
                return False
            delta = event.angleDelta().y()
            if delta == 0:
                return False
            scale = 0.85 if delta > 0 else 1.15
            vb = plot.getViewBox()
            try:
                scene_pt = self._glw.mapToScene(event.position().toPoint())
                center = vb.mapSceneToView(scene_pt)
                vb.scaleBy((scale, scale), center=center)
            except Exception:
                vb.scaleBy((scale, scale))
            self._refresh_scale_bars()
            self.draw_idle()
            self.viewChanged.emit()
            return True

        if et == QEvent.Type.MouseButtonDblClick and isinstance(event, QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                if self._text_hit_at(event.position().toPoint()) is not None:
                    return True  # avoid reset when finishing a text click
                self.reset_view()
                return True

        if et == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                hit = self._text_hit_at(event.position().toPoint())
                if hit is not None:
                    self._start_text_edit(hit)
                    return True
            if event.button() in (
                Qt.MouseButton.MiddleButton,
                Qt.MouseButton.RightButton,
            ):
                plot = self._plot_at(event.position().toPoint())
                if plot is None:
                    return False
                self._pan_vb = plot.getViewBox()
                self._pan_last = event.position()
                return True

        if et == QEvent.Type.MouseMove and isinstance(event, QMouseEvent):
            if self._pan_vb is None or self._pan_last is None:
                return False
            if not (
                event.buttons()
                & (Qt.MouseButton.MiddleButton | Qt.MouseButton.RightButton)
            ):
                return False
            vb = self._pan_vb
            last = self._pan_last
            self._pan_last = event.position()
            try:
                p1 = vb.mapSceneToView(self._glw.mapToScene(last.toPoint()))
                p2 = vb.mapSceneToView(
                    self._glw.mapToScene(event.position().toPoint())
                )
                dx = float(p1.x() - p2.x())
                dy = float(p1.y() - p2.y())
                (x0, x1), (y0, y1) = vb.viewRange()
                vb.setRange(
                    xRange=(x0 + dx, x1 + dx),
                    yRange=(y0 + dy, y1 + dy),
                    padding=0,
                )
                self._refresh_scale_bars()
                self.draw_idle()
            except Exception:
                pass
            return True

        if et == QEvent.Type.MouseButtonRelease and isinstance(event, QMouseEvent):
            if self._pan_vb is not None:
                self._pan_vb = None
                self._pan_last = None
                self.viewChanged.emit()
                return True

        return super().eventFilter(obj, event)

    def _plot_at(self, pos) -> pg.PlotItem | None:
        if not self._plots:
            return None
        try:
            scene_pos = self._glw.mapToScene(pos)
        except Exception:
            return self._plots[0]
        for plot in self._plots:
            try:
                if plot.sceneBoundingRect().contains(scene_pos):
                    return plot
            except Exception:
                continue
        return self._plots[0]

    def _draw_placeholder(self, spec: PanelSpec) -> None:
        plot = self._glw.addPlot(row=0, col=0)
        self._style_plot(plot)
        self._configure_mouse(plot)
        self._plots = [plot]
        self.figure.axes = [_AxisShim(plot)]
        msg = spec.status_message or spec.title or "—"
        text = pg.TextItem(msg, color="#64748b", anchor=(0.5, 0.5))
        text.setFont(QFont("Segoe UI", 9))
        plot.addItem(text)
        text.setPos(0.5, 0.5)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setXRange(0, 1, padding=0)
        plot.setYRange(0, 1, padding=0)
        self._text_items.append(text)

    def _draw_series(self, spec: PanelSpec) -> list[tuple[str, str]]:
        """Draw series; return unique ``(label, color)`` for a below-graph legend."""
        legend_entries: list[tuple[str, str]] = []
        seen_labels: set[str] = set()
        for item in spec.series:
            row = int(getattr(item, "row", 0) or 0)
            if row < 0 or row >= len(self._plots):
                row = 0
            plot = self._plots[row]
            label = str(getattr(item, "label", "") or "").strip()
            if (
                label
                and label not in seen_labels
                and label != "_nolegend_"
                and isinstance(item, (CurveSeries, ScatterSeries, HistSeries))
            ):
                seen_labels.add(label)
                legend_entries.append((label, str(getattr(item, "color", "#334155"))))
            if isinstance(item, CurveSeries):
                if item.x.size == 0 or item.y.size == 0:
                    continue
                pen = pg.mkPen(item.color, width=max(0.5, float(item.linewidth)))
                # Pas de ``name`` : la légende est placée sous les graphs, pas dans le plot.
                plot.plot(
                    np.asarray(item.x, dtype=np.float64),
                    np.asarray(item.y, dtype=np.float64),
                    pen=pen,
                    connect="finite",
                )
            elif isinstance(item, ScatterSeries):
                if item.x.size == 0:
                    continue
                symbol = item.symbol if item.symbol in {"o", "t", "s", "d", "+"} else "o"
                # "|" → vertical ticks via short line segments
                if item.symbol == "|":
                    xs = np.asarray(item.x, dtype=np.float64)
                    ys = np.asarray(item.y, dtype=np.float64)
                    # Cap dense rasters for interactivity.
                    if xs.size > 8000:
                        step = max(1, xs.size // 8000)
                        xs, ys = xs[::step], ys[::step]
                    for x, y in zip(xs, ys):
                        plot.plot(
                            [x, x],
                            [y - 0.35, y + 0.35],
                            pen=pg.mkPen(item.color, width=1.0),
                        )
                else:
                    plot.plot(
                        np.asarray(item.x, dtype=np.float64),
                        np.asarray(item.y, dtype=np.float64),
                        pen=None,
                        symbol=symbol,
                        symbolSize=max(2, float(item.size)),
                        symbolBrush=pg.mkBrush(item.color),
                        symbolPen=pg.mkPen(item.color),
                    )
            elif isinstance(item, HistSeries):
                edges = np.asarray(item.edges, dtype=np.float64)
                counts = np.asarray(item.counts, dtype=np.float64)
                if edges.size < 2 or counts.size == 0:
                    continue
                centers = 0.5 * (edges[:-1] + edges[1:])
                width = float(np.median(np.diff(edges))) if edges.size > 2 else 1.0
                color = pg.mkColor(item.color)
                color.setAlpha(160)
                bar = pg.BarGraphItem(
                    x=centers,
                    height=counts,
                    width=width * 0.9,
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen(item.color),
                )
                plot.addItem(bar)

        # Row highlight + grid
        for row, plot in enumerate(self._plots):
            if row < len(spec.row_highlight) and spec.row_highlight[row]:
                plot.getViewBox().setBackgroundColor(_CHANNEL_HIGHLIGHT)
            if spec.grid:
                plot.showGrid(x=True, y=True, alpha=float(spec.grid_alpha))
        return legend_entries

    def _draw_annotations(self, spec: PanelSpec) -> None:
        for ann in spec.annotations:
            row = int(ann.row or 0)
            if row < 0 or row >= len(self._plots):
                row = 0
            plot = self._plots[row]
            if ann.kind == "vline":
                line = pg.InfiniteLine(
                    pos=float(ann.x),
                    angle=90,
                    pen=pg.mkPen(ann.color, width=1.0, style=Qt.PenStyle.DashLine),
                )
                plot.addItem(line)
            elif ann.kind == "hline":
                line = pg.InfiniteLine(
                    pos=float(ann.y),
                    angle=0,
                    pen=pg.mkPen(ann.color, width=1.0, style=Qt.PenStyle.DotLine),
                )
                plot.addItem(line)
            elif ann.kind == "span":
                region = pg.LinearRegionItem(
                    values=(float(ann.x), float(ann.x2)),
                    brush=pg.mkBrush(ann.color + "30"),
                    movable=False,
                )
                plot.addItem(region)
            elif ann.kind == "text":
                text = pg.TextItem(ann.text or "", color=ann.color, anchor=(0.5, 0.5))
                plot.addItem(text)
                # Place in view center after limits applied — use viewbox center later.
                text.setPos(0.0, 0.0)
                self._text_items.append(text)

    def _apply_limits(self, spec: PanelSpec) -> None:
        for row, plot in enumerate(self._plots):
            if spec.xlim is not None:
                plot.setXRange(float(spec.xlim[0]), float(spec.xlim[1]), padding=0)
            else:
                plot.enableAutoRange(axis=pg.ViewBox.XAxis)
            ylim = None
            if row < len(spec.row_ylims):
                ylim = spec.row_ylims[row]
            if ylim is None and spec.ylim is not None:
                ylim = spec.ylim
            if ylim is not None and ylim[0] is not None and ylim[1] is not None:
                plot.setYRange(float(ylim[0]), float(ylim[1]), padding=0)
            else:
                plot.enableAutoRange(axis=pg.ViewBox.YAxis)
                # PSTH: keep y ≥ 0
                if row < len(spec.row_kinds) and str(spec.row_kinds[row]) == "psth":
                    try:
                        (_x0, _x1), (y0, y1) = plot.viewRange()
                        plot.setYRange(0.0, max(float(y1), 1.0), padding=0)
                    except Exception:
                        pass

    def _apply_labels(self, spec: PanelSpec, *, is_montage: bool = False) -> None:
        c = self._colors
        if self._plots:
            self._plots[0].setTitle(spec.title or "", color=c["title"], size="10pt")
            title_item = getattr(self._plots[0], "titleLabel", None)
            if title_item is not None:
                self._editable.append(
                    _EditableText(title_item, "title", axes_index=0, item_index=0)
                )
        for row, plot in enumerate(self._plots):
            if not is_montage:
                ylabel = ""
                if row < len(spec.row_labels) and spec.row_labels[row]:
                    ylabel = spec.row_labels[row]
                elif row == 0:
                    ylabel = spec.ylabel or ""
                if ylabel:
                    plot.setLabel("left", ylabel, color=c["fg"])
                    axis = plot.getAxis("left")
                    label_item = getattr(axis, "label", None)
                    if label_item is not None:
                        self._editable.append(
                            _EditableText(
                                label_item, "ylabel", axes_index=row, item_index=0
                            )
                        )
            if row == len(self._plots) - 1:
                plot.getAxis("bottom").setStyle(showValues=True)
                if spec.xlabel:
                    plot.setLabel("bottom", spec.xlabel, color=c["fg"])
                    axis = plot.getAxis("bottom")
                    label_item = getattr(axis, "label", None)
                    if label_item is not None:
                        self._editable.append(
                            _EditableText(
                                label_item, "xlabel", axes_index=row, item_index=0
                            )
                        )
            else:
                plot.getAxis("bottom").setStyle(showValues=False)

        # Center placeholder-like text annotations that had pos 0,0
        for text in self._text_items:
            try:
                parent = text.getViewBox()
                if parent is not None:
                    (x0, x1), (y0, y1) = parent.viewRange()
                    text.setPos(0.5 * (x0 + x1), 0.5 * (y0 + y1))
            except Exception:
                pass

    def _add_montage_row_labels(
        self, spec: PanelSpec, *, n_rows: int, label_col: int
    ) -> None:
        """Libellés de canal horizontaux (un LabelItem par ligne), éditables."""
        c = self._colors
        font = QFont("Segoe UI", 8)
        metrics = QFontMetrics(font)
        max_w = 48
        for row in range(n_rows):
            text = ""
            if row < len(spec.row_labels) and spec.row_labels[row]:
                text = str(spec.row_labels[row])
            max_w = max(max_w, int(metrics.horizontalAdvance(text)) + 10)
            label = pg.LabelItem(
                text or " ",
                justify="right",
                color=c["fg"],
                size=_MONTAGE_LABEL_SIZE,
            )
            self._glw.addItem(label, row=row, col=label_col)
            self._row_labels.append(label)
            self._editable.append(
                _EditableText(label, "ylabel", axes_index=row, item_index=0)
            )
        try:
            self._glw.ci.layout.setColumnFixedWidth(label_col, max_w)
            self._glw.ci.layout.setColumnStretchFactor(label_col, 0)
            self._glw.ci.layout.setColumnStretchFactor(label_col + 1, 1)
        except Exception:
            pass

    def _place_legend_below(
        self,
        entries: list[tuple[str, str]],
        *,
        n_rows: int,
        plot_col: int,
    ) -> None:
        """Légende sous les graphs (pas en overlay sur les courbes)."""
        if not entries:
            return
        c = self._colors
        legend = pg.LegendItem(
            offset=(0, 0),
            horSpacing=8,
            verSpacing=2,
            labelTextSize="8pt",
            labelTextColor=c["fg"],
            colCount=1,
            frame=True,
            brush=pg.mkBrush(255, 255, 255, 235),
            pen=pg.mkPen(c["spine"], width=0.8),
        )
        for name, color in entries:
            sample = pg.PlotDataItem(pen=pg.mkPen(color, width=2.0))
            legend.addItem(sample, name)
        self._glw.addItem(legend, row=n_rows, col=plot_col)
        try:
            self._glw.ci.layout.setRowStretchFactor(n_rows, 0)
        except Exception:
            pass
        # Entrées de légende éditables unitairement.
        for index, (_sample, label_item) in enumerate(list(getattr(legend, "items", []) or [])):
            self._editable.append(
                _EditableText(label_item, "legend", axes_index=0, item_index=index)
            )

    # ----------------------------------------------------------- scale bars

    def _disconnect_scale_bar_range(self) -> None:
        for vb in self._scale_bar_range_connected:
            try:
                vb.sigRangeChanged.disconnect(self._on_scale_bar_range_changed)
            except (TypeError, RuntimeError):
                pass
        self._scale_bar_range_connected.clear()

    def _clear_scale_bar_items(self) -> None:
        for plot, item in self._scale_bar_items:
            try:
                plot.removeItem(item)
            except Exception:
                try:
                    vb = plot.getViewBox()
                    vb.removeItem(item)
                except Exception:
                    pass
        self._scale_bar_items.clear()

    def _apply_scale_bars(self, spec: PanelSpec, *, is_montage: bool) -> None:
        bars = getattr(spec, "scale_bars", None)
        if bars is None or not bool(getattr(bars, "enabled", False)):
            return
        if not self._plots:
            return
        # Masquer graduations / libellés d’unités (style montage EEG).
        for row, plot in enumerate(self._plots):
            plot.getAxis("left").setStyle(showValues=False)
            plot.getAxis("bottom").setStyle(showValues=False)
            for axis_name in ("left", "bottom", "right", "top"):
                try:
                    plot.getAxis(axis_name).setPen(pg.mkPen(None))
                except Exception:
                    pass
            if not is_montage:
                try:
                    plot.getAxis("left").setLabel("")
                except Exception:
                    pass
            if row == len(self._plots) - 1:
                try:
                    plot.getAxis("bottom").setLabel("")
                except Exception:
                    pass
        self._scale_bar_lengths = self._resolve_scale_bar_lengths(bars)
        self._connect_scale_bar_range()
        self._refresh_scale_bars()

    def _resolve_scale_bar_lengths(
        self, bars: ScaleBarSpec
    ) -> list[tuple[float | None, float | None]]:
        """Longueurs X/Y en unités de données, une fois par rendu."""
        from plot_utils import resolve_scale_bar_length

        out: list[tuple[float | None, float | None]] = []
        n = len(self._plots)
        multi = n > 1
        for row, plot in enumerate(self._plots):
            try:
                (x0, x1), (y0, y1) = plot.viewRange()
            except Exception:
                out.append((None, None))
                continue
            x_span = abs(float(x1) - float(x0))
            y_span = abs(float(y1) - float(y0))
            x_size: float | None = None
            y_size: float | None = None
            if bars.y_unit is not None and y_span > 0:
                y_size = resolve_scale_bar_length(
                    y_span,
                    manual=bool(bars.amp_manual),
                    value=float(bars.amplitude),
                )
            show_x = bars.x_unit is not None and (row == n - 1 or not multi)
            if show_x and x_span > 0:
                manual_x = float(bars.time_s)
                if bars.x_unit == "ms":
                    manual_x *= 1000.0
                x_size = resolve_scale_bar_length(
                    x_span,
                    manual=bool(bars.time_manual),
                    value=manual_x,
                )
            out.append((x_size, y_size))
        return out

    def _connect_scale_bar_range(self) -> None:
        self._disconnect_scale_bar_range()
        for plot in self._plots:
            vb = plot.getViewBox()
            try:
                vb.sigRangeChanged.connect(self._on_scale_bar_range_changed)
                self._scale_bar_range_connected.append(vb)
            except Exception:
                pass

    def _on_scale_bar_range_changed(self, *_args: Any) -> None:
        if self._scale_bars_refreshing:
            return
        self._refresh_scale_bars()

    def _refresh_scale_bars(self) -> None:
        spec = self._spec
        if spec is None:
            return
        bars = getattr(spec, "scale_bars", None)
        if bars is None or not bool(getattr(bars, "enabled", False)):
            return
        if not self._plots:
            return
        if self._scale_bars_refreshing:
            return
        self._scale_bars_refreshing = True
        try:
            self._clear_scale_bar_items()
            self._draw_scale_bar_artists(bars)
        finally:
            self._scale_bars_refreshing = False

    def _draw_scale_bar_artists(self, bars: ScaleBarSpec) -> None:
        from plot_utils import format_scale_label

        color = str(bars.color or "#111827")
        pen = pg.mkPen(color, width=float(bars.linewidth))
        font = QFont("Segoe UI", max(7, int(round(float(bars.fontsize)))))
        lengths = self._scale_bar_lengths
        if len(lengths) != len(self._plots):
            lengths = self._resolve_scale_bar_lengths(bars)
            self._scale_bar_lengths = lengths

        for row, plot in enumerate(self._plots):
            try:
                (x0, x1), (y0, y1) = plot.viewRange()
            except Exception:
                continue
            x_span = abs(float(x1) - float(x0))
            y_span = abs(float(y1) - float(y0))
            if x_span <= 0 or y_span <= 0:
                continue
            x_size, y_size = lengths[row] if row < len(lengths) else (None, None)

            # Ancrage bord droit / bas de la vue courante.
            x_edge = float(x1) - 0.02 * x_span
            y_mid = 0.5 * (float(y0) + float(y1))
            y_bot = float(y0) + 0.06 * y_span

            if y_size is not None and y_size > 0 and bars.y_unit is not None:
                y_a = y_mid - 0.5 * float(y_size)
                y_b = y_a + float(y_size)
                vline = pg.PlotDataItem(
                    [x_edge, x_edge],
                    [y_a, y_b],
                    pen=pen,
                    clipToView=False,
                )
                plot.addItem(vline, ignoreBounds=True)
                self._scale_bar_items.append((plot, vline))
                y_label = format_scale_label(float(y_size), str(bars.y_unit))
                y_text = pg.TextItem(
                    y_label, color=color, anchor=(0.0, 0.5), fill=None
                )
                y_text.setFont(font)
                try:
                    y_text.setClipToView(False)
                except Exception:
                    pass
                plot.addItem(y_text, ignoreBounds=True)
                y_text.setPos(x_edge + 0.01 * x_span, y_mid)
                self._scale_bar_items.append((plot, y_text))

            if x_size is not None and x_size > 0 and bars.x_unit is not None:
                x_right = x_edge
                x_left = x_right - float(x_size)
                if x_left < float(x0):
                    x_left = float(x0) + 0.02 * x_span
                    x_right = x_left + float(x_size)
                hline = pg.PlotDataItem(
                    [x_left, x_right],
                    [y_bot, y_bot],
                    pen=pen,
                    clipToView=False,
                )
                plot.addItem(hline, ignoreBounds=True)
                self._scale_bar_items.append((plot, hline))
                x_label = format_scale_label(float(x_size), str(bars.x_unit))
                x_text = pg.TextItem(
                    x_label, color=color, anchor=(0.5, 0.0), fill=None
                )
                x_text.setFont(font)
                try:
                    x_text.setClipToView(False)
                except Exception:
                    pass
                plot.addItem(x_text, ignoreBounds=True)
                x_text.setPos(0.5 * (x_left + x_right), y_bot - 0.02 * y_span)
                self._scale_bar_items.append((plot, x_text))

    # ----------------------------------------------------------- text editing

    def _reapply_text_pins(self) -> None:
        if not self._text_pins:
            return
        by_key = {edit.key: edit for edit in self._editable}
        for key, text in list(self._text_pins.items()):
            edit = by_key.get(key)
            if edit is None:
                continue
            try:
                edit.set_text(text)
            except Exception:
                pass

    def _text_hit_at(self, pos: QPoint) -> _EditableText | None:
        if not self._editable:
            return None
        try:
            scene_pos = self._glw.mapToScene(pos)
        except Exception:
            return None
        try:
            scene = self._glw.scene()
            items = scene.items(scene_pos) if scene is not None else []
        except Exception:
            items = []
        item_ids = {id(it) for it in items}
        # Inclure les parents (LabelItem contient un QGraphicsTextItem enfant).
        for it in list(items):
            parent = it
            for _ in range(6):
                parent = parent.parentItem() if parent is not None else None
                if parent is None:
                    break
                item_ids.add(id(parent))
        for edit in self._editable:
            if id(edit.item) in item_ids:
                return edit
            # Match enfant texte d’un LabelItem
            if isinstance(edit.item, pg.LabelItem):
                child = getattr(edit.item, "item", None)
                if child is not None and id(child) in item_ids:
                    return edit
        # Fallback géométrique (marges de clic plus larges).
        for edit in self._editable:
            try:
                rect = edit.item.sceneBoundingRect().adjusted(-4, -4, 4, 4)
                if rect.contains(scene_pos):
                    return edit
            except Exception:
                continue
        return None

    def _start_text_edit(self, hit: _EditableText) -> None:
        self._cancel_text_edit()
        edit = QLineEdit(self)
        edit.setText(hit.get_text())
        edit.setStyleSheet(
            "QLineEdit { background: #fffbeb; border: 1px solid #d97706; "
            "padding: 1px 4px; font-size: 9pt; }"
        )
        try:
            rect = hit.item.sceneBoundingRect()
            top_left = self._glw.mapFromScene(rect.topLeft())
            bottom_right = self._glw.mapFromScene(rect.bottomRight())
            # mapFromScene → coords du viewport GLW ; convertir en coords PlotHost.
            glw_origin = self._glw.mapTo(self, QPoint(0, 0))
            x = glw_origin.x() + min(top_left.x(), bottom_right.x()) - 2
            y = glw_origin.y() + min(top_left.y(), bottom_right.y()) - 2
            w = max(80, abs(bottom_right.x() - top_left.x()) + 8)
            h = max(22, abs(bottom_right.y() - top_left.y()) + 4)
            edit.setGeometry(int(x), int(y), int(w), int(h))
        except Exception:
            edit.setGeometry(40, 40, 220, 24)
        self._text_edit = edit
        self._text_hit = hit
        edit.returnPressed.connect(self._commit_text_edit)
        edit.editingFinished.connect(self._on_text_edit_finished)
        edit.show()
        edit.setFocus(Qt.FocusReason.MouseFocusReason)
        edit.selectAll()

    def _on_text_edit_finished(self) -> None:
        if self._text_closing or self._text_edit is None:
            return
        self._commit_text_edit()

    def _commit_text_edit(self) -> None:
        if self._text_closing or self._text_edit is None or self._text_hit is None:
            return
        self._text_closing = True
        try:
            new_text = self._text_edit.text()
            hit = self._text_hit
            self._close_text_editor()
            self._text_pins[hit.key] = new_text
            try:
                hit.set_text(new_text)
            except Exception:
                pass
            self.draw_idle()
            self.textEdited.emit(
                {
                    "role": hit.role,
                    "text": new_text,
                    "axes_index": hit.axes_index,
                    "item_index": hit.item_index,
                }
            )
        finally:
            self._text_closing = False

    def _cancel_text_edit(self) -> None:
        if self._text_closing:
            return
        self._text_closing = True
        try:
            self._close_text_editor()
        finally:
            self._text_closing = False

    def _close_text_editor(self) -> None:
        edit = self._text_edit
        self._text_edit = None
        self._text_hit = None
        if edit is None:
            return
        try:
            edit.returnPressed.disconnect(self._commit_text_edit)
        except (TypeError, RuntimeError):
            pass
        try:
            edit.editingFinished.disconnect(self._on_text_edit_finished)
        except (TypeError, RuntimeError):
            pass
        edit.hide()
        edit.deleteLater()
