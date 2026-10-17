"""Pyqtgraph host for screen panels (Phase 3).

On-screen rendering host (pyqtgraph). Matplotlib
remains the PDF / export drawing path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontInfo,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QTextCharFormat,
    QTextCursor,
    QTransform,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsSimpleTextItem,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from panel_prepare import (
    CurveSeries,
    HistSeries,
    PanelSpec,
    ScaleBarSpec,
    ScatterSeries,
    montage_vline_span_keys,
    montage_vline_spans,
)
from plot_utils import preview_title_row_layout


from .constants import (
    INTERACTION_HINT, INTERACTION_HINT_NO_ZOOM, _CHANNEL_HIGHLIGHT, _COALESCE_MS,
    _MONTAGE_AXIS_WIDTH_MAX, _MONTAGE_AXIS_WIDTH_MIN, _MONTAGE_DATA_BOTTOM_AXIS_PX,
    _MONTAGE_DATA_ROW_SPACING_PX, _MONTAGE_STREAM_KINDS, _MONTAGE_X_AXIS_HEIGHT_PX,
    _SCALE_BAR_BOTTOM_GUTTER_PX, _SCALE_BAR_PAD_PX, _SCALE_BAR_RIGHT_GUTTER_PX,
)
from .helpers import (
    _montage_label_column_extra, _montage_label_shift,
    montage_shares_x as _montage_shares_x,
    scale_bar_offset_px as _scale_bar_offset_px,
    scale_bar_pair_px as _scale_bar_pair_px,
    scale_bar_time_offset_px as _scale_bar_time_offset_px,
    uncrop_scale_axis as _uncrop_scale_axis,
)
from .mixins import (
    MontageMixin, ScaleBarsMixin, StimZoneMixin, TextEditMixin, ViewLimitsMixin,
)
from .montage_items import _MontageRowLabel, _MontageStimSpanItem
from .shims import _AxisShim, _FigureShim
from .text_model import TextRole, _EditableText, _StimLabelItem, _TextPinKey, _ZoneLabelItem
from .text_style import (
    _apply_axis_label_style, _apply_label_face, _editor_font, _editor_stylesheet,
    _face_qfont, _scale_label_item, _scope_colors, _stamp_text_face, _tick_qfont,
)

# Tick-face painters must stay in this module (offscreen AV if in text_style).
def _tick_text_top_left(
    rect: Any, flags: int, text: str, metrics: QFontMetrics
) -> tuple[float, float, float]:
    """``(left, top, advance)`` du libellé de tick dans ``rect``."""
    align_right = int(Qt.AlignmentFlag.AlignRight)
    align_hcenter = int(Qt.AlignmentFlag.AlignHCenter)
    align_vcenter = int(Qt.AlignmentFlag.AlignVCenter)
    align_bottom = int(Qt.AlignmentFlag.AlignBottom)
    fl = int(flags)
    advance = float(metrics.horizontalAdvance(text))
    height = float(metrics.height())
    if fl & align_vcenter:
        top = float(rect.center().y()) - height / 2.0
    elif fl & align_bottom:
        top = float(rect.bottom()) - height
    else:
        top = float(rect.top())
    if fl & align_right:
        left = float(rect.right()) - advance
    elif fl & align_hcenter:
        left = float(rect.center().x()) - advance / 2.0
    else:
        left = float(rect.left())
    return left, top, advance

def _draw_tick_labels_into_picture(
    axis: Any, painter: Any, text_specs: Any
) -> None:
    """Glyphes en contours dans le QPicture : gras / italique / souligné survivent.

    ``QPainter.drawText`` enregistré dans un ``QPicture`` perd le graisse et
    l'italique au replay. ``QPainterPath.addText`` bake la forme des glyphes.
    On ne monkey-patch **pas** ``paint`` (vtable C++ / PySide peu fiable) :
    seul ``drawPicture``, appelé depuis le ``paint`` Python de pyqtgraph.
    """
    if not text_specs:
        return
    style = getattr(axis, "style", None)
    font = style.get("tickFont") if isinstance(style, dict) else None
    if font is None:
        return
    draw_font = QFont(font)
    draw_font.setUnderline(False)
    metrics = QFontMetrics(draw_font)
    try:
        brush = QBrush(axis.textPen().color())
    except Exception:
        brush = QBrush(QColor("#111827"))
    want_underline = bool(font.underline())
    underline_pen: QPen | None = None
    if want_underline:
        underline_pen = QPen(brush.color())
        underline_pen.setWidthF(max(1.0, float(metrics.lineWidth())))
        underline_pen.setCosmetic(True)
    descent = max(1, int(metrics.underlinePos())) if want_underline else 0
    painter.save()
    try:
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        except Exception:
            pass
        try:
            painter.setClipRect(axis.boundingRect())
        except Exception:
            pass
        for rect, flags, text in text_specs:
            shown = str(text or "")
            if not shown:
                continue
            left, top, advance = _tick_text_top_left(
                rect, int(flags), shown, metrics
            )
            baseline = top + float(metrics.ascent())
            path = QPainterPath()
            path.addText(QPointF(left, baseline), draw_font, shown)
            painter.fillPath(path, brush)
            if underline_pen is not None:
                y = baseline + float(descent)
                painter.setPen(underline_pen)
                painter.drawLine(QPointF(left, y), QPointF(left + advance, y))
    finally:
        painter.restore()

def _ensure_tick_face_painter(axis: Any) -> None:
    """Enregistrer gras / italique / souligné des ticks dans le QPicture d'axe."""
    if getattr(axis, "_erg_tick_face_picture", False):
        return
    original_draw = axis.drawPicture

    def draw_picture(
        painter: Any, axis_spec: Any, tick_specs: Any, text_specs: Any
    ) -> None:
        # Traits d'axe sans drawText (sinon le QPicture perd la face).
        original_draw(painter, axis_spec, tick_specs, [])
        _draw_tick_labels_into_picture(axis, painter, text_specs)

    axis.drawPicture = draw_picture
    axis._erg_tick_face_picture = True
    # Ancien flag éventuel (wrap paint) : ne plus s'y fier.
    axis._erg_tick_underline = True

def _apply_plot_tick_font(
    plot: pg.PlotItem, spec: PanelSpec | None, *, montage: bool = False
) -> None:
    """Gras / italique / souligné des graduations, y compris après un refresh."""
    tick_pt = 8.0 if montage else 9.0
    bold = italic = underline = False
    if spec is not None:
        tick_pt = float(getattr(spec, "tick_font_size", tick_pt) or tick_pt)
        bold = bool(getattr(spec, "tick_bold", False))
        italic = bool(getattr(spec, "tick_italic", False))
        underline = bool(getattr(spec, "tick_underline", False))
    font = _tick_qfont(tick_pt, bold=bold, italic=italic, underline=underline)
    for name in ("left", "bottom", "top", "right"):
        axis = plot.getAxis(name)
        _ensure_tick_face_painter(axis)
        axis.setStyle(tickFont=font)


def _relax_label_width(label: Any) -> None:
    """Ne pas laisser un texte imposer la largeur du graphique.

    ``LabelItem.updateMin`` fige la largeur minimale sur le texte. Le layout
    pyqtgraph refuse alors de rétrécir : le widget (échelle temps) devient
    plus petit que le graphique, et la courbe est coupée.
    """
    if label is None or not hasattr(label, "updateMin"):
        return
    if not getattr(label, "_erg_width_relaxed", False):

        def update_min(*_args: Any, **_kwargs: Any) -> None:
            try:
                height = float(label.itemRect().height())
            except Exception:
                height = float(label.minimumHeight() or 0.0)
            label.setMinimumWidth(0)
            label.setPreferredWidth(0)
            label.setMinimumHeight(height)
            label._sizeHint = {
                Qt.SizeHint.MinimumSize: (0.0, height),
                Qt.SizeHint.PreferredSize: (0.0, height),
                Qt.SizeHint.MaximumSize: (-1.0, -1.0),
                Qt.SizeHint.MinimumDescent: (0.0, 0.0),
            }
            try:
                label.updateGeometry()
            except Exception:
                pass

        label.updateMin = update_min  # type: ignore[method-assign]
        label._erg_width_relaxed = True
    try:
        label.updateMin()
    except Exception:
        pass


class PlotHost(QWidget):
    """QWidget wrapping ``GraphicsLayoutWidget`` with coalesce + view API.

    Mixin methods (montage, scale bars, stim/zone, view limits, text edit) live
    in ``gui.widgets.plot_host.mixins`` and are attached below via ``setattr``.
    They are intentionally *not* placed in the class MRO: multiple inheritance
    with Shiboken ``QWidget`` triggers access violations in pyqtgraph paint
    under ``QT_QPA_PLATFORM=offscreen`` (multi-panel smoke).
    """

    viewChanged = Signal()
    textEdited = Signal(object)
    # Émis après double-clic / ⌂ : ``(x0, x1)`` partagé à propager aux peers.
    xAutoscaleRequested = Signal(object)
    # Après autoscale : le parent adapte hauteur / largeur (échelle amp. / temps).
    scaleBarHeightFitRequested = Signal()
    scaleBarWidthFitRequested = Signal()

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
        del figsize, layout  # accepted for legacy callers; unused by pyqtgraph host
        super().__init__(parent)
        self._allow_zoom = bool(allow_zoom)
        self._colors = _scope_colors()
        self._spec: PanelSpec | None = None
        self._plots: list[pg.PlotItem] = []
        self._text_items: list[pg.TextItem] = []
        self._stim_labels: list[_StimLabelItem] = []
        self._zone_labels: list[_ZoneLabelItem] = []
        self._row_labels: list[pg.LabelItem] = []
        self._montage_title_label: pg.LabelItem | None = None
        # Axe X partagé sous les courbes montage (ticks + xlabel hors ViewBox).
        self._montage_x_axis_plot: pg.PlotItem | None = None
        # Index de grille de la 1ʳᵉ ligne de données (1 si titre montage dédié).
        self._data_row0 = 0
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
        self._geometry_key: tuple[Any, ...] | None = None
        self._legend_item: Any | None = None
        self._cached_spec: PanelSpec | None = None
        self._structure_key: tuple[Any, ...] | None = None
        self._table: QTableWidget | None = None

        # AA global = très coûteux au scroll / pan ; activé au cas par cas sur les courbes.
        pg.setConfigOptions(antialias=False, foreground=self._colors["fg"])

        self.figure = _FigureShim(self)
        self.canvas = self  # grab() / attach callers expect a QWidget
        self.toolbar: QWidget | None = None

        self._glw = pg.GraphicsLayoutWidget(self)
        self._glw.setBackground(self._colors["bg"])
        self._glw.setMinimumHeight(int(min_height))
        self._glw.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        # Un titre plus large que la zone de tracé ne doit pas figer le layout.
        try:
            self._glw.ci.setMinimumWidth(0)
        except Exception:
            pass
        # Paint ciblé : évite de redessiner toute la scène à chaque cran de scroll.
        try:
            from PySide6.QtWidgets import QGraphicsView

            self._glw.setViewportUpdateMode(
                QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate
            )
            self._glw.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        except Exception:
            pass
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

        self._draw_timer = QTimer(self)
        self._draw_timer.setSingleShot(True)
        self._draw_timer.setInterval(_COALESCE_MS)
        self._draw_timer.timeout.connect(self._flush_draw)
        # Barres d’échelle : coalescer les refreshes (sinon freeze au pan/zoom).
        self._scale_bar_timer = QTimer(self)
        self._scale_bar_timer.setSingleShot(True)
        self._scale_bar_timer.setInterval(40)
        self._scale_bar_timer.timeout.connect(self._refresh_scale_bars)
        # Traits de stim continus : recalcul après pan / zoom / redimensionnement.
        self._stim_span_item: _MontageStimSpanItem | None = None
        self._stim_span_specs: list[tuple[float, str, tuple[int, ...]]] = []
        self._stim_span_width = 1.0
        self._stim_span_connected: list[Any] = []
        self._stim_span_timer = QTimer(self)
        self._stim_span_timer.setSingleShot(True)
        self._stim_span_timer.setInterval(16)
        self._stim_span_timer.timeout.connect(self._update_stim_spans)

        # Limites appliquées après le premier paint (courbes déjà à l’écran).
        self._autoscale_gen = 0
        self._pending_xlim: tuple[float, float] | None = None
        self._pending_auto_x = False
        self._pending_ylims: list[tuple[float, float] | None] = []
        self._pending_auto_y: list[bool] = []
        self._limits_after_paint = False

        self._install_interactions()

    # ---------------------------------------------------------------- API


    def draw_idle(self) -> None:
        if not self._draw_timer.isActive():
            self._draw_timer.start()

    def _flush_draw(self) -> None:
        try:
            self._glw.update()
        except Exception:
            pass
        if self._limits_after_paint:
            # Après mise à jour scène : les courbes sont affichables → autoscale.
            gen = self._autoscale_gen
            QTimer.singleShot(0, lambda g=gen: self._apply_limits_after_paint(g))

    def _hide_table(self) -> None:
        if self._table is not None:
            self._table.hide()
        self._glw.show()

    def _show_table(self, spec: PanelSpec) -> None:
        self._glw.hide()
        if self._table is None:
            self._table = QTableWidget(self)
            self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self._table.setAlternatingRowColors(True)
            layout = self.layout()
            if layout is not None:
                layout.insertWidget(1, self._table, 1)
        headers = [str(h) for h in spec.table_headers]
        rows = [list(row) for row in spec.table_rows]
        self._table.clear()
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()
        highlight = int(getattr(spec, "table_highlight_row", -1) or -1)
        if 0 <= highlight < len(rows):
            self._table.selectRow(highlight)
            current = self._table.item(highlight, 0)
            if current is not None:
                self._table.scrollToItem(current)
        title = str(spec.title or "").strip()
        self._cursor_label.setText(title or "Table")
        self._table.show()

    def _apply_x_ticks(self, spec: PanelSpec | None) -> None:
        ticks = list(getattr(spec, "x_ticks", ()) or []) if spec is not None else []
        if not ticks or not self._plots:
            return
        target = self._montage_x_axis_plot or self._plots[0]
        try:
            target.getAxis("bottom").setTicks([ticks])
        except Exception:
            pass


    @staticmethod
    def _geometry_key_for(spec: PanelSpec) -> tuple[Any, ...]:
        """Identity of plot layout — same key → reuse PlotItems (no glw.clear).

        Montage : on ignore les ``row_keys`` (indices de canaux) pour qu’un
        changement de page réutilise la grille au lieu de tout reconstruire.
        """
        layout = str(spec.layout_kind or "single")
        if layout == "montage":
            # Motif de kinds par canal (ex. raw,hp) — stable entre pages.
            kinds = tuple(spec.row_kinds) if spec.row_kinds else ()
            unit: tuple[str, ...] = ()
            if kinds:
                # Déduire la période (kinds répétés par canal).
                for period in range(1, len(kinds) + 1):
                    if len(kinds) % period == 0 and kinds == kinds[:period] * (
                        len(kinds) // period
                    ):
                        unit = kinds[:period]
                        break
                if not unit:
                    unit = kinds
            return (
                layout,
                unit,
                bool(spec.show_legend),
                str(spec.legend_location or "below"),
                bool(getattr(spec.scale_bars, "enabled", False)),
            )
        return (
            layout,
            max(1, int(spec.n_rows or 1)),
            tuple(spec.row_keys) if spec.row_keys else (),
            tuple(spec.row_kinds) if spec.row_kinds else (),
            int(len(spec.series)),
            bool(spec.show_legend),
            str(spec.legend_location or "below"),
            bool(getattr(spec.scale_bars, "enabled", False)),
        )

    def render_spec(self, spec: PanelSpec) -> str:
        self._cancel_text_edit()
        self._spec = spec
        self._cached_spec = spec
        if spec.layout_kind == "table" and spec.status == "ok":
            self._clear_plots()
            self._geometry_key = ("table", tuple(spec.table_headers), len(spec.table_rows))
            self._show_table(spec)
            self.figure._erg_status = spec.status
            return spec.status
        self._hide_table()
        placeholder = spec.layout_kind == "placeholder" or (
            spec.status in {"unavailable", "pending", "empty"} and not spec.series
        )
        if placeholder:
            self._clear_plots()
            self._geometry_key = None
            self._draw_placeholder(spec)
            self.figure._erg_status = spec.status
            self.draw_idle()
            return spec.status

        geo = self._geometry_key_for(spec)
        n_rows = max(1, int(spec.n_rows or 1))
        is_montage = spec.layout_kind == "montage"

        # Montage page suivante : même structure → resize éventuel + refresh données.
        if (
            is_montage
            and geo == self._geometry_key
            and self._plots
        ):
            if len(self._plots) != n_rows:
                self._resize_montage_rows(spec, n_rows=n_rows)
            if len(self._plots) == n_rows and self._plots:
                return self._refresh_spec(spec)

        if (
            not is_montage
            and geo == self._geometry_key
            and len(self._plots) == n_rows
            and self._plots
        ):
            return self._refresh_spec(spec)

        self._clear_plots()
        self._geometry_key = geo
        label_col = 0 if is_montage else -1
        plot_col = 1 if is_montage else 0
        self._plot_col = plot_col
        self._plots = []
        self._row_labels = []
        self.figure.axes = []
        # Titre montage hors du 1er PlotItem : sinon la ligne 0 est plus haute
        # que les libellés et tout le bloc paraît décalé.
        self._data_row0 = 1 if is_montage else 0
        if is_montage:
            self._add_montage_title_row(spec, label_col=label_col, plot_col=plot_col)
            self._add_montage_row_labels(spec, n_rows=n_rows, label_col=label_col)

        for row in range(n_rows):
            grid_row = row + self._data_row0
            plot = self._glw.addPlot(row=grid_row, col=plot_col)
            self._style_plot(plot, montage=is_montage, spec=spec)
            self._configure_mouse(plot)
            if is_montage:
                # Placeholder ; largeur réelle calculée après les ylim (uniforme).
                plot.getAxis("left").setWidth(_MONTAGE_AXIS_WIDTH_MIN)
            self._plots.append(plot)
            self.figure.axes.append(_AxisShim(plot))
            if row > 0 and (not is_montage or _montage_shares_x(spec)):
                plot.setXLink(self._plots[0])
            try:
                self._glw.ci.layout.setRowStretchFactor(grid_row, 1)
            except Exception:
                pass

        if is_montage:
            self._bind_montage_row_label_anchors()
        return self._finish_draw(spec, is_montage=is_montage, n_rows=n_rows)








    def _refresh_spec(self, spec: PanelSpec) -> str:
        """Reuse existing PlotItems — clear data items only, then redraw."""
        is_montage = spec.layout_kind == "montage"
        n_rows = max(1, int(spec.n_rows or 1))
        # Conserver le zoom X entre pages de montage (mêmes bornes temporelles).
        prev_xlim: tuple[float, float] | None = None
        if is_montage and self._plots:
            try:
                x0, x1 = self._plots[0].viewRange()[0]
                if np.isfinite([x0, x1]).all() and float(x0) != float(x1):
                    prev_xlim = (float(x0), float(x1))
            except Exception:
                prev_xlim = None
        self._disconnect_scale_bar_range()
        self._clear_scale_bar_items()
        self._scale_bar_lengths = []
        self._remove_legend()
        self._text_items.clear()
        self._clear_stim_labels()
        self._clear_zone_labels()
        self._editable.clear()
        # Keep montage row labels / axis shims; re-register edit targets.
        if is_montage:
            self._update_montage_row_labels(spec)
            for row, label in enumerate(self._row_labels):
                self._editable.append(
                    _EditableText(label, "ylabel", axes_index=row, item_index=0)
                )
        c = self._colors
        for row, plot in enumerate(self._plots):
            try:
                plot.clear()
            except Exception:
                pass
            # Highlight / fond : pas de re-style axes complet (coûteux à chaque page).
            if row < len(spec.row_highlight) and spec.row_highlight[row]:
                plot.getViewBox().setBackgroundColor(_CHANNEL_HIGHLIGHT)
            else:
                plot.getViewBox().setBackgroundColor(c["axes"])
            if not is_montage:
                self._style_plot(plot, montage=False, spec=spec)
            else:
                # Pas de reconstruction d’axes, mais bordures, ticks et police
                # doivent suivre les réglages d’affichage.
                self._apply_axis_frame(plot, montage=True, spec=spec)
                if spec.grid:
                    plot.showGrid(x=True, y=True, alpha=float(spec.grid_alpha))
                else:
                    plot.showGrid(x=False, y=False)
        status = self._finish_draw(spec, is_montage=is_montage, n_rows=n_rows)
        # Zoom X partagé entre pages : seulement si les lignes partagent le même X.
        if (
            prev_xlim is not None
            and spec.xlim is None
            and (not is_montage or _montage_shares_x(spec))
            and not any(rx is not None for rx in (spec.row_xlims or ()))
        ):
            self._pending_xlim = prev_xlim
            self._pending_auto_x = False
        return status

    def _finish_draw(
        self, spec: PanelSpec, *, is_montage: bool, n_rows: int
    ) -> str:
        legend_entries = self._draw_series(spec)
        self._draw_annotations(spec)
        seen_labels = {label for label, _color in legend_entries}
        for ann in spec.annotations:
            ann_label = str(getattr(ann, "label", "") or "").strip()
            if ann.kind == "hline" and ann_label and ann_label not in seen_labels:
                seen_labels.add(ann_label)
                legend_entries.append((ann_label, str(ann.color)))
        # Autoscale / limites après affichage des courbes (pas pendant le build).
        self._queue_limits_after_paint(spec)
        self._apply_labels(spec, is_montage=is_montage)
        if is_montage:
            # Axe X hors stack avant légende / barres : sinon la dernière
            # ligne grossit et l’espacement ViewBox n’est plus uniforme.
            self._collapse_montage_data_bottom_axes()
            self._ensure_montage_x_axis_row(
                spec, n_rows=n_rows, plot_col=self._plot_col
            )
        self._apply_scale_bars(spec, is_montage=is_montage)
        if spec.show_legend and legend_entries:
            self._place_legend(
                legend_entries,
                spec=spec,
                n_rows=n_rows,
                plot_col=self._plot_col,
            )
        self._reapply_text_pins()
        if is_montage:
            self._apply_montage_row_minimums(spec)
            self._bind_montage_row_label_anchors()
            # Après layout (gouttières échelle / légende) : recentrer les libellés.
            QTimer.singleShot(0, self._bind_montage_row_label_anchors)

        # Montage metadata for view restore / range bars.
        if spec.row_keys:
            self.figure._erg_montage_state = {
                "kind": "continuous_review",
                "row_keys": list(spec.row_keys),
            }
        self.figure._erg_montage_row_kinds = (
            list(spec.row_kinds) if spec.row_kinds else None
        )
        self.figure._erg_montage_row_streams = (
            [
                kind if kind in _MONTAGE_STREAM_KINDS else "raw"
                for kind in spec.row_kinds
            ]
            if spec.row_kinds
            else None
        )
        self.figure._erg_montage_row_channels = (
            list(spec.row_channels) if spec.row_channels else None
        )
        self.figure._erg_status = spec.status
        self.draw_idle()
        if is_montage and n_rows > 4:
            # Après layout : masquer les lignes hors viewport scroll parent.
            QTimer.singleShot(0, self.cull_rows_outside_scroll_viewport)
        return spec.status



    def _remove_legend(self) -> None:
        legend = self._legend_item
        self._legend_item = None
        if legend is None:
            return
        self._discard_layout_item(legend)
        try:
            legend.close()
        except Exception:
            pass

    def plot_items(self) -> list[pg.PlotItem]:
        return list(self._plots)











    # ---------------------------------------------------------------- draw

    def _qt_graphics_at(self, row: int, col: int) -> Any | None:
        """Item Qt réellement posé en (row, col), même si le suivi pyqtgraph diverge."""
        layout = self._glw.ci.layout
        try:
            row_count = int(layout.rowCount())
            col_count = int(layout.columnCount())
        except Exception:
            return None
        # Hors grille, itemAt() logue « invalid row, column » et renvoie null.
        if row < 0 or col < 0 or row >= row_count or col >= col_count:
            return None
        try:
            layout_item = layout.itemAt(int(row), int(col))
        except TypeError:
            return None
        except Exception:
            return None
        if layout_item is None:
            return None
        try:
            return layout_item.graphicsItem()
        except Exception:
            return layout_item

    def _free_grid_cell(self, row: int, col: int) -> None:
        """Libérer une cellule avant d’y poser un nouvel item."""
        occupants: list[Any] = []
        graphic = self._qt_graphics_at(row, col)
        if graphic is not None:
            occupants.append(graphic)
        try:
            tracked = self._glw.getItem(int(row), int(col))
        except Exception:
            tracked = None
        if tracked is not None and all(tracked is not item for item in occupants):
            occupants.append(tracked)
        for item in occupants:
            self._discard_layout_item(item)

    def _discard_layout_item(self, item: Any) -> None:
        """Retirer un item même quand ``GraphicsLayout.removeItem`` ne trouve plus son index."""
        if item is None:
            return
        ci = self._glw.ci
        try:
            self._glw.removeItem(item)
            return
        except Exception:
            pass
        layout = ci.layout
        if hasattr(layout, "count"):
            try:
                index = 0
                while index < layout.count():
                    layout_item = layout.itemAt(index)
                    graphic = None
                    try:
                        graphic = (
                            layout_item.graphicsItem()
                            if layout_item is not None
                            else None
                        )
                    except Exception:
                        graphic = None
                    if layout_item is item or graphic is item:
                        layout.removeAt(index)
                        continue
                    index += 1
            except Exception:
                pass
        try:
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
        except Exception:
            pass
        for row, col in list(ci.items.get(item, ())):
            row_map = ci.rows.get(row)
            if row_map is not None and row_map.get(col) is item:
                del row_map[col]
        ci.items.pop(item, None)
        # PySide6: disconnect déjà fait → RuntimeWarning, pas une exception.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                item.geometryChanged.disconnect(ci._updateItemBorder)
            except Exception:
                pass
        border = ci.itemBorders.pop(item, None)
        if border is not None:
            try:
                border_scene = border.scene()
                if border_scene is not None:
                    border_scene.removeItem(border)
            except Exception:
                pass

    def _force_clear_layout(self) -> bool:
        """Vider la grille après un ``clear()`` interrompu. False si Qt garde des cellules."""
        ci = self._glw.ci
        for item in list(ci.items.keys()):
            self._discard_layout_item(item)
        layout = ci.layout
        if hasattr(layout, "count"):
            try:
                guard = int(layout.count()) + 1
                while layout.count() and guard:
                    guard -= 1
                    before = int(layout.count())
                    layout_item = layout.itemAt(0)
                    graphic = None
                    try:
                        graphic = (
                            layout_item.graphicsItem()
                            if layout_item is not None
                            else None
                        )
                    except Exception:
                        graphic = None
                    layout.removeAt(0)
                    if graphic is not None:
                        try:
                            scene = graphic.scene()
                            if scene is not None:
                                scene.removeItem(graphic)
                        except Exception:
                            pass
                    if int(layout.count()) >= before:
                        break
            except Exception:
                pass
        for border in list(ci.itemBorders.values()):
            try:
                scene = border.scene()
                if scene is not None:
                    scene.removeItem(border)
            except Exception:
                pass
        ci.items.clear()
        ci.rows.clear()
        ci.itemBorders.clear()
        ci.currentRow = 0
        ci.currentCol = 0
        remaining = 0
        if hasattr(layout, "count"):
            try:
                remaining = int(layout.count())
            except Exception:
                remaining = 0
        return remaining == 0 and not ci.items

    def _replace_graphics_layout(self) -> None:
        """Remplacer la grille quand QGraphicsGridLayout garde des cellules fantômes."""
        glw = self._glw
        new = pg.GraphicsLayout()
        for name in (
            "nextRow",
            "nextCol",
            "nextColumn",
            "addPlot",
            "addViewBox",
            "addItem",
            "getItem",
            "addLayout",
            "addLabel",
            "removeItem",
            "itemIndex",
            "clear",
        ):
            setattr(glw, name, getattr(new, name))
        glw.ci = new
        glw.setCentralItem(new)

    def _empty_graphics_layout(self) -> None:
        try:
            self._glw.clear()
        except Exception:
            if not self._force_clear_layout():
                self._replace_graphics_layout()
            return
        ci = self._glw.ci
        leftover = bool(ci.items)
        if not leftover and hasattr(ci.layout, "count"):
            try:
                leftover = int(ci.layout.count()) > 0
            except Exception:
                leftover = False
        if leftover and not self._force_clear_layout():
            self._replace_graphics_layout()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        if getattr(self, "_stim_span_specs", None):
            self._schedule_stim_span_update()

    def _clear_plots(self) -> None:
        self._limits_after_paint = False
        self._autoscale_gen += 1
        self._disconnect_scale_bar_range()
        self._clear_stim_spans()
        self._clear_scale_bar_items()
        self._scale_bar_lengths = []
        self._legend_item = None
        self._clear_stim_labels()
        self._clear_zone_labels()
        self._empty_graphics_layout()
        self._plots.clear()
        self._text_items.clear()
        self._row_labels.clear()
        self._montage_title_label = None
        self._montage_x_axis_plot = None
        self._data_row0 = 0
        self._editable.clear()
        self._geometry_key = None
        self.figure.clear()

    def _style_plot(
        self,
        plot: pg.PlotItem,
        *,
        montage: bool = False,
        spec: PanelSpec | None = None,
    ) -> None:
        c = self._colors
        plot.setMenuEnabled(False)
        plot.showGrid(x=False, y=False)
        vb = plot.getViewBox()
        vb.setBackgroundColor(c["axes"])
        # Empêche l’autoscale pyqtgraph à l’ajout des PlotDataItem.
        try:
            vb.enableAutoRange(enable=False)
            vb.setDefaultPadding(0.0)
        except Exception:
            pass
        if spec is not None and bool(getattr(spec, "y_log", False)):
            try:
                plot.setLogMode(x=False, y=True)
            except Exception:
                pass
        self._apply_axis_frame(plot, montage=montage, spec=spec)

    def _apply_axis_frame(
        self,
        plot: pg.PlotItem,
        *,
        montage: bool,
        spec: PanelSpec | None,
    ) -> None:
        """Bordures, sens des graduations et police des ticks."""
        c = self._colors
        show_borders = True if spec is None else bool(spec.show_borders)
        ticks_inside = False if spec is None else bool(spec.ticks_inside)
        spine_pen = (
            pg.mkPen(c["spine"], width=0.8)
            if show_borders
            else pg.mkPen(None)
        )
        for axis_name in ("left", "bottom", "right", "top"):
            axis = plot.getAxis(axis_name)
            axis.setPen(spine_pen)
            axis.setTextPen(pg.mkPen(c["fg"]))
            try:
                axis.setStyle(tickLength=(-6 if ticks_inside else 6))
            except Exception:
                pass
        _apply_plot_tick_font(plot, spec, montage=montage)
        try:
            plot.showAxis("top", show=bool(ticks_inside and show_borders))
            plot.showAxis("right", show=bool(ticks_inside and show_borders))
            if ticks_inside and show_borders:
                plot.getAxis("top").setStyle(showValues=False)
                plot.getAxis("right").setStyle(showValues=False)
        except Exception:
            pass
        if montage:
            plot.getAxis("bottom").setStyle(showValues=False)
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
            self._schedule_scale_bar_refresh()
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
                self._schedule_scale_bar_refresh()
                self.draw_idle()
            except Exception:
                pass
            return True

        if et == QEvent.Type.MouseButtonRelease and isinstance(event, QMouseEvent):
            if self._pan_vb is not None:
                self._pan_vb = None
                self._pan_last = None
                # Frame final net après pan coalescé.
                self._refresh_scale_bars()
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
        self._style_plot(plot, spec=spec)
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

    @staticmethod
    def _tune_plot_data_item(curve: Any, *, dense: bool = True) -> None:
        """clipToView + downsample auto : paint bien plus léger au scroll/pan/zoom."""
        if curve is None:
            return
        try:
            curve.setClipToView(True)
        except Exception:
            pass
        if dense:
            try:
                curve.setDownsampling(auto=True, method="peak")
            except Exception:
                try:
                    curve.setDownsampling(ds=True, auto=True, method="peak")
                except Exception:
                    pass
        try:
            # Évite un second passage NaN coûteux (données déjà nettoyées côté prepare).
            curve.setSkipFiniteCheck(True)
        except Exception:
            pass

    def _draw_series(self, spec: PanelSpec) -> list[tuple[str, str]]:
        """Draw series; return unique ``(label, color)`` for a below-graph legend."""
        legend_entries: list[tuple[str, str]] = []
        seen_labels: set[str] = set()
        # AA seulement sur un graphe simple ; montages / multi-axes = trop cher au scroll.
        use_aa = len(self._plots) <= 1 and str(spec.layout_kind or "") != "montage"
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
                width = max(0.5, float(item.linewidth))
                raw_alpha = getattr(item, "alpha", 1.0)
                alpha = 1.0 if raw_alpha is None else float(raw_alpha)
                alpha = max(0.0, min(1.0, alpha))
                if alpha < 0.999:
                    qcolor = pg.mkColor(item.color)
                    qcolor.setAlphaF(alpha)
                    pen = pg.mkPen(qcolor, width=width)
                else:
                    pen = pg.mkPen(item.color, width=width)
                # Pas de ``name`` : la légende est placée sous les graphs, pas dans le plot.
                # float32 : moins de bande passante GPU/CPU au paint.
                curve = plot.plot(
                    np.asarray(item.x, dtype=np.float32),
                    np.asarray(item.y, dtype=np.float32),
                    pen=pen,
                    connect="finite",
                    antialias=use_aa,
                )
                try:
                    curve.setZValue(4 if alpha >= 0.999 else 1)
                except Exception:
                    pass
                self._tune_plot_data_item(curve, dense=True)
            elif isinstance(item, ScatterSeries):
                if item.x.size == 0:
                    continue
                symbol = item.symbol if item.symbol in {"o", "t", "s", "d", "+"} else "o"
                # "|" → un seul path (connect=pairs) au lieu d’un plot() par spike.
                if item.symbol == "|":
                    xs = np.asarray(item.x, dtype=np.float32)
                    ys = np.asarray(item.y, dtype=np.float32)
                    # Cap dense rasters for interactivity.
                    if xs.size > 8000:
                        step = max(1, int(xs.size // 8000))
                        xs, ys = xs[::step], ys[::step]
                    n = int(xs.size)
                    if n == 0:
                        continue
                    x_seg = np.repeat(xs, 2)
                    y_seg = np.empty(n * 2, dtype=np.float32)
                    y_seg[0::2] = ys - 0.35
                    y_seg[1::2] = ys + 0.35
                    curve = plot.plot(
                        x_seg,
                        y_seg,
                        pen=pg.mkPen(item.color, width=1.0),
                        connect="pairs",
                        antialias=False,
                    )
                    self._tune_plot_data_item(curve, dense=True)
                else:
                    curve = plot.plot(
                        np.asarray(item.x, dtype=np.float32),
                        np.asarray(item.y, dtype=np.float32),
                        pen=None,
                        symbol=symbol,
                        symbolSize=max(2, float(item.size)),
                        symbolBrush=pg.mkBrush(item.color),
                        symbolPen=pg.mkPen(item.color),
                    )
                    self._tune_plot_data_item(curve, dense=False)
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
            else:
                plot.showGrid(x=False, y=False)
        return legend_entries

    def _draw_annotations(self, spec: PanelSpec) -> None:
        spans = (
            montage_vline_spans(spec) if spec.layout_kind == "montage" else []
        )
        covered = montage_vline_span_keys(spans)
        self._stim_span_specs = spans
        stim_width = max(0.2, float(getattr(spec, "stim_linewidth", 1.0) or 1.0))
        self._stim_span_width = stim_width
        for ann in spec.annotations:
            row = int(ann.row or 0)
            if row < 0 or row >= len(self._plots):
                row = 0
            plot = self._plots[row]
            if ann.kind == "vline":
                x_key = round(float(ann.x), 6)
                if (row, x_key, str(ann.color or "")) in covered:
                    continue
                line = pg.InfiniteLine(
                    pos=float(ann.x),
                    angle=90,
                    pen=pg.mkPen(ann.color, width=stim_width, style=Qt.PenStyle.DashLine),
                )
                plot.addItem(line)
            elif ann.kind == "hline":
                style_name = str(getattr(ann, "linestyle", ":") or ":")
                if style_name in {"--", "dashed"}:
                    pen_style = Qt.PenStyle.DashLine
                elif style_name in {"-", "solid"}:
                    pen_style = Qt.PenStyle.SolidLine
                else:
                    pen_style = Qt.PenStyle.DotLine
                line = pg.InfiniteLine(
                    pos=float(ann.y),
                    angle=0,
                    pen=pg.mkPen(ann.color, width=1.2, style=pen_style),
                )
                plot.addItem(line)
            elif ann.kind == "span":
                region = pg.LinearRegionItem(
                    values=(float(ann.x), float(ann.x2)),
                    brush=pg.mkBrush(ann.color + "30"),
                    movable=False,
                )
                plot.addItem(region)
            elif ann.kind == "stim_label":
                self._add_stim_label(plot, ann, row=row)
            elif ann.kind == "zone_label":
                self._add_zone_label_from_ann(ann)
            elif ann.kind == "text":
                text = pg.TextItem(ann.text or "", color=ann.color, anchor=(0.5, 0.5))
                plot.addItem(text)
                # Place in view center after limits applied — use viewbox center later.
                text.setPos(0.0, 0.0)
                self._text_items.append(text)
        self._arm_stim_spans()
        self._reposition_stim_labels()
        if self._zone_labels:
            self._reposition_zone_labels()


























    def _apply_labels(self, spec: PanelSpec, *, is_montage: bool = False) -> None:
        c = self._colors
        title_pt = max(5, int(round(float(spec.title_font_size or 10.0))))
        label_pt = max(5, int(round(float(spec.label_font_size or 9.0))))
        title_size = f"{title_pt}pt"
        if is_montage and self._montage_title_label is not None:
            # Titre sur ligne dédiée — pas dans le 1er PlotItem.
            # ``setTitle('')`` laisse 30 px vides ; ``None`` masque vraiment.
            if self._plots:
                try:
                    self._plots[0].setTitle(None)
                except Exception:
                    pass
            title_item = self._montage_title_label
            try:
                title_item.setText(
                    spec.title or " ", size=title_size, color=c["title"]
                )
            except Exception:
                pass
            _apply_label_face(
                title_item,
                bold=bool(spec.title_bold),
                italic=bool(spec.title_italic),
                underline=bool(getattr(spec, "title_underline", False)),
            )
            self._editable.append(
                _EditableText(title_item, "title", axes_index=0, item_index=0)
            )
            self._apply_montage_title_row_height(spec)
            self._apply_title_pixel_offset(title_item, spec)
            _relax_label_width(title_item)
        elif self._plots:
            self._plots[0].setTitle(
                spec.title or "", color=c["title"], size=title_size
            )
            title_item = getattr(self._plots[0], "titleLabel", None)
            if title_item is not None:
                _apply_label_face(
                    title_item,
                    bold=bool(spec.title_bold),
                    italic=bool(spec.title_italic),
                    underline=bool(getattr(spec, "title_underline", False)),
                )
                self._editable.append(
                    _EditableText(title_item, "title", axes_index=0, item_index=0)
                )
            self._apply_preview_title_gap(self._plots[0], spec)
            if title_item is not None:
                self._apply_title_pixel_offset(title_item, spec)
                _relax_label_width(title_item)
        for row, plot in enumerate(self._plots):
            if not is_montage:
                ylabel = ""
                if row < len(spec.row_labels) and spec.row_labels[row]:
                    ylabel = spec.row_labels[row]
                elif row == 0:
                    ylabel = spec.ylabel or ""
                if ylabel:
                    plot.setLabel(
                        "left",
                        ylabel,
                        color=c["fg"],
                        **{"font-size": f"{label_pt}pt"},
                    )
                    axis = plot.getAxis("left")
                    _apply_label_face(
                        getattr(axis, "label", None),
                        bold=bool(spec.label_bold),
                        italic=bool(spec.label_italic),
                        underline=bool(getattr(spec, "label_underline", False)),
                    )
                    label_item = getattr(axis, "label", None)
                    if label_item is not None:
                        self._editable.append(
                            _EditableText(
                                label_item, "ylabel", axes_index=row, item_index=0
                            )
                        )
            is_last = row == len(self._plots) - 1
            bottom = plot.getAxis("bottom")
            if is_montage:
                # Xlabel / ticks sur la ligne d’axe partagée (ViewBox égales).
                self._clear_bottom_axis_label(bottom, collapse=True)
                continue
            if not is_last:
                bottom.setStyle(showValues=False)
                continue
            bottom.setStyle(showValues=True)
            if spec.xlabel:
                plot.setLabel(
                    "bottom",
                    spec.xlabel,
                    color=c["fg"],
                    **{"font-size": f"{label_pt}pt"},
                )
                _apply_label_face(
                    getattr(bottom, "label", None),
                    bold=bool(spec.label_bold),
                    italic=bool(spec.label_italic),
                    underline=bool(getattr(spec, "label_underline", False)),
                )
                label_item = getattr(bottom, "label", None)
                if label_item is not None:
                    self._editable.append(
                        _EditableText(
                            label_item, "xlabel", axes_index=row, item_index=0
                        )
                    )

        # Center placeholder-like text annotations that had pos 0,0
        for text in self._text_items:
            try:
                parent = text.getViewBox()
                if parent is not None:
                    (x0, x1), (y0, y1) = parent.viewRange()
                    text.setPos(0.5 * (x0 + x1), 0.5 * (y0 + y1))
            except Exception:
                pass
        self._reposition_stim_labels()
        self._release_plot_width_floor()

    def _release_plot_width_floor(self) -> None:
        """Oublier une largeur minimale héritée d’un titre ou d’une légende."""
        try:
            self._glw.ci.setMinimumWidth(0)
            self._glw.ci.updateGeometry()
        except Exception:
            pass
        for plot in self._plots:
            try:
                plot.setMinimumWidth(0)
                plot.updateGeometry()
            except Exception:
                pass

    def _apply_title_pixel_offset(self, title_item: Any, spec: PanelSpec) -> None:
        """Décaler le titre en pixels (+dx droite, +dy haut).

        Utilise une transform sur le ``LabelItem``. Les marges du layout
        (voir ``_apply_title_layout_margins``) réservent la place pour que le
        titre reste dans le widget.
        """
        if title_item is None:
            return
        dx = float(getattr(spec, "title_dx_px", 0.0) or 0.0)
        dy = float(getattr(spec, "title_dy_px", 0.0) or 0.0)
        try:
            title_item.setFlag(
                title_item.GraphicsItemFlag.ItemClipsChildrenToShape, False
            )
        except Exception:
            pass
        try:
            # Scène pyqtgraph : Y positif vers le bas → inverser dy.
            if dx or dy:
                title_item.setTransform(QTransform.fromTranslate(dx, -dy))
            else:
                title_item.resetTransform()
        except Exception:
            pass

    def _apply_preview_title_gap(self, plot: pg.PlotItem, spec: PanelSpec) -> None:
        """Écarter ou rapprocher le titre de la courbe (aperçu, hors montage)."""
        title = getattr(plot, "titleLabel", None)
        layout = getattr(plot, "layout", None)
        if title is None or layout is None or not title.isVisible():
            return
        text_h = 14.0
        try:
            text_h = max(1.0, float(title.itemRect().height()))
        except Exception:
            text_h = 14.0
        row_h, spacing = preview_title_row_layout(
            text_h,
            int(getattr(spec, "title_gap_px", 0) or 0),
            title_dx_px=float(getattr(spec, "title_dx_px", 0.0) or 0.0),
            title_dy_px=float(getattr(spec, "title_dy_px", 0.0) or 0.0),
        )
        try:
            title.setMaximumHeight(row_h)
            layout.setRowFixedHeight(0, row_h)
            layout.setRowSpacing(0, spacing)
        except Exception:
            pass
        # Marges titre sur le GLW (pas le PlotItem) : sinon transform + marge
        # interne s’annulent et le titre ne bouge pas à l’écran.
        self._apply_stim_label_band_margin(
            0 if not self._stim_labels else None
        )


    def _clear_bottom_axis_label(self, axis: Any, *, collapse: bool = True) -> None:
        """Retirer le titre d’axe X. ``collapse`` enlève aussi la hauteur réservée."""
        if collapse:
            try:
                axis.setStyle(showValues=False)
            except Exception:
                pass
        try:
            if str(getattr(axis, "labelText", "") or ""):
                axis.setLabel("")
            else:
                axis.showLabel(False)
        except Exception:
            try:
                axis.showLabel(False)
            except Exception:
                pass
        if collapse:
            try:
                axis.setHeight(None)
            except Exception:
                pass




    def _legend_overlay_offset(self, location: str, *, extra_px: int = 0) -> tuple[int, int]:
        """Offset pyqtgraph (positif = haut/gauche, négatif = bas/droite)."""
        loc = (location or "below").strip().lower()
        mapping = {
            "upper right": (-10, 10),
            "upper left": (10, 10),
            "lower right": (-10, -10),
            "lower left": (10, -10),
            "center right": (-10, 0),
            "center left": (10, 0),
            "best": (-10, 10),
        }
        dx, dy = mapping.get(loc, (-10, 10))
        extra = max(0, int(extra_px))
        if extra:
            if dx < 0:
                dx -= extra
            elif dx > 0:
                dx += extra
            if dy > 0:
                dy += extra
            elif dy < 0:
                dy -= extra
        return dx, dy

    @staticmethod
    def _legend_gap_px(spec: PanelSpec) -> int:
        """Écart légende ↔ graphique (fraction d’axes → pixels)."""
        return max(0, int(round(float(getattr(spec, "legend_gap", 0.0) or 0.0) * 160.0)))

    def _place_legend(
        self,
        entries: list[tuple[str, str]],
        *,
        spec: PanelSpec,
        n_rows: int,
        plot_col: int,
    ) -> None:
        """Légende sous le panneau (défaut) ou en overlay selon ``legend_location``."""
        if not entries:
            return
        c = self._colors
        font_pt = max(5, int(round(float(spec.legend_font_size or 9.0))))
        columns = max(1, int(spec.legend_columns or 1))
        frame = bool(spec.legend_frame)
        gap_px = self._legend_gap_px(spec)
        location = str(spec.legend_location or "below")
        below = location.strip().lower() in {"below", ""}
        self._remove_legend()
        legend = pg.LegendItem(
            offset=(0, 0) if below else self._legend_overlay_offset(location, extra_px=gap_px),
            horSpacing=8,
            verSpacing=4,
            labelTextSize=f"{font_pt}pt",
            labelTextColor=c["fg"],
            colCount=columns,
            frame=frame,
            brush=pg.mkBrush(255, 255, 255, 235) if frame else pg.mkBrush(None),
            pen=pg.mkPen(c["spine"], width=0.8) if frame else pg.mkPen(None),
        )
        self._legend_item = legend
        for name, color in entries:
            sample = pg.PlotDataItem(pen=pg.mkPen(color, width=2.0))
            legend.addItem(sample, name)
        if below or not self._plots:
            # Sous l’axe X partagé (montage) pour ne pas élargir la dernière courbe.
            footer = 1 if (
                str(spec.layout_kind or "") == "montage"
                and self._montage_x_axis_plot is not None
            ) else 0
            legend_row = int(n_rows) + int(self._data_row0) + footer
            self._free_grid_cell(legend_row, plot_col)
            self._glw.addItem(legend, row=legend_row, col=plot_col)
            try:
                self._glw.ci.layout.setRowStretchFactor(legend_row, 0)
                if legend_row > 0:
                    # Gap légende après l’axe X / dernière ligne — pas entre canaux.
                    self._glw.ci.layout.setRowSpacing(legend_row - 1, gap_px)
            except Exception:
                pass
        else:
            legend.setParentItem(self._plots[0])
            try:
                legend.anchor(
                    itemPos=(1, 0) if "right" in location else (0, 0),
                    parentPos=(1, 0) if "right" in location else (0, 0),
                    offset=self._legend_overlay_offset(location, extra_px=gap_px),
                )
            except Exception:
                pass
        # Gras / italique / souligné sur les libellés de légende.
        for _sample, label_item in list(getattr(legend, "items", []) or []):
            _apply_label_face(
                label_item,
                bold=bool(spec.legend_bold),
                italic=bool(spec.legend_italic),
                underline=bool(getattr(spec, "legend_underline", False)),
            )
            _relax_label_width(label_item)
        try:
            legend.setMinimumWidth(0)
            legend.setPreferredWidth(0)
            legend.updateGeometry()
        except Exception:
            pass
        self._release_plot_width_floor()
        # Entrées de légende éditables unitairement.
        for index, (_sample, label_item) in enumerate(
            list(getattr(legend, "items", []) or [])
        ):
            self._editable.append(
                _EditableText(label_item, "legend", axes_index=0, item_index=index)
            )

    # ----------------------------------------------------------- scale bars
















    # ----------------------------------------------------------- text editing



def _attach_mixin_methods() -> None:
    for mixin in (
        MontageMixin,
        ScaleBarsMixin,
        StimZoneMixin,
        ViewLimitsMixin,
        TextEditMixin,
    ):
        for name, value in mixin.__dict__.items():
            if name.startswith("__"):
                continue
            if callable(value):
                setattr(PlotHost, name, value)


_attach_mixin_methods()
