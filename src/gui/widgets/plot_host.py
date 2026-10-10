"""Pyqtgraph host for screen panels (Phase 3).

On-screen rendering host (pyqtgraph). Matplotlib
remains the PDF / export drawing path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPen
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsSimpleTextItem,
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
_MONTAGE_AXIS_WIDTH_MIN = 36  # px — largeur mini ticks Y (échelle X uniforme)
_MONTAGE_AXIS_WIDTH_MAX = 96
# Gouttières hors ViewBox : échelle Y à droite, X sous le dernier graph.
_SCALE_BAR_PAD_PX = 6
_SCALE_BAR_RIGHT_GUTTER_PX = 58
# Place pour bâton temps + libellé (+ xlabel éventuel) sous le dernier graph.
_SCALE_BAR_BOTTOM_GUTTER_PX = 48
_MONTAGE_STREAM_KINDS = frozenset({"raw", "hp", "lp"})


def _montage_shares_x(spec: PanelSpec) -> bool:
    """True si toutes les lignes montage partagent la même famille temporelle."""
    kinds = list(spec.row_kinds or ())
    if not kinds:
        return True
    return all(str(kind) in _MONTAGE_STREAM_KINDS for kind in kinds)

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


class _MontageRowLabel(pg.LabelItem):
    """Libellé de ligne montage : droite + centre vertical du ViewBox.

    ``LabelItem(justify='right')`` de pyqtgraph aligne le texte en haut de la
    cellule — d’où le décalage croissant par rapport aux graphes (centrés).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._anchor_plot: pg.PlotItem | None = None
        super().__init__(*args, **kwargs)

    def set_anchor_plot(self, plot: pg.PlotItem | None) -> None:
        self._anchor_plot = plot
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
        bounds.moveCenter(QPointF(rect.center().x(), cy))
        bounds.moveRight(rect.right())
        self.item.setPos(bounds.topLeft() - self.itemRect().topLeft())
        self.updateMin()


def _apply_label_face(
    item: Any,
    *,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
) -> None:
    """Gras / italique / souligné sur LabelItem ou TextItem pyqtgraph."""
    if item is None:
        return
    for method, value in (
        ("setBold", bold),
        ("setItalic", italic),
        ("setUnderline", underline),
    ):
        fn = getattr(item, method, None)
        if callable(fn):
            try:
                fn(bool(value))
            except Exception:
                pass
    try:
        opt = getattr(item, "opts", None)
        if isinstance(opt, dict):
            opt["bold"] = bool(bold)
            opt["italic"] = bool(italic)
            opt["underline"] = bool(underline)
    except Exception:
        pass
    # TextItem / axes : souligné via QFont si exposé.
    try:
        font = item.font() if callable(getattr(item, "font", None)) else None
        if font is not None:
            font.setBold(bool(bold))
            font.setItalic(bool(italic))
            font.setUnderline(bool(underline))
            item.setFont(font)
    except Exception:
        pass


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
    # Émis après double-clic / ⌂ : ``(x0, x1)`` partagé à propager aux peers.
    xAutoscaleRequested = Signal(object)

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
        self._row_labels: list[pg.LabelItem] = []
        self._montage_title_label: pg.LabelItem | None = None
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

        self._draw_pending = False
        self._draw_timer = QTimer(self)
        self._draw_timer.setSingleShot(True)
        self._draw_timer.setInterval(_COALESCE_MS)
        self._draw_timer.timeout.connect(self._flush_draw)
        # Barres d’échelle : coalescer les refreshes (sinon freeze au pan/zoom).
        self._scale_bar_timer = QTimer(self)
        self._scale_bar_timer.setSingleShot(True)
        self._scale_bar_timer.setInterval(40)
        self._scale_bar_timer.timeout.connect(self._refresh_scale_bars)

        # Limites appliquées après le premier paint (courbes déjà à l’écran).
        self._autoscale_gen = 0
        self._pending_xlim: tuple[float, float] | None = None
        self._pending_auto_x = False
        self._pending_ylims: list[tuple[float, float] | None] = []
        self._pending_auto_y: list[bool] = []
        self._limits_after_paint = False

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
        if self._limits_after_paint:
            # Après mise à jour scène : les courbes sont affichables → autoscale.
            gen = self._autoscale_gen
            QTimer.singleShot(0, lambda g=gen: self._apply_limits_after_paint(g))

    def render_request(self, request: Any) -> str:
        spec = prepare_panel(request)
        return self.render_spec(spec)

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

    def _bind_montage_row_label_anchors(self) -> None:
        """Lier chaque libellé au ViewBox de sa ligne (centrage vertical exact)."""
        for label, plot in zip(self._row_labels, self._plots):
            if isinstance(label, _MontageRowLabel):
                label.set_anchor_plot(plot)

    def _resize_montage_rows(self, spec: PanelSpec, *, n_rows: int) -> None:
        """Ajouter / retirer des lignes sans ``glw.clear`` (changement de page)."""
        plot_col = 1
        label_col = 0
        self._plot_col = plot_col
        self._data_row0 = 1
        # Trim
        while len(self._plots) > n_rows:
            plot = self._plots.pop()
            try:
                self._glw.removeItem(plot)
            except Exception:
                pass
            if self._row_labels:
                label = self._row_labels.pop()
                try:
                    self._glw.removeItem(label)
                except Exception:
                    pass
            if self.figure.axes:
                self.figure.axes.pop()
        # Extend
        while len(self._plots) < n_rows:
            row = len(self._plots)
            grid_row = row + self._data_row0
            text = ""
            if row < len(spec.row_labels) and spec.row_labels[row]:
                text = str(spec.row_labels[row])
            label_pt = max(5, int(round(float(spec.label_font_size or 8.0))))
            label = _MontageRowLabel(
                text or " ",
                justify="right",
                color=self._colors["fg"],
                size=f"{label_pt}pt",
            )
            self._glw.addItem(label, row=grid_row, col=label_col)
            self._row_labels.append(label)
            plot = self._glw.addPlot(row=grid_row, col=plot_col)
            self._style_plot(plot, montage=True, spec=spec)
            self._configure_mouse(plot)
            plot.getAxis("left").setWidth(_MONTAGE_AXIS_WIDTH_MIN)
            self._plots.append(plot)
            self.figure.axes.append(_AxisShim(plot))
            if row > 0 and _montage_shares_x(spec):
                plot.setXLink(self._plots[0])
            try:
                self._glw.ci.layout.setRowStretchFactor(grid_row, 1)
            except Exception:
                pass
        self._bind_montage_row_label_anchors()

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
            elif spec.grid:
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
        # Autoscale / limites après affichage des courbes (pas pendant le build).
        self._queue_limits_after_paint(spec)
        self._apply_labels(spec, is_montage=is_montage)
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

    def _update_montage_row_labels(self, spec: PanelSpec) -> None:
        """Refresh LabelItem texts without recreating the layout column."""
        c = self._colors
        label_pt = max(5, int(round(float(spec.label_font_size or 8.0))))
        size = f"{label_pt}pt"
        for row, label in enumerate(self._row_labels):
            text = ""
            if row < len(spec.row_labels) and spec.row_labels[row]:
                text = str(spec.row_labels[row])
            try:
                label.setText(text or " ")
                label.setAttr("size", size)
                label.setAttr("color", c["fg"])
            except Exception:
                pass
            try:
                _apply_label_face(
                    label,
                    bold=bool(spec.label_bold),
                    italic=bool(spec.label_italic),
                    underline=bool(getattr(spec, "label_underline", False)),
                )
            except Exception:
                pass

    def _remove_legend(self) -> None:
        legend = self._legend_item
        self._legend_item = None
        if legend is None:
            return
        try:
            self._glw.removeItem(legend)
        except Exception:
            try:
                legend.scene().removeItem(legend)
            except Exception:
                pass
        try:
            legend.close()
        except Exception:
            pass

    def plot_items(self) -> list[pg.PlotItem]:
        return list(self._plots)

    def cull_rows_outside_scroll_viewport(self, *, margin_px: int = 160) -> None:
        """Masquer les courbes des lignes hors du QScrollArea parent (montage haut).

        Les axes / labels restent en place (pas de reflow) ; seul le paint des
        ``PlotDataItem`` est sauté → scroll nettement plus fluide.
        """
        if len(self._plots) <= 4:
            return
        from PySide6.QtWidgets import QAbstractScrollArea

        scroll: QWidget | None = self.parentWidget()
        while scroll is not None and not isinstance(scroll, QAbstractScrollArea):
            scroll = scroll.parentWidget()
        if scroll is None:
            return
        vp = scroll.viewport()
        try:
            y0 = vp.mapToGlobal(vp.rect().topLeft()).y() - int(margin_px)
            y1 = vp.mapToGlobal(vp.rect().bottomRight()).y() + int(margin_px)
        except Exception:
            return
        for plot in self._plots:
            try:
                br = plot.sceneBoundingRect()
                top_left = self._glw.mapFromScene(br.topLeft())
                bot_right = self._glw.mapFromScene(br.bottomRight())
                top = self._glw.mapToGlobal(top_left).y()
                bot = self._glw.mapToGlobal(bot_right).y()
            except Exception:
                continue
            show = bot >= y0 and top <= y1
            try:
                for item in plot.listDataItems():
                    if item.isVisible() != show:
                        item.setVisible(show)
            except Exception:
                pass

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
                    if index < len(self._pending_auto_y):
                        self._pending_auto_y[index] = False
                        if index < len(self._pending_ylims):
                            self._pending_ylims[index] = None
            except Exception:
                pass
        if restore_x:
            self._pending_auto_x = False
            self._pending_xlim = None
            if len(self._plots) > n and limits:
                x0, x1 = limits[0][0]
                for plot in self._plots[n:]:
                    try:
                        plot.setXRange(x0, x1, padding=0)
                    except Exception:
                        pass
        if self._spec is not None and self._spec.layout_kind == "montage":
            self._fit_montage_left_axis_widths(self._spec)
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
        restored_any_x = False
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
                    restored_any_x = True
                do_y = (
                    y_mask[index]
                    if y_mask is not None and index < len(y_mask)
                    else (y_mask is None and restore_y_all)
                )
                if do_y and np.isfinite(ylim).all() and ylim[0] != ylim[1]:
                    plot.setYRange(ylim[0], ylim[1], padding=0)
                    if index < len(self._pending_auto_y):
                        self._pending_auto_y[index] = False
                        if index < len(self._pending_ylims):
                            self._pending_ylims[index] = None
            except Exception:
                pass
        # Changement de page : autres canaux, mêmes bornes temporelles.
        if restore_x and not restored_any_x:
            sample = next(iter(limits.values()), None)
            if sample is not None:
                xlim = sample[0]
                if np.isfinite(xlim).all() and xlim[0] != xlim[1]:
                    for plot in self._plots:
                        try:
                            plot.setXRange(xlim[0], xlim[1], padding=0)
                        except Exception:
                            pass
                    restored_any_x = True
        if restore_x and restored_any_x:
            self._pending_auto_x = False
            self._pending_xlim = None
        if self._spec is not None and self._spec.layout_kind == "montage":
            self._fit_montage_left_axis_widths(self._spec)
        self._refresh_scale_bars()

    def _show_all_data_items(self) -> None:
        """Réafficher toutes les courbes (après cull scroll)."""
        for plot in self._plots:
            try:
                for item in plot.listDataItems():
                    if not item.isVisible():
                        item.setVisible(True)
            except Exception:
                pass

    def reset_view(self) -> None:
        """Autoscale idempotent : une seule échelle X pour toutes les lignes liées."""
        self._show_all_data_items()
        # Retirer les barres d’échelle avant mesure (sinon elles dilatent la plage).
        self._clear_scale_bar_items()
        self._limits_after_paint = False
        self._autoscale_gen += 1
        xlim: tuple[float, float] | None
        if self._spec is not None:
            self._queue_limits_after_paint(self._spec)
            self._limits_after_paint = False
            xlim = self._resolve_shared_xlim(self._spec)
            self._pending_xlim = xlim
            self._pending_auto_x = False
            self._apply_pending_limits()
            if self._spec.layout_kind == "montage":
                self._fit_montage_left_axis_widths(self._spec)
            self._resync_scale_bar_lengths()
            self._refresh_scale_bars()
            self.draw_idle()
            self.xAutoscaleRequested.emit(xlim)
            return
        xlim = self._union_x_from_all_plots()
        if xlim is not None:
            for plot in self._plots:
                plot.setXRange(xlim[0], xlim[1], padding=0)
        for plot in self._plots:
            self._autoscale_y_from_data(plot)
        self._resync_scale_bar_lengths()
        self._refresh_scale_bars()
        self.draw_idle()
        self.xAutoscaleRequested.emit(xlim)

    def data_xlim(self) -> tuple[float, float] | None:
        """Étendue X des courbes affichées (hors barres d’échelle)."""
        if self._spec is not None and self._spec.xlim is not None:
            return (float(self._spec.xlim[0]), float(self._spec.xlim[1]))
        return self._union_x_from_all_plots()

    def apply_shared_xlim(self, xlim: tuple[float, float] | None) -> None:
        """Appliquer une échelle X externe (sync entre panels) sans retoucher Y."""
        if xlim is None or not self._plots:
            return
        x0, x1 = float(xlim[0]), float(xlim[1])
        if not (x1 > x0) or not np.isfinite([x0, x1]).all():
            return
        self._pending_auto_x = False
        self._pending_xlim = (x0, x1)
        for plot in self._plots:
            try:
                plot.setXRange(x0, x1, padding=0)
            except Exception:
                pass
        self._refresh_scale_bars()
        self.draw_idle()

    # ---------------------------------------------------------------- draw

    def _clear_plots(self) -> None:
        self._limits_after_paint = False
        self._autoscale_gen += 1
        self._disconnect_scale_bar_range()
        self._clear_scale_bar_items()
        self._scale_bar_lengths = []
        self._legend_item = None
        self._glw.clear()
        self._plots.clear()
        self._text_items.clear()
        self._row_labels.clear()
        self._montage_title_label = None
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
        show_borders = True if spec is None else bool(spec.show_borders)
        ticks_inside = False if spec is None else bool(spec.ticks_inside)
        tick_pt = 8 if montage else 9
        tick_bold = False
        tick_italic = False
        tick_underline = False
        if spec is not None:
            tick_pt = max(5, int(round(float(spec.tick_font_size or tick_pt))))
            tick_bold = bool(spec.tick_bold)
            tick_italic = bool(spec.tick_italic)
            tick_underline = bool(getattr(spec, "tick_underline", False))
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
        font = QFont()
        font.setPointSize(tick_pt)
        font.setBold(tick_bold)
        font.setItalic(tick_italic)
        font.setUnderline(tick_underline)
        plot.getAxis("left").setStyle(tickFont=font)
        plot.getAxis("bottom").setStyle(tickFont=font)
        # Masquer axes haut / droite sauf ticks_inside (repères intérieurs).
        try:
            plot.showAxis("top", show=bool(ticks_inside and show_borders))
            plot.showAxis("right", show=bool(ticks_inside and show_borders))
            if ticks_inside and show_borders:
                plot.getAxis("top").setStyle(showValues=False, tickFont=font)
                plot.getAxis("right").setStyle(showValues=False, tickFont=font)
        except Exception:
            pass
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
                pen = pg.mkPen(item.color, width=max(0.5, float(item.linewidth)))
                # Pas de ``name`` : la légende est placée sous les graphs, pas dans le plot.
                # float32 : moins de bande passante GPU/CPU au paint.
                curve = plot.plot(
                    np.asarray(item.x, dtype=np.float32),
                    np.asarray(item.y, dtype=np.float32),
                    pen=pen,
                    connect="finite",
                    antialias=use_aa,
                )
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

    def _scale_bar_item_ids(self) -> set[int]:
        return {id(item) for _plot, item in self._scale_bar_items}

    def _data_range_from_plot(
        self, plot: pg.PlotItem, *, axis: str
    ) -> tuple[float, float] | None:
        """Min/max des courbes/scatter (ignore vlines, barres d’échelle, légende)."""
        lo: float | None = None
        hi: float | None = None
        attr = "xData" if axis == "x" else "yData"
        skip = self._scale_bar_item_ids()
        try:
            items = list(plot.listDataItems())
        except Exception:
            items = []
        for item in items:
            if id(item) in skip:
                continue
            try:
                data = getattr(item, attr, None)
                if data is None:
                    continue
                arr = np.asarray(data, dtype=np.float64).ravel()
                if arr.size == 0:
                    continue
                # Légende / samples vides : ignorer.
                if arr.size <= 2 and not np.any(np.isfinite(arr)):
                    continue
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    continue
                a, b = float(np.min(finite)), float(np.max(finite))
            except Exception:
                continue
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
        if lo is None or hi is None or not (hi > lo):
            return None
        return (lo, hi)

    def _union_x_from_all_plots(self) -> tuple[float, float] | None:
        """Union X de toutes les lignes (échelle horizontale unique, stable)."""
        x_min: float | None = None
        x_max: float | None = None
        for plot in self._plots:
            span = self._data_range_from_plot(plot, axis="x")
            if span is None:
                continue
            x_min = span[0] if x_min is None else min(x_min, span[0])
            x_max = span[1] if x_max is None else max(x_max, span[1])
        if x_min is None or x_max is None or not (x_max > x_min):
            return None
        return (x_min, x_max)

    def _resolve_shared_xlim(self, spec: PanelSpec) -> tuple[float, float] | None:
        """xlim du spec si défini, sinon union des courbes affichées."""
        if spec.xlim is not None:
            return (float(spec.xlim[0]), float(spec.xlim[1]))
        return self._union_x_from_all_plots()

    def _autoscale_x_from_data(self, plot: pg.PlotItem) -> None:
        """Étendue X d’une ligne (préférer ``_resolve_shared_xlim`` si multi-lignes)."""
        span = self._data_range_from_plot(plot, axis="x")
        if span is not None:
            plot.setXRange(span[0], span[1], padding=0)
            return
        try:
            vb = plot.getViewBox()
            vb.setDefaultPadding(0.0)
            plot.enableAutoRange(axis=pg.ViewBox.XAxis)
        except Exception:
            pass

    def _autoscale_y_from_data(
        self, plot: pg.PlotItem, *, non_negative: bool = False
    ) -> None:
        """Étendue Y des courbes/scatter uniquement (padding=0)."""
        span = self._data_range_from_plot(plot, axis="y")
        if span is not None:
            y0, y1 = span
            if non_negative:
                y0 = 0.0
                y1 = max(y1, 1.0)
            plot.setYRange(y0, y1, padding=0)
            return
        try:
            vb = plot.getViewBox()
            vb.setDefaultPadding(0.0)
            plot.enableAutoRange(axis=pg.ViewBox.YAxis)
            if non_negative:
                (_x0, _x1), (y0, y1) = plot.viewRange()
                plot.setYRange(0.0, max(float(y1), 1.0), padding=0)
        except Exception:
            pass

    def _queue_limits_after_paint(self, spec: PanelSpec) -> None:
        """Mémorise les limites à appliquer après le premier paint des courbes."""
        self._autoscale_gen += 1
        self._pending_xlim = (
            (float(spec.xlim[0]), float(spec.xlim[1]))
            if spec.xlim is not None
            else None
        )
        self._pending_auto_x = spec.xlim is None
        pending_ylims: list[tuple[float, float] | None] = []
        pending_auto_y: list[bool] = []
        for row in range(len(self._plots)):
            ylim = None
            if row < len(spec.row_ylims):
                ylim = spec.row_ylims[row]
            if ylim is None and spec.ylim is not None:
                ylim = spec.ylim
            if ylim is not None and ylim[0] is not None and ylim[1] is not None:
                pending_ylims.append((float(ylim[0]), float(ylim[1])))
                pending_auto_y.append(False)
            else:
                pending_ylims.append(None)
                pending_auto_y.append(True)
        self._pending_ylims = pending_ylims
        self._pending_auto_y = pending_auto_y
        self._limits_after_paint = True

    def _apply_limits_after_paint(self, gen: int) -> None:
        """Autoscale / limites manuelles une fois les courbes affichées."""
        if gen != self._autoscale_gen or not self._limits_after_paint:
            return
        if not self._plots or self._spec is None:
            self._limits_after_paint = False
            return
        self._limits_after_paint = False
        self._apply_pending_limits()
        if self._spec.layout_kind == "montage":
            self._fit_montage_left_axis_widths(self._spec)
        # Longueurs auto basées sur le vrai viewRange (pas celui d’avant l’autoscale).
        self._resync_scale_bar_lengths()
        self._refresh_scale_bars()
        self.draw_idle()

    def _apply_pending_limits(self) -> None:
        spec = self._spec
        row_xlims = list(spec.row_xlims) if spec is not None else []
        has_row_xlims = any(rx is not None for rx in row_xlims)
        share_x = bool(
            spec is None
            or spec.layout_kind != "montage"
            or _montage_shares_x(spec)
        )

        if has_row_xlims or (spec is not None and not share_x):
            # Limites X par ligne (extras Pipeline : unités incompatibles).
            for row, plot in enumerate(self._plots):
                rx = row_xlims[row] if row < len(row_xlims) else None
                if rx is not None and float(rx[1]) > float(rx[0]):
                    plot.setXRange(float(rx[0]), float(rx[1]), padding=0)
                elif self._pending_xlim is not None and share_x:
                    plot.setXRange(
                        float(self._pending_xlim[0]),
                        float(self._pending_xlim[1]),
                        padding=0,
                    )
                else:
                    self._autoscale_x_from_data(plot)
        else:
            # Une seule échelle X pour toutes les lignes (XLink) — idempotent.
            xlim: tuple[float, float] | None = None
            if self._pending_xlim is not None:
                xlim = (
                    float(self._pending_xlim[0]),
                    float(self._pending_xlim[1]),
                )
            elif self._pending_auto_x:
                xlim = (
                    self._resolve_shared_xlim(spec)
                    if spec is not None
                    else self._union_x_from_all_plots()
                )
            if xlim is not None and xlim[1] > xlim[0]:
                for plot in self._plots:
                    plot.setXRange(xlim[0], xlim[1], padding=0)

        for row, plot in enumerate(self._plots):
            ylim = (
                self._pending_ylims[row]
                if row < len(self._pending_ylims)
                else None
            )
            auto_y = (
                self._pending_auto_y[row]
                if row < len(self._pending_auto_y)
                else False
            )
            if ylim is not None:
                plot.setYRange(float(ylim[0]), float(ylim[1]), padding=0)
            elif auto_y:
                nonneg = bool(
                    spec is not None
                    and row < len(spec.row_kinds)
                    and str(spec.row_kinds[row]) in {"psth", "trial_rate", "isi", "rms"}
                )
                self._autoscale_y_from_data(plot, non_negative=nonneg)

    def _apply_limits(self, spec: PanelSpec) -> None:
        """Application synchrone (reset vue / tests) — pas le chemin rendu usuel."""
        self._clear_scale_bar_items()
        self._queue_limits_after_paint(spec)
        self._limits_after_paint = False
        # Pending xlim du spec ; si auto, résoudre depuis les courbes maintenant.
        if self._pending_auto_x:
            self._pending_xlim = self._resolve_shared_xlim(spec)
            self._pending_auto_x = False
        self._apply_pending_limits()
        self._resync_scale_bar_lengths()

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
        for row, plot in enumerate(self._plots):
            if not is_montage:
                ylabel = ""
                if row < len(spec.row_labels) and spec.row_labels[row]:
                    ylabel = spec.row_labels[row]
                elif row == 0:
                    ylabel = spec.ylabel or ""
                if ylabel:
                    plot.setLabel("left", ylabel, color=c["fg"], size=f"{label_pt}pt")
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
            if row == len(self._plots) - 1:
                plot.getAxis("bottom").setStyle(showValues=True)
                if spec.xlabel:
                    plot.setLabel(
                        "bottom", spec.xlabel, color=c["fg"], size=f"{label_pt}pt"
                    )
                    axis = plot.getAxis("bottom")
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

    def _fit_montage_left_axis_widths(self, spec: PanelSpec) -> None:
        """Une seule largeur d’axe Y pour toutes les lignes → échelle X visuelle identique."""
        if not self._plots:
            return
        tick_pt = max(5, int(round(float(spec.tick_font_size or 8.0))))
        font = QFont("Segoe UI", tick_pt)
        font.setBold(bool(spec.tick_bold))
        font.setItalic(bool(spec.tick_italic))
        font.setUnderline(bool(getattr(spec, "tick_underline", False)))
        metrics = QFontMetrics(font)
        needed = _MONTAGE_AXIS_WIDTH_MIN
        for plot in self._plots:
            try:
                (_x0, _x1), (y0, y1) = plot.viewRange()
            except Exception:
                continue
            if not np.isfinite([y0, y1]).all() or y0 == y1:
                continue
            # Échantillon de graduations représentatives (bornes + milieu).
            samples = (float(y0), float(y1), 0.5 * (float(y0) + float(y1)))
            for value in samples:
                # Format proche des ticks pyqtgraph (compact).
                text = f"{value:g}"
                needed = max(needed, int(metrics.horizontalAdvance(text)) + 14)
        width = max(_MONTAGE_AXIS_WIDTH_MIN, min(_MONTAGE_AXIS_WIDTH_MAX, needed))
        for plot in self._plots:
            plot.getAxis("left").setWidth(width)

    def _add_montage_title_row(
        self, spec: PanelSpec, *, label_col: int, plot_col: int
    ) -> None:
        """Ligne de titre au-dessus des graphes (ne compte pas dans les hauteurs de ligne)."""
        c = self._colors
        title_pt = max(5, int(round(float(spec.title_font_size or 10.0))))
        title = pg.LabelItem(
            spec.title or " ",
            justify="center",
            color=c["title"],
            size=f"{title_pt}pt",
        )
        _apply_label_face(
            title,
            bold=bool(spec.title_bold),
            italic=bool(spec.title_italic),
            underline=bool(getattr(spec, "title_underline", False)),
        )
        # Colspan sur libellés + graphes.
        self._glw.addItem(title, row=0, col=label_col, colspan=2)
        self._montage_title_label = title
        try:
            self._glw.ci.layout.setRowStretchFactor(0, 0)
            self._glw.ci.layout.setColumnStretchFactor(label_col, 0)
            self._glw.ci.layout.setColumnStretchFactor(plot_col, 1)
        except Exception:
            pass

    def _add_montage_row_labels(
        self, spec: PanelSpec, *, n_rows: int, label_col: int
    ) -> None:
        """Libellés de canal horizontaux (un LabelItem par ligne), éditables."""
        c = self._colors
        label_pt = max(5, int(round(float(spec.label_font_size or 8.0))))
        font = QFont("Segoe UI", label_pt)
        font.setBold(bool(spec.label_bold))
        font.setItalic(bool(spec.label_italic))
        font.setUnderline(bool(getattr(spec, "label_underline", False)))
        metrics = QFontMetrics(font)
        max_w = 48
        size = f"{label_pt}pt"
        row0 = int(self._data_row0)
        for row in range(n_rows):
            text = ""
            if row < len(spec.row_labels) and spec.row_labels[row]:
                text = str(spec.row_labels[row])
            max_w = max(max_w, int(metrics.horizontalAdvance(text)) + 10)
            label = _MontageRowLabel(
                text or " ",
                justify="right",
                color=c["fg"],
                size=size,
            )
            _apply_label_face(
                label,
                bold=bool(spec.label_bold),
                italic=bool(spec.label_italic),
                underline=bool(getattr(spec, "label_underline", False)),
            )
            self._glw.addItem(label, row=row + row0, col=label_col)
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

    def _legend_overlay_offset(self, location: str) -> tuple[int, int]:
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
        return mapping.get(loc, (-10, 10))

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
        # gap (fraction) → espacement vertical entre entrées (px approximatif).
        ver_spacing = max(0, int(round(2.0 + 18.0 * float(spec.legend_gap or 0.0))))
        location = str(spec.legend_location or "below")
        below = location.strip().lower() in {"below", ""}
        self._remove_legend()
        legend = pg.LegendItem(
            offset=(0, 0) if below else self._legend_overlay_offset(location),
            horSpacing=8,
            verSpacing=ver_spacing,
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
            legend_row = int(n_rows) + int(self._data_row0)
            self._glw.addItem(legend, row=legend_row, col=plot_col)
            try:
                self._glw.ci.layout.setRowStretchFactor(legend_row, 0)
            except Exception:
                pass
        else:
            legend.setParentItem(self._plots[0])
            try:
                legend.anchor(
                    itemPos=(1, 0) if "right" in location else (0, 0),
                    parentPos=(1, 0) if "right" in location else (0, 0),
                    offset=self._legend_overlay_offset(location),
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
        # Entrées de légende éditables unitairement.
        for index, (_sample, label_item) in enumerate(
            list(getattr(legend, "items", []) or [])
        ):
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
            try:
                vb.sigResized.disconnect(self._on_scale_bar_range_changed)
            except (TypeError, RuntimeError, AttributeError):
                pass
        self._scale_bar_range_connected.clear()

    def _clear_scale_bar_items(self) -> None:
        for _plot, item in self._scale_bar_items:
            try:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
                else:
                    item.setParentItem(None)
            except Exception:
                try:
                    item.setParentItem(None)
                except Exception:
                    pass
        self._scale_bar_items.clear()

    def _reserve_scale_bar_gutters(self, bars: ScaleBarSpec) -> None:
        """Réserver axe droit (Y) et hauteur bas (X) hors zone de tracé."""
        from plot_utils import format_scale_label

        font = QFont("Segoe UI", max(7, int(round(float(bars.fontsize)))))
        metrics = QFontMetrics(font)
        label_w = 28
        for row, (_x_size, y_size) in enumerate(self._scale_bar_lengths):
            _x_unit, y_unit = self._scale_bar_units_for_row(bars, row)
            if y_size is not None and y_unit is not None:
                lab = format_scale_label(float(y_size), str(y_unit))
                label_w = max(label_w, int(metrics.horizontalAdvance(lab)) + 14)
        right_w = max(_SCALE_BAR_RIGHT_GUTTER_PX, label_w)
        n = len(self._plots)
        for row, plot in enumerate(self._plots):
            plot.showAxis("right", True)
            ra = plot.getAxis("right")
            ra.setStyle(showValues=False, tickLength=0)
            try:
                ra.setTicks([])
            except Exception:
                pass
            ra.setPen(pg.mkPen(None))
            ra.setTextPen(pg.mkPen(None))
            ra.setWidth(right_w)
            ba = plot.getAxis("bottom")
            if row == n - 1 and bars.x_unit is not None:
                ba.setHeight(_SCALE_BAR_BOTTOM_GUTTER_PX)
            else:
                try:
                    ba.setHeight(None)
                except Exception:
                    ba.setHeight(15)

    def _apply_scale_bars(self, spec: PanelSpec, *, is_montage: bool = False) -> None:
        bars = getattr(spec, "scale_bars", None)
        enabled = bars is not None and bool(getattr(bars, "enabled", False))
        if not enabled:
            # Décoche / refresh : retirer les barres et restaurer ticks + cadres.
            self._teardown_scale_bars(spec, is_montage=is_montage)
            return
        if not self._plots:
            return
        # Masquer les graduations numériques seulement — garder les titres d’axes.
        for plot in self._plots:
            plot.getAxis("left").setStyle(showValues=False)
            plot.getAxis("bottom").setStyle(showValues=False)
            for axis_name in ("left", "bottom", "right", "top"):
                try:
                    plot.getAxis(axis_name).setPen(pg.mkPen(None))
                except Exception:
                    pass
        self._scale_bar_lengths = self._resolve_scale_bar_lengths(bars)
        self._reserve_scale_bar_gutters(bars)
        self._connect_scale_bar_range()
        self._refresh_scale_bars()
        # Après layout : les axes ont une taille pixel réelle → redessiner.
        QTimer.singleShot(0, self._refresh_scale_bars)

    def _teardown_scale_bars(self, spec: PanelSpec, *, is_montage: bool) -> None:
        """Retirer les barres flottantes et remettre graduations / cadres d’axes."""
        try:
            self._scale_bar_timer.stop()
        except Exception:
            pass
        self._disconnect_scale_bar_range()
        self._clear_scale_bar_items()
        self._scale_bar_lengths = []
        if not self._plots:
            return
        c = self._colors
        show_borders = bool(getattr(spec, "show_borders", True))
        ticks_inside = bool(getattr(spec, "ticks_inside", False))
        spine_pen = (
            pg.mkPen(c["spine"], width=0.8) if show_borders else pg.mkPen(None)
        )
        text_pen = pg.mkPen(c["fg"])
        n = len(self._plots)
        show_right = bool(ticks_inside and show_borders)
        for row, plot in enumerate(self._plots):
            for axis_name in ("left", "bottom", "right", "top"):
                try:
                    axis = plot.getAxis(axis_name)
                    axis.setPen(spine_pen)
                    axis.setTextPen(text_pen)
                except Exception:
                    pass
            # Ticks Y numériques ; ticks X sur la dernière ligne seulement.
            plot.getAxis("left").setStyle(showValues=True)
            plot.getAxis("bottom").setStyle(showValues=(row == n - 1))
            try:
                plot.getAxis("bottom").setHeight(None)
            except Exception:
                pass
            try:
                plot.showAxis("right", show_right)
                if not show_right:
                    plot.getAxis("right").setWidth(0)
            except Exception:
                pass

    def _resync_scale_bar_lengths(self) -> None:
        """Recalculer les longueurs après que le viewRange soit final (limites/⌂)."""
        spec = self._spec
        if spec is None or not self._plots:
            return
        bars = getattr(spec, "scale_bars", None)
        if bars is None or not bool(getattr(bars, "enabled", False)):
            return
        self._scale_bar_lengths = self._resolve_scale_bar_lengths(bars)
        self._reserve_scale_bar_gutters(bars)

    def _scale_bar_units_for_row(
        self, bars: ScaleBarSpec, row: int
    ) -> tuple[str | None, str | None]:
        """Unités X/Y pour une ligne (overlay montage = ms, pas secondes)."""
        x_unit = bars.x_unit
        y_unit = bars.y_unit
        spec = self._spec
        kinds = list(getattr(spec, "row_kinds", None) or []) if spec else []
        if row < len(kinds):
            kind = str(kinds[row])
            if kind == "overlay":
                return ("ms", "µV")
            if kind == "psth":
                return ("s", "Hz")
            if kind == "trial_rate":
                return (None, "Hz")
            if kind == "isi":
                return ("s", "ms")
            if kind == "raster":
                return ("s", None)
            if kind == "rms":
                return ("s", "µV")
        return (x_unit, y_unit)

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
            x_unit, y_unit = self._scale_bar_units_for_row(bars, row)
            x_size: float | None = None
            y_size: float | None = None
            if y_unit is not None and y_span > 0:
                y_size = resolve_scale_bar_length(
                    y_span,
                    manual=bool(bars.amp_manual) and y_unit == "µV",
                    value=float(bars.amplitude),
                )
            show_x = x_unit is not None and (row == n - 1 or not multi)
            if show_x and x_span > 0:
                manual_x = float(bars.time_s)
                if x_unit == "ms":
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
            try:
                vb.sigResized.connect(self._on_scale_bar_range_changed)
            except Exception:
                pass

    def _on_scale_bar_range_changed(self, *_args: Any) -> None:
        if self._scale_bars_refreshing:
            return
        self._schedule_scale_bar_refresh()

    def _schedule_scale_bar_refresh(self) -> None:
        """Coalescer les rebuilds de barres d’échelle pendant pan/zoom/XLink."""
        if self._scale_bars_refreshing:
            return
        if not self._scale_bar_timer.isActive():
            self._scale_bar_timer.start()

    def _refresh_scale_bars(self) -> None:
        self._scale_bar_timer.stop()
        spec = self._spec
        if spec is None:
            return
        bars = getattr(spec, "scale_bars", None)
        if bars is None or not bool(getattr(bars, "enabled", False)):
            # Style désactivé après un pan/zoom programmé : ne pas redessiner.
            self._disconnect_scale_bar_range()
            self._clear_scale_bar_items()
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
        """Barres dans les gouttières d’axes (droite / bas), hors ViewBox."""
        from plot_utils import format_scale_label

        color = str(bars.color or "#111827")
        qcolor = QColor(color)
        pen = QPen(qcolor, float(bars.linewidth))
        pen.setCosmetic(True)
        font = QFont("Segoe UI", max(7, int(round(float(bars.fontsize)))))
        metrics = QFontMetrics(font)
        lengths = self._scale_bar_lengths
        if len(lengths) != len(self._plots):
            lengths = self._resolve_scale_bar_lengths(bars)
            self._scale_bar_lengths = lengths
            self._reserve_scale_bar_gutters(bars)

        pad = float(_SCALE_BAR_PAD_PX)
        n = len(self._plots)
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
            x_unit, y_unit = self._scale_bar_units_for_row(bars, row)

            if y_size is not None and y_size > 0 and y_unit is not None:
                self._draw_y_scale_in_gutter(
                    plot,
                    y_size=float(y_size),
                    y_span=y_span,
                    label=format_scale_label(float(y_size), str(y_unit)),
                    pen=pen,
                    brush=QBrush(qcolor),
                    font=font,
                    metrics=metrics,
                    pad=pad,
                )

            if (
                x_size is not None
                and x_size > 0
                and x_unit is not None
                and row == n - 1
            ):
                self._draw_x_scale_in_gutter(
                    plot,
                    x_size=float(x_size),
                    x_span=x_span,
                    label=format_scale_label(float(x_size), str(x_unit)),
                    pen=pen,
                    brush=QBrush(qcolor),
                    font=font,
                    metrics=metrics,
                    pad=pad,
                )

    def _draw_y_scale_in_gutter(
        self,
        plot: pg.PlotItem,
        *,
        y_size: float,
        y_span: float,
        label: str,
        pen: QPen,
        brush: QBrush,
        font: QFont,
        metrics: QFontMetrics,
        pad: float,
    ) -> None:
        """Bâton + libellé Y dans l’axe droit (à côté du graph)."""
        axis = plot.getAxis("right")
        try:
            h = float(axis.height())
            w = float(axis.width())
        except Exception:
            return
        if h < 4.0 or w < 4.0:
            return
        # Longueur pixel strictement proportionnelle à y_size / y_span
        # (pas de clamp : sinon le libellé ne correspond plus au bâton).
        bar_h = max(1.0, (float(y_size) / float(y_span)) * h)
        y_a = 0.5 * (h - min(bar_h, h))
        y_b = y_a + bar_h
        x_line = pad
        line = QGraphicsLineItem(x_line, y_a, x_line, y_b)
        line.setPen(pen)
        line.setParentItem(axis)
        line.setZValue(50)
        self._scale_bar_items.append((plot, line))

        text = QGraphicsSimpleTextItem(label)
        text.setFont(font)
        text.setBrush(brush)
        text.setParentItem(axis)
        text.setZValue(50)
        text_h = float(metrics.height())
        text.setPos(QPointF(x_line + pad, 0.5 * (y_a + y_b) - 0.5 * text_h))
        self._scale_bar_items.append((plot, text))

    def _draw_x_scale_in_gutter(
        self,
        plot: pg.PlotItem,
        *,
        x_size: float,
        x_span: float,
        label: str,
        pen: QPen,
        brush: QBrush,
        font: QFont,
        metrics: QFontMetrics,
        pad: float,
    ) -> None:
        """Bâton + libellé X dans l’axe bas (sous le graph)."""
        axis = plot.getAxis("bottom")
        try:
            h = float(axis.height())
            w = float(axis.width())
        except Exception:
            return
        if h < 4.0 or w < 8.0:
            return
        # Longueur pixel = x_size / x_span (libellé = vraie valeur données).
        bar_w = max(1.0, (float(x_size) / float(x_span)) * w)
        x_right = w - pad
        x_left = x_right - bar_w
        if x_left < pad:
            # Ancrage à gauche si la barre dépasse : garder bar_w exact.
            x_left = pad
            x_right = x_left + bar_w
        y_line = pad
        line = QGraphicsLineItem(x_left, y_line, x_right, y_line)
        line.setPen(pen)
        line.setParentItem(axis)
        line.setZValue(50)
        self._scale_bar_items.append((plot, line))

        text = QGraphicsSimpleTextItem(label)
        text.setFont(font)
        text.setBrush(brush)
        text.setParentItem(axis)
        text.setZValue(50)
        text_w = float(metrics.horizontalAdvance(label))
        text.setPos(
            QPointF(
                0.5 * (x_left + x_right) - 0.5 * text_w,
                y_line + pad,
            )
        )
        self._scale_bar_items.append((plot, text))

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
