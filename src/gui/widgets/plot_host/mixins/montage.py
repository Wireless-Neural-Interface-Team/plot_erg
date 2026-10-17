"""Layout / lignes / axe X / libellés montage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QGraphicsItem, QWidget

from panel_prepare import PanelSpec
from plot_utils import preview_title_row_layout

from ..constants import (
    _MONTAGE_AXIS_WIDTH_MAX,
    _MONTAGE_AXIS_WIDTH_MIN,
    _MONTAGE_DATA_BOTTOM_AXIS_PX,
    _MONTAGE_DATA_ROW_SPACING_PX,
    _MONTAGE_X_AXIS_HEIGHT_PX,
)
from ..helpers import (
    _montage_label_column_extra,
    _montage_label_shift,
    montage_shares_x as _montage_shares_x,
)
from ..montage_items import _MontageRowLabel
from ..shims import _AxisShim
from ..text_model import _EditableText
from ..text_style import _apply_label_face, _face_qfont

def _host_apply_plot_tick_font(*args, **kwargs):
    from .. import host as _host_mod
    return _host_mod._apply_plot_tick_font(*args, **kwargs)



class MontageMixin:
    def _apply_montage_row_minimums(self, spec: PanelSpec) -> None:
        """Chaque ligne de courbe : même hauteur mini + même écart vertical."""
        row_min = max(0, int(getattr(spec, "row_min_height_px", 0) or 0))
        layout = getattr(getattr(self._glw, "ci", None), "layout", None)
        if layout is None or not hasattr(layout, "setRowMinimumHeight"):
            return
        try:
            row_count = int(layout.rowCount())
        except Exception:
            row_count = 0
        for row in range(row_count):
            try:
                layout.setRowMinimumHeight(row, 0)
            except Exception:
                return
        n_plots = len(self._plots)
        row0 = int(self._data_row0)
        # Écarts entre lignes de données identiques (pas le gap légende).
        try:
            layout.setVerticalSpacing(_MONTAGE_DATA_ROW_SPACING_PX)
        except Exception:
            pass
        for index in range(max(0, n_plots - 1)):
            try:
                layout.setRowSpacing(
                    row0 + index, _MONTAGE_DATA_ROW_SPACING_PX
                )
            except Exception:
                pass
        if row_min <= 0:
            return
        for index in range(n_plots):
            try:
                layout.setRowMinimumHeight(row0 + index, row_min)
            except Exception:
                return

    def _collapse_montage_data_bottom_axes(self) -> None:
        """Retirer ticks/xlabel des lignes de courbes (hauteur ViewBox uniforme)."""
        for plot in self._plots:
            bottom = plot.getAxis("bottom")
            self._clear_bottom_axis_label(bottom, collapse=True)
            try:
                bottom.setStyle(showValues=False, tickLength=0)
            except Exception:
                try:
                    bottom.setStyle(showValues=False)
                except Exception:
                    pass
            try:
                bottom.setHeight(_MONTAGE_DATA_BOTTOM_AXIS_PX)
            except Exception:
                pass

    def _montage_x_axis_grid_row(self, n_rows: int) -> int:
        return int(n_rows) + int(self._data_row0)

    def _ensure_montage_x_axis_row(
        self, spec: PanelSpec, *, n_rows: int, plot_col: int
    ) -> None:
        """Ligne d’axe X sous le stack : ticks + xlabel hors des ViewBox."""
        c = self._colors
        grid_row = self._montage_x_axis_grid_row(n_rows)
        plot = self._montage_x_axis_plot
        if plot is not None:
            pos = getattr(getattr(self._glw, "ci", None), "items", {}).get(plot)
            if not (isinstance(pos, tuple) and len(pos) >= 1 and int(pos[0]) == grid_row):
                self._discard_layout_item(plot)
                self._montage_x_axis_plot = None
                plot = None
        if plot is None:
            self._free_grid_cell(grid_row, plot_col)
            plot = self._glw.addPlot(row=grid_row, col=plot_col)
            self._montage_x_axis_plot = plot
            plot.setMenuEnabled(False)
            vb = plot.getViewBox()
            try:
                vb.enableAutoRange(enable=False)
                vb.setDefaultPadding(0.0)
                vb.setMouseEnabled(x=False, y=False)
                vb.setMenuEnabled(False)
            except Exception:
                pass
            plot.hideAxis("left")
            plot.hideAxis("right")
            plot.hideAxis("top")
            plot.showAxis("bottom", True)
            try:
                plot.setTitle(None)
            except Exception:
                pass
            try:
                self._glw.ci.layout.setRowStretchFactor(grid_row, 0)
                if hasattr(self._glw.ci.layout, "setRowFixedHeight"):
                    self._glw.ci.layout.setRowFixedHeight(
                        grid_row, _MONTAGE_X_AXIS_HEIGHT_PX
                    )
                else:
                    self._glw.ci.layout.setRowMinimumHeight(
                        grid_row, _MONTAGE_X_AXIS_HEIGHT_PX
                    )
            except Exception:
                pass
        if self._plots:
            try:
                plot.setXLink(self._plots[0])
            except Exception:
                pass
        bottom = plot.getAxis("bottom")
        show_borders = bool(spec.show_borders)
        spine_pen = (
            pg.mkPen(c["spine"], width=0.8) if show_borders else pg.mkPen(None)
        )
        bottom.setPen(spine_pen)
        bottom.setTextPen(pg.mkPen(c["fg"]))
        ticks_inside = bool(spec.ticks_inside)
        try:
            bottom.setStyle(
                showValues=True,
                tickLength=(-6 if ticks_inside else 6),
            )
        except Exception:
            bottom.setStyle(showValues=True)
        _host_apply_plot_tick_font(plot, spec, montage=True)
        try:
            bottom.setHeight(None)
        except Exception:
            pass
        label_pt = max(5, int(round(float(spec.label_font_size or 9.0))))
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
                        label_item, "xlabel", axes_index=max(0, n_rows - 1), item_index=0
                    )
                )
        else:
            self._clear_bottom_axis_label(bottom, collapse=False)
        try:
            # Hauteur de la ligne de chrome : ticks (+ xlabel).
            needed = _MONTAGE_X_AXIS_HEIGHT_PX
            if spec.xlabel:
                needed = max(needed, _MONTAGE_X_AXIS_HEIGHT_PX + 14)
            plot.setMaximumHeight(needed + 8)
            plot.setMinimumHeight(needed)
            layout = self._glw.ci.layout
            if hasattr(layout, "setRowFixedHeight"):
                layout.setRowFixedHeight(grid_row, needed + 4)
            else:
                layout.setRowMinimumHeight(grid_row, needed)
            layout.setRowStretchFactor(grid_row, 0)
            # Petit écart fixe après la dernière courbe (identique quel que soit n).
            if n_rows > 0:
                layout.setRowSpacing(
                    int(self._data_row0) + n_rows - 1,
                    _MONTAGE_DATA_ROW_SPACING_PX,
                )
        except Exception:
            pass

    def montage_needed_glw_height_px(self) -> int | None:
        """Hauteur de vue nécessaire pour montrer toutes les lignes et la légende.

        ``None`` si ce n’est pas un montage. La valeur peut dépasser la vue
        actuelle : le panneau doit alors grandir, sinon le bas est rogné.
        """
        spec = self._spec
        if spec is None or spec.layout_kind != "montage" or not self._plots:
            return None
        self._apply_montage_row_minimums(spec)
        layout = getattr(getattr(self._glw, "ci", None), "layout", None)
        needed = 0.0
        if layout is not None:
            try:
                layout.activate()
            except Exception:
                pass
            try:
                hint = layout.effectiveSizeHint(Qt.SizeHint.MinimumSize)
                needed = float(hint.height())
            except Exception:
                needed = 0.0
        bottom = 0.0
        items = [self._plots[-1]]
        if self._montage_x_axis_plot is not None:
            items.append(self._montage_x_axis_plot)
        if self._legend_item is not None:
            items.append(self._legend_item)
        for item in items:
            try:
                bottom = max(bottom, float(item.sceneBoundingRect().bottom()))
            except Exception:
                pass
        view_h = float(self._glw.height())
        if bottom > view_h + 0.5:
            # Le contenu dépasse la vue : garder une marge pour la légende.
            needed = max(needed, bottom + 10.0)
        elif bottom > 0.0:
            needed = max(needed, bottom)
        if needed <= 0.0:
            return None
        gap = needed - int(needed)
        return int(needed) + (1 if gap > 0.01 else 0)

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
        # La légende occupe la ligne juste sous les graphes. Sans ça, la
        # première ligne ajoutée tombe sur une cellule prise : Qt refuse
        # l’item, pyqtgraph le garde quand même, et le prochain clear() plante.
        self._remove_legend()
        # Trim
        while len(self._plots) > n_rows:
            plot = self._plots.pop()
            self._discard_layout_item(plot)
            if self._row_labels:
                label = self._row_labels.pop()
                self._discard_layout_item(label)
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
            self._free_grid_cell(grid_row, label_col)
            self._glw.addItem(label, row=grid_row, col=label_col)
            self._row_labels.append(label)
            self._free_grid_cell(grid_row, plot_col)
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
        self._apply_montage_label_offsets(spec)

    def _update_montage_row_labels(self, spec: PanelSpec) -> None:
        """Refresh LabelItem texts without recreating the layout column."""
        c = self._colors
        label_pt = max(5, int(round(float(spec.label_font_size or 8.0))))
        size = f"{label_pt}pt"
        font = QFont("Segoe UI", label_pt)
        font.setBold(bool(spec.label_bold))
        font.setItalic(bool(spec.label_italic))
        font.setUnderline(bool(getattr(spec, "label_underline", False)))
        metrics = QFontMetrics(font)
        extra = _montage_label_column_extra(spec)
        max_w = 48 + extra
        for row, label in enumerate(self._row_labels):
            text = ""
            if row < len(spec.row_labels) and spec.row_labels[row]:
                text = str(spec.row_labels[row])
            max_w = max(max_w, int(metrics.horizontalAdvance(text)) + 10 + extra)
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
        try:
            self._glw.ci.layout.setColumnFixedWidth(0, max_w)
        except Exception:
            pass
        self._apply_montage_label_offsets(spec)

    def _apply_montage_label_offsets(self, spec: PanelSpec) -> None:
        dx, dy = _montage_label_shift(spec)
        for label in self._row_labels:
            if isinstance(label, _MontageRowLabel):
                label.set_pixel_offset(dx, dy)

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

    def _apply_montage_title_row_height(self, spec: PanelSpec) -> None:
        """Hauteur de la ligne de titre montage (marges via le layout GLW)."""
        if self._montage_title_label is None:
            return
        layout = getattr(getattr(self._glw, "ci", None), "layout", None)
        if layout is None:
            return
        text_h = 14.0
        try:
            text_h = max(
                1.0, float(self._montage_title_label.itemRect().height())
            )
        except Exception:
            text_h = 14.0
        row_h, _spacing = preview_title_row_layout(
            text_h,
            0,
            title_dx_px=float(getattr(spec, "title_dx_px", 0.0) or 0.0),
            title_dy_px=float(getattr(spec, "title_dy_px", 0.0) or 0.0),
        )
        try:
            self._montage_title_label.setMaximumHeight(row_h)
            layout.setRowFixedHeight(0, row_h)
            layout.setRowMinimumHeight(0, row_h)
        except Exception:
            pass
        self._apply_stim_label_band_margin(
            0 if not self._stim_labels else None
        )

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
        extra = _montage_label_column_extra(spec)
        max_w = 48 + extra
        size = f"{label_pt}pt"
        row0 = int(self._data_row0)
        for row in range(n_rows):
            text = ""
            if row < len(spec.row_labels) and spec.row_labels[row]:
                text = str(spec.row_labels[row])
            max_w = max(max_w, int(metrics.horizontalAdvance(text)) + 10 + extra)
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
        self._apply_montage_label_offsets(spec)

