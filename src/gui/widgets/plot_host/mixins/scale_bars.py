"""Barres d’échelle."""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsLineItem, QGraphicsSimpleTextItem

from panel_prepare import PanelSpec, ScaleBarSpec

from ..constants import (
    _MONTAGE_DATA_BOTTOM_AXIS_PX,
    _SCALE_BAR_BOTTOM_GUTTER_PX,
    _SCALE_BAR_PAD_PX,
    _SCALE_BAR_RIGHT_GUTTER_PX,
)
from ..helpers import (
    scale_bar_offset_px as _scale_bar_offset_px,
    scale_bar_pair_px as _scale_bar_pair_px,
    scale_bar_time_offset_px as _scale_bar_time_offset_px,
    uncrop_scale_axis as _uncrop_scale_axis,
)
from ..text_style import _face_qfont, _scale_label_item, _tick_qfont


class ScaleBarsMixin:
    def scale_bar_needed_view_height_px(self) -> int | None:
        """Hauteur ViewBox (px) pour que amplitude = taille amp., données visibles.

        Somme sur les lignes : chaque span Y données × ``amp_length_px / amplitude``.
        ``None`` si l’échelle ne pilote pas la hauteur, ou si un ylim Canal est fixé.
        """
        spec = getattr(self, "_spec", None)
        if spec is None:
            return None
        bars = getattr(spec, "scale_bars", None)
        if bars is None or not bool(getattr(bars, "enabled", False)):
            return None
        if not bool(getattr(bars, "amp_manual", False)):
            return None
        if not bool(getattr(bars, "amp_length_manual", False)):
            return None
        # ylim Canal / manuel : ne pas écraser par un fit hauteur.
        if getattr(spec, "ylim", None) is not None:
            return None
        row_ylims = list(getattr(spec, "row_ylims", None) or [])
        if any(y is not None for y in row_ylims):
            return None
        try:
            amp = float(getattr(bars, "amplitude", 0.0) or 0.0)
            length = float(getattr(bars, "amp_length_px", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if not (amp > 0.0 and length > 0.0):
            return None
        plots = list(getattr(self, "_plots", None) or [])
        if not plots:
            return None
        total = 0.0
        any_span = False
        data_range = getattr(self, "_data_range_from_plot", None)
        for row, plot in enumerate(plots):
            # Seulement les lignes tension (µV) — pas ISI / PSTH / raster.
            _x_unit, y_unit = self._scale_bar_units_for_row(bars, row)
            if y_unit != "µV":
                continue
            span = None
            if callable(data_range):
                try:
                    span = data_range(plot, axis="y")
                except Exception:
                    span = None
            if span is None:
                try:
                    (_x0, _x1), (y0, y1) = plot.viewRange()
                    if float(y1) > float(y0):
                        span = (float(y0), float(y1))
                except Exception:
                    span = None
            if span is None:
                continue
            dy = float(span[1]) - float(span[0])
            if not (dy > 0.0) or dy != dy:
                continue
            any_span = True
            total += dy * (length / amp)
        if not any_span or not (total > 0.0) or total != total:
            return None
        return max(40, int(round(total)))

    def scale_bar_needed_view_width_px(self) -> int | None:
        """Largeur ViewBox (px) pour que durée = taille temps, données visibles.

        ``x_span × time_length_px / time_value`` (unités de l’axe : s ou ms).
        ``None`` si l’échelle ne pilote pas la largeur.
        """
        spec = getattr(self, "_spec", None)
        if spec is None:
            return None
        bars = getattr(spec, "scale_bars", None)
        if bars is None or not bool(getattr(bars, "enabled", False)):
            return None
        if not bool(getattr(bars, "time_manual", False)):
            return None
        if not bool(getattr(bars, "time_length_manual", False)):
            return None
        try:
            length = float(getattr(bars, "time_length_px", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if not (length > 0.0):
            return None
        plots = list(getattr(self, "_plots", None) or [])
        if not plots:
            return None
        # Échelle X = dernière ligne (barre temps unique sous le montage).
        row = len(plots) - 1
        plot = plots[row]
        x_unit, _y_unit = self._scale_bar_units_for_row(bars, row)
        if x_unit is None:
            return None
        try:
            time_s = float(getattr(bars, "time_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        time_val = time_s * 1000.0 if x_unit == "ms" else time_s
        if not (time_val > 0.0) or time_val != time_val:
            return None
        span = None
        data_range = getattr(self, "_data_range_from_plot", None)
        if callable(data_range):
            try:
                span = data_range(plot, axis="x")
            except Exception:
                span = None
        if span is None:
            try:
                (x0, x1), _y = plot.viewRange()
                if float(x1) > float(x0):
                    span = (float(x0), float(x1))
            except Exception:
                span = None
        if span is None:
            return None
        dx = float(span[1]) - float(span[0])
        if not (dx > 0.0) or dx != dx:
            return None
        needed = dx * (length / time_val)
        if not (needed > 0.0) or needed != needed:
            return None
        return max(80, int(round(needed)))

    def _scale_bar_item_ids(self) -> set[int]:
        return {id(item) for _plot, item in self._scale_bar_items}

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

        spec = self._spec
        font = _tick_qfont(
            max(7, int(round(float(bars.fontsize)))),
            bold=bool(getattr(spec, "tick_bold", False)),
            italic=bool(getattr(spec, "tick_italic", False)),
            underline=bool(getattr(spec, "tick_underline", False)),
        )
        metrics = QFontMetrics(font)
        amp_dx, _amp_dy = _scale_bar_offset_px(bars)
        _time_dx, time_dy = _scale_bar_time_offset_px(bars)
        pad = float(_SCALE_BAR_PAD_PX)
        label_w = 28
        label_h = int(metrics.height())
        for row, (_x_size, y_size) in enumerate(self._scale_bar_lengths):
            _x_unit, y_unit = self._scale_bar_units_for_row(bars, row)
            if y_size is not None and y_unit is not None:
                lab = format_scale_label(float(y_size), str(y_unit))
                label_w = max(label_w, int(metrics.horizontalAdvance(lab)) + 14)
        # Élargir la gouttière pour que bâton + libellé restent dans le panneau.
        # +dx amplitude → plus de place à droite ; dx négatif : le bâton entre dans la courbe.
        right_needed = int(round(pad + max(0.0, amp_dx) + pad + max(0, label_w - 14) + 8))
        right_w = max(_SCALE_BAR_RIGHT_GUTTER_PX, label_w, right_needed)
        # +dy temps (haut) : le bâton monte vers la courbe ; -dy : descendre → plus bas.
        bottom_needed = int(
            round(pad + max(0.0, -time_dy) + pad + float(label_h) + 4.0)
        )
        bottom_h = max(_SCALE_BAR_BOTTOM_GUTTER_PX, bottom_needed)
        n = len(self._plots)
        is_montage = bool(
            spec is not None and str(getattr(spec, "layout_kind", "")) == "montage"
        )
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
            _uncrop_scale_axis(ra)
            ba = plot.getAxis("bottom")
            if is_montage:
                try:
                    ba.setHeight(_MONTAGE_DATA_BOTTOM_AXIS_PX)
                except Exception:
                    pass
            elif row == n - 1 and bars.x_unit is not None:
                ba.setHeight(bottom_h)
                _uncrop_scale_axis(ba)
            else:
                try:
                    ba.setHeight(None)
                except Exception:
                    ba.setHeight(15)
        if (
            is_montage
            and bars.x_unit is not None
            and self._montage_x_axis_plot is not None
        ):
            x_plot = self._montage_x_axis_plot
            x_ba = x_plot.getAxis("bottom")
            try:
                x_ba.setStyle(showValues=False, tickLength=0)
            except Exception:
                try:
                    x_ba.setStyle(showValues=False)
                except Exception:
                    pass
            x_ba.setPen(pg.mkPen(None))
            x_ba.setTextPen(pg.mkPen(None))
            x_ba.setHeight(bottom_h)
            _uncrop_scale_axis(x_ba)
            try:
                needed = bottom_h + 8
                x_plot.setMinimumHeight(needed)
                x_plot.setMaximumHeight(needed + 12)
                grid_row = self._montage_x_axis_grid_row(n)
                layout = self._glw.ci.layout
                if hasattr(layout, "setRowFixedHeight"):
                    layout.setRowFixedHeight(grid_row, needed)
                else:
                    layout.setRowMinimumHeight(grid_row, needed)
            except Exception:
                pass

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
        # Axe X partagé (montage) : même règle — barres flottantes sans ticks.
        if is_montage and self._montage_x_axis_plot is not None:
            x_plot = self._montage_x_axis_plot
            try:
                x_plot.getAxis("bottom").setStyle(showValues=False, tickLength=0)
            except Exception:
                try:
                    x_plot.getAxis("bottom").setStyle(showValues=False)
                except Exception:
                    pass
            for axis_name in ("left", "bottom", "right", "top"):
                try:
                    x_plot.getAxis(axis_name).setPen(pg.mkPen(None))
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
            # Ticks Y numériques ; ticks X sur l’axe partagé (montage) ou dernière ligne.
            plot.getAxis("left").setStyle(showValues=True)
            if is_montage:
                plot.getAxis("bottom").setStyle(showValues=False)
                try:
                    plot.getAxis("bottom").setHeight(_MONTAGE_DATA_BOTTOM_AXIS_PX)
                except Exception:
                    pass
            else:
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
        if is_montage and self._montage_x_axis_plot is not None:
            try:
                self._ensure_montage_x_axis_row(
                    spec, n_rows=n, plot_col=self._plot_col
                )
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
                # Histogramme : X = ISI (ms), Y = comptages.
                return ("ms", None)
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
        """Barres dans les gouttières d’axes (droite / bas), hors ViewBox.

        Valeur+taille manuelles → libellé = valeur (taille widget pilotée).
        Taille seule → libellé calculé. Valeur seule → bâton proportionnel.
        """
        color = str(bars.color or "#111827")
        qcolor = QColor(color)
        pen = QPen(qcolor, float(bars.linewidth))
        pen.setCosmetic(True)
        spec = self._spec
        font = _tick_qfont(
            max(7, int(round(float(bars.fontsize)))),
            bold=bool(getattr(spec, "tick_bold", False)),
            italic=bool(getattr(spec, "tick_italic", False)),
            underline=bool(getattr(spec, "tick_underline", False)),
        )
        metrics = QFontMetrics(font)
        lengths = self._scale_bar_lengths
        if len(lengths) != len(self._plots):
            lengths = self._resolve_scale_bar_lengths(bars)
            self._scale_bar_lengths = lengths
            self._reserve_scale_bar_gutters(bars)

        pad = float(_SCALE_BAR_PAD_PX)
        amp_dx, amp_dy = _scale_bar_offset_px(bars)
        time_dx, time_dy = _scale_bar_time_offset_px(bars)
        n = len(self._plots)
        for row, plot in enumerate(self._plots):
            try:
                (x0, x1), (y0, y1) = plot.viewRange()
            except Exception:
                continue
            x_span = abs(float(x1) - float(x0))
            y_span = abs(float(y1) - float(y0))
            x_size, y_size = lengths[row] if row < len(lengths) else (None, None)
            x_unit, y_unit = self._scale_bar_units_for_row(bars, row)

            amp_len_px = (
                float(bars.amp_length_px)
                if bool(getattr(bars, "amp_length_manual", False))
                and float(getattr(bars, "amp_length_px", 0.0) or 0.0) > 0
                else None
            )
            time_len_px = (
                float(bars.time_length_px)
                if bool(getattr(bars, "time_length_manual", False))
                and float(getattr(bars, "time_length_px", 0.0) or 0.0) > 0
                else None
            )
            amp_value_manual = bool(bars.amp_manual) and y_unit == "µV"
            time_value_manual = bool(bars.time_manual)

            if (
                y_size is not None
                and y_size > 0
                and y_unit is not None
                and y_span > 0
            ):
                self._draw_y_scale_in_gutter(
                    plot,
                    y_size=float(y_size),
                    y_span=y_span,
                    y_unit=str(y_unit),
                    pen=pen,
                    brush=QBrush(qcolor),
                    font=font,
                    metrics=metrics,
                    pad=pad,
                    dx=amp_dx,
                    dy=amp_dy,
                    length_px=amp_len_px,
                    value_manual=amp_value_manual,
                )

            if (
                x_size is not None
                and x_size > 0
                and x_unit is not None
                and x_span > 0
                and row == n - 1
            ):
                x_host = self._montage_x_axis_plot or plot
                self._draw_x_scale_in_gutter(
                    x_host,
                    x_size=float(x_size),
                    x_span=x_span,
                    x_unit=str(x_unit),
                    pen=pen,
                    brush=QBrush(qcolor),
                    font=font,
                    metrics=metrics,
                    pad=pad,
                    dx=time_dx,
                    dy=time_dy,
                    length_px=time_len_px,
                    value_manual=time_value_manual,
                )

    def _draw_y_scale_in_gutter(
        self,
        plot: pg.PlotItem,
        *,
        y_size: float,
        y_span: float,
        y_unit: str,
        pen: QPen,
        brush: QBrush,
        font: QFont,
        metrics: QFontMetrics,
        pad: float,
        dx: float = 0.0,
        dy: float = 0.0,
        length_px: float | None = None,
        value_manual: bool = False,
    ) -> None:
        """Bâton + libellé Y dans l’axe droit (à côté du graph)."""
        from plot_utils import (
            format_scale_label,
            scale_bar_label_value,
            scale_bar_pixel_length,
        )

        axis = plot.getAxis("right")
        try:
            h = float(axis.height())
            w = float(axis.width())
        except Exception:
            return
        if h < 4.0 or w < 4.0:
            return
        length_manual = length_px is not None and float(length_px) > 0
        bar_h = scale_bar_pixel_length(
            data_value=float(y_size),
            span=float(y_span),
            axis_px=h,
            length_px=length_px if length_manual else None,
            value_manual=bool(value_manual),
        )
        if bar_h < 1.0:
            return
        label_value = scale_bar_label_value(
            data_value=float(y_size),
            span=float(y_span),
            axis_px=h,
            bar_px=bar_h,
            value_manual=bool(value_manual),
            length_manual=length_manual,
        )
        label = format_scale_label(label_value, str(y_unit))
        # Qt : y vers le bas. dy positif (haut) diminue y.
        y_a = 0.5 * (h - bar_h) - float(dy)
        y_b = y_a + bar_h
        if y_a < 0.0:
            y_b -= y_a
            y_a = 0.0
        if y_b > h:
            y_a -= y_b - h
            y_b = h
            y_a = max(0.0, y_a)
        # Taille seule : si le clamp raccourcit le bâton, recalculer le libellé.
        drawn_h = max(0.0, y_b - y_a)
        if (
            length_manual
            and not value_manual
            and drawn_h + 0.5 < bar_h
        ):
            label_value = scale_bar_label_value(
                data_value=float(y_size),
                span=float(y_span),
                axis_px=h,
                bar_px=drawn_h,
                value_manual=False,
                length_manual=True,
            )
            label = format_scale_label(label_value, str(y_unit))
        text_w = float(metrics.horizontalAdvance(label))
        text_h = float(metrics.height())
        # Garder bâton + libellé dans la gouttière droite (élargie si besoin).
        x_line = pad + float(dx)
        x_line = max(0.0, min(x_line, max(0.0, w - pad - text_w)))
        line = QGraphicsLineItem(x_line, y_a, x_line, y_b)
        line.setPen(pen)
        line.setParentItem(axis)
        line.setZValue(50)
        self._scale_bar_items.append((plot, line))

        text = _scale_label_item(label, font, brush)
        text.setParentItem(axis)
        text.setZValue(50)
        text_x = min(x_line + pad, max(0.0, w - text_w))
        text_y = max(0.0, min(0.5 * (y_a + y_b) - 0.5 * text_h, h - text_h))
        text.setPos(QPointF(text_x, text_y))
        self._scale_bar_items.append((plot, text))

    def _draw_x_scale_in_gutter(
        self,
        plot: pg.PlotItem,
        *,
        x_size: float,
        x_span: float,
        x_unit: str,
        pen: QPen,
        brush: QBrush,
        font: QFont,
        metrics: QFontMetrics,
        pad: float,
        dx: float = 0.0,
        dy: float = 0.0,
        length_px: float | None = None,
        value_manual: bool = False,
    ) -> None:
        """Bâton + libellé X dans l’axe bas (sous le graph)."""
        from plot_utils import (
            format_scale_label,
            scale_bar_label_value,
            scale_bar_pixel_length,
        )

        axis = plot.getAxis("bottom")
        try:
            h = float(axis.height())
            w = float(axis.width())
        except Exception:
            return
        if h < 4.0 or w < 8.0:
            return
        length_manual = length_px is not None and float(length_px) > 0
        bar_w = scale_bar_pixel_length(
            data_value=float(x_size),
            span=float(x_span),
            axis_px=w,
            length_px=length_px if length_manual else None,
            value_manual=bool(value_manual),
        )
        if bar_w < 1.0:
            return
        label_value = scale_bar_label_value(
            data_value=float(x_size),
            span=float(x_span),
            axis_px=w,
            bar_px=bar_w,
            value_manual=bool(value_manual),
            length_manual=length_manual,
        )
        label = format_scale_label(label_value, str(x_unit))
        text_w = float(metrics.horizontalAdvance(label))
        text_h = float(metrics.height())
        x_right = w - pad
        x_left = x_right - bar_w
        if x_left < pad:
            # Ancrage à gauche si la barre dépasse : garder bar_w exact.
            x_left = pad
            x_right = x_left + bar_w
        x_left += float(dx)
        x_right += float(dx)
        # Recentrer dans [0, w] si le décalage dépasse encore la gouttière.
        if x_left < 0.0:
            x_right -= x_left
            x_left = 0.0
        if x_right > w:
            x_left -= x_right - w
            x_right = w
            x_left = max(0.0, x_left)
        drawn_w = max(0.0, x_right - x_left)
        if (
            length_manual
            and not value_manual
            and drawn_w + 0.5 < bar_w
        ):
            label_value = scale_bar_label_value(
                data_value=float(x_size),
                span=float(x_span),
                axis_px=w,
                bar_px=drawn_w,
                value_manual=False,
                length_manual=True,
            )
            label = format_scale_label(label_value, str(x_unit))
            text_w = float(metrics.horizontalAdvance(label))
        # Qt : y vers le bas. dy positif (haut) rapproche la barre du graph.
        y_line = pad - float(dy)
        y_max = max(pad, h - pad - text_h)
        y_line = max(0.0, min(y_line, y_max))
        line = QGraphicsLineItem(x_left, y_line, x_right, y_line)
        line.setPen(pen)
        line.setParentItem(axis)
        line.setZValue(50)
        self._scale_bar_items.append((plot, line))

        text = _scale_label_item(label, font, brush)
        text.setParentItem(axis)
        text.setZValue(50)
        text_x = max(0.0, min(0.5 * (x_left + x_right) - 0.5 * text_w, w - text_w))
        text_y = max(0.0, min(y_line + pad, h - text_h))
        text.setPos(QPointF(text_x, text_y))
        self._scale_bar_items.append((plot, text))

