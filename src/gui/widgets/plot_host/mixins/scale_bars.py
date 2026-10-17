"""Barres d’échelle flottantes (écran pyqtgraph) — via ``scale_bar_model``."""

from __future__ import annotations

import math
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QPointF, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPen
from PySide6.QtWidgets import QGraphicsLineItem

from panel_prepare import PanelSpec, ScaleBarSpec, scale_bar_spec_settings_key
from scale_bar_model import (
    amp_manual_applies,
    format_scale_label,
    major_ticks_for_axis,
    resolve_axis_bar,
    time_value_in_axis_units,
    units_for_kind,
    uses_shared_manual_scale,
)

from ..constants import (
    _MONTAGE_DATA_BOTTOM_AXIS_PX,
    _SCALE_BAR_BOTTOM_GUTTER_PX,
    _SCALE_BAR_PAD_PX,
    _SCALE_BAR_RIGHT_GUTTER_PX,
)
from ..helpers import (
    scale_bar_offset_px as _scale_bar_offset_px,
    scale_bar_time_offset_px as _scale_bar_time_offset_px,
    uncrop_scale_axis as _uncrop_scale_axis,
)
from ..text_style import _scale_label_item, _tick_qfont

def _manual_y_params(
    bars: ScaleBarSpec, y_unit: str, *, shared: bool = True
) -> tuple[float | None, float | None]:
    if not shared or not amp_manual_applies(y_unit):
        return None, None
    val = float(bars.amplitude) if bool(bars.amp_manual) else None
    length = (
        float(bars.amp_length_px) if bool(bars.amp_length_manual) else None
    )
    return val, length


def _manual_x_params(
    bars: ScaleBarSpec, x_unit: str, *, shared: bool = True
) -> tuple[float | None, float | None]:
    if not shared:
        return None, None
    val = None
    if bool(bars.time_manual):
        val = time_value_in_axis_units(float(bars.time_s), x_unit)
        if not (val and val > 0):
            val = None
    length = (
        float(bars.time_length_px) if bool(bars.time_length_manual) else None
    )
    return val, length


def _resolve_y_bar(
    bars: ScaleBarSpec,
    *,
    span: float,
    axis_px: float,
    y_unit: str,
    shared: bool = True,
):
    if not (span > 0) or not (axis_px > 0) or not y_unit:
        return None
    val, length = _manual_y_params(bars, y_unit, shared=shared)
    if val is None and length is None:
        return resolve_axis_bar(span=span, axis_px=axis_px)
    return resolve_axis_bar(span=span, axis_px=axis_px, value=val, length_px=length)


def _resolve_x_bar(
    bars: ScaleBarSpec,
    *,
    span: float,
    axis_px: float,
    x_unit: str,
    shared: bool = True,
):
    if not (span > 0) or not (axis_px > 0) or not x_unit:
        return None
    val, length = _manual_x_params(bars, x_unit, shared=shared)
    if val is None and length is None:
        return resolve_axis_bar(span=span, axis_px=axis_px)
    return resolve_axis_bar(span=span, axis_px=axis_px, value=val, length_px=length)


class ScaleBarsMixin:
    def scale_bar_widget_fit_enabled(self) -> bool:
        """True si l’utilisateur a défini valeur + taille px (temps ou amplitude)."""
        bars = self._active_scale_bars()
        if bars is None:
            return False
        amp_fit = bool(bars.amp_manual) and bool(bars.amp_length_manual)
        time_fit = bool(bars.time_manual) and bool(bars.time_length_manual)
        return amp_fit or time_fit

    def scale_bar_needed_view_height_px(self) -> int | None:
        """Hauteur ViewBox totale pour que amplitude (µV) = taille amp. (px).

        Utilise la plage Y *visible* (ylim manuel compris) :
        ``hauteur = span × taille_px / amplitude``. Pas de plancher : un
        minimum en pixels change le gain µV/px.
        """
        if not self.scale_bar_widget_fit_enabled():
            return None
        bars = self._active_scale_bars()
        if bars is None or not (
            bool(bars.amp_manual) and bool(bars.amp_length_manual)
        ):
            return None
        try:
            amp = float(bars.amplitude)
            length = float(bars.amp_length_px)
        except (TypeError, ValueError):
            return None
        if not (amp > 0.0 and length > 0.0):
            return None
        plots = list(getattr(self, "_plots", None) or [])
        if not plots:
            return None
        total = 0.0
        any_span = False
        for row, plot in enumerate(plots):
            _x_unit, y_unit = self._scale_bar_units_for_row(bars, row)
            if not self._row_shares_manual_scale(row):
                # Spike scope : hauteur de ligne fixe, pas le gain µV des traces.
                any_span = True
                total += float(self._independent_row_height_px())
                continue
            if not amp_manual_applies(y_unit):
                continue
            try:
                (_x0, _x1), (y0, y1) = plot.viewRange()
            except Exception:
                continue
            dy = abs(float(y1) - float(y0))
            if not (dy > 0.0) or dy != dy:
                continue
            bar = resolve_axis_bar(
                span=dy, axis_px=100.0, value=amp, length_px=length
            )
            if bar.needed_axis_px is None:
                continue
            any_span = True
            total += float(bar.needed_axis_px)
        if not any_span or not (total > 0.0) or total != total:
            return None
        return max(1, int(round(total)))

    def scale_bar_needed_view_width_px(self) -> int | None:
        """Largeur ViewBox pour temps (s) = taille temps (px), plage X visible."""
        if not self.scale_bar_widget_fit_enabled():
            return None
        bars = self._active_scale_bars()
        if bars is None or not (
            bool(bars.time_manual) and bool(bars.time_length_manual)
        ):
            return None
        try:
            length = float(bars.time_length_px)
        except (TypeError, ValueError):
            return None
        if not (length > 0.0):
            return None
        plots = list(getattr(self, "_plots", None) or [])
        if not plots:
            return None
        row = self._shared_scale_width_row()
        if row is None:
            return None
        plot = plots[row]
        x_unit, _y_unit = self._scale_bar_units_for_row(bars, row)
        if x_unit is None:
            return None
        time_val = time_value_in_axis_units(float(bars.time_s), x_unit)
        if not (time_val > 0.0):
            return None
        try:
            (x0, x1), _y = plot.viewRange()
            if not (float(x1) > float(x0)):
                return None
            dx = abs(float(x1) - float(x0))
        except Exception:
            return None
        bar = resolve_axis_bar(
            span=dx, axis_px=100.0, value=time_val, length_px=length
        )
        if bar.needed_axis_px is None or not (bar.needed_axis_px > 0):
            return None
        needed = float(bar.needed_axis_px)
        if not (needed > 0.0) or needed != needed:
            return None
        # Pas de plancher : 80 px minimum étirait les zooms courts
        # (le bâton ne valait plus la durée affichée).
        return max(1, int(round(needed)))

    def _active_scale_bars(self) -> ScaleBarSpec | None:
        spec = getattr(self, "_spec", None)
        if spec is None:
            return None
        bars = getattr(spec, "scale_bars", None)
        if bars is None or not bool(getattr(bars, "enabled", False)):
            return None
        return bars

    def _axis_span(
        self, plot: pg.PlotItem, *, axis: str
    ) -> tuple[float, float] | None:
        data_range = getattr(self, "_data_range_from_plot", None)
        if callable(data_range):
            try:
                span = data_range(plot, axis=axis)
                if span is not None and float(span[1]) > float(span[0]):
                    return (float(span[0]), float(span[1]))
            except Exception:
                pass
        try:
            (x0, x1), (y0, y1) = plot.viewRange()
            if axis == "x" and float(x1) > float(x0):
                return (float(x0), float(x1))
            if axis == "y" and float(y1) > float(y0):
                return (float(y0), float(y1))
        except Exception:
            pass
        return None

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

    def _row_kind(self, row: int) -> str:
        spec = getattr(self, "_spec", None)
        kinds = list(getattr(spec, "row_kinds", None) or []) if spec else []
        if 0 <= row < len(kinds):
            return str(kinds[row])
        return ""

    def _row_shares_manual_scale(self, row: int) -> bool:
        """False pour une ligne spike scope : échelle propre à ses formes d’onde."""
        return uses_shared_manual_scale(kind=self._row_kind(row))

    def _independent_row_height_px(self) -> int:
        spec = getattr(self, "_spec", None)
        raw = int(getattr(spec, "row_min_height_px", 0) or 0) if spec else 0
        return max(36, raw or 52)

    def _shared_scale_width_row(self) -> int | None:
        """Dernière ligne dont l’axe X suit le gain temps partagé (pas le spike scope)."""
        plots = list(getattr(self, "_plots", None) or [])
        bars = self._active_scale_bars()
        if bars is None:
            return None
        for row in range(len(plots) - 1, -1, -1):
            if not self._row_shares_manual_scale(row):
                continue
            x_unit, _y_unit = self._scale_bar_units_for_row(bars, row)
            if x_unit is not None:
                return row
        return None

    def _pin_independent_scale_rows(self) -> None:
        """Le spike scope garde une hauteur de ligne fixe quand le gain µV est manuel.

        Les autres lignes continuent de se partager le reste, au gain demandé.
        """
        spec = getattr(self, "_spec", None)
        if spec is None or str(getattr(spec, "layout_kind", "") or "") != "montage":
            return
        layout = getattr(getattr(self._glw, "ci", None), "layout", None)
        if layout is None or not hasattr(layout, "setRowFixedHeight"):
            return
        bars = self._active_scale_bars()
        amp_fit = bool(
            bars is not None
            and bool(bars.amp_manual)
            and bool(bars.amp_length_manual)
        )
        row0 = int(getattr(self, "_data_row0", 0) or 0)
        height = self._independent_row_height_px()
        for row in range(len(self._plots)):
            grid_row = row0 + row
            try:
                if amp_fit and not self._row_shares_manual_scale(row):
                    layout.setRowFixedHeight(grid_row, height)
                    layout.setRowStretchFactor(grid_row, 0)
                else:
                    layout.setRowMaximumHeight(grid_row, 16777215)
                    layout.setRowStretchFactor(grid_row, 1)
            except Exception:
                continue

    def _scale_bar_units_for_row(
        self, bars: ScaleBarSpec, row: int
    ) -> tuple[str | None, str | None]:
        spec = self._spec
        kinds = list(getattr(spec, "row_kinds", None) or []) if spec else []
        kind = str(kinds[row]) if row < len(kinds) else ""
        return units_for_kind(kind, default_x=bars.x_unit, default_y=bars.y_unit)

    def _resolve_scale_bar_lengths(
        self, bars: ScaleBarSpec
    ) -> list[tuple[float | None, float | None]]:
        """Cibles X/Y en unités de données (une par ligne)."""
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
            shared = self._row_shares_manual_scale(row)
            x_size: float | None = None
            y_size: float | None = None
            y_bar = (
                _resolve_y_bar(
                    bars,
                    span=y_span,
                    axis_px=100.0,
                    y_unit=str(y_unit),
                    shared=shared,
                )
                if y_unit is not None and y_span > 0
                else None
            )
            y_size = float(y_bar.label) if y_bar is not None else None
            show_x = x_unit is not None and (row == n - 1 or not multi)
            x_bar = (
                _resolve_x_bar(
                    bars,
                    span=x_span,
                    axis_px=100.0,
                    x_unit=str(x_unit),
                    shared=shared,
                )
                if show_x and x_span > 0
                else None
            )
            x_size = float(x_bar.label) if x_bar is not None else None
            out.append((x_size, y_size))
        return out

    def _reserve_scale_bar_gutters(self, bars: ScaleBarSpec) -> None:
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
        # Le décalage élargit la gouttière. Avec valeur + taille px fixes, le
        # fit du widget (empreinte de placement) rend ces pixels au graphique :
        # la zone de tracé ne change pas, le widget si.
        right_needed = int(round(pad + max(0.0, amp_dx) + pad + max(0, label_w - 14) + 8))
        right_w = max(_SCALE_BAR_RIGHT_GUTTER_PX, label_w, right_needed)
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

    def _request_scale_widget_fit(self) -> None:
        """Demander au parent un recalage H/W (échelle amp. / temps)."""
        try:
            self.scaleBarHeightFitRequested.emit()
        except Exception:
            pass
        try:
            self.scaleBarWidthFitRequested.emit()
        except Exception:
            pass

    def _apply_scale_bars(self, spec: PanelSpec, *, is_montage: bool = False) -> None:
        bars = getattr(spec, "scale_bars", None)
        enabled = bars is not None and bool(getattr(bars, "enabled", False))
        if not enabled:
            self._teardown_scale_bars(spec, is_montage=is_montage)
            return
        if not self._plots:
            return
        self._pin_independent_scale_rows()
        floating = bool(getattr(bars, "floating", True))
        if not floating:
            key = scale_bar_spec_settings_key(bars)
            if key != getattr(self, "_scale_bar_settings_key", None):
                self._scale_bar_settings_key = key
                self._clear_scale_bar_items()
            self._scale_bar_lengths = self._resolve_scale_bar_lengths(bars)
            self._connect_scale_bar_range()
            self._apply_scale_axis_ticks(bars)
            self._request_scale_widget_fit()
            QTimer.singleShot(0, lambda: self._apply_scale_axis_ticks(bars))
            return
        for plot in self._plots:
            plot.getAxis("left").setStyle(showValues=False)
            plot.getAxis("bottom").setStyle(showValues=False)
            for axis_name in ("left", "bottom", "right", "top"):
                try:
                    plot.getAxis(axis_name).setPen(pg.mkPen(None))
                except Exception:
                    pass
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
        key = scale_bar_spec_settings_key(bars)
        if key != getattr(self, "_scale_bar_settings_key", None):
            self._scale_bar_settings_key = key
            self._clear_scale_bar_items()
        self._scale_bar_lengths = self._resolve_scale_bar_lengths(bars)
        self._reserve_scale_bar_gutters(bars)
        self._connect_scale_bar_range()
        self._refresh_scale_bars()
        self._request_scale_widget_fit()
        QTimer.singleShot(0, self._refresh_scale_bars)

    def _teardown_scale_bars(self, spec: PanelSpec, *, is_montage: bool) -> None:
        try:
            self._scale_bar_timer.stop()
        except Exception:
            pass
        self._disconnect_scale_bar_range()
        self._clear_scale_bar_items()
        self._scale_bar_lengths = []
        self._scale_bar_settings_key = None
        self._pin_independent_scale_rows()
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
        spec = self._spec
        if spec is None or not self._plots:
            return
        bars = getattr(spec, "scale_bars", None)
        if bars is None or not bool(getattr(bars, "enabled", False)):
            return
        self._scale_bar_lengths = self._resolve_scale_bar_lengths(bars)
        self._reserve_scale_bar_gutters(bars)

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
        # Pas de fit widget ici (sigRange/sigResized) : boucle resize ↔ layout.

    def _schedule_scale_bar_refresh(self) -> None:
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
            self._disconnect_scale_bar_range()
            self._clear_scale_bar_items()
            return
        if not self._plots or self._scale_bars_refreshing:
            return
        self._scale_bar_lengths = self._resolve_scale_bar_lengths(bars)
        self._scale_bars_refreshing = True
        try:
            if not bool(getattr(bars, "floating", True)):
                self._clear_scale_bar_items()
                self._apply_scale_axis_ticks(bars)
            else:
                self._clear_scale_bar_items()
                self._draw_scale_bar_artists(bars)
        finally:
            self._scale_bars_refreshing = False

    def _scale_ticks_for_axis(
        self,
        *,
        lo: float,
        hi: float,
        span: float,
        axis_px: float,
        bars: ScaleBarSpec,
        axis: str,
        unit: str,
        shared: bool = True,
    ) -> list[tuple[float, str]] | None:
        if axis == "y":
            val, length = _manual_y_params(bars, unit, shared=shared)
        else:
            val, length = _manual_x_params(bars, unit, shared=shared)
        val_manual = val is not None
        length_manual = length is not None
        ticks = major_ticks_for_axis(
            lo=lo,
            hi=hi,
            span=span,
            axis_px=axis_px,
            unit=unit,
            manual_value=val,
            value_manual=val_manual,
            length_px=length,
            length_manual=length_manual,
        )
        return ticks or None

    def _apply_scale_axis_ticks(self, bars: ScaleBarSpec) -> None:
        """Graduations classiques calées sur le même modèle que les barres flottantes."""
        n = len(self._plots)
        for row, plot in enumerate(self._plots):
            try:
                (x0, x1), (y0, y1) = plot.viewRange()
            except Exception:
                continue
            x_unit, y_unit = self._scale_bar_units_for_row(bars, row)
            shared = self._row_shares_manual_scale(row)
            left = plot.getAxis("left")
            try:
                h = float(left.height())
            except Exception:
                h = 0.0
            y_span = abs(float(y1) - float(y0))
            if y_unit is not None and y_span > 0 and h > 4:
                y_ticks = self._scale_ticks_for_axis(
                    lo=float(y0),
                    hi=float(y1),
                    span=y_span,
                    axis_px=h,
                    bars=bars,
                    axis="y",
                    unit=str(y_unit),
                    shared=shared,
                )
                if y_ticks:
                    left.setTicks([y_ticks])
                left.setStyle(showValues=True)
            if row == n - 1 and x_unit is not None:
                bottom = plot.getAxis("bottom")
                try:
                    w = float(bottom.width())
                except Exception:
                    w = 0.0
                x_span = abs(float(x1) - float(x0))
                if x_span > 0 and w > 4:
                    x_ticks = self._scale_ticks_for_axis(
                        lo=float(x0),
                        hi=float(x1),
                        span=x_span,
                        axis_px=w,
                        bars=bars,
                        axis="x",
                        unit=str(x_unit),
                        shared=shared,
                    )
                    if x_ticks:
                        bottom.setTicks([x_ticks])
                    bottom.setStyle(showValues=True)

    def _draw_scale_bar_artists(self, bars: ScaleBarSpec) -> None:
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
            x_unit, y_unit = self._scale_bar_units_for_row(bars, row)
            shared = self._row_shares_manual_scale(row)

            if y_unit is not None and y_span > 0:
                self._draw_y_scale_in_gutter(
                    plot,
                    bars=bars,
                    y_span=y_span,
                    y_unit=str(y_unit),
                    shared=shared,
                    pen=pen,
                    brush=QBrush(qcolor),
                    font=font,
                    metrics=metrics,
                    pad=pad,
                    dx=amp_dx,
                    dy=amp_dy,
                )

            if x_unit is not None and x_span > 0 and row == n - 1:
                x_host = self._montage_x_axis_plot or plot
                self._draw_x_scale_in_gutter(
                    x_host,
                    bars=bars,
                    x_span=x_span,
                    x_unit=str(x_unit),
                    shared=shared,
                    pen=pen,
                    brush=QBrush(qcolor),
                    font=font,
                    metrics=metrics,
                    pad=pad,
                    dx=time_dx,
                    dy=time_dy,
                )

    def _draw_y_scale_in_gutter(
        self,
        plot: pg.PlotItem,
        *,
        bars: ScaleBarSpec,
        y_span: float,
        y_unit: str,
        shared: bool = True,
        pen: QPen,
        brush: QBrush,
        font: QFont,
        metrics: QFontMetrics,
        pad: float,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> None:
        axis = plot.getAxis("right")
        try:
            h = float(axis.height())
            w = float(axis.width())
        except Exception:
            return
        if h < 4.0 or w < 4.0:
            return
        bar = _resolve_y_bar(
            bars, span=y_span, axis_px=h, y_unit=y_unit, shared=shared
        )
        if bar is None:
            return
        val, length = _manual_y_params(bars, y_unit, shared=shared)
        both_manual = val is not None and length is not None
        length_only = length is not None and val is None
        bar_h = float(bar.bar_px)
        if bar_h < 1.0:
            return
        label = format_scale_label(float(bar.label), str(y_unit))
        y_a = 0.5 * (h - bar_h) - float(dy)
        y_b = y_a + bar_h
        if not both_manual:
            if y_a < 0.0:
                y_b -= y_a
                y_a = 0.0
            if y_b > h:
                y_a -= y_b - h
                y_b = h
                y_a = max(0.0, y_a)
        drawn_h = max(0.0, y_b - y_a)
        # Taille seule : si clamp, libellé = géométrie réelle du bâton dessiné.
        if length_only and drawn_h + 0.5 < bar_h:
            label = format_scale_label(
                resolve_axis_bar(
                    span=y_span, axis_px=h, value=None, length_px=drawn_h
                ).label,
                str(y_unit),
            )
        text_w = float(metrics.horizontalAdvance(label))
        text_h = float(metrics.height())
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
        bars: ScaleBarSpec,
        x_span: float,
        x_unit: str,
        shared: bool = True,
        pen: QPen,
        brush: QBrush,
        font: QFont,
        metrics: QFontMetrics,
        pad: float,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> None:
        axis = plot.getAxis("bottom")
        try:
            h = float(axis.height())
            w = float(axis.width())
        except Exception:
            return
        if h < 4.0 or w < 8.0:
            return
        bar = _resolve_x_bar(
            bars, span=x_span, axis_px=w, x_unit=x_unit, shared=shared
        )
        if bar is None:
            return
        val, length = _manual_x_params(bars, x_unit, shared=shared)
        both_manual = val is not None and length is not None
        length_only = length is not None and val is None
        bar_w = float(bar.bar_px)
        if bar_w < 1.0:
            return
        label = format_scale_label(float(bar.label), str(x_unit))
        text_w = float(metrics.horizontalAdvance(label))
        text_h = float(metrics.height())
        x_right = w - pad
        x_left = x_right - bar_w
        if not both_manual:
            if x_left < pad:
                x_left = pad
                x_right = x_left + bar_w
            x_left += float(dx)
            x_right += float(dx)
            if x_left < 0.0:
                x_right -= x_left
                x_left = 0.0
            if x_right > w:
                x_left -= x_right - w
                x_right = w
                x_left = max(0.0, x_left)
        else:
            x_left += float(dx)
            x_right += float(dx)
        drawn_w = max(0.0, x_right - x_left)
        if length_only and drawn_w + 0.5 < bar_w:
            label = format_scale_label(
                resolve_axis_bar(
                    span=x_span, axis_px=w, value=None, length_px=drawn_w
                ).label,
                str(x_unit),
            )
            text_w = float(metrics.horizontalAdvance(label))
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
