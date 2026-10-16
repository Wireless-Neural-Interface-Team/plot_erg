"""Capture / restore / reset / xlim partagé."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg

from panel_prepare import PanelSpec

from ..helpers import montage_shares_x as _montage_shares_x


class ViewLimitsMixin:
    def apply_pending_limits_now(self) -> None:
        """Forcer l’application des xlim/ylim en attente (fenêtres zoom sync)."""
        if self._limits_after_paint:
            self._apply_limits_after_paint(self._autoscale_gen)
            return
        if (
            self._pending_xlim is not None
            or self._pending_auto_x
            or any(self._pending_auto_y)
        ):
            self._apply_pending_limits()

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
                # Overlay : temps en ms autour du spike — ne pas réappliquer
                # l’échelle d’une autre famille (secondes) ni l’ancienne fenêtre.
                kind = str(key[1])
                if restore_x and kind != "overlay":
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
        self._apply_x_ticks(self._spec)
        if self._spec.layout_kind == "montage":
            self._fit_montage_left_axis_widths(self._spec)
            self._update_stim_spans()
            self._schedule_stim_span_update()
        # Libellés hors ViewBox (bande layout) — les pointillés restent dans les courbes.
        self._reposition_stim_labels()
        if self._zone_labels and not self._stim_labels:
            self._reposition_zone_labels()
        # Longueurs auto basées sur le vrai viewRange (pas celui d’avant l’autoscale).
        self._resync_scale_bar_lengths()
        self._refresh_scale_bars()
        self.draw_idle()
        # Amp. + taille amp. → hauteur ; temps + taille temps → largeur.
        try:
            needed_h = self.scale_bar_needed_view_height_px()
        except Exception:
            needed_h = None
        if needed_h is not None:
            try:
                self.scaleBarHeightFitRequested.emit()
            except Exception:
                pass
        try:
            needed_w = self.scale_bar_needed_view_width_px()
        except Exception:
            needed_w = None
        if needed_w is not None:
            try:
                self.scaleBarWidthFitRequested.emit()
            except Exception:
                pass

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
