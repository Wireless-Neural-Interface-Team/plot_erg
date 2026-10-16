"""Spans stim + libellés stim / zone."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsSimpleTextItem

from panel_prepare import montage_vline_span_keys, montage_vline_spans

from ..montage_items import _MontageStimSpanItem
from ..text_model import TextRole, _EditableText, _StimLabelItem, _ZoneLabelItem
from ..text_style import _face_qfont, _stamp_text_face


class StimZoneMixin:
    def _clear_stim_labels(self) -> None:
        """Retirer les overlays on/off de la scène."""
        scene = None
        try:
            scene = self._glw.scene()
        except Exception:
            scene = None
        for entry in self._stim_labels:
            try:
                item = entry.item
                if scene is not None and item.scene() is scene:
                    scene.removeItem(item)
                elif item.scene() is not None:
                    item.scene().removeItem(item)
            except Exception:
                pass
        self._stim_labels.clear()
        self._apply_stim_label_band_margin(0)

    def _add_stim_label(self, plot: pg.PlotItem, ann: Any, *, row: int) -> None:
        """Libellé « on » / « off » au-dessus de toutes les courbes (hors ViewBox)."""
        label_text = str(getattr(ann, "text", "") or "")
        if not label_text.strip():
            return
        role_raw = str(getattr(ann, "role", "") or "")
        edit_role: TextRole = (
            "stim_off" if role_raw == "stim_offset" else "stim_on"
        )
        color = str(getattr(ann, "color", "") or "#111827")
        fontsize = max(4.0, float(getattr(ann, "fontsize", 8.0) or 8.0))
        bold = bool(getattr(ann, "bold", False))
        italic = bool(getattr(ann, "italic", False))
        underline = bool(getattr(ann, "underline", False))
        dx_px = float(getattr(ann, "dx_px", 0.0) or 0.0)
        dy_px = float(getattr(ann, "dy_px", 0.0) or 0.0)
        font = _face_qfont(
            fontsize, bold=bold, italic=italic, underline=underline
        )
        text = QGraphicsSimpleTextItem(label_text)
        text.setFont(font)
        try:
            text.setBrush(QBrush(pg.mkColor(color)))
        except Exception:
            text.setBrush(QBrush(QColor(color)))
        try:
            text.setZValue(2000)
        except Exception:
            pass
        try:
            text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        except Exception:
            pass
        try:
            self._glw.scene().addItem(text)
        except Exception:
            return
        # Ancre X sur le 1er plot (haut de la pile), pour rester au-dessus de tout.
        anchor_plot = self._plots[0] if self._plots else plot
        item_index = len(self._stim_labels)
        self._stim_labels.append(
            _StimLabelItem(
                item=text,
                plot=anchor_plot,
                x=float(ann.x),
                dx_px=dx_px,
                dy_px=dy_px,
                fontsize=fontsize,
                color=color,
            )
        )
        self._editable.append(
            _EditableText(text, edit_role, axes_index=row, item_index=item_index)
        )

    def _stim_label_pad_px(self, entries: list[_StimLabelItem] | None = None) -> float:
        """Hauteur (px) de la bande au-dessus des courbes pour les libellés."""
        items = entries if entries is not None else self._stim_labels
        pad = 0.0
        for entry in items:
            try:
                rect = entry.item.boundingRect()
                text_h = max(float(rect.height()), float(entry.fontsize) * 1.35)
            except Exception:
                text_h = max(12.0, float(entry.fontsize) * 1.35)
            need = text_h + 6.0 + max(0.0, float(entry.dy_px))
            pad = max(pad, need)
        return pad

    def _zone_label_pad_px(self, entries: list[_ZoneLabelItem] | None = None) -> float:
        """Hauteur (px) de la bande sous les courbes pour les libellés Zone."""
        items = entries if entries is not None else self._zone_labels
        pad = 0.0
        for entry in items:
            try:
                rect = entry.item.boundingRect()
                text_h = max(float(rect.height()), float(entry.fontsize) * 1.35)
            except Exception:
                text_h = max(12.0, float(entry.fontsize) * 1.35)
            # dy positif = haut (vers le graph) ; dy négatif = plus bas → plus de marge.
            need = text_h + 6.0 + max(0.0, -float(entry.dy_px))
            pad = max(pad, need)
        return pad

    def _title_offset_margin_px(self) -> tuple[int, int, int]:
        """``(left, top, right)`` réservés pour le décalage du titre."""
        spec = self._spec
        if spec is None:
            return (0, 0, 0)
        dx = int(round(float(getattr(spec, "title_dx_px", 0.0) or 0.0)))
        dy = int(round(float(getattr(spec, "title_dy_px", 0.0) or 0.0)))
        return (max(0, -dx), max(0, dy), max(0, dx))

    def _apply_stim_label_band_margin(self, height_px: float | None = None) -> None:
        """Marges du layout GLW : libellés stim (haut) + zones (bas) + titre.

        ``height_px`` force la bande stim (``0`` = pas de libellés). ``None`` =
        recalculer depuis les libellés présents. Le pad titre est toujours
        fusionné pour ne pas écraser un décalage d’aperçu.
        """
        if height_px is None:
            stim_top = (
                int(round(self._stim_label_pad_px())) if self._stim_labels else 0
            )
        else:
            stim_top = max(0, int(round(float(height_px))))
        zone_bottom = (
            int(round(self._zone_label_pad_px())) if self._zone_labels else 0
        )
        title_left, title_top, title_right = self._title_offset_margin_px()
        top = max(stim_top, title_top)
        try:
            layout = self._glw.ci.layout
            layout.setContentsMargins(title_left, top, title_right, zone_bottom)
        except Exception:
            try:
                self._glw.setContentsMargins(title_left, top, title_right, zone_bottom)
            except Exception:
                pass

    def _stim_label_stack_top_scene_y(self) -> float | None:
        """Y scène du haut du contenu (titre ou 1ʳᵉ courbe) — sous la bande libellés."""
        tops: list[float] = []
        if self._montage_title_label is not None:
            try:
                tops.append(
                    float(self._montage_title_label.sceneBoundingRect().top())
                )
            except Exception:
                pass
        if self._plots:
            try:
                tops.append(
                    float(self._plots[0].getViewBox().sceneBoundingRect().top())
                )
            except Exception:
                pass
        if not tops:
            return None
        return min(tops)

    def _zone_label_stack_bottom_scene_y(self) -> float | None:
        """Y scène du bas de la pile de courbes — au-dessus de la bande Zone."""
        if not self._plots:
            return None
        bottoms: list[float] = []
        # Dernière courbe (ou axe X montage dédié) = bas du stack.
        candidates = list(self._plots)
        if self._montage_x_axis_plot is not None:
            candidates.append(self._montage_x_axis_plot)
        for plot in candidates:
            try:
                bottoms.append(
                    float(plot.getViewBox().sceneBoundingRect().bottom())
                )
            except Exception:
                pass
        if not bottoms:
            return None
        return max(bottoms)

    def _reposition_stim_labels(self) -> None:
        """Placer les libellés dans la marge haute, au-dessus de toutes les courbes."""
        if not self._stim_labels:
            self._apply_stim_label_band_margin(0 if not self._zone_labels else None)
            if self._zone_labels:
                self._reposition_zone_labels()
            return
        self._apply_stim_label_band_margin()
        stack_top = self._stim_label_stack_top_scene_y()
        if stack_top is None:
            return
        for entry in self._stim_labels:
            try:
                mapped = self._stim_span_scene_point(entry.plot, float(entry.x))
                if mapped is None:
                    vb = entry.plot.getViewBox()
                    rect = vb.sceneBoundingRect()
                    (x0, x1), _y = vb.viewRange()
                    if not np.isfinite([x0, x1]).all() or float(x0) == float(x1):
                        entry.item.hide()
                        continue
                    frac = (float(entry.x) - float(x0)) / (float(x1) - float(x0))
                    scene_x = float(rect.left()) + frac * float(rect.width())
                else:
                    scene_x = float(mapped[0])
                br = entry.item.boundingRect()
                text_w = float(br.width())
                text_h = float(br.height())
                # Qt : y vers le bas. dy positif = vers le haut de l’écran.
                x = scene_x - 0.5 * text_w + float(entry.dx_px)
                y = float(stack_top) - text_h - 2.0 - float(entry.dy_px)
                entry.item.setPos(QPointF(x, y))
                entry.item.show()
            except Exception:
                continue
        if self._zone_labels:
            self._reposition_zone_labels()

    def _clear_zone_labels(self) -> None:
        """Retirer les overlays Zone de la scène."""
        scene = None
        try:
            scene = self._glw.scene()
        except Exception:
            scene = None
        for entry in self._zone_labels:
            try:
                item = entry.item
                if scene is not None and item.scene() is scene:
                    scene.removeItem(item)
                elif item.scene() is not None:
                    item.scene().removeItem(item)
            except Exception:
                pass
        self._zone_labels.clear()
        self._editable = [e for e in self._editable if e.role != "zone"]
        # Recalculer marges (stim éventuellement encore présents).
        self._apply_stim_label_band_margin(
            0 if not self._stim_labels else None
        )

    def _add_zone_label_from_ann(self, ann: Any) -> None:
        """Ajouter un libellé Zone depuis une AnnotationSpec (sous le graph)."""
        label_text = str(getattr(ann, "text", "") or "").strip()
        if not label_text or not self._plots:
            return
        color = str(getattr(ann, "color", "") or "#111827")
        fontsize = max(4.0, float(getattr(ann, "fontsize", 8.0) or 8.0))
        bold = bool(getattr(ann, "bold", False))
        italic = bool(getattr(ann, "italic", False))
        underline = bool(getattr(ann, "underline", False))
        dx_px = float(getattr(ann, "dx_px", 0.0) or 0.0)
        dy_px = float(getattr(ann, "dy_px", 0.0) or 0.0)
        self._append_zone_overlay(
            x=float(ann.x),
            text=label_text,
            color=color,
            fontsize=fontsize,
            bold=bold,
            italic=italic,
            underline=underline,
            dx_px=dx_px,
            dy_px=dy_px,
        )

    def _append_zone_overlay(
        self,
        *,
        x: float,
        text: str,
        color: str,
        fontsize: float,
        bold: bool,
        italic: bool,
        underline: bool,
        dx_px: float,
        dy_px: float,
    ) -> None:
        if not self._plots or not str(text).strip():
            return
        anchor_plot = self._plots[0]
        fs = max(4.0, float(fontsize))
        font = _face_qfont(fs, bold=bold, italic=italic, underline=underline)
        item = QGraphicsSimpleTextItem(str(text))
        item.setFont(font)
        try:
            item.setBrush(QBrush(pg.mkColor(color)))
        except Exception:
            item.setBrush(QBrush(QColor(color)))
        try:
            item.setZValue(2000)
        except Exception:
            pass
        try:
            item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
        except Exception:
            pass
        try:
            self._glw.scene().addItem(item)
        except Exception:
            return
        index = len(self._zone_labels)
        self._zone_labels.append(
            _ZoneLabelItem(
                item=item,
                plot=anchor_plot,
                x=float(x),
                dx_px=float(dx_px),
                dy_px=float(dy_px),
                fontsize=fs,
                color=str(color),
            )
        )
        self._editable.append(
            _EditableText(item, "zone", axes_index=0, item_index=index)
        )

    def set_zone_labels(
        self,
        labels: Sequence[tuple[float, str, str]] | None,
        *,
        fontsize: float = 8.0,
        bold: bool = False,
        italic: bool = False,
        underline: bool = False,
        dx_px: float = 0.0,
        dy_px: float = 0.0,
    ) -> None:
        """Overlays « Zone N » sous le graph (même mécanique que on/off).

        ``labels`` : ``(x_centre, texte, couleur)`` en coords d’affichage.
        """
        self._clear_zone_labels()
        if not labels or not self._plots:
            self.draw_idle()
            return
        for entry in labels:
            if len(entry) < 2:
                continue
            text = str(entry[1] or "").strip()
            if not text:
                continue
            color = str(entry[2] if len(entry) >= 3 else "") or "#111827"
            self._append_zone_overlay(
                x=float(entry[0]),
                text=text,
                color=color,
                fontsize=float(fontsize),
                bold=bold,
                italic=italic,
                underline=underline,
                dx_px=float(dx_px),
                dy_px=float(dy_px),
            )
        self._reposition_zone_labels()
        # Brancher pan/zoom pour suivre la marge basse (comme on/off en haut).
        self._arm_stim_spans()
        self.draw_idle()

    def _reposition_zone_labels(self) -> None:
        """Placer les libellés Zone dans la marge basse, sous toutes les courbes."""
        if not self._zone_labels:
            self._apply_stim_label_band_margin(
                0 if not self._stim_labels else None
            )
            return
        self._apply_stim_label_band_margin()
        stack_bottom = self._zone_label_stack_bottom_scene_y()
        if stack_bottom is None:
            return
        for entry in self._zone_labels:
            try:
                mapped = self._stim_span_scene_point(entry.plot, float(entry.x))
                if mapped is None:
                    vb = entry.plot.getViewBox()
                    rect = vb.sceneBoundingRect()
                    (x0, x1), _y = vb.viewRange()
                    if not np.isfinite([x0, x1]).all() or float(x0) == float(x1):
                        entry.item.hide()
                        continue
                    frac = (float(entry.x) - float(x0)) / (float(x1) - float(x0))
                    scene_x = float(rect.left()) + frac * float(rect.width())
                else:
                    scene_x = float(mapped[0])
                br = entry.item.boundingRect()
                text_w = float(br.width())
                # Qt : y vers le bas. dy positif = vers le haut (vers le graph).
                x = scene_x - 0.5 * text_w + float(entry.dx_px)
                y = float(stack_bottom) + 2.0 - float(entry.dy_px)
                entry.item.setPos(QPointF(x, y))
                entry.item.show()
            except Exception:
                continue

    def _arm_stim_spans(self) -> None:
        """Brancher le trait continu et le positionner dès que la géométrie existe."""
        if (
            not self._stim_span_specs
            and not self._stim_labels
            and not self._zone_labels
        ):
            self._clear_stim_spans()
            return
        if self._stim_span_specs:
            self._ensure_stim_span_item()
        self._connect_stim_spans()
        self._schedule_stim_span_update()

    def _ensure_stim_span_item(self) -> _MontageStimSpanItem:
        item = self._stim_span_item
        if item is None:
            item = _MontageStimSpanItem()
            self._glw.scene().addItem(item)
            self._stim_span_item = item
        elif item.scene() is None:
            self._glw.scene().addItem(item)
        return item

    def _clear_stim_spans(self) -> None:
        self._stim_span_timer.stop()
        self._stim_span_specs = []
        self._disconnect_stim_spans()
        item = self._stim_span_item
        if item is None:
            return
        try:
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
        except Exception:
            pass
        self._stim_span_item = None

    def _disconnect_stim_spans(self) -> None:
        for vb in self._stim_span_connected:
            try:
                vb.sigRangeChanged.disconnect(self._schedule_stim_span_update)
            except Exception:
                pass
            try:
                vb.sigResized.disconnect(self._schedule_stim_span_update)
            except Exception:
                pass
        self._stim_span_connected.clear()

    def _connect_stim_spans(self) -> None:
        self._disconnect_stim_spans()
        seen: set[int] = set()
        rows: list[int] = []
        for _x, _color, span_rows in self._stim_span_specs:
            rows.extend(int(r) for r in span_rows)
        for entry in list(self._stim_labels) + list(self._zone_labels):
            try:
                rows.append(self._plots.index(entry.plot))
            except ValueError:
                vb = entry.plot.getViewBox()
                try:
                    vb.sigRangeChanged.connect(self._schedule_stim_span_update)
                    self._stim_span_connected.append(vb)
                    vb.sigResized.connect(self._schedule_stim_span_update)
                except Exception:
                    pass
        # Zone / stim hors ViewBox : brancher aussi le bas de pile (dernier plot).
        if self._zone_labels and self._plots:
            rows.append(len(self._plots) - 1)
        for row in rows:
            if row < 0 or row >= len(self._plots) or row in seen:
                continue
            seen.add(row)
            vb = self._plots[row].getViewBox()
            try:
                vb.sigRangeChanged.connect(self._schedule_stim_span_update)
                self._stim_span_connected.append(vb)
            except Exception:
                continue
            try:
                vb.sigResized.connect(self._schedule_stim_span_update)
            except Exception:
                pass

    def _schedule_stim_span_update(self, *_args: Any) -> None:
        # Les libellés on/off (haut) et Zone (bas) suivent pan / zoom.
        if self._stim_labels:
            self._reposition_stim_labels()
        elif self._zone_labels:
            self._reposition_zone_labels()
        if not self._stim_span_specs:
            return
        if not self._stim_span_timer.isActive():
            self._stim_span_timer.start()

    def _stim_span_scene_point(
        self, plot: pg.PlotItem, x: float
    ) -> tuple[float, float, float] | None:
        """``(x scène, haut, bas)`` du ViewBox si ``x`` est visible."""
        vb = plot.getViewBox()
        try:
            rect = vb.sceneBoundingRect()
            (x0, x1), _y = vb.viewRange()
        except Exception:
            return None
        x0, x1 = float(x0), float(x1)
        if rect.width() < 2.0 or rect.height() < 2.0:
            return None
        if not np.isfinite([x0, x1, x]).all() or x0 == x1:
            return None
        lo, hi = (x0, x1) if x0 < x1 else (x1, x0)
        if x < lo or x > hi:
            return None
        frac = (float(x) - x0) / (x1 - x0)
        scene_x = float(rect.left()) + frac * float(rect.width())
        return (scene_x, float(rect.top()), float(rect.bottom()))

    def _update_stim_spans(self) -> None:
        """Recaler les pointillés sur les ViewBox (y compris l’espace entre canaux)."""
        self._stim_span_timer.stop()
        specs = self._stim_span_specs
        if not specs or not self._plots:
            if self._stim_span_item is not None:
                self._stim_span_item.set_lines([])
            return
        item = self._ensure_stim_span_item()
        lines: list[tuple[float, float, float, QPen]] = []
        align_px = 2.0
        stim_width = max(0.2, float(getattr(self, "_stim_span_width", 1.0) or 1.0))
        for x, color, rows in specs:
            pen = QPen(QColor(color))
            pen.setWidthF(stim_width)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            pending: list[tuple[float, float, float]] = []

            def flush() -> None:
                if not pending:
                    return
                xs = [point[0] for point in pending]
                if max(xs) - min(xs) <= align_px:
                    scene_x = sum(xs) / len(xs)
                    lines.append((scene_x, pending[0][1], pending[-1][2], pen))
                else:
                    for scene_x, top, bottom in pending:
                        lines.append((scene_x, top, bottom, pen))
                pending.clear()

            for row in rows:
                if row < 0 or row >= len(self._plots):
                    flush()
                    continue
                mapped = self._stim_span_scene_point(self._plots[row], float(x))
                if mapped is None:
                    flush()
                    continue
                if pending and abs(mapped[0] - pending[-1][0]) > align_px:
                    flush()
                pending.append(mapped)
            flush()
        item.set_lines(lines)
