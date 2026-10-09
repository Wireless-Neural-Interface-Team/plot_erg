"""Clickable MEA electrode map.

Draws the probe geometry natively (no matplotlib) so it stays responsive, and
turns a click or arrow key into a channel selection. Contacts can be shaded by a
per-channel metric such as mean RMS, which makes the map a quick way to spot the
channels worth looking at.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

_UNMAPPED_FILL = QColor("#e2e8f0")
_UNMAPPED_EDGE = QColor("#94a3b8")
_MAPPED_FILL = QColor("#bfdbfe")
_MAPPED_EDGE = QColor("#475569")
_HIDDEN_FILL = QColor("#f1f5f9")
_HIDDEN_EDGE = QColor("#cbd5e1")
_SELECTED_EDGE = QColor("#dc2626")
_HOVER_EDGE = QColor("#2563eb")
_LABEL_COLOR = QColor("#0f172a")
_HIDDEN_LABEL = QColor("#94a3b8")
_BACKGROUND = QColor("#f8fafc")
_CLUSTER_FILL = QColor(226, 232, 240, 90)
_CLUSTER_EDGE = QColor("#cbd5e1")

# Low -> high gradient used when a per-channel metric is provided.
_METRIC_LOW = QColor("#dbeafe")
_METRIC_HIGH = QColor("#b91c1c")


@dataclass(frozen=True)
class _Contact:
    index: int
    contact_id: str
    x_um: float
    y_um: float
    channel_name: str | None
    short_label: str


def _lerp_color(low: QColor, high: QColor, t: float) -> QColor:
    ratio = max(0.0, min(1.0, float(t)))
    return QColor(
        int(low.red() + (high.red() - low.red()) * ratio),
        int(low.green() + (high.green() - low.green()) * ratio),
        int(low.blue() + (high.blue() - low.blue()) * ratio),
    )


def _short_label(contact_id: str) -> str:
    """Compact label that fits inside a contact circle (full id stays in tooltip)."""
    raw = str(contact_id).strip()
    if not raw:
        return "?"
    # A-033 / A_033 / amp-A-033 → keep letter + number when possible.
    match = re.search(r"([A-Za-z]+)\s*[-_]?\s*(\d+)\s*$", raw)
    if match:
        letter = match.group(1)
        number = match.group(2).lstrip("0") or "0"
        if len(letter) <= 2 and len(number) <= 3:
            return f"{letter}{number}"
        if len(number) <= 3:
            return number
    # Pure numeric id.
    digits = re.sub(r"\D+", "", raw)
    if digits:
        compact = digits.lstrip("0") or "0"
        return compact[-3:] if len(compact) > 3 else compact
    return raw[-4:] if len(raw) > 4 else raw


def _median_nearest_pitch_um(contacts: Sequence[_Contact]) -> float:
    """Typical local electrode pitch — robust for multi-cluster probes."""
    if len(contacts) < 2:
        return 1.0
    pitches: list[float] = []
    for i, contact in enumerate(contacts):
        best = float("inf")
        for j, other in enumerate(contacts):
            if i == j:
                continue
            dx = contact.x_um - other.x_um
            dy = contact.y_um - other.y_um
            dist = math.hypot(dx, dy)
            if 0.0 < dist < best:
                best = dist
        if best < float("inf"):
            pitches.append(best)
    if not pitches:
        return 1.0
    pitches.sort()
    return float(pitches[len(pitches) // 2])


def _cluster_bounds(
    contacts: Sequence[_Contact], gap_um: float
) -> list[tuple[float, float, float, float]]:
    """Axis-aligned bounds of contact clusters separated by more than ``gap_um``."""
    if not contacts:
        return []
    parent = list(range(len(contacts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(contacts):
        for j in range(i + 1, len(contacts)):
            b = contacts[j]
            if math.hypot(a.x_um - b.x_um, a.y_um - b.y_um) <= gap_um:
                union(i, j)

    groups: dict[int, list[_Contact]] = {}
    for i, contact in enumerate(contacts):
        groups.setdefault(find(i), []).append(contact)

    bounds: list[tuple[float, float, float, float]] = []
    pad = max(gap_um * 0.35, 1.0)
    for members in groups.values():
        xs = [c.x_um for c in members]
        ys = [c.y_um for c in members]
        bounds.append(
            (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
        )
    return bounds


class MeaMapWidget(QWidget):
    """Electrode map that emits the channel name of the contact being clicked."""

    channelSelected = Signal(str)
    channelActivated = Signal(str)  # double-clic → Analyse le canal

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._contacts: list[_Contact] = []
        self._selected: str | None = None
        self._hover_index: int | None = None
        self._hidden: set[str] = set()
        self._metric: dict[str, float] = {}
        self._metric_label = ""
        self._metric_range: tuple[float, float] = (0.0, 1.0)
        self._show_labels = True
        self._bounds_um: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
        self._pitch_um = 1.0
        self._cluster_bounds_um: list[tuple[float, float, float, float]] = []
        self._contact_radius_px = 8.0
        self._label_fit_cache: dict[tuple[int, int], tuple[str, float] | None] = {}
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ------------------------------------------------------------------ inputs

    def set_probe(self, layout: Any | None, channel_names: Sequence[str]) -> None:
        """Bind a :class:`probe_layout.ProbeLayout` to the recording channels."""
        self._contacts = []
        self._label_fit_cache.clear()
        if layout is not None:
            from probe_layout import _is_nc_contact, _contact_name_matches_rhs

            positions = layout.positions_um
            for index, contact_id in enumerate(layout.contact_ids):
                cid = str(contact_id).strip()
                if _is_nc_contact(cid):
                    continue
                matched: str | None = None
                for name in channel_names:
                    if _contact_name_matches_rhs(cid, str(name)):
                        matched = str(name)
                        break
                self._contacts.append(
                    _Contact(
                        index=index,
                        contact_id=cid,
                        x_um=float(positions[index, 0]),
                        y_um=float(positions[index, 1]),
                        channel_name=matched,
                        short_label=_short_label(cid),
                    )
                )
        self._recompute_bounds()
        self.update()

    def set_selected_channel(self, channel_name: str | None) -> None:
        if channel_name == self._selected:
            return
        self._selected = channel_name
        self.update()

    def set_metric(self, values: Mapping[str, float] | None, label: str = "") -> None:
        """Shade contacts by a per-channel value (e.g. mean RMS in µV)."""
        self._metric = {str(k): float(v) for k, v in (values or {}).items()}
        self._metric_label = label
        finite = [v for v in self._metric.values() if v == v]
        if finite:
            low, high = min(finite), max(finite)
            self._metric_range = (low, high if high > low else low + 1.0)
        else:
            self._metric_range = (0.0, 1.0)
        self.update()

    def set_labels_visible(self, visible: bool) -> None:
        self._show_labels = bool(visible)
        self.update()

    def set_hidden_channels(self, names: Sequence[str] | set[str] | None) -> None:
        """Atténuer les contacts masqués dans le montage."""
        hidden = {str(name) for name in (names or ())}
        if hidden == self._hidden:
            return
        self._hidden = hidden
        self.update()

    @property
    def has_probe(self) -> bool:
        return bool(self._contacts)

    def mapped_channels(self) -> list[str]:
        return [c.channel_name for c in self._contacts if c.channel_name]

    # ------------------------------------------------------------- geometry

    def _recompute_bounds(self) -> None:
        if not self._contacts:
            self._bounds_um = (0.0, 1.0, 0.0, 1.0)
            self._pitch_um = 1.0
            self._cluster_bounds_um = []
            return
        xs = [c.x_um for c in self._contacts]
        ys = [c.y_um for c in self._contacts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        self._pitch_um = _median_nearest_pitch_um(self._contacts)
        pad = max(self._pitch_um * 0.75, (x_max - x_min) * 0.06, (y_max - y_min) * 0.06, 1.0)
        self._bounds_um = (x_min - pad, x_max + pad, y_min - pad, y_max + pad)
        # Clusters séparés par > ~2.8 pas locaux.
        self._cluster_bounds_um = _cluster_bounds(self._contacts, self._pitch_um * 2.8)

    def _legend_height(self) -> float:
        """Espace réservé en bas pour la légende couleurs / nuances."""
        if not self._contacts:
            return 0.0
        # Gradient + valeurs + pastilles d’état.
        return 52.0 if self._metric_label else 28.0

    def _plot_rect(self) -> QRectF:
        margin = 8.0
        legend_h = self._legend_height()
        rect = QRectF(self.rect()).adjusted(margin, margin, -margin, -margin)
        if legend_h > 0.0:
            rect.setBottom(rect.bottom() - legend_h)
        x_min, x_max, y_min, y_max = self._bounds_um
        span_x = max(x_max - x_min, 1e-6)
        span_y = max(y_max - y_min, 1e-6)
        aspect = span_x / span_y
        width, height = rect.width(), rect.height()
        if width / max(height, 1e-6) > aspect:
            new_width = height * aspect
            rect.setLeft(rect.left() + (width - new_width) / 2.0)
            rect.setWidth(new_width)
        else:
            new_height = width / aspect
            rect.setTop(rect.top() + (height - new_height) / 2.0)
            rect.setHeight(new_height)
        return rect

    def _um_to_px_scale(self, rect: QRectF) -> float:
        x_min, x_max, y_min, y_max = self._bounds_um
        sx = rect.width() / max(x_max - x_min, 1e-6)
        sy = rect.height() / max(y_max - y_min, 1e-6)
        return min(sx, sy)

    def _to_widget(self, x_um: float, y_um: float, rect: QRectF) -> QPointF:
        x_min, x_max, y_min, y_max = self._bounds_um
        fx = (x_um - x_min) / max(x_max - x_min, 1e-6)
        fy = (y_um - y_min) / max(y_max - y_min, 1e-6)
        return QPointF(
            rect.left() + fx * rect.width(),
            rect.bottom() - fy * rect.height(),
        )

    def _contact_at(self, position: QPointF) -> _Contact | None:
        if not self._contacts:
            return None
        rect = self._plot_rect()
        best: _Contact | None = None
        best_distance = float("inf")
        for contact in self._contacts:
            point = self._to_widget(contact.x_um, contact.y_um, rect)
            dx = point.x() - position.x()
            dy = point.y() - position.y()
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance = distance
                best = contact
        reach = max(self._contact_radius_px * 1.85, 11.0)
        return best if best_distance <= reach * reach else None

    def _label_candidates(self, contact: _Contact) -> list[str]:
        """Progressively shorter labels until one fits the disk."""
        primary = contact.short_label
        candidates = [primary]
        digits = re.sub(r"\D+", "", primary)
        if digits and digits != primary:
            candidates.append(digits)
        if digits and len(digits) > 2:
            candidates.append(digits[-2:])
        if digits:
            candidates.append(digits[-1:])
        # Unique-ish fallback from contact index when nothing else fits.
        candidates.append(str(contact.index % 100))
        seen: set[str] = set()
        out: list[str] = []
        for item in candidates:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _fit_label(
        self, contact: _Contact, diameter: float, base: QFont
    ) -> tuple[str, QFont] | None:
        """Pick a short label + font that fit inside the contact disk."""
        # Sous ~11 px de diamètre, le texte devient illisible : tooltip seulement.
        if diameter < 11.0:
            return None
        bucket = int(diameter * 2.0)  # ~0.5 px buckets — stable across hover paints
        cache_key = (contact.index, bucket)
        cached = self._label_fit_cache.get(cache_key)
        if cached is not None or cache_key in self._label_fit_cache:
            if cached is None:
                return None
            label, size = cached
            font = QFont(base)
            font.setBold(True)
            font.setPointSizeF(size)
            return label, font
        max_size = min(11.0, max(6.0, diameter * 0.50))
        min_size = 6.0
        if max_size < min_size:
            self._label_fit_cache[cache_key] = None
            return None
        for label in self._label_candidates(contact):
            size = max_size
            font = QFont(base)
            font.setBold(True)
            while size >= min_size - 1e-6:
                font.setPointSizeF(size)
                metrics = QFontMetricsF(font)
                max_w = diameter * 0.90
                max_h = diameter * 0.78
                if metrics.horizontalAdvance(label) <= max_w and metrics.height() <= max_h:
                    self._label_fit_cache[cache_key] = (label, float(size))
                    return label, font
                size -= 0.5
        self._label_fit_cache[cache_key] = None
        return None

    # --------------------------------------------------------------- painting

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: D102
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), _BACKGROUND)
        if not self._contacts:
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.drawText(
                QRectF(self.rect()),
                int(Qt.AlignmentFlag.AlignCenter),
                "Aucun mapping chargé.\n"
                "Utilisez Charger… juste au-dessus\n"
                "pour ouvrir un JSON de sonde MEA.",
            )
            painter.end()
            return

        rect = self._plot_rect()
        scale = self._um_to_px_scale(rect)
        pitch_px = max(self._pitch_um * scale, 1.0)
        # Cercles un peu plus petits que le pas local (lisibilité sans chevauchement).
        self._contact_radius_px = max(5.0, min(17.0, pitch_px * 0.40))
        diameter = self._contact_radius_px * 2.0

        # Fond léger par grappe (aide à lire les blocs séparés).
        if len(self._cluster_bounds_um) > 1:
            for x0, x1, y0, y1 in self._cluster_bounds_um:
                p0 = self._to_widget(x0, y1, rect)
                p1 = self._to_widget(x1, y0, rect)
                cluster_rect = QRectF(p0, p1).normalized().adjusted(-4.0, -4.0, 4.0, 4.0)
                painter.setBrush(_CLUSTER_FILL)
                painter.setPen(QPen(_CLUSTER_EDGE, 1.0))
                painter.drawRoundedRect(cluster_rect, 6.0, 6.0)

        base_font = QFont(self.font())
        low, high = self._metric_range
        for contact in self._contacts:
            point = self._to_widget(contact.x_um, contact.y_um, rect)
            radius = self._contact_radius_px
            mapped = contact.channel_name is not None
            is_hidden = bool(mapped and contact.channel_name in self._hidden)
            if is_hidden:
                fill, edge = _HIDDEN_FILL, _HIDDEN_EDGE
            elif mapped and contact.channel_name in self._metric:
                value = self._metric[contact.channel_name]
                fill = _lerp_color(_METRIC_LOW, _METRIC_HIGH, (value - low) / max(high - low, 1e-9))
                edge = _MAPPED_EDGE
            elif mapped:
                fill, edge = _MAPPED_FILL, _MAPPED_EDGE
            else:
                fill, edge = _UNMAPPED_FILL, _UNMAPPED_EDGE
            width = 1.1
            if self._hover_index == contact.index:
                edge, width = _HOVER_EDGE, 2.0
            if mapped and contact.channel_name == self._selected:
                edge, width = _SELECTED_EDGE, 2.4
                radius *= 1.08
            painter.setBrush(fill)
            painter.setPen(QPen(edge, width))
            painter.drawEllipse(point, radius, radius)

            if not self._show_labels:
                continue
            fitted = self._fit_label(contact, diameter, base_font)
            if fitted is None:
                continue
            label, font = fitted
            painter.setFont(font)
            metrics = QFontMetricsF(font)
            text_rect = QRectF(
                point.x() - radius,
                point.y() - metrics.height() / 2.0,
                radius * 2.0,
                metrics.height(),
            )
            painter.setPen(QPen(_HIDDEN_LABEL if is_hidden else _LABEL_COLOR))
            painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignCenter), label)

        self._draw_legend(painter)
        painter.end()

    def _draw_legend(self, painter: QPainter) -> None:
        """Légende : nuances de la métrique + signification des couleurs d’état."""
        if not self._contacts:
            return
        legend_h = self._legend_height()
        if legend_h <= 0.0:
            return

        left = 8.0
        right = float(self.width()) - 8.0
        width = max(right - left, 1.0)
        top = float(self.height()) - legend_h
        text_color = QColor("#475569")
        legend_font = QFont(self.font())
        legend_font.setPointSizeF(max(7.5, min(9.0, legend_font.pointSizeF())))
        painter.setFont(legend_font)
        metrics = QFontMetricsF(legend_font)
        y = top + 2.0

        if self._metric_label:
            low, high = self._metric_range
            title = self._metric_label
            painter.setPen(QPen(text_color))
            painter.drawText(
                QRectF(left, y, width, metrics.height()),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                title,
            )
            y += metrics.height() + 2.0
            bar_h = 10.0
            bar_rect = QRectF(left, y, width, bar_h)
            gradient = QLinearGradient(bar_rect.topLeft(), bar_rect.topRight())
            gradient.setColorAt(0.0, _METRIC_LOW)
            gradient.setColorAt(1.0, _METRIC_HIGH)
            painter.setBrush(gradient)
            painter.setPen(QPen(QColor("#94a3b8"), 1.0))
            painter.drawRoundedRect(bar_rect, 3.0, 3.0)
            y += bar_h + 1.0
            painter.setPen(QPen(text_color))
            value_h = metrics.height()
            painter.drawText(
                QRectF(left, y, width * 0.5, value_h),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"{low:.2f}",
            )
            painter.drawText(
                QRectF(left + width * 0.5, y, width * 0.5, value_h),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                f"{high:.2f}",
            )
            y += value_h + 3.0
        else:
            y += 4.0

        # Pastilles d’état (toujours visibles quand une sonde est chargée).
        swatch_r = 4.5
        gap = 10.0
        items: list[tuple[QColor, QColor, float, str]] = []
        if self._metric_label:
            items.append((_METRIC_LOW, _MAPPED_EDGE, 1.1, "faible"))
            items.append((_METRIC_HIGH, _MAPPED_EDGE, 1.1, "élevé"))
        else:
            items.append((_MAPPED_FILL, _MAPPED_EDGE, 1.1, "mappé"))
        items.extend(
            [
                (_UNMAPPED_FILL, _UNMAPPED_EDGE, 1.1, "non mappé"),
                (_HIDDEN_FILL, _HIDDEN_EDGE, 1.1, "masqué"),
                (_MAPPED_FILL, _SELECTED_EDGE, 2.2, "sélectionné"),
            ]
        )

        x = left + swatch_r
        baseline = y + max(swatch_r, metrics.height() / 2.0)
        for fill, edge, edge_w, label in items:
            text_w = metrics.horizontalAdvance(label)
            needed = swatch_r * 2.0 + 4.0 + text_w + gap
            if x + needed - swatch_r > right and x > left + swatch_r + 1.0:
                break
            painter.setBrush(fill)
            painter.setPen(QPen(edge, edge_w))
            painter.drawEllipse(QPointF(x, baseline), swatch_r, swatch_r)
            painter.setPen(QPen(text_color))
            text_rect = QRectF(
                x + swatch_r + 4.0,
                baseline - metrics.height() / 2.0,
                text_w + 2.0,
                metrics.height(),
            )
            painter.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                label,
            )
            x = text_rect.right() + gap + swatch_r

    # ----------------------------------------------------------------- events

    def mousePressEvent(self, event: Any) -> None:  # noqa: D102
        contact = self._contact_at(QPointF(event.position()))
        if contact is not None and contact.channel_name:
            self.set_selected_channel(contact.channel_name)
            self.channelSelected.emit(contact.channel_name)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: D102
        contact = self._contact_at(QPointF(event.position()))
        if contact is not None and contact.channel_name:
            self.set_selected_channel(contact.channel_name)
            self.channelSelected.emit(contact.channel_name)
            self.channelActivated.emit(contact.channel_name)
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: D102
        contact = self._contact_at(QPointF(event.position()))
        index = contact.index if contact is not None else None
        if index != self._hover_index:
            self._hover_index = index
            self.update()
        if contact is None:
            self.setToolTip("")
        elif contact.channel_name:
            metric = self._metric.get(contact.channel_name)
            detail = (
                f"\n{self._metric_label}: {metric:.3f}"
                if metric is not None and self._metric_label
                else ""
            )
            hidden = "\n(masqué dans le montage)" if contact.channel_name in self._hidden else ""
            self.setToolTip(f"{contact.contact_id} → {contact.channel_name}{detail}{hidden}")
        else:
            self.setToolTip(f"{contact.contact_id} (non enregistré)")
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: D102
        self._hover_index = None
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: Any) -> None:  # noqa: D102
        mapped = [c for c in self._contacts if c.channel_name]
        if not mapped:
            super().keyPressEvent(event)
            return
        names = [c.channel_name for c in mapped]
        try:
            current = names.index(self._selected) if self._selected in names else 0
        except ValueError:
            current = 0
        step = {
            Qt.Key.Key_Right: 1,
            Qt.Key.Key_Down: 1,
            Qt.Key.Key_Left: -1,
            Qt.Key.Key_Up: -1,
        }.get(event.key())
        if step is None:
            super().keyPressEvent(event)
            return
        target = names[(current + step) % len(names)]
        self.set_selected_channel(target)
        self.channelSelected.emit(str(target))

    def sizeHint(self):  # noqa: D102
        from PySide6.QtCore import QSize

        # Compact : un sizeHint trop large force le dock Session et mange la marge droite.
        return QSize(220, 220)
