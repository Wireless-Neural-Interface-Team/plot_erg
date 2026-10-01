"""Clickable MEA electrode map.

Draws the probe geometry natively (no matplotlib) so it stays responsive, and
turns a click or arrow key into a channel selection. Contacts can be shaded by a
per-channel metric such as mean RMS, which makes the map a quick way to spot the
channels worth looking at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

_UNMAPPED_FILL = QColor("#e2e8f0")
_UNMAPPED_EDGE = QColor("#cbd5e1")
_MAPPED_FILL = QColor("#93c5fd")
_MAPPED_EDGE = QColor("#64748b")
_SELECTED_EDGE = QColor("#dc2626")
_HOVER_EDGE = QColor("#2563eb")
_LABEL_COLOR = QColor("#0f172a")
_BACKGROUND = QColor("#f8fafc")

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


def _lerp_color(low: QColor, high: QColor, t: float) -> QColor:
    ratio = max(0.0, min(1.0, float(t)))
    return QColor(
        int(low.red() + (high.red() - low.red()) * ratio),
        int(low.green() + (high.green() - low.green()) * ratio),
        int(low.blue() + (high.blue() - low.blue()) * ratio),
    )


class MeaMapWidget(QWidget):
    """Electrode map that emits the channel name of the contact being clicked."""

    channelSelected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._contacts: list[_Contact] = []
        self._selected: str | None = None
        self._hover_index: int | None = None
        self._metric: dict[str, float] = {}
        self._metric_label = ""
        self._metric_range: tuple[float, float] = (0.0, 1.0)
        self._show_labels = True
        self._bounds_um: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
        self._contact_radius_px = 8.0
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ------------------------------------------------------------------ inputs

    def set_probe(self, layout: Any | None, channel_names: Sequence[str]) -> None:
        """Bind a :class:`probe_layout.ProbeLayout` to the recording channels."""
        self._contacts = []
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

    @property
    def has_probe(self) -> bool:
        return bool(self._contacts)

    def mapped_channels(self) -> list[str]:
        return [c.channel_name for c in self._contacts if c.channel_name]

    # ------------------------------------------------------------- geometry

    def _recompute_bounds(self) -> None:
        if not self._contacts:
            self._bounds_um = (0.0, 1.0, 0.0, 1.0)
            return
        xs = [c.x_um for c in self._contacts]
        ys = [c.y_um for c in self._contacts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad_x = max((x_max - x_min) * 0.08, 1.0)
        pad_y = max((y_max - y_min) * 0.08, 1.0)
        self._bounds_um = (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)

    def _plot_rect(self) -> QRectF:
        margin = 6.0
        rect = QRectF(self.rect()).adjusted(margin, margin, -margin, -margin)
        if self._metric_label:
            rect.setBottom(rect.bottom() - 16.0)
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

    def _to_widget(self, contact: _Contact, rect: QRectF) -> QPointF:
        x_min, x_max, y_min, y_max = self._bounds_um
        fx = (contact.x_um - x_min) / max(x_max - x_min, 1e-6)
        fy = (contact.y_um - y_min) / max(y_max - y_min, 1e-6)
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
            point = self._to_widget(contact, rect)
            dx = point.x() - position.x()
            dy = point.y() - position.y()
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance = distance
                best = contact
        reach = max(self._contact_radius_px * 1.9, 12.0)
        return best if best_distance <= reach * reach else None

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
                "No MEA probe loaded.\nSelect a probe JSON in the parameters.",
            )
            painter.end()
            return

        rect = self._plot_rect()
        n_cols = max(1, len({round(c.x_um, 3) for c in self._contacts}))
        n_rows = max(1, len({round(c.y_um, 3) for c in self._contacts}))
        cell = min(rect.width() / n_cols, rect.height() / n_rows)
        self._contact_radius_px = max(4.0, min(22.0, cell * 0.42))
        font_size = max(5.0, min(10.0, self._contact_radius_px * 0.95))
        font = QFont(self.font())
        font.setPointSizeF(font_size)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        label_fits = self._show_labels and metrics.height() <= cell * 0.92

        low, high = self._metric_range
        for contact in self._contacts:
            point = self._to_widget(contact, rect)
            radius = self._contact_radius_px
            mapped = contact.channel_name is not None
            if mapped and contact.channel_name in self._metric:
                value = self._metric[contact.channel_name]
                fill = _lerp_color(_METRIC_LOW, _METRIC_HIGH, (value - low) / max(high - low, 1e-9))
            elif mapped:
                fill = _MAPPED_FILL
            else:
                fill = _UNMAPPED_FILL
            edge = _MAPPED_EDGE if mapped else _UNMAPPED_EDGE
            width = 1.0
            if self._hover_index == contact.index:
                edge, width = _HOVER_EDGE, 2.0
            if mapped and contact.channel_name == self._selected:
                edge, width = _SELECTED_EDGE, 2.6
                radius *= 1.05
            painter.setBrush(fill)
            painter.setPen(QPen(edge, width))
            painter.drawEllipse(point, radius, radius)
            if label_fits:
                painter.setPen(QPen(_LABEL_COLOR))
                text_rect = QRectF(
                    point.x() - cell / 2.0,
                    point.y() - metrics.height() / 2.0,
                    cell,
                    metrics.height(),
                )
                painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignCenter), contact.contact_id)

        if self._metric_label:
            painter.setPen(QPen(QColor("#475569")))
            legend_rect = QRectF(
                6.0, float(self.height()) - 18.0, float(self.width()) - 12.0, 16.0
            )
            painter.drawText(
                legend_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"{self._metric_label}: {low:.2f} (light) → {high:.2f} (dark)",
            )
        painter.end()

    # ----------------------------------------------------------------- events

    def mousePressEvent(self, event: Any) -> None:  # noqa: D102
        contact = self._contact_at(QPointF(event.position()))
        if contact is not None and contact.channel_name:
            self.set_selected_channel(contact.channel_name)
            self.channelSelected.emit(contact.channel_name)
        super().mousePressEvent(event)

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
            self.setToolTip(f"{contact.contact_id} → {contact.channel_name}{detail}")
        else:
            self.setToolTip(f"{contact.contact_id} (not recorded)")
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

        return QSize(360, 300)
