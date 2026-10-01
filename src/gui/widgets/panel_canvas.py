"""Panneau graphique interactif (matplotlib Qt) — zoom, pan, curseur toujours actifs."""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.interactive_canvas import InteractiveCanvas
from panel_registry import RenderRequest, panel_info
from view_config import PanelPlacement

_STATUS_COLORS = {
    "ok": "#16a34a",
    "empty": "#ca8a04",
    "unavailable": "#dc2626",
    "pending": "#64748b",
}


class PanelCanvas(QFrame):
    """Un panneau : en-tête + canvas interactif (toolbar toujours visible)."""

    removeRequested = Signal(object)  # PanelPlacement
    detachRequested = Signal(object)  # PanelPlacement

    def __init__(self, placement: PanelPlacement, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.placement = placement
        self.setObjectName("panelCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._dirty = True
        self._last_render_s = 0.0
        self._status = "pending"

        info = panel_info(placement.panel)
        self._plot = InteractiveCanvas(
            figsize=(4.2, 2.8),
            parent=self,
            show_toolbar=True,
            show_cursor=True,
            min_height=max(120, min(220, info.preferred_height_px // 3)),
        )
        self.figure = self._plot.figure
        self.canvas = self._plot.canvas
        self.toolbar = self._plot.toolbar

        self._title = QLabel(placement.title())
        self._title.setObjectName("panelTitle")
        self._title.setToolTip(
            f"{placement.title()}\n"
            "Molette = zoom · Clic droit = pan · Double-clic = reset · "
            "Boutons Home / Pan / Zoom dans la barre"
        )
        self._title.setWordWrap(False)
        self._title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._status_label = QLabel("—")
        self._status_label.setObjectName("panelStatus")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._detach_button = self._make_tool_button("⤢", "Ouvrir ce panneau dans sa propre fenêtre")
        self._detach_button.clicked.connect(lambda: self.detachRequested.emit(self.placement))
        self._close_button = self._make_tool_button("✕", "Retirer ce panneau de la vue")
        self._close_button.clicked.connect(lambda: self.removeRequested.emit(self.placement))

        header = QHBoxLayout()
        header.setContentsMargins(8, 4, 4, 0)
        header.setSpacing(4)
        header.addWidget(self._title, 1)
        header.addWidget(self._status_label, 0)
        header.addWidget(self._detach_button, 0)
        header.addWidget(self._close_button, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 4)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self._plot, 1)

    def _make_tool_button(self, text: str, tooltip: str) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName("panelToolButton")
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    # ----------------------------------------------------------------- drawing

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def invalidate(self) -> None:
        self._dirty = True

    def mark_pending(self) -> None:
        self._set_status("pending", "redraw pending")

    def mark_offscreen(self) -> None:
        self._set_status("pending", "scroll to draw")

    def render(self, request: RenderRequest) -> str:
        """Dessiner le panneau et activer les interactions."""
        from gui.mpl_theme import apply_intan_scope_style
        from panel_registry import render_panel

        started = time.perf_counter()
        try:
            status = render_panel(self.figure, request)
            apply_intan_scope_style(self.figure)
        except Exception as exc:
            self.figure.clear()
            axes = self.figure.add_subplot(111)
            axes.text(
                0.5,
                0.5,
                f"Render error:\n{exc}",
                ha="center",
                va="center",
                transform=axes.transAxes,
                fontsize=8,
                color="#ff6b6b",
                wrap=True,
            )
            axes.set_axis_off()
            apply_intan_scope_style(self.figure)
            status = "unavailable"
        self._last_render_s = time.perf_counter() - started
        self._plot.draw_idle()
        self._plot.enable_default_pan()
        self._dirty = False
        self._set_status(status, f"{self._last_render_s * 1000:.0f} ms")
        return status

    def _set_status(self, status: str, text: str) -> None:
        self._status = status
        color = _STATUS_COLORS.get(status, "#64748b")
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color};")
        tooltip = {
            "ok": "Interactif — molette zoom, clic droit pan, double-clic reset",
            "empty": "Rien à dessiner avec les réglages actuels",
            "unavailable": "Données indisponibles pour ce panneau",
            "pending": "Dessiné à l’entrée dans la vue",
        }.get(status, status)
        self._status_label.setToolTip(f"{tooltip} — dernier rendu {self._last_render_s * 1000:.0f} ms")

    @property
    def last_render_s(self) -> float:
        return self._last_render_s


class DetachedPanelWindow(QDialog):
    """Panneau détaché : même canvas interactif."""

    closed = Signal(object)  # PanelPlacement

    def __init__(self, placement: PanelPlacement, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.placement = placement
        info = panel_info(placement.panel)
        self.setWindowTitle(placement.title())
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(960, max(440, info.preferred_height_px + 140))

        self._plot = InteractiveCanvas(
            figsize=(8.0, 5.0),
            parent=self,
            show_toolbar=True,
            show_cursor=True,
            min_height=280,
        )
        self.figure = self._plot.figure
        self.canvas = self._plot.canvas

        self._status = QLabel("—")
        self._status.setObjectName("panelStatus")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        header = QHBoxLayout()
        header.addWidget(QLabel(placement.title()), 1)
        header.addWidget(self._status, 0)
        layout.addLayout(header)
        layout.addWidget(self._plot, 1)

    def render(self, request: RenderRequest) -> str:
        from gui.mpl_theme import apply_intan_scope_style
        from panel_registry import render_panel

        started = time.perf_counter()
        try:
            status = render_panel(self.figure, request)
            apply_intan_scope_style(self.figure)
        except Exception as exc:
            self.figure.clear()
            axes = self.figure.add_subplot(111)
            axes.text(
                0.5,
                0.5,
                f"Render error:\n{exc}",
                ha="center",
                va="center",
                transform=axes.transAxes,
                fontsize=10,
                color="#ff6b6b",
                wrap=True,
            )
            axes.set_axis_off()
            apply_intan_scope_style(self.figure)
            status = "unavailable"
        elapsed = time.perf_counter() - started
        self._plot.draw_idle()
        self._plot.enable_default_pan()
        self._status.setText(f"{elapsed * 1000:.0f} ms")
        self._status.setStyleSheet(f"color: {_STATUS_COLORS.get(status, '#64748b')};")
        return status

    def closeEvent(self, event: Any) -> None:  # noqa: D102
        self.closed.emit(self.placement)
        super().closeEvent(event)
