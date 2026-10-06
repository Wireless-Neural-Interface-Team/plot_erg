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
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.interactive_canvas import InteractiveCanvas
from gui.widgets.view_params import LocalViewParams
from panel_registry import RenderRequest, panel_info
from view_config import PanelPlacement, ViewerSettings

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

    def update_placement(self, placement: PanelPlacement) -> None:
        """Mettre à jour le placement (ex. bornes de plage) sans recréer le widget."""
        self.placement = placement
        title = placement.title()
        self._title.setText(title)
        self._title.setToolTip(
            f"{title}\n"
            "Molette = zoom · Clic droit = pan · Double-clic = reset · "
            "Boutons Home / Pan / Zoom dans la barre"
        )
        self._dirty = True

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
            # Montage multi-axes : éviter le post-pass O(n_axes) (freeze à chaque coche).
            if self.placement.panel != "montage_continuous_raw":
                apply_intan_scope_style(
                    self.figure,
                    grid=request.style.grid,
                    grid_alpha=request.style.grid_alpha,
                    show_borders=getattr(request.style, "show_borders", None),
                    ticks_inside=getattr(request.style, "ticks_inside", None),
                )
            else:
                try:
                    from gui.theme import SCOPE_BG

                    self.figure.patch.set_facecolor(SCOPE_BG)
                except Exception:
                    pass
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
            apply_intan_scope_style(self.figure, grid=False, show_borders=False)
            status = "unavailable"
        self._last_render_s = time.perf_counter() - started
        self._plot.draw_idle()
        # Laisser le clic gauche aux barres de plage (pan = clic droit / molette zoom).
        if self.placement.panel not in {"montage_continuous_raw", "full_recording"}:
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
    """Panneau détaché : canvas interactif + légende/style locaux."""

    closed = Signal(object)  # PanelPlacement
    refreshRequested = Signal(object)  # DetachedPanelWindow

    def __init__(
        self,
        placement: PanelPlacement,
        settings: ViewerSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.placement = placement
        self.settings = settings
        info = panel_info(placement.panel)
        self.setWindowTitle(placement.title())
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(1100, max(480, info.preferred_height_px + 160))

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

        self.params = LocalViewParams(
            settings,
            show_analysis=False,
            show_continuous=False,
            show_display_extras=True,
            parent=self,
        )
        self.params.changed.connect(self._on_local_params_changed)

        plot_side = QWidget(self)
        plot_layout = QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.setSpacing(4)
        header = QHBoxLayout()
        header.addWidget(QLabel(placement.title()), 1)
        header.addWidget(self._status, 0)
        plot_layout.addLayout(header)
        plot_layout.addWidget(self._plot, 1)

        params_side = QWidget(self)
        params_layout = QVBoxLayout(params_side)
        params_layout.setContentsMargins(6, 6, 6, 6)
        params_layout.addWidget(self.params, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(plot_side)
        splitter.addWidget(params_side)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([780, 280])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def local_settings(self) -> ViewerSettings:
        return self.params.settings()

    def _on_local_params_changed(self) -> None:
        self.settings = self.local_settings()
        self.refreshRequested.emit(self)

    def render(self, request: RenderRequest) -> str:
        from gui.mpl_theme import apply_intan_scope_style
        from panel_registry import render_panel

        started = time.perf_counter()
        try:
            status = render_panel(self.figure, request)
            apply_intan_scope_style(
                self.figure,
                grid=request.style.grid,
                grid_alpha=request.style.grid_alpha,
                show_borders=request.style.show_borders,
                ticks_inside=request.style.ticks_inside,
            )
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
            apply_intan_scope_style(self.figure, grid=False, show_borders=False)
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
