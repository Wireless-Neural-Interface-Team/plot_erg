"""Canvas matplotlib réellement interactif (zoom molette, pan, curseur, home).

Tous les graphiques GUI passent par ce widget — plus d’affichage « image morte ».
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class CompactNavToolbar(NavigationToolbar2QT):
    """Barre de navigation compacte, toujours visible."""

    toolitems = [
        t
        for t in NavigationToolbar2QT.toolitems
        if t[0] in {"Home", "Back", "Forward", "Pan", "Zoom", "Save"}
    ]


class _ScopeCanvas(FigureCanvasQTAgg):
    """Canvas qui garde la molette pour le zoom (ne la laisse pas au scroll parent)."""

    def __init__(self, figure: Figure, owner: "InteractiveCanvas") -> None:
        super().__init__(figure)
        self._owner = owner

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        # Laisser matplotlib émettre scroll_event, puis consommer si on a zoomé.
        super().wheelEvent(event)
        if getattr(self._owner, "_did_zoom", False):
            event.accept()
            self._owner._did_zoom = False
        else:
            event.ignore()


class InteractiveCanvas(QWidget):
    """Figure + toolbar + interactions souris (molette, pan, curseur)."""

    def __init__(
        self,
        *,
        figsize: tuple[float, float] = (4.2, 2.8),
        parent: QWidget | None = None,
        show_toolbar: bool = True,
        show_cursor: bool = True,
        min_height: int = 120,
    ) -> None:
        super().__init__(parent)
        self._did_zoom = False
        self.figure = Figure(figsize=figsize, layout="constrained", facecolor="#ffffff")
        self.canvas = _ScopeCanvas(self.figure, self)
        self.canvas.setStyleSheet("background-color: #ffffff;")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumHeight(int(min_height))
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.setMouseTracking(True)

        self.toolbar = CompactNavToolbar(self.canvas, self)
        self.toolbar.setIconSize(self.toolbar.iconSize() * 0.75)
        self.toolbar.setVisible(bool(show_toolbar))
        self.toolbar.setMaximumHeight(28)

        self._cursor_label = QLabel(
            "Molette zoom · clic droit pan · double-clic reset", self
        )
        self._cursor_label.setObjectName("panelStatus")
        self._cursor_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._cursor_label.setVisible(bool(show_cursor))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self._cursor_label)

        self._press_ax: Any | None = None
        self._pan_origin: tuple[float, float] | None = None
        self._xlim0: tuple[float, float] | None = None
        self._ylim0: tuple[float, float] | None = None

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # ---------------------------------------------------------------- drawing

    def draw_idle(self) -> None:
        self.canvas.draw_idle()

    def enable_default_pan(self) -> None:
        """Activer le mode Pan de la toolbar après un redraw (si pas déjà actif)."""
        try:
            mode = getattr(self.toolbar, "mode", None)
            name = getattr(mode, "name", None) or str(mode or "")
            if str(name).upper() in {"PAN", "PAN/ZOOM"}:
                return
            if "pan" in str(name).lower():
                return
            self.toolbar.pan()
        except Exception:
            pass

    # ----------------------------------------------------------- interactions

    def _on_scroll(self, event: Any) -> None:
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            self._did_zoom = False
            return
        scale = 0.8 if getattr(event, "step", 0) > 0 else 1.25
        try:
            self._zoom_around(ax, float(event.xdata), float(event.ydata), scale)
            self.canvas.draw_idle()
            self._did_zoom = True
        except Exception:
            self._did_zoom = False

    @staticmethod
    def _zoom_around(ax: Any, x: float, y: float, scale: float) -> None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        new_xmin = x - (x - xmin) * scale
        new_xmax = x + (xmax - x) * scale
        new_ymin = y - (y - ymin) * scale
        new_ymax = y + (ymax - y) * scale
        if np.isfinite([new_xmin, new_xmax, new_ymin, new_ymax]).all() and new_xmin != new_xmax:
            ax.set_xlim(new_xmin, new_xmax)
            ax.set_ylim(new_ymin, new_ymax)

    def _on_press(self, event: Any) -> None:
        if getattr(event, "dblclick", False) and event.button == 1:
            try:
                self.toolbar.home()
            except Exception:
                pass
            return

        # Clic milieu ou droit : pan manuel.
        if event.button not in (2, 3) or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._press_ax = event.inaxes
        self._pan_origin = (float(event.xdata), float(event.ydata))
        self._xlim0 = event.inaxes.get_xlim()
        self._ylim0 = event.inaxes.get_ylim()
        self.canvas.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def _on_release(self, event: Any) -> None:
        del event
        self._press_ax = None
        self._pan_origin = None
        self._xlim0 = None
        self._ylim0 = None
        self.canvas.unsetCursor()

    def _on_motion(self, event: Any) -> None:
        if event.inaxes is not None and event.xdata is not None and event.ydata is not None:
            self._cursor_label.setText(
                f"t = {event.xdata:.4g}   y = {event.ydata:.4g}   "
                f"(molette zoom · clic droit pan · double-clic reset)"
            )
        elif self._press_ax is None:
            self._cursor_label.setText(
                "Molette zoom · clic droit pan · double-clic reset"
            )

        if self._press_ax is None or self._pan_origin is None or self._xlim0 is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Convertir le déplacement en données via le delta écran pour un pan stable.
        try:
            x0, y0 = self._pan_origin
            # Recalcule depuis le pixel pour éviter le feedback de set_xlim.
            inv = self._press_ax.transData.inverted()
            x1, y1 = inv.transform((event.x, event.y))
            dx = x1 - x0
            dy = y1 - y0
            # Origin doit être en coords au moment du press en display... 
            # Approche simple : utiliser event.xdata avec reset d'origin chaque frame
            # est instable. On utilise plutôt le delta depuis le press en pixels.
        except Exception:
            return

        # Pan basé sur pixels (robuste).
        try:
            display_origin = self._press_ax.transData.transform(self._pan_origin)
            cur = (float(event.x), float(event.y))
            dx_pix = cur[0] - display_origin[0]
            dy_pix = cur[1] - display_origin[1]
            inv = self._press_ax.transData.inverted()
            x_a, y_a = inv.transform(display_origin)
            x_b, y_b = inv.transform((display_origin[0] + dx_pix, display_origin[1] + dy_pix))
            dx = x_b - x_a
            dy = y_b - y_a
            self._press_ax.set_xlim(self._xlim0[0] - dx, self._xlim0[1] - dx)
            if self._ylim0 is not None:
                self._press_ax.set_ylim(self._ylim0[0] - dy, self._ylim0[1] - dy)
            self.canvas.draw_idle()
        except Exception:
            pass
