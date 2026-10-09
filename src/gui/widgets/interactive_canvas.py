"""Canvas matplotlib réellement interactif (zoom molette, pan, curseur, home).

Tous les graphiques GUI passent par ce widget — plus d’affichage « image morte ».
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

# Pan = clic milieu ou droit (le clic gauche reste libre pour les barres de plage).
INTERACTION_HINT = "Molette zoom · clic milieu/droit pan · double-clic reset"
INTERACTION_HINT_NO_ZOOM = "Clic milieu/droit = pan (zoom désactivé sur cet aperçu)"
_PAN_FALLBACK_MS = 16


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
        # "constrained" coûte cher avec beaucoup d’axes (montage) — passer None.
        layout: str | None = "constrained",
        allow_zoom: bool = True,
    ) -> None:
        super().__init__(parent)
        self._did_zoom = False
        self._allow_zoom = bool(allow_zoom)
        self.figure = Figure(figsize=figsize, layout=layout, facecolor="#ffffff")
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
        if not self._allow_zoom:
            # Masquer Zoom toolbar ; pan / home restent utiles.
            try:
                for action in self.toolbar.actions():
                    text = str(action.text() or action.iconText() or "")
                    if "zoom" in text.lower():
                        action.setVisible(False)
            except Exception:
                pass

        hint = INTERACTION_HINT if self._allow_zoom else INTERACTION_HINT_NO_ZOOM
        self._cursor_label = QLabel(hint, self)
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
        self._press_xy: tuple[float, float] | None = None
        self._xlim0: tuple[float, float] | None = None
        self._ylim0: tuple[float, float] | None = None
        self._pan_trans: Any | None = None
        # Blit cache for interactive pan/zoom (axes bbox snapshot).
        self._blit_bg: Any | None = None
        self._blit_ax: Any | None = None
        self._interaction_draw_pending = False
        self._interaction_timer = QTimer(self)
        self._interaction_timer.setSingleShot(True)
        self._interaction_timer.setInterval(_PAN_FALLBACK_MS)
        self._interaction_timer.timeout.connect(self._flush_interaction_draw)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # ---------------------------------------------------------------- drawing

    def draw_idle(self) -> None:
        self.invalidate_blit()
        self.canvas.draw_idle()

    def invalidate_blit(self) -> None:
        """Drop cached background after a full panel redraw."""
        self._blit_bg = None
        self._blit_ax = None

    def _try_blit(self, ax: Any) -> bool:
        """Fast path: redraw one axes via blit. Falls back if the backend rejects it."""
        try:
            renderer = self.canvas.get_renderer()
            if self._blit_bg is None or self._blit_ax is not ax:
                # Snapshot current frame, then we'll redraw on the next call.
                ax.figure.canvas.draw()
                self._blit_bg = renderer.copy_from_bbox(ax.bbox)
                self._blit_ax = ax
            self.canvas.restore_region(self._blit_bg)
            ax.draw_artist(ax)
            self.canvas.blit(ax.bbox)
            # Refresh cache for the next motion event (new view already drawn).
            self._blit_bg = renderer.copy_from_bbox(ax.bbox)
            self._blit_ax = ax
            return True
        except Exception:
            self._blit_bg = None
            self._blit_ax = None
            return False

    def _request_interaction_draw(self, ax: Any | None = None) -> None:
        """Prefer blit; otherwise coalesce draw_idle to ~60 Hz during pan/zoom."""
        if ax is not None and self._try_blit(ax):
            return
        self._interaction_draw_pending = True
        if not self._interaction_timer.isActive():
            self._interaction_timer.start()

    def _flush_interaction_draw(self) -> None:
        if not self._interaction_draw_pending:
            return
        self._interaction_draw_pending = False
        self.canvas.draw_idle()

    def capture_view_limits(
        self,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]] | None:
        """Snapshot des (xlim, ylim) de chaque axe — pour survivre à un redessin."""
        axes = list(self.figure.axes)
        if not axes:
            return None
        limits: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for ax in axes:
            try:
                xlim = (float(ax.get_xlim()[0]), float(ax.get_xlim()[1]))
                ylim = (float(ax.get_ylim()[0]), float(ax.get_ylim()[1]))
            except Exception:
                return None
            if not np.isfinite([*xlim, *ylim]).all() or xlim[0] == xlim[1]:
                return None
            limits.append((xlim, ylim))
        return limits

    def restore_view_limits(
        self,
        limits: list[tuple[tuple[float, float], tuple[float, float]]] | None,
    ) -> None:
        """Rétablir un snapshot de vue (ignore les axes en trop / manquants)."""
        if not limits:
            return
        axes = list(self.figure.axes)
        if not axes:
            return
        n = min(len(axes), len(limits))
        for index in range(n):
            xlim, ylim = limits[index]
            ax = axes[index]
            try:
                ax.set_xlim(xlim[0], xlim[1])
                if np.isfinite(ylim).all() and ylim[0] != ylim[1]:
                    ax.set_ylim(ylim[0], ylim[1])
            except Exception:
                pass
        # Même base temporelle si le nombre d’axes a changé (ex. flux ajouté).
        if len(axes) > n and limits:
            xlim0 = limits[0][0]
            for ax in axes[n:]:
                try:
                    ax.set_xlim(xlim0[0], xlim0[1])
                except Exception:
                    pass

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
        if not self._allow_zoom:
            self._did_zoom = False
            return
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            self._did_zoom = False
            return
        scale = 0.8 if getattr(event, "step", 0) > 0 else 1.25
        try:
            self._zoom_around(ax, float(event.xdata), float(event.ydata), scale)
            self._request_interaction_draw(ax)
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

        # Clic milieu ou droit : pan (pixels + transform figé au press).
        if event.button not in (2, 3) or event.inaxes is None:
            return
        if event.x is None or event.y is None:
            return
        self._press_ax = event.inaxes
        self._press_xy = (float(event.x), float(event.y))
        self._xlim0 = tuple(event.inaxes.get_xlim())
        self._ylim0 = tuple(event.inaxes.get_ylim())
        # Figé : sinon chaque set_xlim change le mapping pixel→données et le pan décroche.
        self._pan_trans = event.inaxes.transData.frozen()
        self.invalidate_blit()
        self.canvas.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def _on_release(self, event: Any) -> None:
        del event
        was_panning = self._press_ax is not None
        self._press_ax = None
        self._press_xy = None
        self._xlim0 = None
        self._ylim0 = None
        self._pan_trans = None
        self.canvas.unsetCursor()
        if was_panning:
            # Final crisp frame after the last coalesced/blit update.
            self.invalidate_blit()
            self.canvas.draw_idle()

    def _on_motion(self, event: Any) -> None:
        # Pendant un pan : suivre les pixels (pas besoin de xdata / inaxes).
        if (
            self._press_ax is not None
            and self._press_xy is not None
            and self._xlim0 is not None
            and self._ylim0 is not None
            and self._pan_trans is not None
        ):
            if event.x is None or event.y is None:
                return
            try:
                inv = self._pan_trans.inverted()
                x0, y0 = inv.transform(self._press_xy)
                x1, y1 = inv.transform((float(event.x), float(event.y)))
                dx = float(x1 - x0)
                dy = float(y1 - y0)
                self._press_ax.set_xlim(self._xlim0[0] - dx, self._xlim0[1] - dx)
                self._press_ax.set_ylim(self._ylim0[0] - dy, self._ylim0[1] - dy)
                self._request_interaction_draw(self._press_ax)
            except Exception:
                pass
            return

        if event.inaxes is not None and event.xdata is not None and event.ydata is not None:
            self._cursor_label.setText(
                f"t = {event.xdata:.4g}   y = {event.ydata:.4g}   ({INTERACTION_HINT})"
            )
        else:
            self._cursor_label.setText(INTERACTION_HINT)
