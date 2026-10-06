"""Barres de plage temporelle déplaçables sur un canvas matplotlib."""

from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import Any, Sequence

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from view_config import TimeRangeBar

_BAR_COLORS = ("#16a34a", "#2563eb", "#ca8a04", "#dc2626", "#7c3aed")


class RangeBarToolbar(QWidget):
    """Boutons pour ajouter / supprimer / activer une plage et lancer le traitement."""

    addRequested = Signal()
    addRelativeRequested = Signal()
    removeRequested = Signal()
    activeChanged = Signal(int)
    processRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._count_label = QLabel("0 plage(s)", self)
        self._active = QSpinBox(self)
        self._active.setMinimum(1)
        self._active.setMaximum(1)
        self._active.setValue(1)
        self._active.setToolTip("Plage active (utilisée pour le zoom / traitement)")
        self._active.valueChanged.connect(lambda v: self.activeChanged.emit(int(v) - 1))

        btn_add = QPushButton("+ Plage", self)
        btn_add.setToolTip(
            "Ajouter une paire de barres en temps absolu (sur la trace continue)"
        )
        btn_add.clicked.connect(self.addRequested.emit)
        btn_add_rel = QPushButton("+ Rel. stim", self)
        btn_add_rel.setToolTip(
            "Ajouter une plage [t₀, t₁] relative au début de stimulation "
            "(mode moyenne / une stim)"
        )
        btn_add_rel.clicked.connect(self.addRelativeRequested.emit)
        btn_remove = QPushButton("− Plage", self)
        btn_remove.setToolTip("Supprimer la plage active")
        btn_remove.clicked.connect(self.removeRequested.emit)
        btn_process = QPushButton("Appliquer les zooms", self)
        btn_process.setObjectName("primaryButton")
        btn_process.setToolTip(
            "Créer un zoom (et les graphs cochés) pour chaque plage — "
            "ne relance pas le traitement F5"
        )
        btn_process.clicked.connect(self.processRequested.emit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._count_label)
        layout.addWidget(QLabel("Active", self))
        layout.addWidget(self._active)
        layout.addWidget(btn_add)
        layout.addWidget(btn_add_rel)
        layout.addWidget(btn_remove)
        layout.addWidget(btn_process)
        layout.addStretch(1)

    def set_bar_count(self, count: int, active_index: int = 0) -> None:
        n = max(0, int(count))
        self._count_label.setText(f"{n} plage(s)")
        self._active.blockSignals(True)
        self._active.setMaximum(max(1, n))
        self._active.setEnabled(n > 0)
        if n > 0:
            self._active.setValue(max(1, min(n, int(active_index) + 1)))
        self._active.blockSignals(False)


class RangeBarController(QObject):
    """Gère des paires de barres verticales déplaçables sur une figure matplotlib.

    - Clic gauche près d’une barre + glisser : déplace cette barre
    - Les deux barres d’une plage définissent [t0, t1]
    - ``barsChanged`` émis pendant le glisser (debouncé côté parent si besoin)
    """

    barsChanged = Signal(object)  # tuple[TimeRangeBar, ...]

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bars: list[TimeRangeBar] = []
        self._active_index = 0
        self._canvas: Any | None = None
        self._axes: list[Any] = []
        self._artists: list[Any] = []
        self._cid_press: int | None = None
        self._cid_release: int | None = None
        self._cid_motion: int | None = None
        self._drag: tuple[int, str] | None = None  # (bar_index, "t0"|"t1")
        self._t_min = 0.0
        self._t_max = 1.0
        self._pick_tol_frac = 0.012
        self._last_draw_s = 0.0
        self._motion_dirty = False

    @property
    def bars(self) -> tuple[TimeRangeBar, ...]:
        return tuple(self._bars)

    @property
    def active_index(self) -> int:
        return self._active_index

    def set_active_index(self, index: int) -> None:
        if not self._bars:
            self._active_index = 0
            return
        self._active_index = max(0, min(len(self._bars) - 1, int(index)))
        self._redraw_artists()

    def set_time_span(self, t_min: float, t_max: float) -> None:
        self._t_min = float(t_min)
        self._t_max = float(t_max)
        if self._t_max <= self._t_min:
            self._t_max = self._t_min + 1.0

    def ensure_default_bar(self) -> None:
        if self._bars:
            return
        self._bars = [
            TimeRangeBar(
                t0_s=self._t_min,
                t1_s=self._t_max,
                label="Plage 1",
                bar_id=uuid.uuid4().hex[:8],
            )
        ]
        self._active_index = 0
        self.barsChanged.emit(self.bars)

    def set_bars(self, bars: Sequence[TimeRangeBar], *, active_index: int = 0) -> None:
        self._bars = [replace(b) for b in bars]
        self._active_index = max(0, min(max(0, len(self._bars) - 1), int(active_index)))
        self._redraw_artists()

    def add_bar(self, *, t0: float | None = None, t1: float | None = None) -> None:
        a = self._t_min if t0 is None else float(t0)
        b = self._t_max if t1 is None else float(t1)
        if b <= a:
            b = a + max(0.01, (self._t_max - self._t_min) * 0.1)
        n = len(self._bars) + 1
        self._bars.append(
            TimeRangeBar(
                t0_s=a,
                t1_s=b,
                label=f"Plage {n}",
                bar_id=uuid.uuid4().hex[:8],
            )
        )
        self._active_index = len(self._bars) - 1
        self._redraw_artists()
        self.barsChanged.emit(self.bars)

    def remove_active_bar(self) -> None:
        if not self._bars:
            return
        self._bars.pop(self._active_index)
        self._active_index = max(0, min(len(self._bars) - 1, self._active_index))
        self._redraw_artists()
        self.barsChanged.emit(self.bars)

    def attach(self, canvas: Any, axes: Sequence[Any] | Any) -> None:
        """Attacher aux axes d’une figure (après un render)."""
        self.detach()
        self._canvas = canvas
        if axes is None:
            self._axes = []
        elif isinstance(axes, (list, tuple)):
            self._axes = [ax for ax in axes if ax is not None]
        else:
            # Figure.axes or single Axes
            try:
                self._axes = list(axes) if hasattr(axes, "__iter__") else [axes]
            except TypeError:
                self._axes = [axes]
        if self._canvas is None or not self._axes:
            return
        self._cid_press = self._canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_release = self._canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_motion = self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._redraw_artists()

    def detach(self) -> None:
        if self._canvas is not None:
            for cid in (self._cid_press, self._cid_release, self._cid_motion):
                if cid is not None:
                    try:
                        self._canvas.mpl_disconnect(cid)
                    except Exception:
                        pass
        self._clear_artists()
        self._canvas = None
        self._axes = []
        self._cid_press = self._cid_release = self._cid_motion = None
        self._drag = None

    def _clear_artists(self) -> None:
        for artist in self._artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._artists.clear()

    def _redraw_artists(self) -> None:
        self._clear_artists()
        if not self._axes or not self._bars:
            if self._canvas is not None:
                try:
                    self._canvas.draw_idle()
                except Exception:
                    pass
            return
        for index, bar in enumerate(self._bars):
            t0, t1 = bar.ordered()
            color = _BAR_COLORS[index % len(_BAR_COLORS)]
            alpha_span = 0.18 if index == self._active_index else 0.08
            lw = 1.6 if index == self._active_index else 1.0
            for ax in self._axes:
                span = ax.axvspan(t0, t1, alpha=alpha_span, color=color, zorder=2.5)
                left = ax.axvline(t0, color=color, linewidth=lw, linestyle="-", zorder=3)
                right = ax.axvline(t1, color=color, linewidth=lw, linestyle="-", zorder=3)
                self._artists.extend([span, left, right])
        if self._canvas is not None:
            try:
                self._canvas.draw_idle()
            except Exception:
                pass

    def _pick_edge(self, xdata: float) -> tuple[int, str] | None:
        if not self._bars:
            return None
        span = max(1e-9, self._t_max - self._t_min)
        tol = span * self._pick_tol_frac
        best: tuple[float, int, str] | None = None
        for index, bar in enumerate(self._bars):
            for edge, value in (("t0", float(bar.t0_s)), ("t1", float(bar.t1_s))):
                dist = abs(xdata - value)
                if dist <= tol and (best is None or dist < best[0]):
                    best = (dist, index, edge)
        if best is None:
            return None
        return best[1], best[2]

    def _on_press(self, event: Any) -> None:
        if event.button != 1 or event.xdata is None or event.inaxes is None:
            return
        if event.inaxes not in self._axes:
            return
        hit = self._pick_edge(float(event.xdata))
        if hit is None:
            return
        self._drag = hit
        self._active_index = hit[0]

    def _on_release(self, event: Any) -> None:
        del event
        if self._drag is None:
            return
        self._drag = None
        if self._motion_dirty:
            self._redraw_artists()
            self._motion_dirty = False
        self.barsChanged.emit(self.bars)

    def _on_motion(self, event: Any) -> None:
        if self._drag is None or event.xdata is None:
            return
        index, edge = self._drag
        if not (0 <= index < len(self._bars)):
            return
        x = float(event.xdata)
        x = max(self._t_min, min(self._t_max, x))
        bar = self._bars[index]
        if edge == "t0":
            # Empêcher croisement : laisser un epsilon.
            eps = max(1e-4, (self._t_max - self._t_min) * 1e-4)
            other = float(bar.t1_s)
            if x >= other:
                x = other - eps
            self._bars[index] = bar.with_bounds(x, other)
        else:
            eps = max(1e-4, (self._t_max - self._t_min) * 1e-4)
            other = float(bar.t0_s)
            if x <= other:
                x = other + eps
            self._bars[index] = bar.with_bounds(other, x)
        now = time.perf_counter()
        if now - self._last_draw_s < 0.016:
            self._motion_dirty = True
            return
        self._last_draw_s = now
        self._motion_dirty = False
        self._redraw_artists()
