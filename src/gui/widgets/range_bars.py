"""Barres de plage temporelle déplaçables (toolbar + contrôleur pyqtgraph)."""

from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import Any, Sequence

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from view_config import TimeRangeBar

_BAR_COLORS = ("#16a34a", "#2563eb", "#ca8a04", "#dc2626", "#7c3aed")


def _qt_alive(obj: Any) -> bool:
    """True si le wrapper Python pointe encore vers un objet C++ valide."""
    if obj is None:
        return False
    try:
        from shiboken6 import isValid

        return bool(isValid(obj))
    except Exception:
        try:
            obj.objectName()  # type: ignore[attr-defined]
            return True
        except RuntimeError:
            return False


def _plot_alive(plot: Any) -> bool:
    """True si le PlotItem et sa ViewBox sont encore utilisables."""
    if not _qt_alive(plot):
        return False
    try:
        vb = getattr(plot, "vb", None)
        if vb is None:
            getter = getattr(plot, "getViewBox", None)
            vb = getter() if callable(getter) else None
        if vb is None or not _qt_alive(vb):
            return False
        # Accès léger : échoue si le C++ a déjà été détruit.
        plot.zValue()
        vb.zValue()
        return True
    except RuntimeError:
        return False


def _full_width_button(text: str, parent: QWidget) -> QPushButton:
    """Bouton qui occupe la largeur dispo sans écraser son libellé."""
    btn = QPushButton(text, parent)
    btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    btn.setMinimumWidth(0)
    return btn


class RangeBarToolbar(QWidget):
    """Boutons pour ajouter / supprimer / activer une plage et lancer le traitement.

    Empilé verticalement dans un GroupBox pour rester lisible dans le panneau
    latéral étroit (Inspecter / Analyse) — plus de texte coupé.
    """

    addRequested = Signal()
    addRelativeRequested = Signal()
    removeRequested = Signal()
    activeChanged = Signal(int)
    processRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        box = QGroupBox("Plages", self)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(8, 10, 8, 8)
        box_layout.setSpacing(6)

        self._count_label = QLabel("0 plage(s)", box)
        self._count_label.setObjectName("hintLabel")
        self._active = QSpinBox(box)
        self._active.setMinimum(1)
        self._active.setMaximum(1)
        self._active.setValue(1)
        self._active.setFixedWidth(52)
        self._active.setToolTip("Plage active (utilisée pour le zoom / traitement)")
        self._active.valueChanged.connect(lambda v: self.activeChanged.emit(int(v) - 1))

        active_label = QLabel("Active", box)
        active_label.setBuddy(self._active)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self._count_label, 1)
        header.addWidget(active_label)
        header.addWidget(self._active)

        btn_add = _full_width_button("+ Absolue", box)
        btn_add.setToolTip(
            "Ajouter une plage centrée (10 % du graph visible), "
            "déplaçable au clic gauche — puis « Appliquer les zooms »"
        )
        btn_add.clicked.connect(self.addRequested.emit)

        btn_add_rel = _full_width_button("+ Rel. stim", box)
        btn_add_rel.setToolTip(
            "Ajouter une plage [t₀, t₁] relative au début de stimulation — "
            "puis « Appliquer les zooms » pour ouvrir la fenêtre"
        )
        btn_add_rel.clicked.connect(self.addRelativeRequested.emit)

        btn_remove = _full_width_button("− Supprimer", box)
        btn_remove.setToolTip("Supprimer la plage active")
        btn_remove.clicked.connect(self.removeRequested.emit)

        btn_process = _full_width_button("Appliquer les zooms", box)
        btn_process.setObjectName("primaryButton")
        btn_process.setToolTip(
            "Ouvrir les zooms du canal dans une fenêtre dédiée "
            "(continuous : tous flux / plages ; moyenne / stimulation : "
            "graphs d’analyse cochés) — ne relance pas le traitement F5"
        )
        btn_process.clicked.connect(self.processRequested.emit)

        actions = QGridLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setHorizontalSpacing(4)
        actions.setVerticalSpacing(4)
        actions.addWidget(btn_add, 0, 0)
        actions.addWidget(btn_add_rel, 0, 1)
        actions.addWidget(btn_remove, 1, 0, 1, 2)
        actions.setColumnStretch(0, 1)
        actions.setColumnStretch(1, 1)

        box_layout.addLayout(header)
        box_layout.addLayout(actions)
        box_layout.addWidget(btn_process)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(box)

    def set_bar_count(self, count: int, active_index: int = 0) -> None:
        n = max(0, int(count))
        self._count_label.setText(f"{n} plage(s)")
        self._active.blockSignals(True)
        self._active.setMaximum(max(1, n))
        self._active.setEnabled(n > 0)
        if n > 0:
            self._active.setValue(max(1, min(n, int(active_index) + 1)))
        self._active.blockSignals(False)


class RangeBarControllerPg(QObject):
    """Range bars on a :class:`~gui.widgets.plot_host.PlotHost` (pyqtgraph).

    Uses ``LinearRegionItem`` / ``InfiniteLine``; left-drag moves edges.
    Absolute / relative-stim modes are encoded in :class:`TimeRangeBar` and
    interpreted by the parent window — this controller only edits ``t0``/``t1``.
    """

    barsChanged = Signal(object)  # tuple[TimeRangeBar, ...]

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bars: list[TimeRangeBar] = []
        self._active_index = 0
        self._host: Any | None = None
        self._regions: list[Any] = []
        self._lines: list[Any] = []
        self._t_min = 0.0
        self._t_max = 1.0
        self._display_offset = 0.0
        self._drag: tuple[int, str] | None = None
        self._last_draw_s = 0.0
        self._motion_dirty = False
        self._updating = False

    @property
    def bars(self) -> tuple[TimeRangeBar, ...]:
        return tuple(self._bars)

    @property
    def active_index(self) -> int:
        return self._active_index

    @property
    def is_dragging(self) -> bool:
        return self._drag is not None

    def set_active_index(self, index: int) -> None:
        if not self._bars:
            self._active_index = 0
            return
        if int(index) < 0:
            self._active_index = -1
            self._redraw_items()
            return
        self._active_index = max(0, min(len(self._bars) - 1, int(index)))
        self._redraw_items()

    def set_display_offset(self, offset_s: float) -> None:
        new_offset = float(offset_s)
        if abs(new_offset - self._display_offset) < 1e-15:
            return
        self._display_offset = new_offset
        self._redraw_items()

    def set_time_span(self, t_min: float, t_max: float) -> None:
        self._t_min = float(t_min)
        self._t_max = float(t_max)
        if self._t_max <= self._t_min:
            self._t_max = self._t_min + 1.0


    def set_bars(self, bars: Sequence[TimeRangeBar], *, active_index: int = 0) -> None:
        self._bars = [replace(b) for b in bars]
        self._active_index = max(0, min(max(0, len(self._bars) - 1), int(active_index)))
        self._redraw_items()

    def add_bar(self, *, t0: float | None = None, t1: float | None = None) -> None:
        """Ajouter une plage. Sans bornes → 10 % de la span, centrée sur le graph."""
        span = max(1e-6, self._t_max - self._t_min)
        if t0 is None or t1 is None:
            width = span * 0.1
            mid = (self._t_min + self._t_max) * 0.5
            a = mid - width * 0.5 if t0 is None else float(t0)
            b = mid + width * 0.5 if t1 is None else float(t1)
        else:
            a, b = float(t0), float(t1)
        if b <= a:
            b = a + max(0.01, span * 0.1)
        a = max(self._t_min, min(self._t_max, a))
        b = max(self._t_min, min(self._t_max, b))
        if b <= a:
            b = min(self._t_max, a + max(0.01, span * 0.1))
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
        self._redraw_items()
        self.barsChanged.emit(self.bars)


    def attach(self, host: Any, axes: Sequence[Any] | Any | None = None) -> None:
        """Attach to a PlotHost (``axes`` ignored — kept for API parity)."""
        del axes
        self.detach()
        if host is not None and not _qt_alive(host):
            return
        self._host = host
        self._redraw_items()

    def detach(self) -> None:
        self._clear_items()
        self._host = None
        self._drag = None

    def _ensure_host(self) -> Any | None:
        host = self._host
        if host is None:
            return None
        if not _qt_alive(host):
            self._regions.clear()
            self._lines.clear()
            self._host = None
            self._drag = None
            return None
        return host

    def _plot_items(self) -> list[Any]:
        host = self._ensure_host()
        if host is None:
            return []
        try:
            getter = getattr(host, "plot_items", None)
            if callable(getter):
                raw = list(getter() or [])
            else:
                figure = getattr(host, "figure", None)
                axes = (
                    list(getattr(figure, "axes", []) or [])
                    if figure is not None
                    else []
                )
                raw = []
                for ax in axes:
                    plot = getattr(ax, "_plot", None)
                    if plot is not None:
                        raw.append(plot)
        except RuntimeError:
            self._host = None
            return []
        return [plot for plot in raw if _plot_alive(plot)]

    def _clear_items(self) -> None:
        plots = self._plot_items()
        for item in self._regions + self._lines:
            try:
                if not _qt_alive(item):
                    continue
                parent = item.parentItem()
                if parent is not None and hasattr(parent, "removeItem"):
                    parent.removeItem(item)
                else:
                    for plot in plots:
                        try:
                            plot.removeItem(item)
                        except Exception:
                            pass
            except Exception:
                pass
        self._regions.clear()
        self._lines.clear()

    def _redraw_items(self) -> None:
        self._clear_items()
        host = self._ensure_host()
        plots = self._plot_items()
        if not plots or not self._bars:
            if host is not None:
                try:
                    host.draw_idle()
                except Exception:
                    pass
            return
        import pyqtgraph as pg
        from PySide6.QtCore import Qt as _Qt

        self._updating = True
        try:
            off = float(self._display_offset)
            for index, bar in enumerate(self._bars):
                t0, t1 = bar.ordered()
                t0_d, t1_d = float(t0) - off, float(t1) - off
                color = _BAR_COLORS[index % len(_BAR_COLORS)]
                active = index == self._active_index
                brush_color = pg.mkColor(color)
                brush_color.setAlpha(60 if active else 30)
                brush = pg.mkBrush(brush_color)
                pen = pg.mkPen(color, width=1.6 if active else 1.0)
                for plot in plots:
                    if not _plot_alive(plot):
                        continue
                    region = pg.LinearRegionItem(
                        values=(t0_d, t1_d),
                        brush=brush,
                        pen=pen,
                        movable=True,
                        swapMode="block",
                    )
                    region.setZValue(10)
                    try:
                        region.setAcceptedMouseButtons(_Qt.MouseButton.LeftButton)
                    except Exception:
                        pass
                    try:
                        plot.addItem(region)
                    except RuntimeError:
                        # PlotHost / ViewBox détruit entre-temps (rebuild grille).
                        continue
                    self._regions.append(region)
                    region.sigRegionChanged.connect(
                        lambda _r=region, i=index: self._on_region_changed(i, _r)
                    )
                    region.sigRegionChangeFinished.connect(
                        lambda _r=region, i=index: self._on_region_finished(i, _r)
                    )
        finally:
            self._updating = False
        host = self._ensure_host()
        if host is not None:
            try:
                host.draw_idle()
            except Exception:
                pass

    def _on_region_changed(self, index: int, region: Any) -> None:
        if self._updating or not (0 <= index < len(self._bars)):
            return
        self._drag = (index, "region")
        self._active_index = index
        now = time.perf_counter()
        if now - self._last_draw_s < 0.016:
            self._motion_dirty = True
            return
        self._last_draw_s = now
        self._motion_dirty = False
        self._apply_region(index, region, emit=True)

    def _on_region_finished(self, index: int, region: Any) -> None:
        if not (0 <= index < len(self._bars)):
            return
        self._apply_region(index, region, emit=True)
        self._drag = None
        if self._motion_dirty:
            self._motion_dirty = False
            self._redraw_items()

    def _apply_region(self, index: int, region: Any, *, emit: bool) -> None:
        try:
            a, b = region.getRegion()
        except Exception:
            return
        off = float(self._display_offset)
        t0 = max(self._t_min, min(self._t_max, float(a) + off))
        t1 = max(self._t_min, min(self._t_max, float(b) + off))
        if t1 < t0:
            t0, t1 = t1, t0
        eps = max(1e-4, (self._t_max - self._t_min) * 1e-4)
        if t1 - t0 < eps:
            t1 = t0 + eps
        self._bars[index] = self._bars[index].with_bounds(t0, t1)
        # Sync sibling regions (same bar on other rows) without feedback loops.
        self._updating = True
        try:
            display = (float(t0) - off, float(t1) - off)
            n_plots = max(1, len(self._plot_items()))
            for reg_i, reg in enumerate(self._regions):
                bar_i = reg_i // n_plots
                if bar_i != index or reg is region:
                    continue
                try:
                    reg.setRegion(display)
                except Exception:
                    pass
        finally:
            self._updating = False
        if emit:
            self.barsChanged.emit(self.bars)
