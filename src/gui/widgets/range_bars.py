"""Barres de plage temporelle déplaçables (toolbar + liste + contrôleur pyqtgraph)."""

from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import Any, Sequence

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.form_widgets import ColorSwatchButton, make_double_spin
from view_config import (
    TextFace,
    TimeRangeBar,
    default_zone_label,
    default_zoom_range_color,
    resolved_range_bar_color,
    resolved_zone_label,
)


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


class _RangeBarRow(QFrame):
    """Une plage / zoom : couleur, nom, [t0, t1], type abs/rel."""

    activated = Signal(int)
    changed = Signal(int, object)  # index, TimeRangeBar

    def __init__(self, index: int, bar: TimeRangeBar, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = int(index)
        self._bar = replace(bar)
        self._updating = False
        self.setObjectName("rangeBarRow")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        color = resolved_range_bar_color(bar, index)
        self._color = ColorSwatchButton(
            color,
            dialog_title="Couleur de la zone de zoom",
            parent=self,
        )
        self._color.setToolTip("Couleur de cette plage / fenêtre zoom")
        self._color.changed.connect(self._emit_changed)

        self._label = QLineEdit(resolved_zone_label(bar, index), self)
        self._label.setPlaceholderText(default_zone_label(index))
        self._label.setMinimumWidth(0)
        self._label.setToolTip("Nom affiché sous la zone de zoom sur le graphique")
        self._label.editingFinished.connect(self._emit_changed)

        kind = "rel. stim" if bar.relative_to_stim else "abs."
        self._kind = QLabel(kind, self)
        self._kind.setObjectName("hintLabel")
        self._kind.setToolTip(
            "Relatif au début de stimulation"
            if bar.relative_to_stim
            else "Temps absolu sur l’enregistrement"
        )

        t0, t1 = bar.ordered()
        self._t0 = make_double_spin(-1e6, 1e6, t0, decimals=3, step=0.01, suffix=" s")
        self._t1 = make_double_spin(-1e6, 1e6, t1, decimals=3, step=0.01, suffix=" s")
        self._t0.setToolTip("Début de la plage")
        self._t1.setToolTip("Fin de la plage")
        self._t0.valueChanged.connect(self._emit_changed)
        self._t1.valueChanged.connect(self._emit_changed)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        top.addWidget(self._color, 0)
        top.addWidget(self._label, 1)
        top.addWidget(self._kind, 0)

        times = QHBoxLayout()
        times.setContentsMargins(0, 0, 0, 0)
        times.setSpacing(4)
        times.addWidget(self._t0, 1)
        times.addWidget(QLabel("→", self), 0)
        times.addWidget(self._t1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addLayout(times)

    def mousePressEvent(self, event: Any) -> None:  # noqa: D102
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self._index)
        super().mousePressEvent(event)

    def set_active(self, active: bool) -> None:
        self.setProperty("activeRange", bool(active))
        self.style().unpolish(self)
        self.style().polish(self)

    def set_bar(self, index: int, bar: TimeRangeBar, *, active: bool) -> None:
        self._index = int(index)
        self._bar = replace(bar)
        self._updating = True
        try:
            self._color.blockSignals(True)
            self._color.set_color(resolved_range_bar_color(bar, index))
            self._color.blockSignals(False)
            self._label.blockSignals(True)
            self._label.setText(resolved_zone_label(bar, index))
            self._label.setPlaceholderText(default_zone_label(index))
            self._label.blockSignals(False)
            self._kind.setText("rel. stim" if bar.relative_to_stim else "abs.")
            t0, t1 = bar.ordered()
            self._t0.blockSignals(True)
            self._t1.blockSignals(True)
            self._t0.setValue(t0)
            self._t1.setValue(t1)
            self._t0.blockSignals(False)
            self._t1.blockSignals(False)
        finally:
            self._updating = False
        self.set_active(active)

    def _current_bar(self) -> TimeRangeBar:
        t0 = float(self._t0.value())
        t1 = float(self._t1.value())
        if t1 < t0:
            t0, t1 = t1, t0
        label = self._label.text().strip() or default_zone_label(self._index)
        return replace(
            self._bar,
            t0_s=t0,
            t1_s=t1,
            label=label,
            color=self._color.color_value(),
        )

    def _emit_changed(self, *_args: Any) -> None:
        if self._updating:
            return
        bar = self._current_bar()
        self._bar = bar
        self.changed.emit(self._index, bar)


class RangeBarToolbar(QWidget):
    """Liste des plages / zooms + actions (ajouter, supprimer, appliquer)."""

    addRequested = Signal()
    addRelativeRequested = Signal()
    removeRequested = Signal()
    activeChanged = Signal(int)
    colorChanged = Signal(str)
    barEdited = Signal(int, object)  # index, TimeRangeBar
    processRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._rows: list[_RangeBarRow] = []
        self._updating_list = False

        box = QGroupBox("Plages / zooms", self)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(8, 10, 8, 8)
        box_layout.setSpacing(6)

        self._count_label = QLabel("0 plage(s)", box)
        self._count_label.setObjectName("hintLabel")
        self._active = QSpinBox(box)
        self._active.setMinimum(0)
        self._active.setMaximum(1)
        self._active.setSpecialValueText("—")
        self._active.setValue(0)
        self._active.setFixedWidth(52)
        self._active.setToolTip(
            "Plage active (surlignage / suppression). "
            "« — » = aucune. Clic à côté d’une bande sur le graph pour tout désélectionner."
        )
        self._active.valueChanged.connect(lambda v: self.activeChanged.emit(int(v) - 1))

        active_label = QLabel("Active", box)
        active_label.setBuddy(self._active)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self._count_label, 1)
        header.addWidget(active_label)
        header.addWidget(self._active)

        self._list_host = QWidget(box)
        self._list_layout = QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(4)
        self._list_layout.addStretch(1)

        self._list_scroll = QScrollArea(box)
        self._list_scroll.setWidgetResizable(True)
        self._list_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._list_scroll.setMinimumHeight(72)
        self._list_scroll.setMaximumHeight(220)
        self._list_scroll.setWidget(self._list_host)

        self._empty_label = QLabel(
            "Aucune plage — ajoutez-en une, puis « Appliquer les zooms ».",
            box,
        )
        self._empty_label.setObjectName("hintLabel")
        self._empty_label.setWordWrap(True)

        btn_add = _full_width_button("+ Absolue", box)
        btn_add.setToolTip(
            "Ajouter une plage centrée (10 % du graph visible), "
            "déplaçable au clic gauche — puis « Appliquer les zooms »"
        )
        btn_add.clicked.connect(self.addRequested.emit)

        btn_add_rel = _full_width_button("+ Rel. stim", box)
        btn_add_rel.setToolTip(
            "Ajouter une plage [t₀, t₁] relative au début de stimulation — "
            "puis « Appliquer les zooms » pour ouvrir une fenêtre par plage"
        )
        btn_add_rel.clicked.connect(self.addRelativeRequested.emit)

        btn_remove = _full_width_button("− Supprimer", box)
        btn_remove.setToolTip("Supprimer la plage active")
        btn_remove.clicked.connect(self.removeRequested.emit)

        btn_process = _full_width_button("Appliquer les zooms", box)
        btn_process.setObjectName("primaryButton")
        btn_process.setToolTip(
            "Ouvrir une fenêtre de zoom par plage "
            "(uniquement la portion zoomée — pas la courbe aperçu complète)"
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
        box_layout.addWidget(self._empty_label)
        box_layout.addWidget(self._list_scroll, 1)
        box_layout.addLayout(actions)
        box_layout.addWidget(btn_process)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(box)

        # Compat API (couleur active via la ligne sélectionnée).
        self._color = ColorSwatchButton(
            default_zoom_range_color(0),
            dialog_title="Couleur de la zone de zoom",
            parent=box,
        )
        self._color.hide()
        self._color.changed.connect(self._emit_color)

    def _emit_color(self) -> None:
        self.colorChanged.emit(self._color.color_value())

    def set_active_color(self, color: str, *, enabled: bool = True) -> None:
        """Synchroniser le nuancier legacy (sans émettre)."""
        self._color.blockSignals(True)
        self._color.set_color(str(color or default_zoom_range_color(0)))
        self._color.setEnabled(bool(enabled))
        self._color.blockSignals(False)

    def set_bar_count(self, count: int, active_index: int = 0) -> None:
        n = max(0, int(count))
        self._count_label.setText(f"{n} plage(s)")
        active = int(active_index)
        if n <= 0:
            active = -1
        elif active >= 0:
            active = max(0, min(n - 1, active))
        else:
            active = -1
        self._active.blockSignals(True)
        self._active.setMaximum(max(1, n))
        self._active.setEnabled(n > 0)
        # 0 = « — » (aucune) ; 1…n = plage active.
        self._active.setValue(0 if active < 0 else active + 1)
        self._active.blockSignals(False)
        self._color.setEnabled(n > 0 and active >= 0)
        self._empty_label.setVisible(n == 0)
        self._list_scroll.setVisible(n > 0)

    def set_bars(self, bars: Sequence[TimeRangeBar], active_index: int = 0) -> None:
        """Rebuild / sync the editable list of zooms."""
        bars = tuple(bars)
        n = len(bars)
        active = int(active_index)
        if n <= 0:
            active = -1
        elif active >= 0:
            active = max(0, min(n - 1, active))
        else:
            active = -1
        self._updating_list = True
        try:
            while len(self._rows) > n:
                row = self._rows.pop()
                self._list_layout.removeWidget(row)
                row.deleteLater()
            while len(self._rows) < n:
                index = len(self._rows)
                row = _RangeBarRow(index, bars[index], self._list_host)
                row.activated.connect(self._on_row_activated)
                row.changed.connect(self._on_row_changed)
                self._list_layout.insertWidget(index, row)
                self._rows.append(row)
            for index, bar in enumerate(bars):
                self._rows[index].set_bar(index, bar, active=(index == active))
            self.set_bar_count(n, active)
            if n and active >= 0:
                self.set_active_color(
                    resolved_range_bar_color(bars[active], active), enabled=True
                )
            elif n:
                self.set_active_color(default_zoom_range_color(0), enabled=False)
        finally:
            self._updating_list = False

    def _on_row_activated(self, index: int) -> None:
        if self._updating_list:
            return
        self.activeChanged.emit(int(index))

    def _on_row_changed(self, index: int, bar: TimeRangeBar) -> None:
        if self._updating_list:
            return
        self.barEdited.emit(int(index), bar)


class RangeBarControllerPg(QObject):
    """Range bars on a :class:`~gui.widgets.plot_host.PlotHost` (pyqtgraph).

    Uses ``LinearRegionItem`` / ``InfiniteLine``; left-drag moves edges.
    Absolute / relative-stim modes are encoded in :class:`TimeRangeBar` and
    interpreted by the parent window — this controller only edits ``t0``/``t1``.
    """

    barsChanged = Signal(object)  # tuple[TimeRangeBar, ...]
    activeChanged = Signal(int)  # index local, -1 = aucune

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bars: list[TimeRangeBar] = []
        self._active_index = -1
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
        self._show_zone_labels = True
        self._zone_font_size = 8.0
        self._zone_face = TextFace()
        self._zone_dx_px = 0.0
        self._zone_dy_px = 0.0
        self._scene_connected: list[Any] = []

    @property
    def bars(self) -> tuple[TimeRangeBar, ...]:
        return tuple(self._bars)

    @property
    def active_index(self) -> int:
        return self._active_index

    @property
    def is_dragging(self) -> bool:
        return self._drag is not None

    def set_active_index(self, index: int, *, emit: bool = False) -> None:
        if not self._bars:
            new_index = -1
        elif int(index) < 0:
            new_index = -1
        else:
            new_index = max(0, min(len(self._bars) - 1, int(index)))
        changed = new_index != self._active_index
        self._active_index = new_index
        if changed:
            self._redraw_items()
            if emit:
                self.activeChanged.emit(int(self._active_index))

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
        if not self._bars:
            self._active_index = -1
        elif int(active_index) < 0:
            self._active_index = -1
        else:
            self._active_index = max(0, min(len(self._bars) - 1, int(active_index)))
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
                label=default_zone_label(n - 1),
                bar_id=uuid.uuid4().hex[:8],
                color=default_zoom_range_color(n - 1),
            )
        )
        self._active_index = len(self._bars) - 1
        self._redraw_items()
        self.barsChanged.emit(self.bars)
        self.activeChanged.emit(int(self._active_index))

    def set_zone_label_style(self, style: Any | None) -> None:
        """Police / placement des libellés « Zone N » (PanelStyle ou None)."""
        if style is None:
            show = True
            fontsize = 8.0
            face = TextFace()
            dx = 0.0
            dy = 0.0
        else:
            show = bool(getattr(style, "show_zone_labels", True))
            fontsize = float(getattr(style, "zone_font_size", 8.0) or 8.0)
            face = getattr(style, "zone_face", None) or TextFace()
            dx = float(getattr(style, "zone_dx_px", 0.0) or 0.0)
            dy = float(getattr(style, "zone_dy_px", 0.0) or 0.0)
        changed = (
            show != self._show_zone_labels
            or abs(fontsize - self._zone_font_size) > 1e-9
            or face != self._zone_face
            or abs(dx - self._zone_dx_px) > 1e-9
            or abs(dy - self._zone_dy_px) > 1e-9
        )
        self._show_zone_labels = show
        self._zone_font_size = max(4.0, fontsize)
        self._zone_face = face
        self._zone_dx_px = dx
        self._zone_dy_px = dy
        if changed:
            self._redraw_items()

    def attach(self, host: Any, axes: Sequence[Any] | Any | None = None) -> None:
        """Attach to a PlotHost (``axes`` ignored — kept for API parity)."""
        del axes
        self.detach()
        if host is not None and not _qt_alive(host):
            return
        self._host = host
        self._redraw_items()
        self._connect_scene_clicks()

    def detach(self) -> None:
        self._disconnect_scene_clicks()
        host = self._host
        self._clear_items()
        if host is not None and _qt_alive(host):
            clearer = getattr(host, "set_zone_labels", None)
            if callable(clearer):
                try:
                    clearer(())
                except Exception:
                    pass
        self._host = None
        self._drag = None

    def _disconnect_scene_clicks(self) -> None:
        for scene in self._scene_connected:
            try:
                scene.sigMouseClicked.disconnect(self._on_scene_clicked)
            except (TypeError, RuntimeError):
                pass
        self._scene_connected.clear()

    def _connect_scene_clicks(self) -> None:
        self._disconnect_scene_clicks()
        host = self._ensure_host()
        if host is None:
            return
        scenes: list[Any] = []
        try:
            glw = getattr(host, "_glw", None)
            scene = glw.scene() if glw is not None else None
            if scene is not None:
                scenes.append(scene)
        except Exception:
            pass
        for plot in self._plot_items():
            try:
                scene = plot.scene()
            except Exception:
                scene = None
            if scene is not None and scene not in scenes:
                scenes.append(scene)
        for scene in scenes:
            try:
                scene.sigMouseClicked.connect(self._on_scene_clicked)
                self._scene_connected.append(scene)
            except Exception:
                pass

    def _bar_index_at_x(self, x_display: float) -> int | None:
        """Index de plage contenant ``x`` (coords d’affichage), ou None."""
        off = float(self._display_offset)
        x = float(x_display)
        for index, bar in enumerate(self._bars):
            t0, t1 = bar.ordered()
            a, b = float(t0) - off, float(t1) - off
            if a > b:
                a, b = b, a
            if a <= x <= b:
                return index
        return None

    def _on_scene_clicked(self, event: Any) -> None:
        """Clic gauche hors bande → désélectionner ; sur une bande → activer."""
        if self._updating or self._drag is not None or not self._bars:
            return
        try:
            from PySide6.QtCore import Qt as _Qt

            if event.button() != _Qt.MouseButton.LeftButton:
                return
            if bool(event.double()):
                return
            scene_pos = event.scenePos()
        except Exception:
            return
        x_hit: float | None = None
        for plot in self._plot_items():
            if not _plot_alive(plot):
                continue
            try:
                vb = plot.getViewBox()
                rect = vb.sceneBoundingRect()
                if not rect.contains(scene_pos):
                    continue
                view_pt = vb.mapSceneToView(scene_pos)
                x_hit = float(view_pt.x())
                break
            except Exception:
                continue
        if x_hit is None:
            return
        hit = self._bar_index_at_x(x_hit)
        if hit is None:
            if self._active_index >= 0:
                self.set_active_index(-1, emit=True)
            return
        if hit != self._active_index:
            self.set_active_index(hit, emit=True)

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

    def _push_zone_labels_to_host(self) -> None:
        """Overlays Zone sous le graph via PlotHost (même mécanique que on/off)."""
        host = self._ensure_host()
        if host is None:
            return
        setter = getattr(host, "set_zone_labels", None)
        if not callable(setter):
            return
        face = self._zone_face
        if not self._show_zone_labels or not self._bars:
            try:
                setter(())
            except Exception:
                pass
            return
        off = float(self._display_offset)
        labels: list[tuple[float, str, str]] = []
        for index, bar in enumerate(self._bars):
            t0, t1 = bar.ordered()
            mid = 0.5 * ((float(t0) - off) + (float(t1) - off))
            text = resolved_zone_label(bar, index)
            if not text.strip():
                continue
            color = resolved_range_bar_color(bar, index)
            labels.append((mid, text, color))
        try:
            setter(
                labels,
                fontsize=float(self._zone_font_size),
                bold=bool(getattr(face, "bold", False)),
                italic=bool(getattr(face, "italic", False)),
                underline=bool(getattr(face, "underline", False)),
                dx_px=float(self._zone_dx_px),
                dy_px=float(self._zone_dy_px),
            )
        except Exception:
            pass

    def _redraw_items(self) -> None:
        self._clear_items()
        host = self._ensure_host()
        plots = self._plot_items()
        if not plots or not self._bars:
            if host is not None:
                try:
                    setter = getattr(host, "set_zone_labels", None)
                    if callable(setter):
                        setter(())
                except Exception:
                    pass
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
                color = resolved_range_bar_color(bar, index)
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
        self._push_zone_labels_to_host()
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
        if index != self._active_index:
            self._active_index = index
            self.activeChanged.emit(int(index))
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
        prev = self._active_index
        self._active_index = index
        if prev != index:
            self.activeChanged.emit(int(index))
        # Redraw : bornes finales + surlignage de la plage active.
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
        self._push_zone_labels_to_host()
        if emit:
            self.barsChanged.emit(self.bars)
