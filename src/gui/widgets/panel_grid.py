"""Scrollable, reconfigurable grid of panels for one view tab.

Two things keep redraws fast. Panels are drawn one per event-loop turn, so the
window never freezes while a dozen panels catch up; and panels scrolled out of
view are deferred until they come close to the viewport, which is what makes a
view with dozens of panels still feel immediate.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from panel_registry import RenderRequest, panel_info
from view_config import PanelPlacement

RequestFactory = Callable[[PanelPlacement], RenderRequest]


class PanelGrid(QScrollArea):
    """Lays out :class:`PanelCanvas` widgets and drives their redraws."""

    removeRequested = Signal(object)  # PanelPlacement
    detachRequested = Signal(object)  # PanelPlacement
    renderFinished = Signal(int, float)  # panels drawn, total seconds

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._container = QWidget()
        self._layout = QGridLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(8)
        self.setWidget(self._container)

        self._panels: dict[str, object] = {}
        self._order: list[PanelPlacement] = []
        self._columns = 2
        self._panel_height = 300
        self._placeholder: QLabel | None = None
        self._pending: list[PanelPlacement] = []
        self._deferred: list[PanelPlacement] = []
        self._factory: RequestFactory | None = None
        self._elapsed = 0.0
        self._drawn = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._render_next)
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(90)
        self._scroll_timer.timeout.connect(self._render_newly_visible)
        self.verticalScrollBar().valueChanged.connect(lambda _v: self._scroll_timer.start())

    # ------------------------------------------------------------------ layout

    def configure(
        self,
        placements: Sequence[PanelPlacement],
        *,
        columns: int,
        panel_height: int,
    ) -> None:
        """Rebuild the grid for the given panels."""
        from .panel_canvas import PanelCanvas

        self._timer.stop()
        self._pending.clear()
        self._columns = max(1, int(columns))
        self._panel_height = max(140, int(panel_height))
        # One widget per key: section-independent panels collapse to a single copy.
        wanted: dict[str, PanelPlacement] = {}
        for placement in placements:
            wanted.setdefault(placement.key, placement)
        placements = list(wanted.values())

        for key in list(self._panels):
            if key not in wanted:
                widget = self._panels.pop(key)
                self._layout.removeWidget(widget)  # type: ignore[arg-type]
                widget.setParent(None)  # type: ignore[attr-defined]
                widget.deleteLater()  # type: ignore[attr-defined]

        for placement in placements:
            if placement.key not in self._panels:
                canvas = PanelCanvas(placement, self._container)
                canvas.removeRequested.connect(self.removeRequested.emit)
                canvas.detachRequested.connect(self.detachRequested.emit)
                self._panels[placement.key] = canvas
            else:
                widget = self._panels[placement.key]
                update = getattr(widget, "update_placement", None)
                if callable(update):
                    update(placement)
                else:
                    widget.placement = placement  # type: ignore[attr-defined]
                    widget.invalidate()  # type: ignore[attr-defined]

        self._order = list(placements)
        self._relayout()

    def set_panel_height(
        self, panel_height: int, *, panels: Sequence[str] | None = None
    ) -> None:
        """Changer la hauteur min sans reconstruire la grille (rapide)."""
        height = max(140, int(panel_height))
        if height == self._panel_height and panels is None:
            return
        self._panel_height = height
        wanted = set(panels) if panels is not None else None
        for placement in self._order:
            if wanted is not None and placement.panel not in wanted:
                continue
            widget = self._panels.get(placement.key)
            if widget is None:
                continue
            info = panel_info(placement.panel)
            min_h = max(height, info.preferred_height_px)
            if placement.panel == "montage_continuous_raw":
                min_h = max(min_h, height)
            widget.setMinimumHeight(min_h)  # type: ignore[attr-defined]

    def _relayout(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        if self._placeholder is not None:
            self._placeholder.deleteLater()
            self._placeholder = None

        if not self._order:
            self._placeholder = QLabel(
                "Aperçu du canal sélectionné.\n\n"
                "1. Session → Ajouter un .rhs\n"
                "2. Traiter (F5)\n"
                "3. Choisir un canal · Inspecter (Ctrl+I)\n\n"
                "Revue montage : Ctrl+M"
            )
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._placeholder.setObjectName("workflowHint")
            self._placeholder.setWordWrap(True)
            self._layout.addWidget(self._placeholder, 0, 0)
            return

        for index, placement in enumerate(self._order):
            widget = self._panels[placement.key]
            info = panel_info(placement.panel)
            height = max(self._panel_height, info.preferred_height_px)
            # Montage continu : hauteur proportionnelle au nombre de lignes canal×flux.
            if placement.panel == "montage_continuous_raw":
                height = max(height, self._panel_height)
            widget.setMinimumHeight(height)  # type: ignore[attr-defined]
            widget.setParent(self._container)  # type: ignore[attr-defined]
            row, column = divmod(index, self._columns)
            span = self._columns if info.is_global and info.key.startswith("montage_") else 1
            span = min(span, self._columns)
            self._layout.addWidget(widget, row, column, 1, span)  # type: ignore[arg-type]
            widget.show()  # type: ignore[attr-defined]
        for column in range(self._columns):
            self._layout.setColumnStretch(column, 1)

    @property
    def placements(self) -> list[PanelPlacement]:
        return list(self._order)

    def panel_widget(self, placement: PanelPlacement) -> Any | None:
        """The :class:`PanelCanvas` showing this placement, if it is laid out."""
        return self._panels.get(placement.key)

    @property
    def panel_count(self) -> int:
        return len(self._order)

    # ----------------------------------------------------------------- drawing

    def invalidate_all(self) -> None:
        for widget in self._panels.values():
            widget.invalidate()  # type: ignore[attr-defined]
            widget.mark_pending()  # type: ignore[attr-defined]

    def _near_viewport(self, widget: QWidget) -> bool:
        """True when the panel is on screen, or within half a screen of it."""
        viewport = self.viewport()
        height = viewport.height()
        if height <= 0 or widget.height() <= 0:
            return True
        margin = height // 2
        top_left = widget.mapTo(viewport, widget.rect().topLeft())
        rect = QRect(top_left, widget.size())
        return viewport.rect().adjusted(0, -margin, 0, margin).intersects(rect)

    def schedule_render(self, factory: RequestFactory, *, force: bool = False) -> None:
        """Queue a redraw of the dirty panels, nearest to the viewport first."""
        self._factory = factory
        if force:
            self.invalidate_all()
        dirty = [
            placement
            for placement in self._order
            if force or self._panels[placement.key].is_dirty  # type: ignore[attr-defined]
        ]
        self._pending = []
        self._deferred = []
        for placement in dirty:
            widget = self._panels[placement.key]
            if self._near_viewport(widget):  # type: ignore[arg-type]
                self._pending.append(placement)
            else:
                self._deferred.append(placement)
        self._elapsed = 0.0
        self._drawn = 0
        for placement in self._pending:
            self._panels[placement.key].mark_pending()  # type: ignore[attr-defined]
        for placement in self._deferred:
            self._panels[placement.key].mark_offscreen()  # type: ignore[attr-defined]
        if not self._pending:
            self.renderFinished.emit(0, 0.0)
            return
        self._timer.start()

    def render_dirty_now(self, factory: RequestFactory) -> int:
        """Draw every dirty panel synchronously, viewport or not (used by export)."""
        self._timer.stop()
        drawn = 0
        for placement in self._order:
            widget = self._panels.get(placement.key)
            if widget is None or not widget.is_dirty:  # type: ignore[attr-defined]
                continue
            widget.render(factory(placement))  # type: ignore[attr-defined]
            drawn += 1
        self._pending.clear()
        self._deferred.clear()
        return drawn

    def cancel_render(self) -> None:
        self._timer.stop()
        self._scroll_timer.stop()
        self._pending.clear()

    def _render_next(self) -> None:
        if not self._pending or self._factory is None:
            self.renderFinished.emit(self._drawn, self._elapsed)
            return
        placement = self._pending.pop(0)
        widget = self._panels.get(placement.key)
        if widget is not None:
            widget.render(self._factory(placement))  # type: ignore[attr-defined]
            self._elapsed += widget.last_render_s  # type: ignore[attr-defined]
            self._drawn += 1
        if self._pending:
            self._timer.start()
        else:
            self.renderFinished.emit(self._drawn, self._elapsed)

    def _render_newly_visible(self) -> None:
        """After a scroll, draw the deferred panels that just came into reach."""
        if self._factory is None or not self._deferred:
            return
        ready = [
            placement
            for placement in self._deferred
            if self._near_viewport(self._panels[placement.key])  # type: ignore[arg-type]
        ]
        if not ready:
            return
        self._deferred = [p for p in self._deferred if p not in ready]
        self._pending.extend(ready)
        for placement in ready:
            self._panels[placement.key].mark_pending()  # type: ignore[attr-defined]
        if not self._timer.isActive():
            self._elapsed = 0.0
            self._drawn = 0
            self._timer.start()


class ViewTabPage(QWidget):
    """A view tab: its toolbar lives in the main window, the grid lives here."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.grid = PanelGrid(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.grid, 1)
