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
    QSizePolicy,
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

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        allow_zoom: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._allow_zoom = bool(allow_zoom)

        self._container = QWidget()
        self._layout = QGridLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(8)
        self.setWidget(self._container)

        self._panels: dict[str, object] = {}
        self._order: list[PanelPlacement] = []
        self._columns = 2
        self._panel_height = 300
        # True = même hauteur (et donc même largeur de cellule) pour tous les panels.
        self._uniform = False
        # True = panneaux Expanding qui se partagent le viewport (aperçu canal).
        self._fill = False
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
        self._fill_timer = QTimer(self)
        self._fill_timer.setSingleShot(True)
        self._fill_timer.setInterval(50)
        self._fill_timer.timeout.connect(self._reflow_fill_heights)

    # ------------------------------------------------------------------ layout

    def configure(
        self,
        placements: Sequence[PanelPlacement],
        *,
        columns: int,
        panel_height: int,
        uniform: bool = False,
        fill: bool = False,
    ) -> None:
        """Rebuild the grid for the given panels."""
        from .panel_canvas import PanelCanvas

        # Ne pas faire sauter le viewport (ex. après déplacement de plages).
        vbar = self.verticalScrollBar()
        hbar = self.horizontalScrollBar()
        scroll_v = int(vbar.value())
        scroll_h = int(hbar.value())

        self._timer.stop()
        self._pending.clear()
        self._columns = max(1, int(columns))
        self._panel_height = max(140, int(panel_height))
        self._uniform = bool(uniform)
        # fill et uniform sont exclusifs : uniform = hauteur figée (montage).
        self._fill = bool(fill) and not self._uniform
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
                canvas = PanelCanvas(
                    placement, self._container, allow_zoom=self._allow_zoom
                )
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
        self._restore_scroll(scroll_v, scroll_h)

    def _restore_scroll(self, scroll_v: int, scroll_h: int) -> None:
        """Rétablir la position de scroll après un relayout (immédiat + tick suivant)."""
        vbar = self.verticalScrollBar()
        hbar = self.horizontalScrollBar()

        def _apply() -> None:
            vbar.setValue(min(scroll_v, vbar.maximum()))
            hbar.setValue(min(scroll_h, hbar.maximum()))

        _apply()
        QTimer.singleShot(0, _apply)
    def set_panel_height(
        self,
        panel_height: int,
        *,
        panels: Sequence[str] | None = None,
        uniform: bool | None = None,
        fill: bool | None = None,
    ) -> None:
        """Changer la hauteur min sans reconstruire la grille (rapide)."""
        height = max(140, int(panel_height))
        if uniform is not None:
            self._uniform = bool(uniform)
            if self._uniform:
                self._fill = False
        if fill is not None:
            self._fill = bool(fill) and not self._uniform
        if (
            height == self._panel_height
            and panels is None
            and uniform is None
            and fill is None
        ):
            return
        self._panel_height = height
        wanted = set(panels) if panels is not None else None
        for placement in self._order:
            if wanted is not None and placement.panel not in wanted:
                continue
            widget = self._panels.get(placement.key)
            if widget is None:
                continue
            self._apply_panel_height(widget, placement)

    def _panel_min_height(self, placement: PanelPlacement) -> int:
        """Hauteur cible d’un panneau (uniforme, fill viewport, ou preferred_height)."""
        if self._fill or self._uniform or placement.panel == "montage_continuous_raw":
            # Fill : hauteur déjà calculée via preferred_height_px dans _reflow_fill_heights.
            return self._panel_height
        info = panel_info(placement.panel)
        return max(self._panel_height, int(info.preferred_height_px))

    def _apply_panel_height(self, widget: object, placement: PanelPlacement) -> None:
        height = self._panel_min_height(placement)
        widget.setMinimumHeight(height)  # type: ignore[attr-defined]
        # Montage / uniform : hauteur fixe (= budget lignes) pour forcer le scroll.
        # fill : Expanding pour occuper tout le viewport (continuous / moyenne / stim).
        fix_height = bool(self._uniform or placement.panel == "montage_continuous_raw")
        if fix_height:
            widget.setMaximumHeight(height)  # type: ignore[attr-defined]
            v_policy = QSizePolicy.Policy.Fixed
        else:
            widget.setMaximumHeight(16777215)  # type: ignore[attr-defined]
            v_policy = (
                QSizePolicy.Policy.Expanding if self._fill else QSizePolicy.Policy.Preferred
            )
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, v_policy)  # type: ignore[attr-defined]

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fill and self._order:
            self._fill_timer.start()

    def _reflow_fill_heights(self) -> None:
        """Recalculer la hauteur min pour coller au viewport (mode fill)."""
        if not self._fill or not self._order:
            return
        n = max(1, len(self._order))
        vp = int(self.viewport().height())
        if vp <= 80:
            return
        margins = int(self._layout.contentsMargins().top()) + int(
            self._layout.contentsMargins().bottom()
        )
        gaps = max(0, n - 1) * int(self._layout.spacing())
        usable = max(140, vp - margins - gaps)
        # Plancher = preferred_height_px du catalogue (paramètre hauteur de graph).
        preferreds = [
            int(panel_info(p.panel).preferred_height_px) for p in self._order
        ]
        if n == 1:
            # Un graph : occupe tout le viewport, au moins sa hauteur catalogue.
            height = max(max(preferreds, default=260), usable)
        else:
            # Plusieurs : parts égales ; plancher lisible (pas le preferred plein).
            height = max(260, usable // n)
        if height == self._panel_height:
            return
        self.set_panel_height(height)

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
                "Aucun panneau dans cette vue.\n\n"
                "1. Session → Ajouter un .rhs\n"
                "2. Traiter (F5)\n"
                "3. Choisir un canal (aperçu) · Analyse (Ctrl+I)\n\n"
                "Revue montage : Ctrl+M"
            )
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._placeholder.setObjectName("workflowHint")
            self._placeholder.setWordWrap(True)
            self._layout.addWidget(self._placeholder, 0, 0)
            return

        n_rows = 0
        for index, placement in enumerate(self._order):
            widget = self._panels[placement.key]
            info = panel_info(placement.panel)
            self._apply_panel_height(widget, placement)
            widget.setMinimumWidth(0)  # type: ignore[attr-defined]
            widget.setParent(self._container)  # type: ignore[attr-defined]
            row, column = divmod(index, self._columns)
            n_rows = max(n_rows, row + 1)
            span = self._columns if info.is_global and info.key.startswith("montage_") else 1
            span = min(span, self._columns)
            self._layout.addWidget(widget, row, column, 1, span)  # type: ignore[arg-type]
            widget.show()  # type: ignore[attr-defined]
        for column in range(self._columns):
            self._layout.setColumnStretch(column, 1)
            self._layout.setColumnMinimumWidth(column, 0)
        # Mode fill : chaque ligne partage également l’espace vertical.
        for row in range(max(n_rows, 1)):
            self._layout.setRowStretch(row, 1 if self._fill else 0)
        if self._fill:
            QTimer.singleShot(0, self._reflow_fill_heights)

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
        # Fenêtre principale : pan / home seulement (pas de zoom molette / outil Zoom).
        self.grid = PanelGrid(self, allow_zoom=False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.grid, 1)
