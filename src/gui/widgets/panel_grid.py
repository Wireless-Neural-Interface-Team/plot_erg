"""Scrollable, reconfigurable grid of panels for one view tab.

Two things keep redraws fast. Panels are drawn one per event-loop turn, so the
window never freezes while a dozen panels catch up; and panels scrolled out of
view are deferred until they come close to the viewport, which is what makes a
view with dozens of panels still feel immediate.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from PySide6.QtCore import QRect, QSize, Qt, QTimer, Signal
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
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(0)
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
        # Hauteur des graphs non-montage (ex. RMS / spikes à côté du montage).
        self._graph_height = 400
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
        # Pendant un défilement actif : pas de paint / render lourd (évite les freezes).
        self._scroll_active = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._render_next)
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(140)
        self._scroll_timer.timeout.connect(self._on_scroll_settled)
        self._cull_timer = QTimer(self)
        self._cull_timer.setSingleShot(True)
        self._cull_timer.setInterval(16)
        self._cull_timer.timeout.connect(self._cull_montage_rows)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll_moved)
        self.horizontalScrollBar().valueChanged.connect(self._on_scroll_moved)
        self._fill_timer = QTimer(self)
        self._fill_timer.setSingleShot(True)
        self._fill_timer.setInterval(50)
        self._fill_timer.timeout.connect(self._reflow_fill_heights)

    def sizeHint(self) -> QSize:  # noqa: D102
        if self.isVisible() and self.width() > 0:
            return QSize(self.width(), max(120, self.height()))
        return QSize(320, 400)

    def minimumSizeHint(self) -> QSize:  # noqa: D102
        return QSize(0, 80)

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
        new_height = max(140, int(panel_height))
        shrinking = new_height < int(self._panel_height)
        self._panel_height = new_height
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
                canvas.styleChanged.connect(self._on_panel_style_changed)
                canvas.xScaleSyncRequested.connect(self._on_x_scale_sync)
                self._panels[placement.key] = canvas
            else:
                widget = self._panels[placement.key]
                # Budget en baisse : oublier le fit grow-only (moins de kinds).
                if (
                    shrinking
                    and str(placement.panel).startswith("montage_")
                    and hasattr(widget, "fitted_height_px")
                ):
                    widget.fitted_height_px = None  # type: ignore[attr-defined]
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
        # Budget en baisse (ex. décocher WIDE) : oublier le fit grow-only
        # pour que les lignes ne restent pas étirées dans l’ancien panneau.
        shrinking = height < int(self._panel_height)
        self._panel_height = height
        # Toujours réappliquer aux widgets (hauteur liée à la taille d’échelle).
        wanted = set(panels) if panels is not None else None
        for placement in self._order:
            if wanted is not None and placement.panel not in wanted:
                continue
            widget = self._panels.get(placement.key)
            if widget is None:
                continue
            if shrinking and str(placement.panel).startswith("montage_"):
                if hasattr(widget, "fitted_height_px"):
                    widget.fitted_height_px = None  # type: ignore[attr-defined]
            self._apply_panel_height(widget, placement)

    def apply_fitted_height(
        self, placement: PanelPlacement, height: int, *, scale_fit: bool = False
    ) -> None:
        """Hauteur panneau : ``scale_fit`` = valeur exacte (échelle amp.), sinon grow-only."""
        widget = self._panels.get(placement.key)
        if widget is None:
            return
        try:
            current_h = max(1, int(widget.height()))
        except Exception:
            current_h = 160
        fitted = max(140, int(height))
        if not scale_fit:
            fitted = max(fitted, current_h)
        previous = getattr(widget, "fitted_height_px", None)
        widget.fitted_height_px = fitted  # type: ignore[attr-defined]
        if (
            not scale_fit
            and previous is not None
            and abs(int(previous) - fitted) <= 2
        ):
            return
        self._apply_panel_height(widget, placement)

    def apply_fitted_width(
        self, placement: PanelPlacement, width: int, *, scale_fit: bool = False
    ) -> None:
        """Largeur panneau : ``scale_fit`` = exacte (échelle temps), sinon grow-only."""
        widget = self._panels.get(placement.key)
        if widget is None:
            return
        try:
            current_w = max(1, int(widget.width()))
        except Exception:
            current_w = 200
        fitted = max(200, int(width))
        if not scale_fit:
            fitted = max(fitted, current_w)
        previous = getattr(widget, "fitted_width_px", None)
        widget.fitted_width_px = fitted  # type: ignore[attr-defined]
        if (
            not scale_fit
            and previous is not None
            and abs(int(previous) - fitted) <= 2
        ):
            return
        self._apply_panel_width(widget, placement)

    def _apply_panel_width(self, widget: object, placement: PanelPlacement) -> None:
        fitted = getattr(widget, "fitted_width_px", None)
        if fitted is None:
            widget.setMinimumWidth(0)  # type: ignore[attr-defined]
            widget.setMaximumWidth(16777215)  # type: ignore[attr-defined]
            return
        width = max(200, int(fitted))
        widget.setMinimumWidth(width)  # type: ignore[attr-defined]
        # Largeur max libre : le graph peut dépasser → scroll, pas de bande étroite figée.
        widget.setMaximumWidth(16777215)  # type: ignore[attr-defined]

    def set_graph_height(self, graph_height: int) -> None:
        """Hauteur des panneaux hors montage (RMS, spikes, résumés…)."""
        height = max(140, int(graph_height))
        self._graph_height = height
        for placement in self._order:
            if str(placement.panel).startswith("montage_"):
                continue
            widget = self._panels.get(placement.key)
            if widget is None:
                continue
            self._apply_panel_height(widget, placement)

    def _panel_min_height(self, placement: PanelPlacement) -> int:
        """Hauteur cible — override par panneau si style dédié, sinon hauteur de grille.

        Montage : jamais ``graph_height_px`` — hauteur imposée par la grille
        (dérivée de la hauteur de ligne). Autres graphs : ``_graph_height``.
        """
        if str(placement.panel).startswith("montage_"):
            return self._panel_height
        widget = self._panels.get(placement.key)
        override = getattr(widget, "style_height_px", None) if widget is not None else None
        if override is not None:
            return max(140, int(override))
        # Revue montage mixte : ne pas étirer RMS/spikes à la hauteur du montage.
        if any(str(p.panel).startswith("montage_") for p in self._order):
            return max(140, int(self._graph_height))
        return self._panel_height

    def _apply_panel_height(self, widget: object, placement: PanelPlacement) -> None:
        height = self._panel_min_height(placement)
        fitted = getattr(widget, "fitted_height_px", None)
        if fitted is not None:
            if str(placement.panel).startswith("montage_"):
                # Montage : ne pas descendre sous le budget de grille.
                height = max(height, int(fitted))
            else:
                # Échelle flottante : hauteur exacte pour le gain µV/px.
                height = max(140, int(fitted))
        widget.setMinimumHeight(height)  # type: ignore[attr-defined]
        # Hauteur fixe pour tous les graphs (scroll si la fenêtre est plus petite).
        # fill (legacy) : Expanding pour occuper le viewport.
        # Un override de style par panneau impose aussi une hauteur fixe.
        has_style_height = getattr(widget, "style_height_px", None) is not None
        fix_height = bool(
            self._uniform
            or not self._fill
            or has_style_height
            or placement.panel == "montage_continuous_raw"
        )
        if fix_height:
            widget.setMaximumHeight(height)  # type: ignore[attr-defined]
            v_policy = QSizePolicy.Policy.Fixed
        else:
            widget.setMaximumHeight(16777215)  # type: ignore[attr-defined]
            v_policy = QSizePolicy.Policy.Expanding
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, v_policy)  # type: ignore[attr-defined]

    def _on_panel_style_changed(self, placement: PanelPlacement) -> None:
        """Recaler la hauteur après un override d’affichage / style local."""
        widget = self._panels.get(placement.key)
        if widget is None:
            return
        self._apply_panel_height(widget, placement)

    def _on_x_scale_sync(self, family: str, xlim: object) -> None:
        """Double-clic : même échelle X pour tous les panels de la même famille."""
        from .panel_canvas import x_scale_family

        fam = str(family or "")
        if not fam:
            return
        try:
            x0, x1 = float(xlim[0]), float(xlim[1])  # type: ignore[index]
        except Exception:
            return
        peers: list[Any] = []
        for widget in self._panels.values():
            placement = getattr(widget, "placement", None)
            if placement is None:
                continue
            if x_scale_family(str(placement.panel)) != fam:
                continue
            peers.append(widget)
        if not peers:
            return
        # Union des étendues de données → résultat stable si on reclique.
        for widget in peers:
            host = getattr(widget, "plot_host", None)
            if host is None:
                continue
            span = host.data_xlim()
            if span is None:
                continue
            x0 = min(x0, float(span[0]))
            x1 = max(x1, float(span[1]))
        if not (x1 > x0):
            return
        shared = (x0, x1)
        for widget in peers:
            apply = getattr(widget, "apply_shared_x_scale", None)
            if callable(apply):
                apply(shared)

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

    def sync_scale_bars_from(self, style_or_settings: Any) -> None:
        """Propager légende / hauteur / échelle du dock vers les overrides locaux."""
        for widget in self._panels.values():
            sync = getattr(widget, "sync_scale_bars_from", None)
            if callable(sync):
                try:
                    sync(style_or_settings)
                except Exception:
                    pass


    def invalidate_matching(self, predicate: Callable[[PanelPlacement], bool]) -> None:
        """Marquer dirty uniquement les panneaux qui matchent ``predicate``."""
        for placement in self._order:
            if not predicate(placement):
                continue
            widget = self._panels.get(placement.key)
            if widget is not None:
                widget.invalidate()  # type: ignore[attr-defined]

    def _near_viewport(self, widget: QWidget, *, margin_factor: float = 0.5) -> bool:
        """True when the panel is on screen, or within ``margin_factor`` screens of it."""
        viewport = self.viewport()
        height = viewport.height()
        if height <= 0 or widget.height() <= 0:
            return True
        margin = max(1, int(height * float(margin_factor)))
        top_left = widget.mapTo(viewport, widget.rect().topLeft())
        rect = QRect(top_left, widget.size())
        return viewport.rect().adjusted(0, -margin, 0, margin).intersects(rect)

    def _on_scroll_moved(self, *_args: Any) -> None:
        """Pendant le scroll : suspendre les rendus lourds (prepare/draw) pour rester fluide."""
        if not self._scroll_active:
            self._scroll_active = True
            self._timer.stop()
        if not self._cull_timer.isActive():
            self._cull_timer.start()
        self._scroll_timer.start()

    def _on_scroll_settled(self) -> None:
        """Fin de scroll : dessiner les panneaux entrés en vue, reprendre la file."""
        self._scroll_active = False
        self._cull_timer.stop()
        self._cull_montage_rows()
        self._render_newly_visible()
        if self._pending and self._factory is not None and not self._timer.isActive():
            self._timer.start()

    def _cull_montage_rows(self) -> None:
        """Sur les montages hauts : ne peindre que les lignes proches du viewport."""
        for placement in self._order:
            if not str(placement.panel).startswith("montage_"):
                continue
            widget = self._panels.get(placement.key)
            if widget is None:
                continue
            host = getattr(widget, "plot_host", None)
            cull = getattr(host, "cull_rows_outside_scroll_viewport", None)
            if callable(cull):
                try:
                    cull(margin_px=200)
                except Exception:
                    pass

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
        # Ne pas démarrer un rendu lourd au milieu d’un scroll.
        if self._scroll_active:
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


    def _render_next(self) -> None:
        if self._scroll_active:
            return
        if not self._pending or self._factory is None:
            self.renderFinished.emit(self._drawn, self._elapsed)
            return
        placement = self._pending.pop(0)
        widget = self._panels.get(placement.key)
        render_s = 0.0
        if widget is not None:
            # Ne pas bloquer le thread UI sur un panneau déjà hors vue.
            if not self._near_viewport(widget):  # type: ignore[arg-type]
                self._deferred.append(placement)
                widget.mark_offscreen()  # type: ignore[attr-defined]
            else:
                widget.render(self._factory(placement))  # type: ignore[attr-defined]
                render_s = float(widget.last_render_s)  # type: ignore[attr-defined]
                self._elapsed += render_s
                self._drawn += 1
        if self._pending:
            # Budget : laisser la molette / le layout respirer après un rendu lourd.
            delay_ms = 16 if render_s > 0.025 else 0
            self._timer.setInterval(delay_ms)
            self._timer.start()
        else:
            self._timer.setInterval(0)
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
        ready_set = set(ready)
        self._deferred = [p for p in self._deferred if p not in ready_set]
        self._pending.extend(ready)
        for placement in ready:
            self._panels[placement.key].mark_pending()  # type: ignore[attr-defined]
        if not self._timer.isActive() and not self._scroll_active:
            self._elapsed = 0.0
            self._drawn = 0
            self._timer.start()


class ViewTabPage(QWidget):
    """A view tab: its toolbar lives in the main window, the grid lives here."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Ctrl+molette = zoom ; molette seule = scroll de la grille.
        self.grid = PanelGrid(self, allow_zoom=True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.grid, 1)
