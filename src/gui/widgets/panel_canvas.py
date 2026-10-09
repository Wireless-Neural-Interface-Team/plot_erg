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
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.interactive_canvas import (
    INTERACTION_HINT,
    INTERACTION_HINT_NO_ZOOM,
    InteractiveCanvas,
)
from gui.widgets.view_params import LocalViewParams
from panel_registry import RenderRequest, panel_info
from view_config import PanelPlacement, ViewerSettings

_STATUS_COLORS = {
    "ok": "#16a34a",
    "empty": "#ca8a04",
    "unavailable": "#dc2626",
    "pending": "#64748b",
}

_TITLE_INTERACT_HINT = (
    "Molette = zoom · Clic milieu/droit = pan · Double-clic = reset · "
    "Boutons Home / Pan / Zoom dans la barre"
)
_TITLE_INTERACT_HINT_NO_ZOOM = (
    "Clic milieu/droit = pan · Double-clic = reset · "
    "Zoom désactivé sur cet aperçu"
)


def _apply_post_render_style(figure: Any, request: RenderRequest, panel: str) -> None:
    """Thème scope clair — montage : spines/ticks sans écraser la surbrillance."""
    from gui.mpl_theme import apply_intan_scope_style

    if panel == "montage_continuous_raw":
        try:
            from gui.theme import SCOPE_BG, SCOPE_FG, SCOPE_GRID, SCOPE_SPINE

            figure.patch.set_facecolor(SCOPE_BG)
            show_borders = bool(getattr(request.style, "show_borders", True))
            show_grid = bool(request.style.grid)
            grid_alpha = float(request.style.grid_alpha)
            for ax in list(getattr(figure, "axes", []) or []):
                for spine in ax.spines.values():
                    spine.set_color(SCOPE_SPINE)
                    spine.set_linewidth(0.8)
                    spine.set_visible(show_borders)
                ax.tick_params(colors=SCOPE_FG)
                try:
                    ax.xaxis.label.set_color(SCOPE_FG)
                    ax.yaxis.label.set_color(SCOPE_FG)
                except Exception:
                    pass
                if show_grid:
                    ax.grid(True, color=SCOPE_GRID, linewidth=0.7, alpha=grid_alpha)
                    ax.set_axisbelow(True)
                # facecolor (surbrillance canal) volontairement non touchée.
        except Exception:
            pass
        return
    apply_intan_scope_style(
        figure,
        grid=request.style.grid,
        grid_alpha=request.style.grid_alpha,
        show_borders=getattr(request.style, "show_borders", None),
        ticks_inside=getattr(request.style, "ticks_inside", None),
    )


def _paint_render_error(figure: Any, exc: BaseException) -> None:
    from gui.mpl_theme import apply_intan_scope_style

    figure.clear()
    axes = figure.add_subplot(111)
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
    apply_intan_scope_style(figure, grid=False, show_borders=False)


def _finish_panel_render(
    plot: InteractiveCanvas,
    *,
    figure: Any,
    request: RenderRequest,
    panel: str,
    preserve_limits: list | None,
) -> str:
    """Rendu commun : style, restore vue, draw. Pas de pan toolbar auto (clic gauche libre)."""
    from panel_registry import render_panel

    try:
        status = render_panel(figure, request)
        _apply_post_render_style(figure, request, panel)
    except Exception as exc:
        _paint_render_error(figure, exc)
        status = "unavailable"
    if preserve_limits is not None:
        plot.restore_view_limits(preserve_limits)
    plot.draw_idle()
    return status


class PanelCanvas(QFrame):
    """Un panneau : en-tête + canvas interactif (toolbar toujours visible)."""

    removeRequested = Signal(object)  # PanelPlacement
    detachRequested = Signal(object)  # PanelPlacement

    def __init__(
        self,
        placement: PanelPlacement,
        parent: QWidget | None = None,
        *,
        allow_zoom: bool = True,
    ) -> None:
        super().__init__(parent)
        self.placement = placement
        self._allow_zoom = bool(allow_zoom)
        self.setObjectName("panelCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # Expanding : occupe toute la largeur de la cellule de grille (même largeur).
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self._dirty = True
        self._last_render_s = 0.0
        self._status = "pending"

        info = panel_info(placement.panel)
        # Montage multi-lignes : pas de constrained layout (solveur O(n_axes) au draw).
        fig_layout = (
            None
            if placement.panel.startswith("montage_")
            else "constrained"
        )
        # figsize / min_height dérivés du paramètre preferred_height_px du catalogue.
        pref_h = max(160, int(info.preferred_height_px))
        fig_h_in = max(2.0, min(6.0, pref_h / 120.0))
        self._plot = InteractiveCanvas(
            figsize=(4.2, fig_h_in),
            parent=self,
            show_toolbar=True,
            show_cursor=True,
            min_height=max(120, pref_h // 3),
            layout=fig_layout,
            allow_zoom=self._allow_zoom,
        )
        self.figure = self._plot.figure
        self.canvas = self._plot.canvas
        self.toolbar = self._plot.toolbar

        interact_hint = (
            _TITLE_INTERACT_HINT if self._allow_zoom else _TITLE_INTERACT_HINT_NO_ZOOM
        )
        self._title = QLabel(placement.title())
        self._title.setObjectName("panelTitle")
        self._title.setToolTip(f"{placement.title()}\n{interact_hint}")
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
        hint = (
            _TITLE_INTERACT_HINT if self._allow_zoom else _TITLE_INTERACT_HINT_NO_ZOOM
        )
        self._title.setToolTip(f"{title}\n{hint}")
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
        """Dessiner le panneau (pan = clic milieu/droit ; clic gauche libre pour plages)."""
        saved_limits = (
            self._plot.capture_view_limits() if request.preserve_view else None
        )
        started = time.perf_counter()
        status = _finish_panel_render(
            self._plot,
            figure=self.figure,
            request=request,
            panel=self.placement.panel,
            preserve_limits=saved_limits,
        )
        self._last_render_s = time.perf_counter() - started
        self._dirty = False
        self._set_status(status, f"{self._last_render_s * 1000:.0f} ms")
        return status

    def _set_status(self, status: str, text: str) -> None:
        self._status = status
        color = _STATUS_COLORS.get(status, "#64748b")
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color};")
        hint = INTERACTION_HINT if self._allow_zoom else INTERACTION_HINT_NO_ZOOM
        tooltip = {
            "ok": f"Interactif — {hint}",
            "empty": "Rien à dessiner avec les réglages actuels",
            "unavailable": "Données indisponibles pour ce panneau",
            "pending": "Dessiné à l’entrée dans la vue",
        }.get(status, status)
        self._status_label.setToolTip(
            f"{tooltip} — dernier rendu {self._last_render_s * 1000:.0f} ms"
        )

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
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(placement.title())
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(1100, max(480, info.preferred_height_px + 160))

        fig_layout = (
            None if placement.panel == "montage_continuous_raw" else "constrained"
        )
        self._plot = InteractiveCanvas(
            figsize=(8.0, 5.0),
            parent=self,
            show_toolbar=True,
            show_cursor=True,
            min_height=280,
            layout=fig_layout,
        )
        self.figure = self._plot.figure
        self.canvas = self._plot.canvas

        self._status = QLabel("—")
        self._status.setObjectName("panelStatus")
        self._last_render_s = 0.0

        self.params = LocalViewParams(
            settings,
            show_sync=True,
            show_display_extras=True,
            parent=self,
        )
        self.params.changed.connect(self._on_local_params_changed)

        plot_side = QWidget(self)
        plot_layout = QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.setSpacing(4)
        header = QHBoxLayout()
        self._title_label = QLabel(placement.title())
        header.addWidget(self._title_label, 1)
        header.addWidget(self._status, 0)
        plot_layout.addLayout(header)
        plot_layout.addWidget(self._plot, 1)

        from gui.form_widgets import FitWidthScrollArea

        params_inner = QWidget(self)
        params_layout = QVBoxLayout(params_inner)
        params_layout.setContentsMargins(6, 6, 6, 6)
        params_layout.addWidget(self.params, 1)

        params_side = FitWidthScrollArea(self)
        params_side.setWidget(params_inner)
        params_side.setMinimumWidth(220)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.addWidget(plot_side)
        splitter.addWidget(params_side)
        plot_side.setMinimumWidth(320)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([740, 360])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def local_settings(self) -> ViewerSettings:
        return self.params.settings()

    def update_placement(self, placement: PanelPlacement) -> None:
        """Mettre à jour le zoom / titre sans recréer la fenêtre."""
        self.placement = placement
        title = placement.title()
        self.setWindowTitle(title)
        self._title_label.setText(title)

    def _on_local_params_changed(self) -> None:
        self.settings = self.local_settings()
        self.refreshRequested.emit(self)

    def render(self, request: RenderRequest) -> str:
        saved_limits = (
            self._plot.capture_view_limits() if request.preserve_view else None
        )
        started = time.perf_counter()
        status = _finish_panel_render(
            self._plot,
            figure=self.figure,
            request=request,
            panel=self.placement.panel,
            preserve_limits=saved_limits,
        )
        self._last_render_s = time.perf_counter() - started
        self._status.setText(f"{self._last_render_s * 1000:.0f} ms")
        self._status.setStyleSheet(f"color: {_STATUS_COLORS.get(status, '#64748b')};")
        self._status.setToolTip(
            f"Interactif — {INTERACTION_HINT} — "
            f"dernier rendu {self._last_render_s * 1000:.0f} ms"
        )
        return status

    def closeEvent(self, event: Any) -> None:  # noqa: D102
        self.closed.emit(self.placement)
        super().closeEvent(event)
