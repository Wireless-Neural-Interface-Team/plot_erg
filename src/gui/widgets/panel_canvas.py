"""Panneau graphique interactif (pyqtgraph) — zoom, pan, curseur toujours actifs.

Matplotlib reste utilisé pour l’export PDF ; l’écran passe par PlotHost.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.plot_host import (
    INTERACTION_HINT,
    INTERACTION_HINT_NO_ZOOM,
    PlotHost,
)
from gui.widgets.view_params import LocalViewParams
from panel_prepare import prepare_panel
from panel_registry import RenderRequest, panel_info
from view_config import (
    PanelPlacement,
    ViewerSettings,
    apply_local_display_settings,
    continuous_sync_offset_s,
)

_STATUS_COLORS = {
    "ok": "#16a34a",
    "empty": "#ca8a04",
    "unavailable": "#dc2626",
    "pending": "#64748b",
}

_TITLE_INTERACT_HINT = (
    "Clic sur un texte = éditer · Ctrl+molette = zoom · Clic milieu/droit = pan · "
    "Double-clic = reset · Boutons Home / Pan / Zoom dans la barre"
)
_TITLE_INTERACT_HINT_NO_ZOOM = (
    "Clic sur un texte = éditer · Clic milieu/droit = pan · Double-clic = reset · "
    "Zoom désactivé sur cet aperçu"
)


def _request_sync_offset(request: RenderRequest) -> float:
    """Offset d’origine du temps (s) pour le 1er enregistrement de la requête."""
    if not request.recordings:
        return 0.0
    try:
        return float(
            continuous_sync_offset_s(
                request.recordings[0].stimulation_times_s(),
                request.settings.time_sync,
            )
        )
    except Exception:
        return 0.0


def _shift_limits_for_sync_delta(
    limits: list[tuple[tuple[float, float], tuple[float, float]]],
    delta_s: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Décale les xlim d’affichage quand l’origine du temps change (même fenêtre physique)."""
    if not delta_s:
        return limits
    return [((x0 - delta_s, x1 - delta_s), ylim) for (x0, x1), ylim in limits]


def _panel_stream_key(panel: str, *, placement_stream: str = "") -> str | None:
    """Flux WIDE/HIGH/LOW d’un panneau, ou ``None`` si non applicable."""
    key = str(panel)
    if "rms" in key:
        return "rms"
    if key == "full_recording":
        stream = str(placement_stream or "raw").strip().lower()
        return stream if stream in {"raw", "hp", "lp"} else "raw"
    if "_hp" in key or key.endswith("_hp"):
        return "hp"
    if "_lp" in key or key.endswith("_lp"):
        return "lp"
    if "_raw" in key or key.endswith("_raw"):
        return "raw"
    return None


def _axis_ylim_setting(
    figure: Any, request: RenderRequest, axis_index: int
) -> tuple[float, float] | None:
    """Limites Y manuelles configurées pour un axe, ou ``None`` (auto)."""
    settings = request.settings
    row_kinds = getattr(figure, "_erg_montage_row_kinds", None)
    if isinstance(row_kinds, (list, tuple)) and axis_index < len(row_kinds):
        kind = str(row_kinds[axis_index]).strip().lower()
        if kind == "rms":
            return settings.rms_ylim.as_tuple()
        if kind in {"raw", "hp", "lp"}:
            return settings.ylim_for_stream(kind).as_tuple()
        return None
    row_streams = getattr(figure, "_erg_montage_row_streams", None)
    if isinstance(row_streams, (list, tuple)) and axis_index < len(row_streams):
        stream = str(row_streams[axis_index]).strip().lower()
    else:
        stream = _panel_stream_key(
            str(request.panel),
            placement_stream=str(getattr(request.placement, "stream", "") or ""),
        )
    if stream is None:
        return None
    if stream == "rms":
        return settings.rms_ylim.as_tuple()
    return settings.ylim_for_stream(stream).as_tuple()


_MONTAGE_AUTO_Y_KINDS = frozenset(
    {"psth", "trial_rate", "isi", "overlay", "raster"}
)


def _axis_keeps_rendered_ylim(
    figure: Any,
    request: RenderRequest,
    axis_index: int,
    *,
    previous: RenderRequest | None = None,
) -> bool:
    """True → garder le ylim du rendu (pas le zoom interactif sauvegardé)."""
    row_kinds = getattr(figure, "_erg_montage_row_kinds", None)
    if (
        str(request.panel) == "montage_continuous_raw"
        and isinstance(row_kinds, (list, tuple))
        and axis_index < len(row_kinds)
        and str(row_kinds[axis_index]).strip().lower() in _MONTAGE_AUTO_Y_KINDS
    ):
        return True
    current = _axis_ylim_setting(figure, request, axis_index)
    if current is not None:
        return True
    if previous is not None:
        previous_ylim = _axis_ylim_setting(figure, previous, axis_index)
        if previous_ylim != current:
            return True
    return False


def _finish_panel_render(
    plot: PlotHost,
    *,
    figure: Any,
    request: RenderRequest,
    panel: str,
    preserve_limits: list | dict | None,
    previous_request: RenderRequest | None = None,
) -> str:
    """Prepare PanelSpec → PlotHost.render; restore view; coalesce draw."""
    del panel  # status comes from spec; panel kept for call-site compatibility
    try:
        spec = prepare_panel(request)
        status = plot.render_spec(spec)
    except Exception as exc:
        from panel_prepare import PanelSpec

        status = plot.render_spec(
            PanelSpec(
                title="Erreur de rendu",
                status="unavailable",
                status_message=f"Erreur de rendu :\n{exc}",
                layout_kind="placeholder",
            )
        )
        status = "unavailable"
    if preserve_limits is not None:
        restore_x = request.settings.x_limits.as_tuple() is None
        n_axes = len(list(getattr(figure, "axes", []) or []))
        restore_y = [
            not _axis_keeps_rendered_ylim(
                figure, request, index, previous=previous_request
            )
            for index in range(n_axes)
        ]
        if isinstance(preserve_limits, dict):
            plot.restore_montage_view_limits(
                preserve_limits,
                restore_x=restore_x,
                restore_y=restore_y,
            )
        else:
            plot.restore_view_limits(
                preserve_limits,
                restore_x=restore_x,
                restore_y=restore_y,
            )
    plot.draw_idle()
    return status


class PanelStyleDialog(QDialog):
    """Dialogue non modal : affichage / style propres à un seul graphique."""

    resetRequested = Signal()

    def __init__(
        self,
        placement: PanelPlacement,
        settings: ViewerSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(f"Style — {placement.title()}")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.resize(420, 640)
        self._placement = placement
        # Montage : hauteur pilotée par « hauteur de ligne » (dock Paramètres).
        show_graph_height = not str(placement.panel).startswith("montage_")

        self._tabs = QTabWidget(self)
        self.params = LocalViewParams(
            settings,
            show_sync=True,
            show_display_extras=True,
            show_graph_height=show_graph_height,
            host_tabs=self._tabs,
            parent=self,
        )

        reset_btn = QPushButton("Réinitialiser", self)
        reset_btn.setToolTip(
            "Supprimer les réglages spécifiques à ce graphique et "
            "reprendre ceux de la fenêtre."
        )
        reset_btn.clicked.connect(self.resetRequested.emit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.close)
        buttons.addButton(reset_btn, QDialogButtonBox.ButtonRole.ResetRole)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        hint = QLabel(
            "Ces réglages s’appliquent uniquement à ce graphique. "
            "Les autres panneaux de la fenêtre restent inchangés.",
            self,
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self._tabs, 1)
        layout.addWidget(buttons)

    def load_settings(self, settings: ViewerSettings) -> None:
        """Recharger le formulaire (ex. après réinitialisation)."""
        while self._tabs.count():
            page = self._tabs.widget(0)
            self._tabs.removeTab(0)
            if page is not None:
                page.deleteLater()
        old = self.params
        try:
            old.changed.disconnect()
        except (TypeError, RuntimeError):
            pass
        show_graph_height = not str(self._placement.panel).startswith("montage_")
        self.params = LocalViewParams(
            settings,
            show_sync=True,
            show_display_extras=True,
            show_graph_height=show_graph_height,
            host_tabs=self._tabs,
            parent=self,
        )
        old.deleteLater()


class PanelCanvas(QFrame):
    """Un panneau : en-tête + PlotHost pyqtgraph (toolbar toujours visible)."""

    removeRequested = Signal(object)  # PanelPlacement
    detachRequested = Signal(object)  # PanelPlacement
    styleChanged = Signal(object)  # PanelPlacement — overrides d’affichage modifiés

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
        self._style_overrides: ViewerSettings | None = None
        self._last_base_settings: ViewerSettings | None = None
        self._last_request: RenderRequest | None = None
        self._style_dialog: PanelStyleDialog | None = None

        self._plot = PlotHost(
            parent=self,
            show_toolbar=True,
            show_cursor=True,
            min_height=80,
            allow_zoom=self._allow_zoom,
        )
        self.figure = self._plot.figure
        self.canvas = self._plot  # QWidget for grab() / range-bar attach
        self.toolbar = self._plot.toolbar
        self.plot_host = self._plot

        interact_hint = (
            _TITLE_INTERACT_HINT if self._allow_zoom else _TITLE_INTERACT_HINT_NO_ZOOM
        )
        self._title = QLabel(placement.title())
        self._title.setObjectName("panelTitle")
        self._title.setToolTip(f"{placement.title()}\n{interact_hint}")
        self._title.setWordWrap(False)
        self._title.setMinimumWidth(0)
        self._title.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._status_label = QLabel("—")
        self._status_label.setObjectName("panelStatus")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._style_button = self._make_tool_button(
            "⚙",
            "Affichage / style de ce graphique uniquement",
        )
        self._style_button.setCheckable(True)
        self._style_button.clicked.connect(self._open_style_dialog)
        self._detach_button = self._make_tool_button("⤢", "Ouvrir ce panneau dans sa propre fenêtre")
        self._detach_button.clicked.connect(lambda: self.detachRequested.emit(self.placement))
        self._close_button = self._make_tool_button("✕", "Retirer ce panneau de la vue")
        self._close_button.clicked.connect(lambda: self.removeRequested.emit(self.placement))

        header = QHBoxLayout()
        header.setContentsMargins(8, 4, 4, 0)
        header.setSpacing(4)
        header.addWidget(self._title, 1)
        header.addWidget(self._status_label, 0)
        header.addWidget(self._style_button, 0)
        header.addWidget(self._detach_button, 0)
        header.addWidget(self._close_button, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 4)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self._plot, 1)
        self._refresh_style_button()

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
        if self._style_dialog is not None:
            self._style_dialog.setWindowTitle(f"Style — {title}")

    # ----------------------------------------------------------------- style

    @property
    def has_style_overrides(self) -> bool:
        return self._style_overrides is not None

    @property
    def style_height_px(self) -> int | None:
        """Hauteur dédiée si ce panneau a un override d’affichage.

        Les montages ignorent ``graph_height_px`` : leur hauteur vient de
        ``montage_row_min_height_px`` (via la grille / ``_adapt_montage_height``).
        """
        if self._style_overrides is None:
            return None
        if str(self.placement.panel).startswith("montage_"):
            return None
        return max(140, int(self._style_overrides.graph_height_px))

    def _refresh_style_button(self) -> None:
        active = self._style_overrides is not None
        self._style_button.blockSignals(True)
        self._style_button.setChecked(active)
        self._style_button.blockSignals(False)
        tip = "Affichage / style de ce graphique uniquement"
        if active:
            tip += "\n(réglages spécifiques actifs — Réinitialiser dans le dialogue)"
        self._style_button.setToolTip(tip)

    def _seed_settings_for_dialog(self) -> ViewerSettings:
        if self._style_overrides is not None:
            return self._style_overrides
        if self._last_base_settings is not None:
            return self._last_base_settings
        return ViewerSettings()

    def _open_style_dialog(self) -> None:
        # setCheckable : le clic bascule l’état — on rétablit l’indicateur d’override.
        self._refresh_style_button()
        if self._style_dialog is not None:
            self._style_dialog.raise_()
            self._style_dialog.activateWindow()
            return
        dialog = PanelStyleDialog(
            self.placement, self._seed_settings_for_dialog(), parent=self
        )
        dialog.params.changed.connect(self._on_style_params_changed)
        dialog.resetRequested.connect(self._reset_style_overrides)
        dialog.destroyed.connect(self._on_style_dialog_destroyed)
        self._style_dialog = dialog
        dialog.show()

    def _on_style_dialog_destroyed(self, *_args: Any) -> None:
        self._style_dialog = None

    def _on_style_params_changed(self) -> None:
        if self._style_dialog is None:
            return
        self._style_overrides = self._style_dialog.params.settings()
        self._refresh_style_button()
        self._rerender_with_overrides()
        self.styleChanged.emit(self.placement)

    def _reset_style_overrides(self) -> None:
        self._style_overrides = None
        self._plot.clear_text_pins()
        self._refresh_style_button()
        base = self._last_base_settings
        if base is not None and self._style_dialog is not None:
            # Reconnecter après rebuild du formulaire.
            self._style_dialog.load_settings(base)
            self._style_dialog.params.changed.connect(self._on_style_params_changed)
        self._rerender_with_overrides()
        self.styleChanged.emit(self.placement)

    def _rerender_with_overrides(self) -> None:
        if self._last_request is None:
            self._dirty = True
            return
        # Conserver la vue (zoom/pan) pendant l’ajustement de style.
        request = replace(self._last_request, preserve_view=True)
        self.render(request)

    def _with_style_overrides(self, request: RenderRequest) -> RenderRequest:
        self._last_base_settings = request.settings
        self._last_request = request
        if self._style_overrides is None:
            return request
        settings = apply_local_display_settings(request.settings, self._style_overrides)
        return replace(request, settings=settings)

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
        prev_request = self._last_request
        old_offset = _request_sync_offset(prev_request) if prev_request else 0.0
        request = self._with_style_overrides(request)
        is_montage = str(self.placement.panel) == "montage_continuous_raw"
        saved_limits: list | dict | None = None
        if request.preserve_view:
            if is_montage:
                saved_limits = self._plot.capture_montage_view_limits()
            else:
                saved_limits = self._plot.capture_view_limits()
        if (
            isinstance(saved_limits, list)
            and saved_limits is not None
            and prev_request is not None
        ):
            delta = _request_sync_offset(request) - old_offset
            if delta:
                saved_limits = _shift_limits_for_sync_delta(saved_limits, delta)
        elif isinstance(saved_limits, dict) and prev_request is not None:
            delta = _request_sync_offset(request) - old_offset
            if delta:
                saved_limits = {
                    key: ((xlim[0] + delta, xlim[1] + delta), ylim)
                    for key, (xlim, ylim) in saved_limits.items()
                }
        started = time.perf_counter()
        status = _finish_panel_render(
            self._plot,
            figure=self.figure,
            request=request,
            panel=self.placement.panel,
            preserve_limits=saved_limits,
            previous_request=prev_request,
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

        self._plot = PlotHost(
            parent=self,
            show_toolbar=True,
            show_cursor=True,
            min_height=280,
            allow_zoom=True,
        )
        self.figure = self._plot.figure
        self.canvas = self._plot
        self.plot_host = self._plot

        self._status = QLabel("—")
        self._status.setObjectName("panelStatus")
        self._last_render_s = 0.0
        self._last_request: RenderRequest | None = None

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
        prev_request = self._last_request
        old_offset = _request_sync_offset(prev_request) if prev_request else 0.0
        self._last_request = request
        is_montage = str(self.placement.panel) == "montage_continuous_raw"
        saved_limits: list | dict | None = None
        if request.preserve_view:
            if is_montage:
                saved_limits = self._plot.capture_montage_view_limits()
            else:
                saved_limits = self._plot.capture_view_limits()
        if isinstance(saved_limits, list) and prev_request is not None:
            delta = _request_sync_offset(request) - old_offset
            if delta:
                saved_limits = _shift_limits_for_sync_delta(saved_limits, delta)
        elif isinstance(saved_limits, dict) and prev_request is not None:
            delta = _request_sync_offset(request) - old_offset
            if delta:
                saved_limits = {
                    key: ((xlim[0] + delta, xlim[1] + delta), ylim)
                    for key, (xlim, ylim) in saved_limits.items()
                }
        started = time.perf_counter()
        status = _finish_panel_render(
            self._plot,
            figure=self.figure,
            request=request,
            panel=self.placement.panel,
            preserve_limits=saved_limits,
            previous_request=prev_request,
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


class DetachedZoomWindow(QDialog):
    """Zooms continuous d’un canal regroupés dans une seule fenêtre."""

    closed = Signal()
    refreshRequested = Signal(object)  # DetachedZoomWindow

    def __init__(
        self,
        channel_name: str,
        placements: Sequence[PanelPlacement],
        settings: ViewerSettings,
        *,
        panel_height: int = 400,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.channel_name = str(channel_name)
        self.settings = settings
        self._placements: tuple[PanelPlacement, ...] = tuple(placements)
        self._panel_height = max(160, int(panel_height))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(f"{self.channel_name} — Zooms")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(1100, max(560, self._panel_height + 220))

        from gui.widgets.panel_grid import PanelGrid

        self.grid = PanelGrid(self, allow_zoom=True)
        self._status = QLabel("—")
        self._status.setObjectName("panelStatus")
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
        self._title_label = QLabel(self._header_text())
        self._title_label.setWordWrap(True)
        header.addWidget(self._title_label, 1)
        header.addWidget(self._status, 0)
        plot_layout.addLayout(header)
        plot_layout.addWidget(self.grid, 1)

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

        self.grid.renderFinished.connect(self._on_render_finished)
        self._configure_grid()

    def _header_text(self) -> str:
        n = len(self._placements)
        return f"{self.channel_name} — {n} zoom(s)"

    def _configure_grid(self) -> None:
        self.grid.configure(
            self._placements,
            columns=1,
            panel_height=self._panel_height,
            uniform=True,
            fill=False,
        )

    def local_settings(self) -> ViewerSettings:
        return self.params.settings()

    def update_placements(
        self,
        placements: Sequence[PanelPlacement],
        *,
        panel_height: int | None = None,
    ) -> None:
        """Mettre à jour la grille sans recréer la fenêtre."""
        self._placements = tuple(placements)
        if panel_height is not None:
            self._panel_height = max(160, int(panel_height))
        self.setWindowTitle(f"{self.channel_name} — Zooms")
        self._title_label.setText(self._header_text())
        self._configure_grid()

    @property
    def placements(self) -> tuple[PanelPlacement, ...]:
        return self._placements

    def contains_key(self, key: str) -> bool:
        return any(p.key == key for p in self._placements)

    def _on_local_params_changed(self) -> None:
        self.settings = self.local_settings()
        self.refreshRequested.emit(self)

    def schedule_render(self, factory: Any, *, force: bool = True) -> None:
        self.grid.schedule_render(factory, force=force)

    def _on_render_finished(self, drawn: int, elapsed_s: float) -> None:
        self._status.setText(f"{drawn} panneau(x) · {elapsed_s * 1000:.0f} ms")

    def closeEvent(self, event: Any) -> None:  # noqa: D102
        self.closed.emit()
        super().closeEvent(event)
