"""Fenêtre de graphique : canal figé, paramètres locaux, canvas interactif."""

from __future__ import annotations

import itertools
import time
import uuid
from dataclasses import replace
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.custom_zoom_dialog import ask_custom_zoom
from gui.widgets.interactive_canvas import InteractiveCanvas
from gui.widgets.view_params import LocalViewParams
from panel_registry import RenderRequest, panel_info, render_panel
from view_config import PanelPlacement, ViewerSettings, is_section_independent, panel_label

_STATUS_COLORS = {
    "ok": "#16a34a",
    "empty": "#ca8a04",
    "unavailable": "#dc2626",
    "pending": "#64748b",
}

_WINDOW_COUNTER = itertools.count(1)


class FrozenGraphWindow(QMainWindow):
    """Fenêtre indépendante sur un canal, avec canvas interactif."""

    closed = Signal(str)
    refreshRequested = Signal(object)

    def __init__(
        self,
        *,
        placement: PanelPlacement,
        channel_name: str,
        channel_index: int,
        base_settings: ViewerSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.window_id = f"frozen-{next(_WINDOW_COUNTER)}-{uuid.uuid4().hex[:6]}"
        self.placement = placement
        self.channel_name = str(channel_name)
        self.channel_index = int(channel_index)
        self.settings = replace(base_settings)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlag(Qt.WindowType.Window, True)

        info = panel_info(placement.panel)
        self.setWindowTitle(f"{self.channel_name} — {placement.title()}")
        self.resize(1040, max(520, info.preferred_height_px + 180))

        self._plot = InteractiveCanvas(
            figsize=(8.0, 5.0),
            parent=self,
            show_toolbar=True,
            show_cursor=True,
            min_height=320,
        )
        self.figure = self._plot.figure
        self.canvas = self._plot.canvas
        self._status = QLabel("—")
        self._status.setObjectName("panelStatus")

        params = QWidget(self)
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 6)
        params_layout.setSpacing(8)

        identity = QGroupBox("Identité", params)
        identity_form = QFormLayout(identity)
        self._graph_label = QLabel(panel_label(placement.panel))
        self._section_label = QLabel(self._section_caption())
        identity_form.addRow("Canal", QLabel(self.channel_name))
        identity_form.addRow("Graphique", self._graph_label)
        identity_form.addRow("Fenêtre", self._section_label)
        params_layout.addWidget(identity)

        self._btn_zoom = QPushButton("Définir un zoom…", params)
        self._btn_zoom.setObjectName("primaryButton")
        self._btn_zoom.setEnabled(not is_section_independent(placement.panel))
        self._btn_zoom.setToolTip(
            "Choisir une fenêtre temporelle [t₀, t₁] relative à la stimulation."
        )
        self._btn_zoom.clicked.connect(self.apply_custom_zoom)
        self._btn_full = QPushButton("Vue complète", params)
        self._btn_full.setEnabled(placement.has_custom_zoom)
        self._btn_full.clicked.connect(self.clear_custom_zoom)
        params_layout.addWidget(self._btn_zoom)
        params_layout.addWidget(self._btn_full)

        self.params = LocalViewParams(self.settings, parent=params)
        self.params.changed.connect(self._on_local_params_changed)
        params_layout.addWidget(self.params, 1)

        plot_side = QWidget(self)
        plot_layout = QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        header = QHBoxLayout()
        header.addWidget(QLabel(f"<b>{self.channel_name}</b>"), 1)
        header.addWidget(self._status, 0)
        plot_layout.addLayout(header)
        plot_layout.addWidget(self._plot, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(plot_side)
        splitter.addWidget(params)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([760, 280])
        self.setCentralWidget(splitter)

        self._request_factory: Callable[[FrozenGraphWindow], RenderRequest | None] | None = None

    def _section_caption(self) -> str:
        if self.placement.has_custom_zoom:
            name = self.placement.zoom_label.strip() or "zoom"
            return f"{name} [{self.placement.zoom_t0_s:g} … {self.placement.zoom_t1_s:g} s]"
        return "complète"

    def set_request_factory(
        self, factory: Callable[[FrozenGraphWindow], RenderRequest | None]
    ) -> None:
        self._request_factory = factory

    def set_trial_count(self, n_trials: int) -> None:
        self.params.set_trial_count(n_trials)

    def needs_channel_compute(self) -> bool:
        panel = self.placement.panel
        if panel in {"full_recording", "mea_layout", "impedance", "montage_continuous_raw"}:
            return False
        return True

    def local_settings(self) -> ViewerSettings:
        return self.params.settings()

    def apply_custom_zoom(self) -> None:
        settings = self.local_settings()
        spec = ask_custom_zoom(
            self,
            default_t0=float(settings.zoom_onset_t0_s),
            default_t1=float(settings.zoom_onset_t1_s),
        )
        if spec is None:
            return
        self.placement = self.placement.with_custom_zoom(
            spec.t0_s, spec.t1_s, label=spec.label
        )
        self._section_label.setText(self._section_caption())
        self._btn_full.setEnabled(True)
        self.setWindowTitle(f"{self.channel_name} — {self.placement.title()}")
        self.refreshRequested.emit(self)

    def clear_custom_zoom(self) -> None:
        self.placement = PanelPlacement(panel=self.placement.panel, section="full")
        self._section_label.setText(self._section_caption())
        self._btn_full.setEnabled(False)
        self.setWindowTitle(f"{self.channel_name} — {self.placement.title()}")
        self.refreshRequested.emit(self)

    def redraw(self) -> None:
        self.settings = self.local_settings()
        if self._request_factory is None:
            return
        request = self._request_factory(self)
        if request is None:
            self._status.setText("aucune donnée")
            self._status.setStyleSheet(f"color: {_STATUS_COLORS['unavailable']};")
            return
        started = time.perf_counter()
        try:
            status = render_panel(self.figure, request)
            from gui.mpl_theme import apply_intan_scope_style

            apply_intan_scope_style(self.figure)
        except Exception as exc:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Erreur de rendu :\n{exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#ff6b6b",
            )
            ax.set_axis_off()
            from gui.mpl_theme import apply_intan_scope_style

            apply_intan_scope_style(self.figure)
            status = "unavailable"
        elapsed = time.perf_counter() - started
        self._plot.draw_idle()
        self._plot.enable_default_pan()
        self._status.setText(f"{elapsed * 1000:.0f} ms")
        self._status.setStyleSheet(f"color: {_STATUS_COLORS.get(status, '#64748b')};")

    def _on_local_params_changed(self, *_args: Any) -> None:
        self.settings = self.local_settings()
        self.refreshRequested.emit(self)

    def closeEvent(self, event: Any) -> None:  # noqa: D102
        self.closed.emit(self.window_id)
        super().closeEvent(event)
