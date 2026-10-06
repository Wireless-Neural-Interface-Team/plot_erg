"""Fenêtre de vue multi-panneaux avec paramètres locaux et zooms personnalisés."""

from __future__ import annotations

import itertools
import time
import uuid
from dataclasses import replace
from typing import Any, Callable, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.custom_zoom_dialog import ask_custom_zoom
from gui.widgets.panel_grid import PanelGrid
from gui.widgets.view_params import LocalViewParams
from panel_registry import RenderRequest
from view_config import PanelPlacement, ViewerSettings, is_section_independent

_WINDOW_COUNTER = itertools.count(1)

RequestFactory = Callable[["ViewSessionWindow", PanelPlacement], RenderRequest | None]


class ViewSessionWindow(QMainWindow):
    """Vue ouverte depuis un bouton : panneaux + paramètres locaux à cette fenêtre."""

    closed = Signal(str)
    refreshRequested = Signal(object)

    def __init__(
        self,
        *,
        title: str,
        placements: Sequence[PanelPlacement],
        channel_name: str,
        channel_index: int,
        base_settings: ViewerSettings,
        columns: int = 2,
        panel_height: int = 320,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.window_id = f"view-{next(_WINDOW_COUNTER)}-{uuid.uuid4().hex[:6]}"
        self.channel_name = str(channel_name)
        self.channel_index = int(channel_index)
        self._base_placements = tuple(
            p for p in placements if not p.has_custom_zoom
        ) or tuple(placements)
        self.placements = tuple(placements)
        self.settings = replace(base_settings)
        self._columns = int(columns)
        self._panel_height = int(panel_height)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle(f"{title} — {self.channel_name}")
        self.resize(1280, 820)
        self._title = title

        self._status = QLabel("—")
        self._status.setObjectName("panelStatus")

        self.grid = PanelGrid(self)
        self.grid.configure(
            self.placements, columns=self._columns, panel_height=self._panel_height
        )
        self.grid.removeRequested.connect(self._on_panel_removed)
        self.grid.detachRequested.connect(lambda _p: None)

        self.params = LocalViewParams(self.settings, parent=self)
        self.params.changed.connect(self._on_params_changed)

        self._btn_add_zoom = QPushButton("Ajouter un zoom…", self)
        self._btn_add_zoom.setObjectName("primaryButton")
        self._btn_add_zoom.setToolTip(
            "Définir une fenêtre temporelle [t₀, t₁] relative à la stimulation "
            "et ajouter les graphiques correspondants dans cette vue."
        )
        self._btn_add_zoom.clicked.connect(self.add_custom_zoom)

        header = QHBoxLayout()
        title_label = QLabel(f"<b>{title}</b> · canal {self.channel_name}")
        title_label.setObjectName("sectionTitle")
        header.addWidget(title_label, 1)
        header.addWidget(self._status)

        plot_side = QWidget(self)
        plot_layout = QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.setSpacing(4)
        plot_layout.addLayout(header)
        plot_layout.addWidget(self.grid, 1)

        params_side = QWidget(self)
        params_layout = QVBoxLayout(params_side)
        params_layout.setContentsMargins(6, 6, 6, 6)
        params_layout.addWidget(self._btn_add_zoom)
        params_layout.addWidget(self.params, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.addWidget(plot_side)
        splitter.addWidget(params_side)
        plot_side.setMinimumWidth(320)
        params_side.setMinimumWidth(240)
        params_side.setMaximumWidth(16777215)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([860, 380])
        self.setCentralWidget(splitter)

        self._request_factory: RequestFactory | None = None
        self._last_redraw = 0.0

    def set_request_factory(self, factory: RequestFactory) -> None:
        self._request_factory = factory

    def set_trial_count(self, n_trials: int) -> None:
        self.params.set_trial_count(n_trials)

    def local_settings(self) -> ViewerSettings:
        return self.params.settings()

    def needs_channel_compute(self) -> bool:
        return any(
            p.panel
            not in {"full_recording", "mea_layout", "impedance", "montage_continuous_raw"}
            for p in self.placements
        )

    def add_custom_zoom(self) -> None:
        """Dupliquer les panneaux de base avec une fenêtre [t0, t1] définie par l’utilisateur."""
        settings = self.local_settings()
        spec = ask_custom_zoom(
            self,
            default_t0=float(settings.zoom_onset_t0_s),
            default_t1=float(settings.zoom_onset_t1_s),
        )
        if spec is None:
            return
        sources = [
            p
            for p in self._base_placements
            if not is_section_independent(p.panel)
        ]
        if not sources:
            QMessageBox.information(
                self,
                "Ajouter un zoom",
                "Cette vue n’a pas de graphique zoomable (canaux / analyse).",
            )
            return
        added: list[PanelPlacement] = []
        existing = {p.key for p in self.placements}
        for placement in sources:
            zoomed = placement.with_custom_zoom(
                spec.t0_s, spec.t1_s, label=spec.label
            )
            if zoomed.key in existing:
                continue
            added.append(zoomed)
            existing.add(zoomed.key)
        if not added:
            QMessageBox.information(
                self, "Ajouter un zoom", "Ce zoom est déjà présent dans la vue."
            )
            return
        self.placements = tuple([*self.placements, *added])
        self.grid.configure(
            self.placements, columns=self._columns, panel_height=self._panel_height
        )
        self.refreshRequested.emit(self)

    def _on_panel_removed(self, placement: PanelPlacement) -> None:
        self.placements = tuple(p for p in self.placements if p.key != placement.key)
        # Ne pas retirer le dernier panneau de base.
        if not self.placements:
            self.placements = self._base_placements
        self.grid.configure(
            self.placements, columns=self._columns, panel_height=self._panel_height
        )
        self.refreshRequested.emit(self)

    def redraw(self) -> None:
        self.settings = self.local_settings()
        if self._request_factory is None:
            return

        def factory(placement: PanelPlacement) -> RenderRequest:
            request = self._request_factory(self, placement)  # type: ignore[misc]
            if request is not None:
                return request
            return RenderRequest(
                placement=placement,
                recordings=[],
                labels=[],
                colors=[],
                legend_flags=[],
                channel_index=self.channel_index,
                channel_name=self.channel_name,
                settings=self.settings,
                probe_layout=None,
                impedance_sessions=[],
            )

        started = time.perf_counter()
        self.grid.invalidate_all()
        drawn = self.grid.render_dirty_now(factory)
        self._last_redraw = time.perf_counter() - started
        self._status.setText(f"{drawn} panneau(x) · {self._last_redraw * 1000:.0f} ms")

    def _on_params_changed(self) -> None:
        self.settings = self.local_settings()
        self.refreshRequested.emit(self)

    def closeEvent(self, event: Any) -> None:  # noqa: D102
        self.closed.emit(self.window_id)
        super().closeEvent(event)
