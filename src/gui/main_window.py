"""Visionneuse interactive pour enregistrements Intan RHS / ERG.

Paradigme canal d’abord :
- gauche  : Session (zone Mapping + enregistrements / liste)
- centre  : aperçu léger du canal sélectionné
- Inspecter / double-clic → fenêtre canal (barres de plage + graphs)
- montage multi-canaux : optionnel (Revue montage)
- bas     : Control Panel (flux WIDE / HIGH / LOW)
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from PySide6.QtCore import QByteArray, QSettings, Qt
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from config import AnalysisConfig
from display_config import resolve_recording_plot_colors
from gui.defaults import (
    probe_path_from_defaults,
    viewer_settings_from_defaults,
)
from gui.jobs import BuildRequest, BuildWorker, ChannelEnsureRequest, ChannelEnsureWorker, Debouncer, TaskWorker
from gui.styles import APP_STYLESHEET
from gui.widgets.channel_panel import ChannelPanel
from gui.widgets.control_panel import ControlPanel
from gui.widgets.dialogs import CacheDialog, ExportDatasetDialog
from gui.widgets.panel_canvas import DetachedPanelWindow
from gui.widgets.panel_grid import ViewTabPage
from gui.widgets.panel_picker import pick_panels
from gui.widgets.params_panel import ParamsPanel
from gui.widgets.recordings_panel import RecordingEntry, RecordingsPanel
from gui.widgets.session_panel import SessionPanel
from gui.widgets.status_panel import StatusPanel
from gui.widgets.view_session_window import ViewSessionWindow
from gui.widgets.channel_analysis_window import ChannelAnalysisWindow
from panel_registry import RenderRequest, highlight_zooms_from_placements
from view_config import (
    AnalysisSettings,
    GLOBAL_PANEL_FIELD_NAMES,
    PanelPlacement,
    ViewTab,
    ViewerSettings,
    WorkspaceLayout,
)

_ORG = "plot_erg"
_APP = "viewer"


def _impedance_sessions(entries: list[RecordingEntry]) -> list[Any]:
    """Sessions d’impédance uniques, dans l’ordre d’apparition."""
    sessions: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        for session in getattr(entry.recording, "impedance_sessions", []) or []:
            marker = (str(getattr(session, "label", "")), str(getattr(session, "csv_path", "")))
            if marker in seen:
                continue
            seen.add(marker)
            sessions.append(session)
    return sessions


class ViewerWindow(QMainWindow):
    """Fenêtre principale style RHX : session, scope, control panel."""

    def __init__(
        self,
        defaults: dict[str, Any] | None = None,
        *,
        pdf_callback: Callable[[list[AnalysisConfig]], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._defaults = dict(defaults or {})
        self._pdf_callback = pdf_callback
        self._workspace = WorkspaceLayout()
        self._settings = viewer_settings_from_defaults(self._defaults)
        self._probe_path: Path | None = None
        self._probe_layout: Any | None = None
        self._build_worker: BuildWorker | None = None
        self._ensure_worker: ChannelEnsureWorker | None = None
        self._task_worker: TaskWorker | None = None
        self._detached: dict[str, DetachedPanelWindow] = {}
        self._view_sessions: dict[str, ViewSessionWindow] = {}
        self._channel_windows: dict[str, ChannelAnalysisWindow] = {}
        self._config_dirty = False
        self._force_redraw = False
        self._cache_root: Path | None = None
        self._warned_channel_mismatch = False
        self._pending_ensure_all = False
        self._pending_window_id: str | None = None
        self._ensure_queue: list[tuple[list[int], str | None, dict[str, bool]]] = []

        self.setWindowTitle("plot_erg — Intan RHX viewer")
        self.resize(1600, 960)

        # Centre = aperçu canal + control panel.
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("centralViews")
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.control_panel = ControlPanel(self)
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.tabs, 1)
        central_layout.addWidget(self.control_panel, 0)
        self.setCentralWidget(central)

        self.recordings_panel = RecordingsPanel(self)
        self.channel_panel = ChannelPanel(self)
        self.session_panel = SessionPanel(self.recordings_panel, self.channel_panel, self)
        self.params_panel = ParamsPanel(self._defaults, self)
        self.status_panel = StatusPanel(self)

        self._dock_session = self._add_dock(
            "Session", self.session_panel, Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self._dock_params = self._add_dock(
            "Paramètres", self.params_panel, Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._dock_status = self._add_dock(
            "Journal", self.status_panel, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._dock_status.hide()
        self.resizeDocks([self._dock_session], [360], Qt.Orientation.Horizontal)
        self.resizeDocks([self._dock_params], [340], Qt.Orientation.Horizontal)

        self._redraw_debouncer = Debouncer(110, self)
        self._redraw_debouncer.triggered.connect(self._redraw_active_tab)
        # Masquer/afficher des canaux : debounce long (montage multi-axes coûteux).
        self._visibility_redraw_debouncer = Debouncer(350, self)
        self._visibility_redraw_debouncer.triggered.connect(
            lambda: self._schedule_redraw(force=True)
        )

        self.recordings_panel.entriesChanged.connect(self._on_entries_changed)
        self.recordings_panel.styleChanged.connect(lambda: self._schedule_redraw(force=True))
        self.channel_panel.channelChanged.connect(self._on_channel_changed)
        self.channel_panel.visibilityChanged.connect(self._on_channel_visibility_changed)
        self.channel_panel.inspectChannelRequested.connect(self.open_channel_focus)
        self.params_panel.viewChanged.connect(self._on_params_view_changed)
        self.params_panel.configChanged.connect(self._on_params_config_changed)
        self.session_panel.probePathChanged.connect(self._on_session_probe_path)
        self.control_panel.filterChanged.connect(self._on_analysis_changed)

        # Sync display settings from the params dock.
        self._settings = self.params_panel.viewer_settings()

        self._status_progress = QProgressBar()
        self._status_progress.setRange(0, 1000)
        self._status_progress.setValue(0)
        self._status_progress.setMaximumWidth(160)
        self._status_progress.setMaximumHeight(14)
        self._status_progress.setTextVisible(False)
        self._status_progress.hide()
        self._status_message = QLabel(
            "Ajoutez un .rhs, Traiter (F5), puis double-clic un canal pour inspecter."
        )
        self.statusBar().addWidget(self._status_progress, 0)
        self.statusBar().addWidget(self._status_message, 1)

        self._build_actions()
        self._rebuild_tabs()
        self.tabs.tabBar().hide()
        self._load_probe(probe_path_from_defaults(self._defaults), sync_ui=True)
        self._restore_state()
        self.status_panel.set_idle("Prêt.")

    # ------------------------------------------------------------------- setup

    def _add_dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower().replace(' ', '_').replace('&', '')}")
        dock.setWidget(widget)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(area, dock)
        return dock

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&Fichier")
        act_add = QAction("Ajouter Intan .rhs…", self)
        act_add.setShortcut(QKeySequence.StandardKey.Open)
        act_add.triggered.connect(self.recordings_panel.browse_rhs)
        act_open_processed = QAction("Ouvrir traité…", self)
        act_open_processed.setToolTip(
            "Ouvrir un dataset déjà exporté (.ergdataset / .zip) pour le comparer."
        )
        act_open_processed.triggered.connect(self.recordings_panel.browse_processed)
        act_export_dataset = QAction("Exporter un dataset traité…", self)
        act_export_dataset.setToolTip(
            "Écrire un dataset réutilisable qui se rouvre instantanément plus tard."
        )
        act_export_dataset.triggered.connect(self.export_processed_dataset)
        act_export_pdf = QAction("Exporter le rapport PDF…", self)
        act_export_pdf.triggered.connect(self.export_pdf_report)
        act_export_figure = QAction("Enregistrer la vue actuelle en image…", self)
        act_export_figure.triggered.connect(self.save_view_images)
        act_quit = QAction("Quitter", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        for action in (act_add, act_open_processed):
            file_menu.addAction(action)
        file_menu.addSeparator()
        for action in (act_export_dataset, act_export_pdf, act_export_figure):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        run_menu = self.menuBar().addMenu("&Traitement")
        self._act_process = QAction("Traiter", self)
        self._act_process.setShortcut(QKeySequence("F5"))
        self._act_process.triggered.connect(lambda: self.start_processing())
        self._act_cancel = QAction("Annuler", self)
        self._act_cancel.setShortcut(QKeySequence("Esc"))
        self._act_cancel.setEnabled(False)
        self._act_cancel.triggered.connect(self.cancel_processing)
        act_reprocess_all = QAction("Retraiter tous les enregistrements", self)
        act_reprocess_all.triggered.connect(lambda: self.start_processing(all_rows=True))
        self._act_ensure_channel = QAction("Calculer le canal sélectionné", self)
        self._act_ensure_channel.setShortcut(QKeySequence("F6"))
        self._act_ensure_channel.setToolTip(
            "Filtrer et analyser uniquement le canal actuellement sélectionné."
        )
        self._act_ensure_channel.triggered.connect(self.ensure_selected_channels)
        self._act_ensure_all = QAction("Calculer tous les canaux", self)
        self._act_ensure_all.setShortcut(QKeySequence("F7"))
        self._act_ensure_all.setToolTip(
            "Calculer tous les canaux (résumés, revue montage)."
        )
        self._act_ensure_all.triggered.connect(self.ensure_all_channels)
        for action in (
            self._act_process,
            self._act_cancel,
            act_reprocess_all,
            self._act_ensure_channel,
            self._act_ensure_all,
        ):
            run_menu.addAction(action)
        run_menu.addSeparator()
        act_cache = QAction("Gestionnaire de cache…", self)
        act_cache.triggered.connect(self.open_cache_manager)
        run_menu.addAction(act_cache)

        view_menu = self.menuBar().addMenu("&Vue")
        act_configure = QAction("Configurer les panneaux…", self)
        act_configure.setShortcut(QKeySequence("Ctrl+P"))
        act_configure.triggered.connect(self.configure_current_view)
        act_redraw = QAction("Redessiner", self)
        act_redraw.setShortcut(QKeySequence("Ctrl+R"))
        act_redraw.triggered.connect(lambda: self._schedule_redraw(force=True))
        act_montage = QAction("Revue montage", self)
        act_montage.setShortcut(QKeySequence("Ctrl+M"))
        act_montage.setToolTip(
            "Montage continu de tous les canaux visibles (scrollable). "
            "Les barres de plage restent dans Inspecter."
        )
        act_montage.triggered.connect(self.open_montage_review)
        act_preview = QAction("Retour à l’aperçu canal", self)
        act_preview.setToolTip("Vue centrale légère : canal sélectionné uniquement.")
        act_preview.triggered.connect(self.return_to_channel_preview)
        act_inspect = QAction("Inspecter", self)
        act_inspect.setShortcuts(
            [QKeySequence("Ctrl+I"), QKeySequence("Ctrl+Return")]
        )
        act_inspect.setToolTip(
            "Canal sélectionné : traces, barres de plage, graphs. "
            "Mode moyenne / stimulation dans la fenêtre. "
            "Aussi : double-clic MEA / liste."
        )
        act_inspect.triggered.connect(lambda: self.open_channel_focus())
        view_menu.addAction(act_configure)
        view_menu.addAction(act_redraw)
        view_menu.addSeparator()
        view_menu.addAction(act_inspect)
        view_menu.addAction(act_montage)
        view_menu.addAction(act_preview)
        view_menu.addSeparator()
        for dock in (self._dock_session, self._dock_params, self._dock_status):
            view_menu.addAction(dock.toggleViewAction())

        channel_menu = self.menuBar().addMenu("&Canal")
        act_prev = QAction("Canal précédent", self)
        act_prev.setShortcut(QKeySequence("Ctrl+Left"))
        act_prev.triggered.connect(lambda: self.channel_panel.step(-1))
        act_next = QAction("Canal suivant", self)
        act_next.setShortcut(QKeySequence("Ctrl+Right"))
        act_next.triggered.connect(lambda: self.channel_panel.step(1))
        channel_menu.addAction(act_prev)
        channel_menu.addAction(act_next)
        channel_menu.addSeparator()
        channel_menu.addAction(act_inspect)
        channel_menu.addAction(self._act_ensure_channel)
        channel_menu.addAction(self._act_ensure_all)

        help_menu = self.menuBar().addMenu("&Aide")
        act_about = QAction("À propos", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

        # Barre : flux principal unique (pas de doublon dans le panneau Recordings).
        toolbar = QToolBar("Principal", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for action in (
            act_add,
            act_open_processed,
            self._act_process,
            self._act_cancel,
        ):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(act_inspect)
        toolbar.addSeparator()
        toolbar.addAction(act_montage)

    # -------------------------------------------------------------- view tabs

    def _rebuild_tabs(self) -> None:
        current = self.tabs.currentIndex()
        self.tabs.blockSignals(True)
        while self.tabs.count():
            page = self.tabs.widget(0)
            self.tabs.removeTab(0)
            page.deleteLater()
        for tab in self._workspace.tabs:
            page = ViewTabPage(self)
            page.grid.removeRequested.connect(self._on_panel_removed)
            page.grid.detachRequested.connect(self._on_panel_detached)
            page.grid.renderFinished.connect(self._on_render_finished)
            page.grid.configure(
                tab.panels, columns=tab.columns, panel_height=tab.panel_height_px
            )
            self.tabs.addTab(page, tab.name)
        index = min(max(0, current if current >= 0 else self._workspace.active_index),
                    max(0, self.tabs.count() - 1))
        self.tabs.setCurrentIndex(index)
        self.tabs.blockSignals(False)
        self.tabs.tabBar().hide()
        self._schedule_redraw(force=True)

    def _current_page(self) -> ViewTabPage | None:
        page = self.tabs.currentWidget()
        return page if isinstance(page, ViewTabPage) else None

    def _current_tab(self) -> ViewTab | None:
        index = self.tabs.currentIndex()
        if 0 <= index < len(self._workspace.tabs):
            return self._workspace.tabs[index]
        return None

    def _replace_tab(self, index: int, tab: ViewTab) -> None:
        tabs = list(self._workspace.tabs)
        if not (0 <= index < len(tabs)):
            return
        tabs[index] = tab
        self._workspace = self._workspace.with_tabs(tabs)
        self.tabs.setTabText(index, tab.name)
        page = self.tabs.widget(index)
        if isinstance(page, ViewTabPage):
            page.grid.configure(tab.panels, columns=tab.columns, panel_height=tab.panel_height_px)
        self._schedule_redraw(force=True)

    def configure_current_view(self) -> None:
        tab = self._current_tab()
        if tab is None:
            return
        updated = pick_panels(tab, self)
        if updated is not None:
            self._replace_tab(self.tabs.currentIndex(), updated)

    def reset_views(self) -> None:
        self._workspace = WorkspaceLayout()
        self._rebuild_tabs()

    def _on_panel_removed(self, placement: PanelPlacement) -> None:
        tab = self._current_tab()
        if tab is None:
            return
        panels = [p for p in tab.panels if p.key != placement.key]
        self._replace_tab(self.tabs.currentIndex(), tab.with_panels(panels))

    def _on_panel_detached(self, placement: PanelPlacement) -> None:
        existing = self._detached.get(placement.key)
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return
        window = DetachedPanelWindow(
            placement, self._default_viewer_settings(), self
        )
        window.closed.connect(lambda key=placement.key: self._detached.pop(key, None))
        window.refreshRequested.connect(self._on_detached_refresh)
        self._detached[placement.key] = window
        window.show()
        self._render_detached(window)

    def _on_detached_refresh(self, window: DetachedPanelWindow) -> None:
        self._render_detached(window)

    def _render_detached(self, window: DetachedPanelWindow) -> None:
        from dataclasses import replace as dc_replace

        request = self._make_request(window.placement)
        if request is None:
            return
        # Affichage local ; données / flux restent ceux de la vue centrale.
        local = window.local_settings()
        settings = dc_replace(
            request.settings,
            legend=local.legend,
            style=local.style,
            psth_bin_window_s=local.psth_bin_window_s,
            sampling_percent=local.sampling_percent,
        )
        window.render(dc_replace(request, settings=settings))

    # --------------------------------------------------------------- rendering

    def _schedule_redraw(self, *, force: bool = False) -> None:
        self._force_redraw = self._force_redraw or force
        self._redraw_debouncer.request()

    def _redraw_active_tab(self) -> None:
        force = self._force_redraw
        self._force_redraw = False
        self._settings = self._default_viewer_settings()
        self._adapt_montage_height()
        page = self._current_page()
        if page is None:
            return
        if force:
            for index in range(self.tabs.count()):
                other = self.tabs.widget(index)
                if isinstance(other, ViewTabPage) and other is not page:
                    other.grid.invalidate_all()
        page.grid.schedule_render(self._make_request_or_blank, force=force)
        for window in list(self._detached.values()):
            self._render_detached(window)

    def _on_params_view_changed(self) -> None:
        self._settings = self.params_panel.viewer_settings()
        self._schedule_redraw(force=True)

    def _on_params_config_changed(self) -> None:
        self._config_dirty = True
        self.params_panel.set_dirty_message(
            "Paramètres de traitement modifiés — appuyez sur Traiter (F5) pour recalculer."
        )
        self._set_status("Paramètres modifiés — F5 pour retraiter.")
        # Mapping MEA : appliquer tout de suite si le chemin a changé (pas besoin de F5).
        probe = self.params_panel.probe_layout_path()
        if probe != self._probe_path:
            self._load_probe(probe, sync_ui=True)

    def _on_session_probe_path(self, path: object) -> None:
        self._load_probe(Path(path) if path else None, sync_ui=True)

    def _on_channel_changed(self, channel: str) -> None:
        self.control_panel.set_channel(channel)
        tab = self._current_tab()
        # En revue montage, un clic sur une case change souvent la sélection :
        # coalescer avec le debounce de visibilité pour éviter un freeze par clic.
        if tab is not None and any(
            p.panel == "montage_continuous_raw" or str(p.panel).startswith("montage_")
            for p in tab.panels
        ):
            self._visibility_redraw_debouncer.request()
            return
        # N’invalider que l’onglet actif — les autres se rafraîchiront à l’activation.
        page = self._current_page()
        if isinstance(page, ViewTabPage):
            page.grid.invalidate_all()
        self._schedule_redraw(force=True)

    def _on_channel_visibility_changed(self) -> None:
        """Masquer / afficher des canaux → rafraîchir seulement si un montage est affiché."""
        tab = self._current_tab()
        if tab is None:
            return
        if not any(
            p.panel == "montage_continuous_raw" or str(p.panel).startswith("montage_")
            for p in tab.panels
        ):
            return
        self._visibility_redraw_debouncer.request()

    def _adapt_montage_height(self) -> None:
        """Hauteur du panneau montage = canaux visibles × flux × hauteur min (scrollable)."""
        tab = self._current_tab()
        page = self._current_page()
        if tab is None or page is None:
            return
        if not any(p.panel == "montage_continuous_raw" for p in tab.panels):
            return
        entries = self._plotted()
        if not entries:
            return
        recording = entries[0].recording
        n_channels = int(getattr(recording, "n_channels", 0) or 0)
        if n_channels <= 0:
            return
        hidden = set(self.channel_panel.hidden_channels)
        names = list(getattr(recording, "channel_names", ()) or ())
        n_visible = sum(
            1
            for index in range(n_channels)
            if (names[index] if index < len(names) else f"CH{index}") not in hidden
        )
        n_visible = max(1, n_visible)
        streams = self._settings.resolved_continuous_streams()
        rows = n_visible * max(1, len(streams))
        min_row = max(48, int(self._settings.montage_row_min_height_px))
        height = max(tab.panel_height_px, rows * min_row + 80)
        # Ajuster la hauteur sans relayout complet (évite un freeze à chaque coche).
        setter = getattr(page.grid, "set_panel_height", None)
        if callable(setter):
            setter(height, panels=("montage_continuous_raw",))
        elif getattr(page.grid, "_panel_height", None) != height:
            page.grid.configure(tab.panels, columns=tab.columns, panel_height=height)

    def _default_viewer_settings(self) -> ViewerSettings:
        """Réglages d’affichage de la vue centrale (ParamsPanel + ControlPanel)."""
        settings = self.params_panel.viewer_settings()
        streams = self.control_panel.continuous_streams()
        return replace(
            settings,
            analysis=AnalysisSettings(),
            continuous_stream=streams[0] if streams else "raw",  # type: ignore[arg-type]
            continuous_streams=streams,
            continuous_mark_stims=self.control_panel.mark_stimulations(),
            hidden_channels=self.channel_panel.hidden_channels,
            # Pas de barres sur la vue centrale.
            range_bars=(),
            active_range_index=0,
        )

    def _seed_settings_for_window(self) -> ViewerSettings:
        """Valeurs par défaut injectées à l’ouverture d’une fenêtre de vue."""
        return self._default_viewer_settings()

    def _build_config(self, rhs_file: Path) -> AnalysisConfig:
        return self.params_panel.build_config(rhs_file)

    def _ensure_channels_for_indices(
        self,
        channels: list[int],
        *,
        then_redraw_window: str | None = None,
        need_means: bool = True,
        need_rms: bool = True,
        need_spikes: bool = True,
        need_overlay: bool = True,
    ) -> None:
        """Lancer le calcul des canaux manquants (vue active ou fenêtre détachée)."""
        self._pending_window_id = then_redraw_window
        self._start_channel_ensure(
            channels,
            allow_empty_redraw=then_redraw_window is None,
            need_means=need_means,
            need_rms=need_rms,
            need_spikes=need_spikes,
            need_overlay=need_overlay,
        )

    def _on_analysis_changed(self) -> None:
        self._schedule_redraw(force=True)

    def _make_session_request(
        self, window: ViewSessionWindow, placement: PanelPlacement
    ) -> RenderRequest | None:
        entries = self._plotted() or self.recordings_panel.ready_entries()
        if not entries:
            return None
        return self._build_render_request(
            placement,
            entries,
            channel_index=int(window.channel_index),
            channel_name=window.channel_name,
            settings=window.local_settings(),
            highlight_zooms=highlight_zooms_from_placements(window.placements),
        )

    def _on_view_session_closed(self, window_id: str) -> None:
        self._view_sessions.pop(str(window_id), None)

    def _on_view_session_refresh(self, window: ViewSessionWindow) -> None:
        if window.needs_channel_compute():
            self._ensure_channels_for_indices(
                [int(window.channel_index)], then_redraw_window=window.window_id
            )
        else:
            window.redraw()

    @staticmethod
    def _split_key(key: str) -> tuple[str, str]:
        if "@" in key:
            panel, section = key.split("@", 1)
            return panel, section
        return key, "full"

    def _on_tab_changed(self, index: int) -> None:
        self._workspace = replace(self._workspace, active_index=max(0, index))
        self._schedule_redraw()

    def _pipeline_busy(self) -> bool:
        return (
            (self._build_worker is not None and self._build_worker.isRunning())
            or (self._ensure_worker is not None and self._ensure_worker.isRunning())
        )

    def _mark_display_loaded(self) -> None:
        """100 % quand les données affichées sont prêtes."""
        self.status_panel.set_progress_complete()
        self._status_progress.setValue(1000)

    def _on_render_finished(self, panels: int, seconds: float) -> None:
        if panels <= 0:
            return
        tab = self._current_tab()
        self.status_panel.set_render_summary(panels, seconds, tab=tab.name if tab else "")
        self._set_status(f"{panels} panneau(x) redessiné(s) en {seconds * 1000:.0f} ms")
        if not self._pipeline_busy():
            self._mark_display_loaded()

    def open_montage_review(self) -> None:
        """Ouvrir le montage multi-canaux (tous les canaux, scrollable)."""
        ready = self.recordings_panel.ready_entries()
        if not ready:
            QMessageBox.information(
                self,
                "Revue montage",
                "Traitez un enregistrement (F5) avant d’ouvrir le montage.",
            )
            return
        tab = self._current_tab()
        if tab is not None and any(p.panel == "montage_continuous_raw" for p in tab.panels):
            self._set_status("Revue montage déjà affichée.")
            return
        n_channels = int(getattr(ready[0].recording, "n_channels", 0) or 0)
        montage_tab = ViewTab(
            name="Revue montage",
            columns=1,
            panel_height_px=900,
            panels=(PanelPlacement("montage_continuous_raw"),),
        )
        self._workspace = WorkspaceLayout(tabs=(montage_tab,), active_index=0)
        self._rebuild_tabs()
        n_hidden = len(self.channel_panel.hidden_channels)
        visible = max(0, n_channels - n_hidden)
        self._set_status(
            f"Revue montage — {visible}/{n_channels} canaux visibles "
            "(décocher / Masquer dans Session → Channels). "
            "Pour les barres de plage : double-clic un canal (Inspecter)."
        )

    def return_to_channel_preview(self) -> None:
        """Revenir à l’aperçu léger du canal sélectionné."""
        self._workspace = WorkspaceLayout()
        self._rebuild_tabs()
        channel = self.channel_panel.current_channel or ""
        self._set_status(
            f"Aperçu canal{f' — {channel}' if channel else ''}. "
            "Double-clic pour inspecter avec barres."
        )

    def open_channel_focus(self, channel: str | None = None) -> None:
        """Fenêtre d’inspection : traces + barres de plage + graphs d’analyse.

        Entrées : toolbar / menu / Ctrl+I / Ctrl+Entrée / double-clic MEA.
        Le mode moyenne / stimulation se règle dans la fenêtre ouverte.
        """
        channel = str(channel or self.channel_panel.current_channel or "").strip()
        if not channel:
            QMessageBox.information(self, "Inspecter", "Sélectionnez d’abord un canal.")
            return
        ready = self.recordings_panel.ready_entries()
        if not ready:
            QMessageBox.information(
                self,
                "Inspecter",
                "Traitez un enregistrement (F5) avant d’inspecter un canal.",
            )
            return
        recording = ready[0].recording
        resolved = recording.channel_index(channel)
        if resolved is None:
            QMessageBox.warning(self, "Inspecter", f"Canal introuvable : {channel}")
            return

        # Réutiliser une fenêtre déjà ouverte pour ce canal.
        for existing in self._channel_windows.values():
            if existing.channel_name == channel:
                existing.raise_()
                existing.activateWindow()
                self._set_status(f"Inspection déjà ouverte — {channel}")
                return

        mode_label = "Moyenne"
        # Un graph d’analyse par défaut ; l’utilisateur coche le reste.
        analysis = AnalysisSettings(
            mode="average",
            stim_index=0,
            show_raw=True,
            show_hp=False,
            show_lp=False,
            show_rms=False,
        )
        self.channel_panel.select(channel, emit=False)
        self.control_panel.set_channel(channel)
        window = ChannelAnalysisWindow(
            channel_name=channel,
            channel_index=int(resolved),
            analysis=analysis,
            base_settings=self._seed_settings_for_window(),
            mode_label=mode_label,
            parent=self,
        )
        window.set_trial_count(int(getattr(recording, "n_trials", 0) or 0))
        try:
            t, _ = recording.continuous_trace("raw", int(resolved))
            if t.size:
                window.set_time_span(float(t[0]), float(t[-1]))
        except Exception:
            window.set_time_span(0.0, 1.0)
        window.set_request_factory(self._make_channel_analysis_request)
        window.closed.connect(self._on_channel_window_closed)
        window.refreshRequested.connect(self._on_channel_window_refresh)
        self._channel_windows[window.window_id] = window
        window.show()
        if window.needs_channel_compute():
            self._ensure_channels_for_indices(
                [int(resolved)], then_redraw_window=window.window_id
            )
        else:
            window.redraw()
        self._set_status(f"Inspection {channel} — {mode_label}")

    def _make_channel_analysis_request(
        self, window: ChannelAnalysisWindow, placement: PanelPlacement
    ) -> RenderRequest | None:
        entries = self._plotted() or self.recordings_panel.ready_entries()
        if not entries:
            return None
        return self._build_render_request(
            placement,
            entries,
            channel_index=int(window.channel_index),
            channel_name=window.channel_name,
            settings=window.local_settings(),
            highlight_zooms=highlight_zooms_from_placements(window.placements),
        )

    def _on_channel_window_closed(self, window_id: str) -> None:
        self._channel_windows.pop(str(window_id), None)

    def _on_channel_window_refresh(self, window: ChannelAnalysisWindow) -> None:
        if window.needs_channel_compute():
            self._ensure_channels_for_indices(
                [int(window.channel_index)], then_redraw_window=window.window_id
            )
        else:
            window.redraw()

    def _plotted(self) -> list[RecordingEntry]:
        return self.recordings_panel.plotted_entries()

    def _build_render_request(
        self,
        placement: PanelPlacement,
        entries: list[RecordingEntry],
        *,
        channel_index: int,
        channel_name: str,
        settings: ViewerSettings,
        highlight_zooms: tuple[tuple[float, float, str], ...] = (),
    ) -> RenderRequest:
        """Construire une requête de rendu à partir des enregistrements prêts."""
        return RenderRequest(
            placement=placement,
            recordings=[entry.recording for entry in entries],
            labels=[entry.display_label for entry in entries],
            colors=resolve_recording_plot_colors(
                [entry.style for entry in entries], list(range(len(entries)))
            ),
            legend_flags=[entry.style.legend_visible for entry in entries],
            channel_index=channel_index,
            channel_name=channel_name,
            settings=settings,
            probe_layout=self._probe_layout,
            impedance_sessions=_impedance_sessions(self.recordings_panel.ready_entries()),
            highlight_zooms=highlight_zooms,
        )

    def _make_request_or_blank(self, placement: PanelPlacement) -> RenderRequest:
        request = self._make_request(placement)
        if request is not None:
            return request
        return RenderRequest(
            placement=placement,
            recordings=[],
            labels=[],
            colors=[],
            legend_flags=[],
            channel_index=0,
            channel_name="",
            settings=self._settings,
            probe_layout=self._probe_layout,
            impedance_sessions=[],
        )

    def _make_request(self, placement: PanelPlacement) -> RenderRequest | None:
        entries = self._plotted()
        channel = self.channel_panel.current_channel or ""
        channel_index = 0
        if entries:
            recordings = [entry.recording for entry in entries]
            resolved = recordings[0].channel_index(channel)
            channel_index = int(resolved) if resolved is not None else 0
            self._check_channel_alignment(recordings, channel_index, channel)
        return self._build_render_request(
            placement,
            entries,
            channel_index=channel_index,
            channel_name=channel,
            settings=self._settings,
        )

    def _check_channel_alignment(
        self, recordings: list[Any], channel_index: int, channel: str
    ) -> None:
        if self._warned_channel_mismatch or len(recordings) < 2:
            return
        for recording in recordings[1:]:
            names = recording.channel_names
            if not (0 <= channel_index < len(names)) or names[channel_index] != channel:
                self._warned_channel_mismatch = True
                self.status_panel.append_log(
                    "Attention : les enregistrements comparés n’ont pas le même ordre de canaux ; "
                    "les panneaux utilisent la position du premier enregistrement."
                )
                return

    # -------------------------------------------------------------- processing

    def _on_entries_changed(self) -> None:
        self._refresh_channels()
        self._schedule_redraw(force=True)

    def _apply_probe_from_defaults(self) -> None:
        self._load_probe(probe_path_from_defaults(self._defaults), sync_ui=True)

    def _load_probe(self, path: Path | str | None, *, sync_ui: bool = True) -> None:
        """Charge le JSON de géométrie MEA pour la carte (et l’inset PDF)."""
        resolved: Path | None
        if path is None or not str(path).strip():
            resolved = None
        else:
            resolved = Path(str(path))

        same_path = resolved == self._probe_path
        if same_path and (resolved is None or self._probe_layout is not None):
            if sync_ui:
                self.session_panel.set_probe_path(resolved)
                self.params_panel.set_probe_path(resolved)
            return

        self._probe_path = resolved
        self._probe_layout = None
        self._defaults["default_probe_layout_json"] = str(resolved) if resolved else ""

        if resolved is not None:
            if not resolved.exists():
                self.status_panel.append_log(f"Mapping introuvable : {resolved}")
                self._set_status(f"Mapping introuvable : {resolved.name}")
            else:
                try:
                    from probe_layout import load_probe_layout_json

                    self._probe_layout = load_probe_layout_json(resolved)
                    self.status_panel.append_log(f"Mapping chargé : {resolved}")
                    self._set_status(f"Mapping MEA : {resolved.name}")
                except Exception as exc:
                    self.status_panel.append_log(f"Impossible de lire le mapping : {exc}")
                    self._set_status(f"Mapping invalide : {exc}")

        if sync_ui:
            self.session_panel.set_probe_path(resolved)
            self.params_panel.set_probe_path(resolved)
        self.channel_panel.set_probe(self._probe_layout)
        self._schedule_redraw(force=True)

    def start_processing(self, *, all_rows: bool = False) -> None:
        if self._build_worker is not None and self._build_worker.isRunning():
            return
        entries = self.recordings_panel.entries
        if not entries:
            QMessageBox.information(
                self, "Rien à traiter", "Ajoutez au moins un fichier .rhs ou un dataset traité."
            )
            return
        targets = (
            entries
            if (all_rows or self._config_dirty)
            else [e for e in entries if not e.is_ready]
        )
        if not targets:
            self._set_status("Tout est déjà prêt — Inspecter un canal, ou F6 pour recalculer.")
            return

        requests: list[BuildRequest] = []
        processed_entries: list[RecordingEntry] = []
        for entry in targets:
            if entry.is_processed:
                processed_entries.append(entry)
                continue
            config = self._build_config(entry.path)
            config = replace(config, recording_label=entry.label or None, recording_style=entry.style)
            requests.append(
                BuildRequest(
                    config=config,
                    label=entry.display_label,
                    style=entry.style,
                    row_id=entry.row_id,
                )
            )
            self.recordings_panel.set_status(entry.row_id, "queued", "waiting")
        if requests:
            self._cache_root = self._resolve_cache_root(requests[0].config)

        for entry in processed_entries:
            self._open_processed_entry(entry)

        if not requests:
            self._config_dirty = False
            self._schedule_redraw(force=True)
            return

        worker = BuildWorker(requests, self._cache_root, self)
        worker.progressed.connect(self._on_pipeline_progress)
        worker.logged.connect(self.status_panel.append_log)
        worker.recording_ready.connect(self._on_recording_ready)
        worker.recording_failed.connect(self._on_recording_failed)
        worker.finished_all.connect(self._on_build_finished)
        self._build_worker = worker
        self._set_busy(True)
        self.status_panel.set_headline(f"Traitement de {len(requests)} enregistrement(s)…")
        worker.start()

    def _resolve_cache_root(self, config: AnalysisConfig) -> Path:
        from erg_cache import default_cache_root

        return default_cache_root(config)

    def _open_processed_entry(self, entry: RecordingEntry) -> None:
        from dataset_builder import open_dataset

        self.recordings_panel.set_status(entry.row_id, "building", "ouverture du dataset")
        started = time.perf_counter()
        path = entry.path
        label = entry.label or None
        style = entry.style
        row_id = entry.row_id
        display = entry.display_label

        def task() -> Any:
            return open_dataset(path, label=label, style=style)

        def done(recording: Any) -> None:
            elapsed = time.perf_counter() - started
            report = SimpleNamespace(
                recording=display,
                timings=(),
                total_s=elapsed,
                reused_bundle=True,
            )
            self.recordings_panel.set_result(row_id, recording, report)
            self.status_panel.append_log(f"Dataset traité ouvert : {path}")
            self.status_panel.add_timing(display, "Ouvert depuis un dataset traité", elapsed)
            self._refresh_channels()
            self._schedule_redraw(force=True)

        def failed(message: str) -> None:
            self.recordings_panel.set_status(row_id, "failed", message)
            self.status_panel.append_log(f"Impossible d’ouvrir {path.name} : {message}")

        # Réutilise TaskWorker pour ne pas bloquer l’UI.
        if self._task_worker is not None and self._task_worker.isRunning():
            try:
                recording = open_dataset(path, label=label, style=style)
                done(recording)
            except Exception as exc:
                failed(str(exc))
            return
        worker = TaskWorker(task, self)
        worker.succeeded.connect(done)
        worker.failed.connect(failed)
        worker.logged.connect(self.status_panel.append_log)
        self._task_worker = worker
        self._set_busy(True)
        worker.finished.connect(lambda: self._set_busy(False))
        worker.start()

    def cancel_processing(self) -> None:
        if self._build_worker is not None and self._build_worker.isRunning():
            self._build_worker.request_stop()
            self.status_panel.set_headline("Annulation…")
        if self._ensure_worker is not None and self._ensure_worker.isRunning():
            self._ensure_worker.request_stop()
            self.status_panel.set_headline("Annulation du calcul de canal…")
        if self._task_worker is not None and self._task_worker.isRunning():
            self._task_worker.request_stop()

    def _on_recording_ready(self, row_id: int, recording: Any, report: Any) -> None:
        self.recordings_panel.set_result(row_id, recording, report)
        self.status_panel.add_report(report)
        self._refresh_channels()

    def _on_recording_failed(self, row_id: int, message: str) -> None:
        self.recordings_panel.set_status(row_id, "failed", message)
        self._set_status(f"Échec du traitement : {message}")

    def _on_build_finished(self, ok: bool) -> None:
        self._set_busy(False)
        self._build_worker = None
        self._config_dirty = False
        self.params_panel.set_dirty_message("")
        if ok:
            self.status_panel.set_headline("Enregistrement prêt.")
            self._mark_display_loaded()
            self._set_status("Prêt — choisissez un canal, puis Inspecter (ou double-clic).")
            self.session_panel.show_channels()
        else:
            self.status_panel.set_headline("Traitement terminé avec des erreurs ou annulé.")
        self._refresh_channels()
        self._schedule_redraw(force=True)

    def _set_busy(self, busy: bool) -> None:
        self.recordings_panel.set_busy(busy)
        self.control_panel.set_busy(busy)
        self._act_process.setEnabled(not busy)
        self._act_cancel.setEnabled(busy)
        self._act_ensure_channel.setEnabled(not busy)
        self._act_ensure_all.setEnabled(not busy)
        self._status_progress.setVisible(busy)
        if busy:
            self._dock_status.show()
            self._dock_status.raise_()

    def _on_pipeline_progress(self, event: Any) -> None:
        self.status_panel.on_progress(event)
        overall = float(getattr(event, "overall_fraction", 0.0) or 0.0)
        self._status_progress.setValue(int(max(0.0, min(1.0, overall)) * 1000))
        stage = str(getattr(event, "stage_label", "") or "")
        recording = str(getattr(event, "recording", "") or "")
        if recording and stage:
            self._status_message.setText(f"{recording} — {stage}")
        elif stage:
            self._status_message.setText(stage)

    def _channels_for_current_view(self, *, all_channels: bool = False) -> list[int]:
        """Channel indices the GUI needs right now."""
        plotted = self._plotted()
        if not plotted:
            return []
        recording = plotted[0].recording
        n_channels = int(getattr(recording, "n_channels", 0))
        if n_channels <= 0:
            return []
        if all_channels or self._pending_ensure_all:
            return list(range(n_channels))

        tab = self._current_tab()
        wants_global = False
        if tab is not None:
            wants_global = any(
                placement.panel in GLOBAL_PANEL_FIELD_NAMES for placement in tab.panels
            )
        if wants_global and recording.ready_channels and len(recording.ready_channels) >= n_channels:
            return list(range(n_channels))

        channel = self.channel_panel.current_channel or ""
        resolved = recording.channel_index(channel)
        if resolved is None:
            return [0] if n_channels else []
        return [int(resolved)]

    def ensure_selected_channels(self) -> None:
        self._pending_ensure_all = False
        self._start_channel_ensure(self._channels_for_current_view(all_channels=False))

    def ensure_all_channels(self) -> None:
        self._pending_ensure_all = True
        self._start_channel_ensure(self._channels_for_current_view(all_channels=True))

    def _start_channel_ensure(
        self,
        channels: list[int],
        *,
        allow_empty_redraw: bool = True,
        need_means: bool = True,
        need_rms: bool = True,
        need_spikes: bool = True,
        need_overlay: bool = True,
    ) -> None:
        products = {
            "need_means": need_means,
            "need_rms": need_rms,
            "need_spikes": need_spikes,
            "need_overlay": need_overlay,
        }
        if self._build_worker is not None and self._build_worker.isRunning():
            return
        if self._ensure_worker is not None and self._ensure_worker.isRunning():
            self._ensure_queue.append((list(channels), self._pending_window_id, products))
            self.status_panel.append_log(
                f"Calcul en file d’attente : {len(channels)} canal(aux)."
            )
            return
        plotted = self._plotted() or self.recordings_panel.ready_entries()
        window_id = self._pending_window_id
        if not plotted or not channels:
            if window_id:
                self._redraw_pending_window(window_id)
            elif allow_empty_redraw:
                self._schedule_redraw(force=True)
            return

        def _needs_work(recording: Any, ch: int) -> bool:
            if not recording.is_channel_ready(ch):
                return True
            data = getattr(recording, "_channel_data", {}).get(ch)
            if data is None:
                return True
            if need_means and not getattr(data, "means_ready", True):
                return True
            if need_rms and not getattr(data, "rms_ready", True):
                return True
            if need_spikes and not getattr(data, "spikes_ready", True):
                return True
            if need_overlay and not getattr(data, "overlay_ready", True):
                return True
            return False

        requests: list[ChannelEnsureRequest] = []
        for entry in plotted:
            recording = entry.recording
            if recording is None or getattr(recording, "source", None) is None:
                continue
            missing = [
                ch
                for ch in channels
                if 0 <= ch < recording.n_channels and _needs_work(recording, ch)
            ]
            if not missing:
                continue
            config = self._build_config(entry.path)
            config = replace(config, recording_label=entry.label or None, recording_style=entry.style)
            requests.append(
                ChannelEnsureRequest(
                    recording=recording,
                    config=config,
                    channels=missing,
                    label=entry.display_label,
                    need_means=need_means,
                    need_rms=need_rms,
                    need_spikes=need_spikes,
                    need_overlay=need_overlay,
                )
            )

        if not requests:
            if window_id:
                self._redraw_pending_window(window_id)
            else:
                self._schedule_redraw(force=True)
            plotted_now = self._plotted()
            self.channel_panel.set_reference_recording(
                plotted_now[0].recording if plotted_now else None
            )
            return

        worker = ChannelEnsureWorker(requests, self)
        worker.progressed.connect(self._on_pipeline_progress)
        worker.logged.connect(self.status_panel.append_log)
        worker.failed.connect(lambda message: self._set_status(f"Échec du calcul de canal : {message}"))
        worker.finished_all.connect(self._on_ensure_finished)
        self._ensure_worker = worker
        self._set_busy(True)
        n = sum(len(request.channels) for request in requests)
        self.status_panel.set_headline(f"Calcul de {n} canal(aux)…")
        worker.start()

    def _on_ensure_finished(self, ok: bool) -> None:
        self._set_busy(False)
        self._ensure_worker = None
        self._pending_ensure_all = False
        window_id = self._pending_window_id
        self._pending_window_id = None
        if ok:
            self.status_panel.set_headline("Canaux prêts.")
            self._set_status("Canaux prêts.")
        else:
            self.status_panel.set_headline("Calcul des canaux terminé avec des erreurs ou annulé.")
        plotted = self._plotted()
        self.channel_panel.set_reference_recording(plotted[0].recording if plotted else None)
        self._redraw_pending_window(window_id)
        self._schedule_redraw(force=True)
        if self._ensure_queue:
            channels, pending_id, products = self._ensure_queue.pop(0)
            self._pending_window_id = pending_id
            self._start_channel_ensure(
                channels, allow_empty_redraw=pending_id is None, **products
            )
        elif ok:
            self._mark_display_loaded()

    def _redraw_pending_window(self, window_id: str | None) -> None:
        if not window_id:
            return
        if window_id in self._view_sessions:
            self._view_sessions[window_id].redraw()
            return
        if window_id in self._channel_windows:
            self._channel_windows[window_id].redraw()
            return

    def _refresh_channels(self) -> None:
        names = self.recordings_panel.common_channel_names()
        if not names:
            ready = self.recordings_panel.first_ready()
            names = list(ready.recording.channel_names) if ready is not None else []
        self.channel_panel.set_channels(names)
        self.channel_panel.set_probe(self._probe_layout)
        plotted = self._plotted()
        recording = plotted[0].recording if plotted else None
        if recording is None:
            ready = self.recordings_panel.first_ready()
            recording = ready.recording if ready is not None else None
        self.channel_panel.set_reference_recording(recording)
        n_trials = int(getattr(recording, "n_trials", 0) or 0) if recording is not None else 0
        self.control_panel.set_trial_count(n_trials)
        self.control_panel.set_channel(self.channel_panel.current_channel)

    # ------------------------------------------------------------------ export

    def export_processed_dataset(self) -> None:
        ready = self.recordings_panel.ready_entries()
        if not ready:
            QMessageBox.information(
                self, "Export", "Traitez au moins un enregistrement avant d’exporter."
            )
            return
        default_dir = ready[0].path.parent
        dialog = ExportDatasetDialog(default_dir, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        options = dialog.options()
        options.directory.mkdir(parents=True, exist_ok=True)

        skipped_windows = False
        if options.include_trigger_windows:
            for entry in ready:
                recording = entry.recording
                source = getattr(recording, "source", None)
                derived = getattr(recording, "derived", None)
                windows = getattr(derived, "trigger_windows", None) if derived is not None else None
                if source is not None and not windows:
                    skipped_windows = True
                    break
        if skipped_windows:
            reply = QMessageBox.question(
                self,
                "Export partiel",
                "Les fenêtres de stimulation individuelles ne sont pas encore "
                "matérielles (canaux à la demande). Elles seront omises de l’export.\n\n"
                "Continuer quand même ?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        def task() -> list[str]:
            from processed_dataset import archive_bundle, dataset_target_path
            from dataset_builder import export_dataset

            written: list[str] = []
            for entry in ready:
                target = dataset_target_path(options.directory, entry.display_label)
                bundle = export_dataset(
                    entry.recording,
                    target,
                    include_streams=options.include_streams,
                    include_trigger_windows=options.include_trigger_windows,
                    include_overlay=options.include_overlay,
                    progress=print,
                )
                written.append(str(bundle))
                if options.make_archive:
                    archive = archive_bundle(bundle, bundle.with_suffix(".zip"))
                    written.append(str(archive))
            return written

        def done(result: Any) -> None:
            paths = result or []
            self.status_panel.append_log("Exporté :\n" + "\n".join(str(p) for p in paths))
            QMessageBox.information(
                self,
                "Export terminé",
                "Dataset(s) traité(s) écrit(s) :\n\n" + "\n".join(str(p) for p in paths),
            )

        self._run_task(task, done, "Export du dataset traité…")

    def export_pdf_report(self) -> None:
        if self._pdf_callback is None:
            QMessageBox.information(
                self, "Rapport PDF", "Le pipeline PDF est indisponible dans cette session."
            )
            return
        entries = [e for e in self.recordings_panel.entries if not e.is_processed]
        if not entries:
            QMessageBox.information(
                self,
                "Rapport PDF",
                "Le rapport PDF est généré à partir d’enregistrements .rhs. Ajoutez-en au moins un.",
            )
            return
        default_name = f"{entries[0].display_label}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Enregistrer le rapport PDF",
            str(entries[0].path.parent / default_name),
            "PDF (*.pdf)",
        )
        if not path:
            return
        target = Path(path)
        display = self._workspace.to_plot_display()
        configs: list[AnalysisConfig] = []
        for entry in entries:
            configs.append(
                replace(
                    self._build_config(entry.path),
                    save_dir=target.parent,
                    pdf_title=target.stem,
                    recording_label=entry.label or None,
                    recording_style=entry.style,
                    plot_display=display,
                    zoom_mode=self._workspace.zoom_mode(),
                )
            )
        callback = self._pdf_callback

        def task() -> str:
            callback(configs)
            return str(target)

        def done(result: Any) -> None:
            self.status_panel.append_log(f"Rapport PDF écrit près de {result}")
            QMessageBox.information(
                self, "Rapport PDF", f"Rapport généré dans :\n{target.parent}"
            )

        self._run_task(task, done, "Génération du rapport PDF…")

    def save_view_images(self) -> None:
        page = self._current_page()
        tab = self._current_tab()
        if page is None or tab is None or page.grid.panel_count == 0:
            QMessageBox.information(self, "Enregistrer des images", "Cette vue n’a aucun panneau à enregistrer.")
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Choisir un dossier pour les images des panneaux", str(Path.home())
        )
        if not directory:
            return
        root = Path(directory)
        page.grid.render_dirty_now(self._make_request_or_blank)
        written = 0
        for placement in page.grid.placements:
            canvas = page.grid.panel_widget(placement)
            if canvas is None:
                continue
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in placement.key)
            out = root / f"{tab.name}_{safe}.png"
            canvas.figure.savefig(out, dpi=200, bbox_inches="tight")
            written += 1
        self._set_status(f"{written} image(s) écrite(s) dans {root}")
        self.status_panel.append_log(f"{written} image(s) de panneau écrite(s) dans {root}")

    def _run_task(
        self, task: Callable[[], Any], on_success: Callable[[Any], None], headline: str
    ) -> None:
        if self._task_worker is not None and self._task_worker.isRunning():
            QMessageBox.information(self, "Occupé", "Un autre export est déjà en cours.")
            return
        worker = TaskWorker(task, self)
        worker.logged.connect(self.status_panel.append_log)

        def _success(result: Any) -> None:
            self._task_worker = None
            self._set_busy(False)
            self.status_panel.set_headline("Terminé.")
            on_success(result)

        def _failed(message: str) -> None:
            self._task_worker = None
            self._set_busy(False)
            self.status_panel.set_headline("Échec.")
            QMessageBox.warning(self, "Tâche échouée", message)

        worker.succeeded.connect(_success)
        worker.failed.connect(_failed)
        self._task_worker = worker
        self._set_busy(True)
        self.status_panel.set_headline(headline)
        worker.start()

    # ------------------------------------------------------------------- cache

    def open_cache_manager(self) -> None:
        root = self._cache_root
        if root is None:
            entries = self.recordings_panel.entries
            if entries:
                root = self._resolve_cache_root(self._build_config(entries[0].path))
        if root is None:
            QMessageBox.information(
                self, "Cache", "Ajoutez d’abord un enregistrement pour résoudre le dossier de cache."
            )
            return
        protected = {
            Path(getattr(entry.recording, "bundle_root", "") or "")
            for entry in self.recordings_panel.ready_entries()
            if getattr(entry.recording, "bundle_root", None)
        }
        CacheDialog(root, protected, self).exec()

    # ------------------------------------------------------------------- misc

    def _set_status(self, message: str) -> None:
        self._status_message.setText(message)

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "À propos de plot_erg",
            "Visionneuse ERG — paradigme canal d’abord.\n\n"
            "1. Session → Ajouter un .rhs, puis Traiter (F5).\n"
            "2. Clic sur la MEA : aperçu central du canal.\n"
            "3. Inspecter (ou double-clic) : traces, barres, graphs.\n"
            "4. Cocher les graphs ; « Traiter la plage » ajoute les zooms.\n"
            "5. Revue montage : vue globale de tous les canaux (optionnel).",
        )

    def _restore_state(self) -> None:
        settings = QSettings(_ORG, _APP)
        geometry = settings.value("geometry")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        state = settings.value("windowState")
        if isinstance(state, QByteArray):
            self.restoreState(state)
        # Journal optionnel : ne pas le rouvrir automatiquement.
        self._dock_status.hide()

    def closeEvent(self, event: Any) -> None:  # noqa: D102
        settings = QSettings(_ORG, _APP)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        if self._build_worker is not None and self._build_worker.isRunning():
            self._build_worker.request_stop()
            self._build_worker.wait(4000)
        if self._ensure_worker is not None and self._ensure_worker.isRunning():
            self._ensure_worker.request_stop()
            self._ensure_worker.wait(4000)
        if self._task_worker is not None and self._task_worker.isRunning():
            self._task_worker.request_stop()
            self._task_worker.wait(4000)
        for window in list(self._detached.values()):
            try:
                window.close()
            except RuntimeError:
                pass
        self._detached.clear()
        for window in list(self._view_sessions.values()):
            try:
                window.close()
            except RuntimeError:
                pass
        self._view_sessions.clear()
        for window in list(self._channel_windows.values()):
            try:
                window.close()
            except RuntimeError:
                pass
        self._channel_windows.clear()
        self.recordings_panel.clear()
        super().closeEvent(event)


def launch_qt_gui(
    run_callback: Callable[[AnalysisConfig], None] | None = None,
    run_comparison_callback: Callable[[AnalysisConfig, AnalysisConfig], None] | None = None,
    run_multi_comparison_callback: Callable[[list[AnalysisConfig]], None] | None = None,
    **defaults: Any,
) -> int:
    """Ouvrir la visionneuse interactive. Les callbacks pilotent l’export PDF classique."""
    del run_callback, run_comparison_callback  # the viewer always uses the multi path
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    font = QFont("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(APP_STYLESHEET)
    window = ViewerWindow(defaults, pdf_callback=run_multi_comparison_callback)
    window.show()
    initial = defaults.get("initial_rhs_files") or []
    if initial:
        window.recordings_panel.add_paths(Path(p) for p in initial)
    return app.exec()
