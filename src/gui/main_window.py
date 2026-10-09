"""Visionneuse interactive pour enregistrements Intan RHS / ERG.

Paradigme canal d’abord :
- gauche  : Session (zone Mapping + enregistrements / liste)
- centre  : aperçu canal (continuous / moyenne / stimulation)
- Paramètres → Canal : mode / courbes / spikes + Pipeline (F5)
- montage multi-canaux : optionnel (Revue montage)
- bas     : Control Panel (canal · Analyse = bascule moyenne)
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
    QStackedWidget,
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
from gui.widgets.channel_analysis_window import ChannelAnalysisWindow
from panel_registry import RenderRequest, highlight_zooms_from_placements
from view_config import (
    GLOBAL_PANEL_FIELD_NAMES,
    PanelPlacement,
    ViewTab,
    ViewerSettings,
    WorkspaceLayout,
    apply_local_display_settings,
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
        self._workspace = WorkspaceLayout(tabs=(), active_index=0)
        self._settings = viewer_settings_from_defaults(self._defaults)
        self._probe_path: Path | None = None
        self._probe_layout: Any | None = None
        self._build_worker: BuildWorker | None = None
        self._ensure_worker: ChannelEnsureWorker | None = None
        self._task_worker: TaskWorker | None = None
        self._detached: dict[str, DetachedPanelWindow] = {}
        self._channel_inspect: ChannelAnalysisWindow | None = None
        self._config_dirty = False
        self._force_redraw = False
        self._redraw_inspectors = False
        self._preserve_view = False
        self._cache_root: Path | None = None
        self._warned_channel_mismatch = False
        self._pending_ensure_all = False
        self._pending_window_id: str | None = None
        self._ensure_background = False
        self._ensure_accepting = False
        self._channel_compute_active = False
        self._in_montage = False
        self._pending_progress: Any | None = None
        self._pending_open_entries: list[RecordingEntry] = []

        self.setWindowTitle("plot_erg — Visionneuse Intan / ERG")
        self.resize(1600, 960)

        # Centre = aperçu canal (inspect embarqué) | revue montage + control panel.
        self._view_stack = QStackedWidget(self)
        self._preview_page = QWidget(self)
        self._preview_layout = QVBoxLayout(self._preview_page)
        self._preview_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_layout.setSpacing(0)
        self._preview_placeholder = QLabel(
            "Aperçu du canal sélectionné.\n\n"
            "1. Session → Ajouter un .rhs\n"
            "2. Traiter (F5)\n"
            "3. Choisir un canal\n\n"
            "Mode aperçu : Paramètres → Canal · Revue montage (Ctrl+M)",
            self._preview_page,
        )
        self._preview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_placeholder.setObjectName("workflowHint")
        self._preview_placeholder.setWordWrap(True)
        self._preview_layout.addWidget(self._preview_placeholder)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("centralViews")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._view_stack.addWidget(self._preview_page)  # 0 = aperçu canal
        self._view_stack.addWidget(self.tabs)  # 1 = revue montage

        self.control_panel = ControlPanel(self)
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self._view_stack, 1)
        central_layout.addWidget(self.control_panel, 0)
        self.setCentralWidget(central)

        self.recordings_panel = RecordingsPanel(self)
        self.channel_panel = ChannelPanel(self)
        self.session_panel = SessionPanel(self.recordings_panel, self.channel_panel, self)
        self.params_panel = ParamsPanel(self._defaults, self)
        self.status_panel = StatusPanel(self)

        self._dock_session = self._add_dock(
            "Session",
            self.session_panel,
            Qt.DockWidgetArea.LeftDockWidgetArea,
            object_name="dock_session",
        )
        self._dock_params = self._add_dock(
            "Paramètres",
            self.params_panel,
            Qt.DockWidgetArea.RightDockWidgetArea,
            object_name="dock_params",
        )
        self._dock_status = self._add_dock(
            "Journal",
            self.status_panel,
            Qt.DockWidgetArea.BottomDockWidgetArea,
            object_name="dock_status",
        )
        self._dock_status.hide()
        # Mins bas : le contenu utilise SizePolicy.Ignored + sizeHint courant
        # pour ne plus « aspirer » la largeur au drag du séparateur.
        self._dock_session.setMinimumWidth(200)
        self._dock_params.setMinimumWidth(200)
        self.resizeDocks([self._dock_session], [360], Qt.Orientation.Horizontal)
        self.resizeDocks([self._dock_params], [380], Qt.Orientation.Horizontal)

        self._redraw_debouncer = Debouncer(110, self)
        self._redraw_debouncer.triggered.connect(self._redraw_active_tab)
        # Masquer/afficher des canaux : debounce long (montage multi-axes coûteux).
        self._visibility_redraw_debouncer = Debouncer(350, self)
        self._visibility_redraw_debouncer.triggered.connect(
            lambda: self._schedule_redraw(force=True)
        )
        # Progression pipeline : coalescer pour ne pas saturer la boucle Qt.
        self._progress_debouncer = Debouncer(80, self)
        self._progress_debouncer.triggered.connect(self._flush_pipeline_progress)
        # Badges ✓/○ : ne pas rescanner toute la liste à chaque canal prefetch.
        self._channel_badge_debouncer = Debouncer(200, self)
        self._channel_badge_debouncer.triggered.connect(self._flush_channel_badges)

        self.recordings_panel.entriesChanged.connect(self._on_entries_changed)
        self.recordings_panel.styleChanged.connect(
            lambda: self._schedule_redraw(force=True, inspectors=True)
        )
        self.recordings_panel.processRequested.connect(lambda: self.start_processing())
        self.channel_panel.channelChanged.connect(self._on_channel_changed)
        self.channel_panel.visibilityChanged.connect(self._on_channel_visibility_changed)
        self.channel_panel.inspectChannelRequested.connect(self.open_channel_analysis)
        self.params_panel.viewChanged.connect(self._on_params_view_changed)
        self.params_panel.configChanged.connect(self._on_params_config_changed)
        self.params_panel.processRequested.connect(lambda: self.start_processing())
        self.session_panel.probePathChanged.connect(self._on_session_probe_path)
        self.control_panel.analysisRequested.connect(self.open_channel_analysis)
        self.params_panel.streamsChanged.connect(self._on_streams_changed)
        # Pour détecter les flux nouvellement cochés → déplier Pipeline (Canal).
        self._prev_control_streams: set[str] = set(
            self.params_panel.continuous_streams()
        )

        # Sync display settings from the params dock.
        self._settings = self.params_panel.viewer_settings()

        self._status_progress = QProgressBar()
        self._status_progress.setRange(0, 1000)
        self._status_progress.setValue(0)
        self._status_progress.setMaximumWidth(180)
        self._status_progress.setMaximumHeight(16)
        self._status_progress.setTextVisible(True)
        self._status_progress.setFormat("%p%")
        self._status_progress.hide()
        self._status_message = QLabel(
            "Ajoutez un .rhs, Traiter (F5), puis sélectionnez un canal (Analyse : Ctrl+I)."
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

    def _add_dock(
        self,
        title: str,
        widget: QWidget,
        area: Qt.DockWidgetArea,
        *,
        object_name: str | None = None,
    ) -> QDockWidget:
        dock = QDockWidget(title, self)
        # ASCII objectName (évite accents dans les sélecteurs / outils).
        if object_name:
            dock.setObjectName(object_name)
        else:
            ascii_title = (
                title.lower()
                .replace(" ", "_")
                .replace("&", "")
                .replace("è", "e")
                .replace("é", "e")
                .replace("à", "a")
            )
            dock.setObjectName(f"dock_{ascii_title}")
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
        self._act_add = QAction("Ajouter Intan .rhs…", self)
        self._act_add.setShortcut(QKeySequence.StandardKey.Open)
        self._act_add.setToolTip("Ajouter un ou plusieurs fichiers .rhs (Ctrl+O)")
        self._act_add.triggered.connect(self._add_rhs_and_focus)
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
        for action in (self._act_add, act_open_processed):
            file_menu.addAction(action)
        file_menu.addSeparator()
        for action in (act_export_dataset, act_export_pdf, act_export_figure):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        self._act_process = QAction("Traiter", self)
        self._act_process.setShortcut(QKeySequence("F5"))
        self._act_process.setToolTip("Traiter les enregistrements en attente (F5)")
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
            "Prioriser le canal affiché (le prefetch en arrière-plan continue ensuite)."
        )
        self._act_ensure_channel.triggered.connect(self.ensure_selected_channels)
        self._act_ensure_all = QAction("Calculer tous les canaux", self)
        self._act_ensure_all.setShortcut(QKeySequence("F7"))
        self._act_ensure_all.setToolTip(
            "Lancer / relancer le calcul de tous les canaux en arrière-plan "
            "(le canal affiché reste prioritaire)."
        )
        self._act_ensure_all.triggered.connect(self.ensure_all_channels)
        act_cache = QAction("Gestionnaire de cache…", self)
        act_cache.triggered.connect(self.open_cache_manager)

        act_configure = QAction("Configurer les panneaux…", self)
        act_configure.setShortcut(QKeySequence("Ctrl+P"))
        act_configure.triggered.connect(self.configure_current_view)
        act_redraw = QAction("Redessiner", self)
        act_redraw.setShortcut(QKeySequence("Ctrl+R"))
        act_redraw.triggered.connect(lambda: self._schedule_redraw(force=True))
        self._act_montage = QAction("Revue montage", self)
        self._act_montage.setShortcut(QKeySequence("Ctrl+M"))
        self._act_montage.setToolTip(
            "Montage continu de tous les canaux visibles (scrollable). "
            "Les barres de plage restent dans l’aperçu canal."
        )
        self._act_montage.triggered.connect(self.open_montage_review)
        self._act_preview = QAction("Retour à l’aperçu canal", self)
        self._act_preview.setShortcut(QKeySequence("Ctrl+Shift+M"))
        self._act_preview.setToolTip(
            "Quitter la revue montage et revenir à l’aperçu du canal sélectionné (Ctrl+Shift+M)."
        )
        self._act_preview.setEnabled(False)
        self._act_preview.triggered.connect(self.return_to_channel_preview)
        self._act_analyse = QAction("Analyse", self)
        self._act_analyse.setShortcuts(
            [QKeySequence("Ctrl+I"), QKeySequence("Ctrl+Return")]
        )
        self._act_analyse.setToolTip(
            "Afficher la moyenne d’essais dans l’aperçu "
            "(Paramètres → Canal : continuous / moyenne / stimulation). "
            "Ctrl+I · aussi double-clic MEA / liste."
        )
        self._act_analyse.triggered.connect(lambda: self.open_channel_analysis())
        act_prev = QAction("Canal précédent", self)
        act_prev.setShortcut(QKeySequence("Ctrl+Left"))
        act_prev.triggered.connect(lambda: self.channel_panel.step(-1))
        act_next = QAction("Canal suivant", self)
        act_next.setShortcut(QKeySequence("Ctrl+Right"))
        act_next.triggered.connect(lambda: self.channel_panel.step(1))
        act_about = QAction("À propos", self)
        act_about.triggered.connect(self._show_about)

        # Raccourcis sans menus Traitement / Vue / Canal / Aide.
        for action in (
            self._act_process,
            self._act_cancel,
            act_reprocess_all,
            self._act_ensure_channel,
            self._act_ensure_all,
            act_cache,
            act_configure,
            act_redraw,
            self._act_analyse,
            self._act_montage,
            self._act_preview,
            act_prev,
            act_next,
            act_about,
        ):
            self.addAction(action)
        for dock in (self._dock_session, self._dock_params, self._dock_status):
            action = dock.toggleViewAction()
            self.addAction(action)
            if dock is self._dock_params:
                action.setShortcut(QKeySequence("Ctrl+P"))
                action.setToolTip("Afficher / masquer le panneau Paramètres (Ctrl+P)")
            elif dock is self._dock_status:
                action.setShortcut(QKeySequence("Ctrl+J"))
                action.setToolTip("Afficher / masquer le journal (Ctrl+J)")

        toolbar = QToolBar("Principal", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)
        # Ajouter / Traiter : uniquement dans le dock Session (onglet Enregistrements).
        toolbar.addAction(self._act_cancel)
        toolbar.addSeparator()
        toolbar.addAction(self._act_montage)
        toolbar.addAction(self._act_preview)

    def _add_rhs_and_focus(self) -> None:
        before = len(self.recordings_panel.entries)
        self.recordings_panel.browse_rhs()
        if len(self.recordings_panel.entries) > before:
            self.session_panel.show_recordings()
            self._set_status("Fichier(s) ajouté(s) — cliquez Traiter (F5) pour calculer.")

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
        # Revue montage = tous les panneaux sont du montage continu.
        tab = self._current_tab()
        is_montage = bool(
            tab is not None
            and bool(tab.panels)
            and all(p.panel == "montage_continuous_raw" for p in tab.panels)
        )
        self._set_view_mode(montage=is_montage)
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
        self._workspace = WorkspaceLayout(tabs=(), active_index=0)
        self._rebuild_tabs()
        self._set_view_mode(montage=False)
        self._sync_channel_inspect(redraw=True)

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
        # Affichage = panneau de *cette* fenêtre ; données / flux = vue centrale.
        settings = apply_local_display_settings(
            request.settings, window.local_settings()
        )
        window.render(dc_replace(request, settings=settings))

    # --------------------------------------------------------------- rendering

    def _schedule_redraw(
        self,
        *,
        force: bool = False,
        inspectors: bool = False,
        preserve_view: bool = False,
    ) -> None:
        self._force_redraw = self._force_redraw or force
        self._redraw_inspectors = self._redraw_inspectors or inspectors
        if preserve_view:
            self._preserve_view = True
        self._redraw_debouncer.request()

    def _redraw_detached_windows(self) -> None:
        """Rafraîchir les panneaux détachés (chaque fenêtre garde son affichage local)."""
        for window in list(self._detached.values()):
            self._render_detached(window)

    def _redraw_inspector_windows(self) -> None:
        """Rafraîchir l’aperçu canal embarqué sans reset zoom / pan."""
        if self._channel_inspect is None:
            return
        try:
            self._channel_inspect.redraw(preserve_view=True)
        except Exception:
            pass

    def _redraw_active_tab(self) -> None:
        force = self._force_redraw
        redraw_inspectors = self._redraw_inspectors
        preserve_view = self._preserve_view
        self._force_redraw = False
        self._redraw_inspectors = False
        # Garder preserve_view pendant toute la file de rendus async.
        self._settings = self._default_viewer_settings()
        if not preserve_view:
            self._adapt_montage_height()

        if not self._showing_channel_preview():
            page = self._current_page()
            if page is not None:
                if force and not preserve_view:
                    for index in range(self.tabs.count()):
                        other = self.tabs.widget(index)
                        if isinstance(other, ViewTabPage) and other is not page:
                            other.grid.invalidate_all()
                page.grid.schedule_render(self._make_request_or_blank, force=force)
            self._redraw_detached_windows()
            if redraw_inspectors:
                self._redraw_inspector_windows()
            if page is None or not getattr(page.grid, "_pending", None):
                self._preserve_view = False
            return

        # Mode aperçu : vue canal embarquée (barres + continuous).
        self._sync_channel_inspect(redraw=False)
        if self._channel_inspect is not None:
            self._channel_inspect.apply_viewer_settings(self._settings)
            self._channel_inspect.redraw(preserve_view=preserve_view)
        elif redraw_inspectors:
            self._redraw_inspector_windows()
        self._redraw_detached_windows()
        self._preserve_view = False

    def _on_params_view_changed(self) -> None:
        self._settings = self.params_panel.viewer_settings()
        if self._channel_inspect is not None and self._showing_channel_preview():
            self._channel_inspect.apply_viewer_settings(self._settings)
            # Affichage (échelles, sync…) : garder zoom/pan.
            self._channel_inspect.redraw(preserve_view=True)
            self._redraw_detached_windows()
            return
        self._schedule_redraw(force=True)

    def _on_params_config_changed(self) -> None:
        self._config_dirty = True
        self.params_panel.set_dirty_message(
            "Paramètres de traitement modifiés — Traiter (F5) pour recalculer."
        )
        # Ne pas forcer l’onglet Canal : l’utilisateur peut être sur Affichage.
        self._set_status("Paramètres modifiés — F5 pour retraiter.")
        # Mapping MEA : appliquer tout de suite si le chemin a changé (pas besoin de F5).
        probe = self.params_panel.probe_layout_path()
        if probe != self._probe_path:
            self._load_probe(probe, sync_ui=True)

    def _on_session_probe_path(self, path: object) -> None:
        self._load_probe(Path(path) if path else None, sync_ui=True)

    def _on_channel_changed(self, channel: str) -> None:
        self.control_panel.set_channel(channel)
        self._preserve_view = False
        # Prioriser le canal regardé, même pendant un prefetch en arrière-plan.
        self._prioritize_viewed_channels()
        self._prefetch_neighbor_channels()
        if self._in_montage:
            # En revue montage, un clic sur une case change souvent la sélection :
            # coalescer avec le debounce de visibilité pour éviter un freeze par clic.
            self._visibility_redraw_debouncer.request()
            return
        self._schedule_redraw(force=True)

    def _neighbor_channel_indices(self, center: int, n_channels: int) -> list[int]:
        """Canaux proches (±2 en index, plus voisins MEA si la sonde est chargée)."""
        neighbors: set[int] = set()
        for delta in (-2, -1, 1, 2):
            ch = int(center) + delta
            if 0 <= ch < n_channels:
                neighbors.add(ch)
        probe = self._probe_layout
        if probe is not None:
            try:
                from probe_layout import neighboring_channel_indices

                for ch in neighboring_channel_indices(probe, int(center), max_count=6):
                    if 0 <= int(ch) < n_channels and int(ch) != int(center):
                        neighbors.add(int(ch))
            except Exception:
                pass
        return sorted(neighbors)

    def _prefetch_neighbor_channels(self) -> None:
        """Chauffer le cache filtre / moyennes des canaux voisins (basse priorité)."""
        plotted = self._plotted()
        if not plotted:
            return
        recording = plotted[0].recording
        n_channels = int(getattr(recording, "n_channels", 0) or 0)
        if n_channels <= 0:
            return
        channel = self.channel_panel.current_channel or ""
        resolved = recording.channel_index(channel)
        if resolved is None:
            return
        neighbors = self._neighbor_channel_indices(int(resolved), n_channels)
        if not neighbors:
            return
        # Prefetch filter rows immediately (disk/RAM), then soft ensure means.
        source = getattr(recording, "source", None)
        if source is not None:
            for bank_name in ("highpass", "lowpass"):
                bank = getattr(source, bank_name, None)
                prefetch = getattr(bank, "prefetch", None)
                if callable(prefetch):
                    try:
                        prefetch(neighbors)
                    except Exception:
                        pass
        self._ensure_channels_for_indices(
            neighbors,
            need_means=True,
            need_rms=False,
            need_spikes=False,
            need_overlay=False,
            priority=False,
            background=True,
        )

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

    def _montage_visible_indices(self, recording: Any) -> list[int]:
        """Indices des canaux non masqués (ordre d’origine)."""
        n_channels = int(getattr(recording, "n_channels", 0) or 0)
        if n_channels <= 0:
            return []
        hidden = set(self.channel_panel.hidden_channels)
        names = list(getattr(recording, "channel_names", ()) or ())
        return [
            index
            for index in range(n_channels)
            if (names[index] if index < len(names) else f"CH{index}") not in hidden
        ]

    def _montage_visible_count(self, recording: Any) -> int:
        """Nombre de canaux non masqués pour la revue montage."""
        return len(self._montage_visible_indices(recording))

    def _montage_review_placements(self) -> tuple[PanelPlacement, ...]:
        """Un seul panneau : toutes les traces empilées sans séparateur."""
        return (PanelPlacement("montage_continuous_raw"),)

    def _montage_total_height_px(self, n_visible: int) -> int:
        """Hauteur du montage (canaux×flux × ligne + chrome panneau)."""
        streams = self._settings.resolved_continuous_streams()
        rows = max(1, int(n_visible)) * max(1, len(streams))
        min_row = max(36, int(self._settings.montage_row_min_height_px))
        # Chrome : en-tête + toolbar matplotlib + curseur + marges.
        chrome = 96
        return max(220, rows * min_row + chrome)

    def _adapt_montage_height(self) -> None:
        """Resynchroniser la hauteur du montage (visibles / flux)."""
        tab = self._current_tab()
        page = self._current_page()
        if tab is None or page is None:
            return
        if not any(p.panel == "montage_continuous_raw" for p in tab.panels):
            return
        entries = self._plotted()
        n_visible = 1
        if entries:
            n_visible = max(1, self._montage_visible_count(entries[0].recording))
        height = self._montage_total_height_px(n_visible)
        if int(getattr(tab, "panel_height_px", 0) or 0) != height:
            new_tab = replace(tab, panel_height_px=height)
            tabs = list(self._workspace.tabs)
            active = max(0, min(len(tabs) - 1, int(self._workspace.active_index)))
            tabs[active] = new_tab
            self._workspace = replace(self._workspace, tabs=tuple(tabs))
            tab = new_tab
        setter = getattr(page.grid, "set_panel_height", None)
        if callable(setter):
            setter(height, panels=("montage_continuous_raw",))
        elif getattr(page.grid, "_panel_height", None) != height:
            page.grid.configure(tab.panels, columns=tab.columns, panel_height=height)

    def _default_viewer_settings(self) -> ViewerSettings:
        """Réglages d’affichage de la vue centrale (ParamsPanel)."""
        settings = self.params_panel.viewer_settings()
        streams = settings.resolved_continuous_streams()
        # Conserver le mode / courbes choisis dans l’aperçu (onglet Canal).
        analysis = (
            self._channel_inspect.local_settings().analysis
            if self._channel_inspect is not None
            else settings.analysis
        )
        return replace(
            settings,
            analysis=analysis,
            continuous_stream=streams[0] if streams else "raw",  # type: ignore[arg-type]
            continuous_streams=streams,
            hidden_channels=self.channel_panel.hidden_channels,
            range_bars=(),
            active_range_index=0,
        )

    def _seed_settings_for_window(self) -> ViewerSettings:
        """Valeurs par défaut injectées à l’ouverture d’une fenêtre de vue."""
        return self._default_viewer_settings()

    def _build_config(self, rhs_file: Path) -> AnalysisConfig:
        from core import gui_channel_workers

        config = self.params_panel.build_config(rhs_file)
        # Laisser au moins un cœur libre pour que l’UI reste réactive.
        return replace(config, channel_workers=gui_channel_workers(config.channel_workers))

    def _ensure_channels_for_indices(
        self,
        channels: list[int],
        *,
        then_redraw_window: str | None = None,
        need_means: bool = True,
        need_rms: bool = True,
        need_spikes: bool = True,
        need_overlay: bool = True,
        priority: bool = True,
        background: bool = False,
    ) -> None:
        """Lancer le calcul des canaux manquants (vue active ou fenêtre détachée)."""
        if then_redraw_window is not None:
            self._pending_window_id = then_redraw_window
        self._start_channel_ensure(
            channels,
            allow_empty_redraw=then_redraw_window is None,
            need_means=need_means,
            need_rms=need_rms,
            need_spikes=need_spikes,
            need_overlay=need_overlay,
            priority=priority,
            background=background,
        )

    def _on_streams_changed(self, streams: object) -> None:
        """Coches WIDE/HIGH/LOW dans Paramètres → Affichage → aperçu continuous."""
        resolved = tuple(streams) if isinstance(streams, (list, tuple)) else ()
        current = set(resolved) or set(self.params_panel.continuous_streams())
        added = current - self._prev_control_streams
        self._prev_control_streams = current
        if added:
            # Déplie le Pipeline Canal correspondant (sans voler le focus Affichage).
            self.params_panel.reveal_curve_params(*sorted(added), focus_tab=False)
        self._settings = self._default_viewer_settings()
        self._preserve_view = False
        if self._channel_inspect is not None and self._showing_channel_preview():
            self._channel_inspect.apply_control_streams(
                self.params_panel.continuous_streams(),
                mark_stims=self.params_panel.mark_stimulations(),
                redraw=False,
            )
            # Nouveau flux → redessin sans conserver les axes (subplots ajoutés).
            self._channel_inspect.redraw(preserve_view=False)
            return
        self._schedule_redraw(force=True)

    @staticmethod
    def _split_key(key: str) -> tuple[str, str]:
        if "@" in key:
            panel, section = key.split("@", 1)
            return panel, section
        return key, "full"

    def _on_tab_changed(self, index: int) -> None:
        self._workspace = replace(self._workspace, active_index=max(0, index))
        self._prioritize_viewed_channels()
        self._schedule_redraw()

    def _pipeline_busy(self) -> bool:
        return (
            (self._build_worker is not None and self._build_worker.isRunning())
            or (
                self._ensure_worker is not None
                and self._ensure_worker.isRunning()
                and not self._ensure_background
            )
        )

    def _pipeline_ui_locked(self) -> bool:
        """True tant qu’un build ou un calcul de canaux possède la barre de statut."""
        if self._build_worker is not None and self._build_worker.isRunning():
            return True
        return bool(self._channel_compute_active)

    def _mark_display_loaded(self) -> None:
        """100 % quand les données affichées sont prêtes (pas pendant un calcul en cours)."""
        if self._pipeline_ui_locked():
            return
        self.status_panel.set_progress_complete()
        self._status_progress.setValue(1000)

    def _on_render_finished(self, panels: int, seconds: float) -> None:
        if panels <= 0:
            return
        self._preserve_view = False
        # Ne pas écraser headline / barre pendant un calcul — sinon ça clignote.
        if self._pipeline_ui_locked():
            return
        tab = self._current_tab()
        self.status_panel.set_render_summary(panels, seconds, tab=tab.name if tab else "")
        self._set_status(f"{panels} panneau(x) redessiné(s) en {seconds * 1000:.0f} ms")
        self._mark_display_loaded()

    def open_montage_review(self) -> None:
        """Ouvrir le montage multi-canaux (traces empilées, un seul graphe)."""
        ready = self.recordings_panel.ready_entries()
        if not ready:
            QMessageBox.information(
                self,
                "Revue montage",
                "Traitez un enregistrement (F5) avant d’ouvrir le montage.",
            )
            return
        plotted = self._plotted()
        if not plotted:
            QMessageBox.information(
                self,
                "Revue montage",
                "Cochez au moins un enregistrement dans Session pour le tracer.",
            )
            return
        tab = self._current_tab()
        already = bool(
            self._in_montage
            or (
                tab is not None
                and bool(tab.panels)
                and all(p.panel == "montage_continuous_raw" for p in tab.panels)
            )
        )
        if already:
            self._set_status("Revue montage déjà affichée.")
            self._set_view_mode(montage=True)
            self._adapt_montage_height()
            self._schedule_redraw(force=True)
            return
        # Aligner sur ParamsPanel avant le 1er redraw.
        self._settings = self._default_viewer_settings()
        recording = plotted[0].recording
        n_channels = int(getattr(recording, "n_channels", 0) or 0)
        n_visible = max(1, self._montage_visible_count(recording))
        placements = self._montage_review_placements()
        height = self._montage_total_height_px(n_visible)
        montage_tab = ViewTab(
            name="Revue montage",
            columns=1,
            panel_height_px=height,
            panels=placements,
        )
        self._workspace = WorkspaceLayout(tabs=(montage_tab,), active_index=0)
        self._rebuild_tabs()
        self._set_view_mode(montage=True)
        self.session_panel.show_channels()
        self._set_status(
            f"Revue montage — {n_visible}/{n_channels} canaux empilés"
            f"{f' · {len(plotted)} fichiers' if len(plotted) > 1 else ''}. "
            "Retour aperçu : Ctrl+Shift+M."
        )

    def return_to_channel_preview(self) -> None:
        """Revenir à l’aperçu canal (continuous / moyenne / stimulation)."""
        self._workspace = WorkspaceLayout(tabs=(), active_index=0)
        self._rebuild_tabs()
        self._set_view_mode(montage=False)
        self._sync_channel_inspect(redraw=True)
        channel = self.channel_panel.current_channel or ""
        self._set_status(
            f"Aperçu canal{f' — {channel}' if channel else ''}. "
            "Mode : Paramètres → Canal."
        )

    def _set_view_mode(self, *, montage: bool) -> None:
        self._in_montage = bool(montage)
        mode = "montage" if montage else "preview"
        self.control_panel.set_view_mode(mode)
        self.params_panel.set_view_mode(mode)
        if hasattr(self, "_act_preview"):
            self._act_preview.setEnabled(self._in_montage)
        if hasattr(self, "_act_montage"):
            self._act_montage.setEnabled(not self._in_montage)
        self._update_view_stack()

    def _workspace_has_tab_panels(self) -> bool:
        return any(bool(tab.panels) for tab in self._workspace.tabs)

    def _update_view_stack(self) -> None:
        """Onglets (montage / vue custom) ou aperçu canal embarqué."""
        if not hasattr(self, "_view_stack"):
            return
        if self._in_montage or self._workspace_has_tab_panels():
            self._view_stack.setCurrentWidget(self.tabs)
        else:
            self._view_stack.setCurrentWidget(self._preview_page)

    def _showing_channel_preview(self) -> bool:
        return (
            hasattr(self, "_view_stack")
            and self._view_stack.currentWidget() is self._preview_page
        )

    def _sync_channel_inspect(self, *, redraw: bool = True) -> None:
        """Créer / lier la vue aperçu au canal sélectionné."""
        channel = str(self.channel_panel.current_channel or "").strip()
        ready = self.recordings_panel.ready_entries()
        if not channel or not ready:
            if self._channel_inspect is not None:
                self._channel_inspect.hide()
            self.params_panel.set_channel_side_panel(None)
            self._preview_placeholder.show()
            return

        recording = ready[0].recording
        resolved = recording.channel_index(channel)
        if resolved is None:
            if self._channel_inspect is not None:
                self._channel_inspect.hide()
            self.params_panel.set_channel_side_panel(None)
            self._preview_placeholder.show()
            return

        self._preview_placeholder.hide()
        created = False
        if self._channel_inspect is None:
            seed = self._seed_settings_for_window()
            window = ChannelAnalysisWindow(
                channel_name=channel,
                channel_index=int(resolved),
                analysis=seed.analysis,
                base_settings=seed,
                parent=self._preview_page,
                embedded=True,
            )
            window.set_request_factory(self._make_channel_analysis_request)
            window.refreshRequested.connect(self._on_channel_window_refresh)
            window.previewModeChanged.connect(self._on_preview_mode_changed)
            self._preview_layout.addWidget(window, 1)
            self._channel_inspect = window
            created = True
        else:
            self._channel_inspect.rebind_channel(
                channel_name=channel,
                channel_index=int(resolved),
            )

        window = self._channel_inspect
        # Contrôles aperçu → onglet Canal (focus seulement à la 1re attache).
        self.params_panel.set_channel_side_panel(
            window.side_panel, focus=created
        )
        window.set_trial_count(int(getattr(recording, "n_trials", 0) or 0))
        try:
            stims = recording.stimulation_times_s()
            window.set_stim_times([float(t) for t in stims])
        except Exception:
            window.set_stim_times([])
        try:
            t, _ = recording.continuous_trace("raw", int(resolved))
            if t.size:
                window.set_time_span(float(t[0]), float(t[-1]))
        except Exception:
            window.set_time_span(0.0, 1.0)
        window.apply_viewer_settings(self._default_viewer_settings())
        window.apply_control_streams(
            self.params_panel.continuous_streams(),
            mark_stims=self.params_panel.mark_stimulations(),
            redraw=False,
        )
        channel_ready = bool(
            hasattr(recording, "is_channel_ready")
            and recording.is_channel_ready(int(resolved))
        )
        window.set_channel_ready(channel_ready)
        window.show()
        if window.needs_channel_compute():
            self._ensure_channels_for_indices(
                [int(resolved)], then_redraw_window=window.window_id
            )
        elif redraw:
            window.redraw(preserve_view=False)

    def open_channel_analysis(self, channel: str | None = None) -> None:
        """Basculer l’aperçu en mode moyenne (ou sélectionner le canal d’abord).

        Entrées : bouton Analyse / menu / Ctrl+I / Ctrl+Entrée / double-clic MEA.
        Les graphs s’affichent dans l’aperçu ; choix continuous / moyenne /
        stimulation dans Paramètres → Canal.
        """
        if channel:
            channel = str(channel).strip()
            if channel:
                self.channel_panel.select(channel, emit=True)
        if self._in_montage:
            self.return_to_channel_preview()
        else:
            self._sync_channel_inspect(redraw=False)
        window = self._channel_inspect
        if window is None:
            if not str(self.channel_panel.current_channel or "").strip():
                QMessageBox.information(
                    self, "Analyse", "Sélectionnez d’abord un canal."
                )
            elif not self.recordings_panel.ready_entries():
                QMessageBox.information(
                    self,
                    "Analyse",
                    "Traitez un enregistrement (F5) avant d’ouvrir l’analyse.",
                )
            return
        window.set_preview_mode("average", redraw=True)
        self.params_panel.focus_channel_tab()
        self._set_status(
            f"Aperçu — moyenne d’essais ({window.channel_name}). "
            "Mode : Paramètres → Canal."
        )

    def _make_channel_analysis_request(
        self, window: ChannelAnalysisWindow, placement: PanelPlacement
    ) -> RenderRequest | None:
        # Respecter la case « Afficher » : pas de repli sur tous les fichiers prêts.
        entries = self._plotted()
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

    @staticmethod
    def _reveal_keys_from_analysis(analysis: object) -> tuple[str, ...]:
        """Courbes Pipeline (Canal) à déplier depuis le paramétrage d’analyse."""
        keys: list[str] = []
        if bool(getattr(analysis, "show_raw", False)):
            keys.append("raw")
        if bool(getattr(analysis, "show_hp", False)):
            keys.append("hp")
        if bool(getattr(analysis, "show_lp", False)):
            keys.append("lp")
        if bool(getattr(analysis, "show_rms", False) or getattr(analysis, "show_summary_rms", False)):
            keys.append("rms")
        if any(
            bool(getattr(analysis, attr, False))
            for attr in (
                "show_isi",
                "show_overlay",
                "show_psth",
                "show_trial_rate",
                "show_raster",
            )
        ):
            keys.append("spikes")
        return tuple(keys)

    def _on_preview_mode_changed(self, mode: object) -> None:
        """Mode continuous / moyenne / stimulation dans l’aperçu → précharger les données."""
        window = self._channel_inspect
        if window is None:
            return
        # Continuous = traces brutes seulement (pas de courbe moyennée).
        if str(mode) == "continuous":
            return
        analysis = window.local_settings().analysis
        reveal = self._reveal_keys_from_analysis(analysis)
        if reveal:
            self.params_panel.reveal_curve_params(*reveal, focus_tab=False)
        self._ensure_channels_for_indices(
            [int(window.channel_index)],
            then_redraw_window=window.window_id,
            need_means=True,
            need_rms=bool(analysis.show_rms or analysis.show_summary_rms),
            need_spikes=bool(
                analysis.show_isi
                or analysis.show_overlay
                or analysis.show_psth
                or analysis.show_trial_rate
                or analysis.show_raster
            ),
            need_overlay=bool(analysis.show_overlay),
        )

    def _on_channel_window_refresh(self, window: ChannelAnalysisWindow) -> None:
        analysis = window.local_settings().analysis
        has_analysis_panels = bool(getattr(window, "_analysis_placements", ()) or ())
        need_analysis = bool(window.is_analysis_view()) or has_analysis_panels
        if need_analysis:
            reveal = self._reveal_keys_from_analysis(analysis)
            if reveal:
                self.params_panel.reveal_curve_params(*reveal, focus_tab=False)
        if window.needs_channel_compute() or need_analysis:
            self._ensure_channels_for_indices(
                [int(window.channel_index)],
                then_redraw_window=window.window_id,
                need_means=True,
                need_rms=bool(
                    need_analysis
                    and (analysis.show_rms or analysis.show_summary_rms)
                ),
                need_spikes=bool(
                    need_analysis
                    and (
                        analysis.show_isi
                        or analysis.show_overlay
                        or analysis.show_psth
                        or analysis.show_trial_rate
                        or analysis.show_raster
                    )
                ),
                need_overlay=bool(need_analysis and analysis.show_overlay),
            )
        else:
            # Respecter le flag local (False après changement de flux / courbes).
            window.redraw(preserve_view=bool(getattr(window, "_preserve_view", True)))

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
            impedance_sessions=_impedance_sessions(entries),
            highlight_zooms=highlight_zooms,
            preserve_view=bool(self._preserve_view),
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
            preserve_view=bool(self._preserve_view),
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
        problems = self.params_panel.validate()
        if problems:
            QMessageBox.warning(
                self,
                "Paramètres invalides",
                "Corrigez avant de traiter :\n\n• " + "\n• ".join(problems),
            )
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
            self._set_status(
                "Tout est déjà prêt — Paramètres → Canal pour le mode, ou F6."
            )
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

        # Ne jamais ouvrir sur le thread UI : mettre en file si un export tourne déjà.
        if self._task_worker is not None and self._task_worker.isRunning():
            self._pending_open_entries.append(entry)
            self.recordings_panel.set_status(
                entry.row_id, "queued", "ouverture en attente"
            )
            return

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
            # Datasets exportés peuvent déjà avoir tous les canaux ; sinon prefetch.
            self._start_background_prefetch()
            self._drain_pending_opens()

        def failed(message: str) -> None:
            self.recordings_panel.set_status(row_id, "failed", message)
            self.status_panel.append_log(f"Impossible d’ouvrir {path.name} : {message}")
            self._drain_pending_opens()

        worker = TaskWorker(task, self)
        worker.succeeded.connect(done)
        worker.failed.connect(failed)
        worker.logged.connect(self.status_panel.append_log)
        self._task_worker = worker
        self._set_busy(True)
        worker.finished.connect(lambda: self._set_busy(False))
        worker.start()

    def _drain_pending_opens(self) -> None:
        """Enchaîner les ouvertures de datasets mises en file (toujours hors UI)."""
        if not self._pending_open_entries:
            return
        if self._task_worker is not None and self._task_worker.isRunning():
            return
        next_entry = self._pending_open_entries.pop(0)
        self._open_processed_entry(next_entry)

    def cancel_processing(self) -> None:
        if self._build_worker is not None and self._build_worker.isRunning():
            self._build_worker.request_stop()
            self.status_panel.set_headline("Annulation…")
        if self._ensure_worker is not None and self._ensure_accepting:
            self._ensure_worker.request_stop()
            self._ensure_accepting = False
            self.status_panel.set_headline("Annulation du calcul de canal…")
        if self._task_worker is not None and self._task_worker.isRunning():
            self._task_worker.request_stop()

    def _on_recording_ready(self, row_id: int, recording: Any, report: Any) -> None:
        self.recordings_panel.set_result(row_id, recording, report)
        self.status_panel.add_report(report)
        self._refresh_channels()

    def _on_recording_failed(self, row_id: int, message: str) -> None:
        self.recordings_panel.set_status(row_id, "failed", message)
        self._set_status(f"Échec du traitement : {message}", force=True)

    def _on_build_finished(self, ok: bool) -> None:
        self._progress_debouncer.flush()
        self._set_busy(False)
        self._build_worker = None
        self._config_dirty = False
        self.params_panel.set_dirty_message("")
        if ok:
            self.status_panel.set_headline("Enregistrement prêt — calcul des canaux…")
            self._mark_display_loaded()
            self._set_status("Prêt — le canal affiché est prioritaire ; le reste suit en arrière-plan.")
        else:
            self.status_panel.set_headline("Traitement terminé avec des erreurs ou annulé.")
        self._refresh_channels()
        # Conserver onglets / zooms / layout : Traiter ne doit pas changer l’affichage.
        self._schedule_redraw(force=True, preserve_view=True)
        if ok:
            self._start_background_prefetch()

    def _set_busy(self, busy: bool) -> None:
        """Verrouiller seulement Traiter — navigation / params / canaux restent libres."""
        self.recordings_panel.set_busy(busy)
        self.control_panel.set_busy(busy)
        self._act_process.setEnabled(not busy)
        build_or_channels = busy or self._channel_compute_active
        self._act_cancel.setEnabled(build_or_channels)
        self._act_ensure_channel.setEnabled(not busy)
        self._act_ensure_all.setEnabled(not busy)
        if self._status_progress.isVisible() != build_or_channels:
            self._status_progress.setVisible(build_or_channels)
        # Ne pas ouvrir le dock Journal : ça recompose toute la fenêtre.

    def _set_channel_compute_active(self, active: bool, *, background: bool = False) -> None:
        self._channel_compute_active = bool(active)
        self._ensure_background = bool(background) if active else False
        build_busy = self._build_worker is not None and self._build_worker.isRunning()
        self._act_cancel.setEnabled(active or build_busy)
        show_progress = bool(active or build_busy)
        if self._status_progress.isVisible() != show_progress:
            self._status_progress.setVisible(show_progress)
        # Ne pas ouvrir / voler le focus du Journal pendant le calcul.

    def _on_pipeline_progress(self, event: Any) -> None:
        self._pending_progress = event
        self._progress_debouncer.request()

    def _flush_pipeline_progress(self) -> None:
        event = self._pending_progress
        if event is None:
            return
        self._pending_progress = None
        self.status_panel.on_progress(event)
        overall = float(getattr(event, "overall_fraction", 0.0) or 0.0)
        value = int(max(0.0, min(1.0, overall)) * 1000)
        if self._status_progress.value() != value:
            self._status_progress.setValue(value)
        stage = str(getattr(event, "stage_label", "") or "")
        recording = str(getattr(event, "recording", "") or "")
        # Barres par fichier : uniquement pendant le build .rhs (pas le prefetch canaux).
        building = self._build_worker is not None and self._build_worker.isRunning()
        if building and recording:
            entry = self.recordings_panel.entry_by_label(recording)
            if entry is not None and entry.status in {"queued", "building"}:
                self.recordings_panel.set_progress(entry.row_id, overall, stage)
        if recording and stage:
            text = f"{recording} — {stage}"
        elif stage:
            text = stage
        else:
            return
        if self._status_message.text() != text:
            self._status_message.setText(text)

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
        self._start_channel_ensure(
            self._channels_for_current_view(all_channels=False),
            priority=True,
            background=False,
        )

    def ensure_all_channels(self) -> None:
        self._pending_ensure_all = True
        self._start_channel_ensure(
            self._channels_for_current_view(all_channels=True),
            priority=False,
            background=True,
        )

    def _prioritize_viewed_channels(self) -> None:
        """Bump the currently viewed channel(s) ahead of background prefetch."""
        if self._build_worker is not None and self._build_worker.isRunning():
            return
        channels = self._channels_for_current_view(all_channels=False)
        if not channels:
            return
        self._start_channel_ensure(
            channels,
            allow_empty_redraw=False,
            priority=True,
            background=self._ensure_background,
        )

    def _start_background_prefetch(self) -> None:
        """Compute every channel in the background after the skeleton is ready."""
        plotted_or_ready = self.recordings_panel.ready_entries()
        if not plotted_or_ready:
            return
        recording = plotted_or_ready[0].recording
        n_channels = int(getattr(recording, "n_channels", 0) or 0)
        if n_channels <= 0:
            return
        # Vue active d’abord, puis le reste.
        viewed = self._channels_for_current_view(all_channels=False)
        if viewed:
            self._start_channel_ensure(
                viewed,
                allow_empty_redraw=False,
                priority=True,
                background=True,
            )
        remaining = [ch for ch in range(n_channels) if ch not in set(viewed)]
        if remaining:
            self._start_channel_ensure(
                remaining,
                allow_empty_redraw=False,
                priority=False,
                background=True,
            )

    def _build_ensure_requests(
        self,
        channels: list[int],
        *,
        need_means: bool,
        need_rms: bool,
        need_spikes: bool,
        need_overlay: bool,
    ) -> list[ChannelEnsureRequest]:
        ready = self.recordings_panel.ready_entries()

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
        for entry in ready:
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
        return requests

    def _start_channel_ensure(
        self,
        channels: list[int],
        *,
        allow_empty_redraw: bool = True,
        need_means: bool = True,
        need_rms: bool = True,
        need_spikes: bool = True,
        need_overlay: bool = True,
        priority: bool = True,
        background: bool = False,
    ) -> None:
        if self._build_worker is not None and self._build_worker.isRunning():
            return

        window_id = self._pending_window_id
        if not channels:
            if window_id:
                self._redraw_pending_window(window_id)
            elif allow_empty_redraw:
                self._schedule_redraw(force=True)
            return

        requests = self._build_ensure_requests(
            channels,
            need_means=need_means,
            need_rms=need_rms,
            need_spikes=need_spikes,
            need_overlay=need_overlay,
        )
        if not requests:
            if window_id:
                self._redraw_pending_window(window_id)
            elif allow_empty_redraw:
                self._schedule_redraw(force=True)
            plotted_now = self._plotted()
            self.channel_panel.set_reference_recording(
                plotted_now[0].recording if plotted_now else None
            )
            return

        worker = self._ensure_worker
        if worker is not None and self._ensure_accepting:
            worker.submit(requests, priority=priority)
            if not background:
                self._ensure_background = False
                self._set_channel_compute_active(True, background=False)
                n = sum(len(request.channels) for request in requests)
                self.status_panel.set_headline(
                    f"Priorité : calcul de {n} canal(aux) regardé(s)…"
                )
                self.status_panel.append_log(
                    f"Canal(aux) prioritaire(s) : {len(channels)} — le prefetch continue ensuite."
                )
            return

        worker = ChannelEnsureWorker(requests, self, priority=priority)
        worker.progressed.connect(self._on_pipeline_progress)
        worker.logged.connect(self.status_panel.append_log)
        worker.succeeded.connect(self._on_channel_computed)
        worker.queue_changed.connect(self._on_ensure_queue_changed)
        worker.failed.connect(
            lambda message: self._set_status(f"Échec du calcul de canal : {message}", force=True)
        )
        worker.finished_all.connect(self._on_ensure_finished)
        self._ensure_worker = worker
        self._ensure_accepting = True
        self._set_channel_compute_active(True, background=background)
        n = sum(len(request.channels) for request in requests)
        if background and not priority:
            self.status_panel.set_headline(f"Calcul en arrière-plan : {n} canal(aux)…")
        elif background:
            self.status_panel.set_headline(
                f"Canal affiché d’abord, puis calcul en arrière-plan…"
            )
        else:
            self.status_panel.set_headline(f"Calcul de {n} canal(aux)…")
        worker.start()

    def _on_channel_computed(self, recording: Any, channels: list) -> None:
        """Un canal vient d’être calculé — rafraîchir s’il est visible."""
        ready_set = {int(ch) for ch in channels}
        viewed = set(self._channels_for_current_view(all_channels=False))
        if ready_set & viewed:
            self._mark_display_loaded()
            self._schedule_redraw(force=True, preserve_view=True)
            window_id = self._pending_window_id
            if window_id:
                self._redraw_pending_window(window_id)
                self._pending_window_id = None

        if self._channel_inspect is not None and int(
            self._channel_inspect.channel_index
        ) in ready_set:
            self._channel_inspect.set_channel_ready(True)
            # Après Traiter / prefetch : garder zoom et pan de l’aperçu.
            self._channel_inspect.redraw(preserve_view=True)

        # Badges / carte MEA : coalescer pendant le prefetch (évite un freeze/liste).
        if recording is not None:
            self._channel_badge_debouncer.request()

    def _flush_channel_badges(self) -> None:
        plotted = self._plotted()
        if plotted:
            self.channel_panel.set_reference_recording(plotted[0].recording)
            return
        ready = self.recordings_panel.first_ready()
        self.channel_panel.set_reference_recording(
            ready.recording if ready is not None else None
        )

    def _on_ensure_queue_changed(self, high: int, low: int) -> None:
        # Ne pas écraser le headline de progression (sinon clignotement à chaque canal).
        # On n’affiche la file que s’il n’y a pas encore de progression active.
        if "Computing selected channels" in self.status_panel.headline_text():
            return
        if high > 0:
            self.status_panel.set_headline(
                f"Priorité vue : {high} canal(aux), puis {low} en arrière-plan…"
            )
        elif low > 0:
            self.status_panel.set_headline(f"Calcul en arrière-plan : {low} canal(aux) restant(s)…")

    def _on_ensure_finished(self, ok: bool) -> None:
        self._progress_debouncer.flush()
        self._channel_badge_debouncer.flush()
        self._ensure_accepting = False
        self._set_channel_compute_active(False)
        self._ensure_worker = None
        self._pending_ensure_all = False
        window_id = self._pending_window_id
        self._pending_window_id = None
        if ok:
            self.status_panel.set_headline("Tous les canaux sont prêts.")
            self._set_status("Canaux prêts.")
            self._mark_display_loaded()
        else:
            self.status_panel.set_headline("Calcul des canaux terminé avec des erreurs ou annulé.")
        plotted = self._plotted()
        self.channel_panel.set_reference_recording(plotted[0].recording if plotted else None)
        self._redraw_pending_window(window_id)
        self._schedule_redraw(
            force=True,
            inspectors=window_id is None,
            preserve_view=True,
        )
    def _redraw_pending_window(self, window_id: str | None) -> None:
        if not window_id:
            return
        window = self._channel_inspect
        if window is not None and window.window_id == window_id:
            window.set_channel_ready(True)
            window.redraw(preserve_view=True)

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

    def _set_status(self, message: str, *, force: bool = False) -> None:
        if not force and self._pipeline_ui_locked():
            return
        if self._status_message.text() != message:
            self._status_message.setText(message)

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "À propos de plot_erg",
            "Visionneuse ERG — paradigme canal d’abord.\n\n"
            "1. Ajouter un .rhs (Ctrl+O), puis Traiter (F5).\n"
            "2. Canaux / carte MEA : clic = aperçu (continuous + plages).\n"
            "3. Paramètres → Canal : continuous / moyenne / stimulation.\n"
            "4. Revue montage (Ctrl+M) ; retour aperçu (Ctrl+Shift+M).\n\n"
            "Mapping MEA optionnel : Session → Mapping MEA.\n"
            "Journal : Ctrl+J.",
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
        # restoreState / sizeHints de tables peuvent avoir gonflé les mins — reset.
        self._dock_session.setMinimumWidth(200)
        self._dock_session.setMaximumWidth(16777215)
        self._dock_params.setMinimumWidth(200)
        self._dock_params.setMaximumWidth(16777215)
        # Répartir Session / Paramètres si un ancien état a écrasé Paramètres.
        if self.width() >= 1100 and (
            self._dock_params.width() < 280 or self._dock_session.width() > self.width() // 2
        ):
            self.resizeDocks(
                [self._dock_session, self._dock_params],
                [360, 340],
                Qt.Orientation.Horizontal,
            )

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
        if self._channel_inspect is not None:
            try:
                self.params_panel.set_channel_side_panel(None)
                self._channel_inspect.shutdown()
            except RuntimeError:
                pass
            self._channel_inspect = None
        self.recordings_panel.clear()
        super().closeEvent(event)


def launch_qt_gui(
    run_multi_comparison_callback: Callable[[list[AnalysisConfig]], None] | None = None,
    **defaults: Any,
) -> int:
    """Ouvrir la visionneuse interactive. Le callback pilote l’export PDF classique."""
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
