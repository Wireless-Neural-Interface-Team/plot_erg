"""Visionneuse interactive pour enregistrements Intan RHS / ERG.

Paradigme canal d’abord :
- gauche  : Session (zone Mapping + enregistrements / liste)
- centre  : aperçu canal (continuous / moyenne / stimulation)
- Paramètres → Canal : mode / plages + Traitement (F5)
- Paramètres → Affichage : axe X, sync, légendes, apparence (redessin immédiat)
- Paramètres → Canal → sections courbes : échelles Y par type (redessin immédiat)
- montage multi-canaux : optionnel (Revue montage)
- bas     : Control Panel (canal · pagination montage)
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
from gui.defaults import (
    probe_path_from_defaults,
    viewer_settings_from_defaults,
)
from gui.jobs import BuildRequest, BuildWorker, ChannelEnsureRequest, ChannelEnsureWorker, Debouncer, TaskWorker
from gui.services import (
    ChannelPreviewController,
    ExportService,
    MontageController,
    PipelineController,
    RedrawScheduler,
    RenderRequestFactory,
)
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
from panel_catalog import merge_product_needs, panel_product_needs
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
        self._cache_root: Path | None = None
        self._cache_roots: set[Path] = set()
        self._warned_channel_mismatch = False
        self._pending_ensure_all = False
        self._pending_window_id: str | None = None
        self._ensure_background = False
        self._ensure_accepting = False
        self._channel_compute_active = False
        self._pending_progress: Any | None = None
        self._pending_open_entries: list[RecordingEntry] = []

        # Services (assembleur mince) — flags redraw / montage via proxies.
        self._redraw = RedrawScheduler(self, on_redraw=self._redraw_active_tab)
        self._pipeline = PipelineController(self)
        self._montage = MontageController(self)
        self._render_requests = RenderRequestFactory(self)
        self._export = ExportService(self)

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
        self._channel_preview = ChannelPreviewController(self._preview_placeholder)

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

        # Masquer/afficher des canaux : debounce dédié (pas de force/invalidate_all).
        self._visibility_redraw_debouncer = Debouncer(160, self)
        self._visibility_redraw_debouncer.triggered.connect(
            self._redraw_channel_visibility
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
        self.control_panel.montagePrevPage.connect(
            lambda: self._shift_montage_review_page(-1)
        )
        self.control_panel.montageNextPage.connect(
            lambda: self._shift_montage_review_page(+1)
        )
        self.params_panel.pipelineVisibilityChanged.connect(
            self._on_pipeline_visibility_changed
        )
        self._syncing_pipeline_visibility = False

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
        act_reprocess_all.setShortcut(QKeySequence("Ctrl+Shift+F5"))
        act_reprocess_all.setToolTip("Retraiter tous les enregistrements (Ctrl+Shift+F5)")
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
        act_cache.setShortcut(QKeySequence("Ctrl+Shift+C"))
        act_cache.setToolTip("Inspecter / nettoyer le cache disque (Ctrl+Shift+C)")
        act_cache.triggered.connect(self.open_cache_manager)

        act_configure = QAction("Configurer les panneaux…", self)
        act_configure.setShortcut(QKeySequence("Ctrl+Shift+P"))
        act_configure.setToolTip(
            "Choisir les panneaux de la revue montage (Ctrl+Shift+P)."
        )
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

        process_menu = self.menuBar().addMenu("&Traitement")
        for action in (
            self._act_process,
            self._act_cancel,
            act_reprocess_all,
            self._act_ensure_channel,
            self._act_ensure_all,
            act_cache,
        ):
            process_menu.addAction(action)
            self.addAction(action)

        view_menu = self.menuBar().addMenu("&Vue")
        for action in (
            act_configure,
            act_redraw,
            self._act_analyse,
            self._act_montage,
            self._act_preview,
            act_prev,
            act_next,
        ):
            view_menu.addAction(action)
            self.addAction(action)
        view_menu.addSeparator()
        for dock in (self._dock_session, self._dock_params, self._dock_status):
            action = dock.toggleViewAction()
            view_menu.addAction(action)
            self.addAction(action)
            if dock is self._dock_params:
                action.setShortcut(QKeySequence("Ctrl+P"))
                action.setToolTip("Afficher / masquer le panneau Paramètres (Ctrl+P)")
            elif dock is self._dock_status:
                action.setShortcut(QKeySequence("Ctrl+J"))
                action.setToolTip("Afficher / masquer le journal (Ctrl+J)")

        help_menu = self.menuBar().addMenu("&Aide")
        help_menu.addAction(act_about)
        self.addAction(act_about)

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
                tab.panels,
                columns=tab.columns,
                panel_height=tab.panel_height_px,
                uniform=True,
            )
            self.tabs.addTab(page, tab.name)
        index = min(max(0, current if current >= 0 else self._workspace.active_index),
                    max(0, self.tabs.count() - 1))
        self.tabs.setCurrentIndex(index)
        self.tabs.blockSignals(False)
        self.tabs.tabBar().hide()
        # Revue montage = au moins le panneau multi-canaux (éventuellement + graphs).
        tab = self._current_tab()
        is_montage = bool(
            tab is not None
            and bool(tab.panels)
            and any(p.panel == "montage_continuous_raw" for p in tab.panels)
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
            page.grid.configure(
                tab.panels,
                columns=tab.columns,
                panel_height=tab.panel_height_px,
                uniform=True,
            )
        self._schedule_redraw(force=True)

    def configure_current_view(self) -> None:
        if not self._in_montage:
            QMessageBox.information(
                self,
                "Configurer les panneaux",
                "Ouvrez d’abord la revue montage (Ctrl+M), puis configurez ses panneaux "
                "(Ctrl+Shift+P).\n\n"
                "En aperçu canal, les graphs se choisissent dans Paramètres → Canal.",
            )
            return
        tab = self._current_tab()
        if tab is None:
            QMessageBox.information(
                self,
                "Configurer les panneaux",
                "Aucun onglet de montage à configurer.",
            )
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

    @property
    def _preserve_view(self) -> bool:
        return self._redraw.preserve_view

    @_preserve_view.setter
    def _preserve_view(self, value: bool) -> None:
        self._redraw.preserve_view = bool(value)

    @property
    def _block_preserve_view(self) -> bool:
        return self._redraw.block_preserve_view

    @_block_preserve_view.setter
    def _block_preserve_view(self, value: bool) -> None:
        self._redraw.block_preserve_view = bool(value)

    @property
    def _in_montage(self) -> bool:
        return self._montage.in_montage

    @_in_montage.setter
    def _in_montage(self, value: bool) -> None:
        self._montage.set_in_montage(value)

    def _schedule_redraw(
        self,
        *,
        force: bool = False,
        inspectors: bool = False,
        preserve_view: bool = False,
        reset_view: bool = False,
    ) -> None:
        self._redraw.schedule(
            force=force,
            inspectors=inspectors,
            preserve_view=preserve_view,
            reset_view=reset_view,
        )

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
        force, redraw_inspectors, preserve_view = self._redraw.begin_redraw()
        self._settings = self._default_viewer_settings()
        # Toujours resync la hauteur montage (indépendant du zoom/pan).
        if self._in_montage:
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
        # Toujours passer par le merge complet (mode, flux, canaux masqués…).
        self._settings = self._default_viewer_settings()
        # Montage : hauteur = lignes × hauteur de ligne (jamais graph_height_px).
        if self._in_montage:
            self._clamp_montage_review_page()
            self._settings = self._default_viewer_settings()
            self._adapt_montage_height()
            self._sync_montage_page_controls()
        else:
            self._apply_graph_height_to_tabs()
        if self._channel_inspect is not None and self._showing_channel_preview():
            self._channel_inspect.apply_viewer_settings(self._settings)
            # Affichage (échelles, sync, hauteur…) : garder zoom/pan.
            self._channel_inspect.redraw(preserve_view=True)
            self._redraw_detached_windows()
            return
        # Revue montage / onglets : même règle — ne pas reset zoom/pan
        # (ex. changement d’« Origine du temps » = sync seule).
        self._schedule_redraw(force=True, preserve_view=True)

    def _apply_graph_height_to_tabs(self) -> None:
        """Propager la hauteur fixe Affichage aux onglets (hors revue montage)."""
        if self._in_montage:
            return
        height = max(160, int(self._settings.graph_height_px))
        tabs = list(self._workspace.tabs)
        changed = False
        for index, tab in enumerate(tabs):
            if tab.panels and all(
                p.panel == "montage_continuous_raw" or str(p.panel).startswith("montage_")
                for p in tab.panels
            ):
                continue
            if int(tab.panel_height_px) == height:
                continue
            tabs[index] = replace(tab, panel_height_px=height)
            changed = True
            page = self.tabs.widget(index)
            if isinstance(page, ViewTabPage):
                page.grid.set_panel_height(height, uniform=True, fill=False)
        if changed:
            self._workspace = self._workspace.with_tabs(tabs)

    def _on_params_config_changed(self) -> None:
        self._config_dirty = True
        self.params_panel.set_dirty_message(
            "Paramètres de traitement modifiés — Traiter (F5) pour recalculer."
        )
        # Ne pas forcer l’onglet Canal : l’utilisateur peut être sur Affichage.
        self._set_status("Paramètres modifiés — F5 pour retraiter.")
        # Stop channel workers before closing recordings they may still hold.
        self._stop_ensure_worker(wait_ms=250)
        # Marquer les lignes « prêt » comme à retraiter (évite un état stale).
        self.recordings_panel.invalidate_results()
        # Mapping MEA : appliquer tout de suite si le chemin a changé (pas besoin de F5).
        probe = self.params_panel.probe_layout_path()
        if probe != self._probe_path:
            self._load_probe(probe, sync_ui=True)

    def _on_session_probe_path(self, path: object) -> None:
        self._load_probe(Path(path) if path else None, sync_ui=True)

    def _on_channel_changed(self, channel: str) -> None:
        self.control_panel.set_channel(channel)
        self._preserve_view = False
        if self._is_continuous_montage_view():
            # Surbrillance seule (style refresh) — pas de prefetch means/spikes.
            self._schedule_redraw(force=True, preserve_view=True)
            return
        # Prioriser le canal regardé, même pendant un prefetch en arrière-plan.
        self._prioritize_viewed_channels()
        self._prefetch_neighbor_channels()
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
        """Chauffer filtre / moyennes des voisins hors thread UI (basse priorité)."""
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
        need_means, need_rms, need_spikes, _need_overlay = self._view_product_needs()
        # Always enqueue filter warm on the worker — never LazyFilterBank.prefetch here.
        self._ensure_channels_for_indices(
            neighbors,
            need_means=need_means,
            need_rms=need_rms,
            need_spikes=need_spikes,
            need_overlay=False,
            need_filters=True,
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

    @staticmethod
    def _is_montage_placement(placement: Any) -> bool:
        panel = str(getattr(placement, "panel", "") or "")
        return panel == "montage_continuous_raw" or panel.startswith("montage_")

    def _redraw_channel_visibility(self) -> None:
        """Redraw ciblé après coches Canaux — sans invalidate_all ni 2ᵉ debounce."""
        self._settings = self._default_viewer_settings()
        if self._in_montage:
            self._clamp_montage_review_page()
            self._settings = self._default_viewer_settings()
            self._adapt_montage_height()
            self._sync_montage_page_controls()
        if self._showing_channel_preview():
            return
        page = self._current_page()
        if page is not None:
            page.grid.invalidate_matching(self._is_montage_placement)
            page.grid.schedule_render(self._make_request_or_blank, force=False)
        for window in list(self._detached.values()):
            if self._is_montage_placement(window.placement):
                self._render_detached(window)

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

    def _montage_review_per_page(self) -> int:
        """Canaux par page (Paramètres → Affichage → Revue montage)."""
        return max(1, int(getattr(self._settings, "montage_review_channels", 10) or 10))

    def _montage_review_page_info(
        self, recording: Any
    ) -> tuple[list[int], int, int, int]:
        """``(indices page, page, n_pages, n_visible)`` pour la revue montage."""
        visible = self._montage_visible_indices(recording)
        per_page = self._montage_review_per_page()
        n_visible = len(visible)
        n_pages = max(1, (n_visible + per_page - 1) // per_page) if n_visible else 1
        page = max(0, min(n_pages - 1, int(getattr(self._settings, "montage_review_page", 0) or 0)))
        start = page * per_page
        return visible[start : start + per_page], page, n_pages, n_visible

    def _montage_page_channel_count(self, recording: Any) -> int:
        """Nombre de canaux sur la page courante (hauteur du montage)."""
        indices, _page, _n_pages, _n_visible = self._montage_review_page_info(recording)
        return max(1, len(indices)) if indices else 1

    def _clamp_montage_review_page(self) -> None:
        """Ramener la page dans les bornes après masquage / changement de taille."""
        entries = self._plotted()
        n_visible = self._montage_visible_count(entries[0].recording) if entries else 0
        self.params_panel.clamp_montage_review_page(n_visible=n_visible, emit=False)

    def _shift_montage_review_page(self, delta: int) -> None:
        """Boutons Suivant / Précédent de la revue montage."""
        if not self._in_montage or not delta:
            return
        entries = self._plotted()
        if not entries:
            return
        _indices, page, n_pages, _n_visible = self._montage_review_page_info(
            entries[0].recording
        )
        new_page = max(0, min(n_pages - 1, page + int(delta)))
        if new_page == page:
            return
        # Changement de structure (autres canaux) : rebuild, pas style-only.
        self.params_panel.set_montage_review_page(new_page, emit=False)
        self._settings = self._default_viewer_settings()
        self._adapt_montage_height()
        self._sync_montage_page_controls()
        page_widget = self._current_page()
        if page_widget is not None:
            page_widget.grid.invalidate_matching(self._is_montage_placement)
        self._schedule_redraw(force=True, reset_view=True)

    def _sync_montage_page_controls(self) -> None:
        """Rafraîchir le libellé / boutons de pagination du bandeau."""
        if not self._in_montage:
            return
        entries = self._plotted()
        if not entries:
            self.control_panel.set_montage_page_info(
                page=0, n_pages=1, start=0, end=0, n_visible=0
            )
            return
        indices, page, n_pages, n_visible = self._montage_review_page_info(
            entries[0].recording
        )
        if indices:
            start = indices[0] + 1
            end = indices[-1] + 1
        else:
            start = end = 0
        self.control_panel.set_montage_page_info(
            page=page,
            n_pages=n_pages,
            start=start,
            end=end,
            n_visible=n_visible,
        )

    def _montage_review_placements(self) -> tuple[PanelPlacement, ...]:
        """Un seul panneau : tous les graphs Pipeline intégrés en lignes multi-canaux."""
        return (PanelPlacement("montage_continuous_raw"),)

    def _montage_review_kind_count(self) -> int:
        """Nombre de lignes par canal (flux + extras Pipeline)."""
        streams = self._settings.resolved_continuous_streams() or ("raw",)
        analysis = self._settings.analysis
        extras = sum(
            1
            for flag in (
                analysis.show_rms,
                analysis.show_psth,
                analysis.show_trial_rate,
                analysis.show_raster,
                analysis.show_isi,
                analysis.show_overlay,
            )
            if flag
        )
        return max(1, len(streams) + extras)

    def _montage_total_height_px(self, n_visible: int) -> int:
        """Hauteur du montage (canaux×kinds × ligne + chrome panneau)."""
        rows = max(1, int(n_visible)) * self._montage_review_kind_count()
        min_row = max(36, int(self._settings.montage_row_min_height_px))
        # Chrome : en-tête + toolbar PlotHost + curseur + marges.
        chrome = 96
        return max(220, rows * min_row + chrome)

    def _sync_montage_review_panels(self) -> None:
        """Resynchroniser la grille montage sur Mode / Pipeline / contexte."""
        if not self._in_montage:
            return
        tab = self._current_tab()
        if tab is None:
            return
        self._clamp_montage_review_page()
        self._settings = self._default_viewer_settings()
        placements = self._montage_review_placements()
        entries = self._plotted()
        n_on_page = 1
        if entries:
            n_on_page = self._montage_page_channel_count(entries[0].recording)
        height = self._montage_total_height_px(n_on_page)
        same_panels = tuple(p.key for p in tab.panels) == tuple(p.key for p in placements)
        if same_panels and int(tab.panel_height_px) == height:
            self._adapt_montage_height()
            self._sync_montage_page_controls()
            self._ensure_montage_review_data()
            return
        updated = replace(
            tab,
            panels=placements,
            columns=1,
            panel_height_px=height,
        )
        self._replace_tab(self.tabs.currentIndex(), updated)
        self._adapt_montage_height()
        self._sync_montage_page_controls()
        self._ensure_montage_review_data()

    def _adapt_montage_height(self) -> None:
        """Resynchroniser la hauteur du montage (page courante / flux)."""
        tab = self._current_tab()
        page = self._current_page()
        if tab is None or page is None:
            return
        if not any(p.panel == "montage_continuous_raw" for p in tab.panels):
            return
        entries = self._plotted()
        n_on_page = 1
        if entries:
            n_on_page = self._montage_page_channel_count(entries[0].recording)
        height = self._montage_total_height_px(n_on_page)
        graph_h = max(160, int(self._settings.graph_height_px))
        if int(getattr(tab, "panel_height_px", 0) or 0) != height:
            new_tab = replace(tab, panel_height_px=height)
            tabs = list(self._workspace.tabs)
            active = max(0, min(len(tabs) - 1, int(self._workspace.active_index)))
            tabs[active] = new_tab
            self._workspace = replace(self._workspace, tabs=tuple(tabs))
            tab = new_tab
        setter = getattr(page.grid, "set_panel_height", None)
        if callable(setter):
            setter(height, panels=("montage_continuous_raw",), uniform=True, fill=False)
        elif getattr(page.grid, "_panel_height", None) != height:
            page.grid.configure(
                tab.panels,
                columns=tab.columns,
                panel_height=height,
                uniform=True,
            )
        graph_setter = getattr(page.grid, "set_graph_height", None)
        if callable(graph_setter):
            graph_setter(graph_h)

    def _default_viewer_settings(self) -> ViewerSettings:
        """Réglages d’affichage de la vue centrale (ParamsPanel)."""
        settings = self.params_panel.viewer_settings()
        streams = settings.resolved_continuous_streams()
        # Pipeline (cases Paramètres) = source de vérité des courbes affichées.
        # L’aperçu canal ne sert qu’au mode / stim_index / résumés locaux.
        analysis = settings.analysis
        preview_content = settings.preview_content
        if self._channel_inspect is not None:
            local = self._channel_inspect.local_settings()
            preview_content = self._channel_inspect.preview_mode()
            analysis = replace(
                analysis,
                mode=local.analysis.mode,
                stim_index=local.analysis.stim_index,
                show_summary_rms=local.analysis.show_summary_rms,
                show_summary_rms_table=local.analysis.show_summary_rms_table,
            )
        return replace(
            settings,
            analysis=analysis,
            preview_content=preview_content,  # type: ignore[arg-type]
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
        need_filters: bool = True,
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
            need_filters=need_filters,
            priority=priority,
            background=background,
        )

    def _on_pipeline_visibility_changed(self) -> None:
        """Cases Pipeline → même chemin pour continuous / moyenne / stimulation."""
        if self._syncing_pipeline_visibility:
            return
        self._settings = self._default_viewer_settings()
        self._preserve_view = False
        self._block_preserve_view = True
        window = self._channel_inspect
        if window is not None:
            self._syncing_pipeline_visibility = True
            try:
                window.apply_pipeline_visibility(
                    self.params_panel.pipeline_visibility_flags(),
                    mark_stims=self.params_panel.mark_stimulations(),
                    redraw=self._showing_channel_preview(),
                )
            finally:
                self._syncing_pipeline_visibility = False
            if self._showing_channel_preview():
                self._block_preserve_view = False
                return
        if self._in_montage:
            self._settings = self._default_viewer_settings()
            self._sync_montage_review_panels()
            # Invalider la figure montage : interdit style-only / shrink stale.
            page = self._current_page()
            if page is not None:
                page.grid.invalidate_matching(self._is_montage_placement)
        self._schedule_redraw(force=True, reset_view=True)

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
        return self._pipeline.pipeline_busy()

    def _pipeline_ui_locked(self) -> bool:
        """True tant qu’un build ou un calcul de canaux possède la barre de statut."""
        return self._pipeline.pipeline_ui_locked()

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
                and any(p.panel == "montage_continuous_raw" for p in tab.panels)
            )
        )
        if already:
            self._settings = self._default_viewer_settings()
            self._set_view_mode(montage=True)
            self._sync_montage_review_panels()
            self._schedule_redraw(force=True)
            self._set_status("Revue montage déjà affichée — graphs resynchronisés.")
            return
        # Aligner sur ParamsPanel / aperçu (mode + flux + extras) avant le 1er redraw.
        self._settings = self._default_viewer_settings()
        recording = plotted[0].recording
        n_channels = int(getattr(recording, "n_channels", 0) or 0)
        self._clamp_montage_review_page()
        self._settings = self._default_viewer_settings()
        page_indices, page, n_pages, n_visible = self._montage_review_page_info(recording)
        n_on_page = max(1, len(page_indices)) if page_indices else 1
        placements = self._montage_review_placements()
        height = self._montage_total_height_px(n_on_page)
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
        self._adapt_montage_height()
        self._sync_montage_page_controls()
        self._ensure_montage_review_data()
        mode = str(self._settings.preview_content or "continuous")
        streams = self._settings.resolved_continuous_streams()
        stream_txt = "+".join(
            {"raw": "WIDE", "hp": "HIGH", "lp": "LOW"}.get(s, s.upper()) for s in streams
        )
        mode_txt = {
            "continuous": "continuous",
            "average": "moyenne",
            "stimulation": f"stim. n°{int(self._settings.analysis.stim_index) + 1}",
        }.get(mode, mode)
        n_kinds = self._montage_review_kind_count()
        per_page = self._montage_review_per_page()
        self._set_status(
            f"Revue montage — {n_on_page}/{n_visible} canaux (page {page + 1}/{n_pages}"
            f", {per_page}/page) · {n_channels} total · {mode_txt}"
            f" · {stream_txt or 'WIDE'} · {n_kinds} graph(s)/canal"
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
        if montage:
            self._montage.enter()
        else:
            self._montage.leave()

    def sync_montage_ui(self, *, montage: bool) -> None:
        """Callback MontageController : sync control panel / params / actions."""
        mode = "montage" if montage else "preview"
        self.control_panel.set_view_mode(mode)
        self.params_panel.set_view_mode(mode)
        if hasattr(self, "_act_preview"):
            self._act_preview.setEnabled(bool(montage))
        if hasattr(self, "_act_montage"):
            self._act_montage.setEnabled(not bool(montage))
        if montage:
            self._sync_montage_page_controls()
        self._update_view_stack()

    def _is_continuous_montage_view(self) -> bool:
        """True si le montage ne lit que des traces memmap (pas RMS/spikes/moyennes)."""
        if not self._in_montage:
            return False
        tab = self._current_tab()
        if tab is None or not tab.panels:
            return False
        if not all(p.panel == "montage_continuous_raw" for p in tab.panels):
            return False
        if str(self._settings.preview_content or "continuous") != "continuous":
            return False
        need_rms, need_spikes, need_overlay = self._montage_extra_flags()
        return not (need_rms or need_spikes or need_overlay)

    def _montage_review_needs_means(self) -> bool:
        """True si la revue montage lit des moyennes / fenêtres de stim."""
        if not self._in_montage:
            return False
        return str(self._settings.preview_content or "continuous") in {
            "average",
            "stimulation",
        }

    def _montage_extra_flags(self) -> tuple[bool, bool, bool]:
        """``(need_rms, need_spikes, need_overlay)`` pour les graphs hors traces."""
        return self._analysis_extra_flags(self._settings.analysis)

    @staticmethod
    def _analysis_extra_flags(analysis: Any) -> tuple[bool, bool, bool]:
        """``(need_rms, need_spikes, need_overlay)`` depuis les flags Pipeline / Analyse."""
        need_rms = bool(
            getattr(analysis, "show_rms", False)
            or getattr(analysis, "show_summary_rms", False)
            or getattr(analysis, "show_summary_rms_table", False)
        )
        need_spikes = bool(
            getattr(analysis, "show_isi", False)
            or getattr(analysis, "show_overlay", False)
            or getattr(analysis, "show_psth", False)
            or getattr(analysis, "show_trial_rate", False)
            or getattr(analysis, "show_raster", False)
        )
        need_overlay = bool(getattr(analysis, "show_overlay", False))
        return need_rms, need_spikes, need_overlay

    def _preview_product_needs(
        self, window: ChannelAnalysisWindow | None = None
    ) -> tuple[bool, bool, bool, bool]:
        """Produits requis par l’aperçu canal (mode + panels d’analyse)."""
        win = window if window is not None else self._channel_inspect
        if win is None:
            analysis = self._settings.analysis
            mode = str(self._settings.preview_content or "continuous")
            need_rms, need_spikes, need_overlay = self._analysis_extra_flags(analysis)
            need_means = mode in {"average", "stimulation"}
            return need_means, need_rms, need_spikes, need_overlay

        analysis = win.local_settings().analysis
        need_rms, need_spikes, need_overlay = self._analysis_extra_flags(analysis)
        need_means = bool(win.is_analysis_view())
        for placement in getattr(win, "_analysis_placements", ()) or ():
            means, rms, spikes, overlay = panel_product_needs(
                getattr(placement, "panel", "")
            )
            need_means = need_means or means
            need_rms = need_rms or rms
            need_spikes = need_spikes or spikes
            need_overlay = need_overlay or overlay
        return need_means, need_rms, need_spikes, need_overlay

    def _view_product_needs(self) -> tuple[bool, bool, bool, bool]:
        """Produits à calculer pour ce qui est réellement affiché."""
        if self._is_continuous_montage_view():
            return False, False, False, False

        if self._in_montage:
            need_means = self._montage_review_needs_means()
            need_rms, need_spikes, need_overlay = self._montage_extra_flags()
            if need_means or need_rms or need_spikes or need_overlay:
                need_means = need_means or need_rms or need_spikes
            return need_means, need_rms, need_spikes, need_overlay

        workspace_needs = self._workspace.product_needs()
        if self._showing_channel_preview() or self._channel_inspect is not None:
            preview_needs = self._preview_product_needs(self._channel_inspect)
            return merge_product_needs(workspace_needs, preview_needs)
        if any(workspace_needs):
            return workspace_needs
        return self._preview_product_needs(None)

    def _view_needs_all_channels(self) -> bool:
        """True si la vue courante lit tous les canaux (montage / résumés)."""
        if self._in_montage:
            need_means, need_rms, need_spikes, need_overlay = self._view_product_needs()
            return bool(need_means or need_rms or need_spikes or need_overlay)
        return bool(self._workspace.needs_all_channels())

    def _ensure_montage_review_data(self) -> None:
        """Précharger moyennes / RMS / spikes pour tous les canaux visibles."""
        plotted = self._plotted()
        if not plotted:
            return
        need_means = self._montage_review_needs_means()
        need_rms, need_spikes, need_overlay = self._montage_extra_flags()
        if not (need_means or need_rms or need_spikes or need_overlay):
            return
        indices = self._montage_visible_indices(plotted[0].recording)
        if not indices:
            return
        self._ensure_channels_for_indices(
            indices,
            need_means=need_means or need_rms or need_spikes,
            need_rms=need_rms,
            need_spikes=need_spikes,
            need_overlay=need_overlay,
            priority=True,
            background=True,
        )

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
            self._channel_preview.show_placeholder()
            return

        recording = ready[0].recording
        resolved = recording.channel_index(channel)
        if resolved is None:
            if self._channel_inspect is not None:
                self._channel_inspect.hide()
            self.params_panel.set_channel_side_panel(None)
            self._channel_preview.show_placeholder()
            return

        self._channel_preview.hide_placeholder()
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
            window.analysisCurvesChanged.connect(
                self._on_preview_analysis_curves_changed
            )
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
        window.apply_pipeline_visibility(
            self.params_panel.pipeline_visibility_flags(),
            mark_stims=self.params_panel.mark_stimulations(),
            redraw=False,
        )
        need_means, need_rms, need_spikes, need_overlay = self._preview_product_needs(
            window
        )
        needs_products = bool(need_means or need_rms or need_spikes or need_overlay)
        channel_ready = bool(
            hasattr(recording, "is_channel_ready")
            and recording.is_channel_ready(int(resolved))
        )
        # Continuous / memmap : prêt sans ensure ; sinon selon les produits demandés.
        if not needs_products:
            channel_ready = True
        window.set_channel_ready(channel_ready)
        window.show()
        if needs_products and window.needs_channel_compute():
            self._ensure_channels_for_indices(
                [int(resolved)],
                then_redraw_window=window.window_id,
                need_means=need_means,
                need_rms=need_rms,
                need_spikes=need_spikes,
                need_overlay=need_overlay,
            )
        elif redraw:
            window.redraw(preserve_view=False)

    def open_channel_analysis(self, channel: str | None = None) -> None:
        """Basculer l’aperçu en mode moyenne (ou sélectionner le canal d’abord).

        Entrées : menu Vue / Ctrl+I / Ctrl+Entrée / double-clic MEA.
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

    def _on_preview_analysis_curves_changed(self, analysis: object) -> None:
        """Fermeture d’un panneau → réaligner les cases Pipeline."""
        if self._syncing_pipeline_visibility:
            return
        self._syncing_pipeline_visibility = True
        try:
            self.params_panel.sync_analysis_checks_from(analysis)
        finally:
            self._syncing_pipeline_visibility = False

    def _on_preview_mode_changed(self, mode: object) -> None:
        """Mode continuous / moyenne / stimulation → mêmes flags Pipeline + prefetch."""
        window = self._channel_inspect
        if window is None:
            return
        # Source de vérité unique = cases Pipeline (Canal).
        self._syncing_pipeline_visibility = True
        try:
            window.apply_pipeline_visibility(
                self.params_panel.pipeline_visibility_flags(),
                mark_stims=self.params_panel.mark_stimulations(),
                redraw=False,
            )
        finally:
            self._syncing_pipeline_visibility = False
        if self._in_montage:
            # Revue montage : même mode / graphs que l’aperçu.
            self._settings = self._default_viewer_settings()
            self._sync_montage_review_panels()
            self._schedule_redraw(force=True)
            return
        # Même chemin pour continuous / moyenne / stimulation (RMS & spikes inclus).
        self._on_channel_window_refresh(window)

    def _on_channel_window_refresh(self, window: ChannelAnalysisWindow) -> None:
        if self._in_montage:
            # Mode / stim / contexte / résumés → resync grille montage.
            self._settings = self._default_viewer_settings()
            self._sync_montage_review_panels()
            self._schedule_redraw(force=True)
            return
        need_means, need_rms, need_spikes, need_overlay = self._preview_product_needs(
            window
        )
        needs_products = bool(need_means or need_rms or need_spikes or need_overlay)
        if needs_products:
            self._ensure_channels_for_indices(
                [int(window.channel_index)],
                then_redraw_window=window.window_id,
                need_means=need_means,
                need_rms=need_rms,
                need_spikes=need_spikes,
                need_overlay=need_overlay,
            )
        else:
            # Continuous / memmap : pas d’ensure, redraw immédiat.
            window.set_channel_ready(True)
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
        return self._render_requests.build(
            placement,
            entries,
            channel_index=channel_index,
            channel_name=channel_name,
            settings=settings,
            highlight_zooms=highlight_zooms,
        )

    def _make_request_or_blank(self, placement: PanelPlacement) -> RenderRequest:
        request = self._make_request(placement)
        if request is not None:
            return request
        return self._render_requests.blank(placement, settings=self._settings)

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

        # Ensure workers must not keep writing into recordings being rebuilt.
        self._stop_ensure_worker(wait_ms=250)

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

        root = Path(default_cache_root(config))
        self._cache_roots.add(root)
        self._cache_root = root
        return root

    def _purge_session_caches(self) -> None:
        """Supprimer les caches disque créés pendant la session (`.erg_cache` / work_dir)."""
        from erg_cache import clear_cache

        roots = {Path(p) for p in self._cache_roots}
        if self._cache_root is not None:
            roots.add(Path(self._cache_root))
        for root in roots:
            try:
                clear_cache(root, keep=None, include_raw=True, include_filters=True)
            except Exception:
                pass
        self._cache_roots.clear()
        self._cache_root = None

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
            self._task_worker = None
            self._drain_pending_opens()
            # Keep busy while another open was chained; only unlock when idle.
            if self._task_worker is None and not self._pending_open_entries:
                self._set_busy(False)

        def failed(message: str) -> None:
            self.recordings_panel.set_status(row_id, "failed", message)
            self.status_panel.append_log(f"Impossible d’ouvrir {path.name} : {message}")
            self._task_worker = None
            self._drain_pending_opens()
            if self._task_worker is None and not self._pending_open_entries:
                self._set_busy(False)

        worker = TaskWorker(task, self)
        worker.succeeded.connect(done)
        worker.failed.connect(failed)
        worker.logged.connect(self.status_panel.append_log)
        self._task_worker = worker
        self._set_busy(True)
        worker.start()

    def _drain_pending_opens(self) -> None:
        """Enchaîner les ouvertures de datasets mises en file (toujours hors UI)."""
        if not self._pending_open_entries:
            return
        if self._task_worker is not None and self._task_worker.isRunning():
            return
        next_entry = self._pending_open_entries.pop(0)
        self._open_processed_entry(next_entry)

    def _stop_ensure_worker(self, wait_ms: int = 250) -> None:
        """Stop channel ensure work; short wait only (never freeze the UI for seconds)."""
        self._pipeline.stop_ensure_worker(wait_ms=wait_ms)

    def cancel_processing(self) -> None:
        if self._build_worker is not None and self._build_worker.isRunning():
            self._build_worker.request_stop()
            self.status_panel.set_headline("Annulation…")
        if self._ensure_worker is not None:
            self._stop_ensure_worker(wait_ms=250)
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
        need_means, need_rms, need_spikes, need_overlay = self._view_product_needs()
        if not (need_means or need_rms or need_spikes or need_overlay):
            self._set_status("Rien à calculer pour la vue affichée (traces continues).")
            return
        self._start_channel_ensure(
            self._channels_for_current_view(all_channels=False),
            need_means=need_means,
            need_rms=need_rms,
            need_spikes=need_spikes,
            need_overlay=need_overlay,
            priority=True,
            background=False,
        )

    def ensure_all_channels(self) -> None:
        self._pending_ensure_all = True
        need_means, need_rms, need_spikes, need_overlay = self._view_product_needs()
        if not (need_means or need_rms or need_spikes or need_overlay):
            self._set_status(
                "Rien à précalculer : la vue n’utilise que des traces continues."
            )
            return
        self._start_channel_ensure(
            self._channels_for_current_view(all_channels=True),
            need_means=need_means,
            need_rms=need_rms,
            need_spikes=need_spikes,
            need_overlay=need_overlay,
            priority=False,
            background=True,
        )

    def _prioritize_viewed_channels(self) -> None:
        """Bump the currently viewed channel(s) ahead of background prefetch."""
        if self._build_worker is not None and self._build_worker.isRunning():
            return
        need_means, need_rms, need_spikes, need_overlay = self._view_product_needs()
        channels = self._channels_for_current_view(all_channels=False)
        if not channels:
            return
        # Continuous HP/LP still needs filter warm even when no products are required.
        self._start_channel_ensure(
            channels,
            allow_empty_redraw=False,
            need_means=need_means,
            need_rms=need_rms,
            need_spikes=need_spikes,
            need_overlay=need_overlay,
            need_filters=True,
            priority=True,
            background=self._ensure_background,
        )

    def _start_background_prefetch(self) -> None:
        """Précalcul ciblé après le squelette : vue active, puis tous canaux si besoin."""
        plotted_or_ready = self.recordings_panel.ready_entries()
        if not plotted_or_ready:
            return
        recording = plotted_or_ready[0].recording
        n_channels = int(getattr(recording, "n_channels", 0) or 0)
        if n_channels <= 0:
            return
        need_means, need_rms, need_spikes, need_overlay = self._view_product_needs()
        # Vue active d’abord (filtres + produits), puis le reste si montage/résumés.
        viewed = self._channels_for_current_view(all_channels=False)
        if viewed:
            self._start_channel_ensure(
                viewed,
                allow_empty_redraw=False,
                need_means=need_means,
                need_rms=need_rms,
                need_spikes=need_spikes,
                need_overlay=need_overlay,
                need_filters=True,
                priority=True,
                background=True,
            )
        if not self._view_needs_all_channels():
            return
        remaining = [ch for ch in range(n_channels) if ch not in set(viewed)]
        if remaining:
            self._start_channel_ensure(
                remaining,
                allow_empty_redraw=False,
                need_means=need_means,
                need_rms=need_rms,
                need_spikes=need_spikes,
                need_overlay=need_overlay,
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
        need_filters: bool = True,
    ) -> list[ChannelEnsureRequest]:
        ready = self.recordings_panel.ready_entries()

        def _filters_pending(recording: Any, ch: int) -> bool:
            if not need_filters:
                return False
            return not (
                recording.stream_ready("hp", ch) and recording.stream_ready("lp", ch)
            )

        def _needs_work(recording: Any, ch: int) -> bool:
            if _filters_pending(recording, ch):
                return True
            products = need_means or need_rms or need_spikes or need_overlay
            if not products:
                return False
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
                    need_filters=need_filters,
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
        need_filters: bool = True,
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
            need_filters=need_filters,
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
            accepted = worker.submit(requests, priority=priority)
            if accepted:
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
            # Worker is shutting down — start a fresh one after a short wait.
            self._stop_ensure_worker(wait_ms=250)

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
        will_redraw_preview = bool(ready_set & viewed) and not self._is_continuous_montage_view()
        if ready_set & viewed:
            self._mark_display_loaded()
            # Montage continu = memmap ; means/spikes n’y changent rien.
            if will_redraw_preview:
                self._schedule_redraw(force=True, preserve_view=True)
            window_id = self._pending_window_id
            if window_id and not will_redraw_preview:
                self._redraw_pending_window(window_id)
            self._pending_window_id = None

        if self._channel_inspect is not None and int(
            self._channel_inspect.channel_index
        ) in ready_set:
            self._channel_inspect.set_channel_ready(True)
            # Aperçu : déjà redessiné via _schedule_redraw — éviter le double paint.
            if not (will_redraw_preview and self._showing_channel_preview()):
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
        if self._is_continuous_montage_view():
            # Pas de rebuild montage : les produits canal ne nourrissent pas ce panneau.
            return
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
        ready = [
            e
            for e in self.recordings_panel.ready_entries()
            if e.recording is not None and getattr(e.recording, "source", None) is not None
        ]
        entries = ready or [e for e in self.recordings_panel.entries if not e.is_processed]
        if not entries:
            QMessageBox.information(
                self,
                "Rapport PDF",
                "Ajoutez et traitez au moins un enregistrement .rhs (F5) avant l’export PDF.",
            )
            return
        if self._pdf_callback is None and not ready:
            QMessageBox.information(
                self, "Rapport PDF", "Le pipeline PDF est indisponible dans cette session."
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
        # Workspace vide (aperçu canal seul) → to_plot_display() = all_on ;
        # zoom_mode() renverrait "none" et exclurait les zooms du PDF.
        zoom_mode = self._workspace.zoom_mode()
        if not any(tab.panels for tab in self._workspace.tabs):
            zoom_mode = "both"
        configs: list[AnalysisConfig] = []
        recordings: list[Any] = []
        for entry in entries:
            configs.append(
                replace(
                    self._build_config(entry.path),
                    save_dir=target.parent,
                    pdf_title=target.stem,
                    recording_label=entry.label or None,
                    recording_style=entry.style,
                    plot_display=display,
                    zoom_mode=zoom_mode,
                )
            )
            if entry.recording is not None and getattr(entry.recording, "source", None) is not None:
                recordings.append(entry.recording)
        callback = self._pdf_callback

        def task() -> str:
            if recordings and len(recordings) == len(configs):
                from plotting import plot_processed_recordings_pdf

                plot_processed_recordings_pdf(
                    recordings, configs, output_dir=target.parent
                )
            elif callback is not None:
                callback(configs)
            else:
                raise RuntimeError("Aucun chemin PDF disponible.")
            return str(target)

        def done(result: Any) -> None:
            self.status_panel.append_log(f"Rapport PDF écrit près de {result}")
            QMessageBox.information(
                self, "Rapport PDF", f"Rapport généré dans :\n{target.parent}"
            )

        self._export.run_pdf_task(task, done)

    def save_view_images(self) -> None:
        self._export.save_view_images()

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
            self._build_worker.wait(250)
        self._stop_ensure_worker(wait_ms=250)
        if self._task_worker is not None and self._task_worker.isRunning():
            self._task_worker.request_stop()
            self._task_worker.wait(250)
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
        # Collecter les racines de cache avant de fermer les enregistrements.
        for entry in self.recordings_panel.entries:
            if entry.is_processed:
                continue
            try:
                self._resolve_cache_root(self._build_config(entry.path))
            except Exception:
                pass
        # Fermer les memmaps d’abord, sinon Windows refuse souvent la suppression.
        self.recordings_panel.clear()
        self._purge_session_caches()
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
