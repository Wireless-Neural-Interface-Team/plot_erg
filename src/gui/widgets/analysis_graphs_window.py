"""Fenêtre dédiée aux graphs d’analyse (hors continuous) : plages + params d’affichage."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.form_widgets import (
    AxisLimitRow,
    FitWidthScrollArea,
    configure_narrow_form,
    make_int_spin,
)
from gui.widgets.panel_canvas import DetachedPanelWindow
from gui.widgets.panel_grid import PanelGrid
from gui.widgets.range_bars import RangeBarToolbar
from gui.widgets.view_params import LocalViewParams
from panel_registry import RenderRequest
from view_config import (
    AnalysisMode,
    AnalysisSettings,
    PanelPlacement,
    ViewerSettings,
    apply_local_display_settings,
)

RequestFactory = Callable[["AnalysisGraphsWindow", PanelPlacement], RenderRequest | None]

_DEFAULT_PANEL_HEIGHT = 300


def _scrollable(inner: QWidget) -> FitWidthScrollArea:
    area = FitWidthScrollArea()
    area.setWidget(inner)
    return area


class AnalysisGraphsWindow(QMainWindow):
    """Graphs d’analyse dans leur propre fenêtre (plages + onglets de paramétrage)."""

    closed = Signal()
    refreshRequested = Signal(object)
    modeChanged = Signal()
    curvesChanged = Signal()  # coches Courbes / Spikes
    addRangeRequested = Signal()
    addRelativeRequested = Signal()
    removeRangeRequested = Signal()
    activeRangeChanged = Signal(int)
    applyZoomsRequested = Signal()
    renderFinished = Signal(int, float)  # drawn, elapsed_s — pour réattacher les barres

    def __init__(
        self,
        *,
        channel_name: str,
        channel_index: int,
        base_settings: ViewerSettings,
        mode_label: str = "",
        n_trials: int = 1,
        panel_height: int = _DEFAULT_PANEL_HEIGHT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.channel_name = str(channel_name)
        self.channel_index = int(channel_index)
        self.settings = replace(base_settings)
        self.placements: tuple[PanelPlacement, ...] = ()
        self._detached: dict[str, DetachedPanelWindow] = {}
        analysis = base_settings.analysis
        self._mode_label = mode_label or analysis.describe()
        self._n_trials = max(1, int(n_trials))
        self._panel_height = max(140, int(panel_height))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle(self._title_text())
        self.resize(1280, 860)

        self._status = QLabel(
            "Fenêtre Analyse — plages déplaçables sur le graph temporel "
            "(clic gauche = barre, clic milieu/droit = pan). "
            "Indépendantes de l’aperçu canal."
        )
        self._status.setObjectName("panelStatus")
        self._status.setWordWrap(True)

        self._title_label = QLabel(
            f"<b>{self.channel_name}</b><br>Analyse — {self._mode_label}"
        )
        self._title_label.setObjectName("sectionTitle")
        self._title_label.setTextFormat(Qt.TextFormat.RichText)

        # ---- Onglet Analyse : mode, courbes, spikes, plages, hauteur ----
        analyse_page = QWidget()
        analyse_layout = QVBoxLayout(analyse_page)
        analyse_layout.setContentsMargins(8, 8, 8, 8)
        analyse_layout.setSpacing(8)
        analyse_layout.addWidget(self._title_label)

        mode_box = QGroupBox("Mode d’analyse", analyse_page)
        mode_form = configure_narrow_form(QFormLayout(mode_box))
        self._mode_combo = QComboBox(self)
        self._mode_combo.addItem("Moyenne d’essais", "average")
        self._mode_combo.addItem("Une stimulation", "stimulation")
        mode_idx = self._mode_combo.findData(analysis.mode)
        self._mode_combo.setCurrentIndex(mode_idx if mode_idx >= 0 else 0)
        self._stim_spin = QSpinBox(self)
        self._stim_spin.setMinimum(1)
        self._stim_spin.setMaximum(self._n_trials)
        self._stim_spin.setValue(max(1, int(analysis.stim_index) + 1))
        self._stim_spin.setKeyboardTracking(False)
        self._stim_spin.setToolTip(
            "Numéro de stimulation à afficher (1 … N). "
            "Référence aussi les plages relatives à la stim."
        )
        mode_form.addRow("Mode", self._mode_combo)
        mode_form.addRow("Stimulation n°", self._stim_spin)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_ui_changed)
        self._stim_spin.valueChanged.connect(self._on_mode_ui_changed)
        self._sync_stim_enabled()
        analyse_layout.addWidget(mode_box)

        curves_box = QGroupBox("Courbes", analyse_page)
        curves_layout = QVBoxLayout(curves_box)
        self._cb_raw = QCheckBox("WIDE (brut)")
        self._cb_hp = QCheckBox("HIGH (passe-haut)")
        self._cb_lp = QCheckBox("LOW (passe-bas)")
        self._cb_rms = QCheckBox("RMS")
        self._cb_isi = QCheckBox("ISI")
        self._cb_overlay = QCheckBox("Spike scope")
        self._cb_raw.setChecked(analysis.show_raw)
        self._cb_hp.setChecked(analysis.show_hp)
        self._cb_lp.setChecked(analysis.show_lp)
        self._cb_rms.setChecked(analysis.show_rms)
        self._cb_isi.setChecked(analysis.show_isi)
        self._cb_overlay.setChecked(analysis.show_overlay)
        for box in (self._cb_raw, self._cb_hp, self._cb_lp):
            curves_layout.addWidget(box)
            box.toggled.connect(self._on_curves_toggled)
        curves_layout.addWidget(self._cb_rms)
        self._cb_rms.toggled.connect(self._on_curves_toggled)
        self._rms_ylim = AxisLimitRow(
            self.settings.rms_ylim, unit=" µV", step=1.0
        )
        self._rms_ylim.setToolTip("Échelle Y des panneaux RMS.")
        rms_scale_wrap = QWidget(curves_box)
        rms_scale_layout = QVBoxLayout(rms_scale_wrap)
        rms_scale_layout.setContentsMargins(18, 0, 0, 4)
        rms_scale_layout.setSpacing(2)
        rms_scale_label = QLabel("Échelle Y :", rms_scale_wrap)
        rms_scale_label.setObjectName("hintLabel")
        rms_scale_layout.addWidget(rms_scale_label)
        rms_scale_layout.addWidget(self._rms_ylim)
        curves_layout.addWidget(rms_scale_wrap)
        self._rms_ylim.changed.connect(self._on_rms_ylim_changed)
        for box in (self._cb_isi, self._cb_overlay):
            curves_layout.addWidget(box)
            box.toggled.connect(self._on_curves_toggled)
        analyse_layout.addWidget(curves_box)

        spikes_box = QGroupBox("Spikes", analyse_page)
        spikes_layout = QVBoxLayout(spikes_box)
        self._cb_psth = QCheckBox("PSTH")
        self._cb_trial_rate = QCheckBox("Firing rate / essai")
        self._cb_raster = QCheckBox("Raster")
        self._cb_psth.setChecked(analysis.show_psth)
        self._cb_trial_rate.setChecked(analysis.show_trial_rate)
        self._cb_raster.setChecked(analysis.show_raster)
        for box in (self._cb_psth, self._cb_trial_rate, self._cb_raster):
            spikes_layout.addWidget(box)
            box.toggled.connect(self._on_curves_toggled)
        analyse_layout.addWidget(spikes_box)

        layout_box = QGroupBox("Affichage des graphs", analyse_page)
        layout_form = configure_narrow_form(QFormLayout(layout_box))
        self._height_spin = make_int_spin(140, 1200, self._panel_height, step=20)
        self._height_spin.setSuffix(" px")
        self._height_spin.setToolTip(
            "Hauteur d’affichage commune à tous les graphs de cette fenêtre "
            "(même largeur et même hauteur)."
        )
        self._height_spin.valueChanged.connect(self._on_height_changed)
        layout_form.addRow("Hauteur des graphs :", self._height_spin)
        analyse_layout.addWidget(layout_box)

        self._range_toolbar = RangeBarToolbar(self)
        self._range_toolbar.addRequested.connect(self.addRangeRequested.emit)
        self._range_toolbar.addRelativeRequested.connect(self.addRelativeRequested.emit)
        self._range_toolbar.removeRequested.connect(self.removeRangeRequested.emit)
        self._range_toolbar.activeChanged.connect(self.activeRangeChanged.emit)
        self._range_toolbar.processRequested.connect(self.applyZoomsRequested.emit)
        analyse_layout.addWidget(self._range_toolbar)

        self._relative_label = QLabel("0 plage(s) relative(s) à la stim", self)
        self._relative_label.setObjectName("hintLabel")
        self._relative_label.setWordWrap(True)
        self._relative_label.setMinimumWidth(0)
        analyse_layout.addWidget(self._relative_label)

        self._btn_redraw = QPushButton("Redessiner", self)
        self._btn_redraw.clicked.connect(lambda: self.refreshRequested.emit(self))
        analyse_layout.addWidget(self._btn_redraw)
        analyse_layout.addWidget(self._status)
        analyse_layout.addStretch(1)

        # ---- Onglets comme le dock Paramètres de la fenêtre principale ----
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("paramsTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(False)
        tab_bar = self.tabs.tabBar()
        tab_bar.setElideMode(Qt.TextElideMode.ElideRight)
        tab_bar.setExpanding(True)

        self.tabs.addTab(_scrollable(analyse_page), "Analyse")
        self.tabs.setTabToolTip(
            0, "Mode, courbes / spikes, plages et hauteur des graphs"
        )

        # Affichage / Style locaux à cette fenêtre (pas de sync continuous ici).
        self.params = LocalViewParams(
            self.settings,
            show_sync=False,
            show_display_extras=True,
            host_tabs=self.tabs,
            parent=self,
        )
        self.params.changed.connect(self._on_params_changed)

        side = QWidget(self)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        side_layout.addWidget(self.tabs, 1)
        side.setMinimumWidth(260)

        self.grid = PanelGrid(self)
        self.grid.removeRequested.connect(self._on_panel_removed)
        self.grid.detachRequested.connect(self._on_panel_detached)
        self.grid.renderFinished.connect(self._on_render_finished)

        header_label = QLabel(
            "Fenêtre Analyse : mode / courbes / spikes = onglet Analyse. "
            "Échelle RMS = sous l’option RMS · autres échelles / bin PSTH = Affichage · légendes = Style. "
            "Plages = barres sur le premier graph temporel. "
            "Molette = zoom · clic milieu/droit = pan.",
            self,
        )
        header_label.setWordWrap(True)
        header = QHBoxLayout()
        header.addWidget(header_label, 1)

        plot_side = QWidget(self)
        plot_layout = QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.addLayout(header)
        plot_layout.addWidget(self.grid, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.addWidget(plot_side)
        splitter.addWidget(side)
        plot_side.setMinimumWidth(360)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([880, 400])
        self.setCentralWidget(splitter)

        self._request_factory: RequestFactory | None = None
        self._preserve_view = False

    def _title_text(self) -> str:
        suffix = f" — {self._mode_label}" if self._mode_label else ""
        return f"Analyse {self.channel_name}{suffix}"

    def _sync_stim_enabled(self) -> None:
        is_stim = str(self._mode_combo.currentData() or "") == "stimulation"
        self._stim_spin.setEnabled(is_stim)

    def mode_from_ui(self) -> tuple[AnalysisMode, int]:
        """Mode et indice de stimulation (0-based) choisis dans cette fenêtre."""
        mode = str(self._mode_combo.currentData() or "average")
        if mode not in ("average", "stimulation"):
            mode = "average"
        stim_index = max(0, int(self._stim_spin.value()) - 1)
        return mode, stim_index  # type: ignore[return-value]

    def analysis_from_ui(self, base: AnalysisSettings | None = None) -> AnalysisSettings:
        """``AnalysisSettings`` avec mode / stim / courbes / spikes issus de l’UI."""
        seed = base if base is not None else self.settings.analysis
        mode, stim_index = self.mode_from_ui()
        return replace(
            seed,
            mode=mode,
            stim_index=stim_index,
            show_raw=self._cb_raw.isChecked(),
            show_hp=self._cb_hp.isChecked(),
            show_lp=self._cb_lp.isChecked(),
            show_rms=self._cb_rms.isChecked(),
            show_isi=self._cb_isi.isChecked(),
            show_overlay=self._cb_overlay.isChecked(),
            show_psth=self._cb_psth.isChecked(),
            show_trial_rate=self._cb_trial_rate.isChecked(),
            show_raster=self._cb_raster.isChecked(),
        )

    def _on_curves_toggled(self, *_args: Any) -> None:
        analysis = self.analysis_from_ui()
        self.settings = replace(self.settings, analysis=analysis)
        self.curvesChanged.emit()

    def _checkbox_for_panel(self, panel: str) -> QCheckBox | None:
        mapping = {
            "analysis_raw": self._cb_raw,
            "analysis_hp": self._cb_hp,
            "analysis_lp": self._cb_lp,
            "analysis_rms": self._cb_rms,
            "analysis_isi": self._cb_isi,
            "analysis_overlay": self._cb_overlay,
            "analysis_psth": self._cb_psth,
            "analysis_trial_rate": self._cb_trial_rate,
            "analysis_raster": self._cb_raster,
        }
        return mapping.get(str(panel))

    def set_trial_count(self, n_trials: int) -> None:
        """Borne le sélecteur de stimulation à 1 … N."""
        self._n_trials = max(1, int(n_trials))
        self._stim_spin.blockSignals(True)
        self._stim_spin.setMaximum(self._n_trials)
        if self._stim_spin.value() > self._n_trials:
            self._stim_spin.setValue(self._n_trials)
        self._stim_spin.blockSignals(False)
        self._sync_stim_enabled()

    def set_mode_label(self, label: str) -> None:
        self._mode_label = str(label or "")
        self._title_label.setText(
            f"<b>{self.channel_name}</b><br>Analyse — {self._mode_label}"
        )
        self.setWindowTitle(self._title_text())

    def rebind_channel(self, *, channel_name: str, channel_index: int) -> None:
        """Mettre à jour le canal affiché sans fermer la fenêtre."""
        self.channel_name = str(channel_name)
        self.channel_index = int(channel_index)
        self._title_label.setText(
            f"<b>{self.channel_name}</b><br>Analyse — {self._mode_label}"
        )
        self.setWindowTitle(self._title_text())
        for win in self._detached.values():
            win.setWindowTitle(f"{self.channel_name} — {win.placement.title()}")

    def _on_mode_ui_changed(self, *_args: Any) -> None:
        self._sync_stim_enabled()
        analysis = self.analysis_from_ui()
        self._mode_label = analysis.describe()
        self._title_label.setText(
            f"<b>{self.channel_name}</b><br>Analyse — {self._mode_label}"
        )
        self.setWindowTitle(self._title_text())
        self.settings = replace(self.settings, analysis=analysis)
        self.modeChanged.emit()

    def _on_height_changed(self, value: int) -> None:
        self._panel_height = max(140, int(value))
        if self.placements:
            self.grid.set_panel_height(self._panel_height, uniform=True)

    def set_request_factory(self, factory: RequestFactory) -> None:
        self._request_factory = factory

    def set_range_counts(self, n_abs: int, active_index: int, relative_text: str) -> None:
        self._range_toolbar.set_bar_count(n_abs, active_index)
        self._relative_label.setText(relative_text)

    def local_settings(self) -> ViewerSettings:
        base = self.params.settings()
        return replace(
            base,
            analysis=self.analysis_from_ui(base.analysis),
            rms_ylim=self._rms_ylim.value(),
        )

    def sync_base_settings(self, settings: ViewerSettings) -> None:
        """Aligner les réglages de base tout en gardant style local et mode UI."""
        # Conserver le mode / les coches choisis ici : ne pas écraser depuis le parent.
        local_analysis = self.analysis_from_ui(settings.analysis)
        merged = replace(
            settings,
            analysis=local_analysis,
            rms_ylim=self._rms_ylim.value(),
        )
        self.params.sync_base_settings(merged)
        self.settings = self.local_settings()

    def _on_rms_ylim_changed(self) -> None:
        self.settings = self.local_settings()
        self.refreshRequested.emit(self)

    @property
    def panel_height(self) -> int:
        return self._panel_height

    def configure_placements(
        self,
        placements: Sequence[PanelPlacement],
        *,
        panel_height: int | None = None,
        columns: int = 1,
    ) -> None:
        self.placements = tuple(placements)
        if panel_height is not None:
            self._panel_height = max(140, int(panel_height))
            self._height_spin.blockSignals(True)
            self._height_spin.setValue(self._panel_height)
            self._height_spin.blockSignals(False)
        height = self._panel_height
        if not self.placements:
            self.grid.configure(
                (), columns=columns, panel_height=height, uniform=True
            )
            self._status.setText("Aucun graph d’analyse coché.")
            return
        self.grid.configure(
            self.placements,
            columns=int(columns),
            panel_height=int(height),
            uniform=True,
        )

    def _on_params_changed(self) -> None:
        self.settings = self.local_settings()
        self.refreshRequested.emit(self)

    def _on_panel_removed(self, placement: PanelPlacement) -> None:
        box = self._checkbox_for_panel(placement.panel)
        if box is not None and box.isChecked() and not placement.has_custom_zoom:
            box.blockSignals(True)
            box.setChecked(False)
            box.blockSignals(False)
            analysis = self.analysis_from_ui()
            self.settings = replace(self.settings, analysis=analysis)
            self.curvesChanged.emit()
            return
        self.placements = tuple(p for p in self.placements if p.key != placement.key)
        self.grid.configure(
            self.placements,
            columns=1,
            panel_height=self._panel_height,
            uniform=True,
        )
        self.refreshRequested.emit(self)

    def _on_panel_detached(self, placement: PanelPlacement) -> None:
        """Ouvrir un graph d’analyse dans sa propre fenêtre (bouton ⤢)."""
        key = placement.key
        existing = self._detached.get(key)
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return
        win = DetachedPanelWindow(placement, self.local_settings(), parent=self)
        win.setWindowTitle(f"{self.channel_name} — {placement.title()}")
        win.closed.connect(self._on_detached_closed)
        win.refreshRequested.connect(self._on_detached_refresh)
        self._detached[key] = win
        win.show()
        win.raise_()
        self._render_detached(win)

    def _on_detached_closed(self, placement: PanelPlacement) -> None:
        self._detached.pop(placement.key, None)

    def _on_detached_refresh(self, window: DetachedPanelWindow) -> None:
        self._render_detached(window)

    def _render_detached(self, window: DetachedPanelWindow) -> None:
        if self._request_factory is None:
            return
        request = self._request_factory(self, window.placement)
        if request is None:
            return
        # Axes / style / sync de la fenêtre détachée, pas de la fenêtre Analyse.
        settings = apply_local_display_settings(
            request.settings, window.local_settings()
        )
        window.render(
            replace(request, settings=settings, preserve_view=bool(self._preserve_view))
        )

    def _close_all_detached(self) -> None:
        for key in list(self._detached):
            win = self._detached.pop(key)
            try:
                win.closed.disconnect(self._on_detached_closed)
            except (TypeError, RuntimeError):
                pass
            try:
                win.refreshRequested.disconnect(self._on_detached_refresh)
            except (TypeError, RuntimeError):
                pass
            win.close()

    def redraw(self, *, preserve_view: bool = False) -> None:
        self.settings = self.local_settings()
        if self._request_factory is None:
            return
        self._preserve_view = bool(preserve_view)

        def factory(placement: PanelPlacement) -> RenderRequest:
            preserve = bool(self._preserve_view) and not placement.has_custom_zoom
            request = self._request_factory(self, placement)  # type: ignore[misc]
            if request is not None:
                if preserve:
                    return replace(request, preserve_view=True)
                return (
                    replace(request, preserve_view=False)
                    if self._preserve_view
                    else request
                )
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
                preserve_view=preserve,
            )

        self._status.setText("Rendu…")
        self.grid.schedule_render(factory, force=True)
        for detached in list(self._detached.values()):
            self._render_detached(detached)

    def _on_render_finished(self, drawn: int, elapsed_s: float) -> None:
        self._preserve_view = False
        self._status.setText(f"{drawn} panneau(x) · {elapsed_s * 1000:.0f} ms")
        self.renderFinished.emit(drawn, elapsed_s)

    def panel_widget(self, placement: PanelPlacement):
        """Widget panneau de la grille (pour attacher les barres de plage)."""
        return self.grid.panel_widget(placement)

    def closeEvent(self, event: Any) -> None:  # noqa: D102
        self._close_all_detached()
        self.closed.emit()
        super().closeEvent(event)
