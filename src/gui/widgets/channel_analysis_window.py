"""Vue canal (aperçu / inspection) : continuous, moyenne ou stimulation.

Embarquée dans la fenêtre principale en mode aperçu, ou flottante.
Le mode d’affichage (continuous / moyenne / stimulation) se choisit dans
Paramètres → Canal ; les graphs d’analyse s’affichent dans la même grille.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import replace
from typing import Any, Callable, Literal, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.form_widgets import AxisLimitRow, FitWidthScrollArea, configure_narrow_form
from gui.jobs import Debouncer
from gui.widgets.custom_zoom_dialog import ask_custom_zoom
from gui.widgets.panel_canvas import DetachedPanelWindow
from gui.widgets.panel_grid import PanelGrid
from gui.widgets.range_bars import RangeBarController, RangeBarToolbar
from gui.widgets.view_params import LocalViewParams
from panel_registry import RenderRequest
from view_config import (
    AnalysisMode,
    AnalysisSettings,
    AnalysisStream,
    PanelPlacement,
    TimeRangeBar,
    ViewerSettings,
    apply_local_display_settings,
    continuous_sync_offset_s,
)

_WINDOW_COUNTER = itertools.count(1)

PreviewDisplayMode = Literal["continuous", "average", "stimulation"]
RequestFactory = Callable[["ChannelAnalysisWindow", PanelPlacement], RenderRequest | None]
# Hauteur mini d’un panneau d’aperçu (fill viewport si plus grand).
_PREVIEW_PANEL_MIN_HEIGHT = 260

# Panneaux temporels où les barres Analyse peuvent être glissées (même UX que continuous).
_ANALYSIS_TEMPORAL_PANELS = frozenset(
    {
        "analysis_raw",
        "analysis_hp",
        "analysis_lp",
        "analysis_rms",
        "mean_raw",
        "mean_hp",
        "mean_lp",
        "mean_rms",
    }
)


class ChannelAnalysisWindow(QWidget):
    """Vue canal unifiée : continuous, moyenne d’essais ou une stimulation.

    En mode ``embedded`` : intégrée à l’aperçu de la fenêtre principale
    (flux WIDE/HIGH/LOW pilotés par Paramètres → Affichage). Sinon : fenêtre flottante.
    """

    closed = Signal(str)
    refreshRequested = Signal(object)
    previewModeChanged = Signal(str)

    def __init__(
        self,
        *,
        channel_name: str,
        channel_index: int,
        analysis: AnalysisSettings,
        base_settings: ViewerSettings,
        parent: QWidget | None = None,
        embedded: bool = False,
    ) -> None:
        super().__init__(parent)
        self._embedded = bool(embedded)
        self.window_id = f"chan-{next(_WINDOW_COUNTER)}-{uuid.uuid4().hex[:6]}"
        self.channel_name = str(channel_name)
        self.channel_index = int(channel_index)
        self.settings = replace(base_settings, analysis=analysis)
        # Zooms / plages : continuous et analyse gardent leurs listes séparées.
        self._continuous_zooms_applied = False
        self._analysis_zooms_applied = False
        self._stream_updating = False
        self._mode_updating = False
        self._stim_times_s: tuple[float, ...] = ()
        self._relative_ranges: list[TimeRangeBar] = []
        self._analysis_ranges: list[TimeRangeBar] = []
        self._analysis_active_index = 0
        self._zoom_placements: tuple[PanelPlacement, ...] = ()
        self._analysis_placements: tuple[PanelPlacement, ...] = ()
        self._zoom_windows: dict[str, DetachedPanelWindow] = {}
        self._detached: dict[str, DetachedPanelWindow] = {}
        self._user_closed_zooms: set[str] = set()
        self._placements: tuple[PanelPlacement, ...] = ()
        self._preview_mode: PreviewDisplayMode = "continuous"
        if self._embedded:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            self.setWindowFlag(Qt.WindowType.Window, True)
            self.setWindowTitle(f"Canal {self.channel_name}")
            self.resize(1400, 900)

        self._status = QLabel("")
        self._status.setObjectName("panelStatus")
        self._status.setWordWrap(True)
        self._n_trials = max(1, int(analysis.stim_index) + 1)
        self._title_label = QLabel("")
        self._title_label.setObjectName("sectionTitle")
        self._title_label.setTextFormat(Qt.TextFormat.RichText)
        self._channel_ready = False

        # --- Mode d’affichage (continuous / moyenne / stimulation) ---
        mode_box = QGroupBox("Mode d’affichage", self)
        mode_form = configure_narrow_form(QFormLayout(mode_box))
        self._mode_combo = QComboBox(self)
        self._mode_combo.addItem("Continuous", "continuous")
        self._mode_combo.addItem("Moyenne d’essais", "average")
        self._mode_combo.addItem("Une stimulation", "stimulation")
        self._mode_combo.setCurrentIndex(0)
        self._mode_combo.setToolTip(
            "Continuous = traces brutes sur tout l’enregistrement. "
            "Moyenne / stimulation = graphs d’analyse dans la même vue."
        )
        self._stim_spin = QSpinBox(self)
        self._stim_spin.setMinimum(1)
        self._stim_spin.setMaximum(self._n_trials)
        self._stim_spin.setValue(max(1, int(analysis.stim_index) + 1))
        self._stim_spin.setKeyboardTracking(False)
        self._stim_spin.setToolTip("Numéro de stimulation à afficher (1 … N).")
        mode_form.addRow("Afficher", self._mode_combo)
        mode_form.addRow("Stimulation n°", self._stim_spin)
        self._mode_combo.currentIndexChanged.connect(self._on_preview_mode_ui_changed)
        self._stim_spin.valueChanged.connect(self._on_stim_index_changed)

        # --- flux continus ---
        # Embarqué : pilotés par Paramètres → Affichage (masqués ici).
        self._streams_box = QGroupBox("Traces continues", self)
        streams_layout = QVBoxLayout(self._streams_box)
        seed_streams = set(base_settings.resolved_continuous_streams())
        self._cb_stream_wide = QCheckBox("WIDE (brut)")
        self._cb_stream_high = QCheckBox("HIGH (passe-haut)")
        self._cb_stream_low = QCheckBox("LOW (passe-bas)")
        self._cb_stream_wide.setChecked("raw" in seed_streams)
        self._cb_stream_high.setChecked("hp" in seed_streams)
        self._cb_stream_low.setChecked("lp" in seed_streams)
        if not any(
            box.isChecked()
            for box in (self._cb_stream_wide, self._cb_stream_high, self._cb_stream_low)
        ):
            self._cb_stream_wide.setChecked(True)
        self._cb_mark_stims = QCheckBox("Marqueurs de stimulation")
        self._cb_mark_stims.setChecked(bool(base_settings.continuous_mark_stims))
        for box in (self._cb_stream_wide, self._cb_stream_high, self._cb_stream_low):
            streams_layout.addWidget(box)
            box.toggled.connect(self._on_streams_changed)
        streams_layout.addWidget(self._cb_mark_stims)
        self._cb_mark_stims.toggled.connect(self._on_stim_markers_changed)
        self._preserve_view = False
        if self._embedded:
            self._streams_box.hide()

        # --- Courbes / spikes (modes moyenne & stimulation uniquement) ---
        self._curves_box = QGroupBox("Courbes", self)
        curves_layout = QVBoxLayout(self._curves_box)
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
            box.toggled.connect(self._on_analysis_curves_toggled)
        curves_layout.addWidget(self._cb_rms)
        self._cb_rms.toggled.connect(self._on_analysis_curves_toggled)
        # Échelle Y RMS juste sous l’option RMS (pas dans Affichage → axes).
        self._rms_ylim = AxisLimitRow(
            base_settings.rms_ylim, unit=" µV", step=1.0
        )
        self._rms_ylim.setToolTip(
            "Échelle Y des panneaux et résumés RMS."
        )
        rms_scale_wrap = QWidget(self._curves_box)
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
            box.toggled.connect(self._on_analysis_curves_toggled)

        self._spikes_box = QGroupBox("Spikes", self)
        spikes_layout = QVBoxLayout(self._spikes_box)
        self._cb_psth = QCheckBox("PSTH")
        self._cb_trial_rate = QCheckBox("Firing rate / essai")
        self._cb_raster = QCheckBox("Raster")
        self._cb_psth.setChecked(analysis.show_psth)
        self._cb_trial_rate.setChecked(analysis.show_trial_rate)
        self._cb_raster.setChecked(analysis.show_raster)
        for box in (self._cb_psth, self._cb_trial_rate, self._cb_raster):
            spikes_layout.addWidget(box)
            box.toggled.connect(self._on_analysis_curves_toggled)

        # --- contexte / résumés (surtout continuous) ---
        self._context_box = QGroupBox("Contexte", self)
        context_layout = QVBoxLayout(self._context_box)
        self._cb_impedance = QCheckBox("Impédance")
        context_layout.addWidget(self._cb_impedance)
        self._cb_impedance.toggled.connect(self._on_extra_toggles_changed)

        self._summary_box = QGroupBox("Résumés", self)
        summary_layout = QVBoxLayout(self._summary_box)
        self._cb_summary_rms = QCheckBox("Mean RMS / enregistrement")
        self._cb_summary_rms_table = QCheckBox("Mean RMS / canal")
        self._cb_summary_rms.setToolTip(
            "Profil RMS moyen sur tous les canaux — une courbe par enregistrement."
        )
        self._cb_summary_rms_table.setToolTip(
            "Table du RMS moyen par canal (une colonne par enregistrement)."
        )
        self._cb_summary_rms.setChecked(analysis.show_summary_rms)
        self._cb_summary_rms_table.setChecked(analysis.show_summary_rms_table)
        for box in (self._cb_summary_rms, self._cb_summary_rms_table):
            summary_layout.addWidget(box)
            box.toggled.connect(self._on_toggles_changed)

        self._range_toolbar = RangeBarToolbar(self)
        self._range_ctrl = RangeBarController(self)
        self._range_ctrl.barsChanged.connect(self._on_bars_changed)
        self._analysis_range_ctrl = RangeBarController(self)
        self._analysis_range_ctrl.barsChanged.connect(self._on_analysis_bars_changed)
        self._relative_label = QLabel("0 plage(s) relative(s) à la stim", self)
        self._relative_label.setObjectName("hintLabel")
        self._relative_label.setWordWrap(True)
        self._relative_label.setMinimumWidth(0)

        self._btn_redraw = QPushButton("Redessiner", self)
        self._btn_redraw.clicked.connect(lambda: self.refreshRequested.emit(self))

        # Embarqué : légende / style / axes X·traces·HP = dock Paramètres.
        # Échelle RMS = sous la case RMS (ci-dessus).
        self.params: LocalViewParams | None
        if self._embedded:
            self.params = None
        else:
            self.params = LocalViewParams(
                self.settings,
                show_sync=True,
                show_display_extras=False,
                parent=self,
            )
            self.params.changed.connect(self._on_local_params_changed)

        side_inner = QWidget()
        side_inner_layout = QVBoxLayout(side_inner)
        side_inner_layout.setContentsMargins(6, 6, 6, 6)
        side_inner_layout.setSpacing(6)
        side_inner_layout.addWidget(self._title_label)
        side_inner_layout.addWidget(mode_box)
        side_inner_layout.addWidget(self._range_toolbar)
        side_inner_layout.addWidget(self._relative_label)
        side_inner_layout.addWidget(self._streams_box)
        side_inner_layout.addWidget(self._curves_box)
        side_inner_layout.addWidget(self._spikes_box)
        side_inner_layout.addWidget(self._context_box)
        side_inner_layout.addWidget(self._summary_box)
        side_inner_layout.addWidget(self._btn_redraw)
        if self.params is not None:
            side_inner_layout.addWidget(self.params, 1)
        else:
            side_inner_layout.addStretch(1)
        side_inner_layout.addWidget(self._status)

        self.side_panel = FitWidthScrollArea()
        self.side_panel.setWidget(side_inner)
        self.side_panel.setWidgetResizable(True)
        if self._embedded:
            self.side_panel.setMinimumWidth(0)
        else:
            self.side_panel.setMinimumWidth(220)

        self.grid = PanelGrid(self, allow_zoom=not self._embedded)
        self.grid.removeRequested.connect(self._on_panel_removed)
        self.grid.detachRequested.connect(self._on_panel_detached)
        self.grid.renderFinished.connect(self._on_render_finished)

        self._header_label = QLabel("", self)
        self._header_label.setWordWrap(True)
        header = QHBoxLayout()
        header.addWidget(self._header_label, 1)

        plot_side = QWidget(self)
        plot_layout = QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.addLayout(header)
        plot_layout.addWidget(self.grid, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if self._embedded:
            root.addWidget(plot_side)
        else:
            splitter = QSplitter(Qt.Orientation.Horizontal, self)
            splitter.setChildrenCollapsible(False)
            splitter.setHandleWidth(6)
            splitter.addWidget(plot_side)
            splitter.addWidget(self.side_panel)
            plot_side.setMinimumWidth(360)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([900, 400])
            root.addWidget(splitter)

        self._request_factory: RequestFactory | None = None
        self._bars_debouncer = Debouncer(80, self)
        self._bars_debouncer.triggered.connect(self._after_bars_moved)
        self._analysis_bars_debouncer = Debouncer(80, self)
        self._analysis_bars_debouncer.triggered.connect(self._after_analysis_bars_moved)
        self._wire_range_toolbar()
        self._apply_preview_mode_ui()
        self._rebuild_placements()

    # ---------------------------------------------------------------- wiring

    def set_request_factory(self, factory: RequestFactory) -> None:
        self._request_factory = factory

    def preview_mode(self) -> PreviewDisplayMode:
        return self._preview_mode

    def is_analysis_view(self) -> bool:
        return self._preview_mode in ("average", "stimulation")

    def set_preview_mode(
        self,
        mode: PreviewDisplayMode | AnalysisMode | str,
        *,
        redraw: bool = True,
    ) -> None:
        """Basculer continuous / moyenne / stimulation dans la même vue."""
        resolved: PreviewDisplayMode
        raw = str(mode or "continuous")
        if raw in ("continuous", "average", "stimulation"):
            resolved = raw  # type: ignore[assignment]
        else:
            resolved = "continuous"
        if resolved == self._preview_mode and not self._mode_updating:
            if redraw:
                self.refreshRequested.emit(self)
            return
        self._mode_updating = True
        idx = self._mode_combo.findData(resolved)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        self._mode_updating = False
        self._preview_mode = resolved
        if resolved in ("average", "stimulation"):
            analysis = self._analysis_from_toggles()
            analysis = replace(analysis, mode=resolved)  # type: ignore[arg-type]
            if resolved == "stimulation":
                analysis = replace(
                    analysis,
                    stim_index=max(0, int(self._stim_spin.value()) - 1),
                )
            if not analysis.selected_analysis_panels():
                self._cb_raw.setChecked(True)
                self._cb_hp.setChecked(True)
                self._cb_lp.setChecked(True)
                self._cb_rms.setChecked(True)
                analysis = self._analysis_from_toggles()
                analysis = replace(analysis, mode=resolved)  # type: ignore[arg-type]
            self.settings = replace(
                self.settings,
                analysis=analysis,
                preview_content=resolved,
            )
        else:
            self.settings = replace(self.settings, preview_content="continuous")
        self._wire_range_toolbar()
        self._apply_preview_mode_ui()
        self._rebuild_placements()
        self.previewModeChanged.emit(self._preview_mode)
        if redraw:
            self.refreshRequested.emit(self)

    def set_trial_count(self, n_trials: int) -> None:
        """Borne le sélecteur de stimulation à 1 … N."""
        self._n_trials = max(1, int(n_trials))
        self._stim_spin.blockSignals(True)
        self._stim_spin.setMaximum(self._n_trials)
        if self._stim_spin.value() > self._n_trials:
            self._stim_spin.setValue(self._n_trials)
        self._stim_spin.blockSignals(False)
        self._sync_stim_enabled()

    def set_stim_times(self, stim_times_s: Sequence[float]) -> None:
        """Temps d’onset des stimulations (s, absolus) pour placer les plages relatives."""
        self._stim_times_s = tuple(float(t) for t in stim_times_s)

    def apply_control_streams(
        self,
        streams: Sequence[AnalysisStream],
        *,
        mark_stims: bool,
        redraw: bool = True,
    ) -> None:
        """Synchroniser les flux depuis Paramètres → Affichage (mode embarqué)."""
        wanted = set(streams) or {"raw"}
        self._stream_updating = True
        self._cb_stream_wide.setChecked("raw" in wanted)
        self._cb_stream_high.setChecked("hp" in wanted)
        self._cb_stream_low.setChecked("lp" in wanted)
        if not any(
            box.isChecked()
            for box in (self._cb_stream_wide, self._cb_stream_high, self._cb_stream_low)
        ):
            self._cb_stream_wide.setChecked(True)
        self._cb_mark_stims.setChecked(bool(mark_stims))
        self._stream_updating = False
        self._preserve_view = False
        self.settings = self.local_settings()
        self._sync_params_base()
        self._rebuild_placements()
        if redraw:
            self.refreshRequested.emit(self)

    def apply_viewer_settings(self, settings: ViewerSettings) -> None:
        """Pousser les réglages du dock Paramètres (mode embarqué)."""
        analysis = self._analysis_from_toggles()
        bars = self._all_range_bars()
        active = self._range_ctrl.active_index
        if self._embedded:
            # Flux continuous = Paramètres → Affichage.
            # Mode / courbes d’analyse = onglet Canal (locaux).
            streams = settings.resolved_continuous_streams() or ("raw",)
            mark = bool(settings.continuous_mark_stims)
            self._stream_updating = True
            self._cb_stream_wide.setChecked("raw" in streams)
            self._cb_stream_high.setChecked("hp" in streams)
            self._cb_stream_low.setChecked("lp" in streams)
            if not any(
                box.isChecked()
                for box in (
                    self._cb_stream_wide,
                    self._cb_stream_high,
                    self._cb_stream_low,
                )
            ):
                self._cb_stream_wide.setChecked(True)
                streams = ("raw",)
            self._cb_mark_stims.setChecked(mark)
            self._stream_updating = False
        else:
            streams = self._selected_streams()
            mark = self._cb_mark_stims.isChecked()
        self.settings = replace(
            settings,
            analysis=analysis,
            preview_content=self._preview_mode,
            continuous_stream=streams[0],
            continuous_streams=streams,
            continuous_mark_stims=mark,
            range_bars=bars,
            active_range_index=active,
            # Échelle RMS = contrôle local (sous la case RMS), pas le dock Affichage.
            rms_ylim=self._rms_ylim.value(),
        )
        self._sync_params_base()

    def _sync_params_base(self) -> None:
        if self.params is not None:
            self.params.sync_base_settings(self.settings)

    def rebind_channel(
        self,
        *,
        channel_name: str,
        channel_index: int,
        reset_ranges: bool = True,
    ) -> None:
        """Changer de canal sans recréer la vue (conserve mode et coches)."""
        name = str(channel_name)
        index = int(channel_index)
        same = name == self.channel_name and index == self.channel_index
        self.channel_name = name
        self.channel_index = index
        self._update_titles()
        if not self._embedded:
            self.setWindowTitle(f"Canal {self.channel_name}")
        if same:
            return
        self._close_all_zoom_windows()
        self._close_all_detached()
        self._continuous_zooms_applied = False
        self._analysis_zooms_applied = False
        self._channel_ready = False
        if reset_ranges:
            self._range_ctrl.set_bars(())
            self._relative_ranges.clear()
            self._analysis_ranges.clear()
            self._analysis_active_index = 0
            self._range_ctrl.ensure_default_bar()
            self._sync_settings_from_bars()
            self._update_range_counts()
        self._rebuild_placements()

    def set_time_span(self, t_min: float, t_max: float) -> None:
        self._range_ctrl.set_time_span(t_min, t_max)
        if not self._range_ctrl.bars:
            if self.settings.range_bars:
                abs_bars = [
                    b for b in self.settings.range_bars if not b.relative_to_stim
                ]
                rel_bars = [b for b in self.settings.range_bars if b.relative_to_stim]
                if abs_bars:
                    self._range_ctrl.set_bars(
                        abs_bars,
                        active_index=self.settings.active_range_index,
                    )
                else:
                    self._range_ctrl.ensure_default_bar()
                self._relative_ranges = [replace(b) for b in rel_bars]
            else:
                self._range_ctrl.ensure_default_bar()
            self._sync_settings_from_bars()
            self._update_range_counts()

    def _selected_streams(self) -> tuple[AnalysisStream, ...]:
        streams: list[AnalysisStream] = []
        if self._cb_stream_wide.isChecked():
            streams.append("raw")
        if self._cb_stream_high.isChecked():
            streams.append("hp")
        if self._cb_stream_low.isChecked():
            streams.append("lp")
        return tuple(streams) or ("raw",)

    def _all_range_bars(self) -> tuple[TimeRangeBar, ...]:
        return tuple(self._range_ctrl.bars) + tuple(self._relative_ranges)

    def local_settings(self) -> ViewerSettings:
        streams = self._selected_streams()
        # Embarqué : base = dock Paramètres (déjà dans self.settings).
        # Flottant : base = panneau local.
        base = self.params.settings() if self.params is not None else self.settings
        if self.is_analysis_view():
            bars = tuple(self._analysis_ranges)
            active = self._analysis_active_index
        else:
            bars = self._all_range_bars()
            active = self._range_ctrl.active_index
        return replace(
            base,
            analysis=self._analysis_from_toggles(),
            preview_content=self._preview_mode,
            continuous_stream=streams[0],
            continuous_streams=streams,
            continuous_mark_stims=self._cb_mark_stims.isChecked(),
            range_bars=bars,
            active_range_index=active,
            rms_ylim=self._rms_ylim.value(),
        )

    def _on_rms_ylim_changed(self) -> None:
        """Échelle Y RMS : redessiner en gardant zoom/pan."""
        self._preserve_view = True
        self.settings = self.local_settings()
        self._sync_params_base()
        self.refreshRequested.emit(self)

    def _on_local_params_changed(self) -> None:
        self.settings = self.local_settings()
        self.refreshRequested.emit(self)

    def _on_streams_changed(self, *_args: Any) -> None:
        if self._stream_updating:
            return
        if not any(
            box.isChecked()
            for box in (self._cb_stream_wide, self._cb_stream_high, self._cb_stream_low)
        ):
            self._stream_updating = True
            self._cb_stream_wide.setChecked(True)
            self._stream_updating = False
        self._preserve_view = False
        self.settings = self.local_settings()
        self._sync_params_base()
        # Recréer la grille : un axe par flux coché.
        self._rebuild_placements()
        self.refreshRequested.emit(self)

    def _on_stim_markers_changed(self, *_args: Any) -> None:
        """Bascule des marqueurs stim : redessiner sans reset zoom/position."""
        self._preserve_view = True
        self.settings = self.local_settings()
        self._sync_params_base()
        self.refreshRequested.emit(self)

    def needs_channel_compute(self) -> bool:
        return not self._channel_ready

    def set_channel_ready(self, ready: bool) -> None:
        self._channel_ready = bool(ready)

    @property
    def placements(self) -> tuple[PanelPlacement, ...]:
        # En mode analyse, les graphs sont déjà dans ``_placements`` (pas de doublon).
        if self.is_analysis_view():
            return self._placements
        return self._placements + self._zoom_placements

    # ----------------------------------------------------------- placements

    def _summary_from_ui(self) -> tuple[bool, bool]:
        return (
            self._cb_summary_rms.isChecked(),
            self._cb_summary_rms_table.isChecked(),
        )

    def _analysis_from_toggles(self) -> AnalysisSettings:
        """Mode / courbes / spikes / résumés depuis l’UI locale."""
        show_summary_rms, show_summary_rms_table = self._summary_from_ui()
        mode: AnalysisMode = "average"
        if self._preview_mode == "stimulation":
            mode = "stimulation"
        elif self._preview_mode == "average":
            mode = "average"
        else:
            mode = self.settings.analysis.mode
        stim_index = max(0, int(self._stim_spin.value()) - 1)
        return replace(
            self.settings.analysis,
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
            show_summary_rms=show_summary_rms,
            show_summary_rms_table=show_summary_rms_table,
        )

    def _on_preview_mode_ui_changed(self, *_args: Any) -> None:
        if self._mode_updating:
            return
        data = str(self._mode_combo.currentData() or "continuous")
        self.set_preview_mode(data, redraw=True)

    def _on_stim_index_changed(self, *_args: Any) -> None:
        if self._mode_updating or self._preview_mode != "stimulation":
            return
        self.settings = self.local_settings()
        self._sync_params_base()
        self._update_titles()
        if self._any_zooms_applied():
            self._rebuild_placements()
        self.refreshRequested.emit(self)

    def _on_analysis_curves_toggled(self, *_args: Any) -> None:
        """Coches LOW / HIGH / RMS… → graphs correspondants dans l’aperçu."""
        if not self.is_analysis_view():
            return
        analysis = self._analysis_from_toggles()
        if not analysis.selected_analysis_panels():
            self._cb_raw.blockSignals(True)
            self._cb_raw.setChecked(True)
            self._cb_raw.blockSignals(False)
            analysis = self._analysis_from_toggles()
        self._preserve_view = False
        self.settings = replace(self.settings, analysis=analysis)
        self._sync_params_base()
        self._rebuild_placements()
        self.refreshRequested.emit(self)

    def _sync_stim_enabled(self) -> None:
        self._stim_spin.setEnabled(self._preview_mode == "stimulation")

    def _update_titles(self) -> None:
        role = "Aperçu" if self._embedded else "Canal"
        if self._preview_mode == "continuous":
            subtitle = f"{role} — continuous"
        elif self._preview_mode == "stimulation":
            subtitle = f"{role} — stim. n°{int(self._stim_spin.value())}"
        else:
            subtitle = f"{role} — moyenne d’essais"
        self._title_label.setText(f"<b>{self.channel_name}</b><br>{subtitle}")

    def _apply_preview_mode_ui(self) -> None:
        """Montrer / masquer les groupes selon continuous vs analyse."""
        analysis = self.is_analysis_view()
        self._sync_stim_enabled()
        # Courbes moyennées / spikes : uniquement moyenne et stimulation.
        self._curves_box.setVisible(analysis)
        self._spikes_box.setVisible(analysis)
        self._context_box.setVisible(not analysis)
        self._summary_box.setVisible(not analysis)
        if not self._embedded:
            self._streams_box.setVisible(not analysis)
        self._update_titles()
        if analysis:
            self._header_label.setText(
                "Analyse — barres = clic gauche · pan = clic milieu/droit. "
                "Mode / courbes / spikes → onglet Canal."
            )
            self._status.setText(
                "Mode moyenne ou stimulation : graphs dans l’aperçu. "
                "Plages locales à cette vue."
            )
        else:
            self._header_label.setText(
                "Continuous — barres = clic gauche · pan = clic milieu/droit. "
                "Plages / contexte / résumés → onglet Canal."
            )
            self._status.setText(
                "Plages = barres sur le brut. "
                "« + Rel. stim » = [t₀, t₁] relatifs à la stim."
            )

    def _wire_range_toolbar(self) -> None:
        """Connecter la toolbar plages au mode actif (continuous ou analyse)."""
        toolbar = self._range_toolbar
        scope = "analysis" if self.is_analysis_view() else "continuous"
        prev = getattr(self, "_range_toolbar_scope", None)
        if prev == scope:
            if scope == "analysis":
                self._push_range_counts_to_analysis()
            else:
                self._update_range_counts()
            return

        pairs = (
            (toolbar.addRequested, self._add_bar_analysis, self._add_bar_continuous),
            (
                toolbar.addRelativeRequested,
                self._add_relative_range_analysis,
                self._add_relative_range_continuous,
            ),
            (
                toolbar.removeRequested,
                self._remove_bar_analysis,
                self._remove_bar_continuous,
            ),
            (
                toolbar.activeChanged,
                self._on_analysis_active_changed,
                self._on_active_changed,
            ),
            (
                toolbar.processRequested,
                self._apply_zooms_analysis,
                self._apply_zooms_continuous,
            ),
        )
        for signal, analysis_slot, continuous_slot in pairs:
            if prev == "analysis":
                try:
                    signal.disconnect(analysis_slot)
                except (TypeError, RuntimeError):
                    pass
            elif prev == "continuous":
                try:
                    signal.disconnect(continuous_slot)
                except (TypeError, RuntimeError):
                    pass
            signal.connect(analysis_slot if scope == "analysis" else continuous_slot)

        self._range_toolbar_scope = scope
        if scope == "analysis":
            self._push_range_counts_to_analysis()
        else:
            self._update_range_counts()

    def open_analysis_window(self) -> None:
        """Compat : basculer vers le mode moyenne (plus de fenêtre séparée)."""
        self.set_preview_mode("average", redraw=True)

    def _context_panels(self) -> tuple[str, ...]:
        if self._cb_impedance.isChecked():
            return ("impedance",)
        return ()

    def _stim_reference_s(self) -> float | None:
        """Onset de référence pour convertir une plage relative → absolue."""
        if not self._stim_times_s:
            return None
        analysis = self._analysis_from_toggles()
        index = analysis.trigger_index()
        if index is None:
            index = 0
        if 0 <= int(index) < len(self._stim_times_s):
            return float(self._stim_times_s[int(index)])
        return float(self._stim_times_s[0])

    def _any_zooms_applied(self) -> bool:
        return self._continuous_zooms_applied or self._analysis_zooms_applied

    def _ensure_analysis_curves(self) -> AnalysisSettings:
        """Garantir au moins une courbe d’analyse (paramétrage Canal)."""
        analysis = self._analysis_from_toggles()
        if analysis.selected_analysis_panels():
            return analysis
        self._cb_raw.blockSignals(True)
        self._cb_hp.blockSignals(True)
        self._cb_lp.blockSignals(True)
        self._cb_rms.blockSignals(True)
        self._cb_raw.setChecked(True)
        self._cb_hp.setChecked(True)
        self._cb_lp.setChecked(True)
        self._cb_rms.setChecked(True)
        self._cb_raw.blockSignals(False)
        self._cb_hp.blockSignals(False)
        self._cb_lp.blockSignals(False)
        self._cb_rms.blockSignals(False)
        return self._analysis_from_toggles()

    def _propagate_continuous_ranges_to_analysis(self) -> AnalysisSettings:
        """Recopier les plages continuous vers moyenne / une stimulation.

        Utilise le paramétrage d’analyse (mode, stim, courbes cochées).
        """
        analysis = self._ensure_analysis_curves()
        # En continuous, le mode d’analyse n’est pas le combo aperçu : conserver
        # average / stimulation déjà choisis (défaut = moyenne).
        if analysis.mode not in ("average", "stimulation"):
            analysis = replace(analysis, mode="average")
        copied: list[TimeRangeBar] = []
        for index, bar in enumerate(self._range_ctrl.bars or ()):
            t0, t1 = bar.ordered()
            copied.append(
                TimeRangeBar(
                    t0_s=t0,
                    t1_s=t1,
                    label=bar.label.strip() or f"Plage {index + 1}",
                    bar_id=str(bar.bar_id or f"bar{index}"),
                    relative_to_stim=False,
                )
            )
        for index, bar in enumerate(self._relative_ranges):
            t0, t1 = bar.ordered()
            copied.append(
                TimeRangeBar(
                    t0_s=t0,
                    t1_s=t1,
                    label=bar.label.strip() or f"Rel. stim {index + 1}",
                    bar_id=str(bar.bar_id or f"rel{index}"),
                    relative_to_stim=True,
                )
            )
        self._analysis_ranges = copied
        self._analysis_active_index = 0 if copied else 0
        self._analysis_zooms_applied = bool(copied)
        self.settings = replace(self.settings, analysis=analysis)
        return analysis

    def _append_zoom_panels(
        self,
        analysis_panels: list[PanelPlacement],
        zoom_panels: list[PanelPlacement],
        *,
        analysis: AnalysisSettings,
        t0: float,
        t1: float,
        label: str,
        bar_id: str,
        absolute: bool,
        for_continuous: bool,
        for_analysis: bool,
    ) -> None:
        # Zooms continuous → fenêtres détachées (brut) uniquement.
        if for_continuous:
            if absolute:
                zoom_panels.append(
                    PanelPlacement("full_recording").with_custom_zoom(
                        t0,
                        t1,
                        label=f"Zoom brut {label}",
                        instance_id=bar_id,
                        absolute=True,
                    )
                )
            else:
                stim_t = self._stim_reference_s()
                if stim_t is not None:
                    zoom_panels.append(
                        PanelPlacement("full_recording").with_custom_zoom(
                            stim_t + t0,
                            stim_t + t1,
                            label=f"Zoom brut {label}",
                            instance_id=bar_id,
                            absolute=True,
                        )
                    )
        # Zooms d’analyse → grille aperçu (mode moyenne / stimulation).
        if for_analysis:
            for key in analysis.selected_analysis_panels():
                analysis_panels.append(
                    PanelPlacement(key).with_custom_zoom(
                        t0,
                        t1,
                        label=label,
                        instance_id=bar_id,
                        absolute=absolute,
                    )
                )

    def _rebuild_placements(self, *, reopen_zooms: bool = False) -> None:
        analysis = self._analysis_from_toggles()
        analysis_panels: list[PanelPlacement] = []
        zoom_panels: list[PanelPlacement] = []

        if self.is_analysis_view():
            # Graphs d'analyse dans la grille principale (fusion avec l'aperçu).
            for key in analysis.selected_analysis_panels():
                analysis_panels.append(PanelPlacement(key, section="full"))
            if not analysis_panels:
                analysis_panels.append(PanelPlacement("analysis_raw", section="full"))
            if self._analysis_zooms_applied:
                for index, bar in enumerate(self._analysis_ranges):
                    t0, t1 = bar.ordered()
                    label = bar.label.strip() or f"Plage {index + 1}"
                    bar_id = str(bar.bar_id or f"an{index}")
                    self._append_zoom_panels(
                        analysis_panels,
                        zoom_panels,
                        analysis=analysis,
                        t0=t0,
                        t1=t1,
                        label=label,
                        bar_id=bar_id,
                        absolute=not bool(bar.relative_to_stim),
                        for_continuous=False,
                        for_analysis=True,
                    )
            self._placements = tuple(analysis_panels)
            self._zoom_placements = ()
            self._analysis_placements = tuple(analysis_panels)
            self._close_all_zoom_windows()
            self._configure_preview_grid()
            return

        # Continuous : traces continues + contexte + résumés (pas de courbe moyennée).
        panels: list[PanelPlacement] = [PanelPlacement("full_recording")]
        for key in self._context_panels():
            panels.append(PanelPlacement(key))
        for key in analysis.selected_global_panels():
            panels.append(PanelPlacement(key, section="full"))

        if self._continuous_zooms_applied:
            # Zooms continuous uniquement (fenêtres détachées) — pas d’analyse.
            self._propagate_continuous_ranges_to_analysis()
            _unused_analysis, zoom_panels = self._continuous_zoom_panel_lists(
                for_analysis=False
            )

        self._placements = tuple(panels)
        self._zoom_placements = tuple(zoom_panels)
        self._analysis_placements = ()
        self._configure_preview_grid()
        self._sync_zoom_windows(reopen=reopen_zooms)

    def _sync_zoom_windows(self, *, reopen: bool = False) -> None:
        """Ouvrir / mettre à jour / fermer les fenêtres de zoom courbe complète."""
        if reopen:
            self._user_closed_zooms.clear()
        wanted = {p.key: p for p in self._zoom_placements}

        for key in list(self._zoom_windows):
            if key not in wanted:
                win = self._zoom_windows.pop(key)
                try:
                    win.closed.disconnect(self._on_zoom_window_closed)
                except (TypeError, RuntimeError):
                    pass
                try:
                    win.refreshRequested.disconnect(self._on_zoom_window_refresh)
                except (TypeError, RuntimeError):
                    pass
                win.close()

        for index, (key, placement) in enumerate(wanted.items()):
            existing = self._zoom_windows.get(key)
            if existing is not None:
                existing.update_placement(placement)
                existing.setWindowTitle(
                    f"{self.channel_name} — {placement.title()}"
                )
                continue
            if key in self._user_closed_zooms and not reopen:
                continue
            win = DetachedPanelWindow(
                placement, self.local_settings(), parent=self
            )
            win.setWindowTitle(f"{self.channel_name} — {placement.title()}")
            win.resize(1100, max(520, self._panel_height() + 180))
            offset = 36 * (index % 10)
            win.move(self.x() + 48 + offset, self.y() + 48 + offset)
            win.closed.connect(self._on_zoom_window_closed)
            win.refreshRequested.connect(self._on_zoom_window_refresh)
            self._zoom_windows[key] = win
            win.show()
            win.raise_()

    def _on_zoom_window_closed(self, placement: PanelPlacement) -> None:
        key = placement.key
        self._zoom_windows.pop(key, None)
        self._user_closed_zooms.add(key)

    def _on_zoom_window_refresh(self, window: DetachedPanelWindow) -> None:
        self._render_zoom_window(window)

    def _render_zoom_window(self, window: DetachedPanelWindow) -> None:
        if self._request_factory is None:
            return
        request = self._request_factory(self, window.placement)
        if request is None:
            return
        # Affichage local à la fenêtre zoom / détachée (pas l’aperçu canal).
        settings = apply_local_display_settings(
            request.settings, window.local_settings()
        )
        # Les bornes X viennent de la plage : ne pas conserver un ancien zoom.
        preserve = bool(self._preserve_view) and not window.placement.has_custom_zoom
        window.render(replace(request, settings=settings, preserve_view=preserve))

    def _close_all_zoom_windows(self) -> None:
        for key in list(self._zoom_windows):
            win = self._zoom_windows.pop(key)
            try:
                win.closed.disconnect(self._on_zoom_window_closed)
            except (TypeError, RuntimeError):
                pass
            try:
                win.refreshRequested.disconnect(self._on_zoom_window_refresh)
            except (TypeError, RuntimeError):
                pass
            win.close()
        self._zoom_placements = ()
        self._user_closed_zooms.clear()

    def _sync_analysis_window(self, *, force_open: bool = False) -> None:
        """Compat : plus de fenêtre séparée — graphs déjà dans la grille."""
        del force_open
        self._push_range_counts_to_analysis()
        self._sync_analysis_range_ctrl_from_list()

    def _push_range_counts_to_analysis(self) -> None:
        n = len(self._analysis_ranges)
        if n == 0:
            rel_text = "0 plage(s) Analyse (locales à cette vue)"
        else:
            parts = []
            for bar in self._analysis_ranges:
                t0, t1 = bar.ordered()
                name = bar.label.strip() or "plage"
                kind = "rel" if bar.relative_to_stim else "abs"
                parts.append(f"{name} ({kind}) [{t0:g}…{t1:g} s]")
            rel_text = f"{n} plage(s) Analyse : " + " · ".join(parts)
        if self.is_analysis_view():
            self._range_toolbar.set_bar_count(n, self._analysis_active_index)
            self._relative_label.setText(rel_text)

    def _close_analysis_window(self) -> None:
        """Compat : rien à fermer (graphs fusionnés dans l'aperçu)."""
        self._analysis_range_ctrl.detach()

    def _on_extra_toggles_changed(self, *_args: Any) -> None:
        self.settings = self.local_settings()
        self._sync_params_base()
        self._rebuild_placements()
        self.refreshRequested.emit(self)

    def _apply_zooms_continuous(self) -> None:
        self.apply_processing(scope="continuous")

    def _apply_zooms_analysis(self) -> None:
        self.apply_processing(scope="analysis")

    def apply_processing(
        self, *, scope: Literal["continuous", "analysis"] = "continuous"
    ) -> None:
        """« Appliquer les zooms » : continuous et/ou moyenne / une stimulation."""
        # Conserver zoom / pan du continuous pendant le refresh qui suit.
        self._preserve_view = True
        if scope == "analysis":
            self._ensure_analysis_curves()
            self._analysis_zooms_applied = bool(self._analysis_ranges)
            self._sync_analysis_range_ctrl_from_list()
            self._push_range_counts_to_analysis()
            self._rebuild_placements(reopen_zooms=False)
            n_zoom = sum(1 for p in self._analysis_placements if p.has_custom_zoom)
            mode = self._analysis_from_toggles().describe()
            self._status.setText(
                f"{len(self._analysis_ranges)} plage(s) Analyse ({mode}) — "
                f"{n_zoom} zoom(s) analyse."
            )
        else:
            self._continuous_zooms_applied = bool(
                self._range_ctrl.bars or self._relative_ranges
            )
            self._sync_settings_from_bars()
            self._propagate_continuous_ranges_to_analysis()
            self._rebuild_placements(reopen_zooms=True)
            self._status.setText(
                f"{len(self._range_ctrl.bars)} plage(s) abs. + "
                f"{len(self._relative_ranges)} relative(s) — "
                f"{len(self._zoom_windows)} zoom(s) continuous."
            )
        self.refreshRequested.emit(self)

    # -------------------------------------------------------------- bars UI

    def _update_range_counts(self) -> None:
        """Compteurs de la toolbar aperçu canal uniquement."""
        self._range_toolbar.set_bar_count(
            len(self._range_ctrl.bars), self._range_ctrl.active_index
        )
        n_rel = len(self._relative_ranges)
        if n_rel == 0:
            self._relative_label.setText("0 plage(s) relative(s) à la stim")
        else:
            parts = []
            for bar in self._relative_ranges:
                t0, t1 = bar.ordered()
                name = bar.label.strip() or "rel"
                parts.append(f"{name} [{t0:g}…{t1:g} s]")
            self._relative_label.setText(
                f"{n_rel} relative(s) : " + " · ".join(parts)
            )

    def _add_bar_continuous(self) -> None:
        self._range_ctrl.add_bar()
        self._update_range_counts()
        self._sync_settings_from_bars()
        if self._continuous_zooms_applied:
            self._rebuild_placements(reopen_zooms=True)
            self.refreshRequested.emit(self)

    def _add_bar_analysis(self) -> None:
        """+ Plage en mode analyse — n’affecte pas les plages continuous."""
        settings = self.local_settings()
        n = len(self._analysis_ranges) + 1
        self._analysis_ranges.append(
            TimeRangeBar(
                t0_s=float(settings.zoom_onset_t0_s),
                t1_s=float(settings.zoom_onset_t1_s),
                label=f"Plage {n}",
                bar_id=uuid.uuid4().hex[:8],
                relative_to_stim=True,
            )
        )
        self._analysis_active_index = len(self._analysis_ranges) - 1
        self._analysis_zooms_applied = True
        self._sync_analysis_range_ctrl_from_list()
        self._push_range_counts_to_analysis()
        self._rebuild_placements(reopen_zooms=False)
        self._status.setText(
            f"Plage Analyse [{settings.zoom_onset_t0_s:g} … "
            f"{settings.zoom_onset_t1_s:g}] s — zoom analyse."
        )
        self.refreshRequested.emit(self)

    def _add_relative_range_continuous(self) -> None:
        settings = self.local_settings()
        spec = ask_custom_zoom(
            self,
            default_t0=float(settings.zoom_onset_t0_s),
            default_t1=float(settings.zoom_onset_t1_s),
        )
        if spec is None:
            return
        n = len(self._relative_ranges) + 1
        label = spec.label.strip() or f"Rel. stim {n}"
        self._relative_ranges.append(
            TimeRangeBar(
                t0_s=float(spec.t0_s),
                t1_s=float(spec.t1_s),
                label=label,
                bar_id=uuid.uuid4().hex[:8],
                relative_to_stim=True,
            )
        )
        self._update_range_counts()
        self._sync_settings_from_bars()
        self._continuous_zooms_applied = True
        self._propagate_continuous_ranges_to_analysis()
        self._rebuild_placements(reopen_zooms=True)
        self._status.setText(
            f"Plage relative [{spec.t0_s:g} … {spec.t1_s:g}] s — "
            f"zoom continuous."
        )
        self.refreshRequested.emit(self)

    def _add_relative_range_analysis(self) -> None:
        settings = self.local_settings()
        parent: QWidget = (
            self
        )
        spec = ask_custom_zoom(
            parent,
            default_t0=float(settings.zoom_onset_t0_s),
            default_t1=float(settings.zoom_onset_t1_s),
        )
        if spec is None:
            return
        n = len(self._analysis_ranges) + 1
        label = spec.label.strip() or f"Rel. stim {n}"
        self._analysis_ranges.append(
            TimeRangeBar(
                t0_s=float(spec.t0_s),
                t1_s=float(spec.t1_s),
                label=label,
                bar_id=uuid.uuid4().hex[:8],
                relative_to_stim=True,
            )
        )
        self._analysis_active_index = len(self._analysis_ranges) - 1
        self._analysis_zooms_applied = True
        self._sync_analysis_range_ctrl_from_list()
        self._push_range_counts_to_analysis()
        self._rebuild_placements(reopen_zooms=False)
        self._status.setText(
            f"Plage Analyse [{spec.t0_s:g} … {spec.t1_s:g}] s — "
            "zoom analyse."
        )
        self.refreshRequested.emit(self)

    def _remove_bar_continuous(self) -> None:
        if self._range_ctrl.bars:
            self._range_ctrl.remove_active_bar()
        elif self._relative_ranges:
            self._relative_ranges.pop()
        else:
            QMessageBox.information(self, "Plages", "Aucune plage à supprimer.")
            return
        if not self._range_ctrl.bars and not self._relative_ranges:
            self._continuous_zooms_applied = False
        self._update_range_counts()
        self._sync_settings_from_bars()
        if self._continuous_zooms_applied or not (
            self._range_ctrl.bars or self._relative_ranges
        ):
            self._rebuild_placements(reopen_zooms=False)
            self.refreshRequested.emit(self)

    def _remove_bar_analysis(self) -> None:
        if not self._analysis_ranges:
            parent: QWidget = (
                self
            )
            QMessageBox.information(parent, "Plages", "Aucune plage à supprimer.")
            return
        idx = max(0, min(len(self._analysis_ranges) - 1, self._analysis_active_index))
        self._analysis_ranges.pop(idx)
        self._analysis_active_index = max(
            0, min(len(self._analysis_ranges) - 1, idx)
        )
        if not self._analysis_ranges:
            self._analysis_zooms_applied = False
        self._sync_analysis_range_ctrl_from_list()
        self._push_range_counts_to_analysis()
        self._rebuild_placements(reopen_zooms=False)
        self.refreshRequested.emit(self)

    def _on_active_changed(self, index: int) -> None:
        self._range_ctrl.set_active_index(index)
        self._sync_settings_from_bars()

    def _on_analysis_active_changed(self, index: int) -> None:
        if not self._analysis_ranges:
            self._analysis_active_index = 0
            self._analysis_range_ctrl.set_active_index(0)
            return
        self._analysis_active_index = max(
            0, min(len(self._analysis_ranges) - 1, int(index))
        )
        self._analysis_range_ctrl.set_active_index(self._analysis_active_index)
        self._push_range_counts_to_analysis()

    def _on_bars_changed(self, _bars: object) -> None:
        self._update_range_counts()
        self._sync_settings_from_bars()
        self._bars_debouncer.request()

    def _on_analysis_bars_changed(self, bars: object) -> None:
        """Glisser une barre sur un graph Analyse → synchroniser la liste locale."""
        self._analysis_ranges = [replace(b) for b in bars]  # type: ignore[arg-type]
        self._analysis_active_index = self._analysis_range_ctrl.active_index
        self._push_range_counts_to_analysis()
        self._analysis_bars_debouncer.request()

    def _after_bars_moved(self) -> None:
        """Déplacement des barres aperçu → zooms continuous (fenêtres détachées).

        Sans zooms appliqués : rien à redessiner (les barres sont déjà sur le canvas).
        N’actualise pas toute la grille (évite de casser le glisser).
        """
        if not self._continuous_zooms_applied:
            return
        self._propagate_continuous_ranges_to_analysis()
        _unused, zoom_panels = self._continuous_zoom_panel_lists(for_analysis=False)
        self._zoom_placements = tuple(zoom_panels)
        self._analysis_placements = ()
        self._sync_zoom_windows(reopen=False)
        for win in list(self._zoom_windows.values()):
            self._render_zoom_window(win)
        self._status.setText("Barres déplacées — zooms continuous actualisés.")

    def _continuous_zoom_panel_lists(
        self,
        analysis: AnalysisSettings | None = None,
        *,
        for_analysis: bool = False,
    ) -> tuple[list[PanelPlacement], list[PanelPlacement]]:
        """Zooms continuous depuis les barres (+ panels analyse si demandé)."""
        analysis = analysis or self._ensure_analysis_curves()
        analysis_panels: list[PanelPlacement] = []
        zoom_panels: list[PanelPlacement] = []
        abs_bars = self._range_ctrl.bars or ()
        for index, bar in enumerate(abs_bars):
            t0, t1 = bar.ordered()
            label = bar.label.strip() or f"Plage {index + 1}"
            bar_id = str(bar.bar_id or f"bar{index}")
            self._append_zoom_panels(
                analysis_panels,
                zoom_panels,
                analysis=analysis,
                t0=t0,
                t1=t1,
                label=label,
                bar_id=bar_id,
                absolute=True,
                for_continuous=True,
                for_analysis=for_analysis,
            )
        for index, bar in enumerate(self._relative_ranges):
            t0, t1 = bar.ordered()
            label = bar.label.strip() or f"Rel. stim {index + 1}"
            bar_id = str(bar.bar_id or f"rel{index}")
            self._append_zoom_panels(
                analysis_panels,
                zoom_panels,
                analysis=analysis,
                t0=t0,
                t1=t1,
                label=label,
                bar_id=bar_id,
                absolute=False,
                for_continuous=True,
                for_analysis=for_analysis,
            )
        return analysis_panels, zoom_panels

    def _live_update_analysis_zoom_panels(
        self, analysis_panels: list[PanelPlacement]
    ) -> None:
        """Mettre à jour les graphs analyse dans la grille sans ``configure``."""
        if self._request_factory is None or not analysis_panels:
            self._analysis_placements = tuple(analysis_panels)
            return
        for placement in analysis_panels:
            widget = self.grid.panel_widget(placement)
            if widget is None:
                # Clé absente (1ʳᵉ applique ou courbe ajoutée) → rebuild complet.
                self._rebuild_placements(reopen_zooms=False)
                self.refreshRequested.emit(self)
                return
            updater = getattr(widget, "update_placement", None)
            if callable(updater):
                updater(placement)
            else:
                widget.placement = placement
            request = self._request_factory(self, placement)
            if request is None:
                continue
            # Bornes = plage courante ; conserver le paramétrage d’analyse.
            settings = apply_local_display_settings(
                request.settings,
                self.local_settings(),
                include_analysis=True,
            )
            widget.render(replace(request, settings=settings, preserve_view=False))
        self._analysis_placements = tuple(analysis_panels)

    def _after_analysis_bars_moved(self) -> None:
        """Déplacement des barres Analyse → zooms Analyse seulement."""
        if not self._analysis_zooms_applied:
            return
        # Conserver la vue des graphs de base ; les zooms custom suivent la plage.
        self._preserve_view = True
        self._rebuild_placements(reopen_zooms=False)
        self._status.setText("Barres Analyse déplacées — zooms Analyse actualisés.")
        self.refreshRequested.emit(self)
    def _sync_analysis_range_ctrl_from_list(self) -> None:
        """Pousser ``_analysis_ranges`` vers le controller (sans boucle barsChanged)."""
        ctrl = self._analysis_range_ctrl
        try:
            ctrl.barsChanged.disconnect(self._on_analysis_bars_changed)
        except (TypeError, RuntimeError):
            pass
        ctrl.set_bars(
            self._analysis_ranges, active_index=self._analysis_active_index
        )
        ctrl.barsChanged.connect(self._on_analysis_bars_changed)

    def _on_analysis_render_finished(self, _drawn: int = 0, _elapsed_s: float = 0.0) -> None:
        self._attach_range_bars_to_analysis()

    def _attach_range_bars_to_analysis(self) -> None:
        """Réattacher les barres sur le premier graph temporel d'analyse (grille principale)."""
        if not self.is_analysis_view():
            self._analysis_range_ctrl.detach()
            return
        target = None
        for placement in self._placements:
            if placement.panel not in _ANALYSIS_TEMPORAL_PANELS:
                continue
            if placement.has_custom_zoom:
                continue
            widget = self.grid.panel_widget(placement)
            if widget is None:
                continue
            figure = getattr(widget, "figure", None)
            canvas = getattr(widget, "canvas", None)
            if figure is None or canvas is None:
                continue
            axes = list(getattr(figure, "axes", []) or [])
            if not axes:
                continue
            target = (canvas, axes)
            break
        if target is None:
            self._analysis_range_ctrl.detach()
            return
        canvas, axes = target
        self._analysis_range_ctrl.set_display_offset(0.0)
        try:
            xlim = axes[0].get_xlim()
            self._analysis_range_ctrl.set_time_span(float(xlim[0]), float(xlim[1]))
        except Exception:
            pass
        self._sync_analysis_range_ctrl_from_list()
        self._analysis_range_ctrl.attach(canvas, axes)

    def _sync_settings_from_bars(self) -> None:
        self.settings = self.local_settings()
        self._sync_params_base()

    def _on_toggles_changed(self, *_args: Any) -> None:
        """Coches continuous (contexte / résumés)."""
        self.settings = self.local_settings()
        self._sync_params_base()
        self._rebuild_placements()
        self.refreshRequested.emit(self)

    def _configure_preview_grid(self) -> None:
        """Même layout fill pour continuous / moyenne / stimulation."""
        n = max(1, len(self._placements))
        self.grid.configure(
            self._placements,
            columns=1,
            panel_height=self._panel_height(n_panels=n),
            fill=True,
        )

    def _panel_height(self, *, n_panels: int | None = None) -> int:
        """Hauteur cible pour remplir le viewport (partagée entre les panneaux).

        Plancher = ``preferred_height_px`` du catalogue quand un seul graph.
        """
        from panel_registry import panel_info

        n = max(1, int(n_panels) if n_panels is not None else len(self._placements) or 1)
        preferreds = [
            int(panel_info(p.panel).preferred_height_px) for p in self._placements
        ]
        floor = (
            max(preferreds)
            if preferreds and n == 1
            else _PREVIEW_PANEL_MIN_HEIGHT
        )
        vp = 0
        try:
            vp = int(self.grid.viewport().height())
        except Exception:
            pass
        if vp > 80:
            margins = 12
            gaps = max(0, n - 1) * 8
            share = (vp - margins - gaps) // n
            return max(floor, share) if n == 1 else max(_PREVIEW_PANEL_MIN_HEIGHT, share)
        # Fallback avant le 1er layout / show.
        return max(floor, _PREVIEW_PANEL_MIN_HEIGHT)

    def _checkbox_for_panel(self, panel: str) -> QCheckBox | None:
        mapping = {
            "summary_rms": self._cb_summary_rms,
            "summary_rms_table": self._cb_summary_rms_table,
            "impedance": self._cb_impedance,
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

    def _on_panel_detached(self, placement: PanelPlacement) -> None:
        """Ouvrir un panneau dans sa propre fenêtre (bouton ⤢)."""
        key = placement.key
        existing = self._detached.get(key)
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return
        # Ne pas doubler une fenêtre de zoom déjà ouverte pour la même clé.
        zoom = self._zoom_windows.get(key)
        if zoom is not None:
            zoom.raise_()
            zoom.activateWindow()
            return
        win = DetachedPanelWindow(placement, self.local_settings(), parent=self)
        win.setWindowTitle(f"{self.channel_name} — {placement.title()}")
        win.closed.connect(self._on_detached_closed)
        win.refreshRequested.connect(self._on_detached_refresh)
        self._detached[key] = win
        win.show()
        win.raise_()
        self._render_zoom_window(win)

    def _on_detached_closed(self, placement: PanelPlacement) -> None:
        self._detached.pop(placement.key, None)

    def _on_detached_refresh(self, window: DetachedPanelWindow) -> None:
        self._render_zoom_window(window)

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

    def _checkbox_for_analysis_panel(self, panel: str) -> QCheckBox | None:
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

    def _on_panel_removed(self, placement: PanelPlacement) -> None:
        if placement.panel == "full_recording" and not placement.has_custom_zoom:
            return
        box = self._checkbox_for_panel(placement.panel)
        if box is None:
            box = self._checkbox_for_analysis_panel(placement.panel)
        if box is not None and box.isChecked() and not placement.has_custom_zoom:
            box.blockSignals(True)
            box.setChecked(False)
            box.blockSignals(False)
            self.settings = self.local_settings()
            self._rebuild_placements()
            self.refreshRequested.emit(self)
            return
        self._placements = tuple(p for p in self._placements if p.key != placement.key)
        if not self._placements:
            self._rebuild_placements()
        else:
            self._configure_preview_grid()
        self.refreshRequested.emit(self)

    # ---------------------------------------------------------------- redraw

    def redraw(self, *, preserve_view: bool = False) -> None:
        """Redessiner. ``preserve_view=True`` conserve zoom / pan (ex. après Traiter)."""
        self.settings = self.local_settings()
        if self._request_factory is None:
            return
        keep = bool(preserve_view) or bool(self._preserve_view)
        self._preserve_view = keep

        def factory(placement: PanelPlacement) -> RenderRequest:
            # Zoom custom = bornes de plage : toujours les appliquer.
            preserve = keep and not placement.has_custom_zoom
            request = self._request_factory(self, placement)  # type: ignore[misc]
            if request is not None:
                if preserve:
                    return replace(request, preserve_view=True)
                return replace(request, preserve_view=False) if keep else request
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
        for zoom_win in list(self._zoom_windows.values()):
            self._render_zoom_window(zoom_win)
        for detached in list(self._detached.values()):
            self._render_zoom_window(detached)

    def _on_render_finished(self, drawn: int, elapsed_s: float) -> None:
        self._preserve_view = False
        n_zoom = len(self._zoom_windows)
        n_an = len(self._analysis_placements)
        parts: list[str] = []
        if n_zoom:
            parts.append(f"{n_zoom} zoom(s)")
        if n_an:
            parts.append(f"{n_an} analyse")
        suffix = (" · " + " · ".join(parts)) if parts else ""
        self._status.setText(f"{drawn} panneau(x) · {elapsed_s * 1000:.0f} ms{suffix}")
        if self.is_analysis_view():
            # Ne pas réattacher pendant un glisser (sinon le drag est annulé).
            if not self._analysis_range_ctrl.is_dragging:
                self._attach_range_bars_to_analysis()
        else:
            if not self._range_ctrl.is_dragging:
                self._attach_range_bars_to_full_recording()

    def _attach_range_bars_to_full_recording(self) -> None:
        """Réattacher les barres interactives sur le panneau brut (non zoomé)."""
        for placement in self._placements:
            if placement.panel != "full_recording" or placement.has_custom_zoom:
                continue
            widget = self.grid.panel_widget(placement)
            if widget is None:
                continue
            figure = getattr(widget, "figure", None)
            canvas = getattr(widget, "canvas", None)
            if figure is None or canvas is None:
                continue
            axes = list(getattr(figure, "axes", []) or [])
            if not axes:
                continue
            offset = continuous_sync_offset_s(
                self._stim_times_s, self.settings.time_sync
            )
            self._range_ctrl.set_display_offset(offset)
            # Limites X de l’axe = coords d’affichage → bornes absolues pour le stockage.
            try:
                xlim = axes[0].get_xlim()
                self._range_ctrl.set_time_span(
                    float(xlim[0]) + offset, float(xlim[1]) + offset
                )
            except Exception:
                pass
            self._range_ctrl.attach(canvas, axes)
            break

    def closeEvent(self, event: Any) -> None:  # noqa: D102
        self._range_ctrl.detach()
        self._analysis_range_ctrl.detach()
        self._close_all_zoom_windows()
        self._close_all_detached()
        self._close_analysis_window()
        if not self._embedded:
            self.closed.emit(self.window_id)
        super().closeEvent(event)

    def shutdown(self) -> None:
        """Nettoyage explicite (vue embarquée dans la fenêtre principale)."""
        self._range_ctrl.detach()
        self._analysis_range_ctrl.detach()
        self._close_all_zoom_windows()
        self._close_all_detached()
        self._close_analysis_window()
