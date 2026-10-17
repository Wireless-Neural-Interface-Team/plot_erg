"""Vue canal (aperçu / inspection) : continuous, moyenne ou stimulation.

Embarquée dans la fenêtre principale en mode aperçu, ou flottante.
Mêmes paramètres (Pipeline, plages, axes) pour les trois modes — seul le
contenu des traces change. Le mode se choisit dans Paramètres → Canal.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import replace
from typing import Any, Callable, Literal, Mapping, Sequence

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

from gui.form_widgets import FitWidthScrollArea, configure_narrow_form
from gui.jobs import Debouncer
from gui.services.view_scopes import settings_for_zoom_window
from gui.widgets.custom_zoom_dialog import ask_custom_zoom
from gui.widgets.panel_canvas import (
    DetachedPanelWindow,
    DetachedZoomWindow,
    initial_zoom_panel_height,
    zoom_grid_panel_height,
    zoom_scale_bar_amp_fit_active,
    _Y_SCALE_NONNEG_FAMILIES,
    _axis_ylim_setting,
    y_scale_family,
)
from gui.widgets.panel_grid import PanelGrid
from gui.widgets.range_bars import RangeBarControllerPg, RangeBarToolbar
from gui.widgets.view_params import LocalViewParams
from panel_registry import RenderRequest
from view_config import (
    AnalysisMode,
    AnalysisSettings,
    AnalysisStream,
    PanelPlacement,
    STREAM_SHORT_LABELS,
    TimeRangeBar,
    ViewerSettings,
    apply_local_display_settings,
    continuous_sync_offset_s,
    default_zone_label,
    default_zoom_range_color,
    resolved_range_bar_color,
    resolved_zone_label,
)

_WINDOW_COUNTER = itertools.count(1)

PreviewDisplayMode = Literal["continuous", "average", "stimulation"]
RequestFactory = Callable[["ChannelAnalysisWindow", PanelPlacement], RenderRequest | None]
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
# Spike scope : formes d’onde, axe en ms. La sélection des spikes suit
# la fenêtre de temps des traces, pas ces panneaux.
_SPIKE_SCOPE_PANELS = frozenset({"analysis_overlay", "spike_overlay"})
_SPIKE_SCOPE_TIME_PANELS = _ANALYSIS_TEMPORAL_PANELS | {"full_recording"}


def _as_time_span(value: object) -> tuple[float, float] | None:
    if not isinstance(value, tuple) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if end < start:
        start, end = end, start
    if not (end > start):
        return None
    return start, end


def _view_covers_data(
    view: tuple[float, float], data: tuple[float, float] | None
) -> bool:
    """True si le zoom utilisateur n’a pas rétréci l’étendue des données."""
    if data is None:
        return True
    span = max(float(data[1]) - float(data[0]), 1e-9)
    tol = max(1e-3, 0.002 * span)
    return float(view[0]) <= float(data[0]) + tol and float(view[1]) >= float(data[1]) - tol


def _host_view_xlim(host: Any) -> tuple[float, float] | None:
    try:
        axes = list(getattr(getattr(host, "figure", None), "axes", []) or [])
        if not axes:
            return None
        xlim = axes[0].get_xlim()
        return _as_time_span((float(xlim[0]), float(xlim[1])))
    except Exception:
        return None


def _qt_alive(obj: Any) -> bool:
    """True si le wrapper Python pointe encore vers un objet Qt C++."""
    if obj is None:
        return False
    try:
        from shiboken6 import isValid

        return bool(isValid(obj))
    except Exception:
        try:
            obj.objectName()  # type: ignore[attr-defined]
            return True
        except RuntimeError:
            return False


def _grid_of(widget: Any) -> PanelGrid | None:
    parent = widget
    while parent is not None:
        if not _qt_alive(parent):
            return None
        if isinstance(parent, PanelGrid):
            return parent
        try:
            parent = parent.parentWidget()
        except RuntimeError:
            return None
    return None


class ChannelAnalysisWindow(QWidget):
    """Vue canal unifiée : continuous, moyenne d’essais ou une stimulation.

    Une seule source de paramètres (Pipeline / plages). Le mode ne change que
    la source des traces. Embarqué : coches Pipeline du dock Paramètres.
    """

    closed = Signal(str)
    refreshRequested = Signal(object)
    previewModeChanged = Signal(str)
    # Mode édité depuis le dock alors que la revue montage est affichée
    # (n’altère pas le mode réel de l’aperçu).
    externalContentEdited = Signal(str, int)
    # Coches d’analyse modifiées localement (ex. fermeture d’un panneau) → sync Pipeline.
    analysisCurvesChanged = Signal(object)

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
        # Plages partagées (continuous / moyenne / stimulation) — une seule liste.
        self._zooms_applied = False
        self._mode_updating = False
        self._pipeline_applying = False
        self._stim_times_s: tuple[float, ...] = ()
        self._relative_ranges: list[TimeRangeBar] = []
        self._active_range_index = 0
        self._zoom_placements: tuple[PanelPlacement, ...] = ()
        self._analysis_placements: tuple[PanelPlacement, ...] = ()
        self._zoom_windows: dict[str, DetachedZoomWindow] = {}
        self._detached: dict[str, DetachedPanelWindow] = {}
        self._user_closed_zoom_keys: set[str] = set()
        self._placements: tuple[PanelPlacement, ...] = ()
        self._preview_mode: PreviewDisplayMode = "continuous"
        # « preview » : le combo édite cette fenêtre. « montage » : il édite
        # seulement le mode affiché dans le dock (profil montage).
        self._mode_editor = "preview"
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
            "Même paramètres (Pipeline, plages, axes) pour les trois modes. "
            "Seul le contenu des traces change : brut, moyenne d’essais, ou une stim."
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

        self._preserve_view = False

        # --- Courbes / spikes (mêmes coches pour continuous, moyenne, stimulation) ---
        # Embarqué : pilotés par Paramètres → Canal → Pipeline (masqués ici).
        self._curves_box = QGroupBox("Courbes", self)
        curves_layout = QVBoxLayout(self._curves_box)
        # Aligner sur les flux continuous si les flags d’analyse sont vides.
        seed_streams = set(base_settings.resolved_continuous_streams()) or {"raw"}
        show_raw = bool(analysis.show_raw) or "raw" in seed_streams
        show_hp = bool(analysis.show_hp) or "hp" in seed_streams
        show_lp = bool(analysis.show_lp) or "lp" in seed_streams
        if not (show_raw or show_hp or show_lp):
            show_raw = True
        self._cb_raw = QCheckBox("WIDE (brut)")
        self._cb_hp = QCheckBox("HIGH (passe-haut)")
        self._cb_lp = QCheckBox("LOW (passe-bas)")
        self._cb_rms = QCheckBox("RMS")
        self._cb_isi = QCheckBox("ISI")
        self._cb_overlay = QCheckBox("Spike scope")
        self._cb_raw.setChecked(show_raw)
        self._cb_hp.setChecked(show_hp)
        self._cb_lp.setChecked(show_lp)
        self._cb_rms.setChecked(analysis.show_rms)
        self._cb_isi.setChecked(analysis.show_isi)
        self._cb_overlay.setChecked(analysis.show_overlay)
        self._cb_mark_stims = QCheckBox("Marqueurs de stimulation")
        self._cb_mark_stims.setChecked(bool(base_settings.continuous_mark_stims))
        self._cb_mark_stims.setToolTip(
            "Marquer les stimulations sur les traces continuous / montage."
        )
        curves_layout.addWidget(self._cb_mark_stims)
        self._cb_mark_stims.toggled.connect(self._on_stim_markers_changed)
        for box in (self._cb_raw, self._cb_hp, self._cb_lp):
            curves_layout.addWidget(box)
            box.toggled.connect(self._on_curves_toggled)
        curves_layout.addWidget(self._cb_rms)
        self._cb_rms.toggled.connect(self._on_curves_toggled)
        for box in (self._cb_isi, self._cb_overlay):
            curves_layout.addWidget(box)
            box.toggled.connect(self._on_curves_toggled)

        self._spikes_box = QGroupBox("Spikes", self)
        spikes_layout = QVBoxLayout(self._spikes_box)
        self._cb_psth = QCheckBox("PSTH")
        self._cb_trial_rate = QCheckBox("Firing rate / essai")
        self._cb_raster = QCheckBox("Raster")
        self._cb_raster.setToolTip("Raster compact (tous essais sur une ligne).")
        self._cb_raster_trials = QCheckBox("Raster / essai")
        self._cb_raster_trials.setToolTip("Raster avec une ligne par essai.")
        self._cb_psth.setChecked(analysis.show_psth)
        self._cb_trial_rate.setChecked(analysis.show_trial_rate)
        self._cb_raster.setChecked(analysis.show_raster)
        self._cb_raster_trials.setChecked(analysis.show_raster_trials)
        for box in (
            self._cb_psth,
            self._cb_trial_rate,
            self._cb_raster,
            self._cb_raster_trials,
        ):
            spikes_layout.addWidget(box)
            box.toggled.connect(self._on_curves_toggled)

        # --- contexte / résumés (tous modes) ---
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
            "Profil RMS moyen sur les canaux cochés — une courbe par enregistrement."
        )
        self._cb_summary_rms_table.setToolTip(
            "Table du RMS moyen des canaux cochés (une colonne par enregistrement)."
        )
        self._cb_summary_rms.setChecked(analysis.show_summary_rms)
        self._cb_summary_rms_table.setChecked(analysis.show_summary_rms_table)
        for box in (self._cb_summary_rms, self._cb_summary_rms_table):
            summary_layout.addWidget(box)
            box.toggled.connect(self._on_extra_toggles_changed)

        self._range_toolbar = RangeBarToolbar(self)
        # Controller continuous (barres abs. sur full_recording) — pyqtgraph.
        self._range_ctrl = RangeBarControllerPg(self)
        self._range_ctrl.barsChanged.connect(self._on_bars_changed)
        self._range_ctrl.activeChanged.connect(self._on_range_ctrl_active_changed)
        # Controller analyse : mêmes plages, attaché aux graphs temporels.
        self._analysis_range_ctrl = RangeBarControllerPg(self)
        self._analysis_range_ctrl.barsChanged.connect(self._on_analysis_bars_changed)
        self._analysis_range_ctrl.activeChanged.connect(
            self._on_analysis_ctrl_active_changed
        )

        self._btn_redraw = QPushButton("Redessiner", self)
        self._btn_redraw.clicked.connect(lambda: self.refreshRequested.emit(self))

        # Embarqué : légende / style / échelles = dock Paramètres (Affichage).
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

        self.grid = PanelGrid(self, allow_zoom=True)
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
        self._spike_scope_view_host: Any = None
        self._spike_scope_refreshing = False
        self._spike_scope_window_key: dict[int, tuple[Any, ...]] = {}
        self._bars_debouncer = Debouncer(80, self)
        self._bars_debouncer.triggered.connect(self._after_bars_moved)
        self._analysis_bars_debouncer = Debouncer(80, self)
        self._analysis_bars_debouncer.triggered.connect(self._after_analysis_bars_moved)
        self._spike_view_debouncer = Debouncer(120, self)
        self._spike_view_debouncer.triggered.connect(
            self._refresh_spike_scope_for_displayed_window
        )
        self._wire_range_toolbar_once()
        self._apply_preview_mode_ui()
        self._rebuild_placements()

    # ---------------------------------------------------------------- wiring

    def set_request_factory(self, factory: RequestFactory) -> None:
        self._request_factory = factory

    def preview_mode(self) -> PreviewDisplayMode:
        return self._preview_mode

    def stim_index(self) -> int:
        """Indice 0-based de la stimulation de *cette* vue aperçu."""
        if self._mode_editor == "montage":
            return max(0, int(self.settings.analysis.stim_index))
        return max(0, int(self._stim_spin.value()) - 1)

    def set_dock_view(self, mode: str) -> None:
        """Montrer les contrôles de l’aperçu, ou seulement le mode en revue montage."""
        preview = str(mode) != "montage"
        self._mode_editor = "preview" if preview else "montage"
        for widget in (
            self._range_toolbar,
            self._context_box,
            self._summary_box,
            self._btn_redraw,
        ):
            widget.setVisible(preview)

    def display_content_mode(self, mode: str, stim_index: int) -> None:
        """Afficher un mode dans le combo sans changer le mode réel de l’aperçu."""
        raw = str(mode or "continuous")
        if raw not in ("continuous", "average", "stimulation"):
            raw = "continuous"
        self._mode_updating = True
        self._mode_combo.blockSignals(True)
        self._stim_spin.blockSignals(True)
        index = self._mode_combo.findData(raw)
        if index >= 0:
            self._mode_combo.setCurrentIndex(index)
        self._stim_spin.setValue(max(1, int(stim_index) + 1))
        self._stim_spin.setEnabled(raw == "stimulation")
        self._mode_combo.blockSignals(False)
        self._stim_spin.blockSignals(False)
        self._mode_updating = False

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
        analysis = self._analysis_from_toggles()
        if resolved in ("average", "stimulation"):
            analysis = replace(analysis, mode=resolved)  # type: ignore[arg-type]
            if resolved == "stimulation":
                analysis = replace(
                    analysis,
                    stim_index=max(0, int(self._stim_spin.value()) - 1),
                )
            if not analysis.selected_analysis_panels():
                # Même repli pour tous les modes : au moins WIDE.
                self._cb_raw.setChecked(True)
                analysis = self._analysis_from_toggles()
                analysis = replace(analysis, mode=resolved)  # type: ignore[arg-type]
        self.settings = replace(
            self.settings,
            analysis=analysis,
            preview_content=resolved,
            continuous_streams=self._selected_streams(),
            continuous_stream=self._selected_streams()[0],
        )
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

    def apply_viewer_settings(self, settings: ViewerSettings) -> None:
        """Pousser les réglages du dock Paramètres (mode embarqué)."""
        analysis = self._analysis_from_toggles()
        bars = self._all_range_bars()
        active = self._active_range_index
        if self._embedded:
            streams = settings.resolved_continuous_streams() or ("raw",)
            mark = bool(settings.continuous_mark_stims)
            self._pipeline_applying = True
            try:
                # Streams from viewer settings are authoritative (do not OR with
                # stale analysis toggles — that can re-enable a hidden stream).
                self._cb_raw.setChecked("raw" in streams)
                self._cb_hp.setChecked("hp" in streams)
                self._cb_lp.setChecked("lp" in streams)
                if not any(
                    box.isChecked() for box in (self._cb_raw, self._cb_hp, self._cb_lp)
                ):
                    self._cb_raw.setChecked(True)
                    streams = ("raw",)
                self._cb_mark_stims.setChecked(mark)
            finally:
                self._pipeline_applying = False
            analysis = self._analysis_from_toggles()
            rms_ylim = settings.rms_ylim
        else:
            streams = self._selected_streams()
            mark = self._cb_mark_stims.isChecked()
            # Échelle RMS : panneau Affichage local (pas le dock principal).
            rms_ylim = (
                self.params.settings().rms_ylim
                if self.params is not None
                else settings.rms_ylim
            )
        self.settings = replace(
            settings,
            analysis=analysis,
            preview_content=self._preview_mode,
            continuous_stream=streams[0],
            continuous_streams=streams,
            continuous_mark_stims=mark,
            range_bars=bars,
            active_range_index=active,
            rms_ylim=rms_ylim,
        )
        # Overrides locaux (édition texte / Style) : ne pas bloquer le toggle global.
        try:
            self.grid.sync_scale_bars_from(self.settings)
        except Exception:
            pass
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
        self._zooms_applied = False
        self._channel_ready = False
        if reset_ranges:
            self._range_ctrl.set_bars(())
            self._relative_ranges.clear()
            self._active_range_index = 0
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
                self._relative_ranges = [replace(b) for b in rel_bars]
            # Sinon : aucune plage par défaut (l’utilisateur en ajoute une).
            self._sync_settings_from_bars()
            self._update_range_counts()

    def _selected_streams(self) -> tuple[AnalysisStream, ...]:
        """WIDE/HIGH/LOW — mêmes coches que l’analyse (show_raw/hp/lp)."""
        streams: list[AnalysisStream] = []
        if self._cb_raw.isChecked():
            streams.append("raw")
        if self._cb_hp.isChecked():
            streams.append("hp")
        if self._cb_lp.isChecked():
            streams.append("lp")
        return tuple(streams) or ("raw",)

    def _all_range_bars(self) -> tuple[TimeRangeBar, ...]:
        """Plages partagées : absolues + relatives (tous modes)."""
        bars: list[TimeRangeBar] = []
        for index, bar in enumerate(self._range_ctrl.bars or ()):
            t0, t1 = bar.ordered()
            bars.append(
                TimeRangeBar(
                    t0_s=t0,
                    t1_s=t1,
                    label=resolved_zone_label(bar, index),
                    bar_id=str(bar.bar_id or f"bar{index}"),
                    relative_to_stim=False,
                    color=str(bar.color or default_zoom_range_color(index)),
                )
            )
        abs_count = len(bars)
        for index, bar in enumerate(self._relative_ranges):
            t0, t1 = bar.ordered()
            global_index = abs_count + index
            bars.append(
                TimeRangeBar(
                    t0_s=t0,
                    t1_s=t1,
                    label=bar.label.strip() or f"Rel. stim {index + 1}",
                    bar_id=str(bar.bar_id or f"rel{index}"),
                    relative_to_stim=True,
                    color=str(bar.color or default_zoom_range_color(global_index)),
                )
            )
        return tuple(bars)

    def local_settings(self) -> ViewerSettings:
        streams = self._selected_streams()
        # Embarqué : base = dock Paramètres (déjà dans self.settings).
        # Flottant : base = panneau local.
        base = self.params.settings() if self.params is not None else self.settings
        bars = self._all_range_bars()
        active = self._active_range_index
        return replace(
            base,
            analysis=self._analysis_from_toggles(),
            preview_content=self._preview_mode,
            continuous_stream=streams[0],
            continuous_streams=streams,
            continuous_mark_stims=self._cb_mark_stims.isChecked(),
            range_bars=bars,
            active_range_index=active,
        )

    def _on_local_params_changed(self) -> None:
        self.settings = self.local_settings()
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
        # Zooms = fenêtres détachées uniquement ; inclus ici pour le surlignage.
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
            show_raster_trials=self._cb_raster_trials.isChecked(),
            show_summary_rms=show_summary_rms,
            show_summary_rms_table=show_summary_rms_table,
        )

    def _on_preview_mode_ui_changed(self, *_args: Any) -> None:
        if self._mode_updating:
            return
        data = str(self._mode_combo.currentData() or "continuous")
        if self._mode_editor == "montage":
            self._stim_spin.setEnabled(data == "stimulation")
            self.externalContentEdited.emit(
                data, max(0, int(self._stim_spin.value()) - 1)
            )
            return
        self.set_preview_mode(data, redraw=True)

    def _on_stim_index_changed(self, *_args: Any) -> None:
        if self._mode_updating:
            return
        if self._mode_editor == "montage":
            data = str(self._mode_combo.currentData() or "continuous")
            self.externalContentEdited.emit(
                data, max(0, int(self._stim_spin.value()) - 1)
            )
            return
        if self._preview_mode != "stimulation":
            return
        self.settings = self.local_settings()
        self._sync_params_base()
        self._update_titles()
        if self._any_zooms_applied():
            self._rebuild_placements()
        self.refreshRequested.emit(self)

    def _on_curves_toggled(self, *_args: Any) -> None:
        """Coches WIDE/HIGH/LOW/RMS… → continuous ou graphs d’analyse."""
        if self._pipeline_applying:
            return
        if not any(box.isChecked() for box in (self._cb_raw, self._cb_hp, self._cb_lp)):
            self._cb_raw.blockSignals(True)
            self._cb_raw.setChecked(True)
            self._cb_raw.blockSignals(False)
        analysis = self._analysis_from_toggles()
        if self.is_analysis_view() and not analysis.selected_analysis_panels():
            self._cb_raw.blockSignals(True)
            self._cb_raw.setChecked(True)
            self._cb_raw.blockSignals(False)
            analysis = self._analysis_from_toggles()
        streams = self._selected_streams()
        self._preserve_view = False
        self.settings = replace(
            self.settings,
            analysis=analysis,
            continuous_streams=streams,
            continuous_stream=streams[0],
        )
        self._sync_params_base()
        self._rebuild_placements()
        self.refreshRequested.emit(self)
        self.analysisCurvesChanged.emit(analysis)

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
        """UI commune : seuls le spin stim et les titres dépendent du mode."""
        self._sync_stim_enabled()
        # Embarqué : courbes / spikes = cases Pipeline (Canal). Flottant : toujours.
        if self._embedded:
            self._curves_box.hide()
            self._spikes_box.hide()
        else:
            self._curves_box.show()
            self._spikes_box.show()
        self._context_box.show()
        self._summary_box.show()
        self._update_titles()
        self._header_label.setText(
            "Barres = clic gauche · pan = clic milieu/droit. "
            "Courbes → Canal → Traitement — mêmes cases pour tous les modes. "
            "Échelles → Affichage."
        )
        self._update_range_counts()

    def apply_pipeline_visibility(
        self,
        flags: Mapping[str, bool] | AnalysisSettings,
        *,
        mark_stims: bool | None = None,
        redraw: bool = True,
    ) -> None:
        """Appliquer les cases Pipeline — même chemin pour les 3 modes."""
        if isinstance(flags, AnalysisSettings):
            raw = {
                "show_raw": flags.show_raw,
                "show_hp": flags.show_hp,
                "show_lp": flags.show_lp,
                "show_rms": flags.show_rms,
                "show_isi": flags.show_isi,
                "show_overlay": flags.show_overlay,
                "show_psth": flags.show_psth,
                "show_trial_rate": flags.show_trial_rate,
                "show_raster": flags.show_raster,
                "show_raster_trials": flags.show_raster_trials,
            }
        else:
            raw = dict(flags)
        pairs = (
            (self._cb_raw, "show_raw"),
            (self._cb_hp, "show_hp"),
            (self._cb_lp, "show_lp"),
            (self._cb_rms, "show_rms"),
            (self._cb_isi, "show_isi"),
            (self._cb_overlay, "show_overlay"),
            (self._cb_psth, "show_psth"),
            (self._cb_trial_rate, "show_trial_rate"),
            (self._cb_raster, "show_raster"),
            (self._cb_raster_trials, "show_raster_trials"),
        )
        self._pipeline_applying = True
        try:
            for box, key in pairs:
                box.blockSignals(True)
                box.setChecked(bool(raw.get(key, False)))
                box.blockSignals(False)
            if mark_stims is not None:
                self._cb_mark_stims.blockSignals(True)
                self._cb_mark_stims.setChecked(bool(mark_stims))
                self._cb_mark_stims.blockSignals(False)
            analysis = self._analysis_from_toggles()
            # Continuous : au moins un flux. Analyse : au moins un graph.
            need_fallback = (
                not analysis.selected_analysis_panels()
                if self.is_analysis_view()
                else not any(
                    box.isChecked() for box in (self._cb_raw, self._cb_hp, self._cb_lp)
                )
            )
            if need_fallback:
                self._cb_raw.blockSignals(True)
                self._cb_raw.setChecked(True)
                self._cb_raw.blockSignals(False)
                analysis = self._analysis_from_toggles()
            streams = self._selected_streams()
            self._preserve_view = False
            self.settings = replace(
                self.settings,
                analysis=analysis,
                continuous_streams=streams,
                continuous_stream=streams[0],
                continuous_mark_stims=self._cb_mark_stims.isChecked(),
            )
            self._sync_params_base()
            self._rebuild_placements()
            if redraw:
                self.refreshRequested.emit(self)
        finally:
            self._pipeline_applying = False

    # Alias historique.

    def _wire_range_toolbar_once(self) -> None:
        """Une seule toolbar plages pour continuous / moyenne / stimulation."""
        toolbar = self._range_toolbar
        toolbar.addRequested.connect(self._add_bar)
        toolbar.addRelativeRequested.connect(self._add_relative_range)
        toolbar.removeRequested.connect(self._remove_bar)
        toolbar.activeChanged.connect(self._on_active_changed)
        toolbar.colorChanged.connect(self._on_range_color_changed)
        toolbar.barEdited.connect(self._on_bar_edited)
        toolbar.processRequested.connect(self._apply_zooms)
        self._update_range_counts()


    def _context_panels(self) -> tuple[str, ...]:
        if self._cb_impedance.isChecked():
            return ("impedance",)
        return ()

    def _stim_reference_s(self) -> float | None:
        """Onset de la stimulation choisie (le spin, même hors mode stimulation)."""
        if not self._stim_times_s:
            return None
        index = max(0, int(self._stim_spin.value()) - 1)
        if index >= len(self._stim_times_s):
            index = 0
        return float(self._stim_times_s[index])

    def _local_stim_for_absolute_range(self, t0: float, t1: float) -> float | None:
        """Stim dans / la plus proche d’une plage absolue (pour afficher en t_rel)."""
        if not self._stim_times_s:
            return None
        import numpy as np

        stims = np.asarray(self._stim_times_s, dtype=np.float64).ravel()
        if stims.size == 0:
            return None
        lo, hi = (float(t0), float(t1)) if t0 <= t1 else (float(t1), float(t0))
        inside = stims[(stims >= lo) & (stims <= hi)]
        if inside.size:
            return float(inside[0])
        mid = 0.5 * (lo + hi)
        return float(stims[int(np.argmin(np.abs(stims - mid)))])

    def _bars_for_analysis_display(self) -> list[TimeRangeBar]:
        """Plages en coords t_rel pour le graph moyenne / stimulation."""
        shown: list[TimeRangeBar] = []
        for bar in self._all_range_bars():
            if bar.relative_to_stim:
                shown.append(replace(bar))
                continue
            t0, t1 = bar.ordered()
            ref = self._local_stim_for_absolute_range(t0, t1)
            if ref is None:
                continue
            shown.append(
                replace(bar, t0_s=float(t0) - ref, t1_s=float(t1) - ref)
            )
        return shown

    def _any_zooms_applied(self) -> bool:
        return bool(self._zooms_applied)

    def _ensure_analysis_curves(self) -> AnalysisSettings:
        """Garantir au moins WIDE (même repli pour tous les modes)."""
        analysis = self._analysis_from_toggles()
        if analysis.selected_analysis_panels() or any(
            box.isChecked() for box in (self._cb_raw, self._cb_hp, self._cb_lp)
        ):
            return analysis
        self._cb_raw.blockSignals(True)
        self._cb_raw.setChecked(True)
        self._cb_raw.blockSignals(False)
        return self._analysis_from_toggles()

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
        # Zooms continuous → une fenêtre détachée par plage (tous flux).
        # Toujours raw/hp/lp : le filtrage Canal est local à la fenêtre zoom.
        if for_continuous:
            streams: tuple[AnalysisStream, ...] = ("raw", "hp", "lp")
            for stream in streams:
                short = STREAM_SHORT_LABELS.get(stream, str(stream).upper())
                zoom_label = f"Zoom {short} {label}"
                zoom_id = f"{stream}:{bar_id}"
                base = PanelPlacement("full_recording", stream=str(stream))
                if absolute:
                    zoom_panels.append(
                        base.with_custom_zoom(
                            t0,
                            t1,
                            label=zoom_label,
                            instance_id=zoom_id,
                            absolute=True,
                        )
                    )
                else:
                    stim_t = self._stim_reference_s()
                    if stim_t is None:
                        self._relative_zoom_skipped = True
                    else:
                        zoom_panels.append(
                            base.with_custom_zoom(
                                stim_t + t0,
                                stim_t + t1,
                                label=zoom_label,
                                instance_id=zoom_id,
                                absolute=True,
                            )
                        )
        # Zooms d’analyse / extras → une fenêtre par plage ; filtrage Canal local.
        if for_analysis:
            if for_continuous:
                # Continuous : WIDE/HIGH/LOW déjà en full_recording — ici RMS / spikes…
                extras = replace(
                    analysis,
                    show_raw=False,
                    show_hp=False,
                    show_lp=False,
                    show_rms=True,
                    show_isi=True,
                    show_overlay=True,
                    show_psth=True,
                    show_trial_rate=True,
                    show_raster=True,
                    show_raster_trials=True,
                )
                panel_keys = extras.selected_continuous_extra_panels()
            else:
                full_analysis = replace(
                    analysis,
                    show_raw=True,
                    show_hp=True,
                    show_lp=True,
                    show_rms=True,
                    show_isi=True,
                    show_overlay=True,
                    show_psth=True,
                    show_trial_rate=True,
                    show_raster=True,
                    show_raster_trials=True,
                )
                panel_keys = full_analysis.selected_analysis_panels()
            for key in panel_keys:
                zoom_panels.append(
                    PanelPlacement(key).with_custom_zoom(
                        t0,
                        t1,
                        label=label,
                        instance_id=f"{key}:{bar_id}",
                        absolute=absolute,
                    )
                )

    def _rebuild_placements(self, *, reopen_zooms: bool = False) -> None:
        self._relative_zoom_skipped = False
        analysis = self._analysis_from_toggles()
        analysis_panels: list[PanelPlacement] = []
        zoom_panels: list[PanelPlacement] = []
        shared_bars = self._all_range_bars()

        if self.is_analysis_view():
            # Graphs d'analyse dans la grille (+ contexte / résumés si cochés).
            # Les zooms vont uniquement dans les fenêtres détachées (1 / plage).
            for key in analysis.selected_analysis_panels():
                analysis_panels.append(PanelPlacement(key, section="full"))
            if not analysis_panels:
                analysis_panels.append(PanelPlacement("analysis_raw", section="full"))
            for key in self._context_panels():
                analysis_panels.append(PanelPlacement(key))
            for key in analysis.selected_global_panels():
                analysis_panels.append(PanelPlacement(key, section="full"))
            if self._zooms_applied:
                for index, bar in enumerate(shared_bars):
                    t0, t1 = bar.ordered()
                    label = resolved_zone_label(bar, index)
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
            self._zoom_placements = tuple(zoom_panels)
            self._analysis_placements = tuple(analysis_panels)
            self._configure_preview_grid()
            self._sync_zoom_windows(reopen=reopen_zooms)
            return

        # Continuous : un panneau par flux (WIDE/HIGH/LOW), comme moyenne / stim.
        panels: list[PanelPlacement] = [
            PanelPlacement("full_recording", stream=str(stream))
            for stream in self._selected_streams()
        ]
        if not panels:
            panels.append(PanelPlacement("full_recording", stream="raw"))
        extra_panels: list[PanelPlacement] = []
        for key in analysis.selected_continuous_extra_panels():
            extra = PanelPlacement(key, section="full")
            panels.append(extra)
            extra_panels.append(extra)
        for key in self._context_panels():
            panels.append(PanelPlacement(key))
        for key in analysis.selected_global_panels():
            panels.append(PanelPlacement(key, section="full"))

        if self._zooms_applied:
            _unused_analysis, zoom_panels = self._continuous_zoom_panel_lists(
                for_analysis=True
            )

        self._placements = tuple(panels)
        self._zoom_placements = tuple(zoom_panels)
        # Pour prefetch spikes/RMS même en continuous.
        self._analysis_placements = tuple(extra_panels)
        self._configure_preview_grid()
        self._sync_zoom_windows(reopen=reopen_zooms)

    @staticmethod
    def _bar_id_from_placement(placement: PanelPlacement) -> str:
        iid = str(placement.instance_id or "").strip()
        if not iid:
            return placement.key
        # ``raw:barId`` / ``analysis_rms:barId`` → barId (dernier segment).
        bar_id = iid.rsplit(":", 1)[-1].strip()
        return bar_id or placement.key

    @staticmethod
    def _zoom_group_label(placements: Sequence[PanelPlacement]) -> str:
        if not placements:
            return "Zoom"
        zl = str(placements[0].zoom_label or "").strip()
        # Continuous : « Zoom WIDE Zone 1 » → « Zone 1 »
        if zl.startswith("Zoom "):
            parts = zl.split(" ", 2)
            if len(parts) >= 3:
                return parts[2]
        return zl or "Zoom"

    def _group_zoom_placements(
        self, placements: Sequence[PanelPlacement]
    ) -> dict[str, tuple[str, tuple[PanelPlacement, ...]]]:
        """Regrouper les panneaux zoom par plage (bar_id)."""
        order: list[str] = []
        groups: dict[str, list[PanelPlacement]] = {}
        for placement in placements:
            key = self._bar_id_from_placement(placement)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(placement)
        return {
            key: (self._zoom_group_label(groups[key]), tuple(groups[key]))
            for key in order
        }

    def _sync_zoom_windows(self, *, reopen: bool = False) -> None:
        """Ouvrir / mettre à jour / fermer une fenêtre de zooms par plage."""
        if reopen:
            self._user_closed_zoom_keys.clear()
        placements = self._zoom_placements

        if not placements:
            self._close_all_zoom_windows()
            return

        groups = self._group_zoom_placements(placements)
        wanted = set(groups)

        for key in list(self._zoom_windows):
            if key not in wanted:
                self._close_zoom_window(key, mark_user_closed=False)

        for index, (key, (label, group)) in enumerate(groups.items()):
            existing = self._zoom_windows.get(key)
            if existing is not None:
                existing.update_placements(
                    group,
                    panel_height=zoom_grid_panel_height(existing.local_settings()),
                    zoom_label=label,
                )
                self._render_zoom_window(existing)
                if reopen:
                    try:
                        existing.show()
                        existing.raise_()
                    except RuntimeError:
                        pass
                continue

            if key in self._user_closed_zoom_keys and not reopen:
                continue

            # Scope zoom : pas d’échelle X aperçu (sinon toute la courbe revient).
            # Parent = fenêtre top-level (aperçu embarqué) ; coords écran via mapToGlobal.
            parent_win = self.window() if self.window() is not None else self
            zoom_seed = settings_for_zoom_window(self.local_settings())
            win = DetachedZoomWindow(
                self.channel_name,
                group,
                zoom_seed,
                panel_height=initial_zoom_panel_height(zoom_seed),
                zoom_key=key,
                zoom_label=label,
                parent=parent_win,
            )
            try:
                origin = self.mapToGlobal(self.rect().topLeft())
                win.move(origin.x() + 48 + index * 36, origin.y() + 48 + index * 36)
            except Exception:
                win.move(80 + index * 36, 80 + index * 36)
            win.closed.connect(self._on_zoom_window_closed)
            win.refreshRequested.connect(self._on_zoom_window_refresh)
            self._zoom_windows[key] = win
            win.show()
            win.raise_()
            self._render_zoom_window(win)

        # Même échelle Y entre fenêtres zoom (même flux) — X reste propre à chaque plage.
        self._sync_zoom_shared_ylims()

    def _widget_skips_zoom_y_sync(self, widget: Any) -> bool:
        """Ne pas fusionner l’axe Y si échelle locale (ylim ou barre amplitude fit)."""
        request = getattr(widget, "_last_request", None)
        host = getattr(widget, "plot_host", None) or getattr(widget, "_plot", None)
        if request is not None and host is not None:
            figure = getattr(host, "figure", None)
            if figure is not None and _axis_ylim_setting(figure, request, 0) is not None:
                return True
            if zoom_scale_bar_amp_fit_active(request.settings):
                return True
            plot = getattr(host, "_plot", host)
            fit = getattr(plot, "scale_bar_widget_fit_enabled", None)
            if callable(fit) and bool(fit()):
                return True
        return False

    def _refit_zoom_scale_bars(self) -> None:
        """Re-appliquer le fit échelle sur chaque panneau de chaque fenêtre zoom."""
        for win in self._zoom_windows.values():
            if not isinstance(win, DetachedZoomWindow):
                continue
            panels = getattr(getattr(win, "grid", None), "_panels", None) or {}
            for widget in panels.values():
                run_fit = getattr(widget, "_run_scale_bar_widget_fit", None)
                if not callable(run_fit):
                    continue
                try:
                    run_fit()
                except Exception:
                    pass

    def _sync_zoom_shared_ylims(self) -> None:
        """Une échelle Y commune aux zooms affichés d’une même famille de courbes.

        WIDE avec WIDE, HIGH avec HIGH, etc. L’axe X n’est pas synchronisé
        (chaque fenêtre garde sa fenêtre temporelle). Une échelle Y manuelle
        (onglet Canal) ou une barre d’amplitude valeur+px (fit widget) reste
        propre à chaque fenêtre.
        """
        windows = [
            win
            for win in self._zoom_windows.values()
            if isinstance(win, DetachedZoomWindow)
        ]
        if len(windows) < 2:
            return

        peers: dict[str, list[Any]] = {}
        for win in windows:
            panels = getattr(getattr(win, "grid", None), "_panels", None) or {}
            for widget in panels.values():
                placement = getattr(widget, "placement", None)
                if placement is None or not getattr(placement, "has_custom_zoom", False):
                    continue
                family = y_scale_family(
                    str(placement.panel),
                    placement_stream=str(getattr(placement, "stream", "") or ""),
                )
                if not family:
                    continue
                if self._widget_skips_zoom_y_sync(widget):
                    continue
                peers.setdefault(family, []).append(widget)

        for family, widgets in peers.items():
            if len(widgets) < 2:
                continue
            y0: float | None = None
            y1: float | None = None
            for widget in widgets:
                host = getattr(widget, "plot_host", None) or getattr(widget, "_plot", None)
                if host is None:
                    continue
                span = host.data_ylim() if hasattr(host, "data_ylim") else None
                if span is None:
                    continue
                a, b = float(span[0]), float(span[1])
                y0 = a if y0 is None else min(y0, a)
                y1 = b if y1 is None else max(y1, b)
            if y0 is None or y1 is None or not (y1 > y0):
                continue
            shared = (y0, y1)
            nonneg = family in _Y_SCALE_NONNEG_FAMILIES or any(
                token in family for token in ("psth", "trial_rate", "isi", ":rms")
            )
            for widget in widgets:
                if self._widget_skips_zoom_y_sync(widget):
                    continue
                apply = getattr(widget, "apply_shared_y_scale", None)
                if callable(apply):
                    apply(shared, non_negative=nonneg)
        self._refit_zoom_scale_bars()

    def _on_zoom_window_closed(self) -> None:
        sender = self.sender()
        if not isinstance(sender, DetachedZoomWindow):
            return
        key = str(sender.zoom_key or "")
        if key and self._zoom_windows.get(key) is sender:
            self._zoom_windows.pop(key, None)
        if key:
            self._user_closed_zoom_keys.add(key)
        self._sync_zoom_shared_ylims()

    def _refresh_zoom_placement_catalog(self) -> tuple[PanelPlacement, ...]:
        """Recataloguer tous les zooms possibles (filtrage Canal = fenêtre zoom)."""
        if not self._zooms_applied:
            self._zoom_placements = ()
            return ()
        if self.is_analysis_view():
            analysis = self._ensure_analysis_curves()
            zoom_panels: list[PanelPlacement] = []
            for index, bar in enumerate(self._all_range_bars()):
                t0, t1 = bar.ordered()
                label = resolved_zone_label(bar, index)
                bar_id = str(bar.bar_id or f"an{index}")
                self._append_zoom_panels(
                    [],
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
            self._zoom_placements = tuple(zoom_panels)
        else:
            _unused, zoom_panels = self._continuous_zoom_panel_lists(
                for_analysis=True
            )
            self._zoom_placements = tuple(zoom_panels)
        return self._zoom_placements

    def _on_zoom_window_refresh(self, window: DetachedZoomWindow) -> None:
        # Recataloguer (tous flux / RMS / spikes), puis MainWindow assure les
        # produits manquants et redraw (re-rend aussi les fenêtres zoom).
        placements = self._refresh_zoom_placement_catalog()
        key = str(window.zoom_key or "")
        group = tuple(
            p for p in placements if self._bar_id_from_placement(p) == key
        )
        if group:
            window.update_placements(
                group,
                panel_height=zoom_grid_panel_height(window.local_settings()),
                zoom_label=self._zoom_group_label(group),
            )
        self.refreshRequested.emit(self)

    def _render_zoom_window(self, window: DetachedZoomWindow | DetachedPanelWindow) -> None:
        if self._request_factory is None:
            return
        if isinstance(window, DetachedZoomWindow):
            def factory(placement: PanelPlacement) -> RenderRequest:
                # Affichage / échelle : uniquement le panneau local de *cette* fenêtre zoom
                # (ne pas fusionner avec l’aperçu canal — sinon échelle incohérente).
                zoom_settings = settings_for_zoom_window(window.local_settings())
                request = self._request_factory(self, placement)  # type: ignore[misc]
                if request is None:
                    return RenderRequest(
                        placement=placement,
                        recordings=[],
                        labels=[],
                        colors=[],
                        legend_flags=[],
                        channel_index=self.channel_index,
                        channel_name=self.channel_name,
                        settings=zoom_settings,
                        probe_layout=None,
                        impedance_sessions=[],
                        preserve_view=False,
                    )
                return replace(
                    request,
                    settings=zoom_settings,
                    preserve_view=False,
                    highlight_zooms=(),
                )

            window.schedule_render(factory, force=True)
            self._bind_spike_scope_views(window.grid)
            return

        request = self._request_factory(self, window.placement)
        if request is None:
            return
        # Affichage local à la fenêtre détachée (pas l’aperçu canal).
        settings = apply_local_display_settings(
            request.settings, window.local_settings()
        )
        # Les bornes X viennent de la plage : ne pas conserver un ancien zoom.
        preserve = bool(self._preserve_view) and not window.placement.has_custom_zoom
        window.render(replace(request, settings=settings, preserve_view=preserve))

    def _close_zoom_window(self, key: str, *, mark_user_closed: bool) -> None:
        win = self._zoom_windows.pop(key, None)
        if mark_user_closed:
            self._user_closed_zoom_keys.add(key)
        if win is None:
            return
        try:
            win.closed.disconnect(self._on_zoom_window_closed)
        except (TypeError, RuntimeError):
            pass
        try:
            win.refreshRequested.disconnect(self._on_zoom_window_refresh)
        except (TypeError, RuntimeError):
            pass
        win.close()

    def _close_all_zoom_windows(self) -> None:
        for key in list(self._zoom_windows):
            self._close_zoom_window(key, mark_user_closed=False)
        self._zoom_placements = ()
        self._user_closed_zoom_keys.clear()


    def _close_analysis_window(self) -> None:
        """Compat : rien à fermer (graphs fusionnés dans l'aperçu)."""
        self._analysis_range_ctrl.detach()

    def _on_extra_toggles_changed(self, *_args: Any) -> None:
        self.settings = self.local_settings()
        self._sync_params_base()
        self._rebuild_placements()
        self.refreshRequested.emit(self)

    def _apply_zooms(self) -> None:
        self.apply_processing()

    def apply_processing(
        self, *, scope: Literal["continuous", "analysis"] | None = None
    ) -> None:
        """« Appliquer les zooms » — une fenêtre par plage, uniquement le zoom."""
        del scope  # Plus de scope séparé : une seule liste de plages.
        bars = self._all_range_bars()
        if not bars:
            self._zooms_applied = False
            self._close_all_zoom_windows()
            self._status.setText("Aucune plage — ajoutez-en une avant d’appliquer.")
            return
        self._preserve_view = True
        self._ensure_analysis_curves()
        self._zooms_applied = True
        self._user_closed_zoom_keys.clear()
        self._sync_settings_from_bars()
        self._sync_analysis_range_ctrl_from_list()
        self._update_range_counts()
        self._rebuild_placements(reopen_zooms=True)
        # Garantir que chaque fenêtre zoom est visible (pas masquée derrière).
        for win in self._zoom_windows.values():
            try:
                win.show()
                win.raise_()
            except RuntimeError:
                pass
        n_windows = len(self._zoom_windows)
        n_groups = len(self._group_zoom_placements(self._zoom_placements))
        kind = "analyse" if self.is_analysis_view() else "continuous"
        warn = ""
        if getattr(self, "_relative_zoom_skipped", False):
            warn = " Plage relative ignorée : aucun onset de stimulation."
        if n_windows < n_groups:
            warn += f" {n_groups - n_windows} fenêtre(s) manquante(s)."
        self._status.setText(
            f"{len(bars)} plage(s) — {n_windows}/{n_groups} fenêtre(s) zoom {kind}.{warn}"
        )
        self.refreshRequested.emit(self)

    # -------------------------------------------------------------- bars UI

    def _update_range_counts(self) -> None:
        """Liste éditable + compteurs — mêmes plages pour tous les modes."""
        bars = self._all_range_bars()
        n = len(bars)
        active = int(self._active_range_index)
        if n <= 0:
            active = -1
        elif active >= 0:
            active = max(0, min(n - 1, active))
        else:
            active = -1
        self._active_range_index = active
        self._range_toolbar.set_bars(bars, active)

    def _visible_graph_time_span(self) -> tuple[float, float] | None:
        """Bornes X du graph temporel courant (coords stockées des plages)."""
        if self.is_analysis_view():
            for placement in self._placements:
                if placement.panel not in _ANALYSIS_TEMPORAL_PANELS:
                    continue
                if placement.has_custom_zoom:
                    continue
                widget = self.grid.panel_widget(placement)
                if widget is None:
                    continue
                host = getattr(widget, "plot_host", None) or getattr(
                    widget, "_plot", None
                )
                if host is None:
                    continue
                try:
                    axes = list(
                        getattr(getattr(host, "figure", None), "axes", []) or []
                    )
                    if axes:
                        xlim = axes[0].get_xlim()
                        return float(xlim[0]), float(xlim[1])
                except Exception:
                    pass
            return None
        for placement in self._placements:
            if placement.panel != "full_recording" or placement.has_custom_zoom:
                continue
            widget = self.grid.panel_widget(placement)
            if widget is None:
                continue
            host = getattr(widget, "plot_host", None) or getattr(widget, "_plot", None)
            if host is None:
                continue
            offset = continuous_sync_offset_s(
                self._stim_times_s, self.settings.time_sync
            )
            try:
                axes = list(getattr(getattr(host, "figure", None), "axes", []) or [])
                if axes:
                    xlim = axes[0].get_xlim()
                    return float(xlim[0]) + offset, float(xlim[1]) + offset
            except Exception:
                pass
            break
        return None

    def _add_bar(self) -> None:
        """+ Absolue : barre centrée (10 % du graph) — n’ouvre pas les zooms.

        En moyenne / stimulation le graph est en t_rel : on crée une plage
        *relative* (sinon les bornes t_rel seraient traitées comme temps fichier).
        """
        if self.is_analysis_view():
            span = self._visible_graph_time_span()
            if span is None:
                settings = self.local_settings()
                t0 = float(settings.zoom_onset_t0_s)
                t1 = float(settings.zoom_onset_t1_s)
            else:
                width = max(1e-3, 0.1 * (span[1] - span[0]))
                mid = 0.5 * (span[0] + span[1])
                t0, t1 = mid - 0.5 * width, mid + 0.5 * width
            n = len(self._relative_ranges) + 1
            color_index = len(self._all_range_bars())
            self._relative_ranges.append(
                TimeRangeBar(
                    t0_s=float(t0),
                    t1_s=float(t1),
                    label=default_zone_label(len(self._all_range_bars())),
                    bar_id=uuid.uuid4().hex[:8],
                    relative_to_stim=True,
                    color=default_zoom_range_color(color_index),
                )
            )
            self._active_range_index = len(self._all_range_bars()) - 1
            self._update_range_counts()
            self._sync_settings_from_bars()
            self._sync_analysis_range_ctrl_from_list()
            self._attach_range_bars_to_analysis()
            if self._zooms_applied:
                self._rebuild_placements(reopen_zooms=True)
                self.refreshRequested.emit(self)
            self._status.setText(
                f"Plage relative [{t0:g} … {t1:g}] s (graph analyse) — "
                "« Appliquer les zooms » pour ouvrir."
            )
            return

        span = self._visible_graph_time_span()
        if span is not None:
            self._range_ctrl.set_time_span(span[0], span[1])
        # Ne pas redessiner sur un ViewBox déjà détruit par un rebuild de grille.
        self._range_ctrl.detach()
        before = len(self._range_ctrl.bars or ())
        self._range_ctrl.add_bar()
        bars = self._range_ctrl.bars or ()
        if len(bars) <= before:
            return
        self._active_range_index = len(self._all_range_bars()) - 1
        self._update_range_counts()
        self._sync_settings_from_bars()
        self._sync_analysis_range_ctrl_from_list()
        # Réattacher sur un host vivant (évite un redraw sur ViewBox détruite).
        self._attach_range_bars_to_full_recording()
        if self._zooms_applied:
            self._rebuild_placements(reopen_zooms=True)
            self.refreshRequested.emit(self)
        t0, t1 = bars[-1].ordered()
        self._status.setText(
            f"Plage absolue [{t0:g} … {t1:g}] s — « Appliquer les zooms » pour ouvrir."
        )

    def _add_relative_range(self) -> None:
        """+ Rel. stim : dialogue custom — n’ouvre pas les zooms."""
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
        color_index = len(self._all_range_bars())
        self._relative_ranges.append(
            TimeRangeBar(
                t0_s=float(spec.t0_s),
                t1_s=float(spec.t1_s),
                label=label,
                bar_id=uuid.uuid4().hex[:8],
                relative_to_stim=True,
                color=default_zoom_range_color(color_index),
            )
        )
        self._active_range_index = len(self._all_range_bars()) - 1
        self._update_range_counts()
        self._sync_settings_from_bars()
        self._sync_analysis_range_ctrl_from_list()
        if self.is_analysis_view():
            self._attach_range_bars_to_analysis()
        else:
            self._attach_relative_bars_to_continuous()
        if self._zooms_applied:
            self._rebuild_placements(reopen_zooms=True)
            self.refreshRequested.emit(self)
        self._status.setText(
            f"Plage relative [{spec.t0_s:g} … {spec.t1_s:g}] s — "
            "« Appliquer les zooms » pour ouvrir."
        )

    def _remove_bar(self) -> None:
        bars = list(self._all_range_bars())
        if not bars:
            QMessageBox.information(self, "Plages", "Aucune plage à supprimer.")
            return
        if self._active_range_index < 0:
            QMessageBox.information(
                self,
                "Plages",
                "Aucune plage sélectionnée — cliquez une plage (liste ou bande).",
            )
            return
        idx = max(0, min(len(bars) - 1, self._active_range_index))
        target = bars[idx]
        if target.relative_to_stim:
            # Retirer dans _relative_ranges (même bar_id ou même index relatif).
            rel_idx = sum(1 for b in bars[:idx] if b.relative_to_stim)
            if 0 <= rel_idx < len(self._relative_ranges):
                self._relative_ranges.pop(rel_idx)
        else:
            abs_idx = sum(1 for b in bars[:idx] if not b.relative_to_stim)
            abs_bars = list(self._range_ctrl.bars or ())
            if 0 <= abs_idx < len(abs_bars):
                abs_bars.pop(abs_idx)
                self._range_ctrl.detach()
                self._range_ctrl.set_bars(abs_bars, active_index=max(0, abs_idx - 1))
        remaining = self._all_range_bars()
        if not remaining:
            self._active_range_index = -1
            self._zooms_applied = False
        else:
            self._active_range_index = max(0, min(len(remaining) - 1, idx))
        self._update_range_counts()
        self._sync_settings_from_bars()
        self._sync_analysis_range_ctrl_from_list()
        self._rebuild_placements(reopen_zooms=False)
        self.refreshRequested.emit(self)

    def _on_active_changed(self, index: int) -> None:
        bars = self._all_range_bars()
        if not bars:
            self._active_range_index = -1
            self._range_ctrl.set_active_index(-1)
            self._analysis_range_ctrl.set_active_index(-1)
            self._sync_settings_from_bars()
            self._update_range_counts()
            return
        raw = int(index)
        if raw < 0:
            self._active_range_index = -1
            self._range_ctrl.set_active_index(-1)
            self._analysis_range_ctrl.set_active_index(-1)
        else:
            self._active_range_index = max(0, min(len(bars) - 1, raw))
            n_abs = len(self._range_ctrl.bars or ())
            if self.is_analysis_view():
                # Le contrôleur d’analyse porte absolues puis relatives, dans cet ordre.
                self._analysis_range_ctrl.set_active_index(self._active_range_index)
            elif self._active_range_index < n_abs:
                self._range_ctrl.set_active_index(self._active_range_index)
                self._analysis_range_ctrl.set_active_index(-1)
            else:
                self._range_ctrl.set_active_index(-1)
                self._analysis_range_ctrl.set_active_index(
                    self._active_range_index - n_abs
                )
        self._sync_settings_from_bars()
        self._update_range_counts()

    def _on_range_ctrl_active_changed(self, local_index: int) -> None:
        """Sélection / désélection depuis une bande continuous (abs)."""
        if int(local_index) >= 0:
            self._on_active_changed(int(local_index))
            return
        # Clic à côté : ne vider le global que si l’autre contrôleur n’est pas actif.
        if int(self._analysis_range_ctrl.active_index) < 0:
            self._on_active_changed(-1)

    def _on_analysis_ctrl_active_changed(self, local_index: int) -> None:
        """Sélection / désélection depuis une bande analyse / relative."""
        if int(local_index) >= 0:
            if self.is_analysis_view():
                self._on_active_changed(int(local_index))
            else:
                n_abs = len(self._range_ctrl.bars or ())
                self._on_active_changed(n_abs + int(local_index))
            return
        if int(self._range_ctrl.active_index) < 0:
            self._on_active_changed(-1)

    def _on_range_color_changed(self, color: str) -> None:
        """Compat nuancier legacy → délègue à l’édition de la plage active."""
        bars = list(self._all_range_bars())
        if not bars or self._active_range_index < 0:
            return
        idx = max(0, min(len(bars) - 1, self._active_range_index))
        self._on_bar_edited(idx, bars[idx].with_color(str(color)))

    def _on_bar_edited(self, index: int, bar: object) -> None:
        """Édition depuis la liste Plages (couleur, nom, t0/t1)."""
        if not isinstance(bar, TimeRangeBar):
            return
        bars = list(self._all_range_bars())
        if not bars:
            return
        idx = max(0, min(len(bars) - 1, int(index)))
        target = replace(bar)
        t0, t1 = target.ordered()
        if t1 <= t0:
            t1 = t0 + 1e-3
            target = target.with_bounds(t0, t1)
        # Conserver le type abs/rel et l’id de la plage d’origine.
        previous = bars[idx]
        target = replace(
            target,
            bar_id=previous.bar_id or target.bar_id,
            relative_to_stim=bool(previous.relative_to_stim),
        )
        if target.relative_to_stim:
            rel_idx = sum(1 for b in bars[:idx] if b.relative_to_stim)
            if 0 <= rel_idx < len(self._relative_ranges):
                self._relative_ranges[rel_idx] = target
        else:
            abs_idx = sum(1 for b in bars[:idx] if not b.relative_to_stim)
            abs_bars = list(self._range_ctrl.bars or ())
            if 0 <= abs_idx < len(abs_bars):
                abs_bars[abs_idx] = target
                self._range_ctrl.set_bars(abs_bars, active_index=abs_idx)
        self._active_range_index = idx
        self._sync_settings_from_bars()
        self._sync_analysis_range_ctrl_from_list()
        self._update_range_counts()
        if self.is_analysis_view():
            self._attach_range_bars_to_analysis()
        elif target.relative_to_stim:
            self._attach_relative_bars_to_continuous()
        else:
            self._attach_range_bars_to_full_recording()
        if self._zooms_applied:
            self._rebuild_placements(reopen_zooms=False)
            self.refreshRequested.emit(self)

    def _on_bars_changed(self, _bars: object) -> None:
        self._update_range_counts()
        self._sync_settings_from_bars()
        self._bars_debouncer.request()

    def _on_analysis_bars_changed(self, bars: object) -> None:
        """Glisser une barre Analyse → resynchroniser la liste partagée."""
        shared = [replace(b) for b in bars]  # type: ignore[arg-type]
        if not self.is_analysis_view():
            stim = self._stim_reference_s()
            if stim is None:
                return
            self._relative_ranges = [
                replace(bar, t0_s=float(bar.t0_s) - float(stim), t1_s=float(bar.t1_s) - float(stim))
                for bar in shared
                if bar.relative_to_stim
            ]
            n_abs = len(self._range_ctrl.bars or ())
            local = int(self._analysis_range_ctrl.active_index)
            self._active_range_index = -1 if local < 0 else n_abs + local
            self._update_range_counts()
            self._analysis_bars_debouncer.request()
            return
        # Graph analyse = t_rel. Les plages « absolues » y sont affichées converties :
        # au drop, reconvertir vers temps fichier via la stim locale d’origine.
        prev_abs = {
            str(b.bar_id): b for b in (self._range_ctrl.bars or ()) if b.bar_id
        }
        abs_bars: list[TimeRangeBar] = []
        rel_bars: list[TimeRangeBar] = []
        for bar in shared:
            if bar.relative_to_stim:
                rel_bars.append(replace(bar))
                continue
            prev = prev_abs.get(str(bar.bar_id or ""))
            ref = None
            if prev is not None:
                ref = self._local_stim_for_absolute_range(*prev.ordered())
            if ref is None:
                ref = self._stim_reference_s()
            if ref is None:
                continue
            abs_bars.append(
                replace(
                    bar,
                    t0_s=float(bar.t0_s) + float(ref),
                    t1_s=float(bar.t1_s) + float(ref),
                    relative_to_stim=False,
                )
            )
        try:
            self._range_ctrl.barsChanged.disconnect(self._on_bars_changed)
        except (TypeError, RuntimeError):
            pass
        self._range_ctrl.set_bars(
            abs_bars, active_index=min(self._active_range_index, max(0, len(abs_bars) - 1))
        )
        self._range_ctrl.barsChanged.connect(self._on_bars_changed)
        self._relative_ranges = rel_bars
        self._active_range_index = self._analysis_range_ctrl.active_index
        self._update_range_counts()
        self._analysis_bars_debouncer.request()

    def _window_from_temporal_host(
        self, host: Any
    ) -> tuple[tuple[float, float] | None, bool]:
        """Fenêtre affichée d’un graph temporel, ou (None, …) si elle couvre tout.

        Le second booléen est True quand les bornes sont en temps fichier.
        """
        if not _qt_alive(host):
            return None, False
        try:
            canvas = host.parentWidget()
        except RuntimeError:
            return None, False
        placement = getattr(canvas, "placement", None)
        view = _host_view_xlim(host)
        data = None
        reader = getattr(host, "data_xlim", None)
        if callable(reader):
            try:
                data = _as_time_span(reader())
            except Exception:
                data = None
        if view is None or _view_covers_data(view, data):
            return None, False
        absolute = placement is not None and str(placement.panel) == "full_recording"
        if absolute:
            offset = continuous_sync_offset_s(
                self._stim_times_s, self.settings.time_sync
            )
            return (float(view[0]) + offset, float(view[1]) + offset), True
        return view, False

    def _displayed_spike_window_for_grid(
        self, grid: Any
    ) -> tuple[tuple[float, float] | None, bool]:
        host = self._spike_scope_view_host
        if not _qt_alive(host):
            self._spike_scope_view_host = None
            host = None
        if host is not None and _grid_of(host) is grid:
            return self._window_from_temporal_host(host)
        panels = getattr(grid, "_panels", {}) or {}
        for widget in panels.values():
            placement = getattr(widget, "placement", None)
            if placement is None or str(placement.panel) not in _SPIKE_SCOPE_TIME_PANELS:
                continue
            candidate = getattr(widget, "plot_host", None) or getattr(widget, "_plot", None)
            if not _qt_alive(candidate):
                continue
            return self._window_from_temporal_host(candidate)
        return None, False

    def _bind_spike_scope_views(self, grid: Any) -> None:
        panels = getattr(grid, "_panels", {}) or {}
        for widget in panels.values():
            placement = getattr(widget, "placement", None)
            if placement is None or str(placement.panel) not in _SPIKE_SCOPE_TIME_PANELS:
                continue
            host = getattr(widget, "plot_host", None) or getattr(widget, "_plot", None)
            if not _qt_alive(host):
                continue
            signal = getattr(host, "viewChanged", None)
            # PySide avertit (RuntimeWarning) si on déconnecte un slot jamais branché.
            if signal is not None and not getattr(host, "_spike_view_bound", False):
                signal.connect(self._on_temporal_view_changed)
                host._spike_view_bound = True
            self._bind_temporal_x_range(host)

    def _bind_temporal_x_range(self, host: Any) -> None:
        """Tout changement d’axe X (après le premier affichage) relance la détection."""
        if not _qt_alive(host) or getattr(host, "_spike_x_bound", False):
            return
        plots = list(getattr(host, "plot_items", lambda: [])() or [])
        if not plots:
            return
        bound = False
        for plot in plots:
            try:
                vb = plot.getViewBox()
            except Exception:
                continue
            signal = getattr(vb, "sigXRangeChanged", None)
            if signal is None:
                continue
            signal.connect(lambda *_a, h=host: self._note_temporal_host(h))
            bound = True
        host._spike_x_bound = bound

    def _note_temporal_host(self, host: Any) -> None:
        if not _qt_alive(host) or self._spike_scope_refreshing:
            return
        self._spike_scope_view_host = host
        self._spike_view_debouncer.request()

    def _on_temporal_view_changed(self) -> None:
        self._note_temporal_host(self.sender())

    def _zoom_window_owning_grid(self, grid: Any) -> DetachedZoomWindow | None:
        for win in list(self._zoom_windows.values()):
            if getattr(win, "grid", None) is grid:
                return win
        return None

    def _refresh_spike_scope_for_displayed_window(self) -> None:
        """Recalculer le spike scope de la grille dont la fenêtre de temps a bougé."""
        if self._request_factory is None or self._spike_scope_refreshing:
            return
        host = self._spike_scope_view_host
        if not _qt_alive(host):
            self._spike_scope_view_host = None
            return
        grid = _grid_of(host)
        if grid is None:
            return
        window, absolute = self._window_from_temporal_host(host)
        key = (
            None
            if window is None
            else (round(float(window[0]), 6), round(float(window[1]), 6)),
            bool(absolute),
        )
        if self._spike_scope_window_key.get(id(grid)) == key:
            return
        self._spike_scope_window_key[id(grid)] = key
        self._rerender_spike_scopes(grid, window, absolute)

    def _rerender_spike_scopes(
        self,
        grid: Any,
        window: tuple[float, float] | None,
        absolute: bool,
    ) -> None:
        panels = getattr(grid, "_panels", {}) or {}
        targets = [
            widget
            for widget in panels.values()
            if str(getattr(getattr(widget, "placement", None), "panel", ""))
            in _SPIKE_SCOPE_PANELS
        ]
        if not targets or self._request_factory is None:
            return
        zoom = self._zoom_window_owning_grid(grid)
        self._spike_scope_refreshing = True
        try:
            for widget in targets:
                placement = widget.placement
                request = self._request_factory(self, placement)  # type: ignore[misc]
                if request is None:
                    continue
                if zoom is not None:
                    request = replace(
                        request,
                        settings=settings_for_zoom_window(zoom.local_settings()),
                        highlight_zooms=(),
                    )
                widget.render(
                    replace(
                        request,
                        preserve_view=True,
                        visible_time_s=window,
                        visible_time_absolute=absolute,
                    )
                )
        finally:
            self._spike_scope_refreshing = False

    def _refresh_after_range_moved(self) -> None:
        """PSTH / raster suivent la plage. Le spike scope suit la fenêtre affichée."""
        self._preserve_view = True
        self._sync_settings_from_bars()
        self.refreshRequested.emit(self)

    def _after_bars_moved(self) -> None:
        """Déplacement des barres continuous → zooms (si appliqués)."""
        if not self._zooms_applied:
            self._refresh_after_range_moved()
            return
        if self.is_analysis_view():
            self._after_analysis_bars_moved()
            return
        _unused, zoom_panels = self._continuous_zoom_panel_lists(for_analysis=True)
        self._zoom_placements = tuple(zoom_panels)
        # Conserver les panels RMS/spikes déjà dans la grille (prefetch).
        analysis = self._analysis_from_toggles()
        self._analysis_placements = tuple(
            PanelPlacement(key, section="full")
            for key in analysis.selected_continuous_extra_panels()
        )
        self._sync_zoom_windows(reopen=False)
        for zoom in list(self._zoom_windows.values()):
            self._render_zoom_window(zoom)
        self._sync_zoom_shared_ylims()
        self._status.setText("Barres déplacées — zooms actualisés.")
        self._refresh_after_range_moved()

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
            label = resolved_zone_label(bar, index)
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


    def _after_analysis_bars_moved(self) -> None:
        """Déplacement des barres → zooms analyse (fenêtre détachée)."""
        if not self._zooms_applied:
            self._refresh_after_range_moved()
            return
        analysis = self._ensure_analysis_curves()
        analysis_panels: list[PanelPlacement] = []
        zoom_panels: list[PanelPlacement] = []
        for index, bar in enumerate(self._all_range_bars()):
            t0, t1 = bar.ordered()
            label = resolved_zone_label(bar, index)
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
        self._zoom_placements = tuple(zoom_panels)
        self._sync_zoom_windows(reopen=False)
        for zoom in list(self._zoom_windows.values()):
            self._render_zoom_window(zoom)
        self._sync_zoom_shared_ylims()
        self._status.setText("Barres déplacées — zooms actualisés.")
        self._refresh_after_range_moved()

    def _sync_analysis_range_ctrl_from_list(self) -> None:
        """Pousser les plages partagées vers le controller analyse (coords t_rel)."""
        ctrl = self._analysis_range_ctrl
        try:
            ctrl.barsChanged.disconnect(self._on_analysis_bars_changed)
        except (TypeError, RuntimeError):
            pass
        # Toujours en t_rel sur le graph analyse (abs converties via stim locale).
        bars = self._bars_for_analysis_display()
        active = int(self._active_range_index)
        if not bars:
            active = -1
        elif active >= 0:
            active = max(0, min(len(bars) - 1, active))
        else:
            active = -1
        ctrl.set_bars(bars, active_index=active)
        ctrl.barsChanged.connect(self._on_analysis_bars_changed)


    def _attach_range_bars_to_analysis(self) -> None:
        """Réattacher les barres sur le premier graph temporel d'analyse (grille principale)."""
        if not self.is_analysis_view():
            self._analysis_range_ctrl.detach()
            return
        host = None
        for placement in self._placements:
            if placement.panel not in _ANALYSIS_TEMPORAL_PANELS:
                continue
            if placement.has_custom_zoom:
                continue
            widget = self.grid.panel_widget(placement)
            if widget is None:
                continue
            candidate = getattr(widget, "plot_host", None) or getattr(
                widget, "_plot", None
            )
            if candidate is None:
                continue
            plots = list(getattr(candidate, "plot_items", lambda: [])() or [])
            if not plots:
                continue
            host = candidate
            break
        if host is None:
            self._analysis_range_ctrl.detach()
            return
        self._analysis_range_ctrl.set_display_offset(0.0)
        try:
            axes = list(getattr(getattr(host, "figure", None), "axes", []) or [])
            if axes:
                xlim = axes[0].get_xlim()
                self._analysis_range_ctrl.set_time_span(float(xlim[0]), float(xlim[1]))
        except Exception:
            pass
        self._sync_analysis_range_ctrl_from_list()
        self._analysis_range_ctrl.set_zone_label_style(self.settings.style)
        self._analysis_range_ctrl.attach(host)

    def _sync_settings_from_bars(self) -> None:
        self.settings = self.local_settings()
        self._sync_params_base()
        self._sync_zone_label_style()

    def _sync_zone_label_style(self) -> None:
        """Police / placement des libellés Zone N (Affichage, toutes les vues)."""
        style = getattr(self.settings, "style", None)
        self._range_ctrl.set_zone_label_style(style)
        self._analysis_range_ctrl.set_zone_label_style(style)

    def _configure_preview_grid(self) -> None:
        """Hauteur fixe pour continuous / moyenne / stimulation (pas de fill viewport)."""
        self.grid.configure(
            self._placements,
            columns=1,
            panel_height=self._panel_height(),
            uniform=True,
            fill=False,
        )

    def _panel_height(self) -> int:
        """Hauteur fixe de chaque graphique (réglage Affichage → Hauteur des graphs)."""
        return max(160, int(getattr(self.settings, "graph_height_px", 400) or 400))

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
            "analysis_raster_channel": self._cb_raster,
            "analysis_raster": self._cb_raster_trials,
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
        # Ne pas doubler un panneau déjà présent dans une fenêtre de zooms.
        for zoom in self._zoom_windows.values():
            if zoom.contains_key(key):
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
            "analysis_raster_channel": self._cb_raster,
            "analysis_raster": self._cb_raster_trials,
        }
        return mapping.get(str(panel))

    def _on_panel_removed(self, placement: PanelPlacement) -> None:
        # Continuous : retirer un flux = décocher WIDE / HIGH / LOW.
        if placement.panel == "full_recording" and not placement.has_custom_zoom:
            stream_box = {
                "raw": self._cb_raw,
                "hp": self._cb_hp,
                "lp": self._cb_lp,
            }.get(str(placement.stream or "raw"))
            if stream_box is not None and stream_box.isChecked():
                stream_box.blockSignals(True)
                stream_box.setChecked(False)
                stream_box.blockSignals(False)
                # Garder au moins un flux.
                if not any(
                    box.isChecked()
                    for box in (self._cb_raw, self._cb_hp, self._cb_lp)
                ):
                    self._cb_raw.blockSignals(True)
                    self._cb_raw.setChecked(True)
                    self._cb_raw.blockSignals(False)
                self._on_curves_toggled()
            return
        box = self._checkbox_for_panel(placement.panel)
        if box is None:
            box = self._checkbox_for_analysis_panel(placement.panel)
        if box is not None and box.isChecked() and not placement.has_custom_zoom:
            box.blockSignals(True)
            box.setChecked(False)
            box.blockSignals(False)
            self.settings = self.local_settings()
            analysis = self.settings.analysis
            if not analysis.selected_analysis_panels() and self.is_analysis_view():
                self._cb_raw.blockSignals(True)
                self._cb_raw.setChecked(True)
                self._cb_raw.blockSignals(False)
                self.settings = self.local_settings()
                analysis = self.settings.analysis
            self._rebuild_placements()
            self.refreshRequested.emit(self)
            if self._embedded:
                self.analysisCurvesChanged.emit(analysis)
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
        self._sync_zone_label_style()
        try:
            self.grid.sync_scale_bars_from(self.settings)
        except Exception:
            pass
        if self._placements:
            # Appliquer la hauteur fixe sans reconstruire la grille.
            height = self._panel_height()
            self.grid.set_panel_height(height, uniform=True, fill=False)
            self.grid.set_graph_height(height)
        if self._request_factory is None:
            return
        keep = bool(preserve_view) or bool(self._preserve_view)
        self._preserve_view = keep

        def factory(placement: PanelPlacement) -> RenderRequest:
            # Zoom custom = bornes de plage : toujours les appliquer.
            preserve = keep and not placement.has_custom_zoom
            request = self._request_factory(self, placement)  # type: ignore[misc]
            if request is not None:
                if (
                    preserve
                    and str(placement.panel) in _SPIKE_SCOPE_PANELS
                ):
                    window, absolute = self._displayed_spike_window_for_grid(self.grid)
                    request = replace(
                        request,
                        visible_time_s=window,
                        visible_time_absolute=absolute,
                    )
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
        for zoom in list(self._zoom_windows.values()):
            self._render_zoom_window(zoom)
        self._sync_zoom_shared_ylims()
        for detached in list(self._detached.values()):
            self._render_zoom_window(detached)

    def _on_render_finished(self, drawn: int, elapsed_s: float) -> None:
        self._preserve_view = False
        self._bind_spike_scope_views(self.grid)
        n_zoom = len(self._zoom_placements) if self._zoom_windows else 0
        n_an = len(self._analysis_placements)
        parts: list[str] = []
        if n_zoom:
            parts.append(f"{n_zoom} zoom(s)")
        if n_an:
            parts.append(f"{n_an} analyse")
        suffix = (" · " + " · ".join(parts)) if parts else ""
        self._status.setText(f"{drawn} panneau(x) · {elapsed_s * 1000:.0f} ms{suffix}")
        if self.is_analysis_view():
            # Le host continuous peut avoir été détruit au changement de mode.
            self._range_ctrl.detach()
            # Ne pas réattacher pendant un glisser (sinon le drag est annulé).
            if not self._analysis_range_ctrl.is_dragging:
                self._attach_range_bars_to_analysis()
        else:
            if not self._range_ctrl.is_dragging:
                self._attach_range_bars_to_full_recording()
            if not self._analysis_range_ctrl.is_dragging:
                self._attach_relative_bars_to_continuous()

    def _attach_range_bars_to_full_recording(self) -> None:
        """Réattacher les barres interactives sur le panneau brut (non zoomé)."""
        for placement in self._placements:
            if placement.panel != "full_recording" or placement.has_custom_zoom:
                continue
            widget = self.grid.panel_widget(placement)
            if widget is None:
                continue
            host = getattr(widget, "plot_host", None) or getattr(widget, "_plot", None)
            if host is None:
                continue
            plots = list(getattr(host, "plot_items", lambda: [])() or [])
            if not plots:
                continue
            offset = continuous_sync_offset_s(
                self._stim_times_s, self.settings.time_sync
            )
            self._range_ctrl.set_display_offset(offset)
            # Limites X de l’axe = coords d’affichage → bornes absolues pour le stockage.
            try:
                axes = list(getattr(getattr(host, "figure", None), "axes", []) or [])
                if axes:
                    xlim = axes[0].get_xlim()
                    self._range_ctrl.set_time_span(
                        float(xlim[0]) + offset, float(xlim[1]) + offset
                    )
            except Exception:
                pass
            self._range_ctrl.set_zone_label_style(self.settings.style)
            self._range_ctrl.attach(host)
            break

    def _attach_relative_bars_to_continuous(self) -> None:
        """Dessiner les plages relatives sur l’axe continu (temps fichier)."""
        if self.is_analysis_view():
            return
        host = None
        offset = 0.0
        for placement in self._placements:
            if placement.panel != "full_recording" or placement.has_custom_zoom:
                continue
            widget = self.grid.panel_widget(placement)
            if widget is None:
                continue
            candidate = getattr(widget, "plot_host", None) or getattr(widget, "_plot", None)
            plots = list(getattr(candidate, "plot_items", lambda: [])() or [])
            if candidate is None or not plots:
                continue
            host = candidate
            offset = continuous_sync_offset_s(self._stim_times_s, self.settings.time_sync)
            break
        rel = list(self._relative_ranges)
        stim = self._stim_reference_s()
        if host is None or not rel or stim is None:
            self._analysis_range_ctrl.detach()
            return
        shown = [
            replace(bar, t0_s=float(stim) + float(bar.t0_s), t1_s=float(stim) + float(bar.t1_s))
            for bar in rel
        ]
        n_abs = len(self._range_ctrl.bars or ())
        active = self._active_range_index - n_abs
        try:
            self._analysis_range_ctrl.barsChanged.disconnect(self._on_analysis_bars_changed)
        except (TypeError, RuntimeError):
            pass
        self._analysis_range_ctrl.set_display_offset(offset)
        rel_active = -1 if active < 0 else max(0, active)
        self._analysis_range_ctrl.set_bars(shown, active_index=rel_active)
        if active < 0:
            self._analysis_range_ctrl.set_active_index(-1)
        self._analysis_range_ctrl.barsChanged.connect(self._on_analysis_bars_changed)
        self._analysis_range_ctrl.set_zone_label_style(self.settings.style)
        self._analysis_range_ctrl.attach(host)

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
