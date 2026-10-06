"""Fenêtre d’inspection d’un canal : brut + barres de plage + graphs d’analyse."""

from __future__ import annotations

import itertools
import uuid
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
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.jobs import Debouncer
from gui.widgets.custom_zoom_dialog import ask_custom_zoom
from gui.widgets.panel_grid import PanelGrid
from gui.widgets.range_bars import RangeBarController, RangeBarToolbar
from gui.widgets.view_params import LocalViewParams
from panel_registry import RenderRequest
from view_config import (
    AnalysisSettings,
    AnalysisStream,
    PanelPlacement,
    TimeRangeBar,
    ViewerSettings,
)

_WINDOW_COUNTER = itertools.count(1)

RequestFactory = Callable[["ChannelAnalysisWindow", PanelPlacement], RenderRequest | None]


class ChannelAnalysisWindow(QMainWindow):
    """Inspection d’un canal : traces continues + barres locales + graphs d’analyse.

    Les barres de plage n’existent que dans cette fenêtre (pas sur le montage global).
    """

    closed = Signal(str)
    refreshRequested = Signal(object)

    def __init__(
        self,
        *,
        channel_name: str,
        channel_index: int,
        analysis: AnalysisSettings,
        base_settings: ViewerSettings,
        mode_label: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.window_id = f"chan-{next(_WINDOW_COUNTER)}-{uuid.uuid4().hex[:6]}"
        self.channel_name = str(channel_name)
        self.channel_index = int(channel_index)
        self._mode_label = mode_label or analysis.describe()
        self.settings = replace(base_settings, analysis=analysis)
        self._processed_applied = False
        self._stream_updating = False
        self._stim_times_s: tuple[float, ...] = ()
        self._relative_ranges: list[TimeRangeBar] = []
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle(f"Inspecter {self.channel_name} — {self._mode_label}")
        self.resize(1400, 900)

        self._status = QLabel(
            "Plages absolues sur le brut, ou « + Rel. stim » pour [t₀, t₁] "
            "relatifs à la stimulation (moyenne / une stim)."
        )
        self._status.setObjectName("panelStatus")
        self._status.setWordWrap(True)
        self._n_trials = max(1, int(analysis.stim_index) + 1)
        self._title_label = QLabel(f"<b>{self.channel_name}</b><br>{self._mode_label}")
        self._title_label.setObjectName("sectionTitle")
        self._title_label.setTextFormat(Qt.TextFormat.RichText)
        self._channel_ready = False

        # --- mode moyenne / stimulation ---
        mode_box = QGroupBox("Mode d’analyse", self)
        mode_form = QFormLayout(mode_box)
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
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._stim_spin.valueChanged.connect(self._on_mode_changed)
        self._sync_stim_enabled()

        # --- flux continus (aperçu + barres) ---
        streams_box = QGroupBox("Traces continues", self)
        streams_layout = QVBoxLayout(streams_box)
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

        # --- contexte ---
        context_box = QGroupBox("Contexte", self)
        context_layout = QVBoxLayout(context_box)
        self._cb_mea = QCheckBox("Carte MEA")
        self._cb_impedance = QCheckBox("Impédance")
        for box in (self._cb_mea, self._cb_impedance):
            context_layout.addWidget(box)
            box.toggled.connect(self._on_extra_toggles_changed)

        # --- graphs d’analyse (plein + zooms après « Appliquer les zooms ») ---
        toggles = QGroupBox("Graphs d’analyse", self)
        toggle_layout = QVBoxLayout(toggles)
        self._cb_raw = QCheckBox("WIDE (brut)")
        self._cb_hp = QCheckBox("HIGH (passe-haut)")
        self._cb_lp = QCheckBox("LOW (passe-bas)")
        self._cb_rms = QCheckBox("RMS")
        self._cb_raster = QCheckBox("Raster")
        self._cb_psth = QCheckBox("PSTH")
        self._cb_isi = QCheckBox("ISI")
        self._cb_overlay = QCheckBox("Superposition spikes")
        self._cb_raw.setChecked(analysis.show_raw)
        self._cb_hp.setChecked(analysis.show_hp)
        self._cb_lp.setChecked(analysis.show_lp)
        self._cb_rms.setChecked(analysis.show_rms)
        self._cb_raster.setChecked(analysis.show_raster)
        self._cb_psth.setChecked(analysis.show_psth)
        self._cb_isi.setChecked(analysis.show_isi)
        self._cb_overlay.setChecked(analysis.show_overlay)
        for box in (
            self._cb_raw,
            self._cb_hp,
            self._cb_lp,
            self._cb_rms,
            self._cb_raster,
            self._cb_psth,
            self._cb_isi,
            self._cb_overlay,
        ):
            toggle_layout.addWidget(box)
            box.toggled.connect(self._on_toggles_changed)

        self._range_toolbar = RangeBarToolbar(self)
        self._range_ctrl = RangeBarController(self)
        self._range_ctrl.barsChanged.connect(self._on_bars_changed)
        self._range_toolbar.addRequested.connect(self._add_bar)
        self._range_toolbar.addRelativeRequested.connect(self._add_relative_range)
        self._range_toolbar.removeRequested.connect(self._remove_bar)
        self._range_toolbar.activeChanged.connect(self._on_active_changed)
        self._range_toolbar.processRequested.connect(self.apply_processing)
        self._relative_label = QLabel("0 plage(s) relative(s) à la stim", self)
        self._relative_label.setObjectName("hintLabel")

        self._btn_redraw = QPushButton("Redessiner", self)
        self._btn_redraw.clicked.connect(lambda: self.refreshRequested.emit(self))

        # Légende / style locaux à cette fenêtre (partagés par tous ses graphs).
        self.params = LocalViewParams(
            self.settings,
            show_analysis=False,
            show_continuous=False,
            show_display_extras=True,
            parent=self,
        )
        self.params.changed.connect(self._on_local_params_changed)

        side = QWidget(self)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(6, 6, 6, 6)
        side_layout.addWidget(self._title_label)
        side_layout.addWidget(mode_box)
        side_layout.addWidget(self._range_toolbar)
        side_layout.addWidget(self._relative_label)
        side_layout.addWidget(streams_box)
        side_layout.addWidget(context_box)
        side_layout.addWidget(toggles)
        side_layout.addWidget(self._btn_redraw)
        side_layout.addWidget(self.params, 1)
        side_layout.addWidget(self._status)

        self.grid = PanelGrid(self)
        self.grid.removeRequested.connect(self._on_panel_removed)
        self.grid.detachRequested.connect(lambda _p: None)
        self.grid.renderFinished.connect(self._on_render_finished)

        header = QHBoxLayout()
        header.addWidget(
            QLabel(
                "Traces continues + graphs cochés. "
                "« Appliquer les zooms » crée un zoom par plage "
                "(absolue ou relative à la stim).",
                self,
            ),
            1,
        )

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
        side.setMinimumWidth(260)
        side.setMaximumWidth(16777215)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 400])
        self.setCentralWidget(splitter)

        self._request_factory: RequestFactory | None = None
        self._bars_debouncer = Debouncer(80, self)
        self._bars_debouncer.triggered.connect(self._after_bars_moved)
        self._rebuild_placements(apply_zooms=False)

    # ---------------------------------------------------------------- wiring

    def set_request_factory(self, factory: RequestFactory) -> None:
        self._request_factory = factory

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
        base = self.params.settings()
        return replace(
            base,
            analysis=self._analysis_from_toggles(),
            continuous_stream=streams[0],
            continuous_streams=streams,
            continuous_mark_stims=self._cb_mark_stims.isChecked(),
            range_bars=self._all_range_bars(),
            active_range_index=self._range_ctrl.active_index,
        )

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
        self.params.sync_base_settings(self.settings)
        # Recréer la grille : un axe par flux coché.
        self._rebuild_placements(apply_zooms=self._processed_applied)
        self.refreshRequested.emit(self)

    def _on_stim_markers_changed(self, *_args: Any) -> None:
        """Bascule des marqueurs stim : redessiner sans reset zoom/position."""
        self._preserve_view = True
        self.settings = self.local_settings()
        self.params.sync_base_settings(self.settings)
        self.refreshRequested.emit(self)

    def needs_channel_compute(self) -> bool:
        return not self._channel_ready

    def set_channel_ready(self, ready: bool) -> None:
        self._channel_ready = bool(ready)

    @property
    def placements(self) -> tuple[PanelPlacement, ...]:
        return self._placements

    # ----------------------------------------------------------- placements

    def _analysis_from_toggles(self) -> AnalysisSettings:
        base = self.settings.analysis
        mode = str(self._mode_combo.currentData() or "average")
        return replace(
            base,
            mode=mode,  # type: ignore[arg-type]
            stim_index=max(0, int(self._stim_spin.value()) - 1),
            show_raw=self._cb_raw.isChecked(),
            show_hp=self._cb_hp.isChecked(),
            show_lp=self._cb_lp.isChecked(),
            show_rms=self._cb_rms.isChecked(),
            show_raster=self._cb_raster.isChecked(),
            show_psth=self._cb_psth.isChecked(),
            show_isi=self._cb_isi.isChecked(),
            show_overlay=self._cb_overlay.isChecked(),
        )

    def _sync_stim_enabled(self) -> None:
        is_stim = str(self._mode_combo.currentData() or "") == "stimulation"
        self._stim_spin.setEnabled(is_stim)

    def _mode_label_from_ui(self) -> str:
        analysis = self._analysis_from_toggles()
        return analysis.describe()

    def _on_mode_changed(self, *_args: Any) -> None:
        self._sync_stim_enabled()
        self._mode_label = self._mode_label_from_ui()
        self._title_label.setText(f"<b>{self.channel_name}</b><br>{self._mode_label}")
        self.setWindowTitle(f"Inspecter {self.channel_name} — {self._mode_label}")
        self.settings = self.local_settings()
        self.params.sync_base_settings(self.settings)
        if self._processed_applied:
            self._rebuild_placements(apply_zooms=True)
        self.refreshRequested.emit(self)

    def _context_panels(self) -> tuple[str, ...]:
        panels: list[str] = []
        if self._cb_mea.isChecked():
            panels.append("mea_layout")
        if self._cb_impedance.isChecked():
            panels.append("impedance")
        return tuple(panels)

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

    def _append_zoom_panels(
        self,
        panels: list[PanelPlacement],
        *,
        analysis: AnalysisSettings,
        t0: float,
        t1: float,
        label: str,
        bar_id: str,
        absolute: bool,
    ) -> None:
        if absolute:
            panels.append(
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
                panels.append(
                    PanelPlacement("full_recording").with_custom_zoom(
                        stim_t + t0,
                        stim_t + t1,
                        label=f"Zoom brut {label}",
                        instance_id=bar_id,
                        absolute=True,
                    )
                )
        for key in analysis.selected_analysis_panels():
            panels.append(
                PanelPlacement(key).with_custom_zoom(
                    t0,
                    t1,
                    label=label,
                    instance_id=bar_id,
                    absolute=absolute,
                )
            )

    def _rebuild_placements(self, *, apply_zooms: bool) -> None:
        panels: list[PanelPlacement] = [PanelPlacement("full_recording")]
        for key in self._context_panels():
            panels.append(PanelPlacement(key))

        analysis = self._analysis_from_toggles()
        # Graphs cochés en fenêtre pleine (remplace l’ancien groupe « moyennes »).
        for key in analysis.selected_analysis_panels():
            panels.append(PanelPlacement(key, section="full"))

        if apply_zooms:
            abs_bars = self._range_ctrl.bars or ()
            for index, bar in enumerate(abs_bars):
                t0, t1 = bar.ordered()
                label = bar.label.strip() or f"Plage {index + 1}"
                bar_id = str(bar.bar_id or f"bar{index}")
                self._append_zoom_panels(
                    panels,
                    analysis=analysis,
                    t0=t0,
                    t1=t1,
                    label=label,
                    bar_id=bar_id,
                    absolute=True,
                )
            for index, bar in enumerate(self._relative_ranges):
                t0, t1 = bar.ordered()
                label = bar.label.strip() or f"Rel. stim {index + 1}"
                bar_id = str(bar.bar_id or f"rel{index}")
                self._append_zoom_panels(
                    panels,
                    analysis=analysis,
                    t0=t0,
                    t1=t1,
                    label=label,
                    bar_id=bar_id,
                    absolute=False,
                )

        self._placements = tuple(panels)
        # Un sous-graphique par flux continu → hauteur proportionnelle.
        self.grid.configure(
            self._placements, columns=1, panel_height=self._panel_height()
        )

    def _on_extra_toggles_changed(self, *_args: Any) -> None:
        self.settings = self.local_settings()
        self.params.sync_base_settings(self.settings)
        self._rebuild_placements(apply_zooms=self._processed_applied)
        self.refreshRequested.emit(self)

    def apply_processing(self) -> None:
        """Bouton « Appliquer les zooms » : ajouter les zooms sur chaque plage."""
        if not self._range_ctrl.bars and not self._relative_ranges:
            self._range_ctrl.ensure_default_bar()
        self._processed_applied = True
        self._sync_settings_from_bars()
        self._rebuild_placements(apply_zooms=True)
        n_abs = len(self._range_ctrl.bars)
        n_rel = len(self._relative_ranges)
        self._status.setText(
            f"{n_abs} plage(s) abs. + {n_rel} relative(s) — zooms ajoutés."
        )
        self.refreshRequested.emit(self)

    # -------------------------------------------------------------- bars UI

    def _update_range_counts(self) -> None:
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

    def _add_bar(self) -> None:
        self._range_ctrl.add_bar()
        self._update_range_counts()
        self._sync_settings_from_bars()
        if self._processed_applied:
            self._rebuild_placements(apply_zooms=True)
            self.refreshRequested.emit(self)

    def _add_relative_range(self) -> None:
        """Saisir une plage [t₀, t₁] relative au début de stimulation."""
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
        self._processed_applied = True
        self._rebuild_placements(apply_zooms=True)
        self._status.setText(
            f"Plage relative [{spec.t0_s:g} … {spec.t1_s:g}] s — zooms ajoutés."
        )
        self.refreshRequested.emit(self)

    def _remove_bar(self) -> None:
        # Priorité : supprimer la plage absolue active ; sinon la dernière relative.
        if self._range_ctrl.bars:
            if len(self._range_ctrl.bars) <= 1 and not self._relative_ranges:
                QMessageBox.information(
                    self, "Plages", "Il doit rester au moins une plage."
                )
                return
            if len(self._range_ctrl.bars) > 1:
                self._range_ctrl.remove_active_bar()
            elif self._relative_ranges:
                self._relative_ranges.pop()
        elif self._relative_ranges:
            self._relative_ranges.pop()
        else:
            QMessageBox.information(self, "Plages", "Aucune plage à supprimer.")
            return
        self._update_range_counts()
        self._sync_settings_from_bars()
        if self._processed_applied:
            self._rebuild_placements(apply_zooms=True)
            self.refreshRequested.emit(self)

    def _on_active_changed(self, index: int) -> None:
        self._range_ctrl.set_active_index(index)
        self._sync_settings_from_bars()

    def _on_bars_changed(self, _bars: object) -> None:
        self._update_range_counts()
        self._sync_settings_from_bars()
        self._bars_debouncer.request()

    def _after_bars_moved(self) -> None:
        """Déplacement des barres → mise à jour auto des graphs dépendants."""
        if not self._processed_applied:
            # Mettre à jour le surlignage sur le brut uniquement.
            self.refreshRequested.emit(self)
            return
        self._rebuild_placements(apply_zooms=True)
        self._status.setText("Barres déplacées — graphs actualisés.")
        self.refreshRequested.emit(self)

    def _sync_settings_from_bars(self) -> None:
        self.settings = self.local_settings()
        self.params.sync_base_settings(self.settings)

    def _on_toggles_changed(self, *_args: Any) -> None:
        self.settings = self.local_settings()
        self.params.sync_base_settings(self.settings)
        self._rebuild_placements(apply_zooms=self._processed_applied)
        self.refreshRequested.emit(self)

    def _panel_height(self) -> int:
        n_streams = max(1, len(self._selected_streams()))
        return max(280, 220 * n_streams)

    def _on_panel_removed(self, placement: PanelPlacement) -> None:
        if placement.panel == "full_recording" and not placement.has_custom_zoom:
            return
        self._placements = tuple(p for p in self._placements if p.key != placement.key)
        if not self._placements:
            self._rebuild_placements(apply_zooms=self._processed_applied)
        else:
            self.grid.configure(
                self._placements, columns=1, panel_height=self._panel_height()
            )
        self.refreshRequested.emit(self)

    # ---------------------------------------------------------------- redraw

    def redraw(self) -> None:
        self.settings = self.local_settings()
        if self._request_factory is None:
            return
        preserve_view = bool(self._preserve_view)

        def factory(placement: PanelPlacement) -> RenderRequest:
            request = self._request_factory(self, placement)  # type: ignore[misc]
            if request is not None:
                if preserve_view:
                    return replace(request, preserve_view=True)
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
                preserve_view=preserve_view,
            )

        self._status.setText("Rendu…")
        self.grid.schedule_render(factory, force=True)

    def _on_render_finished(self, drawn: int, elapsed_s: float) -> None:
        self._preserve_view = False
        self._status.setText(f"{drawn} panneau(x) · {elapsed_s * 1000:.0f} ms")
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
            # Estimer le span temporel depuis les limites X.
            try:
                xlim = axes[0].get_xlim()
                self._range_ctrl.set_time_span(float(xlim[0]), float(xlim[1]))
            except Exception:
                pass
            if not self._range_ctrl.bars:
                self._range_ctrl.ensure_default_bar()
                self._sync_settings_from_bars()
                self._update_range_counts()
            self._range_ctrl.attach(canvas, axes)
            break

    def closeEvent(self, event: Any) -> None:  # noqa: D102
        self._range_ctrl.detach()
        self.closed.emit(self.window_id)
        super().closeEvent(event)
