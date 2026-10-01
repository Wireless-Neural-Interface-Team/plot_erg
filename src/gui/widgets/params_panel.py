"""Every parameter of the analysis, split by how expensive a change is.

- *Display* and *Legend & style* edits only change how cached data is drawn, so
  they emit :attr:`ParamsPanel.viewChanged` and the panels redraw immediately.
- *Processing* edits change what gets computed, so they emit
  :attr:`ParamsPanel.configChanged`; the hierarchical cache means only the
  affected stages are recomputed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import AnalysisConfig
from view_config import (
    LEGEND_LOCATIONS,
    AxisLimits,
    LegendSettings,
    PanelStyle,
    ViewerSettings,
)

_EDGE_CHOICES = (
    ("Front descendant (ANALOG-IN-0)", "falling"),
    ("Front montant (ANALOG-IN-0)", "rising"),
    ("Sans déclencheur (sections fixes)", "none"),
)
_FILTER_TYPES = (("Bessel", "bessel"), ("Butterworth", "butterworth"))
_THRESHOLD_MODES = (("Fixe (µV)", "fixed"), ("Multiple du RMS", "rms_multiple"))
_POLARITIES = (("Négative", "negative"), ("Positive", "positive"))
_SECTION_SPECS = (("Nombre de sections", "count"), ("Durée de section (s)", "duration"))


def _spin(
    minimum: float,
    maximum: float,
    value: float,
    *,
    decimals: int = 3,
    step: float = 0.01,
    suffix: str = "",
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(float(value))
    box.setKeyboardTracking(False)
    if suffix:
        box.setSuffix(suffix)
    return box


def _int_spin(
    minimum: int, maximum: int, value: int, *, step: int = 1, suffix: str = ""
) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(int(value))
    box.setKeyboardTracking(False)
    if suffix:
        box.setSuffix(suffix)
    return box


def _choice(options: tuple[tuple[str, str], ...], value: str) -> QComboBox:
    box = QComboBox()
    for label, key in options:
        box.addItem(label, key)
    index = box.findData(value)
    box.setCurrentIndex(index if index >= 0 else 0)
    return box


def _scrollable(inner: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setWidget(inner)
    return area


class _AxisLimitRow(QWidget):
    """Enable checkbox plus min/max spin boxes for one y-axis family."""

    changed = Signal()

    def __init__(self, limits: AxisLimits, unit: str = " µV", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._enabled = QCheckBox("Manuel")
        self._enabled.setChecked(bool(limits.enabled))
        self._min = _spin(-1e6, 1e6, limits.minimum, decimals=2, step=10.0, suffix=unit)
        self._max = _spin(-1e6, 1e6, limits.maximum, decimals=2, step=10.0, suffix=unit)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._enabled)
        layout.addWidget(self._min)
        layout.addWidget(QLabel("à"))
        layout.addWidget(self._max)
        self._enabled.toggled.connect(self._on_enabled)
        self._min.valueChanged.connect(lambda _v: self.changed.emit())
        self._max.valueChanged.connect(lambda _v: self.changed.emit())
        self._on_enabled(self._enabled.isChecked(), emit=False)

    def _on_enabled(self, checked: bool, *, emit: bool = True) -> None:
        self._min.setEnabled(bool(checked))
        self._max.setEnabled(bool(checked))
        if emit:
            self.changed.emit()

    def value(self) -> AxisLimits:
        return AxisLimits(
            enabled=self._enabled.isChecked(),
            minimum=float(self._min.value()),
            maximum=float(self._max.value()),
        )

    def set_value(self, limits: AxisLimits) -> None:
        self._enabled.setChecked(bool(limits.enabled))
        self._min.setValue(float(limits.minimum))
        self._max.setValue(float(limits.maximum))


class ParamsPanel(QWidget):
    """Live display settings and processing settings in one dock."""

    viewChanged = Signal()
    configChanged = Signal()

    def __init__(self, defaults: Mapping[str, Any] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._defaults = dict(defaults or {})
        self._loading = True

        self.tabs = QTabWidget(self)
        self.tabs.addTab(_scrollable(self._build_display_tab()), "Affichage")
        self.tabs.addTab(_scrollable(self._build_legend_tab()), "Légende && style")
        self.tabs.addTab(_scrollable(self._build_processing_tab()), "Traitement")

        self._dirty_label = QLabel("")
        self._dirty_label.setObjectName("warningLabel")
        self._dirty_label.setWordWrap(True)
        self._dirty_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self._dirty_label)

        self._loading = False

    # ----------------------------------------------------------- display tab

    def _build_display_tab(self) -> QWidget:
        defaults = self._defaults
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(8)

        zoom_group = QGroupBox("Fenêtres de zoom (relatives à la stimulation)")
        zoom_form = QFormLayout(zoom_group)
        self._zoom_onset_t0 = _spin(
            -60.0, 60.0, defaults.get("default_zoom_onset_t0_s", -0.1), suffix=" s"
        )
        self._zoom_onset_t1 = _spin(
            -60.0, 60.0, defaults.get("default_zoom_onset_t1_s", 0.2), suffix=" s"
        )
        self._zoom_end_t0 = _spin(
            -60.0, 60.0, defaults.get("default_zoom_end_t0_s", -0.1), suffix=" s"
        )
        self._zoom_end_t1 = _spin(
            -60.0, 60.0, defaults.get("default_zoom_end_t1_s", 0.2), suffix=" s"
        )
        zoom_form.addRow("Zoom début — départ :", self._zoom_onset_t0)
        zoom_form.addRow("Zoom début — fin :", self._zoom_onset_t1)
        zoom_form.addRow("Zoom fin — départ :", self._zoom_end_t0)
        zoom_form.addRow("Zoom fin — fin :", self._zoom_end_t1)
        zoom_group.setToolTip(
            "Les fenêtres « début » sont relatives au début de stimulation ; "
            "les fenêtres « fin » au dernier front montant."
        )

        spike_group = QGroupBox("Affichage des spikes")
        spike_form = QFormLayout(spike_group)
        self._psth_bin = _spin(
            0.001, 10.0, defaults.get("default_psth_bin_window_s", 0.025), suffix=" s"
        )
        self._psth_bin.setToolTip("Largeur de bin PSTH / taux de décharge. Rebin instantané.")
        self._sampling = _int_spin(
            1, 100, int(defaults.get("default_sampling_percent", 100) or 100), suffix=" %"
        )
        self._sampling.setToolTip(
            "Fraction des spikes dessinés dans les rasters et superpositions "
            "(les taux restent calculés sur tous les spikes)."
        )
        spike_form.addRow("Bin PSTH :", self._psth_bin)
        spike_form.addRow("Spikes dessinés :", self._sampling)

        axis_group = QGroupBox("Limites de l’axe Y")
        axis_form = QFormLayout(axis_group)
        self._stim_hp_ylim = _AxisLimitRow(
            AxisLimits(
                enabled=bool(defaults.get("default_first_trigger_hp_ylim_enabled", False)),
                minimum=float(defaults.get("default_first_trigger_hp_ylim_min_uv", -200.0)),
                maximum=float(defaults.get("default_first_trigger_hp_ylim_max_uv", 200.0)),
            )
        )
        self._rms_ylim = _AxisLimitRow(AxisLimits(enabled=True, minimum=0.0, maximum=20.0))
        self._trace_ylim = _AxisLimitRow(AxisLimits())
        axis_form.addRow("Passe-haut stimulation :", self._stim_hp_ylim)
        axis_form.addRow("Panneaux RMS :", self._rms_ylim)
        axis_form.addRow("Panneaux de traces :", self._trace_ylim)

        montage_group = QGroupBox("Panneaux montage")
        montage_form = QFormLayout(montage_group)
        self._montage_channels = _int_spin(2, 128, 32)
        self._montage_channels.setToolTip("Canaux empilés dans un panneau montage.")
        self._montage_page = _int_spin(0, 64, 0)
        self._montage_page.setToolTip(
            "Bloc de canaux affiché par les montages (0 = première page)."
        )
        montage_form.addRow("Canaux par montage :", self._montage_channels)
        montage_form.addRow("Page de montage :", self._montage_page)

        for group in (zoom_group, spike_group, axis_group, montage_group):
            page_layout.addWidget(group)
        page_layout.addStretch(1)

        for widget in (
            self._zoom_onset_t0,
            self._zoom_onset_t1,
            self._zoom_end_t0,
            self._zoom_end_t1,
            self._psth_bin,
        ):
            widget.valueChanged.connect(lambda _v: self._emit_view())
        for int_widget in (self._sampling, self._montage_channels, self._montage_page):
            int_widget.valueChanged.connect(lambda _v: self._emit_view())
        for row in (self._stim_hp_ylim, self._rms_ylim, self._trace_ylim):
            row.changed.connect(self._emit_view)
        return page

    # ------------------------------------------------------------ legend tab

    def _build_legend_tab(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(8)

        legend_group = QGroupBox("Légendes")
        legend_form = QFormLayout(legend_group)
        self._legend_visible = QCheckBox("Afficher les légendes")
        self._legend_visible.setChecked(True)
        self._legend_location = QComboBox()
        for location in LEGEND_LOCATIONS:
            self._legend_location.addItem(
                "Sous le panneau" if location == "below" else location.capitalize(), location
            )
        self._legend_font = _spin(4.0, 24.0, 9.0, decimals=1, step=0.5, suffix=" pt")
        self._legend_columns = _int_spin(1, 6, 1)
        self._legend_frame = QCheckBox("Cadre autour des légendes")
        self._legend_frame.setChecked(True)
        self._legend_filters = QCheckBox("Détails du filtre (type, ordre, coupure)")
        self._legend_filters.setChecked(True)
        self._legend_counts = QCheckBox("Compteurs d’échantillons / spikes")
        self._legend_counts.setChecked(True)
        self._legend_markers = QCheckBox("Marqueurs de stimulation dans les légendes")
        self._legend_markers.setChecked(True)
        legend_form.addRow(self._legend_visible)
        legend_form.addRow("Position :", self._legend_location)
        legend_form.addRow("Taille de police :", self._legend_font)
        legend_form.addRow("Colonnes :", self._legend_columns)
        legend_form.addRow(self._legend_frame)
        legend_form.addRow(self._legend_filters)
        legend_form.addRow(self._legend_counts)
        legend_form.addRow(self._legend_markers)

        style_group = QGroupBox("Style des panneaux")
        style_form = QFormLayout(style_group)
        self._title_font = _spin(5.0, 24.0, 10.0, decimals=1, step=0.5, suffix=" pt")
        self._label_font = _spin(5.0, 24.0, 9.0, decimals=1, step=0.5, suffix=" pt")
        self._tick_font = _spin(4.0, 20.0, 8.0, decimals=1, step=0.5, suffix=" pt")
        self._line_width = _spin(0.2, 5.0, 1.2, decimals=2, step=0.1)
        self._grid = QCheckBox("Afficher la grille")
        self._grid.setChecked(True)
        self._grid_alpha = _spin(0.0, 1.0, 0.3, decimals=2, step=0.05)
        self._max_points = _int_spin(500, 200000, 6000, step=500)
        self._max_points.setToolTip(
            "Points dessinés par courbe. Au-delà, une enveloppe min/max conserve "
            "tous les pics tout en restant rapide."
        )
        style_form.addRow("Police du titre :", self._title_font)
        style_form.addRow("Police des axes :", self._label_font)
        style_form.addRow("Police des ticks :", self._tick_font)
        style_form.addRow("Épaisseur de trait :", self._line_width)
        style_form.addRow(self._grid)
        style_form.addRow("Opacité de la grille :", self._grid_alpha)
        style_form.addRow("Points max / courbe :", self._max_points)

        page_layout.addWidget(legend_group)
        page_layout.addWidget(style_group)
        page_layout.addStretch(1)

        for box in (
            self._legend_visible,
            self._legend_frame,
            self._legend_filters,
            self._legend_counts,
            self._legend_markers,
            self._grid,
        ):
            box.toggled.connect(lambda _c: self._emit_view())
        self._legend_location.currentIndexChanged.connect(lambda _i: self._emit_view())
        for spin in (
            self._legend_font,
            self._title_font,
            self._label_font,
            self._tick_font,
            self._line_width,
            self._grid_alpha,
        ):
            spin.valueChanged.connect(lambda _v: self._emit_view())
        for int_spin in (self._legend_columns, self._max_points):
            int_spin.valueChanged.connect(lambda _v: self._emit_view())
        return page

    # -------------------------------------------------------- processing tab

    def _build_processing_tab(self) -> QWidget:
        defaults = self._defaults
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(8)

        trigger_group = QGroupBox("Détection des stimulations")
        trigger_form = QFormLayout(trigger_group)
        self._edge = _choice(_EDGE_CHOICES, str(defaults.get("default_edge", "falling")))
        self._threshold = _spin(
            -100.0, 100.0, defaults.get("default_threshold", 1.0), decimals=3, suffix=" V"
        )
        self._pre_s = _spin(0.0, 600.0, defaults.get("default_pre_s", 2.0), suffix=" s")
        self._post_s = _spin(0.001, 3600.0, defaults.get("default_post_s", 10.0), suffix=" s")
        self._section_spec = _choice(
            _SECTION_SPECS, str(defaults.get("default_section_spec", "count"))
        )
        self._section_count = _int_spin(
            1, 10000, int(defaults.get("default_section_count", 10) or 10)
        )
        self._section_duration = _spin(
            0.001, 3600.0, defaults.get("default_section_duration_s") or 10.0, suffix=" s"
        )
        self._section_start = _spin(
            0.0, 3600.0, defaults.get("default_section_trigger_start_s", 1.0), suffix=" s"
        )
        self._section_end = _spin(
            0.0, 3600.0, defaults.get("default_section_trigger_end_s", 4.0), suffix=" s"
        )
        trigger_form.addRow("Front de déclenchement :", self._edge)
        trigger_form.addRow("Seuil de déclenchement :", self._threshold)
        trigger_form.addRow("Fenêtre avant :", self._pre_s)
        trigger_form.addRow("Fenêtre après :", self._post_s)
        trigger_form.addRow("Mode sans déclencheur :", self._section_spec)
        trigger_form.addRow("Nombre de sections :", self._section_count)
        trigger_form.addRow("Durée de section :", self._section_duration)
        trigger_form.addRow("Début virtuel dans la section :", self._section_start)
        trigger_form.addRow("Fin virtuelle dans la section :", self._section_end)
        self._section_spec.currentIndexChanged.connect(lambda _i: self._sync_section_rows())
        self._edge.currentIndexChanged.connect(lambda _i: self._sync_section_rows())

        filter_group = QGroupBox("Filtres Intan (passe-haut et passe-bas)")
        filter_form = QFormLayout(filter_group)
        self._filter_type = _choice(
            _FILTER_TYPES, str(defaults.get("default_intan_filter_type", "bessel"))
        )
        self._filter_order = _int_spin(
            1, 8, int(defaults.get("default_intan_filter_order", 2) or 2)
        )
        self._filter_cutoff = _spin(
            0.1,
            20000.0,
            defaults.get("default_intan_filter_cutoff_hz", 250.0),
            decimals=2,
            step=10.0,
            suffix=" Hz",
        )
        self._artifact_enabled = QCheckBox("Supprimer les artefacts de stimulation")
        self._artifact_enabled.setChecked(True)
        self._artifact_threshold = _spin(
            1.0, 100000.0, 2500.0, decimals=1, step=50.0, suffix=" µV"
        )
        self._rms_window = _spin(
            0.001, 60.0, defaults.get("default_rms_window_s", 1.0), suffix=" s"
        )
        self._rms_window.setToolTip("Fenêtre glissante utilisée pour les profils RMS.")
        filter_form.addRow("Type de filtre :", self._filter_type)
        filter_form.addRow("Ordre du filtre :", self._filter_order)
        filter_form.addRow("Coupure :", self._filter_cutoff)
        filter_form.addRow(self._artifact_enabled)
        filter_form.addRow("Seuil d’artefact :", self._artifact_threshold)
        filter_form.addRow("Fenêtre RMS :", self._rms_window)

        spike_group = QGroupBox("Détection de spikes")
        spike_form = QFormLayout(spike_group)
        self._threshold_mode = _choice(
            _THRESHOLD_MODES, str(defaults.get("default_spike_threshold_mode", "fixed"))
        )
        self._spike_threshold = _spin(
            0.1,
            100000.0,
            abs(float(defaults.get("default_spike_threshold_uv", 70.0))),
            decimals=2,
            step=5.0,
            suffix=" µV",
        )
        self._polarity = _choice(
            _POLARITIES, str(defaults.get("default_spike_threshold_polarity", "negative"))
        )
        self._rms_multiplier = _spin(
            0.1,
            100.0,
            defaults.get("default_spike_threshold_rms_multiplier", 4.0),
            decimals=2,
            step=0.5,
        )
        self._overlay_pre = _spin(
            0.0, 500.0, defaults.get("default_spike_overlay_pre_ms", 2.0), decimals=2, suffix=" ms"
        )
        self._overlay_post = _spin(
            0.01,
            500.0,
            defaults.get("default_spike_overlay_post_ms", 4.0),
            decimals=2,
            suffix=" ms",
        )
        for widget in (self._overlay_pre, self._overlay_post):
            widget.setToolTip(
                "Fenêtre de forme d’onde autour de chaque détection. La modifier "
                "ré-extrait les snippets (rapide : les flux filtrés restent en cache)."
            )
        spike_form.addRow("Mode de seuil :", self._threshold_mode)
        spike_form.addRow("Seuil fixe :", self._spike_threshold)
        spike_form.addRow("Polarité :", self._polarity)
        spike_form.addRow("Multiplicateur RMS :", self._rms_multiplier)
        spike_form.addRow("Avant le spike :", self._overlay_pre)
        spike_form.addRow("Après le spike :", self._overlay_post)
        self._threshold_mode.currentIndexChanged.connect(lambda _i: self._sync_threshold_rows())

        resources_group = QGroupBox("Sonde et ressources")
        resources_form = QFormLayout(resources_group)
        self._probe_edit = QLineEdit(str(defaults.get("default_probe_layout_json") or ""))
        self._probe_edit.setPlaceholderText("JSON de sonde MEA (optionnel)")
        probe_button = QPushButton("Parcourir…")
        probe_button.clicked.connect(self._browse_probe)
        probe_row = QWidget()
        probe_layout = QHBoxLayout(probe_row)
        probe_layout.setContentsMargins(0, 0, 0, 0)
        probe_layout.setSpacing(4)
        probe_layout.addWidget(self._probe_edit, 1)
        probe_layout.addWidget(probe_button)
        self._work_edit = QLineEdit("")
        self._work_edit.setPlaceholderText("Par défaut : à côté du fichier .rhs")
        work_button = QPushButton("Parcourir…")
        work_button.clicked.connect(self._browse_work_dir)
        work_row = QWidget()
        work_layout = QHBoxLayout(work_row)
        work_layout.setContentsMargins(0, 0, 0, 0)
        work_layout.setSpacing(4)
        work_layout.addWidget(self._work_edit, 1)
        work_layout.addWidget(work_button)
        self._channel_workers = _int_spin(
            0, 64, int(defaults.get("default_channel_workers") or 0)
        )
        self._channel_workers.setSpecialValueText("Auto")
        self._channel_workers.setToolTip(
            "Travailleurs parallèles pendant le filtrage (0 = automatique)."
        )
        resources_form.addRow("JSON sonde MEA :", probe_row)
        resources_form.addRow("Dossier cache / travail :", work_row)
        resources_form.addRow("Travailleurs canaux :", self._channel_workers)

        for group in (trigger_group, filter_group, spike_group, resources_group):
            page_layout.addWidget(group)
        hint = QLabel(
            "Modifier un paramètre de traitement exige un retraitement (F5). "
            "Grâce au cache par étapes, seules les étapes concernées sont recalculées."
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        page_layout.addWidget(hint)
        page_layout.addStretch(1)

        for spin in (
            self._threshold,
            self._pre_s,
            self._post_s,
            self._section_duration,
            self._section_start,
            self._section_end,
            self._filter_cutoff,
            self._artifact_threshold,
            self._rms_window,
            self._spike_threshold,
            self._rms_multiplier,
            self._overlay_pre,
            self._overlay_post,
        ):
            spin.valueChanged.connect(lambda _v: self._emit_config())
        for int_spin in (self._section_count, self._filter_order, self._channel_workers):
            int_spin.valueChanged.connect(lambda _v: self._emit_config())
        for box in (
            self._edge,
            self._section_spec,
            self._filter_type,
            self._threshold_mode,
            self._polarity,
        ):
            box.currentIndexChanged.connect(lambda _i: self._emit_config())
        self._artifact_enabled.toggled.connect(lambda _c: self._emit_config())
        self._probe_edit.textChanged.connect(lambda _t: self._emit_config())
        self._work_edit.textChanged.connect(lambda _t: self._emit_config())

        self._sync_section_rows()
        self._sync_threshold_rows()
        return page

    # -------------------------------------------------------------- behaviour

    def _sync_section_rows(self) -> None:
        no_trigger = str(self._edge.currentData()) == "none"
        by_count = str(self._section_spec.currentData()) == "count"
        self._threshold.setEnabled(not no_trigger)
        self._section_spec.setEnabled(no_trigger)
        self._section_count.setEnabled(no_trigger and by_count)
        self._section_duration.setEnabled(no_trigger and not by_count)
        self._section_start.setEnabled(no_trigger)
        self._section_end.setEnabled(no_trigger)

    def _sync_threshold_rows(self) -> None:
        fixed = str(self._threshold_mode.currentData()) == "fixed"
        self._spike_threshold.setEnabled(fixed)
        self._rms_multiplier.setEnabled(not fixed)

    def _browse_probe(self) -> None:
        start = self._probe_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir un JSON de sonde", start, "JSON (*.json);;Tous les fichiers (*)"
        )
        if path:
            self._probe_edit.setText(path)

    def _browse_work_dir(self) -> None:
        start = self._work_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Choisir le dossier de cache", start)
        if path:
            self._work_edit.setText(path)

    def _emit_view(self) -> None:
        if self._loading:
            return
        self.viewChanged.emit()

    def _emit_config(self) -> None:
        if self._loading:
            return
        self.configChanged.emit()

    def set_dirty_message(self, message: str) -> None:
        self._dirty_label.setText(message)
        self._dirty_label.setVisible(bool(message))

    def focus_processing_tab(self) -> None:
        self.tabs.setCurrentIndex(2)

    # ----------------------------------------------------------------- values

    def viewer_settings(self) -> ViewerSettings:
        legend = LegendSettings(
            visible=self._legend_visible.isChecked(),
            location=str(self._legend_location.currentData()),  # type: ignore[arg-type]
            font_size=float(self._legend_font.value()),
            columns=int(self._legend_columns.value()),
            frame=self._legend_frame.isChecked(),
            show_filter_details=self._legend_filters.isChecked(),
            show_sample_counts=self._legend_counts.isChecked(),
            show_reference_markers=self._legend_markers.isChecked(),
        )
        style = PanelStyle(
            title_font_size=float(self._title_font.value()),
            label_font_size=float(self._label_font.value()),
            tick_font_size=float(self._tick_font.value()),
            line_width=float(self._line_width.value()),
            grid=self._grid.isChecked(),
            grid_alpha=float(self._grid_alpha.value()),
            max_points_per_curve=int(self._max_points.value()),
        )
        return ViewerSettings(
            zoom_onset_t0_s=float(self._zoom_onset_t0.value()),
            zoom_onset_t1_s=float(self._zoom_onset_t1.value()),
            zoom_end_t0_s=float(self._zoom_end_t0.value()),
            zoom_end_t1_s=float(self._zoom_end_t1.value()),
            psth_bin_window_s=float(self._psth_bin.value()),
            sampling_percent=int(self._sampling.value()),
            spike_overlay_pre_ms=float(self._overlay_pre.value()),
            spike_overlay_post_ms=float(self._overlay_post.value()),
            stim_hp_ylim=self._stim_hp_ylim.value(),
            rms_ylim=self._rms_ylim.value(),
            trace_ylim=self._trace_ylim.value(),
            legend=legend,
            style=style,
            montage_channels=int(self._montage_channels.value()),
            montage_page=int(self._montage_page.value()),
        )

    def probe_layout_path(self) -> Path | None:
        text = self._probe_edit.text().strip()
        return Path(text) if text else None

    def build_config(self, rhs_file: Path, *, base: AnalysisConfig | None = None) -> AnalysisConfig:
        """Processing configuration for one recording."""
        work_text = self._work_edit.text().strip()
        polarity = str(self._polarity.currentData())
        magnitude = abs(float(self._spike_threshold.value()))
        signed = -magnitude if polarity == "negative" else magnitude
        section_duration = (
            float(self._section_duration.value())
            if str(self._section_spec.currentData()) == "duration"
            else None
        )
        config = AnalysisConfig(
            rhs_file=Path(rhs_file),
            threshold=float(self._threshold.value()),
            edge=str(self._edge.currentData()),  # type: ignore[arg-type]
            pre_s=float(self._pre_s.value()),
            post_s=float(self._post_s.value()),
            section_count=int(self._section_count.value()),
            section_duration_s=section_duration,
            section_spec=str(self._section_spec.currentData()),  # type: ignore[arg-type]
            section_trigger_start_s=float(self._section_start.value()),
            section_trigger_end_s=float(self._section_end.value()),
            spike_threshold_uv=signed,
            spike_threshold_polarity=polarity,  # type: ignore[arg-type]
            spike_threshold_mode=str(self._threshold_mode.currentData()),  # type: ignore[arg-type]
            spike_threshold_rms_multiplier=float(self._rms_multiplier.value()),
            psth_bin_window_s=float(self._psth_bin.value()),
            spike_overlay_pre_ms=float(self._overlay_pre.value()),
            spike_overlay_post_ms=float(self._overlay_post.value()),
            zoom_onset_t0_s=float(self._zoom_onset_t0.value()),
            zoom_onset_t1_s=float(self._zoom_onset_t1.value()),
            zoom_end_t0_s=float(self._zoom_end_t0.value()),
            zoom_end_t1_s=float(self._zoom_end_t1.value()),
            first_trigger_hp_ylim_enabled=self._stim_hp_ylim.value().enabled,
            first_trigger_hp_ylim_min_uv=self._stim_hp_ylim.value().minimum,
            first_trigger_hp_ylim_max_uv=self._stim_hp_ylim.value().maximum,
            rms_window_s=float(self._rms_window.value()),
            intan_filter_order=int(self._filter_order.value()),
            intan_filter_type=str(self._filter_type.currentData()),  # type: ignore[arg-type]
            intan_filter_cutoff_hz=float(self._filter_cutoff.value()),
            intan_artifact_threshold_uv=float(self._artifact_threshold.value()),
            intan_artifact_suppression_enabled=self._artifact_enabled.isChecked(),
            work_dir=Path(work_text) if work_text else None,
            channel_workers=int(self._channel_workers.value()) or None,
            sampling_percent=int(self._sampling.value()),
            probe_layout_json=self.probe_layout_path(),
        )
        if base is not None:
            config = replace(
                config,
                save_dir=base.save_dir,
                pdf_title=base.pdf_title,
                comparison_workers=base.comparison_workers,
            )
        return config

    def validate(self) -> list[str]:
        problems = self.viewer_settings().validate()
        if self._post_s.value() <= 0:
            problems.append("La fenêtre après la stimulation doit être > 0 s.")
        if str(self._edge.currentData()) == "none" and self._section_end.value() <= self._section_start.value():
            problems.append(
                "Mode sans déclencheur : la fin virtuelle doit être après le début virtuel."
            )
        if self._overlay_post.value() <= 0:
            problems.append("Superposition de spikes : le temps après détection doit être > 0 ms.")
        return problems
