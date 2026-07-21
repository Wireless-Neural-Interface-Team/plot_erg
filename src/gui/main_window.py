"""Tabbed Qt main window for plot_erg."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from config import AnalysisConfig
from display_config import (
    PANEL_FIELD_NAMES,
    PANEL_LABELS,
    PlotDisplaySettings,
    RecordingStyle,
    SectionPanels,
    resolve_display_label,
)
from gui.analysis_thread import create_analysis_thread_class
from gui.styles import APP_STYLESHEET


def launch_qt_gui(
    run_callback: Callable[[AnalysisConfig], None],
    run_comparison_callback: Callable[[AnalysisConfig, AnalysisConfig], None],
    run_multi_comparison_callback: Callable[[list[AnalysisConfig]], None] | None = None,
    **defaults,
) -> int:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QBrush, QColor
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QProgressBar,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QTabWidget,
            QTableWidget,
            QTableWidgetItem,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("PySide6 is not installed. Run: pip install PySide6") from exc

    AnalysisThread = create_analysis_thread_class()
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(APP_STYLESHEET)

    # ------------------------------------------------------------------ helpers
    def _spin(
        value: float,
        *,
        minimum: float = -1e6,
        maximum: float = 1e6,
        decimals: int = 4,
        step: float = 0.01,
    ) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(minimum, maximum)
        w.setDecimals(decimals)
        w.setSingleStep(step)
        w.setValue(float(value))
        return w

    def _int_spin(value: int, *, minimum: int = 0, maximum: int = 10000) -> QSpinBox:
        w = QSpinBox()
        w.setRange(minimum, maximum)
        w.setValue(int(value))
        return w

  # ===========================================================================
    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Intan RHS Stimulation Plotter")
            self.resize(1040, 820)
            self._run_callback = run_callback
            self._run_comparison_callback = run_comparison_callback
            self._run_multi_callback = run_multi_comparison_callback
            self._analysis_thread = None
            self._section_spec = "count"
            self._cached_duration_s: float | None = None
            self._section_sync_guard = False
            self._file_rows: list[MainWindow.FileEntryRow] = []
            self._display_checkboxes: dict[str, dict[str, QCheckBox]] = {}
            self._display_cell_wrappers: dict[str, dict[str, QWidget]] = {}
            self._last_auto_pdf_title: str = ""
            self._build_ui()
            self._apply_defaults(defaults)

        class FileEntryRow(QWidget):
            def __init__(self, outer: MainWindow, *, initial_path: str = "") -> None:
                super().__init__()
                self.outer = outer
                self.setObjectName("fileEntryRow")
                layout = QHBoxLayout(self)
                layout.setContentsMargins(10, 8, 10, 8)
                layout.setSpacing(8)
                self.path_edit = QLineEdit(initial_path)
                self.path_edit.setPlaceholderText("Path to .rhs file")
                self.legend_edit = QLineEdit()
                self.legend_edit.setPlaceholderText("Custom legend (empty = file name)")
                self.plot_check = QCheckBox("Plot")
                self.plot_check.setChecked(True)
                self.plot_check.setToolTip("Show plots for this recording")
                self.legend_check = QCheckBox("Legend")
                self.legend_check.setChecked(True)
                self.legend_check.setToolTip("Include this recording in legends")
                browse_btn = QPushButton("Browse…")
                browse_btn.setObjectName("secondaryButton")
                remove_btn = QPushButton("Remove")
                remove_btn.setObjectName("dangerButton")
                browse_btn.clicked.connect(self._browse)
                remove_btn.clicked.connect(self._remove)
                self.path_edit.textChanged.connect(outer._on_files_changed)
                self.legend_edit.textChanged.connect(outer._on_files_changed)
                layout.addWidget(self.path_edit, stretch=3)
                layout.addWidget(self.legend_edit, stretch=2)
                layout.addWidget(self.plot_check)
                layout.addWidget(self.legend_check)
                layout.addWidget(browse_btn)
                layout.addWidget(remove_btn)

            def _browse(self) -> None:
                selected, _ = QFileDialog.getOpenFileName(
                    self.outer,
                    "RHS file",
                    "",
                    "Intan RHS (*.rhs);;All files (*)",
                )
                if selected:
                    self.path_edit.setText(selected)
                    if not self.legend_edit.text().strip():
                        self.legend_edit.setText(Path(selected).stem)

            def _remove(self) -> None:
                self.outer._remove_file_row(self)

            def path(self) -> str:
                return self.path_edit.text().strip()

            def style(self) -> RecordingStyle:
                return RecordingStyle(
                    plot_visible=self.plot_check.isChecked(),
                    legend_visible=self.legend_check.isChecked(),
                )

            def label(self) -> str | None:
                text = self.legend_edit.text().strip()
                return text or None

        def _build_ui(self) -> None:
            central = QWidget()
            central.setObjectName("centralWidget")
            self.setCentralWidget(central)
            root = QVBoxLayout(central)
            root.setContentsMargins(14, 14, 14, 14)
            root.setSpacing(12)

            self.tabs = QTabWidget()
            self.tabs.addTab(self._build_files_tab(), "Files")
            self.tabs.addTab(self._build_trigger_tab(), "Stimulation")
            self.tabs.addTab(self._build_output_tab(), "PDF output")
            self.tabs.addTab(self._build_filter_tab(), "Intan filter")
            self.tabs.addTab(self._build_spikes_tab(), "Spikes / PSTH")
            self.tabs.addTab(self._build_zoom_tab(), "Zoom")
            self.tabs.addTab(self._build_display_tab(), "Display")
            self.tabs.addTab(self._build_perf_tab(), "Performance")
            root.addWidget(self.tabs)

            action_row = QHBoxLayout()
            self.run_btn = QPushButton("Run analysis")
            self.run_btn.clicked.connect(self._run_analysis)
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.setObjectName("dangerButton")
            self.stop_btn.setEnabled(False)
            self.stop_btn.clicked.connect(self._stop_analysis)
            action_row.addWidget(self.run_btn)
            action_row.addWidget(self.stop_btn)
            action_row.addStretch()
            root.addLayout(action_row)

            self.progress = QProgressBar()
            self.progress.setRange(0, 0)
            self.progress.setVisible(False)
            root.addWidget(self.progress)

            self.status_label = QLabel("Add one or more .rhs files, then run the analysis.")
            self.status_label.setObjectName("statusLabel")
            root.addWidget(self.status_label)

            self.log_view = QTextEdit()
            self.log_view.setObjectName("logView")
            self.log_view.setReadOnly(True)
            self.log_view.setPlaceholderText("Run logs appear here…")
            self.log_view.setMinimumHeight(160)
            root.addWidget(self.log_view)

        def _build_files_tab(self) -> QWidget:
            w = QWidget()
            w.setObjectName("filesTab")
            layout = QVBoxLayout(w)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)
            info = QLabel(
                "Add recordings to compare. Set a custom legend "
                "and control plot and legend entry visibility."
            )
            info.setObjectName("hintLabel")
            info.setWordWrap(True)
            layout.addWidget(info)

            header = QHBoxLayout()
            for text, stretch in ((".rhs file", 3), ("Legend", 2), ("Display", 1)):
                lbl = QLabel(text)
                lbl.setObjectName("columnHeader")
                header.addWidget(lbl, stretch)
            header.addStretch()
            layout.addLayout(header)

            self.files_container = QWidget()
            self.files_container.setObjectName("filesContainer")
            self.files_layout = QVBoxLayout(self.files_container)
            self.files_layout.setContentsMargins(0, 0, 0, 0)
            self.files_layout.setSpacing(10)
            scroll = QScrollArea()
            scroll.setObjectName("filesScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.viewport().setObjectName("filesScrollViewport")
            scroll.setWidget(self.files_container)
            layout.addWidget(scroll, stretch=1)

            add_btn = QPushButton("Add file")
            add_btn.setObjectName("secondaryButton")
            add_btn.clicked.connect(lambda: self._add_file_row(""))
            layout.addWidget(add_btn)
            return w

        def _build_trigger_tab(self) -> QWidget:
            w = QWidget()
            form = QFormLayout(w)
            form.setContentsMargins(16, 16, 16, 16)
            form.setSpacing(12)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.edge_combo = QComboBox()
            self.edge_combo.addItem("Falling edge", "falling")
            self.edge_combo.addItem("Rising edge", "rising")
            self.edge_combo.addItem("No stimulation", "none")
            self.edge_combo.currentIndexChanged.connect(self._update_trigger_visibility)
            self.threshold_spin = _spin(defaults.get("default_threshold", 1.0), minimum=0.0)
            self.pre_spin = _spin(defaults.get("default_pre_s", 2.0), minimum=0.0)
            self.post_spin = _spin(defaults.get("default_post_s", 10.0), minimum=0.0)
            self.section_count_spin = _int_spin(defaults.get("default_section_count", 10), minimum=1)
            self.section_duration_spin = _spin(
                defaults.get("default_section_duration_s") or 1.0, minimum=0.001
            )
            self.section_trigger_start_spin = _spin(
                defaults.get("default_section_trigger_start_s", 1.0), minimum=0.0
            )
            self.section_trigger_end_spin = _spin(
                defaults.get("default_section_trigger_end_s", 4.0), minimum=0.0
            )
            self.threshold_label = QLabel("ANALOG_IN 0 threshold:")
            self.pre_label = QLabel("Pre-stimulation (s):")
            self.post_label = QLabel("Post-stimulation (s):")
            self.section_count_label = QLabel("Number of sections:")
            self.section_duration_label = QLabel("Section duration (s):")
            self.section_trigger_start_label = QLabel("Imaginary stimulation start (s):")
            self.section_trigger_end_label = QLabel("Imaginary stimulation end (s):")
            form.addRow("Stimulation mode:", self.edge_combo)
            form.addRow(self.threshold_label, self.threshold_spin)
            form.addRow(self.pre_label, self.pre_spin)
            form.addRow(self.post_label, self.post_spin)
            form.addRow(self.section_count_label, self.section_count_spin)
            form.addRow(self.section_duration_label, self.section_duration_spin)
            form.addRow(self.section_trigger_start_label, self.section_trigger_start_spin)
            form.addRow(self.section_trigger_end_label, self.section_trigger_end_spin)
            self.section_count_spin.valueChanged.connect(lambda _v: self._sync_sections("count"))
            self.section_duration_spin.valueChanged.connect(lambda _v: self._sync_sections("duration"))
            return w

        def _build_output_tab(self) -> QWidget:
            w = QWidget()
            form = QFormLayout(w)
            form.setContentsMargins(16, 16, 16, 16)
            form.setSpacing(12)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.save_dir_edit = QLineEdit()
            self.save_dir_edit.setPlaceholderText("Empty = .rhs file folder")
            browse_save = QPushButton("Browse…")
            browse_save.setObjectName("secondaryButton")
            save_row = QHBoxLayout()
            save_row.addWidget(self.save_dir_edit)
            save_row.addWidget(browse_save)
            browse_save.clicked.connect(self._browse_save_dir)
            self.pdf_title_edit = QLineEdit()
            self.pdf_title_edit.setPlaceholderText(
                "Empty = automatic name from .rhs files"
            )
            self.probe_json_edit = QLineEdit()
            self.probe_json_edit.setPlaceholderText("Optional — probeinterface JSON (MEA map)")
            browse_probe = QPushButton("Browse…")
            browse_probe.setObjectName("secondaryButton")
            probe_row = QHBoxLayout()
            probe_row.addWidget(self.probe_json_edit)
            probe_row.addWidget(browse_probe)
            browse_probe.clicked.connect(self._browse_probe_json)
            form.addRow("PDF output folder:", save_row)
            form.addRow("PDF name:", self.pdf_title_edit)
            form.addRow("MEA probe (JSON):", probe_row)
            return w

        def _build_filter_tab(self) -> QWidget:
            w = QWidget()
            form = QFormLayout(w)
            form.setContentsMargins(16, 16, 16, 16)
            form.setSpacing(12)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.filter_kind_combo = QComboBox()
            self.filter_kind_combo.addItem("High-pass", "highpass")
            self.filter_kind_combo.addItem("Low-pass", "lowpass")
            self.filter_type_combo = QComboBox()
            self.filter_type_combo.addItem("Bessel", "bessel")
            self.filter_type_combo.addItem("Butterworth", "butterworth")
            self.filter_order_spin = _int_spin(defaults.get("default_intan_filter_order", 2), minimum=1, maximum=8)
            self.filter_cutoff_spin = _spin(
                defaults.get("default_intan_filter_cutoff_hz", 250.0), minimum=0.1, maximum=50000.0, decimals=1
            )
            form.addRow("Filter type (HP/LP):", self.filter_kind_combo)
            form.addRow("Prototype:", self.filter_type_combo)
            form.addRow("Order (1–8):", self.filter_order_spin)
            form.addRow("Cutoff frequency (Hz):", self.filter_cutoff_spin)
            return w

        def _build_spikes_tab(self) -> QWidget:
            w = QWidget()
            form = QFormLayout(w)
            form.setContentsMargins(16, 16, 16, 16)
            form.setSpacing(12)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.spike_mode_combo = QComboBox()
            self.spike_mode_combo.addItem("Fixed threshold (same for all channels)", "fixed")
            self.spike_mode_combo.addItem("× mean RMS multiplier per channel", "rms_multiple")
            self.spike_mode_combo.currentIndexChanged.connect(self._update_spike_mode_visibility)
            self.spike_polarity_combo = QComboBox()
            self.spike_polarity_combo.addItem("Negative — below threshold", "negative")
            self.spike_polarity_combo.addItem("Positive — above threshold", "positive")
            self.spike_fixed_spin = _spin(defaults.get("default_spike_threshold_uv", 70.0), minimum=0.001)
            self.spike_rms_mult_spin = _spin(
                defaults.get("default_spike_threshold_rms_multiplier", 4.0), minimum=0.001
            )
            self.psth_bin_spin = _spin(defaults.get("default_psth_bin_window_s", 0.025), minimum=0.001)
            self.rms_window_spin = _spin(defaults.get("default_rms_window_s", 1.0), minimum=0.001)
            self.spike_fixed_label = QLabel("Fixed threshold (µV):")
            self.spike_rms_label = QLabel("RMS multiplier:")
            form.addRow("Spike threshold mode:", self.spike_mode_combo)
            form.addRow("Polarity:", self.spike_polarity_combo)
            form.addRow(self.spike_fixed_label, self.spike_fixed_spin)
            form.addRow(self.spike_rms_label, self.spike_rms_mult_spin)
            form.addRow("PSTH window (s):", self.psth_bin_spin)
            form.addRow("RMS window (s):", self.rms_window_spin)
            return w

        def _build_zoom_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.zoom_mode_combo = QComboBox()
            self.zoom_mode_combo.addItem("No zoom", "none")
            self.zoom_mode_combo.addItem("Zoom at stimulation onset", "onset")
            self.zoom_mode_combo.addItem("Zoom at stimulation end", "trigger_end")
            self.zoom_mode_combo.addItem("Both zooms", "both")
            self.zoom_mode_combo.currentIndexChanged.connect(self._update_zoom_visibility)
            form.addRow("Zoom mode:", self.zoom_mode_combo)
            layout.addLayout(form)

            onset_group = QGroupBox("Stimulation onset zoom")
            onset_form = QFormLayout(onset_group)
            onset_form.setSpacing(10)
            onset_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.zoom_onset_t0_spin = _spin(defaults.get("default_zoom_onset_t0_s", -0.1))
            self.zoom_onset_t1_spin = _spin(defaults.get("default_zoom_onset_t1_s", 0.2))
            onset_form.addRow("Start (s rel. stimulation):", self.zoom_onset_t0_spin)
            onset_form.addRow("End (s rel. stimulation):", self.zoom_onset_t1_spin)
            self.onset_group = onset_group
            layout.addWidget(onset_group)

            end_group = QGroupBox("Stimulation end zoom (next rising edge)")
            end_form = QFormLayout(end_group)
            end_form.setSpacing(10)
            end_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.zoom_end_t0_spin = _spin(defaults.get("default_zoom_end_t0_s", -0.1))
            self.zoom_end_t1_spin = _spin(defaults.get("default_zoom_end_t1_s", 0.2))
            end_form.addRow("Start (s rel. stimulation end):", self.zoom_end_t0_spin)
            end_form.addRow("End (s rel. stimulation end):", self.zoom_end_t1_spin)
            self.end_group = end_group
            layout.addWidget(end_group)

            hp_group = QGroupBox("Y axis — first filtered stimulation")
            hp_form = QFormLayout(hp_group)
            hp_form.setSpacing(10)
            hp_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.hp_ylim_check = QCheckBox("Fix Y axis (µV) on first filtered stimulation panels")
            self.hp_ylim_check.toggled.connect(self._update_hp_ylim_visibility)
            self.hp_ylim_min_spin = _spin(defaults.get("default_first_trigger_hp_ylim_min_uv", -200.0))
            self.hp_ylim_max_spin = _spin(defaults.get("default_first_trigger_hp_ylim_max_uv", 200.0))
            hp_form.addRow(self.hp_ylim_check)
            hp_form.addRow("Y min (µV):", self.hp_ylim_min_spin)
            hp_form.addRow("Y max (µV):", self.hp_ylim_max_spin)
            layout.addWidget(hp_group)
            layout.addStretch()
            return w

        def _build_display_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)
            top = QHBoxLayout()
            self.mea_check = QCheckBox("MEA map")
            self.mea_check.setChecked(True)
            self.impedance_check = QCheckBox("Impedance panel")
            self.impedance_check.setChecked(True)
            self.summary_rms_check = QCheckBox("RMS summary page")
            self.summary_rms_check.setChecked(True)
            self.summary_imp_check = QCheckBox("Impedance summary page")
            self.summary_imp_check.setChecked(True)
            for cb in (
                self.mea_check,
                self.impedance_check,
                self.summary_rms_check,
                self.summary_imp_check,
            ):
                top.addWidget(cb)
            top.addStretch()
            layout.addLayout(top)

            self.display_table = QTableWidget(len(PANEL_FIELD_NAMES), 4)
            self.display_table.setAlternatingRowColors(True)
            self.display_table.setShowGrid(True)
            self.display_table.setHorizontalHeaderLabels(
                ["Panel", "Full view", "Onset zoom", "End zoom"]
            )
            self.display_table.verticalHeader().setVisible(False)
            self.display_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.Stretch
            )
            for col in range(1, 4):
                self.display_table.horizontalHeader().setSectionResizeMode(
                    col, QHeaderView.ResizeMode.ResizeToContents
                )
            section_keys = ("full_view", "zoom_onset", "zoom_trigger_end")
            for row, panel_key in enumerate(PANEL_FIELD_NAMES):
                self.display_table.setItem(row, 0, QTableWidgetItem(PANEL_LABELS[panel_key]))
                item = self.display_table.item(row, 0)
                if item is not None:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._display_checkboxes[panel_key] = {}
                self._display_cell_wrappers[panel_key] = {}
                for col, section_key in enumerate(section_keys, start=1):
                    cb = QCheckBox()
                    cb.setChecked(True)
                    wrapper = QWidget()
                    wl = QHBoxLayout(wrapper)
                    wl.addWidget(cb)
                    wl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    wl.setContentsMargins(0, 0, 0, 0)
                    self.display_table.setCellWidget(row, col, wrapper)
                    self._display_checkboxes[panel_key][section_key] = cb
                    self._display_cell_wrappers[panel_key][section_key] = wrapper
            layout.addWidget(self.display_table)
            return w

        def _build_perf_tab(self) -> QWidget:
            w = QWidget()
            form = QFormLayout(w)
            form.setContentsMargins(16, 16, 16, 16)
            form.setSpacing(12)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.channel_workers_combo = QComboBox()
            for n in range(16, 0, -1):
                label = f"{n} worker{'s' if n > 1 else ''}"
                if n == 16:
                    label += " (max)"
                self.channel_workers_combo.addItem(label, n)
            self.channel_workers_combo.addItem("Auto (max CPU, capped at 16)", None)
            self.sampling_spin = _int_spin(defaults.get("default_sampling_percent", 100), minimum=1, maximum=100)
            form.addRow("Workers per channel:", self.channel_workers_combo)
            form.addRow("Spike display sampling (%):", self.sampling_spin)
            return w

        def _apply_defaults(self, d: dict) -> None:
            edge = d.get("default_edge", "falling")
            idx = self.edge_combo.findData(edge)
            if idx >= 0:
                self.edge_combo.setCurrentIndex(idx)
            for combo, key, fallback in (
                (self.filter_kind_combo, "default_intan_spike_filter_kind", "highpass"),
                (self.filter_type_combo, "default_intan_filter_type", "bessel"),
                (self.spike_mode_combo, "default_spike_threshold_mode", "fixed"),
                (self.spike_polarity_combo, "default_spike_threshold_polarity", "negative"),
            ):
                i = combo.findData(d.get(key, fallback))
                if i >= 0:
                    combo.setCurrentIndex(i)
            if d.get("default_probe_layout_json"):
                self.probe_json_edit.setText(str(d["default_probe_layout_json"]))
            if d.get("default_channel_workers") is not None:
                i = self.channel_workers_combo.findData(int(d["default_channel_workers"]))
                if i >= 0:
                    self.channel_workers_combo.setCurrentIndex(i)
            self.hp_ylim_check.setChecked(bool(d.get("default_first_trigger_hp_ylim_enabled", False)))
            self._add_file_row("")
            self._update_trigger_visibility()
            self._update_spike_mode_visibility()
            self._update_zoom_visibility()
            self._update_hp_ylim_visibility()

        def _add_file_row(self, path: str) -> None:
            row = self.FileEntryRow(self, initial_path=path)
            if path:
                row.legend_edit.setText(Path(path).stem)
            self._file_rows.append(row)
            self.files_layout.addWidget(row)
            self._on_files_changed()

        def _remove_file_row(self, row: FileEntryRow) -> None:
            if row in self._file_rows:
                self._file_rows.remove(row)
            self.files_layout.removeWidget(row)
            row.deleteLater()
            self._on_files_changed()

        def _suggest_pdf_title(self) -> str:
            paths = self._collect_paths()
            if len(paths) >= 2:
                return f"{Path(paths[0]).stem}_vs_{len(paths) - 1}_others"
            if len(paths) == 1:
                return Path(paths[0]).stem
            return ""

        def _refresh_pdf_title(self) -> None:
            suggested = self._suggest_pdf_title()
            current = self.pdf_title_edit.text().strip()
            # Keep user edits; auto-update while empty or still matching last suggestion.
            if not current or current == self._last_auto_pdf_title:
                self.pdf_title_edit.setText(suggested)
                self._last_auto_pdf_title = suggested

        def _on_files_changed(self) -> None:
            self._refresh_duration()
            self._refresh_pdf_title()

        def _collect_paths(self) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for row in self._file_rows:
                p = row.path()
                if not p:
                    continue
                resolved = str(Path(p).resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    out.append(p)
            return out

        def _refresh_duration(self) -> None:
            from core import peek_rhs_recording_info

            paths = self._collect_paths()
            durations: list[float] = []
            for path in paths:
                try:
                    n_samples, fs = peek_rhs_recording_info(Path(path))
                    durations.append(float(n_samples) / float(fs))
                except Exception:
                    continue
            self._cached_duration_s = min(durations) if durations else None
            if self.edge_combo.currentData() == "none":
                self._sync_sections(self._section_spec)

        def _sync_sections(self, changed: str) -> None:
            if self._section_sync_guard or self.edge_combo.currentData() != "none":
                return
            if self._cached_duration_s is None or self._cached_duration_s <= 0:
                return
            self._section_sync_guard = True
            try:
                self._section_spec = changed
                if changed == "count":
                    count = max(1, self.section_count_spin.value())
                    self.section_duration_spin.setValue(self._cached_duration_s / float(count))
                else:
                    duration = max(0.001, self.section_duration_spin.value())
                    count = max(1, int(self._cached_duration_s / duration))
                    exact = self._cached_duration_s / float(count)
                    self.section_count_spin.setValue(count)
                    self.section_duration_spin.setValue(exact)
            finally:
                self._section_sync_guard = False

        def _update_trigger_visibility(self) -> None:
            no_trigger = self.edge_combo.currentData() == "none"
            for w, show in (
                (self.threshold_label, not no_trigger),
                (self.threshold_spin, not no_trigger),
                (self.pre_label, not no_trigger),
                (self.pre_spin, not no_trigger),
                (self.post_label, not no_trigger),
                (self.post_spin, not no_trigger),
                (self.section_count_label, no_trigger),
                (self.section_count_spin, no_trigger),
                (self.section_duration_label, no_trigger),
                (self.section_duration_spin, no_trigger),
                (self.section_trigger_start_label, no_trigger),
                (self.section_trigger_start_spin, no_trigger),
                (self.section_trigger_end_label, no_trigger),
                (self.section_trigger_end_spin, no_trigger),
            ):
                w.setVisible(show)
            if no_trigger:
                self._refresh_duration()

        def _update_spike_mode_visibility(self) -> None:
            fixed = self.spike_mode_combo.currentData() == "fixed"
            self.spike_fixed_label.setVisible(fixed)
            self.spike_fixed_spin.setVisible(fixed)
            self.spike_rms_label.setVisible(not fixed)
            self.spike_rms_mult_spin.setVisible(not fixed)

        def _update_zoom_visibility(self) -> None:
            mode = str(self.zoom_mode_combo.currentData() or "both")
            onset_on = mode in ("onset", "both")
            end_on = mode in ("trigger_end", "both")
            self.onset_group.setVisible(onset_on)
            self.end_group.setVisible(end_on)
            self._sync_display_columns_with_zoom(onset_on=onset_on, end_on=end_on)

        def _sync_display_columns_with_zoom(self, *, onset_on: bool, end_on: bool) -> None:
            """Enable/disable Display columns according to the Zoom mode."""
            if not self._display_checkboxes:
                return
            for panel_key in PANEL_FIELD_NAMES:
                cbs = self._display_checkboxes[panel_key]
                wrappers = self._display_cell_wrappers.get(panel_key, {})
                cbs["zoom_onset"].setEnabled(onset_on)
                cbs["zoom_trigger_end"].setEnabled(end_on)
                onset_wrap = wrappers.get("zoom_onset")
                end_wrap = wrappers.get("zoom_trigger_end")
                if onset_wrap is not None:
                    onset_wrap.setEnabled(onset_on)
                    onset_wrap.setObjectName(
                        "" if onset_on else "displayColumnDisabled"
                    )
                    onset_wrap.style().unpolish(onset_wrap)
                    onset_wrap.style().polish(onset_wrap)
                if end_wrap is not None:
                    end_wrap.setEnabled(end_on)
                    end_wrap.setObjectName("" if end_on else "displayColumnDisabled")
                    end_wrap.style().unpolish(end_wrap)
                    end_wrap.style().polish(end_wrap)
            if hasattr(self, "display_table"):
                onset_header = "Onset zoom" if onset_on else "Onset zoom (disabled)"
                end_header = "End zoom" if end_on else "End zoom (disabled)"
                self.display_table.setHorizontalHeaderLabels(
                    ["Panel", "Full view", onset_header, end_header]
                )
                active = QBrush(QColor("#f8fafc"))
                active_bg = QBrush(QColor("#334155"))
                muted = QBrush(QColor("#e2e8f0"))
                muted_bg = QBrush(QColor("#94a3b8"))
                for col, enabled in ((2, onset_on), (3, end_on)):
                    item = self.display_table.horizontalHeaderItem(col)
                    if item is None:
                        continue
                    item.setForeground(active if enabled else muted)
                    item.setBackground(active_bg if enabled else muted_bg)

        def _update_hp_ylim_visibility(self) -> None:
            enabled = self.hp_ylim_check.isChecked()
            self.hp_ylim_min_spin.setEnabled(enabled)
            self.hp_ylim_max_spin.setEnabled(enabled)

        def _browse_save_dir(self) -> None:
            selected = QFileDialog.getExistingDirectory(self, "PDF output folder")
            if selected:
                self.save_dir_edit.setText(selected)

        def _browse_probe_json(self) -> None:
            selected, _ = QFileDialog.getOpenFileName(
                self, "JSON probeinterface", "", "JSON (*.json);;All (*)"
            )
            if selected:
                self.probe_json_edit.setText(selected)

        def _section_panels_from_ui(
            self, section_key: str, *, enabled: bool = True
        ) -> SectionPanels:
            if not enabled:
                return SectionPanels(
                    **{panel_key: False for panel_key in PANEL_FIELD_NAMES}
                )
            kwargs = {
                panel_key: self._display_checkboxes[panel_key][section_key].isChecked()
                for panel_key in PANEL_FIELD_NAMES
            }
            return SectionPanels(**kwargs)

        def _build_plot_display(self) -> PlotDisplaySettings:
            mode = str(self.zoom_mode_combo.currentData() or "both")
            onset_on = mode in ("onset", "both")
            end_on = mode in ("trigger_end", "both")
            return PlotDisplaySettings(
                mea_layout=self.mea_check.isChecked(),
                impedance=self.impedance_check.isChecked(),
                summary_rms_page=self.summary_rms_check.isChecked(),
                summary_impedance_page=self.summary_imp_check.isChecked(),
                full_view=self._section_panels_from_ui("full_view"),
                zoom_onset=self._section_panels_from_ui("zoom_onset", enabled=onset_on),
                zoom_trigger_end=self._section_panels_from_ui(
                    "zoom_trigger_end", enabled=end_on
                ),
            )

        def _build_configs(self) -> list[AnalysisConfig]:
            from core import validate_section_trigger_window
            from probe_layout import load_probe_layout_json

            paths_with_meta: list[tuple[str, FileEntryRow]] = [
                (row.path(), row) for row in self._file_rows if row.path()
            ]
            if not paths_with_meta:
                raise ValueError("Add at least one .rhs file.")
            seen: set[str] = set()
            unique: list[tuple[str, FileEntryRow]] = []
            for path, row in paths_with_meta:
                key = str(Path(path).resolve())
                if key not in seen:
                    seen.add(key)
                    unique.append((path, row))

            edge = str(self.edge_combo.currentData() or "falling")
            if edge not in ("falling", "rising", "none"):
                raise ValueError("Invalid stimulation mode.")
            zoom_mode = str(self.zoom_mode_combo.currentData() or "both")
            zoom_onset_t0 = float(self.zoom_onset_t0_spin.value())
            zoom_onset_t1 = float(self.zoom_onset_t1_spin.value())
            zoom_end_t0 = float(self.zoom_end_t0_spin.value())
            zoom_end_t1 = float(self.zoom_end_t1_spin.value())
            if zoom_mode in ("onset", "both") and zoom_onset_t1 <= zoom_onset_t0:
                raise ValueError("Onset zoom: end must be strictly greater than start.")
            if zoom_mode in ("trigger_end", "both") and zoom_end_t1 <= zoom_end_t0:
                raise ValueError("End zoom: end must be strictly greater than start.")

            section_count = int(self.section_count_spin.value())
            section_duration = float(self.section_duration_spin.value())
            section_duration_s = section_duration if self._section_spec == "duration" else None
            section_trigger_start = float(self.section_trigger_start_spin.value())
            section_trigger_end = float(self.section_trigger_end_spin.value())
            if edge == "none":
                if section_count < 1:
                    raise ValueError("Number of sections must be ≥ 1.")
                seg_dur = section_duration_s
                if seg_dur is None and self._cached_duration_s is not None:
                    seg_dur = self._cached_duration_s / float(section_count)
                if seg_dur is not None:
                    validate_section_trigger_window(
                        seg_dur, section_trigger_start, section_trigger_end
                    )

            save_text = self.save_dir_edit.text().strip()
            save_dir = Path(save_text) if save_text else None
            pdf_title = self.pdf_title_edit.text().strip() or self._suggest_pdf_title() or None
            probe_text = self.probe_json_edit.text().strip()
            probe_path: Path | None = None
            if probe_text:
                probe_path = Path(probe_text)
                if not probe_path.exists():
                    raise ValueError(f"Probe JSON file not found: {probe_path}")
                load_probe_layout_json(probe_path)

            channel_workers = self.channel_workers_combo.currentData()
            if channel_workers is not None:
                channel_workers = int(channel_workers)
                if channel_workers <= 0 or channel_workers > 16:
                    raise ValueError("Workers per channel: between 1 and 16.")

            sampling = int(self.sampling_spin.value())
            if sampling < 1 or sampling > 100:
                raise ValueError("Sampling: between 1 and 100%.")

            hp_enabled = self.hp_ylim_check.isChecked()
            hp_min = float(self.hp_ylim_min_spin.value())
            hp_max = float(self.hp_ylim_max_spin.value())
            if hp_enabled and hp_max <= hp_min:
                raise ValueError("HP Y axis: max must be strictly greater than min.")

            plot_display = self._build_plot_display()
            shared = dict(
                threshold=float(self.threshold_spin.value()),
                edge=edge,
                pre_s=float(self.pre_spin.value()),
                post_s=float(self.post_spin.value()),
                section_count=section_count,
                section_duration_s=section_duration_s,
                section_spec=self._section_spec,
                section_trigger_start_s=section_trigger_start,
                section_trigger_end_s=section_trigger_end,
                save_dir=save_dir,
                pdf_title=pdf_title,
                spike_threshold_uv=float(self.spike_fixed_spin.value()),
                spike_threshold_polarity=str(self.spike_polarity_combo.currentData()),
                spike_threshold_mode=str(self.spike_mode_combo.currentData()),
                spike_threshold_rms_multiplier=float(self.spike_rms_mult_spin.value()),
                psth_bin_window_s=float(self.psth_bin_spin.value()),
                rms_window_s=float(self.rms_window_spin.value()),
                zoom_mode=zoom_mode,
                zoom_onset_t0_s=zoom_onset_t0,
                zoom_onset_t1_s=zoom_onset_t1,
                zoom_end_t0_s=zoom_end_t0,
                zoom_end_t1_s=zoom_end_t1,
                first_trigger_hp_ylim_enabled=hp_enabled,
                first_trigger_hp_ylim_min_uv=hp_min,
                first_trigger_hp_ylim_max_uv=hp_max,
                intan_spike_filter_kind=str(self.filter_kind_combo.currentData()),
                intan_filter_order=int(self.filter_order_spin.value()),
                intan_filter_type=str(self.filter_type_combo.currentData()),
                intan_filter_cutoff_hz=float(self.filter_cutoff_spin.value()),
                channel_workers=channel_workers,
                sampling_percent=sampling,
                probe_layout_json=probe_path,
                plot_display=plot_display,
            )

            configs: list[AnalysisConfig] = []
            for path, row in unique:
                configs.append(
                    AnalysisConfig(
                        rhs_file=Path(path),
                        recording_label=row.label(),
                        recording_style=row.style(),
                        **shared,  # type: ignore[arg-type]
                    )
                )
            return configs

        def _append_log(self, message: str) -> None:
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_view.append(f"[{ts}] {message}")
            self.log_view.ensureCursorVisible()

        def _set_busy(self, running: bool) -> None:
            self.progress.setVisible(running)
            self.stop_btn.setEnabled(running)
            self.run_btn.setEnabled(not running)
            self.tabs.setEnabled(not running)

        def _finalize_thread(self) -> None:
            self._set_busy(False)
            if self._analysis_thread is not None:
                self._analysis_thread.deleteLater()
            self._analysis_thread = None

        def _on_ok(self, output: str) -> None:
            self._finalize_thread()
            if output:
                for line in output.splitlines():
                    if line.strip():
                        self._append_log(line)
            pdf_m = re.search(r"(?:Comparison )?PDF written: (.+)", output)
            pdf_path = pdf_m.group(1).strip() if pdf_m else None
            self.status_label.setText(
                f"Done — {pdf_path}" if pdf_path else "Analysis complete."
            )
            self._show_message(
                "Success",
                "Analysis complete."
                if pdf_path
                else "Analysis complete.\nSee the log for the PDF path.",
                QMessageBox.Icon.Information,
                detail=pdf_path,
                detail_label="PDF file",
            )

        def _on_err(self, msg: str) -> None:
            self._finalize_thread()
            self.status_label.setText("Failed.")
            self._append_log(f"Error: {msg}")
            self._show_message("Error", msg, QMessageBox.Icon.Critical)

        def _on_interrupted(self, msg: str) -> None:
            self._finalize_thread()
            self.status_label.setText("Analysis interrupted.")
            self._append_log(msg)

        def _show_message(
            self,
            title: str,
            text: str,
            icon: QMessageBox.Icon,
            *,
            detail: str | None = None,
            detail_label: str | None = None,
        ) -> None:
            """Readable modal dialog (avoids broken QMessageBox path layout)."""
            dlg = QDialog(self)
            dlg.setWindowTitle(title)
            dlg.setModal(True)
            dlg.setMinimumWidth(460)
            dlg.setMaximumWidth(640)
            dlg.setObjectName("appMessageDialog")
            dlg.setStyleSheet(
                "QDialog#appMessageDialog {"
                "  background-color: #ffffff;"
                "  color: #0f172a;"
                "}"
                "QDialog#appMessageDialog QLabel {"
                "  background: transparent;"
                "  color: #0f172a;"
                "  font-weight: 500;"
                "}"
                "QDialog#appMessageDialog QLabel#msgTitle {"
                "  font-size: 15px;"
                "  font-weight: 700;"
                "  color: #0f172a;"
                "}"
                "QDialog#appMessageDialog QLabel#msgBody {"
                "  font-size: 13px;"
                "  color: #334155;"
                "}"
                "QDialog#appMessageDialog QLabel#msgDetailLabel {"
                "  font-size: 12px;"
                "  font-weight: 600;"
                "  color: #475569;"
                "}"
                "QDialog#appMessageDialog QLineEdit {"
                "  background-color: #f8fafc;"
                "  color: #0f172a;"
                "  border: 2px solid #8896ab;"
                "  border-radius: 8px;"
                "  padding: 8px 10px;"
                "  font-size: 12px;"
                "  selection-background-color: #99f6e4;"
                "  selection-color: #0f172a;"
                "}"
                "QDialog#appMessageDialog QPushButton {"
                "  background-color: #0f766e;"
                "  color: #ffffff;"
                "  border: 2px solid #0d5c56;"
                "  border-radius: 8px;"
                "  padding: 8px 22px;"
                "  font-weight: 700;"
                "  min-width: 88px;"
                "}"
                "QDialog#appMessageDialog QPushButton:hover {"
                "  background-color: #0d9488;"
                "}"
            )

            root = QVBoxLayout(dlg)
            root.setContentsMargins(20, 18, 20, 16)
            root.setSpacing(12)

            top = QHBoxLayout()
            top.setSpacing(14)
            icon_label = QLabel()
            icon_label.setFixedSize(36, 36)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            std = dlg.style()
            if std is not None:
                pm = std.standardIcon(self._dialog_icon(icon)).pixmap(36, 36)
                icon_label.setPixmap(pm)
            top.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignTop)

            text_col = QVBoxLayout()
            text_col.setSpacing(6)
            title_lbl = QLabel(title)
            title_lbl.setObjectName("msgTitle")
            text_col.addWidget(title_lbl)
            body = QLabel(text)
            body.setObjectName("msgBody")
            body.setWordWrap(True)
            body.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            text_col.addWidget(body)
            top.addLayout(text_col, stretch=1)
            root.addLayout(top)

            if detail:
                if detail_label:
                    dl = QLabel(detail_label)
                    dl.setObjectName("msgDetailLabel")
                    root.addWidget(dl)
                path_edit = QLineEdit(detail)
                path_edit.setReadOnly(True)
                path_edit.setCursorPosition(0)
                path_edit.setToolTip(detail)
                root.addWidget(path_edit)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
            buttons.accepted.connect(dlg.accept)
            root.addWidget(buttons, alignment=Qt.AlignmentFlag.AlignRight)
            dlg.exec()

        @staticmethod
        def _dialog_icon(icon: QMessageBox.Icon):
            from PySide6.QtWidgets import QStyle

            mapping = {
                QMessageBox.Icon.Information: QStyle.StandardPixmap.SP_MessageBoxInformation,
                QMessageBox.Icon.Warning: QStyle.StandardPixmap.SP_MessageBoxWarning,
                QMessageBox.Icon.Critical: QStyle.StandardPixmap.SP_MessageBoxCritical,
                QMessageBox.Icon.Question: QStyle.StandardPixmap.SP_MessageBoxQuestion,
            }
            return mapping.get(icon, QStyle.StandardPixmap.SP_MessageBoxInformation)

        def _stop_analysis(self) -> None:
            if self._analysis_thread is not None and self._analysis_thread.isRunning():
                self._append_log("Stop requested — waiting for checkpoints…")
                self._analysis_thread.request_stop()

        def _run_analysis(self) -> None:
            if self._analysis_thread is not None and self._analysis_thread.isRunning():
                return
            try:
                configs = self._build_configs()
            except Exception as exc:
                self._append_log(f"Error: {exc}")
                self._show_message("Validation", str(exc), QMessageBox.Icon.Warning)
                return

            def task() -> None:
                if len(configs) == 1:
                    self._run_callback(configs[0])
                    return
                if len(configs) == 2:
                    self._run_comparison_callback(configs[0], configs[1])
                    return
                if self._run_multi_callback is not None:
                    self._run_multi_callback(configs)
                    return
                raise RuntimeError("Multi-file comparison unavailable.")

            thread = AnalysisThread(task)
            thread.finished_ok.connect(self._on_ok)
            thread.finished_err.connect(self._on_err)
            thread.finished_interrupted.connect(self._on_interrupted)
            self._analysis_thread = thread
            labels = [
                resolve_display_label(cfg.rhs_file.stem, cfg.recording_label) for cfg in configs
            ]
            self.status_label.setText("Analysis in progress…")
            self._append_log("Starting: " + " | ".join(labels))
            out_dir = configs[0].save_dir or configs[0].rhs_file.parent
            self._append_log(f"PDF folder: {out_dir}")
            self._set_busy(True)
            thread.start()

        def closeEvent(self, event) -> None:  # noqa: N802
            if self._analysis_thread is not None and self._analysis_thread.isRunning():
                self._append_log("Closing: stopping current processing…")
                self._analysis_thread.request_stop()
                self._analysis_thread.wait()
            super().closeEvent(event)

    window = MainWindow()
    window.show()
    return app.exec()
