"""Qt GUI for selecting RHS files, parameters, and running analysis / comparison."""

from __future__ import annotations

import contextlib
import io
import re
from datetime import datetime
from pathlib import Path
import threading
from typing import Callable

from config import AnalysisConfig
from core import analysis_cancel_scope
from probe_layout import load_probe_layout_json


def launch_qt_gui(
    run_callback: Callable[[AnalysisConfig], None],
    run_comparison_callback: Callable[[AnalysisConfig, AnalysisConfig], None],
    run_multi_comparison_callback: Callable[[list[AnalysisConfig]], None] | None = None,
    default_threshold: float = 1.0,
    default_edge: str = "falling",
    default_pre_s: float = 2.0,
    default_post_s: float = 10.0,
    default_lowpass_hz: float | None = None,
    default_curve_filter: str = "no filter",
    default_curve_filter_low_hz: float | None = None,
    default_curve_filter_high_hz: float | None = None,
    default_spike_threshold_uv: float = -40.0,
    default_psth_bin_window_s: float = 0.025,
    default_rms_window_s: float = 0.050,
    default_zoom_t0_s: float = -0.1,
    default_zoom_t1_s: float = 0.2,
    default_spike_bandpass_low_hz: float | None = None,
    default_spike_bandpass_high_hz: float | None = None,
    default_channel_workers: int | None = None,
    default_lightweight_plot: bool = False,
    default_sampling_percent: int = 100,
    default_probe_layout_json: Path | None = None,
) -> int:
    try:
        from PySide6.QtCore import QThread, Signal
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QProgressBar,
            QPushButton,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
            QCheckBox,
        )
    except ImportError as exc:
        raise RuntimeError("PySide6 is not installed. Run: pip install PySide6") from exc

    class AnalysisThread(QThread):
        finished_ok = Signal(str)
        finished_err = Signal(str)
        finished_interrupted = Signal(str)

        def __init__(
            self,
            target: Callable[[], None],
        ) -> None:
            super().__init__()
            self._target = target
            self._cancel_event = threading.Event()

        def request_stop(self) -> None:
            self._cancel_event.set()

        def run(self) -> None:
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    with analysis_cancel_scope(self._cancel_event):
                        self._target()
                self.finished_ok.emit(buf.getvalue().strip())
            except InterruptedError as exc:
                self.finished_interrupted.emit(str(exc))
            except Exception as exc:
                self.finished_err.emit(str(exc))

    app = QApplication.instance() or QApplication([])

    window = QWidget()
    window.setWindowTitle("Intan RHS Trigger Plotter")
    window.resize(860, 720)

    # Tabs (files only)
    tabs = QTabWidget()

    tab_single = QWidget()
    single_layout = QVBoxLayout(tab_single)
    rhs_path_edit = QLineEdit()
    browse_rhs_btn = QPushButton("Browse...")
    rhs_row = QHBoxLayout()
    rhs_row.addWidget(rhs_path_edit)
    rhs_row.addWidget(browse_rhs_btn)
    fl_single = QFormLayout()
    fl_single.addRow("RHS file:", rhs_row)
    single_layout.addLayout(fl_single)
    run_btn = QPushButton("Run analysis")
    single_layout.addWidget(run_btn)
    tabs.addTab(tab_single, "Analysis")

    tab_compare = QWidget()
    compare_layout = QVBoxLayout(tab_compare)
    rhs1_edit = QLineEdit()
    rhs2_edit = QLineEdit()
    browse1_btn = QPushButton("Browse...")
    browse2_btn = QPushButton("Browse...")
    add_rhs_field_btn = QPushButton("Add file")
    row1 = QHBoxLayout()
    row1.addWidget(rhs1_edit)
    row1.addWidget(browse1_btn)
    row2 = QHBoxLayout()
    row2.addWidget(rhs2_edit)
    row2.addWidget(browse2_btn)
    extra_files_widget = QWidget()
    extra_files_layout = QVBoxLayout(extra_files_widget)
    extra_files_layout.setContentsMargins(0, 0, 0, 0)
    extra_files_layout.setSpacing(6)
    fl_cmp = QFormLayout()
    fl_cmp.addRow("Recording 1 (.rhs):", row1)
    fl_cmp.addRow("Recording 2 (.rhs):", row2)
    fl_cmp.addRow("", add_rhs_field_btn)
    fl_cmp.addRow("Extra recordings (.rhs):", extra_files_widget)
    compare_layout.addLayout(fl_cmp)
    run_compare_btn = QPushButton("Compare recordings")
    compare_layout.addWidget(run_compare_btn)
    tabs.addTab(tab_compare, "Comparison")

    # Shared parameters
    edge_combo = QComboBox()
    edge_combo.addItem("Falling edge", "falling")
    edge_combo.addItem("Rising edge", "rising")
    idx = edge_combo.findData(default_edge)
    if idx >= 0:
        edge_combo.setCurrentIndex(idx)
    threshold_edit = QLineEdit(str(default_threshold))
    pre_edit = QLineEdit(str(default_pre_s))
    post_edit = QLineEdit(str(default_post_s))
    save_dir_edit = QLineEdit()
    filter_combo = QComboBox()
    filter_combo.addItem("highpass", "highpass")
    filter_combo.addItem("lowpass", "lowpass")
    filter_combo.addItem("bandpass", "bandpass")
    filter_combo.addItem("no filter", "no filter")
    curve_cutoff_low_edit = QLineEdit()
    curve_cutoff_high_edit = QLineEdit()
    curve_cutoff_low_edit.setPlaceholderText("Cutoff (Hz)")
    curve_cutoff_high_edit.setPlaceholderText("High cutoff (Hz)")
    curve_cutoff_row_widget = QWidget()
    curve_cutoff_row = QHBoxLayout(curve_cutoff_row_widget)
    curve_cutoff_row.setContentsMargins(0, 0, 0, 0)
    curve_cutoff_row.setSpacing(6)
    curve_cutoff_row.addWidget(curve_cutoff_low_edit)
    curve_cutoff_row.addWidget(curve_cutoff_high_edit)

    # Backward compatibility with old low-pass default.
    initial_filter_kind = default_curve_filter if default_curve_filter else "no filter"
    if (
        initial_filter_kind == "no filter"
        and default_curve_filter_low_hz is None
        and default_curve_filter_high_hz is None
        and default_lowpass_hz is not None
    ):
        initial_filter_kind = "lowpass"
        default_curve_filter_low_hz = default_lowpass_hz
    idx_filter = filter_combo.findData(initial_filter_kind)
    if idx_filter < 0:
        idx_filter = filter_combo.findData("no filter")
    if idx_filter >= 0:
        filter_combo.setCurrentIndex(idx_filter)
    if default_curve_filter_low_hz is not None:
        curve_cutoff_low_edit.setText(str(default_curve_filter_low_hz))
    if default_curve_filter_high_hz is not None:
        curve_cutoff_high_edit.setText(str(default_curve_filter_high_hz))
    spike_threshold_edit = QLineEdit(str(default_spike_threshold_uv))
    spike_threshold_edit.setToolTip(
        "Threshold in µV (amplifier signal; optionally band-pass filtered). "
        "Value >= 0 : spike = rising crossing (signal crosses upward). "
        "Value < 0 : spike = falling crossing (signal crosses downward, e.g. negative peak). "
        "Detected times feed raster, PSTH / firing rate, and ISI. "
        "Independent of ANALOG_IN threshold for the trigger."
    )
    psth_bin_window_edit = QLineEdit(str(default_psth_bin_window_s))
    psth_bin_window_edit.setToolTip(
        "PSTH time window (seconds) used for each PSTH point."
    )
    rms_window_edit = QLineEdit(str(default_rms_window_s))
    rms_window_edit.setToolTip(
        "RMS computation window (seconds) used for moving-RMS calculation."
    )
    zoom_t0_edit = QLineEdit(str(default_zoom_t0_s))
    zoom_t1_edit = QLineEdit(str(default_zoom_t1_s))
    zoom_t0_edit.setToolTip("Zoom window start (seconds relative to trigger).")
    zoom_t1_edit.setToolTip("Zoom window end (seconds relative to trigger).")
    bandpass_spikes_low_edit = QLineEdit()
    bandpass_spikes_high_edit = QLineEdit()
    if default_spike_bandpass_low_hz is not None:
        bandpass_spikes_low_edit.setText(str(default_spike_bandpass_low_hz))
    if default_spike_bandpass_high_hz is not None:
        bandpass_spikes_high_edit.setText(str(default_spike_bandpass_high_hz))
    bandpass_spikes_low_edit.setPlaceholderText("empty = raw — e.g. 300")
    bandpass_spikes_high_edit.setPlaceholderText("empty = raw — e.g. 3000")
    _bp_tip = (
        "Butterworth band-pass (order 4) per channel before raster, PSTH, and ISI. "
        "Both empty = raw mmap signal. Both set = low and high cutoff (Hz); "
        "high cutoff must stay below Nyquist."
    )
    bandpass_spikes_low_edit.setToolTip("Low frequency (Hz). " + _bp_tip)
    bandpass_spikes_high_edit.setToolTip("High frequency (Hz). " + _bp_tip)
    keep_work_cb = QCheckBox("Keep work folder (amplifier_raw.npy)")
    keep_work_cb.setToolTip(
        "Otherwise the intermediate folder is removed after the PDF is generated (save disk)."
    )

    save_row = QHBoxLayout()
    save_row.addWidget(save_dir_edit)
    browse_save_btn = QPushButton("Browse...")
    save_row.addWidget(browse_save_btn)
    pdf_title_edit = QLineEdit()
    pdf_title_edit.setPlaceholderText("empty = auto name from .rhs file")
    probe_layout_json_edit = QLineEdit()
    if default_probe_layout_json is not None:
        probe_layout_json_edit.setText(str(default_probe_layout_json))
    probe_layout_json_edit.setPlaceholderText("optional — probeinterface JSON (MEA map)")
    browse_probe_json_btn = QPushButton("Browse…")
    probe_json_row = QHBoxLayout()
    probe_json_row.addWidget(probe_layout_json_edit)
    probe_json_row.addWidget(browse_probe_json_btn)
    channel_workers_edit = QLineEdit()
    if default_channel_workers is not None:
        channel_workers_edit.setText(str(default_channel_workers))
    channel_workers_edit.setPlaceholderText("auto (default)")
    lightweight_plot_cb = QCheckBox("Lightweight PDF mode (raster/ISI downsample, lower dpi)")
    lightweight_plot_cb.setChecked(default_lightweight_plot)
    sampling_percent_edit = QLineEdit(str(default_sampling_percent))
    sampling_percent_edit.setPlaceholderText("1..100")

    def update_curve_filter_inputs_visibility() -> None:
        kind = str(filter_combo.currentData() or "no filter")
        if kind in {"highpass", "lowpass"}:
            curve_cutoff_low_edit.setVisible(True)
            curve_cutoff_low_edit.setPlaceholderText("Cutoff (Hz)")
            curve_cutoff_high_edit.setVisible(False)
            curve_cutoff_high_edit.clear()
        elif kind == "bandpass":
            curve_cutoff_low_edit.setVisible(True)
            curve_cutoff_low_edit.setPlaceholderText("Low cutoff (Hz)")
            curve_cutoff_high_edit.setVisible(True)
        else:
            curve_cutoff_low_edit.setVisible(False)
            curve_cutoff_high_edit.setVisible(False)
            curve_cutoff_low_edit.clear()
            curve_cutoff_high_edit.clear()
    filter_combo.currentIndexChanged.connect(update_curve_filter_inputs_visibility)
    update_curve_filter_inputs_visibility()

    general_form = QFormLayout()
    general_form.addRow("ANALOG_IN 0 edge:", edge_combo)
    general_form.addRow("ANALOG_IN 0 trigger threshold:", threshold_edit)
    general_form.addRow("Pre-trigger (s):", pre_edit)
    general_form.addRow("Post-trigger (s):", post_edit)
    general_form.addRow("Filter:", filter_combo)
    general_form.addRow("", curve_cutoff_row_widget)
    general_form.addRow("", keep_work_cb)
    general_form.addRow("PDF output folder (empty = .rhs folder):", save_row)
    general_form.addRow("PDF title/name:", pdf_title_edit)
    general_form.addRow("Probe MEA (JSON probeinterface):", probe_json_row)
    general_form.addRow("Channel workers (max 16, empty = auto):", channel_workers_edit)
    general_form.addRow("", lightweight_plot_cb)
    general_form.addRow("Spike display sampling (%):", sampling_percent_edit)

    general_group = QGroupBox("General settings — ANALOG_IN trigger, amplifier averages, files")
    general_group.setLayout(general_form)

    spike_form = QFormLayout()
    spike_form.addRow("Amplifier spike threshold (µV) — raster, PSTH and ISI:", spike_threshold_edit)
    spike_form.addRow("PSTH time window (s):", psth_bin_window_edit)
    spike_form.addRow("RMS window (s):", rms_window_edit)
    spike_form.addRow("Zoom window start (s, relative to trigger):", zoom_t0_edit)
    spike_form.addRow("Zoom window end (s, relative to trigger):", zoom_t1_edit)
    spike_form.addRow("Band-pass signal (raster, PSTH, ISI) low f (Hz):", bandpass_spikes_low_edit)
    spike_form.addRow("Band-pass signal (raster, PSTH, ISI) high f (Hz):", bandpass_spikes_high_edit)

    spike_group = QGroupBox("Raster, firing rate (PSTH) and ISI — amplifier PDF panels")
    spike_group.setLayout(spike_form)
    spike_group.setToolTip(
        "These settings apply only to amplifier spike panels in the PDF. "
        "Raster, PSTH (rate), and ISI share the same spike times (same threshold and band-pass). "
        "Zoom window is configured in this panel."
    )

    params_stack = QWidget()
    params_stack_layout = QVBoxLayout(params_stack)
    params_stack_layout.setContentsMargins(0, 0, 0, 0)
    params_stack_layout.addWidget(general_group)
    params_stack_layout.addWidget(spike_group)

    status_label = QLabel("Choose a tab, one or more .rhs files, then run.")
    log_view = QTextEdit()
    log_view.setReadOnly(True)
    log_view.setPlaceholderText("Execution logs appear here...")

    progress = QProgressBar()
    progress.setRange(0, 0)
    progress.setFormat("Processing...")
    progress.setTextVisible(True)
    progress.setVisible(False)
    progress.setMinimumHeight(22)

    stop_btn = QPushButton("Stop")
    stop_btn.setToolTip("Request stop of the current run (may take a few seconds).")
    stop_btn.setEnabled(False)
    stop_btn.setMinimumWidth(100)

    progress_row = QHBoxLayout()
    progress_row.addWidget(progress, stretch=1)
    progress_row.addWidget(stop_btn)

    main_layout = QVBoxLayout()
    main_layout.addWidget(tabs)
    main_layout.addWidget(params_stack)
    main_layout.addLayout(progress_row)
    main_layout.addWidget(status_label)
    main_layout.addWidget(log_view)
    window.setLayout(main_layout)

    def append_log(message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        log_view.append(f"[{ts}] {message}")
        log_view.ensureCursorVisible()

    def build_shared_params() -> tuple[
        float,
        str,
        float,
        float,
        str,
        float | None,
        float | None,
        Path | None,
        str | None,
        float,
        float,
        float,
        float | None,
        float | None,
        Path | None,
        bool,
        int | None,
        bool,
        int,
    ]:
        curve_filter_kind = str(filter_combo.currentData() or "no filter")
        cutoff_low_text = curve_cutoff_low_edit.text().strip()
        cutoff_high_text = curve_cutoff_high_edit.text().strip()
        curve_filter_low_hz: float | None = None
        curve_filter_high_hz: float | None = None
        if curve_filter_kind in {"highpass", "lowpass"}:
            if not cutoff_low_text:
                raise ValueError(f"Filter {curve_filter_kind}: enter cutoff (Hz).")
            curve_filter_low_hz = float(cutoff_low_text)
            if curve_filter_low_hz <= 0:
                raise ValueError("Filter cutoff must be > 0 Hz.")
            if cutoff_high_text:
                raise ValueError(
                    f"Filter {curve_filter_kind}: only one cutoff is required."
                )
        elif curve_filter_kind == "bandpass":
            if not cutoff_low_text or not cutoff_high_text:
                raise ValueError("Filter bandpass: enter low and high cutoffs (Hz).")
            curve_filter_low_hz = float(cutoff_low_text)
            curve_filter_high_hz = float(cutoff_high_text)
            if curve_filter_low_hz <= 0 or curve_filter_high_hz <= 0:
                raise ValueError("Filter bandpass: both frequencies must be > 0 Hz.")
            if curve_filter_low_hz >= curve_filter_high_hz:
                raise ValueError("Filter bandpass: low cutoff must be < high cutoff.")
        elif curve_filter_kind == "no filter":
            if cutoff_low_text or cutoff_high_text:
                raise ValueError("Filter no filter: leave cutoffs empty.")
        else:
            raise ValueError("Filter: invalid option.")
        save_text = save_dir_edit.text().strip()
        edge = edge_combo.currentData()
        if edge not in ("falling", "rising"):
            edge = "falling"
        bp_lo_text = bandpass_spikes_low_edit.text().strip()
        bp_hi_text = bandpass_spikes_high_edit.text().strip()
        bp_lo: float | None = None
        bp_hi: float | None = None
        if bp_lo_text or bp_hi_text:
            if not bp_lo_text or not bp_hi_text:
                raise ValueError(
                    "Spike band-pass: set both frequencies (Hz) or leave both fields empty."
                )
            bp_lo = float(bp_lo_text)
            bp_hi = float(bp_hi_text)
            if bp_lo <= 0 or bp_hi <= 0:
                raise ValueError("Spike band-pass: each frequency must be > 0 Hz.")
            if bp_lo >= bp_hi:
                raise ValueError("Spike band-pass: low frequency must be < high frequency.")
        pdf_title_text = pdf_title_edit.text().strip()
        cw_text = channel_workers_edit.text().strip()
        channel_workers: int | None = None
        if cw_text:
            channel_workers = int(cw_text)
            if channel_workers <= 0:
                raise ValueError("Channel workers: value must be > 0 (or leave empty for auto).")
            if channel_workers > 16:
                raise ValueError("Channel workers: maximum allowed is 16.")
        sampling_percent = int(sampling_percent_edit.text().strip())
        if sampling_percent < 1 or sampling_percent > 100:
            raise ValueError("Sampling (%): enter a value between 1 and 100.")
        zoom_t0_s = float(zoom_t0_edit.text().strip())
        zoom_t1_s = float(zoom_t1_edit.text().strip())
        if zoom_t1_s <= zoom_t0_s:
            raise ValueError("Zoom window: end must be strictly greater than start.")
        rms_window_s = float(rms_window_edit.text().strip())
        if rms_window_s <= 0:
            raise ValueError("RMS window (s): value must be > 0.")
        return (
            float(threshold_edit.text().strip()),
            edge,
            float(pre_edit.text().strip()),
            float(post_edit.text().strip()),
            curve_filter_kind,
            curve_filter_low_hz,
            curve_filter_high_hz,
            Path(save_text) if save_text else None,
            pdf_title_text if pdf_title_text else None,
            float(spike_threshold_edit.text().strip()),
            float(psth_bin_window_edit.text().strip()),
            rms_window_s,
            zoom_t0_s,
            zoom_t1_s,
            bp_lo,
            bp_hi,
            None,
            keep_work_cb.isChecked(),
            channel_workers,
            lightweight_plot_cb.isChecked(),
            sampling_percent,
        )

    def _suggest_pdf_title() -> str:
        """Build a PDF title string from the selected RHS file paths."""
        p_single = rhs_path_edit.text().strip()
        compare_paths = []
        p1 = rhs1_edit.text().strip()
        p2 = rhs2_edit.text().strip()
        if p1:
            compare_paths.append(p1)
        if p2:
            compare_paths.append(p2)
        for edit in extra_rhs_edits:
            p = edit.text().strip()
            if p:
                compare_paths.append(p)

        if p_single:
            return Path(p_single).stem
        if len(compare_paths) >= 2:
            return f"{Path(compare_paths[0]).stem}_vs_{len(compare_paths) - 1}_autres"
        if len(compare_paths) == 1:
            return Path(compare_paths[0]).stem
        return ""

    def refresh_pdf_title() -> None:
        suggested = _suggest_pdf_title()
        pdf_title_edit.setText(suggested)

    def browse_rhs() -> None:
        selected, _ = QFileDialog.getOpenFileName(
            window,
            "Select Intan RHS file",
            "",
            "Intan RHS files (*.rhs);;All files (*)",
        )
        if selected:
            rhs_path_edit.setText(selected)
            refresh_pdf_title()

    def browse_rhs1() -> None:
        selected, _ = QFileDialog.getOpenFileName(
            window,
            "Recording 1 — RHS file",
            "",
            "Intan RHS files (*.rhs);;All files (*)",
        )
        if selected:
            rhs1_edit.setText(selected)
            refresh_pdf_title()

    def browse_rhs2() -> None:
        selected, _ = QFileDialog.getOpenFileName(
            window,
            "Recording 2 — RHS file",
            "",
            "Intan RHS files (*.rhs);;All files (*)",
        )
        if selected:
            rhs2_edit.setText(selected)
            refresh_pdf_title()

    extra_rhs_edits: list[QLineEdit] = []
    extra_rhs_browse_buttons: list[QPushButton] = []
    extra_rhs_remove_buttons: list[QPushButton] = []
    extra_rhs_rows: list[QWidget] = []

    def add_rhs_field(initial_path: str = "") -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        path_edit = QLineEdit()
        if initial_path:
            path_edit.setText(initial_path)
        browse_btn = QPushButton("Browse...")
        remove_btn = QPushButton("Remove")
        row_layout.addWidget(path_edit)
        row_layout.addWidget(browse_btn)
        row_layout.addWidget(remove_btn)
        extra_files_layout.addWidget(row_widget)

        def browse_for_this_field() -> None:
            selected, _ = QFileDialog.getOpenFileName(
                window,
                "Extra recording — RHS file",
                "",
                "Intan RHS files (*.rhs);;All files (*)",
            )
            if selected:
                path_edit.setText(selected)
                refresh_pdf_title()

        def remove_this_field() -> None:
            if row_widget in extra_rhs_rows:
                idx = extra_rhs_rows.index(row_widget)
                extra_rhs_rows.pop(idx)
                extra_rhs_edits.pop(idx)
                extra_rhs_browse_buttons.pop(idx)
                extra_rhs_remove_buttons.pop(idx)
            extra_files_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            refresh_pdf_title()

        browse_btn.clicked.connect(browse_for_this_field)
        remove_btn.clicked.connect(remove_this_field)
        path_edit.textChanged.connect(refresh_pdf_title)

        extra_rhs_rows.append(row_widget)
        extra_rhs_edits.append(path_edit)
        extra_rhs_browse_buttons.append(browse_btn)
        extra_rhs_remove_buttons.append(remove_btn)
        refresh_pdf_title()

    def browse_save_dir() -> None:
        selected = QFileDialog.getExistingDirectory(window, "PDF output folder")
        if selected:
            save_dir_edit.setText(selected)

    def browse_probe_layout_json() -> None:
        selected, _ = QFileDialog.getOpenFileName(
            window,
            "JSON probeinterface (MEA)",
            "",
            "JSON (*.json);;All files (*)",
        )
        if selected:
            probe_layout_json_edit.setText(selected)

    def resolve_probe_layout_json_param() -> Path | None:
        pj = probe_layout_json_edit.text().strip()
        if not pj:
            return None
        pp = Path(pj)
        if not pp.exists():
            raise ValueError(f"Probe JSON file not found: {pp}")
        try:
            load_probe_layout_json(pp)
        except Exception as exc:
            raise ValueError(f"Invalid probe JSON: {exc}") from exc
        return pp

    analysis_thread: QThread | None = None

    def stop_analysis_thread_on_exit() -> None:
        """Avoid destroying QThread before the worker finishes."""
        nonlocal analysis_thread
        if analysis_thread is None:
            return
        if analysis_thread.isRunning():
            append_log("Shutdown: stopping current processing...")
            analysis_thread.request_stop()
            # Prefer graceful shutdown over destroying the thread too early.
            analysis_thread.wait()
        analysis_thread = None

    def set_busy(running: bool) -> None:
        progress.setVisible(running)
        stop_btn.setEnabled(running)
        run_btn.setEnabled(not running)
        run_compare_btn.setEnabled(not running)
        browse_rhs_btn.setEnabled(not running)
        browse1_btn.setEnabled(not running)
        browse2_btn.setEnabled(not running)
        add_rhs_field_btn.setEnabled(not running)
        for btn in extra_rhs_browse_buttons:
            btn.setEnabled(not running)
        for btn in extra_rhs_remove_buttons:
            btn.setEnabled(not running)
        for edit in extra_rhs_edits:
            edit.setEnabled(not running)
        browse_save_btn.setEnabled(not running)
        edge_combo.setEnabled(not running)
        filter_combo.setEnabled(not running)
        curve_cutoff_low_edit.setEnabled(not running)
        curve_cutoff_high_edit.setEnabled(not running)
        threshold_edit.setEnabled(not running)
        spike_threshold_edit.setEnabled(not running)
        psth_bin_window_edit.setEnabled(not running)
        rms_window_edit.setEnabled(not running)
        zoom_t0_edit.setEnabled(not running)
        zoom_t1_edit.setEnabled(not running)
        bandpass_spikes_low_edit.setEnabled(not running)
        bandpass_spikes_high_edit.setEnabled(not running)
        pre_edit.setEnabled(not running)
        post_edit.setEnabled(not running)
        save_dir_edit.setEnabled(not running)
        keep_work_cb.setEnabled(not running)
        channel_workers_edit.setEnabled(not running)
        lightweight_plot_cb.setEnabled(not running)
        sampling_percent_edit.setEnabled(not running)
        probe_layout_json_edit.setEnabled(not running)
        browse_probe_json_btn.setEnabled(not running)
        tabs.setEnabled(not running)

    def on_analysis_ok(output: str) -> None:
        nonlocal analysis_thread
        set_busy(False)
        if analysis_thread is not None:
            analysis_thread.deleteLater()
        analysis_thread = None
        if output:
            append_log(output.replace("\n", "<br>"))
        total_m = re.search(r"Total triggers detected: (\d+)", output)
        used_m = re.search(r"Triggers used for average: (\d+)", output)
        if total_m and used_m:
            n_tot, n_used = total_m.group(1), used_m.group(1)
            status_label.setText(
                f"Done — {n_tot} trigger(s) detected, {n_used} used for average."
            )
            append_log(f"Summary: {n_tot} trigger(s) total, {n_used} used for average.")
            QMessageBox.information(
                window,
                "Success",
                f"Analysis completed.\n\n"
                f"Total triggers detected: {n_tot}\n"
                f"Triggers used for average: {n_used}",
            )
        else:
            status_label.setText("Analysis completed successfully.")
            append_log("Analysis completed successfully.")
            QMessageBox.information(window, "Success", "Analysis completed.")

    def on_compare_ok(output: str) -> None:
        nonlocal analysis_thread
        set_busy(False)
        if analysis_thread is not None:
            analysis_thread.deleteLater()
        analysis_thread = None
        if output:
            append_log(output.replace("\n", "<br>"))
        pdf_m = re.search(r"Comparison PDF written: (.+)", output)
        if pdf_m:
            status_label.setText(f"Comparison completed — {pdf_m.group(1)}")
        else:
            status_label.setText("Comparison completed.")
        QMessageBox.information(window, "Success", "Comparison completed. See log for PDF path.")

    def on_analysis_err(msg: str) -> None:
        nonlocal analysis_thread
        set_busy(False)
        if analysis_thread is not None:
            analysis_thread.deleteLater()
        analysis_thread = None
        status_label.setText("Failed.")
        append_log(f"Error: {msg}")
        QMessageBox.critical(window, "Error", msg)

    def on_interrupted(msg: str) -> None:
        nonlocal analysis_thread
        set_busy(False)
        if analysis_thread is not None:
            analysis_thread.deleteLater()
        analysis_thread = None
        status_label.setText("Processing interrupted.")
        append_log(msg)

    def on_stop_clicked() -> None:
        if analysis_thread is not None and analysis_thread.isRunning():
            append_log("Stop requested — waiting for safe checkpoints...")
            analysis_thread.request_stop()

    def run_analysis() -> None:
        nonlocal analysis_thread
        if analysis_thread is not None and analysis_thread.isRunning():
            return
        try:
            rhs_file_path_text = rhs_path_edit.text().strip()
            if not rhs_file_path_text:
                raise ValueError("Choose an RHS file.")
            trigger_threshold, edge_mode, pre_window_s, post_window_s, curve_filter_kind, curve_filter_low_hz, curve_filter_high_hz, save_dir_path, pdf_title, spike_threshold_uv, psth_bin_window_s, rms_window_s, zoom_start_s, zoom_end_s, bandpass_low_hz, bandpass_high_hz, work_dir_path, keep_work_files, channel_worker_count, lightweight_mode_enabled, sampling_percent = (
                build_shared_params()
            )
            if psth_bin_window_s <= 0:
                raise ValueError("PSTH time window (s) must be > 0.")
            probe_layout_path = resolve_probe_layout_json_param()
            config = AnalysisConfig(
                rhs_file=Path(rhs_file_path_text),
                threshold=trigger_threshold,
                edge=edge_mode,
                pre_s=pre_window_s,
                post_s=post_window_s,
                lowpass_cutoff_hz=curve_filter_low_hz if curve_filter_kind == "lowpass" else None,
                curve_filter=curve_filter_kind,  # type: ignore[arg-type]
                curve_filter_low_hz=curve_filter_low_hz,
                curve_filter_high_hz=curve_filter_high_hz,
                save_dir=save_dir_path,
                pdf_title=pdf_title,
                spike_threshold_uv=spike_threshold_uv,
                psth_bin_window_s=psth_bin_window_s,
                rms_window_s=rms_window_s,
                zoom_t0_s=zoom_start_s,
                zoom_t1_s=zoom_end_s,
                spike_bandpass_low_hz=bandpass_low_hz,
                spike_bandpass_high_hz=bandpass_high_hz,
                work_dir=work_dir_path,
                keep_intermediate_files=keep_work_files,
                channel_workers=channel_worker_count,
                lightweight_plot=lightweight_mode_enabled,
                sampling_percent=sampling_percent,
                probe_layout_json=probe_layout_path,
            )
        except ValueError as exc:
            append_log(f"Error: {exc}")
            QMessageBox.warning(window, "Validation", str(exc))
            return
        except Exception as exc:
            append_log(f"Error: {exc}")
            QMessageBox.critical(window, "Error", str(exc))
            return

        def task() -> None:
            run_callback(config)

        thread = AnalysisThread(task)
        thread.finished_ok.connect(on_analysis_ok)
        thread.finished_err.connect(on_analysis_err)
        thread.finished_interrupted.connect(on_interrupted)

        status_label.setText("Analysis running...")
        append_log(f"Starting analysis: {config.rhs_file}")
        output_dir = config.save_dir if config.save_dir is not None else config.rhs_file.parent
        append_log(f"PDF will be saved to: {output_dir}")

        analysis_thread = thread
        set_busy(True)
        thread.start()

    def run_compare() -> None:
        nonlocal analysis_thread
        if analysis_thread is not None and analysis_thread.isRunning():
            return
        try:
            selected_paths: list[str] = []
            recording_path_1 = rhs1_edit.text().strip()
            recording_path_2 = rhs2_edit.text().strip()
            if recording_path_1:
                selected_paths.append(recording_path_1)
            if recording_path_2:
                selected_paths.append(recording_path_2)
            for edit in extra_rhs_edits:
                path_value = edit.text().strip()
                if path_value:
                    selected_paths.append(path_value)
            unique_paths: list[str] = []
            resolved_seen_paths: set[str] = set()
            for candidate_path in selected_paths:
                resolved_path = str(Path(candidate_path).resolve())
                if resolved_path not in resolved_seen_paths:
                    resolved_seen_paths.add(resolved_path)
                    unique_paths.append(candidate_path)
            if len(unique_paths) < 2:
                raise ValueError("Add at least two RHS files for comparison.")
            trigger_threshold, edge_mode, pre_window_s, post_window_s, curve_filter_kind, curve_filter_low_hz, curve_filter_high_hz, save_dir_path, pdf_title, spike_threshold_uv, psth_bin_window_s, rms_window_s, zoom_start_s, zoom_end_s, bandpass_low_hz, bandpass_high_hz, work_dir_path, keep_work_files, channel_worker_count, lightweight_mode_enabled, sampling_percent = (
                build_shared_params()
            )
            if psth_bin_window_s <= 0:
                raise ValueError("PSTH time window (s) must be > 0.")
            probe_layout_path = resolve_probe_layout_json_param()
            configs_to_compare: list[AnalysisConfig] = []
            for rhs_path in unique_paths:
                configs_to_compare.append(
                    AnalysisConfig(
                        rhs_file=Path(rhs_path),
                        threshold=trigger_threshold,
                        edge=edge_mode,
                        pre_s=pre_window_s,
                        post_s=post_window_s,
                        lowpass_cutoff_hz=curve_filter_low_hz if curve_filter_kind == "lowpass" else None,
                        curve_filter=curve_filter_kind,  # type: ignore[arg-type]
                        curve_filter_low_hz=curve_filter_low_hz,
                        curve_filter_high_hz=curve_filter_high_hz,
                        save_dir=save_dir_path,
                        pdf_title=pdf_title,
                        spike_threshold_uv=spike_threshold_uv,
                        psth_bin_window_s=psth_bin_window_s,
                        rms_window_s=rms_window_s,
                        zoom_t0_s=zoom_start_s,
                        zoom_t1_s=zoom_end_s,
                        spike_bandpass_low_hz=bandpass_low_hz,
                        spike_bandpass_high_hz=bandpass_high_hz,
                        work_dir=work_dir_path,
                        keep_intermediate_files=keep_work_files,
                        channel_workers=channel_worker_count,
                        lightweight_plot=lightweight_mode_enabled,
                        sampling_percent=sampling_percent,
                        probe_layout_json=probe_layout_path,
                    )
                )
        except ValueError as exc:
            append_log(f"Error: {exc}")
            QMessageBox.warning(window, "Validation", str(exc))
            return
        except Exception as exc:
            append_log(f"Error: {exc}")
            QMessageBox.critical(window, "Error", str(exc))
            return

        def task() -> None:
            if len(configs_to_compare) == 2:
                run_comparison_callback(configs_to_compare[0], configs_to_compare[1])
                return
            if run_multi_comparison_callback is None:
                raise RuntimeError(
                    "Multi-file comparison unavailable in this build (missing backend)."
                )
            run_multi_comparison_callback(configs_to_compare)

        thread = AnalysisThread(task)
        thread.finished_ok.connect(on_compare_ok)
        thread.finished_err.connect(on_analysis_err)
        thread.finished_interrupted.connect(on_interrupted)

        status_label.setText("Comparison running...")
        if len(configs_to_compare) == 2:
            append_log(f"Comparison: {configs_to_compare[0].rhs_file.name} vs {configs_to_compare[1].rhs_file.name}")
        else:
            append_log(
                "Multi comparison: "
                + " | ".join(cfg.rhs_file.name for cfg in configs_to_compare)
            )
        output_dir = (
            configs_to_compare[0].save_dir
            if configs_to_compare[0].save_dir is not None
            else configs_to_compare[0].rhs_file.parent
        )
        append_log(f"Comparison PDF folder: {output_dir}")

        analysis_thread = thread
        set_busy(True)
        thread.start()

    browse_rhs_btn.clicked.connect(browse_rhs)
    browse1_btn.clicked.connect(browse_rhs1)
    browse2_btn.clicked.connect(browse_rhs2)
    add_rhs_field_btn.clicked.connect(lambda: add_rhs_field(""))
    browse_save_btn.clicked.connect(browse_save_dir)
    browse_probe_json_btn.clicked.connect(browse_probe_layout_json)
    rhs_path_edit.textChanged.connect(refresh_pdf_title)
    rhs1_edit.textChanged.connect(refresh_pdf_title)
    rhs2_edit.textChanged.connect(refresh_pdf_title)
    run_btn.clicked.connect(run_analysis)
    run_compare_btn.clicked.connect(run_compare)
    stop_btn.clicked.connect(on_stop_clicked)
    app.aboutToQuit.connect(stop_analysis_thread_on_exit)

    window.show()
    return app.exec()
