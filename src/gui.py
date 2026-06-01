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
from core import analysis_cancel_scope, peek_rhs_recording_info, validate_section_trigger_window
from probe_layout import load_probe_layout_json


def launch_qt_gui(
    run_callback: Callable[[AnalysisConfig], None],
    run_comparison_callback: Callable[[AnalysisConfig, AnalysisConfig], None],
    run_multi_comparison_callback: Callable[[list[AnalysisConfig]], None] | None = None,
    default_threshold: float = 1.0,
    default_edge: str = "falling",
    default_pre_s: float = 2.0,
    default_post_s: float = 10.0,
    default_section_count: int = 10,
    default_section_duration_s: float | None = None,
    default_section_spec: str = "count",
    default_section_trigger_start_s: float = 1.0,
    default_section_trigger_end_s: float = 4.0,
    default_lowpass_hz: float | None = None,
    default_curve_filter: str = "no filter",
    default_curve_filter_low_hz: float | None = None,
    default_curve_filter_high_hz: float | None = None,
    default_spike_threshold_uv: float = 15.0,
    default_spike_threshold_mode: str = "fixed",
    default_spike_threshold_rms_multiplier: float = 4.0,
    default_psth_bin_window_s: float = 0.025,
    default_rms_window_s: float = 0.050,
    default_zoom_t0_s: float = -0.1,
    default_zoom_t1_s: float = 0.2,
    default_spike_bandpass_low_hz: float | None = None,
    default_spike_bandpass_high_hz: float | None = None,
    default_channel_workers: int | None = None,
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

    # Unified files panel (1+ recordings), no tabs.
    files_panel = QWidget()
    compare_layout = QVBoxLayout(files_panel)
    rhs1_edit = QLineEdit()
    browse1_btn = QPushButton("Browse...")
    add_rhs_field_btn = QPushButton("Add file")
    row1 = QHBoxLayout()
    row1.addWidget(rhs1_edit)
    row1.addWidget(browse1_btn)
    extra_files_widget = QWidget()
    extra_files_layout = QVBoxLayout(extra_files_widget)
    extra_files_layout.setContentsMargins(0, 0, 0, 0)
    extra_files_layout.setSpacing(6)
    fl_cmp = QFormLayout()
    fl_cmp.addRow("Recording (.rhs):", row1)
    fl_cmp.addRow("", add_rhs_field_btn)
    fl_cmp.addRow("Additional recordings (.rhs):", extra_files_widget)
    compare_layout.addLayout(fl_cmp)
    run_compare_btn = QPushButton("Run analysis")
    compare_layout.addWidget(run_compare_btn)

    # Shared parameters
    edge_combo = QComboBox()
    edge_combo.addItem("Falling edge", "falling")
    edge_combo.addItem("Rising edge", "rising")
    edge_combo.addItem("No trigger", "none")
    idx = edge_combo.findData(default_edge if default_edge in {"falling", "rising", "none"} else "falling")
    if idx >= 0:
        edge_combo.setCurrentIndex(idx)
    threshold_edit = QLineEdit(str(default_threshold))
    pre_edit = QLineEdit(str(default_pre_s))
    post_edit = QLineEdit(str(default_post_s))
    section_count_edit = QLineEdit(str(default_section_count))
    section_count_edit.setToolTip(
        "Number of equal contiguous sections per recording (each section is averaged like one trigger)."
    )
    section_duration_edit = QLineEdit(
        "" if default_section_duration_s is None else str(default_section_duration_s)
    )
    section_duration_edit.setToolTip(
        "Duration of each section (s). Linked to section count from the shortest selected recording."
    )
    section_trigger_start_edit = QLineEdit(str(default_section_trigger_start_s))
    section_trigger_start_edit.setToolTip(
        "Imaginary trigger onset within each segment (seconds from segment start). Default: 1 s."
    )
    section_trigger_end_edit = QLineEdit(str(default_section_trigger_end_s))
    section_trigger_end_edit.setToolTip(
        "Imaginary trigger end within each segment (seconds from segment start). Default: 4 s."
    )
    _section_spec = str(default_section_spec if default_section_spec in {"count", "duration"} else "count")
    _section_sync_guard = False
    _cached_duration_s: float | None = None
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
    spike_threshold_mode_combo = QComboBox()
    spike_threshold_mode_combo.addItem("Fixed threshold (same for all channels)", "fixed")
    spike_threshold_mode_combo.addItem("Multiplier x mean RMS per channel", "rms_multiple")
    idx_mode = spike_threshold_mode_combo.findData(default_spike_threshold_mode)
    if idx_mode < 0:
        idx_mode = spike_threshold_mode_combo.findData("fixed")
    if idx_mode >= 0:
        spike_threshold_mode_combo.setCurrentIndex(idx_mode)
    spike_threshold_fixed_edit = QLineEdit(str(default_spike_threshold_uv))
    spike_threshold_fixed_edit.setToolTip(
        "Fixed threshold in µV (same value for every channel). "
        "Value >= 0: spike = rising crossing. "
        "Value < 0: spike = falling crossing (negative peaks)."
    )
    spike_threshold_rms_multiplier_edit = QLineEdit(str(default_spike_threshold_rms_multiplier))
    spike_threshold_rms_multiplier_edit.setToolTip(
        "Multiplier applied to the mean RMS computed for each channel. "
        "Effective threshold per channel = multiplier x mean RMS(channel)."
    )
    spike_threshold_value_row_widget = QWidget()
    spike_threshold_value_row = QHBoxLayout(spike_threshold_value_row_widget)
    spike_threshold_value_row.setContentsMargins(0, 0, 0, 0)
    spike_threshold_value_row.setSpacing(6)
    spike_threshold_fixed_label = QLabel("Fixed threshold (µV):")
    spike_threshold_rms_multiplier_label = QLabel("RMS multiplier:")
    spike_threshold_value_row.addWidget(spike_threshold_fixed_label)
    spike_threshold_value_row.addWidget(spike_threshold_fixed_edit)
    spike_threshold_value_row.addWidget(spike_threshold_rms_multiplier_label)
    spike_threshold_value_row.addWidget(spike_threshold_rms_multiplier_edit)
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

    def update_spike_threshold_inputs_visibility() -> None:
        mode = str(spike_threshold_mode_combo.currentData() or "fixed")
        is_fixed = mode == "fixed"
        spike_threshold_fixed_label.setVisible(is_fixed)
        spike_threshold_fixed_edit.setVisible(is_fixed)
        spike_threshold_rms_multiplier_label.setVisible(not is_fixed)
        spike_threshold_rms_multiplier_edit.setVisible(not is_fixed)

    spike_threshold_mode_combo.currentIndexChanged.connect(update_spike_threshold_inputs_visibility)
    update_spike_threshold_inputs_visibility()

    def _selected_rhs_paths() -> list[str]:
        paths: list[str] = []
        p1 = rhs1_edit.text().strip()
        if p1:
            paths.append(p1)
        for edit in extra_rhs_edits:
            p = edit.text().strip()
            if p:
                paths.append(p)
        return paths

    def refresh_recording_duration() -> None:
        nonlocal _cached_duration_s
        paths = _selected_rhs_paths()
        if not paths:
            _cached_duration_s = None
            return
        durations: list[float] = []
        for path_text in paths:
            try:
                n_samples, fs = peek_rhs_recording_info(Path(path_text))
                durations.append(float(n_samples) / float(fs))
            except Exception:
                continue
        _cached_duration_s = min(durations) if durations else None
        if edge_combo.currentData() == "none":
            sync_section_fields(_section_spec)

    def sync_section_fields(changed: str) -> None:
        nonlocal _section_sync_guard, _section_spec
        if _section_sync_guard or edge_combo.currentData() != "none":
            return
        if _cached_duration_s is None or _cached_duration_s <= 0:
            return
        _section_sync_guard = True
        try:
            _section_spec = changed
            if changed == "count":
                count_text = section_count_edit.text().strip()
                if not count_text:
                    return
                count = max(1, int(count_text))
                duration = _cached_duration_s / float(count)
                section_duration_edit.setText(f"{duration:.6g}")
            else:
                dur_text = section_duration_edit.text().strip()
                if not dur_text:
                    return
                duration = float(dur_text)
                if duration <= 0:
                    raise ValueError("Section duration must be > 0.")
                count = max(1, int(_cached_duration_s / duration))
                exact_duration = _cached_duration_s / float(count)
                section_count_edit.setText(str(count))
                section_duration_edit.setText(f"{exact_duration:.6g}")
        except ValueError:
            pass
        finally:
            _section_sync_guard = False

    def update_trigger_mode_visibility() -> None:
        is_no_trigger = str(edge_combo.currentData() or "falling") == "none"
        threshold_edit.setVisible(not is_no_trigger)
        pre_edit.setVisible(not is_no_trigger)
        post_edit.setVisible(not is_no_trigger)
        threshold_label.setVisible(not is_no_trigger)
        pre_label.setVisible(not is_no_trigger)
        post_label.setVisible(not is_no_trigger)
        section_count_edit.setVisible(is_no_trigger)
        section_duration_edit.setVisible(is_no_trigger)
        section_count_label.setVisible(is_no_trigger)
        section_duration_label.setVisible(is_no_trigger)
        section_trigger_start_edit.setVisible(is_no_trigger)
        section_trigger_end_edit.setVisible(is_no_trigger)
        section_trigger_start_label.setVisible(is_no_trigger)
        section_trigger_end_label.setVisible(is_no_trigger)
        if is_no_trigger:
            refresh_recording_duration()

    edge_combo.currentIndexChanged.connect(update_trigger_mode_visibility)

    general_form = QFormLayout()
    general_form.addRow("Trigger mode:", edge_combo)
    threshold_label = QLabel("ANALOG_IN 0 trigger threshold:")
    pre_label = QLabel("Pre-trigger (s):")
    post_label = QLabel("Post-trigger (s):")
    section_count_label = QLabel("Number of sections:")
    section_duration_label = QLabel("Section duration (s):")
    section_trigger_start_label = QLabel("Imaginary trigger start (s in segment):")
    section_trigger_end_label = QLabel("Imaginary trigger end (s in segment):")
    general_form.addRow(threshold_label, threshold_edit)
    general_form.addRow(pre_label, pre_edit)
    general_form.addRow(post_label, post_edit)
    general_form.addRow(section_count_label, section_count_edit)
    general_form.addRow(section_duration_label, section_duration_edit)
    general_form.addRow(section_trigger_start_label, section_trigger_start_edit)
    general_form.addRow(section_trigger_end_label, section_trigger_end_edit)
    general_form.addRow("Filter:", filter_combo)
    general_form.addRow("", curve_cutoff_row_widget)
    general_form.addRow("PDF output folder (empty = .rhs folder):", save_row)
    general_form.addRow("PDF title/name:", pdf_title_edit)
    general_form.addRow("Probe MEA (JSON probeinterface):", probe_json_row)
    general_form.addRow("Channel workers (max 16, empty = auto):", channel_workers_edit)
    general_form.addRow("Spike display sampling (%):", sampling_percent_edit)

    general_group = QGroupBox("General settings — segmentation, amplifier averages, files")
    general_group.setLayout(general_form)

    spike_form = QFormLayout()
    spike_form.addRow("Spike threshold mode — raster, PSTH and ISI:", spike_threshold_mode_combo)
    spike_form.addRow("Spike threshold parameters:", spike_threshold_value_row_widget)
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

    status_label = QLabel("Choose one or more .rhs files, then run.")
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
    main_layout.addWidget(files_panel)
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
        str,
        float,
        float,
        float,
        float,
        float | None,
        float | None,
        Path | None,
        int,
        float | None,
        str,
        float,
        float,
        int | None,
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
        edge = str(edge_combo.currentData() or "falling")
        if edge not in ("falling", "rising", "none"):
            edge = "falling"
        section_count = int(section_count_edit.text().strip() or "0")
        section_duration_text = section_duration_edit.text().strip()
        section_duration_s: float | None = None
        if section_duration_text:
            section_duration_s = float(section_duration_text)
        section_trigger_start_s = float(section_trigger_start_edit.text().strip())
        section_trigger_end_s = float(section_trigger_end_edit.text().strip())
        if edge == "none":
            if section_count < 1:
                raise ValueError("Number of sections must be >= 1.")
            if _section_spec == "duration":
                if section_duration_s is None or section_duration_s <= 0:
                    raise ValueError("Section duration (s) must be > 0.")
            segment_duration_s: float | None = None
            if _section_spec == "duration" and section_duration_s is not None:
                segment_duration_s = float(section_duration_s)
            elif _cached_duration_s is not None and section_count >= 1:
                segment_duration_s = float(_cached_duration_s) / float(section_count)
            if segment_duration_s is not None:
                validate_section_trigger_window(
                    segment_duration_s,
                    section_trigger_start_s,
                    section_trigger_end_s,
                )
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
        spike_threshold_mode = str(spike_threshold_mode_combo.currentData() or "fixed")
        spike_threshold_fixed_uv = float(spike_threshold_fixed_edit.text().strip())
        spike_threshold_rms_multiplier = float(
            spike_threshold_rms_multiplier_edit.text().strip()
        )
        if spike_threshold_mode not in {"fixed", "rms_multiple"}:
            raise ValueError("Spike threshold mode: invalid option.")
        if spike_threshold_mode == "rms_multiple" and spike_threshold_rms_multiplier <= 0:
            raise ValueError("RMS multiplier: value must be > 0.")
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
            spike_threshold_fixed_uv,
            spike_threshold_mode,
            spike_threshold_rms_multiplier,
            float(psth_bin_window_edit.text().strip()),
            rms_window_s,
            zoom_t0_s,
            zoom_t1_s,
            bp_lo,
            bp_hi,
            None,
            section_count,
            section_duration_s,
            _section_spec,
            section_trigger_start_s,
            section_trigger_end_s,
            channel_workers,
            sampling_percent,
        )

    def _suggest_pdf_title() -> str:
        """Build a PDF title string from the selected RHS file paths."""
        compare_paths = []
        p1 = rhs1_edit.text().strip()
        if p1:
            compare_paths.append(p1)
        for edit in extra_rhs_edits:
            p = edit.text().strip()
            if p:
                compare_paths.append(p)

        if len(compare_paths) >= 2:
            return f"{Path(compare_paths[0]).stem}_vs_{len(compare_paths) - 1}_autres"
        if len(compare_paths) == 1:
            return Path(compare_paths[0]).stem
        return ""

    def refresh_pdf_title() -> None:
        suggested = _suggest_pdf_title()
        pdf_title_edit.setText(suggested)

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
            refresh_recording_duration()

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
                refresh_recording_duration()

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
            refresh_recording_duration()

        browse_btn.clicked.connect(browse_for_this_field)
        remove_btn.clicked.connect(remove_this_field)
        path_edit.textChanged.connect(refresh_pdf_title)
        path_edit.textChanged.connect(refresh_recording_duration)

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
        run_compare_btn.setEnabled(not running)
        browse1_btn.setEnabled(not running)
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
        spike_threshold_mode_combo.setEnabled(not running)
        spike_threshold_fixed_edit.setEnabled(not running)
        spike_threshold_rms_multiplier_edit.setEnabled(not running)
        psth_bin_window_edit.setEnabled(not running)
        rms_window_edit.setEnabled(not running)
        zoom_t0_edit.setEnabled(not running)
        zoom_t1_edit.setEnabled(not running)
        bandpass_spikes_low_edit.setEnabled(not running)
        bandpass_spikes_high_edit.setEnabled(not running)
        pre_edit.setEnabled(not running)
        post_edit.setEnabled(not running)
        section_count_edit.setEnabled(not running)
        section_duration_edit.setEnabled(not running)
        section_trigger_start_edit.setEnabled(not running)
        section_trigger_end_edit.setEnabled(not running)
        save_dir_edit.setEnabled(not running)
        channel_workers_edit.setEnabled(not running)
        sampling_percent_edit.setEnabled(not running)
        probe_layout_json_edit.setEnabled(not running)
        browse_probe_json_btn.setEnabled(not running)

    def _finalize_thread_idle() -> None:
        nonlocal analysis_thread
        set_busy(False)
        if analysis_thread is not None:
            analysis_thread.deleteLater()
        analysis_thread = None

    def on_compare_ok(output: str) -> None:
        _finalize_thread_idle()
        if output:
            for line in output.splitlines():
                if line.strip():
                    append_log(line)
        pdf_m = re.search(r"(?:Comparison )?PDF written: (.+)", output)
        if pdf_m:
            status_label.setText(f"Processing completed — {pdf_m.group(1)}")
        else:
            status_label.setText("Processing completed.")
        QMessageBox.information(window, "Success", "Processing completed. See log for PDF path.")

    def on_analysis_err(msg: str) -> None:
        _finalize_thread_idle()
        status_label.setText("Failed.")
        append_log(f"Error: {msg}")
        QMessageBox.critical(window, "Error", msg)

    def on_interrupted(msg: str) -> None:
        _finalize_thread_idle()
        status_label.setText("Processing interrupted.")
        append_log(msg)

    def on_stop_clicked() -> None:
        if analysis_thread is not None and analysis_thread.isRunning():
            append_log("Stop requested — waiting for safe checkpoints...")
            analysis_thread.request_stop()

    def _dedupe_paths(paths: list[str]) -> list[str]:
        unique_paths: list[str] = []
        resolved_seen_paths: set[str] = set()
        for candidate_path in paths:
            resolved_path = str(Path(candidate_path).resolve())
            if resolved_path not in resolved_seen_paths:
                resolved_seen_paths.add(resolved_path)
                unique_paths.append(candidate_path)
        return unique_paths

    def _build_configs_from_paths(paths: list[str]) -> list[AnalysisConfig]:
        trigger_threshold, edge_mode, pre_window_s, post_window_s, curve_filter_kind, curve_filter_low_hz, curve_filter_high_hz, save_dir_path, pdf_title, spike_threshold_uv, spike_threshold_mode, spike_threshold_rms_multiplier, psth_bin_window_s, rms_window_s, zoom_start_s, zoom_end_s, bandpass_low_hz, bandpass_high_hz, work_dir_path, section_count, section_duration_s, section_spec, section_trigger_start_s, section_trigger_end_s, channel_worker_count, sampling_percent = (
            build_shared_params()
        )
        if psth_bin_window_s <= 0:
            raise ValueError("PSTH time window (s) must be > 0.")
        probe_layout_path = resolve_probe_layout_json_param()
        configs: list[AnalysisConfig] = []
        for rhs_path in paths:
            configs.append(
                AnalysisConfig(
                    rhs_file=Path(rhs_path),
                    threshold=trigger_threshold,
                    edge=edge_mode,  # type: ignore[arg-type]
                    pre_s=pre_window_s,
                    post_s=post_window_s,
                    section_count=section_count,
                    section_duration_s=section_duration_s,
                    section_spec=section_spec,  # type: ignore[arg-type]
                    section_trigger_start_s=section_trigger_start_s,
                    section_trigger_end_s=section_trigger_end_s,
                    lowpass_cutoff_hz=curve_filter_low_hz if curve_filter_kind == "lowpass" else None,
                    curve_filter=curve_filter_kind,  # type: ignore[arg-type]
                    curve_filter_low_hz=curve_filter_low_hz,
                    curve_filter_high_hz=curve_filter_high_hz,
                    save_dir=save_dir_path,
                    pdf_title=pdf_title,
                    spike_threshold_uv=spike_threshold_uv,
                    spike_threshold_mode=spike_threshold_mode,  # type: ignore[arg-type]
                    spike_threshold_rms_multiplier=spike_threshold_rms_multiplier,
                    psth_bin_window_s=psth_bin_window_s,
                    rms_window_s=rms_window_s,
                    zoom_t0_s=zoom_start_s,
                    zoom_t1_s=zoom_end_s,
                    spike_bandpass_low_hz=bandpass_low_hz,
                    spike_bandpass_high_hz=bandpass_high_hz,
                    work_dir=work_dir_path,
                    channel_workers=channel_worker_count,
                    sampling_percent=sampling_percent,
                    probe_layout_json=probe_layout_path,
                )
            )
        return configs

    def _start_batch_processing(configs: list[AnalysisConfig], source_label: str) -> None:
        nonlocal analysis_thread
        if analysis_thread is not None and analysis_thread.isRunning():
            return

        def task() -> None:
            if len(configs) == 1:
                run_callback(configs[0])
                return
            if len(configs) == 2:
                run_comparison_callback(configs[0], configs[1])
                return
            if run_multi_comparison_callback is not None:
                run_multi_comparison_callback(configs)
                return
            raise RuntimeError("Multi-file processing unavailable in this build.")

        thread = AnalysisThread(task)
        thread.finished_ok.connect(on_compare_ok)
        thread.finished_err.connect(on_analysis_err)
        thread.finished_interrupted.connect(on_interrupted)

        status_label.setText("Processing running...")
        append_log(f"{source_label}: " + " | ".join(cfg.rhs_file.name for cfg in configs))
        output_dir = (
            configs[0].save_dir
            if configs[0].save_dir is not None
            else configs[0].rhs_file.parent
        )
        append_log(f"PDF folder: {output_dir}")

        analysis_thread = thread
        set_busy(True)
        thread.start()

    def run_compare() -> None:
        try:
            selected_paths: list[str] = []
            recording_path_1 = rhs1_edit.text().strip()
            if recording_path_1:
                selected_paths.append(recording_path_1)
            for edit in extra_rhs_edits:
                path_value = edit.text().strip()
                if path_value:
                    selected_paths.append(path_value)
            unique_paths = _dedupe_paths(selected_paths)
            if len(unique_paths) < 1:
                raise ValueError("Add at least one RHS file.")
            configs_to_compare = _build_configs_from_paths(unique_paths)
        except ValueError as exc:
            append_log(f"Error: {exc}")
            QMessageBox.warning(window, "Validation", str(exc))
            return
        except Exception as exc:
            append_log(f"Error: {exc}")
            QMessageBox.critical(window, "Error", str(exc))
            return
        _start_batch_processing(configs_to_compare, "Batch processing")

    browse1_btn.clicked.connect(browse_rhs1)
    add_rhs_field_btn.clicked.connect(lambda: add_rhs_field(""))
    browse_save_btn.clicked.connect(browse_save_dir)
    browse_probe_json_btn.clicked.connect(browse_probe_layout_json)
    rhs1_edit.textChanged.connect(refresh_pdf_title)
    rhs1_edit.textChanged.connect(refresh_recording_duration)
    section_count_edit.textChanged.connect(lambda _text: sync_section_fields("count"))
    section_duration_edit.textChanged.connect(lambda _text: sync_section_fields("duration"))
    run_compare_btn.clicked.connect(run_compare)
    stop_btn.clicked.connect(on_stop_clicked)
    app.aboutToQuit.connect(stop_analysis_thread_on_exit)

    update_trigger_mode_visibility()

    window.show()
    return app.exec()
