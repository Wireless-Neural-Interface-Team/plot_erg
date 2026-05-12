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
    default_spike_threshold_uv: float = -40.0,
    default_firing_rate_window_s: float = 0.025,
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
        raise RuntimeError("PySide6 n'est pas installe. Execute: pip install PySide6") from exc

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

    # --- Onglets (fichiers seulement) ---
    tabs = QTabWidget()

    tab_single = QWidget()
    single_layout = QVBoxLayout(tab_single)
    rhs_path_edit = QLineEdit()
    browse_rhs_btn = QPushButton("Browse...")
    rhs_row = QHBoxLayout()
    rhs_row.addWidget(rhs_path_edit)
    rhs_row.addWidget(browse_rhs_btn)
    fl_single = QFormLayout()
    fl_single.addRow("Fichier RHS:", rhs_row)
    single_layout.addLayout(fl_single)
    run_btn = QPushButton("Lancer l'analyse")
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
    fl_cmp.addRow("Enregistrement 1 (.rhs):", row1)
    fl_cmp.addRow("Enregistrement 2 (.rhs):", row2)
    fl_cmp.addRow("", add_rhs_field_btn)
    fl_cmp.addRow("Fichiers supplementaires (.rhs):", extra_files_widget)
    compare_layout.addLayout(fl_cmp)
    run_compare_btn = QPushButton("Comparer les enregistrements")
    compare_layout.addWidget(run_compare_btn)
    tabs.addTab(tab_compare, "Comparison")

    # --- Parametres communs ---
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
    lowpass_edit = QLineEdit()
    if default_lowpass_hz is not None:
        lowpass_edit.setText(str(default_lowpass_hz))
    lowpass_edit.setPlaceholderText("ex: 300 — vide = pas de filtre")
    spike_threshold_edit = QLineEdit(str(default_spike_threshold_uv))
    spike_threshold_edit.setToolTip(
        "Seuil en µV (signal amplificateur, éventuellement filtré passe-bande si renseigné). "
        "Valeur ≥ 0 : spike = front montant (signal dépasse le seuil vers le haut). "
        "Valeur < 0 : spike = front descendant (signal passe sous le seuil, ex. pic négatif). "
        "Les instants détectés servent pour le raster, le PSTH / taux de décharge et l’ISI. "
        "Indépendant du seuil ANALOG_IN pour le trigger."
    )
    firing_rate_window_edit = QLineEdit(str(default_firing_rate_window_s))
    firing_rate_window_edit.setToolTip(
        "Largeur σ (secondes) du noyau gaussien appliqué au PSTH pour la courbe de taux (Hz)."
    )
    zoom_t0_edit = QLineEdit(str(default_zoom_t0_s))
    zoom_t1_edit = QLineEdit(str(default_zoom_t1_s))
    zoom_t0_edit.setToolTip("Début de la fenêtre de zoom (secondes relatives au trigger).")
    zoom_t1_edit.setToolTip("Fin de la fenêtre de zoom (secondes relatives au trigger).")
    bandpass_spikes_low_edit = QLineEdit()
    bandpass_spikes_high_edit = QLineEdit()
    if default_spike_bandpass_low_hz is not None:
        bandpass_spikes_low_edit.setText(str(default_spike_bandpass_low_hz))
    if default_spike_bandpass_high_hz is not None:
        bandpass_spikes_high_edit.setText(str(default_spike_bandpass_high_hz))
    bandpass_spikes_low_edit.setPlaceholderText("vide = brut — ex: 300")
    bandpass_spikes_high_edit.setPlaceholderText("vide = brut — ex: 3000")
    _bp_tip = (
        "Passe-bande Butterworth (ordre 4) sur chaque canal avant raster, PSTH et ISI. "
        "Les deux champs vides = signal brut (mmap). Les deux renseignés = fc basse et fc haute (Hz) ; "
        "fc haute doit rester sous la fréquence de Nyquist."
    )
    bandpass_spikes_low_edit.setToolTip("Fréquence basse (Hz). " + _bp_tip)
    bandpass_spikes_high_edit.setToolTip("Fréquence haute (Hz). " + _bp_tip)
    keep_work_cb = QCheckBox("Conserver le dossier travail (amplifier_raw.npy)")
    keep_work_cb.setToolTip(
        "Sinon le dossier intermédiaire est supprimé après génération du PDF (économie disque)."
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
    probe_layout_json_edit.setPlaceholderText("optionnel — JSON probeinterface (carte MEA)")
    browse_probe_json_btn = QPushButton("Parcourir…")
    probe_json_row = QHBoxLayout()
    probe_json_row.addWidget(probe_layout_json_edit)
    probe_json_row.addWidget(browse_probe_json_btn)
    channel_workers_edit = QLineEdit()
    if default_channel_workers is not None:
        channel_workers_edit.setText(str(default_channel_workers))
    channel_workers_edit.setPlaceholderText("auto (defaut)")
    lightweight_plot_cb = QCheckBox("Lightweight PDF mode (raster/ISI downsample, lower dpi)")
    lightweight_plot_cb.setChecked(default_lightweight_plot)
    sampling_percent_edit = QLineEdit(str(default_sampling_percent))
    sampling_percent_edit.setPlaceholderText("1..100")

    general_form = QFormLayout()
    general_form.addRow("ANALOG_IN 0 edge:", edge_combo)
    general_form.addRow("ANALOG_IN 0 trigger threshold:", threshold_edit)
    general_form.addRow("Pre-trigger (s):", pre_edit)
    general_form.addRow("Post-trigger (s):", post_edit)
    general_form.addRow("Passe-bas Butterworth fc (Hz):", lowpass_edit)
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
    spike_form.addRow("PSTH / firing-rate Gaussian smoothing (σ, s):", firing_rate_window_edit)
    spike_form.addRow("Zoom window start (s, relative to trigger):", zoom_t0_edit)
    spike_form.addRow("Zoom window end (s, relative to trigger):", zoom_t1_edit)
    spike_form.addRow("Band-pass signal (raster, PSTH, ISI) low f (Hz):", bandpass_spikes_low_edit)
    spike_form.addRow("Band-pass signal (raster, PSTH, ISI) high f (Hz):", bandpass_spikes_high_edit)

    spike_group = QGroupBox("Raster, firing rate (PSTH) and ISI — amplifier PDF panels")
    spike_group.setLayout(spike_form)
    spike_group.setToolTip(
        "Ces réglages concernent uniquement les graphiques spikes du PDF. "
        "Le raster, le PSTH (taux) et l’ISI utilisent les mêmes instants de spike (même seuil et même filtre passe-bande). "
        "La fenêtre de zoom est configurable dans ce panneau."
    )

    params_stack = QWidget()
    params_stack_layout = QVBoxLayout(params_stack)
    params_stack_layout.setContentsMargins(0, 0, 0, 0)
    params_stack_layout.addWidget(general_group)
    params_stack_layout.addWidget(spike_group)

    status_label = QLabel("Choose a tab, one or more .rhs files, then run.")
    log_view = QTextEdit()
    log_view.setReadOnly(True)
    log_view.setPlaceholderText("Les logs d'execution s'afficheront ici...")

    progress = QProgressBar()
    progress.setRange(0, 0)
    progress.setFormat("Traitement en cours...")
    progress.setTextVisible(True)
    progress.setVisible(False)
    progress.setMinimumHeight(22)

    stop_btn = QPushButton("Stop")
    stop_btn.setToolTip("Demander l'arrêt du traitement en cours (peut prendre quelques secondes).")
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
        float,
        float,
        float | None,
        Path | None,
        str | None,
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
        lp_text = lowpass_edit.text().strip()
        lowpass_hz: float | None = None
        if lp_text:
            lowpass_hz = float(lp_text)
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
                    "Passe-bande spikes : renseigner les deux fréquences (Hz) ou laisser les deux champs vides."
                )
            bp_lo = float(bp_lo_text)
            bp_hi = float(bp_hi_text)
            if bp_lo <= 0 or bp_hi <= 0:
                raise ValueError("Passe-bande spikes : chaque fréquence doit être > 0 Hz.")
            if bp_lo >= bp_hi:
                raise ValueError("Passe-bande spikes : la fréquence basse doit être < la fréquence haute.")
        pdf_title_text = pdf_title_edit.text().strip()
        cw_text = channel_workers_edit.text().strip()
        channel_workers: int | None = None
        if cw_text:
            channel_workers = int(cw_text)
            if channel_workers <= 0:
                raise ValueError("Workers canaux : valeur > 0 requise (ou laisser vide pour auto).")
            if channel_workers > 16:
                raise ValueError("Workers canaux : maximum autorisé = 16.")
        sampling_percent = int(sampling_percent_edit.text().strip())
        if sampling_percent < 1 or sampling_percent > 100:
            raise ValueError("Sampling (%) : renseigner une valeur entre 1 et 100.")
        zoom_t0_s = float(zoom_t0_edit.text().strip())
        zoom_t1_s = float(zoom_t1_edit.text().strip())
        if zoom_t1_s <= zoom_t0_s:
            raise ValueError("Fenêtre de zoom : la fin doit être strictement > au début.")
        return (
            float(threshold_edit.text().strip()),
            edge,
            float(pre_edit.text().strip()),
            float(post_edit.text().strip()),
            lowpass_hz,
            Path(save_text) if save_text else None,
            pdf_title_text if pdf_title_text else None,
            float(spike_threshold_edit.text().strip()),
            float(firing_rate_window_edit.text().strip()),
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
        """Construit un titre PDF à partir des fichiers RHS renseignés."""
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
            "Fichiers Intan RHS (*.rhs);;Tous les fichiers (*)",
        )
        if selected:
            rhs_path_edit.setText(selected)
            refresh_pdf_title()

    def browse_rhs1() -> None:
        selected, _ = QFileDialog.getOpenFileName(
            window,
            "Recording 1 — RHS file",
            "",
            "Fichiers Intan RHS (*.rhs);;Tous les fichiers (*)",
        )
        if selected:
            rhs1_edit.setText(selected)
            refresh_pdf_title()

    def browse_rhs2() -> None:
        selected, _ = QFileDialog.getOpenFileName(
            window,
            "Recording 2 — RHS file",
            "",
            "Fichiers Intan RHS (*.rhs);;Tous les fichiers (*)",
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
        browse_btn = QPushButton("Parcourir...")
        remove_btn = QPushButton("Supprimer")
        row_layout.addWidget(path_edit)
        row_layout.addWidget(browse_btn)
        row_layout.addWidget(remove_btn)
        extra_files_layout.addWidget(row_widget)

        def browse_for_this_field() -> None:
            selected, _ = QFileDialog.getOpenFileName(
                window,
                "Enregistrement supplementaire — fichier RHS",
                "",
                "Fichiers Intan RHS (*.rhs);;Tous les fichiers (*)",
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
            "JSON (*.json);;Tous les fichiers (*)",
        )
        if selected:
            probe_layout_json_edit.setText(selected)

    def resolve_probe_layout_json_param() -> Path | None:
        pj = probe_layout_json_edit.text().strip()
        if not pj:
            return None
        pp = Path(pj)
        if not pp.exists():
            raise ValueError(f"Fichier probe JSON introuvable : {pp}")
        try:
            load_probe_layout_json(pp)
        except Exception as exc:
            raise ValueError(f"JSON probe invalide : {exc}") from exc
        return pp

    analysis_thread: QThread | None = None

    def stop_analysis_thread_on_exit() -> None:
        """Évite QThread détruit avant la fin du worker."""
        nonlocal analysis_thread
        if analysis_thread is None:
            return
        if analysis_thread.isRunning():
            append_log("Fermeture: arrêt du traitement en cours...")
            analysis_thread.request_stop()
            # Au shutdown, on privilégie un arrêt propre plutôt qu'une destruction prématurée.
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
        lowpass_edit.setEnabled(not running)
        threshold_edit.setEnabled(not running)
        spike_threshold_edit.setEnabled(not running)
        firing_rate_window_edit.setEnabled(not running)
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
        total_m = re.search(r"Nombre total de triggers detectes: (\d+)", output)
        used_m = re.search(r"Nombre de triggers utilises pour la moyenne: (\d+)", output)
        if total_m and used_m:
            n_tot, n_used = total_m.group(1), used_m.group(1)
            status_label.setText(
                f"Termine — {n_tot} trigger(s) detecte(s), {n_used} utilise(s) pour la moyenne."
            )
            append_log(f"Resume: {n_tot} trigger(s) au total, {n_used} pour la moyenne.")
            QMessageBox.information(
                window,
                "Success",
                f"Analysis completed.\n\n"
                f"Nombre de triggers detectes: {n_tot}\n"
                f"Nombre utilises pour la moyenne: {n_used}",
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
        pdf_m = re.search(r"PDF comparaison genere: (.+)", output)
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
        status_label.setText("Echec.")
        append_log(f"Erreur: {msg}")
        QMessageBox.critical(window, "Erreur", msg)

    def on_interrupted(msg: str) -> None:
        nonlocal analysis_thread
        set_busy(False)
        if analysis_thread is not None:
            analysis_thread.deleteLater()
        analysis_thread = None
        status_label.setText("Traitement interrompu.")
        append_log(msg)

    def on_stop_clicked() -> None:
        if analysis_thread is not None and analysis_thread.isRunning():
            append_log("Arrêt demandé — attente des points de contrôle...")
            analysis_thread.request_stop()

    def run_analysis() -> None:
        nonlocal analysis_thread
        if analysis_thread is not None and analysis_thread.isRunning():
            return
        try:
            rhs_text = rhs_path_edit.text().strip()
            if not rhs_text:
                raise ValueError("Choisis un fichier RHS.")
            thr, edge, pre, post, lp_hz, save_p, pdf_title, sp_thr, fr_w, zoom_t0_s, zoom_t1_s, bp_lo, bp_hi, work_p, keep_w, ch_w, light_plot, samp_pct = (
                build_shared_params()
            )
            if fr_w <= 0:
                raise ValueError("La fenêtre de lissage du taux (s) doit être > 0.")
            probe_p = resolve_probe_layout_json_param()
            config = AnalysisConfig(
                rhs_file=Path(rhs_text),
                threshold=thr,
                edge=edge,
                pre_s=pre,
                post_s=post,
                lowpass_cutoff_hz=lp_hz,
                save_dir=save_p,
                pdf_title=pdf_title,
                spike_threshold_uv=sp_thr,
                firing_rate_window_s=fr_w,
                zoom_t0_s=zoom_t0_s,
                zoom_t1_s=zoom_t1_s,
                spike_bandpass_low_hz=bp_lo,
                spike_bandpass_high_hz=bp_hi,
                work_dir=work_p,
                keep_intermediate_files=keep_w,
                channel_workers=ch_w,
                lightweight_plot=light_plot,
                sampling_percent=samp_pct,
                probe_layout_json=probe_p,
            )
        except ValueError as exc:
            append_log(f"Erreur: {exc}")
            QMessageBox.warning(window, "Validation", str(exc))
            return
        except Exception as exc:
            append_log(f"Erreur: {exc}")
            QMessageBox.critical(window, "Erreur", str(exc))
            return

        def task() -> None:
            run_callback(config)

        thread = AnalysisThread(task)
        thread.finished_ok.connect(on_analysis_ok)
        thread.finished_err.connect(on_analysis_err)
        thread.finished_interrupted.connect(on_interrupted)

        status_label.setText("Analysis running...")
        append_log(f"Debut analyse: {config.rhs_file}")
        out_dir = config.save_dir if config.save_dir is not None else config.rhs_file.parent
        append_log(f"PDF sera enregistre dans: {out_dir}")

        analysis_thread = thread
        set_busy(True)
        thread.start()

    def run_compare() -> None:
        nonlocal analysis_thread
        if analysis_thread is not None and analysis_thread.isRunning():
            return
        try:
            selected_paths: list[str] = []
            p1 = rhs1_edit.text().strip()
            p2 = rhs2_edit.text().strip()
            if p1:
                selected_paths.append(p1)
            if p2:
                selected_paths.append(p2)
            for edit in extra_rhs_edits:
                p = edit.text().strip()
                if p:
                    selected_paths.append(p)
            uniq_paths: list[str] = []
            seen: set[str] = set()
            for p in selected_paths:
                pr = str(Path(p).resolve())
                if pr not in seen:
                    seen.add(pr)
                    uniq_paths.append(p)
            if len(uniq_paths) < 2:
                raise ValueError("Ajoute au moins deux fichiers RHS pour la comparaison.")
            thr, edge, pre, post, lp_hz, save_p, pdf_title, sp_thr, fr_w, zoom_t0_s, zoom_t1_s, bp_lo, bp_hi, work_p, keep_w, ch_w, light_plot, samp_pct = (
                build_shared_params()
            )
            if fr_w <= 0:
                raise ValueError("La fenêtre de lissage du taux (s) doit être > 0.")
            probe_p = resolve_probe_layout_json_param()
            cfgs: list[AnalysisConfig] = []
            for p in uniq_paths:
                cfgs.append(
                    AnalysisConfig(
                        rhs_file=Path(p),
                        threshold=thr,
                        edge=edge,
                        pre_s=pre,
                        post_s=post,
                        lowpass_cutoff_hz=lp_hz,
                        save_dir=save_p,
                        pdf_title=pdf_title,
                        spike_threshold_uv=sp_thr,
                        firing_rate_window_s=fr_w,
                        zoom_t0_s=zoom_t0_s,
                        zoom_t1_s=zoom_t1_s,
                        spike_bandpass_low_hz=bp_lo,
                        spike_bandpass_high_hz=bp_hi,
                        work_dir=work_p,
                        keep_intermediate_files=keep_w,
                        channel_workers=ch_w,
                        lightweight_plot=light_plot,
                        sampling_percent=samp_pct,
                        probe_layout_json=probe_p,
                    )
                )
        except ValueError as exc:
            append_log(f"Erreur: {exc}")
            QMessageBox.warning(window, "Validation", str(exc))
            return
        except Exception as exc:
            append_log(f"Erreur: {exc}")
            QMessageBox.critical(window, "Erreur", str(exc))
            return

        def task() -> None:
            if len(cfgs) == 2:
                run_comparison_callback(cfgs[0], cfgs[1])
                return
            if run_multi_comparison_callback is None:
                raise RuntimeError(
                    "Comparaison multi-fichiers indisponible dans cette version (backend manquant)."
                )
            run_multi_comparison_callback(cfgs)

        thread = AnalysisThread(task)
        thread.finished_ok.connect(on_compare_ok)
        thread.finished_err.connect(on_analysis_err)
        thread.finished_interrupted.connect(on_interrupted)

        status_label.setText("Comparison running...")
        if len(cfgs) == 2:
            append_log(f"Comparaison: {cfgs[0].rhs_file.name} vs {cfgs[1].rhs_file.name}")
        else:
            append_log(
                "Comparaison multi: "
                + " | ".join(c.rhs_file.name for c in cfgs)
            )
        out_dir = cfgs[0].save_dir if cfgs[0].save_dir is not None else cfgs[0].rhs_file.parent
        append_log(f"PDF comparaison dans: {out_dir}")

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
