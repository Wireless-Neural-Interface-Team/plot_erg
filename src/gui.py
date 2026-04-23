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


def launch_qt_gui(
    run_callback: Callable[[AnalysisConfig], None],
    run_comparison_callback: Callable[[AnalysisConfig, AnalysisConfig], None],
    default_threshold: float = 1.0,
    default_edge: str = "falling",
    default_lowpass_hz: float | None = None,
    default_spike_threshold_uv: float = -40.0,
    default_firing_rate_window_s: float = 0.025,
    default_spike_bandpass_low_hz: float | None = None,
    default_spike_bandpass_high_hz: float | None = None,
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
    browse_rhs_btn = QPushButton("Parcourir...")
    rhs_row = QHBoxLayout()
    rhs_row.addWidget(rhs_path_edit)
    rhs_row.addWidget(browse_rhs_btn)
    fl_single = QFormLayout()
    fl_single.addRow("Fichier RHS:", rhs_row)
    single_layout.addLayout(fl_single)
    run_btn = QPushButton("Lancer l'analyse")
    single_layout.addWidget(run_btn)
    tabs.addTab(tab_single, "Analyse")

    tab_compare = QWidget()
    compare_layout = QVBoxLayout(tab_compare)
    rhs1_edit = QLineEdit()
    rhs2_edit = QLineEdit()
    browse1_btn = QPushButton("Parcourir...")
    browse2_btn = QPushButton("Parcourir...")
    row1 = QHBoxLayout()
    row1.addWidget(rhs1_edit)
    row1.addWidget(browse1_btn)
    row2 = QHBoxLayout()
    row2.addWidget(rhs2_edit)
    row2.addWidget(browse2_btn)
    fl_cmp = QFormLayout()
    fl_cmp.addRow("Enregistrement 1 (.rhs):", row1)
    fl_cmp.addRow("Enregistrement 2 (.rhs):", row2)
    compare_layout.addLayout(fl_cmp)
    run_compare_btn = QPushButton("Comparer les deux enregistrements")
    compare_layout.addWidget(run_compare_btn)
    tabs.addTab(tab_compare, "Comparaison")

    # --- Parametres communs ---
    edge_combo = QComboBox()
    edge_combo.addItem("Front descendant", "falling")
    edge_combo.addItem("Front montant", "rising")
    idx = edge_combo.findData(default_edge)
    if idx >= 0:
        edge_combo.setCurrentIndex(idx)
    threshold_edit = QLineEdit(str(default_threshold))
    pre_edit = QLineEdit("2.0")
    post_edit = QLineEdit("10.0")
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
    work_dir_edit = QLineEdit()
    work_dir_edit.setPlaceholderText("vide = auto sous le dossier PDF (.plot_erg/…)")
    browse_work_btn = QPushButton("Parcourir…")
    work_dir_row = QHBoxLayout()
    work_dir_row.addWidget(work_dir_edit)
    work_dir_row.addWidget(browse_work_btn)
    keep_work_cb = QCheckBox("Conserver le dossier travail (amplifier_raw.npy)")
    keep_work_cb.setToolTip(
        "Sinon le dossier intermédiaire est supprimé après génération du PDF (économie disque)."
    )

    save_row = QHBoxLayout()
    save_row.addWidget(save_dir_edit)
    browse_save_btn = QPushButton("Parcourir...")
    save_row.addWidget(browse_save_btn)

    general_form = QFormLayout()
    general_form.addRow("Front sur ANALOG_IN 0:", edge_combo)
    general_form.addRow("Seuil trigger ANALOG_IN 0:", threshold_edit)
    general_form.addRow("Pre-trigger (s):", pre_edit)
    general_form.addRow("Post-trigger (s):", post_edit)
    general_form.addRow("Passe-bas Butterworth fc (Hz):", lowpass_edit)
    general_form.addRow("Dossier travail (mmap):", work_dir_row)
    general_form.addRow("", keep_work_cb)
    general_form.addRow("Dossier PDF (vide = dossier du .rhs):", save_row)

    general_group = QGroupBox("Paramètres généraux — trigger ANALOG_IN, moyennes amplificateur, fichiers")
    general_group.setLayout(general_form)

    spike_form = QFormLayout()
    spike_form.addRow("Seuil spikes amplificateur (µV) — raster, PSTH et ISI:", spike_threshold_edit)
    spike_form.addRow("Lissage gaussien PSTH / taux de décharge (σ, s):", firing_rate_window_edit)
    spike_form.addRow("Passe-bande signal (raster, PSTH, ISI) f_bas (Hz):", bandpass_spikes_low_edit)
    spike_form.addRow("Passe-bande signal (raster, PSTH, ISI) f_haut (Hz):", bandpass_spikes_high_edit)

    spike_group = QGroupBox("Raster, taux de décharge (PSTH) et ISI — panneaux PDF amplificateur")
    spike_group.setLayout(spike_form)
    spike_group.setToolTip(
        "Ces réglages concernent uniquement les graphiques spikes du PDF. "
        "Le raster, le PSTH (taux) et l’ISI utilisent les mêmes instants de spike (même seuil et même filtre passe-bande). "
        "Les fenêtres temporelles affichées (vue complète / zoom) sont définies dans le code (plotting.py)."
    )

    params_stack = QWidget()
    params_stack_layout = QVBoxLayout(params_stack)
    params_stack_layout.setContentsMargins(0, 0, 0, 0)
    params_stack_layout.addWidget(general_group)
    params_stack_layout.addWidget(spike_group)

    status_label = QLabel("Choisis un onglet, un ou deux fichiers .rhs, puis lance.")
    log_view = QTextEdit()
    log_view.setReadOnly(True)
    log_view.setPlaceholderText("Les logs d'execution s'afficheront ici...")

    progress = QProgressBar()
    progress.setRange(0, 0)
    progress.setFormat("Traitement en cours...")
    progress.setTextVisible(True)
    progress.setVisible(False)
    progress.setMinimumHeight(22)

    stop_btn = QPushButton("Arrêter")
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
        float | None,
        Path | None,
        float,
        float,
        float | None,
        float | None,
        Path | None,
        bool,
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
        work_text = work_dir_edit.text().strip()
        return (
            float(threshold_edit.text().strip()),
            edge,
            float(pre_edit.text().strip()),
            float(post_edit.text().strip()),
            lowpass_hz,
            Path(save_text) if save_text else None,
            float(spike_threshold_edit.text().strip()),
            float(firing_rate_window_edit.text().strip()),
            bp_lo,
            bp_hi,
            Path(work_text) if work_text else None,
            keep_work_cb.isChecked(),
        )

    def browse_rhs() -> None:
        selected, _ = QFileDialog.getOpenFileName(
            window,
            "Selectionner un fichier Intan RHS",
            "",
            "Fichiers Intan RHS (*.rhs);;Tous les fichiers (*)",
        )
        if selected:
            rhs_path_edit.setText(selected)

    def browse_rhs1() -> None:
        selected, _ = QFileDialog.getOpenFileName(
            window,
            "Enregistrement 1 — fichier RHS",
            "",
            "Fichiers Intan RHS (*.rhs);;Tous les fichiers (*)",
        )
        if selected:
            rhs1_edit.setText(selected)

    def browse_rhs2() -> None:
        selected, _ = QFileDialog.getOpenFileName(
            window,
            "Enregistrement 2 — fichier RHS",
            "",
            "Fichiers Intan RHS (*.rhs);;Tous les fichiers (*)",
        )
        if selected:
            rhs2_edit.setText(selected)

    def browse_save_dir() -> None:
        selected = QFileDialog.getExistingDirectory(window, "Dossier de sortie du PDF")
        if selected:
            save_dir_edit.setText(selected)

    def browse_work_dir() -> None:
        selected = QFileDialog.getExistingDirectory(window, "Dossier travail (fichiers intermédiaires mmap)")
        if selected:
            work_dir_edit.setText(selected)

    analysis_thread: QThread | None = None

    def set_busy(running: bool) -> None:
        progress.setVisible(running)
        stop_btn.setEnabled(running)
        run_btn.setEnabled(not running)
        run_compare_btn.setEnabled(not running)
        browse_rhs_btn.setEnabled(not running)
        browse1_btn.setEnabled(not running)
        browse2_btn.setEnabled(not running)
        browse_save_btn.setEnabled(not running)
        edge_combo.setEnabled(not running)
        lowpass_edit.setEnabled(not running)
        threshold_edit.setEnabled(not running)
        spike_threshold_edit.setEnabled(not running)
        firing_rate_window_edit.setEnabled(not running)
        bandpass_spikes_low_edit.setEnabled(not running)
        bandpass_spikes_high_edit.setEnabled(not running)
        pre_edit.setEnabled(not running)
        post_edit.setEnabled(not running)
        save_dir_edit.setEnabled(not running)
        work_dir_edit.setEnabled(not running)
        browse_work_btn.setEnabled(not running)
        keep_work_cb.setEnabled(not running)
        tabs.setEnabled(not running)

    def on_analysis_ok(output: str) -> None:
        nonlocal analysis_thread
        set_busy(False)
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
                "Succes",
                f"Analyse terminee.\n\n"
                f"Nombre de triggers detectes: {n_tot}\n"
                f"Nombre utilises pour la moyenne: {n_used}",
            )
        else:
            status_label.setText("Analyse terminee avec succes.")
            append_log("Analyse terminee avec succes.")
            QMessageBox.information(window, "Succes", "Analyse terminee.")

    def on_compare_ok(output: str) -> None:
        nonlocal analysis_thread
        set_busy(False)
        analysis_thread = None
        if output:
            append_log(output.replace("\n", "<br>"))
        pdf_m = re.search(r"PDF comparaison genere: (.+)", output)
        if pdf_m:
            status_label.setText(f"Comparaison terminee — {pdf_m.group(1)}")
        else:
            status_label.setText("Comparaison terminee.")
        QMessageBox.information(window, "Succes", "Comparaison terminee. Voir le log pour le chemin du PDF.")

    def on_analysis_err(msg: str) -> None:
        nonlocal analysis_thread
        set_busy(False)
        analysis_thread = None
        status_label.setText("Echec.")
        append_log(f"Erreur: {msg}")
        QMessageBox.critical(window, "Erreur", msg)

    def on_interrupted(msg: str) -> None:
        nonlocal analysis_thread
        set_busy(False)
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
            thr, edge, pre, post, lp_hz, save_p, sp_thr, fr_w, bp_lo, bp_hi, work_p, keep_w = (
                build_shared_params()
            )
            if fr_w <= 0:
                raise ValueError("La fenêtre de lissage du taux (s) doit être > 0.")
            config = AnalysisConfig(
                rhs_file=Path(rhs_text),
                threshold=thr,
                edge=edge,
                pre_s=pre,
                post_s=post,
                lowpass_cutoff_hz=lp_hz,
                save_dir=save_p,
                spike_threshold_uv=sp_thr,
                firing_rate_window_s=fr_w,
                spike_bandpass_low_hz=bp_lo,
                spike_bandpass_high_hz=bp_hi,
                work_dir=work_p,
                keep_intermediate_files=keep_w,
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

        status_label.setText("Analyse en cours...")
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
            p1 = rhs1_edit.text().strip()
            p2 = rhs2_edit.text().strip()
            if not p1 or not p2:
                raise ValueError("Choisis deux fichiers RHS.")
            if Path(p1).resolve() == Path(p2).resolve():
                raise ValueError("Les deux fichiers doivent etre differents.")
            thr, edge, pre, post, lp_hz, save_p, sp_thr, fr_w, bp_lo, bp_hi, work_p, keep_w = (
                build_shared_params()
            )
            if fr_w <= 0:
                raise ValueError("La fenêtre de lissage du taux (s) doit être > 0.")
            cfg_a = AnalysisConfig(
                rhs_file=Path(p1),
                threshold=thr,
                edge=edge,
                pre_s=pre,
                post_s=post,
                lowpass_cutoff_hz=lp_hz,
                save_dir=save_p,
                spike_threshold_uv=sp_thr,
                firing_rate_window_s=fr_w,
                spike_bandpass_low_hz=bp_lo,
                spike_bandpass_high_hz=bp_hi,
                work_dir=work_p,
                keep_intermediate_files=keep_w,
            )
            cfg_b = AnalysisConfig(
                rhs_file=Path(p2),
                threshold=thr,
                edge=edge,
                pre_s=pre,
                post_s=post,
                lowpass_cutoff_hz=lp_hz,
                save_dir=save_p,
                spike_threshold_uv=sp_thr,
                firing_rate_window_s=fr_w,
                spike_bandpass_low_hz=bp_lo,
                spike_bandpass_high_hz=bp_hi,
                work_dir=work_p,
                keep_intermediate_files=keep_w,
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
            run_comparison_callback(cfg_a, cfg_b)

        thread = AnalysisThread(task)
        thread.finished_ok.connect(on_compare_ok)
        thread.finished_err.connect(on_analysis_err)
        thread.finished_interrupted.connect(on_interrupted)

        status_label.setText("Comparaison en cours...")
        append_log(f"Comparaison: {cfg_a.rhs_file.name} vs {cfg_b.rhs_file.name}")
        out_dir = cfg_a.save_dir if cfg_a.save_dir is not None else cfg_a.rhs_file.parent
        append_log(f"PDF comparaison dans: {out_dir}")

        analysis_thread = thread
        set_busy(True)
        thread.start()

    browse_rhs_btn.clicked.connect(browse_rhs)
    browse1_btn.clicked.connect(browse_rhs1)
    browse2_btn.clicked.connect(browse_rhs2)
    browse_save_btn.clicked.connect(browse_save_dir)
    browse_work_btn.clicked.connect(browse_work_dir)
    run_btn.clicked.connect(run_analysis)
    run_compare_btn.clicked.connect(run_compare)
    stop_btn.clicked.connect(on_stop_clicked)

    window.show()
    return app.exec()
