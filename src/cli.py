from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from config import AnalysisConfig
from core import (
    AmplifierSpikeSource,
    check_analysis_cancelled,
    compute_average_per_channel,
    detect_edges,
    get_analog_in0_signal,
    get_channel_names,
    get_sampling_rate,
    load_rhs_file,
    mean_time_to_next_rising_edge_s,
    persist_amplifier_float32,
    resolve_work_dir,
    valid_triggers_and_timebase,
)
from gui import launch_qt_gui
from impedance_tracking import collect_impedance_sessions
from plotting import plot_channel_averages, plot_channel_comparison, plot_channel_multi_comparison


def _to_temp_mmap(arr: np.ndarray, folder: Path, name: str) -> np.ndarray:
    """Persist array to temporary .npy and reopen as read-only memmap."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npy"
    np.save(path, np.ascontiguousarray(arr, dtype=np.float32))
    return np.load(path, mmap_mode="r")


def _compute_payload_for_comparison(config: AnalysisConfig) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    int,
    int,
    float,
    float | None,
    np.ndarray,
    np.ndarray,
    int,
    int,
    str,
]:
    """Calcule une analyse sérialisable pour exécution en process séparé."""
    worker_cfg = replace(config, keep_intermediate_files=True)
    (
        mean_per_channel,
        t_rel,
        channel_names,
        n_valid,
        n_total,
        fs,
        end_rising_s,
        spike_src,
        mean_per_channel_raw,
    ) = compute_average_per_channel(worker_cfg)
    try:
        if spike_src.work_dir is None:
            raise RuntimeError("Spike work dir not found for parallel comparison.")
        amp_path = spike_src.work_dir / "amplifier_raw.npy"
        return (
            mean_per_channel,
            t_rel,
            channel_names,
            n_valid,
            n_total,
            fs,
            end_rising_s,
            mean_per_channel_raw,
            spike_src.valid_triggers.copy(),
            int(spike_src.pre_n),
            int(spike_src.post_n),
            str(amp_path),
        )
    finally:
        # Le worker libère son mmap ; le parent rechargera son propre mmap.
        spike_src.close()


def _compute_payload_for_streaming(config: AnalysisConfig) -> tuple[
    np.ndarray,
    list[str],
    int,
    int,
    float,
    float | None,
    np.ndarray,
    int,
    int,
    str,
]:
    """Payload léger pour pipeline streaming multi (sans moyennes globales)."""
    worker_cfg = replace(config, keep_intermediate_files=True)
    if not worker_cfg.rhs_file.exists():
        raise FileNotFoundError(f"Fichier introuvable: {worker_cfg.rhs_file}")
    data = load_rhs_file(worker_cfg.rhs_file)
    fs = get_sampling_rate(data)
    analog_in0 = get_analog_in0_signal(data)
    trigger_indices = detect_edges(analog_in0, threshold=worker_cfg.threshold, edge=worker_cfg.edge)
    if trigger_indices.size == 0:
        edge_fr = "descendant" if worker_cfg.edge == "falling" else "montant"
        raise RuntimeError(f"No {edge_fr} edge detected on ANALOG_IN 0.")
    amplifier_raw = np.asarray(data.get("amplifier_data"))
    if amplifier_raw.size == 0:
        raise RuntimeError("RHS file does not contain amplifier_data.")
    _, n_samples = amplifier_raw.shape
    valid_triggers, t_rel, pre_n, post_n = valid_triggers_and_timebase(
        n_samples=n_samples,
        trigger_indices=trigger_indices,
        fs=fs,
        pre_s=worker_cfg.pre_s,
        post_s=worker_cfg.post_s,
    )
    channel_names = get_channel_names(data, amplifier_raw.shape[0])
    n_valid = int(valid_triggers.shape[0])
    n_total = int(trigger_indices.size)
    end_rising_s = mean_time_to_next_rising_edge_s(analog_in0, trigger_indices, worker_cfg.threshold, fs)
    work_dir = resolve_work_dir(worker_cfg)
    amp_path = work_dir / "amplifier_raw.npy"
    persist_amplifier_float32(amplifier_raw, amp_path)
    return (
        t_rel,
        channel_names,
        n_valid,
        n_total,
        fs,
        end_rising_s,
        valid_triggers.copy(),
        int(pre_n),
        int(post_n),
        str(amp_path),
    )


def parse_args() -> argparse.Namespace:
    defaults = AnalysisConfig(rhs_file=Path("."))
    parser = argparse.ArgumentParser(
        description=(
            "Lit un fichier Intan RHS, detecte un front sur ANALOG_IN 0 (montant ou descendant), "
            "et calcule les moyennes par canal sur [-2s, +10s]."
        )
    )
    parser.add_argument("rhs_file", nargs="?", type=Path, help="Chemin vers le fichier .rhs")
    parser.add_argument("--gui", action="store_true", help="Lancer l'interface graphique Qt")
    parser.add_argument(
        "--edge",
        choices=("falling", "rising"),
        default=defaults.edge,
        help="Type de front sur ANALOG_IN 0: falling=descendant, rising=montant (defaut: falling)",
    )
    parser.add_argument("--threshold", type=float, default=defaults.threshold, help="Seuil de detection (defaut: 1.0)")
    parser.add_argument("--pre", type=float, default=defaults.pre_s, help="Temps avant trigger en secondes")
    parser.add_argument("--post", type=float, default=defaults.post_s, help="Temps apres trigger en secondes")
    parser.add_argument("--save-dir", type=Path, default=None, help="Dossier de sauvegarde du PDF")
    parser.add_argument(
        "--pdf-title",
        type=str,
        default=None,
        help="Nom/titre du PDF de sortie (sans extension ou avec .pdf)",
    )
    parser.add_argument(
        "--lowpass-hz",
        type=float,
        default=defaults.lowpass_cutoff_hz,
        help="Frequence de coupure (Hz) du passe-bas Butterworth sur les canaux amplificateur (defaut: pas de filtre)",
    )
    parser.add_argument(
        "--spike-threshold-uv",
        type=float,
        default=defaults.spike_threshold_uv,
        help=(
            "Seuil (µV) spikes amplificateur : >=0 = passage montant au-dessus ; "
            "<0 = passage descendant en dessous (pics negatifs, defaut: -40)"
        ),
    )
    parser.add_argument(
        "--firing-rate-window-s",
        type=float,
        default=defaults.firing_rate_window_s,
        help="Largeur (s) du lissage gaussien du PSTH / taux de decharge (defaut: 0.025)",
    )
    parser.add_argument(
        "--zoom-t0-s",
        type=float,
        default=defaults.zoom_t0_s,
        help="Debut de la fenetre de zoom (s, relatif au trigger).",
    )
    parser.add_argument(
        "--zoom-t1-s",
        type=float,
        default=defaults.zoom_t1_s,
        help="Fin de la fenetre de zoom (s, relatif au trigger).",
    )
    parser.add_argument(
        "--spike-bandpass-low-hz",
        type=float,
        default=defaults.spike_bandpass_low_hz,
        help=(
            "Frequence basse (Hz) du passe-bande Butterworth sur l'amplificateur pour "
            "raster / PSTH / ISI (avec --spike-bandpass-high-hz ; defaut: desactive = brut mmap)"
        ),
    )
    parser.add_argument(
        "--spike-bandpass-high-hz",
        type=float,
        default=defaults.spike_bandpass_high_hz,
        help=(
            "Frequence haute (Hz) du passe-bande Butterworth spikes "
            "(avec --spike-bandpass-low-hz ; defaut: desactive = brut mmap)"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Dossier fichiers intermediaires (amplificateur .npy mmap). Defaut: auto sous le PDF",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Conserver le dossier de travail (amplifier_raw.npy) apres le PDF",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Nombre de processus pour la comparaison A/B (defaut: 2)",
    )
    parser.add_argument(
        "--channel-workers",
        type=int,
        default=None,
        help="Nombre max de workers canal (defaut: auto, cap 16).",
    )
    parser.add_argument(
        "--lightweight-plot",
        action="store_true",
        help="Mode affichage léger PDF (downsample raster/ISI + dpi reduit).",
    )
    parser.add_argument(
        "--sampling-percent",
        type=int,
        default=100,
        help="Pourcentage de points conserves dans raster/ISI (1..100, defaut: 100).",
    )
    return parser.parse_args()


def run(config: AnalysisConfig) -> None:
    spike_src: AmplifierSpikeSource | None = None
    t0 = time.perf_counter()
    try:
        (
            mean_per_channel,
            t_rel,
            channel_names,
            n_valid,
            n_total,
            fs,
            end_rising_s,
            spike_src,
            mean_per_channel_raw,
        ) = compute_average_per_channel(config)
        check_analysis_cancelled()
        out_dir = config.save_dir if config.save_dir is not None else config.rhs_file.parent
        with tempfile.TemporaryDirectory(prefix="plot_erg_means_") as td:
            mm_dir = Path(td)
            mean_mm = _to_temp_mmap(mean_per_channel, mm_dir, "mean_filtered_or_raw")
            mean_raw_mm = (
                _to_temp_mmap(mean_per_channel_raw, mm_dir, "mean_raw")
                if config.lowpass_cutoff_hz is not None
                else None
            )
            del mean_per_channel
            del mean_per_channel_raw
            pdf_path = plot_channel_averages(
                t_rel=t_rel,
                mean_per_channel=mean_mm,
                channel_names=channel_names,
                output_dir=out_dir,
                rhs_file=config.rhs_file,
                pdf_title=config.pdf_title,
                lowpass_cutoff_hz=config.lowpass_cutoff_hz,
                trigger_end_rising_rel_s=end_rising_s,
                spike_source=spike_src,
                windows=None,
                fs=fs,
                spike_threshold_uv=config.spike_threshold_uv,
                firing_rate_window_s=config.firing_rate_window_s,
                zoom_t0_s=config.zoom_t0_s,
                zoom_t1_s=config.zoom_t1_s,
                mean_per_channel_raw=mean_raw_mm,
                spike_bandpass_low_hz=config.spike_bandpass_low_hz,
                spike_bandpass_high_hz=config.spike_bandpass_high_hz,
                lightweight_mode=config.lightweight_plot,
                sampling_percent=config.sampling_percent,
            )
        print(f"Frequence d'echantillonnage: {fs:.2f} Hz")
        print("--- Triggers (ANALOG_IN 0) ---")
        print(f"Nombre total de triggers detectes: {n_total}")
        print(f"Nombre de triggers utilises pour la moyenne: {n_valid}")
        if n_total > n_valid:
            print(f"  ({n_total - n_valid} trigger(s) exclus: fenetre [-pre,+post] hors du signal)")
        print(f"Nb canaux amplificateur: {len(channel_names)}")
        print(f"Fenetre temporelle: [-{config.pre_s:.3f}s, +{config.post_s:.3f}s]")
        print(f"Front ANALOG_IN 0: {config.edge}")
        if end_rising_s is not None:
            print(f"Delai moyen jusqu'au prochain front montant (fin de pulse typique): {end_rising_s*1e3:.3f} ms")
        else:
            print("Next rising edge at threshold: not computed (no rising edge after triggers).")
        if config.lowpass_cutoff_hz is not None:
            print(f"Passe-bas Butterworth: fc = {config.lowpass_cutoff_hz} Hz (ordre 4, filtfilt)")
        else:
            print("Butterworth low-pass: disabled")
        spike_rule = (
            "front descendant (pic négatif)"
            if config.spike_threshold_uv < 0
            else "front montant"
        )
        bp_txt = (
            f" | passe-bande spikes {config.spike_bandpass_low_hz:g}–{config.spike_bandpass_high_hz:g} Hz"
            if config.spike_bandpass_low_hz is not None
            else " | passe-bande spikes: desactive (brut)"
        )
        print(
            f"Spikes (PDF amplificateur): seuil {config.spike_threshold_uv} µV ({spike_rule}) | "
            f"lissage taux σ = {config.firing_rate_window_s} s{bp_txt}"
        )
        print(f"Fenetre zoom PDF: [{config.zoom_t0_s:.3f}s, {config.zoom_t1_s:.3f}s]")
        print(f"PDF genere: {pdf_path}")
        elapsed_s = time.perf_counter() - t0
        print(f"Temps total analyse + PDF: {elapsed_s:.2f} s")
        print(
            f"Dossier travail (mmap amplificateur): {resolve_work_dir(config)} — "
            + (
                "conserve (--keep-intermediate)."
                if config.keep_intermediate_files
                else "supprime apres succes (economie disque)."
            )
        )
    finally:
        if spike_src is not None:
            spike_src.close()


def run_comparison(config_a: AnalysisConfig, config_b: AnalysisConfig) -> Path:
    """Deux enregistrements via moteur streaming unifié."""
    pdf_path, stats = _run_streaming_comparison([config_a, config_b], "comparaison A/B")
    print(f"Frequence echantillonnage A: {stats['fs_values'][0]:.2f} Hz | B: {stats['fs_values'][1]:.2f} Hz")
    print("--- Recording A ---")
    print(f"  Triggers detectes: {stats['n_totals'][0]} | utilises: {stats['n_valids'][0]}")
    print("--- Recording B ---")
    print(f"  Triggers detectes: {stats['n_totals'][1]} | utilises: {stats['n_valids'][1]}")
    print(f"Canaux compares (superposition): {stats['n_ch']}")
    print(f"Workers multiprocessing (comparaison A/B): {stats['workers']}")
    print(f"Temps calcul A/B (multiprocessing): {stats['t_compute_s']:.2f} s")
    print(f"Temps rendu PDF A/B: {stats['t_render_s']:.2f} s")
    print(f"Fenetre temporelle: [-{config_a.pre_s:.3f}s, +{config_a.post_s:.3f}s]")
    print(f"Front ANALOG_IN 0: {config_a.edge}")
    if stats["end_markers"][0] is not None:
        print(f"Delai moyen fin trigger (↗) — A: {stats['end_markers'][0]*1e3:.3f} ms")
    if stats["end_markers"][1] is not None:
        print(f"Delai moyen fin trigger (↗) — B: {stats['end_markers'][1]*1e3:.3f} ms")
    if config_a.lowpass_cutoff_hz is not None:
        print(f"Passe-bas Butterworth: fc = {config_a.lowpass_cutoff_hz} Hz")
    else:
        print("Butterworth low-pass: disabled")
    print(f"PDF comparaison genere: {pdf_path}")
    print(f"Temps total comparaison + PDF: {stats['t_total_s']:.2f} s")
    return pdf_path


def run_multi_comparison(configs: list[AnalysisConfig]) -> None:
    """Compare N enregistrements sur les mêmes graphes (superposition multi-courbes)."""
    pdf_path, stats = _run_streaming_comparison(configs, "comparaison multi")
    print("--- Triggers per recording ---")
    for i, cfg in enumerate(configs):
        print(f"{cfg.rhs_file.name}: detectes={stats['n_totals'][i]} | utilises={stats['n_valids'][i]}")
    print(f"Canaux compares (superposition): {stats['n_ch']}")
    print(f"Workers multiprocessing (comparaison multi): {stats['workers']}")
    print(f"Temps calcul multi (multiprocessing): {stats['t_compute_s']:.2f} s")
    print(f"Temps rendu PDF multi: {stats['t_render_s']:.2f} s")
    print(f"Front ANALOG_IN 0: {configs[0].edge}")
    print(f"PDF comparaison genere: {pdf_path}")
    print(f"Temps total comparaison + PDF: {stats['t_total_s']:.2f} s")


def _autotune_config(cfg: AnalysisConfig, n_files: int) -> AnalysisConfig:
    """Politique auto orientée vitesse stable."""
    workers = max(1, min(int(cfg.comparison_workers), max(1, min(6, n_files))))
    channel_workers = cfg.channel_workers
    if channel_workers is None:
        channel_workers = 8 if n_files <= 2 else 4
    if cfg.pre_s + cfg.post_s > 20:
        workers = min(workers, 3)
        channel_workers = min(channel_workers, 4)
    sampling_percent = cfg.sampling_percent
    lightweight = cfg.lightweight_plot
    if n_files >= 4 and sampling_percent > 35:
        sampling_percent = 35
    if n_files >= 6 and sampling_percent > 20:
        sampling_percent = 20
    if n_files >= 4:
        lightweight = True
    return replace(
        cfg,
        comparison_workers=workers,
        channel_workers=channel_workers,
        sampling_percent=sampling_percent,
        lightweight_plot=lightweight,
    )


def _run_streaming_comparison(configs: list[AnalysisConfig], label: str) -> tuple[Path, dict[str, object]]:
    if len(configs) < 2:
        raise ValueError("Multi-file comparison requires at least 2 recordings.")
    t0 = time.perf_counter()
    tuned = [_autotune_config(cfg, len(configs)) for cfg in configs]
    workers = max(1, int(tuned[0].comparison_workers))
    labels = [cfg.rhs_file.stem for cfg in tuned]
    print(f"{label.capitalize()}: {len(tuned)} fichiers.")
    print("Files: " + " | ".join(cfg.rhs_file.name for cfg in tuned))
    if tuned[0].pre_s + tuned[0].post_s > 20:
        print("Guardrail mode: large window detected, parallelism limited for memory stability.")
    if tuned[0].sampling_percent != configs[0].sampling_percent:
        print(f"Auto-tuning sampling: {configs[0].sampling_percent}% -> {tuned[0].sampling_percent}%")

    payloads = []
    t_compute0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(workers, len(tuned))) as pool:
        futures = [pool.submit(_compute_payload_for_streaming, cfg) for cfg in tuned]
        for fut in futures:
            payloads.append(fut.result())
    t_compute_s = time.perf_counter() - t_compute0

    spike_sources: list[AmplifierSpikeSource] = []
    try:
        t_arrays: list[np.ndarray] = []
        names_per_rec: list[list[str]] = []
        fs_values: list[float] = []
        end_markers: list[float | None] = []
        n_valids: list[int] = []
        n_totals: list[int] = []
        pre_vals: list[int] = []
        post_vals: list[int] = []
        for payload, cfg in zip(payloads, tuned):
            (
                t_rel,
                channel_names,
                n_valid,
                n_total,
                fs,
                end_rising_s,
                valid_triggers,
                pre_n,
                post_n,
                amp_path,
            ) = payload
            amp_mm = np.load(Path(amp_path), mmap_mode="r")
            spike_sources.append(
                AmplifierSpikeSource(
                    amplifier=amp_mm,
                    valid_triggers=valid_triggers,
                    pre_n=int(pre_n),
                    post_n=int(post_n),
                    work_dir=Path(amp_path).parent,
                    keep_intermediate_files=cfg.keep_intermediate_files,
                    fs=float(fs),
                    bandpass_low_hz=cfg.spike_bandpass_low_hz,
                    bandpass_high_hz=cfg.spike_bandpass_high_hz,
                )
            )
            t_arrays.append(np.asarray(t_rel))
            names_per_rec.append(channel_names)
            fs_values.append(float(fs))
            end_markers.append(end_rising_s)
            n_valids.append(int(n_valid))
            n_totals.append(int(n_total))
            pre_vals.append(int(pre_n))
            post_vals.append(int(post_n))

        if max(fs_values) - min(fs_values) > 1e-3:
            print("Warning: different sampling rates detected.")
        fs_ref = fs_values[0]
        t_min = min(len(t) for t in t_arrays)
        n_ch = min(src.amplifier.shape[0] for src in spike_sources)
        t_ref = np.asarray(t_arrays[0][:t_min])
        channel_names = names_per_rec[0][:n_ch]
        pre_n_common = min(pre_vals)
        post_n_common = min(post_vals)
        out_dir = tuned[0].save_dir if tuned[0].save_dir is not None else tuned[0].rhs_file.parent
        imp_sessions = collect_impedance_sessions([cfg.rhs_file for cfg in tuned])
        if imp_sessions:
            n_skip = len(tuned) - len(imp_sessions)
            print(
                f"Impedance |Z| @ 1 kHz: {len(imp_sessions)} session(s) with companion CSV "
                f"(_YYMMDD_HHMMSS suffix, chronological order) — panel at bottom of each multi-comparison channel page."
                + (f" ({n_skip} RHS file(s) skipped: no CSV.)" if n_skip else "")
            )
        t_render0 = time.perf_counter()
        pdf_path = plot_channel_multi_comparison(
            t_rel=t_ref,
            means=[],
            channel_names=channel_names,
            output_dir=out_dir,
            labels=labels,
            pdf_title=tuned[0].pdf_title,
            lowpass_cutoff_hz=tuned[0].lowpass_cutoff_hz,
            trigger_end_rising_rel_s_list=end_markers,
            means_raw=None,
            spike_sources=spike_sources,
            fs=float(fs_ref),
            spike_threshold_uv=tuned[0].spike_threshold_uv,
            firing_rate_window_s=tuned[0].firing_rate_window_s,
            zoom_t0_s=tuned[0].zoom_t0_s,
            zoom_t1_s=tuned[0].zoom_t1_s,
            spike_bandpass_low_hz=tuned[0].spike_bandpass_low_hz,
            spike_bandpass_high_hz=tuned[0].spike_bandpass_high_hz,
            lightweight_mode=tuned[0].lightweight_plot,
            sampling_percent=tuned[0].sampling_percent,
            streaming_mode=True,
            pre_n_common=pre_n_common,
            post_n_common=post_n_common,
            impedance_sessions=imp_sessions if imp_sessions else None,
        )
        t_render_s = time.perf_counter() - t_render0
        stats: dict[str, object] = {
            "n_valids": n_valids,
            "n_totals": n_totals,
            "n_ch": n_ch,
            "workers": min(workers, len(tuned)),
            "t_compute_s": t_compute_s,
            "t_render_s": t_render_s,
            "t_total_s": time.perf_counter() - t0,
            "end_markers": end_markers,
            "fs_values": fs_values,
        }
        return pdf_path, stats
    finally:
        for src in spike_sources:
            src.close()


def main() -> None:
    args = parse_args()
    rhs_path = args.rhs_file
    if args.gui or rhs_path is None:
        exit_code = launch_qt_gui(
            run_callback=run,
            run_comparison_callback=run_comparison,
            run_multi_comparison_callback=run_multi_comparison,
            default_threshold=args.threshold,
            default_edge=args.edge,
            default_pre_s=args.pre,
            default_post_s=args.post,
            default_lowpass_hz=args.lowpass_hz,
            default_spike_threshold_uv=args.spike_threshold_uv,
            default_firing_rate_window_s=args.firing_rate_window_s,
            default_zoom_t0_s=args.zoom_t0_s,
            default_zoom_t1_s=args.zoom_t1_s,
            default_spike_bandpass_low_hz=args.spike_bandpass_low_hz,
            default_spike_bandpass_high_hz=args.spike_bandpass_high_hz,
            default_channel_workers=args.channel_workers,
            default_lightweight_plot=args.lightweight_plot,
            default_sampling_percent=args.sampling_percent,
        )
        if exit_code != 0:
            sys.exit(exit_code)
        return

    config = AnalysisConfig(
        rhs_file=rhs_path,
        threshold=args.threshold,
        edge=args.edge,
        pre_s=args.pre,
        post_s=args.post,
        lowpass_cutoff_hz=args.lowpass_hz,
        save_dir=args.save_dir,
        pdf_title=args.pdf_title,
        spike_threshold_uv=args.spike_threshold_uv,
        firing_rate_window_s=args.firing_rate_window_s,
        zoom_t0_s=args.zoom_t0_s,
        zoom_t1_s=args.zoom_t1_s,
        spike_bandpass_low_hz=args.spike_bandpass_low_hz,
        spike_bandpass_high_hz=args.spike_bandpass_high_hz,
        work_dir=args.work_dir,
        keep_intermediate_files=args.keep_intermediate,
        comparison_workers=args.workers,
        channel_workers=args.channel_workers,
        lightweight_plot=args.lightweight_plot,
        sampling_percent=args.sampling_percent,
    )
    if config.zoom_t1_s <= config.zoom_t0_s:
        print("Erreur: --zoom-t1-s doit être strictement supérieur à --zoom-t0-s.", file=sys.stderr)
        sys.exit(2)
    try:
        run(config)
    except Exception as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
