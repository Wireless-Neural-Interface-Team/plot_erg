from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from config import AnalysisConfig
from core import (
    AmplifierSpikeSource,
    check_analysis_cancelled,
    compute_average_per_channel,
    resolve_work_dir,
)
from gui import launch_qt_gui
from plotting import plot_channel_averages, plot_channel_comparison


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
    return parser.parse_args()


def run(config: AnalysisConfig) -> None:
    spike_src: AmplifierSpikeSource | None = None
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
        pdf_path = plot_channel_averages(
            t_rel=t_rel,
            mean_per_channel=mean_per_channel,
            channel_names=channel_names,
            output_dir=out_dir,
            rhs_file=config.rhs_file,
            lowpass_cutoff_hz=config.lowpass_cutoff_hz,
            trigger_end_rising_rel_s=end_rising_s,
            spike_source=spike_src,
            windows=None,
            fs=fs,
            spike_threshold_uv=config.spike_threshold_uv,
            firing_rate_window_s=config.firing_rate_window_s,
            mean_per_channel_raw=mean_per_channel_raw,
            spike_bandpass_low_hz=config.spike_bandpass_low_hz,
            spike_bandpass_high_hz=config.spike_bandpass_high_hz,
        )
        print(f"Frequence d'echantillonnage: {fs:.2f} Hz")
        print("--- Triggers (ANALOG_IN 0) ---")
        print(f"Nombre total de triggers detectes: {n_total}")
        print(f"Nombre de triggers utilises pour la moyenne: {n_valid}")
        if n_total > n_valid:
            print(f"  ({n_total - n_valid} trigger(s) exclus: fenetre [-pre,+post] hors du signal)")
        print(f"Nb canaux amplificateur: {mean_per_channel.shape[0]}")
        print(f"Fenetre temporelle: [-{config.pre_s:.3f}s, +{config.post_s:.3f}s]")
        print(f"Front ANALOG_IN 0: {config.edge}")
        if end_rising_s is not None:
            print(f"Delai moyen jusqu'au prochain front montant (fin de pulse typique): {end_rising_s*1e3:.3f} ms")
        else:
            print("Prochain front montant au seuil: non calcule (aucun front montant apres les triggers).")
        if config.lowpass_cutoff_hz is not None:
            print(f"Passe-bas Butterworth: fc = {config.lowpass_cutoff_hz} Hz (ordre 4, filtfilt)")
        else:
            print("Passe-bas Butterworth: desactive")
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
        print(f"PDF genere: {pdf_path}")
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


def run_comparison(config_a: AnalysisConfig, config_b: AnalysisConfig) -> None:
    """Deux enregistrements, memes parametres (sauf fichiers), PDF superpose."""
    spike_a: AmplifierSpikeSource | None = None
    spike_b: AmplifierSpikeSource | None = None
    try:
        ma, ta, names_a, va, tta, fsa, end_a, spike_a, ma_raw = compute_average_per_channel(config_a)
        check_analysis_cancelled()
        mb, tb, names_b, vb, ttb, fsb, end_b, spike_b, mb_raw = compute_average_per_channel(config_b)
        check_analysis_cancelled()

        if abs(fsa - fsb) > 1e-3:
            print(f"Attention: frequences d'echantillonnage differentes ({fsa} vs {fsb} Hz).")

        t_min = min(len(ta), len(tb))
        n_ch = min(ma.shape[0], mb.shape[0])
        ta = np.asarray(ta[:t_min])
        ma = np.asarray(ma[:n_ch, :t_min])
        mb = np.asarray(mb[:n_ch, :t_min])
        ma_raw = np.asarray(ma_raw[:n_ch, :t_min])
        mb_raw = np.asarray(mb_raw[:n_ch, :t_min])
        names = [f"{names_a[i]} / {names_b[i]}" if names_a[i] != names_b[i] else names_a[i] for i in range(n_ch)]

        if ma.shape[1] != mb.shape[1]:
            raise RuntimeError("Longueurs de fenetres incompatibles apres alignement.")

        check_analysis_cancelled()

        out_dir = config_a.save_dir if config_a.save_dir is not None else config_a.rhs_file.parent
        if config_b.save_dir is not None and config_a.save_dir != config_b.save_dir:
            print("Note: dossier PDF = celui de l'enregistrement 1 (ou --save-dir commun).")

        la = config_a.rhs_file.stem
        lb = config_b.rhs_file.stem
        pdf_path = plot_channel_comparison(
            t_rel=ta,
            mean_a=ma,
            mean_b=mb,
            channel_names=names,
            output_dir=out_dir,
            label_a=la,
            label_b=lb,
            lowpass_cutoff_hz=config_a.lowpass_cutoff_hz,
            trigger_end_rising_rel_s_a=end_a,
            trigger_end_rising_rel_s_b=end_b,
            mean_a_raw=ma_raw,
            mean_b_raw=mb_raw,
            spike_source_a=spike_a,
            spike_source_b=spike_b,
            fs=float(fsa),
            spike_threshold_uv=config_a.spike_threshold_uv,
            firing_rate_window_s=config_a.firing_rate_window_s,
            spike_bandpass_low_hz=config_a.spike_bandpass_low_hz,
            spike_bandpass_high_hz=config_a.spike_bandpass_high_hz,
        )
        print(f"Frequence echantillonnage A: {fsa:.2f} Hz | B: {fsb:.2f} Hz")
        print("--- Enregistrement A ---")
        print(f"  Triggers detectes: {tta} | utilises: {va}")
        print("--- Enregistrement B ---")
        print(f"  Triggers detectes: {ttb} | utilises: {vb}")
        print(f"Canaux compares (superposition): {n_ch}")
        print(f"Fenetre temporelle: [-{config_a.pre_s:.3f}s, +{config_a.post_s:.3f}s]")
        print(f"Front ANALOG_IN 0: {config_a.edge}")
        if end_a is not None:
            print(f"Delai moyen fin trigger (↗) — A: {end_a*1e3:.3f} ms")
        if end_b is not None:
            print(f"Delai moyen fin trigger (↗) — B: {end_b*1e3:.3f} ms")
        if config_a.lowpass_cutoff_hz is not None:
            print(f"Passe-bas Butterworth: fc = {config_a.lowpass_cutoff_hz} Hz")
        else:
            print("Passe-bas Butterworth: desactive")
        print(f"PDF comparaison genere: {pdf_path}")
    finally:
        if spike_a is not None:
            spike_a.close()
        if spike_b is not None:
            spike_b.close()


def main() -> None:
    args = parse_args()
    rhs_path = args.rhs_file
    if args.gui or rhs_path is None:
        exit_code = launch_qt_gui(
            run_callback=run,
            run_comparison_callback=run_comparison,
            default_threshold=args.threshold,
            default_edge=args.edge,
            default_pre_s=args.pre,
            default_post_s=args.post,
            default_lowpass_hz=args.lowpass_hz,
            default_spike_threshold_uv=args.spike_threshold_uv,
            default_firing_rate_window_s=args.firing_rate_window_s,
            default_spike_bandpass_low_hz=args.spike_bandpass_low_hz,
            default_spike_bandpass_high_hz=args.spike_bandpass_high_hz,
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
        spike_threshold_uv=args.spike_threshold_uv,
        firing_rate_window_s=args.firing_rate_window_s,
        spike_bandpass_low_hz=args.spike_bandpass_low_hz,
        spike_bandpass_high_hz=args.spike_bandpass_high_hz,
        work_dir=args.work_dir,
        keep_intermediate_files=args.keep_intermediate,
    )
    try:
        run(config)
    except Exception as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
