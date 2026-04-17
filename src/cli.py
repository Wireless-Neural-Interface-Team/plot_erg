from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from config import AnalysisConfig
from core import check_analysis_cancelled, compute_average_per_channel
from gui import launch_qt_gui
from plotting import plot_channel_averages, plot_channel_comparison


def parse_args() -> argparse.Namespace:
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
        default="falling",
        help="Type de front sur ANALOG_IN 0: falling=descendant, rising=montant (defaut: falling)",
    )
    parser.add_argument("--threshold", type=float, default=1.0, help="Seuil de detection (defaut: 1.0)")
    parser.add_argument("--pre", type=float, default=2.0, help="Temps avant trigger en secondes")
    parser.add_argument("--post", type=float, default=10.0, help="Temps apres trigger en secondes")
    parser.add_argument("--save-dir", type=Path, default=None, help="Dossier de sauvegarde du PDF")
    parser.add_argument(
        "--lowpass-hz",
        type=float,
        default=None,
        help="Frequence de coupure (Hz) du passe-bas Butterworth sur les canaux amplificateur (defaut: pas de filtre)",
    )
    return parser.parse_args()


def run(config: AnalysisConfig) -> None:
    mean_per_channel, t_rel, channel_names, n_valid, n_total, fs, end_rising_s = (
        compute_average_per_channel(config)
    )
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
    print(f"PDF genere: {pdf_path}")


def run_comparison(config_a: AnalysisConfig, config_b: AnalysisConfig) -> None:
    """Deux enregistrements, memes parametres (sauf fichiers), PDF superpose."""
    ma, ta, names_a, va, tta, fsa, end_a = compute_average_per_channel(config_a)
    check_analysis_cancelled()
    mb, tb, names_b, vb, ttb, fsb, end_b = compute_average_per_channel(config_b)
    check_analysis_cancelled()

    if abs(fsa - fsb) > 1e-3:
        print(f"Attention: frequences d'echantillonnage differentes ({fsa} vs {fsb} Hz).")

    t_min = min(len(ta), len(tb))
    n_ch = min(ma.shape[0], mb.shape[0])
    ta = np.asarray(ta[:t_min])
    ma = np.asarray(ma[:n_ch, :t_min])
    mb = np.asarray(mb[:n_ch, :t_min])
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


def main() -> None:
    args = parse_args()
    rhs_path = args.rhs_file
    if args.gui or rhs_path is None:
        exit_code = launch_qt_gui(
            run_callback=run,
            run_comparison_callback=run_comparison,
            default_threshold=args.threshold,
            default_edge=args.edge,
            default_lowpass_hz=args.lowpass_hz,
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
    )
    try:
        run(config)
    except Exception as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
