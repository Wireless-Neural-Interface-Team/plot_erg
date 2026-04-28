from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

EdgeKind = Literal["falling", "rising"]


@dataclass(frozen=True)
class AnalysisConfig:
    rhs_file: Path
    threshold: float = 1.0
    edge: EdgeKind = "falling"  # falling=descendant, rising=montant au seuil sur ANALOG_IN 0
    pre_s: float = 2.0
    post_s: float = 10.0
    # Passe-bas Butterworth sur amplifier_data (None = desactive)
    lowpass_cutoff_hz: Optional[float] = None
    save_dir: Path | None = None
    # Titre/nom du PDF de sortie (sans extension .pdf = ajouté automatiquement).
    pdf_title: str | None = None
    # Spikes (amplificateur, µV) : seuil >= 0 = passage vers le haut ; seuil < 0 = passage vers le bas
    spike_threshold_uv: float = -40.0
    # Largeur de lissage du taux de decharge (s), noyau gaussien sur le PSTH
    firing_rate_window_s: float = 0.025
    # Passe-bande Butterworth sur l'amplificateur pour raster / PSTH / ISI (les deux None = brut)
    spike_bandpass_low_hz: Optional[float] = None
    spike_bandpass_high_hz: Optional[float] = None
    # None = auto : (save_dir ou dossier du .rhs) / ".plot_erg" / <stem>
    work_dir: Path | None = None
    # Ne pas supprimer work_dir à la fin (amplificateur .npy, etc.)
    keep_intermediate_files: bool = False
    # Nb de workers process pour comparaison A/B (>=1).
    comparison_workers: int = 32
    # Nb max de workers canaux (None = auto, cap 16).
    channel_workers: int | None = None
    # Mode affichage léger PDF (downsample raster/ISI + DPI réduit).
    lightweight_plot: bool = False
    # Pourcentage de points conservés dans les graphes spikes (1..100).
    sampling_percent: int = 100
