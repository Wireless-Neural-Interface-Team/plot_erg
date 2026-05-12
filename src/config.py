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
    lowpass_cutoff_hz: Optional[float] = 250.0
    save_dir: Path | None = None
    # Titre/nom du PDF de sortie (sans extension .pdf = ajouté automatiquement).
    pdf_title: str | None = None
    # Spikes (amplificateur, µV) : seuil >= 0 = passage vers le haut ; seuil < 0 = passage vers le bas
    spike_threshold_uv: float = -15.0
    # Largeur de lissage du taux de decharge (s), noyau gaussien sur le PSTH
    firing_rate_window_s: float = 0.010
    # Fenetre de zoom des panneaux PDF (s, temps relatif au trigger).
    zoom_t0_s: float = -0.1
    zoom_t1_s: float = 0.5
    # Passe-bande Butterworth sur l'amplificateur pour raster / PSTH / ISI (les deux None = brut)
    spike_bandpass_low_hz: Optional[float] = 250.0
    spike_bandpass_high_hz: Optional[float] = 7500.0
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
    # JSON probeinterface (carte MEA en encart PDF si le canal y est mappé).
    probe_layout_json: Path | None = None
