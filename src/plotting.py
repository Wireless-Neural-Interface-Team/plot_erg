from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from core import check_analysis_cancelled

# Fenetre du panneau zoom (s), temps relatif au trigger (t=0)
ZOOM_T0 = -0.1
ZOOM_T1 = 0.2

def plot_channel_averages(
    t_rel: np.ndarray,
    mean_per_channel: np.ndarray,
    channel_names: Sequence[str],
    output_dir: Path,
    rhs_file: Path,
    lowpass_cutoff_hz: Optional[float] = None,
    trigger_end_rising_rel_s: Optional[float] = None,
) -> Path:
    """Un seul PDF multi-pages (une page par canal), sans fenêtre graphique.

    Les valeurs amplificateur Intan sont affichées en microvolts (µV), comme
    fournies par load_intan_rhs_format (amplifier_data).
    """
    n_channels = mean_per_channel.shape[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{rhs_file.stem}.pdf"

    zoom_t0, zoom_t1 = ZOOM_T0, ZOOM_T1
    filt_note = (
        f" — Butterworth passe-bas {lowpass_cutoff_hz:g} Hz"
        if lowpass_cutoff_hz is not None
        else ""
    )
    zoom_title = f"Zoom : {zoom_t0:.1f} à {zoom_t1:.1f} s (relatif au trigger){filt_note}"

    with PdfPages(pdf_path) as pdf:
        for ch in range(n_channels):
            check_analysis_cancelled()
            y = mean_per_channel[ch]
            fig, (ax_full, ax_zoom) = plt.subplots(
                2,
                1,
                figsize=(10, 7),
                height_ratios=[1.15, 1.0],
                sharex=False,
            )
            ax_full.plot(t_rel, y, linewidth=1.2, color="C0")
            ax_full.axvline(0.0, linestyle="--", linewidth=1.0, color="red", label="Trigger (début)")
            if trigger_end_rising_rel_s is not None:
                ax_full.axvline(
                    trigger_end_rising_rel_s,
                    linestyle=":",
                    linewidth=1.0,
                    color="darkorange",
                    label="Fin trigger (↗)",
                )
            ax_full.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zone zoom")
            ax_full.set_title(f"Moyenne trigger — {channel_names[ch]} (vue complète){filt_note}")
            ax_full.set_ylabel("Amplitude (µV)")
            ax_full.grid(True, alpha=0.3)
            ax_full.legend(loc="best", fontsize=8)

            zmask = (t_rel >= zoom_t0) & (t_rel <= zoom_t1)
            ax_zoom.plot(t_rel[zmask], y[zmask], linewidth=1.4, color="C0")
            ax_zoom.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
            if trigger_end_rising_rel_s is not None:
                ax_zoom.axvline(
                    trigger_end_rising_rel_s,
                    linestyle=":",
                    linewidth=1.0,
                    color="darkorange",
                )
            ax_zoom.set_xlim(zoom_t0, zoom_t1)
            ax_zoom.set_title(zoom_title)
            ax_zoom.set_xlabel("Temps relatif (s)")
            ax_zoom.set_ylabel("Amplitude (µV)")
            ax_zoom.grid(True, alpha=0.3)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path


def plot_channel_comparison(
    t_rel: np.ndarray,
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    channel_names: Sequence[str],
    output_dir: Path,
    label_a: str,
    label_b: str,
    lowpass_cutoff_hz: Optional[float] = None,
    trigger_end_rising_rel_s_a: Optional[float] = None,
    trigger_end_rising_rel_s_b: Optional[float] = None,
) -> Path:
    """PDF multi-pages, meme mise en page que l'analyse unitaire : une page par canal, vue complete + zoom."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_a = "".join(c if c.isalnum() or c in "._-" else "_" for c in label_a)[:80]
    safe_b = "".join(c if c.isalnum() or c in "._-" else "_" for c in label_b)[:80]
    pdf_path = output_dir / f"{safe_a}_vs_{safe_b}.pdf"

    zoom_t0, zoom_t1 = ZOOM_T0, ZOOM_T1
    filt_note = (
        f" — Butterworth passe-bas {lowpass_cutoff_hz:g} Hz"
        if lowpass_cutoff_hz is not None
        else ""
    )
    zoom_title = f"Zoom : {zoom_t0:.1f} à {zoom_t1:.1f} s (relatif au trigger){filt_note}"
    n_channels = mean_a.shape[0]
    zmask = (t_rel >= zoom_t0) & (t_rel <= zoom_t1)

    with PdfPages(pdf_path) as pdf:
        for ch in range(n_channels):
            check_analysis_cancelled()
            ya = mean_a[ch]
            yb = mean_b[ch]
            fig, (ax_full, ax_zoom) = plt.subplots(
                2,
                1,
                figsize=(10, 7),
                height_ratios=[1.15, 1.0],
                sharex=False,
            )
            ax_full.plot(t_rel, ya, linewidth=1.2, color="C0", label=label_a)
            ax_full.plot(t_rel, yb, linewidth=1.2, color="C1", label=label_b)
            ax_full.axvline(0.0, linestyle="--", linewidth=1.0, color="red", label="Trigger (début)")
            if trigger_end_rising_rel_s_a is not None:
                ax_full.axvline(
                    trigger_end_rising_rel_s_a,
                    linestyle=":",
                    linewidth=1.0,
                    color="darkorange",
                    label=f"Fin (↗) {label_a}",
                )
            if trigger_end_rising_rel_s_b is not None:
                ax_full.axvline(
                    trigger_end_rising_rel_s_b,
                    linestyle=":",
                    linewidth=1.0,
                    color="purple",
                    label=f"Fin (↗) {label_b}",
                )
            ax_full.axvspan(zoom_t0, zoom_t1, alpha=0.12, color="green", label="Zone zoom")
            ax_full.set_title(
                f"Comparaison — {channel_names[ch]} (vue complète){filt_note}"
            )
            ax_full.set_ylabel("Amplitude (µV)")
            ax_full.grid(True, alpha=0.3)
            ax_full.legend(loc="best", fontsize=7)

            ax_zoom.plot(t_rel[zmask], ya[zmask], linewidth=1.4, color="C0", label=label_a)
            ax_zoom.plot(t_rel[zmask], yb[zmask], linewidth=1.4, color="C1", label=label_b)
            ax_zoom.axvline(0.0, linestyle="--", linewidth=1.0, color="red")
            if trigger_end_rising_rel_s_a is not None:
                ax_zoom.axvline(
                    trigger_end_rising_rel_s_a,
                    linestyle=":",
                    linewidth=1.0,
                    color="darkorange",
                )
            if trigger_end_rising_rel_s_b is not None:
                ax_zoom.axvline(
                    trigger_end_rising_rel_s_b,
                    linestyle=":",
                    linewidth=1.0,
                    color="purple",
                )
            ax_zoom.set_xlim(zoom_t0, zoom_t1)
            ax_zoom.set_title(zoom_title)
            ax_zoom.set_xlabel("Temps relatif (s)")
            ax_zoom.set_ylabel("Amplitude (µV)")
            ax_zoom.grid(True, alpha=0.3)
            ax_zoom.legend(loc="best", fontsize=7)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path
