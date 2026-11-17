"""Visibilité des paramètres d’affichage par vue (aperçu / montage).

Carte issue de la sélection utilisateur (canvas Paramètres par vue).
Chaque paramètre a des valeurs **indépendantes** dans les profils preview /
montage du dock ; cette carte ne gère que la présence dans l’UI.
"""

from __future__ import annotations

from typing import Literal

ViewMode = Literal["preview", "montage"]

# True = visible dans cette vue. Absent d’une vue = masqué.
PARAM_VISIBILITY: dict[str, dict[ViewMode, bool]] = {
    # A. Titre
    "title_text": {"preview": True, "montage": True},
    "title_font": {"preview": True, "montage": True},
    "title_dx": {"preview": True, "montage": True},
    # B. Stim
    "stim_markers": {"preview": True, "montage": True},
    "stim_linewidth": {"preview": True, "montage": True},
    "stim_onset_color": {"preview": True, "montage": True},
    "stim_offset_color": {"preview": True, "montage": True},
    "stim_labels": {"preview": True, "montage": True},
    "stim_on_text": {"preview": True, "montage": True},
    "stim_on_font": {"preview": True, "montage": True},
    "stim_on_offset": {"preview": True, "montage": True},
    "stim_off_text": {"preview": True, "montage": True},
    "stim_off_font": {"preview": True, "montage": True},
    "stim_off_offset": {"preview": True, "montage": True},
    # C. Labels verticaux / titres canal
    "ylabel": {"preview": True, "montage": True},
    "label_font": {"preview": True, "montage": True},
    "montage_show_kind": {"preview": False, "montage": True},
    "montage_label_dx": {"preview": False, "montage": True},
    "montage_label_dy": {"preview": False, "montage": True},
    # D. Graph
    "graph_height": {"preview": True, "montage": False},
    "montage_row_height": {"preview": False, "montage": True},
    "graph_width": {"preview": True, "montage": True},
    "max_points": {"preview": True, "montage": True},
    "line_width": {"preview": True, "montage": True},
    "grid": {"preview": True, "montage": True},
    "grid_alpha": {"preview": True, "montage": True},
    "show_borders": {"preview": True, "montage": True},
    "tick_font": {"preview": True, "montage": True},
    "montage_shared_ylim": {"preview": False, "montage": True},
    # E / F. Échelles
    "scale_bars": {"preview": True, "montage": True},
    "scale_amp": {"preview": True, "montage": True},
    "scale_amp_len": {"preview": True, "montage": True},
    "scale_amp_offset": {"preview": True, "montage": True},
    "scale_linewidth": {"preview": True, "montage": True},
    "scale_time": {"preview": True, "montage": True},
    "scale_time_len": {"preview": True, "montage": True},
    "scale_time_offset": {"preview": True, "montage": True},
    # G. Axe X
    "xlabel": {"preview": True, "montage": True},
    "x_limits": {"preview": True, "montage": True},
    # H. Légende
    "legend_visible": {"preview": True, "montage": True},
    "legend_location": {"preview": True, "montage": True},
    "legend_font": {"preview": True, "montage": True},
    "legend_columns": {"preview": True, "montage": True},
    "legend_frame": {"preview": True, "montage": True},
    "legend_filters": {"preview": True, "montage": True},
    "legend_counts": {"preview": True, "montage": True},
    "series_suffix": {"preview": True, "montage": True},
    # I. Zones
    "zone_labels": {"preview": True, "montage": True},
    "zone_font": {"preview": True, "montage": True},
    "zone_offset": {"preview": True, "montage": True},
    # J. Plages Y
    "raw_ylim": {"preview": True, "montage": False},
    # K. Temps & contenu
    "time_sync": {"preview": True, "montage": True},
    "content_mode": {"preview": True, "montage": True},
    "stim_index": {"preview": True, "montage": True},
    "streams": {"preview": True, "montage": True},
    "extra_panels": {"preview": True, "montage": True},
    # L. Spikes
    "psth_bin": {"preview": True, "montage": True},
    "sampling": {"preview": True, "montage": True},
    "overlay_window": {"preview": True, "montage": True},
    "overlay_alpha": {"preview": True, "montage": True},
    "spike_threshold": {"preview": True, "montage": True},
    "spike_scale_amp": {"preview": True, "montage": True},
    "spike_scale_time": {"preview": True, "montage": True},
    # M. Pagination revue
    "montage_review_channels": {"preview": True, "montage": True},
}


def visible_in(param_id: str, mode: ViewMode) -> bool:
    """True si le paramètre doit apparaître dans ``mode``."""
    entry = PARAM_VISIBILITY.get(param_id)
    if entry is None:
        return True
    return bool(entry.get(mode, False))
