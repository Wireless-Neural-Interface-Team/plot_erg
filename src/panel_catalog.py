"""Catalogue unique des panneaux (clés, libellés, métadonnées).

Source de vérité partagée par l’affichage live, le PDF et le sélecteur GUI.
"""

from __future__ import annotations

from typing import Literal

PanelScope = Literal["channel", "global"]

# ---- Panneaux de section (PDF / vues temporelles) ----------------------------

SECTION_PANEL_FIELD_NAMES: tuple[str, ...] = (
    "mean_raw",
    "first_trigger_raw",
    "second_trigger_raw",
    "mean_hp",
    "first_trigger_hp",
    "second_trigger_hp",
    "mean_lp",
    "first_trigger_lp",
    "second_trigger_lp",
    "rms",
    "first_rms",
    "second_rms",
    "psth",
    "first_psth",
    "second_psth",
    "isi",
    "first_isi",
    "second_isi",
    "trial_rate",
    "raster",
    "spike_overlay",
)

SECTION_PANEL_LABELS: dict[str, str] = {
    "mean_raw": "Moyenne d’essais — brut",
    "mean_hp": "Moyenne d’essais — passe-haut",
    "mean_lp": "Moyenne d’essais — passe-bas",
    "first_trigger_raw": "1re stimulation (brut)",
    "first_trigger_hp": "1re stimulation (passe-haut)",
    "first_trigger_lp": "1re stimulation (passe-bas)",
    "second_trigger_raw": "2e stimulation (brut)",
    "second_trigger_hp": "2e stimulation (passe-haut)",
    "second_trigger_lp": "2e stimulation (passe-bas)",
    "rms": "RMS (moyenne d’essais)",
    "first_rms": "RMS (1re stimulation)",
    "second_rms": "RMS (2e stimulation)",
    "psth": "PSTH / FR moyenne",
    "first_psth": "PSTH / FR (1re stimulation)",
    "second_psth": "PSTH / FR (2e stimulation)",
    "isi": "ISI (toutes stimulations)",
    "first_isi": "ISI (1re stimulation)",
    "second_isi": "ISI (2e stimulation)",
    "trial_rate": "Taux par essai (non moyenné)",
    "raster": "Raster (toutes stimulations)",
    "spike_overlay": "Superposition de spikes",
}

SECTION_PANEL_GROUPS: dict[str, str] = {
    "mean_raw": "Tension — brut",
    "first_trigger_raw": "Tension — brut",
    "second_trigger_raw": "Tension — brut",
    "mean_hp": "Tension — passe-haut",
    "first_trigger_hp": "Tension — passe-haut",
    "second_trigger_hp": "Tension — passe-haut",
    "mean_lp": "Tension — passe-bas",
    "first_trigger_lp": "Tension — passe-bas",
    "second_trigger_lp": "Tension — passe-bas",
    "rms": "RMS",
    "first_rms": "RMS",
    "second_rms": "RMS",
    "psth": "Spikes — taux",
    "first_psth": "Spikes — taux",
    "second_psth": "Spikes — taux",
    "trial_rate": "Spikes — taux",
    "isi": "Spikes — intervalles",
    "first_isi": "Spikes — intervalles",
    "second_isi": "Spikes — intervalles",
    "raster": "Spikes — raster",
    "spike_overlay": "Spikes — formes d’onde",
}

# ---- Panneaux globaux (montages / résumés) -----------------------------------

GLOBAL_PANEL_FIELD_NAMES: tuple[str, ...] = (
    "montage_continuous_raw",
    "summary_rms",
    "summary_rms_table",
    "summary_impedance",
    "montage_mean_raw",
    "montage_mean_hp",
    "montage_mean_lp",
    "montage_second_raw",
    "montage_second_hp",
    "montage_second_lp",
    "montage_second_to_third_lp",
)

GLOBAL_PANEL_LABELS: dict[str, str] = {
    "montage_continuous_raw": "Montage — tous canaux (mêmes courbes que l’aperçu)",
    "summary_rms": "Résumé — mean RMS par enregistrement",
    "summary_rms_table": "Résumé — mean RMS par canal",
    "summary_impedance": "Résumé — impédance moyenne |Z| @ 1 kHz",
    "montage_mean_raw": "Montage — tous canaux, moyenne brut",
    "montage_mean_hp": "Montage — tous canaux, moyenne passe-haut",
    "montage_mean_lp": "Montage — tous canaux, moyenne passe-bas",
    "montage_second_raw": "Montage — tous canaux, 2e stim brut",
    "montage_second_hp": "Montage — tous canaux, 2e stim passe-haut",
    "montage_second_lp": "Montage — tous canaux, 2e stim passe-bas",
    "montage_second_to_third_lp": "Montage — tous canaux, 2e→3e stim passe-bas",
}

# ---- Contexte canal ---------------------------------------------------------

EXTRA_CHANNEL_PANEL_FIELD_NAMES: tuple[str, ...] = ("mea_layout", "impedance")

EXTRA_CHANNEL_PANEL_LABELS: dict[str, str] = {
    "mea_layout": "Carte MEA (canal sélectionné)",
    "impedance": "Impédance |Z| @ 1 kHz (canal sélectionné)",
}

# ---- Analyse progressive ----------------------------------------------------

ANALYSIS_PANEL_FIELD_NAMES: tuple[str, ...] = (
    "full_recording",
    "analysis_raw",
    "analysis_hp",
    "analysis_lp",
    "analysis_rms",
    "analysis_psth",
    "analysis_trial_rate",
    "analysis_isi",
    "analysis_raster_channel",
    "analysis_raster",
    "analysis_overlay",
)

ANALYSIS_PANEL_LABELS: dict[str, str] = {
    "full_recording": "Enregistrement complet (continu)",
    "analysis_raw": "Analyse — brut",
    "analysis_hp": "Analyse — passe-haut",
    "analysis_lp": "Analyse — passe-bas",
    "analysis_rms": "Analyse — RMS",
    "analysis_psth": "Analyse — PSTH / FR",
    "analysis_trial_rate": "Analyse — taux de décharge / essai",
    "analysis_isi": "Analyse — ISI",
    "analysis_raster_channel": "Analyse — raster par canal (compact)",
    "analysis_raster": "Analyse — raster (tous essais)",
    "analysis_overlay": "Analyse — spike scope",
}

# ---- Métadonnées dérivées ---------------------------------------------------

SPIKE_PANELS: frozenset[str] = frozenset(
    {
        "psth",
        "first_psth",
        "second_psth",
        "isi",
        "first_isi",
        "second_isi",
        "trial_rate",
        "raster",
        "spike_overlay",
        "analysis_psth",
        "analysis_trial_rate",
        "analysis_isi",
        "analysis_raster_channel",
        "analysis_raster",
        "analysis_overlay",
    }
)

RMS_PANELS: frozenset[str] = frozenset(
    {
        "rms",
        "first_rms",
        "second_rms",
        "analysis_rms",
        "summary_rms",
        "summary_rms_table",
    }
)

OVERLAY_PANELS: frozenset[str] = frozenset(
    {
        "spike_overlay",
        "analysis_overlay",
    }
)

MEANS_PANELS: frozenset[str] = frozenset(
    {
        "mean_raw",
        "mean_hp",
        "mean_lp",
        "first_trigger_raw",
        "first_trigger_hp",
        "first_trigger_lp",
        "second_trigger_raw",
        "second_trigger_hp",
        "second_trigger_lp",
        "analysis_raw",
        "analysis_hp",
        "analysis_lp",
        "montage_mean_raw",
        "montage_mean_hp",
        "montage_mean_lp",
        "montage_second_raw",
        "montage_second_hp",
        "montage_second_lp",
        "montage_second_to_third_lp",
    }
)

_ALL_LABELS: dict[str, str] = {
    **SECTION_PANEL_LABELS,
    **ANALYSIS_PANEL_LABELS,
    **EXTRA_CHANNEL_PANEL_LABELS,
    **GLOBAL_PANEL_LABELS,
}


def panel_label(field_name: str) -> str:
    return _ALL_LABELS.get(field_name, field_name)


def panel_needs_spikes(field_name: str) -> bool:
    return field_name in SPIKE_PANELS


def panel_product_needs(field_name: str) -> tuple[bool, bool, bool, bool]:
    """``(need_means, need_rms, need_spikes, need_overlay)`` for one panel key.

    Continuous / memmap panels (``full_recording``, ``montage_continuous_raw``)
    and context-only panels return all ``False``.
    """
    name = str(field_name or "")
    need_means = name in MEANS_PANELS
    need_rms = name in RMS_PANELS
    need_spikes = name in SPIKE_PANELS
    need_overlay = name in OVERLAY_PANELS
    return need_means, need_rms, need_spikes, need_overlay


def merge_product_needs(
    *needs: tuple[bool, bool, bool, bool],
) -> tuple[bool, bool, bool, bool]:
    means = rms = spikes = overlay = False
    for item in needs:
        means = means or bool(item[0])
        rms = rms or bool(item[1])
        spikes = spikes or bool(item[2])
        overlay = overlay or bool(item[3])
    return means, rms, spikes, overlay


def is_global_panel(field_name: str) -> bool:
    return field_name in GLOBAL_PANEL_FIELD_NAMES


def is_section_panel(field_name: str) -> bool:
    if field_name in SECTION_PANEL_FIELD_NAMES:
        return True
    return field_name in ANALYSIS_PANEL_FIELD_NAMES and field_name != "full_recording"


def is_section_independent(field_name: str) -> bool:
    return (
        field_name in GLOBAL_PANEL_FIELD_NAMES
        or field_name in EXTRA_CHANNEL_PANEL_FIELD_NAMES
        or field_name == "full_recording"
    )


def panel_group(field_name: str) -> str:
    if field_name == "full_recording":
        return "Enregistrement"
    if field_name in ANALYSIS_PANEL_FIELD_NAMES:
        return "Analyse (configurée)"
    if field_name in SECTION_PANEL_GROUPS:
        return SECTION_PANEL_GROUPS[field_name]
    if field_name in EXTRA_CHANNEL_PANEL_FIELD_NAMES:
        return "Contexte"
    if field_name.startswith("montage_"):
        return "Montages (tous canaux)"
    if field_name in GLOBAL_PANEL_FIELD_NAMES:
        return "Résumés"
    return "Autre"


def preferred_height_px(field_name: str) -> int:
    """Hauteur d’affichage indicative (px) — figsize initial ; la grille impose la hauteur fixe."""
    if field_name == "full_recording" or field_name.startswith("analysis_"):
        # Même budget vertical que continuous (WIDE / HIGH / LOW / RMS / spikes…).
        return 560
    if field_name == "analysis_raster_channel":
        return 220
    if field_name in {"raster"}:
        return 320
    if field_name in {"impedance", "mea_layout"}:
        return 260
    if field_name.startswith("montage_"):
        return 620
    if field_name == "summary_rms_table":
        # Table dense (tous les canaux) : plus de hauteur pour rester lisible.
        return 560
    if field_name in GLOBAL_PANEL_FIELD_NAMES:
        return 360
    return 300
