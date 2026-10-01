"""Paramètres par défaut (non éditables dans la fenêtre principale).

Les valeurs viennent du lanceur / de la CLI. L’utilisateur les modifie ensuite
uniquement dans chaque fenêtre de vue ouverte.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from config import AnalysisConfig
from view_config import (
    AxisLimits,
    LegendSettings,
    PanelStyle,
    ViewerSettings,
)


def viewer_settings_from_defaults(defaults: Mapping[str, Any] | None = None) -> ViewerSettings:
    """Réglages d’affichage initiaux pour une nouvelle fenêtre de vue."""
    d = dict(defaults or {})
    return ViewerSettings(
        zoom_onset_t0_s=float(d.get("default_zoom_onset_t0_s", -0.1)),
        zoom_onset_t1_s=float(d.get("default_zoom_onset_t1_s", 0.2)),
        zoom_end_t0_s=float(d.get("default_zoom_end_t0_s", -0.1)),
        zoom_end_t1_s=float(d.get("default_zoom_end_t1_s", 0.2)),
        psth_bin_window_s=float(d.get("default_psth_bin_window_s", 0.025)),
        sampling_percent=int(d.get("default_sampling_percent", 100) or 100),
        spike_overlay_pre_ms=float(d.get("default_spike_overlay_pre_ms", 2.0)),
        spike_overlay_post_ms=float(d.get("default_spike_overlay_post_ms", 4.0)),
        stim_hp_ylim=AxisLimits(
            enabled=bool(d.get("default_first_trigger_hp_ylim_enabled", False)),
            minimum=float(d.get("default_first_trigger_hp_ylim_min_uv", -200.0)),
            maximum=float(d.get("default_first_trigger_hp_ylim_max_uv", 200.0)),
        ),
        rms_ylim=AxisLimits(enabled=True, minimum=0.0, maximum=20.0),
        legend=LegendSettings(),
        style=PanelStyle(),
        montage_channels=32,
        montage_page=0,
    )


def probe_path_from_defaults(defaults: Mapping[str, Any] | None = None) -> Path | None:
    raw = (defaults or {}).get("default_probe_layout_json")
    if not raw:
        return None
    path = Path(str(raw))
    return path if str(path).strip() else None


def build_config_from_defaults(
    defaults: Mapping[str, Any] | None,
    rhs_file: Path,
    *,
    base: AnalysisConfig | None = None,
) -> AnalysisConfig:
    """Configuration de traitement pour un enregistrement (valeurs par défaut fixes)."""
    d = dict(defaults or {})
    polarity = str(d.get("default_spike_threshold_polarity", "negative"))
    magnitude = abs(float(d.get("default_spike_threshold_uv", 70.0)))
    signed = -magnitude if polarity == "negative" else magnitude
    section_spec = str(d.get("default_section_spec", "count"))
    section_duration = (
        float(d["default_section_duration_s"])
        if section_spec == "duration" and d.get("default_section_duration_s") is not None
        else (
            float(d.get("default_section_duration_s") or 10.0)
            if section_spec == "duration"
            else None
        )
    )
    workers = d.get("default_channel_workers")
    probe = probe_path_from_defaults(d)
    display = viewer_settings_from_defaults(d)
    config = AnalysisConfig(
        rhs_file=Path(rhs_file),
        threshold=float(d.get("default_threshold", 1.0)),
        edge=str(d.get("default_edge", "falling")),  # type: ignore[arg-type]
        pre_s=float(d.get("default_pre_s", 2.0)),
        post_s=float(d.get("default_post_s", 10.0)),
        section_count=int(d.get("default_section_count", 10) or 10),
        section_duration_s=section_duration,
        section_spec=section_spec,  # type: ignore[arg-type]
        section_trigger_start_s=float(d.get("default_section_trigger_start_s", 1.0)),
        section_trigger_end_s=float(d.get("default_section_trigger_end_s", 4.0)),
        spike_threshold_uv=signed,
        spike_threshold_polarity=polarity,  # type: ignore[arg-type]
        spike_threshold_mode=str(d.get("default_spike_threshold_mode", "fixed")),  # type: ignore[arg-type]
        spike_threshold_rms_multiplier=float(
            d.get("default_spike_threshold_rms_multiplier", 4.0)
        ),
        psth_bin_window_s=float(display.psth_bin_window_s),
        spike_overlay_pre_ms=float(display.spike_overlay_pre_ms),
        spike_overlay_post_ms=float(display.spike_overlay_post_ms),
        zoom_onset_t0_s=float(display.zoom_onset_t0_s),
        zoom_onset_t1_s=float(display.zoom_onset_t1_s),
        zoom_end_t0_s=float(display.zoom_end_t0_s),
        zoom_end_t1_s=float(display.zoom_end_t1_s),
        first_trigger_hp_ylim_enabled=display.stim_hp_ylim.enabled,
        first_trigger_hp_ylim_min_uv=display.stim_hp_ylim.minimum,
        first_trigger_hp_ylim_max_uv=display.stim_hp_ylim.maximum,
        rms_window_s=float(d.get("default_rms_window_s", 1.0)),
        intan_filter_order=int(d.get("default_intan_filter_order", 2) or 2),
        intan_filter_type=str(d.get("default_intan_filter_type", "bessel")),  # type: ignore[arg-type]
        intan_filter_cutoff_hz=float(d.get("default_intan_filter_cutoff_hz", 250.0)),
        work_dir=None,
        channel_workers=int(workers) if workers else None,
        sampling_percent=int(display.sampling_percent),
        probe_layout_json=probe,
    )
    if base is not None:
        config = replace(
            config,
            save_dir=base.save_dir,
            pdf_title=base.pdf_title,
            comparison_workers=base.comparison_workers,
        )
    return config
