"""Paramètres par défaut unifiés (lanceur → GUI → AnalysisConfig)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from config import AnalysisConfig
from view_config import (
    EDGE_TO_TRIGGER_POLARITY,
    AxisLimits,
    LegendSettings,
    PanelStyle,
    ViewerSettings,
)

# Défauts d’affichage cohérents (ViewerSettings + PDF).
DEFAULT_ZOOM_T0_S = -0.1
DEFAULT_ZOOM_T1_S = 0.4
DEFAULT_PSTH_BIN_S = 0.050


def app_defaults_from_config(cfg: AnalysisConfig | None = None) -> dict[str, Any]:
    """Dictionnaire de défauts GUI dérivé d’un AnalysisConfig (ou des constantes)."""
    cfg = cfg or AnalysisConfig(rhs_file=Path("."))
    return {
        "default_threshold": cfg.threshold,
        "default_edge": cfg.edge,
        "default_pre_s": cfg.pre_s,
        "default_post_s": cfg.post_s,
        "default_section_count": cfg.section_count,
        "default_section_duration_s": cfg.section_duration_s,
        "default_section_spec": cfg.section_spec,
        "default_section_trigger_start_s": cfg.section_trigger_start_s,
        "default_section_trigger_end_s": cfg.section_trigger_end_s,
        "default_spike_threshold_uv": cfg.spike_threshold_uv,
        "default_spike_threshold_polarity": cfg.spike_threshold_polarity,
        "default_spike_threshold_mode": cfg.spike_threshold_mode,
        "default_spike_threshold_rms_multiplier": cfg.spike_threshold_rms_multiplier,
        "default_psth_bin_window_s": float(cfg.psth_bin_window_s or DEFAULT_PSTH_BIN_S),
        "default_spike_overlay_pre_ms": cfg.spike_overlay_pre_ms,
        "default_spike_overlay_post_ms": cfg.spike_overlay_post_ms,
        "default_rms_window_s": cfg.rms_window_s,
        "default_zoom_onset_t0_s": float(cfg.zoom_onset_t0_s if cfg.zoom_onset_t0_s is not None else DEFAULT_ZOOM_T0_S),
        "default_zoom_onset_t1_s": float(cfg.zoom_onset_t1_s if cfg.zoom_onset_t1_s is not None else DEFAULT_ZOOM_T1_S),
        "default_zoom_end_t0_s": float(cfg.zoom_end_t0_s if cfg.zoom_end_t0_s is not None else DEFAULT_ZOOM_T0_S),
        "default_zoom_end_t1_s": float(cfg.zoom_end_t1_s if cfg.zoom_end_t1_s is not None else DEFAULT_ZOOM_T1_S),
        "default_intan_hp_filter_order": cfg.intan_hp_filter_order,
        "default_intan_hp_filter_type": cfg.intan_hp_filter_type,
        "default_intan_hp_filter_cutoff_hz": cfg.intan_hp_filter_cutoff_hz,
        "default_intan_lp_filter_order": cfg.intan_lp_filter_order,
        "default_intan_lp_filter_type": cfg.intan_lp_filter_type,
        "default_intan_lp_filter_cutoff_hz": cfg.intan_lp_filter_cutoff_hz,
        "default_software_notch_hz": int(getattr(cfg, "software_notch_hz", 0) or 0),
        "default_channel_workers": cfg.channel_workers,
        "default_sampling_percent": cfg.sampling_percent,
        "default_probe_layout_json": cfg.probe_layout_json,
        "default_first_trigger_hp_ylim_enabled": cfg.first_trigger_hp_ylim_enabled,
        "default_first_trigger_hp_ylim_min_uv": cfg.first_trigger_hp_ylim_min_uv,
        "default_first_trigger_hp_ylim_max_uv": cfg.first_trigger_hp_ylim_max_uv,
    }


def viewer_settings_from_defaults(defaults: Mapping[str, Any] | None = None) -> ViewerSettings:
    """Réglages d’affichage initiaux."""
    d = dict(defaults or {})
    default_edge = str(d.get("default_edge", "falling"))
    polarity = EDGE_TO_TRIGGER_POLARITY.get(default_edge, "low")
    return ViewerSettings(
        zoom_onset_t0_s=float(d.get("default_zoom_onset_t0_s", DEFAULT_ZOOM_T0_S)),
        zoom_onset_t1_s=float(d.get("default_zoom_onset_t1_s", DEFAULT_ZOOM_T1_S)),
        zoom_end_t0_s=float(d.get("default_zoom_end_t0_s", DEFAULT_ZOOM_T0_S)),
        zoom_end_t1_s=float(d.get("default_zoom_end_t1_s", DEFAULT_ZOOM_T1_S)),
        psth_bin_window_s=float(d.get("default_psth_bin_window_s", DEFAULT_PSTH_BIN_S)),
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
        montage_channels=12,
        montage_page=0,
        time_sync=str(d.get("default_time_sync", "recording_start")),  # type: ignore[arg-type]
        trigger_polarity=str(d.get("default_trigger_polarity", polarity)),  # type: ignore[arg-type]
        trigger_threshold=float(d.get("default_threshold", 1.0)),
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
    """Configuration de traitement pour un enregistrement."""
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
        intan_hp_filter_order=int(d.get("default_intan_hp_filter_order", 2) or 2),
        intan_hp_filter_type=str(d.get("default_intan_hp_filter_type", "bessel")),  # type: ignore[arg-type]
        intan_hp_filter_cutoff_hz=float(d.get("default_intan_hp_filter_cutoff_hz", 250.0)),
        intan_lp_filter_order=int(d.get("default_intan_lp_filter_order", 2) or 2),
        intan_lp_filter_type=str(d.get("default_intan_lp_filter_type", "bessel")),  # type: ignore[arg-type]
        intan_lp_filter_cutoff_hz=float(d.get("default_intan_lp_filter_cutoff_hz", 250.0)),
        software_notch_hz=int(d.get("default_software_notch_hz", 0) or 0),  # type: ignore[arg-type]
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
