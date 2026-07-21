"""Fast GUI entry point — heavy analysis modules load only when a run starts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from config import AnalysisConfig


def _lazy_run(config: AnalysisConfig) -> None:
    from cli import run

    run(config)


def _lazy_run_comparison(config_a: AnalysisConfig, config_b: AnalysisConfig) -> None:
    from cli import run_comparison

    run_comparison(config_a, config_b)


def _lazy_run_multi_comparison(configs: list[AnalysisConfig]) -> None:
    from cli import run_multi_comparison

    run_multi_comparison(configs)


def _default_gui_kwargs() -> dict[str, Any]:
    """GUI defaults (slightly tuned for typical lab use)."""
    cfg = AnalysisConfig(rhs_file=Path("."))
    return {
        "default_threshold": cfg.threshold,
        "default_edge": cfg.edge,
        "default_pre_s": 2.0,
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
        "default_psth_bin_window_s": 0.025,
        "default_rms_window_s": cfg.rms_window_s,
        "default_zoom_onset_t0_s": cfg.zoom_onset_t0_s,
        "default_zoom_onset_t1_s": 0.2,
        "default_zoom_end_t0_s": cfg.zoom_end_t0_s,
        "default_zoom_end_t1_s": 0.2,
        "default_intan_spike_filter_kind": cfg.intan_spike_filter_kind,
        "default_intan_filter_order": cfg.intan_filter_order,
        "default_intan_filter_type": cfg.intan_filter_type,
        "default_intan_filter_cutoff_hz": cfg.intan_filter_cutoff_hz,
        "default_channel_workers": cfg.channel_workers,
        "default_sampling_percent": cfg.sampling_percent,
        "default_probe_layout_json": cfg.probe_layout_json,
        "default_first_trigger_hp_ylim_enabled": cfg.first_trigger_hp_ylim_enabled,
        "default_first_trigger_hp_ylim_min_uv": cfg.first_trigger_hp_ylim_min_uv,
        "default_first_trigger_hp_ylim_max_uv": cfg.first_trigger_hp_ylim_max_uv,
    }


def gui_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    from intan_rhx_dsp import normalize_spike_threshold

    thr_uv, thr_pol = normalize_spike_threshold(
        args.spike_threshold_uv, args.spike_threshold_polarity
    )
    return {
        "default_threshold": args.threshold,
        "default_edge": args.edge,
        "default_pre_s": args.pre,
        "default_post_s": args.post,
        "default_section_count": args.section_count,
        "default_section_duration_s": args.section_duration_s,
        "default_section_spec": args.section_spec,
        "default_section_trigger_start_s": args.section_trigger_start_s,
        "default_section_trigger_end_s": args.section_trigger_end_s,
        "default_spike_threshold_uv": thr_uv,
        "default_spike_threshold_polarity": thr_pol,
        "default_spike_threshold_mode": args.spike_threshold_mode,
        "default_spike_threshold_rms_multiplier": args.spike_threshold_rms_multiplier,
        "default_psth_bin_window_s": args.psth_bin_window_s,
        "default_rms_window_s": args.rms_window_s,
        "default_zoom_onset_t0_s": args.zoom_onset_t0_s,
        "default_zoom_onset_t1_s": args.zoom_onset_t1_s,
        "default_zoom_end_t0_s": args.zoom_end_t0_s,
        "default_zoom_end_t1_s": args.zoom_end_t1_s,
        "default_intan_spike_filter_kind": args.intan_spike_filter,
        "default_intan_filter_order": args.intan_filter_order,
        "default_intan_filter_type": args.intan_filter_type,
        "default_intan_filter_cutoff_hz": args.intan_filter_cutoff_hz,
        "default_channel_workers": args.channel_workers,
        "default_sampling_percent": args.sampling_percent,
        "default_probe_layout_json": args.probe_layout_json,
        "default_first_trigger_hp_ylim_enabled": args.first_trigger_hp_ylim,
        "default_first_trigger_hp_ylim_min_uv": args.first_trigger_hp_ylim_min_uv,
        "default_first_trigger_hp_ylim_max_uv": args.first_trigger_hp_ylim_max_uv,
    }


def launch_gui(**overrides: Any) -> int:
    from gui.main_window import launch_qt_gui

    kwargs = _default_gui_kwargs()
    kwargs.update(overrides)
    return launch_qt_gui(
        run_callback=_lazy_run,
        run_comparison_callback=_lazy_run_comparison,
        run_multi_comparison_callback=_lazy_run_multi_comparison,
        **kwargs,
    )


def launch_gui_from_args(args: argparse.Namespace) -> int:
    return launch_gui(**gui_kwargs_from_args(args))
