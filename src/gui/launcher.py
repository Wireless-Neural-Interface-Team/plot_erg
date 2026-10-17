"""Point d’entrée GUI — les modules d’analyse lourds ne chargent qu’au premier run."""

from __future__ import annotations

import argparse
from typing import Any

from gui.defaults import app_defaults_from_config


def _lazy_run_multi_comparison(configs: list) -> None:
    from cli import run_multi_comparison

    run_multi_comparison(configs)


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
        "default_spike_overlay_pre_ms": args.spike_overlay_pre_ms,
        "default_spike_overlay_post_ms": args.spike_overlay_post_ms,
        "default_rms_window_s": args.rms_window_s,
        "default_zoom_onset_t0_s": args.zoom_onset_t0_s,
        "default_zoom_onset_t1_s": args.zoom_onset_t1_s,
        "default_zoom_end_t0_s": args.zoom_end_t0_s,
        "default_zoom_end_t1_s": args.zoom_end_t1_s,
        "default_intan_hp_filter_order": int(
            args.intan_filter_order
            if getattr(args, "intan_filter_order", None) is not None
            else args.intan_hp_filter_order
        ),
        "default_intan_hp_filter_type": (
            args.intan_filter_type
            if getattr(args, "intan_filter_type", None) is not None
            else args.intan_hp_filter_type
        ),
        "default_intan_hp_filter_cutoff_hz": float(
            args.intan_filter_cutoff_hz
            if getattr(args, "intan_filter_cutoff_hz", None) is not None
            else args.intan_hp_filter_cutoff_hz
        ),
        "default_intan_lp_filter_order": int(
            args.intan_filter_order
            if getattr(args, "intan_filter_order", None) is not None
            else args.intan_lp_filter_order
        ),
        "default_intan_lp_filter_type": (
            args.intan_filter_type
            if getattr(args, "intan_filter_type", None) is not None
            else args.intan_lp_filter_type
        ),
        "default_intan_lp_filter_cutoff_hz": float(
            args.intan_filter_cutoff_hz
            if getattr(args, "intan_filter_cutoff_hz", None) is not None
            else args.intan_lp_filter_cutoff_hz
        ),
        "default_software_notch_hz": int(getattr(args, "software_notch_hz", 0) or 0),
        "default_channel_workers": args.channel_workers,
        "default_sampling_percent": args.sampling_percent,
        "default_probe_layout_json": args.probe_layout_json,
        "default_first_trigger_hp_ylim_enabled": args.first_trigger_hp_ylim,
        "default_first_trigger_hp_ylim_min_uv": args.first_trigger_hp_ylim_min_uv,
        "default_first_trigger_hp_ylim_max_uv": args.first_trigger_hp_ylim_max_uv,
    }


def launch_gui(**overrides: Any) -> int:
    from gui.main_window import launch_qt_gui

    kwargs = app_defaults_from_config()
    kwargs.update(overrides)
    return launch_qt_gui(
        run_multi_comparison_callback=_lazy_run_multi_comparison,
        **kwargs,
    )


def launch_gui_from_args(args: argparse.Namespace) -> int:
    return launch_gui(**gui_kwargs_from_args(args))
