"""Command-line entry: single-file analysis, A/B and multi comparison, optional Qt GUI."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

from config import AnalysisConfig
from intan_rhx_dsp import normalize_spike_threshold
from core import (
    AmplifierSpikeSource,
    build_intan_dsp_settings,
    get_analog_in0_signal,
    get_channel_names,
    get_sampling_rate,
    load_rhs_file,
    persist_amp_and_filtered_stacks,
    resolve_recording_windows,
    resolve_work_dir,
    uses_analog_trigger,
)
from memmap_io import load_readonly_memmap
from gui import launch_qt_gui
from impedance_tracking import collect_impedance_sessions
from plotting import plot_channel_multi_comparison
from probe_layout import load_probe_layout_json


def _compute_payload_for_streaming(config: AnalysisConfig) -> tuple[
    np.ndarray,
    list[str],
    int,
    int,
    float,
    float | None,
    np.ndarray,
    int,
    int,
    str,
]:
    """Lightweight payload for multi-recording streaming (no global means)."""
    if not config.rhs_file.exists():
        raise FileNotFoundError(f"File not found: {config.rhs_file}")
    data = load_rhs_file(config.rhs_file)
    fs = get_sampling_rate(data)
    analog_in0 = get_analog_in0_signal(data) if uses_analog_trigger(config) else np.array([], dtype=np.float64)
    amplifier_raw = np.asarray(data.get("amplifier_data"))
    if amplifier_raw.size == 0:
        raise RuntimeError("RHS file does not contain amplifier_data.")
    _, n_samples = amplifier_raw.shape
    valid_triggers, t_rel, pre_n, post_n, n_valid, n_total, end_rising_s = resolve_recording_windows(
        config=config,
        n_samples=n_samples,
        fs=fs,
        analog_in0=analog_in0,
    )
    channel_names = get_channel_names(data, amplifier_raw.shape[0])
    work_dir = resolve_work_dir(config)
    intan_dsp = build_intan_dsp_settings(data, config)
    stack_shape = (int(amplifier_raw.shape[0]), int(amplifier_raw.shape[1]))
    amp_path, _ = persist_amp_and_filtered_stacks(
        work_dir,
        intan_dsp,
        stack_shape,
        amplifier_2d=amplifier_raw,
        channel_workers=config.channel_workers,
    )
    del amplifier_raw
    if isinstance(data, dict):
        data.pop("amplifier_data", None)
    gc.collect()
    return (
        t_rel,
        channel_names,
        n_valid,
        n_total,
        fs,
        end_rising_s,
        valid_triggers.copy(),
        int(pre_n),
        int(post_n),
        str(amp_path),
    )


def parse_args() -> argparse.Namespace:
    defaults = AnalysisConfig(rhs_file=Path("."))
    parser = argparse.ArgumentParser(
        description=(
            "Read an Intan RHS file, detect an edge on ANALOG_IN 0 (rising or falling), "
            "and compute per-channel averages on [-pre, +post] seconds."
        )
    )
    parser.add_argument("rhs_file", nargs="?", type=Path, help="Path to the .rhs file")
    parser.add_argument("--gui", action="store_true", help="Launch the Qt GUI")
    parser.add_argument(
        "--edge",
        choices=("falling", "rising", "none"),
        default=defaults.edge,
        help="ANALOG_IN 0 edge (falling/rising) or none for fixed equal sections",
    )
    parser.add_argument("--threshold", type=float, default=defaults.threshold, help="Detection threshold (default: 1.0)")
    parser.add_argument("--pre", type=float, default=defaults.pre_s, help="Time before trigger (seconds)")
    parser.add_argument("--post", type=float, default=defaults.post_s, help="Time after trigger (seconds)")
    parser.add_argument(
        "--section-count",
        type=int,
        default=defaults.section_count,
        help="No-trigger mode: number of equal sections per recording (default: 10)",
    )
    parser.add_argument(
        "--section-duration-s",
        type=float,
        default=None,
        help="No-trigger mode: section duration (s); sets section count from recording length",
    )
    parser.add_argument(
        "--section-spec",
        choices=("count", "duration"),
        default=defaults.section_spec,
        help="No-trigger mode: whether --section-count or --section-duration-s is authoritative",
    )
    parser.add_argument(
        "--section-trigger-start-s",
        type=float,
        default=defaults.section_trigger_start_s,
        help="No-trigger mode: imaginary trigger start within each segment (s, default: 1.0)",
    )
    parser.add_argument(
        "--section-trigger-end-s",
        type=float,
        default=defaults.section_trigger_end_s,
        help="No-trigger mode: imaginary trigger end within each segment (s, default: 4.0)",
    )
    parser.add_argument("--save-dir", type=Path, default=None, help="Folder for the output PDF")
    parser.add_argument(
        "--pdf-title",
        type=str,
        default=None,
        help="PDF output name/title (with or without .pdf)",
    )
    parser.add_argument(
        "--spike-threshold-uv",
        type=float,
        default=defaults.spike_threshold_uv,
        help=(
            "Spike threshold magnitude (µV). Use --spike-threshold-polarity for above vs below "
            "(legacy: negative value implies negative polarity)."
        ),
    )
    parser.add_argument(
        "--spike-threshold-polarity",
        choices=("negative", "positive"),
        default=defaults.spike_threshold_polarity,
        help=(
            "Spike detection polarity: negative = below threshold (default), "
            "positive = above threshold."
        ),
    )
    parser.add_argument(
        "--spike-threshold-mode",
        choices=("fixed", "rms_multiple"),
        default=defaults.spike_threshold_mode,
        help="Spike threshold mode: fixed value or per-channel RMS multiple.",
    )
    parser.add_argument(
        "--spike-threshold-rms-multiplier",
        type=float,
        default=defaults.spike_threshold_rms_multiplier,
        help="Multiplier applied to mean channel RMS when --spike-threshold-mode=rms_multiple.",
    )
    parser.add_argument(
        "--psth-bin-window-s",
        "--firing-rate-window-s",
        dest="psth_bin_window_s",
        type=float,
        default=defaults.psth_bin_window_s,
        help="PSTH time window (s) used for each PSTH point (default: from config)",
    )
    parser.add_argument(
        "--zoom-t0-s",
        type=float,
        default=defaults.zoom_t0_s,
        help="Zoom window start (s, relative to trigger).",
    )
    parser.add_argument(
        "--zoom-t1-s",
        type=float,
        default=defaults.zoom_t1_s,
        help="Zoom window end (s, relative to trigger).",
    )
    parser.add_argument(
        "--rms-window-s",
        "--rms-smoothing-window-s",
        type=float,
        default=defaults.rms_window_s,
        help="RMS computation window (s) for moving-RMS profile.",
    )
    parser.add_argument(
        "--intan-spike-filter",
        choices=("highpass", "lowpass"),
        default=defaults.intan_spike_filter_kind,
        help="Intan software filter for raster/PSTH/ISI (default: highpass).",
    )
    parser.add_argument(
        "--intan-filter-order",
        type=int,
        default=defaults.intan_filter_order,
        help="Intan software filter order 1–8 (default: 2).",
    )
    parser.add_argument(
        "--intan-filter-type",
        choices=("bessel", "butterworth"),
        default=defaults.intan_filter_type,
        help="Intan filter prototype: bessel or butterworth (default: bessel).",
    )
    parser.add_argument(
        "--intan-filter-cutoff-hz",
        type=float,
        default=defaults.intan_filter_cutoff_hz,
        help="Intan software filter cutoff in Hz (default: 250).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Intermediate files folder (amplifier .npy mmap). Default: auto next to PDF",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of processes for A/B comparison (default: 2)",
    )
    parser.add_argument(
        "--channel-workers",
        type=int,
        default=None,
        help="Max channel worker threads (default: auto, cap 16).",
    )
    parser.add_argument(
        "--sampling-percent",
        type=int,
        default=100,
        help="Fraction of points kept in raster/ISI plots (1..100, default: 100).",
    )
    parser.add_argument(
        "--probe-layout-json",
        type=Path,
        default=None,
        help="probeinterface JSON (MEA): electrode map inset when channel maps.",
    )
    return parser.parse_args()


def run(config: AnalysisConfig) -> None:
    if config.probe_layout_json is not None:
        probe_json_path = config.probe_layout_json
        if not probe_json_path.exists():
            raise FileNotFoundError(f"Probe JSON not found: {probe_json_path}")
        load_probe_layout_json(probe_json_path)

    pdf_path, stats = _run_streaming_comparison([config], "analysis")
    fs = float(stats["fs_values"][0])  # type: ignore[index]
    n_total = int(stats["n_totals"][0])  # type: ignore[index]
    n_valid = int(stats["n_valids"][0])  # type: ignore[index]
    end_marker = stats["end_markers"][0]  # type: ignore[index]

    print(f"Sample rate: {fs:.2f} Hz")
    if config.edge == "none":
        print("--- Sections (no ANALOG_IN trigger) ---")
        print(f"Section spec: {config.section_spec}")
        if config.section_spec == "duration" and config.section_duration_s is not None:
            print(f"Target section duration: {config.section_duration_s:g} s")
        else:
            print(f"Target section count: {config.section_count}")
        print(f"Sections used for average: {n_valid}")
        print(
            f"Imaginary trigger window per segment: "
            f"{config.section_trigger_start_s:g}–{config.section_trigger_end_s:g} s "
            f"(t=0 at segment start + {config.section_trigger_start_s:g} s)"
        )
    else:
        print("--- Triggers (ANALOG_IN 0) ---")
        print(f"Total triggers detected: {n_total}")
        print(f"Triggers used for average: {n_valid}")
        if n_total > n_valid:
            print(f"  ({n_total - n_valid} trigger(s) excluded: [-pre,+post] window outside signal)")
        print(f"Time window: [-{config.pre_s:.3f}s, +{config.post_s:.3f}s]")
        print(f"ANALOG_IN 0 edge: {config.edge}")
        if end_marker is not None:
            print(f"Mean delay to trigger end (rising): {float(end_marker)*1e3:.3f} ms")
        else:
            print("Next rising edge at threshold: not computed (no rising edge after triggers).")
    print(f"Channels compared (overlay): {stats['n_ch']}")
    if config.edge == "none":
        print("Segmentation: equal sections with imaginary trigger window")
    print(
        "Mean traces in PDF: raw amplifier + Intan software filter "
        f"({config.intan_filter_type} "
        f"{'HP' if config.intan_spike_filter_kind == 'highpass' else 'LP'} "
        f"{config.intan_filter_cutoff_hz:g} Hz, order {config.intan_filter_order})"
    )
    print(f"PDF written: {pdf_path}")
    print(f"Compute time (multiprocessing): {stats['t_compute_s']:.2f} s")
    print(f"PDF render time: {stats['t_render_s']:.2f} s")
    print(f"Total time (analysis + PDF): {stats['t_total_s']:.2f} s")


def run_comparison(config_a: AnalysisConfig, config_b: AnalysisConfig) -> Path:
    """Two recordings via the unified streaming engine."""
    pdf_path, stats = _run_streaming_comparison([config_a, config_b], "A/B comparison")
    print(f"Sample rate A: {stats['fs_values'][0]:.2f} Hz | B: {stats['fs_values'][1]:.2f} Hz")
    if config_a.edge == "none":
        print("--- Sections per recording (no ANALOG_IN trigger) ---")
        for i, cfg in enumerate((config_a, config_b)):
            label = "A" if i == 0 else "B"
            print(f"  Recording {label}: sections used={stats['n_valids'][i]}")
    else:
        print("--- Recording A ---")
        print(f"  Triggers detected: {stats['n_totals'][0]} | used: {stats['n_valids'][0]}")
        print("--- Recording B ---")
        print(f"  Triggers detected: {stats['n_totals'][1]} | used: {stats['n_valids'][1]}")
        print(f"Time window: [-{config_a.pre_s:.3f}s, +{config_a.post_s:.3f}s]")
        print(f"ANALOG_IN 0 edge: {config_a.edge}")
        if stats["end_markers"][0] is not None:
            print(f"Mean delay to trigger end (rising) — A: {stats['end_markers'][0]*1e3:.3f} ms")
        if stats["end_markers"][1] is not None:
            print(f"Mean delay to trigger end (rising) — B: {stats['end_markers'][1]*1e3:.3f} ms")
    print(f"Channels compared (overlay): {stats['n_ch']}")
    print(f"Multiprocessing workers (A/B comparison): {stats['workers']}")
    print(f"A/B compute time (multiprocessing): {stats['t_compute_s']:.2f} s")
    print(f"A/B PDF render time: {stats['t_render_s']:.2f} s")
    print(
        "Mean traces in PDF: raw amplifier + Intan software filter "
        f"({config_a.intan_filter_type} "
        f"{'HP' if config_a.intan_spike_filter_kind == 'highpass' else 'LP'} "
        f"{config_a.intan_filter_cutoff_hz:g} Hz, order {config_a.intan_filter_order})"
    )
    print(f"Comparison PDF written: {pdf_path}")
    print(f"Total time (comparison + PDF): {stats['t_total_s']:.2f} s")
    return pdf_path


def run_multi_comparison(configs: list[AnalysisConfig]) -> None:
    """Process N recordings on a unified multi-trace plotting pipeline."""
    pdf_path, stats = _run_streaming_comparison(configs, "multi comparison")
    if configs[0].edge == "none":
        print("--- Sections per recording (no ANALOG_IN trigger) ---")
        for i, cfg in enumerate(configs):
            print(f"{cfg.rhs_file.name}: sections used={stats['n_valids'][i]}")
    else:
        print("--- Triggers per recording ---")
        for i, cfg in enumerate(configs):
            print(f"{cfg.rhs_file.name}: detected={stats['n_totals'][i]} | used={stats['n_valids'][i]}")
    print(f"Channels compared (overlay): {stats['n_ch']}")
    print(f"Multiprocessing workers (multi comparison): {stats['workers']}")
    print(f"Multi compute time (multiprocessing): {stats['t_compute_s']:.2f} s")
    print(f"Multi PDF render time: {stats['t_render_s']:.2f} s")
    if configs[0].edge == "none":
        print("Segmentation: no trigger — equal contiguous sections")
    else:
        print(f"ANALOG_IN 0 edge: {configs[0].edge}")
    print(f"Comparison PDF written: {pdf_path}")
    print(f"Total time (comparison + PDF): {stats['t_total_s']:.2f} s")


def _autotune_config(cfg: AnalysisConfig, n_files: int) -> AnalysisConfig:
    """Auto-tuning policy biased toward stable throughput."""
    if n_files <= 1:
        return replace(cfg, comparison_workers=1)
    workers = max(1, min(int(cfg.comparison_workers), max(1, min(6, n_files))))
    channel_workers = cfg.channel_workers
    if channel_workers is None:
        channel_workers = 8 if n_files <= 3 else 4
    if uses_analog_trigger(cfg) and cfg.pre_s + cfg.post_s > 20:
        workers = min(workers, 3)
        channel_workers = min(channel_workers, 4)
    sampling_percent = cfg.sampling_percent
    if n_files >= 2 and sampling_percent > 50:
        sampling_percent = 50
    if n_files >= 4 and sampling_percent > 35:
        sampling_percent = 35
    if n_files >= 6 and sampling_percent > 20:
        sampling_percent = 20
    return replace(
        cfg,
        comparison_workers=workers,
        channel_workers=channel_workers,
        sampling_percent=sampling_percent,
    )


def _run_streaming_comparison(configs: list[AnalysisConfig], label: str) -> tuple[Path, dict[str, object]]:
    if len(configs) < 1:
        raise ValueError("At least 1 recording is required.")
    t0 = time.perf_counter()
    tuned = [_autotune_config(cfg, len(configs)) for cfg in configs]
    workers = max(1, int(tuned[0].comparison_workers))
    labels = [cfg.rhs_file.stem for cfg in tuned]
    print(f"{label.capitalize()}: {len(tuned)} file(s).")
    print("Files: " + " | ".join(cfg.rhs_file.name for cfg in tuned))
    if uses_analog_trigger(tuned[0]) and tuned[0].pre_s + tuned[0].post_s > 20:
        print("Guardrail mode: large window detected, parallelism limited for memory stability.")
    if tuned[0].sampling_percent != configs[0].sampling_percent:
        print(f"Auto-tuning sampling: {configs[0].sampling_percent}% -> {tuned[0].sampling_percent}%")
    if os.environ.get("PLOT_ERG_HIGH_QUALITY_PDF", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print(
            "PDF fast layout: ~38 in page height, DPI=72 "
            "(set PLOT_ERG_HIGH_QUALITY_PDF=1 for legacy tall export @ 120 DPI)."
        )
    payloads = []
    t_compute0 = time.perf_counter()
    if len(tuned) == 1:
        # In-process load so RHS reader progress appears in the GUI log (stdout redirect).
        payloads = [_compute_payload_for_streaming(tuned[0])]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tuned))) as pool:
            futures = [pool.submit(_compute_payload_for_streaming, cfg) for cfg in tuned]
            for fut in futures:
                payloads.append(fut.result())
    t_compute_s = time.perf_counter() - t_compute0

    spike_sources: list[AmplifierSpikeSource] = []
    try:
        t_arrays: list[np.ndarray] = []
        names_per_rec: list[list[str]] = []
        fs_values: list[float] = []
        end_markers: list[float | None] = []
        n_valids: list[int] = []
        n_totals: list[int] = []
        pre_vals: list[int] = []
        post_vals: list[int] = []
        for payload, cfg in zip(payloads, tuned):
            (
                t_rel,
                channel_names,
                n_valid,
                n_total,
                fs,
                end_rising_s,
                valid_triggers,
                pre_n,
                post_n,
                amp_path,
            ) = payload
            work = Path(amp_path).parent
            amp_mm = load_readonly_memmap(Path(amp_path))
            high_mm = load_readonly_memmap(work / "high_intan.npy")
            from intan_rhx_dsp import IntanDspSettings

            intan_dsp = IntanDspSettings.load_json(work / "intan_dsp.json")
            spike_sources.append(
                AmplifierSpikeSource(
                    amplifier=amp_mm,
                    highpass=high_mm,
                    valid_triggers=valid_triggers,
                    pre_n=int(pre_n),
                    post_n=int(post_n),
                    work_dir=work,
                    intan_dsp=intan_dsp,
                )
            )
            t_arrays.append(np.asarray(t_rel))
            names_per_rec.append(channel_names)
            fs_values.append(float(fs))
            end_markers.append(end_rising_s)
            n_valids.append(int(n_valid))
            n_totals.append(int(n_total))
            pre_vals.append(int(pre_n))
            post_vals.append(int(post_n))

        if max(fs_values) - min(fs_values) > 1e-3:
            print("Warning: different sampling rates detected.")
        fs_ref = fs_values[0]
        t_min = min(len(t) for t in t_arrays)
        n_ch = min(src.amplifier.shape[0] for src in spike_sources)
        t_ref = np.asarray(t_arrays[0][:t_min])
        channel_names = names_per_rec[0][:n_ch]
        pre_n_common = min(pre_vals)
        post_n_common = min(post_vals)
        out_dir = tuned[0].save_dir if tuned[0].save_dir is not None else tuned[0].rhs_file.parent
        imp_sessions = collect_impedance_sessions([cfg.rhs_file for cfg in tuned])
        if imp_sessions:
            n_skip = len(tuned) - len(imp_sessions)
            print(
                f"Impedance |Z| @ 1 kHz: {len(imp_sessions)} session(s) with companion CSV "
                f"(_YYMMDD_HHMMSS suffix) — panel per channel page + summary page (mean across channels)."
                + (f" ({n_skip} RHS file(s) skipped: no CSV.)" if n_skip else "")
            )
        t_render0 = time.perf_counter()
        pdf_path = plot_channel_multi_comparison(
            t_rel=t_ref,
            channel_names=channel_names,
            output_dir=out_dir,
            labels=labels,
            pdf_title=tuned[0].pdf_title,
            trigger_end_rising_rel_s_list=end_markers,
            spike_sources=spike_sources,
            fs=float(fs_ref),
            spike_threshold_uv=tuned[0].spike_threshold_uv,
            spike_threshold_polarity=tuned[0].spike_threshold_polarity,
            spike_threshold_mode=tuned[0].spike_threshold_mode,
            spike_threshold_rms_multiplier=tuned[0].spike_threshold_rms_multiplier,
            psth_bin_window_s=tuned[0].psth_bin_window_s,
            rms_window_s=tuned[0].rms_window_s,
            zoom_t0_s=tuned[0].zoom_t0_s,
            zoom_t1_s=tuned[0].zoom_t1_s,
            sampling_percent=tuned[0].sampling_percent,
            pre_n_common=pre_n_common,
            post_n_common=post_n_common,
            impedance_sessions=imp_sessions if imp_sessions else None,
            probe_layout_json=tuned[0].probe_layout_json,
            channel_workers=tuned[0].channel_workers,
        )
        t_render_s = time.perf_counter() - t_render0
        stats: dict[str, object] = {
            "n_valids": n_valids,
            "n_totals": n_totals,
            "n_ch": n_ch,
            "workers": min(workers, len(tuned)),
            "t_compute_s": t_compute_s,
            "t_render_s": t_render_s,
            "t_total_s": time.perf_counter() - t0,
            "end_markers": end_markers,
            "fs_values": fs_values,
        }
        return pdf_path, stats
    finally:
        for src in spike_sources:
            src.close()


def main() -> None:
    args = parse_args()
    rhs_path = args.rhs_file
    if args.gui or rhs_path is None:
        exit_code = launch_qt_gui(
            run_callback=run,
            run_comparison_callback=run_comparison,
            run_multi_comparison_callback=run_multi_comparison,
            default_threshold=args.threshold,
            default_edge=args.edge,
            default_pre_s=args.pre,
            default_post_s=args.post,
            default_spike_threshold_uv=normalize_spike_threshold(
                args.spike_threshold_uv, args.spike_threshold_polarity
            )[0],
            default_spike_threshold_polarity=normalize_spike_threshold(
                args.spike_threshold_uv, args.spike_threshold_polarity
            )[1],
            default_spike_threshold_mode=args.spike_threshold_mode,
            default_spike_threshold_rms_multiplier=args.spike_threshold_rms_multiplier,
            default_psth_bin_window_s=args.psth_bin_window_s,
            default_rms_window_s=args.rms_window_s,
            default_zoom_t0_s=args.zoom_t0_s,
            default_zoom_t1_s=args.zoom_t1_s,
            default_intan_spike_filter_kind=args.intan_spike_filter,
            default_intan_filter_order=args.intan_filter_order,
            default_intan_filter_type=args.intan_filter_type,
            default_intan_filter_cutoff_hz=args.intan_filter_cutoff_hz,
            default_channel_workers=args.channel_workers,
            default_sampling_percent=args.sampling_percent,
            default_probe_layout_json=args.probe_layout_json,
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
        section_count=args.section_count,
        section_duration_s=args.section_duration_s,
        section_spec=args.section_spec,
        section_trigger_start_s=args.section_trigger_start_s,
        section_trigger_end_s=args.section_trigger_end_s,
        save_dir=args.save_dir,
        pdf_title=args.pdf_title,
        spike_threshold_uv=normalize_spike_threshold(
            args.spike_threshold_uv, args.spike_threshold_polarity
        )[0],
        spike_threshold_polarity=normalize_spike_threshold(
            args.spike_threshold_uv, args.spike_threshold_polarity
        )[1],
        spike_threshold_mode=args.spike_threshold_mode,
        spike_threshold_rms_multiplier=args.spike_threshold_rms_multiplier,
        psth_bin_window_s=args.psth_bin_window_s,
        rms_window_s=args.rms_window_s,
        zoom_t0_s=args.zoom_t0_s,
        zoom_t1_s=args.zoom_t1_s,
        intan_spike_filter_kind=args.intan_spike_filter,
        intan_filter_order=args.intan_filter_order,
        intan_filter_type=args.intan_filter_type,
        intan_filter_cutoff_hz=args.intan_filter_cutoff_hz,
        work_dir=args.work_dir,
        comparison_workers=args.workers,
        channel_workers=args.channel_workers,
        sampling_percent=args.sampling_percent,
        probe_layout_json=args.probe_layout_json,
    )
    if config.zoom_t1_s <= config.zoom_t0_s:
        print("Error: --zoom-t1-s must be strictly greater than --zoom-t0-s.", file=sys.stderr)
        sys.exit(2)
    if config.rms_window_s <= 0:
        print("Error: --rms-window-s / --rms-smoothing-window-s must be > 0.", file=sys.stderr)
        sys.exit(2)
    if config.intan_filter_order < 1 or config.intan_filter_order > 8:
        print("Error: --intan-filter-order must be between 1 and 8.", file=sys.stderr)
        sys.exit(2)
    if config.intan_filter_cutoff_hz <= 0:
        print("Error: --intan-filter-cutoff-hz must be > 0.", file=sys.stderr)
        sys.exit(2)
    if config.spike_threshold_mode == "rms_multiple" and config.spike_threshold_rms_multiplier <= 0:
        print(
            "Error: --spike-threshold-rms-multiplier must be > 0 when --spike-threshold-mode=rms_multiple.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        run(config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
