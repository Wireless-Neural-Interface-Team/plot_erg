"""Command-line entry: single-file analysis, A/B and multi comparison, optional Qt GUI."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from config import AnalysisConfig
from core import (
    AmplifierSpikeSource,
    check_analysis_cancelled,
    compute_average_per_channel,
    detect_edges,
    get_analog_in0_signal,
    get_channel_names,
    get_sampling_rate,
    load_rhs_file,
    mean_time_to_next_rising_edge_s,
    persist_amplifier_float32,
    resolve_work_dir,
    valid_triggers_and_timebase,
)
from gui import launch_qt_gui
from impedance_tracking import collect_impedance_sessions
from plotting import plot_channel_averages, plot_channel_comparison, plot_channel_multi_comparison
from probe_layout import load_probe_layout_json


def _to_temp_mmap(arr: np.ndarray, folder: Path, name: str) -> np.ndarray:
    """Persist array to temporary .npy and reopen as read-only memmap."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npy"
    np.save(path, np.ascontiguousarray(arr, dtype=np.float32))
    return np.load(path, mmap_mode="r")


def _compute_payload_for_comparison(config: AnalysisConfig) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    int,
    int,
    float,
    float | None,
    np.ndarray,
    np.ndarray,
    int,
    int,
    str,
]:
    """Compute a serializable analysis payload for a separate worker process."""
    worker_cfg = replace(config, keep_intermediate_files=True)
    (
        mean_per_channel,
        t_rel,
        channel_names,
        n_valid,
        n_total,
        fs,
        end_rising_s,
        spike_src,
        mean_per_channel_raw,
    ) = compute_average_per_channel(worker_cfg)
    try:
        if spike_src.work_dir is None:
            raise RuntimeError("Spike work dir not found for parallel comparison.")
        amp_path = spike_src.work_dir / "amplifier_raw.npy"
        return (
            mean_per_channel,
            t_rel,
            channel_names,
            n_valid,
            n_total,
            fs,
            end_rising_s,
            mean_per_channel_raw,
            spike_src.valid_triggers.copy(),
            int(spike_src.pre_n),
            int(spike_src.post_n),
            str(amp_path),
        )
    finally:
        # Worker releases its mmap; the parent reloads its own mmap.
        spike_src.close()


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
    worker_cfg = replace(config, keep_intermediate_files=True)
    if not worker_cfg.rhs_file.exists():
        raise FileNotFoundError(f"File not found: {worker_cfg.rhs_file}")
    data = load_rhs_file(worker_cfg.rhs_file)
    fs = get_sampling_rate(data)
    analog_in0 = get_analog_in0_signal(data)
    trigger_indices = detect_edges(analog_in0, threshold=worker_cfg.threshold, edge=worker_cfg.edge)
    if trigger_indices.size == 0:
        edge_fr = "falling" if worker_cfg.edge == "falling" else "rising"
        raise RuntimeError(f"No {edge_fr} edge detected on ANALOG_IN 0.")
    amplifier_raw = np.asarray(data.get("amplifier_data"))
    if amplifier_raw.size == 0:
        raise RuntimeError("RHS file does not contain amplifier_data.")
    _, n_samples = amplifier_raw.shape
    valid_triggers, t_rel, pre_n, post_n = valid_triggers_and_timebase(
        n_samples=n_samples,
        trigger_indices=trigger_indices,
        fs=fs,
        pre_s=worker_cfg.pre_s,
        post_s=worker_cfg.post_s,
    )
    channel_names = get_channel_names(data, amplifier_raw.shape[0])
    n_valid = int(valid_triggers.shape[0])
    n_total = int(trigger_indices.size)
    end_rising_s = mean_time_to_next_rising_edge_s(analog_in0, trigger_indices, worker_cfg.threshold, fs)
    work_dir = resolve_work_dir(worker_cfg)
    amp_path = work_dir / "amplifier_raw.npy"
    persist_amplifier_float32(amplifier_raw, amp_path)
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
        choices=("falling", "rising"),
        default=defaults.edge,
        help="ANALOG_IN 0 edge type: falling vs rising (default: falling)",
    )
    parser.add_argument("--threshold", type=float, default=defaults.threshold, help="Detection threshold (default: 1.0)")
    parser.add_argument("--pre", type=float, default=defaults.pre_s, help="Time before trigger (seconds)")
    parser.add_argument("--post", type=float, default=defaults.post_s, help="Time after trigger (seconds)")
    parser.add_argument("--save-dir", type=Path, default=None, help="Folder for the output PDF")
    parser.add_argument(
        "--pdf-title",
        type=str,
        default=None,
        help="PDF output name/title (with or without .pdf)",
    )
    parser.add_argument(
        "--lowpass-hz",
        type=float,
        default=defaults.lowpass_cutoff_hz,
        help="Butterworth low-pass corner (Hz) on amplifier channels (default: no filter)",
    )
    parser.add_argument(
        "--spike-threshold-uv",
        type=float,
        default=defaults.spike_threshold_uv,
        help=(
            "Spike threshold (µV) on amplifier: >=0 = upward crossing; "
            "<0 = downward crossing (negative peaks, default -40)"
        ),
    )
    parser.add_argument(
        "--firing-rate-window-s",
        type=float,
        default=defaults.firing_rate_window_s,
        help="Gaussian smoothing width (s) for PSTH / firing rate (default 0.025)",
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
        type=float,
        default=defaults.rms_window_s,
        help="RMS window duration (s) after trigger onset for RMS-evolution plot.",
    )
    parser.add_argument(
        "--spike-bandpass-low-hz",
        type=float,
        default=defaults.spike_bandpass_low_hz,
        help=(
            "Spike band-pass low corner (Hz) on amplifier for raster / PSTH / ISI "
            "(use with --spike-bandpass-high-hz; default: disabled = raw mmap)"
        ),
    )
    parser.add_argument(
        "--spike-bandpass-high-hz",
        type=float,
        default=defaults.spike_bandpass_high_hz,
        help=(
            "Spike band-pass high corner (Hz) "
            "(use with --spike-bandpass-low-hz; default: disabled = raw mmap)"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Intermediate files folder (amplifier .npy mmap). Default: auto next to PDF",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep work folder (amplifier_raw.npy) after the PDF",
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
        "--lightweight-plot",
        action="store_true",
        help="Lightweight PDF mode (raster/ISI downsample + reduced dpi).",
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
    spike_source: AmplifierSpikeSource | None = None
    run_start_time_s = time.perf_counter()
    try:
        if config.probe_layout_json is not None:
            probe_json_path = config.probe_layout_json
            if not probe_json_path.exists():
                raise FileNotFoundError(f"Probe JSON not found: {probe_json_path}")
            load_probe_layout_json(probe_json_path)
        (
            mean_per_channel,
            t_rel,
            channel_names,
            n_valid,
            n_total,
            fs,
            end_rising_s,
            spike_source,
            mean_per_channel_raw,
        ) = compute_average_per_channel(config)
        check_analysis_cancelled()
        output_dir = config.save_dir if config.save_dir is not None else config.rhs_file.parent
        with tempfile.TemporaryDirectory(prefix="plot_erg_means_") as temporary_dir:
            mmap_dir = Path(temporary_dir)
            mean_per_channel_mmap = _to_temp_mmap(mean_per_channel, mmap_dir, "mean_filtered_or_raw")
            mean_per_channel_raw_mmap = (
                _to_temp_mmap(mean_per_channel_raw, mmap_dir, "mean_raw")
                if config.lowpass_cutoff_hz is not None
                else None
            )
            del mean_per_channel
            del mean_per_channel_raw
            pdf_path = plot_channel_averages(
                t_rel=t_rel,
                mean_per_channel=mean_per_channel_mmap,
                channel_names=channel_names,
                output_dir=output_dir,
                rhs_file=config.rhs_file,
                pdf_title=config.pdf_title,
                lowpass_cutoff_hz=config.lowpass_cutoff_hz,
                trigger_end_rising_rel_s=end_rising_s,
                spike_source=spike_source,
                windows=None,
                fs=fs,
                spike_threshold_uv=config.spike_threshold_uv,
                firing_rate_window_s=config.firing_rate_window_s,
                rms_window_s=config.rms_window_s,
                zoom_t0_s=config.zoom_t0_s,
                zoom_t1_s=config.zoom_t1_s,
                mean_per_channel_raw=mean_per_channel_raw_mmap,
                spike_bandpass_low_hz=config.spike_bandpass_low_hz,
                spike_bandpass_high_hz=config.spike_bandpass_high_hz,
                lightweight_mode=config.lightweight_plot,
                sampling_percent=config.sampling_percent,
                probe_layout_json=config.probe_layout_json,
            )
        print(f"Sample rate: {fs:.2f} Hz")
        print("--- Triggers (ANALOG_IN 0) ---")
        print(f"Total triggers detected: {n_total}")
        print(f"Triggers used for average: {n_valid}")
        if n_total > n_valid:
            print(f"  ({n_total - n_valid} trigger(s) excluded: [-pre,+post] window outside signal)")
        print(f"Amplifier channels: {len(channel_names)}")
        print(f"Time window: [-{config.pre_s:.3f}s, +{config.post_s:.3f}s]")
        print(f"ANALOG_IN 0 edge: {config.edge}")
        if end_rising_s is not None:
            print(f"Mean delay to next rising edge (typical pulse end): {end_rising_s*1e3:.3f} ms")
        else:
            print("Next rising edge at threshold: not computed (no rising edge after triggers).")
        if config.lowpass_cutoff_hz is not None:
            print(f"Butterworth low-pass: fc = {config.lowpass_cutoff_hz} Hz (order 4, filtfilt)")
        else:
            print("Butterworth low-pass: disabled")
        spike_rule = (
            "falling edge (negative peak)"
            if config.spike_threshold_uv < 0
            else "rising edge"
        )
        bp_txt = (
            f" | spike band-pass {config.spike_bandpass_low_hz:g}–{config.spike_bandpass_high_hz:g} Hz"
            if config.spike_bandpass_low_hz is not None
            else " | spike band-pass: disabled (raw)"
        )
        print(
            f"Spikes (PDF amplifier): threshold {config.spike_threshold_uv} µV ({spike_rule}) | "
            f"firing-rate smooth σ = {config.firing_rate_window_s} s{bp_txt}"
        )
        print(f"PDF zoom window: [{config.zoom_t0_s:.3f}s, {config.zoom_t1_s:.3f}s]")
        print(f"RMS evolution window: [0.000s, {config.rms_window_s:.3f}s] after trigger onset")
        print(f"PDF written: {pdf_path}")
        elapsed_s = time.perf_counter() - run_start_time_s
        print(f"Total time (analysis + PDF): {elapsed_s:.2f} s")
        print(
            f"Work dir (amplifier mmap): {resolve_work_dir(config)} — "
            + (
                "kept (--keep-intermediate)."
                if config.keep_intermediate_files
                else "removed after success (save disk)."
            )
        )
    finally:
        if spike_source is not None:
            spike_source.close()


def run_comparison(config_a: AnalysisConfig, config_b: AnalysisConfig) -> Path:
    """Two recordings via the unified streaming engine."""
    pdf_path, stats = _run_streaming_comparison([config_a, config_b], "A/B comparison")
    print(f"Sample rate A: {stats['fs_values'][0]:.2f} Hz | B: {stats['fs_values'][1]:.2f} Hz")
    print("--- Recording A ---")
    print(f"  Triggers detected: {stats['n_totals'][0]} | used: {stats['n_valids'][0]}")
    print("--- Recording B ---")
    print(f"  Triggers detected: {stats['n_totals'][1]} | used: {stats['n_valids'][1]}")
    print(f"Channels compared (overlay): {stats['n_ch']}")
    print(f"Multiprocessing workers (A/B comparison): {stats['workers']}")
    print(f"A/B compute time (multiprocessing): {stats['t_compute_s']:.2f} s")
    print(f"A/B PDF render time: {stats['t_render_s']:.2f} s")
    print(f"Time window: [-{config_a.pre_s:.3f}s, +{config_a.post_s:.3f}s]")
    print(f"ANALOG_IN 0 edge: {config_a.edge}")
    if stats["end_markers"][0] is not None:
        print(f"Mean delay to trigger end (rising) — A: {stats['end_markers'][0]*1e3:.3f} ms")
    if stats["end_markers"][1] is not None:
        print(f"Mean delay to trigger end (rising) — B: {stats['end_markers'][1]*1e3:.3f} ms")
    if config_a.lowpass_cutoff_hz is not None:
        print(f"Butterworth low-pass: fc = {config_a.lowpass_cutoff_hz} Hz")
    else:
        print("Butterworth low-pass: disabled")
    print(f"Comparison PDF written: {pdf_path}")
    print(f"Total time (comparison + PDF): {stats['t_total_s']:.2f} s")
    return pdf_path


def run_multi_comparison(configs: list[AnalysisConfig]) -> None:
    """Compare N recordings on the same plots (multi-trace overlay)."""
    pdf_path, stats = _run_streaming_comparison(configs, "multi comparison")
    print("--- Triggers per recording ---")
    for i, cfg in enumerate(configs):
        print(f"{cfg.rhs_file.name}: detected={stats['n_totals'][i]} | used={stats['n_valids'][i]}")
    print(f"Channels compared (overlay): {stats['n_ch']}")
    print(f"Multiprocessing workers (multi comparison): {stats['workers']}")
    print(f"Multi compute time (multiprocessing): {stats['t_compute_s']:.2f} s")
    print(f"Multi PDF render time: {stats['t_render_s']:.2f} s")
    print(f"ANALOG_IN 0 edge: {configs[0].edge}")
    print(f"Comparison PDF written: {pdf_path}")
    print(f"Total time (comparison + PDF): {stats['t_total_s']:.2f} s")


def _autotune_config(cfg: AnalysisConfig, n_files: int) -> AnalysisConfig:
    """Auto-tuning policy biased toward stable throughput."""
    workers = max(1, min(int(cfg.comparison_workers), max(1, min(6, n_files))))
    channel_workers = cfg.channel_workers
    if channel_workers is None:
        channel_workers = 8 if n_files <= 2 else 4
    if cfg.pre_s + cfg.post_s > 20:
        workers = min(workers, 3)
        channel_workers = min(channel_workers, 4)
    sampling_percent = cfg.sampling_percent
    lightweight = cfg.lightweight_plot
    if n_files >= 4 and sampling_percent > 35:
        sampling_percent = 35
    if n_files >= 6 and sampling_percent > 20:
        sampling_percent = 20
    if n_files >= 4:
        lightweight = True
    return replace(
        cfg,
        comparison_workers=workers,
        channel_workers=channel_workers,
        sampling_percent=sampling_percent,
        lightweight_plot=lightweight,
    )


def _run_streaming_comparison(configs: list[AnalysisConfig], label: str) -> tuple[Path, dict[str, object]]:
    if len(configs) < 2:
        raise ValueError("Multi-file comparison requires at least 2 recordings.")
    t0 = time.perf_counter()
    tuned = [_autotune_config(cfg, len(configs)) for cfg in configs]
    workers = max(1, int(tuned[0].comparison_workers))
    labels = [cfg.rhs_file.stem for cfg in tuned]
    print(f"{label.capitalize()}: {len(tuned)} file(s).")
    print("Files: " + " | ".join(cfg.rhs_file.name for cfg in tuned))
    if tuned[0].pre_s + tuned[0].post_s > 20:
        print("Guardrail mode: large window detected, parallelism limited for memory stability.")
    if tuned[0].sampling_percent != configs[0].sampling_percent:
        print(f"Auto-tuning sampling: {configs[0].sampling_percent}% -> {tuned[0].sampling_percent}%")

    payloads = []
    t_compute0 = time.perf_counter()
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
            amp_mm = np.load(Path(amp_path), mmap_mode="r")
            spike_sources.append(
                AmplifierSpikeSource(
                    amplifier=amp_mm,
                    valid_triggers=valid_triggers,
                    pre_n=int(pre_n),
                    post_n=int(post_n),
                    work_dir=Path(amp_path).parent,
                    keep_intermediate_files=cfg.keep_intermediate_files,
                    fs=float(fs),
                    bandpass_low_hz=cfg.spike_bandpass_low_hz,
                    bandpass_high_hz=cfg.spike_bandpass_high_hz,
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
            means=[],
            channel_names=channel_names,
            output_dir=out_dir,
            labels=labels,
            pdf_title=tuned[0].pdf_title,
            lowpass_cutoff_hz=tuned[0].lowpass_cutoff_hz,
            trigger_end_rising_rel_s_list=end_markers,
            means_raw=None,
            spike_sources=spike_sources,
            fs=float(fs_ref),
            spike_threshold_uv=tuned[0].spike_threshold_uv,
            firing_rate_window_s=tuned[0].firing_rate_window_s,
            rms_window_s=tuned[0].rms_window_s,
            zoom_t0_s=tuned[0].zoom_t0_s,
            zoom_t1_s=tuned[0].zoom_t1_s,
            spike_bandpass_low_hz=tuned[0].spike_bandpass_low_hz,
            spike_bandpass_high_hz=tuned[0].spike_bandpass_high_hz,
            lightweight_mode=tuned[0].lightweight_plot,
            sampling_percent=tuned[0].sampling_percent,
            streaming_mode=True,
            pre_n_common=pre_n_common,
            post_n_common=post_n_common,
            impedance_sessions=imp_sessions if imp_sessions else None,
            probe_layout_json=tuned[0].probe_layout_json,
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
            default_lowpass_hz=args.lowpass_hz,
            default_spike_threshold_uv=args.spike_threshold_uv,
            default_firing_rate_window_s=args.firing_rate_window_s,
            default_rms_window_s=args.rms_window_s,
            default_zoom_t0_s=args.zoom_t0_s,
            default_zoom_t1_s=args.zoom_t1_s,
            default_spike_bandpass_low_hz=args.spike_bandpass_low_hz,
            default_spike_bandpass_high_hz=args.spike_bandpass_high_hz,
            default_channel_workers=args.channel_workers,
            default_lightweight_plot=args.lightweight_plot,
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
        lowpass_cutoff_hz=args.lowpass_hz,
        save_dir=args.save_dir,
        pdf_title=args.pdf_title,
        spike_threshold_uv=args.spike_threshold_uv,
        firing_rate_window_s=args.firing_rate_window_s,
        rms_window_s=args.rms_window_s,
        zoom_t0_s=args.zoom_t0_s,
        zoom_t1_s=args.zoom_t1_s,
        spike_bandpass_low_hz=args.spike_bandpass_low_hz,
        spike_bandpass_high_hz=args.spike_bandpass_high_hz,
        work_dir=args.work_dir,
        keep_intermediate_files=args.keep_intermediate,
        comparison_workers=args.workers,
        channel_workers=args.channel_workers,
        lightweight_plot=args.lightweight_plot,
        sampling_percent=args.sampling_percent,
        probe_layout_json=args.probe_layout_json,
    )
    if config.zoom_t1_s <= config.zoom_t0_s:
        print("Error: --zoom-t1-s must be strictly greater than --zoom-t0-s.", file=sys.stderr)
        sys.exit(2)
    if config.rms_window_s <= 0:
        print("Error: --rms-window-s must be > 0.", file=sys.stderr)
        sys.exit(2)
    try:
        run(config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
