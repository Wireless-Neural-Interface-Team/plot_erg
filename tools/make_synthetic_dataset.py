"""Build a small synthetic processed recording, for tests without a .rhs file.

Used by ``tools/smoke_render.py`` to exercise every panel renderer and the
processed-dataset round trip.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intan_rhx_dsp import IntanDspSettings  # noqa: E402
from processed_dataset import (  # noqa: E402
    DerivedArrays,
    OverlaySnippets,
    ProcessedRecording,
    RecordingMeta,
    SegmentationInfo,
    SpikeDetectionInfo,
    SpikeTrains,
)


def build_synthetic(
    *,
    n_channels: int = 8,
    n_trials: int = 4,
    fs: float = 2000.0,
    pre_s: float = 0.5,
    post_s: float = 2.0,
    label: str = "synthetic",
    seed: int = 0,
) -> ProcessedRecording:
    rng = np.random.default_rng(seed)
    pre_n = int(round(pre_s * fs))
    post_n = int(round(post_s * fs))
    window = pre_n + post_n
    t_rel = (np.arange(window, dtype=np.float64) - pre_n) / fs

    names = tuple(f"A-{index:03d}" for index in range(n_channels))
    response = np.exp(-np.clip(t_rel, 0.0, None) * 3.0) * (t_rel >= 0.0)

    means: dict[str, np.ndarray] = {}
    for stream, gain, noise in (("raw", 120.0, 8.0), ("hp", 25.0, 4.0), ("lp", 110.0, 2.0)):
        base = np.outer(np.linspace(0.6, 1.4, n_channels), response * gain)
        means[stream] = (base + rng.normal(0.0, noise, base.shape)).astype(np.float32)

    trigger_windows: dict[tuple[int, str], np.ndarray] = {}
    for trigger in (0, 1):
        for stream in ("raw", "hp", "lp"):
            jitter = rng.normal(0.0, 6.0, means[stream].shape)
            trigger_windows[(trigger, stream)] = (
                means[stream] * (1.0 + 0.1 * trigger) + jitter
            ).astype(np.float32)

    rms_time = t_rel.copy()
    rms_profiles = {
        kind: np.abs(
            np.outer(np.linspace(3.0, 9.0, n_channels), 1.0 + 0.5 * response)
            + rng.normal(0.0, 0.3, (n_channels, window))
        ).astype(np.float32)
        for kind in ("mean", "first", "second")
    }
    channel_rms = np.asarray(
        [float(np.nanmean(rms_profiles["mean"][ch])) for ch in range(n_channels)],
        dtype=np.float32,
    )
    thresholds = (-4.0 * channel_rms).astype(np.float32)

    per_channel: list[list[np.ndarray]] = []
    for ch in range(n_channels):
        trials: list[np.ndarray] = []
        for _trial in range(n_trials):
            count = int(rng.integers(5, 40))
            times = np.sort(rng.uniform(float(t_rel[0]), float(t_rel[-1]), count))
            trials.append(times.astype(np.float32))
        per_channel.append(trials)
    spikes = SpikeTrains.from_lists(per_channel, n_trials)

    overlay_pre_ms, overlay_post_ms = 2.0, 4.0
    snippet = int(round((overlay_pre_ms + overlay_post_ms) * fs / 1000.0)) or 12
    t_ms = np.linspace(-overlay_pre_ms, overlay_post_ms, snippet)
    shape = -np.exp(-((t_ms / 0.8) ** 2))
    counts: list[int] = []
    totals: list[int] = []
    waves_chunks: list[np.ndarray] = []
    t_rel_chunks: list[np.ndarray] = []
    exact_means = np.zeros((n_channels, snippet), dtype=np.float32)
    for ch in range(n_channels):
        all_times = np.concatenate(per_channel[ch])
        kept = all_times[: min(all_times.size, 250)]
        waves = (
            shape[None, :] * abs(float(thresholds[ch])) * 1.4
            + rng.normal(0.0, 4.0, (kept.size, snippet))
        ).astype(np.float32)
        counts.append(int(kept.size))
        totals.append(int(all_times.size))
        waves_chunks.append(waves)
        t_rel_chunks.append(kept.astype(np.float32))
        exact_means[ch] = waves.mean(axis=0) if kept.size else 0.0
    overlay = OverlaySnippets(
        t_ms,
        np.asarray(counts, dtype=np.int32),
        np.asarray(totals, dtype=np.int32),
        np.concatenate(waves_chunks) if waves_chunks else np.empty((0, snippet), np.float32),
        np.concatenate(t_rel_chunks) if t_rel_chunks else np.empty(0, np.float32),
        exact_means,
    )

    dsp = IntanDspSettings(
        fs=fs,
        rhs_version_major=3,
        notch_filter_frequency_hz=0.0,
        hp_filter_order=2,
        hp_filter_type="bessel",
        hp_filter_cutoff_hz=250.0,
        lp_filter_order=2,
        lp_filter_type="bessel",
        lp_filter_cutoff_hz=250.0,
        spike_threshold_uv=-70.0,
        artifact_threshold_uv=2500.0,
        artifact_suppression_enabled=True,
    )
    meta = RecordingMeta(
        source_path=Path(f"{label}.rhs"),
        source_name=f"{label}.rhs",
        fs=fs,
        n_channels=n_channels,
        n_samples=window * (n_trials + 2),
        channel_names=names,
        segmentation=SegmentationInfo(
            mode="falling",
            n_trials=n_trials,
            n_triggers_detected=n_trials,
            pre_n=pre_n,
            post_n=post_n,
            pre_s=pre_s,
            post_s=post_s,
            end_rising_s=1.2,
        ),
        spike_detection=SpikeDetectionInfo(
            mode="rms_multiple",
            threshold_uv=-70.0,
            polarity="negative",
            rms_multiplier=4.0,
            artifact_threshold_uv=2500.0,
            artifact_suppression=True,
        ),
        dsp=dsp,
        cache_keys={"raw": "synthetic", "derived": "synthetic"},
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        overlay_pre_ms=overlay_pre_ms,
        overlay_post_ms=overlay_post_ms,
    )
    derived = DerivedArrays(
        t_rel=t_rel,
        triggers=np.arange(n_trials, dtype=np.float64) * (pre_s + post_s) + pre_s,
        means=means,
        trigger_windows=trigger_windows,
        rms_time=rms_time,
        rms_profiles=rms_profiles,
        channel_rms_uv=channel_rms,
        thresholds_uv=thresholds,
    )
    return ProcessedRecording(
        meta,
        derived,
        spikes,
        overlay,
        label=label,
        threshold_captions=[
            f"threshold −{abs(float(thresholds[ch])):.1f} µV (4.0 × RMS)"
            for ch in range(n_channels)
        ],
    )


if __name__ == "__main__":
    recording = build_synthetic()
    print(recording.describe())
