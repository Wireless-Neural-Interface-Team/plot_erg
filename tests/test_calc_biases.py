"""Regression tests for RMS length, PSTH edge rate, and overlay mean."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from channel_metrics import mean_rms_profile_from_row  # noqa: E402
from core import valid_triggers_and_timebase  # noqa: E402
from draw_primitives import _extract_spike_waveforms, _psth_mean_hz  # noqa: E402
from intan_rhx_dsp import IntanDspSettings  # noqa: E402
from processed_dataset import MAX_OVERLAY_SNIPPETS_PER_CHANNEL  # noqa: E402


def test_rms_profile_matches_mean_window_length() -> None:
    fs = 30_000.0
    pre_s, post_s = 0.05, 0.2
    n_samples = 300_000
    trigs = np.array([100_000, 150_000, 200_000], dtype=np.int64)
    valid, t_rel, pre_n, post_n = valid_triggers_and_timebase(
        n_samples, trigs, fs, pre_s, post_s
    )
    row = np.random.default_rng(0).normal(0, 1, n_samples).astype(np.float32)
    dsp = IntanDspSettings(fs=fs, rms_window_s=0.01)
    _axis, rms = mean_rms_profile_from_row(
        row,
        fs=fs,
        valid_triggers=valid,
        t0_s=float(t_rel[0]),
        t1_s=float(t_rel[-1]),
        dsp=dsp,
    )
    assert rms.size == pre_n + post_n
    assert rms.size == t_rel.size


def test_psth_edges_flat_for_uniform_bin_counts() -> None:
    """Effective-width normalization keeps a flat rate when every bin is equal.

    The previous ``convolve(..., mode='same') / full_window`` form under-estimated
    edges by ~window_bins/2 / window_bins (e.g. 0.6× for a 50 ms bin).
    """
    dt = 0.001
    t_rel = np.arange(0.0, 1.0, dt)
    # One spike per 1 ms bin, one trial → true rate = 1/dt = 1000 Hz everywhere.
    spikes = t_rel + 0.5 * dt
    _centers, rate = _psth_mean_hz(
        [spikes], t_rel, n_trials=1, bin_width_s=0.05
    )
    assert rate.size > 0
    assert float(np.min(rate)) == pytest.approx(1000.0, rel=1e-9)
    assert float(np.max(rate)) == pytest.approx(1000.0, rel=1e-9)


def test_overlay_mean_uses_all_detections_when_capped() -> None:
    fs = 1_000.0
    n_samples = 50_000
    row = np.zeros(n_samples, dtype=np.float32)
    n_spikes = MAX_OVERLAY_SNIPPETS_PER_CHANNEL + 500
    centers = np.arange(100, 100 + n_spikes * 10, 10, dtype=np.int64)
    assert centers.size == n_spikes
    # Step amplitudes: early spikes tiny, late spikes huge — a display subsample
    # mean must not replace the exact all-detection mean.
    for i, c in enumerate(centers.tolist()):
        row[c] = 1.0 if i < n_spikes // 2 else 1000.0

    source = SimpleNamespace(
        fs=fs,
        valid_triggers=np.array([0], dtype=np.int64),
        high_trace_for_channel=lambda _ch: row,
    )
    trains = [centers.astype(np.float64) / fs]
    _t_ms, waves, _times, total, exact_mean = _extract_spike_waveforms(
        source,  # type: ignore[arg-type]
        0,
        trains,
        pre_ms=0.0,
        post_ms=1.0,
    )
    assert total == n_spikes
    assert waves.shape[0] == MAX_OVERLAY_SNIPPETS_PER_CHANNEL
    assert exact_mean is not None
    expected = float(np.mean(row[centers]))
    assert float(exact_mean[0]) == pytest.approx(expected, rel=1e-9)
