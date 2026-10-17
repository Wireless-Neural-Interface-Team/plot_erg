"""Unit tests for stimulation / section window resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AnalysisConfig  # noqa: E402
from core import detect_edges, resolve_recording_windows  # noqa: E402


def test_detect_falling_edges() -> None:
    signal = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0], dtype=np.float64)
    edges = detect_edges(signal, threshold=1.0, edge="falling")
    assert edges.tolist() == [3, 6]


def test_detect_rising_edges() -> None:
    signal = np.array([0.0, 0.0, 2.0, 2.0, 0.0, 2.0], dtype=np.float64)
    edges = detect_edges(signal, threshold=1.0, edge="rising")
    assert edges.tolist() == [2, 5]


def test_resolve_recording_windows_with_triggers() -> None:
    fs = 1000.0
    n_samples = 5000
    analog = np.zeros(n_samples, dtype=np.float64)
    analog[1000:1500] = 2.0
    analog[3000:3400] = 2.0
    config = AnalysisConfig(
        rhs_file=Path("dummy.rhs"),
        edge="falling",
        threshold=1.0,
        pre_s=0.1,
        post_s=0.2,
    )
    triggers, t_rel, pre_n, post_n, n_valid, n_total, end_rising, _ = (
        resolve_recording_windows(config, n_samples, fs, analog)
    )
    assert n_total == 2
    assert n_valid == 2
    assert pre_n == 100
    assert post_n == 200
    assert triggers.size == 2
    assert t_rel.size == pre_n + post_n
    assert end_rising is not None
    # Falling at 1500/3400; mean delay to next rising (3000) = 1.5 s.
    assert end_rising == pytest.approx(1.5, abs=1e-6)


def test_resolve_recording_windows_no_edge_raises() -> None:
    fs = 1000.0
    n_samples = 1000
    analog = np.zeros(n_samples, dtype=np.float64)
    config = AnalysisConfig(rhs_file=Path("dummy.rhs"), edge="falling")
    with pytest.raises(RuntimeError, match="No falling edge"):
        resolve_recording_windows(config, n_samples, fs, analog)


def test_resolve_sections_without_stimulation() -> None:
    fs = 100.0
    n_samples = 1000
    analog = np.zeros(n_samples, dtype=np.float64)
    config = AnalysisConfig(
        rhs_file=Path("dummy.rhs"),
        edge="none",
        section_count=5,
        section_spec="count",
        section_trigger_start_s=0.5,
        section_trigger_end_s=1.5,
        pre_s=0.2,
        post_s=0.2,
    )
    triggers, t_rel, pre_n, post_n, n_valid, n_total, end_rising, section_dur = (
        resolve_recording_windows(config, n_samples, fs, analog)
    )
    assert n_total == 5
    assert n_valid >= 1
    assert pre_n == 20
    assert post_n == 20
    assert triggers.size == n_valid
    assert t_rel.size == pre_n + post_n
    assert section_dur == pytest.approx(2.0, abs=1e-6)
    assert end_rising == pytest.approx(1.0, abs=1e-6)
