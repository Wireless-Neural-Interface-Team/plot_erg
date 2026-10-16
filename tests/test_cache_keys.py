"""Unit tests for content-addressed cache keys and prune policy."""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AnalysisConfig  # noqa: E402
from erg_cache import (  # noqa: E402
    CacheKeys,
    cache_size_bytes,
    human_bytes,
    prune_cache,
)


@pytest.fixture()
def rhs_file(tmp_path: Path) -> Path:
    path = tmp_path / "rec.rhs"
    path.write_bytes(b"INTAN_FAKE_RHS_v1")
    return path


def test_cache_keys_stable_for_same_config(rhs_file: Path) -> None:
    cfg = AnalysisConfig(rhs_file=rhs_file)
    a = CacheKeys.from_config(cfg)
    b = CacheKeys.from_config(cfg)
    assert a == b
    assert a.as_dict()["raw"] == b.as_dict()["raw"]


def test_cache_keys_change_with_filter_but_not_spike(rhs_file: Path) -> None:
    base = AnalysisConfig(rhs_file=rhs_file)
    filtered = replace(base, intan_hp_filter_cutoff_hz=300.0)
    spiked = replace(base, spike_threshold_uv=90.0)
    k0 = CacheKeys.from_config(base)
    k_f = CacheKeys.from_config(filtered)
    k_s = CacheKeys.from_config(spiked)
    assert k0.raw == k_f.raw == k_s.raw
    assert k0.filtered != k_f.filtered
    assert k0.filtered == k_s.filtered
    assert k0.spike != k_s.spike


def test_prune_cache_includes_raw(tmp_path: Path) -> None:
    root = tmp_path / ".erg_cache"
    raw = root / "_raw_old"
    filt = root / "_filter_old"
    raw.mkdir(parents=True)
    filt.mkdir(parents=True)
    (raw / "amplifier.npy").write_bytes(b"x" * 4096)
    (filt / "hp.npy").write_bytes(b"y" * 4096)
    # Age the raw tree so it is pruned first.
    old = time.time() - 10_000
    Path(raw).touch()
    import os

    os.utime(raw, (old, old))
    os.utime(filt, (time.time(), time.time()))
    before = cache_size_bytes(root)
    assert before >= 8000
    removed = prune_cache(root, max_bytes=5000, include_raw=True)
    assert removed >= 1
    assert cache_size_bytes(root) <= before


def test_human_bytes() -> None:
    assert human_bytes(512) == "512 B"
    assert "KB" in human_bytes(2048)
    assert "MB" in human_bytes(3 * 1024**2)
