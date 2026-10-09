"""Timing smoke for filter cache, ensure_channels, and incremental redraw.

Run with:
    si_env\\Scripts\\python.exe tools\\smoke_perf.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from dataclasses import replace  # noqa: E402

from channel_metrics import mean_trial_windows  # noqa: E402
from make_synthetic_dataset import build_synthetic  # noqa: E402
from panel_registry import RenderRequest, render_panel  # noqa: E402
from view_config import PanelPlacement, ViewerSettings  # noqa: E402


def _time(label: str, fn) -> float:
    started = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - started
    print(f"  {label:42s} {elapsed * 1000:8.1f} ms")
    return elapsed


def _bench_means() -> None:
    print("mean_trial_windows (vectorized):")
    rng = np.random.default_rng(0)
    row = rng.standard_normal(200_000).astype(np.float32)
    triggers = np.arange(5_000, 180_000, 8_000, dtype=np.int64)
    pre_n, post_n = 1000, 4000

    def once() -> None:
        mean_trial_windows(row, triggers, pre_n, post_n)

    once()  # warm
    _time("20 trials × 5k samples", once)


def _bench_redraw() -> None:
    print("incremental redraw:")
    recording = build_synthetic(label="perf", seed=3)
    settings = ViewerSettings()
    placement = PanelPlacement("mean_raw", "full")
    request = RenderRequest(
        placement=placement,
        recordings=[recording],
        labels=[recording.label],
        colors=["#1f77b4"],
        legend_flags=[True],
        channel_index=1,
        channel_name=recording.channel_names[1],
        settings=settings,
    )
    figure = Figure(figsize=(6.0, 3.5), layout="constrained")

    def full() -> None:
        figure._erg_structure_key = None  # type: ignore[attr-defined]
        render_panel(figure, request)

    def style_only() -> None:
        # Same data fingerprint → style refresh path.
        tweaked = RenderRequest(
            placement=placement,
            recordings=[recording],
            labels=[recording.label],
            colors=["#dc2626"],
            legend_flags=[True],
            channel_index=1,
            channel_name=recording.channel_names[1],
            settings=replace(
                settings,
                style=replace(
                    settings.style,
                    line_width=float(settings.style.line_width) + 0.2,
                    grid=not bool(settings.style.grid),
                ),
            ),
        )
        render_panel(figure, tweaked)

    _time("full rebuild mean_raw", full)
    _time("display-only refresh (color/grid)", style_only)
    _time("display-only refresh (2nd)", style_only)


def _bench_filter_disk_roundtrip() -> None:
    print("LazyFilterBank disk cache (synthetic memmap path):")
    from collections import OrderedDict

    from dataset_builder import LazyFilterBank, _FilteredDiskStore, LAZY_FILTER_CACHE_MAX_CHANNELS
    from erg_cache import FilteredStreamLayout
    from intan_rhx_dsp import IntanDspSettings, build_intan_filter_sos

    n_ch, n_samp = 8, 50_000
    amp = np.random.default_rng(1).standard_normal((n_ch, n_samp)).astype(np.float32)
    dsp = IntanDspSettings(
        fs=20_000.0,
        hp_filter_order=2,
        hp_filter_type="bessel",
        hp_filter_cutoff_hz=250.0,
        lp_filter_order=2,
        lp_filter_type="bessel",
        lp_filter_cutoff_hz=250.0,
    )
    notch_sos, hp_sos, lp_sos = build_intan_filter_sos(dsp)
    with tempfile.TemporaryDirectory() as tmp:
        layout = FilteredStreamLayout(root=Path(tmp) / "_filter_test")
        store = _FilteredDiskStore(layout, n_ch, n_samp, with_notch=notch_sos is not None)
        bank = LazyFilterBank(
            amp,
            dsp,
            "hp",
            max_cached_channels=2,  # force disk hits after eviction
            notch_cache=OrderedDict() if notch_sos is not None else None,
            filter_sos=hp_sos,
            notch_sos=notch_sos,
            disk_store=store,
        )

        def cold() -> None:
            bank.clear()
            for ch in range(n_ch):
                _ = bank[ch]

        def warm_disk() -> None:
            bank.clear()  # drop RAM; disk rows stay ready
            for ch in range(n_ch):
                _ = bank[ch]

        _time(f"cold filter {n_ch} ch × {n_samp} samp", cold)
        _time(f"warm disk reload (LRU={LAZY_FILTER_CACHE_MAX_CHANNELS})", warm_disk)
        # Release Windows file locks before TemporaryDirectory cleanup.
        bank.clear()
        for attr in ("hp", "lp", "notch", "ready_hp", "ready_lp", "ready_notch"):
            mm = getattr(store, attr, None)
            if mm is not None:
                try:
                    if hasattr(mm, "_mmap") and mm._mmap is not None:
                        mm._mmap.close()
                except Exception:
                    pass
        del bank, store


def main() -> int:
    print("=== smoke_perf ===")
    _bench_means()
    _bench_redraw()
    try:
        _bench_filter_disk_roundtrip()
    except Exception as exc:  # noqa: BLE001
        print(f"  filter disk bench skipped: {type(exc).__name__}: {exc}")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
