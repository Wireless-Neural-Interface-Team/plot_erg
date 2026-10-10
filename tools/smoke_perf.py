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


def _bench_montage_visibility() -> None:
    print("montage continuous visibility toggle:")
    from types import SimpleNamespace

    recording = build_synthetic(label="perf-montage", seed=5, n_channels=16)
    n_ch = int(recording.n_channels)
    n_samp = 20_000
    rng = np.random.default_rng(7)
    amp = rng.standard_normal((n_ch, n_samp), dtype=np.float32)
    # Fake continuous source so montage_continuous_raw has streams.
    recording.source = SimpleNamespace(
        amplifier=amp,
        highpass=amp * 0.3,
        lowpass=amp * 0.8,
    )
    recording.meta = replace(recording.meta, fs=2000.0) if hasattr(recording.meta, "fs") else recording.meta
    names = list(recording.channel_names)
    settings = ViewerSettings(preview_content="continuous")  # type: ignore[arg-type]
    placement = PanelPlacement("montage_continuous_raw")
    figure = Figure(figsize=(6.0, 10.0))

    def _request(hidden: tuple[str, ...]) -> RenderRequest:
        return RenderRequest(
            placement=placement,
            recordings=[recording],
            labels=[recording.label],
            colors=["#1f77b4"],
            legend_flags=[True],
            channel_index=0,
            channel_name=names[0],
            settings=replace(settings, hidden_channels=hidden),
        )

    def initial() -> None:
        figure._erg_structure_key = None  # type: ignore[attr-defined]
        status = render_panel(figure, _request(()))
        assert status == "ok", status
        assert getattr(figure, "_erg_montage_state", None) is not None

    def hide_one() -> None:
        before = len(figure.axes)
        status = render_panel(figure, _request((names[1],)))
        assert status == "ok", status
        # Shrink or rebuild may change axis count depending on kinds × channels.
        assert len(figure.axes) <= before

    def hide_more() -> None:
        before = len(figure.axes)
        status = render_panel(figure, _request(tuple(names[1:5])))
        assert status == "ok", status
        assert len(figure.axes) <= before

    def show_again() -> None:
        status = render_panel(figure, _request((names[1],)))
        assert status == "ok", status

    _time(f"initial montage ({n_ch} ch)", initial)
    _time("hide 1 channel (shrink path)", hide_one)
    _time("hide 4 channels (shrink path)", hide_more)
    _time("unhide 3 channels (cached rebuild)", show_again)


def _bench_continuous_require_ready() -> None:
    print("continuous_trace require_ready (UI-safe path):")
    from types import SimpleNamespace

    from dataset_builder import LazyFilterBank
    from intan_rhx_dsp import IntanDspSettings, build_intan_filter_sos

    recording = build_synthetic(label="perf-ready", seed=9, n_channels=4)
    n_ch, n_samp = 4, 40_000
    amp = np.random.default_rng(2).standard_normal((n_ch, n_samp)).astype(np.float32)
    dsp = IntanDspSettings(
        fs=20_000.0,
        hp_filter_order=2,
        hp_filter_type="bessel",
        hp_filter_cutoff_hz=250.0,
        lp_filter_order=2,
        lp_filter_type="bessel",
        lp_filter_cutoff_hz=250.0,
    )
    _notch, hp_sos, lp_sos = build_intan_filter_sos(dsp)
    bank = LazyFilterBank(amp, dsp, "hp", filter_sos=hp_sos, notch_sos=_notch)
    recording.source = SimpleNamespace(amplifier=amp, highpass=bank, lowpass=amp)
    recording.meta = replace(recording.meta, fs=20_000.0)

    def cold_ui() -> None:
        # Must not run sosfilt — empty when not ready.
        t, y = recording.continuous_trace("hp", 0, max_points=2000, require_ready=True)
        assert t.size == 0 and y.size == 0
        assert not recording.stream_ready("hp", 0)

    def after_prefetch() -> None:
        bank.prefetch([0])
        t, y = recording.continuous_trace("hp", 0, max_points=2000, require_ready=True)
        assert y.size > 0

    _time("require_ready cold (no sosfilt)", cold_ui)
    _time("prefetch + require_ready hit", after_prefetch)


def _bench_panel_prepare_pyqtgraph() -> None:
    print("panel_prepare + PlotHost (offscreen):")
    from PySide6.QtWidgets import QApplication

    from panel_prepare import prepare_panel
    from gui.widgets.plot_host import PlotHost

    app = QApplication.instance() or QApplication([])
    recording = build_synthetic(label="perf-pg", seed=11)
    settings = ViewerSettings()
    request = RenderRequest(
        placement=PanelPlacement("mean_raw", "full"),
        recordings=[recording],
        labels=[recording.label],
        colors=["#1f77b4"],
        legend_flags=[True],
        channel_index=0,
        channel_name=recording.channel_names[0],
        settings=settings,
    )

    def prepare() -> None:
        spec = prepare_panel(request)
        assert spec is not None

    host = PlotHost()

    def draw() -> None:
        spec = prepare_panel(request)
        host.render_spec(spec)

    _time("prepare_panel mean_raw", prepare)
    _time("PlotHost.render_spec mean_raw", draw)
    host.deleteLater()
    del host


def _bench_batch_flush_store() -> None:
    print("FilteredDiskStore batch flush:")
    from dataset_builder import _FilteredDiskStore
    from erg_cache import FilteredStreamLayout

    n_ch, n_samp = 6, 10_000
    with tempfile.TemporaryDirectory() as tmp:
        layout = FilteredStreamLayout(root=Path(tmp) / "_filter_batch")
        store = _FilteredDiskStore(layout, n_ch, n_samp, with_notch=False)
        row = np.ones(n_samp, dtype=np.float32)

        def writes_no_flush() -> None:
            for ch in range(n_ch):
                store.write("hp", ch, row, flush=False)
            store.flush()

        _time(f"write {n_ch} rows + one flush", writes_no_flush)
        assert store.is_ready("hp", 0)
        view = store.read_ready("hp", 0, copy=False)
        assert view is not None and view.shape == (n_samp,)
        store.close()


def main() -> int:
    print("=== smoke_perf ===")
    _bench_means()
    _bench_redraw()
    _bench_montage_visibility()
    _bench_continuous_require_ready()
    _bench_batch_flush_store()
    try:
        _bench_filter_disk_roundtrip()
    except Exception as exc:  # noqa: BLE001
        print(f"  filter disk bench skipped: {type(exc).__name__}: {exc}")
        return 1
    try:
        _bench_panel_prepare_pyqtgraph()
    except Exception as exc:  # noqa: BLE001
        print(f"  pyqtgraph bench skipped: {type(exc).__name__}: {exc}")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
