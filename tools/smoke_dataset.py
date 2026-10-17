"""Round-trip the processed dataset format: write, reopen, compare, render.

Run with:
    si_env\\Scripts\\python.exe tools\\smoke_dataset.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.figure import Figure  # noqa: E402

from dataset_builder import export_dataset, open_dataset  # noqa: E402
from make_synthetic_dataset import build_synthetic  # noqa: E402
from panel_registry import RenderRequest, render_panel  # noqa: E402
from processed_dataset import (  # noqa: E402
    ChannelData,
    DerivedArrays,
    OverlaySnippets,
    SpikeTrains,
    archive_bundle,
    dataset_target_path,
)
from view_config import PanelPlacement, ViewerSettings  # noqa: E402


def _assert_close(name: str, a: np.ndarray | None, b: np.ndarray | None) -> None:
    if a is None and b is None:
        return
    if a is None or b is None:
        raise AssertionError(f"{name}: one side is missing ({a is None}, {b is None})")
    if not np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True):
        raise AssertionError(f"{name}: arrays differ")


def main() -> int:
    original = build_synthetic(label="roundtrip", seed=7)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = dataset_target_path(root, original.label)

        started = time.perf_counter()
        bundle = export_dataset(original, target, progress=print)
        write_s = time.perf_counter() - started

        started = time.perf_counter()
        reopened = open_dataset(bundle, label="reopened")
        read_s = time.perf_counter() - started

        assert reopened.n_channels == original.n_channels
        assert reopened.n_trials == original.n_trials
        assert reopened.channel_names == original.channel_names
        assert reopened.end_marker_s == original.end_marker_s
        _assert_close("t_rel", reopened.t_rel, original.t_rel)
        for stream in ("raw", "hp", "lp"):
            _assert_close(f"mean[{stream}]", reopened.mean(stream, 3), original.mean(stream, 3))
            for trigger in (0, 1):
                _assert_close(
                    f"trigger[{trigger}][{stream}]",
                    reopened.trigger_window(trigger, stream, 3),
                    original.trigger_window(trigger, stream, 3),
                )
        for kind in ("mean", "first", "second"):
            t_a, v_a = original.rms_profile(kind, 3)
            t_b, v_b = reopened.rms_profile(kind, 3)
            _assert_close(f"rms_time[{kind}]", t_a, t_b)
            _assert_close(f"rms[{kind}]", v_a, v_b)
        assert abs(reopened.channel_rms_uv(3) - original.channel_rms_uv(3)) < 1e-4
        assert abs(reopened.threshold_uv(3) - original.threshold_uv(3)) < 1e-4
        assert reopened.threshold_caption(3) == original.threshold_caption(3)
        for ch in range(original.n_channels):
            left = original.spike_times(ch)
            right = reopened.spike_times(ch)
            assert len(left) == len(right), f"channel {ch}: trial count differs"
            for trial, (a, b) in enumerate(zip(left, right)):
                _assert_close(f"spikes[{ch}][{trial}]", a, b)
        t_ms_a, waves_a, times_a, total_a, mean_a = original.overlay_for_channel(3)
        t_ms_b, waves_b, times_b, total_b, mean_b = reopened.overlay_for_channel(3)
        _assert_close("overlay t_ms", t_ms_a, t_ms_b)
        _assert_close("overlay waves", waves_a, waves_b)
        _assert_close("overlay times", times_a, times_b)
        _assert_close("overlay mean", mean_a, mean_b)
        assert total_a == total_b

        figure = Figure(figsize=(6.0, 4.0), layout="constrained")
        settings = ViewerSettings()
        statuses: dict[str, int] = {}
        for key in ("mean_raw", "mean_hp", "rms", "raster", "psth", "isi", "spike_overlay"):
            request = RenderRequest(
                placement=PanelPlacement(key, "full"),
                recordings=[reopened],
                labels=[reopened.label],
                colors=["#1f77b4"],
                legend_flags=[True],
                channel_index=3,
                channel_name=reopened.channel_names[3],
                settings=settings,
            )
            status = render_panel(figure, request)
            statuses[status] = statuses.get(status, 0) + 1

        archive = archive_bundle(bundle, bundle.with_suffix(".zip"))
        size_mb = archive.stat().st_size / (1024.0 * 1024.0)
        from_zip = open_dataset(archive, label="from zip")
        _assert_close("zip mean[hp]", from_zip.mean("hp", 3), original.mean("hp", 3))

        print("\nround trip OK")
        print(f"  write:   {write_s:.2f} s")
        print(f"  reopen:  {read_s * 1000:.0f} ms")
        print(f"  archive: {size_mb:.2f} MB")
        print(f"  panels:  {statuses}")
        reopened.close()
        from_zip.close()

        # GUI-like lazy state: products only in ``_channel_data``.
        lazy = build_synthetic(label="lazy-export", seed=11, n_channels=4, n_trials=3)
        for ch in range(lazy.n_channels):
            means = {s: lazy.mean(s, ch) for s in ("raw", "hp", "lp")}
            t_rms, v_mean = lazy.rms_profile("mean", ch)
            _, v_first = lazy.rms_profile("first", ch)
            _, v_second = lazy.rms_profile("second", ch)
            t_ms, waves, times, total, mean_w = lazy.overlay_for_channel(ch)
            lazy.store_channel(
                ch,
                ChannelData(
                    means={k: np.asarray(v) for k, v in means.items() if v is not None},
                    rms_time=np.asarray(t_rms),
                    rms_profiles={
                        "mean": np.asarray(v_mean),
                        "first": np.asarray(v_first),
                        "second": np.asarray(v_second),
                    },
                    channel_rms_uv=float(lazy.channel_rms_uv(ch)),
                    threshold_uv=float(lazy.threshold_uv(ch)),
                    threshold_caption=str(lazy.threshold_caption(ch)),
                    spike_trains=list(lazy.spike_times(ch)),
                    overlay_t_ms=np.asarray(t_ms),
                    overlay_waves=np.asarray(waves),
                    overlay_times=np.asarray(times),
                    overlay_total=int(total),
                    overlay_mean=None if mean_w is None else np.asarray(mean_w),
                ),
            )
        lazy.derived = DerivedArrays(
            t_rel=lazy.derived.t_rel,
            triggers=lazy.derived.triggers,
            means={},
            trigger_windows={},
            rms_time=None,
            rms_profiles={},
            channel_rms_uv=None,
            thresholds_uv=None,
        )
        lazy.spikes = SpikeTrains.empty(lazy.n_channels, lazy.n_trials)
        lazy.overlay = OverlaySnippets.empty(lazy.n_channels, 0)
        expected_raw = np.asarray(lazy.mean("raw", 1), dtype=np.float64).copy()
        assert expected_raw is not None and expected_raw.size
        lazy_target = dataset_target_path(root, "lazy-export")
        lazy_bundle = export_dataset(lazy, lazy_target)
        lazy_reopened = open_dataset(lazy_bundle, label="lazy-reopened")
        assert lazy_reopened.ready_channels == set(range(lazy.n_channels))
        _assert_close("lazy mean[raw]", lazy_reopened.mean("raw", 1), expected_raw)
        assert any(len(t) for t in lazy_reopened.spike_times(1))
        lazy_reopened.close()
        lazy.close()
        print("lazy channel-cache export OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
