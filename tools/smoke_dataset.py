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
from processed_dataset import archive_bundle, dataset_target_path  # noqa: E402
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
