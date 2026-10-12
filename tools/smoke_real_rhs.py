"""Exercise the pipeline on a real multi-gigabyte RHS file.

Looks for ``exemple.rhs`` in the repo root. Skips (exit 0) when the file is
absent so CI without the recording still passes.

Run with:
    si_env\\Scripts\\python.exe tools\\smoke_real_rhs.py
"""

from __future__ import annotations

import gc
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import AnalysisConfig  # noqa: E402
from dataset_builder import build_recording, ensure_channels  # noqa: E402
from intanutil.data import (  # noqa: E402
    advance_index,
    calculate_data_size,
    initialize_memory,
    read_all_data_blocks_to_memmap,
    read_one_data_block,
)
from intanutil.header import read_header  # noqa: E402
from memmap_io import close_memmap, open_writable_memmap  # noqa: E402
from panel_registry import RenderRequest, render_panel  # noqa: E402
from view_config import PanelPlacement, ViewerSettings  # noqa: E402

EXAMPLE = ROOT / "exemple.rhs"
PREFIX_BLOCKS = 64


def _allocated_bytes(path: Path) -> int:
    if sys.platform != "win32":
        return path.stat().st_size
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    high = wintypes.DWORD(0)
    low = kernel32.GetCompressedFileSizeW(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.get_last_error() != 0:
        return path.stat().st_size
    return (int(high.value) << 32) + int(low)


def _check_sparse_memmap() -> None:
    print("sparse memmap:")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "big.npy"
        n = 8_000_000  # 32 MB logical
        mm = open_writable_memmap(path, (4, n), np.dtype(np.float32))
        mm[0, :1000] = np.linspace(-1, 1, 1000, dtype=np.float32)
        mm.flush()
        logical = path.stat().st_size
        allocated = _allocated_bytes(path)
        close_memmap(mm)
        del mm
        reloaded = np.load(path, mmap_mode="r")
        try:
            if not np.allclose(reloaded[0, :1000], np.linspace(-1, 1, 1000, dtype=np.float32)):
                raise AssertionError("sparse memmap round-trip mismatch")
            if not np.all(reloaded[1, :100] == 0):
                raise AssertionError("unwritten sparse region is not zero")
        finally:
            close_memmap(reloaded)
        print(f"  logical {logical / 1e6:.1f} MB, allocated {allocated / 1e6:.1f} MB")
        if sys.platform == "win32" and allocated > logical * 0.25:
            raise AssertionError(
                f"memmap was fully allocated ({allocated} bytes); sparse create failed"
            )


def _check_prefix(path: Path) -> None:
    print(f"prefix vs legacy reader ({PREFIX_BLOCKS} blocks):")
    with path.open("rb") as fid:
        header = read_header(fid)
        _present, _size, total_blocks, _samples = calculate_data_size(header, str(path), fid)
        n_blocks = min(PREFIX_BLOCKS, int(total_blocks))
        n_samples = n_blocks * int(header["num_samples_per_data_block"])
        data_pos = fid.tell()
        legacy, index = initialize_memory(header, n_samples)
        for _ in range(n_blocks):
            read_one_data_block(legacy, header, index, fid)
            index = advance_index(index, header["num_samples_per_data_block"])
        legacy_amp = legacy["amplifier_data"].astype(np.float32)
        legacy_amp -= np.float32(32768.0)
        legacy_amp *= np.float32(0.195)
        legacy_adc = legacy["board_adc_data"].astype(np.float32)
        legacy_adc -= np.float32(32768.0)
        legacy_adc *= np.float32(312.5e-6)

        fid.seek(data_pos)
        with tempfile.TemporaryDirectory() as tmp:
            mm = open_writable_memmap(
                Path(tmp) / "amp.npy",
                (int(header["num_amplifier_channels"]), n_samples),
                np.dtype(np.float32),
            )
            try:
                streamed = read_all_data_blocks_to_memmap(
                    header, n_samples, n_blocks, fid, mm
                )
                new_amp = np.array(mm, dtype=np.float32, copy=True)
                new_adc = np.array(streamed["board_adc_data"], dtype=np.float32, copy=True)
            finally:
                close_memmap(mm)
                del mm
    if not np.allclose(new_amp, legacy_amp, rtol=0, atol=0):
        diff = np.max(np.abs(new_amp - legacy_amp))
        raise AssertionError(f"amplifier prefix mismatch, max abs {diff}")
    if not np.allclose(new_adc, legacy_adc, rtol=0, atol=1e-6):
        diff = np.max(np.abs(new_adc - legacy_adc))
        raise AssertionError(f"ADC prefix mismatch, max abs {diff}")
    print(f"  {n_blocks} blocks match ({new_amp.shape[0]} ch × {new_amp.shape[1]} samples)")


def _cache_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="plot_erg_real_"))


def _render(recording, channel: int) -> dict[str, str]:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(6.0, 4.0), layout="constrained")
    settings = ViewerSettings()
    # ISI stays empty when a channel has fewer than two spikes in the same trial.
    required_ok = {
        "mean_raw",
        "mean_hp",
        "rms",
        "raster",
        "psth",
        "spike_overlay",
        "full_recording",
    }
    keys = ("mean_raw", "mean_hp", "rms", "raster", "psth", "isi", "spike_overlay", "full_recording")
    statuses: dict[str, str] = {}
    for key in keys:
        request = RenderRequest(
            placement=PanelPlacement(key, "full"),
            recordings=[recording],
            labels=[recording.label],
            colors=["#1f77b4"],
            legend_flags=[True],
            channel_index=channel,
            channel_name=recording.channel_names[channel],
            settings=settings,
        )
        status = render_panel(figure, request)
        statuses[key] = status
        if status not in {"ok", "empty"}:
            raise AssertionError(f"panel {key} rendered as {status}")
        if key in required_ok and status != "ok":
            raise AssertionError(f"panel {key} is {status} on {recording.channel_names[channel]}")
    t, y = recording.continuous_trace("raw", channel, max_points=4000)
    if t.size == 0 or y.size == 0:
        raise AssertionError("continuous trace is empty")
    if y.size > 8000:
        raise AssertionError(f"continuous trace was not decimated ({y.size} points)")
    return statuses


def main() -> int:
    if not EXAMPLE.exists():
        print(f"skip: {EXAMPLE.name} not found")
        return 0
    _check_sparse_memmap()
    _check_prefix(EXAMPLE)

    cache = _cache_dir()
    print(f"cache: {cache}")
    recording = None
    try:
        config = AnalysisConfig(rhs_file=EXAMPLE, work_dir=cache)
        started = time.perf_counter()
        try:
            recording, report = build_recording(config, progress=lambda event: None)
        except RuntimeError as exc:
            if "edge" not in str(exc).lower():
                raise
            print(f"  default trigger failed: {exc}")
            recording = None
            raise
        read_s = time.perf_counter() - started
        print(
            f"build: {read_s:.1f} s — {recording.n_channels} ch, "
            f"{recording.meta.n_samples} samples, {recording.n_trials} trials, "
            f"fs {recording.meta.fs:.0f}"
        )
        if recording.n_trials < 1:
            raise AssertionError("no trials after segmentation")
        n_trials = int(recording.n_trials)

        channels = list(range(recording.n_channels))
        started = time.perf_counter()
        computed = ensure_channels(recording, channels, config, progress=lambda event: None)
        channel_s = time.perf_counter() - started
        print(f"ensure {len(computed)} channels: {channel_s:.1f} s")
        if len(computed) != recording.n_channels:
            raise AssertionError(f"computed {len(computed)} / {recording.n_channels}")

        scores = []
        for index in range(recording.n_channels):
            curve = recording.mean("hp", index)
            scores.append(0.0 if curve is None else float(np.max(np.abs(curve))))
        ch = int(np.argmax(scores))
        print(f"strongest channel: {ch} {recording.channel_names[ch]} peak {scores[ch]:.1f} µV")
        mean = recording.mean("hp", ch)
        if mean is None or mean.size < 10 or not np.isfinite(mean).all():
            raise AssertionError("mean HP trace is missing or non-finite")
        mean_ref = np.array(mean, dtype=np.float32, copy=True)
        statuses = _render(recording, ch)
        print(f"panels: {statuses}")

        recording.close()
        recording = None
        gc.collect()

        started = time.perf_counter()
        reopened, _report = build_recording(config, progress=lambda event: None)
        reopen_s = time.perf_counter() - started
        print(f"reopen from cache: {reopen_s:.2f} s")
        if int(reopened.n_trials) != n_trials:
            raise AssertionError(f"reopened trials {reopened.n_trials} != {n_trials}")
        if not reopened.source.highpass.is_ready(ch) or not reopened.source.lowpass.is_ready(ch):
            raise AssertionError("filtered rows were not reused from disk")
        ensure_channels(reopened, [ch], config, progress=lambda event: None)
        mean_again = reopened.mean("hp", ch)
        if mean_again is None or not np.allclose(mean_again, mean_ref, rtol=1e-5, atol=1e-3):
            raise AssertionError("reopened mean HP does not match")
        reopened.close()
        print("real RHS pipeline OK")
        return 0
    finally:
        if recording is not None:
            try:
                recording.close()
            except Exception:
                pass
        gc.collect()
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
