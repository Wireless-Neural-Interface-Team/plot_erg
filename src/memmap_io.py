"""Channel-by-channel NumPy memmap I/O to limit peak RAM."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

CancelCheck = Callable[[], None] | None


def open_writable_memmap(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)


def write_2d_channelwise(
    source: np.ndarray,
    path: Path,
    *,
    dtype: np.dtype = np.dtype(np.float32),
    cancel_check: CancelCheck = None,
    progress_every: int = 0,
    on_channel: Callable[[int, int], None] | None = None,
) -> Path:
    """Copy a 2D array to disk one channel at a time (shape preserved)."""
    arr = np.asarray(source)
    if arr.ndim != 2:
        raise ValueError("write_2d_channelwise expects a 2D array.")
    n_ch, n_samp = int(arr.shape[0]), int(arr.shape[1])
    mm = open_writable_memmap(path, (n_ch, n_samp), dtype)
    try:
        for ch in range(n_ch):
            if cancel_check is not None:
                cancel_check()
            mm[ch] = np.asarray(arr[ch], dtype=dtype)
            if on_channel is not None:
                on_channel(ch, n_ch)
            elif progress_every > 0 and (ch + 1) % progress_every == 0:
                print(f"  memmap {path.name}: channel {ch + 1}/{n_ch}")
        mm.flush()
    finally:
        del mm
    return path


def load_readonly_memmap(path: Path) -> np.memmap:
    return np.load(path, mmap_mode="r")
