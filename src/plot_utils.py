"""Shared plotting helpers used by PDF rendering modules."""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np


def shorten_filename_for_windows(output_dir: Path, filename: str, max_total_len: int = 240) -> str:
    """Shorten filename if total path length may exceed Windows limits."""
    full_len = len(str(output_dir / filename))
    if full_len <= max_total_len:
        return filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".pdf"
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
    budget = max_total_len - len(str(output_dir)) - len(suffix) - len(digest) - 2
    budget = max(24, budget)
    short_stem = stem[:budget]
    return f"{short_stem}_{digest}{suffix}"


def downsample_points(x: np.ndarray, y: np.ndarray, sampling_percent: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically downsample points based on a percentage in [1, 100]."""
    if sampling_percent >= 100:
        return x, y
    pct = max(1, min(100, int(sampling_percent)))
    step = max(1, int(np.ceil(100.0 / float(pct))))
    return x[::step], y[::step]


def decimate_envelope(
    x: np.ndarray, y: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope decimation: keeps peaks while bounding the point count."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    n = int(x_arr.size)
    limit = max(16, int(max_points))
    if n <= limit:
        return x_arr, y_arr
    n_bins = max(8, limit // 2)
    edges = np.linspace(0, n, n_bins + 1).astype(np.int64)
    starts = edges[:-1]
    valid = starts < n
    starts = starts[valid]
    lows = np.minimum.reduceat(y_arr, starts)
    highs = np.maximum.reduceat(y_arr, starts)
    mids = np.add.reduceat(x_arr, starts) / np.diff(np.append(starts, n))
    out_x = np.repeat(mids, 2)
    out_y = np.empty(out_x.size, dtype=np.float64)
    out_y[0::2] = lows
    out_y[1::2] = highs
    return out_x, out_y
