"""Shared plotting helpers used by PDF rendering modules."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def shift_axes_down(axes: Sequence[Any], delta: float) -> None:
    """Shift a group of axes downward (figure coordinates)."""
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - delta, pos.width, pos.height])


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
