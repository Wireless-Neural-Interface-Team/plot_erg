"""Regression: removing a montage curve kind must not scramble later channels."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from make_synthetic_dataset import build_synthetic  # noqa: E402
from panel_registry import (  # noqa: E402
    RenderRequest,
    _montage_review_kinds,
    render_panel,
)
from view_config import AnalysisSettings, PanelPlacement, ViewerSettings  # noqa: E402


def _capture_montage_limits(figure: Figure):
    stored = getattr(figure, "_erg_montage_state", None)
    if not isinstance(stored, dict):
        return None
    row_keys = list(stored.get("row_keys") or [])
    axes = list(figure.axes)
    if not row_keys or len(row_keys) != len(axes):
        return None
    limits = {}
    for key, ax in zip(row_keys, axes):
        ch_kind = (int(key[0]), str(key[1]))
        xlim = (float(ax.get_xlim()[0]), float(ax.get_xlim()[1]))
        ylim = (float(ax.get_ylim()[0]), float(ax.get_ylim()[1]))
        limits[ch_kind] = (xlim, ylim)
    return limits


def _restore_montage_limits(figure: Figure, limits: dict) -> None:
    stored = getattr(figure, "_erg_montage_state", None)
    row_keys = list((stored or {}).get("row_keys") or [])
    axes = list(figure.axes)
    for key, ax in zip(row_keys, axes):
        ch_kind = (int(key[0]), str(key[1]))
        saved = limits.get(ch_kind)
        if saved is None:
            continue
        xlim, ylim = saved
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])


def main() -> int:
    recording = build_synthetic(label="bug-montage", seed=5, n_channels=4)
    n_ch = int(recording.n_channels)
    n_samp = 20_000
    rng = np.random.default_rng(7)
    amp = rng.standard_normal((n_ch, n_samp), dtype=np.float32)
    recording.source = SimpleNamespace(
        amplifier=amp, highpass=amp * 0.3, lowpass=amp * 0.8
    )
    names = list(recording.channel_names)

    analysis_full = AnalysisSettings(
        mode="average",
        show_raw=True,
        show_hp=True,
        show_lp=False,
        show_rms=True,
        show_psth=True,
        show_trial_rate=True,
        show_raster=True,
        show_overlay=True,
        show_isi=False,
    )
    analysis_less = replace(analysis_full, show_hp=False, show_trial_rate=False)
    placement = PanelPlacement("montage_continuous_raw")

    def req(analysis: AnalysisSettings, streams: tuple[str, ...]) -> RenderRequest:
        return RenderRequest(
            placement=placement,
            recordings=[recording],
            labels=[recording.label],
            colors=["#1f77b4"],
            legend_flags=[True],
            channel_index=0,
            channel_name=names[0],
            settings=ViewerSettings(
                preview_content="continuous",  # type: ignore[arg-type]
                continuous_streams=streams,  # type: ignore[arg-type]
                continuous_stream=streams[0],  # type: ignore[arg-type]
                analysis=analysis,
            ),
            preserve_view=False,
        )

    fig = Figure(figsize=(6, 12), layout=None)
    r1 = req(analysis_full, ("raw", "hp"))
    assert render_panel(fig, r1) == "ok", "initial render failed"
    saved = _capture_montage_limits(fig)
    assert saved is not None and (1, "raw") in saved

    # Old buggy path: restore by axis index after kind removal.
    index_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes]

    r2 = req(analysis_less, ("raw",))
    assert render_panel(fig, r2) == "ok", "rebuild after kind removal failed"
    kinds = _montage_review_kinds(r2)
    assert len(fig.axes) == 4 * len(kinds)

    # Demonstrate the old bug would scramble ch1 WIDE.
    fig_bug = Figure(figsize=(6, 12), layout=None)
    assert render_panel(fig_bug, r1) == "ok"
    assert render_panel(fig_bug, r2) == "ok"
    for index, ax in enumerate(fig_bug.axes):
        if index >= len(index_limits):
            break
        xlim, ylim = index_limits[index]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    row_kinds = list(getattr(fig_bug, "_erg_montage_row_kinds", []) or [])
    row_chs = list(getattr(fig_bug, "_erg_montage_row_channels", []) or [])
    buggy_xlim = None
    for i, (ch, kind) in enumerate(zip(row_chs, row_kinds)):
        if ch == 1 and kind == "raw":
            buggy_xlim = fig_bug.axes[i].get_xlim()
            break
    assert buggy_xlim is not None
    assert buggy_xlim[1] - buggy_xlim[0] < 5.0, "expected index restore to scramble"

    # Fixed path: restore by (ch, kind).
    _restore_montage_limits(fig, saved)
    for i, (ch, kind) in enumerate(
        zip(
            getattr(fig, "_erg_montage_row_channels", []),
            getattr(fig, "_erg_montage_row_kinds", []),
        )
    ):
        if ch == 1 and kind == "raw":
            xlim = fig.axes[i].get_xlim()
            assert xlim[1] - xlim[0] > 5.0, f"ch1 WIDE xlim still scrambled: {xlim}"
            assert abs(xlim[0] - saved[(1, "raw")][0][0]) < 1e-6
            break
    else:
        raise AssertionError("missing ch1 raw row")

    print("ok — index restore scrambles; key restore keeps ch1 WIDE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
