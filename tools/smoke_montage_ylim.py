"""Regression: montage extras must re-autoscale; preserve_view must not freeze 0–1."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from make_synthetic_dataset import build_synthetic  # noqa: E402
from gui.widgets.interactive_canvas import (  # noqa: E402
    _data_exceeds_ylim,
    _is_mpl_default_ylim,
)
from gui.widgets.panel_canvas import _axis_keeps_rendered_ylim  # noqa: E402
from panel_registry import RenderRequest, render_panel  # noqa: E402
from view_config import AnalysisSettings, PanelPlacement, ViewerSettings  # noqa: E402


def _ylim(ax) -> tuple[float, float]:
    y0, y1 = ax.get_ylim()
    return float(y0), float(y1)


def _span(ylim: tuple[float, float]) -> float:
    return abs(ylim[1] - ylim[0])


def _make_recording():
    recording = build_synthetic(label="montage-ylim", seed=5, n_channels=2)
    n_ch = int(recording.n_channels)
    amp = np.random.default_rng(7).standard_normal((n_ch, 20_000), dtype=np.float32) * 40.0
    recording.source = SimpleNamespace(
        amplifier=amp, highpass=amp * 0.3, lowpass=amp * 0.8
    )
    return recording


def _request(recording, *, analysis: AnalysisSettings | None = None) -> RenderRequest:
    analysis = analysis or AnalysisSettings(
        mode="average",
        show_raw=True,
        show_hp=True,
        show_lp=False,
        show_rms=False,
        show_psth=True,
        show_trial_rate=True,
        show_raster=True,
        show_overlay=True,
        show_isi=True,
    )
    return RenderRequest(
        placement=PanelPlacement("montage_continuous_raw"),
        recordings=[recording],
        labels=[recording.label],
        colors=["#1f77b4"],
        legend_flags=[True],
        channel_index=0,
        channel_name=recording.channel_names[0],
        settings=ViewerSettings(
            preview_content="continuous",  # type: ignore[arg-type]
            continuous_streams=("raw", "hp"),  # type: ignore[arg-type]
            continuous_stream="raw",  # type: ignore[arg-type]
            analysis=analysis,
            montage_review_channels=1,
        ),
        preserve_view=False,
    )


def _assert_kind_autoscaled(figure: Figure, kind: str, *, min_span: float) -> None:
    kinds = list(getattr(figure, "_erg_montage_row_kinds", []) or [])
    for index, ax in enumerate(figure.axes):
        if index >= len(kinds) or str(kinds[index]) != kind:
            continue
        y = _ylim(ax)
        if _is_mpl_default_ylim(y) or _span(y) < min_span:
            raise AssertionError(
                f"{kind}: ylim trop étroit / défaut matplotlib {y} (span={_span(y):.3f})"
            )
        return
    raise AssertionError(f"aucune ligne {kind} dans le montage")


def main() -> int:
    recording = _make_recording()
    req = _request(recording)
    fig = Figure(figsize=(6.0, 10.0), layout=None)
    status = render_panel(fig, req)
    if status != "ok":
        raise AssertionError(f"rendu initial status={status}")

    _assert_kind_autoscaled(fig, "raw", min_span=20.0)
    _assert_kind_autoscaled(fig, "overlay", min_span=5.0)
    _assert_kind_autoscaled(fig, "trial_rate", min_span=1.0)
    _assert_kind_autoscaled(fig, "isi", min_span=1.0)
    _assert_kind_autoscaled(fig, "psth", min_span=1.0)
    print("ok  premier rendu autoscalé")

    # Empoisonner comme un vieux snapshot 0–1, puis refresh style.
    kinds = list(fig._erg_montage_row_kinds)
    for ax, kind in zip(fig.axes, kinds):
        if kind in {"raw", "hp", "overlay", "trial_rate", "isi", "psth"}:
            ax.set_ylim(0.0, 1.0)

    status = render_panel(fig, req)
    if status != "ok":
        raise AssertionError(f"refresh style status={status}")
    _assert_kind_autoscaled(fig, "raw", min_span=20.0)
    _assert_kind_autoscaled(fig, "overlay", min_span=5.0)
    _assert_kind_autoscaled(fig, "trial_rate", min_span=1.0)
    _assert_kind_autoscaled(fig, "isi", min_span=1.0)
    print("ok  refresh style ré-autoscalé après 0–1")

    # preserve_view : extras doivent garder le ylim du rendu.
    for index, kind in enumerate(kinds):
        keeps = _axis_keeps_rendered_ylim(fig, req, index, previous=req)
        if kind in {"psth", "trial_rate", "isi", "overlay", "raster"}:
            if not keeps:
                raise AssertionError(f"{kind}: doit garder ylim rendu (pas restore)")
        elif kind in {"raw", "hp"}:
            if keeps:
                raise AssertionError(f"{kind}: auto stable doit autoriser restore zoom")
    print("ok  keep-rendered pour extras montage")

    # Garde-fou défaut matplotlib.
    ax0 = fig.axes[0]
    if not _is_mpl_default_ylim((0.0, 1.0)):
        raise AssertionError("détecteur (0,1) cassé")
    ax0.set_ylim(-100.0, 100.0)
    # dataLim encore large après autoscale précédent
    if not _data_exceeds_ylim(ax0, (0.0, 1.0)):
        raise AssertionError("dataLim devrait dépasser (0,1) sur WIDE")
    print("ok  détecteur ylim défaut / data exceed")

    print("smoke_montage_ylim: all passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        raise
