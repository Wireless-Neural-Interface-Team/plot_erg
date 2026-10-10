"""Regression: HIGH/LOW Y auto-scale after leaving manual mode.

Manuel → auto must re-autoscale on style refresh (same structure fingerprint)
and must not be overwritten by preserve_view restore logic.
"""

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
from panel_registry import RenderRequest, render_panel  # noqa: E402
from view_config import AxisLimits, PanelPlacement, ViewerSettings  # noqa: E402


def _ylim(ax) -> tuple[float, float]:
    y0, y1 = ax.get_ylim()
    return float(y0), float(y1)


def _span(ylim: tuple[float, float]) -> float:
    return abs(ylim[1] - ylim[0])


def _base_request(
    recording,
    placement: PanelPlacement,
    settings: ViewerSettings,
) -> RenderRequest:
    return RenderRequest(
        placement=placement,
        recordings=[recording],
        labels=[recording.label],
        colors=["#1f77b4"],
        legend_flags=[True],
        channel_index=2,
        channel_name=recording.channel_names[2],
        settings=settings,
    )


def _assert_manual_to_auto(
    recording,
    *,
    placement: PanelPlacement,
    stream: str,
) -> None:
    manual = AxisLimits(enabled=True, minimum=-5.0, maximum=5.0)
    auto = AxisLimits(enabled=False, minimum=-5.0, maximum=5.0)
    if stream == "hp":
        settings_manual = ViewerSettings(hp_ylim=manual)
        settings_auto = ViewerSettings(hp_ylim=auto)
    else:
        settings_manual = ViewerSettings(lp_ylim=manual)
        settings_auto = ViewerSettings(lp_ylim=auto)

    figure = Figure(figsize=(6.0, 3.5), layout="constrained")
    req_m = _base_request(recording, placement, settings_manual)
    req_a = _base_request(recording, placement, settings_auto)

    status = render_panel(figure, req_m)
    if status != "ok":
        raise AssertionError(f"{placement.key}: rendu manuel status={status}")
    ax = figure.axes[0]
    y_manual = _ylim(ax)
    if abs(y_manual[0] + 5.0) > 1e-6 or abs(y_manual[1] - 5.0) > 1e-6:
        raise AssertionError(
            f"{placement.key}: ylim manuel attendu (-5, 5), got {y_manual}"
        )

    # Même fingerprint structure → chemin refresh style (pas de rebuild).
    status = render_panel(figure, req_a)
    if status != "ok":
        raise AssertionError(f"{placement.key}: rendu auto status={status}")
    y_auto = _ylim(ax)
    if _span(y_auto) <= 10.0 + 1e-6:
        raise AssertionError(
            f"{placement.key}: autoscale attendu après manuel→auto, "
            f"ylim resté étroit {y_auto} (span={_span(y_auto):.3f})"
        )


def _assert_preserve_keeps_autoscale(recording) -> None:
    from gui.widgets.panel_canvas import _axis_keeps_rendered_ylim

    panel = "analysis_hp"
    placement = PanelPlacement(panel)
    manual = AxisLimits(enabled=True, minimum=-5.0, maximum=5.0)
    auto = AxisLimits(enabled=False, minimum=-5.0, maximum=5.0)
    prev = _base_request(recording, placement, ViewerSettings(hp_ylim=manual))
    curr = _base_request(recording, placement, ViewerSettings(hp_ylim=auto))

    figure = Figure(figsize=(6.0, 3.5), layout="constrained")
    render_panel(figure, prev)
    render_panel(figure, curr)
    y_auto = _ylim(figure.axes[0])

    keeps = _axis_keeps_rendered_ylim(figure, curr, 0, previous=prev)
    if not keeps:
        raise AssertionError(
            "manuel→auto doit garder le ylim du rendu (pas restore_view)"
        )
    if _span(y_auto) <= 10.0 + 1e-6:
        raise AssertionError(f"autoscale trop étroit après manuel→auto: {y_auto}")

    # Auto inchangé : on doit pouvoir restaurer le zoom interactif.
    keeps_stable = _axis_keeps_rendered_ylim(figure, curr, 0, previous=curr)
    if keeps_stable:
        raise AssertionError("auto inchangé doit autoriser restore_view (zoom/pan)")


def _attach_continuous_source(recording) -> None:
    n_ch = int(recording.n_channels)
    n_samp = 20_000
    rng = np.random.default_rng(13)
    amp = rng.standard_normal((n_ch, n_samp), dtype=np.float32) * 40.0
    recording.source = SimpleNamespace(
        amplifier=amp, highpass=amp * 0.35, lowpass=amp * 0.8
    )


def main() -> int:
    recording = build_synthetic(label="ylim-auto", seed=11)
    _attach_continuous_source(recording)

    for panel, stream in (("analysis_hp", "hp"), ("analysis_lp", "lp")):
        _assert_manual_to_auto(
            recording, placement=PanelPlacement(panel), stream=stream
        )
        print(f"ok  manuel->auto  {panel}")

    for stream in ("hp", "lp"):
        _assert_manual_to_auto(
            recording,
            placement=PanelPlacement("full_recording", stream=stream),
            stream=stream,
        )
        print(f"ok  manuel->auto  full_recording/{stream}")

    _assert_preserve_keeps_autoscale(recording)
    print("ok  preserve_view / keep-rendered logic")
    print("smoke_ylim_autoscale: all passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        raise
