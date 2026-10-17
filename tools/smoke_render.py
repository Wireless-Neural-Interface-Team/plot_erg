"""Render every catalogued panel against synthetic data and report the status.

Run with:
    si_env\\Scripts\\python.exe tools\\smoke_render.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from matplotlib.figure import Figure  # noqa: E402

from make_synthetic_dataset import build_synthetic  # noqa: E402
from panel_registry import PANEL_CATALOG, RenderRequest, render_panel  # noqa: E402
from view_config import PanelPlacement, ViewerSettings  # noqa: E402


def main() -> int:
    first = build_synthetic(label="control", seed=1)
    second = build_synthetic(label="treated", seed=2)
    recordings = [first, second]
    settings = ViewerSettings()

    placements: list[PanelPlacement] = []
    for info in PANEL_CATALOG:
        if info.is_global:
            placements.append(PanelPlacement(info.key))
        else:
            for section in ("full", "zoom_onset", "zoom_trigger_end"):
                placements.append(PanelPlacement(info.key, section))  # type: ignore[arg-type]

    figure = Figure(figsize=(7.0, 4.5), layout="constrained")
    totals: dict[str, int] = {}
    not_ok: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    slowest: list[tuple[float, str]] = []
    started = time.perf_counter()
    for placement in placements:
        request = RenderRequest(
            placement=placement,
            recordings=recordings,
            labels=[r.label for r in recordings],
            colors=["#1f77b4", "#ff7f0e"],
            legend_flags=[True, True],
            channel_index=2,
            channel_name=first.channel_names[2],
            settings=settings,
        )
        panel_started = time.perf_counter()
        try:
            status = render_panel(figure, request)
        except Exception as exc:  # noqa: BLE001 - reported below
            status = "error"
            failures.append((placement.key, f"{type(exc).__name__}: {exc}"))
        slowest.append((time.perf_counter() - panel_started, placement.key))
        totals[status] = totals.get(status, 0) + 1
        if status != "ok":
            not_ok.append((placement.key, status))

    print(f"placements: {len(placements)}")
    print(f"statuses:   {totals}")
    if not_ok:
        print("not drawn:")
        for key, status in not_ok:
            print(f"  {status:12s} {key}")
    print(f"elapsed:    {time.perf_counter() - started:.2f} s")
    slowest.sort(reverse=True)
    print("slowest panels:")
    for seconds, key in slowest[:6]:
        print(f"  {seconds * 1000:7.1f} ms  {key}")
    if failures:
        print("\nFAILURES:")
        for key, message in failures:
            print(f"  {key}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
