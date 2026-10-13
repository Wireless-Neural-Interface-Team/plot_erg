"""Diagnostic: are montage rows clipped once the panel gets tall?"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PySide6.QtWidgets import QApplication

from gui.services.montage_controller import MontageController
from gui.widgets.panel_canvas import PanelCanvas
from panel_prepare import CurveSeries, PanelSpec
from view_config import PanelPlacement, ViewerSettings


def make_spec(n: int, kinds: tuple[str, ...]) -> PanelSpec:
    x = np.linspace(0, 1, 40)
    y = np.sin(x * 6)
    series: list[CurveSeries] = []
    labels: list[str] = []
    rkinds: list[str] = []
    chans: list[int] = []
    for ch in range(n):
        for kind in kinds:
            series.append(
                CurveSeries(
                    x=x,
                    y=y + ch,
                    color="#2563eb",
                    row=len(labels),
                    label="WIDE" if kind == "raw" else kind,
                )
            )
            labels.append(f"CH{ch} {kind}")
            rkinds.append(kind)
            chans.append(ch)
    return PanelSpec(
        title="Montage",
        xlabel="Temps (s)",
        status="ok",
        layout_kind="montage",
        n_rows=len(labels),
        row_labels=labels,
        row_kinds=rkinds,
        row_channels=chans,
        series=series,
        show_legend=True,
        legend_location="below",
    )


class _Settings:
    def __init__(self, row_h: int, kinds: tuple[str, ...]) -> None:
        self.montage_row_min_height_px = row_h
        self.analysis = None
        self._kinds = kinds

    def resolved_continuous_streams(self) -> tuple[str, ...]:
        return self._kinds


def report(n: int, row_h: int, kinds: tuple[str, ...]) -> None:
    from PySide6.QtWidgets import QScrollArea, QWidget, QVBoxLayout

    settings = _Settings(row_h, kinds)
    height = MontageController.total_height_px(n, settings)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.resize(1000, 700)
    container = QWidget()
    lay = QVBoxLayout(container)
    canvas = PanelCanvas(PanelPlacement("montage_continuous_raw"))
    canvas.setMinimumHeight(height)
    canvas.setMaximumHeight(height)
    lay.addWidget(canvas)
    scroll.setWidget(container)
    scroll.show()
    canvas.plot_host.render_spec(make_spec(n, kinds))
    app = QApplication.instance()
    assert app is not None
    for _ in range(4):
        app.processEvents()
    glw = canvas.plot_host._glw
    plots = canvas.plot_host._plots
    heights = [round(float(p.sceneBoundingRect().height()), 1) for p in plots]
    last = plots[-1].sceneBoundingRect() if plots else None
    legend = canvas.plot_host._legend_item
    legend_bottom = None
    if legend is not None:
        legend_bottom = round(float(legend.sceneBoundingRect().bottom()), 1)
    print(
        f"n={n:3d} kinds={len(kinds)} row={row_h:3d} alloc={height:5d} "
        f"canvas={canvas.height():5d} container={container.height():5d} "
        f"scrollMax={scroll.verticalScrollBar().maximum():5d} glw={glw.height():5d} "
        f"plots={len(plots):3d} h0={heights[0] if heights else None} "
        f"hLast={heights[-1] if heights else None} "
        f"lastBot={None if last is None else round(float(last.bottom()),1)} "
        f"legendBot={legend_bottom} glwH={glw.height()}"
    )
    scroll.close()


def main() -> None:
    QApplication([])
    cases = [
        (4, 52, ("raw",)),
        (10, 52, ("raw",)),
        (10, 120, ("raw",)),
        (10, 200, ("raw",)),
        (8, 80, ("raw", "hp", "lp")),
        (10, 52, ("raw", "hp", "lp")),
        (12, 160, ("raw", "hp")),
        (20, 100, ("raw",)),
        (30, 80, ("raw", "hp")),
    ]
    for n, row_h, kinds in cases:
        report(n, row_h, kinds)


if __name__ == "__main__":
    main()
