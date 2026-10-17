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
from gui.widgets.plot_host import PlotHost
from panel_prepare import CurveSeries, PanelSpec
from view_config import PanelPlacement, ViewerSettings


def make_spec(n: int, kinds: tuple[str, ...], row_h: int = 0) -> PanelSpec:
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
        row_min_height_px=int(row_h),
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
    canvas.plot_host.render_spec(make_spec(n, kinds, row_h))
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
    needed = canvas.plot_host.montage_needed_glw_height_px()
    if needed is not None and int(needed) > glw.height() + 1:
        grow = int(needed) - glw.height()
        target = canvas.height() + grow
        canvas.setMinimumHeight(target)
        canvas.setMaximumHeight(target)
        for _ in range(4):
            app.processEvents()
        heights = [round(float(p.sceneBoundingRect().height()), 1) for p in plots]
        legend = canvas.plot_host._legend_item
        legend_bottom = (
            round(float(legend.sceneBoundingRect().bottom()), 1) if legend is not None else None
        )
        glw = canvas.plot_host._glw
    needed_after = canvas.plot_host.montage_needed_glw_height_px()
    clipped = legend_bottom is not None and legend_bottom > glw.height() + 1
    short_rows = [h for h in heights if h + 1 < row_h]
    print(
        f"n={n:3d} kinds={len(kinds)} row={row_h:3d} alloc={height:5d} "
        f"glw={glw.height():5d} needed={needed} plots={len(plots):3d} "
        f"h0={heights[0] if heights else None} hLast={heights[-1] if heights else None} "
        f"legendBot={legend_bottom} clipped={clipped} short_rows={len(short_rows)} "
        f"needed2={needed_after}"
    )
    scroll.close()


def paint_check(height: int) -> None:
    """True if the bottom of a tall GraphicsView actually paints."""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QScrollArea

    host = PlotHost()
    n = max(2, height // 80)
    host.resize(900, height)
    host.show()
    host.render_spec(make_spec(n, ("raw",)))
    app = QApplication.instance()
    assert app is not None
    for _ in range(3):
        app.processEvents()
    image = host.grab().toImage()
    # Sample a pixel inside the last plot.
    last = host._plots[-1].sceneBoundingRect()
    x = int(host.width() * 0.55)
    y = int(min(image.height() - 2, max(0, last.center().y())))
    color = QColor(image.pixel(x, y))
    # Background is near-white; a curve or axes should not be pure background.
    bg = QColor(image.pixel(x, 4))
    painted = color.rgb() != bg.rgb()
    # Also sample the very bottom strip.
    bottom = QColor(image.pixel(x, max(0, image.height() - 3)))
    print(
        f"paint h={height:5d} img={image.width()}x{image.height()} "
        f"lastCenterY={y} sample={color.name()} bg={bg.name()} "
        f"bottom={bottom.name()} curveish={painted} plots={len(host._plots)}"
    )
    host.close()


def main() -> None:
    QApplication([])
    cases = [
        (4, 52, ("raw",)),
        (10, 52, ("raw",)),
        (10, 200, ("raw",)),
        (10, 52, ("raw", "hp", "lp")),
        (8, 160, ("raw", "hp")),
    ]
    for n, row_h, kinds in cases:
        report(n, row_h, kinds)


if __name__ == "__main__":
    main()
