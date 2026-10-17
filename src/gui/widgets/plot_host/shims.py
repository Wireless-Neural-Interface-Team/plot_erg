"""Shims."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyqtgraph as pg

if TYPE_CHECKING:
    from .host import PlotHost

class _AxisShim:
    """Minimal matplotlib-like axis for capture/restore and range-bar attach."""

    def __init__(self, plot_item: pg.PlotItem) -> None:
        self._plot = plot_item

    def get_xlim(self) -> tuple[float, float]:
        x0, x1 = self._plot.viewRange()[0]
        return float(x0), float(x1)

    def get_ylim(self) -> tuple[float, float]:
        y0, y1 = self._plot.viewRange()[1]
        return float(y0), float(y1)

    def set_xlim(self, a: float, b: float) -> None:
        self._plot.setXRange(float(a), float(b), padding=0)

    def set_ylim(self, a: float, b: float) -> None:
        self._plot.setYRange(float(a), float(b), padding=0)

class _FigureShim:
    """Compatibility shim so callers can read ``figure.axes`` / montage attrs."""

    def __init__(self, host: "PlotHost") -> None:
        self._host = host
        self.axes: list[_AxisShim] = []
        self._erg_montage_state: dict[str, Any] | None = None
        self._erg_montage_row_kinds: list[str] | None = None
        self._erg_montage_row_streams: list[str] | None = None
        self._erg_montage_row_channels: list[int] | None = None
        self._erg_status: str = "pending"

    def clear(self) -> None:
        self.axes.clear()
        self._erg_montage_state = None
        self._erg_montage_row_kinds = None
        self._erg_montage_row_streams = None
        self._erg_montage_row_channels = None
