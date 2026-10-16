"""Pyqtgraph host for screen panels (Phase 3).

On-screen rendering host (pyqtgraph). Matplotlib
remains the PDF / export drawing path.
"""

from __future__ import annotations

from .constants import INTERACTION_HINT, INTERACTION_HINT_NO_ZOOM
from .host import PlotHost

__all__ = [
    "INTERACTION_HINT",
    "INTERACTION_HINT_NO_ZOOM",
    "PlotHost",
]
