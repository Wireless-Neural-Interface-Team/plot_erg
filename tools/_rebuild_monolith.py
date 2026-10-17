"""Rebuild a temporary monolith from the package for smoke diagnosis."""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

pkg = Path("src/gui/widgets/plot_host")

methods: dict[str, str] = {}
order: list[str] = []
for p in [
    pkg / "mixins/montage.py",
    pkg / "mixins/scale_bars.py",
    pkg / "mixins/stim_zone.py",
    pkg / "mixins/view_limits.py",
    pkg / "mixins/text_edit.py",
    pkg / "host.py",
]:
    text = p.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name not in (
            "PlotHost",
            "MontageMixin",
            "ScaleBarsMixin",
            "StimZoneMixin",
            "ViewLimitsMixin",
            "TextEditMixin",
        ):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                src = ast.get_source_segment(text, item) or ""
                lines = src.splitlines(True)
                ded = "".join(l[4:] if l.startswith("    ") else l for l in lines)
                if item.name not in methods:
                    order.append(item.name)
                methods[item.name] = ded


def strip_rel_imports(src: str) -> str:
    out = []
    for line in src.splitlines(True):
        s = line.lstrip()
        if s.startswith("from .") or s.startswith("from .."):
            continue
        if s.startswith("from __future__"):
            continue
        if s.startswith("from typing import TYPE_CHECKING"):
            continue
        if s.startswith("if TYPE_CHECKING"):
            continue
        if "from .host import PlotHost" in s:
            continue
        out.append(line)
    return "".join(out)


def keep_defs(src: str) -> str:
    out = []
    for line in src.splitlines(True):
        s = line.lstrip()
        if s.startswith("import ") or s.startswith("from "):
            continue
        out.append(line)
    return "".join(out)


constants = keep_defs(strip_rel_imports((pkg / "constants.py").read_text(encoding="utf-8")))
helpers = keep_defs(strip_rel_imports((pkg / "helpers.py").read_text(encoding="utf-8")))
text_style = keep_defs(strip_rel_imports((pkg / "text_style.py").read_text(encoding="utf-8")))
text_model = keep_defs(strip_rel_imports((pkg / "text_model.py").read_text(encoding="utf-8")))
montage_items = keep_defs(strip_rel_imports((pkg / "montage_items.py").read_text(encoding="utf-8")))
shims = keep_defs(strip_rel_imports((pkg / "shims.py").read_text(encoding="utf-8")))

aliases = """
_montage_shares_x = montage_shares_x
_scale_bar_offset_px = scale_bar_offset_px
_scale_bar_pair_px = scale_bar_pair_px
_scale_bar_time_offset_px = scale_bar_time_offset_px
_uncrop_scale_axis = uncrop_scale_axis
"""

header = '''"""Reconstructed monolith for smoke diagnosis."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontInfo,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QTextCharFormat,
    QTextCursor,
    QTransform,
)
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsSimpleTextItem,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from panel_prepare import (
    CurveSeries,
    HistSeries,
    PanelSpec,
    ScaleBarSpec,
    ScatterSeries,
    montage_vline_span_keys,
    montage_vline_spans,
)
from plot_utils import preview_title_row_layout

'''

class_head = '''
class PlotHost(QWidget):
    """QWidget wrapping GraphicsLayoutWidget with coalesce + view API."""

    viewChanged = Signal()
    textEdited = Signal(object)
    xAutoscaleRequested = Signal(object)

'''

host_order: list[str] = []
host_text = (pkg / "host.py").read_text(encoding="utf-8")
for node in ast.parse(host_text).body:
    if isinstance(node, ast.ClassDef) and node.name == "PlotHost":
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                host_order.append(item.name)

all_order = list(host_order)
for name in order:
    if name not in all_order:
        all_order.append(name)

body_parts: list[str] = []
for name in all_order:
    src = methods[name]
    indented = "".join(("    " + l if l.strip() else l) for l in src.splitlines(True))
    body_parts.append(indented)
    if not indented.endswith("\n"):
        body_parts.append("\n")
    body_parts.append("\n")

mono = (
    header
    + constants
    + "\n"
    + helpers
    + aliases
    + text_style
    + text_model
    + montage_items
    + shims
    + class_head
    + "".join(body_parts)
)

ast.parse(mono)

bak = Path("src/gui/widgets/_plot_host_pkg")
if bak.exists():
    shutil.rmtree(bak)
pkg.rename(bak)
Path("src/gui/widgets/plot_host.py").write_text(mono, encoding="utf-8")
print(f"wrote monolith {len(mono.splitlines())} lines, {len(methods)} methods")
