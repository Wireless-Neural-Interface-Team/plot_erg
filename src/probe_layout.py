"""Load probeinterface JSON (MEA) and draw the electrode layout inset for PDF figures."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Circle


@dataclass(frozen=True)
class ProbeLayout:
    """2D geometry (µm), contact ids, and optional device channel index mapping."""

    positions_um: np.ndarray  # (n, 2)
    contact_ids: tuple[str, ...]
    device_channel_indices: tuple[int | None, ...]

    def __post_init__(self) -> None:
        n = int(self.positions_um.shape[0])
        if len(self.contact_ids) != n or len(self.device_channel_indices) != n:
            raise ValueError("positions_um, contact_ids, and device_channel_indices must have the same length.")


# Header band: [left, bottom, width, height] in parent transAxes (MEA inset; Part 1 title on the left).
PROBE_INSET_HEADER_RECT: tuple[float, float, float, float] = (0.20, 0.00, 1, 1)


def _as_int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return None
    if iv < 0:
        return None
    return iv


def _normalize_label(s: str) -> str:
    return "".join(str(s).split()).casefold()


def _is_nc_contact(cid: str) -> bool:
    u = str(cid).strip().upper()
    return not u or u.startswith("NC")


def _contact_name_matches_rhs(cid: str, channel_name: str) -> bool:
    """True only if the recording channel name identifies the same label as ``contact_id`` (full id, e.g. A-005).

    No match on the number alone: ``A-005`` != ``B-005``. Underscore vs hyphen and token splitting (prefix + ``A-005``)
    are accepted when the Intan name contains separators.
    """
    cid_s = str(cid).strip()
    if _is_nc_contact(cid_s):
        return False
    cid_norm = _normalize_label(cid_s.replace("_", "-"))
    ch_full = _normalize_label(str(channel_name).replace("_", "-"))
    if cid_norm == ch_full:
        return True
    for tok in re.split(r"[\s,;|/]+", str(channel_name).strip()):
        if not tok:
            continue
        if _normalize_label(tok.replace("_", "-")) == cid_norm:
            return True
    return False


def load_probe_layout_json(path: Path) -> ProbeLayout:
    """Read a probeinterface JSON (``probes`` key). Raises if invalid."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    probes = raw.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError("Probe JSON: missing or empty 'probes' section.")
    p0 = probes[0]
    if not isinstance(p0, dict):
        raise ValueError("Probe JSON: invalid first element of 'probes'.")
    pos = p0.get("contact_positions")
    if not isinstance(pos, list) or not pos:
        raise ValueError("Probe JSON: 'contact_positions' missing or empty.")
    arr = np.asarray(pos, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Probe JSON: contact_positions must be a list of [x, y].")
    n = int(arr.shape[0])
    ids_raw = p0.get("contact_ids")
    dev_raw = p0.get("device_channel_indices")
    ids: list[str] = []
    if isinstance(ids_raw, list) and len(ids_raw) >= n:
        for i in range(n):
            ids.append(str(ids_raw[i]).strip())
    else:
        ids = [str(i) for i in range(n)]
    devs: list[int | None] = []
    if isinstance(dev_raw, list) and len(dev_raw) >= n:
        for i in range(n):
            devs.append(_as_int_or_none(dev_raw[i]))
    else:
        devs = [None] * n
    return ProbeLayout(
        positions_um=np.ascontiguousarray(arr),
        contact_ids=tuple(ids),
        device_channel_indices=tuple(devs),
    )


def match_contact_index(layout: ProbeLayout, channel_name: str) -> int | None:
    """Return the contact index if the recording channel name matches ``contact_id`` (full name).

    See ``_contact_name_matches_rhs``: no partial match on the numeric suffix alone.
    """
    for i, cid in enumerate(layout.contact_ids):
        if _contact_name_matches_rhs(str(cid), channel_name):
            return i
    return None


def draw_probe_inset_on_axes(
    parent_ax: Any,  # matplotlib.axes.Axes
    layout: ProbeLayout,
    channel_name: str,
    *,
    inset_rect: tuple[float, float, float, float] = PROBE_INSET_HEADER_RECT,
) -> None:
    """Inset on the header: one marker per electrode, ``contact_id`` labels, highlight for the active channel."""
    inset_left, inset_bottom, inset_width, inset_height = inset_rect
    inset_axes = parent_ax.inset_axes(
        [inset_left, inset_bottom, inset_width, inset_height],
        transform=parent_ax.transAxes,
    )
    draw_probe_layout_on_axes(inset_axes, layout, channel_name)


def draw_probe_layout_on_axes(
    target_ax: Any,  # matplotlib.axes.Axes
    layout: ProbeLayout,
    channel_name: str,
) -> None:
    """Draw the MEA layout directly on an axes (no inset), highlight active channel."""
    active_contact_index = match_contact_index(layout, channel_name)
    if active_contact_index is None:
        return
    contact_count = len(layout.contact_ids)
    contact_positions_um = layout.positions_um
    span_x_um = float(np.ptp(contact_positions_um[:, 0])) + 1e-6
    span_y_um = float(np.ptp(contact_positions_um[:, 1])) + 1e-6
    outer_padding_um = 0.06 * max(span_x_um, span_y_um)
    x_min_um = float(np.min(contact_positions_um[:, 0]) - outer_padding_um)
    x_max_um = float(np.max(contact_positions_um[:, 0]) + outer_padding_um)
    # Asymmetric vertical padding: more space above pushes the MEA cloud lower
    # in its own subplot without moving any other graph.
    y_padding_bottom_um = outer_padding_um * 0.55
    y_padding_top_um = outer_padding_um * 2.20
    y_min_um = float(np.min(contact_positions_um[:, 1]) - y_padding_bottom_um)
    y_max_um = float(np.max(contact_positions_um[:, 1]) + y_padding_top_um)

    layout_ax = target_ax
    layout_ax.set_facecolor((1.0, 1.0, 1.0, 0.88))
    for spine in layout_ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("0.45")

    point_size = max(5, min(18, int(780 // max(contact_count, 1))))
    layout_ax.scatter(
        contact_positions_um[:, 0],
        contact_positions_um[:, 1],
        s=point_size,
        c="0.55",
        edgecolors="0.35",
        linewidths=0.2,
        zorder=1,
    )

    active_x_um = float(contact_positions_um[active_contact_index, 0])
    active_y_um = float(contact_positions_um[active_contact_index, 1])
    highlight_radius_um = 0.035 * max(span_x_um, span_y_um)
    layout_ax.add_patch(
        Circle(
            (active_x_um, active_y_um),
            radius=highlight_radius_um,
            facecolor="none",
            edgecolor="crimson",
            linewidth=2.0,
            zorder=4,
        )
    )

    label_font_size = max(2.6, min(4.6, 420.0 / max(float(contact_count) ** 0.48, 1.0)))
    text_stroke_width = max(1.8, min(2.8, label_font_size * 0.55))
    for contact_index in range(contact_count):
        contact_id = str(layout.contact_ids[contact_index]).strip()
        if _is_nc_contact(contact_id):
            continue
        text_artist = layout_ax.annotate(
            contact_id,
            (float(contact_positions_um[contact_index, 0]), float(contact_positions_um[contact_index, 1])),
            ha="center",
            va="center",
            fontsize=label_font_size,
            color="0.12",
            fontweight="normal",
            zorder=5,
        )
        text_artist.set_path_effects([pe.withStroke(linewidth=text_stroke_width, foreground="white")])

    layout_ax.set_xlim(x_min_um, x_max_um)
    layout_ax.set_ylim(y_min_um, y_max_um)
    layout_ax.set_aspect("equal", adjustable="box")
    layout_ax.set_xticks([])
    layout_ax.set_yticks([])
    layout_ax.set_title("MEA layout", fontsize=7, pad=2)
    layout_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
