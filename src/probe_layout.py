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


def _cluster_count(values: np.ndarray) -> tuple[int, float]:
    """Number of distinct columns/rows and typical pitch (same units as ``values``)."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return 1, 1.0
    uniq = np.unique(np.round(vals, 8))
    if uniq.size <= 1:
        return 1, 1.0
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    pitch = float(np.median(diffs)) if diffs.size else 1.0
    count = 1
    anchor = float(uniq[0])
    for raw in uniq[1:]:
        value = float(raw)
        if value - anchor > pitch * 0.4:
            count += 1
            anchor = value
    return max(1, count), max(pitch, 1e-6)


def mea_panel_size_in(
    layout: ProbeLayout,
    *,
    max_width_in: float,
    max_height_in: float = 10.5,
    font_min: float = 8.0,
    font_max: float = 14.0,
    font_scale: float = 120.0,
) -> tuple[float, float]:
    """Width/height in inches so contact labels fit (equal aspect when possible)."""
    positions = np.asarray(layout.positions_um, dtype=np.float64)
    n_cols, pitch_x = _cluster_count(positions[:, 0])
    n_rows, pitch_y = _cluster_count(positions[:, 1])
    labels = [str(cid).strip() for cid in layout.contact_ids if not _is_nc_contact(str(cid))]
    max_chars = max((len(lab) for lab in labels), default=5)
    contact_count = max(len(layout.contact_ids), 1)
    font = max(float(font_min), min(float(font_max), float(font_scale) / max(float(contact_count) ** 0.48, 1.0)))
    char_w_in = font * 0.58 / 72.0
    cell_w_in = max(char_w_in * (max_chars + 0.9), font * 1.15 / 72.0)
    cell_h_in = font * 1.65 / 72.0
    min_width_in = n_cols * cell_w_in + 0.28
    min_height_in = n_rows * cell_h_in + 0.28
    span_x = max(float(np.ptp(positions[:, 0])), pitch_x * max(n_cols - 1, 1), 1e-6)
    span_y = max(float(np.ptp(positions[:, 1])), pitch_y * max(n_rows - 1, 1), 1e-6)
    aspect = span_x / span_y

    width_in = min(float(max_width_in), max(min_width_in, min_height_in * aspect))
    height_in = width_in / aspect
    if height_in < min_height_in:
        height_in = min_height_in
        width_in = min(float(max_width_in), max(min_width_in, height_in * aspect))
    if height_in > float(max_height_in):
        height_in = float(max_height_in)
        width_in = min(float(max_width_in), max(min_width_in, height_in * aspect))
    width_in = min(float(max_width_in), max(min_width_in, width_in))
    height_in = max(min_height_in, min(float(max_height_in), height_in))
    return float(width_in), float(height_in)


def draw_probe_layout_on_axes(
    target_ax: Any,  # matplotlib.axes.Axes
    layout: ProbeLayout,
    channel_name: str,
    *,
    set_mea_title: bool = True,
    title_fontsize: float = 26,
    contact_label_font_min: float = 10.0,
    contact_label_font_max: float = 18.0,
    contact_label_font_scale: float = 1620.0,
) -> None:
    """Draw the MEA layout directly on an axes (no inset), highlight active channel."""
    active_contact_index = match_contact_index(layout, channel_name)
    if active_contact_index is None:
        return
    contact_count = len(layout.contact_ids)
    contact_positions_um = layout.positions_um
    n_cols, pitch_x = _cluster_count(contact_positions_um[:, 0])
    n_rows, pitch_y = _cluster_count(contact_positions_um[:, 1])
    span_x_um = max(float(np.ptp(contact_positions_um[:, 0])), pitch_x, 1e-6)
    span_y_um = max(float(np.ptp(contact_positions_um[:, 1])), pitch_y, 1e-6)
    pad_um = 0.55 * max(pitch_x, pitch_y)
    x_min_um = float(np.min(contact_positions_um[:, 0]) - pad_um)
    x_max_um = float(np.max(contact_positions_um[:, 0]) + pad_um)
    y_min_um = float(np.min(contact_positions_um[:, 1]) - pad_um)
    y_max_um = float(np.max(contact_positions_um[:, 1]) + pad_um)

    layout_ax = target_ax
    layout_ax.set_facecolor((1.0, 1.0, 1.0, 0.88))
    for spine in layout_ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("0.45")

    layout_ax.set_xlim(x_min_um, x_max_um)
    layout_ax.set_ylim(y_min_um, y_max_um)
    data_aspect = (x_max_um - x_min_um) / max(y_max_um - y_min_um, 1e-6)
    fig = layout_ax.figure
    pos = layout_ax.get_position()
    box_aspect = (pos.width * float(fig.get_figwidth())) / max(pos.height * float(fig.get_figheight()), 1e-6)
    # Keep physical proportions only when the allocated box is close; otherwise fill the box
    # so labels stay readable instead of collapsing into a thin strip.
    if 0.72 <= (box_aspect / data_aspect) <= 1.35:
        layout_ax.set_aspect("equal", adjustable="box")
    else:
        layout_ax.set_aspect("auto")

    axes_w_in = max(pos.width * float(fig.get_figwidth()), 0.4)
    axes_h_in = max(pos.height * float(fig.get_figheight()), 0.4)
    labels = [str(cid).strip() for cid in layout.contact_ids if not _is_nc_contact(str(cid))]
    max_chars = max((len(lab) for lab in labels), default=5)
    fs_from_width = (axes_w_in / max(n_cols, 1)) * 72.0 / (max_chars * 0.62)
    fs_from_height = (axes_h_in / max(n_rows, 1)) * 72.0 / 1.70
    fs_from_count = float(contact_label_font_scale) / max(float(contact_count) ** 0.48, 1.0)
    label_font_size = max(
        float(contact_label_font_min),
        min(float(contact_label_font_max), fs_from_width, fs_from_height, fs_from_count),
    )
    text_stroke_width = max(1.6, min(3.0, label_font_size * 0.40))

    point_size = max(8.0, min(36.0, (axes_w_in / max(n_cols, 1)) * 72.0 * 0.45))
    layout_ax.scatter(
        contact_positions_um[:, 0],
        contact_positions_um[:, 1],
        s=point_size,
        c="0.78",
        edgecolors="0.35",
        linewidths=0.4,
        zorder=1,
    )

    active_x_um = float(contact_positions_um[active_contact_index, 0])
    active_y_um = float(contact_positions_um[active_contact_index, 1])
    highlight_radius_um = 0.42 * min(pitch_x, pitch_y)
    layout_ax.add_patch(
        Circle(
            (active_x_um, active_y_um),
            radius=highlight_radius_um,
            facecolor="none",
            edgecolor="crimson",
            linewidth=1.6,
            zorder=4,
        )
    )

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
            color="0.10",
            fontweight="normal",
            zorder=5,
            clip_on=True,
        )
        text_artist.set_path_effects([pe.withStroke(linewidth=text_stroke_width, foreground="white")])

    layout_ax.set_xticks([])
    layout_ax.set_yticks([])
    if set_mea_title:
        layout_ax.set_title("MEA layout", fontsize=title_fontsize, pad=2)
    layout_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
