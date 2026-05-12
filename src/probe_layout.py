"""Chargement JSON probeinterface (MEA) et encart position des électrodes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


@dataclass(frozen=True)
class ProbeLayout:
    """Géométrie 2D (µm) + identifiants + mapping vers indices d'enregistrement."""

    positions_um: np.ndarray  # (n, 2)
    contact_ids: tuple[str, ...]
    device_channel_indices: tuple[int | None, ...]

    def __post_init__(self) -> None:
        n = int(self.positions_um.shape[0])
        if len(self.contact_ids) != n or len(self.device_channel_indices) != n:
            raise ValueError("positions_um, contact_ids et device_channel_indices doivent avoir la même longueur.")


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


def load_probe_layout_json(path: Path) -> ProbeLayout:
    """Lit un JSON probeinterface (clé ``probes``). Lève une exception si invalide."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    probes = raw.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError("JSON probe : section 'probes' absente ou vide.")
    p0 = probes[0]
    if not isinstance(p0, dict):
        raise ValueError("JSON probe : premier élément de 'probes' invalide.")
    pos = p0.get("contact_positions")
    if not isinstance(pos, list) or not pos:
        raise ValueError("JSON probe : 'contact_positions' manquant ou vide.")
    arr = np.asarray(pos, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("JSON probe : contact_positions doit être une liste de [x, y].")
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


def match_contact_index(layout: ProbeLayout, recording_channel_index: int, channel_name: str) -> int | None:
    """Trouve l'index de contact pour le canal d'enregistrement (0-based) ou le nom de canal RHS."""
    ch = int(recording_channel_index)
    for i, d in enumerate(layout.device_channel_indices):
        if d is not None and d == ch:
            return i
    name_n = _normalize_label(channel_name)
    m = re.fullmatch(r"ch0*(\d+)", name_n)
    if m:
        k = int(m.group(1))
        for i, d in enumerate(layout.device_channel_indices):
            if d is not None and d == k:
                return i
    for i, cid in enumerate(layout.contact_ids):
        cid_s = str(cid).strip()
        up = cid_s.upper()
        if not cid_s or up in ("NC", "NC1", "NC2", "NC3", "NC4", "NC5"):
            continue
        if _normalize_label(cid_s) == name_n:
            return i
        cdn = _normalize_label(cid_s.replace("_", "-"))
        if cdn == name_n:
            return i
    return None


def draw_probe_inset_on_axes(
    parent_ax: Any,  # matplotlib.axes.Axes
    layout: ProbeLayout,
    recording_channel_index: int,
    channel_name: str,
) -> None:
    """Encart haut-droite du parent : carte + étiquettes ; cercle sur l'électrode du canal si trouvée."""
    idx = match_contact_index(layout, recording_channel_index, channel_name)
    if idx is None:
        return
    n = len(layout.contact_ids)
    xy = layout.positions_um
    span_x = float(np.ptp(xy[:, 0])) + 1e-6
    span_y = float(np.ptp(xy[:, 1])) + 1e-6
    pad = 0.06 * max(span_x, span_y)
    x0, x1 = float(np.min(xy[:, 0]) - pad), float(np.max(xy[:, 0]) + pad)
    y0, y1 = float(np.min(xy[:, 1]) - pad), float(np.max(xy[:, 1]) + pad)

    ax_in = inset_axes(
        parent_ax,
        width="24%",
        height="14%",
        loc="upper right",
        bbox_to_anchor=(0.99, 0.98),
        bbox_transform=parent_ax.transAxes,
        borderpad=0.02,
    )
    ax_in.set_facecolor((1.0, 1.0, 1.0, 0.88))
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("0.45")
    fs = max(3.0, min(5.5, 320.0 / max(n, 1)))
    ax_in.scatter(xy[:, 0], xy[:, 1], s=max(6, 180 // max(n, 1)), c="0.55", edgecolors="0.35", linewidths=0.2, zorder=1)
    for i in range(n):
        ax_in.text(
            float(xy[i, 0]),
            float(xy[i, 1]),
            layout.contact_ids[i],
            ha="center",
            va="center",
            fontsize=fs,
            color="0.1",
            clip_on=True,
            zorder=2,
        )
    hx, hy = float(xy[idx, 0]), float(xy[idx, 1])
    r = 0.035 * max(span_x, span_y)
    ax_in.add_patch(
        Circle(
            (hx, hy),
            radius=r,
            facecolor="none",
            edgecolor="crimson",
            linewidth=2.0,
            zorder=4,
        )
    )
    ax_in.set_xlim(x0, x1)
    ax_in.set_ylim(y0, y1)
    ax_in.set_aspect("equal", adjustable="box")
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_title("MEA layout", fontsize=6.5, pad=1)
    ax_in.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
