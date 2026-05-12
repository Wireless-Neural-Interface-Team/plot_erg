"""Chargement JSON probeinterface (MEA) et encart position des électrodes."""

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
    """Géométrie 2D (µm) + identifiants + mapping vers indices d'enregistrement."""

    positions_um: np.ndarray  # (n, 2)
    contact_ids: tuple[str, ...]
    device_channel_indices: tuple[int | None, ...]

    def __post_init__(self) -> None:
        n = int(self.positions_um.shape[0])
        if len(self.contact_ids) != n or len(self.device_channel_indices) != n:
            raise ValueError("positions_um, contact_ids et device_channel_indices doivent avoir la même longueur.")


# Bandeau d'entête : [left, bottom, width, height] en transAxes (MEA à droite, titre « Part 1 » à gauche).
PROBE_INSET_HEADER_RECT: tuple[float, float, float, float] = (0.17, 0.01, 0.82, 0.98)


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
    """True uniquement si le nom de canal RHS désigne le même identifiant que ``contact_id`` (nom entier, ex. A-005).

    Pas de correspondance sur le seul numéro : ``A-005`` ≠ ``B-005``. On accepte ``_`` ↔ ``-`` et des jetons si le
    nom Intan contient des séparateurs (ex. préfixe + ``A-005``).
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


def match_contact_index(layout: ProbeLayout, channel_name: str) -> int | None:
    """Trouve l'index de contact si le nom de canal RHS correspond **exactement** au ``contact_id`` (nom entier).

    Voir ``_contact_name_matches_rhs`` : pas de correspondance partielle sur le numéro seul.
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
    """Encart sur le parent (bandeau) : une pastille par électrode, ``contact_id`` sur chaque point, surbrillance du canal RHS."""
    idx = match_contact_index(layout, channel_name)
    if idx is None:
        return
    n = len(layout.contact_ids)
    xy = layout.positions_um
    span_x = float(np.ptp(xy[:, 0])) + 1e-6
    span_y = float(np.ptp(xy[:, 1])) + 1e-6
    pad = 0.06 * max(span_x, span_y)
    x0, x1 = float(np.min(xy[:, 0]) - pad), float(np.max(xy[:, 0]) + pad)
    y0, y1 = float(np.min(xy[:, 1]) - pad), float(np.max(xy[:, 1]) + pad)

    left, bottom, width, height = inset_rect
    ax_in = parent_ax.inset_axes(
        [left, bottom, width, height],
        transform=parent_ax.transAxes,
    )
    ax_in.set_facecolor((1.0, 1.0, 1.0, 0.88))
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("0.45")

    pt_size = max(5, min(18, int(780 // max(n, 1))))
    ax_in.scatter(
        xy[:, 0],
        xy[:, 1],
        s=pt_size,
        c="0.55",
        edgecolors="0.35",
        linewidths=0.2,
        zorder=1,
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

    label_fs = max(2.6, min(4.6, 420.0 / max(float(n) ** 0.48, 1.0)))
    stroke_w = max(1.8, min(2.8, label_fs * 0.55))
    for i in range(n):
        cid_s = str(layout.contact_ids[i]).strip()
        if _is_nc_contact(cid_s):
            continue
        t = ax_in.annotate(
            cid_s,
            (float(xy[i, 0]), float(xy[i, 1])),
            ha="center",
            va="center",
            fontsize=label_fs,
            color="0.12",
            fontweight="normal",
            zorder=5,
        )
        t.set_path_effects([pe.withStroke(linewidth=stroke_w, foreground="white")])

    ax_in.set_xlim(x0, x1)
    ax_in.set_ylim(y0, y1)
    ax_in.set_aspect("equal", adjustable="box")
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_title("MEA layout", fontsize=7, pad=2)
    ax_in.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
