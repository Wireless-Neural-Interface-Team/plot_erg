"""Modèle unique des barres d’échelle flottantes (écran + PDF).

Contrat par axe (Y = amplitude, X = temps), à partir de ``span`` (étendue
visible) et ``axis_px`` (taille pixel de l’axe) :

+------------------+-------------------+------------------+------------------+
| Mode             | value / length_px | bâton            | libellé          |
+------------------+-------------------+------------------+------------------+
| Auto             | None / None       | ∝ nice            | nice             |
| Valeur seule     | V / None          | ∝ V               | V                |
| Taille seule     | None / L          | L (clamp axe)     | L/axis × span    |
| Les deux         | V / L             | L (clamp) + fit   | V                |
+------------------+-------------------+------------------+------------------+

Invariant géométrique : ``true = bar_px / axis_px × span``.
En mode « les deux », le bâton fait toujours ``L`` px (donc la taille
temps/amp. réagit tout de suite) et le libellé reste ``V``. Le fit widget
cale l’axe pour que ``true == V`` dès que c’est possible ; sinon le bâton
reste à ``L`` (annotation fixe) plutôt que d’ignorer la taille saisie.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AxisBar:
    """Résultat d’une résolution d’échelle pour un axe."""

    value: float
    """Longueur cible en unités de données (nice ou manuelle)."""

    bar_px: float
    """Longueur du bâton en pixels."""

    label: float
    """Nombre affiché à côté du bâton."""

    needed_axis_px: float | None
    """Si mode « les deux » : hauteur/largeur d’axe pour que bâton ≈ length_px."""

    data_length: float
    """Longueur en unités de données correspondant à ``bar_px`` (dessin MPL)."""


def nice_scale_length(span: float, *, target_frac: float = 0.18) -> float:
    """Longueur « ronde » (~1 / 2 / 5 × 10ⁿ) proche de ``target_frac`` du span."""
    span = abs(float(span))
    if not math.isfinite(span) or span <= 0:
        return 1.0
    target = span * float(target_frac)
    if target <= 0:
        return span * target_frac
    exponent = math.floor(math.log10(target))
    fraction = target / (10.0**exponent)
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if fraction <= candidate * 1.05:
            return candidate * (10.0**exponent)
    return 10.0 * (10.0**exponent)


def _nice_bar_length(span: float) -> float:
    """Longueur ronde bornée à ~45 % du span visible."""
    size = nice_scale_length(span)
    while size > span * 0.45 and size > 0:
        size *= 0.5
    return size


def _finite_positive(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not (out > 0.0) or out != out:
        return None
    return out


def resolve_axis_bar(
    *,
    span: float,
    axis_px: float,
    value: float | None = None,
    length_px: float | None = None,
) -> AxisBar:
    """Résoudre bâton + libellé (+ fit) pour un axe.

    ``value`` / ``length_px`` à ``None`` = non manuel (auto / absent).
    """
    try:
        data_span = abs(float(span))
    except (TypeError, ValueError):
        data_span = 0.0
    try:
        axis = float(axis_px)
    except (TypeError, ValueError):
        axis = 0.0
    if not (data_span > 0.0) or data_span != data_span:
        return AxisBar(value=0.0, bar_px=0.0, label=0.0, needed_axis_px=None, data_length=0.0)
    if not (axis > 0.0) or axis != axis:
        return AxisBar(value=0.0, bar_px=0.0, label=0.0, needed_axis_px=None, data_length=0.0)

    manual_value = _finite_positive(value)
    manual_length = _finite_positive(length_px)
    value_manual = manual_value is not None
    length_manual = manual_length is not None

    # Cible en données : manuelle ou « nice ».
    if value_manual:
        target = float(manual_value)
    else:
        target = _nice_bar_length(data_span)

    if length_manual and not value_manual:
        # Taille seule : bâton fixe, libellé géométrique.
        bar = max(1.0, min(float(manual_length), axis))
        label = (bar / axis) * data_span
        return AxisBar(
            value=target,
            bar_px=bar,
            label=label,
            needed_axis_px=None,
            data_length=label,
        )

    if value_manual and length_manual:
        # Les deux : bâton = L px (fixe), libellé = V ; le widget se cale sur needed_axis.
        needed = data_span * (float(manual_length) / float(manual_value))
        bar_px = max(1.0, float(manual_length))
        data_length = (bar_px / axis) * data_span if axis > 0 else float(target)
        return AxisBar(
            value=target,
            bar_px=bar_px,
            label=target,
            needed_axis_px=needed,
            data_length=data_length,
        )

    # Valeur seule ou auto : bâton proportionnel à la cible.
    bar = max(1.0, min((target / data_span) * axis, axis))
    data_length = (bar / axis) * data_span
    return AxisBar(
        value=target,
        bar_px=bar,
        label=target,
        needed_axis_px=None,
        data_length=data_length,
    )


def major_ticks_for_axis(
    *,
    lo: float,
    hi: float,
    span: float,
    axis_px: float,
    unit: str,
    manual_value: float | None = None,
    value_manual: bool = False,
    length_px: float | None = None,
    length_manual: bool = False,
) -> list[tuple[float, str]]:
    """Graduations majeures (position, libellé) alignées sur ``resolve_axis_bar``."""
    if length_manual and not value_manual and length_px is not None:
        bar = resolve_axis_bar(
            span=span, axis_px=axis_px, value=None, length_px=float(length_px)
        )
    elif value_manual and manual_value is not None and float(manual_value) > 0:
        bar = resolve_axis_bar(
            span=span,
            axis_px=axis_px,
            value=float(manual_value),
            length_px=float(length_px) if length_manual and length_px else None,
        )
    else:
        bar = resolve_axis_bar(span=span, axis_px=axis_px)
    step = float(bar.label)
    if not (step > 0.0) or step != step:
        return []
    if lo > hi:
        lo, hi = hi, lo
    start = math.floor(lo / step) * step
    out: list[tuple[float, str]] = []
    v = start
    guard = 0
    while v <= hi + step * 0.01 and guard < 500:
        if v >= lo - step * 0.01:
            out.append((v, format_scale_label(v, unit)))
        v += step
        guard += 1
    return out


def format_scale_label(value: float, unit: str) -> str:
    """Libellé compact (ex. ``100ms``, ``200µV``)."""
    unit = (unit or "").strip()
    magnitude = abs(float(value))
    if unit in {"s", "sec", "seconds"}:
        if magnitude >= 1.0:
            return f"{magnitude:g}s"
        ms = magnitude * 1000.0
        if ms >= 1.0:
            return f"{ms:g}ms"
        us = ms * 1000.0
        return f"{us:g}µs"
    if unit in {"ms"}:
        return f"{magnitude:g}ms"
    if not unit:
        return f"{magnitude:g}"
    return f"{magnitude:g}{unit}"


def time_value_in_axis_units(time_s: float, x_unit: str | None) -> float:
    """``scale_bar_time_s`` (stocké en s) → unités de l’axe (s ou ms)."""
    try:
        t = float(time_s)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(t) or t <= 0.0:
        return 0.0
    if x_unit == "ms":
        return t * 1000.0
    return t


# --- Unités par kind de ligne / panneau --------------------------------------

_KIND_UNITS: dict[str, tuple[str | None, str | None]] = {
    "overlay": ("ms", "µV"),
    "psth": ("s", "Hz"),
    "trial_rate": (None, "Hz"),
    "isi": ("ms", None),
    "raster": ("s", None),
    "rms": ("s", "µV"),
    "raw": ("s", "µV"),
    "hp": ("s", "µV"),
    "lp": ("s", "µV"),
}

_PANEL_UNITS: dict[str, tuple[str | None, str | None]] = {
    "mea_layout": (None, None),
    "summary_rms_table": (None, None),
    "impedance": (None, "Ω"),
    "summary_impedance": (None, "Ω"),
    "analysis_overlay": ("ms", "µV"),
    "spike_overlay": ("ms", "µV"),
    "analysis_psth": ("s", "Hz"),
    "psth": ("s", "Hz"),
    "first_psth": ("s", "Hz"),
    "second_psth": ("s", "Hz"),
    "analysis_trial_rate": (None, "Hz"),
    "trial_rate": (None, "Hz"),
    "analysis_isi": ("ms", None),
    "isi": ("ms", None),
    "first_isi": ("ms", None),
    "second_isi": ("ms", None),
    "analysis_raster": ("s", None),
    "analysis_raster_channel": ("s", None),
    "raster": ("s", None),
    "analysis_rms": ("s", "µV"),
    "rms": ("s", "µV"),
    "first_rms": ("s", "µV"),
    "second_rms": ("s", "µV"),
    "summary_rms": ("s", "µV"),
}


def units_for_panel(panel: str) -> tuple[str | None, str | None]:
    """Unités (x, y) pour un panneau ; ``None`` = barre omise."""
    key = str(panel or "")
    if key in _PANEL_UNITS:
        return _PANEL_UNITS[key]
    return ("s", "µV")


def units_for_kind(
    kind: str | None,
    *,
    default_x: str | None = "s",
    default_y: str | None = "µV",
) -> tuple[str | None, str | None]:
    """Unités (x, y) pour une ligne de montage (sinon défauts panneau)."""
    key = str(kind or "")
    if key in _KIND_UNITS:
        return _KIND_UNITS[key]
    return (default_x, default_y)


def amp_manual_applies(y_unit: str | None) -> bool:
    """Le spin amplitude (µV) ne s’applique qu’aux axes tension."""
    return y_unit == "µV"


# Spike scope : fenêtre de quelques ms et amplitude des formes d’onde.
# Le gain partagé (Amplitude / Temps des traces) ne le dimensionne pas.
_INDEPENDENT_SCALE_KINDS = frozenset({"overlay"})
_INDEPENDENT_SCALE_PANELS = frozenset({"analysis_overlay", "spike_overlay"})


def uses_shared_manual_scale(
    *, kind: str | None = None, panel: str | None = None
) -> bool:
    """False pour le spike scope : échelle propre, pas le gain des traces."""
    if str(kind or "") in _INDEPENDENT_SCALE_KINDS:
        return False
    if str(panel or "") in _INDEPENDENT_SCALE_PANELS:
        return False
    return True
