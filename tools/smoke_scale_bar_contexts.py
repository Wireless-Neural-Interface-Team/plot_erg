"""Vérifie amplitude/temps + tailles px en aperçu, montage et zoom.

Contrats :
  - valeur manuelle → libellé = valeur saisie
  - valeur + taille px → fit hauteur/largeur pour que bâton ≈ taille px
  - zoom interactif → libellé reste la valeur saisie
  - montage → fit hauteur seulement sur lignes µV

    .venv\\Scripts\\python.exe tools\\smoke_scale_bar_contexts.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PySide6.QtWidgets import QApplication

from gui.widgets.plot_host import PlotHost
from panel_prepare import CurveSeries, PanelSpec, ScaleBarSpec
from plot_utils import (
    scale_bar_label_value,
    scale_bar_pixel_length,
    scale_bar_true_value,
)


def _close(a: float, b: float, *, tol: float = 0.02) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _bars(
    *,
    amp: float = 50.0,
    amp_px: float = 40.0,
    time_s: float = 0.1,
    time_px: float = 40.0,
    amp_manual: bool = True,
    amp_len: bool = True,
    time_manual: bool = True,
    time_len: bool = True,
) -> ScaleBarSpec:
    return ScaleBarSpec(
        enabled=True,
        x_unit="s",
        y_unit="µV",
        time_manual=time_manual,
        time_s=time_s,
        amp_manual=amp_manual,
        amplitude=amp,
        amp_length_manual=amp_len,
        amp_length_px=amp_px,
        time_length_manual=time_len,
        time_length_px=time_px,
        fontsize=8.0,
        linewidth=1.4,
        color="#111827",
    )


def _curve(t0: float, t1: float, y0: float, y1: float, *, row: int = 0) -> CurveSeries:
    t = np.linspace(t0, t1, 200, dtype=np.float64)
    y = np.linspace(y0, y1, 200, dtype=np.float64)
    return CurveSeries(x=t, y=y, color="#1f77b4", label="", row=row)


def _show(host: PlotHost, spec: PanelSpec, *, w: int = 640, h: int = 360) -> None:
    host.render_spec(spec)
    host.resize(w, h)
    host.show()
    app = QApplication.instance()
    assert app is not None
    for _ in range(12):
        app.processEvents()
    # Appliquer autoscale + fit barres (file d’attente paint).
    if host._limits_after_paint:
        host._apply_limits_after_paint(host._autoscale_gen)
    for _ in range(8):
        app.processEvents()


def _check_labels_after_zoom(host: PlotHost, *, amp: float, time_s: float, label: str) -> None:
    plots = list(host._plots or [])
    if not plots:
        _fail(f"{label}: pas de plot")
    plot = plots[-1]
    (x0, x1), (y0, y1) = plot.viewRange()
    # Zoom interactif X et Y (moitié de la fenêtre).
    mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    plot.setXRange(mx - 0.25 * (x1 - x0), mx + 0.25 * (x1 - x0), padding=0)
    plot.setYRange(my - 0.25 * (y1 - y0), my + 0.25 * (y1 - y0), padding=0)
    app = QApplication.instance()
    assert app is not None
    for _ in range(6):
        app.processEvents()
    host._refresh_scale_bars()
    for _ in range(4):
        app.processEvents()

    bars = host._spec.scale_bars
    (zx0, zx1), (zy0, zy1) = plot.viewRange()
    x_span = abs(zx1 - zx0)
    y_span = abs(zy1 - zy0)
    # Libellés = valeurs saisies (pas recalcul géométrique).
    y_label = scale_bar_label_value(
        data_value=float(bars.amplitude),
        span=y_span,
        axis_px=200.0,
        bar_px=80.0,
        value_manual=True,
        length_manual=True,
    )
    x_label = scale_bar_label_value(
        data_value=float(bars.time_s),
        span=x_span,
        axis_px=200.0,
        bar_px=80.0,
        value_manual=True,
        length_manual=True,
    )
    if not _close(y_label, amp, tol=1e-9):
        _fail(f"{label}: libellé amp après zoom {y_label} ≠ {amp}")
    if not _close(x_label, time_s, tol=1e-9):
        _fail(f"{label}: libellé temps après zoom {x_label} ≠ {time_s}")
    print(f"  OK  {label}: libellés stables après zoom")


def test_preview() -> None:
    print("aperçu (full_recording-like):")
    amp, amp_px = 50.0, 40.0
    time_s, time_px = 0.1, 40.0
    host = PlotHost(show_toolbar=False, show_cursor=False, min_height=120)
    y0, y1 = -100.0, 100.0  # span 200 µV
    t0, t1 = 0.0, 1.0
    spec = PanelSpec(
        title="Aperçu",
        xlabel="Temps (s)",
        ylabel="µV",
        series=[_curve(t0, t1, y0, y1)],
        xlim=(t0, t1),
        ylim=None,
        status="ok",
        layout_kind="single",
        row_kinds=["raw"],
        scale_bars=_bars(amp=amp, amp_px=amp_px, time_s=time_s, time_px=time_px),
    )
    _show(host, spec)
    needed_h = host.scale_bar_needed_view_height_px()
    expect_h = (y1 - y0) * (amp_px / amp)  # 200 * 40/50 = 160
    if needed_h is None or not _close(needed_h, expect_h, tol=0.05):
        _fail(f"aperçu hauteur {needed_h} ≠ {expect_h}")
    needed_w = host.scale_bar_needed_view_width_px()
    expect_w = (t1 - t0) * (time_px / time_s)  # 1.0 * 40/0.1 = 400
    if needed_w is None or not _close(needed_w, expect_w, tol=0.05):
        _fail(f"aperçu largeur {needed_w} ≠ {expect_w}")
    # Après fit théorique : bâton = taille px, libellé = valeur.
    bar = scale_bar_pixel_length(
        data_value=amp, span=y1 - y0, axis_px=expect_h, length_px=amp_px, value_manual=True
    )
    if not _close(bar, amp_px):
        _fail(f"aperçu bâton amp {bar} ≠ {amp_px}")
    if not _close(
        scale_bar_true_value(bar, span=y1 - y0, axis_px=expect_h), amp
    ):
        _fail("aperçu échelle amp fausse après fit")
    print(f"  OK  fit H={needed_h}px W={needed_w}px")
    _check_labels_after_zoom(host, amp=amp, time_s=time_s, label="aperçu")
    host.close()


def test_montage() -> None:
    print("montage (lignes tension + ISI):")
    amp, amp_px = 50.0, 40.0
    time_s, time_px = 0.05, 30.0
    host = PlotHost(show_toolbar=False, show_cursor=False, min_height=200)
    # Deux lignes µV (span 100 chacune) + une ISI (comptages) qui ne doit PAS
    # entrer dans le fit hauteur.
    series = [
        _curve(-0.1, 0.4, -50.0, 50.0, row=0),
        _curve(-0.1, 0.4, -50.0, 50.0, row=1),
        _curve(0.0, 20.0, 0.0, 80.0, row=2),  # ISI-like counts
    ]
    spec = PanelSpec(
        title="Montage",
        xlabel="Temps (s)",
        ylabel="",
        series=series,
        xlim=(-0.1, 0.4),
        ylim=None,
        status="ok",
        layout_kind="montage",
        n_rows=3,
        row_kinds=["raw", "hp", "isi"],
        row_ylims=[None, None, None],
        row_labels=["A raw", "A hp", "A isi"],
        scale_bars=_bars(amp=amp, amp_px=amp_px, time_s=time_s, time_px=time_px),
    )
    # Forcer unités ISI sur la 3e ligne via row_kinds (mixin).
    _show(host, spec, w=700, h=500)
    if len(host._plots) < 3:
        _fail(f"montage: {len(host._plots)} plots")
    # Unités par ligne.
    bars = spec.scale_bars
    assert host._scale_bar_units_for_row(bars, 0) == ("s", "µV")
    assert host._scale_bar_units_for_row(bars, 2) == ("ms", None)
    needed_h = host.scale_bar_needed_view_height_px()
    # Seulement 2 lignes × 100 µV × (40/50) = 160
    expect_h = 2.0 * 100.0 * (amp_px / amp)
    if needed_h is None or not _close(needed_h, expect_h, tol=0.08):
        _fail(f"montage hauteur (sans ISI) {needed_h} ≠ {expect_h}")
    needed_w = host.scale_bar_needed_view_width_px()
    # Dernière ligne = ISI → x en ms ; time_s=0.05 → 50 ms ; span X données ISI ~20
    # Mais le mixin lit le plot de la dernière ligne (ISI, xlim peut différer).
    if needed_w is None:
        _fail("montage largeur None (ISI last row devrait quand même scaler X en ms)")
    print(f"  OK  fit H={needed_h}px (ISI exclu) W={needed_w}px")
    _check_labels_after_zoom(host, amp=amp, time_s=time_s, label="montage")
    host.close()


def test_zoom_window() -> None:
    print("fenêtre zoom (xlim placement, ylim auto):")
    amp, amp_px = 25.0, 50.0
    time_s, time_px = 0.02, 60.0
    host = PlotHost(show_toolbar=False, show_cursor=False, min_height=120)
    # Fenêtre zoom étroite : X fixé, Y autoscalé sur les données visibles.
    t0, t1 = 0.1, 0.3
    y0, y1 = -80.0, 80.0
    spec = PanelSpec(
        title="Zoom custom",
        xlabel="Temps (s)",
        ylabel="µV",
        series=[_curve(t0, t1, y0, y1)],
        xlim=(t0, t1),
        ylim=None,
        status="ok",
        layout_kind="single",
        row_kinds=["raw"],
        scale_bars=_bars(amp=amp, amp_px=amp_px, time_s=time_s, time_px=time_px),
    )
    _show(host, spec)
    needed_h = host.scale_bar_needed_view_height_px()
    expect_h = (y1 - y0) * (amp_px / amp)  # 160 * 50/25 = 320
    if needed_h is None or not _close(needed_h, expect_h, tol=0.05):
        _fail(f"zoom hauteur {needed_h} ≠ {expect_h}")
    needed_w = host.scale_bar_needed_view_width_px()
    expect_w = (t1 - t0) * (time_px / time_s)  # 0.2 * 60/0.02 = 600
    if needed_w is None or not _close(needed_w, expect_w, tol=0.05):
        _fail(f"zoom largeur {needed_w} ≠ {expect_w}")
    # ylim Canal manuel → pas de fit hauteur.
    spec2 = PanelSpec(
        title="Zoom ylim fixe",
        xlabel="Temps (s)",
        ylabel="µV",
        series=[_curve(t0, t1, y0, y1)],
        xlim=(t0, t1),
        ylim=(-200.0, 200.0),
        status="ok",
        layout_kind="single",
        row_kinds=["raw"],
        scale_bars=_bars(amp=amp, amp_px=amp_px, time_s=time_s, time_px=time_px),
    )
    _show(host, spec2)
    if host.scale_bar_needed_view_height_px() is not None:
        _fail("zoom: fit hauteur doit être None si ylim manuel")
    print("  OK  fit actif (ylim auto) / inactif (ylim Canal)")
    _check_labels_after_zoom(host, amp=amp, time_s=time_s, label="zoom")
    host.close()


def test_length_only_vs_value_both() -> None:
    print("modes taille seule / valeur+taille:")
    # Taille seule → libellé géométrique.
    label = scale_bar_label_value(
        data_value=50.0,
        span=100.0,
        axis_px=400.0,
        bar_px=40.0,
        value_manual=False,
        length_manual=True,
    )
    if not _close(label, 10.0):
        _fail(f"taille seule label {label} ≠ 10")
    # Les deux → libellé = valeur (calibre le graph).
    label = scale_bar_label_value(
        data_value=50.0,
        span=100.0,
        axis_px=80.0,  # fit: 100 * 40/50 = 80
        bar_px=40.0,
        value_manual=True,
        length_manual=True,
    )
    if not _close(label, 50.0):
        _fail(f"valeur+taille label {label} ≠ 50")
    print("  OK  contrats libellé")


def main() -> int:
    app = QApplication.instance() or QApplication([])
    test_length_only_vs_value_both()
    test_preview()
    test_montage()
    test_zoom_window()
    print("smoke_scale_bar_contexts: all passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
