"""Smoke test RHS Studio : fenêtre complète, tous les blocs calculés, captures PNG.

python tools/smoke_rhs_studio.py [fichier.rhs] [--out dossier]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rhs_studio.main_window import MainWindow  # noqa: E402


def wait_all(app, win, timeout=180.0) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        app.processEvents()
        busy = win._pending or any(w._busy for w in win.widgets.values())
        if not busy and win.engine.pool.activeThreadCount() == 0:
            app.processEvents()
            return time.perf_counter() - t0
        time.sleep(0.02)
    raise TimeoutError("calculs non terminés")


def _drag(app, view, start, end) -> None:
    """Survol puis glisser, avec de vrais QMouseEvent (comme un utilisateur)."""
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent

    vp = view.viewport()

    def send(kind, pos, button, buttons):
        local = QPointF(pos)
        ev = QMouseEvent(kind, local, QPointF(vp.mapToGlobal(pos)), button, buttons, Qt.NoModifier)
        QApplication.sendEvent(vp, ev)
        app.processEvents()
        time.sleep(0.015)  # pyqtgraph ignore les déplacements à plus de 100 Hz

    none = Qt.NoButton
    send(QEvent.MouseMove, start - QPoint(3, 0), none, none)
    send(QEvent.MouseMove, start, none, none)
    send(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton)
    steps = 12
    for i in range(1, steps + 1):
        p = QPoint(
            start.x() + (end.x() - start.x()) * i // steps,
            start.y() + (end.y() - start.y()) * i // steps,
        )
        send(QEvent.MouseMove, p, none, Qt.LeftButton)
    send(QEvent.MouseButtonRelease, end, Qt.LeftButton, none)


def check_interactions(app, win) -> None:
    from PySide6.QtCore import QPointF

    cfg = win.blocks[1]
    w = win.widgets[cfg.id]
    win.select_block(win.blocks[0].id)

    # Zone de zoom : glisser le centre de la région vers la droite.
    x0, x1 = w.region.getRegion()
    vb = w.main_plot.getViewBox()
    y_lo, y_hi = vb.viewRange()[1]
    mid = vb.mapViewToScene(QPointF((x0 + x1) / 2, y_lo + 0.65 * (y_hi - y_lo)))
    mid = w.glw.mapFromScene(mid)
    _drag(app, w.glw, mid, mid + QPoint(60, 0))
    zx0, zx1 = w.zoom_plot.getViewBox().viewRange()[0]
    print(f"Zone de zoom : {x0:.1f}-{x1:.1f} -> {cfg.zoom_x0:.1f}-{cfg.zoom_x1:.1f} (encart {zx0:.1f}-{zx1:.1f})")
    assert cfg.zoom_x0 > x0 and abs(zx0 - cfg.zoom_x0) < 1e-6, "zone de zoom non synchronisée"

    # Échelle flottante : deux glissers successifs dans le même bloc que la zone de zoom.
    w.redraw()
    app.processEvents()
    for k in range(2):
        sb = w.scale_bar
        corner = w.glw.mapFromScene(sb.mapToScene(QPointF(-3, -3)))
        before = (cfg.sb_fx, cfg.sb_fy)
        _drag(app, w.glw, corner, corner - QPoint(60, 30))
        print(f"Échelle flottante #{k + 1} : {before[0]:.2f},{before[1]:.2f} -> {cfg.sb_fx:.2f},{cfg.sb_fy:.2f}")
        assert cfg.sb_fx < before[0] and cfg.sb_fy > before[1], "glisser de l'échelle sans effet"

    # Formulaires : changement de signal (recalcul) et de couleur de titre (redessin).
    assert win.selected_id == cfg.id, "un clic dans un bloc doit le sélectionner"
    form = win.block_forms[0]
    combo = form._widgets["signal"][1]
    combo.setCurrentIndex(combo.findData("low"))
    assert cfg.signal == "low" and cfg.id in win._pending
    win.block_forms[1]._set("title_color", "#aa0000")
    assert "#aa0000" in w.title_label.styleSheet()
    # Sélection d'un canal supplémentaire dans la table.
    item = win.channel_picker.table.item(3, 0)
    item.setCheckState(Qt.Checked)
    assert 3 in cfg.channels, cfg.channels
    wait_all(app, win)
    assert not w._error, w._error
    print(f"Formulaires OK : signal={cfg.signal}, canaux={cfg.channels}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rhs", nargs="?")
    ap.add_argument("--out", default=str(ROOT / "tools" / "_smoke_out"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.resize(1900, 1150)
    win.show()
    if args.rhs:
        win.open_recording(args.rhs)
    else:
        win.load_demo()
    dt = wait_all(app, win)
    print(f"{win.rec.describe()}\n{len(win.blocks)} blocs calculés en {dt:.2f} s")
    print("Trigger :", win.trigger_status.text().replace("<br>", " | "))

    errors = 0
    for b in win.blocks:
        w = win.widgets[b.id]
        n = len(w.data.series) if w.data else 0
        state = f"ERREUR {w._error}" if w._error else f"{n} séries — {w.data.info if w.data else ''}"
        errors += bool(w._error)
        print(f"  [{b.signal:6s} {b.mode:10s}] {b.title}: {state}")

    # Interactions : style seul, multi-canaux, zoom, échelle flottante, export
    first = win.blocks[1]
    win.select_block(first.id)
    first.channels, first.colors = [0, 1, 2], []
    win.schedule(first.id)
    first.legend_pos = "bottom-left"
    first.sb_show = True
    win.widgets[first.id].redraw()
    wait_all(app, win)
    win.widgets[first.id].render_image(2.0).save(str(out / "bloc_multicanal.png"))
    win.widgets[first.id].export_svg(str(out / "bloc_multicanal.svg"))

    check_interactions(app, win)

    win.save_config(str(out / "config.json"))
    win.load_config(str(out / "config.json"))
    wait_all(app, win)
    errors += sum(bool(w._error) for w in win.widgets.values())

    win.open_stats()
    for i in range(win.stats_dialog.analysis_combo.count()):
        win.stats_dialog.analysis_combo.setCurrentIndex(i)
        app.processEvents()
    win.stats_dialog.figure.savefig(out / "stats.png", dpi=90)

    win.grab().save(str(out / "fenetre.png"))
    print("Captures :", out)
    win.engine.shutdown()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
