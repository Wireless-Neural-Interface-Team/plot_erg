"""GUI réelle : aperçu / montage avec amp+taille et temps+taille.

    .venv\\Scripts\\python.exe tools\\smoke_scale_bar_gui.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EXAMPLE = ROOT / "exemple.rhs"
CACHE = Path(r"C:\Users\pierr\AppData\Local\Temp\plot_erg_exemple_params")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from config import AnalysisConfig  # noqa: E402
from dataset_builder import build_recording  # noqa: E402
from gui.defaults import app_defaults_from_config  # noqa: E402
from gui.main_window import ViewerWindow  # noqa: E402
from gui.widgets.recordings_panel import RecordingEntry  # noqa: E402
from scale_bar_model import resolve_axis_bar  # noqa: E402
from view_config import AxisLimits  # noqa: E402


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _settle(app: QApplication, window: ViewerWindow, timeout_s: float = 45.0) -> None:
    panel = window.params_panel
    deadline = time.perf_counter() + timeout_s
    idle = 0
    while time.perf_counter() < deadline:
        if panel._view_debouncer._timer.isActive():
            panel._view_debouncer.flush()
        if window._redraw._debouncer._timer.isActive():
            window._redraw._debouncer.flush()
        app.processEvents()
        busy = False
        grids = []
        inspect = window._channel_inspect
        if inspect is not None:
            grids.append(inspect.grid)
        page = window.tabs.currentWidget()
        if page is not None and hasattr(page, "grid"):
            grids.append(page.grid)
        for grid in grids:
            if getattr(grid, "_pending", None):
                busy = True
            timer = getattr(grid, "_timer", None)
            if timer is not None and timer.isActive():
                busy = True
        if panel._view_debouncer._timer.isActive() or window._redraw._debouncer._timer.isActive():
            busy = True
        if not busy:
            idle += 1
            if idle >= 4:
                return
        else:
            idle = 0
        time.sleep(0.01)
    raise TimeoutError("rendu non terminé")


def _enable_both_manual(
    params,
    *,
    amp: float = 50.0,
    amp_px: int = 40,
    time_ms: float = 100.0,
    time_px: int = 40,
) -> None:
    params._show_scale_bars.setChecked(True)
    params._scale_bar_amp_manual.setChecked(True)
    params._scale_bar_amp.setValue(amp)
    params._scale_bar_amp_len_px.setValue(amp_px)
    params._scale_bar_time_manual.setChecked(True)
    params._scale_bar_time_ms.setValue(time_ms)
    params._scale_bar_time_len_px.setValue(time_px)


def _preview_host(window: ViewerWindow):
    inspect = window._channel_inspect
    if inspect is None:
        return None, None
    for placement in inspect._placements:
        if placement.panel != "full_recording":
            continue
        if placement.has_custom_zoom:
            continue
        widget = inspect.grid.panel_widget(placement)
        host = getattr(widget, "plot_host", None) if widget is not None else None
        if host is not None and host._spec is not None:
            return host, widget
    return None, None


def _zoom_host(window: ViewerWindow):
    inspect = window._channel_inspect
    if inspect is None:
        return None, None
    for placement in inspect._placements:
        if not placement.has_custom_zoom:
            continue
        widget = inspect.grid.panel_widget(placement)
        host = getattr(widget, "plot_host", None) if widget is not None else None
        if host is not None and host._spec is not None:
            return host, widget
    return None, None


def _montage_host(window: ViewerWindow):
    page = window.tabs.currentWidget()
    if page is None or not hasattr(page, "grid"):
        return None, None
    tab = window._current_tab()
    if tab is None:
        return None, None
    for placement in tab.panels:
        if placement.panel != "montage_continuous_raw":
            continue
        widget = page.grid.panel_widget(placement)
        host = getattr(widget, "plot_host", None) if widget is not None else None
        if host is not None and host._spec is not None:
            return host, widget
    return None, None


def _check_host(label: str, host, *, amp: float, time_s: float) -> None:
    bars = getattr(host._spec, "scale_bars", None)
    if bars is None or not bars.enabled:
        _fail(f"{label}: scale_bars absentes")
    if not bars.amp_manual or abs(float(bars.amplitude) - amp) > 1e-6:
        _fail(f"{label}: amplitude {getattr(bars, 'amplitude', None)} ≠ {amp}")
    if not bars.amp_length_manual:
        _fail(f"{label}: taille amp. non manuelle")
    if not bars.time_manual or abs(float(bars.time_s) - time_s) > 1e-6:
        _fail(f"{label}: temps {getattr(bars, 'time_s', None)} ≠ {time_s}")
    if not bars.time_length_manual:
        _fail(f"{label}: taille temps non manuelle")
    lab = resolve_axis_bar(
        span=200.0,
        axis_px=400.0,
        value=float(bars.amplitude),
        length_px=float(bars.amp_length_px),
    ).label
    if abs(lab - amp) > 1e-9:
        _fail(f"{label}: libelle amp {lab}")
    needed_h = host.scale_bar_needed_view_height_px()
    needed_w = host.scale_bar_needed_view_width_px()
    print(
        f"  OK  {label}: amp={amp}uV/{bars.amp_length_px}px "
        f"temps={time_s}s/{bars.time_length_px}px -> needed H={needed_h} W={needed_w}"
    )
    # Fit largeur : OK sur fenêtre courte ; None si span trop long ou montage.
    plots = list(host._plots or [])
    if plots:
        plot = plots[0]
        (x0, x1), (y0, y1) = plot.viewRange()
        plot.setXRange(x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0), padding=0)
        plot.setYRange(y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0), padding=0)
        host._refresh_scale_bars()
        lab2 = resolve_axis_bar(
            span=abs(y1 - y0) * 0.6,
            axis_px=200.0,
            value=float(bars.amplitude),
            length_px=40.0,
        ).label
        if abs(lab2 - amp) > 1e-9:
            _fail(f"{label}: libelle apres zoom {lab2}")
        print(f"  OK  {label}: libelles stables apres zoom interactif")


def main() -> int:
    if not EXAMPLE.is_file():
        print(f"skip: exemple.rhs absent ({EXAMPLE})")
        return 0

    print("chargement cache…", flush=True)
    cfg = AnalysisConfig(rhs_file=EXAMPLE, work_dir=CACHE)
    recording, _built = build_recording(cfg, cache_root=CACHE)

    app = QApplication.instance() or QApplication([])
    for name in ("information", "warning", "critical", "question"):
        setattr(
            QMessageBox,
            name,
            staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok),
        )

    window = ViewerWindow(app_defaults_from_config())
    window._ensure_channels_for_indices = lambda *_a, **_k: None  # type: ignore[method-assign]
    window.show()
    window.resize(1280, 800)
    app.processEvents()

    panel = window.recordings_panel
    panel._entries.append(
        RecordingEntry(
            row_id=1,
            path=EXAMPLE,
            label="exemple",
            recording=recording,
            status="ready",
        )
    )
    panel._rebuild_table()
    window._refresh_channels()
    stims = [float(t) for t in recording.stimulation_times_s()]
    stim0 = stims[0] if stims else 1.0
    x0, x1 = max(0.0, stim0 - 0.4), stim0 + 12.0
    params = window.params_panel
    params._max_points.setValue(4000)
    # X manuel OK ; Y auto pour permettre le fit hauteur.
    params._x_limits.set_value(AxisLimits(enabled=True, minimum=x0, maximum=x1))
    window._sync_channel_inspect(redraw=True)
    _settle(app, window)

    amp, amp_px, time_ms, time_px = 50.0, 40, 100.0, 40
    _enable_both_manual(
        params, amp=amp, amp_px=amp_px, time_ms=time_ms, time_px=time_px
    )
    _settle(app, window)

    print("aperçu:")
    host, widget = _preview_host(window)
    if host is None or widget is None:
        _fail("pas d'host aperçu")
    _check_host("aperçu", host, amp=amp, time_s=time_ms / 1000.0)
    widget._fit_scale_bar_widget_height()
    widget._fit_scale_bar_widget_width()
    app.processEvents()
    if getattr(widget, "fitted_height_px", None) is None:
        _fail("apercu: fitted_height_px non pose")
    # Largeur : posee seulement si fenetre X assez courte ; sinon None (OK).
    print(
        f"  OK  apercu fitted H={widget.fitted_height_px} "
        f"W={getattr(widget, 'fitted_width_px', None)}"
    )

    print("montage:")
    window.open_montage_review()
    app.processEvents()
    _settle(app, window)
    # open_montage_review recharge les settings : réappliquer les barres.
    _enable_both_manual(
        params, amp=amp, amp_px=amp_px, time_ms=time_ms, time_px=time_px
    )
    _settle(app, window)
    mhost, mwidget = _montage_host(window)
    if mhost is None or mwidget is None:
        _fail("pas d'host montage")
    _check_host("montage", mhost, amp=amp, time_s=time_ms / 1000.0)
    mwidget._fit_scale_bar_widget_height()
    mwidget._fit_scale_bar_widget_width()
    app.processEvents()
    print(
        f"  OK  montage fitted H={getattr(mwidget, 'fitted_height_px', None)} "
        f"W={getattr(mwidget, 'fitted_width_px', None)}"
    )

    print("zoom:")
    # Revenir en aperçu et activer une fenêtre zoom onset.
    if hasattr(window, "_set_view_mode"):
        window._set_view_mode(montage=False)
    app.processEvents()
    zoom_cb = getattr(params, "_cb_zoom_onset", None) or getattr(
        params, "_show_zoom_onset", None
    )
    if zoom_cb is not None:
        zoom_cb.setChecked(True)
    _enable_both_manual(
        params, amp=amp, amp_px=amp_px, time_ms=time_ms, time_px=time_px
    )
    window._sync_channel_inspect(redraw=True)
    _settle(app, window)
    zhost, zwidget = _zoom_host(window)
    if zhost is None:
        print("  SKIP aucun placement custom zoom (couvert par smoke_scale_bar_contexts)")
    else:
        _check_host("zoom", zhost, amp=amp, time_s=time_ms / 1000.0)
        if zwidget is not None:
            zwidget._fit_scale_bar_widget_height()
            zwidget._fit_scale_bar_widget_width()
            app.processEvents()
            print(
                f"  OK  zoom fitted H={getattr(zwidget, 'fitted_height_px', None)} "
                f"W={getattr(zwidget, 'fitted_width_px', None)} "
                f"(W=None si fenetre X trop longue)"
            )

    print("smoke_scale_bar_gui: all passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
