"""Passe complète des paramètres GUI sur exemple.rhs (aperçu + montage).

Le cache ``plot_erg_exemple_params`` est réutilisé. Les produits lourds
(filtres, spikes) ne sont pas recalculés : ``_ensure_channels_for_indices``
est neutralisé, le tracé continu memmap reste réel.

    .venv\\Scripts\\python.exe tools\\smoke_exemple_params.py
"""

from __future__ import annotations

import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()

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
from view_config import AxisLimits, TextFace  # noqa: E402


class _Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"  OK  {name}", flush=True)
            return
        msg = name if not detail else f"{name} — {detail}"
        self.failures.append(msg)
        print(f"  FAIL {msg}", flush=True)


def _settle(app: QApplication, window: ViewerWindow, timeout_s: float = 60.0) -> None:
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
    raise TimeoutError("le rendu n'a pas fini")


def _preview_host(window: ViewerWindow):
    inspect = window._channel_inspect
    if inspect is None:
        return None, None
    chosen = None
    for placement in inspect._placements:
        if placement.panel != "full_recording":
            continue
        stream = str(getattr(placement, "stream", "") or "raw")
        if stream == "raw":
            chosen = placement
            break
        if chosen is None:
            chosen = placement
    if chosen is None and inspect._placements:
        chosen = inspect._placements[0]
    if chosen is None:
        return None, None
    widget = inspect.grid.panel_widget(chosen)
    host = getattr(widget, "plot_host", None) if widget is not None else None
    return chosen, host


def _preview_spec(window: ViewerWindow):
    _placement, host = _preview_host(window)
    if host is None:
        return None
    return host._spec


def _montage_spec(window: ViewerWindow):
    page = window.tabs.currentWidget()
    if page is None or not hasattr(page, "grid"):
        return None
    tab = window._current_tab()
    if tab is None:
        return None
    for placement in tab.panels:
        if placement.panel != "montage_continuous_raw":
            continue
        widget = page.grid.panel_widget(placement)
        host = getattr(widget, "plot_host", None) if widget is not None else None
        if host is not None and host._spec is not None:
            return host._spec
    return None


def _pen_width(host) -> float | None:
    plots = list(host.plot_items() or [])
    if not plots:
        return None
    items = list(plots[0].listDataItems() or [])
    if not items:
        return None
    pen = items[0].opts.get("pen")
    if pen is None:
        return None
    try:
        return float(pen.widthF())
    except Exception:
        return None


def _set_combo(box, value: str) -> None:
    index = box.findData(value)
    if index < 0:
        raise AssertionError(f"choix absent: {value}")
    box.setCurrentIndex(index)


def main() -> int:
    if not EXAMPLE.is_file():
        print(f"skip: exemple.rhs absent ({EXAMPLE})")
        return 0
    report = _Report()
    print("chargement cache…", flush=True)
    t0 = time.perf_counter()
    cfg = AnalysisConfig(rhs_file=EXAMPLE, work_dir=CACHE)
    recording, _built = build_recording(cfg, cache_root=CACHE)
    print(
        f"  {time.perf_counter() - t0:.1f}s  "
        f"{recording.n_channels} canaux, {recording.meta.n_samples} échantillons, "
        f"{recording.n_trials} stims",
        flush=True,
    )
    stims = [float(t) for t in recording.stimulation_times_s()]
    print(f"  stims (s): {', '.join(f'{t:.3f}' for t in stims[:6])}", flush=True)

    app = QApplication.instance() or QApplication([])
    # Offscreen : une boîte modale (fichier plus « prêt ») bloquerait le script.
    for name in ("information", "warning", "critical", "question"):
        setattr(QMessageBox, name, staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok))
    window = ViewerWindow(app_defaults_from_config())
    ensured: list[dict] = []

    def _blocked_ensure(*_args, **kwargs):
        ensured.append(dict(kwargs))

    window._ensure_channels_for_indices = _blocked_ensure  # type: ignore[method-assign]
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
    stim0 = stims[0] if stims else 1.0
    x0, x1 = max(0.0, stim0 - 0.4), stim0 + 12.0
    params_early = window.params_panel
    params_early._max_points.setValue(4000)
    params_early._x_limits.set_value(AxisLimits(enabled=True, minimum=x0, maximum=x1))
    window._sync_channel_inspect(redraw=True)
    _settle(app, window)
    params = window.params_panel
    params.tabs.setCurrentIndex(1)
    window.show()
    app.processEvents()

    spec = _preview_spec(window)
    report.check("aperçu: une courbe brute est dessinée", spec is not None and bool(spec.series), repr(getattr(spec, "status", None)))
    if spec is None or not spec.series:
        recording.close()
        print("arrêt: pas de courbe")
        return 1
    report.check(
        "aperçu: xlabel par défaut « Temps (s) »",
        "Temps (s)" in str(spec.xlabel) and "trigger" not in str(spec.xlabel).lower(),
        spec.xlabel,
    )
    roles = {str(getattr(a, "role", "") or "") for a in spec.annotations}
    report.check(
        "aperçu: marqueurs de stim présents",
        "stim_onset" in roles and "stim_offset" in roles,
        str(sorted(roles)),
    )
    n0 = int(spec.series[0].x.size)
    report.check("aperçu: la courbe a des points", n0 > 10, str(n0))

    # Fenêtre courte autour de la 1re stim : les redraws suivants restent légers.
    stim0 = stims[0] if stims else 1.0
    x0, x1 = max(0.0, stim0 - 0.4), stim0 + 12.0
    params._x_limits.set_value(AxisLimits(enabled=True, minimum=x0, maximum=x1))
    params._max_points.setValue(800)
    _settle(app, window)
    spec = _preview_spec(window)
    report.check("aperçu: échelle X manuelle", spec is not None and spec.xlim is not None)
    if spec is not None and spec.xlim is not None:
        report.check(
            "aperçu: xlim = fenêtre demandée",
            abs(spec.xlim[0] - x0) < 0.05 and abs(spec.xlim[1] - x1) < 0.05,
            str(spec.xlim),
        )
    if spec is not None and spec.series:
        n1 = int(spec.series[0].x.size)
        report.check(
            "aperçu: max points réduit la courbe",
            n1 < n0 and n1 <= 2000,
            f"{n0} -> {n1}",
        )
        roles = {str(getattr(a, "role", "") or "") for a in spec.annotations}
        report.check(
            "aperçu: marqueurs encore visibles dans la fenêtre",
            "stim_onset" in roles,
            str(sorted(roles)),
        )

    params._line_width.setValue(4.0)
    _settle(app, window)
    spec = _preview_spec(window)
    host = _preview_host(window)[1]
    report.check(
        "aperçu: épaisseur de courbe",
        spec is not None and spec.series and abs(float(spec.series[0].linewidth) - 4.0) < 0.05,
        str(getattr(spec.series[0], "linewidth", None) if spec and spec.series else None),
    )
    drawn_w = _pen_width(host) if host is not None else None
    report.check(
        "aperçu: épaisseur arrivée au tracé",
        drawn_w is not None and abs(drawn_w - 4.0) < 0.2,
        str(drawn_w),
    )

    params._text_title.setText("Titre apercu")
    params._text_xlabel.setText("Axe X apercu")
    params._text_ylabel.setText("Axe Y apercu")
    _settle(app, window)
    spec = _preview_spec(window)
    report.check("aperçu: titre", spec is not None and spec.title == "Titre apercu", getattr(spec, "title", ""))
    report.check("aperçu: xlabel", spec is not None and spec.xlabel == "Axe X apercu", getattr(spec, "xlabel", ""))
    report.check(
        "aperçu: ylabel sur la ligne",
        spec is not None and spec.row_labels and spec.row_labels[0] == "Axe Y apercu",
        str(getattr(spec, "row_labels", None)),
    )

    params._title_font.set_size_value(16.0)
    params._title_font.set_face_value(TextFace(bold=True))
    params._label_font.set_size_value(13.0)
    params._tick_font.set_size_value(11.0)
    _settle(app, window)
    spec = _preview_spec(window)
    report.check("aperçu: taille titre", spec is not None and abs(spec.title_font_size - 16.0) < 0.05, str(getattr(spec, "title_font_size", None)))
    report.check("aperçu: titre gras", spec is not None and bool(spec.title_bold))
    report.check("aperçu: taille labels", spec is not None and abs(spec.label_font_size - 13.0) < 0.05)
    report.check("aperçu: taille graduations", spec is not None and abs(spec.tick_font_size - 11.0) < 0.05)
    params._tick_font.set_size_value(17.0)
    params._tick_font.set_face_value(TextFace(bold=True, italic=True, underline=True))
    _settle(app, window)
    spec = _preview_spec(window)
    host = _preview_host(window)[1]
    axis = host.plot_items()[0].getAxis("left") if host and host.plot_items() else None
    tick_font = axis.style.get("tickFont") if axis is not None else None
    report.check(
        "aperçu: police ticks sur l'axe",
        tick_font is not None
        and abs(float(tick_font.pointSizeF()) - 17.0) < 0.05
        and bool(tick_font.bold())
        and bool(tick_font.italic())
        and bool(tick_font.underline()),
        str(tick_font),
    )
    report.check(
        "aperçu: wrap QPicture ticks",
        axis is not None and bool(getattr(axis, "_erg_tick_face_picture", False)),
    )

    params._grid.setChecked(False)
    params._grid_alpha.setValue(0.65)
    params._show_borders.setChecked(False)
    params._ticks_inside.setChecked(True)
    _settle(app, window)
    spec = _preview_spec(window)
    report.check("aperçu: grille masquée", spec is not None and spec.grid is False)
    params._grid.setChecked(True)
    _settle(app, window)
    spec = _preview_spec(window)
    report.check("aperçu: grille + alpha", spec is not None and spec.grid is True and abs(spec.grid_alpha - 0.65) < 0.02, str(getattr(spec, "grid_alpha", None)))
    report.check("aperçu: bordures masquées", spec is not None and spec.show_borders is False)
    report.check("aperçu: graduations intérieures", spec is not None and spec.ticks_inside is True)

    params._legend_visible.setChecked(False)
    _settle(app, window)
    spec = _preview_spec(window)
    report.check("aperçu: légende masquée", spec is not None and spec.show_legend is False)
    params._legend_visible.setChecked(True)
    params._legend_columns.setValue(3)
    params._legend_gap.setValue(0.41)
    params._legend_frame.setChecked(False)
    params._legend_font.set_size_value(12.0)
    params._legend_font.set_face_value(TextFace(italic=True))
    _settle(app, window)
    spec = _preview_spec(window)
    report.check("aperçu: légende visible", spec is not None and spec.show_legend is True)
    report.check("aperçu: colonnes légende", spec is not None and spec.legend_columns == 3, str(getattr(spec, "legend_columns", None)))
    report.check("aperçu: écart légende", spec is not None and abs(spec.legend_gap - 0.41) < 0.02)
    report.check("aperçu: cadre légende", spec is not None and spec.legend_frame is False)
    report.check("aperçu: police légende", spec is not None and abs(spec.legend_font_size - 12.0) < 0.05 and spec.legend_italic)

    label_before = str(spec.series[0].label) if spec and spec.series else ""
    params._legend_counts.setChecked(False)
    _settle(app, window)
    spec = _preview_spec(window)
    label_counts = str(spec.series[0].label) if spec and spec.series else ""
    report.check(
        "aperçu: compteur d'échantillons retiré",
        "(n=" in label_before and "(n=" not in label_counts,
        f"{label_before!r} -> {label_counts!r}",
    )
    params._text_series_suffix.setChecked(False)
    _settle(app, window)
    spec = _preview_spec(window)
    label_suffix = str(spec.series[0].label) if spec and spec.series else ""
    report.check(
        "aperçu: suffixe de série retiré",
        "continuous" in label_counts and "continuous" not in label_suffix,
        f"{label_counts!r} -> {label_suffix!r}",
    )
    params._text_legend_labels.setPlainText("Legende custom")
    _settle(app, window)
    spec = _preview_spec(window)
    report.check(
        "aperçu: libellé de légende forcé",
        spec is not None and spec.series and spec.series[0].label == "Legende custom",
        str(getattr(spec.series[0], "label", None) if spec and spec.series else None),
    )

    params._text_xlabel.clear()
    params._time_sync.setCurrentIndex(params._time_sync.findData("trigger"))
    _settle(app, window)
    spec = _preview_spec(window)
    report.check(
        "aperçu: synchro trigger",
        spec is not None and "trigger" in str(spec.xlabel).lower(),
        str(getattr(spec, "xlabel", None)),
    )
    params._time_sync.setCurrentIndex(params._time_sync.findData("recording_start"))
    _settle(app, window)

    params._cb_mark_stims.setChecked(False)
    params._stim_markers.setChecked(False)
    _settle(app, window)
    spec = _preview_spec(window)
    roles = {str(getattr(a, "role", "") or "") for a in (spec.annotations if spec else [])}
    report.check("aperçu: marqueurs retirés", "stim_onset" not in roles, str(sorted(roles)))
    params._cb_mark_stims.setChecked(True)
    params._stim_markers.setChecked(True)
    params._stim_linewidth.setValue(3.5)
    params._stim_onset_color.set_color("#112233")
    params._stim_onset_color.changed.emit()
    params._stim_offset_color.set_color("#445566")
    params._stim_offset_color.changed.emit()
    _settle(app, window)
    spec = _preview_spec(window)
    onsets = [a for a in (spec.annotations if spec else []) if getattr(a, "role", "") == "stim_onset"]
    offsets = [a for a in (spec.annotations if spec else []) if getattr(a, "role", "") == "stim_offset"]
    report.check("aperçu: marqueurs rétablis", bool(onsets) and bool(offsets))
    report.check(
        "aperçu: couleur début de stim",
        bool(onsets) and str(onsets[0].color).lower() == "#112233",
        str(getattr(onsets[0], "color", None) if onsets else None),
    )
    report.check(
        "aperçu: couleur fin de stim",
        bool(offsets) and str(offsets[0].color).lower() == "#445566",
        str(getattr(offsets[0], "color", None) if offsets else None),
    )
    report.check(
        "aperçu: épaisseur des pointillés",
        spec is not None and abs(float(spec.stim_linewidth) - 3.5) < 0.05,
        str(getattr(spec, "stim_linewidth", None)),
    )

    params._title_dx.setValue(18)
    params._title_dy.setValue(-10)
    params._label_font.set_face_value(TextFace(italic=True))
    params._tick_font.set_face_value(TextFace(underline=True))
    params._show_stim_labels.setChecked(True)
    params._text_stim_on.setText("ON!")
    params._text_stim_off.setText("OFF!")
    params._stim_on_font.set_size_value(13.0)
    params._stim_on_font.set_face_value(TextFace(bold=True))
    params._stim_off_font.set_size_value(10.0)
    params._stim_off_font.set_face_value(TextFace(italic=True))
    params._stim_on_dx.setValue(6)
    params._stim_on_dy.setValue(4)
    params._stim_off_dx.setValue(-5)
    params._stim_off_dy.setValue(8)
    _settle(app, window)
    spec = _preview_spec(window)
    report.check(
        "aperçu: décalage titre",
        spec is not None
        and abs(float(spec.title_dx_px) - 18.0) < 0.05
        and abs(float(spec.title_dy_px) - (-10.0)) < 0.05,
        f"dx={getattr(spec, 'title_dx_px', None)} dy={getattr(spec, 'title_dy_px', None)}",
    )
    report.check("aperçu: axes italiques", spec is not None and bool(spec.label_italic))
    report.check("aperçu: ticks soulignés", spec is not None and bool(spec.tick_underline))
    stim_labels = [
        a for a in (spec.annotations if spec else [])
        if str(getattr(a, "kind", "")) == "stim_label"
    ]
    on_lbl = [a for a in stim_labels if getattr(a, "role", "") == "stim_onset"]
    off_lbl = [a for a in stim_labels if getattr(a, "role", "") == "stim_offset"]
    report.check(
        "aperçu: libellé début stim",
        bool(on_lbl) and str(on_lbl[0].text) == "ON!",
        str([getattr(a, "text", None) for a in on_lbl]),
    )
    report.check(
        "aperçu: libellé fin stim",
        bool(off_lbl) and str(off_lbl[0].text) == "OFF!",
        str([getattr(a, "text", None) for a in off_lbl]),
    )
    report.check(
        "aperçu: police / position début stim",
        bool(on_lbl)
        and abs(float(on_lbl[0].fontsize) - 13.0) < 0.05
        and bool(on_lbl[0].bold)
        and abs(float(on_lbl[0].dx_px) - 6.0) < 0.05
        and abs(float(on_lbl[0].dy_px) - 4.0) < 0.05,
        str(on_lbl[0] if on_lbl else None),
    )
    report.check(
        "aperçu: police / position fin stim",
        bool(off_lbl)
        and abs(float(off_lbl[0].fontsize) - 10.0) < 0.05
        and bool(off_lbl[0].italic)
        and abs(float(off_lbl[0].dx_px) - (-5.0)) < 0.05
        and abs(float(off_lbl[0].dy_px) - 8.0) < 0.05,
        str(off_lbl[0] if off_lbl else None),
    )
    params._show_stim_labels.setChecked(False)
    _settle(app, window)
    spec = _preview_spec(window)
    leftover = [
        a for a in (spec.annotations if spec else [])
        if str(getattr(a, "kind", "")) == "stim_label"
    ]
    report.check("aperçu: libellés stim masqués", not leftover, str(leftover))
    roles = {str(getattr(a, "role", "") or "") for a in (spec.annotations if spec else [])}
    report.check(
        "aperçu: pointillés gardés sans libellés",
        "stim_onset" in roles and "stim_offset" in roles,
        str(sorted(roles)),
    )
    params._show_stim_labels.setChecked(True)
    _settle(app, window)

    params._show_scale_bars.setChecked(True)
    params._scale_bar_time_manual.setChecked(True)
    params._scale_bar_time_ms.setValue(250)
    params._scale_bar_amp_manual.setChecked(True)
    params._scale_bar_amp.setValue(50)
    params._scale_bar_linewidth.setValue(2.5)
    params._scale_bar_dx.setValue(30)
    params._scale_bar_dy.setValue(-12)
    params._scale_bar_time_dx.setValue(18)
    params._scale_bar_time_dy.setValue(-6)
    _settle(app, window)
    spec = _preview_spec(window)
    bars = getattr(spec, "scale_bars", None) if spec else None
    report.check("aperçu: barres d'échelle actives", bars is not None and bars.enabled)
    report.check(
        "aperçu: barre temps 0,25 s",
        bars is not None and bars.time_manual and abs(bars.time_s - 0.25) < 1e-6,
        str(getattr(bars, "time_s", None)),
    )
    report.check(
        "aperçu: barre amplitude 50",
        bars is not None and bars.amp_manual and abs(bars.amplitude - 50.0) < 1e-6,
    )
    report.check(
        "aperçu: épaisseur barre d'échelle",
        bars is not None and abs(float(bars.linewidth) - 2.5) < 0.05,
        str(getattr(bars, "linewidth", None)),
    )
    report.check(
        "aperçu: position barre amplitude",
        bars is not None and abs(float(bars.dx_px) - 30.0) < 0.05 and abs(float(bars.dy_px) - (-12.0)) < 0.05,
        f"dx={getattr(bars, 'dx_px', None)} dy={getattr(bars, 'dy_px', None)}",
    )
    report.check(
        "aperçu: position barre temps",
        bars is not None
        and abs(float(bars.time_dx_px) - 18.0) < 0.05
        and abs(float(bars.time_dy_px) - (-6.0)) < 0.05,
        f"time_dx={getattr(bars, 'time_dx_px', None)} time_dy={getattr(bars, 'time_dy_px', None)}",
    )

    params._preview_title_gap.setValue(36)
    params._graph_height.setValue(520)
    _settle(app, window)
    spec = _preview_spec(window)
    report.check("aperçu: écart titre", spec is not None and int(spec.title_gap_px) == 36, str(getattr(spec, "title_gap_px", None)))
    height = int(window._channel_inspect.grid._panel_height)
    report.check("aperçu: hauteur de graphique", height == 520, str(height))

    params._raw_ylim.set_value(AxisLimits(enabled=True, minimum=-80.0, maximum=120.0))
    _settle(app, window)
    spec = _preview_spec(window)
    host = _preview_host(window)[1]
    row_ylims = list(getattr(spec, "row_ylims", []) or []) if spec is not None else []
    report.check(
        "aperçu: échelle Y brute",
        bool(row_ylims) and row_ylims[0] == (-80.0, 120.0),
        str(row_ylims or getattr(spec, "ylim", None)),
    )
    if host is not None and host.plot_items():
        y_range = host.plot_items()[0].viewRange()[1]
        report.check(
            "aperçu: échelle Y arrivée au tracé",
            abs(float(y_range[0]) + 80.0) < 1.0 and abs(float(y_range[1]) - 120.0) < 1.0,
            str(y_range),
        )

    params._show_spike_threshold.setChecked(True)
    _settle(app, window)
    spec = _preview_spec(window)
    hlines = [a for a in (spec.annotations if spec else []) if getattr(a, "kind", "") == "hline"]
    report.check("aperçu: ligne de seuil", bool(hlines), str([getattr(a, "kind", "") for a in (spec.annotations if spec else [])]))
    params._show_spike_threshold.setChecked(False)

    n_before = len(window._channel_inspect._placements)
    params._curve_high_box.setChecked(True)
    _settle(app, window)
    streams = [str(getattr(p, "stream", "") or "") for p in window._channel_inspect._placements if p.panel == "full_recording"]
    report.check(
        "aperçu: case HIGH ajoute un flux",
        "hp" in streams and len(window._channel_inspect._placements) > n_before,
        str(streams),
    )
    params._curve_high_box.setChecked(False)
    _settle(app, window)

    # Visibilité des blocs propres à chaque vue.
    app.processEvents()
    report.check(
        "aperçu: groupe revue montage masqué",
        not params._montage_review_group.isVisible(),
    )
    report.check(
        "aperçu: hauteur de graphique visible",
        params._graph_height.isVisible(),
    )

    # Mode de contenu : l'ensure est neutralisé, on vérifie le câblage.
    ensured.clear()
    window._channel_inspect._stim_spin.setValue(3)
    _set_combo(window._channel_inspect._mode_combo, "stimulation")
    _settle(app, window)
    report.check(
        "aperçu: mode stimulation demandé",
        window._channel_inspect.preview_mode() == "stimulation"
        and params.viewer_settings().preview_content == "stimulation"
        and int(params.viewer_settings().analysis.stim_index) == 2,
        f"mode={window._channel_inspect.preview_mode()} content={params.viewer_settings().preview_content} stim={params.viewer_settings().analysis.stim_index}",
    )
    report.check(
        "aperçu: passage en stimulation demande les moyennes",
        any(bool(call.get("need_means")) for call in ensured),
        str(ensured),
    )
    mont_snapshot = params.settings_for("montage")
    report.check(
        "aperçu: le mode n'écrit pas le profil montage",
        mont_snapshot.preview_content != "stimulation",
        mont_snapshot.preview_content,
    )
    _set_combo(window._channel_inspect._mode_combo, "continuous")
    _settle(app, window)
    spec = _preview_spec(window)
    report.check(
        "aperçu: retour continuous redessine le flux",
        spec is not None and bool(spec.series) and spec.series[0].label == "Legende custom",
        str(getattr(spec.series[0], "label", None) if spec and spec.series else None),
    )

    # Zoom : valeurs du profil (fenêtres PDF / dialogue), isolées par vue.
    params._zoom_onset_t0.setValue(-0.25)
    params._zoom_onset_t1.setValue(0.55)
    params._zoom_end_t0.setValue(-0.15)
    params._zoom_end_t1.setValue(0.33)
    _settle(app, window)
    live = params.viewer_settings()
    report.check(
        "aperçu: fenêtres de zoom",
        abs(live.zoom_onset_t0_s + 0.25) < 1e-6
        and abs(live.zoom_onset_t1_s - 0.55) < 1e-6
        and abs(live.zoom_end_t0_s + 0.15) < 1e-6
        and abs(live.zoom_end_t1_s - 0.33) < 1e-6,
    )
    # Ces spins marquent le fichier « à retraiter » et ferment l'enregistrement.
    params._psth_bin.blockSignals(True)
    params._sampling.blockSignals(True)
    params._psth_bin.setValue(0.2)
    params._sampling.setValue(40)
    params._psth_bin.blockSignals(False)
    params._sampling.blockSignals(False)
    params._overlay_pre.setValue(3.5)
    params._overlay_post.setValue(6.5)
    params._overlay_alpha.setValue(0.15)
    _settle(app, window)
    live = params.viewer_settings()
    report.check(
        "aperçu: PSTH / sampling / spike scope",
        abs(live.psth_bin_window_s - 0.2) < 1e-6
        and live.sampling_percent == 40
        and abs(live.spike_overlay_pre_ms - 3.5) < 1e-6
        and abs(live.spike_overlay_post_ms - 6.5) < 1e-6
        and abs(live.spike_overlay_alpha - 0.15) < 1e-6,
    )

    preview_lw = float(params.viewer_settings().style.line_width)
    preview_title = params.viewer_settings().texts.title

    # Montage : 2 canaux pour rester léger.
    from dataclasses import replace

    stored = params.settings_for("montage")
    params._profiles["montage"] = replace(
        stored,
        montage_review_channels=2,
        x_limits=AxisLimits(enabled=True, minimum=x0, maximum=x1),
        style=replace(stored.style, max_points_per_curve=800),
    )
    window.open_montage_review()
    _settle(app, window, timeout_s=90.0)
    report.check("montage: vue active", window._in_montage)
    app.processEvents()
    report.check("montage: groupe revue visible", params._montage_review_group.isVisible())
    report.check("montage: hauteur aperçu masquée", not params._graph_height.isVisible())
    mspec = _montage_spec(window)
    report.check(
        "montage: courbes dessinées",
        mspec is not None and bool(mspec.series) and int(getattr(mspec, "n_rows", 0) or 0) == 2,
        f"rows={getattr(mspec, 'n_rows', None)} series={0 if mspec is None else len(mspec.series)}",
    )
    report.check(
        "montage: le titre aperçu n'a pas fuité",
        mspec is not None and mspec.title != preview_title,
        str(getattr(mspec, "title", None)),
    )
    if mspec is not None and mspec.series:
        report.check(
            "montage: l'épaisseur aperçu n'a pas fuité",
            abs(float(mspec.series[0].linewidth) - preview_lw) > 0.2,
            f"montage={mspec.series[0].linewidth} aperçu={preview_lw}",
        )

    params._line_width.setValue(2.2)
    params._raw_ylim.set_value(AxisLimits(enabled=True, minimum=-30.0, maximum=45.0))
    _settle(app, window)
    mspec = _montage_spec(window)
    expected_lw = max(0.6, 2.2 * 0.85)
    report.check(
        "montage: épaisseur (×0,85)",
        mspec is not None and mspec.series and abs(float(mspec.series[0].linewidth) - expected_lw) < 0.08,
        str(getattr(mspec.series[0], "linewidth", None) if mspec and mspec.series else None),
    )
    m_ylims = list(getattr(mspec, "row_ylims", []) or []) if mspec is not None else []
    report.check(
        "montage: échelle Y propre",
        bool(m_ylims) and m_ylims[0] == (-30.0, 45.0),
        str(m_ylims),
    )

    params._text_title.setText("Titre montage")
    params._text_xlabel.setText("Axe X montage")
    params._montage_show_kind_label.setChecked(True)
    _settle(app, window)
    mspec = _montage_spec(window)
    report.check("montage: titre", mspec is not None and mspec.title == "Titre montage", str(getattr(mspec, "title", None)))
    report.check("montage: xlabel", mspec is not None and mspec.xlabel == "Axe X montage", str(getattr(mspec, "xlabel", None)))
    report.check(
        "montage: type à côté du canal",
        mspec is not None and mspec.row_labels and any("WIDE" in str(label) for label in mspec.row_labels),
        str(getattr(mspec, "row_labels", None)),
    )
    params._montage_show_kind_label.setChecked(False)
    _settle(app, window)
    mspec = _montage_spec(window)
    report.check(
        "montage: type masqué",
        mspec is not None and mspec.row_labels and all("WIDE" not in str(label) for label in mspec.row_labels),
        str(getattr(mspec, "row_labels", None)),
    )
    params._montage_label_dx.setValue(24)
    params._montage_label_dy.setValue(-8)
    _settle(app, window)
    mspec = _montage_spec(window)
    report.check(
        "montage: décalage des titres",
        mspec is not None and abs(mspec.row_label_dx_px - 24) < 0.1 and abs(mspec.row_label_dy_px + 8) < 0.1,
        f"dx={getattr(mspec, 'row_label_dx_px', None)} dy={getattr(mspec, 'row_label_dy_px', None)}",
    )

    params._montage_row_height.setValue(80)
    _settle(app, window)
    page = window.tabs.currentWidget()
    m_height = int(page.grid._panel_height) if page is not None else 0
    report.check("montage: hauteur de ligne change la page", m_height > 80, str(m_height))
    height_80 = m_height
    params._montage_row_height.setValue(110)
    _settle(app, window)
    m_height = int(page.grid._panel_height)
    report.check("montage: hauteur de ligne plus grande", m_height > height_80, f"{height_80} -> {m_height}")

    params._montage_review_channels.setValue(1)
    _settle(app, window)
    mspec = _montage_spec(window)
    report.check(
        "montage: canaux visibles",
        mspec is not None and int(mspec.n_rows) == 1,
        str(getattr(mspec, "n_rows", None)),
    )
    params._montage_review_channels.setValue(2)
    _settle(app, window)

    params._grid.setChecked(False)
    params._show_borders.setChecked(False)
    params._show_scale_bars.setChecked(True)
    params._scale_bar_time_manual.setChecked(True)
    params._scale_bar_time_ms.setValue(100)
    params._scale_bar_dx.setValue(-20)
    params._scale_bar_dy.setValue(18)
    params._scale_bar_time_dx.setValue(-14)
    params._scale_bar_time_dy.setValue(22)
    _settle(app, window)
    mspec = _montage_spec(window)
    report.check("montage: grille masquée", mspec is not None and mspec.grid is False)
    report.check("montage: bordures masquées", mspec is not None and mspec.show_borders is False)
    report.check(
        "montage: barres d'échelle",
        mspec is not None and mspec.scale_bars.enabled and abs(mspec.scale_bars.time_s - 0.1) < 1e-6,
    )
    report.check(
        "montage: position barre amplitude",
        mspec is not None
        and abs(float(mspec.scale_bars.dx_px) - (-20.0)) < 0.05
        and abs(float(mspec.scale_bars.dy_px) - 18.0) < 0.05,
        f"dx={getattr(mspec.scale_bars, 'dx_px', None) if mspec else None} "
        f"dy={getattr(mspec.scale_bars, 'dy_px', None) if mspec else None}",
    )
    report.check(
        "montage: position barre temps",
        mspec is not None
        and abs(float(mspec.scale_bars.time_dx_px) - (-14.0)) < 0.05
        and abs(float(mspec.scale_bars.time_dy_px) - 22.0) < 0.05,
        f"time_dx={getattr(mspec.scale_bars, 'time_dx_px', None) if mspec else None} "
        f"time_dy={getattr(mspec.scale_bars, 'time_dy_px', None) if mspec else None}",
    )

    params._montage_channels.setValue(7)
    params._montage_page.setValue(2)
    _settle(app, window)
    live = params.viewer_settings()
    report.check(
        "montage: pagination PDF stockée",
        live.montage_channels == 7 and live.montage_page == 2,
    )
    mspec = _montage_spec(window)
    report.check(
        "montage: pagination PDF ne change pas la revue",
        mspec is not None and int(mspec.n_rows) == 2,
        str(getattr(mspec, "n_rows", None)),
    )

    ensured.clear()
    _set_combo(window._channel_inspect._mode_combo, "average")
    _settle(app, window)
    report.check(
        "montage: mode moyenne n'écrit pas l'aperçu",
        window._channel_inspect.preview_mode() == "continuous"
        and params.viewer_settings().preview_content == "average",
        f"preview={window._channel_inspect.preview_mode()} montage={params.viewer_settings().preview_content}",
    )
    _set_combo(window._channel_inspect._mode_combo, "continuous")
    _settle(app, window)

    # Retour aperçu : le profil aperçu est intact.
    window.return_to_channel_preview()
    _settle(app, window)
    report.check("retour aperçu", not window._in_montage)
    spec = _preview_spec(window)
    report.check(
        "retour: titre aperçu restauré",
        spec is not None and spec.title == "Titre apercu",
        str(getattr(spec, "title", None)),
    )
    report.check(
        "retour: épaisseur aperçu restaurée",
        spec is not None and spec.series and abs(float(spec.series[0].linewidth) - 4.0) < 0.05,
        str(getattr(spec.series[0], "linewidth", None) if spec and spec.series else None),
    )
    report.check(
        "retour: xlim aperçu restauré",
        spec is not None and spec.xlim is not None and abs(spec.xlim[0] - x0) < 0.05,
        str(getattr(spec, "xlim", None)),
    )
    back_ylims = list(getattr(spec, "row_ylims", []) or []) if spec is not None else []
    report.check(
        "retour: échelle Y aperçu restaurée",
        bool(back_ylims) and back_ylims[0] == (-80.0, 120.0),
        str(back_ylims),
    )
    back_bars = getattr(spec, "scale_bars", None) if spec is not None else None
    report.check(
        "retour: position échelle aperçu restaurée",
        back_bars is not None
        and abs(float(back_bars.dx_px) - 30.0) < 0.05
        and abs(float(back_bars.dy_px) - (-12.0)) < 0.05
        and abs(float(back_bars.time_dx_px) - 18.0) < 0.05
        and abs(float(back_bars.time_dy_px) - (-6.0)) < 0.05,
        f"dx={getattr(back_bars, 'dx_px', None)} dy={getattr(back_bars, 'dy_px', None)} "
        f"time_dx={getattr(back_bars, 'time_dx_px', None)} "
        f"time_dy={getattr(back_bars, 'time_dy_px', None)}",
    )
    back = params.settings_for("montage")
    report.check(
        "profil montage conservé",
        back.texts.title == "Titre montage"
        and abs(back.style.line_width - 2.2) < 0.05
        and back.montage_channels == 7
        and abs(back.style.scale_bar_dx_px - (-20.0)) < 0.05
        and abs(back.style.scale_bar_dy_px - 18.0) < 0.05
        and abs(float(back.style.scale_bar_time_dx_px or 0.0) - (-14.0)) < 0.05
        and abs(float(back.style.scale_bar_time_dy_px or 0.0) - 22.0) < 0.05,
        f"title={back.texts.title} lw={back.style.line_width} ch={back.montage_channels}",
    )
    report.check("aperçu: groupe montage de nouveau masqué", not params._montage_review_group.isVisible())

    # Traitement partagé : build_config, indépendant de la vue.
    shared_before = float(params._threshold.value())
    _set_combo(params._edge, "rising")
    params._threshold.setValue(1.75)
    params._pre_s.setValue(0.4)
    params._post_s.setValue(2.5)
    _set_combo(params._section_spec, "duration")
    params._section_duration.setValue(3.0)
    params._section_count.setValue(4)
    _set_combo(params._hp_filter_type, "bessel")
    params._hp_filter_order.setValue(3)
    params._hp_filter_cutoff.setValue(250.0)
    _set_combo(params._lp_filter_type, "butterworth")
    params._lp_filter_order.setValue(5)
    params._lp_filter_cutoff.setValue(80.0)
    notch_index = params._software_notch.findData(50)
    if notch_index < 0:
        notch_index = 1 if params._software_notch.count() > 1 else 0
    params._software_notch.setCurrentIndex(notch_index)
    params._artifact_enabled.setChecked(True)
    params._artifact_threshold.setValue(400.0)
    _set_combo(params._threshold_mode, "rms_multiple")
    _set_combo(params._polarity, "positive")
    params._spike_threshold.setValue(90.0)
    params._rms_multiplier.setValue(5.5)
    params._rms_window.setValue(0.08)
    params._channel_workers.setValue(3)
    params._work_edit.setText(str(CACHE))
    built = params.build_config(EXAMPLE)
    report.check("traitement: front", built.edge == "rising", built.edge)
    report.check("traitement: seuil trigger", abs(built.threshold - 1.75) < 1e-6, str(built.threshold))
    report.check("traitement: pre/post", abs(built.pre_s - 0.4) < 1e-6 and abs(built.post_s - 2.5) < 1e-6)
    report.check("traitement: section", built.section_spec == "duration" and abs(float(built.section_duration_s or 0) - 3.0) < 1e-6 and built.section_count == 4)
    report.check("traitement: passe-haut", built.intan_hp_filter_type == "bessel" and built.intan_hp_filter_order == 3 and abs(built.intan_hp_filter_cutoff_hz - 250) < 1e-6)
    report.check("traitement: passe-bas", built.intan_lp_filter_type == "butterworth" and built.intan_lp_filter_order == 5 and abs(built.intan_lp_filter_cutoff_hz - 80) < 1e-6)
    report.check("traitement: notch", int(built.software_notch_hz) == int(params._software_notch.currentData() or 0))
    report.check("traitement: artefacts", built.intan_artifact_suppression_enabled and abs(built.intan_artifact_threshold_uv - 400) < 1e-6)
    report.check("traitement: spikes", built.spike_threshold_mode == "rms_multiple" and built.spike_threshold_polarity == "positive" and abs(built.spike_threshold_rms_multiplier - 5.5) < 1e-6 and built.spike_threshold_uv > 0)
    report.check("traitement: fenêtre RMS", abs(built.rms_window_s - 0.08) < 1e-6)
    report.check("traitement: workers", built.channel_workers == 3)
    report.check("traitement: dossier de travail", built.work_dir == CACHE)
    report.check(
        "traitement: PSTH / zoom lus sur la vue active",
        abs(built.psth_bin_window_s - 0.2) < 1e-6 and abs(built.zoom_onset_t0_s + 0.25) < 1e-6,
    )

    # Le changement de traitement a fermé l'enregistrement : on ne rouvre pas
    # la revue, on recharge seulement son profil d'affichage.
    params.set_view_mode("montage", emit=False)
    report.check(
        "traitement: le seuil trigger ne change pas de vue",
        abs(float(params._threshold.value()) - 1.75) < 1e-6 and abs(shared_before - 1.75) > 0.01,
    )
    built_m = params.build_config(EXAMPLE)
    report.check(
        "traitement: filtres identiques en montage",
        built_m.intan_hp_filter_cutoff_hz == built.intan_hp_filter_cutoff_hz
        and built_m.edge == "rising"
        and abs(built_m.threshold - 1.75) < 1e-6,
    )
    report.check(
        "traitement: PSTH du montage != PSTH de l'aperçu",
        abs(built_m.psth_bin_window_s - 0.2) > 1e-6,
        str(built_m.psth_bin_window_s),
    )

    print()
    if report.failures:
        print(f"{len(report.failures)} échec(s):")
        for item in report.failures:
            print(f"  - {item}")
        recording.close()
        return 1
    print("passe exemple.rhs: ok")
    recording.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
