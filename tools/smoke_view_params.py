"""Vérifie que les réglages d’aperçu et de revue montage atteignent le dessin.

    .venv\\Scripts\\python.exe tools\\smoke_view_params.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.widgets.plot_host import PlotHost
from make_synthetic_dataset import build_synthetic
from panel_prepare import prepare_panel, screen_prepare_key
from panel_registry import PanelPlacement, RenderRequest
from view_config import (
    AxisLimits,
    LegendSettings,
    PanelStyle,
    TextOverrides,
    ViewerSettings,
)


def _fail(message: str) -> None:
    raise SystemExit(message)


def _recording():
    recording = build_synthetic(label="params", seed=3, n_channels=4, n_trials=4)
    n_ch = int(recording.n_channels)
    n_samp = int(recording.meta.n_samples)
    rng = np.random.default_rng(1)
    amp = rng.standard_normal((n_ch, n_samp)).astype(np.float32)
    recording.source = SimpleNamespace(
        amplifier=amp, highpass=amp * 0.4, lowpass=amp * 0.7
    )
    return recording


def _request(
    recording,
    panel: str,
    settings: ViewerSettings,
    *,
    stream: str = "",
) -> RenderRequest:
    placement = PanelPlacement(panel, stream=stream) if stream else PanelPlacement(panel)
    return RenderRequest(
        placement=placement,
        recordings=[recording],
        labels=[recording.label],
        colors=["#1f77b4"],
        legend_flags=[True],
        channel_index=0,
        channel_name=recording.channel_names[0],
        settings=settings,
    )


def _roles(spec) -> set[str]:
    return {str(getattr(ann, "role", "") or "") for ann in spec.annotations}


def _check_prepare(recording) -> None:
    base = ViewerSettings(
        continuous_streams=("raw", "hp"),
        continuous_stream="raw",
        montage_review_channels=2,
        style=PanelStyle(line_width=2.4, stim_linewidth=3.0, stim_onset_color="#112233", stim_offset_color="#abcdef"),
    )
    preview = prepare_panel(_request(recording, "full_recording", base))
    montage = prepare_panel(_request(recording, "montage_continuous_raw", base))
    if preview.status != "ok" or montage.status != "ok":
        _fail(f"rendu initial: preview={preview.status} montage={montage.status}")

    if "stim_onset" not in _roles(preview) or "stim_offset" not in _roles(preview):
        _fail(f"aperçu sans début et fin de stim: {_roles(preview)}")
    if "stim_onset" not in _roles(montage) or "stim_offset" not in _roles(montage):
        _fail(f"montage sans début et fin de stim: {_roles(montage)}")
    onset = next(a for a in preview.annotations if a.role == "stim_onset")
    offset = next(a for a in preview.annotations if a.role == "stim_offset")
    if str(onset.color).lower() != "#112233" or str(offset.color).lower() != "#abcdef":
        _fail(f"couleurs stim aperçu: {onset.color} {offset.color}")
    if abs(float(preview.stim_linewidth) - 3.0) > 1e-6:
        _fail(f"épaisseur stim aperçu: {preview.stim_linewidth}")
    if abs(float(preview.series[0].linewidth) - 2.4) > 1e-6:
        _fail(f"épaisseur courbe aperçu: {preview.series[0].linewidth}")

    no_marks = replace(base, continuous_mark_stims=False)
    if _roles(prepare_panel(_request(recording, "full_recording", no_marks))) & {
        "stim_onset",
        "stim_offset",
    }:
        _fail("décocher les marqueurs continus laisse des pointillés en aperçu")
    if _roles(prepare_panel(_request(recording, "montage_continuous_raw", no_marks))) & {
        "stim_onset",
        "stim_offset",
    }:
        _fail("décocher les marqueurs continus laisse des pointillés en montage")

    no_dots = replace(base, legend=replace(base.legend, show_reference_markers=False))
    if _roles(prepare_panel(_request(recording, "full_recording", no_dots))) & {
        "stim_onset",
        "stim_offset",
    }:
        _fail("décocher les pointillés laisse des marques en aperçu")
    if _roles(prepare_panel(_request(recording, "montage_continuous_raw", no_dots))) & {
        "stim_onset",
        "stim_offset",
    }:
        _fail("décocher les pointillés laisse des marques en montage")

    keyed = _request(recording, "full_recording", base)
    if screen_prepare_key(keyed) == screen_prepare_key(
        _request(recording, "full_recording", no_marks)
    ):
        _fail("le cache aperçu ignore les marqueurs de stimulation")
    if screen_prepare_key(keyed) == screen_prepare_key(
        _request(recording, "full_recording", no_dots)
    ):
        _fail("le cache aperçu ignore les pointillés")

    ylabel = replace(base, texts=TextOverrides(ylabel="Amplitude custom"))
    preview_y = prepare_panel(_request(recording, "full_recording", ylabel))
    if preview_y.row_labels != ["Amplitude custom", "Amplitude custom"]:
        _fail(f"axe Y aperçu ignoré: {preview_y.row_labels}")
    montage_y = prepare_panel(_request(recording, "montage_continuous_raw", ylabel))
    if any(label == "Amplitude custom" for label in montage_y.row_labels):
        _fail("l’axe Y a remplacé les noms de canal du montage")

    filters_off = replace(base, legend=replace(base.legend, show_filter_details=False))
    with_filters = prepare_panel(_request(recording, "full_recording", base, stream="hp"))
    without_filters = prepare_panel(
        _request(recording, "full_recording", filters_off, stream="hp")
    )
    label_on = str(with_filters.series[0].label)
    label_off = str(without_filters.series[0].label)
    if "bessel" not in label_on.lower():
        _fail(f"détails de filtre absents: {label_on}")
    if "bessel" in label_off.lower():
        _fail(f"détails de filtre encore présents: {label_off}")
    if screen_prepare_key(_request(recording, "full_recording", base, stream="hp")) == screen_prepare_key(
        _request(recording, "full_recording", filters_off, stream="hp")
    ):
        _fail("le cache ignore les détails de filtre")

    counts_off = replace(base, legend=replace(base.legend, show_sample_counts=False))
    counted = str(prepare_panel(_request(recording, "full_recording", base)).series[0].label)
    uncounted = str(
        prepare_panel(_request(recording, "full_recording", counts_off)).series[0].label
    )
    if "(n=" not in counted:
        _fail(f"compteur d’échantillons absent: {counted}")
    if "(n=" in uncounted:
        _fail(f"compteur encore présent: {uncounted}")

    limited = replace(
        base, x_limits=AxisLimits(enabled=True, minimum=0.2, maximum=0.8)
    )
    xlim = prepare_panel(_request(recording, "full_recording", limited)).xlim
    if xlim is None or abs(xlim[0] - 0.2) > 1e-6 or abs(xlim[1] - 0.8) > 1e-6:
        _fail(f"axe X aperçu ignoré: {xlim}")
    montage_x = prepare_panel(_request(recording, "montage_continuous_raw", limited))
    if montage_x.xlim is None or abs(montage_x.xlim[0] - 0.2) > 1e-6:
        _fail(f"axe X montage ignoré: {montage_x.xlim}")

    synced = replace(base, time_sync="trigger")
    if "trigger" not in prepare_panel(_request(recording, "full_recording", synced)).xlabel.lower():
        _fail("origine du temps ignorée en aperçu")
    if "trigger" not in prepare_panel(
        _request(recording, "montage_continuous_raw", synced)
    ).xlabel.lower():
        _fail("origine du temps ignorée en montage")

    kind_off = replace(base, montage_show_kind_label=False)
    labels_on = prepare_panel(_request(recording, "montage_continuous_raw", base)).row_labels
    labels_off = prepare_panel(
        _request(recording, "montage_continuous_raw", kind_off)
    ).row_labels
    if not any("WIDE" in label or "HIGH" in label for label in labels_on):
        _fail(f"type de flux absent: {labels_on}")
    if any("WIDE" in label or "HIGH" in label for label in labels_off):
        _fail(f"type de flux encore affiché: {labels_off}")

    shifted = replace(base, montage_label_dx_px=40.0, montage_label_dy_px=-12.0)
    shifted_spec = prepare_panel(_request(recording, "montage_continuous_raw", shifted))
    if shifted_spec.row_label_dx_px != 40.0 or shifted_spec.row_label_dy_px != -12.0:
        _fail("décalage des titres montage ignoré")

    gapped = replace(base, preview_title_gap_px=48)
    if prepare_panel(_request(recording, "full_recording", gapped)).title_gap_px != 48:
        _fail("écart titre / courbe ignoré")

    titled = replace(base, texts=TextOverrides(title="Mon titre", xlabel="t custom"))
    titled_spec = prepare_panel(_request(recording, "full_recording", titled))
    if titled_spec.title != "Mon titre" or titled_spec.xlabel != "t custom":
        _fail(f"titre / axe X aperçu ignorés: {titled_spec.title!r} {titled_spec.xlabel!r}")
    montage_title = prepare_panel(_request(recording, "montage_continuous_raw", titled))
    if montage_title.title != "Mon titre" or montage_title.xlabel != "t custom":
        _fail("titre / axe X montage ignorés")

    bars = replace(
        base,
        style=replace(
            base.style,
            show_scale_bars=True,
            scale_bar_time_manual=True,
            scale_bar_time_s=0.25,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=50.0,
            scale_bar_linewidth=2.5,
            grid=False,
            grid_alpha=0.05,
        ),
    )
    bar_spec = prepare_panel(_request(recording, "full_recording", bars))
    if not bar_spec.scale_bars.enabled or abs(bar_spec.scale_bars.time_s - 0.25) > 1e-9:
        _fail("barres d’échelle aperçu ignorées")
    if abs(bar_spec.scale_bars.amplitude - 50.0) > 1e-9 or abs(bar_spec.scale_bars.linewidth - 2.5) > 1e-9:
        _fail("amplitude / épaisseur des barres d’échelle ignorées")
    if bar_spec.grid or abs(bar_spec.grid_alpha - 0.05) > 1e-9:
        _fail("grille aperçu ignorée")
    montage_bars = prepare_panel(_request(recording, "montage_continuous_raw", bars))
    if not montage_bars.scale_bars.enabled or montage_bars.grid:
        _fail("barres d’échelle ou grille montage ignorées")

    hidden = replace(base, legend=replace(base.legend, visible=False, frame=False, columns=3))
    if prepare_panel(_request(recording, "full_recording", hidden)).show_legend:
        _fail("masquer la légende est ignoré en aperçu")
    if prepare_panel(_request(recording, "montage_continuous_raw", hidden)).show_legend:
        _fail("masquer la légende est ignoré en montage")
    legend_style = prepare_panel(_request(recording, "full_recording", hidden))
    if legend_style.legend_columns != 3 or legend_style.legend_frame:
        _fail("colonnes / cadre de légende ignorés")

    dense = prepare_panel(
        _request(recording, "full_recording", replace(base, style=replace(base.style, max_points_per_curve=80_000)))
    )
    sparse = prepare_panel(
        _request(recording, "full_recording", replace(base, style=replace(base.style, max_points_per_curve=800)))
    )
    if sparse.series[0].x.size >= dense.series[0].x.size:
        _fail("points max / courbe sans effet en aperçu")

    page = prepare_panel(
        _request(recording, "montage_continuous_raw", replace(base, montage_review_channels=1))
    )
    if page.n_rows != 2:
        _fail(f"canaux visibles: attendu 2 lignes (1 canal × 2 flux), obtenu {page.n_rows}")

    print("prepare: ok")


def _pen_is_none(axis) -> bool:
    pen = axis.pen() if callable(getattr(axis, "pen", None)) else None
    if pen is None:
        return False
    return pen.style() == Qt.PenStyle.NoPen


def _check_plot_host() -> None:
    app = QApplication.instance() or QApplication([])
    host = PlotHost()
    host.resize(800, 500)
    x = np.linspace(0.0, 1.0, 32)
    y = np.sin(x * 6.0)

    def spec(*, montage: bool, borders: bool, ticks_inside: bool, gap: float, title_gap: int, dx: float):
        from panel_prepare import CurveSeries, PanelSpec

        n_rows = 2 if montage else 1
        series = []
        for row in range(n_rows):
            series.append(
                CurveSeries(
                    x=x,
                    y=y + row,
                    color="#1f77b4",
                    label="trace" if row == 0 else "",
                    linewidth=1.2,
                    row=row,
                )
            )
        return PanelSpec(
            title="Titre",
            xlabel="Temps (s)",
            ylabel="µV",
            series=series,
            xlim=(0.0, 1.0),
            ylim=(-2.0, 4.0),
            layout_kind="montage" if montage else "single",
            n_rows=n_rows,
            row_labels=["A-000 WIDE", "A-000 HIGH"] if montage else ["µV"],
            row_ylims=[(-2.0, 4.0)] * n_rows,
            row_kinds=["raw", "hp"] if montage else ["raw"],
            row_label_dx_px=dx,
            row_label_dy_px=8.0 if montage else 0.0,
            show_borders=borders,
            ticks_inside=ticks_inside,
            show_legend=True,
            legend_gap=gap,
            title_gap_px=title_gap,
            grid=True,
        )

    host.render_spec(spec(montage=False, borders=True, ticks_inside=False, gap=0.1, title_gap=0, dx=0))
    gap0 = int(host.plot_items()[0].layout.rowSpacing(0))
    host.render_spec(spec(montage=False, borders=False, ticks_inside=True, gap=0.8, title_gap=40, dx=0))
    gap40 = int(host.plot_items()[0].layout.rowSpacing(0))
    if gap40 <= gap0:
        _fail(f"écart titre / courbe sans effet: {gap0} → {gap40}")
    app.processEvents()
    plot = host.plot_items()[0]
    if not _pen_is_none(plot.getAxis("bottom")):
        _fail("bordures aperçu: le second rendu laisse le cadre")
    tick = int(plot.getAxis("left").style.get("tickLength", 0))
    if tick >= 0:
        _fail(f"graduations intérieures aperçu ignorées: {tick}")
    spacing = int(host._glw.ci.layout.rowSpacing(0))
    if spacing < 80:
        _fail(f"écart légende aperçu trop petit: {spacing}")
    title = plot.titleLabel
    row_h = int(plot.layout.rowFixedHeight(0)) if hasattr(plot.layout, "rowFixedHeight") else -1
    # setRowFixedHeight is what the host applies; read it back via the layout item if needed.
    del title, row_h
    if int(getattr(host._spec, "title_gap_px", 0) or 0) != 40:
        _fail("écart titre non conservé sur le spec dessiné")

    host.render_spec(spec(montage=True, borders=True, ticks_inside=False, gap=0.05, title_gap=0, dx=0))
    host.render_spec(spec(montage=True, borders=False, ticks_inside=True, gap=0.05, title_gap=0, dx=36))
    app.processEvents()
    montage_plot = host.plot_items()[0]
    if not _pen_is_none(montage_plot.getAxis("left")):
        _fail("bordures montage: le rafraîchissement laisse le cadre")
    montage_tick = int(montage_plot.getAxis("left").style.get("tickLength", 0))
    if montage_tick >= 0:
        _fail(f"graduations intérieures montage ignorées: {montage_tick}")
    if not host._row_labels or abs(float(host._row_labels[0]._dx) - 36.0) > 1e-6:
        _fail("décalage horizontal des titres montage non appliqué au rafraîchissement")
    print("plot host: ok")
    app.quit()


def _check_param_profiles() -> None:
    """Aperçu et montage gardent des valeurs distinctes, mêmes noms de réglages."""
    from gui.widgets.params_panel import ParamsPanel

    app = QApplication.instance() or QApplication([])
    panel = ParamsPanel()
    panel.resize(420, 800)
    panel.show()
    panel.tabs.setCurrentIndex(1)
    app.processEvents()

    panel._line_width.setValue(4.0)
    panel._graph_height.setValue(520)
    panel._curve_high_box.setChecked(True)
    panel._text_title.setText("Titre aperçu")
    panel.set_content_mode("average", 1, emit=False)
    preview = panel.viewer_settings()
    if abs(preview.style.line_width - 4.0) > 1e-6:
        _fail("épaisseur aperçu non lue")
    if panel._montage_review_group.isVisible():
        _fail("la revue montage est visible dans l’aperçu")
    if not panel._graph_height.isVisible():
        _fail("la hauteur des graphs est masquée dans l’aperçu")

    panel.set_view_mode("montage", emit=False)
    app.processEvents()
    montage = panel.viewer_settings()
    if abs(montage.style.line_width - 4.0) < 1e-6:
        _fail("l’épaisseur aperçu a fuité dans le montage")
    if montage.preview_content != "continuous":
        _fail("le mode aperçu a fuité dans le montage")
    if montage.analysis.show_hp:
        _fail("la courbe HIGH aperçu a fuité dans le montage")
    if panel._graph_height.isVisible():
        _fail("la hauteur des graphs est visible en montage")
    if not panel._montage_review_group.isVisible():
        _fail("la revue montage est masquée en montage")
    if not panel._montage_pdf_group.isVisible():
        _fail("les montages PDF sont masqués en montage")

    panel._line_width.setValue(2.2)
    panel._montage_row_height.setValue(80)
    panel._curve_low_box.setChecked(True)
    panel._text_title.setText("Titre montage")
    panel.set_content_mode("stimulation", 4, emit=False)
    threshold = float(panel._threshold.value())
    panel._threshold.setValue(threshold + 1.0)
    shared = float(panel._threshold.value())

    panel.set_view_mode("preview", emit=False)
    app.processEvents()
    back = panel.viewer_settings()
    if abs(back.style.line_width - 4.0) > 1e-6:
        _fail(f"retour aperçu : épaisseur {back.style.line_width}")
    if back.texts.title != "Titre aperçu":
        _fail(f"retour aperçu : titre {back.texts.title!r}")
    if back.preview_content != "average" or back.analysis.stim_index != 1:
        _fail("retour aperçu : mode perdu")
    if not back.analysis.show_hp or back.analysis.show_lp:
        _fail("retour aperçu : courbes mélangées avec le montage")
    if int(back.graph_height_px) != 520:
        _fail("retour aperçu : hauteur des graphs perdue")
    if abs(float(panel._threshold.value()) - shared) > 1e-6:
        _fail("le seuil de déclenchement (traitement commun) a été séparé")
    if panel._montage_review_group.isVisible():
        _fail("la revue montage réapparaît dans l’aperçu")

    panel.set_view_mode("montage", emit=False)
    app.processEvents()
    again = panel.viewer_settings()
    if abs(again.style.line_width - 2.2) > 1e-6:
        _fail("retour montage : épaisseur perdue")
    if again.texts.title != "Titre montage":
        _fail("retour montage : titre perdu")
    if again.preview_content != "stimulation" or again.analysis.stim_index != 4:
        _fail("retour montage : mode perdu")
    if not again.analysis.show_lp or again.analysis.show_hp:
        _fail("retour montage : courbes mélangées avec l’aperçu")
    if int(again.montage_row_min_height_px) != 80:
        _fail("retour montage : hauteur de ligne perdue")
    stored_preview = panel.settings_for("preview")
    if abs(stored_preview.style.line_width - 4.0) > 1e-6:
        _fail("le profil aperçu stocké a été écrasé par le montage")
    if abs(float(panel._threshold.value()) - shared) > 1e-6:
        _fail("le seuil commun a changé en revenant au montage")

    panel.close()
    print("profiles: ok")


def main() -> int:
    recording = _recording()
    _check_prepare(recording)
    _check_param_profiles()
    _check_plot_host()
    print("smoke_view_params: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
