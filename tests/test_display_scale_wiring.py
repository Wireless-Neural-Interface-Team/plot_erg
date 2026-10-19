"""Chaîne réglages → ``ScaleBarSpec`` / plages (sans Qt)."""

from __future__ import annotations

from dataclasses import replace

from panel_prepare import _scale_bar_spec, restyle_panel_spec
from panel_prepare import PanelSpec
from panel_registry import RenderRequest
from view_config import (
    AxisLimits,
    PanelPlacement,
    PanelStyle,
    ViewerSettings,
    apply_local_display_settings,
)


def _request(
    settings: ViewerSettings,
    *,
    panel: str = "full_recording",
) -> RenderRequest:
    return RenderRequest(
        placement=PanelPlacement(panel=panel),
        recordings=[],
        labels=[],
        colors=[],
        legend_flags=[],
        channel_index=0,
        channel_name="A",
        settings=settings,
    )


def test_local_display_settings_propagate_scale_and_plage() -> None:
    base = ViewerSettings()
    local = replace(
        base,
        style=replace(
            base.style,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=77.0,
            scale_bar_time_manual=True,
            scale_bar_time_s=0.25,
            scale_bar_amplitude_length_manual=True,
            scale_bar_amplitude_length_px=55.0,
            scale_bar_time_length_manual=True,
            scale_bar_time_length_px=66.0,
        ),
        x_limits=AxisLimits(enabled=True, minimum=1.0, maximum=9.0),
        raw_ylim=AxisLimits(enabled=True, minimum=-10.0, maximum=10.0),
    )
    merged = apply_local_display_settings(base, local, inherit_scale_bars=False)
    assert merged.style.scale_bar_amplitude == 77.0
    assert merged.style.scale_bar_time_s == 0.25
    assert merged.style.scale_bar_amplitude_length_px == 55.0
    assert merged.style.scale_bar_time_length_px == 66.0
    assert merged.x_limits.as_tuple() == (1.0, 9.0)
    assert merged.raw_ylim.as_tuple() == (-10.0, 10.0)


def test_inherit_scale_bars_overwrites_local_style_from_base() -> None:
    base = replace(
        ViewerSettings(),
        style=replace(
            ViewerSettings().style,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=200.0,
        ),
        graph_height_px=500,
    )
    local = replace(
        ViewerSettings(),
        style=replace(
            ViewerSettings().style,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=50.0,
        ),
        graph_height_px=300,
    )
    inherited = apply_local_display_settings(base, local, inherit_scale_bars=True)
    assert inherited.style.scale_bar_amplitude == 200.0
    assert inherited.graph_height_px == 500


def test_scale_bar_spec_reads_merged_request() -> None:
    settings = replace(
        ViewerSettings(),
        style=replace(
            ViewerSettings().style,
            show_scale_bars=True,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=42.0,
            scale_bar_time_manual=True,
            scale_bar_time_s=0.08,
            scale_bar_amplitude_length_manual=True,
            scale_bar_amplitude_length_px=48.0,
            scale_bar_time_length_manual=True,
            scale_bar_time_length_px=52.0,
        ),
    )
    spec = _scale_bar_spec(_request(settings), panel="full_recording")
    assert spec.enabled is True
    assert spec.floating is True
    assert spec.amplitude == 42.0
    assert spec.time_s == 0.08
    assert spec.amp_length_px == 48.0
    assert spec.time_length_px == 52.0
    assert spec.amp_manual is True
    assert spec.time_manual is True


def test_axis_ticks_mode_when_floating_off() -> None:
    settings = replace(
        ViewerSettings(),
        style=replace(ViewerSettings().style, show_scale_bars=False),
    )
    spec = _scale_bar_spec(_request(settings), panel="analysis_raw")
    assert spec.enabled is True
    assert spec.floating is False


def test_restyle_updates_scale_bars_without_reprepare() -> None:
    cached = PanelSpec(
        title="t",
        status="ok",
        layout_kind="single",
        series=[],
        scale_bars=_scale_bar_spec(
            _request(ViewerSettings()), panel="full_recording"
        ),
    )
    new_settings = replace(
        ViewerSettings(),
        style=replace(
            ViewerSettings().style,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=99.0,
        ),
    )
    out = restyle_panel_spec(cached, _request(new_settings))
    assert out.scale_bars.amplitude == 99.0
    assert out.scale_bars.amp_manual is True


def test_panel_override_path_uses_local_not_dock() -> None:
    """Simule ``PanelCanvas._with_style_overrides`` (inherit_scale_bars=False)."""
    dock = replace(
        ViewerSettings(),
        style=replace(
            ViewerSettings().style,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=1000.0,
        ),
    )
    override = replace(
        ViewerSettings(),
        style=replace(
            ViewerSettings().style,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=12.5,
        ),
    )
    merged = apply_local_display_settings(dock, override, inherit_scale_bars=False)
    bars = _scale_bar_spec(_request(merged), panel="full_recording")
    assert bars.amplitude == 12.5


def test_fitted_width_includes_axes_outside_viewbox() -> None:
    """Pas de largeur tant que la zone de tracé n’a pas intégré axes et libellés."""
    from gui.widgets.panel_canvas import _accept_fit_sample, _fitted_width_px

    # Géométrie déjà calée : chrome 100 px + zone 450 px.
    assert _fitted_width_px(1000, 900, 450, explicit_chrome=80) == 550
    # ViewBox pas encore rétrécie des axes : ne pas poser une taille intermédiaire.
    assert _fitted_width_px(1000, 980, 450, explicit_chrome=125) is None
    # Dernier recours : compter les axes connus, une seule fois.
    assert _fitted_width_px(1000, 980, 450, explicit_chrome=125, strict=False) == 575
    # ViewBox encore plus large que le widget : géométrie périmée, ne pas appliquer.
    assert _fitted_width_px(550, 900, 450, explicit_chrome=125) is None
    assert _fitted_width_px(550, 900, 450, explicit_chrome=125, strict=False) is None
    # Rien de mesurable : ne pas inventer une largeur écrasée.
    assert _fitted_width_px(550, 900, 450) is None
    # Deux mesures différentes : attendre. La même valeur deux fois : accepter.
    state, sample = _accept_fit_sample(None, "ready", 575, force=False)
    assert state == "wait" and sample == 575
    state, sample = _accept_fit_sample(sample, "ready", 575, force=False)
    assert state == "ready" and sample == 575


def test_zoom_panel_height_helpers() -> None:
    from gui.widgets.panel_canvas import (
        initial_zoom_panel_height,
        zoom_grid_panel_height,
        zoom_scale_bar_amp_fit_active,
        zoom_scale_bar_widget_fit_active,
    )

    auto = ViewerSettings()
    assert zoom_grid_panel_height(auto) == max(160, int(auto.graph_height_px))
    assert initial_zoom_panel_height(auto) == zoom_grid_panel_height(auto)
    assert not zoom_scale_bar_amp_fit_active(auto)

    fit = replace(
        ViewerSettings(),
        style=replace(
            ViewerSettings().style,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=50.0,
            scale_bar_amplitude_length_manual=True,
            scale_bar_amplitude_length_px=80.0,
        ),
        graph_height_px=420,
    )
    assert zoom_scale_bar_amp_fit_active(fit)
    assert zoom_scale_bar_widget_fit_active(fit)
    assert zoom_grid_panel_height(fit) is None
    assert initial_zoom_panel_height(fit) == 420


def test_spike_scope_uses_its_own_manual_scale() -> None:
    """L’échelle saisie dans Spikes aboutit au spike scope, pas aux traces."""
    from gui.widgets.plot_host.mixins.scale_bars import (
        _manual_x_params,
        _manual_y_params,
    )
    from panel_registry import _scale_bar_lengths

    settings = replace(
        ViewerSettings(),
        style=replace(
            ViewerSettings().style,
            show_scale_bars=True,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=100.0,
            scale_bar_amplitude_length_manual=True,
            scale_bar_amplitude_length_px=40.0,
            scale_bar_time_manual=True,
            scale_bar_time_s=0.2,
            scale_bar_time_length_manual=True,
            scale_bar_time_length_px=40.0,
            spike_scale_amplitude_manual=True,
            spike_scale_amplitude=25.0,
            spike_scale_amplitude_length_px=80.0,
            spike_scale_time_manual=True,
            spike_scale_time_ms=2.0,
            spike_scale_time_length_px=60.0,
        ),
    )
    scope = _scale_bar_spec(
        _request(settings, panel="analysis_overlay"), panel="analysis_overlay"
    )
    traces = _scale_bar_spec(_request(settings), panel="full_recording")
    assert scope.x_unit == "ms"
    assert scope.amp_manual is False and scope.time_manual is False
    assert scope.scope_amp_manual is True
    assert scope.scope_amplitude == 25.0
    assert scope.scope_amp_length_px == 80.0
    assert scope.scope_time_manual is True
    assert abs(scope.scope_time_s - 0.002) < 1e-9
    assert scope.scope_time_length_px == 60.0
    y_val, y_len = _manual_y_params(scope, "µV", shared=False)
    x_val, x_len = _manual_x_params(scope, "ms", shared=False)
    assert y_val == 25.0 and y_len == 80.0
    assert x_val is not None and abs(x_val - 2.0) < 1e-6 and x_len == 60.0
    trace_y, _trace_y_len = _manual_y_params(traces, "µV", shared=True)
    assert trace_y == 100.0

    class _Ax:
        def get_xlim(self) -> tuple[float, float]:
            return (-2.0, 4.0)

        def get_ylim(self) -> tuple[float, float]:
            return (-80.0, 80.0)

    x_size, y_size = _scale_bar_lengths(
        settings.style, x_unit="ms", y_unit="µV", ax=_Ax(), shared=False
    )
    assert y_size == 25.0
    assert x_size == 2.0
    # Les traces gardent leur propre gain.
    x_trace, y_trace = _scale_bar_lengths(
        settings.style, x_unit="s", y_unit="µV", ax=_Ax(), shared=True
    )
    assert y_trace == 100.0
    assert x_trace == 0.2
    from gui.widgets.panel_canvas import _scale_fit_signature

    base = ViewerSettings()
    assert _scale_fit_signature(settings) != _scale_fit_signature(base)


def test_spike_scope_scale_ignores_shared_manual_gain() -> None:
    """Le spike scope ne reprend pas le gain µV/px ni ms/px des traces."""
    from gui.widgets.panel_canvas import y_scale_family
    from scale_bar_model import uses_shared_manual_scale

    settings = replace(
        ViewerSettings(),
        style=replace(
            ViewerSettings().style,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=100.0,
            scale_bar_amplitude_length_manual=True,
            scale_bar_amplitude_length_px=40.0,
            scale_bar_time_manual=True,
            scale_bar_time_s=0.1,
            scale_bar_time_length_manual=True,
            scale_bar_time_length_px=40.0,
        ),
    )
    scope = _scale_bar_spec(
        _request(settings, panel="analysis_overlay"), panel="analysis_overlay"
    )
    traces = _scale_bar_spec(_request(settings), panel="full_recording")
    assert scope.x_unit == "ms" and scope.y_unit == "µV"
    assert scope.amp_manual is False
    assert scope.time_manual is False
    assert scope.amp_length_manual is False
    assert scope.time_length_manual is False
    assert traces.amp_manual is True
    assert traces.time_manual is True
    assert uses_shared_manual_scale(kind="overlay") is False
    assert uses_shared_manual_scale(kind="hp") is True
    assert y_scale_family("analysis_overlay") == ""
    assert y_scale_family("analysis_hp") == "stream:hp"


def test_scale_fit_signature_tracks_font_size_and_face() -> None:
    """Police (taille, gras, italique, souligné) : le widget se recale, pas le tracé."""
    from gui.widgets.panel_canvas import _scale_fit_signature
    from view_config import TextFace

    base = ViewerSettings()
    sig = _scale_fit_signature(base)

    def style(**kwargs: object) -> ViewerSettings:
        return replace(base, style=replace(base.style, **kwargs))

    assert _scale_fit_signature(style(title_font_size=16.0)) != sig
    assert _scale_fit_signature(style(label_font_size=13.0)) != sig
    assert _scale_fit_signature(style(tick_font_size=18.0)) != sig
    assert _scale_fit_signature(style(title_face=TextFace(bold=True))) != sig
    assert _scale_fit_signature(style(label_face=TextFace(italic=True))) != sig
    assert _scale_fit_signature(style(tick_face=TextFace(underline=True))) != sig
    assert (
        _scale_fit_signature(
            style(stim_on_font_size=14.0, stim_on_face=TextFace(bold=True))
        )
        != sig
    )
    assert (
        _scale_fit_signature(
            style(stim_off_font_size=11.0, stim_off_face=TextFace(italic=True))
        )
        != sig
    )
    assert (
        _scale_fit_signature(
            style(zone_font_size=12.0, zone_face=TextFace(underline=True))
        )
        != sig
    )
    legend = replace(
        base.legend, font_size=15.0, face=TextFace(bold=True, underline=True)
    )
    assert _scale_fit_signature(replace(base, legend=legend)) != sig
    # Placement : la gouttière change, le widget se recale, pas la zone de tracé.
    assert _scale_fit_signature(style(scale_bar_dx_px=40.0)) != sig
    assert _scale_fit_signature(style(scale_bar_dy_px=-24.0)) != sig
    assert _scale_fit_signature(style(scale_bar_time_dx_px=-16.0)) != sig
    assert _scale_fit_signature(style(scale_bar_time_dy_px=30.0)) != sig
    # Un réglage sans effet sur le chrome ne relance pas le calage.
    assert _scale_fit_signature(style(line_width=3.0)) == sig


def test_spike_scope_ylim_stays_per_channel() -> None:
    """Même échelle montage : les traces se partagent, pas le spike scope."""
    import numpy as np

    from panel_prepare import CurveSeries, _apply_montage_shared_ylims

    settings = replace(ViewerSettings(), montage_shared_ylim=True)
    series = [
        CurveSeries(
            x=np.array([0.0, 1.0]), y=np.array([-10.0, 10.0]), color="#000", row=0
        ),
        CurveSeries(
            x=np.array([0.0, 1.0]), y=np.array([-2.0, 2.0]), color="#000", row=1
        ),
        CurveSeries(
            x=np.array([0.0, 1.0]), y=np.array([-100.0, 100.0]), color="#000", row=2
        ),
        CurveSeries(
            x=np.array([0.0, 1.0]), y=np.array([-40.0, 40.0]), color="#000", row=3
        ),
    ]
    out = _apply_montage_shared_ylims(
        settings, ["hp", "overlay", "hp", "overlay"], [None, None, None, None], series
    )
    assert out[0] == out[2]
    assert out[0] == (-100.0, 100.0)
    assert out[1] is None
    assert out[3] is None


def test_spike_scope_manual_scale_is_drawn() -> None:
    """Le bâton du spike scope affiche la valeur saisie, pas celle des traces."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from PySide6.QtWidgets import QApplication, QGraphicsSimpleTextItem

    from gui.widgets.plot_host.host import PlotHost
    from panel_prepare import CurveSeries, PanelSpec

    app = QApplication.instance() or QApplication([])
    settings = replace(
        ViewerSettings(),
        style=replace(
            ViewerSettings().style,
            show_scale_bars=True,
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=100.0,
            scale_bar_amplitude_length_manual=True,
            scale_bar_amplitude_length_px=40.0,
            spike_scale_amplitude_manual=True,
            spike_scale_amplitude=25.0,
            spike_scale_amplitude_length_px=80.0,
            spike_scale_time_manual=True,
            spike_scale_time_ms=2.0,
            spike_scale_time_length_px=60.0,
        ),
    )
    bars = _scale_bar_spec(
        _request(settings, panel="analysis_overlay"), panel="analysis_overlay"
    )
    x = np.linspace(-2.0, 4.0, 64)
    spec = PanelSpec(
        title="Spike scope",
        xlabel="Temps (ms)",
        ylabel="µV",
        series=[
            CurveSeries(x=x, y=np.sin(x) * 30.0, color="#1f77b4", label="spike", row=0)
        ],
        xlim=(-2.0, 4.0),
        ylim=(-80.0, 80.0),
        layout_kind="single",
        n_rows=1,
        row_kinds=["overlay"],
        show_legend=False,
        scale_bars=bars,
    )
    host = PlotHost()
    host.resize(720, 480)
    host.show()
    host.render_spec(spec)
    app.processEvents()
    host._refresh_scale_bars()
    app.processEvents()
    labels = []
    for _plot, item in host._scale_bar_items:
        if isinstance(item, QGraphicsSimpleTextItem):
            labels.append(item.text())
    assert "25µV" in labels, labels
    assert "2ms" in labels, labels
    host.close()
