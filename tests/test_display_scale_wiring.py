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
