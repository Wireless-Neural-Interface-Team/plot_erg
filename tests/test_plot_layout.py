"""Modèle de layout : XOR taille/échelle et chrome additif."""

from __future__ import annotations

from dataclasses import replace

from plot_layout import (
    height_mode_for,
    resolve_graph_area_size,
    widget_size_for_plot_area,
    width_mode_for,
)
from view_config import PanelStyle, ViewerSettings, scale_bar_drives_widget_width


def test_height_xor_fixed_vs_scale() -> None:
    fixed = ViewerSettings(graph_height_px=320)
    assert height_mode_for(fixed) == "fixed"
    area = resolve_graph_area_size(fixed, montage=False)
    assert area.height_px == 320
    assert area.height_mode == "fixed"

    scaled = replace(
        fixed,
        style=replace(
            PanelStyle(),
            scale_bar_amplitude_manual=True,
            scale_bar_amplitude=50.0,
            scale_bar_amplitude_length_manual=True,
            scale_bar_amplitude_length_px=40.0,
        ),
    )
    assert height_mode_for(scaled) == "scale"
    area_s = resolve_graph_area_size(
        scaled, montage=False, scale_needed_height_px=220
    )
    assert area_s.height_px == 220
    assert area_s.height_mode == "scale"


def test_width_xor_fill_fixed_scale() -> None:
    fill = ViewerSettings(graph_width_px=0)
    assert width_mode_for(fill) == "fill"
    assert resolve_graph_area_size(fill).width_px is None

    fixed = ViewerSettings(graph_width_px=640)
    assert width_mode_for(fixed) == "fixed"
    assert resolve_graph_area_size(fixed).width_px == 640

    scaled = replace(
        fixed,
        style=replace(
            PanelStyle(),
            scale_bar_time_manual=True,
            scale_bar_time_s=0.2,
            scale_bar_time_length_manual=True,
            scale_bar_time_length_px=80.0,
        ),
    )
    assert scale_bar_drives_widget_width(scaled)
    assert width_mode_for(scaled) == "scale"
    area = resolve_graph_area_size(
        scaled, scale_needed_width_px=500
    )
    assert area.width_px == 500


def test_montage_fixed_height_uses_row_min() -> None:
    settings = ViewerSettings(montage_row_min_height_px=60, graph_height_px=400)
    area = resolve_graph_area_size(settings, montage=True, n_rows=3)
    assert area.height_px == 180
    assert area.height_mode == "fixed"


def test_widget_grows_outward_when_chrome_grows() -> None:
    # ViewBox 300, chrome 100 → widget 400. Cible ViewBox 300 → widget reste 400.
    h, w = widget_size_for_plot_area(
        plot_area_h=300,
        plot_area_w=500,
        widget_h=400,
        widget_w=600,
        target_h=300,
        target_w=500,
    )
    assert h == 400
    assert w == 600
    # Chrome vertical +50 (stim) : ViewBox tombe à 250 → widget doit passer à 450.
    h2, _w2 = widget_size_for_plot_area(
        plot_area_h=250,
        plot_area_w=500,
        widget_h=400,
        widget_w=600,
        target_h=300,
        target_w=None,
    )
    assert h2 == 450
