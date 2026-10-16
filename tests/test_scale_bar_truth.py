"""Modes d’échelle : valeur seule / taille seule / les deux."""

from __future__ import annotations

import math

from plot_utils import (
    format_scale_label,
    resolve_scale_bar_length,
    scale_bar_label_value,
    scale_bar_pixel_length,
    scale_bar_true_value,
)


def _assert_close(a: float, b: float, *, tol: float = 1e-9) -> None:
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a - b) <= tol * max(1.0, abs(b)), (a, b)


def test_value_and_length_bar_follows_amplitude() -> None:
    """Taille + amplitude : bâton proportionnel (échelle vraie), libellé = amp.

    ``length_px`` ne force plus un bâton décorrélé — il cale le widget.
    """
    span, axis, amp, length_px = 200.0, 400.0, 50.0, 40.0
    bar = scale_bar_pixel_length(
        data_value=amp,
        span=span,
        axis_px=axis,
        length_px=length_px,
        value_manual=True,
    )
    # Proportionnel : 50/200 * 400 = 100 px (pas 40).
    _assert_close(bar, 100.0)
    label = scale_bar_label_value(
        data_value=amp,
        span=span,
        axis_px=axis,
        bar_px=bar,
        value_manual=True,
        length_manual=True,
    )
    _assert_close(label, amp)
    _assert_close(scale_bar_true_value(bar, span=span, axis_px=axis), amp)


def test_length_only_calculates_label() -> None:
    """Taille px seule → bâton fixe, libellé calculé."""
    span, axis, nice, length_px = 100.0, 400.0, 50.0, 40.0
    bar = scale_bar_pixel_length(
        data_value=nice,
        span=span,
        axis_px=axis,
        length_px=length_px,
        value_manual=False,
    )
    _assert_close(bar, 40.0)
    label = scale_bar_label_value(
        data_value=nice,
        span=span,
        axis_px=axis,
        bar_px=bar,
        value_manual=False,
        length_manual=True,
    )
    _assert_close(label, 10.0)
    assert format_scale_label(label, "µV") == "10µV"


def test_amplitude_only_proportional() -> None:
    """Amplitude seule → bâton proportionnel, libellé = amplitude."""
    span, axis, amp = 100.0, 400.0, 50.0
    bar = scale_bar_pixel_length(
        data_value=amp, span=span, axis_px=axis, length_px=None, value_manual=True
    )
    label = scale_bar_label_value(
        data_value=amp,
        span=span,
        axis_px=axis,
        bar_px=bar,
        value_manual=True,
        length_manual=False,
    )
    _assert_close(label, amp)
    _assert_close(scale_bar_true_value(bar, span=span, axis_px=axis), amp)


def test_both_with_matched_widget_height() -> None:
    """Après fit hauteur, bâton ≈ length_px et libellé = amp."""
    amp, length_px, data_span = 50.0, 40.0, 100.0
    axis = data_span * (length_px / amp)
    bar = scale_bar_pixel_length(
        data_value=amp,
        span=data_span,
        axis_px=axis,
        length_px=length_px,
        value_manual=True,
    )
    _assert_close(bar, length_px)
    label = scale_bar_label_value(
        data_value=amp,
        span=data_span,
        axis_px=axis,
        bar_px=bar,
        value_manual=True,
        length_manual=True,
    )
    _assert_close(label, amp)


def test_same_curve_same_visual_scale() -> None:
    """Même courbe + même amp : même échelle visuelle, quelle que soit la hauteur."""
    amp, span = 100.0, 250.0
    length_px = 40.0
    for axis in (200.0, 400.0, 800.0):
        bar = scale_bar_pixel_length(
            data_value=amp,
            span=span,
            axis_px=axis,
            length_px=length_px,
            value_manual=True,
        )
        true = scale_bar_true_value(bar, span=span, axis_px=axis)
        _assert_close(true, amp)


def test_resolve_manual_still_returns_target() -> None:
    assert resolve_scale_bar_length(22.0, manual=True, value=100.0) == 100.0


def test_value_manual_label_always_shows_entered_amp() -> None:
    """Valeur manuelle → libellé = valeur saisie (même si bâton clampé)."""
    span, axis, amp = 50.0, 400.0, 100.0
    bar = scale_bar_pixel_length(
        data_value=amp, span=span, axis_px=axis, length_px=None, value_manual=True
    )
    _assert_close(bar, axis)  # clampé à la hauteur d’axe
    label = scale_bar_label_value(
        data_value=amp,
        span=span,
        axis_px=axis,
        bar_px=bar,
        value_manual=True,
        length_manual=False,
    )
    _assert_close(label, amp)


def test_both_manual_time_label_is_entered_duration() -> None:
    """Temps + taille temps : libellé = durée saisie."""
    span, axis, time_s, length_px = 1.0, 500.0, 0.2, 40.0
    # Après fit largeur : axis = span * length / time.
    axis_fit = span * (length_px / time_s)
    bar = scale_bar_pixel_length(
        data_value=time_s,
        span=span,
        axis_px=axis_fit,
        length_px=length_px,
        value_manual=True,
    )
    _assert_close(bar, length_px)
    label = scale_bar_label_value(
        data_value=time_s,
        span=span,
        axis_px=axis_fit,
        bar_px=bar,
        value_manual=True,
        length_manual=True,
    )
    _assert_close(label, time_s)


def test_isi_scale_bar_units_are_ms_not_counts() -> None:
    """Histogramme ISI : X en ms, pas d’échelle Y (comptages)."""
    from panel_registry import _scale_bar_units

    x_unit, y_unit = _scale_bar_units("analysis_isi")
    assert x_unit == "ms"
    assert y_unit is None
    x_unit, y_unit = _scale_bar_units("isi")
    assert x_unit == "ms"
    assert y_unit is None


def test_montage_mean_decimate_preserves_peak() -> None:
    """Montage moyenne/stim : enveloppe, pas stride (pic conservé)."""
    import numpy as np
    from types import SimpleNamespace

    from panel_registry import _montage_review_curve

    n = 10_000
    t = np.linspace(-0.1, 0.4, n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    peak_i = n // 2
    y[peak_i] = 123.0
    recording = SimpleNamespace(
        t_rel=t,
        n_trials=1,
        mean=lambda _stream, _ch: y,
        stream_ready=lambda _stream, _ch: True,
        trigger_window=lambda _i, _stream, _ch: y,
        continuous_trace=lambda *a, **k: (t, y),
        has_streams=True,
    )
    _tx, yy = _montage_review_curve(
        recording, "raw", 0, mode="average", stim_index=0, max_points=200
    )
    _assert_close(float(np.max(yy)), 123.0)
