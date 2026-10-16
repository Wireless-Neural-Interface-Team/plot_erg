"""Contrat unique des barres d’échelle — ``resolve_axis_bar``."""

from __future__ import annotations

import math

from scale_bar_model import (
    format_scale_label,
    resolve_axis_bar,
    time_value_in_axis_units,
    units_for_kind,
    units_for_panel,
)


def _assert_close(a: float, b: float, *, tol: float = 1e-9) -> None:
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a - b) <= tol * max(1.0, abs(b)), (a, b)


def test_auto_mode_nice_proportional() -> None:
    span, axis = 100.0, 400.0
    bar = resolve_axis_bar(span=span, axis_px=axis)
    assert bar.value > 0
    _assert_close(bar.bar_px, (bar.value / span) * axis)
    _assert_close(bar.label, bar.value)
    _assert_close(bar.data_length, bar.bar_px / axis * span)
    assert bar.needed_axis_px is None


def test_value_only_label_and_proportional_bar() -> None:
    span, axis, amp = 100.0, 400.0, 50.0
    bar = resolve_axis_bar(span=span, axis_px=axis, value=amp)
    _assert_close(bar.bar_px, 200.0)
    _assert_close(bar.label, amp)
    _assert_close(bar.data_length, amp)
    assert bar.needed_axis_px is None


def test_length_only_fixed_bar_geometric_label() -> None:
    span, axis, length_px = 100.0, 400.0, 40.0
    bar = resolve_axis_bar(span=span, axis_px=axis, length_px=length_px)
    _assert_close(bar.bar_px, 40.0)
    _assert_close(bar.label, 10.0)
    _assert_close(bar.data_length, 10.0)
    assert format_scale_label(bar.label, "µV") == "10µV"
    assert bar.needed_axis_px is None


def test_both_manual_fit_and_label() -> None:
    """Valeur + taille : bâton = L tout de suite ; needed pour caler le widget."""
    span, amp, length_px = 200.0, 50.0, 40.0
    # Avant fit : la taille px est respectée (pas un bâton proportionnel).
    bar = resolve_axis_bar(
        span=span, axis_px=400.0, value=amp, length_px=length_px
    )
    _assert_close(bar.bar_px, length_px)
    _assert_close(bar.label, amp)
    assert bar.needed_axis_px is not None
    _assert_close(bar.needed_axis_px, span * (length_px / amp))  # 160

    # Après fit théorique : bâton == length_px et true == amp.
    fitted = resolve_axis_bar(
        span=span,
        axis_px=float(bar.needed_axis_px),
        value=amp,
        length_px=length_px,
    )
    _assert_close(fitted.bar_px, length_px)
    _assert_close(fitted.label, amp)
    _assert_close(fitted.data_length, amp)


def test_both_manual_time() -> None:
    span, time_s, length_px = 1.0, 0.2, 40.0
    bar = resolve_axis_bar(
        span=span, axis_px=100.0, value=time_s, length_px=length_px
    )
    assert bar.needed_axis_px is not None
    fitted = resolve_axis_bar(
        span=span,
        axis_px=float(bar.needed_axis_px),
        value=time_s,
        length_px=length_px,
    )
    _assert_close(fitted.bar_px, length_px)
    _assert_close(fitted.label, time_s)


def test_value_manual_label_stable_after_zoom_span() -> None:
    """Zoom : span change → bâton change, libellé reste la valeur."""
    amp = 50.0
    for span in (200.0, 100.0, 50.0):
        bar = resolve_axis_bar(span=span, axis_px=400.0, value=amp, length_px=40.0)
        _assert_close(bar.label, amp)


def test_both_manual_bar_px_unchanged_when_value_changes() -> None:
    """Valeur ×10 : le bâton reste L px (seul le widget / le gain données bouge)."""
    amp, length_px, span = 50.0, 40.0, 200.0
    for axis in (200.0, 400.0, 800.0):
        b10 = resolve_axis_bar(
            span=span, axis_px=axis, value=10.0, length_px=length_px
        )
        b100 = resolve_axis_bar(
            span=span, axis_px=axis, value=100.0, length_px=length_px
        )
        _assert_close(b10.bar_px, length_px)
        _assert_close(b100.bar_px, length_px)
        _assert_close(b10.label, 10.0)
        _assert_close(b100.label, 100.0)
        assert b100.needed_axis_px is not None and b10.needed_axis_px is not None
        _assert_close(b100.needed_axis_px, b10.needed_axis_px / 10.0)


def test_both_manual_bar_px_stable_across_axis() -> None:
    """Valeur + taille : bâton reste L px quelle que soit la taille d’axe."""
    amp, span, length_px = 100.0, 250.0, 40.0
    for axis in (200.0, 400.0, 800.0):
        bar = resolve_axis_bar(
            span=span, axis_px=axis, value=amp, length_px=length_px
        )
        _assert_close(bar.bar_px, length_px)
        _assert_close(bar.label, amp)
        _assert_close(bar.needed_axis_px or 0.0, span * (length_px / amp))


def test_time_fit_needed_grows_with_span() -> None:
    """Durée + taille : needed_axis croît avec la plage X (zoom inclus)."""
    time_s, length_px = 0.1, 40.0
    short = resolve_axis_bar(span=0.25, axis_px=100.0, value=time_s, length_px=length_px)
    long = resolve_axis_bar(span=12.0, axis_px=100.0, value=time_s, length_px=length_px)
    assert short.needed_axis_px is not None
    assert long.needed_axis_px is not None
    assert long.needed_axis_px > short.needed_axis_px


def test_units_isi_and_overlay() -> None:
    assert units_for_panel("analysis_isi") == ("ms", None)
    assert units_for_panel("isi") == ("ms", None)
    assert units_for_kind("isi") == ("ms", None)
    assert units_for_kind("overlay") == ("ms", "µV")
    assert units_for_kind("raw") == ("s", "µV")
    assert time_value_in_axis_units(0.1, "ms") == 100.0
    assert time_value_in_axis_units(0.1, "s") == 0.1


def test_compat_shims() -> None:
    from plot_utils import (
        resolve_scale_bar_length,
        scale_bar_label_value,
        scale_bar_pixel_length,
        scale_bar_true_value,
    )

    assert resolve_scale_bar_length(22.0, manual=True, value=100.0) == 100.0
    span, axis, amp = 100.0, 400.0, 50.0
    px = scale_bar_pixel_length(
        data_value=amp, span=span, axis_px=axis, length_px=None, value_manual=True
    )
    _assert_close(px, 200.0)
    lab = scale_bar_label_value(
        data_value=amp,
        span=span,
        axis_px=axis,
        bar_px=px,
        value_manual=True,
        length_manual=False,
    )
    _assert_close(lab, amp)
    _assert_close(scale_bar_true_value(px, span=span, axis_px=axis), amp)


def test_montage_mean_decimate_preserves_peak() -> None:
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
