"""Le spike scope suit la fenêtre avant/après le spike et la fenêtre affichée."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from make_synthetic_dataset import build_synthetic  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from panel_prepare import prepare_panel, restyle_panel_spec, screen_prepare_key  # noqa: E402
from panel_registry import RenderRequest, render_panel  # noqa: E402
from view_config import (  # noqa: E402
    AnalysisSettings,
    PanelPlacement,
    TimeRangeBar,
    ViewerSettings,
)


def _request(
    recording,
    *,
    pre: float = 2.0,
    post: float = 4.0,
    placement: PanelPlacement | None = None,
    bars: tuple[TimeRangeBar, ...] = (),
    active: int = 0,
) -> RenderRequest:
    return RenderRequest(
        placement=placement or PanelPlacement("analysis_overlay"),
        recordings=[recording],
        labels=["rec"],
        colors=["#1f77b4"],
        legend_flags=[True],
        channel_index=0,
        channel_name=recording.channel_names[0],
        settings=ViewerSettings(
            spike_overlay_pre_ms=pre,
            spike_overlay_post_ms=post,
            analysis=AnalysisSettings(show_overlay=True, mode="average"),
            range_bars=bars,
            active_range_index=active,
        ),
    )


def _x_extent(spec) -> tuple[float, float]:
    xs = [
        float(item.x[0])
        for item in spec.series
        if getattr(item, "x", None) is not None and len(item.x)
    ]
    xe = [
        float(item.x[-1])
        for item in spec.series
        if getattr(item, "x", None) is not None and len(item.x)
    ]
    if not xs:
        raise AssertionError(f"aucune courbe (status={spec.status})")
    return min(xs), max(xe)


def _attach_source(recording) -> None:
    """Trace HIGH avec une impulsion 15 ms avant l’unique détection."""
    from processed_dataset import SpikeTrains

    fs = float(recording.meta.fs)
    n_trials = int(recording.n_trials)
    pre_n = int(recording.meta.segmentation.pre_n)
    post_n = int(recording.meta.segmentation.post_n)
    window = pre_n + post_n
    triggers = (np.arange(n_trials, dtype=np.int64) * window) + pre_n
    pad = int(round(25.0 * fs / 1000.0)) + 8
    n_samples = int(triggers[-1] + post_n + pad)
    high = np.zeros((int(recording.n_channels), n_samples), dtype=np.float32)
    center = int(triggers[0] + round(0.01 * fs))
    pulse = center - int(round(15.0 * fs / 1000.0))
    high[0, pulse] = 800.0
    high[0, center] = -120.0

    class _Source:
        def __init__(self) -> None:
            self.fs = fs
            self.highpass = high
            self.amplifier = high
            self.lowpass = high
            self.valid_triggers = triggers

        def high_trace_for_channel(self, ch: int) -> np.ndarray:
            return self.highpass[int(ch)]

    recording.spikes = SpikeTrains.from_lists([_trials(n_trials)], n_trials)
    recording.source = _Source()


def _trials(n_trials: int) -> list[np.ndarray]:
    out = [np.empty(0, dtype=np.float32) for _ in range(n_trials)]
    out[0] = np.asarray([0.01], dtype=np.float32)
    return out


def main() -> int:
    rec = build_synthetic(label="scope", seed=1, n_channels=1, n_trials=2)
    _t, _w, times, _total, _mean = rec.overlay_for_channel(0)
    times = np.asarray(times, dtype=np.float64)

    full = prepare_panel(_request(rec, pre=2.0, post=4.0))
    if full.status != "ok" or full.xlim != (-2.0, 4.0):
        raise AssertionError(f"fenêtre défaut {full.status} {full.xlim}")
    x0, x1 = _x_extent(full)
    if x0 < -2.0 - 1e-6 or x1 > 4.0 + 1e-6:
        raise AssertionError(f"données hors fenêtre défaut [{x0}, {x1}]")
    alphas = {round(float(getattr(s, "alpha", 1.0)), 2) for s in full.series}
    if 0.4 not in alphas or 1.0 not in alphas:
        raise AssertionError(f"transparence / moyenne absentes {alphas}")
    faint_req = _request(rec)
    faint_req = replace(
        faint_req,
        settings=replace(faint_req.settings, spike_overlay_alpha=0.15),
    )
    faint = prepare_panel(faint_req)
    trace_alphas = {
        round(float(s.alpha), 2) for s in faint.series if not str(s.label).strip()
    }
    mean_alphas = {
        round(float(s.alpha), 2) for s in faint.series if str(s.label).strip()
    }
    if trace_alphas != {0.15} or mean_alphas != {1.0}:
        raise AssertionError(f"opacité réglable {trace_alphas} moyenne {mean_alphas}")
    restyled_alpha = restyle_panel_spec(full, faint_req)
    restyle_traces = {
        round(float(s.alpha), 2)
        for s in restyled_alpha.series
        if not str(s.label).strip()
    }
    if restyle_traces != {0.15}:
        raise AssertionError(f"restyle n’applique pas l’opacité {restyle_traces}")
    print("ok  fenetre defaut -2..+4 ms")

    narrow = prepare_panel(_request(rec, pre=0.5, post=1.0))
    if narrow.xlim != (-0.5, 1.0):
        raise AssertionError(f"xlim étroit {narrow.xlim}")
    nx0, nx1 = _x_extent(narrow)
    if nx0 < -0.5 - 1e-3 or nx1 > 1.0 + 1e-3:
        raise AssertionError(f"courbes non recadrées [{nx0}, {nx1}]")
    if len(narrow.series[0].x) >= len(full.series[0].x):
        raise AssertionError("le recadrage n’a pas retiré d’échantillons")
    print("ok  fenêtre étroite recadre les courbes")

    wide = prepare_panel(_request(rec, pre=10.0, post=20.0))
    if wide.xlim != (-10.0, 20.0):
        raise AssertionError(f"xlim large sans source {wide.xlim}")
    wx0, wx1 = _x_extent(wide)
    if wx0 < -2.0 - 1e-3 or wx1 > 4.0 + 1e-3:
        raise AssertionError(f"sans source, données inventées [{wx0}, {wx1}]")
    print("ok  fenêtre large sans source : axe élargi, pas de données inventées")

    med = float(np.median(times))
    lo, hi = med - 0.02, med + 0.02
    expected = int(np.count_nonzero((times >= lo) & (times <= hi)))
    if expected <= 0:
        raise AssertionError("fenêtre de test sans spike")
    zoomed = prepare_panel(
        _request(
            rec,
            placement=PanelPlacement("analysis_overlay").with_custom_zoom(lo, hi, label="mid"),
        )
    )
    # Une moyenne en plus des traces (échantillonnage 100 %).
    if zoomed.status != "ok" or len(zoomed.series) != expected + 1:
        raise AssertionError(
            f"zoom {lo:.4f}…{hi:.4f} : {zoomed.status} n={len(zoomed.series)} attendu {expected + 1}"
        )
    if zoomed.xlim != (-2.0, 4.0):
        raise AssertionError("le zoom temporel ne doit pas changer l’axe en ms")
    print(f"ok  zoom relatif garde {expected} spikes")

    outside = prepare_panel(
        _request(
            rec,
            placement=PanelPlacement("analysis_overlay").with_custom_zoom(50.0, 51.0, label="vide"),
        )
    )
    if outside.status != "empty":
        raise AssertionError(f"fenêtre vide devrait être empty, pas {outside.status}")
    print("ok  zoom hors spikes : vide")

    bar = TimeRangeBar(t0_s=lo, t1_s=hi, label="active", relative_to_stim=True)
    scoped = prepare_panel(_request(rec, bars=(bar,), active=0))
    if scoped.status != "ok" or len(scoped.series) != len(full.series):
        raise AssertionError(
            f"la zone de zoom ne doit pas filtrer le spike scope "
            f"n={len(scoped.series)} attendu {len(full.series)} ({scoped.status})"
        )
    visible_req = replace(
        _request(rec),
        visible_time_s=(lo, hi),
        visible_time_absolute=False,
    )
    visible = prepare_panel(visible_req)
    if visible.status != "ok" or len(visible.series) != expected + 1:
        raise AssertionError(
            f"fenêtre affichée n={len(visible.series)} attendu {expected + 1} ({visible.status})"
        )
    # Un zoom plus large sur le panneau, resserré par la fenêtre vraiment affichée.
    tight_over_zoom = replace(
        _request(
            rec,
            placement=PanelPlacement("analysis_overlay").with_custom_zoom(
                float(times.min()) - 1.0,
                float(times.max()) + 1.0,
                label="large",
            ),
        ),
        visible_time_s=(lo, hi),
        visible_time_absolute=False,
    )
    tight = prepare_panel(tight_over_zoom)
    if tight.status != "ok" or len(tight.series) != expected + 1:
        raise AssertionError(
            f"fenêtre affichée prioritaire sur le zoom "
            f"n={len(tight.series)} attendu {expected + 1}"
        )
    key_full = screen_prepare_key(_request(rec))
    key_bar = screen_prepare_key(_request(rec, bars=(bar,), active=0))
    key_pre = screen_prepare_key(_request(rec, pre=0.5, post=1.0))
    key_visible = screen_prepare_key(visible_req)
    if key_full == key_bar or key_full == key_pre or key_full == key_visible:
        raise AssertionError("le cache écran ignore la fenêtre sélectionnée")
    print("ok  fenêtre affichée filtre les spikes ; les zones de zoom non")

    restyled = restyle_panel_spec(full, _request(rec))
    lws = sorted(float(getattr(s, "linewidth", 1.0)) for s in restyled.series)
    if lws[-1] <= lws[0] * 1.5:
        raise AssertionError(f"restyle a aplati la moyenne {lws[:3]}…{lws[-1]}")
    print("ok  restyle garde traces fines et moyenne épaisse")

    # Ré-extraction quand la fenêtre dépasse le cache.
    live = build_synthetic(label="scope-live", seed=2, n_channels=1, n_trials=2)
    _attach_source(live)
    extracted = prepare_panel(_request(live, pre=20.0, post=20.0))
    if extracted.status != "ok":
        raise AssertionError(f"ré-extraction status={extracted.status} {extracted.status_message}")
    ex0, ex1 = _x_extent(extracted)
    if ex0 > -14.0 or ex1 < 4.0:
        raise AssertionError(f"ré-extraction trop courte [{ex0}, {ex1}]")
    peak = max(float(np.nanmax(np.abs(s.y))) for s in extracted.series if len(s.y))
    if peak < 400.0:
        raise AssertionError(f"l’impulsion à −15 ms n’est pas dans la fenêtre (pic {peak})")
    tight = prepare_panel(_request(live, pre=2.0, post=4.0))
    tpeak = max(float(np.nanmax(np.abs(s.y))) for s in tight.series if len(s.y))
    if tpeak >= 400.0:
        raise AssertionError("la fenêtre −2…+4 ms inclut l’impulsion à −15 ms")
    print("ok  fenêtre plus large ré-extrait la trace")

    # Après le premier affichage, une plage plus étroite relance la détection.
    detect_calls: list[tuple[float, float] | None] = []
    stored = [np.asarray(trial, dtype=np.float64) for trial in live.spikes.per_trial(0)]

    def _detect(ch, t_rel, threshold, refractory_s=0.001, t_range_s=None, trigger_index=None):
        del ch, t_rel, threshold, refractory_s, trigger_index
        if t_range_s is None:
            detect_calls.append(None)
            return stored
        lo, hi = float(t_range_s[0]), float(t_range_s[1])
        detect_calls.append((lo, hi))
        return [arr[(arr >= lo) & (arr <= hi)] for arr in stored]

    live.source.spike_times_per_trial_for_channel = _detect  # type: ignore[attr-defined]
    detect_calls.clear()
    prepare_panel(_request(live))
    if detect_calls:
        raise AssertionError(f"l’essai entier ne doit pas relancer la détection {detect_calls}")
    first = replace(
        _request(live),
        visible_time_s=(0.0, 0.02),
        visible_time_absolute=False,
    )
    first_spec = prepare_panel(first)
    if not detect_calls or detect_calls[-1] != (0.0, 0.02):
        raise AssertionError(f"détection non relancée sur la plage {detect_calls}")
    if first_spec.status != "ok":
        raise AssertionError(f"plage étroite {first_spec.status} {first_spec.status_message}")
    detect_calls.clear()
    prepare_panel(first)
    if detect_calls:
        raise AssertionError(f"la même plage a redétecté {detect_calls}")
    second = replace(first, visible_time_s=(0.0, 0.05), visible_time_absolute=False)
    prepare_panel(second)
    if not detect_calls or detect_calls[-1] != (0.0, 0.05):
        raise AssertionError(f"second changement de plage ignoré {detect_calls}")
    print("ok  détection relancée après affichage quand la plage change")

    # Matplotlib : même xlim.
    fig = Figure()
    status = render_panel(fig, _request(rec, pre=0.5, post=1.0))
    if status != "ok":
        raise AssertionError(f"rendu mpl {status}")
    xlim = fig.axes[0].get_xlim()
    if abs(xlim[0] + 0.5) > 1e-6 or abs(xlim[1] - 1.0) > 1e-6:
        raise AssertionError(f"mpl xlim {xlim}")
    print("ok  export matplotlib suit la fenêtre")

    print("smoke_spike_scope: all passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        raise
