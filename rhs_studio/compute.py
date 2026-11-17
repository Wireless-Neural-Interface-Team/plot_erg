"""Calcul des données d'un bloc (exécuté hors du thread graphique)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import seaborn as sns

from . import dsp
from .config import BlockConfig, FilterSettings, TriggerSettings
from .recording import DIGITAL_STREAMS, STREAM_UNITS, Recording

TIME_FACTOR = {"ms": 1000.0, "s": 1.0}
MAX_POINTS = 20_000_000


@dataclass
class Series:
    x: np.ndarray
    y: np.ndarray
    name: str
    k: int  # position du canal dans cfg.channels (couleur résolue au dessin), -1 sinon
    role: str  # main | mean | trial | band | markers | raster | psth | thr
    y2: np.ndarray | None = None


@dataclass
class BlockData:
    series: list[Series] = field(default_factory=list)
    trig_on: np.ndarray = field(default_factory=lambda: np.empty(0))
    trig_off: np.ndarray = field(default_factory=lambda: np.empty(0))
    x_label: str = ""
    y_label: str = ""
    x_unit: str = ""
    y_unit: str = ""
    x_extent: tuple[float, float] | None = None
    info: str = ""
    stats: dict[str, list[dict]] = field(default_factory=dict)


def channel_colors(cfg: BlockConfig) -> list[str]:
    n = len(cfg.channels)
    pal = sns.color_palette(cfg.palette, max(n, 1)).as_hex()
    return [
        cfg.colors[k] if k < len(cfg.colors) and cfg.colors[k] else pal[k % len(pal)]
        for k in range(n)
    ]


def _concat_nan(t: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, L = Y.shape
    x = np.empty((n, L + 1))
    y = np.empty((n, L + 1))
    x[:, :L] = t
    x[:, L] = np.nan
    y[:, :L] = Y
    y[:, L] = np.nan
    return x.ravel(), y.ravel()


def _decimate_minmax(t: np.ndarray, y: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    step = int(np.ceil(y.size / (n_out / 2)))
    m = (y.size // step) * step
    yy = y[:m].reshape(-1, step)
    tt = t[:m].reshape(-1, step)[:, 0]
    out_y = np.column_stack([yy.min(1), yy.max(1)]).ravel()
    out_t = np.repeat(tt, 2)
    return out_t, out_y


def _add_trials(data: BlockData, t: np.ndarray, Y: np.ndarray, name: str, color: int, cfg: BlockConfig):
    n = Y.shape[0]
    mode = cfg.trial_display
    if "overlay" in mode:
        sel = Y
        if n > cfg.max_overlay > 0:
            sel = Y[np.linspace(0, n - 1, cfg.max_overlay).astype(int)]
        x_cat, y_cat = _concat_nan(t, sel)
        data.series.append(Series(x_cat, y_cat, f"{name} (essais)", color, "trial"))
    if mode != "overlay":
        mean = Y.mean(axis=0)
        if mode in ("mean+sd", "mean+sem") and n > 1:
            s = Y.std(axis=0, ddof=1)
            if mode == "mean+sem":
                s = s / np.sqrt(n)
            label = "± SD" if mode == "mean+sd" else "± SEM"
            data.series.append(Series(t, mean - s, f"{name} {label}", color, "band", y2=mean + s))
        data.series.append(Series(t, mean, f"{name} (moyenne, n={n})", color, "mean"))


def _unit_labels(cfg: BlockConfig, stream: str, signal: str) -> tuple[str, str]:
    unit = STREAM_UNITS.get(stream, "")
    if stream in DIGITAL_STREAMS:
        return "État logique", ""
    if signal == "rms":
        return f"RMS ({unit})", unit
    if signal == "spikes" and cfg.spikes_view == "raster":
        return ("Essai" if cfg.mode == "triggered" else "Canal"), ""
    if signal == "spikes" and cfg.spikes_view == "psth":
        return "Fréquence (spikes/s)", "spikes/s"
    return f"Amplitude ({unit})", unit


def compute_block(
    rec: Recording | None,
    cfg: BlockConfig,
    flt: FilterSettings,
    trig: TriggerSettings,
    trig_res: dsp.TriggerResult | None,
) -> BlockData:
    data = BlockData()
    if rec is None:
        data.info = "Aucun enregistrement chargé"
        return data
    names = rec.streams.get(cfg.stream, [])
    pairs = [(k, c) for k, c in enumerate(cfg.channels) if 0 <= int(c) < len(names)]
    if not pairs:
        data.info = "Aucun canal valide sélectionné"
        return data
    chans = [(int(c), names[int(c)], k) for k, c in pairs]
    signal = "wide" if cfg.stream in DIGITAL_STREAMS else cfg.signal

    data.y_label, data.y_unit = _unit_labels(cfg, cfg.stream, signal)
    if cfg.mode == "triggered":
        _compute_triggered(data, rec, cfg, flt, trig, trig_res, chans, signal)
    else:
        _compute_continuous(data, rec, cfg, flt, trig_res, chans, signal)
    if signal == "scope":
        data.x_label, data.x_unit = "Temps autour du spike (ms)", "ms"
        data.y_label, data.y_unit = _unit_labels(cfg, cfg.stream, "high")
    if trig_res is not None and trig_res.count:
        dur = trig_res.durations_ms()
        data.stats["trig_dur"] = [{"durée_ms": float(d)} for d in dur]
    return data


def _spike_pass(h: np.ndarray, fs: float, thr: float, flt: FilterSettings, lo: int, hi: int) -> np.ndarray:
    idx = dsp.detect_spikes(h, fs, thr, flt)
    return idx[(idx >= lo) & (idx < hi)]


def _snippets(h: np.ndarray, idx: np.ndarray, pre: int, post: int) -> np.ndarray:
    ok = idx[(idx - pre >= 0) & (idx + post <= h.size)]
    if ok.size == 0:
        return np.empty((0, pre + post))
    return h[ok[:, None] + np.arange(-pre, post)[None, :]]


def _add_scope(data: BlockData, W: np.ndarray, fs: float, pre: int, name: str, color: int, flt):
    if W.shape[0] == 0:
        return
    t = (np.arange(W.shape[1]) - pre) / fs * 1e3
    n = W.shape[0]
    sel = W
    if n > flt.scope_max_waveforms > 0:
        sel = W[np.linspace(0, n - 1, flt.scope_max_waveforms).astype(int)]
    x_cat, y_cat = _concat_nan(t, sel)
    data.series.append(Series(x_cat, y_cat, f"{name} (formes, n={n})", color, "trial"))
    data.series.append(Series(t, W.mean(axis=0), f"{name} (moyenne)", color, "mean"))
    data.x_extent = (float(t[0]), float(t[-1]))


def _spike_stats(data: BlockData, name: str, spikes_s: np.ndarray, amps: np.ndarray):
    if spikes_s.size > 1:
        isi = np.diff(np.sort(spikes_s)) * 1e3
        data.stats.setdefault("isi", []).extend({"canal": name, "isi_ms": float(v)} for v in isi)
    data.stats.setdefault("spike_amp", []).extend(
        {"canal": name, "amplitude": float(v)} for v in amps
    )


def _compute_continuous(data, rec, cfg, flt, trig_res, chans, signal):
    fs = rec.fs
    tf = TIME_FACTOR.get(cfg.time_unit, 1.0)
    s0 = max(0, int(round(cfg.t_start_s * fs)))
    s1 = rec.n_samples if cfg.t_end_s <= cfg.t_start_s else min(rec.n_samples, int(round(cfg.t_end_s * fs)))
    if s1 - s0 < 4:
        data.info = "Fenêtre temporelle vide"
        return
    t = np.arange(s0, s1) / fs * tf
    data.x_label, data.x_unit = f"Temps ({cfg.time_unit})", cfg.time_unit
    data.x_extent = (float(t[0]), float(t[-1]))
    infos = []

    for k, (ch, name, color) in enumerate(chans):
        if signal in ("wide", "low", "high", "rms"):
            y = dsp.read_signal(rec, cfg.stream, ch, s0, s1, signal, flt)
            if cfg.baseline != "none" and signal != "rms" and cfg.stream not in DIGITAL_STREAMS:
                y = y - y.mean()
            tt, yy = (t, y) if y.size <= MAX_POINTS else _decimate_minmax(t, y, MAX_POINTS)
            data.series.append(Series(tt, yy.astype(np.float32), name, color, "main"))
            continue

        h = dsp.read_signal(rec, cfg.stream, ch, s0, s1, "high", flt)
        thr = dsp.spike_threshold(h, flt)
        idx = _spike_pass(h, fs, thr, flt, 0, h.size)
        infos.append(f"{name}: {idx.size} spikes, seuil {thr:.1f} µV")
        _spike_stats(data, name, (idx + s0) / fs, h[idx])
        if signal == "scope":
            pre = int(round(flt.scope_pre_ms * 1e-3 * fs))
            post = int(round(flt.scope_post_ms * 1e-3 * fs))
            _add_scope(data, _snippets(h, idx, pre, post), fs, pre, name, color, flt)
            continue
        view = cfg.spikes_view
        if view == "trace":
            data.series.append(Series(t, h.astype(np.float32), f"{name} (HIGH)", color, "main"))
            data.series.append(Series(t[idx], h[idx], f"{name} spikes", color, "markers"))
            _add_thresholds(data, thr, flt, cfg, k == 0)
        elif view == "raster":
            data.series.append(Series(t[idx], np.full(idx.size, k + 1.0), name, color, "raster"))
        else:
            bin_s = max(flt.psth_bin_ms, 0.1) * 1e-3
            edges = np.arange(s0 / fs, s1 / fs + bin_s, bin_s)
            counts, _ = np.histogram((idx + s0) / fs, bins=edges)
            data.series.append(Series(edges * tf, counts / bin_s, name, color, "psth"))

    if trig_res is not None and trig_res.count:
        on = trig_res.on[(trig_res.on >= s0) & (trig_res.on < s1)]
        off = trig_res.off[(trig_res.off >= s0) & (trig_res.off < s1)]
        data.trig_on = on / fs * tf
        data.trig_off = off / fs * tf
        infos.insert(0, f"{on.size} triggers dans la fenêtre")
    data.info = " · ".join(infos)


def _add_thresholds(data: BlockData, thr: float, flt: FilterSettings, cfg: BlockConfig, first: bool):
    if not first:
        return
    levels = {"negative": [-thr], "positive": [thr], "both": [-thr, thr]}[flt.spike_polarity]
    for lv in levels:
        data.series.append(
            Series(np.array([0.0]), np.array([lv]), f"Seuil {lv:.1f} µV", -1, "thr")
        )


def _compute_triggered(data, rec, cfg, flt, trig, trig_res, chans, signal):
    fs = rec.fs
    tf = TIME_FACTOR.get(cfg.time_unit, 1000.0)
    data.x_label = f"Temps relatif au trigger ({cfg.time_unit})"
    data.x_unit = cfg.time_unit
    if trig_res is None or not trig.enabled:
        data.info = "Trigger désactivé : passez le bloc en mode continu ou activez le trigger"
        return
    if trig_res.count == 0:
        data.info = "Aucun trigger détecté (vérifiez la source et le seuil)"
        return
    on, off, numbers = dsp.select_trials(trig_res, trig)
    pre = int(round(cfg.pre_ms * 1e-3 * fs))
    post = int(round(cfg.post_ms * 1e-3 * fs))
    L = pre + post
    ok = (on - pre >= 0) & (on + post <= rec.n_samples)
    on, off, numbers = on[ok], off[ok], numbers[ok]
    if on.size == 0 or L < 2:
        data.info = "Aucun essai dans la fenêtre pré/post choisie"
        return
    n_tr = on.size
    t = (np.arange(L) - pre) / fs * tf
    data.x_extent = (float(t[0]), float(t[-1]))
    data.trig_on = np.array([0.0])
    dur = (off[off >= 0] - on[off >= 0]) / fs
    if dur.size:
        data.trig_off = np.array([float(np.median(dur)) * tf])
    infos = [f"{n_tr} essais"]
    if dur.size:
        infos.append(f"durée médiane {np.median(dur) * 1e3:.1f} ms")

    for k, (ch, name, color) in enumerate(chans):
        if signal in ("wide", "low", "high", "rms"):
            Y = np.empty((n_tr, L))
            for i, s in enumerate(on.tolist()):
                Y[i] = dsp.read_signal(rec, cfg.stream, ch, s - pre, s + post, signal, flt)
            if signal != "rms" and cfg.stream not in DIGITAL_STREAMS:
                if cfg.baseline == "pre" and pre > 0:
                    Y -= Y[:, :pre].mean(axis=1, keepdims=True)
                elif cfg.baseline == "mean":
                    Y -= Y.mean(axis=1, keepdims=True)
            _add_trials(data, t, Y, name, color, cfg)
            _trial_stats(data, name, numbers, Y[:, pre:], t[pre:])
            continue

        mpre = int(round(flt.scope_pre_ms * 1e-3 * fs))
        mpost = int(round(flt.scope_post_ms * 1e-3 * fs))
        H = np.empty((n_tr, L + mpre + mpost))
        for i, s in enumerate(on.tolist()):
            H[i] = dsp.read_signal(rec, cfg.stream, ch, s - pre - mpre, s + post + mpost, "high", flt)
        core = H[:, mpre : mpre + L]
        thr = dsp.spike_threshold(core.ravel(), flt)
        rel_all, rows, amps, snippets = [], [], [], []
        counts = []
        for i in range(n_tr):
            idx = _spike_pass(H[i], fs, thr, flt, mpre, mpre + L)
            rel = (idx - mpre - pre) / fs
            rel_all.append(rel)
            rows.append(np.full(idx.size, i))
            amps.append(H[i][idx])
            counts.append(idx.size)
            if signal == "scope":
                snippets.append(_snippets(H[i], idx, mpre, mpost))
            if rel.size > 1:
                data.stats.setdefault("isi", []).extend(
                    {"canal": name, "isi_ms": float(v)} for v in np.diff(rel) * 1e3
                )
        rel_all = np.concatenate(rel_all) if rel_all else np.empty(0)
        rows = np.concatenate(rows) if rows else np.empty(0)
        amps = np.concatenate(amps) if amps else np.empty(0)
        data.stats.setdefault("spike_amp", []).extend({"canal": name, "amplitude": float(v)} for v in amps)
        data.stats.setdefault("spike_counts", []).extend(
            {"canal": name, "essai": int(nb), "spikes": int(c)} for nb, c in zip(numbers, counts)
        )
        infos.append(f"{name}: {rel_all.size} spikes, seuil {thr:.1f} µV")

        if signal == "scope":
            W = np.concatenate(snippets) if snippets else np.empty((0, mpre + mpost))
            _add_scope(data, W, fs, mpre, name, color, flt)
            continue
        view = cfg.spikes_view
        if view == "raster":
            offset = k * (n_tr + 1)
            data.series.append(Series(rel_all * tf, rows + 1.0 + offset, name, color, "raster"))
        elif view == "psth":
            bin_s = max(flt.psth_bin_ms, 0.1) * 1e-3
            edges = np.arange(-pre / fs, post / fs + bin_s, bin_s)
            c, _ = np.histogram(rel_all, bins=edges)
            data.series.append(Series(edges * tf, c / (bin_s * n_tr), name, color, "psth"))
        else:
            _add_trials(data, t, core, f"{name} HIGH", color, cfg)
            xs = rel_all * tf
            data.series.append(Series(xs, amps, f"{name} spikes", color, "markers"))
            _add_thresholds(data, thr, flt, cfg, k == 0)

    data.info = " · ".join(infos)


def _trial_stats(data: BlockData, name: str, numbers: np.ndarray, Ypost: np.ndarray, tpost: np.ndarray):
    if Ypost.shape[1] == 0:
        return
    rms = np.sqrt(np.mean(Ypost * Ypost, axis=1))
    i_min = Ypost.argmin(axis=1)
    i_max = Ypost.argmax(axis=1)
    rows = data.stats.setdefault("trials", [])
    for j, nb in enumerate(numbers.tolist()):
        mn = float(Ypost[j, i_min[j]])
        mx = float(Ypost[j, i_max[j]])
        rows.append(
            {
                "canal": name,
                "essai": int(nb),
                "rms": float(rms[j]),
                "min": mn,
                "t_min": float(tpost[i_min[j]]),
                "max": mx,
                "t_max": float(tpost[i_max[j]]),
                "crête_à_crête": mx - mn,
            }
        )
