"""Traitement du signal : filtres WIDE / LOW / HIGH, spikes, RMS glissant, triggers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d

from .config import FilterSettings, TriggerSettings
from .recording import DIGITAL_STREAMS, Recording


@lru_cache(maxsize=64)
def design_sos(btype: str, ftype: str, order: int, cutoff: float, fs: float) -> np.ndarray:
    wn = min(max(cutoff / (fs / 2.0), 1e-6), 0.999)
    if ftype == "bessel":
        return sps.bessel(order, wn, btype=btype, output="sos", norm="mag")
    if ftype == "chebyshev":
        return sps.cheby1(order, 0.5, wn, btype=btype, output="sos")
    return sps.butter(order, wn, btype=btype, output="sos")


@lru_cache(maxsize=8)
def notch_sos(f0: float, fs: float, bandwidth: float = 10.0) -> np.ndarray:
    b, a = sps.iirnotch(f0, f0 / bandwidth, fs)
    return sps.tf2sos(b, a)


def _apply(sos: np.ndarray, x: np.ndarray, zero_phase: bool) -> np.ndarray:
    if x.size < 4:
        return x
    if zero_phase:
        return sps.sosfiltfilt(sos, x, padtype=None)
    zi = sps.sosfilt_zi(sos) * x[0]
    y, _ = sps.sosfilt(sos, x, zi=zi)
    return y


def filter_signal(x: np.ndarray, fs: float, band: str, flt: FilterSettings) -> np.ndarray:
    """``band`` : 'wide' (notch seul), 'low' (passe-bas) ou 'high' (passe-haut)."""
    zero = flt.phase == "zero"
    y = np.asarray(x, dtype=np.float64)
    if flt.notch_hz > 0 and flt.notch_hz < fs / 2:
        y = _apply(notch_sos(float(flt.notch_hz), fs), y, zero)
    if band == "low":
        sos = design_sos("lowpass", flt.low_type, int(flt.low_order), float(flt.low_cutoff_hz), fs)
        y = _apply(sos, y, zero)
    elif band == "high":
        sos = design_sos("highpass", flt.high_type, int(flt.high_order), float(flt.high_cutoff_hz), fs)
        y = _apply(sos, y, zero)
    return y


def sliding_rms(x: np.ndarray, fs: float, window_ms: float) -> np.ndarray:
    w = max(1, int(round(window_ms * 1e-3 * fs)))
    return np.sqrt(uniform_filter1d(x * x, size=w, mode="nearest"))


def band_for(signal: str, flt: FilterSettings) -> str:
    if signal in ("high", "spikes", "scope"):
        return "high"
    if signal == "rms":
        return flt.rms_source
    return signal


def pad_samples(signal: str, flt: FilterSettings, fs: float) -> int:
    band = band_for(signal, flt)
    pad_s = 0.0
    if flt.notch_hz > 0:
        pad_s = 0.2
    if band == "low":
        pad_s = max(pad_s, 6.0 / max(flt.low_cutoff_hz, 0.1))
    elif band == "high":
        pad_s = max(pad_s, 6.0 / max(flt.high_cutoff_hz, 0.1))
    if signal == "rms":
        pad_s += flt.rms_window_ms * 1e-3
    return int(min(pad_s, 10.0) * fs)


def read_signal(
    rec: Recording, stream: str, idx: int, s0: int, s1: int, signal: str, flt: FilterSettings
) -> np.ndarray:
    """Lit [s0, s1) puis applique la chaîne de traitement avec une marge anti-effets de bord."""
    if stream in DIGITAL_STREAMS:
        return rec.read_padded(stream, idx, s0, s1).astype(np.float64)
    pad = pad_samples(signal, flt, rec.fs)
    x = rec.read_padded(stream, idx, s0 - pad, s1 + pad)
    y = filter_signal(x, rec.fs, band_for(signal, flt), flt)
    if signal == "rms":
        y = sliding_rms(y, rec.fs, flt.rms_window_ms)
    return y[pad : pad + (s1 - s0)]


def spike_threshold(x: np.ndarray, flt: FilterSettings) -> float:
    """Seuil positif (magnitude) en µV selon le mode choisi."""
    if x.size == 0:
        return float(flt.spike_threshold)
    if flt.spike_threshold_mode == "rms":
        return float(flt.spike_threshold) * float(np.sqrt(np.mean(x * x)))
    if flt.spike_threshold_mode == "mad":
        return float(flt.spike_threshold) * float(np.median(np.abs(x)) / 0.6745)
    return abs(float(flt.spike_threshold))


def detect_spikes(x: np.ndarray, fs: float, thr: float, flt: FilterSettings) -> np.ndarray:
    """Indices des spikes : franchissement du seuil, période réfractaire, rejet d'artefacts."""
    if x.size < 3:
        return np.empty(0, dtype=np.int64)
    pol = flt.spike_polarity
    if pol == "negative":
        beyond = x < -thr
    elif pol == "positive":
        beyond = x > thr
    else:
        beyond = np.abs(x) > thr
    cand = np.flatnonzero(beyond[1:] & ~beyond[:-1]) + 1
    if cand.size == 0:
        return cand.astype(np.int64)

    refr = max(1, int(round(flt.spike_refractory_ms * 1e-3 * fs)))
    look = max(1, int(round(0.5e-3 * fs)))
    art = float(flt.spike_artifact_uv)
    kept: list[int] = []
    next_ok = -1
    n = x.size
    for s in cand.tolist():
        if s < next_ok:
            continue
        seg = x[s : min(n, s + look)]
        next_ok = s + refr
        if art > 0 and np.max(np.abs(seg)) >= art:
            continue
        if flt.scope_align == "peak":
            if pol == "negative":
                s = s + int(np.argmin(seg))
            elif pol == "positive":
                s = s + int(np.argmax(seg))
            else:
                s = s + int(np.argmax(np.abs(seg)))
        kept.append(s)
    return np.asarray(kept, dtype=np.int64)


@dataclass
class TriggerResult:
    on: np.ndarray
    off: np.ndarray  # -1 quand le front OFF n'a pas été trouvé
    fs: float
    n_raw_edges: int = 0

    @property
    def count(self) -> int:
        return int(self.on.size)

    def durations_ms(self) -> np.ndarray:
        ok = self.off >= 0
        return (self.off[ok] - self.on[ok]) / self.fs * 1e3


def detect_triggers(rec: Recording, trig: TriggerSettings) -> TriggerResult:
    fs = rec.fs
    names = rec.streams.get(trig.source_stream, [])
    empty = TriggerResult(np.empty(0, np.int64), np.empty(0, np.int64), fs)
    if not names or not (0 <= trig.source_index < len(names)):
        return empty
    a = max(0, int(trig.search_start_s * fs))
    b = rec.n_samples if trig.search_end_s <= trig.search_start_s else min(
        rec.n_samples, int(trig.search_end_s * fs)
    )
    if b - a < 2:
        return empty

    thr = float(trig.threshold)
    chunk = 1 << 22
    ons: list[np.ndarray] = []
    offs: list[np.ndarray] = []
    prev = None
    for c0 in range(a, b, chunk):
        c1 = min(b, c0 + chunk)
        x = rec.read(trig.source_stream, trig.source_index, c0, c1)
        if trig.use_abs:
            x = np.abs(x)
        active = (x > thr) if trig.edge == "rising" else (x < thr)
        if prev is None:
            prev = bool(active[0])
        d = np.diff(active.astype(np.int8), prepend=np.int8(prev))
        ons.append(np.flatnonzero(d == 1) + c0)
        offs.append(np.flatnonzero(d == -1) + c0)
        prev = bool(active[-1])
    on = np.concatenate(ons).astype(np.int64)
    off_all = np.concatenate(offs).astype(np.int64)
    n_raw = int(on.size)

    off = np.full(on.size, -1, dtype=np.int64)
    if off_all.size:
        pos = np.searchsorted(off_all, on, side="right")
        found = pos < off_all.size
        off[found] = off_all[pos[found]]

    min_dur = int(trig.min_duration_ms * 1e-3 * fs)
    if min_dur > 0:
        keep = (off < 0) | ((off - on) >= min_dur)
        on, off = on[keep], off[keep]

    gap = int(trig.min_interval_ms * 1e-3 * fs)
    if gap > 0 and on.size:
        keep = np.zeros(on.size, dtype=bool)
        last = -(1 << 62)
        for i, s in enumerate(on.tolist()):
            if s - last >= gap:
                keep[i] = True
                last = s
        on, off = on[keep], off[keep]
    return TriggerResult(on, off, fs, n_raw)


def parse_index_list(text: str) -> set[int]:
    """'3, 7-9' -> {3, 7, 8, 9} (numéros d'essais à partir de 1)."""
    out: set[int] = set()
    for part in (text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = part.split("-", 1)
                out.update(range(int(lo), int(hi) + 1))
            else:
                out.add(int(part))
        except ValueError:
            continue
    return out


def select_trials(res: TriggerResult, trig: TriggerSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applique premier essai / nombre / pas / exclusions. Renvoie (on, off, numéros 1-based)."""
    numbers = np.arange(1, res.count + 1)
    first = max(1, int(trig.first_trial))
    step = max(1, int(trig.step))
    sel = numbers[first - 1 :: step]
    excluded = parse_index_list(trig.exclude)
    if excluded:
        sel = np.array([k for k in sel if k not in excluded], dtype=np.int64)
    if trig.max_trials > 0:
        sel = sel[: int(trig.max_trials)]
    idx = sel - 1
    return res.on[idx], res.off[idx], sel
