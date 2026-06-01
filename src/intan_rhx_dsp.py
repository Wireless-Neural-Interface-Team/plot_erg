"""Intan RHX-compatible DSP (GPL-3.0 reference: Intan-Technologies/Intan-RHX).

Reimplements the software filter chain and spike/RMS metrics used in:
- Engine/Processing/XPUInterfaces/cpuinterface.cpp (HIGH + spike detection)
- GUI/Widgets/spikeplot.cpp (RMS over the last 1 s of HIGH data)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

# systemstate.h
INTAN_SNIPPET_SIZE = 50
INTAN_FRAMES_PER_BLOCK = 128
INTAN_NOTCH_BANDWIDTH_HZ = 10.0
INTAN_RMS_WINDOW_S = 1.0

FilterType = Literal["bessel", "butterworth"]


@dataclass(frozen=True)
class IntanDspSettings:
    """Defaults match Intan RHX SystemState (v3.5.1)."""

    fs: float
    rhs_version_major: int = 3
    notch_filter_frequency_hz: float = 0.0
    high_order: int = 2
    high_type: FilterType = "bessel"
    high_cutoff_hz: float = 250.0
    spike_threshold_uv: float = -70.0
    artifact_threshold_uv: float = 2500.0
    artifact_suppression_enabled: bool = True
    snippet_size: int = INTAN_SNIPPET_SIZE
    rms_window_s: float = INTAN_RMS_WINDOW_S

    @property
    def rms_window_samples(self) -> int:
        return max(1, int(math.ceil(float(self.fs) * float(self.rms_window_s))))

    def apply_notch_to_wideband(self) -> bool:
        """True when wideband must be notch-filtered before HIGH (pre-RHX v3 files)."""
        return self.notch_filter_frequency_hz > 0 and self.rhs_version_major < 3

    @classmethod
    def from_rhs_data(
        cls,
        data: dict[str, Any],
        *,
        spike_threshold_uv: float = -70.0,
        high_order: int = 2,
        high_type: FilterType = "bessel",
        high_cutoff_hz: float = 250.0,
        artifact_threshold_uv: float = 2500.0,
        artifact_suppression_enabled: bool = True,
    ) -> IntanDspSettings:
        freq = data.get("frequency_parameters") or {}
        fs = float(freq.get("amplifier_sample_rate") or data.get("sample_rate") or 0.0)
        if fs <= 0:
            raise ValueError("RHS data has no amplifier sample rate.")
        version = data.get("version") or {}
        major = int(version.get("major", 3))
        notch = float(freq.get("notch_filter_frequency") or 0.0)
        return cls(
            fs=fs,
            rhs_version_major=major,
            notch_filter_frequency_hz=notch,
            high_order=int(high_order),
            high_type=high_type,
            high_cutoff_hz=float(high_cutoff_hz),
            spike_threshold_uv=float(spike_threshold_uv),
            artifact_threshold_uv=float(artifact_threshold_uv),
            artifact_suppression_enabled=bool(artifact_suppression_enabled),
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> IntanDspSettings:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(**raw)


@dataclass
class _BiquadCoeffs:
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float
    is_dc_gain_zero: bool = False


class _BiquadStream:
    """Direct Form I biquad (filter.cpp / BiquadFilter::filterOne)."""

    def __init__(self, coeffs: _BiquadCoeffs) -> None:
        self.c = coeffs
        self.first = True
        self.prev_in = 0.0
        self.prev_prev_in = 0.0
        self.prev_out = 0.0
        self.prev_prev_out = 0.0

    def filter_one(self, x: float) -> float:
        c = self.c
        if self.first:
            self.prev_in = x
            self.prev_prev_in = x
            if c.is_dc_gain_zero:
                self.prev_out = 0.0
                self.prev_prev_out = 0.0
            else:
                self.prev_out = x
                self.prev_prev_out = x
            self.first = False
        out = (
            c.b0 * x
            + c.b1 * self.prev_in
            + c.b2 * self.prev_prev_in
            - c.a1 * self.prev_out
            - c.a2 * self.prev_prev_out
        )
        self.prev_prev_in = self.prev_in
        self.prev_in = x
        self.prev_prev_out = self.prev_out
        self.prev_out = out
        return out

    def process(self, signal: np.ndarray) -> np.ndarray:
        x = np.asarray(signal, dtype=np.float64).ravel()
        y = np.empty(x.size, dtype=np.float64)
        for i in range(x.size):
            y[i] = self.filter_one(float(x[i]))
        return y


def _first_order_lowpass(fc: float, fs: float) -> _BiquadCoeffs:
    k = math.exp(-2.0 * math.pi * fc / fs)
    return _BiquadCoeffs(1.0 - k, 0.0, 0.0, -k, 0.0, False)


def _first_order_highpass(fc: float, fs: float) -> _BiquadCoeffs:
    k = math.exp(-2.0 * math.pi * fc / fs)
    return _BiquadCoeffs(1.0, -1.0, 0.0, -k, 0.0, True)


def _second_order_lowpass(fc: float, q: float, fs: float) -> _BiquadCoeffs:
    k = math.tan(math.pi * fc / fs)
    norm = 1.0 / (1.0 + k / q + k * k)
    b0 = k * k * norm
    b1 = 2.0 * k * k * norm
    return _BiquadCoeffs(b0, b1, b0, 2.0 * (k * k - 1.0) * norm, (1.0 - k / q + k * k) * norm, False)


def _second_order_highpass(fc: float, q: float, fs: float) -> _BiquadCoeffs:
    k = math.tan(math.pi * fc / fs)
    norm = 1.0 / (1.0 + k / q + k * k)
    b0 = norm
    b1 = -2.0 * norm
    return _BiquadCoeffs(b0, b1, b0, 2.0 * (k * k - 1.0) * norm, (1.0 - k / q + k * k) * norm, True)


def _second_order_notch(f_notch: float, bandwidth: float, fs: float) -> _BiquadCoeffs:
    d = math.exp(-math.pi * bandwidth / fs)
    b = (1.0 + d * d) * math.cos(2.0 * math.pi * f_notch / fs)
    a = (1.0 + d * d) / 2.0
    return _BiquadCoeffs(a, -b, a, b, d * d, False)


def _highpass_biquad_chain(order: int, fc: float, fs: float, ftype: FilterType) -> list[_BiquadCoeffs]:
    """Coefficients for BesselHighpassFilter / ButterworthHighpassFilter (filter.cpp)."""
    if order < 1 or order > 8:
        raise ValueError("Intan high-pass order must be 1..8.")
    if ftype == "bessel":
        specs: dict[int, list[tuple[float, float]]] = {
            1: [(1.0, 0.0)],
            2: [(1 / 1.2736, 0.5773)],
            3: [(1 / 1.3270, 0.0), (1 / 1.4524, 0.6910)],
            4: [(1 / 1.4192, 0.5219), (1 / 1.5912, 0.8055)],
            5: [(1 / 1.5069, 0.0), (1 / 1.5611, 0.5635), (1 / 1.7607, 0.9165)],
            6: [(1 / 1.6060, 0.5103), (1 / 1.6913, 0.6112), (1 / 1.9071, 1.0234)],
            7: [(1 / 1.6853, 0.0), (1 / 1.7174, 0.5324), (1 / 1.8235, 0.6608), (1 / 2.0507, 1.1262)],
            8: [(1 / 1.7837, 0.5060), (1 / 1.8376, 0.5596), (1 / 1.9591, 0.7109), (1 / 2.1953, 1.2258)],
        }
    else:
        specs = {
            1: [(1.0, 0.0)],
            2: [(1.0, 0.7071)],
            3: [(1.0, 0.0), (1.0, 1.0)],
            4: [(1.0, 1.3065), (1.0, 0.5412)],
            5: [(1.0, 0.0), (1.0, 0.6180), (1.0, 1.6181)],
            6: [(1.0, 0.5177), (1.0, 0.7071), (1.0, 1.9320)],
            7: [(1.0, 0.0), (1.0, 0.5549), (1.0, 0.8019), (1.0, 2.2472)],
            8: [(1.0, 0.5098), (1.0, 0.6013), (1.0, 0.8999), (1.0, 2.5628)],
        }
    chain: list[_BiquadCoeffs] = []
    for scale, q in specs[order]:
        f = fc / scale if scale > 0 else fc
        if q <= 0:
            chain.append(_first_order_highpass(f, fs))
        else:
            chain.append(_second_order_highpass(f, q, fs))
    return chain


def _cascade(signal: np.ndarray, coeffs: list[_BiquadCoeffs]) -> np.ndarray:
    out = np.asarray(signal, dtype=np.float64)
    for c in coeffs:
        out = _BiquadStream(c).process(out)
    return out


def wideband_to_high(
    wideband_uv: np.ndarray,
    settings: IntanDspSettings,
) -> np.ndarray:
    """Convert wideband amplifier (µV) to Intan HIGH waveform."""
    x = np.asarray(wideband_uv, dtype=np.float64).ravel()
    if settings.apply_notch_to_wideband():
        wide = _BiquadStream(
            _second_order_notch(
                settings.notch_filter_frequency_hz,
                INTAN_NOTCH_BANDWIDTH_HZ,
                settings.fs,
            )
        ).process(x)
    else:
        wide = x
    high_chain = _highpass_biquad_chain(
        settings.high_order,
        settings.high_cutoff_hz,
        settings.fs,
        settings.high_type,
    )
    return _cascade(wide, high_chain)


def compute_high_channel_stack(
    amplifier_2d: np.ndarray,
    settings: IntanDspSettings,
    channel_workers: int | None = None,
    cancel_check: Any | None = None,
) -> np.ndarray:
    """HIGH waveforms for all channels [n_channels, n_samples]."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    n_ch, _ = amplifier_2d.shape
    out = np.empty_like(np.asarray(amplifier_2d, dtype=np.float32), dtype=np.float32)

    def _one(ch: int) -> tuple[int, np.ndarray]:
        if cancel_check is not None:
            cancel_check()
        hi = wideband_to_high(np.asarray(amplifier_2d[ch], dtype=np.float64), settings)
        return ch, hi.astype(np.float32, copy=False)

    if channel_workers is not None:
        workers = max(1, min(int(channel_workers), 16, n_ch))
    else:
        workers = max(1, min((os.cpu_count() or 2) // 2, 16, n_ch))
    if workers <= 1 or n_ch == 1:
        for c in range(n_ch):
            _, row = _one(c)
            out[c, :] = row
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ch, row in pool.map(_one, range(n_ch)):
            out[ch, :] = row
    return out


def rms_intan_at_index(high: np.ndarray, end_index: int, settings: IntanDspSettings) -> float:
    """RMS over the last `rms_window_s` ending at end_index (spikeplot.cpp)."""
    x = np.asarray(high, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    end = int(end_index)
    if end < 0:
        return 0.0
    end = min(end, x.size - 1)
    n_win = min(settings.rms_window_samples, end + 1)
    start = end + 1 - n_win
    seg = x[start : end + 1]
    return float(math.sqrt(float(np.mean(seg * seg))))


def mean_rms_intan_channel(high: np.ndarray, settings: IntanDspSettings) -> float:
    """Channel RMS used for threshold = multiplier × RMS (last 1 s of recording)."""
    return rms_intan_at_index(high, int(np.asarray(high).size) - 1, settings)


def sliding_rms_intan_profile(high: np.ndarray, settings: IntanDspSettings) -> np.ndarray:
    """RMS (µV) at each sample: sqrt(mean(x²)) over the preceding rms_window."""
    x = np.asarray(high, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)
    n_win = settings.rms_window_samples
    sq = x * x
    cs = np.concatenate(([0.0], np.cumsum(sq)))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i + 1 - n_win)
        count = i + 1 - start
        out[i] = math.sqrt((cs[i + 1] - cs[start]) / count)
    return out


def detect_spikes_intan(
    high: np.ndarray,
    settings: IntanDspSettings,
    *,
    start_sample: int = 0,
    end_sample: int | None = None,
) -> np.ndarray:
    """Spike sample indices on HIGH (simplified Intan cpuinterface, no hoops)."""
    x = np.asarray(high, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return np.array([], dtype=np.int64)
    t0 = max(0, int(start_sample))
    t1 = n if end_sample is None else min(n, int(end_sample))
    if t1 - t0 < settings.snippet_size + 2:
        return np.array([], dtype=np.int64)

    thr = float(settings.spike_threshold_uv)
    artifact = float(settings.artifact_threshold_uv)
    use_artifact = settings.artifact_suppression_enabled
    snippet = int(settings.snippet_size)

    spikes: list[int] = []
    s = t0
    while s < t1 - snippet:
        surpassed = False
        if thr >= 0:
            if x[s] > thr:
                surpassed = True
        elif x[s] < thr:
            surpassed = True
        if not surpassed:
            s += 1
            continue
        snippet_end = min(t1, s + snippet)
        seg = x[s:snippet_end]
        if use_artifact:
            if thr >= 0 and np.any(seg >= artifact):
                s += 1
                continue
            if thr < 0 and np.any(seg <= -artifact):
                s += 1
                continue
        spikes.append(s)
        s += snippet
    return np.asarray(spikes, dtype=np.int64)
