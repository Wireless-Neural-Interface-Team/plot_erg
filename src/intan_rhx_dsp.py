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
from scipy.signal import sosfilt

# systemstate.h
INTAN_SNIPPET_SIZE = 50
INTAN_FRAMES_PER_BLOCK = 128
INTAN_NOTCH_BANDWIDTH_HZ = 10.0
INTAN_RMS_WINDOW_S = 1.0

FilterType = Literal["bessel", "butterworth"]
SpikeFilterKind = Literal["highpass", "lowpass"]
SpikeThresholdPolarity = Literal["negative", "positive"]


def normalize_spike_threshold(
    threshold_uv: float,
    polarity: SpikeThresholdPolarity | None = None,
) -> tuple[float, SpikeThresholdPolarity]:
    """Return (magnitude µV, polarity). Legacy: sign of threshold_uv sets polarity."""
    if polarity in ("negative", "positive"):
        return abs(float(threshold_uv)), polarity
    uv = float(threshold_uv)
    if uv >= 0:
        return abs(uv), "positive"
    return abs(uv), "negative"


def effective_spike_threshold_uv(
    threshold_uv: float,
    polarity: SpikeThresholdPolarity | None = None,
) -> float:
    """Signed threshold for detection: positive = above, negative = below."""
    mag, pol = normalize_spike_threshold(threshold_uv, polarity)
    return mag if pol == "positive" else -mag

_BESSEL_SPECS: dict[int, list[tuple[float, float]]] = {
    1: [(1.0, 0.0)],
    2: [(1 / 1.2736, 0.5773)],
    3: [(1 / 1.3270, 0.0), (1 / 1.4524, 0.6910)],
    4: [(1 / 1.4192, 0.5219), (1 / 1.5912, 0.8055)],
    5: [(1 / 1.5069, 0.0), (1 / 1.5611, 0.5635), (1 / 1.7607, 0.9165)],
    6: [(1 / 1.6060, 0.5103), (1 / 1.6913, 0.6112), (1 / 1.9071, 1.0234)],
    7: [(1 / 1.6853, 0.0), (1 / 1.7174, 0.5324), (1 / 1.8235, 0.6608), (1 / 2.0507, 1.1262)],
    8: [(1 / 1.7837, 0.5060), (1 / 1.8376, 0.5596), (1 / 1.9591, 0.7109), (1 / 2.1953, 1.2258)],
}
_BUTTERWORTH_SPECS: dict[int, list[tuple[float, float]]] = {
    1: [(1.0, 0.0)],
    2: [(1.0, 0.7071)],
    3: [(1.0, 0.0), (1.0, 1.0)],
    4: [(1.0, 1.3065), (1.0, 0.5412)],
    5: [(1.0, 0.0), (1.0, 0.6180), (1.0, 1.6181)],
    6: [(1.0, 0.5177), (1.0, 0.7071), (1.0, 1.9320)],
    7: [(1.0, 0.0), (1.0, 0.5549), (1.0, 0.8019), (1.0, 2.2472)],
    8: [(1.0, 0.5098), (1.0, 0.6013), (1.0, 0.8999), (1.0, 2.5628)],
}


@dataclass(frozen=True)
class IntanDspSettings:
    """Defaults match Intan RHX SystemState (v3.5.1) HIGH filter."""

    fs: float
    rhs_version_major: int = 3
    notch_filter_frequency_hz: float = 0.0
    spike_filter_kind: SpikeFilterKind = "highpass"
    filter_order: int = 2
    filter_type: FilterType = "bessel"
    filter_cutoff_hz: float = 250.0
    spike_threshold_uv: float = -70.0
    artifact_threshold_uv: float = 2500.0
    artifact_suppression_enabled: bool = True
    snippet_size: int = INTAN_SNIPPET_SIZE
    rms_window_s: float = INTAN_RMS_WINDOW_S

    @property
    def rms_window_samples(self) -> int:
        return max(1, int(math.ceil(float(self.fs) * float(self.rms_window_s))))

    def apply_notch_to_wideband(self) -> bool:
        """True when wideband must be notch-filtered before software filter (pre-RHX v3)."""
        return self.notch_filter_frequency_hz > 0 and self.rhs_version_major < 3

    def validate_filter(self) -> None:
        fc = float(self.filter_cutoff_hz)
        if fc <= 0:
            raise ValueError("Spike filter cutoff frequency must be > 0 Hz.")
        order = int(self.filter_order)
        if order < 1 or order > 8:
            raise ValueError("Spike filter order must be between 1 and 8.")
        kind = self.spike_filter_kind
        if kind not in ("highpass", "lowpass"):
            raise ValueError("Spike filter kind must be highpass or lowpass.")
        nyq = float(self.fs) / 2.0
        if fc >= nyq:
            raise ValueError(
                f"Spike filter cutoff ({fc:g} Hz) must be below Nyquist ({nyq:g} Hz)."
            )

    def filter_short_label(self) -> str:
        tag = "HP" if self.spike_filter_kind == "highpass" else "LP"
        return (
            f"{self.filter_type} {tag} ord.{self.filter_order} "
            f"@ {self.filter_cutoff_hz:g} Hz"
        )

    @classmethod
    def from_rhs_data(
        cls,
        data: dict[str, Any],
        *,
        spike_threshold_uv: float = -70.0,
        spike_filter_kind: SpikeFilterKind = "highpass",
        filter_order: int = 2,
        filter_type: FilterType = "bessel",
        filter_cutoff_hz: float = 250.0,
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
        settings = cls(
            fs=fs,
            rhs_version_major=major,
            notch_filter_frequency_hz=notch,
            spike_filter_kind=spike_filter_kind,
            filter_order=int(filter_order),
            filter_type=filter_type,
            filter_cutoff_hz=float(filter_cutoff_hz),
            spike_threshold_uv=float(spike_threshold_uv),
            artifact_threshold_uv=float(artifact_threshold_uv),
            artifact_suppression_enabled=bool(artifact_suppression_enabled),
        )
        settings.validate_filter()
        return settings

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> IntanDspSettings:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if "filter_cutoff_hz" not in raw:
            raw.setdefault("spike_filter_kind", "highpass")
            if "high_order" in raw:
                raw["filter_order"] = raw.pop("high_order")
            if "high_type" in raw:
                raw["filter_type"] = raw.pop("high_type")
            if "high_cutoff_hz" in raw:
                raw["filter_cutoff_hz"] = raw.pop("high_cutoff_hz")
        field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in raw.items() if k in field_names}
        settings = cls(**filtered)
        settings.validate_filter()
        return settings


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


def _filter_biquad_chain(
    kind: SpikeFilterKind,
    order: int,
    fc: float,
    fs: float,
    ftype: FilterType,
) -> list[_BiquadCoeffs]:
    """Bessel/Butterworth HP or LP chain (Intan filter.cpp)."""
    if order < 1 or order > 8:
        raise ValueError("Intan filter order must be 1..8.")
    specs = _BESSEL_SPECS if ftype == "bessel" else _BUTTERWORTH_SPECS
    chain: list[_BiquadCoeffs] = []
    for scale, q in specs[order]:
        f = fc / scale if scale > 0 else fc
        if kind == "highpass":
            if q <= 0:
                chain.append(_first_order_highpass(f, fs))
            else:
                chain.append(_second_order_highpass(f, q, fs))
        else:
            if q <= 0:
                chain.append(_first_order_lowpass(f, fs))
            else:
                chain.append(_second_order_lowpass(f, q, fs))
    return chain


def _highpass_biquad_chain(order: int, fc: float, fs: float, ftype: FilterType) -> list[_BiquadCoeffs]:
    return _filter_biquad_chain("highpass", order, fc, fs, ftype)


def _coeffs_to_sos(coeffs: list[_BiquadCoeffs]) -> np.ndarray:
    sos = np.empty((len(coeffs), 6), dtype=np.float64)
    for i, c in enumerate(coeffs):
        sos[i] = (c.b0, c.b1, c.b2, 1.0, c.a1, c.a2)
    return sos


def _cascade(signal: np.ndarray, coeffs: list[_BiquadCoeffs]) -> np.ndarray:
    """Vectorized IIR cascade (scipy SOS, ~100× faster than sample-wise Python)."""
    if not coeffs:
        return np.asarray(signal, dtype=np.float64).ravel()
    x = np.asarray(signal, dtype=np.float64).ravel()
    return sosfilt(_coeffs_to_sos(coeffs), x)


def build_intan_filter_sos(
    settings: IntanDspSettings,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Precompute SOS sections for notch (optional) + software HP/LP (reuse per channel)."""
    notch_sos: np.ndarray | None = None
    if settings.apply_notch_to_wideband():
        notch_sos = _coeffs_to_sos(
            [
                _second_order_notch(
                    settings.notch_filter_frequency_hz,
                    INTAN_NOTCH_BANDWIDTH_HZ,
                    settings.fs,
                )
            ]
        )
    filt_chain = _filter_biquad_chain(
        settings.spike_filter_kind,
        settings.filter_order,
        settings.filter_cutoff_hz,
        settings.fs,
        settings.filter_type,
    )
    return notch_sos, _coeffs_to_sos(filt_chain)


def filter_wideband_with_sos(
    wideband_uv: np.ndarray,
    notch_sos: np.ndarray | None,
    filter_sos: np.ndarray,
) -> np.ndarray:
    x = np.asarray(wideband_uv, dtype=np.float64).ravel()
    if notch_sos is not None:
        x = sosfilt(notch_sos, x)
    return sosfilt(filter_sos, x)


def wideband_to_filtered(
    wideband_uv: np.ndarray,
    settings: IntanDspSettings,
) -> np.ndarray:
    """Convert wideband amplifier (µV) to Intan software-filtered waveform (HIGH or LOW)."""
    notch_sos, filter_sos = build_intan_filter_sos(settings)
    return filter_wideband_with_sos(wideband_uv, notch_sos, filter_sos)


def wideband_to_high(
    wideband_uv: np.ndarray,
    settings: IntanDspSettings,
) -> np.ndarray:
    """Alias: HIGH path when spike_filter_kind is highpass."""
    return wideband_to_filtered(wideband_uv, settings)


def write_filtered_stack_channelwise(
    amplifier: np.ndarray,
    path: Path,
    settings: IntanDspSettings,
    *,
    channel_workers: int = 1,
    cancel_check: Any | None = None,
    progress_every: int = 0,
) -> Path:
    """Filter wideband to disk per channel (parallel workers, no full output stack in RAM)."""
    from concurrent.futures import ThreadPoolExecutor

    from memmap_io import open_writable_memmap

    arr = np.asarray(amplifier)
    if arr.ndim != 2:
        raise ValueError("write_filtered_stack_channelwise expects [n_channels, n_samples].")
    n_ch, n_samp = int(arr.shape[0]), int(arr.shape[1])
    workers = max(1, min(int(channel_workers), 16, n_ch))
    out = open_writable_memmap(path, (n_ch, n_samp), np.dtype(np.float32))
    notch_sos, filter_sos = build_intan_filter_sos(settings)

    def _one(ch: int) -> tuple[int, np.ndarray]:
        if cancel_check is not None:
            cancel_check()
        filtered = filter_wideband_with_sos(np.asarray(arr[ch], dtype=np.float64), notch_sos, filter_sos)
        return ch, np.asarray(filtered, dtype=np.float32)

    try:
        if workers <= 1 or n_ch == 1:
            for ch in range(n_ch):
                _, row = _one(ch)
                out[ch] = row
                if progress_every > 0 and (ch + 1) % progress_every == 0:
                    print(f"  filter → {path.name}: channel {ch + 1}/{n_ch}")
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for ch, row in pool.map(_one, range(n_ch)):
                    out[ch] = row
            if progress_every > 0:
                print(f"  filter → {path.name}: {n_ch} channels ({workers} workers)")
        out.flush()
    finally:
        del out
    return path


def compute_filtered_channel_stack(
    amplifier_2d: np.ndarray,
    settings: IntanDspSettings,
    channel_workers: int | None = None,
    cancel_check: Any | None = None,
) -> np.ndarray:
    """In-RAM filtered stack (legacy); prefer write_filtered_stack_channelwise for large files."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    n_ch, n_samp = np.asarray(amplifier_2d).shape
    if channel_workers is not None:
        workers = max(1, min(int(channel_workers), 16, n_ch))
    else:
        workers = max(1, min(n_ch, 16, int(os.cpu_count() or 2)))
    out = np.empty((n_ch, n_samp), dtype=np.float32)
    notch_sos, filter_sos = build_intan_filter_sos(settings)

    def _one(ch: int) -> tuple[int, np.ndarray]:
        if cancel_check is not None:
            cancel_check()
        row = np.asarray(
            filter_wideband_with_sos(np.asarray(amplifier_2d[ch], dtype=np.float64), notch_sos, filter_sos),
            dtype=np.float32,
        )
        return ch, row

    if workers <= 1 or n_ch == 1:
        for ch in range(n_ch):
            _, row = _one(ch)
            out[ch] = row
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ch, row in pool.map(_one, range(n_ch)):
                out[ch] = row
    return out


def compute_high_channel_stack(
    amplifier_2d: np.ndarray,
    settings: IntanDspSettings,
    channel_workers: int | None = None,
    cancel_check: Any | None = None,
) -> np.ndarray:
    """Backward-compatible alias for compute_filtered_channel_stack."""
    return compute_filtered_channel_stack(
        amplifier_2d, settings, channel_workers=channel_workers, cancel_check=cancel_check
    )


def rms_intan_at_index(high: np.ndarray, end_index: int, settings: IntanDspSettings) -> float:
    """RMS over the last `rms_window_s` ending at end_index (spikeplot.cpp)."""
    n = int(np.asarray(high).shape[-1]) if np.ndim(high) >= 1 else int(np.asarray(high).size)
    if n == 0:
        return 0.0
    end = int(end_index)
    if end < 0:
        return 0.0
    end = min(end, n - 1)
    n_win = min(settings.rms_window_samples, end + 1)
    start = end + 1 - n_win
    seg = np.asarray(high[start : end + 1], dtype=np.float64).ravel()
    return float(math.sqrt(float(np.mean(seg * seg))))


def mean_rms_intan_channel(high: np.ndarray, settings: IntanDspSettings) -> float:
    """Channel RMS used for threshold = multiplier × RMS (last 1 s of recording)."""
    n = int(np.asarray(high).shape[-1]) if np.ndim(high) >= 1 else int(np.asarray(high).size)
    if n == 0:
        return 0.0
    return rms_intan_at_index(high, n - 1, settings)


def sliding_rms_intan_profile(high: np.ndarray, settings: IntanDspSettings) -> np.ndarray:
    """RMS (µV) at each sample: sqrt(mean(x²)) over the preceding rms_window."""
    return sliding_rms_intan_profile_range(high, settings, 0, None)


def sliding_rms_intan_profile_range(
    high: np.ndarray,
    settings: IntanDspSettings,
    start_index: int = 0,
    end_index: int | None = None,
) -> np.ndarray:
    """RMS profile on [start_index, end_index) only (minimal mmap read)."""
    n = int(np.asarray(high).shape[-1]) if np.ndim(high) >= 1 else int(np.asarray(high).size)
    if n == 0:
        return np.array([], dtype=np.float64)
    start = max(0, int(start_index))
    end = n if end_index is None else min(n, int(end_index))
    if end <= start:
        return np.array([], dtype=np.float64)
    n_win = settings.rms_window_samples
    read_start = max(0, start + 1 - n_win)
    read_end = end
    x = np.asarray(high[read_start:read_end], dtype=np.float64).ravel()
    if x.size == 0:
        return np.array([], dtype=np.float64)
    sq = x * x
    cs = np.concatenate(([0.0], np.cumsum(sq)))
    abs_idx = np.arange(start, end, dtype=np.int64)
    seg_start_abs = np.maximum(read_start, abs_idx + 1 - n_win)
    cs_hi = abs_idx + 1 - read_start
    cs_lo = seg_start_abs - read_start
    counts = abs_idx + 1 - seg_start_abs
    return np.sqrt((cs[cs_hi] - cs[cs_lo]) / counts)


def detect_spikes_intan(
    high: np.ndarray,
    settings: IntanDspSettings,
    *,
    start_sample: int = 0,
    end_sample: int | None = None,
) -> np.ndarray:
    """Spike sample indices on HIGH (simplified Intan cpuinterface, no hoops)."""
    n = int(np.asarray(high).shape[-1]) if np.ndim(high) >= 1 else int(np.asarray(high).size)
    if n < 2:
        return np.array([], dtype=np.int64)
    t0 = max(0, int(start_sample))
    t1 = n if end_sample is None else min(n, int(end_sample))
    if t1 - t0 < settings.snippet_size + 2:
        return np.array([], dtype=np.int64)

    x = np.asarray(high[t0:t1], dtype=np.float32).ravel()
    win_len = x.size
    thr = float(settings.spike_threshold_uv)
    artifact = float(settings.artifact_threshold_uv)
    use_artifact = settings.artifact_suppression_enabled
    snippet = int(settings.snippet_size)

    spikes: list[int] = []
    s = 0
    while s < win_len - snippet:
        surpassed = False
        if thr >= 0:
            if x[s] > thr:
                surpassed = True
        elif x[s] < thr:
            surpassed = True
        if not surpassed:
            s += 1
            continue
        snippet_end = min(win_len, s + snippet)
        seg = x[s:snippet_end]
        if use_artifact:
            if thr >= 0 and np.any(seg >= artifact):
                s += 1
                continue
            if thr < 0 and np.any(seg <= -artifact):
                s += 1
                continue
        spikes.append(s + t0)
        s += snippet
    return np.asarray(spikes, dtype=np.int64)
