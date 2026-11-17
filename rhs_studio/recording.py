"""Accès aux enregistrements : fichiers Intan .rhs (memmap) et données de démonstration."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

SAMPLES_PER_BLOCK = 128
RHS_MAGIC = 0xD69127AC

STREAM_LABELS = {
    "amp": "Amplificateur",
    "dc": "Ampli DC",
    "stim": "Stimulation",
    "adc": "Entrée analogique",
    "dac": "Sortie analogique",
    "din": "Entrée numérique",
    "dout": "Sortie numérique",
}
STREAM_UNITS = {
    "amp": "µV",
    "dc": "V",
    "stim": "µA",
    "adc": "V",
    "dac": "V",
    "din": "",
    "dout": "",
}
DIGITAL_STREAMS = ("din", "dout")


class Recording:
    """Interface commune : lecture d'un canal sur un intervalle d'échantillons."""

    name: str = ""
    path: str = ""
    fs: float = 0.0
    n_samples: int = 0

    def __init__(self) -> None:
        self.streams: dict[str, list[str]] = {}

    @property
    def duration(self) -> float:
        return self.n_samples / self.fs if self.fs > 0 else 0.0

    def read(self, stream: str, idx: int, s0: int, s1: int) -> np.ndarray:
        raise NotImplementedError

    def read_padded(self, stream: str, idx: int, s0: int, s1: int) -> np.ndarray:
        """Comme ``read`` mais accepte un intervalle qui dépasse le fichier (bords répétés)."""
        a = max(0, int(s0))
        b = min(self.n_samples, int(s1))
        if b <= a:
            return np.zeros(max(0, int(s1) - int(s0)), dtype=np.float32)
        x = self.read(stream, idx, a, b)
        if a > s0 or b < s1:
            x = np.pad(x, (a - int(s0), int(s1) - b), mode="edge")
        return x

    def describe(self) -> str:
        parts = [f"{len(v)} {STREAM_LABELS[k].lower()}" for k, v in self.streams.items() if v]
        return (
            f"{self.name} — {self.fs:g} Hz — {self.duration:.2f} s — " + ", ".join(parts)
        )


def _qstring(f) -> str:
    (length,) = struct.unpack("<I", f.read(4))
    if length == 0xFFFFFFFF:
        return ""
    return f.read(length).decode("utf-16-le", errors="replace")


def read_rhs_header(f) -> dict:
    (magic,) = struct.unpack("<I", f.read(4))
    if magic != RHS_MAGIC:
        raise ValueError("Ce fichier n'est pas un fichier Intan RHS.")
    h: dict = {}
    h["version"] = struct.unpack("<hh", f.read(4))
    (h["sample_rate"],) = struct.unpack("<f", f.read(4))
    vals = struct.unpack("<hffffffff", f.read(34))
    h["dsp_enabled"] = vals[0]
    h["actual_dsp_cutoff"], h["actual_lower_bw"], _, h["actual_upper_bw"] = vals[1:5]
    (notch_mode,) = struct.unpack("<h", f.read(2))
    h["notch_hz"] = {1: 50.0, 2: 60.0}.get(notch_mode, 0.0)
    f.read(8)  # fréquences de test d'impédance
    f.read(4)  # amp settle mode + charge recovery mode
    (h["stim_step_size"],) = struct.unpack("<f", f.read(4))
    f.read(8)  # recovery current limit + target voltage
    h["notes"] = [_qstring(f) for _ in range(3)]
    (h["dc_saved"],) = struct.unpack("<h", f.read(2))
    f.read(2)  # eval board mode
    h["reference"] = _qstring(f)

    channels: dict[int, list[dict]] = {t: [] for t in (0, 3, 4, 5, 6)}
    (n_groups,) = struct.unpack("<h", f.read(2))
    for _ in range(n_groups):
        _qstring(f)
        _qstring(f)
        enabled, n_ch, _ = struct.unpack("<hhh", f.read(6))
        if n_ch <= 0 or enabled <= 0:
            continue
        for _ in range(n_ch):
            native = _qstring(f)
            custom = _qstring(f)
            native_order, _, sig_type, ch_enabled, _, _, _ = struct.unpack(
                "<hhhhhHh", f.read(14)
            )
            f.read(8)  # paramètres de trigger matériel
            imp_mag, _ = struct.unpack("<ff", f.read(8))
            if not ch_enabled:
                continue
            if sig_type not in channels:
                raise ValueError(f"Type de canal inconnu : {sig_type}")
            channels[sig_type].append(
                {
                    "native": native,
                    "custom": custom,
                    "order": native_order,
                    "impedance": imp_mag,
                }
            )
    h["channels"] = channels
    h["data_offset"] = f.tell()
    return h


def _label(ch: dict) -> str:
    if ch["custom"] and ch["custom"] != ch["native"]:
        return f"{ch['native']} ({ch['custom']})"
    return ch["native"]


class RhsRecording(Recording):
    """Fichier .rhs mappé en mémoire : aucune copie, lecture par fenêtre."""

    def __init__(self, path: str | os.PathLike) -> None:
        super().__init__()
        self.path = str(path)
        self.name = Path(path).name
        with open(path, "rb") as f:
            h = read_rhs_header(f)
        self.header = h
        self.fs = float(h["sample_rate"])
        ch = h["channels"]
        amp, adc, dac, din, dout = ch[0], ch[3], ch[4], ch[5], ch[6]
        n_amp = len(amp)

        fields: list[tuple] = [("t", "<i4", (SAMPLES_PER_BLOCK,))]
        if n_amp:
            fields.append(("amp", "<u2", (n_amp, SAMPLES_PER_BLOCK)))
            if h["dc_saved"]:
                fields.append(("dc", "<u2", (n_amp, SAMPLES_PER_BLOCK)))
            fields.append(("stim", "<u2", (n_amp, SAMPLES_PER_BLOCK)))
        if adc:
            fields.append(("adc", "<u2", (len(adc), SAMPLES_PER_BLOCK)))
        if dac:
            fields.append(("dac", "<u2", (len(dac), SAMPLES_PER_BLOCK)))
        if din:
            fields.append(("din", "<u2", (SAMPLES_PER_BLOCK,)))
        if dout:
            fields.append(("dout", "<u2", (SAMPLES_PER_BLOCK,)))
        dtype = np.dtype(fields)

        size = os.path.getsize(path)
        n_blocks = (size - h["data_offset"]) // dtype.itemsize
        self.n_samples = int(n_blocks) * SAMPLES_PER_BLOCK
        self._mm = (
            np.memmap(path, dtype=dtype, mode="r", offset=h["data_offset"], shape=(n_blocks,))
            if n_blocks > 0
            else None
        )

        if n_amp:
            self.streams["amp"] = [_label(c) for c in amp]
            if h["dc_saved"]:
                self.streams["dc"] = [_label(c) for c in amp]
            self.streams["stim"] = [_label(c) for c in amp]
        if adc:
            self.streams["adc"] = [_label(c) for c in adc]
        if dac:
            self.streams["dac"] = [_label(c) for c in dac]
        if din:
            self.streams["din"] = [_label(c) for c in din]
        if dout:
            self.streams["dout"] = [_label(c) for c in dout]
        self._bits = {
            "din": [int(c["order"]) for c in din],
            "dout": [int(c["order"]) for c in dout],
        }
        self.stim_step_ua = float(h["stim_step_size"]) * 1e6
        self.impedances = [float(c["impedance"]) for c in amp]

    def read(self, stream: str, idx: int, s0: int, s1: int) -> np.ndarray:
        if self._mm is None or s1 <= s0:
            return np.zeros(max(0, s1 - s0), dtype=np.float32)
        b0 = s0 // SAMPLES_PER_BLOCK
        b1 = -(-s1 // SAMPLES_PER_BLOCK)
        view = self._mm[stream][b0:b1]
        lo = s0 - b0 * SAMPLES_PER_BLOCK
        hi = lo + (s1 - s0)
        if stream in DIGITAL_STREAMS:
            raw = np.asarray(view).reshape(-1)[lo:hi]
            return ((raw >> self._bits[stream][idx]) & 1).astype(np.float32)
        raw = np.asarray(view[:, idx, :]).reshape(-1)[lo:hi]
        if stream == "amp":
            return (raw.astype(np.float32) - 32768.0) * 0.195
        if stream == "dc":
            return (raw.astype(np.float32) - 512.0) * -0.01923
        if stream == "stim":
            mag = (raw & 0xFF).astype(np.float32) * self.stim_step_ua
            return np.where(raw & 0x100, -mag, mag)
        return (raw.astype(np.float32) - 32768.0) * 312.5e-6

    def describe(self) -> str:
        v = self.header["version"]
        return super().describe() + f" — RHS v{v[0]}.{v[1]}"


class SyntheticRecording(Recording):
    """ERG + activité unitaire synthétiques, stimulus lumineux sur ADC et DIN."""

    def __init__(
        self,
        fs: float = 20000.0,
        duration_s: float = 40.0,
        n_channels: int = 8,
        period_s: float = 1.0,
        pulse_s: float = 0.2,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.name = "Démonstration (synthétique)"
        self.fs = float(fs)
        n = int(duration_s * fs)
        self.n_samples = n
        rng = np.random.default_rng(seed)
        t = np.arange(n) / fs

        onsets = np.arange(0.5, duration_s - period_s, period_s)
        onsets = onsets + rng.uniform(-0.01, 0.01, onsets.size)
        on_idx = (onsets * fs).astype(np.int64)
        off_idx = on_idx + int(pulse_s * fs)
        light = np.zeros(n, dtype=np.float32)
        for a, b in zip(on_idx, off_idx):
            light[a:b] = 1.0
        self._adc = np.stack([light * 3.3 + rng.normal(0, 0.01, n).astype(np.float32)])
        self._din = np.stack([light])

        k_len = int(0.5 * fs)
        tk = np.arange(k_len) / fs
        a_wave = -np.exp(-0.5 * ((tk - 0.018) / 0.006) ** 2)
        b_wave = np.exp(-0.5 * ((tk - 0.060) / 0.020) ** 2)
        d_wave = np.exp(-0.5 * ((tk - 0.030) / 0.012) ** 2)

        spike_t = np.arange(int(0.002 * fs)) / fs
        spike_wave = -np.sin(2 * np.pi * spike_t / 0.001) * np.exp(-spike_t / 0.0005)
        spike_wave = spike_wave / np.abs(spike_wave).max()

        amp = np.empty((n_channels, n), dtype=np.float32)
        for c in range(n_channels):
            gain = 1.0 - 0.07 * c
            x = rng.normal(0.0, 8.0, n)
            x += 6.0 * np.sin(2 * np.pi * 50.0 * t + c)
            x += np.cumsum(rng.normal(0, 0.05, n))  # dérive lente
            resp = np.zeros(n)
            for a, b in zip(on_idx, off_idx):
                seg = slice(a, min(n, a + k_len))
                m = seg.stop - seg.start
                resp[seg] += gain * (90.0 * a_wave[:m] + 220.0 * b_wave[:m])
                seg = slice(b, min(n, b + k_len))
                m = seg.stop - seg.start
                resp[seg] += gain * 70.0 * d_wave[:m]
            x += resp
            rate = np.where(light > 0, 70.0, 6.0)
            spikes = np.flatnonzero(rng.random(n) < rate / fs)
            amp_s = rng.normal(80.0 * gain, 8.0, spikes.size)
            for s, ampl in zip(spikes, amp_s):
                m = min(spike_wave.size, n - s)
                x[s : s + m] += ampl * spike_wave[:m]
            amp[c] = x
        self._amp = amp

        self.streams = {
            "amp": [f"A-{c:03d}" for c in range(n_channels)],
            "adc": ["ANALOG-IN-1 (lumière)"],
            "din": ["DIGITAL-IN-01"],
        }

    def read(self, stream: str, idx: int, s0: int, s1: int) -> np.ndarray:
        arr = {"amp": self._amp, "adc": self._adc, "din": self._din}[stream]
        return arr[idx, s0:s1]
