from __future__ import annotations

import contextlib
import contextvars
import gc
import importlib.util
import shutil
from pathlib import Path
import sys
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

from config import AnalysisConfig

_analysis_cancel_event: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "analysis_cancel_event",
    default=None,
)


@contextlib.contextmanager
def analysis_cancel_scope(event: threading.Event | None) -> Iterator[None]:
    """Utilise par le thread GUI : si event est defini et .set(), check_analysis_cancelled leve."""
    token = _analysis_cancel_event.set(event)
    try:
        yield
    finally:
        _analysis_cancel_event.reset(token)


def check_analysis_cancelled() -> None:
    """Leve InterruptedError si l'utilisateur a demande l'arret (interface graphique)."""
    ev = _analysis_cancel_event.get()
    if ev is not None and ev.is_set():
        raise InterruptedError("Traitement interrompu.")


def apply_butterworth_lowpass(
    data: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Filtre passe-bas Butterworth, zero-phase (filtfilt), par canal (axis=1)."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise RuntimeError("apply_butterworth_lowpass attend [n_channels, n_samples].")
    nyq = 0.5 * fs
    if cutoff_hz <= 0:
        raise ValueError("La frequence de coupure doit etre > 0.")
    if cutoff_hz >= nyq:
        raise ValueError(
            f"Frequence de coupure ({cutoff_hz} Hz) doit etre < frequence de Nyquist ({nyq:.1f} Hz)."
        )
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, data, axis=1)


def apply_butterworth_bandpass(
    data: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Filtre passe-bande Butterworth, zero-phase (filtfilt), par canal (axis=1)."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise RuntimeError("apply_butterworth_bandpass attend [n_channels, n_samples].")
    nyq = 0.5 * fs
    if low_hz <= 0 or high_hz <= 0:
        raise ValueError("Les frequences de coupure passe-bande doivent etre > 0.")
    if low_hz >= high_hz:
        raise ValueError("Passe-bande: la frequence basse doit etre < la frequence haute.")
    if high_hz >= nyq:
        raise ValueError(
            f"Frequence haute ({high_hz} Hz) doit etre < frequence de Nyquist ({nyq:.1f} Hz)."
        )
    wn = (low_hz / nyq, high_hz / nyq)
    b, a = butter(order, wn, btype="band")
    return filtfilt(b, a, data, axis=1)


def _import_intan_loader():
    src_dir = Path(__file__).resolve().parent
    loader_path = src_dir / "load_intan_rhs_format.py"
    if not loader_path.exists():
        raise RuntimeError(f"Fichier introuvable: {loader_path}")

    spec = importlib.util.spec_from_file_location("load_intan_rhs_format", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible de construire le loader Python pour load_intan_rhs_format.py.")

    # Permet de resoudre les imports internes du lecteur (ex: intanutil.*) depuis src/.
    inserted_path = False
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        inserted_path = True

    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        read_data = getattr(module, "read_data", None)
        if not callable(read_data):
            raise RuntimeError("La fonction read_data est absente de load_intan_rhs_format.py.")
        return read_data
    except Exception as exc:
        raise RuntimeError(
            "Impossible d'importer 'load_intan_rhs_format.py'. "
            "Verifie que le dossier 'intanutil' est bien present dans src/ (fichiers du zip Intan complet). "
            f"Detail: {exc}"
        ) from exc
    finally:
        if inserted_path:
            try:
                sys.path.remove(str(src_dir))
            except ValueError:
                pass


def load_rhs_file(rhs_path: Path) -> dict[str, Any]:
    read_data = _import_intan_loader()
    data = read_data(str(rhs_path))
    if not isinstance(data, dict):
        raise RuntimeError("Le lecteur Intan n'a pas retourne un dictionnaire.")
    return data


def get_sampling_rate(data: dict[str, Any]) -> float:
    freq = data.get("frequency_parameters", {})
    sample_rate = freq.get("amplifier_sample_rate")
    if sample_rate is None:
        raise RuntimeError("Frequence d'echantillonnage introuvable.")
    return float(sample_rate)


def get_analog_in0_signal(data: dict[str, Any]) -> np.ndarray:
    board_adc_data = np.asarray(data.get("board_adc_data"))
    if board_adc_data.size == 0:
        raise RuntimeError("Le fichier RHS ne contient pas board_adc_data.")
    if board_adc_data.ndim != 2:
        raise RuntimeError("Format inattendu pour board_adc_data.")
    if board_adc_data.shape[0] < 1:
        raise RuntimeError("Aucun canal ADC trouve.")
    return board_adc_data[0]


def detect_edges(signal: np.ndarray, threshold: float, edge: str) -> np.ndarray:
    """Indices d'echantillon au front: 'falling' (descendant) ou 'rising' (montant) au seuil."""
    above = signal > threshold
    if edge == "rising":
        return np.where((~above[:-1]) & (above[1:]))[0] + 1
    if edge == "falling":
        return np.where((above[:-1]) & (~above[1:]))[0] + 1
    raise ValueError("edge doit etre 'falling' ou 'rising'.")


def mean_time_to_next_rising_edge_s(
    signal: np.ndarray,
    trigger_indices: np.ndarray,
    threshold: float,
    fs: float,
) -> float | None:
    """Temps relatif moyen (s) du trigger au prochain front montant au seuil (fin de pulse typique)."""
    trigger_indices = np.asarray(trigger_indices, dtype=np.int64)
    rising_idx = detect_edges(signal, threshold, "rising")
    if rising_idx.size == 0:
        return None
    deltas_s: list[float] = []
    for tr in trigger_indices:
        after = rising_idx[rising_idx > tr]
        if after.size == 0:
            continue
        deltas_s.append(float(after[0] - tr) / fs)
    if not deltas_s:
        return None
    return float(np.mean(deltas_s))


def detect_spikes_at_threshold(
    trace: np.ndarray,
    fs: float,
    threshold: float,
    refractory_s: float = 0.001,
) -> np.ndarray:
    """Indices d'échantillon des passages au seuil, avec réfractaire.

    - Seuil >= 0 : front montant (le signal franchit le seuil vers le haut).
    - Seuil < 0 : front descendant (le signal franchit le seuil vers le bas), adapté aux pics négatifs.
    """
    trace = np.asarray(trace, dtype=np.float64).ravel()
    if trace.size < 2:
        return np.array([], dtype=np.int64)
    if threshold >= 0:
        above = trace > threshold
        cross = np.where((~above[:-1]) & (above[1:]))[0] + 1
    else:
        below = trace < threshold
        cross = np.where((~below[:-1]) & (below[1:]))[0] + 1
    if cross.size == 0:
        return cross.astype(np.int64)
    min_dist = max(1, int(round(refractory_s * fs)))
    kept: list[int] = [int(cross[0])]
    for c in cross[1:]:
        if c - kept[-1] >= min_dist:
            kept.append(int(c))
    return np.asarray(kept, dtype=np.int64)


def detect_spikes_threshold_rising(
    trace: np.ndarray,
    fs: float,
    threshold: float,
    refractory_s: float = 0.001,
) -> np.ndarray:
    """Compatibilité : délègue à detect_spikes_at_threshold (seuil négatif pris en charge)."""
    return detect_spikes_at_threshold(trace, fs, threshold, refractory_s=refractory_s)


def resolve_work_dir(config: AnalysisConfig) -> Path:
    """Répertoire pour amplifier_raw.npy et fichiers intermédiaires."""
    if config.work_dir is not None:
        return config.work_dir
    root = config.save_dir if config.save_dir is not None else config.rhs_file.parent
    return root / ".plot_erg" / f"{config.rhs_file.stem}_work"


class AmplifierSpikeSource:
    """Fenêtres alignées sur triggers, une canal à la fois (pas de tensor 3D complet)."""

    def __init__(
        self,
        amplifier: np.ndarray,
        valid_triggers: np.ndarray,
        pre_n: int,
        post_n: int,
        work_dir: Path | None,
        keep_intermediate_files: bool,
        fs: float,
        bandpass_low_hz: float | None = None,
        bandpass_high_hz: float | None = None,
    ) -> None:
        self.amplifier = amplifier
        self.valid_triggers = np.asarray(valid_triggers, dtype=np.int64)
        self.pre_n = pre_n
        self.post_n = post_n
        self._offsets = np.arange(-pre_n, post_n, dtype=np.int64)
        self.work_dir = work_dir
        self.keep_intermediate_files = keep_intermediate_files
        self.fs = float(fs)
        self.bandpass_low_hz = bandpass_low_hz
        self.bandpass_high_hz = bandpass_high_hz
        self._closed = False

    def windows_2d_for_channel(self, ch: int) -> np.ndarray:
        row = np.asarray(self.amplifier[ch], dtype=np.float64)
        if self.bandpass_low_hz is not None and self.bandpass_high_hz is not None:
            row_2d = row.reshape(1, -1)
            row = apply_butterworth_bandpass(
                row_2d, self.fs, self.bandpass_low_hz, self.bandpass_high_hz
            )[0]
        sample_idx = self.valid_triggers[:, None] + self._offsets[None, :]
        return np.take(row, sample_idx, axis=0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        a = self.amplifier
        try:
            if isinstance(a, np.memmap):
                a._mmap.close()
        except Exception:
            pass
        self.amplifier = np.empty((0,))  # libère la référence
        if self.work_dir is not None and self.work_dir.exists() and not self.keep_intermediate_files:
            shutil.rmtree(self.work_dir, ignore_errors=True)


def valid_triggers_and_timebase(
    n_samples: int,
    trigger_indices: np.ndarray,
    fs: float,
    pre_s: float,
    post_s: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Triggers utilisables et axe temporel relatif, sans extraire les fenêtres."""
    trigger_indices = np.asarray(trigger_indices, dtype=np.int64)
    pre_n = int(round(pre_s * fs))
    post_n = int(round(post_s * fs))
    win_len = pre_n + post_n
    if win_len <= 1:
        raise RuntimeError("Fenetre temporelle invalide.")
    start_idx = trigger_indices - pre_n
    end_idx = trigger_indices + post_n
    valid_mask = (start_idx >= 0) & (end_idx <= n_samples)
    valid_triggers = trigger_indices[valid_mask]
    if valid_triggers.size == 0:
        raise RuntimeError("Aucune fenetre valide autour des triggers.")
    t_rel = np.arange(-pre_n, post_n, dtype=np.float64) / fs
    return valid_triggers, t_rel, pre_n, post_n


def mean_filtered_channelwise(
    amplifier_2d: np.ndarray,
    valid_triggers: np.ndarray,
    fs: float,
    pre_n: int,
    post_n: int,
    cutoff_hz: float,
) -> np.ndarray:
    """Passe-bas canal par canal : jamais de matrice filtrée multi-canaux entière."""
    n_ch, _ = amplifier_2d.shape
    win_len = pre_n + post_n
    offsets = np.arange(-pre_n, post_n, dtype=np.int64)
    sample_idx = valid_triggers[:, None] + offsets[None, :]
    out = np.zeros((n_ch, win_len), dtype=np.float64)
    for c in range(n_ch):
        check_analysis_cancelled()
        row_f = apply_butterworth_lowpass(amplifier_2d[c : c + 1], fs, cutoff_hz)
        w = np.take(row_f, sample_idx, axis=1)[0]
        out[c] = np.mean(w, axis=0)
    return out


def persist_amplifier_float32(amplifier_2d: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(amplifier_2d, dtype=np.float32))


def extract_triggered_windows(
    amplifier_data: np.ndarray,
    trigger_indices: np.ndarray,
    fs: float,
    pre_s: float,
    post_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    amplifier_data = np.asarray(amplifier_data)
    if amplifier_data.ndim != 2:
        raise RuntimeError("Format inattendu pour amplifier_data (attendu [n_channels, n_samples]).")

    _, n_samples = amplifier_data.shape
    pre_n = int(round(pre_s * fs))
    post_n = int(round(post_s * fs))
    win_len = pre_n + post_n
    if win_len <= 1:
        raise RuntimeError("Fenetre temporelle invalide.")

    trigger_indices = np.asarray(trigger_indices, dtype=np.int64)
    start_idx = trigger_indices - pre_n
    end_idx = trigger_indices + post_n
    valid_mask = (start_idx >= 0) & (end_idx <= n_samples)
    valid_triggers = trigger_indices[valid_mask]

    if valid_triggers.size == 0:
        raise RuntimeError("Aucune fenetre valide autour des triggers.")

    offsets = np.arange(-pre_n, post_n, dtype=np.int64)
    sample_idx = valid_triggers[:, None] + offsets[None, :]
    windows = np.take(amplifier_data, sample_idx, axis=1).transpose(1, 0, 2)
    t_rel = np.arange(-pre_n, post_n, dtype=np.float64) / fs
    return windows, t_rel


def get_channel_names(data: dict[str, Any], n_channels: int) -> list[str]:
    channels = data.get("amplifier_channels", [])
    if isinstance(channels, list) and len(channels) == n_channels:
        names = []
        for i, ch in enumerate(channels):
            if isinstance(ch, dict):
                name = ch.get("native_channel_name") or ch.get("custom_channel_name") or f"CH{i}"
            else:
                name = f"CH{i}"
            names.append(str(name))
        return names
    return [f"CH{i}" for i in range(n_channels)]


def compute_average_per_channel(
    config: AnalysisConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    int,
    int,
    float,
    float | None,
    AmplifierSpikeSource,
    np.ndarray,
]:
    check_analysis_cancelled()
    if not config.rhs_file.exists():
        raise FileNotFoundError(f"Fichier introuvable: {config.rhs_file}")

    data = load_rhs_file(config.rhs_file)
    check_analysis_cancelled()
    fs = get_sampling_rate(data)
    analog_in0 = get_analog_in0_signal(data)
    trigger_indices = detect_edges(analog_in0, threshold=config.threshold, edge=config.edge)
    if trigger_indices.size == 0:
        edge_fr = "descendant" if config.edge == "falling" else "montant"
        raise RuntimeError(f"Aucun front {edge_fr} detecte sur ANALOG_IN 0.")

    amplifier_raw = np.asarray(data.get("amplifier_data"))
    if amplifier_raw.size == 0:
        raise RuntimeError("Le fichier RHS ne contient pas amplifier_data.")

    _, n_samples = amplifier_raw.shape
    valid_triggers, t_rel, pre_n, post_n = valid_triggers_and_timebase(
        n_samples=n_samples,
        trigger_indices=trigger_indices,
        fs=fs,
        pre_s=config.pre_s,
        post_s=config.post_s,
    )
    check_analysis_cancelled()

    channel_names = get_channel_names(data, amplifier_raw.shape[0])
    n_valid = int(valid_triggers.shape[0])
    n_total = int(trigger_indices.size)

    # Moyenne brute : extraction de toutes les fenêtres puis moyenne (tensor 3D temporaire).
    windows_raw, t_rel_mean = extract_triggered_windows(
        amplifier_data=amplifier_raw,
        trigger_indices=trigger_indices,
        fs=fs,
        pre_s=config.pre_s,
        post_s=config.post_s,
    )
    check_analysis_cancelled()
    if t_rel_mean.shape != t_rel.shape or not np.allclose(t_rel_mean, t_rel):
        raise RuntimeError("Incohérence grille temporelle (moyenne brute).")
    mean_per_channel_raw = windows_raw.mean(axis=0)
    del windows_raw

    if config.lowpass_cutoff_hz is not None:
        mean_per_channel = mean_filtered_channelwise(
            amplifier_raw,
            valid_triggers,
            fs,
            pre_n,
            post_n,
            config.lowpass_cutoff_hz,
        )
    else:
        mean_per_channel = mean_per_channel_raw

    end_rising_rel_s = mean_time_to_next_rising_edge_s(
        analog_in0, trigger_indices, config.threshold, fs
    )

    bp_lo: float | None = None
    bp_hi: float | None = None
    if config.spike_bandpass_low_hz is not None or config.spike_bandpass_high_hz is not None:
        if config.spike_bandpass_low_hz is None or config.spike_bandpass_high_hz is None:
            raise ValueError(
                "Passe-bande spikes : renseigner spike_bandpass_low_hz et spike_bandpass_high_hz, "
                "ou laisser les deux vides (signal brut)."
            )
        bp_lo = float(config.spike_bandpass_low_hz)
        bp_hi = float(config.spike_bandpass_high_hz)
        nyq = 0.5 * fs
        if bp_lo <= 0 or bp_hi <= 0:
            raise ValueError("Passe-bande spikes : les frequences doivent etre > 0 Hz.")
        if bp_lo >= bp_hi:
            raise ValueError("Passe-bande spikes : la frequence basse doit etre < la frequence haute.")
        if bp_hi >= nyq:
            raise ValueError(
                f"Passe-bande spikes : la frequence haute ({bp_hi:g} Hz) doit etre < Nyquist ({nyq:.1f} Hz)."
            )

    # Fichier mmap : libère le dict RHS et la copie RAM de l'amplificateur avant le PDF.
    work_dir = resolve_work_dir(config)
    amp_path = work_dir / "amplifier_raw.npy"
    persist_amplifier_float32(amplifier_raw, amp_path)
    del amplifier_raw
    if isinstance(data, dict):
        data.pop("amplifier_data", None)
    del data
    gc.collect()

    amp_mm = np.load(amp_path, mmap_mode="r")
    spike_source = AmplifierSpikeSource(
        amplifier=amp_mm,
        valid_triggers=valid_triggers,
        pre_n=pre_n,
        post_n=post_n,
        work_dir=work_dir,
        keep_intermediate_files=config.keep_intermediate_files,
        fs=fs,
        bandpass_low_hz=bp_lo,
        bandpass_high_hz=bp_hi,
    )

    return (
        mean_per_channel,
        t_rel,
        channel_names,
        n_valid,
        n_total,
        fs,
        end_rising_rel_s,
        spike_source,
        mean_per_channel_raw,
    )
