from __future__ import annotations

import contextlib
import contextvars
import importlib.util
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
) -> tuple[np.ndarray, np.ndarray, list[str], int, int, float, float | None]:
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

    amplifier_data = np.asarray(data.get("amplifier_data"))
    if amplifier_data.size == 0:
        raise RuntimeError("Le fichier RHS ne contient pas amplifier_data.")

    if config.lowpass_cutoff_hz is not None:
        amplifier_data = apply_butterworth_lowpass(
            amplifier_data,
            fs=fs,
            cutoff_hz=config.lowpass_cutoff_hz,
        )
        check_analysis_cancelled()

    windows, t_rel = extract_triggered_windows(
        amplifier_data=amplifier_data,
        trigger_indices=trigger_indices,
        fs=fs,
        pre_s=config.pre_s,
        post_s=config.post_s,
    )
    check_analysis_cancelled()
    mean_per_channel = windows.mean(axis=0)
    channel_names = get_channel_names(data, mean_per_channel.shape[0])
    end_rising_rel_s = mean_time_to_next_rising_edge_s(
        analog_in0, trigger_indices, config.threshold, fs
    )
    return (
        mean_per_channel,
        t_rel,
        channel_names,
        windows.shape[0],
        trigger_indices.size,
        fs,
        end_rising_rel_s,
    )
