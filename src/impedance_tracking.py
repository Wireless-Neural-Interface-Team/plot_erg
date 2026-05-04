"""Mesures d’impédance (export Intan CSV) associées aux enregistrements RHS."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

# Export typique RHX / Intan (colonne littérale)
_IMP_MAG_COL = "Impedance Magnitude at 1000 Hz (ohms)"
_CH_COL = "Channel Number"

_TS_SUFFIX = re.compile(r"_(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$")


def find_companion_impedance_csv(rhs_file: Path) -> Path | None:
    """Cherche CSV d’impédance dans le même dossier que le .rhs.

    Convention attendue : ``<dossier_parent>/<nom_du_dossier>.csv``
    (ex. dossier ``..._260430_144115`` + fichier ``..._260430_144115.csv``).
    """
    parent = rhs_file.parent
    for ext in (".csv", ".CSV"):
        candidate = parent / f"{parent.name}{ext}"
        if candidate.is_file():
            return candidate
    return None


def parse_recording_timestamp(rhs_file: Path) -> datetime:
    """Découpe la fin du nom (stem du .rhs puis du dossier) : suffixe ``_YYMMDD_HHMMSS``."""
    for fragment in (rhs_file.stem, rhs_file.parent.name):
        dt = _parse_yymmdd_hhmmss_suffix(fragment)
        if dt is not None:
            return dt
    return datetime.fromtimestamp(rhs_file.stat().st_mtime)


def _parse_yymmdd_hhmmss_suffix(text: str) -> datetime | None:
    m = _TS_SUFFIX.search(text)
    if not m:
        return None
    yy, mo, dd, hh, mi, ss = (int(g) for g in m.groups())
    if not (1 <= mo <= 12 and 1 <= dd <= 31 and 0 <= hh <= 23 and 0 <= mi <= 59 and 0 <= ss <= 59):
        return None
    year = 2000 + yy
    return datetime(year, mo, dd, hh, mi, ss)


def load_impedance_magnitude_1k_ohm(csv_path: Path) -> dict[str, float]:
    """Lit le CSV ; clé = nom de canal (ex. A-000), valeur = |Z| à 1 kHz (Ω)."""
    out: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return out

        def pick(d: dict[str, str | None], *candidates: str) -> str | None:
            for c in candidates:
                if c in d and d[c] is not None:
                    return d[c]
            return None

        for row in reader:
            ch = pick(row, _CH_COL, "Channel Name")
            if not ch or not str(ch).strip():
                continue
            mag_s = pick(row, _IMP_MAG_COL)
            if mag_s is None or not str(mag_s).strip():
                continue
            try:
                out[str(ch).strip()] = float(mag_s)
            except ValueError:
                continue
    return out


@dataclass(frozen=True)
class ImpedanceSession:
    when: datetime
    magnitudes_ohm: dict[str, float]
    rhs_path: Path
    rhs_label: str
    source_csv: Path


def collect_impedance_sessions(rhs_files: Sequence[Path]) -> list[ImpedanceSession]:
    """Pour chaque RHS : si un CSV companion existe, charge |Z|@1kHz et l’horodatage du nom."""
    sessions: list[ImpedanceSession] = []
    for rhs in rhs_files:
        csv_p = find_companion_impedance_csv(rhs)
        if csv_p is None:
            continue
        mag = load_impedance_magnitude_1k_ohm(csv_p)
        if not mag:
            continue
        when = parse_recording_timestamp(rhs)
        sessions.append(
            ImpedanceSession(
                when=when,
                magnitudes_ohm=mag,
                rhs_path=rhs,
                rhs_label=rhs.stem,
                source_csv=csv_p,
            )
        )
    sessions.sort(key=lambda s: (s.when, str(s.rhs_path)))
    return sessions
