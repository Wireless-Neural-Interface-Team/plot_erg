from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

EdgeKind = Literal["falling", "rising"]


@dataclass(frozen=True)
class AnalysisConfig:
    rhs_file: Path
    threshold: float = 1.0
    edge: EdgeKind = "falling"  # falling=descendant, rising=montant au seuil sur ANALOG_IN 0
    pre_s: float = 2.0
    post_s: float = 10.0
    # Passe-bas Butterworth sur amplifier_data (None = desactive)
    lowpass_cutoff_hz: Optional[float] = None
    save_dir: Path | None = None
