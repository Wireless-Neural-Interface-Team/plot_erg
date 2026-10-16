"""Guard: context/summary panels stay on the prepare → MPL path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panel_draw_mpl import PREPARE_MPL_PANELS, uses_prepare_mpl  # noqa: E402


REQUIRED = (
    "mea_layout",
    "impedance",
    "summary_rms",
    "summary_rms_table",
    "summary_impedance",
    "montage_continuous_raw",
    "full_recording",
)


def test_context_panels_use_prepare_mpl() -> None:
    for key in REQUIRED:
        assert key in PREPARE_MPL_PANELS
        assert uses_prepare_mpl(key)
