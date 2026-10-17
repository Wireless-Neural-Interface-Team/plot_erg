"""Regression: montage HIGH/LOW rows stay empty until filters are ready, then fill."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from make_synthetic_dataset import build_synthetic  # noqa: E402
from panel_prepare import prepare_panel  # noqa: E402
from panel_registry import RenderRequest  # noqa: E402
from view_config import AnalysisSettings, PanelPlacement, ViewerSettings  # noqa: E402


class _LazyBank:
    """Minimal LazyFilterBank stub: rows ready only after prefetch."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data
        self._ready: set[int] = set()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def is_ready(self, ch: int) -> bool:
        return int(ch) in self._ready

    def prefetch(self, channels) -> None:
        for ch in channels:
            self._ready.add(int(ch))

    def __getitem__(self, ch: int) -> np.ndarray:
        if not self.is_ready(ch):
            raise RuntimeError(f"filter row {ch} not ready")
        return self._data[ch]


def _request(recording, streams=("raw", "hp", "lp")) -> RenderRequest:
    return RenderRequest(
        placement=PanelPlacement("montage_continuous_raw"),
        recordings=[recording],
        labels=[recording.label],
        colors=["#1f77b4"],
        legend_flags=[True],
        channel_index=0,
        channel_name=recording.channel_names[0],
        settings=ViewerSettings(
            preview_content="continuous",  # type: ignore[arg-type]
            continuous_streams=streams,  # type: ignore[arg-type]
            continuous_stream=streams[0],  # type: ignore[arg-type]
            analysis=AnalysisSettings(
                show_raw="raw" in streams,
                show_hp="hp" in streams,
                show_lp="lp" in streams,
            ),
            montage_review_channels=2,
        ),
        preserve_view=False,
    )


def main() -> int:
    recording = build_synthetic(label="montage-filters", seed=3, n_channels=2)
    n_ch = int(recording.n_channels)
    amp = np.random.default_rng(11).standard_normal((n_ch, 8_000), dtype=np.float32)
    hp = _LazyBank(amp * 0.25)
    lp = _LazyBank(amp * 0.7)
    recording.source = SimpleNamespace(amplifier=amp, highpass=hp, lowpass=lp)

    req = _request(recording)
    spec = prepare_panel(req)
    assert spec.status == "ok", f"expected ok with WIDE drawn, got {spec.status}"
    assert any(getattr(s, "row", None) == 0 for s in spec.series), "WIDE missing"
    texts = [a.text for a in spec.annotations if getattr(a, "kind", "") == "text"]
    assert any("calcul" in t for t in texts), f"expected calcul… annotation, got {texts}"
    hp_rows = [i for i, k in enumerate(spec.row_kinds) if k == "hp"]
    assert hp_rows, "no HIGH rows"
    hp_series = [s for s in spec.series if int(getattr(s, "row", -1)) in set(hp_rows)]
    assert not hp_series, "HIGH curves must be empty before prefetch"

    hp.prefetch([0, 1])
    lp.prefetch([0, 1])
    spec2 = prepare_panel(req)
    assert spec2.status == "ok"
    hp_series2 = [s for s in spec2.series if int(getattr(s, "row", -1)) in set(hp_rows)]
    assert hp_series2, "HIGH curves must appear after prefetch"
    lp_rows = [i for i, k in enumerate(spec2.row_kinds) if k == "lp"]
    lp_series2 = [s for s in spec2.series if int(getattr(s, "row", -1)) in set(lp_rows)]
    assert lp_series2, "LOW curves must appear after prefetch"
    pending_texts = [
        a.text for a in spec2.annotations if getattr(a, "kind", "") == "text" and "calcul" in a.text
    ]
    assert not pending_texts, f"stale calcul… after ready: {pending_texts}"

    print("ok — empty HIGH/LOW while pending, filled after prefetch")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        raise
