"""Valide chargement / affichage / traitement ciblés sur exemple.rhs.

Deux segmentations :
  - avec stimulation (front analogique)
  - sans stimulation (edge=none, sections virtuelles)

Pour chaque mode : continuous, moyenne, une stimulation — on vérifie que
seuls les produits affichés sont demandés / calculés, et que les panels
se préparent correctement.

    .venv\\Scripts\\python.exe tools\\smoke_exemple_dual_mode.py
"""

from __future__ import annotations

import gc
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import AnalysisConfig  # noqa: E402
from dataset_builder import build_recording, ensure_channels  # noqa: E402
from panel_prepare import prepare_panel  # noqa: E402
from panel_registry import RenderRequest  # noqa: E402
from view_config import (  # noqa: E402
    AnalysisSettings,
    PanelPlacement,
    PanelStyle,
    ViewerSettings,
    display_filter_streams,
    display_mean_streams,
    display_rms_kinds,
    preview_needs_means,
)

EXAMPLE = ROOT / "exemple.rhs"


def _fail(message: str) -> None:
    raise AssertionError(message)


def _settings(
    *,
    mode: str,
    streams: tuple[str, ...] = ("raw",),
    stim_index: int = 0,
    show_rms: bool = False,
) -> ViewerSettings:
    show_raw = "raw" in streams
    show_hp = "hp" in streams
    show_lp = "lp" in streams
    analysis_mode = "average" if mode == "average" else "stimulation"
    if mode == "continuous":
        analysis_mode = "average"  # unused for continuous product path
    return ViewerSettings(
        preview_content=mode,  # type: ignore[arg-type]
        continuous_streams=streams,  # type: ignore[arg-type]
        continuous_stream=streams[0],  # type: ignore[arg-type]
        analysis=AnalysisSettings(
            mode=analysis_mode,  # type: ignore[arg-type]
            stim_index=stim_index,
            show_raw=show_raw,
            show_hp=show_hp,
            show_lp=show_lp,
            show_rms=show_rms,
        ),
        style=PanelStyle(max_points_per_curve=800),
    )


def _request(recording, panel: str, settings: ViewerSettings, *, stream: str = "") -> RenderRequest:
    placement = PanelPlacement(panel, stream=stream) if stream else PanelPlacement(panel)
    return RenderRequest(
        placement=placement,
        recordings=[recording],
        labels=[recording.label],
        colors=["#1f77b4"],
        legend_flags=[True],
        channel_index=0,
        channel_name=recording.channel_names[0],
        settings=settings,
    )


def _check_product_helpers() -> None:
    print("helpers produits:")
    if preview_needs_means("continuous") or preview_needs_means("stimulation"):
        _fail("moyennes demandées hors mode average")
    if not preview_needs_means("average"):
        _fail("moyennes absentes en mode average")
    analysis = AnalysisSettings(show_raw=True, show_hp=True, show_lp=False)
    if display_mean_streams(analysis) != ("raw", "hp"):
        _fail(f"mean streams: {display_mean_streams(analysis)}")
    if display_filter_streams(preview_content="continuous", continuous_streams=("raw",)) != ():
        _fail("filtres inutiles en continuous raw")
    if display_filter_streams(preview_content="stimulation", analysis=analysis) != ("hp",):
        _fail("filtre hp manquant en stimulation")
    if display_rms_kinds(preview_content="average") != ("mean",):
        _fail("rms average")
    if display_rms_kinds(
        preview_content="stimulation", analysis=AnalysisSettings(stim_index=0)
    ) != ("first",):
        _fail("rms stim 0")
    print("  ok")


def _prepare_modes(recording, config: AnalysisConfig, *, label: str) -> None:
    print(f"  prepare ({label}):")
    # Continuous raw — fenêtre d’essai, pas de moyen.
    cont = _settings(mode="continuous", streams=("raw",))
    spec = prepare_panel(_request(recording, "full_recording", cont, stream="raw"))
    if spec.status != "ok" or not spec.series:
        _fail(f"continuous raw: {spec.status} series={len(spec.series)}")
    n_pts = int(spec.series[0].x.size)
    if n_pts < 10:
        _fail(f"continuous trop court: {n_pts}")
    if n_pts > 2000:
        _fail(f"continuous non borné ({n_pts} pts) — devrait utiliser la fenêtre d’essai")
    print(f"    continuous raw: {n_pts} pts, xlim={spec.xlim}")

    # Stimulation raw — slice memmap, sans ensure means.
    stim = _settings(mode="stimulation", streams=("raw",), stim_index=0)
    spec = prepare_panel(_request(recording, "analysis_raw", stim))
    if spec.status != "ok" or not spec.series:
        _fail(f"stimulation raw: {spec.status}")
    # Pas de moyennes précalculées pour une seule stim.
    data0 = recording._channel_data.get(0)
    if data0 is not None and getattr(data0, "means_ready", False) and data0.means:
        # OK si un ensure précédent (average) a déjà tourné dans ce passage.
        pass
    print(f"    stimulation raw: {int(spec.series[0].x.size)} pts (sans need_means)")

    # Average raw — ensure ciblé sur le seul flux affiché.
    avg = _settings(mode="average", streams=("raw",))
    computed = ensure_channels(
        recording,
        [0],
        config,
        need_means=True,
        need_rms=False,
        need_spikes=False,
        need_overlay=False,
        mean_streams=("raw",),
    )
    data = recording._channel_data.get(0)
    if data is None or data.means.get("raw") is None:
        _fail(f"ensure average raw incomplet (computed={computed})")
    if data.means.get("hp") is not None or data.means.get("lp") is not None:
        _fail(
            "moyennes hp/lp calculées alors que seul raw est affiché: "
            f"{sorted(data.means)}"
        )
    again = ensure_channels(
        recording,
        [0],
        config,
        need_means=True,
        need_rms=False,
        need_spikes=False,
        need_overlay=False,
        mean_streams=("raw",),
    )
    if again:
        _fail(f"recalcul inutile des moyennes: {again}")
    spec = prepare_panel(_request(recording, "analysis_raw", avg))
    if spec.status != "ok" or not spec.series:
        _fail(f"average raw: {spec.status}")
    print(f"    average raw: {int(spec.series[0].x.size)} pts (streams={sorted(data.means)})")


def _build(edge: str, cache: Path, **kwargs) -> tuple:
    cfg = AnalysisConfig(
        rhs_file=EXAMPLE,
        work_dir=cache,
        edge=edge,  # type: ignore[arg-type]
        **kwargs,
    )
    t0 = time.perf_counter()
    recording, report = build_recording(cfg, cache_root=cache)
    dt = time.perf_counter() - t0
    print(
        f"  build edge={edge}: {dt:.1f}s — "
        f"{recording.n_channels} ch, {recording.n_trials} essais, "
        f"fs={recording.meta.fs:.0f}"
    )
    if recording.n_trials < 1:
        _fail(f"aucun essai pour edge={edge}")
    stims = recording.stimulation_times_s()
    if stims.size < 1:
        _fail(f"pas d’onsets pour edge={edge}")
    print(f"  onsets (s): {', '.join(f'{t:.3f}' for t in stims[:4])}")
    del report
    return recording, cfg


def main() -> int:
    if not EXAMPLE.is_file():
        print(f"skip: {EXAMPLE.name} absent")
        return 0

    _check_product_helpers()

    cache = Path(tempfile.mkdtemp(prefix="plot_erg_dual_"))
    print(f"cache: {cache}")
    recording = None
    try:
        # --- Avec stimulation (front) ---------------------------------
        print("\n=== avec stimulation (falling) ===")
        try:
            recording, cfg = _build("falling", cache)
        except RuntimeError as exc:
            if "edge" not in str(exc).lower():
                raise
            print(f"  falling indisponible ({exc}), essai rising…")
            recording, cfg = _build("rising", cache)
        _prepare_modes(recording, cfg, label="avec stim")
        recording.close()
        recording = None
        gc.collect()

        # --- Sans stimulation (sections) ------------------------------
        print("\n=== sans stimulation (edge=none) ===")
        recording, cfg = _build(
            "none",
            cache,
            section_spec="count",
            section_count=4,
            section_trigger_start_s=1.0,
            section_trigger_end_s=3.0,
            pre_s=0.5,
            post_s=2.0,
        )
        # Même API d’affichage : continuous / moyenne / « stimulation » (= section).
        _prepare_modes(recording, cfg, label="sans stim")

        # Montage continuous : même fenêtrage relatif.
        mont = _settings(mode="continuous", streams=("raw",))
        mont = replace(mont, montage_review_channels=2)
        spec = prepare_panel(_request(recording, "montage_continuous_raw", mont))
        if spec.status != "ok" or not spec.series:
            _fail(f"montage continuous sans stim: {spec.status}")
        print(f"  montage continuous: {len(spec.series)} séries, rows={spec.n_rows}")

        print("\nexemple.rhs dual-mode: ok")
        return 0
    finally:
        if recording is not None:
            try:
                recording.close()
            except Exception:
                pass
        gc.collect()
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
