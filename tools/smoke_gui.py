"""Offscreen smoke test of the interactive viewer.

Loads two synthetic recordings and a synthetic probe into the real window, lays
out every catalogued panel, and checks that each one draws. Run with:

    si_env\\Scripts\\python.exe tools\\smoke_gui.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT.parent / "src"))
sys.path.insert(0, str(_ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.defaults import app_defaults_from_config  # noqa: E402
from gui.main_window import ViewerWindow  # noqa: E402
from gui.widgets.panel_grid import ViewTabPage  # noqa: E402
from gui.widgets.recordings_panel import RecordingEntry  # noqa: E402
from make_synthetic_dataset import build_synthetic  # noqa: E402
from panel_registry import PANEL_CATALOG  # noqa: E402
from view_config import PanelPlacement, ViewTab, WorkspaceLayout  # noqa: E402


# Synthetic dataset: pas de CSV d’impédance ni de flux continus (seulement essais).
_EXPECTED_UNAVAILABLE = {
    "impedance",
    "summary_impedance",
    "full_recording",
    "montage_continuous_raw",
}


def _write_probe_json(directory: Path, channel_names: list[str]) -> Path:
    """An 8-contact probe: half mapped to real channels, half not connected."""
    electrodes = []
    for index in range(len(channel_names) + 4):
        intan_id = channel_names[index] if index < len(channel_names) else ""
        electrodes.append(
            {
                "eid": index,
                "x": float(200 * (index % 4)),
                "y": float(200 * (index // 4)),
                "intan_id": intan_id,
                "potentiostat_id": index,
            }
        )
    path = directory / "probe.json"
    path.write_text(
        json.dumps({"specification": "mea_editor", "electrodes": electrodes}, indent=2),
        encoding="utf-8",
    )
    return path


def _all_placements() -> list[PanelPlacement]:
    placements: list[PanelPlacement] = []
    for info in PANEL_CATALOG:
        if info.is_global:
            placements.append(PanelPlacement(info.key))
        else:
            for section in ("full", "zoom_onset", "zoom_trigger_end"):
                placements.append(PanelPlacement(info.key, section))  # type: ignore[arg-type]
    return placements


def _wait_for_render(app: QApplication, page: ViewTabPage, timeout_s: float = 180.0) -> float:
    """Wait until the queued (on-screen) panels have been drawn."""
    started = time.perf_counter()
    time.sleep(0.2)  # let the redraw debouncer fire
    while time.perf_counter() - started < timeout_s:
        app.processEvents()
        if not page.grid._pending:
            break
        time.sleep(0.005)
    return time.perf_counter() - started


def _scroll_through(app: QApplication, page: ViewTabPage, timeout_s: float = 300.0) -> float:
    """Scroll to the bottom so deferred panels get drawn, as a user would."""
    started = time.perf_counter()
    bar = page.grid.verticalScrollBar()
    step = max(1, page.grid.viewport().height())
    position = 0
    while time.perf_counter() - started < timeout_s:
        bar.setValue(position)
        app.processEvents()
        time.sleep(0.15)
        _wait_for_render(app, page, timeout_s=60.0)
        if not page.grid._deferred:
            break
        if position >= bar.maximum():
            break
        position = min(bar.maximum(), position + step)
    return time.perf_counter() - started


def main() -> int:
    app = QApplication.instance() or QApplication([])
    window = ViewerWindow(app_defaults_from_config())
    window.show()
    app.processEvents()

    recordings = [
        build_synthetic(label="control", seed=1),
        build_synthetic(label="treated", seed=2),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        probe_path = _write_probe_json(tmp_dir, list(recordings[0].channel_names))
        window._defaults["default_probe_layout_json"] = probe_path
        window._apply_probe_from_defaults()
        app.processEvents()
        assert window._probe_layout is not None, "probe layout was not loaded"
        assert window.channel_panel.map.has_probe, "MEA map has no contact"

        panel = window.recordings_panel
        for index, recording in enumerate(recordings):
            entry = RecordingEntry(
                row_id=1000 + index,
                path=tmp_dir / f"{recording.label}.rhs",
                label=recording.label,
                recording=recording,
                status="ready",
            )
            panel._entries.append(entry)
        panel._rebuild_table()
        window._refresh_channels()
        app.processEvents()

        assert window.channel_panel.channels, "no channel listed"
        mapped = window.channel_panel.map.mapped_channels()
        assert mapped, "no probe contact matched a recording channel"

        window._workspace = WorkspaceLayout(
            tabs=(ViewTab(name="All panels", panels=tuple(_all_placements()), columns=3),)
        )
        window._rebuild_tabs()
        page = window.tabs.currentWidget()
        assert isinstance(page, ViewTabPage)
        elapsed = _wait_for_render(app, page)
        print(
            f"first paint:  {page.grid._drawn} on-screen panel(s) in {elapsed:.2f} s, "
            f"{len(page.grid._deferred)} deferred off-screen"
        )
        scroll_s = _scroll_through(app, page)
        print(f"scrolled through the whole view in {scroll_s:.2f} s")
        forced = page.grid.render_dirty_now(window._make_request_or_blank)
        app.processEvents()
        print(f"forced the {forced} remaining panel(s), as an export would")

        statuses: dict[str, int] = {}
        broken: list[str] = []
        notes: list[str] = []
        build_s = 0.0
        slowest: list[tuple[float, str]] = []
        for placement in page.grid.placements:
            widget = page.grid.panel_widget(placement)
            status = getattr(widget, "_status", "?")
            statuses[status] = statuses.get(status, 0) + 1
            build_s += float(getattr(widget, "last_render_s", 0.0))
            slowest.append((float(getattr(widget, "last_render_s", 0.0)), placement.key))
            if status == "unavailable" and placement.panel not in _EXPECTED_UNAVAILABLE:
                broken.append(placement.key)
            elif status in {"empty", "unavailable"}:
                notes.append(placement.key)
        print(f"panels drawn: {page.grid.panel_count} in {elapsed:.2f} s")
        print(f"statuses:     {statuses}")
        print(f"artist build:  {build_s:.2f} s   (rest is Qt painting)")
        slowest.sort(reverse=True)
        for seconds, key in slowest[:5]:
            print(f"  {seconds * 1000:7.1f} ms  {key}")
        if broken:
            print("UNEXPECTEDLY UNAVAILABLE: " + ", ".join(broken))
        if notes:
            print("nothing to draw (expected): " + ", ".join(notes))

        # Changing a channel from the MEA map must trigger a redraw.
        window.channel_panel.map.channelSelected.emit(mapped[-1])
        app.processEvents()
        assert window.channel_panel.current_channel == mapped[-1]
        redraw_s = _wait_for_render(app, page)
        print(f"channel switch redraw: {redraw_s:.2f} s")

        # Aperçu canal embarqué : paramètres d’affichage locaux.
        from dataclasses import replace as dc_replace
        from gui.widgets.channel_analysis_window import ChannelAnalysisWindow
        from view_config import AnalysisSettings

        channel = window.channel_panel.current_channel or mapped[0]
        resolved = recordings[0].channel_index(channel)
        assert resolved is not None
        seed = dc_replace(window._seed_settings_for_window(), psth_bin_window_s=0.1)
        inspector = ChannelAnalysisWindow(
            channel_name=channel,
            channel_index=int(resolved),
            analysis=AnalysisSettings(show_raw=True),
            base_settings=seed,
            parent=window,
            embedded=False,
        )
        assert abs(inspector.settings.psth_bin_window_s - 0.1) < 1e-9
        print("local channel-inspector parameter OK")
        inspector.close()
        app.processEvents()

        # The PDF path must still receive a coherent configuration.
        from gui.defaults import build_config_from_defaults

        config = build_config_from_defaults(window._defaults, tmp_dir / "control.rhs")
        display = window._workspace.to_plot_display()
        assert config.rhs_file.name == "control.rhs"
        assert config.probe_layout_json == probe_path
        assert config.spike_threshold_uv < 0, "negative polarity must give a signed threshold"
        assert abs(config.psth_bin_window_s - 0.025) < 1e-9 or abs(config.psth_bin_window_s - 0.05) < 1e-9 or abs(config.psth_bin_window_s - 0.1) < 1e-9
        assert window._workspace.zoom_mode() in {"none", "onset", "trigger_end", "both"}
        # Catalogue all-panels : au moins une section PDF doit avoir des panneaux.
        assert (
            display.section_panels("full").any_enabled()
            or display.section_panels("zoom_onset").any_enabled()
            or display.mea_layout
            or any(
                p.panel == "full_recording"
                for tab in window._workspace.tabs
                for p in tab.panels
            )
        )
        print(f"PDF config OK — zoom mode {window._workspace.zoom_mode()}")

        # Vue par défaut (aperçu canal embarqué) — ce que voit vraiment l’utilisateur.
        window.reset_views()
        app.processEvents()
        window._sync_channel_inspect(redraw=True)
        app.processEvents()
        inspect = window._channel_inspect
        assert inspect is not None, "aperçu canal embarqué attendu après reset_views"
        # Attendre le rendu de la grille embarquée.
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            app.processEvents()
            if inspect.grid._drawn >= 1 or inspect.grid.panel_count == 0:
                if not getattr(inspect.grid, "_pending", None):
                    break
            time.sleep(0.05)
        print(
            f"default channel preview: "
            f"{inspect.grid._drawn}/{inspect.grid.panel_count} panel(s) "
            f"channel={inspect.channel_name}"
        )
        assert inspect.grid.panel_count >= 1

        window.close()
        app.processEvents()
    if broken:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
