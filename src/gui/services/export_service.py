"""Export images de vue et helpers de tâches PDF / fond."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtWidgets import QFileDialog, QMessageBox

from panel_registry import RenderRequest
from view_config import PanelPlacement


class ExportService:
    """Grab des panneaux + TaskWorker ; helpers pour enrober une tâche PDF."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def save_view_images(self) -> None:
        host = self._host
        directory = QFileDialog.getExistingDirectory(
            host, "Choisir un dossier pour les images des panneaux", str(Path.home())
        )
        if not directory:
            return
        root = Path(directory)
        grabs: list[tuple[Path, Any]] = []

        # Mode aperçu canal (défaut) : exporter la grille de l’inspecteur.
        if host._showing_channel_preview() and host._channel_inspect is not None:
            inspect = host._channel_inspect
            channel = host.channel_panel.current_channel or "channel"
            safe_ch = "".join(c if c.isalnum() or c in "._-" else "_" for c in channel)
            factory = getattr(inspect, "_request_factory", None)

            def _inspect_request(placement: PanelPlacement) -> RenderRequest:
                if callable(factory):
                    request = factory(inspect, placement)
                    if request is not None:
                        return request
                return RenderRequest(
                    placement=placement,
                    recordings=[],
                    labels=[],
                    colors=[],
                    legend_flags=[],
                    channel_index=int(inspect.channel_index),
                    channel_name=str(inspect.channel_name),
                    settings=inspect.local_settings(),
                    probe_layout=None,
                    impedance_sessions=[],
                )

            inspect.grid.render_dirty_now(_inspect_request)
            for placement in inspect.grid.placements:
                canvas = inspect.grid.panel_widget(placement)
                if canvas is None:
                    continue
                safe = "".join(
                    c if c.isalnum() or c in "._-" else "_" for c in placement.key
                )
                out = root / f"{safe_ch}_{safe}.png"
                grabs.append((out, canvas.grab()))
        else:
            page = host._current_page()
            tab = host._current_tab()
            if page is None or tab is None or page.grid.panel_count == 0:
                QMessageBox.information(
                    host,
                    "Enregistrer des images",
                    "Cette vue n’a aucun panneau à enregistrer.",
                )
                return
            page.grid.render_dirty_now(host._make_request_or_blank)
            for placement in page.grid.placements:
                canvas = page.grid.panel_widget(placement)
                if canvas is None:
                    continue
                safe = "".join(
                    c if c.isalnum() or c in "._-" else "_" for c in placement.key
                )
                out = root / f"{tab.name}_{safe}.png"
                grabs.append((out, canvas.grab()))

        if not grabs:
            QMessageBox.information(
                host,
                "Enregistrer des images",
                "Aucun panneau n’a pu être enregistré.",
            )
            return

        def task() -> int:
            written = 0
            for path, pixmap in grabs:
                if pixmap.save(str(path), "PNG"):
                    written += 1
            return written

        def done(written: Any) -> None:
            count = int(written or 0)
            host._set_status(f"{count} image(s) écrite(s) dans {root}")
            host.status_panel.append_log(
                f"{count} image(s) de panneau écrite(s) dans {root}"
            )

        host._run_task(task, done, "Export des images de vue…")

    def run_pdf_task(
        self,
        task: Callable[[], Any],
        on_success: Callable[[Any], None],
        *,
        headline: str = "Génération du rapport PDF…",
    ) -> None:
        """Enrobe une tâche PDF via ``host._run_task``."""
        self._host._run_task(task, on_success, headline)
