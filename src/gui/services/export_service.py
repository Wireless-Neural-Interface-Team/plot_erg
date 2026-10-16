"""Export images de vue, PDF et helpers de tâches de fond."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from panel_registry import RenderRequest
from view_config import PanelPlacement


class ExportService:
    """Grab des panneaux + TaskWorker ; logique PDF / images hors ViewerWindow."""

    def __init__(self, host: Any, *, dialog_parent: QWidget | None = None) -> None:
        self._host = host
        self._dialog_parent = dialog_parent if dialog_parent is not None else host

    def save_view_images(self) -> None:
        host = self._host
        parent = self._dialog_parent
        directory = QFileDialog.getExistingDirectory(
            parent, "Choisir un dossier pour les images des panneaux", str(Path.home())
        )
        if not directory:
            return
        root = Path(directory)
        grabs: list[tuple[Path, Any]] = []

        # Mode aperçu canal (défaut) : exporter la grille de l’inspecteur.
        if host.showing_channel_preview() and host.channel_inspect is not None:
            inspect = host.channel_inspect
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
            page = host.current_page()
            tab = host.current_tab()
            if page is None or tab is None or page.grid.panel_count == 0:
                QMessageBox.information(
                    parent,
                    "Enregistrer des images",
                    "Cette vue n’a aucun panneau à enregistrer.",
                )
                return
            page.grid.render_dirty_now(host.make_request_or_blank)
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
                parent,
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
            host.set_status(f"{count} image(s) écrite(s) dans {root}")
            host.append_log(f"{count} image(s) de panneau écrite(s) dans {root}")

        host.run_task(task, done, "Export des images de vue…")

    def run_pdf_task(
        self,
        task: Callable[[], Any],
        on_success: Callable[[Any], None],
        *,
        headline: str = "Génération du rapport PDF…",
    ) -> None:
        """Enrobe une tâche PDF via ``host.run_task``."""
        self._host.run_task(task, on_success, headline)

    def export_pdf_report(
        self,
        *,
        pdf_callback: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        """Exporter un PDF depuis les enregistrements prêts (RHS live ou .ergproc avec flux)."""
        from config import AnalysisConfig

        host = self._host
        parent = self._dialog_parent
        ready_with_streams = [
            e
            for e in host.recordings_panel.ready_entries()
            if e.recording is not None
            and getattr(e.recording, "source", None) is not None
        ]
        ready_processed = [
            e
            for e in host.recordings_panel.ready_entries()
            if e.recording is not None and e.is_processed
        ]
        entries = ready_with_streams or [
            e for e in host.recordings_panel.entries if not e.is_processed
        ]
        if not entries and ready_processed:
            QMessageBox.information(
                parent,
                "Rapport PDF",
                "Les datasets traités ouverts n’incluent pas de flux continus.\n"
                "Réexportez le dataset avec les flux (ampli / HP / LP), ou rouvrez "
                "le .rhs source puis Traiter (F5).",
            )
            return
        if not entries:
            QMessageBox.information(
                parent,
                "Rapport PDF",
                "Ajoutez et traitez au moins un enregistrement .rhs (F5), "
                "ou ouvrez un dataset .ergproc contenant les flux, avant l’export PDF.",
            )
            return
        if pdf_callback is None and not ready_with_streams:
            QMessageBox.information(
                parent, "Rapport PDF", "Le pipeline PDF est indisponible dans cette session."
            )
            return
        default_name = f"{entries[0].display_label}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            parent,
            "Enregistrer le rapport PDF",
            str(entries[0].path.parent / default_name),
            "PDF (*.pdf)",
        )
        if not path:
            return
        target = Path(path)
        display = host.workspace_plot_display()
        zoom_mode = host.workspace_zoom_mode()
        configs: list[AnalysisConfig] = []
        recordings: list[Any] = []
        for entry in entries:
            configs.append(
                replace(
                    host.build_config(entry.path),
                    save_dir=target.parent,
                    pdf_title=target.stem,
                    recording_label=entry.label or None,
                    recording_style=entry.style,
                    plot_display=display,
                    zoom_mode=zoom_mode,  # type: ignore[arg-type]
                )
            )
            if entry.recording is not None and getattr(entry.recording, "source", None) is not None:
                recordings.append(entry.recording)
        selected = (
            host.processing_channel_indices(recordings[0]) if recordings else None
        )
        style = host.params_panel.viewer_settings().style

        def task() -> str:
            if recordings and len(recordings) == len(configs):
                from plotting import plot_processed_recordings_pdf

                plot_processed_recordings_pdf(
                    recordings,
                    configs,
                    output_dir=target.parent,
                    channels=selected,
                    stim_linewidth=float(style.stim_linewidth),
                    stim_onset_color=str(style.stim_onset_color),
                    stim_offset_color=str(style.stim_offset_color),
                )
            elif pdf_callback is not None:
                pdf_callback(configs)
            else:
                raise RuntimeError("Aucun chemin PDF disponible.")
            return str(target)

        def done(result: Any) -> None:
            host.append_log(f"Rapport PDF écrit près de {result}")
            QMessageBox.information(
                parent, "Rapport PDF", f"Rapport généré dans :\n{target.parent}"
            )

        self.run_pdf_task(task, done)
