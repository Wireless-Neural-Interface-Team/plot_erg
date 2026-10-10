"""Contrôle du pipeline build / ensure / busy pour la visionneuse."""

from __future__ import annotations

import contextlib
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PipelineHost(Protocol):
    """Attributs / helpers attendus sur ViewerWindow (duck-typed)."""

    _ensure_worker: Any
    _ensure_accepting: bool
    _build_worker: Any
    _channel_compute_active: bool
    _ensure_background: bool
    _pending_window_id: str | None

    def _set_channel_compute_active(
        self, active: bool, *, background: bool = False
    ) -> None: ...

    def _set_busy(self, busy: bool) -> None: ...

    def _start_channel_ensure(self, channels: list[int], **kwargs: Any) -> None: ...


class PipelineController:
    """Ensure/busy helpers + voisinage de canaux pour le prefetch."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def stop_ensure_worker(self, wait_ms: int = 250) -> None:
        """Stop channel ensure work; short wait only (never freeze the UI for seconds)."""
        host = self._host
        worker = host._ensure_worker
        host._ensure_accepting = False
        if worker is None:
            return
        if worker.isRunning():
            worker.request_stop()
            if not worker.wait(max(0, int(wait_ms))):
                # Detach: finished_all / deleteLater when the thread exits.
                with contextlib.suppress(RuntimeError):
                    worker.finished_all.disconnect()
                worker.setParent(None)
                worker.finished.connect(worker.deleteLater)
        host._ensure_worker = None
        host._set_channel_compute_active(False)

    def start_channel_ensure(self, channels: list[int], **kwargs: Any) -> None:
        """Façade vers ``host._start_channel_ensure``."""
        self._host._start_channel_ensure(channels, **kwargs)

    def ensure_channels_for_indices(
        self,
        channels: list[int],
        *,
        then_redraw_window: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Lancer le calcul des canaux manquants (vue active ou fenêtre détachée)."""
        if then_redraw_window is not None:
            self._host._pending_window_id = then_redraw_window
        self._host._start_channel_ensure(
            channels,
            allow_empty_redraw=then_redraw_window is None,
            **kwargs,
        )

    def set_busy(self, busy: bool) -> None:
        self._host._set_busy(busy)

    def set_channel_compute_active(
        self, active: bool, *, background: bool = False
    ) -> None:
        self._host._set_channel_compute_active(active, background=background)

    def pipeline_busy(self) -> bool:
        host = self._host
        return (
            (host._build_worker is not None and host._build_worker.isRunning())
            or (
                host._ensure_worker is not None
                and host._ensure_worker.isRunning()
                and not host._ensure_background
            )
        )

    def pipeline_ui_locked(self) -> bool:
        """True tant qu’un build ou un calcul de canaux possède la barre de statut."""
        host = self._host
        if host._build_worker is not None and host._build_worker.isRunning():
            return True
        return bool(host._channel_compute_active)

    @staticmethod
    def neighbor_channel_indices(
        center: int, n_channels: int, *, probe: Any | None = None
    ) -> list[int]:
        """Canaux proches (±2 en index, plus voisins MEA si la sonde est chargée)."""
        neighbors: set[int] = set()
        for delta in (-2, -1, 1, 2):
            ch = int(center) + delta
            if 0 <= ch < n_channels:
                neighbors.add(ch)
        if probe is not None:
            try:
                from probe_layout import neighboring_channel_indices

                for ch in neighboring_channel_indices(probe, int(center), max_count=6):
                    if 0 <= int(ch) < n_channels and int(ch) != int(center):
                        neighbors.add(int(ch))
            except Exception:
                pass
        return sorted(neighbors)
