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

    def _set_channel_compute_active(
        self, active: bool, *, background: bool = False
    ) -> None: ...

    def _set_busy(self, busy: bool) -> None: ...

    def _start_channel_ensure(self, channels: list[int], **kwargs: Any) -> None: ...


class PipelineController:
    """Délègue au host ; encapsule l’arrêt ensure et les helpers busy."""

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
