"""Planification coalescée des redessins de la visionneuse."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject

from gui.jobs import Debouncer


class RedrawScheduler:
    """Possède le Debouncer et les flags force / inspectors / preserve / block."""

    def __init__(
        self,
        parent: QObject | None,
        on_redraw: Callable[[], None],
        *,
        interval_ms: int = 110,
    ) -> None:
        self._force = False
        self._inspectors = False
        self._preserve_view = False
        self._block_preserve_view = False
        self._on_redraw = on_redraw
        self._debouncer = Debouncer(interval_ms, parent)
        self._debouncer.triggered.connect(self._emit_redraw)

    # ------------------------------------------------------------------ flags

    @property
    def force(self) -> bool:
        return self._force

    @force.setter
    def force(self, value: bool) -> None:
        self._force = bool(value)

    @property
    def inspectors(self) -> bool:
        return self._inspectors

    @inspectors.setter
    def inspectors(self, value: bool) -> None:
        self._inspectors = bool(value)

    @property
    def preserve_view(self) -> bool:
        return self._preserve_view

    @preserve_view.setter
    def preserve_view(self, value: bool) -> None:
        self._preserve_view = bool(value)

    @property
    def block_preserve_view(self) -> bool:
        return self._block_preserve_view

    @block_preserve_view.setter
    def block_preserve_view(self, value: bool) -> None:
        self._block_preserve_view = bool(value)

    # ---------------------------------------------------------------- schedule

    def schedule(
        self,
        *,
        force: bool = False,
        inspectors: bool = False,
        preserve_view: bool = False,
        reset_view: bool = False,
    ) -> None:
        self._force = self._force or force
        self._inspectors = self._inspectors or inspectors
        if reset_view:
            self._preserve_view = False
            self._block_preserve_view = True
        elif preserve_view and not self._block_preserve_view:
            self._preserve_view = True
        self._debouncer.request()

    def cancel(self) -> None:
        self._debouncer.cancel()

    def begin_redraw(self) -> tuple[bool, bool, bool]:
        """Consomme les flags et renvoie ``(force, inspectors, preserve_view)``."""
        force = self._force
        inspectors = self._inspectors
        preserve_view = self._preserve_view and not self._block_preserve_view
        self._force = False
        self._inspectors = False
        self._block_preserve_view = False
        # Garder preserve_view pendant toute la file de rendus async.
        self._preserve_view = preserve_view
        return force, inspectors, preserve_view

    def _emit_redraw(self) -> None:
        callback = self._on_redraw
        if callback is not None:
            callback()
