"""État montage (revue multi-canaux) pour ViewerWindow."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MontageHost(Protocol):
    """Callbacks UI attendus quand le mode montage change."""

    def sync_montage_ui(self, *, montage: bool) -> None: ...


class MontageController:
    """Flag ``in_montage`` + enter/leave qui rappellent l’hôte pour la synchro UI."""

    def __init__(self, host: Any) -> None:
        self._host = host
        self._in_montage = False

    @property
    def in_montage(self) -> bool:
        return self._in_montage

    def set_in_montage(self, montage: bool) -> None:
        """Met à jour uniquement le flag (sans callback UI)."""
        self._in_montage = bool(montage)

    def enter(self) -> None:
        self._in_montage = True
        self._notify(True)

    def leave(self) -> None:
        self._in_montage = False
        self._notify(False)

    def _notify(self, montage: bool) -> None:
        host = self._host
        sync = getattr(host, "sync_montage_ui", None)
        if callable(sync):
            sync(montage=montage)
