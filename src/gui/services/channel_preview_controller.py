"""Helpers d’affichage du placeholder d’aperçu canal."""

from __future__ import annotations

from typing import Any


class ChannelPreviewController:
    """Show / hide du placeholder d’aperçu (widget fourni par l’hôte)."""

    def __init__(self, placeholder: Any) -> None:
        self._placeholder = placeholder

    def show_placeholder(self) -> None:
        widget = self._placeholder
        if widget is not None:
            widget.show()

    def hide_placeholder(self) -> None:
        widget = self._placeholder
        if widget is not None:
            widget.hide()

    def set_placeholder_visible(self, visible: bool) -> None:
        if visible:
            self.show_placeholder()
        else:
            self.hide_placeholder()
