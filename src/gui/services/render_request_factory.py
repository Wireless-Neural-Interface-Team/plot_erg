"""Construction de ``RenderRequest`` pour la grille / les fenêtres détachées."""

from __future__ import annotations

from typing import Any, Sequence

from display_config import resolve_recording_plot_colors
from panel_registry import RenderRequest
from view_config import PanelPlacement, ViewerSettings


def impedance_sessions(entries: Sequence[Any]) -> list[Any]:
    """Sessions d’impédance uniques, dans l’ordre d’apparition."""
    sessions: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        for session in getattr(entry.recording, "impedance_sessions", []) or []:
            marker = (str(getattr(session, "label", "")), str(getattr(session, "csv_path", "")))
            if marker in seen:
                continue
            seen.add(marker)
            sessions.append(session)
    return sessions


class RenderRequestFactory:
    """Construit des RenderRequest ; lit probe / preserve_view sur l’hôte."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def build(
        self,
        placement: PanelPlacement,
        entries: Sequence[Any],
        *,
        channel_index: int,
        channel_name: str,
        settings: ViewerSettings,
        highlight_zooms: tuple[tuple[float, float, str], ...] = (),
        preserve_view: bool | None = None,
    ) -> RenderRequest:
        host = self._host
        if preserve_view is None:
            preserve_view = bool(getattr(host, "_preserve_view", False))
        entry_list = list(entries)
        return RenderRequest(
            placement=placement,
            recordings=[entry.recording for entry in entry_list],
            labels=[entry.display_label for entry in entry_list],
            colors=resolve_recording_plot_colors(
                [entry.style for entry in entry_list], list(range(len(entry_list)))
            ),
            legend_flags=[entry.style.legend_visible for entry in entry_list],
            channel_index=channel_index,
            channel_name=channel_name,
            settings=settings,
            probe_layout=getattr(host, "_probe_layout", None),
            impedance_sessions=impedance_sessions(entry_list),
            highlight_zooms=highlight_zooms,
            preserve_view=bool(preserve_view),
        )

    def blank(
        self,
        placement: PanelPlacement,
        *,
        settings: ViewerSettings | None = None,
        preserve_view: bool | None = None,
    ) -> RenderRequest:
        host = self._host
        if settings is None:
            settings = getattr(host, "_settings", None)
        if preserve_view is None:
            preserve_view = bool(getattr(host, "_preserve_view", False))
        return RenderRequest(
            placement=placement,
            recordings=[],
            labels=[],
            colors=[],
            legend_flags=[],
            channel_index=0,
            channel_name="",
            settings=settings,
            probe_layout=getattr(host, "_probe_layout", None),
            impedance_sessions=[],
            preserve_view=bool(preserve_view),
        )

    def make_request(self, placement: PanelPlacement) -> RenderRequest | None:
        """Façade : délègue à ``host._make_request`` si présent."""
        make = getattr(self._host, "_make_request", None)
        if callable(make):
            return make(placement)
        return None

    def make_request_or_blank(self, placement: PanelPlacement) -> RenderRequest:
        """Façade : délègue à ``host._make_request_or_blank`` si présent."""
        make = getattr(self._host, "_make_request_or_blank", None)
        if callable(make):
            return make(placement)
        request = self.make_request(placement)
        if request is not None:
            return request
        return self.blank(placement)
