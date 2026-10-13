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
        # Couleurs Auto stables : index dans la table complète, pas parmi les seuls visibles.
        all_entries = list(getattr(getattr(host, "recordings_panel", None), "entries", []) or [])
        if all_entries:
            id_to_index = {
                int(getattr(entry, "row_id", -1)): i for i, entry in enumerate(all_entries)
            }
            styles = [entry.style for entry in all_entries]
            color_indices = [
                id_to_index.get(int(getattr(entry, "row_id", -1)), i)
                for i, entry in enumerate(entry_list)
            ]
        else:
            styles = [entry.style for entry in entry_list]
            color_indices = list(range(len(entry_list)))
        return RenderRequest(
            placement=placement,
            recordings=[entry.recording for entry in entry_list],
            labels=[entry.display_label for entry in entry_list],
            colors=resolve_recording_plot_colors(styles, color_indices),
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

