"""Pont d’accès pour découpler les services GUI des ``_privé`` de ViewerWindow."""

from __future__ import annotations

from typing import Any, Callable, Optional

from panel_registry import RenderRequest
from view_config import PanelPlacement


class ViewerServiceBridge:
    """API publique stable injectée dans PipelineController / ExportService / etc."""

    def __init__(self, window: Any) -> None:
        self._window = window

    # --- workers / busy ------------------------------------------------------
    @property
    def ensure_worker(self) -> Any:
        return self._window._ensure_worker

    @ensure_worker.setter
    def ensure_worker(self, value: Any) -> None:
        self._window._ensure_worker = value

    @property
    def build_worker(self) -> Any:
        return self._window._build_worker

    @property
    def ensure_accepting(self) -> bool:
        return bool(self._window._ensure_accepting)

    @ensure_accepting.setter
    def ensure_accepting(self, value: bool) -> None:
        self._window._ensure_accepting = bool(value)

    @property
    def ensure_background(self) -> bool:
        return bool(self._window._ensure_background)

    @property
    def channel_compute_active(self) -> bool:
        return bool(self._window._channel_compute_active)

    @property
    def pending_window_id(self) -> Optional[str]:
        return self._window._pending_window_id

    @pending_window_id.setter
    def pending_window_id(self, value: Optional[str]) -> None:
        self._window._pending_window_id = value

    def set_channel_compute_active(self, active: bool) -> None:
        self._window._set_channel_compute_active(bool(active))

    def set_busy(self, busy: bool) -> None:
        self._window._set_busy(bool(busy))

    def start_channel_ensure(self, channels: list[int], **kwargs: Any) -> None:
        self._window._start_channel_ensure(channels, **kwargs)

    # --- export / statut -----------------------------------------------------
    @property
    def channel_inspect(self) -> Any:
        return self._window._channel_inspect

    @property
    def channel_panel(self) -> Any:
        return self._window.channel_panel

    @property
    def status_panel(self) -> Any:
        return self._window.status_panel

    @property
    def params_panel(self) -> Any:
        return self._window.params_panel

    @property
    def recordings_panel(self) -> Any:
        return self._window.recordings_panel

    def showing_channel_preview(self) -> bool:
        return bool(self._window._showing_channel_preview())

    def current_page(self) -> Any:
        return self._window._current_page()

    def current_tab(self) -> Any:
        return self._window._current_tab()

    def make_request_or_blank(self, placement: PanelPlacement) -> RenderRequest:
        return self._window._make_request_or_blank(placement)

    def set_status(self, message: str, *, force: bool = False) -> None:
        self._window._set_status(message, force=force)

    def run_task(
        self,
        task: Callable[[], Any],
        on_success: Callable[[Any], None],
        headline: str,
    ) -> None:
        self._window._run_task(task, on_success, headline)

    def append_log(self, message: str) -> None:
        self._window.status_panel.append_log(message)

    def processing_channel_indices(self, recording: Any) -> list[int] | None:
        return self._window._processing_channel_indices(recording)

    def build_config(self, path: Any) -> Any:
        return self._window._build_config(path)

    def workspace_plot_display(self) -> Any:
        return self._window._workspace.to_plot_display()

    def workspace_zoom_mode(self) -> str:
        zoom_mode = self._window._workspace.zoom_mode()
        if not any(tab.panels for tab in self._window._workspace.tabs):
            return "both"
        return str(zoom_mode)

    @property
    def preserve_view(self) -> bool:
        return bool(getattr(self._window, "_preserve_view", False))

    @property
    def probe_layout(self) -> Any:
        return getattr(self._window, "_probe_layout", None)

    @property
    def settings(self) -> Any:
        return getattr(self._window, "_settings", None)

    def make_request(self, placement: PanelPlacement) -> RenderRequest | None:
        make = getattr(self._window, "_make_request", None)
        if callable(make):
            return make(placement)
        return None
