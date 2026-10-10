"""Fusion des réglages d’affichage pour la vue centrale."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from view_config import ViewerSettings


def merge_central_viewer_settings(
    base: ViewerSettings,
    *,
    hidden_channels: Sequence[str] = (),
    channel_inspect: Any | None = None,
) -> ViewerSettings:
    """Combine ParamsPanel settings with channel-inspector overrides.

    Pipeline checkboxes (in ``base.analysis``) remain the source of truth for
    which curves are drawn; the inspector only overrides mode / stim / summaries.
    """
    streams = base.resolved_continuous_streams()
    analysis = base.analysis
    preview_content = base.preview_content
    if channel_inspect is not None:
        local = channel_inspect.local_settings()
        preview_content = channel_inspect.preview_mode()
        analysis = replace(
            analysis,
            mode=local.analysis.mode,
            stim_index=local.analysis.stim_index,
            show_summary_rms=local.analysis.show_summary_rms,
            show_summary_rms_table=local.analysis.show_summary_rms_table,
        )
    return replace(
        base,
        analysis=analysis,
        preview_content=preview_content,  # type: ignore[arg-type]
        continuous_stream=streams[0] if streams else "raw",  # type: ignore[arg-type]
        continuous_streams=streams,
        hidden_channels=tuple(hidden_channels),
        range_bars=(),
        active_range_index=0,
    )
