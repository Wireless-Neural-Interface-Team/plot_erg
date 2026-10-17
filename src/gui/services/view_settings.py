"""Fusion des réglages d’affichage pour la vue centrale.

Les plages (``range_bars``) ne sont **pas** portées par le dock : elles
appartiennent à ``ChannelAnalysisWindow``. Cette fusion ne les touche plus
(évite d’effacer puis de devoir réinjecter à chaque redraw).
"""

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

    ``range_bars`` / ``active_range_index`` : laissés tels quels dans ``base``
    (souvent vides depuis le dock). L’aperçu canal les réinjecte depuis son
    propre store via ``apply_viewer_settings``.
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
    )
