"""État et helpers de la revue montage (multi-canaux) pour ViewerWindow."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from view_config import PanelPlacement


@runtime_checkable
class MontageHost(Protocol):
    """Callbacks UI attendus quand le mode montage change."""

    def sync_montage_ui(self, *, montage: bool) -> None: ...


class MontageController:
    """Flag ``in_montage`` + helpers purs (pagination, hauteur, produits)."""

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

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def is_montage_placement(placement: Any) -> bool:
        panel = str(getattr(placement, "panel", "") or "")
        return panel == "montage_continuous_raw" or panel.startswith("montage_")

    @staticmethod
    def analysis_extra_flags(analysis: Any) -> tuple[bool, bool, bool]:
        """``(need_rms, need_spikes, need_overlay)`` depuis les flags Pipeline / Analyse."""
        need_rms = bool(
            getattr(analysis, "show_rms", False)
            or getattr(analysis, "show_summary_rms", False)
            or getattr(analysis, "show_summary_rms_table", False)
        )
        need_spikes = bool(
            getattr(analysis, "show_isi", False)
            or getattr(analysis, "show_overlay", False)
            or getattr(analysis, "show_psth", False)
            or getattr(analysis, "show_trial_rate", False)
            or getattr(analysis, "show_raster", False)
        )
        need_overlay = bool(getattr(analysis, "show_overlay", False))
        return need_rms, need_spikes, need_overlay

    @staticmethod
    def visible_indices(recording: Any, hidden: Sequence[str] | set[str]) -> list[int]:
        """Indices des canaux non masqués (ordre d’origine)."""
        n_channels = int(getattr(recording, "n_channels", 0) or 0)
        if n_channels <= 0:
            return []
        hidden_set = set(hidden)
        names = list(getattr(recording, "channel_names", ()) or ())
        return [
            index
            for index in range(n_channels)
            if (names[index] if index < len(names) else f"CH{index}") not in hidden_set
        ]

    @staticmethod
    def page_info(
        visible: Sequence[int],
        *,
        per_page: int,
        page: int,
    ) -> tuple[list[int], int, int, int]:
        """``(indices page, page, n_pages, n_visible)``."""
        per = max(1, int(per_page))
        n_visible = len(visible)
        n_pages = max(1, (n_visible + per - 1) // per) if n_visible else 1
        page_i = max(0, min(n_pages - 1, int(page)))
        start = page_i * per
        return list(visible[start : start + per]), page_i, n_pages, n_visible

    @staticmethod
    def review_placements() -> tuple[PanelPlacement, ...]:
        """Un seul panneau : graphs Pipeline intégrés en lignes multi-canaux."""
        return (PanelPlacement("montage_continuous_raw"),)

    @staticmethod
    def kind_count(settings: Any) -> int:
        """Nombre de lignes par canal (flux + extras Pipeline)."""
        streams = ()
        resolved = getattr(settings, "resolved_continuous_streams", None)
        if callable(resolved):
            streams = resolved() or ("raw",)
        else:
            streams = ("raw",)
        analysis = getattr(settings, "analysis", None)
        extras = 0
        if analysis is not None:
            extras = sum(
                1
                for flag in (
                    getattr(analysis, "show_rms", False),
                    getattr(analysis, "show_psth", False),
                    getattr(analysis, "show_trial_rate", False),
                    getattr(analysis, "show_raster", False),
                    getattr(analysis, "show_isi", False),
                    getattr(analysis, "show_overlay", False),
                )
                if flag
            )
        return max(1, len(tuple(streams)) + extras)

    @staticmethod
    def total_height_px(n_visible: int, settings: Any) -> int:
        """Hauteur du montage (canaux×kinds × ligne + chrome panneau)."""
        rows = max(1, int(n_visible)) * MontageController.kind_count(settings)
        min_row = max(36, int(getattr(settings, "montage_row_min_height_px", 36) or 36))
        chrome = 96
        return max(220, rows * min_row + chrome)

    @staticmethod
    def per_page(settings: Any) -> int:
        return max(1, int(getattr(settings, "montage_review_channels", 10) or 10))

    def needs_means(self, settings: Any) -> bool:
        """True si la revue montage lit des moyennes / fenêtres de stim."""
        if not self._in_montage:
            return False
        return str(getattr(settings, "preview_content", None) or "continuous") in {
            "average",
            "stimulation",
        }

    def extra_flags(self, settings: Any) -> tuple[bool, bool, bool]:
        return self.analysis_extra_flags(getattr(settings, "analysis", None))
