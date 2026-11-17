"""Portée des réglages d’affichage — une source de vérité par contexte.

Trois scopes indépendants (mêmes champs d’affichage, valeurs séparées) :

- **preview** — dock Paramètres en mode aperçu canal + grille centrale
- **montage** — dock Paramètres en mode revue montage
- **zoom** — chaque fenêtre ``DetachedZoomWindow`` (LocalViewParams)

Les fenêtres zoom sont créées **uniquement** depuis l’aperçu canal. Elles
sont amorcées avec une copie du profil preview, puis restent indépendantes
(``LocalViewParams``) : modifier le zoom ne change pas l’aperçu, et
inversement.

Les **plages** (``TimeRangeBar``) n’appartiennent à aucun profil dock : elles
vivent uniquement sur ``ChannelAnalysisWindow`` et ne sont jamais fusionnées
depuis le dock (voir ``merge_central_viewer_settings``).

Le traitement F5 (``AnalysisConfig``) reste commun et hors de ces scopes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from view_config import AxisLimits, ViewerSettings

ViewScope = Literal["preview", "montage", "zoom"]


def settings_for_zoom_window(base: ViewerSettings) -> ViewerSettings:
    """Seed d’une fenêtre zoom depuis le profil **aperçu canal**.

    Hérite styles / courbes / échelles de ``base`` (typiquement
    ``params_panel.settings_for("preview")`` + inspecteur), sans échelle X
    ni plages : la fenêtre X vient du ``PanelPlacement`` custom.

    Après création, la fenêtre a son propre ``LocalViewParams`` — plus de
    lien automatique avec le dock.
    """
    return replace(
        base,
        x_limits=AxisLimits(
            enabled=False,
            minimum=float(base.x_limits.minimum),
            maximum=float(base.x_limits.maximum),
        ),
        # Les plages sont gérées par l’aperçu ; la fenêtre zoom n’en a pas.
        range_bars=(),
        active_range_index=0,
    )


def placement_zoom_window(placement: object) -> tuple[float, float] | None:
    """Bornes [t0, t1] d’un placement à zoom personnalisé, ou None."""
    t0 = getattr(placement, "zoom_t0_s", None)
    t1 = getattr(placement, "zoom_t1_s", None)
    if t0 is None or t1 is None:
        return None
    a, b = float(t0), float(t1)
    return (a, b) if a <= b else (b, a)
