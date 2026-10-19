"""Panneau graphique interactif (pyqtgraph) — zoom, pan, curseur toujours actifs.

Matplotlib reste utilisé pour l’export PDF ; l’écran passe par PlotHost.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Sequence

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.plot_host import (
    INTERACTION_HINT,
    INTERACTION_HINT_NO_ZOOM,
    PlotHost,
)
from gui.widgets.view_params import LocalViewParams
from panel_prepare import prepare_panel, restyle_panel_spec, screen_prepare_key
from panel_registry import RenderRequest, panel_info
from view_config import (
    AnalysisSettings,
    PanelPlacement,
    PanelStyle,
    TextOverrides,
    ViewerSettings,
    apply_local_display_settings,
    continuous_sync_offset_s,
    copy_global_display_style,
    resolved_scale_bar_offset_px,
    resolved_scale_bar_time_offset_px,
)


_FIT_RETRY_MAX = 12
_FIT_MATCH_PX = 2


def _scale_fit_axis_flags(host: Any) -> tuple[bool, bool]:
    """(amplitude, temps) : vrai si valeur + taille px sont demandées."""
    targets = getattr(host, "scale_fit_targets", None)
    if callable(targets):
        try:
            amp, time = targets()
            return bool(amp), bool(time)
        except Exception:
            pass
    try:
        bars = host._active_scale_bars()
    except Exception:
        bars = None
    if bars is None:
        return False, False
    amp = bool(getattr(bars, "amp_manual", False)) and bool(
        getattr(bars, "amp_length_manual", False)
    )
    time = bool(getattr(bars, "time_manual", False)) and bool(
        getattr(bars, "time_length_manual", False)
    )
    return amp, time


def _fitted_width_px(
    widget_w: int,
    viewbox_w: float,
    needed_vb: float,
    *,
    explicit_chrome: float = 0.0,
    strict: bool = True,
) -> int | None:
    """Largeur de widget pour une zone de tracé de ``needed_vb`` px.

    ``None`` tant que la géométrie n’est pas fiable : ViewBox encore plus large
    que le widget, ou axes / libellés pas encore retranchés. Appliquer quand
    même poserait une taille, puis une autre au passage suivant.
    ``strict=False`` : dernier recours, compter le chrome connu si la ViewBox
    ne le reflète toujours pas (elle ne doit toutefois pas dépasser le widget).
    """
    geometry = float(widget_w) - float(viewbox_w)
    explicit = max(0.0, float(explicit_chrome))
    if geometry < -1.0:
        return None
    if strict and explicit > geometry + 2.0:
        return None
    chrome = max(0.0, geometry, explicit)
    # Taille exacte : un plancher (200 px) élargissait la zone de tracé
    # et faussait le gain temporel.
    return max(1, int(round(chrome + float(needed_vb))))


def _measure_scale_fit_height(
    host: Any, widget_h: int
) -> tuple[str, int | None]:
    """``off`` / ``wait`` / ``ready`` pour la hauteur liée à l’amplitude."""
    amp, _time = _scale_fit_axis_flags(host)
    if not amp:
        return "off", None
    try:
        needed = host.scale_bar_needed_view_height_px()
    except Exception:
        needed = None
    if needed is None:
        return "wait", None
    total = 0.0
    for plot in getattr(host, "_plots", None) or []:
        try:
            total += float(plot.getViewBox().height())
        except Exception:
            continue
    if total > float(widget_h) + 2.0 or float(widget_h) < 4.0:
        return "wait", None
    # Zone de tracé écrasée (0 px) : le widget est tout en chrome
    # (axes, libellés). On l’agrandit quand même, sinon le gain reste faux.
    if total < 4.0:
        total = 0.0
    required = int(round(float(widget_h) - total + float(needed)))
    return "ready", max(1, required)


def _scale_fit_matches(host: Any) -> bool:
    """True si la zone de tracé a déjà le gain demandé (µV/px et s/px)."""
    amp, time = _scale_fit_axis_flags(host)
    tol = float(_FIT_MATCH_PX)
    if amp:
        try:
            needed = host.scale_bar_needed_view_height_px()
        except Exception:
            needed = None
        if needed is None:
            return False
        total = 0.0
        n = 0
        for plot in getattr(host, "_plots", None) or []:
            try:
                total += float(plot.getViewBox().height())
                n += 1
            except Exception:
                continue
        if n == 0 or abs(total - float(needed)) > tol:
            return False
    if time:
        try:
            needed_w = host.scale_bar_needed_view_width_px()
        except Exception:
            needed_w = None
        if needed_w is None:
            return False
        plots = list(getattr(host, "_plots", None) or [])
        if not plots:
            return False
        try:
            current = float(plots[-1].getViewBox().width())
        except Exception:
            return False
        if abs(current - float(needed_w)) > tol:
            return False
    return True


def _settle_scale_fit(widget: Any, *, force: bool, tries: int) -> None:
    """Relancer tant que le chrome (labels, axes) n’a pas fini de bouger.

    Un seul passage mesure le chrome *avant* le redimensionnement. Après,
    un libellé qui passe à la ligne mange la zone de tracé et le gain
    n’est plus celui demandé. On recommence jusqu’à convergence.
    """
    host = widget._plot
    if force or _scale_fit_matches(host):
        widget._scale_fit_pending = False
        widget._scale_fit_tries = 0
    else:
        widget._scale_fit_tries = int(tries) + 1
        widget._scale_fit_height_sample = None
        widget._scale_fit_width_sample = None
        widget._scale_fit_retry.start()
    try:
        host._refresh_scale_bars()
    except Exception:
        pass


def _measure_scale_fit_width(
    host: Any, widget_w: int, *, strict: bool
) -> tuple[str, int | None]:
    """``off`` / ``wait`` / ``ready`` pour la largeur liée au temps."""
    _amp, time = _scale_fit_axis_flags(host)
    if not time:
        return "off", None
    try:
        needed = host.scale_bar_needed_view_width_px()
    except Exception:
        needed = None
    if needed is None:
        return "wait", None
    plots = list(getattr(host, "_plots", None) or [])
    if not plots:
        return "wait", None
    try:
        current = float(plots[-1].getViewBox().width())
    except Exception:
        return "wait", None
    if current < 8.0:
        return "wait", None
    required = _fitted_width_px(
        int(widget_w),
        current,
        float(needed),
        explicit_chrome=_explicit_graph_chrome_px(host, float(widget_w)),
        strict=strict,
    )
    if required is None:
        return "wait", None
    return "ready", int(required)


def _accept_fit_sample(
    previous: int | None,
    state: str,
    value: int | None,
    *,
    force: bool,
) -> tuple[str, int | None]:
    """N’accepter une taille qu’après deux mesures voisines identiques."""
    if state != "ready" or value is None:
        return state, previous
    px = int(value)
    if force or (previous is not None and abs(int(previous) - px) <= _FIT_MATCH_PX):
        return "ready", px
    return "wait", px


def _explicit_graph_chrome_px(host: Any, panel_w: float) -> float:
    """Pixels hors zone de tracé : cadre, libellés de ligne, axes gauche et droit."""
    plots = list(getattr(host, "_plots", None) or [])
    if not plots:
        return 0.0
    plot = plots[-1]
    left = right = 0.0
    try:
        left = max(0.0, float(plot.getAxis("left").width()))
    except Exception:
        left = 0.0
    try:
        right = max(0.0, float(plot.getAxis("right").width()))
    except Exception:
        right = 0.0
    glw = getattr(host, "_glw", None)
    try:
        glw_w = float(glw.width()) if glw is not None else float(panel_w)
    except Exception:
        glw_w = float(panel_w)
    frame = max(0.0, float(panel_w) - glw_w)
    label = 0.0
    plot_col = int(getattr(host, "_plot_col", 0) or 0)
    if plot_col > 0 and glw is not None:
        layout = getattr(getattr(glw, "ci", None), "layout", None)
        if layout is not None:
            try:
                label = max(0.0, float(layout.columnMinimumWidth(0)))
            except Exception:
                label = 0.0
            try:
                label += max(0.0, float(layout.columnSpacing(0)))
            except Exception:
                pass
    return frame + label + left + right


def _font_face_key(face: Any) -> tuple[bool, bool, bool]:
    """Gras / italique / souligné : ils changent la largeur et la hauteur du texte."""
    return (
        bool(getattr(face, "bold", False)),
        bool(getattr(face, "italic", False)),
        bool(getattr(face, "underline", False)),
    )


def _font_pt(value: Any, default: float) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return round(float(default), 2)


def _scale_fit_signature(settings: ViewerSettings) -> tuple[Any, ...]:
    """Empreinte échelle (valeur + px), placement et polices.

    Taille, gras, italique, souligné et le placement des barres d’échelle
    élargissent le chrome (gouttières, titre, axes, ticks, légende, libellés
    stim / zone). Avec une échelle fixe (valeur + taille px), c’est le widget
    qui absorbe ce chrome : la zone de tracé reste en pixels.
    """
    style = settings.style
    legend = settings.legend
    amp_dx, amp_dy = resolved_scale_bar_offset_px(style)
    time_dx, time_dy = resolved_scale_bar_time_offset_px(style)
    return (
        bool(style.scale_bar_amplitude_manual),
        round(float(style.scale_bar_amplitude or 0.0), 12),
        bool(style.scale_bar_amplitude_length_manual),
        int(round(float(style.scale_bar_amplitude_length_px or 0.0))),
        bool(style.scale_bar_time_manual),
        round(float(style.scale_bar_time_s or 0.0), 12),
        bool(style.scale_bar_time_length_manual),
        int(round(float(style.scale_bar_time_length_px or 0.0))),
        int(round(amp_dx)),
        int(round(amp_dy)),
        int(round(time_dx)),
        int(round(time_dy)),
        bool(getattr(style, "spike_scale_amplitude_manual", False)),
        round(float(getattr(style, "spike_scale_amplitude", 0.0) or 0.0), 12),
        int(round(float(getattr(style, "spike_scale_amplitude_length_px", 0.0) or 0.0))),
        bool(getattr(style, "spike_scale_time_manual", False)),
        round(float(getattr(style, "spike_scale_time_ms", 0.0) or 0.0), 12),
        int(round(float(getattr(style, "spike_scale_time_length_px", 0.0) or 0.0))),
        _font_pt(style.title_font_size, 10.0),
        _font_face_key(style.title_face),
        _font_pt(style.label_font_size, 9.0),
        _font_face_key(style.label_face),
        _font_pt(style.tick_font_size, 8.0),
        _font_face_key(style.tick_face),
        _font_pt(getattr(style, "stim_on_font_size", 8.0), 8.0),
        _font_face_key(getattr(style, "stim_on_face", None)),
        _font_pt(getattr(style, "stim_off_font_size", 8.0), 8.0),
        _font_face_key(getattr(style, "stim_off_face", None)),
        _font_pt(getattr(style, "zone_font_size", 8.0), 8.0),
        _font_face_key(getattr(style, "zone_face", None)),
        _font_pt(getattr(legend, "font_size", 9.0), 9.0),
        _font_face_key(getattr(legend, "face", None)),
    )


def x_scale_family(panel: str) -> str:
    """Famille d’échelle X partageable au double-clic ('' = ne pas synchroniser)."""
    key = str(panel or "")
    if not key or key.startswith("summary_") or key in {"mea_layout", "impedance"}:
        return ""
    # Unités différentes du temps d’essai / fichier.
    if any(token in key for token in ("overlay", "isi", "trial_rate", "waveform")):
        return ""
    try:
        from panel_registry import _panel_expects_relative_x

        if _panel_expects_relative_x(key):
            return "relative"
    except Exception:
        pass
    return "continuous"


def y_scale_family(panel: str, *, placement_stream: str = "") -> str:
    """Famille d’échelle Y partageable entre fenêtres zoom ('' = ne pas synchroniser)."""
    key = str(panel or "")
    if not key or key.startswith("summary_") or key in {"mea_layout", "impedance", "raster"}:
        return ""
    # Spike scope : chaque fenêtre garde l’échelle de ses formes d’onde.
    if "overlay" in key:
        return ""
    stream = _panel_stream_key(key, placement_stream=placement_stream)
    if stream is not None:
        return f"stream:{stream}"
    return f"panel:{key}"


_Y_SCALE_NONNEG_FAMILIES = frozenset(
    {"stream:rms", "panel:analysis_psth", "panel:analysis_trial_rate", "panel:analysis_isi"}
)


def _patch_text_overrides(texts: TextOverrides, result: dict[str, Any] | Any) -> TextOverrides:
    """Appliquer une édition in-place aux overrides de textes persistés."""
    if isinstance(result, dict):
        role = str(result.get("role", "") or "")
        value = str(result.get("text", "") or "")
        item_index = int(result.get("item_index", 0) or 0)
    else:
        role = str(getattr(result, "role", "") or "")
        value = str(getattr(result, "text", "") or "")
        item_index = int(getattr(result, "item_index", 0) or 0)
    if role == "title":
        return replace(texts, title=value.strip())
    if role == "xlabel":
        return replace(texts, xlabel=value.strip())
    if role == "ylabel":
        return replace(texts, ylabel=value.strip())
    if role == "stim_on":
        return replace(texts, stim_on=value)
    if role == "stim_off":
        return replace(texts, stim_off=value)
    if role == "legend":
        labels = list(texts.legend_labels)
        while len(labels) <= item_index:
            labels.append("")
        labels[item_index] = value
        return replace(texts, legend_labels=tuple(labels))
    return texts

_STATUS_COLORS = {
    "ok": "#16a34a",
    "empty": "#ca8a04",
    "unavailable": "#dc2626",
    "pending": "#64748b",
}

_TITLE_INTERACT_HINT = (
    "Clic sur un texte = éditer · Ctrl+molette = zoom · Clic milieu/droit = pan · "
    "Double-clic = reset · Boutons Home / Pan / Zoom dans la barre"
)
_TITLE_INTERACT_HINT_NO_ZOOM = (
    "Clic sur un texte = éditer · Clic milieu/droit = pan · Double-clic = reset · "
    "Zoom désactivé sur cet aperçu"
)


def _request_sync_offset(request: RenderRequest) -> float:
    """Offset d’origine du temps (s) pour le 1er enregistrement de la requête."""
    if not request.recordings:
        return 0.0
    try:
        return float(
            continuous_sync_offset_s(
                request.recordings[0].stimulation_times_s(),
                request.settings.time_sync,
            )
        )
    except Exception:
        return 0.0


def _shift_limits_for_sync_delta(
    limits: list[tuple[tuple[float, float], tuple[float, float]]]
    | dict[Any, tuple[tuple[float, float], tuple[float, float]]],
    delta_s: float,
) -> (
    list[tuple[tuple[float, float], tuple[float, float]]]
    | dict[Any, tuple[tuple[float, float], tuple[float, float]]]
):
    """Décale les xlim d’affichage quand l’origine du temps change (même fenêtre physique)."""
    if not delta_s:
        return limits
    if isinstance(limits, dict):
        return {
            key: ((xlim[0] - delta_s, xlim[1] - delta_s), ylim)
            for key, (xlim, ylim) in limits.items()
        }
    return [((x0 - delta_s, x1 - delta_s), ylim) for (x0, x1), ylim in limits]


def _panel_stream_key(panel: str, *, placement_stream: str = "") -> str | None:
    """Flux WIDE/HIGH/LOW d’un panneau, ou ``None`` si non applicable."""
    key = str(panel)
    if "rms" in key:
        return "rms"
    if key == "full_recording":
        stream = str(placement_stream or "raw").strip().lower()
        return stream if stream in {"raw", "hp", "lp"} else "raw"
    if "_hp" in key or key.endswith("_hp"):
        return "hp"
    if "_lp" in key or key.endswith("_lp"):
        return "lp"
    if "_raw" in key or key.endswith("_raw"):
        return "raw"
    return None


def _axis_ylim_setting(
    figure: Any, request: RenderRequest, axis_index: int
) -> tuple[float, float] | None:
    """Limites Y manuelles configurées pour un axe, ou ``None`` (auto)."""
    settings = request.settings
    row_kinds = getattr(figure, "_erg_montage_row_kinds", None)
    if isinstance(row_kinds, (list, tuple)) and axis_index < len(row_kinds):
        kind = str(row_kinds[axis_index]).strip().lower()
        if kind == "rms":
            return settings.rms_ylim.as_tuple()
        if kind in {"raw", "hp", "lp"}:
            return settings.ylim_for_stream(kind).as_tuple()
        return None
    row_streams = getattr(figure, "_erg_montage_row_streams", None)
    if isinstance(row_streams, (list, tuple)) and axis_index < len(row_streams):
        stream = str(row_streams[axis_index]).strip().lower()
    else:
        stream = _panel_stream_key(
            str(request.panel),
            placement_stream=str(getattr(request.placement, "stream", "") or ""),
        )
    if stream is None:
        return None
    if stream == "rms":
        return settings.rms_ylim.as_tuple()
    return settings.ylim_for_stream(stream).as_tuple()


_MONTAGE_AUTO_Y_KINDS = frozenset(
    {"psth", "trial_rate", "isi", "overlay", "raster"}
)


def _axis_keeps_rendered_ylim(
    figure: Any,
    request: RenderRequest,
    axis_index: int,
    *,
    previous: RenderRequest | None = None,
) -> bool:
    """True → garder le ylim du rendu (pas le zoom interactif sauvegardé)."""
    if (
        str(request.panel) == "montage_continuous_raw"
        and previous is not None
    ):
        prev_shared = bool(getattr(previous.settings, "montage_shared_ylim", False))
        cur_shared = bool(getattr(request.settings, "montage_shared_ylim", False))
        if prev_shared != cur_shared:
            return True
    row_kinds = getattr(figure, "_erg_montage_row_kinds", None)
    if (
        str(request.panel) == "montage_continuous_raw"
        and isinstance(row_kinds, (list, tuple))
        and axis_index < len(row_kinds)
        and str(row_kinds[axis_index]).strip().lower() in _MONTAGE_AUTO_Y_KINDS
    ):
        return True
    current = _axis_ylim_setting(figure, request, axis_index)
    if current is not None:
        return True
    if previous is not None:
        previous_ylim = _axis_ylim_setting(figure, previous, axis_index)
        if previous_ylim != current:
            return True
    return False


def _keeps_rendered_xlim(
    request: RenderRequest,
    *,
    previous: RenderRequest | None = None,
) -> bool:
    """True → garder le xlim du rendu (manuel, ou manuel→auto)."""
    # Spike scope : l’abscisse est la fenêtre avant/après le spike, pas un zoom.
    if str(request.panel) in {"analysis_overlay", "spike_overlay"}:
        return True
    current = request.settings.x_limits.as_tuple()
    if current is not None:
        return True
    if previous is not None:
        previous_x = previous.settings.x_limits.as_tuple()
        if previous_x != current:
            return True
    return False


def _finish_panel_render(
    plot: PlotHost,
    *,
    figure: Any,
    request: RenderRequest,
    panel: str,
    preserve_limits: list | dict | None,
    previous_request: RenderRequest | None = None,
) -> str:
    """Prepare PanelSpec → PlotHost.render; restore view; coalesce draw."""
    del panel  # status comes from spec; panel kept for call-site compatibility
    try:
        key = screen_prepare_key(request)
        cached_key = getattr(plot, "_structure_key", None)
        cached_spec = getattr(plot, "_cached_spec", None)
        # Même géométrie données → restyle (pas de relecture memmap / filtres).
        if (
            cached_key == key
            and cached_spec is not None
            and cached_spec.layout_kind != "placeholder"
            and cached_spec.series
            and plot.plot_items()
        ):
            spec = restyle_panel_spec(cached_spec, request)
        else:
            spec = prepare_panel(request)
        status = plot.render_spec(spec)
        plot._structure_key = key
        plot._cached_spec = spec
    except Exception as exc:
        import logging

        from panel_prepare import PanelSpec

        logging.getLogger("plot_erg.render").exception(
            "Screen panel render failed: %s", exc
        )
        status = plot.render_spec(
            PanelSpec(
                title="Erreur de rendu",
                status="error",
                status_message=f"Erreur de rendu :\n{exc}",
                layout_kind="placeholder",
            )
        )
        plot._structure_key = None
        plot._cached_spec = None
        status = "error"
    if preserve_limits is not None:
        restore_x = not _keeps_rendered_xlim(request, previous=previous_request)
        n_axes = len(list(getattr(figure, "axes", []) or []))
        restore_y = [
            not _axis_keeps_rendered_ylim(
                figure, request, index, previous=previous_request
            )
            for index in range(n_axes)
        ]
        if isinstance(preserve_limits, dict):
            plot.restore_montage_view_limits(
                preserve_limits,
                restore_x=restore_x,
                restore_y=restore_y,
            )
        else:
            plot.restore_view_limits(
                preserve_limits,
                restore_x=restore_x,
                restore_y=restore_y,
            )
    plot.draw_idle()
    return status


class PanelStyleDialog(QDialog):
    """Dialogue non modal : affichage / style propres à un seul graphique."""

    resetRequested = Signal()

    def __init__(
        self,
        placement: PanelPlacement,
        settings: ViewerSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(f"Style — {placement.title()}")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.resize(420, 640)
        self._placement = placement
        # Montage : hauteur pilotée par « hauteur de ligne » (dock Paramètres).
        show_graph_height = not str(placement.panel).startswith("montage_")

        self._tabs = QTabWidget(self)
        self.params = LocalViewParams(
            settings,
            show_sync=True,
            show_display_extras=True,
            show_graph_height=show_graph_height,
            host_tabs=self._tabs,
            parent=self,
        )

        reset_btn = QPushButton("Réinitialiser", self)
        reset_btn.setToolTip(
            "Supprimer les réglages spécifiques à ce graphique et "
            "reprendre ceux de la fenêtre."
        )
        reset_btn.clicked.connect(self.resetRequested.emit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.close)
        buttons.addButton(reset_btn, QDialogButtonBox.ButtonRole.ResetRole)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        hint = QLabel(
            "Ces réglages s’appliquent uniquement à ce graphique. "
            "Les autres panneaux de la fenêtre restent inchangés.",
            self,
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self._tabs, 1)
        layout.addWidget(buttons)

    def load_settings(self, settings: ViewerSettings) -> None:
        """Recharger le formulaire (ex. après réinitialisation)."""
        old = self.params
        try:
            old.changed.disconnect()
        except (TypeError, RuntimeError):
            pass
        # Retirer les pages sans les détruire tout de suite : les widgets
        # (cases légende, etc.) vivent dans les tabs, pas sous LocalViewParams.
        while self._tabs.count():
            page = self._tabs.widget(0)
            self._tabs.removeTab(0)
            if page is not None:
                page.setParent(old)
        show_graph_height = not str(self._placement.panel).startswith("montage_")
        self.params = LocalViewParams(
            settings,
            show_sync=True,
            show_display_extras=True,
            show_graph_height=show_graph_height,
            host_tabs=self._tabs,
            parent=self,
        )
        old.deleteLater()


class PanelCanvas(QFrame):
    """Un panneau : en-tête + PlotHost pyqtgraph (toolbar toujours visible)."""

    removeRequested = Signal(object)  # PanelPlacement
    detachRequested = Signal(object)  # PanelPlacement
    styleChanged = Signal(object)  # PanelPlacement — overrides d’affichage modifiés
    # Famille d’échelle X + (x0, x1) après double-clic / ⌂ — sync grille.
    xScaleSyncRequested = Signal(str, object)

    def __init__(
        self,
        placement: PanelPlacement,
        parent: QWidget | None = None,
        *,
        allow_zoom: bool = True,
    ) -> None:
        super().__init__(parent)
        self.placement = placement
        self._allow_zoom = bool(allow_zoom)
        self.setObjectName("panelCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # Largeur libre tant que l’échelle temps ne verrouille pas le panneau.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self._dirty = True
        self._last_render_s = 0.0
        self._status = "pending"
        self._style_overrides: ViewerSettings | None = None
        self._last_base_settings: ViewerSettings | None = None
        self._last_request: RenderRequest | None = None
        self._style_dialog: PanelStyleDialog | None = None
        self._x_sync_guard = False

        self._plot = PlotHost(
            parent=self,
            show_toolbar=True,
            show_cursor=True,
            min_height=80,
            allow_zoom=self._allow_zoom,
        )
        self.figure = self._plot.figure
        self.canvas = self._plot  # QWidget for grab() / range-bar attach
        self.toolbar = self._plot.toolbar
        self.plot_host = self._plot
        self._plot.textEdited.connect(self._on_canvas_text_edited)
        self._plot.xAutoscaleRequested.connect(self._on_host_x_autoscale)
        self._plot.scaleBarHeightFitRequested.connect(self._schedule_scale_bar_widget_fit)
        self._plot.scaleBarWidthFitRequested.connect(self._schedule_scale_bar_widget_fit)
        self.fitted_height_px: int | None = None
        self.fitted_width_px: int | None = None
        # Largeur stable (premier affichage ou échelle temps) : ne suit pas la fenêtre.
        self.frozen_width_px: int | None = None
        self._last_scale_fit_signature: tuple[Any, ...] | None = None
        self._scale_fit_pending = False
        self._scale_fit_tries = 0
        self._scale_fit_width_sample: int | None = None
        self._scale_fit_height_sample: int | None = None
        self._scale_fit_coalesce = QTimer(self)
        self._scale_fit_coalesce.setSingleShot(True)
        self._scale_fit_coalesce.setInterval(150)
        self._scale_fit_coalesce.timeout.connect(self._run_scale_bar_widget_fit)
        self._scale_fit_retry = QTimer(self)
        self._scale_fit_retry.setSingleShot(True)
        self._scale_fit_retry.setInterval(40)
        self._scale_fit_retry.timeout.connect(self._run_scale_bar_widget_fit)

        interact_hint = (
            _TITLE_INTERACT_HINT if self._allow_zoom else _TITLE_INTERACT_HINT_NO_ZOOM
        )
        self._title = QLabel(placement.title())
        self._title.setObjectName("panelTitle")
        self._title.setToolTip(f"{placement.title()}\n{interact_hint}")
        self._title.setWordWrap(False)
        self._title.setMinimumWidth(0)
        self._title.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._status_label = QLabel("—")
        self._status_label.setObjectName("panelStatus")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._style_button = self._make_tool_button(
            "⚙",
            "Affichage / style de ce graphique uniquement",
        )
        self._style_button.setCheckable(True)
        self._style_button.clicked.connect(self._open_style_dialog)
        self._detach_button = self._make_tool_button("⤢", "Ouvrir ce panneau dans sa propre fenêtre")
        self._detach_button.clicked.connect(lambda: self.detachRequested.emit(self.placement))
        self._close_button = self._make_tool_button("✕", "Retirer ce panneau de la vue")
        self._close_button.clicked.connect(lambda: self.removeRequested.emit(self.placement))

        header = QHBoxLayout()
        header.setContentsMargins(8, 4, 4, 0)
        header.setSpacing(4)
        header.addWidget(self._title, 1)
        header.addWidget(self._status_label, 0)
        header.addWidget(self._style_button, 0)
        header.addWidget(self._detach_button, 0)
        header.addWidget(self._close_button, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 4)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self._plot, 1)
        self._refresh_style_button()

    def _make_tool_button(self, text: str, tooltip: str) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName("panelToolButton")
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def update_placement(self, placement: PanelPlacement) -> None:
        """Mettre à jour le placement (ex. bornes de plage) sans recréer le widget."""
        self.placement = placement
        title = placement.title()
        self._title.setText(title)
        hint = (
            _TITLE_INTERACT_HINT if self._allow_zoom else _TITLE_INTERACT_HINT_NO_ZOOM
        )
        self._title.setToolTip(f"{title}\n{hint}")
        self._dirty = True
        if self._style_dialog is not None:
            self._style_dialog.setWindowTitle(f"Style — {title}")

    # ----------------------------------------------------------------- style


    @property
    def style_height_px(self) -> int | None:
        """Hauteur dédiée si ce panneau a un override d’affichage.

        Les montages ignorent ``graph_height_px`` : leur hauteur vient de
        ``montage_row_min_height_px`` (via la grille / ``_adapt_montage_height``).
        """
        if self._style_overrides is None:
            return None
        if str(self.placement.panel).startswith("montage_"):
            return None
        return max(140, int(self._style_overrides.graph_height_px))

    def _refresh_style_button(self) -> None:
        active = self._style_overrides is not None
        self._style_button.blockSignals(True)
        self._style_button.setChecked(active)
        self._style_button.blockSignals(False)
        tip = "Affichage / style de ce graphique uniquement"
        if active:
            tip += "\n(réglages spécifiques actifs — Réinitialiser dans le dialogue)"
        self._style_button.setToolTip(tip)

    def _seed_settings_for_dialog(self) -> ViewerSettings:
        if self._style_overrides is not None:
            return self._style_overrides
        if self._last_base_settings is not None:
            return self._last_base_settings
        return ViewerSettings()

    def _open_style_dialog(self) -> None:
        # setCheckable : le clic bascule l’état — on rétablit l’indicateur d’override.
        self._refresh_style_button()
        if self._style_dialog is not None:
            self._style_dialog.raise_()
            self._style_dialog.activateWindow()
            return
        dialog = PanelStyleDialog(
            self.placement, self._seed_settings_for_dialog(), parent=self
        )
        dialog.params.changed.connect(self._on_style_params_changed)
        dialog.resetRequested.connect(self._reset_style_overrides)
        dialog.destroyed.connect(self._on_style_dialog_destroyed)
        self._style_dialog = dialog
        dialog.show()

    def _on_style_dialog_destroyed(self, *_args: Any) -> None:
        self._style_dialog = None

    def _on_style_params_changed(self) -> None:
        if self._style_dialog is None:
            return
        self._style_overrides = self._style_dialog.params.settings()
        # Propager légende / hauteur / échelle locales dans la requête de base,
        # sinon ``_with_style_overrides`` (inherit) les réécraserait avec le dock.
        if self._last_request is not None and self._style_overrides is not None:
            merged = copy_global_display_style(
                self._last_request.settings.style,
                self._style_overrides.style,
            )
            self._last_request = replace(
                self._last_request,
                settings=replace(
                    self._last_request.settings,
                    style=merged,
                    legend=self._style_overrides.legend,
                    graph_height_px=int(self._style_overrides.graph_height_px),
                ),
            )
        self._refresh_style_button()
        self._rerender_with_overrides()
        self.styleChanged.emit(self.placement)

    def sync_scale_bars_from(self, style_or_settings: PanelStyle | Any) -> None:
        """Aligner un override local sur les réglages globaux du dock Affichage.

        Accepte un ``PanelStyle`` (échelle / stim) ou un ``ViewerSettings``
        (échelle / stim + légende + hauteur de graph).
        """
        if self._style_overrides is None:
            return
        legend = getattr(style_or_settings, "legend", None)
        graph_height = getattr(style_or_settings, "graph_height_px", None)
        style = getattr(style_or_settings, "style", style_or_settings)
        updates: dict[str, Any] = {
            "style": copy_global_display_style(self._style_overrides.style, style),
        }
        if legend is not None:
            updates["legend"] = legend
        if graph_height is not None:
            updates["graph_height_px"] = int(graph_height)
        self._style_overrides = replace(self._style_overrides, **updates)

    def _on_host_x_autoscale(self, xlim: object) -> None:
        """Relayer l’échelle X du double-clic vers la grille (même famille)."""
        if self._x_sync_guard:
            return
        family = x_scale_family(str(self.placement.panel))
        if not family or xlim is None:
            return
        try:
            x0, x1 = float(xlim[0]), float(xlim[1])  # type: ignore[index]
        except Exception:
            return
        if not (x1 > x0):
            return
        self.xScaleSyncRequested.emit(family, (x0, x1))

    def apply_shared_x_scale(self, xlim: tuple[float, float]) -> None:
        """Recevoir une échelle X synchronisée depuis un autre panneau."""
        if self._x_sync_guard:
            return
        self._x_sync_guard = True
        try:
            self._plot.apply_shared_xlim(xlim)
        finally:
            self._x_sync_guard = False

    def apply_shared_y_scale(
        self,
        ylim: tuple[float, float],
        *,
        non_negative: bool = False,
    ) -> None:
        """Recevoir une échelle Y synchronisée (zooms) sans retoucher X."""
        self._plot.apply_shared_ylim(ylim, non_negative=non_negative)

    def _on_canvas_text_edited(self, result: object) -> None:
        """Persister une édition de texte faite directement sur le graphique."""
        role = (
            str(result.get("role", "") or "")
            if isinstance(result, dict)
            else str(getattr(result, "role", "") or "")
        )
        if role in {"", "annotation"}:
            return
        seed = self._style_overrides or self._last_base_settings or ViewerSettings()
        texts = _patch_text_overrides(seed.texts, result)
        self._style_overrides = replace(seed, texts=texts)
        self._refresh_style_button()
        if self._style_dialog is not None:
            self._style_dialog.params.apply_texts(texts)
        # Pins locaux déjà appliqués ; pas de re-render (évite flash + perte focus).
        self.styleChanged.emit(self.placement)

    def _reset_style_overrides(self) -> None:
        self._style_overrides = None
        self._plot.clear_text_pins()
        self._refresh_style_button()
        base = self._last_base_settings
        if base is not None and self._style_dialog is not None:
            # Reconnecter après rebuild du formulaire.
            self._style_dialog.load_settings(base)
            self._style_dialog.params.changed.connect(self._on_style_params_changed)
        self._rerender_with_overrides()
        self.styleChanged.emit(self.placement)

    def _rerender_with_overrides(self) -> None:
        if self._last_request is None:
            self._dirty = True
            return
        # Conserver la vue (zoom/pan) pendant l’ajustement de style.
        request = replace(self._last_request, preserve_view=True)
        self.render(request)

    def _with_style_overrides(self, request: RenderRequest) -> RenderRequest:
        self._last_base_settings = request.settings
        self._last_request = request
        if self._style_overrides is None:
            return request
        # Overrides = affichage de *ce* graphique (dialogue Style), pas le dock parent.
        settings = apply_local_display_settings(
            request.settings,
            self._style_overrides,
            inherit_scale_bars=False,
        )
        return replace(request, settings=settings)

    # ----------------------------------------------------------------- drawing

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def invalidate(self) -> None:
        self._dirty = True

    def mark_pending(self) -> None:
        self._set_status("pending", "redraw pending")

    def mark_offscreen(self) -> None:
        self._set_status("pending", "scroll to draw")

    def render(self, request: RenderRequest) -> str:
        """Dessiner le panneau (pan = clic milieu/droit ; clic gauche libre pour plages)."""
        prev_request = self._last_request
        old_offset = _request_sync_offset(prev_request) if prev_request else 0.0
        request = self._with_style_overrides(request)
        is_montage = str(self.placement.panel) == "montage_continuous_raw"
        saved_limits: list | dict | None = None
        if request.preserve_view:
            if is_montage:
                saved_limits = self._plot.capture_montage_view_limits()
            else:
                saved_limits = self._plot.capture_view_limits()
        if (
            isinstance(saved_limits, list)
            and saved_limits is not None
            and prev_request is not None
        ):
            delta = _request_sync_offset(request) - old_offset
            if delta:
                saved_limits = _shift_limits_for_sync_delta(saved_limits, delta)
        elif isinstance(saved_limits, dict) and prev_request is not None:
            delta = _request_sync_offset(request) - old_offset
            if delta:
                saved_limits = _shift_limits_for_sync_delta(saved_limits, delta)
        sig = _scale_fit_signature(request.settings)
        if sig != self._last_scale_fit_signature:
            self._last_scale_fit_signature = sig
            self._arm_scale_fit()
        started = time.perf_counter()
        status = _finish_panel_render(
            self._plot,
            figure=self.figure,
            request=request,
            panel=self.placement.panel,
            preserve_limits=saved_limits,
            previous_request=prev_request,
        )
        self._last_render_s = time.perf_counter() - started
        self._dirty = False
        self._set_status(status, f"{self._last_render_s * 1000:.0f} ms")
        if str(self.placement.panel).startswith("montage_"):
            # Le layout pyqtgraph n’a sa géométrie finale qu’au tick suivant.
            QTimer.singleShot(0, self._fit_montage_height)
        # Échelle valeur/px ou police : le widget absorbe le chrome.
        # Un rendu ordinaire ne redimensionne pas (évite les sauts à répétition).
        if self._scale_fit_pending:
            QTimer.singleShot(0, self._schedule_scale_bar_widget_fit)
        return status

    def _arm_scale_fit(self) -> None:
        """Autoriser un seul recalage, après stabilisation de la géométrie."""
        self._scale_fit_pending = True
        self._scale_fit_tries = 0
        self._scale_fit_width_sample = None
        self._scale_fit_height_sample = None
        self._scale_fit_retry.stop()

    def _schedule_scale_bar_widget_fit(self) -> None:
        """Un seul passage H/W après stabilisation layout (évite sauts continus)."""
        if not self._scale_fit_pending:
            return
        self._scale_fit_coalesce.start()

    def _run_scale_bar_widget_fit(self) -> None:
        if not self._scale_fit_pending:
            return
        tries = int(self._scale_fit_tries)
        force = tries >= _FIT_RETRY_MAX
        h_state, h_px = _measure_scale_fit_height(
            self._plot, max(1, int(self.height()))
        )
        w_state, w_px = _measure_scale_fit_width(
            self._plot, max(1, int(self.width())), strict=not force
        )
        h_state, self._scale_fit_height_sample = _accept_fit_sample(
            self._scale_fit_height_sample, h_state, h_px, force=force
        )
        w_state, self._scale_fit_width_sample = _accept_fit_sample(
            self._scale_fit_width_sample, w_state, w_px, force=force
        )
        if not force and (h_state == "wait" or w_state == "wait"):
            self._scale_fit_tries = tries + 1
            self._scale_fit_retry.start()
            return
        self._apply_scale_fit_height(h_state, h_px if h_state == "ready" else None)
        self._apply_scale_fit_width(w_state, w_px if w_state == "ready" else None)
        _settle_scale_fit(self, force=force, tries=tries)

    def _apply_scale_fit_height(self, state: str, required: int | None) -> None:
        if state == "off":
            if self.fitted_height_px is None:
                return
            self.fitted_height_px = None
            grid = self._grid_host()
            if grid is not None:
                grid._apply_panel_height(self, self.placement)
            return
        if state != "ready" or required is None:
            return
        grid = self._grid_host()
        if grid is not None:
            grid.apply_fitted_height(self.placement, int(required), scale_fit=True)
        else:
            self.fitted_height_px = int(required)
            self.setMinimumHeight(int(required))
            self.setMaximumHeight(int(required))

    def _apply_scale_fit_width(self, state: str, required: int | None) -> None:
        if state == "off":
            self._clear_fitted_width()
            return
        if state != "ready" or required is None:
            return
        grid = self._grid_host()
        if grid is not None:
            grid.apply_fitted_width(self.placement, int(required), scale_fit=True)
        else:
            self.fitted_width_px = int(required)
            self.setFixedWidth(int(required))
            self.setSizePolicy(
                QSizePolicy.Policy.Fixed, self.sizePolicy().verticalPolicy()
            )

    def _grid_host(self) -> Any | None:
        parent = self.parentWidget()
        while parent is not None:
            if callable(getattr(parent, "apply_fitted_height", None)):
                return parent
            parent = parent.parentWidget()
        return None

    def _fit_montage_height(self) -> None:
        """Agrandir le panneau si les canaux demandés ne tiennent pas."""
        if not str(self.placement.panel).startswith("montage_"):
            return
        needed_glw = self._plot.montage_needed_glw_height_px()
        if needed_glw is None:
            return
        glw = getattr(self._plot, "_glw", None)
        glw_h = int(glw.height()) if glw is not None else 0
        if glw_h <= 0:
            return
        required = max(140, int(needed_glw) + int(self.height()) - glw_h)
        grid = self._grid_host()
        if grid is None:
            return
        grid.apply_fitted_height(self.placement, required)

    def _clear_fitted_width(self) -> None:
        """Retirer le verrouillage de largeur (le panneau peut à nouveau suivre la place libre)."""
        if self.fitted_width_px is None:
            return
        self.fitted_width_px = None
        grid = self._grid_host()
        if grid is not None:
            grid._apply_panel_width(self, self.placement)
        else:
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding, self.sizePolicy().verticalPolicy()
            )

    def _set_status(self, status: str, text: str) -> None:
        self._status = status
        color = _STATUS_COLORS.get(status, "#64748b")
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color};")
        hint = INTERACTION_HINT if self._allow_zoom else INTERACTION_HINT_NO_ZOOM
        tooltip = {
            "ok": f"Interactif — {hint}",
            "empty": "Rien à dessiner avec les réglages actuels",
            "unavailable": "Données indisponibles pour ce panneau",
            "pending": "Dessiné à l’entrée dans la vue",
        }.get(status, status)
        self._status_label.setToolTip(
            f"{tooltip} — dernier rendu {self._last_render_s * 1000:.0f} ms"
        )

    @property
    def last_render_s(self) -> float:
        return self._last_render_s


class DetachedPanelWindow(QDialog):
    """Panneau détaché : canvas interactif + légende/style locaux."""

    closed = Signal(object)  # PanelPlacement
    refreshRequested = Signal(object)  # DetachedPanelWindow

    def __init__(
        self,
        placement: PanelPlacement,
        settings: ViewerSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.placement = placement
        self.settings = settings
        info = panel_info(placement.panel)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(placement.title())
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(1100, max(480, info.preferred_height_px + 160))

        self._plot = PlotHost(
            parent=self,
            show_toolbar=True,
            show_cursor=True,
            min_height=280,
            allow_zoom=True,
        )
        self.figure = self._plot.figure
        self.canvas = self._plot
        self.plot_host = self._plot
        self._plot.textEdited.connect(self._on_detached_text_edited)
        self._scale_fit_pending = False
        self._scale_fit_tries = 0
        self._scale_fit_width_sample: int | None = None
        self._scale_fit_height_sample: int | None = None
        self._scale_fit_coalesce = QTimer(self)
        self._scale_fit_coalesce.setSingleShot(True)
        self._scale_fit_coalesce.setInterval(150)
        self._scale_fit_coalesce.timeout.connect(self._run_detached_scale_fit)
        self._scale_fit_retry = QTimer(self)
        self._scale_fit_retry.setSingleShot(True)
        self._scale_fit_retry.setInterval(40)
        self._scale_fit_retry.timeout.connect(self._run_detached_scale_fit)
        self._plot.scaleBarHeightFitRequested.connect(self._schedule_detached_scale_fit)
        self._plot.scaleBarWidthFitRequested.connect(self._schedule_detached_scale_fit)
        self._plot_locked_w: int | None = None
        self._plot_locked_h: int | None = None
        self._plot_lock_timer = QTimer(self)
        self._plot_lock_timer.setSingleShot(True)
        self._plot_lock_timer.setInterval(50)
        self._plot_lock_timer.timeout.connect(self._freeze_plot_chrome_once)

        self._status = QLabel("—")
        self._status.setObjectName("panelStatus")
        self._last_render_s = 0.0
        self._last_request: RenderRequest | None = None
        self._last_scale_fit_signature: tuple[Any, ...] | None = None

        self.params = LocalViewParams(
            settings,
            show_sync=True,
            show_display_extras=True,
            parent=self,
        )
        self.params.changed.connect(self._on_local_params_changed)

        plot_side = QWidget(self)
        plot_layout = QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.setSpacing(4)
        header = QHBoxLayout()
        self._title_label = QLabel(placement.title())
        header.addWidget(self._title_label, 1)
        header.addWidget(self._status, 0)
        plot_layout.addLayout(header)
        plot_layout.addWidget(self._plot, 1)

        from gui.form_widgets import FitWidthScrollArea

        params_inner = QWidget(self)
        params_layout = QVBoxLayout(params_inner)
        params_layout.setContentsMargins(6, 6, 6, 6)
        params_layout.addWidget(self.params, 1)

        params_side = FitWidthScrollArea(self)
        params_side.setWidget(params_inner)
        params_side.setMinimumWidth(220)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.addWidget(plot_side)
        splitter.addWidget(params_side)
        plot_side.setMinimumWidth(320)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([740, 360])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def local_settings(self) -> ViewerSettings:
        return self.params.settings()

    def update_placement(self, placement: PanelPlacement) -> None:
        """Mettre à jour le zoom / titre sans recréer la fenêtre."""
        self.placement = placement
        title = placement.title()
        self.setWindowTitle(title)
        self._title_label.setText(title)

    def _on_local_params_changed(self) -> None:
        self.settings = self.local_settings()
        self.refreshRequested.emit(self)

    def _on_detached_text_edited(self, result: object) -> None:
        """Sync édition in-place → panneau Style local de la fenêtre détachée."""
        role = (
            str(result.get("role", "") or "")
            if isinstance(result, dict)
            else str(getattr(result, "role", "") or "")
        )
        if role in {"", "annotation"}:
            return
        texts = _patch_text_overrides(self.params.settings().texts, result)
        self.params.apply_texts(texts)
        self.settings = self.local_settings()

    def render(self, request: RenderRequest) -> str:
        prev_request = self._last_request
        old_offset = _request_sync_offset(prev_request) if prev_request else 0.0
        self._last_request = request
        is_montage = str(self.placement.panel) == "montage_continuous_raw"
        saved_limits: list | dict | None = None
        if request.preserve_view:
            if is_montage:
                saved_limits = self._plot.capture_montage_view_limits()
            else:
                saved_limits = self._plot.capture_view_limits()
        if isinstance(saved_limits, list) and prev_request is not None:
            delta = _request_sync_offset(request) - old_offset
            if delta:
                saved_limits = _shift_limits_for_sync_delta(saved_limits, delta)
        elif isinstance(saved_limits, dict) and prev_request is not None:
            delta = _request_sync_offset(request) - old_offset
            if delta:
                saved_limits = _shift_limits_for_sync_delta(saved_limits, delta)
        sig = _scale_fit_signature(request.settings)
        if sig != self._last_scale_fit_signature:
            self._last_scale_fit_signature = sig
            self._arm_scale_fit()
        started = time.perf_counter()
        status = _finish_panel_render(
            self._plot,
            figure=self.figure,
            request=request,
            panel=self.placement.panel,
            preserve_limits=saved_limits,
            previous_request=prev_request,
        )
        self._last_render_s = time.perf_counter() - started
        if self._scale_fit_pending:
            QTimer.singleShot(0, self._schedule_detached_scale_fit)
        self._status.setText(f"{self._last_render_s * 1000:.0f} ms")
        self._status.setStyleSheet(f"color: {_STATUS_COLORS.get(status, '#64748b')};")
        self._status.setToolTip(
            f"Interactif — {INTERACTION_HINT} — "
            f"dernier rendu {self._last_render_s * 1000:.0f} ms"
        )
        return status

    def _arm_scale_fit(self) -> None:
        self._scale_fit_pending = True
        self._scale_fit_tries = 0
        self._scale_fit_width_sample = None
        self._scale_fit_height_sample = None
        self._scale_fit_retry.stop()

    def _schedule_detached_scale_fit(self) -> None:
        if not self._scale_fit_pending:
            return
        self._scale_fit_coalesce.start()

    def _run_detached_scale_fit(self) -> None:
        if not self._scale_fit_pending:
            return
        tries = int(self._scale_fit_tries)
        force = tries >= _FIT_RETRY_MAX
        plot_h = max(1, int(self._plot.height()))
        plot_w = max(1, int(self._plot.width()))
        h_state, h_px = _measure_scale_fit_height(self._plot, plot_h)
        w_state, w_px = _measure_scale_fit_width(
            self._plot, plot_w, strict=not force
        )
        h_state, self._scale_fit_height_sample = _accept_fit_sample(
            self._scale_fit_height_sample, h_state, h_px, force=force
        )
        w_state, self._scale_fit_width_sample = _accept_fit_sample(
            self._scale_fit_width_sample, w_state, w_px, force=force
        )
        if not force and (h_state == "wait" or w_state == "wait"):
            self._scale_fit_tries = tries + 1
            self._scale_fit_retry.start()
            return
        self._apply_detached_height(h_state, h_px if h_state == "ready" else None)
        self._apply_detached_width(w_state, w_px if w_state == "ready" else None)
        _settle_scale_fit(self, force=force, tries=tries)

    def _apply_detached_height(self, state: str, required: int | None) -> None:
        """Une seule hauteur figée. Ne pas relâcher le graph vers la fenêtre."""
        if state != "ready" or required is None:
            return
        required = int(required)
        plot_h = max(1, int(self._plot.height()))
        if (
            self._plot_locked_h == required
            and abs(required - plot_h) <= _FIT_MATCH_PX
        ):
            return
        self._plot_locked_h = required
        self._plot.setFixedHeight(required)
        if abs(required - plot_h) <= _FIT_MATCH_PX:
            return
        delta = required - plot_h
        self.resize(self.width(), max(480, int(self.height()) + int(delta)))

    def _apply_detached_width(self, state: str, required: int | None) -> None:
        """Une seule largeur figée. Ne pas relâcher le graph vers la fenêtre."""
        if state != "ready" or required is None:
            return
        required = max(1, int(required))
        plot_w = max(1, int(self._plot.width()))
        if (
            self._plot_locked_w == required
            and abs(required - plot_w) <= _FIT_MATCH_PX
        ):
            return
        self._plot_locked_w = required
        self._plot.setFixedWidth(required)
        if abs(required - plot_w) <= _FIT_MATCH_PX:
            return
        delta = required - plot_w
        self.resize(max(640, int(self.width()) + int(delta)), self.height())

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        if self._plot_locked_w is None or self._plot_locked_h is None:
            self._plot_lock_timer.start()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.isVisible() and (
            self._plot_locked_w is None or self._plot_locked_h is None
        ):
            self._plot_lock_timer.start()

    def _freeze_plot_chrome_once(self) -> None:
        """Garder la taille du graph : fenêtre et séparateur Paramètres ne l’étirent plus."""
        if not self.isVisible():
            return
        plot = self._plot
        width = int(plot.width())
        height = int(plot.height())
        if self._plot_locked_w is None:
            if width >= 320 and int(plot.maximumWidth()) >= 16_777_215:
                self._plot_locked_w = width
                plot.setFixedWidth(width)
            elif width < 320:
                self._plot_lock_timer.start()
        if self._plot_locked_h is None:
            if height >= 200 and int(plot.maximumHeight()) >= 16_777_215:
                self._plot_locked_h = height
                plot.setFixedHeight(height)
            elif height < 200:
                self._plot_lock_timer.start()

    def closeEvent(self, event: Any) -> None:  # noqa: D102
        self.closed.emit(self.placement)
        super().closeEvent(event)


def _zoom_placement_visible(
    placement: PanelPlacement, analysis: AnalysisSettings
) -> bool:
    """Filtrer un panneau zoom selon les coches Canal locales de la fenêtre."""
    panel = str(placement.panel)
    stream = str(getattr(placement, "stream", "") or "")

    if panel == "full_recording":
        if stream == "raw":
            return bool(analysis.show_raw)
        if stream == "hp":
            return bool(analysis.show_hp)
        if stream == "lp":
            return bool(analysis.show_lp)
        return True

    curve_flags = {
        "analysis_raw": analysis.show_raw,
        "mean_raw": analysis.show_raw,
        "analysis_hp": analysis.show_hp,
        "mean_hp": analysis.show_hp,
        "analysis_lp": analysis.show_lp,
        "mean_lp": analysis.show_lp,
        "analysis_rms": analysis.show_rms,
        "mean_rms": analysis.show_rms,
        "analysis_isi": analysis.show_isi,
        "analysis_overlay": analysis.show_overlay,
        "analysis_psth": analysis.show_psth,
        "analysis_trial_rate": analysis.show_trial_rate,
        "analysis_raster": analysis.show_raster_trials,
        "analysis_raster_channel": analysis.show_raster,
    }
    if panel in curve_flags:
        return bool(curve_flags[panel])
    # Impédance / résumés éventuels : toujours visibles s’ils sont dans la liste.
    return True


def zoom_scale_bar_amp_fit_active(settings: ViewerSettings) -> bool:
    """Échelle amplitude valeur + taille px → le fit widget pilote la hauteur."""
    style = settings.style
    return bool(style.scale_bar_amplitude_manual) and bool(
        style.scale_bar_amplitude_length_manual
    )


def zoom_scale_bar_widget_fit_active(settings: ViewerSettings) -> bool:
    """Valeur + taille px manuelles (amplitude ou temps) → fit H/W widget."""
    style = settings.style
    amp = zoom_scale_bar_amp_fit_active(settings)
    time = bool(style.scale_bar_time_manual) and bool(
        style.scale_bar_time_length_manual
    )
    return amp or time


def zoom_grid_panel_height(settings: ViewerSettings) -> int | None:
    """Hauteur fixe des panneaux zoom, ou ``None`` si le fit échelle amp. pilote H."""
    if zoom_scale_bar_amp_fit_active(settings):
        return None
    return max(160, int(settings.graph_height_px))


def initial_zoom_panel_height(settings: ViewerSettings) -> int:
    """Hauteur de départ grille zoom (avant fit échelle)."""
    fixed = zoom_grid_panel_height(settings)
    if fixed is not None:
        return fixed
    return max(160, int(settings.graph_height_px))


class DetachedZoomWindow(QDialog):
    """Zooms d’une plage : grille + Paramètres locaux (Canal / Affichage).

    Affiche uniquement la portion zoomée (``PanelPlacement`` custom zoom) —
    jamais la courbe aperçu complète. Les réglages locaux sont scopés zoom
    (pas d’échelle X héritée de l’aperçu).
    """

    closed = Signal()
    refreshRequested = Signal(object)  # DetachedZoomWindow

    def __init__(
        self,
        channel_name: str,
        placements: Sequence[PanelPlacement],
        settings: ViewerSettings,
        *,
        panel_height: int = 400,
        zoom_key: str = "",
        zoom_label: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        from gui.services.view_scopes import settings_for_zoom_window

        self.channel_name = str(channel_name)
        self.zoom_key = str(zoom_key or "")
        self.zoom_label = str(zoom_label or "").strip() or "Zoom"
        self.settings = settings_for_zoom_window(settings)
        # Liste complète fournie par l’aperçu ; le filtrage Canal est local.
        self._all_placements: tuple[PanelPlacement, ...] = tuple(placements)
        self._placements: tuple[PanelPlacement, ...] = ()
        self._panel_height = max(160, int(panel_height))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(self._window_title())
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(1100, max(560, self._panel_height + 220))

        from gui.widgets.panel_grid import PanelGrid

        self.grid = PanelGrid(self, allow_zoom=True)
        self._status = QLabel("—")
        self._status.setObjectName("panelStatus")

        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("paramsTabs")
        self._tabs.setDocumentMode(True)
        self._tabs.setUsesScrollButtons(False)
        tab_bar = self._tabs.tabBar()
        tab_bar.setElideMode(Qt.TextElideMode.ElideRight)
        tab_bar.setExpanding(True)

        self.params = LocalViewParams(
            self.settings,
            show_sync=True,
            show_display_extras=True,
            show_canal_tab=True,
            host_tabs=self._tabs,
            parent=self,
        )
        self.params.changed.connect(self._on_local_params_changed)

        plot_side = QWidget(self)
        plot_layout = QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.setSpacing(4)
        header = QHBoxLayout()
        self._title_label = QLabel(self._header_text())
        self._title_label.setWordWrap(True)
        header.addWidget(self._title_label, 1)
        header.addWidget(self._status, 0)
        plot_layout.addLayout(header)
        plot_layout.addWidget(self.grid, 1)

        params_side = QWidget(self)
        params_layout = QVBoxLayout(params_side)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(0)
        params_layout.addWidget(self._tabs, 1)
        params_side.setMinimumWidth(240)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.addWidget(plot_side)
        splitter.addWidget(params_side)
        plot_side.setMinimumWidth(320)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([740, 360])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.grid.renderFinished.connect(self._on_render_finished)
        self._configure_grid()

    def _window_title(self) -> str:
        return f"{self.channel_name} — {self.zoom_label}"

    def _header_text(self) -> str:
        n = len(self._placements)
        return f"{self.channel_name} — {self.zoom_label} ({n})"

    def _visible_placements(self) -> tuple[PanelPlacement, ...]:
        analysis = self.params.settings().analysis
        visible = tuple(
            p
            for p in self._all_placements
            if _zoom_placement_visible(p, analysis)
        )
        if visible:
            return visible
        # Repli : au moins les WIDE de la liste complète.
        fallback = tuple(
            p
            for p in self._all_placements
            if _zoom_placement_visible(
                p, replace(analysis, show_raw=True, show_hp=False, show_lp=False)
            )
        )
        return fallback or self._all_placements

    def _configure_grid(self) -> None:
        self._placements = self._visible_placements()
        self.grid.configure(
            self._placements,
            columns=1,
            panel_height=self._panel_height,
            uniform=True,
            fill=False,
        )
        self._title_label.setText(self._header_text())

    def local_settings(self) -> ViewerSettings:
        return self.params.settings()

    def update_placements(
        self,
        placements: Sequence[PanelPlacement],
        *,
        panel_height: int | None = None,
        zoom_label: str | None = None,
    ) -> None:
        """Mettre à jour la grille sans recréer la fenêtre."""
        self._all_placements = tuple(placements)
        if panel_height is not None:
            self._panel_height = max(160, int(panel_height))
        if zoom_label is not None:
            self.zoom_label = str(zoom_label).strip() or self.zoom_label
        self.setWindowTitle(self._window_title())
        self._configure_grid()

    @property
    def placements(self) -> tuple[PanelPlacement, ...]:
        return self._placements

    def contains_key(self, key: str) -> bool:
        return any(p.key == key for p in self._placements)

    def _on_local_params_changed(self) -> None:
        # Le parent (ChannelAnalysisWindow) recatalogue les placements, assure
        # les produits HIGH/RMS/spikes, puis redraw → schedule_render ici.
        self.settings = self.local_settings()
        ph = zoom_grid_panel_height(self.settings)
        if ph is not None:
            self._panel_height = ph
        self._configure_grid()
        self.refreshRequested.emit(self)

    def schedule_render(self, factory: Any, *, force: bool = True) -> None:
        # Rendu synchrone : à l’ouverture la géométrie viewport n’est pas fiable
        # et ``PanelGrid`` peut différer le 1er panneau (« Zoom WIDE / Plage 1 »)
        # comme hors écran — fenêtre vide jusqu’à un scroll.
        if force:
            self.grid.invalidate_all()
            drawn = int(self.grid.render_dirty_now(factory) or 0)
            # Limites puis fit échelle (même ordre que l’aperçu après paint).
            for widget in getattr(self.grid, "_panels", {}).values():
                host = getattr(widget, "plot_host", None) or getattr(
                    widget, "_plot", None
                )
                flush = getattr(host, "apply_pending_limits_now", None)
                if callable(flush):
                    try:
                        flush()
                    except Exception:
                        pass
                run_fit = getattr(widget, "_run_scale_bar_widget_fit", None)
                if callable(run_fit):
                    try:
                        run_fit()
                    except Exception:
                        pass
                else:
                    schedule = getattr(widget, "_schedule_scale_bar_widget_fit", None)
                    if callable(schedule):
                        schedule()
            self._status.setText(f"{drawn} panneau(x) · sync")
        else:
            self.grid.schedule_render(factory, force=False)

    def _on_render_finished(self, drawn: int, elapsed_s: float) -> None:
        self._status.setText(f"{drawn} panneau(x) · {elapsed_s * 1000:.0f} ms")

    def closeEvent(self, event: Any) -> None:  # noqa: D102
        self.closed.emit()
        super().closeEvent(event)
