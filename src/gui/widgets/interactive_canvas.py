"""Canvas matplotlib réellement interactif (zoom Ctrl+molette, pan, curseur, home).

Tous les graphiques GUI passent par ce widget — plus d’affichage « image morte ».
Les textes (titre, axes, légende, annotations) sont éditables in-place au clic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.text import Text as MplText
from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QFont, QKeyEvent
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QSizePolicy, QVBoxLayout, QWidget

# Pan = clic milieu ou droit (le clic gauche reste libre pour les barres de plage).
# Zoom = Ctrl+molette (molette seule = scroll de la grille parente).
# Texte = clic gauche sur un libellé → édition type zone de texte.
INTERACTION_HINT = (
    "Clic texte = éditer · Ctrl+molette zoom · clic milieu/droit pan · double-clic reset"
)
INTERACTION_HINT_NO_ZOOM = (
    "Clic texte = éditer · clic milieu/droit = pan (zoom désactivé sur cet aperçu)"
)
_PAN_FALLBACK_MS = 16

TextRole = Literal["title", "xlabel", "ylabel", "legend", "annotation"]


@dataclass(frozen=True)
class TextPinKey:
    """Identifie un texte précis dans la figure (pas un override global)."""

    role: TextRole
    # -1 = niveau figure (suptitle / fig.texts) ; sinon index dans figure.axes.
    axes_index: int
    # Index de légende ou d’annotation ; 0 pour title / xlabel / ylabel.
    item_index: int = 0


@dataclass(frozen=True)
class TextEditResult:
    """Résultat d’une édition in-place sur un texte du graphique."""

    role: TextRole
    text: str
    axes_index: int = 0
    item_index: int = 0
    legend_index: int | None = None  # compat : alias de item_index pour legend


@dataclass
class _TextHit:
    artist: MplText
    role: TextRole
    axes_index: int
    item_index: int = 0

    @property
    def key(self) -> TextPinKey:
        return TextPinKey(
            role=self.role,
            axes_index=int(self.axes_index),
            item_index=int(self.item_index),
        )


def _is_mpl_default_ylim(ylim: tuple[float, float]) -> bool:
    """True pour la plage Y par défaut de matplotlib ``(0, 1)``."""
    try:
        y0, y1 = float(ylim[0]), float(ylim[1])
    except Exception:
        return False
    return abs(y0) < 1e-12 and abs(y1 - 1.0) < 1e-9


def _data_exceeds_ylim(ax: Any, ylim: tuple[float, float]) -> bool:
    """True si le dataLim de l’axe dépasse nettement ``ylim`` (courbes coupées)."""
    try:
        bbox = ax.dataLim
        y0, y1 = float(ylim[0]), float(ylim[1])
        pad = max(1e-9, 0.02 * abs(y1 - y0))
        return float(bbox.y0) < y0 - pad or float(bbox.y1) > y1 + pad
    except Exception:
        return False


def _ctrl_held(event: Any | None = None) -> bool:
    """True si Ctrl est enfoncé (événement Qt ou état clavier courant)."""
    gui = getattr(event, "guiEvent", None) if event is not None else None
    if gui is not None:
        try:
            return bool(gui.modifiers() & Qt.KeyboardModifier.ControlModifier)
        except Exception:
            pass
    try:
        return bool(
            QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier
        )
    except Exception:
        return False


class CompactNavToolbar(NavigationToolbar2QT):
    """Barre de navigation compacte, toujours visible."""

    toolitems = [
        t
        for t in NavigationToolbar2QT.toolitems
        if t[0] in {"Home", "Back", "Forward", "Pan", "Zoom", "Save"}
    ]


class _ScopeCanvas(FigureCanvasQTAgg):
    """Canvas : Ctrl+molette = zoom ; molette seule transmise au scroll parent."""

    def __init__(self, figure: Figure, owner: "InteractiveCanvas") -> None:
        super().__init__(figure)
        self._owner = owner

    def sizeHint(self) -> QSize:  # noqa: D102
        # FigureCanvasQTAgg base le hint sur figsize×dpi : après agrandissement,
        # la largeur « collée » force un scroll horizontal dans PanelGrid.
        # On ne impose que la hauteur ; la largeur suit le viewport (Expanding).
        hint = super().sizeHint()
        return QSize(200, max(80, int(hint.height())))

    def minimumSizeHint(self) -> QSize:  # noqa: D102
        hint = super().minimumSizeHint()
        return QSize(120, max(60, int(hint.height())))

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        # Sans Ctrl : laisser le scroll à la grille parente.
        if not _ctrl_held():
            self._owner._did_zoom = False
            event.ignore()
            return
        # Ctrl+molette : zoom matplotlib, puis consommer l’événement.
        super().wheelEvent(event)
        if getattr(self._owner, "_did_zoom", False):
            event.accept()
            self._owner._did_zoom = False
        else:
            event.ignore()


class _LineEditKeyFilter(QObject):
    """Escape → annuler ; le reste est géré par QLineEdit."""

    def __init__(self, owner: "InteractiveCanvas") -> None:
        super().__init__(owner)
        self._owner = owner

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if event.key() == Qt.Key.Key_Escape:
                self._owner.cancel_text_edit()
                return True
        return super().eventFilter(obj, event)


class InteractiveCanvas(QWidget):
    """Figure + toolbar + interactions souris (Ctrl+molette, pan, curseur, textes)."""

    textEdited = Signal(object)  # TextEditResult

    def __init__(
        self,
        *,
        figsize: tuple[float, float] = (4.2, 2.8),
        parent: QWidget | None = None,
        show_toolbar: bool = True,
        show_cursor: bool = True,
        min_height: int = 120,
        # "constrained" coûte cher avec beaucoup d’axes (montage) — passer None.
        layout: str | None = "constrained",
        allow_zoom: bool = True,
    ) -> None:
        super().__init__(parent)
        self._did_zoom = False
        self._allow_zoom = bool(allow_zoom)
        self.figure = Figure(figsize=figsize, layout=layout, facecolor="#ffffff")
        self.canvas = _ScopeCanvas(self.figure, self)
        self.canvas.setStyleSheet("background-color: #ffffff;")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumHeight(int(min_height))
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.setMouseTracking(True)

        self.toolbar = CompactNavToolbar(self.canvas, self)
        self.toolbar.setIconSize(self.toolbar.iconSize() * 0.75)
        self.toolbar.setVisible(bool(show_toolbar))
        self.toolbar.setMaximumHeight(28)
        if not self._allow_zoom:
            # Masquer Zoom toolbar ; pan / home restent utiles.
            try:
                for action in self.toolbar.actions():
                    text = str(action.text() or action.iconText() or "")
                    if "zoom" in text.lower():
                        action.setVisible(False)
            except Exception:
                pass

        hint = INTERACTION_HINT if self._allow_zoom else INTERACTION_HINT_NO_ZOOM
        self._cursor_label = QLabel(hint, self)
        self._cursor_label.setObjectName("panelStatus")
        self._cursor_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._cursor_label.setMinimumWidth(0)
        self._cursor_label.setWordWrap(True)
        self._cursor_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._cursor_label.setVisible(bool(show_cursor))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self._cursor_label)

        self._press_ax: Any | None = None
        self._press_xy: tuple[float, float] | None = None
        self._xlim0: tuple[float, float] | None = None
        self._ylim0: tuple[float, float] | None = None
        self._pan_trans: Any | None = None
        # Blit cache for interactive pan/zoom (axes bbox snapshot).
        self._blit_bg: Any | None = None
        self._blit_ax: Any | None = None
        self._interaction_draw_pending = False
        self._interaction_timer = QTimer(self)
        self._interaction_timer.setSingleShot(True)
        self._interaction_timer.setInterval(_PAN_FALLBACK_MS)
        self._interaction_timer.timeout.connect(self._flush_interaction_draw)

        self._text_edit: QLineEdit | None = None
        self._text_hit: _TextHit | None = None
        self._text_was_visible = True
        self._text_closing = False
        self._text_key_filter = _LineEditKeyFilter(self)
        self._hover_text = False
        # Éditions ciblées : une entrée = un seul artiste Text, jamais un broadcast.
        self._text_pins: dict[TextPinKey, str] = {}

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:  # noqa: D102
        canvas_hint = self.canvas.sizeHint()
        return QSize(min(200, int(canvas_hint.width())), int(canvas_hint.height()) + 36)

    def minimumSizeHint(self) -> QSize:  # noqa: D102
        canvas_hint = self.canvas.minimumSizeHint()
        return QSize(0, int(canvas_hint.height()) + 28)

    # ---------------------------------------------------------------- drawing

    def draw_idle(self) -> None:
        self.cancel_text_edit()
        self.invalidate_blit()
        self.reapply_text_pins()
        self.canvas.draw_idle()

    def invalidate_blit(self) -> None:
        """Drop cached background after a full panel redraw."""
        self._blit_bg = None
        self._blit_ax = None

    def _try_blit(self, ax: Any) -> bool:
        """Fast path: redraw one axes via blit. Falls back if the backend rejects it."""
        try:
            renderer = self.canvas.get_renderer()
            if self._blit_bg is None or self._blit_ax is not ax:
                # Snapshot current frame, then we'll redraw on the next call.
                ax.figure.canvas.draw()
                self._blit_bg = renderer.copy_from_bbox(ax.bbox)
                self._blit_ax = ax
            self.canvas.restore_region(self._blit_bg)
            ax.draw_artist(ax)
            self.canvas.blit(ax.bbox)
            # Refresh cache for the next motion event (new view already drawn).
            self._blit_bg = renderer.copy_from_bbox(ax.bbox)
            self._blit_ax = ax
            return True
        except Exception:
            self._blit_bg = None
            self._blit_ax = None
            return False

    def _request_interaction_draw(self, ax: Any | None = None) -> None:
        """Prefer blit; otherwise coalesce draw_idle to ~60 Hz during pan/zoom."""
        if ax is not None and self._try_blit(ax):
            return
        self._interaction_draw_pending = True
        if not self._interaction_timer.isActive():
            self._interaction_timer.start()

    def _flush_interaction_draw(self) -> None:
        if not self._interaction_draw_pending:
            return
        self._interaction_draw_pending = False
        self.canvas.draw_idle()

    def capture_view_limits(
        self,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]] | None:
        """Snapshot des (xlim, ylim) de chaque axe — pour survivre à un redessin."""
        axes = list(self.figure.axes)
        if not axes:
            return None
        limits: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for ax in axes:
            try:
                xlim = (float(ax.get_xlim()[0]), float(ax.get_xlim()[1]))
                ylim = (float(ax.get_ylim()[0]), float(ax.get_ylim()[1]))
            except Exception:
                return None
            if not np.isfinite([*xlim, *ylim]).all() or xlim[0] == xlim[1]:
                return None
            limits.append((xlim, ylim))
        return limits

    def capture_montage_view_limits(
        self,
    ) -> dict[tuple[int, str], tuple[tuple[float, float], tuple[float, float]]] | None:
        """Snapshot (xlim, ylim) indexé par ``(canal, kind)`` — stable si la pile change."""
        stored = getattr(self.figure, "_erg_montage_state", None)
        if not isinstance(stored, dict):
            return None
        row_keys = list(stored.get("row_keys") or [])
        axes = list(self.figure.axes)
        if not row_keys or len(row_keys) != len(axes):
            return None
        limits: dict[tuple[int, str], tuple[tuple[float, float], tuple[float, float]]] = {}
        for key, ax in zip(row_keys, axes):
            if not (isinstance(key, (tuple, list)) and len(key) >= 2):
                return None
            ch_kind = (int(key[0]), str(key[1]))
            try:
                xlim = (float(ax.get_xlim()[0]), float(ax.get_xlim()[1]))
                ylim = (float(ax.get_ylim()[0]), float(ax.get_ylim()[1]))
            except Exception:
                return None
            if not np.isfinite([*xlim, *ylim]).all() or xlim[0] == xlim[1]:
                return None
            limits[ch_kind] = (xlim, ylim)
        return limits

    def restore_view_limits(
        self,
        limits: list[tuple[tuple[float, float], tuple[float, float]]] | None,
        *,
        restore_x: bool = True,
        restore_y: bool | list[bool] | tuple[bool, ...] = True,
    ) -> None:
        """Rétablir un snapshot de vue (ignore les axes en trop / manquants).

        ``restore_y`` peut être un booléen global ou un masque par axe
        (``False`` = garder le ylim du rendu, ex. échelle Y manuelle).
        """
        if not limits:
            return
        axes = list(self.figure.axes)
        if not axes:
            return
        n = min(len(axes), len(limits))
        if isinstance(restore_y, (list, tuple)):
            y_mask: list[bool] | None = [bool(v) for v in restore_y]
        else:
            y_mask = None
            restore_y_all = bool(restore_y)
        for index in range(n):
            xlim, ylim = limits[index]
            ax = axes[index]
            try:
                if restore_x:
                    ax.set_xlim(xlim[0], xlim[1])
                do_y = (
                    y_mask[index]
                    if y_mask is not None and index < len(y_mask)
                    else (y_mask is None and restore_y_all)
                )
                if do_y and np.isfinite(ylim).all() and ylim[0] != ylim[1]:
                    if _is_mpl_default_ylim(ylim) and _data_exceeds_ylim(ax, ylim):
                        pass
                    else:
                        ax.set_ylim(ylim[0], ylim[1])
            except Exception:
                pass
        # Même base temporelle si le nombre d’axes a changé (ex. flux ajouté).
        if restore_x and len(axes) > n and limits:
            xlim0 = limits[0][0]
            for ax in axes[n:]:
                try:
                    ax.set_xlim(xlim0[0], xlim0[1])
                except Exception:
                    pass

    def restore_montage_view_limits(
        self,
        limits: dict[tuple[int, str], tuple[tuple[float, float], tuple[float, float]]] | None,
        *,
        restore_x: bool = True,
        restore_y: bool | list[bool] | tuple[bool, ...] = True,
    ) -> None:
        """Rétablir la vue montage par ``(canal, kind)`` — jamais par index d’axe."""
        if not limits:
            return
        stored = getattr(self.figure, "_erg_montage_state", None)
        if not isinstance(stored, dict):
            return
        row_keys = list(stored.get("row_keys") or [])
        axes = list(self.figure.axes)
        if not row_keys or len(row_keys) != len(axes):
            return
        if isinstance(restore_y, (list, tuple)):
            y_mask: list[bool] | None = [bool(v) for v in restore_y]
        else:
            y_mask = None
            restore_y_all = bool(restore_y)
        shared_x: tuple[float, float] | None = None
        for index, (key, ax) in enumerate(zip(row_keys, axes)):
            if not (isinstance(key, (tuple, list)) and len(key) >= 2):
                continue
            ch_kind = (int(key[0]), str(key[1]))
            saved = limits.get(ch_kind)
            if saved is None:
                continue
            xlim, ylim = saved
            try:
                if restore_x:
                    ax.set_xlim(xlim[0], xlim[1])
                    if shared_x is None:
                        shared_x = (float(xlim[0]), float(xlim[1]))
                do_y = (
                    y_mask[index]
                    if y_mask is not None and index < len(y_mask)
                    else (y_mask is None and restore_y_all)
                )
                if do_y and np.isfinite(ylim).all() and ylim[0] != ylim[1]:
                    # Ne pas réinjecter le ylim matplotlib par défaut (0, 1)
                    # par-dessus un autoscale qui couvre déjà les données.
                    if _is_mpl_default_ylim(ylim) and _data_exceeds_ylim(ax, ylim):
                        pass
                    else:
                        ax.set_ylim(ylim[0], ylim[1])
            except Exception:
                pass
        # Lignes nouvellement ajoutées : reprendre le X d’une ligne connue.
        if restore_x and shared_x is not None:
            known = set(limits)
            for key, ax in zip(row_keys, axes):
                if not (isinstance(key, (tuple, list)) and len(key) >= 2):
                    continue
                if (int(key[0]), str(key[1])) in known:
                    continue
                try:
                    ax.set_xlim(shared_x[0], shared_x[1])
                except Exception:
                    pass

    def enable_default_pan(self) -> None:
        """Activer le mode Pan de la toolbar après un redraw (si pas déjà actif)."""
        try:
            mode = getattr(self.toolbar, "mode", None)
            name = getattr(mode, "name", None) or str(mode or "")
            if str(name).upper() in {"PAN", "PAN/ZOOM"}:
                return
            if "pan" in str(name).lower():
                return
            self.toolbar.pan()
        except Exception:
            pass

    # ----------------------------------------------------------- text editing

    def reapply_text_pins(self) -> None:
        """Réappliquer uniquement les textes édités individuellement (après redraw)."""
        if not self._text_pins:
            return
        for key, text in list(self._text_pins.items()):
            artist = self._resolve_pin_artist(key)
            if artist is None:
                continue
            try:
                artist.set_text(text)
            except Exception:
                pass

    def clear_text_pins(self) -> None:
        """Oublier les éditions in-place (ex. réinitialisation style)."""
        self._text_pins.clear()

    def _resolve_pin_artist(self, key: TextPinKey) -> MplText | None:
        fig = self.figure
        if key.role == "title" and key.axes_index < 0:
            artist = getattr(fig, "_suptitle", None)
            return artist if isinstance(artist, MplText) else None
        if key.role == "annotation" and key.axes_index < 0:
            texts = [
                t
                for t in list(getattr(fig, "texts", []) or [])
                if t is not getattr(fig, "_suptitle", None) and isinstance(t, MplText)
            ]
            if 0 <= key.item_index < len(texts):
                return texts[key.item_index]
            return None
        axes = list(getattr(fig, "axes", []) or [])
        if key.axes_index < 0 or key.axes_index >= len(axes):
            return None
        ax = axes[key.axes_index]
        if key.role == "title":
            artist = getattr(ax, "title", None)
            return artist if isinstance(artist, MplText) else None
        if key.role == "xlabel":
            xaxis = getattr(ax, "xaxis", None)
            artist = getattr(xaxis, "label", None) if xaxis is not None else None
            return artist if isinstance(artist, MplText) else None
        if key.role == "ylabel":
            yaxis = getattr(ax, "yaxis", None)
            artist = getattr(yaxis, "label", None) if yaxis is not None else None
            return artist if isinstance(artist, MplText) else None
        if key.role == "legend":
            legend = ax.get_legend()
            if legend is None:
                return None
            texts = list(legend.get_texts())
            if 0 <= key.item_index < len(texts):
                artist = texts[key.item_index]
                return artist if isinstance(artist, MplText) else None
            return None
        if key.role == "annotation":
            texts = [t for t in list(getattr(ax, "texts", []) or []) if isinstance(t, MplText)]
            if 0 <= key.item_index < len(texts):
                return texts[key.item_index]
            return None
        return None

    def commit_text_edit(self) -> None:
        """Valider l’édition en cours — uniquement le texte sélectionné."""
        if self._text_closing or self._text_edit is None or self._text_hit is None:
            return
        self._text_closing = True
        try:
            new_text = self._text_edit.text()
            hit = self._text_hit
            key = hit.key
            self._close_text_editor(restore_artist=False)
            self._text_pins[key] = new_text
            try:
                hit.artist.set_text(new_text)
                hit.artist.set_visible(True)
            except Exception:
                # Artiste peut avoir été invalidé : résoudre via la pin.
                artist = self._resolve_pin_artist(key)
                if artist is not None:
                    try:
                        artist.set_text(new_text)
                    except Exception:
                        pass
            self.invalidate_blit()
            self.canvas.draw_idle()
            self.textEdited.emit(
                TextEditResult(
                    role=hit.role,
                    text=new_text,
                    axes_index=hit.axes_index,
                    item_index=hit.item_index,
                    legend_index=hit.item_index if hit.role == "legend" else None,
                )
            )
        finally:
            self._text_closing = False

    def cancel_text_edit(self) -> None:
        """Annuler l’édition et restaurer le texte matplotlib."""
        if self._text_closing or self._text_edit is None:
            return
        self._text_closing = True
        try:
            hit = self._text_hit
            was_visible = self._text_was_visible
            self._close_text_editor(restore_artist=False)
            if hit is not None:
                try:
                    hit.artist.set_visible(was_visible)
                except Exception:
                    pass
                self.invalidate_blit()
                self.canvas.draw_idle()
        finally:
            self._text_closing = False

    def _close_text_editor(self, *, restore_artist: bool) -> None:
        edit = self._text_edit
        hit = self._text_hit
        was_visible = self._text_was_visible
        self._text_edit = None
        self._text_hit = None
        if edit is not None:
            try:
                edit.removeEventFilter(self._text_key_filter)
            except Exception:
                pass
            try:
                edit.editingFinished.disconnect(self._on_edit_finished)
            except (TypeError, RuntimeError):
                pass
            try:
                edit.returnPressed.disconnect(self.commit_text_edit)
            except (TypeError, RuntimeError):
                pass
            edit.hide()
            edit.deleteLater()
        if restore_artist and hit is not None:
            try:
                hit.artist.set_visible(was_visible)
            except Exception:
                pass

    def _on_edit_finished(self) -> None:
        # Focus perdu (clic ailleurs) → commit. Escape appelle cancel avant.
        if self._text_closing or self._text_edit is None:
            return
        self.commit_text_edit()

    def _iter_editable_texts(self) -> list[_TextHit]:
        hits: list[_TextHit] = []
        fig = self.figure
        seen: set[int] = set()

        def _add(
            artist: Any,
            role: TextRole,
            *,
            axes_index: int,
            item_index: int = 0,
        ) -> None:
            if not isinstance(artist, MplText):
                return
            if id(artist) in seen:
                return
            if not artist.get_visible():
                return
            seen.add(id(artist))
            hits.append(
                _TextHit(
                    artist=artist,
                    role=role,
                    axes_index=int(axes_index),
                    item_index=int(item_index),
                )
            )

        sup = getattr(fig, "_suptitle", None)
        _add(sup, "title", axes_index=-1)

        for ax_index, ax in enumerate(list(getattr(fig, "axes", []) or [])):
            _add(getattr(ax, "title", None), "title", axes_index=ax_index)
            xaxis = getattr(ax, "xaxis", None)
            yaxis = getattr(ax, "yaxis", None)
            if xaxis is not None:
                _add(getattr(xaxis, "label", None), "xlabel", axes_index=ax_index)
            if yaxis is not None:
                _add(getattr(yaxis, "label", None), "ylabel", axes_index=ax_index)
            legend = ax.get_legend()
            if legend is not None:
                for index, text in enumerate(legend.get_texts()):
                    _add(text, "legend", axes_index=ax_index, item_index=index)
            for text_index, text in enumerate(list(getattr(ax, "texts", []) or [])):
                _add(text, "annotation", axes_index=ax_index, item_index=text_index)

        fig_ann_index = 0
        for text in list(getattr(fig, "texts", []) or []):
            if text is sup:
                continue
            _add(text, "annotation", axes_index=-1, item_index=fig_ann_index)
            fig_ann_index += 1
        return hits

    def _hit_test_text(self, event: Any) -> _TextHit | None:
        if event.x is None or event.y is None:
            return None
        # Agrandir légèrement la zone de clic pour les petits libellés.
        pad = 3.0
        best: tuple[float, _TextHit] | None = None
        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            return None
        for hit in self._iter_editable_texts():
            artist = hit.artist
            try:
                # contains() tient compte de la rotation (ylabel).
                contained, _info = artist.contains(event)
            except Exception:
                contained = False
            if not contained:
                try:
                    bbox = artist.get_window_extent(renderer).padded(pad)
                    contained = bool(bbox.contains(event.x, event.y))
                except Exception:
                    contained = False
            if not contained:
                continue
            try:
                bbox = artist.get_window_extent(renderer)
                area = float(max(1.0, bbox.width * bbox.height))
            except Exception:
                area = 1.0e9
            if best is None or area < best[0]:
                best = (area, hit)
        return best[1] if best is not None else None

    def _start_text_edit(self, hit: _TextHit) -> None:
        self.cancel_text_edit()
        artist = hit.artist
        try:
            renderer = self.canvas.get_renderer()
            bbox = artist.get_window_extent(renderer)
        except Exception:
            return

        # Coords matplotlib : origine bas-gauche ; Qt : haut-gauche.
        canvas_h = float(self.canvas.height())
        x0 = float(bbox.x0) - 4.0
        y_top = canvas_h - float(bbox.y1) - 2.0
        width = max(72.0, float(bbox.width) + 16.0)
        height = max(22.0, float(bbox.height) + 6.0)

        # Libellé Y souvent vertical : zone d’édition horizontale près du centre.
        try:
            rotation = abs(float(artist.get_rotation() or 0.0)) % 180.0
        except Exception:
            rotation = 0.0
        if 45.0 < rotation < 135.0:
            cx = float(bbox.x0 + bbox.x1) * 0.5
            cy = float(bbox.y0 + bbox.y1) * 0.5
            width = max(120.0, width)
            height = max(24.0, min(32.0, height + 8.0))
            x0 = cx - width * 0.5
            y_top = canvas_h - cy - height * 0.5

        # Garder l’éditeur dans le canvas.
        x0 = max(0.0, min(x0, float(self.canvas.width()) - width))
        y_top = max(0.0, min(y_top, canvas_h - height))

        current = str(artist.get_text() or "")
        edit = QLineEdit(self.canvas)
        edit.setObjectName("graphTextEdit")
        edit.setText(current)
        edit.setGeometry(int(round(x0)), int(round(y_top)), int(round(width)), int(round(height)))
        try:
            size_pt = float(artist.get_fontsize() or 10.0)
        except Exception:
            size_pt = 10.0
        font = QFont(edit.font())
        font.setPointSizeF(max(8.0, min(18.0, size_pt)))
        try:
            weight = str(artist.get_fontweight() or "")
            if weight in {"bold", "heavy", "black"} or (
                weight.isdigit() and int(weight) >= 600
            ):
                font.setBold(True)
        except Exception:
            pass
        edit.setFont(font)
        edit.setStyleSheet(
            "QLineEdit#graphTextEdit {"
            " background-color: #fffef7;"
            " color: #111827;"
            " border: 1.5px solid #2563eb;"
            " border-radius: 2px;"
            " padding: 1px 4px;"
            " selection-background-color: #93c5fd;"
            "}"
        )
        edit.setToolTip("Entrée = valider · Échap = annuler")
        edit.installEventFilter(self._text_key_filter)
        edit.editingFinished.connect(self._on_edit_finished)
        edit.returnPressed.connect(self.commit_text_edit)

        self._text_was_visible = bool(artist.get_visible())
        try:
            artist.set_visible(False)
            self.invalidate_blit()
            self.canvas.draw_idle()
        except Exception:
            pass

        self._text_edit = edit
        self._text_hit = hit
        edit.show()
        edit.raise_()
        edit.setFocus(Qt.FocusReason.MouseFocusReason)
        edit.selectAll()

    # ----------------------------------------------------------- interactions

    def _on_scroll(self, event: Any) -> None:
        if not self._allow_zoom or not _ctrl_held(event):
            self._did_zoom = False
            return
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            self._did_zoom = False
            return
        # Molette haut = zoom avant, bas = zoom arrière.
        scale = 0.8 if getattr(event, "step", 0) > 0 else 1.25
        try:
            self._zoom_around(ax, float(event.xdata), float(event.ydata), scale)
            self._request_interaction_draw(ax)
            self._did_zoom = True
        except Exception:
            self._did_zoom = False

    @staticmethod
    def _zoom_around(ax: Any, x: float, y: float, scale: float) -> None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        new_xmin = x - (x - xmin) * scale
        new_xmax = x + (xmax - x) * scale
        new_ymin = y - (y - ymin) * scale
        new_ymax = y + (ymax - y) * scale
        if np.isfinite([new_xmin, new_xmax, new_ymin, new_ymax]).all() and new_xmin != new_xmax:
            ax.set_xlim(new_xmin, new_xmax)
            ax.set_ylim(new_ymin, new_ymax)

    def _on_press(self, event: Any) -> None:
        if event.button == 1:
            hit = self._hit_test_text(event)
            if hit is not None:
                self._start_text_edit(hit)
                return
            if self._text_edit is not None:
                # Clic hors texte : valider puis éventuellement reset.
                self.commit_text_edit()
            if getattr(event, "dblclick", False):
                try:
                    self.toolbar.home()
                except Exception:
                    pass
            return

        # Clic milieu ou droit : pan (pixels + transform figé au press).
        if event.button not in (2, 3) or event.inaxes is None:
            return
        if event.x is None or event.y is None:
            return
        if self._text_edit is not None:
            self.commit_text_edit()
        self._press_ax = event.inaxes
        self._press_xy = (float(event.x), float(event.y))
        self._xlim0 = tuple(event.inaxes.get_xlim())
        self._ylim0 = tuple(event.inaxes.get_ylim())
        # Figé : sinon chaque set_xlim change le mapping pixel→données et le pan décroche.
        self._pan_trans = event.inaxes.transData.frozen()
        self.invalidate_blit()
        self.canvas.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def _on_release(self, event: Any) -> None:
        del event
        was_panning = self._press_ax is not None
        self._press_ax = None
        self._press_xy = None
        self._xlim0 = None
        self._ylim0 = None
        self._pan_trans = None
        if self._hover_text:
            self.canvas.setCursor(QCursor(Qt.CursorShape.IBeamCursor))
        else:
            self.canvas.unsetCursor()
        if was_panning:
            # Final crisp frame after the last coalesced/blit update.
            self.invalidate_blit()
            self.canvas.draw_idle()

    def _on_motion(self, event: Any) -> None:
        # Pendant un pan : suivre les pixels (pas besoin de xdata / inaxes).
        if (
            self._press_ax is not None
            and self._press_xy is not None
            and self._xlim0 is not None
            and self._ylim0 is not None
            and self._pan_trans is not None
        ):
            if event.x is None or event.y is None:
                return
            try:
                inv = self._pan_trans.inverted()
                x0, y0 = inv.transform(self._press_xy)
                x1, y1 = inv.transform((float(event.x), float(event.y)))
                dx = float(x1 - x0)
                dy = float(y1 - y0)
                self._press_ax.set_xlim(self._xlim0[0] - dx, self._xlim0[1] - dx)
                self._press_ax.set_ylim(self._ylim0[0] - dy, self._ylim0[1] - dy)
                self._request_interaction_draw(self._press_ax)
            except Exception:
                pass
            return

        hint = INTERACTION_HINT if self._allow_zoom else INTERACTION_HINT_NO_ZOOM
        over_text = self._text_edit is None and self._hit_test_text(event) is not None
        if over_text != self._hover_text:
            self._hover_text = over_text
            if over_text:
                self.canvas.setCursor(QCursor(Qt.CursorShape.IBeamCursor))
            else:
                self.canvas.unsetCursor()

        if event.inaxes is not None and event.xdata is not None and event.ydata is not None:
            suffix = " · texte" if over_text else ""
            self._cursor_label.setText(
                f"t = {event.xdata:.4g}   y = {event.ydata:.4g}   ({hint}){suffix}"
            )
        elif over_text:
            self._cursor_label.setText(f"Clic pour modifier le texte · {hint}")
        else:
            self._cursor_label.setText(hint)
