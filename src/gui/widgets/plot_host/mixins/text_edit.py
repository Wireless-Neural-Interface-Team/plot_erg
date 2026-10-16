"""Édition inline."""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLineEdit

from ..text_model import _EditableText, _TextPinKey
from ..text_style import _editor_font, _editor_stylesheet


class TextEditMixin:
    def clear_text_pins(self) -> None:
        """Oublier les éditions in-place (ex. réinitialisation style)."""
        self._cancel_text_edit()
        self._text_pins.clear()

    def _reapply_text_pins(self) -> None:
        if not self._text_pins:
            return
        by_key = {edit.key: edit for edit in self._editable}
        for key, text in list(self._text_pins.items()):
            edit = by_key.get(key)
            if edit is None:
                continue
            try:
                edit.set_text(text)
            except Exception:
                pass

    def _text_hit_at(self, pos: QPoint) -> _EditableText | None:
        if not self._editable:
            return None
        try:
            scene_pos = self._glw.mapToScene(pos)
        except Exception:
            return None
        try:
            scene = self._glw.scene()
            items = scene.items(scene_pos) if scene is not None else []
        except Exception:
            items = []
        item_ids = {id(it) for it in items}
        # Inclure les parents (LabelItem contient un QGraphicsTextItem enfant).
        for it in list(items):
            parent = it
            for _ in range(6):
                parent = parent.parentItem() if parent is not None else None
                if parent is None:
                    break
                item_ids.add(id(parent))
        for edit in self._editable:
            if id(edit.item) in item_ids:
                return edit
            # Match enfant texte d’un LabelItem
            if isinstance(edit.item, pg.LabelItem):
                child = getattr(edit.item, "item", None)
                if child is not None and id(child) in item_ids:
                    return edit
        # Fallback géométrique (marges de clic plus larges).
        for edit in self._editable:
            try:
                rect = edit.item.sceneBoundingRect().adjusted(-4, -4, 4, 4)
                if rect.contains(scene_pos):
                    return edit
            except Exception:
                continue
        return None

    def _start_text_edit(self, hit: _EditableText) -> None:
        self._cancel_text_edit()
        edit = QLineEdit(self)
        edit.setText(hit.get_text())
        editor_font = _editor_font(hit.item)
        edit.setFont(editor_font)
        edit.setStyleSheet(_editor_stylesheet(editor_font))
        try:
            rect = hit.item.sceneBoundingRect()
            top_left = self._glw.mapFromScene(rect.topLeft())
            bottom_right = self._glw.mapFromScene(rect.bottomRight())
            # mapFromScene → coords du viewport GLW ; convertir en coords PlotHost.
            glw_origin = self._glw.mapTo(self, QPoint(0, 0))
            x = glw_origin.x() + min(top_left.x(), bottom_right.x()) - 2
            y = glw_origin.y() + min(top_left.y(), bottom_right.y()) - 2
            w = max(80, abs(bottom_right.x() - top_left.x()) + 8)
            h = max(22, abs(bottom_right.y() - top_left.y()) + 4)
            edit.setGeometry(int(x), int(y), int(w), int(h))
        except Exception:
            edit.setGeometry(40, 40, 220, 24)
        self._text_edit = edit
        self._text_hit = hit
        edit.returnPressed.connect(self._commit_text_edit)
        edit.editingFinished.connect(self._on_text_edit_finished)
        edit.show()
        edit.setFocus(Qt.FocusReason.MouseFocusReason)
        edit.selectAll()

    def _on_text_edit_finished(self) -> None:
        if self._text_closing or self._text_edit is None:
            return
        self._commit_text_edit()

    def _commit_text_edit(self) -> None:
        if self._text_closing or self._text_edit is None or self._text_hit is None:
            return
        self._text_closing = True
        try:
            new_text = self._text_edit.text()
            hit = self._text_hit
            self._close_text_editor()
            self._text_pins[hit.key] = new_text
            try:
                hit.set_text(new_text)
            except Exception:
                pass
            self.draw_idle()
            self.textEdited.emit(
                {
                    "role": hit.role,
                    "text": new_text,
                    "axes_index": hit.axes_index,
                    "item_index": hit.item_index,
                }
            )
        finally:
            self._text_closing = False

    def _cancel_text_edit(self) -> None:
        if self._text_closing:
            return
        self._text_closing = True
        try:
            self._close_text_editor()
        finally:
            self._text_closing = False

    def _close_text_editor(self) -> None:
        edit = self._text_edit
        self._text_edit = None
        self._text_hit = None
        if edit is None:
            return
        try:
            edit.returnPressed.disconnect(self._commit_text_edit)
        except (TypeError, RuntimeError):
            pass
        try:
            edit.editingFinished.disconnect(self._on_text_edit_finished)
        except (TypeError, RuntimeError):
            pass
        edit.hide()
        edit.deleteLater()

