"""Widgets d'édition : couleur, formulaire lié à une dataclass, sélection de canaux."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import seaborn as sns
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLineEdit, QPushButton, QScrollArea, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


class ColorButton(QPushButton):
    colorChanged = Signal(str)

    def __init__(self, color: str = "#000000", alpha: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._alpha = alpha
        self._color = color
        self.setFixedHeight(22)
        self.setMinimumWidth(70)
        self.clicked.connect(self._pick)
        self.set_color(color)

    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color or "#000000"
        q = QColor(self._color)
        text = "#000000" if q.lightnessF() > 0.55 or q.alphaF() < 0.4 else "#ffffff"
        self.setText(self._color)
        self.setStyleSheet(
            f"QPushButton {{ background: {q.name(QColor.HexArgb)}; color: {text};"
            " border: 1px solid #8c959f; border-radius: 3px; font-size: 8pt; }"
        )

    def _pick(self) -> None:
        opts = QColorDialog.ShowAlphaChannel if self._alpha else QColorDialog.ColorDialogOption(0)
        q = QColorDialog.getColor(QColor(self._color), self, "Choisir une couleur", opts)
        if not q.isValid():
            return
        if self._alpha and q.alpha() < 255:
            name = "#{:02x}{:02x}{:02x}{:02x}".format(q.red(), q.green(), q.blue(), q.alpha())
        else:
            name = q.name()
        self.set_color(name)
        self.colorChanged.emit(name)


class _NoWheelMixin:
    def wheelEvent(self, ev):  # évite les modifications accidentelles en faisant défiler
        if self.hasFocus():
            super().wheelEvent(ev)
        else:
            ev.ignore()


class SpinBox(_NoWheelMixin, QSpinBox):
    pass


class DoubleSpinBox(_NoWheelMixin, QDoubleSpinBox):
    pass


class ComboBox(_NoWheelMixin, QComboBox):
    pass


@dataclass
class Field:
    name: str
    label: str
    kind: str  # int | float | str | bool | choice | color | color_alpha
    choices: list[tuple[Any, str]] | None = None
    choices_fn: Callable[[Any], list[tuple[Any, str]]] | None = None
    minimum: float = -1e9
    maximum: float = 1e9
    step: float = 1.0
    decimals: int = 3
    suffix: str = ""
    enabled_if: Callable[[Any], bool] | None = None
    tooltip: str = ""


class BoundForm(QWidget):
    """Formulaire en groupes, lié aux attributs d'un objet. Émet ``changed(nom)``."""

    changed = Signal(str)

    def __init__(self, groups: list[tuple[str, list[Field]]], parent=None) -> None:
        super().__init__(parent)
        self.obj = None
        self._loading = False
        self._widgets: dict[str, tuple[Field, QWidget]] = {}
        self._rows: dict[str, tuple[QFormLayout, QWidget]] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        for title, flds in groups:
            box = QGroupBox(title)
            form = QFormLayout(box)
            form.setLabelAlignment(Qt.AlignRight)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            for fld in flds:
                w = self._make(fld)
                if fld.tooltip:
                    w.setToolTip(fld.tooltip)
                form.addRow(fld.label, w)
                self._widgets[fld.name] = (fld, w)
            lay.addWidget(box)
        lay.addStretch(1)

    def extra_layout(self) -> QVBoxLayout:
        return self.layout()

    def _make(self, f: Field) -> QWidget:
        if f.kind == "int":
            w = SpinBox()
            w.setRange(int(f.minimum), int(min(f.maximum, 2**31 - 1)))
            w.setSingleStep(int(f.step))
            w.setSuffix(f.suffix)
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, n=f.name: self._set(n, int(v)))
        elif f.kind == "float":
            w = DoubleSpinBox()
            w.setDecimals(f.decimals)
            w.setRange(f.minimum, f.maximum)
            w.setSingleStep(f.step)
            w.setSuffix(f.suffix)
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, n=f.name: self._set(n, float(v)))
        elif f.kind == "bool":
            w = QCheckBox()
            w.toggled.connect(lambda v, n=f.name: self._set(n, bool(v)))
        elif f.kind == "choice":
            w = ComboBox()
            for value, label in f.choices or []:
                w.addItem(label, value)
            w.currentIndexChanged.connect(lambda _i, n=f.name, cw=w: self._set(n, cw.currentData()))
        elif f.kind in ("color", "color_alpha"):
            w = ColorButton(alpha=f.kind == "color_alpha")
            w.colorChanged.connect(lambda v, n=f.name: self._set(n, v))
        else:
            w = QLineEdit()
            w.editingFinished.connect(lambda n=f.name, lw=w: self._set(n, lw.text()))
        w.setFocusPolicy(Qt.StrongFocus)
        return w

    def bind(self, obj) -> None:
        self.obj = obj
        self.reload()

    def reload(self) -> None:
        if self.obj is None:
            return
        self._loading = True
        try:
            for name, (f, w) in self._widgets.items():
                v = getattr(self.obj, name)
                if f.kind in ("int", "float"):
                    w.setValue(v)
                elif f.kind == "bool":
                    w.setChecked(bool(v))
                elif f.kind == "choice":
                    if f.choices_fn is not None:
                        w.clear()
                        for value, label in f.choices_fn(self.obj):
                            w.addItem(label, value)
                    i = w.findData(v)
                    w.setCurrentIndex(i if i >= 0 else 0)
                elif f.kind in ("color", "color_alpha"):
                    w.set_color(v)
                else:
                    w.setText(str(v))
        finally:
            self._loading = False
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        for f, w in self._widgets.values():
            if f.enabled_if is not None and self.obj is not None:
                w.setEnabled(bool(f.enabled_if(self.obj)))

    def _set(self, name: str, value) -> None:
        if self._loading or self.obj is None or value is None:
            return
        if getattr(self.obj, name) == value:
            return
        setattr(self.obj, name, value)
        f = self._widgets[name][0]
        dependents = [g for g, _ in self._widgets.values() if g.choices_fn is not None and g is not f]
        if dependents:
            self.reload()
        else:
            self._refresh_enabled()
        self.changed.emit(name)


def scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setWidget(widget)
    return area


class ChannelPicker(QWidget):
    """Table des canaux d'un flux : case à cocher + couleur individuelle."""

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._names: list[str] = []
        self._colors: dict[int, str] = {}
        self._palette = "deep"
        self._loading = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filtrer les canaux…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        top.addWidget(self.filter_edit, 1)
        for text, slot in (("Tous", self._all), ("Aucun", self._none), ("Palette", self._repalette)):
            b = QPushButton(text)
            b.setFixedHeight(22)
            b.clicked.connect(slot)
            top.addWidget(b)
        lay.addLayout(top)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Canal", "Couleur"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setMinimumHeight(150)
        self.table.itemChanged.connect(self._item_changed)
        lay.addWidget(self.table)

    def set_channels(self, names: list[str], selected: list[int], colors: list[str], palette: str) -> None:
        self._loading = True
        self._names = list(names)
        self._palette = palette
        pal = sns.color_palette(palette, max(1, len(selected))).as_hex()
        self._colors = {}
        for k, ch in enumerate(selected):
            self._colors[int(ch)] = colors[k] if k < len(colors) and colors[k] else pal[k % len(pal)]
        self.table.setRowCount(len(names))
        chosen = {int(c) for c in selected}
        for row, name in enumerate(names):
            item = QTableWidgetItem(f"{row:3d}  {name}")
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if row in chosen else Qt.Unchecked)
            self.table.setItem(row, 0, item)
            btn = ColorButton(self._colors.get(row, "#808080"))
            btn.setEnabled(row in chosen)
            btn.colorChanged.connect(lambda c, r=row: self._color_changed(r, c))
            self.table.setCellWidget(row, 1, btn)
        self.table.resizeRowsToContents()
        self._loading = False
        self._apply_filter(self.filter_edit.text())

    def selection(self) -> tuple[list[int], list[str]]:
        chans = [r for r in range(self.table.rowCount()) if self.table.item(r, 0).checkState() == Qt.Checked]
        pal = sns.color_palette(self._palette, max(1, len(chans))).as_hex()
        colors = [self._colors.get(c) or pal[k % len(pal)] for k, c in enumerate(chans)]
        return chans, colors

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() != 0:
            return
        row = item.row()
        on = item.checkState() == Qt.Checked
        btn = self.table.cellWidget(row, 1)
        if on and row not in self._colors:
            n = len(self.selection()[0])
            pal = sns.color_palette(self._palette, max(n, 1)).as_hex()
            self._colors[row] = pal[(n - 1) % len(pal)]
            btn.set_color(self._colors[row])
        btn.setEnabled(on)
        self.changed.emit()

    def _color_changed(self, row: int, color: str) -> None:
        self._colors[row] = color
        self.changed.emit()

    def _set_all(self, state) -> None:
        self._loading = True
        for r in range(self.table.rowCount()):
            if not self.table.isRowHidden(r):
                self.table.item(r, 0).setCheckState(state)
                self.table.cellWidget(r, 1).setEnabled(state == Qt.Checked)
        self._loading = False
        self._repalette()

    def _all(self) -> None:
        self._set_all(Qt.Checked)

    def _none(self) -> None:
        self._set_all(Qt.Unchecked)

    def set_palette(self, palette: str) -> None:
        self._palette = palette
        self._repalette()

    def _repalette(self) -> None:
        chans = self.selection()[0]
        pal = sns.color_palette(self._palette, max(1, len(chans))).as_hex()
        for k, c in enumerate(chans):
            self._colors[c] = pal[k % len(pal)]
            self.table.cellWidget(c, 1).set_color(self._colors[c])
        self.changed.emit()

    def _apply_filter(self, text: str) -> None:
        text = text.strip().lower()
        for r, name in enumerate(self._names):
            self.table.setRowHidden(r, bool(text) and text not in f"{r} {name}".lower())
