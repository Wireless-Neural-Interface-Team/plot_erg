"""Widgets de formulaire partagés (spinboxes, lignes d’axe)."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from view_config import AxisLimits, TextFace


def configure_narrow_form(form: QFormLayout) -> QFormLayout:
    """Formulaire de barre latérale : libellé au-dessus, champ pleine largeur."""
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    form.setFormAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    form.setHorizontalSpacing(6)
    form.setVerticalSpacing(4)
    return form


class FitWidthScrollArea(QScrollArea):
    """Scroll vertical calé sur la largeur du viewport — sans feedback sur les docks.

    ``setWidgetResizable(True)`` + politique ``Ignored`` sur le contenu suffisent
    à empêcher le débordement horizontal. On ne touche **jamais** à
    ``setMaximumWidth`` / ``setFixedWidth`` pendant ``resizeEvent`` : cela
    déclenche ``updateGeometry`` et fait sauter ``QMainWindow::resizeDocks``
    (le panneau Paramètres se retrouve écrasé au minimum).

    ``sizeHint`` suit la largeur courante une fois affiché, pour que le layout
    des docks ne « aspire » pas vers une largeur préférée fantôme.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        # Ignored : le parent (dock / splitter) impose la largeur, pas le contenu.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:  # noqa: D102
        if self.isVisible() and self.width() > 0:
            return QSize(self.width(), max(120, self.height()))
        return QSize(280, 400)

    def minimumSizeHint(self) -> QSize:  # noqa: D102
        return QSize(0, 80)

    def setWidget(self, widget: QWidget | None) -> None:  # noqa: D102
        if widget is not None:
            widget.setMinimumWidth(0)
            widget.setMaximumWidth(16777215)
            widget.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
        super().setWidget(widget)


def make_double_spin(
    minimum: float,
    maximum: float,
    value: float,
    *,
    decimals: int = 3,
    step: float = 0.01,
    suffix: str = "",
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(float(value))
    box.setKeyboardTracking(False)
    box.setMinimumWidth(0)
    box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    if suffix:
        box.setSuffix(suffix)
    return box


def make_int_spin(
    minimum: int,
    maximum: int,
    value: int,
    *,
    step: int = 1,
    suffix: str = "",
) -> QSpinBox:
    box = QSpinBox()
    box.setRange(int(minimum), int(maximum))
    box.setSingleStep(int(step))
    box.setValue(int(value))
    box.setKeyboardTracking(False)
    box.setMinimumWidth(0)
    box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    if suffix:
        box.setSuffix(suffix)
    return box


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Spinbox dont le sizeHint ignore la plage ±1e6."""

    def __init__(self, *, hint_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hint_text = hint_text
        # Expanding + min bas : se comprime dans un dock étroit sans forcer la largeur.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        height = super().sizeHint().height()
        width = self.fontMetrics().horizontalAdvance(self._hint_text) + 28
        return QSize(max(56, width), height)

    def minimumSizeHint(self) -> QSize:
        height = super().minimumSizeHint().height()
        return QSize(40, height)


def make_axis_spin(
    value: float,
    *,
    decimals: int,
    step: float,
    suffix: str,
) -> QDoubleSpinBox:
    hint = f"-000.{'0' * decimals}{suffix}"
    box = CompactDoubleSpinBox(hint_text=hint)
    box.setDecimals(decimals)
    box.setRange(-1e6, 1e6)
    box.setSingleStep(step)
    box.setValue(float(value))
    box.setKeyboardTracking(False)
    if suffix:
        box.setSuffix(suffix)
    return box


class FontFaceRow(QWidget):
    """Cases Gras / Italique / Souligné pour un rôle typographique."""

    changed = Signal()

    def __init__(
        self, face: TextFace | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        face = face or TextFace()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
        self._bold = QCheckBox("Gras")
        self._italic = QCheckBox("Italique")
        self._underline = QCheckBox("Souligné")
        bold_font = self._bold.font()
        bold_font.setBold(True)
        self._bold.setFont(bold_font)
        italic_font = self._italic.font()
        italic_font.setItalic(True)
        self._italic.setFont(italic_font)
        underline_font = self._underline.font()
        underline_font.setUnderline(True)
        self._underline.setFont(underline_font)
        self._bold.setChecked(bool(face.bold))
        self._italic.setChecked(bool(face.italic))
        self._underline.setChecked(bool(face.underline))
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.addWidget(self._bold)
        root.addWidget(self._italic)
        root.addWidget(self._underline)
        root.addStretch(1)
        for box in (self._bold, self._italic, self._underline):
            box.toggled.connect(lambda _c: self.changed.emit())

    def value(self) -> TextFace:
        return TextFace(
            bold=self._bold.isChecked(),
            italic=self._italic.isChecked(),
            underline=self._underline.isChecked(),
        )

    def set_value(self, face: TextFace) -> None:
        self._bold.setChecked(bool(face.bold))
        self._italic.setChecked(bool(face.italic))
        self._underline.setChecked(bool(face.underline))


class FontSizeFaceRow(QWidget):
    """Taille de police (pt) + Gras / Italique / Souligné."""

    changed = Signal()

    def __init__(
        self,
        minimum: float,
        maximum: float,
        value: float,
        face: TextFace | None = None,
        *,
        decimals: int = 1,
        step: float = 0.5,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
        self._size = make_double_spin(
            minimum, maximum, value, decimals=decimals, step=step, suffix=" pt"
        )
        self._face = FontFaceRow(face, self)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.addWidget(self._size)
        root.addWidget(self._face)
        self._size.valueChanged.connect(lambda _v: self.changed.emit())
        self._face.changed.connect(self.changed.emit)

    def size_value(self) -> float:
        return float(self._size.value())

    def face_value(self) -> TextFace:
        return self._face.value()

    def set_size_value(self, size: float) -> None:
        self._size.setValue(float(size))

    def set_face_value(self, face: TextFace) -> None:
        self._face.set_value(face)


class AxisLimitRow(QWidget):
    """Case « Manuel » + spinboxes min/max (empilés pour rester dans un dock étroit)."""

    changed = Signal()

    def __init__(
        self,
        limits: AxisLimits,
        *,
        unit: str = " µV",
        decimals: int = 2,
        step: float = 10.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
        self._enabled = QCheckBox("Manuel")
        self._enabled.setChecked(bool(limits.enabled))
        self._min = make_axis_spin(
            limits.minimum, decimals=decimals, step=step, suffix=unit
        )
        self._max = make_axis_spin(
            limits.maximum, decimals=decimals, step=step, suffix=unit
        )
        # Deux lignes : la case seule, puis min/max — évite de couper le bord droit.
        range_row = QWidget(self)
        range_layout = QHBoxLayout(range_row)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(4)
        range_layout.addWidget(self._min, 1)
        range_layout.addWidget(QLabel("à"), 0)
        range_layout.addWidget(self._max, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.addWidget(self._enabled)
        root.addWidget(range_row)
        self._enabled.toggled.connect(self._on_enabled)
        self._min.valueChanged.connect(lambda _v: self.changed.emit())
        self._max.valueChanged.connect(lambda _v: self.changed.emit())
        self._on_enabled(self._enabled.isChecked(), emit=False)

    def _on_enabled(self, checked: bool, *, emit: bool = True) -> None:
        self._min.setEnabled(bool(checked))
        self._max.setEnabled(bool(checked))
        if emit:
            self.changed.emit()

    def value(self) -> AxisLimits:
        return AxisLimits(
            enabled=self._enabled.isChecked(),
            minimum=float(self._min.value()),
            maximum=float(self._max.value()),
        )

    def set_value(self, limits: AxisLimits) -> None:
        self._enabled.setChecked(bool(limits.enabled))
        self._min.setValue(float(limits.minimum))
        self._max.setValue(float(limits.maximum))


def _swatch_ink(hex_color: str) -> str:
    """Texte lisible sur le fond du nuancier."""
    text = str(hex_color or "").strip().lstrip("#")
    if len(text) != 6:
        return "#111827"
    try:
        red = int(text[0:2], 16)
        green = int(text[2:4], 16)
        blue = int(text[4:6], 16)
    except ValueError:
        return "#111827"
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0
    return "#111827" if luminance > 0.62 else "#f8fafc"


class ColorSwatchButton(QPushButton):
    """Bouton nuancier : clic → sélecteur de couleur, émet ``changed``."""

    changed = Signal()

    def __init__(
        self,
        color: str,
        *,
        dialog_title: str = "Choisir une couleur",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dialog_title = dialog_title
        self._color = str(color or "#000000")
        self.setFixedHeight(26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._apply()
        self.clicked.connect(self._pick)

    def color_value(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = str(color or self._color)
        self._apply()

    def _apply(self) -> None:
        ink = _swatch_ink(self._color)
        self.setText(self._color.upper())
        self.setStyleSheet(
            "QPushButton {"
            f" background: {self._color}; color: {ink};"
            " border: 1px solid #6b7280; border-radius: 4px; padding: 2px 8px;"
            "}"
            "QPushButton:hover { border: 1px solid #111827; }"
        )

    def _pick(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._color), self, self._dialog_title)
        if not chosen.isValid():
            return
        self.set_color(chosen.name())
        self.changed.emit()
