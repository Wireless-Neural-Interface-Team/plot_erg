"""Police / libellés / ticks (hors tick-face painter — voir host)."""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontInfo, QFontMetrics, QPainter, QPen,
    QTextCharFormat, QTextCursor,
)
from PySide6.QtWidgets import QGraphicsSimpleTextItem

_LABEL_FAMILY = "Segoe UI"

_LABEL_FAMILY_FALLBACKS: tuple[str, ...] = (
    "Segoe UI",
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Sans Serif",
)

def _resolved_label_family() -> str:
    """Première famille disponible (Segoe UI peut manquer hors Windows)."""
    for family in _LABEL_FAMILY_FALLBACKS:
        info = QFontInfo(QFont(family))
        resolved = str(info.family() or "").strip()
        if resolved and info.exactMatch():
            return family
        if resolved and family.lower() in resolved.lower():
            return family
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            family = str(app.font().family() or "").strip()
            if family:
                return family
    except Exception:
        pass
    return _LABEL_FAMILY

def _html_text(text: str) -> str:
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )

def _face_qfont(
    size_pt: float, *, bold: bool = False, italic: bool = False, underline: bool = False
) -> QFont:
    """Police explicite : sans famille, Qt ignore gras et italique dans le HTML."""
    font = QFont(_resolved_label_family())
    font.setStyleHint(QFont.StyleHint.SansSerif)
    try:
        font.setStyleStrategy(
            QFont.StyleStrategy.PreferAntialias
            | QFont.StyleStrategy.PreferQuality
        )
    except Exception:
        pass
    font.setPointSizeF(max(5.0, float(size_pt)))
    font.setWeight(QFont.Weight.Bold if bold else QFont.Weight.Normal)
    font.setBold(bool(bold))
    font.setItalic(bool(italic))
    font.setUnderline(bool(underline))
    if bold or italic:
        info = QFontInfo(font)
        if (bold and not info.bold()) or (italic and not info.italic()):
            for family in _LABEL_FAMILY_FALLBACKS:
                font.setFamily(family)
                font.setBold(bool(bold))
                font.setItalic(bool(italic))
                info = QFontInfo(font)
                if ((not bold) or info.bold()) and ((not italic) or info.italic()):
                    break
    return font

def _stamp_text_face(text_item: Any, font: QFont, color: QColor | None = None) -> None:
    """Appliquer gras / italique / souligné aux caractères déjà posés.

    Un ``span`` CSS sans variante de police ne change pas les glyphes : seul
    le soulignement (trait) apparaît. Le format de caractère force les trois.
    """
    document = getattr(text_item, "document", None)
    if not callable(document):
        return
    doc = document()
    if doc is None:
        return
    try:
        doc.setDefaultFont(font)
        text_item.setFont(font)
    except Exception:
        pass
    cursor = QTextCursor(doc)
    cursor.select(QTextCursor.SelectionType.Document)
    fmt = QTextCharFormat()
    fmt.setFontFamily(font.family() or _LABEL_FAMILY)
    if font.pointSizeF() > 0:
        fmt.setFontPointSize(float(font.pointSizeF()))
    fmt.setFontWeight(QFont.Weight.Bold if font.bold() else QFont.Weight.Normal)
    fmt.setFontItalic(bool(font.italic()))
    fmt.setFontUnderline(bool(font.underline()))
    if color is not None:
        fmt.setForeground(QBrush(color))
    cursor.setCharFormat(fmt)

def _axis_owning_label(item: Any) -> Any | None:
    """AxisItem dont ``label`` est ce QGraphicsTextItem, sinon None."""
    parent_fn = getattr(item, "parentItem", None)
    parent = parent_fn() if callable(parent_fn) else None
    if parent is None:
        return None
    if getattr(parent, "label", None) is item and hasattr(parent, "labelStyle"):
        return parent
    return None

def _css_pt(size: str) -> float | None:
    raw = str(size or "").strip().lower().replace(" ", "")
    if raw.endswith("pt"):
        raw = raw[:-2]
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value

def _render_label_item(item: pg.LabelItem, text: str) -> None:
    """Réécrire le HTML du LabelItem sans toucher taille, police ni style.

    ``LabelItem.setText`` ignore le souligné : on le remet dans le span.
    """
    raw = str(text)
    shown = raw if raw != "" else " "
    opts = item.opts
    item.text = raw
    parts: list[str] = []
    color = opts.get("color")
    if color is None:
        try:
            color = pg.getConfigOption("foreground")
        except Exception:
            color = "#000000"
    qcolor: QColor | None = None
    try:
        qcolor = pg.mkColor(color)
        parts.append("color: " + qcolor.name(QColor.NameFormat.HexArgb))
    except Exception:
        qcolor = None
    family = str(opts.get("family") or _LABEL_FAMILY)
    parts.append('font-family: "%s"' % family)
    size = opts.get("size")
    if size:
        parts.append("font-size: " + str(size))
    bold = bool(opts.get("bold"))
    italic = bool(opts.get("italic"))
    underline = bool(opts.get("underline"))
    parts.append("font-weight: " + ("700" if bold else "400"))
    parts.append("font-style: " + ("italic" if italic else "normal"))
    if underline:
        parts.append("text-decoration: underline")
    body = _html_text(shown)
    if underline:
        body = "<u>%s</u>" % body
    if italic:
        body = "<i>%s</i>" % body
    if bold:
        body = "<b>%s</b>" % body
    try:
        item.item.setHtml("<span style='%s'>%s</span>" % ("; ".join(parts), body))
        pt = _css_pt(str(size or "")) or 9.0
        _stamp_text_face(
            item.item,
            _face_qfont(pt, bold=bold, italic=italic, underline=underline),
            qcolor if isinstance(qcolor, QColor) else None,
        )
        item.updateMin()
        item.resizeEvent(None)
        item.updateGeometry()
    except Exception:
        pass

def _ensure_axis_label_stamper(axis: Any) -> None:
    """Réappliquer le style après chaque ``_updateLabel`` (préfixe SI, resize)."""
    if getattr(axis, "_erg_label_face", False):
        return
    original = axis._updateLabel

    def _update() -> None:
        original()
        font = getattr(axis, "_erg_label_font", None)
        label = getattr(axis, "label", None)
        if font is None or label is None:
            return
        _stamp_text_face(label, font, getattr(axis, "_erg_label_color", None))
        try:
            axis._adjustSize()
        except Exception:
            pass

    axis._updateLabel = _update
    axis._erg_label_face = True

def _apply_axis_label_style(
    axis: Any,
    *,
    bold: bool,
    italic: bool,
    underline: bool,
) -> None:
    """Mettre le style dans ``labelStyle`` pour que ``_updateLabel`` le garde."""
    style = dict(getattr(axis, "labelStyle", {}) or {})
    if "size" in style and "font-size" not in style:
        style["font-size"] = style.pop("size")
    style["font-family"] = '"%s"' % _LABEL_FAMILY
    style["font-weight"] = "700" if bold else "400"
    style["font-style"] = "italic" if italic else "normal"
    if underline:
        style["text-decoration"] = "underline"
    else:
        style.pop("text-decoration", None)
    axis.labelStyle = style
    pt = _css_pt(str(style.get("font-size") or style.get("size") or "")) or 9.0
    axis._erg_label_font = _face_qfont(
        pt, bold=bool(bold), italic=bool(italic), underline=bool(underline)
    )
    color = style.get("color")
    try:
        axis._erg_label_color = pg.mkColor(color) if color else None
    except Exception:
        axis._erg_label_color = None
    _ensure_axis_label_stamper(axis)
    try:
        axis._updateLabel()
    except Exception:
        pass

def _editor_font(item: Any) -> QFont:
    """Police affichée pendant l’édition : celle du libellé, pas celle du formulaire."""
    font = QFont()
    bold = italic = underline = False
    size = ""
    family = ""
    if isinstance(item, pg.LabelItem):
        opts = getattr(item, "opts", {}) or {}
        bold = bool(opts.get("bold"))
        italic = bool(opts.get("italic"))
        underline = bool(opts.get("underline"))
        size = str(opts.get("size") or "")
        family = str(opts.get("family") or "")
    else:
        axis = _axis_owning_label(item)
        if axis is None:
            current = item.font() if callable(getattr(item, "font", None)) else None
            return current if isinstance(current, QFont) else font
        style = dict(getattr(axis, "labelStyle", {}) or {})
        weight = str(style.get("font-weight", "")).strip().lower()
        bold = weight in {"bold", "700", "600", "bolder"}
        italic = str(style.get("font-style", "")).strip().lower() == "italic"
        underline = "underline" in str(style.get("text-decoration", "")).lower()
        size = str(style.get("font-size") or style.get("size") or "")
        family = str(style.get("font-family") or "")
    if family:
        font.setFamily(family)
    pt = _css_pt(size)
    if pt is not None:
        font.setPointSizeF(pt)
    font.setBold(bold)
    font.setItalic(italic)
    font.setUnderline(underline)
    return font

def _editor_stylesheet(font: QFont) -> str:
    weight = "700" if font.bold() else "400"
    slant = "italic" if font.italic() else "normal"
    deco = "underline" if font.underline() else "none"
    family = font.family() or "Segoe UI"
    pt = float(font.pointSizeF())
    size = f"{pt:g}pt" if pt > 0 else "9pt"
    return (
        "QLineEdit { background: #fffbeb; border: 1px solid #d97706; "
        "padding: 1px 4px; "
        f"font-family: \"{family}\"; font-size: {size}; "
        f"font-weight: {weight}; font-style: {slant}; "
        f"text-decoration: {deco}; }}"
    )

def _apply_label_face(
    item: Any,
    *,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
) -> None:
    """Gras / italique / souligné, sans changer la taille ni la famille déjà posées."""
    if item is None:
        return
    axis = _axis_owning_label(item)
    if axis is not None:
        _apply_axis_label_style(
            axis, bold=bool(bold), italic=bool(italic), underline=bool(underline)
        )
        return
    if isinstance(item, pg.LabelItem):
        opts = item.opts
        opts["bold"] = bool(bold)
        opts["italic"] = bool(italic)
        opts["underline"] = bool(underline)
        _render_label_item(item, str(getattr(item, "text", "") or ""))
        return
    for method, value in (
        ("setBold", bold),
        ("setItalic", italic),
        ("setUnderline", underline),
    ):
        fn = getattr(item, method, None)
        if callable(fn):
            try:
                fn(bool(value))
            except Exception:
                pass
    try:
        font = item.font() if callable(getattr(item, "font", None)) else None
        if font is not None:
            font.setBold(bool(bold))
            font.setItalic(bool(italic))
            font.setUnderline(bool(underline))
            item.setFont(font)
    except Exception:
        pass

def _tick_qfont(
    size_pt: float, *, bold: bool = False, italic: bool = False, underline: bool = False
) -> QFont:
    """Police de ticks. Le QPicture de l'axe ne rejoue ni gras, ni italique, ni souligné."""
    return _face_qfont(size_pt, bold=bold, italic=italic, underline=underline)

def _scale_label_item(label: str, font: QFont, brush: QBrush) -> QGraphicsSimpleTextItem:
    """Libellé d'échelle : même police que les ticks, souligné à part."""
    text = QGraphicsSimpleTextItem(label)
    text.setFont(font)
    text.setBrush(brush)
    if font.underline():
        metrics = QFontMetrics(font)
        advance = float(metrics.horizontalAdvance(label))
        y = float(metrics.ascent()) + float(max(1, int(metrics.underlinePos())))
        line = QGraphicsLineItem(0.0, y, advance, y, text)
        pen = QPen(brush.color())
        pen.setWidthF(max(1.0, float(metrics.lineWidth())))
        pen.setCosmetic(True)
        line.setPen(pen)
    return text

def _scope_colors() -> dict[str, str]:
    try:
        from gui.theme import (
            SCOPE_ACCENT,
            SCOPE_AXES,
            SCOPE_BG,
            SCOPE_FG,
            SCOPE_GRID,
            SCOPE_SPINE,
            SCOPE_TITLE,
        )

        return {
            "bg": SCOPE_BG,
            "axes": SCOPE_AXES,
            "fg": SCOPE_FG,
            "grid": SCOPE_GRID,
            "spine": SCOPE_SPINE,
            "title": SCOPE_TITLE,
            "accent": SCOPE_ACCENT,
        }
    except Exception:
        return {
            "bg": "#ffffff",
            "axes": "#ffffff",
            "fg": "#1f2937",
            "grid": "#e5e7eb",
            "spine": "#9ca3af",
            "title": "#111827",
            "accent": "#2563eb",
        }

