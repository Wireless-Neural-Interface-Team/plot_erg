"""Feuille de style Qt — chrome clair (Fusion)."""

from __future__ import annotations

from pathlib import Path

from gui.theme import (
    CHROME_BG,
    CHROME_BG_ALT,
    CHROME_BG_LIGHT,
    CHROME_BORDER,
    CHROME_BORDER_DARK,
    DANGER,
    DANGER_BORDER,
    PRIMARY,
    PRIMARY_HOVER,
    SCOPE_ACCENT,
    SCOPE_BG,
    SELECT_BG,
    SELECT_BORDER,
    TEXT,
    TEXT_INVERSE,
    TEXT_MUTED,
)

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _asset_url(filename: str) -> str:
    return f"url({_ASSETS_DIR.joinpath(filename).resolve().as_posix()})"


_CB_UNCHECKED = _asset_url("checkbox_unchecked.svg")
_CB_CHECKED = _asset_url("checkbox_checked.svg")
_CB_CHECKED_HOVER = _asset_url("checkbox_checked_hover.svg")
_CB_UNCHECKED_DISABLED = _asset_url("checkbox_unchecked_disabled.svg")
_CB_CHECKED_DISABLED = _asset_url("checkbox_checked_disabled.svg")

APP_STYLESHEET = f"""
/* ---- Base (chrome RHX) ---- */
QWidget {{
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 12px;
    color: {TEXT};
}}
QMainWindow {{
    background-color: {CHROME_BG_ALT};
}}
/* Poignée entre docks / zone centrale — assez large pour élargir Paramètres. */
QMainWindow::separator {{
    background-color: {CHROME_BORDER};
    width: 6px;
    height: 6px;
}}
QMainWindow::separator:hover {{
    background-color: {SCOPE_ACCENT};
}}

/* ---- Menu / toolbar ---- */
QMenuBar {{
    background-color: {CHROME_BG};
    color: {TEXT};
    border-bottom: 1px solid {CHROME_BORDER};
    padding: 1px;
}}
QMenuBar::item {{
    padding: 5px 10px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background-color: {SELECT_BG};
    color: {TEXT_INVERSE};
}}
QMenu {{
    background-color: {CHROME_BG_LIGHT};
    border: 1px solid {CHROME_BORDER_DARK};
    padding: 3px;
}}
QMenu::item {{
    padding: 6px 22px 6px 12px;
}}
QMenu::item:selected {{
    background-color: {SELECT_BG};
    color: {TEXT_INVERSE};
}}
QMenu::separator {{
    height: 1px;
    background: {CHROME_BORDER};
    margin: 3px 8px;
}}
QToolBar {{
    background-color: {CHROME_BG};
    border-bottom: 1px solid {CHROME_BORDER};
    spacing: 2px;
    padding: 3px 5px;
}}
QToolBar QToolButton {{
    padding: 5px 10px;
    border: 1px solid transparent;
    border-radius: 2px;
    color: {TEXT};
    font-weight: 600;
}}
QToolBar QToolButton:hover {{
    background-color: {CHROME_BG_LIGHT};
    border-color: {CHROME_BORDER};
}}
QToolBar QToolButton:disabled {{
    color: #8a8a8a;
}}
QToolBar::separator {{
    width: 1px;
    background: {CHROME_BORDER};
    margin: 4px 5px;
}}
QStatusBar {{
    background-color: {CHROME_BG};
    border-top: 1px solid {CHROME_BORDER};
}}
QStatusBar QLabel {{
    padding: 1px 8px;
    color: {TEXT_MUTED};
}}

/* ---- Docks ---- */
QDockWidget {{
    font-weight: 700;
    font-size: 12px;
    color: {TEXT};
}}
QDockWidget::title {{
    background-color: {CHROME_BG};
    color: {TEXT};
    padding: 5px 8px;
    text-align: left;
    border-bottom: 1px solid {CHROME_BORDER};
}}
QDockWidget > QWidget {{
    background-color: {CHROME_BG};
}}
QDockWidget QTabWidget::pane {{
    border: none;
    background-color: {CHROME_BG};
}}
QDockWidget QTabBar::tab {{
    background-color: {CHROME_BG_ALT};
    border: 1px solid {CHROME_BORDER};
    border-bottom: none;
    padding: 5px 12px;
    margin-right: 1px;
    min-width: 80px;
}}
QDockWidget QTabBar::tab:selected {{
    background-color: {CHROME_BG_LIGHT};
    border-bottom: 2px solid {SCOPE_ACCENT};
    font-weight: 700;
}}

/* ---- Zone centrale (viewport scope) ---- */
QTabWidget#centralViews::pane {{
    border: 1px solid {CHROME_BORDER_DARK};
    background-color: {SCOPE_BG};
}}

/* ---- Control panel (bas, style RHX) ---- */
QFrame#controlPanel {{
    background-color: {CHROME_BG};
    border-top: 1px solid {CHROME_BORDER_DARK};
}}
QLabel#controlChannel {{
    font-weight: 700;
    font-size: 13px;
    color: {TEXT};
    background-color: {CHROME_BG_LIGHT};
    border: 1px solid {CHROME_BORDER};
    padding: 4px 10px;
    min-width: 120px;
}}
QLabel#viewModeBadge {{
    font-weight: 800;
    font-size: 11px;
    letter-spacing: 0.4px;
    color: {TEXT};
    background-color: {CHROME_BG_LIGHT};
    border: 1px solid {CHROME_BORDER};
    border-radius: 2px;
    padding: 4px 10px;
    min-width: 72px;
}}
QLabel#viewModeBadge[mode="montage"] {{
    background-color: #dbeafe;
    color: #1e3a8a;
    border-color: #93c5fd;
}}
QLabel#viewModeBadge[mode="preview"] {{
    background-color: {CHROME_BG_LIGHT};
    color: {TEXT};
}}
QGroupBox#collapsibleGroup {{
    font-weight: 700;
}}
QGroupBox#collapsibleGroup::title {{
    color: {TEXT};
}}
QPushButton#filterWide, QPushButton#filterLow, QPushButton#filterHigh, QPushButton#filterSpk {{
    min-width: 58px;
    padding: 6px 10px;
    font-weight: 800;
    letter-spacing: 0.5px;
    border-radius: 2px;
}}
QPushButton#filterWide {{
    background-color: #ffffff;
    color: #a16207;
    border: 1px solid {CHROME_BORDER};
}}
QPushButton#filterWide:checked {{
    background-color: #fef08a;
    color: #713f12;
    border: 1px solid #ca8a04;
}}
QPushButton#filterLow {{
    background-color: #ffffff;
    color: #0369a1;
    border: 1px solid {CHROME_BORDER};
}}
QPushButton#filterLow:checked {{
    background-color: #bae6fd;
    color: #0c4a6e;
    border: 1px solid #0284c7;
}}
QPushButton#filterHigh {{
    background-color: #ffffff;
    color: #15803d;
    border: 1px solid {CHROME_BORDER};
}}
QPushButton#filterHigh:checked {{
    background-color: #bbf7d0;
    color: #14532d;
    border: 1px solid #16a34a;
}}
QPushButton#filterSpk {{
    background-color: #ffffff;
    color: #c2410c;
    border: 1px solid {CHROME_BORDER};
}}
QPushButton#filterSpk:checked {{
    background-color: #fed7aa;
    color: #7c2d12;
    border: 1px solid #ea580c;
}}

/* ---- Group boxes ---- */
QGroupBox {{
    font-weight: 700;
    color: {TEXT};
    border: 1px solid {CHROME_BORDER};
    border-radius: 2px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    background-color: {CHROME_BG_LIGHT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 5px;
    color: {TEXT_MUTED};
    background-color: {CHROME_BG_LIGHT};
}}
QGroupBox#meaMapBox {{
    background-color: {SCOPE_BG};
    border: 1px solid {CHROME_BORDER_DARK};
    margin-top: 10px;
    min-height: 240px;
}}
QGroupBox#meaMapBox::title {{
    color: {TEXT};
    font-weight: 800;
    background-color: {SCOPE_BG};
}}

/* ---- Labels ---- */
QLabel {{
    color: {TEXT};
}}
QLabel#sectionTitle {{
    font-weight: 700;
    font-size: 13px;
    color: {TEXT};
}}
QLabel#hintLabel {{
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 400;
}}
QLabel#warningLabel {{
    color: #6b3a00;
    background-color: #f5e6b8;
    border: 1px solid #d4b45a;
    padding: 6px 8px;
}}
QLabel#workflowHint {{
    color: {TEXT};
    background-color: {SCOPE_BG};
    border: none;
    padding: 24px 32px;
    font-size: 13px;
    font-weight: 500;
}}
QLabel#panelTitle {{
    font-weight: 700;
    color: {TEXT};
    font-size: 12px;
    background: transparent;
}}
QLabel#panelStatus {{
    font-size: 10px;
    color: {TEXT_MUTED};
}}

/* ---- Panel cards ---- */
QFrame#panelCard {{
    background-color: {SCOPE_BG};
    border: 1px solid {CHROME_BORDER};
    border-radius: 2px;
}}
QToolButton#panelToolButton {{
    color: {TEXT_MUTED};
    font-size: 12px;
    padding: 2px 4px;
}}
QToolButton#panelToolButton:hover {{
    background-color: {CHROME_BG};
    color: {TEXT};
}}
QToolButton#panelToolButton:checked {{
    background-color: #dbeafe;
    color: {PRIMARY};
}}

/* ---- Inputs ---- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    border: 1px solid {CHROME_BORDER};
    border-radius: 2px;
    padding: 3px 6px;
    background-color: #ffffff;
    color: {TEXT};
    min-height: 18px;
    selection-background-color: {SELECT_BG};
    selection-color: {TEXT_INVERSE};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {SELECT_BG};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {CHROME_BG_ALT};
    color: #8a8a8a;
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background-color: #ffffff;
    border: 1px solid {CHROME_BORDER_DARK};
    selection-background-color: {SELECT_BG};
    selection-color: {TEXT_INVERSE};
}}

/* ---- Buttons ---- */
QPushButton {{
    background-color: {CHROME_BG_LIGHT};
    color: {TEXT};
    border: 1px solid {CHROME_BORDER};
    border-radius: 2px;
    padding: 5px 11px;
    font-weight: 600;
    min-height: 18px;
}}
QPushButton:hover {{
    background-color: #f0f0f0;
    border-color: {CHROME_BORDER_DARK};
}}
QPushButton:pressed {{
    background-color: {CHROME_BG_ALT};
}}
QPushButton:disabled {{
    color: #8a8a8a;
    border-color: #b0b0b0;
}}
QPushButton#primaryButton {{
    background-color: {PRIMARY};
    color: {TEXT_INVERSE};
    border: 1px solid {SELECT_BORDER};
    font-weight: 700;
}}
QPushButton#primaryButton:hover {{
    background-color: {PRIMARY_HOVER};
}}
QPushButton#primaryButton:disabled {{
    background-color: #8a8a8a;
    border-color: #6e6e6e;
}}
QPushButton#dangerButton {{
    background-color: {CHROME_BG_LIGHT};
    color: {DANGER};
    border: 1px solid {DANGER_BORDER};
    font-weight: 700;
}}
QPushButton#dangerButton:hover {{
    background-color: #f5d0d0;
}}
QPushButton#secondaryButton {{
    background-color: {CHROME_BG_LIGHT};
    color: {PRIMARY};
    border: 1px solid {SELECT_BG};
}}
QPushButton#secondaryButton:hover {{
    background-color: #dceaf3;
}}

/* ---- Message boxes ---- */
QMessageBox {{
    background-color: {CHROME_BG_LIGHT};
}}

/* ---- Log ---- */
QPlainTextEdit, QTextEdit#logView {{
    background-color: #ffffff;
    color: {TEXT};
    border: 1px solid {CHROME_BORDER};
    font-family: Consolas, "Cascadia Mono", monospace;
    padding: 5px;
    selection-background-color: {SELECT_BG};
    selection-color: {TEXT_INVERSE};
}}

/* ---- Progress ---- */
QProgressBar {{
    border: 1px solid {CHROME_BORDER};
    border-radius: 2px;
    text-align: center;
    background-color: {CHROME_BG_LIGHT};
    color: {TEXT};
    font-weight: 600;
    min-height: 14px;
    max-height: 16px;
}}
QProgressBar::chunk {{
    background-color: {SELECT_BG};
}}

/* ---- Checkboxes ---- */
QCheckBox {{
    spacing: 7px;
    color: {TEXT};
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: none;
    background: transparent;
}}
QCheckBox::indicator:unchecked {{
    image: {_CB_UNCHECKED};
}}
QCheckBox::indicator:checked {{
    image: {_CB_CHECKED};
}}
QCheckBox::indicator:checked:hover {{
    image: {_CB_CHECKED_HOVER};
}}
QCheckBox:disabled {{
    color: #8a8a8a;
}}
QCheckBox::indicator:disabled {{
    image: {_CB_UNCHECKED_DISABLED};
}}
QCheckBox::indicator:checked:disabled {{
    image: {_CB_CHECKED_DISABLED};
}}

/* ---- Tables / lists ---- */
QTableWidget, QListWidget, QTreeWidget {{
    background-color: #ffffff;
    alternate-background-color: #efefef;
    gridline-color: {CHROME_BORDER};
    border: 1px solid {CHROME_BORDER};
    border-radius: 0px;
    color: {TEXT};
}}
QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
    background-color: {SELECT_BG};
    color: {TEXT_INVERSE};
}}
QHeaderView::section {{
    background-color: {CHROME_BG_ALT};
    color: {TEXT};
    padding: 5px 6px;
    border: 1px solid {CHROME_BORDER};
    font-weight: 700;
}}

/* ---- Splitters / scroll ---- */
QSplitter::handle {{
    background-color: {CHROME_BORDER};
}}
QSplitter::handle:hover {{
    background-color: {SCOPE_ACCENT};
}}
QSplitter::handle:horizontal {{
    width: 6px;
}}
QSplitter::handle:vertical {{
    height: 6px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: {CHROME_BG_ALT};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #8a8a8a;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: #5a5a5a;
}}
QScrollBar:horizontal {{
    background: {CHROME_BG_ALT};
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: #8a8a8a;
    min-width: 20px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
"""
