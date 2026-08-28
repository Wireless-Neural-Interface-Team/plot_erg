"""Application-wide Qt stylesheet — high-contrast, readable palette."""

from __future__ import annotations

from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _asset_url(filename: str) -> str:
    return f"url({_ASSETS_DIR.joinpath(filename).resolve().as_posix()})"


_CB_UNCHECKED = _asset_url("checkbox_unchecked.svg")
_CB_CHECKED = _asset_url("checkbox_checked.svg")
_CB_CHECKED_HOVER = _asset_url("checkbox_checked_hover.svg")
_CB_UNCHECKED_DISABLED = _asset_url("checkbox_unchecked_disabled.svg")
_CB_CHECKED_DISABLED = _asset_url("checkbox_checked_disabled.svg")

APP_STYLESHEET = f"""
/* ---- Base ---- */
QWidget {{
    font-family: "Segoe UI", "SF Pro Text", sans-serif;
    font-size: 13px;
    color: #0f172a;
}}
QMainWindow, QWidget#centralWidget {{
    background-color: #d5dce8;
}}

/* ---- Tabs ---- */
QTabWidget::pane {{
    border: 2px solid #8896ab;
    border-radius: 10px;
    background-color: #ffffff;
    top: -1px;
    padding: 4px;
}}
QTabBar::tab {{
    background-color: #9aa8bc;
    color: #1e293b;
    border: 2px solid #8896ab;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 10px 18px;
    margin-right: 4px;
    min-width: 80px;
}}
QTabBar::tab:selected {{
    background-color: #ffffff;
    color: #0f172a;
    font-weight: 700;
    border-bottom: 3px solid #0f766e;
    margin-bottom: -1px;
}}
QTabBar::tab:hover:!selected {{
    background-color: #b4c0d0;
    color: #0f172a;
}}

/* ---- Group boxes ---- */
QGroupBox {{
    font-weight: 700;
    font-size: 13px;
    color: #0f172a;
    border: 2px solid #8896ab;
    border-radius: 10px;
    margin-top: 18px;
    padding: 16px 12px 12px 12px;
    background-color: #f8fafc;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: #0f766e;
    background-color: #f8fafc;
}}

/* ---- Form labels ---- */
QLabel {{
    color: #1e293b;
    font-weight: 600;
}}
QLabel#hintLabel {{
    color: #334155;
    font-size: 12px;
    font-weight: 400;
    padding: 4px 2px 8px 2px;
    background: transparent;
}}
QLabel#columnHeader {{
    color: #0f172a;
    font-weight: 700;
    font-size: 12px;
    padding: 4px 2px;
}}
QLabel#statusLabel {{
    color: #0f172a;
    font-weight: 600;
    padding: 6px 4px;
    background-color: #ffffff;
    border: 2px solid #8896ab;
    border-radius: 8px;
}}

/* ---- Inputs ---- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    border: 2px solid #8896ab;
    border-radius: 8px;
    padding: 7px 10px;
    background-color: #ffffff;
    color: #0f172a;
    min-height: 22px;
    selection-background-color: #99f6e4;
    selection-color: #0f172a;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 2px solid #0f766e;
    background-color: #ffffff;
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: #e2e8f0;
    color: #64748b;
    border-color: #94a3b8;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: #ffffff;
    color: #0f172a;
    border: 2px solid #8896ab;
    selection-background-color: #ccfbf1;
    selection-color: #0f172a;
}}

/* ---- Buttons ---- */
QPushButton {{
    background-color: #0f766e;
    color: #ffffff;
    border: 2px solid #0d5c56;
    border-radius: 8px;
    padding: 9px 18px;
    font-weight: 700;
    min-height: 20px;
}}
QPushButton:hover {{
    background-color: #0d9488;
    border-color: #0f766e;
}}
QPushButton:pressed {{
    background-color: #115e59;
}}
QPushButton:disabled {{
    background-color: #94a3b8;
    border-color: #64748b;
    color: #e2e8f0;
}}
QPushButton#secondaryButton {{
    background-color: #ffffff;
    color: #0f172a;
    border: 2px solid #475569;
    font-weight: 600;
}}
QPushButton#secondaryButton:hover {{
    background-color: #f1f5f9;
    border-color: #0f172a;
}}
QPushButton#dangerButton {{
    background-color: #b91c1c;
    color: #ffffff;
    border: 2px solid #7f1d1d;
    font-weight: 700;
}}
QPushButton#dangerButton:hover {{
    background-color: #dc2626;
    border-color: #991b1b;
}}

/* ---- Dialogs / message boxes (legacy fallback; app uses custom QDialog) ---- */
QMessageBox {{
    background-color: #ffffff;
    color: #0f172a;
}}
QMessageBox QLabel {{
    color: #0f172a;
    background-color: transparent;
    font-weight: 500;
    font-size: 13px;
}}
QMessageBox QPushButton {{
    background-color: #0f766e;
    color: #ffffff;
    border: 2px solid #0d5c56;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 700;
    min-width: 90px;
    min-height: 22px;
}}
QMessageBox QPushButton:hover {{
    background-color: #0d9488;
    border-color: #0f766e;
}}
QMessageBox QPushButton:pressed {{
    background-color: #115e59;
}}

/* ---- Log ---- */
QTextEdit#logView {{
    background-color: #1a2332;
    color: #e8eef5;
    border: 2px solid #475569;
    border-radius: 10px;
    font-family: Consolas, "Cascadia Mono", monospace;
    font-size: 12px;
    padding: 8px;
    selection-background-color: #0f766e;
    selection-color: #ffffff;
}}

/* ---- Progress ---- */
QProgressBar {{
    border: 2px solid #8896ab;
    border-radius: 8px;
    text-align: center;
    background-color: #e2e8f0;
    color: #0f172a;
    font-weight: 600;
    min-height: 22px;
}}
QProgressBar::chunk {{
    background-color: #0f766e;
    border-radius: 6px;
}}

/* ---- Checkboxes (white cross on teal) ---- */
QCheckBox {{
    spacing: 10px;
    color: #0f172a;
    font-weight: 500;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
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
    color: #94a3b8;
}}
QCheckBox::indicator:disabled {{
    image: {_CB_UNCHECKED_DISABLED};
}}
QCheckBox::indicator:checked:disabled {{
    image: {_CB_CHECKED_DISABLED};
}}

/* ---- Table (display tab) ---- */
QTableWidget {{
    background-color: #ffffff;
    alternate-background-color: #eef2f7;
    gridline-color: #94a3b8;
    border: 2px solid #8896ab;
    border-radius: 8px;
    color: #0f172a;
}}
QTableWidget::item {{
    padding: 6px;
}}
QTableWidget::item:disabled {{
    color: #94a3b8;
    background-color: #e2e8f0;
}}
QHeaderView::section {{
    background-color: #334155;
    color: #f8fafc;
    padding: 8px 6px;
    border: 1px solid #1e293b;
    font-weight: 700;
    font-size: 12px;
}}
QHeaderView::section:disabled {{
    background-color: #94a3b8;
    color: #e2e8f0;
    border-color: #64748b;
}}

/* ---- Files tab ---- */
QWidget#filesTab {{
    background-color: #ffffff;
}}
QScrollArea#filesScroll {{
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
}}
QWidget#filesScrollViewport,
QWidget#filesContainer {{
    background-color: #ffffff;
}}

/* ---- File rows ---- */
QWidget#fileEntryRow {{
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 4px;
    margin: 2px 0;
}}
QWidget#fileEntryRow:hover {{
    background-color: #ffffff;
    border-color: #0f766e;
}}

/* ---- Scroll areas (generic) ---- */
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: #e2e8f0;
    width: 12px;
    border-radius: 6px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #64748b;
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: #475569;
}}

/* ---- Form layout spacing helper ---- */
QFormLayout {{
    spacing: 10px;
}}
"""
