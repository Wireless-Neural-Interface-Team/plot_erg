"""Point d'entrée de RHS Studio."""

from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("RHS Studio")
    app.setStyle("Fusion")

    from .main_window import MainWindow

    win = MainWindow()
    paths = [a for a in argv[1:] if not a.startswith("-")]
    rhs = next((p for p in paths if p.lower().endswith(".rhs")), None)
    cfg = next((p for p in paths if p.lower().endswith(".json")), None)
    if cfg:
        win.load_config(cfg)
    if rhs:
        win.open_recording(rhs)
    elif win.rec is None:
        win.load_demo()
    win.show()
    return app.exec()
