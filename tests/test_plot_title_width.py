"""Un titre long ne doit pas empêcher le graphique de suivre le widget."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

import numpy as np
import pyqtgraph as pg

from gui.widgets.plot_host.host import _relax_label_width


def test_long_title_does_not_keep_plot_wider_than_widget() -> None:
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    window.resize(700, 420)
    layout = QVBoxLayout(window)
    glw = pg.GraphicsLayoutWidget()
    glw.ci.setMinimumWidth(0)
    layout.addWidget(glw)
    plot = glw.addPlot()
    plot.setLabel("left", "Potentiel (uV)")
    plot.showAxis("right", True)
    plot.getAxis("right").setWidth(58)
    plot.setTitle("A-028 — Analyse — brut (moyenne) — canal tres long")
    plot.plot(np.linspace(-0.1, 0.4, 400), np.sin(np.linspace(0, 8, 400)))
    _relax_label_width(plot.titleLabel)
    plot.setMinimumWidth(0)
    window.show()
    for _ in range(6):
        app.processEvents()

    wide_vb = float(plot.getViewBox().geometry().width())
    assert wide_vb > 300

    window.setFixedWidth(280)
    for _ in range(8):
        app.processEvents()

    vb = float(plot.getViewBox().geometry().width())
    ci = float(glw.ci.geometry().width())
    assert abs(ci - float(glw.width())) <= 4
    assert vb < wide_vb - 80
    assert vb < float(glw.width())
    window.close()
