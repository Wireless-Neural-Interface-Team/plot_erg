"""Progression compacte, temps de chargement, journal du pipeline."""

from __future__ import annotations

from typing import Any, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_MAX_LOG_BLOCKS = 4000


class StatusPanel(QWidget):
    """Journal et temps — masqué par défaut ; la barre d’état suffit au quotidien."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._headline = QLabel("En attente.")
        self._headline.setObjectName("sectionTitle")
        self._headline.setWordWrap(True)

        self._stage_label = QLabel("")
        self._stage_label.setObjectName("hintLabel")
        self._stage_label.setWordWrap(True)

        self._overall = QProgressBar()
        self._overall.setRange(0, 1000)
        self._overall.setValue(0)
        self._overall.setFormat("%p%")
        self._overall.setMaximumHeight(16)

        self._stage = QProgressBar()
        self._stage.setRange(0, 1000)
        self._stage.setValue(0)
        self._stage.setFormat("%p% étape")
        self._stage.setMaximumHeight(16)

        self.timings = QTableWidget(0, 3, self)
        self.timings.setHorizontalHeaderLabels(["Enregistrement", "Étape", "Durée"])
        self.timings.verticalHeader().setVisible(False)
        self.timings.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.timings.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.timings.setAlternatingRowColors(True)
        header = self.timings.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(_MAX_LOG_BLOCKS)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        log_font = QFont("Consolas")
        log_font.setStyleHint(QFont.StyleHint.Monospace)
        log_font.setPointSize(9)
        self.log.setFont(log_font)

        clear_log = QPushButton("Effacer le journal")
        clear_log.clicked.connect(self.log.clear)
        clear_timings = QPushButton("Effacer les temps")
        clear_timings.clicked.connect(lambda: self.timings.setRowCount(0))
        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        buttons.addWidget(clear_timings)
        buttons.addWidget(clear_log)
        buttons.addStretch(1)

        timings_holder = QWidget()
        timings_layout = QVBoxLayout(timings_holder)
        timings_layout.setContentsMargins(0, 0, 0, 0)
        timings_layout.setSpacing(2)
        timings_title = QLabel("Temps")
        timings_title.setObjectName("sectionTitle")
        timings_layout.addWidget(timings_title)
        timings_layout.addWidget(self.timings, 1)

        log_holder = QWidget()
        log_layout = QVBoxLayout(log_holder)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(2)
        log_title = QLabel("Journal")
        log_title.setObjectName("sectionTitle")
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(timings_holder)
        splitter.addWidget(log_holder)
        splitter.setSizes([280, 520])

        progress_row = QHBoxLayout()
        progress_row.setSpacing(8)
        progress_row.addWidget(self._overall, 1)
        progress_row.addWidget(self._stage, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(self._headline)
        layout.addWidget(self._stage_label)
        layout.addLayout(progress_row)
        layout.addLayout(buttons)
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------ states

    def set_idle(self, message: str = "En attente.") -> None:
        self._headline.setText(message)
        self._stage_label.setText("")
        self._overall.setValue(0)
        self._stage.setValue(0)

    def set_progress_complete(self) -> None:
        """Afficher 100 % une fois le chargement / l’affichage prêt."""
        self._overall.setValue(1000)
        self._stage.setValue(1000)

    def set_headline(self, message: str) -> None:
        self._headline.setText(message)

    def append_log(self, text: str) -> None:
        for line in str(text).splitlines():
            if line.strip():
                self.log.appendPlainText(line)

    def on_progress(self, event: Any) -> None:
        """Consommer un :class:`dataset_builder.ProgressEvent`."""
        cached = " (cache)" if getattr(event, "cached", False) else ""
        self._headline.setText(f"{event.recording} — {event.stage_label}{cached}")
        message = getattr(event, "message", "") or ""
        elapsed = float(getattr(event, "elapsed_s", 0.0))
        self._stage_label.setText(
            f"{message}  ·  {elapsed:.1f} s" if message else f"{elapsed:.1f} s"
        )
        self._overall.setValue(int(max(0.0, min(1.0, float(event.overall_fraction))) * 1000))
        self._stage.setValue(int(max(0.0, min(1.0, float(event.stage_fraction))) * 1000))

    def progress_fractions(self) -> tuple[float, float]:
        return self._overall.value() / 1000.0, self._stage.value() / 1000.0

    def headline_text(self) -> str:
        return self._headline.text()

    def add_report(self, report: Any) -> None:
        """Ajouter chaque temps d’étape d’un :class:`dataset_builder.BuildReport`."""
        name = str(getattr(report, "recording", "?"))
        for timing in getattr(report, "timings", ()):  # StageTiming
            self._add_row(
                name,
                f"{timing.label}{' (cache)' if timing.cached else ''}",
                f"{timing.seconds:.2f} s",
                cached=bool(timing.cached),
            )
        total = float(getattr(report, "total_s", 0.0))
        reused = bool(getattr(report, "reused_bundle", False))
        self._add_row(
            name,
            "Total" + (" (cache)" if reused else ""),
            f"{total:.2f} s",
            bold=True,
        )
        self.timings.scrollToBottom()

    def add_timing(self, name: str, label: str, seconds: float) -> None:
        self._add_row(name, label, f"{seconds:.2f} s")
        self.timings.scrollToBottom()

    def _add_row(
        self, name: str, label: str, value: str, *, cached: bool = False, bold: bool = False
    ) -> None:
        row = self.timings.rowCount()
        self.timings.insertRow(row)
        for column, text in enumerate((name, label, value)):
            item = QTableWidgetItem(text)
            if cached:
                item.setForeground(QColor("#16a34a"))
            if bold:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.timings.setItem(row, column, item)

    def set_render_summary(self, panels: int, seconds: float, *, tab: str = "") -> None:
        if panels <= 0:
            return
        where = f" « {tab} »" if tab else ""
        self._stage_label.setText(
            f"{panels} panneau(x){where} · {seconds * 1000:.0f} ms"
        )

    def set_cache_summary(self, lines: Sequence[str]) -> None:
        if lines:
            self.append_log("\n".join(lines))
