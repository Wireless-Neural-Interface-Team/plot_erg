"""Background analysis thread with cooperative cancellation."""

from __future__ import annotations

import contextlib
import io
import threading
from typing import Callable


def create_analysis_thread_class():
    from PySide6.QtCore import QThread, Signal

    class AnalysisThread(QThread):
        finished_ok = Signal(str)
        finished_err = Signal(str)
        finished_interrupted = Signal(str)

        def __init__(self, target: Callable[[], None]) -> None:
            super().__init__()
            self._target = target
            self._cancel_event = threading.Event()

        def request_stop(self) -> None:
            self._cancel_event.set()

        def run(self) -> None:
            from core import analysis_cancel_scope

            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    with analysis_cancel_scope(self._cancel_event):
                        self._target()
                self.finished_ok.emit(buf.getvalue().strip())
            except InterruptedError as exc:
                self.finished_interrupted.emit(str(exc))
            except Exception as exc:
                self.finished_err.emit(str(exc))

    return AnalysisThread
