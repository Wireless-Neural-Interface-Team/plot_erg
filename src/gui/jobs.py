"""Background workers and redraw scheduling for the viewer.

Heavy work (reading, filtering, averaging, spike detection) runs on worker
threads that stream progress and log lines back to the GUI. Drawing stays on the
GUI thread but is funnelled through :class:`Debouncer`, so dragging a spin box
coalesces into a single redraw instead of one per keystroke.
"""

from __future__ import annotations

import contextlib
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from PySide6.QtCore import QObject, QTimer, QThread, Signal


class _SignalStream:
    """File-like object forwarding complete lines to a Qt signal."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._emit(line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._emit(self._buffer.rstrip())
        self._buffer = ""


@dataclass
class BuildRequest:
    """One recording to process, as configured in the Recordings panel."""

    config: Any  # AnalysisConfig
    label: str
    style: Any  # RecordingStyle
    row_id: int


class BuildWorker(QThread):
    """Processes recordings one by one, reporting progress and timings."""

    progressed = Signal(object)  # dataset_builder.ProgressEvent
    logged = Signal(str)
    recording_ready = Signal(int, object, object)  # row_id, ProcessedRecording, BuildReport
    recording_failed = Signal(int, str)
    finished_all = Signal(bool)  # True when nothing failed

    def __init__(
        self,
        requests: Sequence[BuildRequest],
        cache_root: Path | None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._requests = list(requests)
        self._cache_root = cache_root
        self._cancel = threading.Event()

    def request_stop(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        import core
        from dataset_builder import build_recording

        stream = _SignalStream(self.logged.emit)
        ok = True
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            with core.analysis_cancel_scope(self._cancel):
                for request in self._requests:
                    if self._cancel.is_set():
                        ok = False
                        break
                    try:
                        recording, report = build_recording(
                            request.config,
                            cache_root=self._cache_root,
                            progress=self.progressed.emit,
                            label=request.label,
                            style=request.style,
                        )
                    except InterruptedError:
                        ok = False
                        self.logged.emit("Processing interrupted.")
                        break
                    except Exception as exc:  # surfaced in the UI, not swallowed
                        ok = False
                        self.logged.emit(traceback.format_exc(limit=4))
                        self.recording_failed.emit(request.row_id, str(exc))
                        continue
                    self.logged.emit(report.summary())
                    self.recording_ready.emit(request.row_id, recording, report)
            stream.flush()
        self.finished_all.emit(ok and not self._cancel.is_set())


@dataclass
class ChannelEnsureRequest:
    """Compute one or more channels for an already prepared recording."""

    recording: Any  # ProcessedRecording
    config: Any  # AnalysisConfig
    channels: list[int]
    label: str
    force: bool = False
    need_means: bool = True
    need_rms: bool = True
    need_spikes: bool = True
    need_overlay: bool = True


class ChannelEnsureWorker(QThread):
    """Computes the channels requested by the GUI, without touching the others."""

    progressed = Signal(object)
    logged = Signal(str)
    succeeded = Signal(object, list)  # recording, computed channel indices
    failed = Signal(str)
    finished_all = Signal(bool)

    def __init__(
        self,
        requests: Sequence[ChannelEnsureRequest],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._requests = list(requests)
        self._cancel = threading.Event()

    def request_stop(self) -> None:
        self._cancel.set()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        import core
        from dataset_builder import ensure_channels

        stream = _SignalStream(self.logged.emit)
        ok = True
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            with core.analysis_cancel_scope(self._cancel):
                for request in self._requests:
                    if self._cancel.is_set():
                        ok = False
                        break
                    try:
                        computed = ensure_channels(
                            request.recording,
                            request.channels,
                            request.config,
                            progress=self.progressed.emit,
                            force=request.force,
                            need_means=request.need_means,
                            need_rms=request.need_rms,
                            need_spikes=request.need_spikes,
                            need_overlay=request.need_overlay,
                        )
                    except InterruptedError:
                        ok = False
                        self.logged.emit("Channel computation interrupted.")
                        break
                    except Exception as exc:
                        ok = False
                        self.logged.emit(traceback.format_exc(limit=4))
                        self.failed.emit(str(exc))
                        continue
                    if computed:
                        names = [
                            request.recording.channel_names[ch]
                            if 0 <= ch < len(request.recording.channel_names)
                            else str(ch)
                            for ch in computed
                        ]
                        self.logged.emit(
                            f"{request.label}: computed {len(computed)} channel(s) — "
                            + ", ".join(names[:8])
                            + ("…" if len(names) > 8 else "")
                        )
                    self.succeeded.emit(request.recording, computed)
            stream.flush()
        self.finished_all.emit(ok and not self._cancel.is_set())


class TaskWorker(QThread):
    """Runs one callable off the GUI thread, streaming its stdout to the log."""

    logged = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, task: Callable[[], Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._task = task
        self._cancel = threading.Event()

    def request_stop(self) -> None:
        self._cancel.set()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        import core

        stream = _SignalStream(self.logged.emit)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            with core.analysis_cancel_scope(self._cancel):
                try:
                    result = self._task()
                except InterruptedError:
                    stream.flush()
                    self.failed.emit("Task interrupted.")
                    return
                except Exception as exc:
                    self.logged.emit(traceback.format_exc(limit=4))
                    stream.flush()
                    self.failed.emit(str(exc))
                    return
            stream.flush()
        self.succeeded.emit(result)


class Debouncer(QObject):
    """Collapses bursts of change notifications into a single callback."""

    triggered = Signal()

    def __init__(self, interval_ms: int = 120, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(max(0, int(interval_ms)))
        self._timer.timeout.connect(self.triggered.emit)

    def set_interval(self, interval_ms: int) -> None:
        self._timer.setInterval(max(0, int(interval_ms)))

    def request(self) -> None:
        self._timer.start()

    def cancel(self) -> None:
        self._timer.stop()

    def flush(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.triggered.emit()
