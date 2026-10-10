"""Background workers and redraw scheduling for the viewer.

Heavy work (reading, filtering, averaging, spike detection) runs on worker
threads that stream progress and log lines back to the GUI. Drawing stays on the
GUI thread but is funnelled through :class:`Debouncer`, so dragging a spin box
coalesces into a single redraw instead of one per keystroke.
"""

from __future__ import annotations

import contextlib
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from PySide6.QtCore import QObject, QTimer, QThread, Signal

# Pause between channel jobs so the Qt event loop can reclaim the GIL.
_GUI_YIELD_S = 0.008


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

        self.setPriority(QThread.Priority.LowPriority)
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
                    # Let the GUI process events between recordings.
                    time.sleep(_GUI_YIELD_S)
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
    # Warm HP/LP rows on the worker thread (never on the Qt UI thread).
    need_filters: bool = True


@dataclass
class _EnsureJob:
    """Single-channel unit of work for the priority scheduler."""

    recording: Any
    config: Any
    channel: int
    label: str
    force: bool = False
    need_means: bool = True
    need_rms: bool = True
    need_spikes: bool = True
    need_overlay: bool = True
    need_filters: bool = True

    def key(self) -> tuple[int, int]:
        return (id(self.recording), int(self.channel))

    def merge_needs(self, other: "_EnsureJob") -> "_EnsureJob":
        return _EnsureJob(
            recording=self.recording,
            config=other.config,
            channel=self.channel,
            label=self.label,
            force=self.force or other.force,
            need_means=self.need_means or other.need_means,
            need_rms=self.need_rms or other.need_rms,
            need_spikes=self.need_spikes or other.need_spikes,
            need_overlay=self.need_overlay or other.need_overlay,
            need_filters=self.need_filters or other.need_filters,
        )

    @property
    def products_requested(self) -> bool:
        return bool(self.need_means or self.need_rms or self.need_spikes or self.need_overlay)


def _jobs_from_requests(requests: Sequence[ChannelEnsureRequest]) -> list[_EnsureJob]:
    jobs: list[_EnsureJob] = []
    for request in requests:
        for channel in request.channels:
            jobs.append(
                _EnsureJob(
                    recording=request.recording,
                    config=request.config,
                    channel=int(channel),
                    label=request.label,
                    force=bool(request.force),
                    need_means=bool(request.need_means),
                    need_rms=bool(request.need_rms),
                    need_spikes=bool(request.need_spikes),
                    need_overlay=bool(request.need_overlay),
                    need_filters=bool(getattr(request, "need_filters", True)),
                )
            )
    return jobs


def _warm_filters(recording: Any, channel: int) -> None:
    """Run LazyFilterBank.prefetch for one channel (worker thread only)."""
    source = getattr(recording, "source", None)
    if source is None:
        return
    for bank_name in ("highpass", "lowpass"):
        bank = getattr(source, bank_name, None)
        prefetch = getattr(bank, "prefetch", None)
        if callable(prefetch):
            prefetch([int(channel)])


class ChannelEnsureWorker(QThread):
    """Computes channels with a live priority queue.

    High-priority jobs (the channel currently viewed) are always taken before
    background prefetch. New jobs can be submitted while the worker is running;
    the next channel after the one in flight will respect the updated order.
    """

    progressed = Signal(object)
    logged = Signal(str)
    succeeded = Signal(object, list)  # recording, computed channel indices
    failed = Signal(str)
    finished_all = Signal(bool)
    queue_changed = Signal(int, int)  # high remaining, low remaining

    def __init__(
        self,
        requests: Sequence[ChannelEnsureRequest] | None = None,
        parent: QObject | None = None,
        *,
        priority: bool = True,
    ) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._high: list[_EnsureJob] = []
        self._low: list[_EnsureJob] = []
        self._cancel = threading.Event()
        self._wake = threading.Event()
        # Once True, submit() refuses new work so the GUI can start a fresh worker.
        self._shutting_down = False
        if requests:
            self.submit(requests, priority=priority)

    def request_stop(self) -> None:
        self._cancel.set()
        self._wake.set()

    def pending_counts(self) -> tuple[int, int]:
        with self._lock:
            return len(self._high), len(self._low)

    def submit(
        self,
        requests: Sequence[ChannelEnsureRequest],
        *,
        priority: bool = False,
    ) -> bool:
        """Enqueue work; ``priority=True`` jumps ahead of background prefetch.

        Returns False when the worker is shutting down (caller must start a new one).
        """
        jobs = _jobs_from_requests(requests)
        if not jobs:
            return True
        with self._lock:
            if self._shutting_down:
                return False
            if priority:
                prepared: list[_EnsureJob] = []
                for job in jobs:
                    prepared.append(self._take_and_merge(job))
                self._high = prepared + self._high
            else:
                for job in jobs:
                    self._low.append(self._take_and_merge(job))
            high, low = len(self._high), len(self._low)
        self._wake.set()
        self.queue_changed.emit(high, low)
        return True

    def _take_and_merge(self, job: _EnsureJob) -> _EnsureJob:
        key = job.key()
        existing: _EnsureJob | None = None
        for bucket in (self._high, self._low):
            for index, current in enumerate(bucket):
                if current.key() == key:
                    existing = current
                    del bucket[index]
                    break
            if existing is not None:
                break
        return existing.merge_needs(job) if existing is not None else job

    def _pop_next(self) -> _EnsureJob | None:
        with self._lock:
            if self._high:
                return self._high.pop(0)
            if self._low:
                return self._low.pop(0)
            return None

    def _has_pending(self) -> bool:
        with self._lock:
            return bool(self._high or self._low)

    def run(self) -> None:  # noqa: D102 - QThread entry point
        import core
        from dataset_builder import ProgressEvent, ensure_channels

        self.setPriority(QThread.Priority.LowPriority)
        stream = _SignalStream(self.logged.emit)
        ok = True
        idle_rounds = 0
        done = 0
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            with core.analysis_cancel_scope(self._cancel):
                while not self._cancel.is_set():
                    job = self._pop_next()
                    if job is None:
                        idle_rounds += 1
                        # Brief wait so a prioritize() right after the last job is seen.
                        if idle_rounds >= 4:
                            with self._lock:
                                if self._high or self._low:
                                    idle_rounds = 0
                                    continue
                                # Refuse further submit() before emitting finished_all.
                                self._shutting_down = True
                            break
                        self._wake.wait(0.05)
                        self._wake.clear()
                        continue
                    idle_rounds = 0
                    high, low = self.pending_counts()
                    # Prefetch only → yield more aggressively to keep the UI snappy.
                    if high == 0:
                        self.setPriority(QThread.Priority.IdlePriority)
                    else:
                        self.setPriority(QThread.Priority.LowPriority)
                    self.queue_changed.emit(high, low)
                    remaining_after = high + low
                    total = max(1, done + 1 + remaining_after)

                    def _progress(event: object, *, _done: int = done, _total: int = total) -> None:
                        # Fraction globale : évite le reset 0 % à chaque canal (clignotement).
                        stage = float(getattr(event, "stage_fraction", 0.0) or 0.0)
                        overall = (_done + max(0.0, min(1.0, stage))) / _total
                        self.progressed.emit(
                            ProgressEvent(
                                recording=getattr(event, "recording", job.label),
                                stage=getattr(event, "stage", "channel"),
                                stage_label=getattr(
                                    event, "stage_label", "Computing selected channels"
                                ),
                                stage_fraction=stage,
                                overall_fraction=overall,
                                message=getattr(event, "message", "") or "",
                                elapsed_s=float(getattr(event, "elapsed_s", 0.0) or 0.0),
                                cached=bool(getattr(event, "cached", False)),
                            )
                        )

                    try:
                        if job.need_filters:
                            _warm_filters(job.recording, job.channel)
                        computed: list[int] = []
                        if job.products_requested:
                            computed = ensure_channels(
                                job.recording,
                                [job.channel],
                                job.config,
                                progress=_progress,
                                force=job.force,
                                need_means=job.need_means,
                                need_rms=job.need_rms,
                                need_spikes=job.need_spikes,
                                need_overlay=job.need_overlay,
                            )
                        else:
                            computed = [job.channel]
                    except InterruptedError:
                        ok = False
                        self.logged.emit("Channel computation interrupted.")
                        break
                    except Exception as exc:
                        ok = False
                        self.logged.emit(traceback.format_exc(limit=4))
                        self.failed.emit(str(exc))
                    else:
                        done += 1
                        if computed and job.products_requested:
                            name = (
                                job.recording.channel_names[job.channel]
                                if 0 <= job.channel < len(job.recording.channel_names)
                                else str(job.channel)
                            )
                            self.logged.emit(f"{job.label}: computed {name}")
                        self.succeeded.emit(job.recording, computed or [job.channel])
                    # Yield the GIL so clicks / redraws are not starved.
                    time.sleep(_GUI_YIELD_S if high > 0 else _GUI_YIELD_S * 2)
            stream.flush()
        high, low = self.pending_counts()
        self.queue_changed.emit(high, low)
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

        self.setPriority(QThread.Priority.LowPriority)
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
