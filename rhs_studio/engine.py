"""Calcul en arrière-plan (QThreadPool) avec cache de détection des triggers."""

from __future__ import annotations

import copy
import os
import threading
import traceback

from PySide6.QtCore import QObject, QThreadPool, Signal

from . import dsp
from .compute import compute_block
from .config import BlockConfig, FilterSettings, TriggerSettings
from .recording import Recording


class ComputeEngine(QObject):
    resultReady = Signal(str, int, object)
    failed = Signal(str, int, str)
    triggersReady = Signal(object)
    triggerStarted = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max(2, min(6, (os.cpu_count() or 4) - 1)))
        self.rec: Recording | None = None
        self.filters = FilterSettings()
        self.trigger = TriggerSettings()
        self._gen: dict[str, int] = {}
        self._lock = threading.Lock()
        self._trig_cache: dict[tuple, dsp.TriggerResult] = {}

    def set_recording(self, rec: Recording | None) -> None:
        self.rec = rec
        with self._lock:
            self._trig_cache.clear()

    def _key(self, rec: Recording, trig: TriggerSettings) -> tuple:
        return (id(rec),) + trig.detection_key()

    def cached_triggers(self) -> dsp.TriggerResult | None:
        if self.rec is None:
            return None
        return self._trig_cache.get(self._key(self.rec, self.trigger))

    def _triggers(self, rec: Recording, trig: TriggerSettings) -> dsp.TriggerResult:
        key = self._key(rec, trig)
        with self._lock:
            res = self._trig_cache.get(key)
            if res is not None:
                return res
            self.triggerStarted.emit()
            res = dsp.detect_triggers(rec, trig)
            if len(self._trig_cache) > 6:
                self._trig_cache.clear()
            self._trig_cache[key] = res
        self.triggersReady.emit(res)
        return res

    def detect_now(self) -> None:
        rec, trig = self.rec, copy.deepcopy(self.trigger)
        if rec is None:
            return

        def job():
            try:
                self._triggers(rec, trig)
            except Exception:
                self.failed.emit("", 0, traceback.format_exc())

        self.pool.start(job)

    def submit(self, cfg: BlockConfig) -> int:
        gen = self._gen.get(cfg.id, 0) + 1
        self._gen[cfg.id] = gen
        rec = self.rec
        cfg = copy.deepcopy(cfg)
        flt = copy.deepcopy(self.filters)
        trig = copy.deepcopy(self.trigger)

        def job():
            if self._gen.get(cfg.id) != gen:
                return
            try:
                res = self._triggers(rec, trig) if rec is not None and trig.enabled else None
                if self._gen.get(cfg.id) != gen:
                    return
                data = compute_block(rec, cfg, flt, trig, res)
                self.resultReady.emit(cfg.id, gen, data)
            except Exception as exc:  # affiché dans le bloc
                self.failed.emit(cfg.id, gen, f"{exc}\n{traceback.format_exc()}")

        self.pool.start(job)
        return gen

    def is_current(self, block_id: str, gen: int) -> bool:
        return self._gen.get(block_id) == gen

    def forget(self, block_id: str) -> None:
        self._gen.pop(block_id, None)

    def shutdown(self) -> None:
        self._gen.clear()
        self.pool.clear()
        self.pool.waitForDone(3000)
