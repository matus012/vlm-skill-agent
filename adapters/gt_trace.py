"""Evaluator-side ground-truth pose trace for the demo video (probes/demo_recorder.py, brief T10). Never seen by agent/.

    trace = GtTrace(gt_pose, run_dir / "gt_trace.jsonl").start()
    try: ...the run...
    finally: trace.stop()

A daemon thread calls `gt_pose()` (SimTruth.gt_pose: one `gz topic -e -n 1` per call) every PERIOD_S (2 Hz) and appends
{"wall_t", "x", "y", "yaw", "call_s"} lines. wall_t = time.time() at the call's midpoint, the clock of steps.jsonl
wall_t0 / wall_t1; the recorder maps it to sim time per step. A recorder failure never reaches the run: every error
is counted (last line {"stop": ..., "samples": n, "errors": k, "last_error": str|null}), start() / stop() never raise,
and stop() waits at most STOP_WAIT_S (a hung gz call is left to the daemon thread).
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

PERIOD_S = 0.5  # 2 Hz: each sample spawns a gz CLI (~0.16 s CPU measured offline); Nav2 vx_max 0.5 m/s -> <= 25 cm apart
STOP_WAIT_S = 1.0


class GtTrace:
    def __init__(self, gt_pose: Callable[[], tuple[float, float, float]], path: Path) -> None:
        self.gt_pose, self.path = gt_pose, Path(path)
        self.samples = self.errors = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._f: IO[str] | None = None

    def start(self) -> GtTrace:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = self.path.open("a", buffering=1)
            self._thread = threading.Thread(target=self._loop, name="gt_trace", daemon=True)
            self._thread.start()
        except Exception as e:  # noqa: BLE001 (production path: the run must never see a recorder error)
            self.errors, self.last_error = self.errors + 1, repr(e)
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                x, y, yaw = self.gt_pose()
                t1 = time.time()
                self._f.write(json.dumps({"wall_t": (t0 + t1) / 2, "x": x, "y": y, "yaw": yaw,
                                          "call_s": round(t1 - t0, 4)}) + "\n")
                self.samples += 1
            except Exception as e:  # noqa: BLE001
                self.errors, self.last_error = self.errors + 1, repr(e)
            self._stop.wait(max(0.0, PERIOD_S - (time.time() - t0)))

    def stop(self) -> dict[str, int | str | None]:
        summary: dict[str, int | str | None] = {}
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(STOP_WAIT_S)
            summary = {"stop": time.time(), "samples": self.samples, "errors": self.errors,
                       "last_error": self.last_error}
            if self._f is not None:
                self._f.write(json.dumps(summary) + "\n")
                self._f.close()
        except Exception:  # noqa: BLE001
            pass
        return summary
