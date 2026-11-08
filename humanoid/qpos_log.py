"""--qpos-log writer (task 013): robot qpos jsonl for the showcase video. No ROS / MuJoCo imports (unit-testable).
Line 1: {"model", "nq", "qpos_names"} (nq = the qpos length: 7 for the free joint "root", 1 per hinge); then
{"t_sim", "qpos"} at ~hz sim-time. Flushed every FLUSH_S wall, so a killed
node leaves a usable file (a last line may be cut: readers skip an unparsable final line)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import IO, Sequence

MODEL = "unitree_robots/g1/g1_29dof.xml"
FLUSH_S = 1.0


class QposLog:
    def __init__(self, path: Path, names: Sequence[str], hz: float = 30.0) -> None:
        self.f: IO[str] = Path(path).open("w")
        self.period, self.next, self.flushed = 1.0 / hz, 0.0, time.monotonic()
        self.nq = sum(7 if n == "root" else 1 for n in names)  # 013 local smoke: len(names) (30) != len(qpos) (36)
        self.f.write(json.dumps({"model": MODEL, "nq": self.nq, "qpos_names": list(names)}) + "\n")
        self.f.flush()

    def write(self, t_sim: float, qpos: Sequence[float]) -> None:
        if t_sim < self.next - 1e-9:
            return
        assert len(qpos) == self.nq, f"qpos has {len(qpos)} values, header says {self.nq}"
        self.next = max(self.next + self.period, t_sim + 1e-9)  # one line per call at most; no catch-up burst
        self.f.write(json.dumps({"t_sim": t_sim, "qpos": [float(v) for v in qpos]}) + "\n")
        if time.monotonic() - self.flushed >= FLUSH_S:
            self.f.flush()
            self.flushed = time.monotonic()

    def close(self) -> None:
        self.f.close()
