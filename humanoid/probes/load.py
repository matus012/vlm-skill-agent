"""Load sampler (no bar): CPU of the g1 node process tree and the whole machine, GPU util / memory, and whether a
Gazebo server is running, every 2 s until SIGTERM. Writes sideH-load-<tag>.json.
  load.py <node_pid> <tag>
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from humanoid.probes.common import write_verdict

TICK = os.sysconf("SC_CLK_TCK")
NCPU = os.cpu_count() or 1


def proc_ticks(pid: int) -> int:
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(f[11]) + int(f[12])  # utime + stime


def sys_ticks() -> tuple[int, int]:
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]  # total, idle + iowait


def gpu() -> tuple[float, float]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split(",")
    return float(out[0]), float(out[1])


def main() -> None:
    pid, tag = int(sys.argv[1]), sys.argv[2]
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    samples = []
    p0, (s0, i0), t0 = proc_ticks(pid), sys_ticks(), time.time()
    while not stop[0]:
        time.sleep(2.0)
        try:
            p1 = proc_ticks(pid)
        except FileNotFoundError:
            break
        s1, i1 = sys_ticks()
        t1 = time.time()
        g_util, g_mem = gpu()
        samples.append({"node_cpu_pct_of_one_core": 100.0 * (p1 - p0) / TICK / (t1 - t0),
                        "machine_cpu_pct": 100.0 * (1 - (i1 - i0) / (s1 - s0)),
                        "gpu_util_pct": g_util, "gpu_mem_mib": g_mem})
        p0, s0, i0, t0 = p1, s1, i1, t1
    gz = subprocess.run(["pgrep", "-af", "gz sim"], capture_output=True, text=True).stdout.strip().splitlines()

    def stat(k: str) -> dict:
        v = [s[k] for s in samples]
        return {"mean": sum(v) / len(v), "max": max(v)}

    write_verdict(f"load-{tag}", {"measure": "load while the g1 node runs (no bar)", "cpus": NCPU,
                                  "n_samples": len(samples),
                                  **{k: stat(k) for k in samples[0]}, "gazebo_running_at_end": gz})


if __name__ == "__main__":
    main()
