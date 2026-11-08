#!/usr/bin/env python3
"""Live cpu pinning of a sim job (006 M5 round 3 evidence): every process of the job (the job shell's descendants) with
its allowed cpus (union over its threads), the cpu it is on and its cpu use over a few seconds, grouped as sim i (pgid
of sim i's `gz sim`, s<i>/pgid.txt), sim i's starter (the chain between the job shell and that pgid's leader -- the
in_sim subshell, taskset, apptainer -- plus everything under the topmost of them pinned to sim i's cpus, e.g. the image's
squashfuse_ll mount that apptainer starts beside the container: 143819's first check flagged exactly those) or client
(everything else), plus a check.

    python3 hpc/pin_snapshot.py <run dir> <label> <job shell pid> <client cpus> <sim 0 cpus> [<sim 1 cpus> ...]

Check: sim i's group only on sim i's cpus; its starter chain on sim i's or the client cpus; every other job process
(vLLM, params checks, sim_run commands: gate probes, agents, coordinators, gz CLI) on the client cpus only. A sim with
no process found -> VOID (the snapshot could not see it), never PASS. Exit 0 PASS, 1 FAIL, 2 VOID. The process tree is
built from every /proc/<pid>/stat (world-readable), so a root-owned process (a setuid apptainer starter) does not hide
its children; a process whose status / environ cannot be read is listed as unreadable. Stdlib only, host python 3.9
(hpc/cpu_split.py)."""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_split import expand  # noqa: E402

SAMPLE_S = 5.0
TICK = os.sysconf("SC_CLK_TCK")


def stats() -> dict[int, tuple[int, int, int, int, str]]:
    """Every pid: (ppid, pgid, processor, utime + stime ticks, comm)."""
    out = {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                st = (p / "stat").read_text()
            except OSError:
                continue
            f = st[st.rindex(")") + 2:].split()  # f[0] = field 3 (state)
            out[int(p.name)] = (int(f[1]), int(f[2]), int(f[36]), int(f[11]) + int(f[12]),
                                st[st.index("(") + 1:st.rindex(")")])
    return out


def under(pid: int, root: int, st: dict[int, tuple[int, int, int, int, str]]) -> list[int]:
    """The ancestors of pid up to root (exclusive of both), or [] if root is not an ancestor; [pid] marker if pid == root."""
    chain, seen = [], set()
    while pid in st and pid not in seen:
        if pid == root:
            return chain or [root]
        seen.add(pid)
        pid = st[pid][0]
        chain.append(pid)
    return []


def allowed_of(pid: int) -> set[int] | None:
    """Union of the threads' Cpus_allowed_list (a thread can re-pin itself); None if unreadable."""
    cpus: set[int] = set()
    try:
        for t in Path(f"/proc/{pid}/task").iterdir():
            for line in (t / "status").read_text().splitlines():
                if line.startswith("Cpus_allowed_list:"):
                    cpus |= set(expand(line.split()[1]))
    except OSError:
        return None
    return cpus or None


def partition(pid: int) -> str:
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return "?"
    return next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"GZ_PARTITION=")), "")


def fmt(cpus: set[int]) -> str:
    out, run = [], []
    for c in sorted(cpus):
        if run and c == run[-1] + 1:
            run.append(c)
        else:
            if run:
                out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = [c]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def main(argv: list[str]) -> int:
    run, label, root, client = Path(argv[1]), argv[2], int(argv[3]), set(expand(argv[4]))
    sims = [set(expand(a)) for a in argv[5:]]
    pg: dict[int, int] = {}
    for i in range(len(sims)):
        t = run / f"s{i}" / "pgid.txt"
        if t.exists() and t.read_text().strip().isdigit():
            pg[int(t.read_text().strip())] = i
    a = stats()
    time.sleep(SAMPLE_S)
    b = stats()
    starter: dict[int, int] = {}  # pid -> sim i: sim i's starter chain + the subtree of its topmost sim-pinned process
    kids: dict[int, list[int]] = {}
    for pid, s in b.items():
        kids.setdefault(s[0], []).append(pid)
    for leader, i in pg.items():
        chain = [x for x in under(leader, root, b) if x != root]
        for anc in chain:
            starter[anc] = i
        top = next((x for x in reversed(chain) if (allowed_of(x) or set()) and (allowed_of(x) or set()) <= sims[i]),
                   leader)
        todo = list(kids.get(top, []))
        while todo:
            x = todo.pop()
            if b[x][1] not in pg:
                starter.setdefault(x, i)
            todo += kids.get(x, [])
    print(f"== {datetime.now().astimezone().isoformat(timespec='seconds')} {label}")
    print(f"cpus: client {fmt(client)} | " + " | ".join(f"sim {i} {fmt(s)}" for i, s in enumerate(sims)))
    bad, load, count = [], {}, {}
    for pid in sorted(b):
        if pid != root and not under(pid, root, b):
            continue
        ppid, pgid, psr, ticks, comm = b[pid]
        cpu = 100.0 * (ticks - a[pid][3]) / TICK / SAMPLE_S if pid in a else 0.0
        allowed = allowed_of(pid)
        if pgid in pg:
            role, want = f"sim{pg[pgid]}", sims[pg[pgid]]
        elif pid in starter:
            role, want = f"start{starter[pid]}", sims[starter[pid]]
        else:
            role, want = "client", client
        if allowed is None:
            ok, shown = True, "unreadable"
        else:
            ok = allowed <= want or (role.startswith("start") and allowed <= client)
            shown = fmt(allowed)
        load[role] = load.get(role, 0.0) + cpu
        count[role] = count.get(role, 0) + 1
        if not ok:
            bad.append(f"{pid} {role} {comm} allowed {shown}")
        print(f"{pid} pgid {pgid} {role:7s} psr {psr:3d} cpu {cpu:6.1f} % allowed {shown:24s} {comm} {partition(pid)}")
    print(f"load (% of one cpu, {SAMPLE_S:.0f} s): " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(load.items())))
    counts = ", ".join(f"{k} {v}" for k, v in sorted(count.items()))
    missing = [i for i in range(len(sims)) if not count.get(f"sim{i}")]
    if bad:
        print(f"pin check: FAIL {'; '.join(bad)} ({counts})")
        return 1
    if missing:
        print(f"pin check: VOID no process seen for sim {missing} ({counts})")
        return 2
    print(f"pin check: PASS ({counts})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
