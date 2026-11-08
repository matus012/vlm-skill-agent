#!/usr/bin/env python3
"""Pick the cpus of sim i among the job's cpus (task 006 M5 round 2): whole L3 groups only, never shared between sims.

    python3 hpc/cpu_split.py <i> <n_sims>        -> sim i's cpus as a comma list, e.g. 64,65,...,79
    python3 hpc/cpu_split.py client <n_sims>     -> the job's cpus no sim holds (vLLM, gate probes, agents, gz CLI)

PERUN gpu nodes (AMD EPYC 9535; sysfs L3 groups of 4 cores): a sim whose cpus shared L3 groups with other jobs ran
slowest in every diag (143808, 143811, 143812: 0.52-0.55 vs 0.60-0.62), and the GPU is not the limit (143812: one GPU
per sim changed nothing). So: group the job's cpus (this process's Cpus_allowed_list) by L3 (sysfs index3
shared_cpu_list), keep the groups the job holds completely, sorted, and give sim i groups [i*k, (i+1)*k),
k = min(ceil(CPUS_PER_SIM / group size), (n_full - 1) // n_sims): 16 cpus = 4 groups on gpu13 (job 143813 first ran with a
cap of 2 groups = 8 cpus per sim: the cap was meant in cpus). Groups on NUMA nodes the job holds a larger share of come
first: a sim on the job's cpus in NUMA 1-2 (gpu13, the socket shared with busy jobs of others) was the slowest in all
four diags (143808 0.55 vs 0.60-0.62, 143811 0.54 vs 0.62-0.63, 143812 0.52 / 0.67 vs 0.61 / 0.75, 143813 0.55 / 0.59
on NUMA 1-2 only). Fewer full groups than sims -> an equal contiguous split into n_sims + 1 shares (the old diag rule
plus a client share). The client (M5 round 3) gets the rest, never a sim's cpu: at least one group / share stays
spare. In 143815 the gate client ran on sim 0's 16 cpus and its per-run sim/wall fell 0.75 idle -> 0.73 (M3, light
client on sim 1) -> 0.683 (M4, ~1.4 gz CLI spawns / s) -> 0.60-0.64 (both dev shards' agents on sim 0's cpus).
Stdlib only; runs on the host, outside the image: PERUN host python3 is 3.9 (job 143814 died on a PEP 604 annotation
before the future import), so keep it 3.9-clean and smoke it on the login node (`python3 hpc/cpu_split.py 0 2`)."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

CPUS_PER_SIM = 16  # a sim uses ~5-10 cpus (143806 / 143808 top); 143808 pinned 16 each: RTF +0.05-0.10


def expand(s: str) -> list[int]:
    out: list[int] = []
    for part in s.strip().split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def allowed() -> list[int]:
    return expand(re.search(r"Cpus_allowed_list:\s*(\S+)", Path("/proc/self/status").read_text()).group(1))


def l3_of(cpu: int) -> tuple[int, ...]:
    p = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list")
    return tuple(expand(p.read_text())) if p.exists() else (cpu,)


def numa_nodes() -> list[set[int]]:
    return [set(expand(p.read_text())) for p in sorted(Path("/sys/devices/system/node").glob("node[0-9]*/cpulist"))]


def split(i: int, n: int, cpus: list[int], l3=l3_of, nodes: list[set[int]] | None = None) -> list[int]:  # noqa: ANN001
    have = set(cpus)
    nodes = numa_nodes() if nodes is None else nodes

    def held(g: tuple[int, ...]) -> float:  # share of g's NUMA node(s) this job holds
        ns = [nd for nd in nodes if nd & set(g)] or [set(g)]
        return min(len(nd & have) / len(nd) for nd in ns)
    groups = sorted({g for g in map(l3, cpus) if set(g) <= have}, key=lambda g: (-held(g), g))
    k = min(math.ceil(CPUS_PER_SIM / len(groups[0])), (len(groups) - 1) // n) if groups else 0
    if k:
        return sorted(c for g in groups[i * k:(i + 1) * k] for c in g)
    w = len(cpus) // (n + 1)
    if not w:  # no share left for the client: overlap would be silent, so refuse (sim_lib cpus_init -> job exit 2)
        raise ValueError(f"{len(cpus)} cpus cannot give {n} sims + the client a cpu each")
    return sorted(cpus)[i * w:(i + 1) * w]


def client(n: int, cpus: list[int], l3=l3_of, nodes: list[set[int]] | None = None) -> list[int]:  # noqa: ANN001
    """The job's cpus no sim holds (never empty: split keeps a group / share spare, or refuses)."""
    nodes = numa_nodes() if nodes is None else nodes
    taken = {c for i in range(n) for c in split(i, n, cpus, l3, nodes)}
    return sorted(set(cpus) - taken)


if __name__ == "__main__":
    n_sims = int(sys.argv[2])
    out = client(n_sims, allowed()) if sys.argv[1] == "client" else split(int(sys.argv[1]), n_sims, allowed())
    print(",".join(map(str, out)))
