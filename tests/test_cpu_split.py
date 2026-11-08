"""hpc/cpu_split.py (006 M5 rounds 2-3): each sim gets whole L3 groups the job holds completely, never shared; the client
(vLLM, gate probes, agents, gz CLI) gets the rest, never a sim's cpu."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hpc"))
import cpu_split as cs  # noqa: E402

PERUN = cs.expand("28-31,36-47,64-111")  # job 143808's cpus
NODES = [set(range(16 * k, 16 * k + 16)) for k in range(8)]  # gpu13: NUMA k = cpus 16k..16k+15


def l3(cpu: int) -> tuple[int, ...]:  # 8 cores per L3
    b = cpu // 8 * 8
    return tuple(range(b, b + 8))


def l3_4(cpu: int) -> tuple[int, ...]:  # PERUN gpu13 sysfs: cpu40 -> 40-43, cpu64 -> 64-67
    b = cpu // 4 * 4
    return tuple(range(b, b + 4))


def test_perun_4core_groups_16_cpus_per_sim() -> None:
    """Job 143813's cpus and gpu13's 4-core L3 groups: 16 cpus per sim (it first got 8: the cap counted groups); the
    client gets the rest."""
    two = [cs.split(i, 2, PERUN, l3_4, NODES) for i in range(2)]
    assert two[0] == list(range(64, 80)) and two[1] == list(range(80, 96))  # NUMA 4, 5: held whole, used first
    assert cs.client(2, PERUN, l3_4, NODES) == cs.expand("28-31,36-47,96-111")
    four = [set(cs.split(i, 4, PERUN, l3_4, NODES)) for i in range(4)]  # 16 groups: 4 x 4 would leave no client share
    assert all(len(x) == 12 for x in four) and set().union(*four) == set(range(64, 112))
    assert cs.client(4, PERUN, l3_4, NODES) == cs.expand("28-31,36-47")  # the partly held NUMA 1-2 cpus go last
    eight = [cs.split(i, 8, PERUN, l3_4, NODES) for i in range(8)]
    assert all(len(x) == 4 for x in eight) and len({c for x in eight for c in x}) == 32


def test_perun_80_cpus_n4_n8() -> None:
    """The measure jobs' shape (80 cpus = 20 whole 4-core groups): N = 4 -> 16 each + 16 client, N = 8 -> 8 each + 16."""
    cpus = list(range(32, 112))
    four = [set(cs.split(i, 4, cpus, l3_4, NODES)) for i in range(4)]
    assert all(len(x) == 16 for x in four) and len(cs.client(4, cpus, l3_4, NODES)) == 16
    eight = [set(cs.split(i, 8, cpus, l3_4, NODES)) for i in range(8)]
    assert all(len(x) == 8 for x in eight) and len(cs.client(8, cpus, l3_4, NODES)) == 16


def test_client_never_on_a_sim_cpu() -> None:
    for n in (1, 2, 4, 8):
        for cpus, l3f in ((PERUN, l3_4), (PERUN, l3), (list(range(40, 104)), l3_4), (list(range(0, 20)), l3)):
            sims = [set(cs.split(i, n, cpus, l3f, NODES)) for i in range(n)]
            cl = set(cs.client(n, cpus, l3f, NODES))
            assert cl and not cl & set().union(*sims), (n, cpus[:3], l3f.__name__)
            assert len(set().union(*sims)) == sum(map(len, sims))  # sims disjoint too
            assert cl | set().union(*sims) == set(cpus)


def test_too_few_cpus_for_sims_and_client_refused() -> None:
    """2 cpus, 2 sims: no share left for the client -> ValueError (cpus_init stops the job), never a silent overlap."""
    import pytest
    with pytest.raises(ValueError):
        cs.split(0, 2, [0, 1], l3, NODES)
    with pytest.raises(ValueError):
        cs.client(2, [0, 1], l3, NODES)


def test_expand() -> None:
    assert cs.expand("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]


def test_partial_groups_skipped_two_per_sim() -> None:
    s = [cs.split(i, 2, PERUN, l3, NODES) for i in range(2)]
    # 28-31 / 36-39 are partial groups: skipped; fully held NUMA nodes (64-111) before NUMA 2 (40-47, 12 of 16 held)
    assert s[0] == list(range(64, 80)) and s[1] == list(range(80, 96))


def test_four_sims_disjoint_whole_groups() -> None:
    """7 full groups (40-47, 64-111) for 4 sims: one group (8 cpus) each, disjoint, whole groups only."""
    s = [set(cs.split(i, 4, PERUN, l3, NODES)) for i in range(4)]
    assert all(len(x) == 8 for x in s) and len(set().union(*s)) == 32
    assert all(set(l3(c)) <= x for x in s for c in x)


def test_four_sims_on_nine_full_groups_get_two_each() -> None:
    """56-63 (NUMA 3, half held) + 64-127: 9 full groups -> 2 each from the wholly held nodes, 56-63 for the client."""
    cpus = list(range(56, 128))
    s = [set(cs.split(i, 4, cpus, l3, NODES)) for i in range(4)]
    assert all(len(x) == 16 for x in s) and set().union(*s) == set(range(64, 128))
    assert cs.client(4, cpus, l3, NODES) == list(range(56, 64))


def test_eight_sims_seven_groups_fall_back() -> None:
    s = [cs.split(i, 8, PERUN, l3, NODES) for i in range(8)]  # 64 cpus in 9 shares of 7: the last 8 for the client
    assert all(len(x) == 7 for x in s) and len({c for x in s for c in x}) == 56
    assert len(cs.client(8, PERUN, l3, NODES)) == 8


def test_numa_share_orders_groups() -> None:
    """A group on a NUMA node the job holds only partly comes after groups on wholly held nodes."""
    cpus = list(range(0, 4)) + list(range(16, 32))  # NUMA 0: 4 of 16 held; NUMA 1: all 16
    assert cs.split(0, 1, cpus, l3_4, NODES) == list(range(16, 32))
    assert cs.client(1, cpus, l3_4, NODES) == [0, 1, 2, 3]


def test_too_few_groups_falls_back_to_equal_split() -> None:
    cpus = list(range(0, 12))  # one full group (0-7) + a partial one: 3 equal shares, the last for the client
    assert cs.split(0, 2, cpus, l3, NODES) == list(range(0, 4))
    assert cs.split(1, 2, cpus, l3, NODES) == list(range(4, 8))
    assert cs.client(2, cpus, l3, NODES) == list(range(8, 12))


def test_runs_on_the_host_python_39() -> None:
    """PERUN host python3 is 3.9 (job 143814 died on `list[...] | None` evaluated at def time): the future import keeps
    annotations unevaluated; no 3.10+ syntax (match, parenthesized context managers) is used."""
    import ast
    for f in ("cpu_split.py", "pin_snapshot.py"):  # both run on the host
        src = (ROOT / "hpc" / f).read_text()
        assert "from __future__ import annotations" in src
        ast.parse(src, feature_version=(3, 9))
