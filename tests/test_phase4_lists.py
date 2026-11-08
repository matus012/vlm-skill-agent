"""Task 007 run lists (tasks/phase4_counted.yaml, tasks/phase4_dev.yaml) obey the selection rule in their headers."""
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probes.phase3_run import load_runs  # noqa: E402
from probes.phase4_gates import NEED, PER_TYPE, load_list, start_type, world_of  # noqa: E402

COUNTED, DEV = ROOT / "tasks" / "phase4_counted.yaml", ROOT / "tasks" / "phase4_dev.yaml"
P3 = [ROOT / "tasks" / "phase3_counted.yaml", ROOT / "tasks" / "phase3_dev.yaml"]
FIND_005 = {"find-toolbox", "find-cabinet", "find-beds-where", "find-chairs-plain"}
COND_FIRST_RIGHT = {"cond-toolbox", "cond-chairs-reception"}  # target in if_not_in


def triple(e: dict) -> tuple:
    return e["task_id"], e["start_place"], e["seed"], world_of(e)


def test_counted_rules() -> None:
    es = load_list(COUNTED)
    kinds = Counter(e["expected"]["kind"] for e in es)
    assert kinds == {"check_state": PER_TYPE, "count": PER_TYPE, "conditional": PER_TYPE, "find": PER_TYPE}
    assert NEED == 7
    for kind in kinds:
        ks = [e for e in es if e["expected"]["kind"] == kind]
        st = Counter(start_type(e, e["expected"]["answer_room"]) for e in ks)
        assert st["answer_room"] <= 2 and st["hall"] >= 1 and st["other_room"] >= 1, (kind, st)
        assert len({(e["task_id"], e["start_place"], world_of(e)) for e in ks}) == PER_TYPE, kind
    cs = [e for e in es if e["expected"]["kind"] == "check_state"]
    assert {(world_of(e), e["expected"]["answer"]) for e in cs} == {(w, a) for w in ("closed", "cabinet_open")
                                                                       for a in ("yes", "no")}
    assert all(e["expected"]["answer_room"] == "storage" for e in cs)
    assert all("world" not in e for e in es if e["expected"]["kind"] != "check_state")
    cond = [e["task_id"] for e in es if e["expected"]["kind"] == "conditional"]
    assert 0 < sum(t in COND_FIRST_RIGHT for t in cond) < len(cond)  # both branches
    assert {e["task_id"] for e in es if e["expected"]["kind"] == "find"} == FIND_005
    assert {e["start_place"] for e in es} == {"hall", "ward", "office", "reception", "storage"}


def test_seeds_by_rule_and_disjoint() -> None:
    c, d = load_runs(COUNTED), load_runs(DEV)
    assert [e["seed"] for e in c] == random.Random("115-007-counted").sample(range(1, 2**31 - 1), len(c))
    assert [e["seed"] for e in d] == random.Random("115-007-dev").sample(range(1, 2**31 - 1), len(d))
    old = {e["seed"] for p in P3 for e in load_runs(p)}
    assert not ({e["seed"] for e in c} & old) and not ({e["seed"] for e in d} & old)
    assert not ({e["seed"] for e in c} & {e["seed"] for e in d})
    assert not ({triple(e) for e in c} & {triple(e) for e in d})
    assert not ({e["run_id"] for e in c} & {e["run_id"] for e in d})


def test_dev_covers_each_new_type_and_both_worlds() -> None:
    d = load_list(DEV)
    kinds = Counter(e["expected"]["kind"] for e in d)
    assert all(2 <= kinds[k] <= 3 for k in ("check_state", "count", "conditional")) and kinds["find"] == 2
    assert {world_of(e) for e in d} == {"closed", "cabinet_open"}
    assert {e["expected"]["answer"] for e in d if e["expected"]["kind"] == "check_state"} == {"yes", "no"}


def test_phase4_counted_list_is_counted_and_pinned() -> None:
    """The runner applies the counted-run rules (once, ledger, refusals) to the Phase 4 list, by path and by sha256."""
    from probes.phase3_run import COUNTED_LISTS, is_counted, sha256_file
    p4 = ROOT / "tasks" / "phase4_counted.yaml"
    assert is_counted(p4) and COUNTED_LISTS[p4] == sha256_file(p4)
    assert not is_counted(ROOT / "tasks" / "phase4_dev.yaml")
