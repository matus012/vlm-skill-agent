"""Task 009 run lists (tasks/phase6_counted.yaml, tasks/phase6_dev.yaml) obey the slot plan and rules of
probes/pick_phase6.py and INTERFACES section 11. True answers come from truth_for(layout_seed, world)
(phase5_gates.load_list), never from the lists. Env P6_LISTS_DIR checks the two lists in another dir (e.g. a trial
pick_phase6.py output); the list tests skip while the files do not exist. The picker tests run without the real P3 file."""
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import tasks as at  # noqa: E402
from agent.truth import run_truth  # noqa: E402
from probes import pick_phase6 as pick  # noqa: E402
from probes.phase3_run import load_runs, split_runs  # noqa: E402
from probes.phase4_gates import start_type, world_of  # noqa: E402
from probes.phase5_gates import load_list, ok_seeds  # noqa: E402

LISTS = Path(os.environ.get("P6_LISTS_DIR", ROOT / "tasks"))
COUNTED, DEV = LISTS / "phase6_counted.yaml", LISTS / "phase6_dev.yaml"
EARLIER = [ROOT / "tasks" / f"phase{p}_{k}.yaml" for p in (3, 4, 5) for k in ("counted", "dev")]
GUARD_WORLDS = ("closed", "cabinet_open")
MAX_ANSWER_STARTS = 2  # task 009 Design: at most 2 runs per set start in the answer room
MIN_FIRST_BLOCKED = 6  # >= 6 of 10 counted blocked runs are two-room Find / Conditional with the first room blocked
MAX_PER_SIM = 4  # 20 counted runs over the 3 x 2 layout: closed <= 8 on 2 sims, cabinet_open <= 4 on 1 sim
TWO_ROOMS = re.compile(r"\bin (?:the )?(\w+) or (?:the )?(\w+)\b")

def blocked_room(e: dict) -> str | None:
    w = world_of(e)
    return w.removeprefix("blocked_") if w.startswith("blocked_") else None


def first_named(e: dict) -> str | None:
    """The first room a two-room Find / a Conditional names (None for other tasks)."""
    t = at.task_by_id(e["task_id"])
    if t.kind == "conditional":
        return t.target["if_not_in"]
    m = TWO_ROOMS.search(t.prompt) if t.kind == "find" else None
    return m[1] if m else None


def comments(path: Path) -> dict[str, str]:
    return {m[1]: m[2].strip() for m in re.finditer(r"run_id: (\w+),.*\}\s*#\s*(.*)", path.read_text())}


def p3_header(path: Path) -> tuple[Path, str]:
    m = re.search(r"^# p3: (.*) sha256 ([0-9a-f]{64})$", path.read_text(), re.M)
    p = Path(m[1])
    return (p if p.is_absolute() else ROOT / p), m[2]


# ----------------------------------------------------------------------------------------- rules (shared checks)
def check_blocked(es: list[dict]) -> None:
    for e in es:
        b = blocked_room(e)
        if b is None:
            assert world_of(e) in GUARD_WORLDS, e
            continue
        assert run_truth(world_of(e), e["layout_seed"])["blocker"]["room"] == b, e
        assert e["expected"]["answer_room"] != b and e["start_place"] != b, e


def check_starts(lists: dict) -> None:
    for name, es in lists.items():
        for part in ([e for e in es if blocked_room(e)], [e for e in es if not blocked_room(e)]):
            st = Counter(start_type(e, e["expected"]["answer_room"]) for e in part)
            assert st["answer_room"] <= MAX_ANSWER_STARTS, (name, st)
    guard = Counter(start_type(e, e["expected"]["answer_room"]) for e in lists["counted"] if not blocked_room(e))
    assert guard["hall"] == 3 and guard["answer_room"] <= 2 and guard["other_room"] >= 5, guard
    by_task: dict[str, list[dict]] = {}
    for e in lists["counted"]:
        if blocked_room(e) and first_named(e):
            by_task.setdefault(e["task_id"], []).append(e)
    others = set()
    for tid, es in by_task.items():
        kinds = Counter(start_type(e, e["expected"]["answer_room"]) for e in es)
        assert kinds == {"hall": 1, "other_room": 1}, (tid, kinds)
        others |= {e["start_place"] for e in es if e["start_place"] != "hall"}
    assert len(others) >= 2, others  # the other rooms vary


def check_two_room_tasks(lists: dict) -> None:
    for e in (e for es in lists.values() for e in es):
        t = at.task_by_id(e["task_id"])
        ans = e["expected"]["answer_room"]
        if t.kind == "find" and (m := TWO_ROOMS.search(t.prompt)):
            assert ans in (m[1], m[2]) and m[1] == blocked_room(e), e
        if t.kind == "conditional" and blocked_room(e):
            assert t.target["if_not_in"] == blocked_room(e) and ans == t.target["else_room"], e
    b = [e for e in lists["counted"] if blocked_room(e)]
    assert len(b) == 10 and sum(first_named(e) == blocked_room(e) for e in b) >= MIN_FIRST_BLOCKED


def check_seeds(lists: dict) -> None:
    ok, every = ok_seeds(), [e["layout_seed"] for es in lists.values() for e in es]
    assert all(s in ok for s in every) and len(set(every)) == len(every)
    p5 = {e["layout_seed"] for k in ("counted", "dev") for e in load_runs(ROOT / "tasks" / f"phase5_{k}.yaml")}
    spot = pick.p5.spot_seeds()
    assert spot and {672, 673, 675, 676, 689, 690, 691, 692, 694, 698, 693} <= pick.excluded_seeds()
    assert not set(every) & (p5 | spot | pick.FIRST_SPOT | {693})


def check_capacity(lists: dict) -> None:
    worlds = {(j, i): "closed" for j in range(3) for i in range(2)} | pick.SHARD_WORLDS
    for es in lists.values():
        load = Counter(tuple(v) for v in split_runs(es, worlds).values())
        assert max(load.values()) <= MAX_PER_SIM, load
    n = Counter(world_of(e) for e in lists["counted"])
    assert n["closed"] <= 2 * MAX_PER_SIM and n["cabinet_open"] <= MAX_PER_SIM, n


def check_types(lists: dict, paths: dict[str, Path]) -> None:
    from agent.truth import seeded
    for name, es in lists.items():
        com = comments(paths[name])
        for e in (e for e in es if e["expected"]["kind"] == "count"):
            room = e["expected"]["answer_room"]
            several = len(set(seeded().count_photos(e["layout_seed"], room, "chair", world_of(e)).values())) >= 2
            assert 1 <= int(e["expected"]["answer"]) <= 3, e
            assert com[e["run_id"]] == ("several photos" if several else "one photo"), e["run_id"]
    c = lists["counted"]
    cnt = [e for e in c if e["expected"]["kind"] == "count"]
    assert any(comments(paths["counted"])[e["run_id"]] == "several photos" for e in cnt)
    guard_cnt = [e for e in cnt if not blocked_room(e)]
    assert len(guard_cnt) == 3 and {world_of(e) for e in guard_cnt} == set(GUARD_WORLDS)
    cs = [e for e in c if e["expected"]["kind"] == "check_state"]
    assert {e["expected"]["answer"] for e in cs if not blocked_room(e)} == {"yes", "no"}
    assert all(e["expected"]["answer"] == "no" for e in cs if blocked_room(e))
    assert Counter(e["expected"]["kind"] for e in c if not blocked_room(e)) == \
        {"find": 3, "check_state": 2, "count": 3, "conditional": 2}
    d = lists["dev"]
    assert Counter(world_of(e) for e in d) == {w: 2 for w in (*GUARD_WORLDS, "blocked_storage", "blocked_reception",
                                                              "blocked_ward")}


def check_p3(lists: dict, p3_ok: set) -> None:
    for e in (e for es in lists.values() for e in es if blocked_room(e)):
        assert (e["layout_seed"], world_of(e)) in p3_ok, e


def check_all(lists: dict, paths: dict[str, Path], p3_ok: set) -> None:
    assert [len(lists["counted"]), len(lists["dev"])] == [20, 10]
    for f in (check_starts, check_two_room_tasks, check_seeds, check_capacity):
        f(lists)
    check_blocked([e for es in lists.values() for e in es])
    check_types(lists, paths)
    check_p3(lists, p3_ok)


# ------------------------------------------------------------------------------------------------ the real lists
@pytest.fixture(scope="module")
def lists() -> dict[str, list[dict]]:
    if not (COUNTED.exists() and DEV.exists()):
        pytest.skip(f"no phase6 lists in {LISTS} yet")
    return {"counted": load_list(COUNTED), "dev": load_list(DEV)}


def test_blocked_entries(lists: dict) -> None:
    check_blocked([e for es in lists.values() for e in es])


def test_starts(lists: dict) -> None:
    check_starts(lists)


def test_two_room_find_and_conditional(lists: dict) -> None:
    check_two_room_tasks(lists)


def test_layout_seeds_disjoint(lists: dict) -> None:
    check_seeds(lists)


def test_world_capacity(lists: dict) -> None:
    check_capacity(lists)


def test_types_counts_checks(lists: dict) -> None:
    check_types(lists, {"counted": COUNTED, "dev": DEV})


def test_sampling_seeds_by_rule_and_disjoint(lists: dict) -> None:
    c, d = lists["counted"], lists["dev"]
    assert [e["seed"] for e in c] == random.Random("115-009-counted").sample(range(1, 2**31 - 1), len(c))
    assert [e["seed"] for e in d] == random.Random("115-009-dev").sample(range(1, 2**31 - 1), len(d))
    old = {e["seed"] for p in EARLIER for e in load_runs(p)}
    assert not {e["seed"] for e in c + d} & old and not {e["seed"] for e in c} & {e["seed"] for e in d}


def test_p3_ok_and_lists_are_the_pickers_output(lists: dict) -> None:
    p3, sha = p3_header(COUNTED)
    if not p3.exists():
        pytest.skip(f"P3 file {p3} not here")
    assert hashlib.sha256(p3.read_bytes()).hexdigest() == sha, f"{p3} changed since the pick"
    p3_ok = pick.read_p3(p3)
    check_p3(lists, p3_ok)
    rows = pick.pick(p3_ok)
    for name, path in (("counted", COUNTED), ("dev", DEV)):
        assert [x for x in path.read_text().splitlines() if x.startswith("- ")] == [pick.line(*r) for r in rows[name]]


# ------------------------------------------------------------------------- the picker, without the real P3 file
def small_pool() -> dict[int, dict]:
    """The first 12 candidates of every slot condition (a small real-seed pool every slot can still fill)."""
    pl = pick.pool()
    keep = set()
    for sl in pick.SLOTS_COUNTED + pick.SLOTS_DEV:
        keep |= set([x for x in sorted(pl) if pick.meets(sl["alts"][0], pl[x])][:12])
    return {x: pl[x] for x in keep}


class TestPicker:
    def test_pick_on_fake_p3(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pool = small_pool()
        monkeypatch.setattr(pick, "pool", lambda: pool)
        recs = [{"seed": x, "world": w, "ok": True} for x in pool for w in ("blocked_storage", "blocked_reception",
                                                                             "blocked_ward")]
        bad = next(x for x in sorted(pool) if pool[x]["toolbox_room"] == "reception")  # never in blocked_storage
        recs = [r | {"ok": False} if (r["seed"], r["world"]) == (bad, "blocked_storage") else r for r in recs]
        (tmp_path / "p3.json").write_text(json.dumps({"records": recs, "about": "fake"}))
        paths = pick.write(tmp_path / "out", tmp_path / "p3.json")
        ls = {k: load_list(p) for k, p in paths.items()}
        assert all(e["layout_seed"] in pool for es in ls.values() for e in es)
        assert (bad, "blocked_storage") not in {(e["layout_seed"], world_of(e)) for es in ls.values() for e in es}
        check_all(ls, paths, pick.read_p3(tmp_path / "p3.json"))

    def test_unfillable_slot_names_it(self, tmp_path: Path) -> None:
        f = tmp_path / "p3.json"
        f.write_text(json.dumps([{"seed": 0, "world": "blocked_storage", "ok": True}]))  # 0: toolbox in reception
        with pytest.raises(SystemExit, match="counted slot 3 \\(find-toolbox-reception-storage"):
            pick.pick(pick.read_p3(f))

    def test_p3_reader_formats(self, tmp_path: Path) -> None:
        r = [{"seed": 3, "world": "blocked_ward", "ok": True}, {"seed": 4, "world": "blocked_ward", "ok": False}]
        for d in (r, {"records": r}, {"results": r}):
            (tmp_path / "p.json").write_text(json.dumps(d))
            assert pick.read_p3(tmp_path / "p.json") == {(3, "blocked_ward")}
        (tmp_path / "p.json").write_text(json.dumps([{"seed": 3, "ok": True}]))
        with pytest.raises(AssertionError):
            pick.read_p3(tmp_path / "p.json")
