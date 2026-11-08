"""Task 008 run lists (tasks/phase5_counted.yaml, tasks/phase5_dev.yaml) obey the selection rule in their headers, and
tasks/phase5_tasks.yaml resolves after tasks/dev.yaml without changing what dev.yaml users get. True answers come from
truth_for(layout_seed, world) (phase5_gates.load_list), never from the lists. Env P5_LISTS_DIR checks the two lists in
another dir (e.g. probes/pick_phase5.py output) with the same tests."""
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
from probes.phase3_run import load_runs  # noqa: E402
from probes.phase4_gates import start_type, world_of  # noqa: E402
from probes import pick_phase5 as pick  # noqa: E402
from probes.phase5_gates import PER_TYPE, TYPES, load_list, ok_seeds  # noqa: E402

LISTS = Path(os.environ.get("P5_LISTS_DIR", ROOT / "tasks"))
COUNTED, DEV = LISTS / "phase5_counted.yaml", LISTS / "phase5_dev.yaml"
EARLIER = sorted((ROOT / "tasks").glob("phase3_*.yaml")) + sorted((ROOT / "tasks").glob("phase4_*.yaml"))
ROOMS = ("office", "reception", "storage", "ward")
MAX_SHARE = 0.4  # task 008 Counted sets: no room > 40 % of a type's answers
MAX_ANSWER_STARTS = 2  # at most 2 of 10 per type start in the answer room
MIN_SEVERAL = 4  # task 008 Design: >= 4 of 10 Count runs spread the class over >= 2 photos


@pytest.fixture(scope="module")
def lists() -> dict[str, list[dict]]:
    return {"counted": load_list(COUNTED), "dev": load_list(DEV)}


def toolbox_rooms() -> list[str]:
    """The G9-passing toolbox rooms the lists were picked for (their `# toolbox rooms:` header line)."""
    return re.search(r"^# toolbox rooms: (.*)$", COUNTED.read_text(), re.M)[1].split(", ")


def of(es: list[dict], kind: str) -> list[dict]:
    return [e for e in es if e["expected"]["kind"] == kind]


def comments(path: Path) -> dict[str, str]:
    """run_id -> the entry line's trailing comment."""
    return {m[1]: m[2].strip() for m in re.finditer(r"run_id: (\w+),.*\}\s*#\s*(.*)", path.read_text())}


# ------------------------------------------------------------------------------------------------- task files
def test_dev_yaml_unchanged_for_its_users() -> None:
    dev = at.load_tasks()
    assert [t.id for t in dev] == [t["id"] for t in yaml.safe_load(at.DEV_YAML.read_text())]
    p5 = at.load_tasks(at.PHASE5_YAML)
    assert not {t.id for t in dev} & {t.id for t in p5}
    n = len(dev) + len(p5)  # dev.yaml first, then phase5 (phase6_tasks.yaml after: tests/test_phase6_gates.py)
    assert [t.id for t in at.all_tasks()][:n] == [t.id for t in dev] + [t.id for t in p5]
    assert at.task_by_id("find-toolbox") == at.task_by_id("find-toolbox", at.DEV_YAML)
    assert at.task_by_id("count-chairs-office") == at.task_by_id("count-chairs-office", at.PHASE5_YAML)
    with pytest.raises(KeyError):
        at.task_by_id("count-chairs-office", at.DEV_YAML)


def test_repeated_id_across_files_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "p5.yaml"
    f.write_text(at.DEV_YAML.read_text())
    monkeypatch.setattr(at, "TASK_FILES", (at.DEV_YAML, f))
    with pytest.raises(at.TaskError):
        at.all_tasks()


def test_phase5_targets_are_g9_classes() -> None:
    for t in at.load_tasks(at.PHASE5_YAML):
        who = t.target.get("class") or t.target["object"].split("_")[0]
        assert who in ("toolbox", "chair", "bed", "cabinet"), t


# ------------------------------------------------------------------------------------------------ both lists
def test_layout_seeds_ok_and_never_shared(lists: dict) -> None:
    ok = ok_seeds()
    every = [e["layout_seed"] for es in lists.values() for e in es]
    assert all(s in ok for s in every)
    assert len(set(every)) == len(every)  # none shared within or between the lists


def test_no_spot_check_seed_in_the_lists(lists: dict) -> None:
    """The live spot-check / G9 layouts (p5_spot.sh `--seeds a,b,...`) are never run by the VLM lists."""
    spot = pick.spot_seeds()
    assert spot and spot <= set(ok_seeds())
    assert not spot & {e["layout_seed"] for es in lists.values() for e in es}


def test_lists_are_the_pickers_output() -> None:
    """The lists are exactly what probes/pick_phase5.py writes for their toolbox rooms (the rule, reproduced)."""
    rows = pick.pick(toolbox_rooms())
    for name, path in (("counted", COUNTED), ("dev", DEV)):
        assert [x for x in path.read_text().splitlines() if x.startswith("- ")] == [pick.line(*r) for r in rows[name]]


def test_seeds_json_rooms_match_truth_for(lists: dict) -> None:
    """The phase5_seeds.json records the lists were chosen from agree with the placement code."""
    ok = ok_seeds()
    for e in (e for es in lists.values() for e in es):
        rec = ok[e["layout_seed"]]
        rooms = {o["id"]: o["room"] for o in run_truth(world_of(e), e["layout_seed"])["objects"]}
        assert {k: rooms[k] for k in rec["rooms"]} == rec["rooms"], e
        assert rooms["toolbox_red_1"] == rec["toolbox_room"]


def test_sampling_seeds_by_rule_and_disjoint(lists: dict) -> None:
    c, d = lists["counted"], lists["dev"]
    assert [e["seed"] for e in c] == random.Random("115-008-counted").sample(range(1, 2**31 - 1), len(c))
    assert [e["seed"] for e in d] == random.Random("115-008-dev").sample(range(1, 2**31 - 1), len(d))
    old = {e["seed"] for p in EARLIER for e in load_runs(p)}
    assert len(EARLIER) == 4 and not ({e["seed"] for e in c + d} & old)
    assert not ({e["seed"] for e in c} & {e["seed"] for e in d})
    trip = [{(e["task_id"], e["start_place"], e["seed"]) for e in es} for es in (c, d)]
    assert not trip[0] & trip[1] and not ({e["run_id"] for e in c} & {e["run_id"] for e in d})


def test_layout_seed_on_every_entry_and_world_only_on_check_state(lists: dict) -> None:
    for e in (e for es in lists.values() for e in es):
        assert isinstance(e["layout_seed"], int)
        assert world_of(e) == "closed" or e["expected"]["kind"] == "check_state", e


@pytest.mark.parametrize("name", ["counted", "dev"])
def test_start_types(lists: dict, name: str) -> None:
    for kind in TYPES:
        st = Counter(start_type(e, e["expected"]["answer_room"]) for e in of(lists[name], kind))
        assert st["answer_room"] <= MAX_ANSWER_STARTS and st["hall"] >= 1 and st["other_room"] >= 1, (name, kind, st)


@pytest.mark.parametrize("name", ["counted", "dev"])
def test_answers_spread(lists: dict, name: str) -> None:
    for kind in ("find", "conditional", "count"):
        es = of(lists[name], kind)
        rooms = Counter(e["expected"]["answer_room"] for e in es)
        assert max(rooms.values()) <= MAX_SHARE * len(es), (name, kind, rooms)
    counts = {int(e["expected"]["answer"]) for e in of(lists[name], "count")}
    assert {1, 2} <= counts and max(counts) >= 3, counts


@pytest.mark.parametrize("name", ["counted", "dev"])
def test_count_photos(lists: dict, name: str) -> None:
    """Each counted room has >= 1 chair (derive raises on 0; checked again here), the one/several split comes from
    phase5_seeds.json, the comment on the line says the same, and >= 4 of 10 counted runs span >= 2 photos."""
    path = COUNTED if name == "counted" else DEV
    ok, com = ok_seeds(), comments(path)
    es = of(lists[name], "count")
    for e in es:
        rec = ok[e["layout_seed"]]
        room = e["expected"]["answer_room"]
        assert rec["chairs_per_room"][room] == int(e["expected"]["answer"]) >= 1
        several = len(set(rec["chair_photos"][room].values())) >= 2
        assert several == rec["spans_2_photos"][room]
        assert e["expected"]["photos"] == ("several" if several else "one")
        assert com[e["run_id"]] == ("several photos" if several else "one photo"), e["run_id"]
    if name == "counted":
        assert sum(e["expected"]["photos"] == "several" for e in es) >= MIN_SEVERAL
        assert sum(e["expected"]["photos"] == "one" for e in es) >= 1
        assert set(e["expected"]["answer_room"] for e in es) == set(ROOMS)


def test_count_photos_match_the_ray_caster(lists: dict) -> None:
    """phase5_seeds.json chair_photos = seeded.count_photos for every Count entry (the split's source)."""
    from agent.truth import seeded
    ok = ok_seeds()
    for e in (e for es in lists.values() for e in of(es, "count")):
        room = e["expected"]["answer_room"]
        assert seeded().count_photos(e["layout_seed"], room, "chair") == ok[e["layout_seed"]]["chair_photos"][room]


def test_counted_per_type(lists: dict) -> None:
    es = lists["counted"]
    assert Counter(e["expected"]["kind"] for e in es) == {k: PER_TYPE for k in TYPES}
    assert {e["start_place"] for e in es} == {"hall", *ROOMS}
    # find: mostly the toolbox, in every G9-passing room; the rest G9 class targets derive accepts (load_list)
    rooms = set(toolbox_rooms())
    find = of(es, "find")
    tb = [e for e in find if at.task_by_id(e["task_id"]).target.get("object") == "toolbox_red_1"]
    assert len(tb) > len(find) / 2 and {e["expected"]["answer"] for e in tb} == rooms
    # conditional: the moving toolbox in every G9-passing room, both branches; fill-ins only on the beds (dev. 17)
    cond = of(es, "conditional")
    targets = [at.task_by_id(e["task_id"]).target for e in cond]
    assert all(t.get("object") == "toolbox_red_1" or t.get("class") == "bed" for t in targets)
    assert {e["expected"]["answer"] for e, t in zip(cond, targets) if "object" in t} == rooms
    right = Counter(e["expected"]["answer"] == t["if_not_in"] for e, t in zip(cond, targets))
    assert right[True] >= 1 and right[False] >= 1
    # check_state: both worlds x both answers, cabinet fixed
    cs = of(es, "check_state")
    assert {(world_of(e), e["expected"]["answer"]) for e in cs} == {(w, a) for w in ("closed", "cabinet_open")
                                                                       for a in ("yes", "no")}
    assert {e["expected"]["answer_room"] for e in cs} == {"storage"}


def test_dev_covers_each_type(lists: dict) -> None:
    d = lists["dev"]
    kinds = Counter(e["expected"]["kind"] for e in d)
    assert set(kinds) == set(TYPES) and all(2 <= n <= 3 for n in kinds.values())
    assert {world_of(e) for e in d} == {"closed", "cabinet_open"}
    assert {e["expected"]["answer"] for e in of(d, "check_state")} == {"yes", "no"}
    cond = of(d, "conditional")
    right = {e["expected"]["answer"] == at.task_by_id(e["task_id"]).target["if_not_in"] for e in cond}
    assert right == {True, False}


def test_phase5_counted_list_is_counted_and_pinned() -> None:
    from probes.phase3_run import COUNTED_LISTS, is_counted, sha256_file
    p5 = ROOT / "tasks" / "phase5_counted.yaml"
    assert is_counted(p5) and COUNTED_LISTS[p5] == sha256_file(p5)
    assert COUNTED_LISTS[p5] == "3b5416cdd5f9add874517587a7fe311606b3f0008524e63cf5899d47f11d6fca"  # dev. 17 re-pick
    assert not is_counted(ROOT / "tasks" / "phase5_dev.yaml")

