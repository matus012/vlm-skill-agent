"""probes/build_phase8.py (task 011, INTERFACES section 12) on SYNTHETIC inputs only: a fake pass file rendered by the
sealed generator's render_pass(1, small seeds), a fake fresh-seed file, and a fake realizer that follows section 12's
placement rules (toolbox parked in the ward, beds parked outside the ward, cabinet parked outside storage). Never reads
anything under a private directory outside this repo:  or unsealed/ except the generator code."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import tasks as at  # noqa: E402
from probes import build_phase8 as b  # noqa: E402

GEN = ROOT.parent / "private-workspace" / "unsealed" / "phase8" / "gen_phase8.py"  # not shipped: the test skips
pytestmark = pytest.mark.skipif(not GEN.is_file(), reason="sealed generator code not on this machine")
PASS_SEEDS = list(range(1, 26))
FRESH = [100_000 + 7 * i for i in range(400)]
PLACES = {"hall", "ward", "office", "reception", "storage"}
LAYOUT = yaml.safe_load((ROOT / "sim/worlds/clinic/layout.yaml").read_text())


def fake_realizer(extra_park: Callable[[dict], set] = lambda s: set(),
                  bad: Callable[[dict], bool] = lambda s: False) -> SimpleNamespace:
    def parked(spec: dict) -> set:
        o = spec["objects"]
        p = set(spec["removed"]) | extra_park(spec)
        p |= {"toolbox_red_1"} if o["toolbox_red_1"]["room"] == "ward" else set()
        p |= {"cabinet_blue_1"} if o["cabinet_blue_1"]["room"] != "storage" else set()
        return p | {k for k in o if k.startswith("bed_") and o[k]["room"] != "ward"}

    def realize(spec: dict) -> dict:
        world = "cabinet_open" if spec["objects"]["cabinet_blue_1"]["state"] == "open" else "closed"
        return {"world": world, "poses": {}, "parked": sorted(parked(spec))}

    def truth_for_spec(spec: dict) -> dict:
        t = b_gen().spec_truth(spec, LAYOUT)
        return t | {"objects": [o for o in t["objects"] if o["id"] not in parked(spec)]}

    return SimpleNamespace(realize=realize, truth_for_spec=truth_for_spec,
                           check_spec=lambda spec: (not bad(spec), ["bad"] if bad(spec) else []))


_GEN: list = []


def b_gen() -> Any:
    if not _GEN:
        _GEN.append(b.load_gen(GEN))
    return _GEN[0]


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    f, s = tmp_path / "phase8_pass1.yaml", tmp_path / "fresh.txt"
    f.write_text(b_gen().render_pass(1, PASS_SEEDS))
    s.write_text("".join(f"{x}\n" for x in FRESH))
    return {"pass": f, "fresh": s, "out": tmp_path / "out"}


def run(files: dict, R: Any = None, used: set | None = None, places: set = PLACES, n: int = 1) -> tuple:
    return b.build(b_gen(), R or fake_realizer(), n, files["pass"], files["fresh"], GEN, used or set(), places)


def by_kind(tasks: list[dict]) -> dict[str, list[dict]]:
    return {k: [t for t in tasks if t["kind"] == k] for k in at.KINDS}


# ------------------------------------------------------------------------------------------------- stream
def test_stream_continues_ids_and_rerenders_pass_file(files: dict) -> None:
    layouts, n_own = b.stream(b_gen(), 1, files["pass"], FRESH[:3])
    assert n_own == 25 and len(layouts) == 28
    assert [lay["seed"] for lay in layouts] == PASS_SEEDS + FRESH[:3]
    assert layouts[25]["tasks"][0]["id"] == "p8-1-26-find"
    files["pass"].write_text(files["pass"].read_text().replace("seed: 3\n", "seed: 4\n", 1))
    with pytest.raises(SystemExit, match="byte for byte"):
        b.stream(b_gen(), 1, files["pass"], FRESH)
    with pytest.raises(SystemExit, match="pass 1, --pass 2"):
        b.stream(b_gen(), 2, files["pass"], FRESH)


def test_fresh_pin_checked(files: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin = tmp_path / "pin.sha256"
    monkeypatch.setattr(b, "FRESH_PINS", pin)
    pin.write_text(f"# c\n{b.sha256(files['fresh'])}  400  fresh.txt\n")
    assert b.read_fresh(files["fresh"]) == FRESH
    pin.write_text(f"{'0' * 64}  400  fresh.txt\n")
    with pytest.raises(SystemExit, match="pin"):
        b.read_fresh(files["fresh"])


# ------------------------------------------------------------------------------------------------- filter
def test_25_per_kind_and_rule_a_holds(files: dict) -> None:
    tasks, entries, rep = run(files)
    assert len(entries) == 100 and {k: len(v) for k, v in by_kind(tasks).items()} == {k: 25 for k in at.KINDS}
    assert rep["kept"] == {k: 25 for k in at.KINDS} and sum(rep["drops"]["a_g9_placement"].values()) > 0
    R = fake_realizer()
    for t, e in zip(tasks, entries):
        assert e["task_id"] == t["id"]
        ids = b.target_ids(t, e["layout_spec"])
        room = e["layout_spec"]["objects"][ids[0]]["room"]
        assert room in b.G9_ROOMS.get(b.cls_of(ids[0]), PLACES - {"hall"})
        assert not set(ids) & set(R.realize(e["layout_spec"])["parked"])
        assert e["world"] == R.realize(e["layout_spec"])["world"]
    for k in at.KINDS:
        assert rep["entries_examined"][k] == 25 + sum(rep["drops"][r][k] for r in b.RULES)
        assert rep["yield"][k] == 25 / rep["entries_examined"][k]


def test_rule_a_parked_target_dropped(files: dict) -> None:
    R = fake_realizer(extra_park=lambda s: {"chair_orange_1"} if s["seed"] % 2 else set())
    tasks, entries, _ = run(files, R)
    assert not [e for t, e in zip(tasks, entries) if b.cls_of(b.target_ids(t, e["layout_spec"])[0]) == "chair"
                and e["layout_spec"]["seed"] % 2]
    with pytest.raises(SystemExit, match="ran out"):  # the cabinet parked everywhere: check_state never fills
        run(files, fake_realizer(extra_park=lambda s: {"cabinet_blue_1"}))


def test_rule_b_check_spec(files: dict) -> None:
    _, base, _ = run(files)
    bad = {e["layout_spec"]["seed"] for e in base[:10]}
    _, entries, rep = run(files, fake_realizer(bad=lambda s: s["seed"] in bad))
    assert not bad & {e["layout_spec"]["seed"] for e in entries}
    assert sum(rep["drops"]["b_check_spec"].values()) > 0


def test_rule_c_seed_collisions(files: dict) -> None:
    _, base, _ = run(files)
    used = {base[0]["layout_spec"]["seed"], base[1]["seed"]}
    _, entries, rep = run(files, used=used)
    assert base[0]["layout_spec"]["seed"] not in {e["layout_spec"]["seed"] for e in entries}
    assert base[1]["task_id"] not in {e["task_id"] for e in entries}
    assert sum(rep["drops"]["c_seed_collision"].values()) >= 2


def test_used_seeds_cover_phase3_to_6_and_spot() -> None:
    used = b.used_seeds()
    for f in ("phase3_counted", "phase3_dev", "phase4_counted", "phase5_counted", "phase5_dev", "phase6_counted",
              "phase6_dev"):
        for e in yaml.safe_load((ROOT / "tasks" / f"{f}.yaml").read_text()):
            assert e["seed"] in used and e.get("layout_seed", e["seed"]) in used
    assert {693, 672, 698} <= used and b.pick_phase6.p5.spot_seeds() <= used


def test_planner_seed() -> None:
    for i in range(1, 60):
        tid = f"p8-1-{i:02d}-count"
        assert b.planner_seed(tid) == int(hashlib.sha256(tid.encode()).hexdigest(), 16) % 2**31


def test_starts_rule(files: dict) -> None:
    tasks, entries, rep = run(files)
    R = fake_realizer()
    for k, ts in by_kind(tasks).items():
        es = [e for t, e in zip(tasks, entries) if t["kind"] == k]
        n_ans = 0
        for i, (t, e) in enumerate(zip(ts, es)):
            assert e["seed"] == b.planner_seed(t["id"])
            ans = b.answer_room(t, R.truth_for_spec(e["layout_spec"]))
            want = b.CYCLE[i % 3]
            if want == "answer_room" and n_ans == 2:
                want = "other_room"
            if want == "hall":
                assert e["start_place"] == "hall"
            elif want == "answer_room":
                assert e["start_place"] == ans
                n_ans += 1
            else:
                named = {t["target"].get(x) for x in ("if_not_in", "else_room")}
                assert e["start_place"] in PLACES - {"hall", ans} - named
        assert n_ans == 2 and rep["starts"][k] == {"hall": 9, "other_room": 14, "answer_room": 2}


def test_rule_d_never_a_missing_place(files: dict) -> None:
    _, entries, rep = run(files, places=PLACES - {"office"})
    assert "office" not in {e["start_place"] for e in entries}  # only an answer-room slot in the office drops
    assert rep["drops"]["d_start_rule"]["check_state"] == 0  # the cabinet's room is storage
    _, entries, rep = run(files, places=PLACES - {"office", "reception"})  # a conditional on ward / storage: no other
    assert {e["start_place"] for e in entries} <= {"hall", "ward", "storage"}
    assert rep["drops"]["d_start_rule"]["conditional"] > 0


# ------------------------------------------------------------------------------------------------- stop
def test_stops_at_25_and_runs_out(files: dict) -> None:
    *_, rep = run(files)
    used = rep["fresh_seeds_used"]
    assert rep["layouts_consumed"] == 25 + used and used > 0
    files["fresh"].write_text("".join(f"{x}\n" for x in FRESH[:used]))
    t2, e2, rep2 = run(files)
    assert rep2["layouts_consumed"] == rep["layouts_consumed"] and rep2["fresh_seeds_available"] == used
    files["fresh"].write_text("".join(f"{x}\n" for x in FRESH[:used - 1]))
    with pytest.raises(SystemExit, match="ran out"):
        run(files)


# ------------------------------------------------------------------------------------------------- outputs
def write_all(files: dict) -> dict[str, Path]:
    tasks, entries, rep = run(files)
    return b.write(files["out"], 1, b_gen(), tasks, entries, rep)


def test_outputs_deterministic_header_and_loadable(files: dict) -> None:
    p1 = write_all(files)
    first = {k: v.read_bytes() for k, v in p1.items()}
    p2 = write_all(files)
    assert {k: v.read_bytes() for k, v in p2.items()} == first
    shas = (b.sha256(files["pass"]), b.sha256(files["fresh"]), b.sha256(GEN))
    for k in ("tasks", "list"):
        head = [ln for ln in p1[k].read_text().splitlines() if ln.startswith("#")]
        assert all(any(s in ln for ln in head) for s in shas)
    tasks = at.load_tasks(p1["tasks"])
    entries = yaml.safe_load(p1["list"].read_text())
    assert len(tasks) == len(entries) == 100 and [t.id for t in tasks] == [e["task_id"] for e in entries]
    assert all(set(e) == {"run_id", "task_id", "start_place", "seed", "world", "layout_spec"} for e in entries)
    assert all(set(e["layout_spec"]) == {"seed", "objects", "removed"} for e in entries)
    rep = json.loads(p1["drops"].read_text())
    assert rep["list_sha256"] == b.sha256(p1["list"]) and rep["tasks_sha256"] == b.sha256(p1["tasks"])
    assert len({e["run_id"] for e in entries}) == len({e["seed"] for e in entries}) == 100


def test_phase8_tasks_resolution_order() -> None:
    assert at.TASK_FILES[:3] == (at.DEV_YAML, at.PHASE5_YAML, at.PHASE6_YAML)
    assert (at.PHASE8_YAML in at.TASK_FILES) == at.PHASE8_YAML.is_file()
    assert at.PHASE8_YAML == ROOT / "tasks" / "phase8_pass1_tasks.yaml"


def test_resolves_through_task_files(files: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    p = write_all(files)
    monkeypatch.setattr(at, "TASK_FILES", (at.DEV_YAML, at.PHASE5_YAML, at.PHASE6_YAML, p["tasks"]))
    e = yaml.safe_load(p["list"].read_text())[0]
    assert at.task_by_id(e["task_id"]).id == e["task_id"]


def test_main_prints_counts_only(files: dict, monkeypatch: pytest.MonkeyPatch,
                                 capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(b, "realizer", fake_realizer)
    monkeypatch.setattr(b, "used_seeds", set)
    assert b.main(["--pass", "1", "--pass-file", str(files["pass"]), "--fresh-seeds", str(files["fresh"]),
                   "--gen", str(GEN), "--out", str(files["out"])]) == 0
    out = capsys.readouterr().out
    rep = json.loads(out)
    assert rep["kept"] == {k: 25 for k in at.KINDS}
    tasks = at.load_tasks(files["out"] / "phase8_pass1_tasks.yaml")
    leaks = [t.prompt for t in tasks] + [t.id for t in tasks] + ["p8-1-", "ward", "office", "reception", "storage"]
    leaks += [str(x) for x in FRESH]
    assert not [x for x in leaks if x in out]


def test_list_loads_with_phase8_gates(files: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from probes import phase4_gates as p4
    from probes import phase8_gates as g
    p = write_all(files)
    R = fake_realizer()
    monkeypatch.setattr(at, "TASK_FILES", (at.DEV_YAML, at.PHASE5_YAML, at.PHASE6_YAML, p["tasks"]))
    monkeypatch.setattr(p4, "run_truth", lambda world, seed, spec: R.truth_for_spec(spec))
    es = g.load_list(p["list"])
    assert len(es) == 100 and {k: sum(e["expected"]["kind"] == k for e in es) for k in at.KINDS} == {
        k: 25 for k in at.KINDS}
    assert sum(e["start_place"] == e["expected"]["answer_room"] for e in es) == 8
