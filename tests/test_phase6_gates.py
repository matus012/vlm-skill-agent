"""probes/phase6_gates.py (task 009 G1-G3 + measure) on synthetic run dirs: no sim, no VLM, no blocked world truth.
Each run dir holds run.json, steps.jsonl and score.json; the per-run G1 rule (phase4_gates.g1_run), the common rules,
the scorer and require_merged are replaced by fakes that read score.json, so these tests check the phase 6 layer only:
the blocked / guard split, the 7-of-10 bars, the crash rule, block_hits, kill_crash and the measure extraction."""
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from agent import tasks as at  # noqa: E402
from probes import phase3_gates as p3  # noqa: E402
from probes import phase4_gates as p4  # noqa: E402
from probes import phase5_gates as p5  # noqa: E402
from probes import phase6_gates as g  # noqa: E402

KINDS = ("find", "conditional", "count", "check_state")
BLOCK = ("storage", "reception", "office", "ward")
ANSWER = {"storage": "reception", "reception": "storage", "office": "storage", "ward": "reception"}
STARTS = ("hall", "hall", "office", "ward", "reception")


def entries() -> list[dict[str, Any]]:
    out = []
    for i in range(20):
        blocked = i < 10
        room = BLOCK[i % 4] if blocked else None
        answer = ANSWER[room] if blocked else "storage"
        start = next(s for s in STARTS[i % 5:] + STARTS if s not in (room,))
        out.append({"run_id": f"p6{'b' if blocked else 'g'}{i:02d}", "task_id": "find-toolbox", "start_place": start,
                    "seed": i, "layout_seed": 900 + i,
                    "world": f"blocked_{room}" if blocked else ("closed", "cabinet_open")[i % 2],
                    "expected": {"kind": KINDS[i % 4], "answer": answer, "answer_room": answer,
                                 "set": "blocked" if blocked else "guard", "blocked_room": room, "photos": None,
                                 "memory_room": None, "memory_eligible": False}})
    return out


def step(n: int, name: str, args: dict, result: str, nav_sim_s: float | None = None) -> dict[str, Any]:
    return {"step": n, "call": {"name": name, "args": args}, "result": result, "skill_sim_s": 31.0,
            "skill_detail": {"nav_sim_s": nav_sim_s} if nav_sim_s is not None else {}, "planner_wall_s": 2.0}


def write_run(d: Path, e: dict[str, Any], passed: bool = True, hit: bool = True, outcome: str = "reported") -> None:
    room, ans = e["expected"]["blocked_room"], e["expected"]["answer_room"]
    steps = []
    if room is not None and hit:
        steps.append(step(1, "navigate_to", {"place": room}, g.BLOCKED, 40.0))
    steps += [step(len(steps) + 1, "navigate_to", {"place": ans}, "reached", 20.0),
              step(len(steps) + 2, "report", {"answer": ans, "photo": 1}, "ok")]
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"outcome": outcome, "steps": len(steps), "sim_s": 60.0, "wall_s": 80.0,
                                            "node": f"n{e['seed'] % 2}", "crash_origin": None}))
    (d / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    (d / "score.json").write_text(json.dumps({"pass": passed and outcome != "crash",
                                              "label": None if passed else "vlm"}))


def score(d: Path) -> dict[str, Any]:
    return json.loads((d / "score.json").read_text())


def fake_g1_run(e: dict[str, Any], d: Path | None) -> dict[str, Any]:
    if d is None:
        return {"run_id": e["run_id"], "pass": False, "problems": ["run dir missing"]}
    return {"run_id": e["run_id"], "pass": score(d)["pass"], "problems": []}


@pytest.fixture
def c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    es, out = entries(), tmp_path / "out"
    for e in es:
        write_run(out / e["run_id"], e)
    merged: list[Path] = []
    monkeypatch.setattr(p3, "require_merged", merged.append)
    monkeypatch.setattr(p4, "g1_run", fake_g1_run)
    monkeypatch.setattr(p4, "score_run", score)
    monkeypatch.setattr(p5, "common_problems", lambda out, entries, counted, list_path: [])
    return {"out": out, "entries": es, "merged": merged}


def ids(c: dict, which: str) -> list[str]:
    return [e["run_id"] for e in c["entries"] if e["expected"]["set"] == which]


def redo(c: dict, run_id: str, **kw: Any) -> None:
    e = next(e for e in c["entries"] if e["run_id"] == run_id)
    write_run(c["out"] / run_id, e, **kw)


# ------------------------------------------------------------------------------------------------------- tasks
def test_phase6_tasks_resolve_after_phase5() -> None:
    p6 = at.load_tasks(at.PHASE6_YAML)
    assert [t.id for t in p6] == ["find-toolbox-storage-reception", "find-toolbox-reception-storage",
                                  "cond-toolbox-storage-reception"]
    assert at.TASK_FILES[:3] == (at.DEV_YAML, at.PHASE5_YAML, at.PHASE6_YAML)
    n = sum(len(at.load_tasks(f)) for f in at.TASK_FILES[:3])  # phase8_pass1_tasks.yaml may follow (011)
    assert [t.id for t in at.all_tasks()][n - 3:n] == [t.id for t in p6]
    t = at.task_by_id("cond-toolbox-storage-reception")
    assert t.kind == "conditional" and t.prompt == "If the toolbox isn't in storage, check reception."
    assert t.target == {"object": "toolbox_red_1", "if_not_in": "storage", "else_room": "reception"}
    assert at.task_by_id("find-toolbox-storage-reception").target == {"object": "toolbox_red_1"}
    assert at.task_by_id("cond-toolbox-reception-storage") == at.task_by_id("cond-toolbox-reception-storage",
                                                                           at.PHASE5_YAML)


def test_phase6_id_repeated_in_phase5_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "p6.yaml"
    f.write_text(at.PHASE5_YAML.read_text())
    monkeypatch.setattr(at, "TASK_FILES", (at.DEV_YAML, at.PHASE5_YAML, f))
    with pytest.raises(at.TaskError):
        at.all_tasks()


# ------------------------------------------------------------------------------------------------------ list
def test_load_list_sets_and_blocked_room_from_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = entries()
    for e in raw:
        for k in ("set", "blocked_room"):
            del e["expected"][k]
    monkeypatch.setattr(p5, "load_list", lambda path, seeds: raw)
    truths = {w: ({"blocker": {"room": w.removeprefix("blocked_")}} if w.startswith("blocked_") else {})
              for w in {e["world"] for e in raw}}
    monkeypatch.setattr(g, "run_truth", lambda world, seed: truths[world])
    got = g.load_list(Path("x"))
    assert [e["expected"]["blocked_room"] for e in got] == [e["expected"]["blocked_room"] for e in entries()]
    assert [e["expected"]["set"] for e in got] == ["blocked"] * 10 + ["guard"] * 10
    truths["blocked_office"] = {"blocker": {"room": "storage"}}  # a blocked answer room is refused
    with pytest.raises(AssertionError):
        g.load_list(Path("x"))


# ------------------------------------------------------------------------------------------------------------ G1
def test_g1_passes_at_seven_fails_at_six(c: dict) -> None:
    b = ids(c, "blocked")
    for k in b[:3]:
        redo(c, k, passed=False)
    v = g.g1_verdict(c["out"], c["entries"], True)
    assert v["pass"] and v["passed"] == 7 and v["total"] == 10 and v["crashes"] == []
    assert sum(t["runs"] for t in v["by_type"].values()) == 10 and set(v["by_type"]) == set(KINDS)
    assert set(v["by_start"]) == {"hall", "answer_room", "other_room"}
    assert sum(t["passed"] for t in v["by_start"].values()) == 7
    assert c["merged"] and c["merged"][0] == c["out"]
    redo(c, b[3], passed=False)
    v = g.g1_verdict(c["out"], c["entries"], True)
    assert not v["pass"] and v["passed"] == 6


def test_g1_fails_on_one_crash_even_at_ten_of_ten(c: dict) -> None:
    b = ids(c, "blocked")
    assert g.g1_verdict(c["out"], c["entries"], True)["passed"] == 10
    redo(c, b[0], outcome="crash")
    sc = json.loads((c["out"] / b[0] / "score.json").read_text())
    sc["pass"] = True  # even if the scorer said pass, a crash fails G1
    (c["out"] / b[0] / "score.json").write_text(json.dumps(sc))
    v = g.g1_verdict(c["out"], c["entries"], True)
    assert v["passed"] == 10 and v["crashes"] == [b[0]] and not v["pass"]
    assert v["kill_crash"] == [{"run_id": b[0], "set": "blocked", "crash_origin": None}]


def test_g1_guard_crash_does_not_fail_g1_but_is_in_kill_crash(c: dict) -> None:
    gd = ids(c, "guard")
    redo(c, gd[0], outcome="crash")
    v = g.g1_verdict(c["out"], c["entries"], True)
    assert v["pass"] and v["kill_crash"] == [{"run_id": gd[0], "set": "guard", "crash_origin": None}]


def test_g1_common_problem_missing_run_and_not_counted(c: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not g.g1_verdict(c["out"], c["entries"], False)["pass"]
    monkeypatch.setattr(p5, "common_problems", lambda out, entries, counted, list_path: ["tree dirty"])
    assert not g.g1_verdict(c["out"], c["entries"], True)["pass"]


def test_block_hits_and_block_mattered(c: dict) -> None:
    b = ids(c, "blocked")
    for k in b[:4]:
        redo(c, k, hit=False)
    v = g.g1_verdict(c["out"], c["entries"], True)
    assert v["block_hits"] == 6 and v["block_mattered"] and v["pass"]
    redo(c, b[4], hit=False)
    v = g.g1_verdict(c["out"], c["entries"], True)
    assert v["block_hits"] == 5 and not v["block_mattered"] and v["pass"]  # recorded, does not change G1


def test_block_hit_needs_the_blocked_room_and_the_blocked_result(c: dict) -> None:
    from test_phase3_gates import edit_steps
    b = ids(c, "blocked")
    edit_steps(c["out"] / b[0], lambda r: r["call"]["args"].update(place="hall") if r["result"] == g.BLOCKED else None)
    edit_steps(c["out"] / b[1], lambda r: r.update(result="reached") if r["result"] == g.BLOCKED else None)
    rows = {r["run_id"]: r for r in g.g1_verdict(c["out"], c["entries"], True)["runs"]}
    assert not rows[b[0]]["block_hit"] and not rows[b[1]]["block_hit"] and rows[b[2]]["block_hit"]


# ------------------------------------------------------------------------------------------------------------ G2
def test_g2_guard_seven_of_ten(c: dict) -> None:
    gd = ids(c, "guard")
    for k in gd[:3]:
        redo(c, k, passed=False)
    for k in ids(c, "blocked"):  # the blocked set does not count for G2
        redo(c, k, passed=False)
    v = g.g2_verdict(c["out"], c["entries"], True)
    assert v["pass"] and v["passed"] == 7 and v["total"] == 10 and set(v["by_type"]) == set(KINDS)
    redo(c, gd[3], passed=False)
    assert not g.g2_verdict(c["out"], c["entries"], True)["pass"]
    assert not g.g1_verdict(c["out"], c["entries"], True)["pass"]


# ------------------------------------------------------------------------------------------------------------ G3
def test_g3_twenty_runs(c: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p3, "run_problems", lambda d: [])
    monkeypatch.setattr(p3, "g3_run", lambda d: [])
    ok = {"returncode": 0}
    v = g.g3_verdict(c["out"], c["entries"], True, ok)
    assert v["pass"] and v["kill_crash"] == []
    assert not g.g3_verdict(c["out"], c["entries"][:19], True, ok)["pass"]
    assert not g.g3_verdict(c["out"], c["entries"], True, {"returncode": 1})["pass"]
    assert not g.g3_verdict(c["out"], c["entries"], False, ok)["pass"]
    bad = ids(c, "guard")[0]
    monkeypatch.setattr(p3, "g3_run", lambda d: [{"echo": False, "text": "1.5"}] if d.name == bad else [])
    v = g.g3_verdict(c["out"], c["entries"], True, ok)
    assert not v["pass"] and bad in v["failing_hits"]
    monkeypatch.setattr(p3, "g3_run", lambda d: [])
    monkeypatch.setattr(p3, "run_problems", lambda d: ["no label"] if d.name == bad else [])
    assert not g.g3_verdict(c["out"], c["entries"], True, ok)["pass"]


# ------------------------------------------------------------------------------------------------------- measure
def test_measure_reaction_time_to_fail_and_repeats(c: dict) -> None:
    from test_phase3_gates import edit_run, edit_steps
    b = ids(c, "blocked")
    redo(c, b[1], hit=False)
    # b[0]: tries its blocked room twice; the second try's reaction is look_around
    e0 = next(e for e in c["entries"] if e["run_id"] == b[0])
    room = e0["expected"]["blocked_room"]
    steps = [step(1, "navigate_to", {"place": room}, g.BLOCKED, 40.0),
             step(2, "navigate_to", {"place": room}, g.BLOCKED, 50.0),
             step(3, "look_around", {}, "4 photos"),
             step(4, "report", {"answer": "x", "photo": 1}, "ok")]
    (c["out"] / b[0] / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    edit_run(c["out"] / b[2], sim_s=10.0, wall_s=100.0)
    edit_steps(c["out"] / ids(c, "guard")[0], lambda r: r.update(result=g.BLOCKED) if r["step"] == 1 else None)
    m = g.measure(c["out"], c["entries"], job_gpu_h=2.0)
    f0 = [f for f in m["blocked_steps"] if f["run_id"] == b[0]]
    assert [x["nav_sim_s"] for x in f0] == [40.0, 50.0] and all(x["skill_sim_s"] == 31.0 for x in f0)
    assert f0[0]["next_call"] == {"name": "navigate_to", "args": {"place": room}}
    assert f0[1]["next_call"] == {"name": "look_around", "args": {}}
    assert all(x["blocked_room"] == room and x["at_blocked_room"] for x in f0)
    guard_fail = [f for f in m["blocked_steps"] if f["run_id"] == ids(c, "guard")[0]]
    assert len(guard_fail) == 1 and guard_fail[0]["blocked_room"] is None and not guard_fail[0]["at_blocked_room"]
    assert m["navigate_to_blocked_room_per_run"][b[0]] == 2 and m["navigate_to_blocked_room_per_run"][b[1]] == 0
    assert set(m["navigate_to_blocked_room_per_run"]) == set(b)
    assert m["time_to_fail_nav_sim_s"]["n"] == 10 and m["time_to_fail_nav_sim_s"]["max"] == 50.0
    assert m["reaction_next_call"] == {"navigate_to": 9, "look_around": 1}
    assert m["steps_per_run"]["blocked"]["n"] == 10 and m["steps_per_run"]["guard"]["mean"] == 2
    assert m["flagged_below_0.6"] == [b[2]] and set(m["sim_wall_per_node"]) == {"n0", "n1"}
    assert m["sets"]["blocked"]["per_type"]["find"]["runs"] == 3 and "memory_signal" in m["sets"]["guard"]
    assert m["h200_h_per_run_job"] == 0.1 and m["kill_crash"] == []
    assert json.loads((c["out"] / "measure.json").read_text())["navigate_to_blocked_room_per_run"][b[0]] == 2


def test_kill_crash_in_every_gate_json(c: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lst = tmp_path / "phase6_counted.yaml"
    lst.write_text("[]\n")
    monkeypatch.setattr(g, "COUNTED", lst)
    monkeypatch.setattr(g, "load_list", lambda path, seeds: c["entries"])
    monkeypatch.setattr(p3, "no_ros_test", lambda: {"returncode": 0})
    monkeypatch.setattr(p3, "run_problems", lambda d: [])
    monkeypatch.setattr(p3, "g3_run", lambda d: [])
    args = ["all", "--out", str(c["out"]), "--list", str(lst)]
    assert g.main(args) == 0
    assert g.main([*args, "--rehearsal"]) == 1
    crashed = ids(c, "guard")[5]
    redo(c, crashed, outcome="crash")
    g.main(args)
    for n in ("g1", "g2", "g3", "measure"):
        assert json.loads((c["out"] / f"{n}.json").read_text())["kill_crash"] == [
            {"run_id": crashed, "set": "guard", "crash_origin": None}], n
