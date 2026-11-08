"""probes/phase8_gates.py (task 011 G1, G2, measure, gallery) on synthetic run dirs: no sim, no VLM, no test entry.
The per-run G1 rule (phase4_gates.g1_run), the common rules, the scorer and require_merged are replaced by fakes that
read score.json, so these tests check the phase 8 layer: the 18-of-25 bar per type, the Wilson interval, the layout_spec
match, the freeze hook, the measure additions and the gallery fields."""
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probes import phase3_gates as p3  # noqa: E402
from probes import phase4_gates as p4  # noqa: E402
from probes import phase5_gates as p5  # noqa: E402
from probes import phase8_gates as g  # noqa: E402

SPEC = {"seed": 1, "objects": {"toolbox_red_1": {"room": "storage"}}, "removed": []}


def entries() -> list[dict[str, Any]]:
    out = []
    for i in range(100):
        kind = g.TYPES[i % 4]
        out.append({"run_id": f"r{i:03d}", "task_id": f"t-{kind}", "start_place": "hall", "seed": i, "world": "closed",
                    "layout_spec": SPEC | {"seed": 1000 + i},
                    "expected": {"kind": kind, "answer": "storage", "answer_room": "storage"}})
    return out


def write_run(d: Path, e: dict[str, Any], passed: bool = True, commit: str = "c0") -> None:
    steps = [{"step": 1, "call": {"name": "look_around", "args": {}}, "result": "ok", "planner_wall_s": 2.0,
              "packet": {"photos": [{"n": 1, "rgb": "photos/0_0_rgb.png", "seg": "photos/0_0_seg.png"}]},
              "planner_out": {"reasoning": "x" * 500 + "END", "content": None}},
             {"step": 2, "call": {"name": "report", "args": {"answer": "ward", "photo": 1}}, "result": "ok",
              "planner_wall_s": 3.0, "packet": {"photos": [{"n": k, "rgb": f"photos/1_{k - 1}_rgb.png", "seg": None}
                                                           for k in (1, 2, 3, 4)]},
              "planner_out": {"reasoning": None, "content": "done"}}]
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({
        "outcome": "reported", "steps": 2, "sim_s": 50.0, "wall_s": 100.0 if e["seed"] % 2 else 60.0,
        "node": f"n{e['seed'] % 2}", "git_commit": commit, "layout_spec": e["layout_spec"],
        "report": {"answer": "storage" if passed else "ward", "photo": 1}}))
    (d / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    (d / "score.json").write_text(json.dumps({"pass": passed, "label": None if passed else "vlm",
                                              "label_why": None if passed else "answer 'ward' != truth 'storage'"}))


def score(d: Path) -> dict[str, Any]:
    return json.loads((d / "score.json").read_text())


def fake_g1_run(e: dict[str, Any], d: Path | None) -> dict[str, Any]:
    if d is None:
        return {"run_id": e["run_id"], "pass": False, "problems": ["run dir missing"]}
    s = score(d)
    return {"run_id": e["run_id"], "pass": s["pass"], "problems": [], "label": s["label"], "label_why": s["label_why"]}


class FakeTask:
    def __init__(self, tid: str) -> None:
        self.id, self.prompt = tid, f"prompt of {tid}"


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
    monkeypatch.setattr(g, "task_by_id", FakeTask)
    return {"out": out, "entries": es, "merged": merged}


def fail(c: dict, kind: str, n: int) -> None:
    for e in [e for e in c["entries"] if e["expected"]["kind"] == kind][:n]:
        write_run(c["out"] / e["run_id"], e, passed=False)


# ------------------------------------------------------------------------------------------------------------ G1
def test_g1_18_of_25_passes_17_fails(c: dict) -> None:
    fail(c, "count", 7)
    v = g.g1_verdict(c["out"], c["entries"], True, Path("l"))
    assert v["types"]["count"]["passed"] == 18 and v["types"]["count"]["pass"] and v["pass"]
    fail(c, "count", 8)
    v = g.g1_verdict(c["out"], c["entries"], True, Path("l"))
    assert v["types"]["count"]["passed"] == 17 and not v["types"]["count"]["pass"] and not v["pass"]
    assert all(v["types"][k]["pass"] for k in ("find", "check_state", "conditional"))
    assert c["merged"]


def test_g1_not_counted_or_common_problem_never_passes(c: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not g.g1_verdict(c["out"], c["entries"], False, Path("l"))["pass"]
    monkeypatch.setattr(p5, "common_problems", lambda out, entries, counted, list_path: ["2 commits"])
    assert not g.g1_verdict(c["out"], c["entries"], True, Path("l"))["pass"]


def test_g1_missing_run(c: dict) -> None:
    assert not g.g1_verdict(c["out"], c["entries"][1:], True, Path("l"))["types"]["find"]["pass"]  # 24 runs listed
    (c["out"] / "r000" / "run.json").unlink()
    v = g.g1_verdict(c["out"], c["entries"], True, Path("l"))
    assert v["types"]["find"]["passed"] == 24 and v["types"]["find"]["runs"][0]["problems"] == ["run dir missing"]


def test_wilson_numbers(c: dict) -> None:
    lo, hi = g.wilson(18, 25)
    assert lo == pytest.approx(0.524233, abs=1e-5) and hi == pytest.approx(0.857163, abs=1e-5)
    lo, hi = g.wilson(25, 25)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.866804, abs=1e-5)
    assert g.wilson(0, 0) is None
    fail(c, "find", 7)
    t = g.g1_verdict(c["out"], c["entries"], True, Path("l"))["types"]["find"]
    assert t["rate"] == 0.72 and t["wilson95"] == pytest.approx((0.524233, 0.857163), abs=1e-5)


# ------------------------------------------------------------------------------------------------------------ G2
def fake_g3(missing: list | None = None) -> Any:
    return lambda out, entries, counted, no_ros: {"missing_runs": missing or [], "log_problems": {},
                                                  "failing_hits": {}, "pass": False}


def test_g2_needs_freeze_and_rules(c: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p4, "g3_verdict", fake_g3())
    ok_ros, ok_fz = {"returncode": 0}, {"pass": True}
    assert g.g2_verdict(c["out"], c["entries"], True, ok_ros, ok_fz)["pass"]
    assert not g.g2_verdict(c["out"], c["entries"], True, ok_ros, {"pass": False})["pass"]
    assert not g.g2_verdict(c["out"], c["entries"], True, {"returncode": 1}, ok_fz)["pass"]
    assert not g.g2_verdict(c["out"], c["entries"], False, ok_ros, ok_fz)["pass"]
    assert not g.g2_verdict(c["out"], c["entries"][:99], True, ok_ros, ok_fz)["pass"]
    monkeypatch.setattr(p4, "g3_verdict", fake_g3(["r001"]))
    assert not g.g2_verdict(c["out"], c["entries"], True, ok_ros, ok_fz)["pass"]


NULLS = list(g.SKIPPED_NULLS)


def crash_record(d: Path) -> None:
    """r001 as adapters/run_task.teleport_failed writes it (setup skipped)."""
    run = json.loads((d / "run.json").read_text())
    run.update(outcome="crash", crash_origin="layout_teleport",
               setup={"start_place": "hall", "reached": None, "gt_err_m": None, "skipped": g.SKIPPED_SETUP})
    (d / "run.json").write_text(json.dumps(run))


def test_g2_accepts_a_teleport_failed_record(c: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deviation 20.3: the setup nulls of a teleport_failed record are complete; anything else still fails G2."""
    crash_record(c["out"] / "r001")
    probs = {"r001": NULLS, "r002": NULLS}  # r002 is no crash record: its nulls stand
    monkeypatch.setattr(p4, "g3_verdict", lambda out, entries, counted, no_ros: {
        "missing_runs": [], "log_problems": probs, "failing_hits": {}, "pass": False})
    v = g.g2_verdict(c["out"], c["entries"], True, {"returncode": 0}, {"pass": True})
    assert v["log_problems"] == {"r002": NULLS} and v["log_problems_accepted_skipped_setup"] == {"r001": NULLS}
    assert not v["pass"] and not v["pass_as_written"] and v["log_problems_as_written"] == probs
    probs.pop("r002")
    probs["r001"] = NULLS + ["check_complete: steps.jsonl missing"]  # an extra problem on the crash record stands
    v = g.g2_verdict(c["out"], c["entries"], True, {"returncode": 0}, {"pass": True})
    assert v["log_problems"] == {"r001": ["check_complete: steps.jsonl missing"]} and not v["pass"]
    probs["r001"] = NULLS
    v = g.g2_verdict(c["out"], c["entries"], True, {"returncode": 0}, {"pass": True})
    assert v["pass"] and not v["pass_as_written"] and v["log_problems"] == {}


def test_freeze_check_against_tag(c: dict) -> None:
    r = subprocess.run(["git", "rev-parse", "-q", "--verify", f"{g.FREEZE_TAG}^{{commit}}"], cwd=ROOT,
                       capture_output=True, text=True)
    tag = r.stdout.strip()
    if r.returncode:  # a one-commit snapshot (the clean showcase repo) has no history to tag
        pytest.skip(f"needs the git history with tag {g.FREEZE_TAG}")
    for e in c["entries"]:
        write_run(c["out"] / e["run_id"], e, commit=tag)
    ok = {"returncode": 0}
    f = g.freeze_check(c["out"], c["entries"], g.FREEZE_TAG, ok)
    assert f["pass"] and f["run_commit"] == tag and f["freeze_json_equal_tag"]
    assert not g.freeze_check(c["out"], c["entries"], g.FREEZE_TAG, {"returncode": 1})["pass"]
    write_run(c["out"] / "r000", c["entries"][0], commit=tag + "^")  # before the freeze: no interface_freeze.json
    f = g.freeze_check(c["out"], c["entries"][:1], g.FREEZE_TAG, ok)
    assert not f["pass"] and not f["freeze_json_equal_tag"]
    f = g.freeze_check(c["out"], c["entries"], g.FREEZE_TAG, ok)  # two commits
    assert not f["pass"] and f["run_commit"] is None and len(f["run_commits"]) == 2


def test_freeze_test_hook_runs_the_real_test() -> None:
    r = g.freeze_test()
    assert r["returncode"] == 0 and r["file"] == "tests/test_interface_freeze.py" and "passed" in r["summary"]


# ------------------------------------------------------------------------------------------------------- measure
def test_measure_per_type_and_node(c: dict) -> None:
    fail(c, "conditional", 3)
    m = g.measure(c["out"], c["entries"])
    assert m["per_type"]["conditional"]["failure_labels"] == {"vlm": 3}
    assert m["sim_wall_per_type"]["find"]["n"] == 25
    assert sorted(m["flagged_below_0.6"]) == sorted(e["run_id"] for e in c["entries"] if e["seed"] % 2)
    assert m["per_node"]["n0"]["vlm_latency_per_step_s"]["mean"] == 2.5
    assert m["per_node"]["n1"]["steps_per_run"]["n"] == 50
    assert json.loads((c["out"] / "measure.json").read_text())["per_node"]


def test_teleports_per_sim_in_run_order(c: dict) -> None:
    for k, e in enumerate(c["entries"][:4]):
        d = c["out"] / e["run_id"]
        run = json.loads((d / "run.json").read_text())
        run.update(slurm_job_id="j1", shard=k % 2, start_utc=f"2026-11-06T0{4 - k}:00:00Z",
                   teleport={"models": 9, "sends": 9 + k, "retries": ["x"] * k})
        (d / "run.json").write_text(json.dumps(run))
    crash_record(c["out"] / "r002")
    m = g.measure(c["out"], c["entries"][:4])["teleports_per_sim"]
    s0, s1 = m["j1/0"], m["j1/1"]
    assert [r["run_id"] for r in s0["runs"]] == ["r002", "r000"] and s0["crash_orders"] == [1]
    assert s0["retries"] == 2 and s1["retries"] == 4 and s1["crashes"] == 0
    assert [r["sends"] for r in s1["runs"]] == [12, 10]


# ------------------------------------------------------------------------------------------------------- gallery
def test_gallery_fields(c: dict) -> None:
    fail(c, "check_state", 2)
    v = g.g1_verdict(c["out"], c["entries"], True, Path("l"))
    rows = g.gallery(c["out"], c["entries"], v)
    assert [r["run_id"] for r in rows] == ["r001", "r005"]
    r = rows[0]
    assert set(r) == {"run_id", "task_id", "prompt", "kind", "truth_answer", "answer", "label", "why", "photos",
                      "steps"}
    assert (r["kind"], r["task_id"], r["prompt"]) == ("check_state", "t-check_state", "prompt of t-check_state")
    assert (r["truth_answer"], r["answer"], r["label"]) == ("storage", "ward", "vlm") and "!= truth" in r["why"]
    assert len(r["photos"]) == 5 and r["photos"][0]["rgb"] == str(c["out"] / "r001" / "photos/0_0_rgb.png")
    assert r["steps"][0]["reasoning_excerpt"] == ("x" * 500 + "END")[-g.EXCERPT:]
    assert r["steps"][1]["reasoning_excerpt"] == "done" and r["steps"][1]["call"]["name"] == "report"
    assert json.loads((c["out"] / "gallery.json").read_text()) == rows
    md = (c["out"] / "gallery.md").read_text()
    assert "## r001" in md and "photos/1_3_rgb.png" in md and "2 failed runs" in md


# ------------------------------------------------------------------------------------------------------------ list
def test_load_list_expected_from_spec_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml
    f = tmp_path / "l.yaml"
    es = [{k: v for k, v in e.items() if k != "expected"} | {"task_id": "find-toolbox"} for e in entries()[:2]]
    f.write_text(yaml.safe_dump(es))
    truth = {"rooms": {r: {} for r in ("hall", "ward", "office", "reception", "storage")},
             "objects": [{"id": "toolbox_red_1", "class": "toolbox", "colour": "red", "room": "reception",
                          "state": None, "label": 7}]}
    seen = []
    monkeypatch.setattr(p4, "run_truth", lambda w, s, spec: seen.append((w, s, spec)) or truth)
    got = g.load_list(f)
    assert [x["expected"] for x in got] == [{"kind": "find", "answer": "reception", "answer_room": "reception"}] * 2
    assert seen[0] == ("closed", None, es[0]["layout_spec"])
    f.write_text(yaml.safe_dump([{k: v for k, v in es[0].items() if k != "layout_spec"}]))
    with pytest.raises(AssertionError, match="without layout_spec"):
        g.load_list(f)
