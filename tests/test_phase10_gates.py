"""probes/phase10_gates.py (task 012) on synthetic run dirs: no sim, no VLM, no test entry. The per-run G1 rule is
replaced by a fake reading score.json; the G3 scan, the G2 git runner and the robot refusal are the real code."""
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
from probes import phase8_gates as p8  # noqa: E402
from probes import phase10_gates as g  # noqa: E402

SPEC = {"seed": 1, "objects": {}, "removed": []}


def entries() -> list[dict[str, Any]]:
    return [{"run_id": f"r{i:03d}", "task_id": "t", "start_place": "hall", "seed": i, "world": "closed",
             "layout_spec": SPEC | {"seed": i}, "expected": {"kind": p8.TYPES[i % 4], "answer": "a", "answer_room": "a"}}
            for i in range(100)]


def write_run(d: Path, passed: bool = True, robot: str = "g1", label: str | None = None, text: str = "ok",
              crash: str | None = None, blocked: bool = False) -> None:
    d.mkdir(parents=True, exist_ok=True)
    run: dict[str, Any] = {"robot": robot, "outcome": "crash" if crash else "reported", "steps": 1, "sim_s": 5.0,
                           "wall_s": 9.0, "node": {"rtf": 1.0, "restarts": 1}, "system_prompt": "sp", "tools": [],
                           "teleport": {}, "crash_origin": "adapter" if crash else None, "crash": crash}
    step = {"step": 1, "call": {"name": "navigate_to" if blocked else "report", "args": {}, "raw": None},
            "result": "failed: blocked" if blocked else "ok", "planner_wall_s": 1.0,
            "skill_detail": {"nav_status": "STALLED", "recentres": 2, "drift_m": 0.05},
            "packet": {"texts": [text], "history": [], "photos": []}, "planner_out": {"content": "x"}}
    (d / "run.json").write_text(json.dumps(run))
    (d / "steps.jsonl").write_text(json.dumps(step) + "\n")
    (d / "score.json").write_text(json.dumps({"pass": passed, "label": None if passed else (label or "vlm"),
                                              "label_why": None}))


def fake_g1_run(e: dict, d: Path | None) -> dict:
    if d is None:
        return {"run_id": e["run_id"], "pass": False, "problems": ["missing"], "label": None}
    s = json.loads((d / "score.json").read_text())
    return {"run_id": e["run_id"], "pass": s["pass"], "problems": [], "label": s["label"]}


@pytest.fixture
def c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    es, out, tb = entries(), tmp_path / "D", tmp_path / "T"
    for root, robot in ((out, "g1"), (tb, "turtlebot4")):
        for e in es:
            write_run(root / e["run_id"], robot=robot)
        (root / "invocation.json").write_text(json.dumps({"invocation_id": "i", **({"robot": "g1"} if root == out else {})}))
    monkeypatch.setattr(p3, "require_merged", lambda o: None)
    monkeypatch.setattr(p4, "g1_run", fake_g1_run)
    monkeypatch.setattr(p4, "score_run", lambda d: json.loads((d / "score.json").read_text()))
    monkeypatch.setattr(p3, "run_problems", lambda d: [])
    monkeypatch.setattr(p5, "common_problems", lambda *a: [])
    return {"out": out, "tb": tb, "entries": es}


# ------------------------------------------------------------------------------------------------ robot refusal
def test_turtlebot4_run_in_d_is_refused(c: dict) -> None:
    g.require_robot(c["out"], c["entries"])
    write_run(c["out"] / "r007", robot="turtlebot4")
    with pytest.raises(SystemExit, match="r007"):
        g.require_robot(c["out"], c["entries"])
    for fn in (lambda: g.side(c["out"], c["tb"], c["entries"]), lambda: g.measure(c["out"], c["entries"]),
               lambda: g.diag(c["out"], c["entries"])):
        with pytest.raises(SystemExit):
            fn()


def test_invocation_robot_must_be_g1(c: dict) -> None:
    (c["out"] / "invocation.json").write_text(json.dumps({"invocation_id": "i"}))  # TB4 invocations carry no robot key
    with pytest.raises(SystemExit, match="invocation robot"):
        g.require_robot(c["out"], c["entries"])


# ------------------------------------------------------------------------------------------------------- G2
def git(cwd: Path, *a: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=cwd, check=True, capture_output=True)


def test_swap_diff_real_git_one_line_agent_diff_fails(tmp_path: Path) -> None:
    r = tmp_path / "repo"
    (r / "agent").mkdir(parents=True)
    (r / "agent" / "a.py").write_text("x = 1\n")
    (r / "other.py").write_text("y = 1\n")
    git(r, "init", "-q")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "base")
    git(r, "tag", "frozen")
    (r / "other.py").write_text("y = 2\n")  # outside the pathspec: ignored
    git(r, "commit", "-qam", "other")
    assert g.swap_diff("frozen", cwd=r) == ""
    (r / "agent" / "a.py").write_text("x = 2\n")
    git(r, "commit", "-qam", "agent")
    assert len(g.swap_diff("frozen", cwd=r).splitlines()) > 0


def test_g2_verdict_diff_pin_freeze(c: dict) -> None:
    ok_fz, ok_pin = {"pass": True}, {"pass": True}
    v = g.g2_verdict(c["out"], c["entries"], "t", "", ok_fz, ok_pin)
    assert v["pass"] and v["diff_lines"] == 0 and (c["out"] / "g2_swap_diff.txt").read_text() == ""
    diff = "diff --git a/agent/x.py b/agent/x.py\n+1\n"
    v = g.g2_verdict(c["out"], c["entries"], "t", diff, ok_fz, ok_pin)
    assert not v["pass"] and (c["out"] / "g2_swap_diff.txt").read_text() == diff
    assert not g.g2_verdict(c["out"], c["entries"], "t", "", {"pass": False}, ok_pin)["pass"]
    assert not g.g2_verdict(c["out"], c["entries"], "t", "", ok_fz, {"pass": False})["pass"]


def test_list_pin(tmp_path: Path) -> None:
    f = tmp_path / "l.yaml"
    f.write_text("x")
    sha = g.sha256_file(f)
    assert g.list_pin(f, {Path("/a/l.yaml"): sha})["pass"]
    assert not g.list_pin(f, {Path("/a/l.yaml"): "0" * 64})["pass"]
    assert not g.list_pin(f, {})["pass"]
    assert g.list_pin(g.LIST)["pass"] is True or not g.LIST.exists()  # sha of the pinned file only, never its content


# ------------------------------------------------------------------------------------------------------- G3
NO_ROS = {"returncode": 0}


def test_g3_clean_passes_unitree_in_packet_fails(c: dict) -> None:
    assert g.g3_verdict(c["out"], c["entries"], True, NO_ROS)["pass"]
    write_run(c["out"] / "r003", text="You are the Unitree robot")
    v = g.g3_verdict(c["out"], c["entries"], True, NO_ROS)
    assert not v["pass"] and "r003" in v["failing_hits"]
    assert not g.g3_verdict(c["out"], c["entries"], True, {"returncode": 1})["pass"]


# ------------------------------------------------------------------------------------------------------ side
def test_side_agreement_labels_and_hosting_drop(c: dict) -> None:
    write_run(c["tb"] / "r000", passed=False)  # find: G1 only
    write_run(c["out"] / "r004", passed=False, label="nav")  # find: TB only
    write_run(c["tb"] / "r008", passed=False)
    write_run(c["out"] / "r008", passed=False, label="skill")  # find: neither
    s = g.side(c["out"], c["tb"], c["entries"])
    f = s["types"]["find"]
    assert (f["n"], f["tb4"], f["g1"], f["both"], f["tb4_only"], f["g1_only"], f["neither"]) == (25, 23, 23, 22, 1, 1, 1)
    assert f["agreement_pct"] == 92.0
    row = {r["run_id"]: r for r in s["runs"]}
    assert row["r004"]["g1_label"] == "nav" and row["r000"]["g1_label"] is None
    assert "prompt" not in json.dumps(s)
    # hosting drop: r004 leaves both columns (n, tb4 and g1 all lose it)
    s = g.side(c["out"], c["tb"], c["entries"], {"r004"})
    f = s["types"]["find"]
    assert f["n"] == 24 and f["tb4"] == 22 and f["g1"] == 23 and f["tb4_only"] == 0 and s["dropped"] == ["r004"]
    assert all(r["run_id"] != "r004" for r in s["runs"])


def test_side_counts_falls_and_stalls_separately(c: dict) -> None:
    write_run(c["out"] / "r001", passed=False, label="sim",
              crash="Traceback ...\nhumanoid.drive.RobotFell: base z 0.31 < 0.5")  # check_state
    write_run(c["out"] / "r005", passed=False, label="sim", crash="Traceback ...\nOdomHang: no odom")  # other sim
    write_run(c["out"] / "r009", passed=False, label="nav", blocked=True)  # stall
    t = g.side(c["out"], c["tb"], c["entries"])["types"]["check_state"]
    assert (t["g1_falls"], t["g1_other_sim"], t["g1_stalls"]) == (1, 1, 1)
    assert g.side(c["out"], c["tb"], c["entries"])["types"]["find"]["g1_falls"] == 0


# ------------------------------------------------------------------------------------- measure / diag
def test_measure_extras_and_missing_keys(c: dict) -> None:
    write_run(c["out"] / "r001", passed=False, label="sim", crash="RobotFell")
    m = g.measure(c["out"], c["entries"])
    x = m["g1"]
    assert x["falls"] == 1 and x["recentres"] == 200 and x["max_nondrive_step_drift_m"] == 0.05
    assert x["robot_resets"] == 100 and x["node_restarts"] == 1
    run = json.loads((c["out"] / "r002" / "run.json").read_text())
    for k in ("node", "teleport"):
        run.pop(k)
    (c["out"] / "r002" / "run.json").write_text(json.dumps(run))
    assert g.measure(c["out"], c["entries"])["g1"]["robot_resets"] == 99


def test_diag_aggregates_only(c: dict) -> None:
    write_run(c["out"] / "r009", passed=False, label="nav", blocked=True)
    write_run(c["out"] / "r010", passed=False, label="nav", blocked=True)
    d = g.diag(c["out"], c["entries"])
    assert d["failed"] == 2 and d["groups"] == [
        {"label": "nav", "crash_origin": "None", "nav_status": "STALLED", "step_skill": "navigate_to", "n": 2}]
    assert "r009" not in json.dumps(d)


def test_repeat_look_arounds_counts_same_place_sets_only():
    from probes.phase10_gates import repeat_look_arounds
    st = lambda name, result="ok": {"call": {"name": name}, "result": result}  # noqa: E731
    assert repeat_look_arounds([st("look_around"), st("navigate_to", "reached"), st("look_around")]) == 0
    assert repeat_look_arounds([st("look_around"), st("look_around"), st("look_around")]) == 2
    assert repeat_look_arounds([st("look_around"), st("navigate_to", "failed: blocked"), st("look_around")]) == 1
    assert repeat_look_arounds([st("navigate_to", "reached"), st("look_around"), st("report")]) == 0
