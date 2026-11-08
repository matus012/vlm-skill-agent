"""Scorer (INTERFACES section 7), tasks (section 6) and replay on synthetic run directories (section 5). No sim."""
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import scorer  # noqa: E402
from agent.truth import TaskError, derive, load_tasks  # noqa: E402

H, W = 377, 640  # cropped TB4 frame (002: rows 59..436 of 480)
# Labels deliberately != list index + 1 (the real truth has label == index + 1, so it cannot catch a wrong source).
SYN_TRUTH = {
    "layout_sha256": "synthetic", "world": "synthetic", "structure_labels": {"wall": 240},
    "rooms": {r: {"polygon": []} for r in ("hall", "storage", "reception", "office")},
    "objects": [
        {"id": "toolbox_red_1", "class": "toolbox", "colour": "red", "label": 40, "room": "storage", "state": None},
        {"id": "cabinet_blue_1", "class": "cabinet", "colour": "blue", "label": 41, "room": "storage",
         "state": "closed"},
        {"id": "chair_1", "class": "chair", "colour": "orange", "label": 50, "room": "reception", "state": None},
        {"id": "chair_2", "class": "chair", "colour": "orange", "label": 51, "room": "reception", "state": None},
        {"id": "chair_3", "class": "chair", "colour": "grey", "label": 52, "room": "office", "state": None},
        {"id": "plant_1", "class": "plant", "colour": "green", "label": 60, "room": "storage", "state": None},
    ]}
SYN_PLACES = {"places": {p: {"x": 0.0, "y": 0.0, "yaw": 0.0} for p in ("hall", "storage", "reception", "office")}}
TOOLBOX = ("find", {"object": "toolbox_red_1"})


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    t, p = tmp_path / "truth.json", tmp_path / "places.yaml"
    t.write_text(json.dumps(SYN_TRUTH))
    p.write_text(yaml.safe_dump(SYN_PLACES))
    return {"truth_path": t, "places_path": p}


def seg_png(path: Path, px: dict[int, int]) -> None:
    flat = np.zeros(H * W, np.uint8)
    i = 0
    for lab, n in px.items():
        flat[i:i + n] = lab
        i += n
    Image.fromarray(flat.reshape(H, W)).save(path)


def step_rec(n: int, name: str | None, args: dict | None, check: str, result: str | None,
             photos: list[dict]) -> dict[str, Any]:
    return {"step": n, "packet": {"room": "storage", "history": [], "last_result": "start", "photos": photos},
            "call": {"raw": [], "name": name, "args": args}, "check": check, "result": result, "photos_out": [],
            "sim_t0": 0.0, "sim_t1": 1.0, "wall_t0": 0.0, "wall_t1": 1.0, "skill_sim_s": 1.0, "skill_wall_s": 1.0,
            "truth": {"gt_pose_after": [0, 0, 0]}, "planner_out": {"content": None, "reasoning": None,
                                                                "finish_reason": None, "usage": None, "error": None}}


def make_run(d: Path, task: tuple[str, dict], answer: str | None, photo: int = 1,
             px: dict[int, dict[int, int]] | None = None, n_steps: int = 3, outcome: str = "reported",
             hand_help: bool = False, blocked_place: str | None = None) -> Path:
    """A run dir: steps 1..n-1 look_around (step 1 navigate_to if blocked_place), step n = report(answer, photo)
    if answer is not None. px: photo number -> {label: pixels} of the packet the report sees (default 4 empty)."""
    (d / "photos").mkdir(parents=True)
    px = px or {}
    photos = []
    for n in range(1, 5):
        seg_png(d / f"photos/0_{n - 1}_seg.png", px.get(n, {}))
        Image.new("RGB", (W, H)).save(d / f"photos/0_{n - 1}_rgb.png")
        photos.append({"n": n, "rgb": f"photos/0_{n - 1}_rgb.png", "seg": f"photos/0_{n - 1}_seg.png"})
    steps = [step_rec(i, "look_around", {}, "ok", "4 photos", photos) for i in range(1, n_steps + 1)]
    if blocked_place:
        steps[0] = step_rec(1, "navigate_to", {"place": blocked_place}, "ok", "failed: blocked", photos)
    report = None
    if answer is not None:
        report = {"answer": answer, "photo": photo, "note": ""}
        steps[-1] = step_rec(n_steps, "report", report, "ok", "reported", photos)
    run = {"task": {"id": "t", "kind": task[0], "prompt": "p", "target": task[1]}, "planner": "scripted:t",
           "robot": "fake", "layout_sha256": "synthetic", "world_tag": "x", "git_commit": "x", "git_dirty": False,
           "start_place": "hall", "start_utc": "x", "end_utc": "x", "outcome": outcome, "steps": n_steps,
           "report": report, "sim_s": 1.0, "wall_s": 1.0,
           "setup": {"start_place": "hall", "reached": True, "gt_err_m": 0.0}, "hand_help": hand_help}
    (d / "run.json").write_text(json.dumps(run))
    (d / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    return d


def failed(sc: dict) -> set[str]:
    return {k for k, v in sc["checks"].items() if not v}


# ---------- each check fails for its reason, and only that one ----------

def test_correct_run_passes(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, "storage", 3, {3: {40: 9100}}), **world)
    assert sc["pass"] and failed(sc) == set() and sc["label"] is None and sc["label_why"] is None
    assert sc["truth_answer"] == "storage" and sc["proof_px"] == 9100 and sc["proof_label"] == 40


def test_score_json_keys(tmp_path: Path, world: dict) -> None:
    d = make_run(tmp_path / "r", TOOLBOX, "storage", 3, {3: {40: 9100}})
    sc = scorer.write_score(d, **world)
    assert json.loads((d / "score.json").read_text()) == sc
    assert {"pass", "checks", "truth_answer", "proof_px", "label", "label_why"} <= set(sc)
    assert set(sc["checks"]) == {"answer", "proof", "steps", "clean"}


def test_wrong_answer_fails_only_answer(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, "office", 3, {3: {40: 9100}}), **world)
    assert failed(sc) == {"answer"} and sc["label"] == "vlm" and "office" in sc["label_why"]


def test_proof_photo_without_object_fails_only_proof(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, "storage", 1, {1: {60: 30000}, 3: {40: 9100}}), **world)
    assert failed(sc) == {"proof"} and sc["proof_px"] == 0 and sc["label"] == "vlm"
    assert "proof photo 1" in sc["label_why"]


@pytest.mark.parametrize("px,ok", [(1000, True), (999, False), (1001, True)])
def test_proof_threshold_exactly_1000(tmp_path: Path, world: dict, px: int, ok: bool) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, "storage", 2, {2: {40: px}}), **world)
    assert sc["checks"]["proof"] is ok and sc["proof_px"] == px


def test_proof_is_one_label_not_a_sum(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", ("count", {"class": "chair", "room": "reception"}), "2", 1,
                                   {1: {50: 600, 51: 600}}), **world)
    assert failed(sc) == {"proof"} and sc["proof_px"] == 600


def test_count_proof_only_from_the_counted_room(tmp_path: Path, world: dict) -> None:
    task = ("count", {"class": "chair", "room": "reception"})
    sc = scorer.score_run(make_run(tmp_path / "a", task, "2", 1, {1: {52: 5000}}), **world)  # office chair
    assert failed(sc) == {"proof"}
    sc = scorer.score_run(make_run(tmp_path / "b", task, "2", 1, {1: {51: 5000}}), **world)
    assert sc["pass"] and sc["proof_label"] == 51


def test_proof_uses_the_reported_photo_number(tmp_path: Path, world: dict) -> None:
    px = {2: {40: 5000}}
    assert scorer.score_run(make_run(tmp_path / "a", TOOLBOX, "storage", 2, px), **world)["pass"]
    for n in (1, 3):
        assert failed(scorer.score_run(make_run(tmp_path / f"b{n}", TOOLBOX, "storage", n, px), **world)) == {"proof"}


def test_twelve_steps_pass_thirteen_fail_only_steps(tmp_path: Path, world: dict) -> None:
    px = {3: {40: 9100}}
    assert scorer.score_run(make_run(tmp_path / "a", TOOLBOX, "storage", 3, px, n_steps=12), **world)["pass"]
    sc = scorer.score_run(make_run(tmp_path / "b", TOOLBOX, "storage", 3, px, n_steps=13), **world)
    assert failed(sc) == {"steps"} and sc["label"] == "vlm"


def test_step_limit_without_report(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, None, n_steps=12, outcome="fail_step_limit"), **world)
    assert failed(sc) == {"answer", "proof", "steps"} and sc["label"] == "vlm" and "no report" in sc["label_why"]


def test_step_check_needs_outcome_reported(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, "storage", 3, {3: {40: 9100}},
                                   outcome="fail_step_limit"), **world)
    assert failed(sc) == {"steps"}


def test_hand_help_fails_only_clean(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, "storage", 3, {3: {40: 9100}}, hand_help=True), **world)
    assert failed(sc) == {"clean"}


def test_crash_fails_clean_and_is_labelled_sim(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, None, n_steps=2, outcome="crash"), **world)
    assert sc["checks"]["clean"] is False and sc["checks"]["steps"] is False and sc["label"] == "sim"


def test_crash_label_wins_over_blocked(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, None, n_steps=2, outcome="crash",
                                   blocked_place="storage"), **world)
    assert sc["label"] == "sim"


def test_blocked_on_reachable_place_is_nav(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, None, n_steps=12, outcome="fail_step_limit",
                                   blocked_place="storage"), **world)
    assert sc["label"] == "nav" and "storage" in sc["label_why"]


def test_blocked_on_unreachable_test_place_is_vlm(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, None, n_steps=12, outcome="fail_step_limit",
                                   blocked_place="in_wall"), **world)
    assert sc["label"] == "vlm"


def test_blocked_then_correct_report_passes(tmp_path: Path, world: dict) -> None:
    sc = scorer.score_run(make_run(tmp_path / "r", TOOLBOX, "storage", 3, {3: {40: 9100}},
                                   blocked_place="storage"), **world)
    assert sc["pass"] and sc["label"] is None


def test_exactly_one_label_on_every_failure(tmp_path: Path, world: dict) -> None:
    runs = [make_run(tmp_path / "a", TOOLBOX, "office", 3, {3: {40: 9100}}),
            make_run(tmp_path / "b", TOOLBOX, None, n_steps=2, outcome="crash"),
            make_run(tmp_path / "c", TOOLBOX, None, n_steps=12, outcome="fail_step_limit", blocked_place="office")]
    for d in runs:
        sc = scorer.score_run(d, **world)
        assert not sc["pass"] and sc["label"] in ("vlm", "skill", "nav", "sim", "scorer")
        assert isinstance(sc["label_why"], str) and sc["label_why"] and "\n" not in sc["label_why"]


def test_layout_mismatch_refuses(tmp_path: Path, world: dict) -> None:
    d = make_run(tmp_path / "r", TOOLBOX, "storage", 3, {3: {40: 9100}})
    run = json.loads((d / "run.json").read_text())
    (d / "run.json").write_text(json.dumps({**run, "layout_sha256": "other"}))
    with pytest.raises(ValueError, match="layout"):
        scorer.score_run(d, **world)


def test_check_state_and_conditional_scoring(tmp_path: Path, world: dict) -> None:
    st = ("check_state", {"object": "cabinet_blue_1", "state": "open"})
    assert scorer.score_run(make_run(tmp_path / "a", st, "no", 4, {4: {41: 40000}}), **world)["pass"]
    assert failed(scorer.score_run(make_run(tmp_path / "b", st, "yes", 4, {4: {41: 40000}}), **world)) == {"answer"}
    cond = ("conditional", {"object": "toolbox_red_1", "if_not_in": "office", "else_room": "storage"})
    assert scorer.score_run(make_run(tmp_path / "c", cond, "storage", 3, {3: {40: 9100}}), **world)["pass"]
    assert failed(scorer.score_run(make_run(tmp_path / "d", cond, "office", 3, {3: {40: 9100}}), **world)) == {
        "answer"}


def test_class_target_find_proof_is_any_instance(tmp_path: Path, world: dict) -> None:
    task = ("find", {"class": "chair", "colour": "orange"})
    sc = scorer.score_run(make_run(tmp_path / "a", task, "reception", 2, {2: {51: 3000}}), **world)
    assert sc["pass"] and sc["truth_answer"] == "reception" and sc["proof_label"] == 51
    sc = scorer.score_run(make_run(tmp_path / "b", task, "reception", 2, {2: {52: 3000}}), **world)  # grey chair
    assert failed(sc) == {"proof"}


# ---------- truth derivation ----------

def test_synthetic_truth_derivation() -> None:
    assert derive("count", {"class": "chair", "room": "reception"}, SYN_TRUTH) == ("2", (50, 51))
    assert derive("count", {"class": "chair", "room": "office"}, SYN_TRUTH) == ("1", (52,))
    assert derive("find", {"object": "cabinet_blue_1"}, SYN_TRUTH) == ("storage", (41,))
    assert derive("check_state", {"object": "cabinet_blue_1", "state": "closed"}, SYN_TRUTH) == ("yes", (41,))
    assert derive("find", {"class": "chair", "colour": "orange"}, SYN_TRUTH) == ("reception", (50, 51))
    assert derive("find", {"class": "toolbox"}, SYN_TRUTH) == ("storage", (40,))
    assert derive("conditional", {"class": "chair", "colour": "orange", "if_not_in": "reception",
                                  "else_room": "office"}, SYN_TRUTH) == ("reception", (50, 51))


@pytest.mark.parametrize("kind,target,match", [
    ("conditional", {"object": "toolbox_red_1", "if_not_in": "office", "else_room": "reception"}, "neither"),
    ("conditional", {"object": "toolbox_red_1", "if_not_in": "storage", "else_room": "storage"}, "different"),
    ("find", {"object": "plant_1"}, "not a target class"),
    ("find", {"object": "nope"}, "found 0 times"),
    ("count", {"class": "plant", "room": "storage"}, "not a target class"),
    ("count", {"class": "bed", "room": "storage"}, "count 0"),
    ("count", {"class": "chair", "room": "attic"}, "unknown room"),
    ("check_state", {"object": "toolbox_red_1", "state": "open"}, "no state"),
    ("check_state", {"object": "cabinet_blue_1", "state": "ajar"}, "not in"),
    ("find", {"object": "toolbox_red_1", "room": "x"}, "keys"),
    ("find", {"class": "chair"}, "2 rooms"),  # orange chairs in reception, a grey one in the office
    ("conditional", {"class": "chair", "if_not_in": "reception", "else_room": "office"}, "2 rooms"),
    ("find", {"class": "chair", "colour": "purple"}, "0 rooms"),
    ("find", {"class": "plant"}, "not a target class"),
    ("find", {"class": "chair", "object": "chair_1"}, "keys"),
    ("find", {"colour": "orange"}, "keys"),
    ("count", {"class": "chair", "colour": "orange", "room": "reception"}, "keys"),
    ("explore", {"object": "toolbox_red_1"}, "unknown kind"),
])
def test_invalid_tasks_fail(kind: str, target: dict, match: str) -> None:
    with pytest.raises(TaskError, match=match):
        derive(kind, target, SYN_TRUTH)


def test_invalid_conditional_fails_loading(tmp_path: Path) -> None:
    y = tmp_path / "dev.yaml"
    y.write_text(yaml.safe_dump([{"id": "c", "kind": "conditional", "prompt": "p",
                                  "target": {"object": "toolbox_red_1", "if_not_in": "office", "else_room": "ward"}}]))
    with pytest.raises(TaskError, match="neither"):
        load_tasks(y)


def test_dev_tasks_truth() -> None:
    """Every dev task's derived truth vs the clinic layout read by hand (truth.json, layout 17db3394...)."""
    got = {t.id: (t.kind, t.truth_answer, t.proof_labels) for t in load_tasks()}
    assert got == {
        "find-toolbox": ("find", "storage", (13,)),
        "find-cabinet": ("find", "storage", (14,)),
        "state-cabinet-open": ("check_state", "no", (14,)),
        "state-cabinet-closed": ("check_state", "yes", (14,)),
        "count-chairs": ("count", "4", (6, 7, 8, 9)),
        "count-beds": ("count", "3", (1, 2, 3)),
        "cond-toolbox": ("conditional", "storage", (13,)),
        "cond-toolbox-office": ("conditional", "storage", (13,)),
        "cond-cabinet-office": ("conditional", "storage", (14,)),
        "find-beds": ("find", "ward", (1, 2, 3)),
        "find-chairs": ("find", "reception", (6, 7, 8, 9)),
        "find-beds-where": ("find", "ward", (1, 2, 3)),
        "find-chairs-plain": ("find", "reception", (6, 7, 8, 9)),
        "cond-beds-ward": ("conditional", "ward", (1, 2, 3)),
        "cond-chairs-reception": ("conditional", "reception", (6, 7, 8, 9)),
    }


def test_dev_set_coverage() -> None:
    tasks = load_tasks()
    for kind in ("find", "check_state", "count", "conditional"):
        assert sum(t.kind == kind for t in tasks) >= 2, kind
    cond = [t for t in tasks if t.kind == "conditional"]
    assert any(t.truth_answer == t.target["if_not_in"] for t in cond)
    assert any(t.truth_answer == t.target["else_room"] for t in cond)
    assert {t.truth_answer for t in tasks if t.kind == "check_state"} == {"yes", "no"}
    for kind in ("find", "conditional"):  # a constant-room planner cannot pass every find / conditional
        assert {t.truth_answer for t in tasks if t.kind == kind} == {"storage", "ward", "reception"}, kind
    branch = {(t.truth_answer, t.truth_answer == t.target["if_not_in"]) for t in cond}
    assert ("ward", False) in branch and ("reception", True) in branch


# ---------- replay ----------

def test_replay_agrees_and_detects_a_changed_verdict(tmp_path: Path) -> None:
    """Uses the real truth (replay's CLI has no truth override): a dev task with its real labels."""
    truth = json.loads((ROOT / "sim/worlds/clinic/generated/truth.json").read_text())
    d = make_run(tmp_path / "r", TOOLBOX, "storage", 3, {3: {13: 9100}})
    run = json.loads((d / "run.json").read_text())
    (d / "run.json").write_text(json.dumps({**run, "layout_sha256": truth["layout_sha256"]}))
    scorer.write_score(d)
    cmd = [sys.executable, str(ROOT / "probes/replay.py"), str(d)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0 and "SAME" in p.stdout, p.stdout + p.stderr
    sc = json.loads((d / "score.json").read_text())
    (d / "score.json").write_text(json.dumps({**sc, "pass": False, "label": "vlm"}))
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 1 and "DIFFERENT" in p.stdout and "pass" in p.stdout
