"""ScriptedPlanner scripts (tasks/scripts/): correct scripts match the derived truth and photo numbers; each negative
control has the shape its G2 row needs. No sim."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.tasks import SCRIPTS_DIR, load_script  # noqa: E402
from agent.truth import load_tasks  # noqa: E402

TASKS = {t.id: t for t in load_tasks()}
PHOTOS = json.loads((ROOT / "tasks/photo_numbers.json").read_text())["tasks"]
SKILLS = {"navigate_to", "look_around", "report"}
PLACES = {"hall", "ward", "office", "reception", "storage"}


def script(name: str) -> list:
    return load_script(SCRIPTS_DIR / f"{name}.yaml")[1]


def test_one_correct_script_per_dev_task() -> None:
    correct = {p.stem for p in SCRIPTS_DIR.glob("*.yaml") if "." not in p.stem}
    assert correct == set(TASKS)
    assert all(load_script(p)[0] in TASKS for p in SCRIPTS_DIR.glob("*.yaml"))


def test_correct_scripts_report_truth_and_derived_photo() -> None:
    for tid, t in TASKS.items():
        steps = script(tid)
        assert all(isinstance(s, dict) and s["name"] in SKILLS for s in steps), tid
        assert steps[0]["name"] == "navigate_to" and len(steps) <= 12, tid
        rep = steps[-1]
        assert rep["name"] == "report" and [s["name"] for s in steps].count("report") == 1, tid
        assert rep["args"]["answer"] == t.truth_answer and isinstance(rep["args"]["answer"], str), tid
        assert rep["args"]["photo"] == PHOTOS[tid]["photo"], tid
        assert steps[-2]["name"] == "look_around", tid  # the reported photo is a look_around photo
        visited = [s["args"]["place"] for s in steps if s["name"] == "navigate_to"]
        assert visited[-1] == PHOTOS[tid]["room"] and set(visited) <= PLACES, tid


def test_conditional_scripts_follow_the_branch() -> None:
    for tid, t in TASKS.items():
        if t.kind != "conditional":
            continue
        visited = [s["args"]["place"] for s in script(tid) if s["name"] == "navigate_to"]
        want = [t.target["if_not_in"]] + ([] if t.truth_answer == t.target["if_not_in"] else [t.target["else_room"]])
        assert visited == want, tid


def test_photo_numbers_are_proof_photos() -> None:
    for tid, rec in PHOTOS.items():
        assert rec["truth_answer"] == TASKS[tid].truth_answer
        for src in rec["by_source"].values():
            assert src["px_by_k"][rec["photo"] - 1] == max(src["px_by_k"]) >= 1000, tid


def test_neg_wrong_answer() -> None:
    rep = script("find-toolbox.neg-wrong-answer")[-1]["args"]
    assert rep["answer"] != TASKS["find-toolbox"].truth_answer and rep["photo"] == PHOTOS["find-toolbox"]["photo"]


def test_neg_no_proof() -> None:
    rep = script("find-toolbox.neg-no-proof")[-1]["args"]
    assert rep["answer"] == TASKS["find-toolbox"].truth_answer
    assert all(src["px_by_k"][rep["photo"] - 1] == 0 for src in PHOTOS["find-toolbox"]["by_source"].values())


def test_neg_13_steps() -> None:
    steps = script("count-chairs.neg-13-steps")
    assert len(steps) == 13 and [s["name"] for s in steps].index("report") == 12


def test_neg_unknown_call() -> None:
    steps = script("find-toolbox.neg-unknown-call")
    assert isinstance(steps[0], list) and steps[0][0]["function"]["name"] not in SKILLS
    assert steps[1]["name"] == "navigate_to" and steps[1]["args"]["place"] not in PLACES
    assert steps[2:] == script("find-toolbox")


def test_neg_bad_format() -> None:
    for tid, bad in (("state-cabinet-open", "maybe"), ("count-chairs", "four")):
        steps = script(f"{tid}.neg-bad-format")
        assert steps[-2]["args"]["answer"] == bad and steps[:-2] + steps[-1:] == script(tid)
