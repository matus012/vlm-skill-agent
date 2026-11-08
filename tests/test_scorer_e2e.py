"""End to end on the fake adapter: the real tasks/scripts files run through runtime.run, scored, replayed.

The fake world (agent/fake_adapter.py) has its own labels and geometry, so a synthetic truth.json matching it is
written here, and the report photo of each real script is mapped to the fake's view (toolbox ahead of the storage
place = photo 1, nothing at heading 90 deg = photo 2, cabinet at 270 deg = photo 4)."""
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import runtime, scorer  # noqa: E402
from agent.fake_adapter import PLACES, FakeAdapter  # noqa: E402
from agent.planners import ScriptedPlanner  # noqa: E402
from agent.runlog import check_complete  # noqa: E402
from agent.skills import SkillContext  # noqa: E402
from agent.tasks import SCRIPTS_DIR, load_script  # noqa: E402
from agent.truth import derive, load_tasks  # noqa: E402
from probes.replay import replay  # noqa: E402

LAYOUT = "f" * 64
FAKE_TRUTH = {"layout_sha256": LAYOUT, "rooms": {r: {} for r in ("hall", "ward", "reception", "office", "storage")},
              "objects": [
                  {"id": "toolbox_red_1", "class": "toolbox", "label": 10, "room": "storage", "state": None},
                  {"id": "cabinet_blue_1", "class": "cabinet", "label": 30, "room": "storage", "state": "closed"},
                  {"id": "chair_orange_1", "class": "chair", "label": 20, "room": "reception", "state": None},
                  {"id": "chair_orange_2", "class": "chair", "label": 21, "room": "reception", "state": None},
                  {"id": "bed_blue_1", "class": "bed", "label": 1, "room": "ward", "state": None}]}
META = {"robot": "fake", "layout_sha256": LAYOUT, "world_tag": "fake", "git_commit": "abc", "git_dirty": False,
        "start_place": "hall", "setup": {"start_place": "hall", "reached": True, "gt_err_m": 0.0}}
ROOMS = ["hall", "ward", "reception", "office", "storage"]
FAKE_PHOTO = {3: 1, 1: 2, 4: 4}  # real script photo -> fake world photo with the same role
# script -> (pass, failed checks, label)
CASES = {
    "find-toolbox": (True, set(), None),
    "find-toolbox.neg-wrong-answer": (False, {"answer"}, "vlm"),
    "find-toolbox.neg-no-proof": (False, {"proof"}, "vlm"),
    "find-toolbox.neg-unknown-call": (True, set(), None),
    "state-cabinet-open.neg-bad-format": (True, set(), None),
    "count-chairs.neg-13-steps": (False, {"answer", "proof", "steps"}, "vlm"),
}


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    t, p = tmp_path / "truth.json", tmp_path / "places.yaml"
    t.write_text(json.dumps(FAKE_TRUTH))
    p.write_text(yaml.safe_dump({"places": {n: {"x": x, "y": y, "yaw": a} for n, (x, y, a) in PLACES.items()}}))
    return {"truth_path": t, "places_path": p}


@pytest.mark.parametrize("name", CASES)
def test_script_on_fake_scored(tmp_path: Path, world: dict, name: str) -> None:
    tid, steps = load_script(SCRIPTS_DIR / f"{name}.yaml")
    real = {t.id: t for t in load_tasks()}[tid]
    answer, labels = derive(real.kind, real.target, FAKE_TRUTH)
    task = type(real)(real.id, real.kind, real.prompt, real.target, answer, labels)
    steps = [{**s, "args": {**s["args"], "photo": FAKE_PHOTO[s["args"]["photo"]]}}
             if isinstance(s, dict) and s["name"] == "report" else s for s in steps]
    if tid == "count-chairs":  # the fake reception has 2 chairs, the clinic 4
        steps = [{**s, "args": {**s["args"], "answer": "2"}} if isinstance(s, dict) and s["name"] == "report"
                 and s["args"]["answer"] == "4" else s for s in steps]
    ad = FakeAdapter("hall")
    d = tmp_path / "run"
    runtime.run(task, SkillContext(ad, PLACES, task.kind, ROOMS), ScriptedPlanner(steps, name), d, META,
                truth=ad.truth)
    assert check_complete(d) == []
    sc = scorer.write_score(d, **world)
    ok, failed, label = CASES[name]
    assert sc["pass"] is ok and {k for k, v in sc["checks"].items() if not v} == failed and sc["label"] == label, sc
    log = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines()]
    errors = [r["check"] for r in log if r["check"] != "ok"]
    if ".neg-unknown-call" in name or ".neg-bad-format" in name:
        assert len(errors) == 2 if "unknown" in name else len(errors) == 1, errors
        assert all(e.startswith("error:") for e in errors)
        assert json.loads((d / "run.json").read_text())["outcome"] == "reported"
    if "13-steps" in name:
        assert len(log) == 12 and json.loads((d / "run.json").read_text())["outcome"] == "fail_step_limit"
    assert replay(d, **world) == (True, [])
