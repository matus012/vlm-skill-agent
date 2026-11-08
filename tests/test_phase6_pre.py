"""probes/phase6_pre.py (task 009 P1 / P2), no sim: P1 on real run dirs from run_one on the FakeAdapter (fake world of
test_phase2_gates), P2 on test_phase1_gates' FakeRobot."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.fake_adapter import PLACES, ROOMS, FakeAdapter  # noqa: E402
from agent.truth import load_truth, world_truth_path  # noqa: E402
from probes import phase6_pre as p6  # noqa: E402
from test_phase1_gates import PLACES as P1_PLACES, FakeRobot  # noqa: E402
from test_phase2_gates import clean_git, edit_json, edit_step, rk, world  # noqa: E402,F401  (fixtures)

RECT = ROOMS["storage"]  # fake world's storage (0, -6, 5, -2)
STORAGE = PLACES["storage"][:2]


@pytest.fixture(autouse=True)
def closed_world(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIM_WORLD", raising=False)  # run_one's world: closed (the fake world's paths are passed in)


def run_p1(tmp_path: Path, world: dict, ad: FakeAdapter | None = None) -> dict:  # noqa: F811
    ad = ad or FakeAdapter("hall", blocked=(STORAGE,))
    return p6.run_p1(ad, ad.pose, "blocked_storage", "storage", RECT, PLACES, tmp_path, **rk(world))


def dirs(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "p1").iterdir())


# ------------------------------------------------------------------------------------------------ world / cycles
def test_refuses_world_without_blocker() -> None:
    with pytest.raises(SystemExit, match="no 'blocker'"):
        p6.blocked_room(load_truth(world_truth_path("closed")), "closed")


def test_blocked_room_and_rect_from_fake_world(tmp_path: Path) -> None:
    t = tmp_path / "generated_blocked_office" / "truth.json"
    t.parent.mkdir()
    t.write_text(json.dumps({"rooms": {"office": {"polygon": [[0.0, 1.5], [5.0, 1.5], [5.0, 6.5], [0.0, 6.5]]}},
                             "blocker": {"id": "blocker_office", "room": "office", "pose": [2.0, 1.4, 0, 0, 0, 0],
                                         "size": [1.4, 0.3, 1.2], "label": 220}}))
    truth = load_truth(t)
    assert p6.blocked_room(truth, "blocked_office") == "office"
    assert p6.room_rect(truth, "office") == (0.0, 1.5, 5.0, 6.5)
    assert p6.inside([2.0, 1.5], (0.0, 1.5, 5.0, 6.5)) and not p6.inside([2.0, 1.2], (0.0, 1.5, 5.0, 6.5))


def test_start_cycle_hall_first() -> None:
    reach = p6.reachable(PLACES, "storage")
    assert reach == ["hall", "ward", "office", "reception"]
    assert p6.p1_starts(reach) == ["hall", "ward", "office", "reception"] * 2 + ["hall", "ward"]
    assert p6.reachable({"office": 0, "hall": 0, "ward": 0, "reception": 0}, "ward") == ["hall", "office", "reception"]


def test_target_cycle() -> None:
    reach = p6.reachable(PLACES, "office")
    assert p6.p2_targets(reach) == ["ward", "reception", "storage", "hall"] * 2 + ["ward", "reception"]


# ------------------------------------------------------------------------------------------------------------ P1
def test_p1_all_blocked_passes(tmp_path: Path, world: dict) -> None:  # noqa: F811
    v = run_p1(tmp_path, world)
    assert v["pass"] and v["passed"] == v["total"] == 10, [t["problems"] for t in v["tries"]]
    assert [t["start"] for t in v["tries"]] == p6.p1_starts(["hall", "ward", "office", "reception"])
    assert v["time_to_fail"]["nav_sim_s"] == {"min": 10.0, "median": 10.0, "max": 10.0}
    assert v["time_to_fail"]["nav_status"] == ["ABORTED"] * 10 and v["blocked_room"] == "storage"
    assert all((d / "steps.jsonl").exists() for d in dirs(tmp_path))


def test_p1_not_blocked_gets_in(tmp_path: Path, world: dict) -> None:  # noqa: F811
    v = run_p1(tmp_path, world, FakeAdapter("hall"))
    assert not v["pass"] and v["passed"] == 0
    p = v["tries"][0]["problems"]
    assert any("want 'failed: blocked'" in x for x in p) and any("it got in" in x for x in p)


def _one(tmp_path: Path, world: dict, **kv) -> list[str]:  # noqa: ANN003,F811
    run_p1(tmp_path, world)
    d = dirs(tmp_path)[3]
    edit_step(d, 0, **kv)
    return p6.p1_problems(d, "storage", RECT)


def test_p1_slow_fail(tmp_path: Path, world: dict) -> None:  # noqa: F811
    assert _one(tmp_path, world, **{"skill_detail.nav_sim_s": 120.0}) == []
    assert _one(tmp_path / "b", world, **{"skill_detail.nav_sim_s": 120.01}) == [
        f"blocked step: nav_sim_s 120.01, want <= {p6.NAV_TIMEOUT_S} s"]


def test_p1_gt_inside_room(tmp_path: Path, world: dict) -> None:  # noqa: F811
    p = _one(tmp_path, world, **{"truth.gt_pose_after": [2.5, -2.1, 0.0]})
    assert len(p) == 1 and "it got in" in p[0]


def test_p1_hall_not_reached(tmp_path: Path, world: dict) -> None:  # noqa: F811
    run_p1(tmp_path, world)
    d = dirs(tmp_path)[0]
    edit_step(d, 1, result="failed: blocked")
    assert p6.p1_problems(d, "storage", RECT) == ["hall step: result 'failed: blocked', want 'reached' (robot stuck)"]


def test_p1_invalid_start(tmp_path: Path, world: dict) -> None:  # noqa: F811
    run_p1(tmp_path, world)
    d = dirs(tmp_path)[1]
    run = json.loads((d / "run.json").read_text())
    edit_json(d / "run.json", setup={**run["setup"], "reached": False, "gt_err_m": 0.4})
    v = p6.p1_verdict(dirs(tmp_path), "blocked_storage", "storage", RECT)
    assert not v["pass"] and v["passed"] == 9 and "invalid start" in v["tries"][1]["problems"][0]


def test_p1_wrong_script_shape(tmp_path: Path, world: dict) -> None:  # noqa: F811
    run_p1(tmp_path, world)
    d = dirs(tmp_path)[0]
    edit_step(d, 0, call={"name": "navigate_to", "args": {"place": "office"}})
    assert "steps 1-2" in p6.p1_problems(d, "storage", RECT)[0]


def test_p1_dirty_tree_fails(tmp_path: Path, world: dict, clean_git: dict) -> None:  # noqa: F811
    clean_git["git_dirty"] = True
    v = run_p1(tmp_path, world)
    assert v["passed"] == 10 and not v["pass"] and v["git_problems"]


def test_p1_nine_tries_fail(tmp_path: Path, world: dict) -> None:  # noqa: F811
    run_p1(tmp_path, world)
    assert not p6.p1_verdict(dirs(tmp_path)[:9], "blocked_storage", "storage", RECT)["pass"]


# ------------------------------------------------------------------------------------------------------------ P2
def run_p2(monkeypatch: pytest.MonkeyPatch, outcomes: dict | None = None, dirty: bool = False) -> tuple[dict, FakeRobot]:
    robot = FakeRobot(outcomes)
    monkeypatch.setattr(p6.mwc, "gt_pose", robot.gt)
    monkeypatch.setattr(p6, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": dirty})
    return p6.run_p2(robot, P1_PLACES, "blocked_storage", "storage"), robot


def test_p2_all_good(monkeypatch: pytest.MonkeyPatch) -> None:
    v, robot = run_p2(monkeypatch)
    assert v["pass"] and v["passed"] == v["total"] == 10
    assert [t["to"] for t in v["tries"]] == ["ward", "office", "reception", "hall"] * 2 + ["ward", "office"]
    assert [t["from"] for t in v["tries"]][:3] == ["hall", "ward", "office"]
    assert robot.limits[0] == p6.mwc.NAV_SIM_LIMIT_S and set(robot.limits[1:]) == {p6.pg1.G1_MAX_SIM_S}


@pytest.mark.parametrize("outcome, why", [
    ({"status": 6}, "aborted"), ({"end_off": 0.301}, "err"), ({"sim_s": 120.01}, "slow"), ({"start_off": 0.301}, "start")])
def test_p2_each_failure_mode(monkeypatch: pytest.MonkeyPatch, outcome: dict, why: str) -> None:
    v, _ = run_p2(monkeypatch, {3: outcome})
    bad = [t for t in v["tries"] if not t["pass"]]
    assert not v["pass"] and bad and bad[0]["to"] == "reception", why
    if why == "start":
        assert not bad[0]["valid_start"]


def test_p2_boundaries_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    v, _ = run_p2(monkeypatch, {2: {"end_off": 0.300, "sim_s": 120.0}})
    assert v["pass"], [t for t in v["tries"] if not t["pass"]]


def test_p2_dirty_tree_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    v, _ = run_p2(monkeypatch, dirty=True)
    assert v["passed"] == 10 and not v["pass"] and v["git_problems"]
