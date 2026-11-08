"""N7 (005 Step 0): hand_help is set by a GT teleport or an /initialpose message during the run, not by driving."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import adapters.run_task as rt  # noqa: E402
from adapters.hand_help import SLACK_M, V_MAX_MPS, HandHelp, InitialPoseWatch, count_messages  # noqa: E402
from agent.fake_adapter import FakeAdapter  # noqa: E402
from agent.planners import ScriptedPlanner  # noqa: E402
from test_phase2_gates import clean_git, fake_task, rk, world  # noqa: E402,F401


def drive(hh: HandHelp, samples: list[tuple[float, tuple[float, float], bool]]) -> dict:
    for i, (t, xy, drove) in enumerate(samples):
        hh.gt(i, t, xy, drove)
    return hh.verdict()


def test_normal_driving_is_not_hand_help() -> None:
    """navigate_to at full speed, then look_around / report with small in-place drift."""
    v = drive(HandHelp(), [(0.0, (0.0, 0.0), False), (40.0, (V_MAX_MPS * 40.0, 0.0), True),
                           (60.0, (V_MAX_MPS * 40.0 + 0.05, 0.03), False), (61.0, (V_MAX_MPS * 40.0 + 0.05, 0.03), False)])
    assert v == {"hand_help": False, "hand_help_why": []}


def test_teleport_in_a_non_driving_step() -> None:
    v = drive(HandHelp(), [(0.0, (0.0, 0.0), False), (20.0, (0.0, SLACK_M + 0.05), False)])
    assert v["hand_help"] is True and "no drive skill" in v["hand_help_why"][0]


def test_teleport_faster_than_the_robot_drives() -> None:
    v = drive(HandHelp(), [(0.0, (0.0, 0.0), False), (10.0, (V_MAX_MPS * 10.0 + SLACK_M + 0.1, 0.0), True)])
    assert v["hand_help"] is True and "navigate_to" in v["hand_help_why"][0]


def test_initialpose_message() -> None:
    hh = HandHelp()
    hh.initialpose(0)
    assert hh.verdict()["hand_help"] is False
    hh.initialpose(count_messages("header:\n  frame_id: map\npose: {}\n---\n"))
    assert hh.verdict() == {"hand_help": True, "hand_help_why": ["1 /initialpose message(s) during the run"]}


def test_initialpose_watch_counts_published_messages(tmp_path: Path) -> None:
    """The watch process (a stand-in for `ros2 topic echo`) prints two messages; stop() counts them."""
    w = InitialPoseWatch(tmp_path / "ip.log", ["sh", "-c", "printf 'a: 1\\n---\\nb: 2\\n---\\n'; sleep 30"])
    import time
    time.sleep(0.5)
    assert w.stop() == 2


def _run(world: dict, d: Path, ad: FakeAdapter, gt) -> dict:  # noqa: ANN001
    steps = [{"name": "look_around", "args": {}}, {"name": "report", "args": {"answer": "hall", "photo": 1, "note": ""}}]
    rt.run_one(ad, gt, fake_task("find-toolbox"), ScriptedPlanner(steps, "hh"), d, robot="fake", **rk(world))
    return json.loads((d / "run.json").read_text())


def test_harness_sets_hand_help_on_a_teleport(tmp_path: Path, world: dict) -> None:
    """A GT pose that jumps 2 m during look_around (no drive) -> run.json hand_help true, scorer clean false."""
    ad = FakeAdapter("ward")
    calls = []

    def gt() -> tuple[float, float, float]:
        calls.append(1)
        return (ad.x + (2.0 if len(calls) > 1 else 0.0), ad.y, ad.yaw)  # setup: 1 sample, then step 1 jumps

    run = _run(world, tmp_path / "r", ad, gt)
    sc = json.loads((tmp_path / "r" / "score.json").read_text())
    assert run["hand_help"] is True and "step 1" in run["hand_help_why"][0] and sc["checks"]["clean"] is False


def test_harness_no_hand_help_when_driving_normally(tmp_path: Path, world: dict) -> None:
    ad = FakeAdapter("ward")
    run = _run(world, tmp_path / "r", ad, ad.pose)
    assert run["hand_help"] is False and run["hand_help_why"] == []
