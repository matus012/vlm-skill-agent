"""Task 012 P3 (design section 2): adapters/robots.make, the ROBOT key of probes/phase3_run invocations and its
refusals, run_one's place_objects / trace_factory hooks. No sim, no ROS: adapters.turtlebot4 and the humanoid modules
are stubs (sys.modules), coded against the design's signatures."""
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import adapters.run_task as rt  # noqa: E402
from adapters import robots  # noqa: E402
from adapters.hand_help import HandHelp  # noqa: E402
from agent.fake_adapter import FakeAdapter  # noqa: E402
from agent.planners import ScriptedPlanner  # noqa: E402
from agent.truth import WORLD_DIRS  # noqa: E402
from probes import phase3_run as pr  # noqa: E402
from test_phase3_run import Sim  # noqa: E402
from test_phase3_run import main as p3main  # noqa: E402
from test_seeded_layout import GOOD, SEEDED, TASK, FakeSeeded, world  # noqa: E402,F401


@pytest.fixture(autouse=True)
def clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for m in (pr, rt):
        monkeypatch.setattr(m, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
    for k in ("SLURM_JOB_ID", "ROBOT", "SIM_WORLD"):
        monkeypatch.delenv(k, raising=False)


class Closer:
    def __init__(self, log: list[str], name: str, **kw: Any) -> None:
        self.log, self.name, self.kw = log, name, kw

    def gt_pose(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def close(self) -> None:
        self.log.append(f"close {self.name}")


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """adapters.turtlebot4 and humanoid.{adapter_g1, scene_g1, sim_truth_g1} as stubs recording their calls."""
    log: list[str] = []

    def mod(name: str, **attrs: Any) -> None:
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))

    class Scene(Closer):
        def place(self, want: dict, log: dict | None = None) -> str | None:
            return None
    mod("adapters.turtlebot4", TB4Adapter=lambda: Closer(log, "tb4"), SimTruth=lambda: Closer(log, "simtruth"))
    mod("humanoid.adapter_g1", G1Adapter=lambda **kw: Closer(log, "g1", **kw))
    mod("humanoid.scene_g1", G1Scene=lambda: Scene(log, "scene"))
    mod("humanoid.sim_truth_g1", G1SimTruth=lambda: Closer(log, "g1truth"), G1Trace="G1Trace")
    return log


# ------------------------------------------------------------------------------------------------------ robots.make
def test_turtlebot4_is_run_one_defaults(stubs: list[str]) -> None:
    rb = robots.make(robots.robot_name())
    assert rb.ad.name == "tb4" and rb.gt_pose.__self__.name == "simtruth" and rb.run_kw == {} and rb.scene_meta == {}
    rb.close()
    assert stubs == ["close tb4"]


def test_g1_paths_and_run_kw(stubs: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for k, v in {"ROBOT": "g1", "G1_SCENES": str(tmp_path), "GZ_PARTITION": "p3", "SIM_WORLD": "cabinet_open"}.items():
        monkeypatch.setenv(k, v)
    (tmp_path / "p3").mkdir()
    (tmp_path / "p3" / "clinic.xml").write_text("<mujoco/>")  # room._write: xml next to scene.json (path from the job)
    (tmp_path / "p3" / "scene.json").write_text(json.dumps({"xml": "/run/scenes/p3/clinic.xml", "world": "cabinet_open",
                                                           "label_of_body": [0, 7, 0], "mocap_bodies": ["m_tb"]}))
    rb = robots.make(robots.robot_name())
    assert rb.ad.kw == {"scene_json": tmp_path / "p3" / "scene.json",
                        "map_path": WORLD_DIRS["cabinet_open"] / "map_knowledge.json"}
    kw = rb.run_kw
    assert kw["robot"] == "g1" and kw["trace_factory"] == "G1Trace" and kw["place_objects"].__self__.name == "scene"
    assert kw["restore_world"].__self__.name == "scene"  # a frozen-world run puts the movables back (task 012 fix 1)
    hh = kw["hand_help"]()
    assert isinstance(hh, HandHelp) and hh.v_max == 0.71 and hh.slack == HandHelp().slack  # SLACK_M unchanged
    assert rb.scene_meta == {"scene": {
        "world": "cabinet_open", "xml_sha256": hashlib.sha256(b"<mujoco/>").hexdigest(),
        "labels_sha256": hashlib.sha256(b"[0,7,0]").hexdigest(), "mocap_bodies": ["m_tb"]}}
    rb.close()
    assert stubs == ["close g1truth", "close scene", "close g1"]  # the adapter last: it shuts the default context down


def test_g1_adapter_closed_when_truth_close_fails(stubs: list[str], monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    for k, v in {"G1_SCENES": str(tmp_path), "GZ_PARTITION": "p3", "SIM_WORLD": "closed"}.items():
        monkeypatch.setenv(k, v)
    (tmp_path / "p3").mkdir()
    (tmp_path / "p3" / "clinic.xml").write_text("<mujoco/>")
    (tmp_path / "p3" / "scene.json").write_text(json.dumps({"xml": "clinic.xml", "world": "closed",
                                                           "label_of_body": [0], "mocap_bodies": []}))
    rb = robots.make("g1")
    monkeypatch.setattr(rb.gt_pose.__self__, "close", lambda: (_ for _ in ()).throw(RuntimeError("truth")))
    with pytest.raises(RuntimeError, match="truth"):
        rb.close()
    assert stubs == ["close g1"]


def test_unknown_robot_refused() -> None:
    with pytest.raises(SystemExit, match="not a robot"):
        robots.make("unitree")


def test_tb4_path_never_imports_humanoid(stubs: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    for m in [m for m in sys.modules if m.startswith("humanoid")]:
        monkeypatch.delitem(sys.modules, m)
    robots.make("turtlebot4")
    assert not [m for m in sys.modules if m.startswith("humanoid")]


# --------------------------------------------------------------------------------------------- phase3_run: robot key
def inv(out: Path, name: str = "invocation.json") -> dict[str, Any]:
    return json.loads((out / name).read_text())


def test_tb4_invocation_has_no_robot_key(tmp_path: Path) -> None:
    assert p3main(tmp_path, "--out", str(tmp_path / "out")) == 0
    assert "robot" not in inv(tmp_path / "out")


def test_g1_invocation_records_robot_and_resume_needs_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    monkeypatch.setenv("ROBOT", "g1")
    p3main(tmp_path, "--out", str(out), sim=Sim(down=[None, "down"]))  # aborted: resumable
    assert inv(out)["robot"] == "g1"
    monkeypatch.delenv("ROBOT")
    with pytest.raises(SystemExit, match=r"refusing --resume: .* robot 'g1', this is ROBOT='turtlebot4'"):
        p3main(tmp_path, "--resume", str(out))
    monkeypatch.setenv("ROBOT", "g1")
    assert p3main(tmp_path, "--resume", str(out)) == 0


def test_tb4_invocation_refuses_a_g1_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    p3main(tmp_path, "--out", str(out), sim=Sim(down=[None, "down"]))
    monkeypatch.setenv("ROBOT", "g1")
    with pytest.raises(SystemExit, match=r"refusing --resume: .* robot 'turtlebot4', this is ROBOT='g1'"):
        p3main(tmp_path, "--resume", str(out))


@pytest.mark.parametrize("robot,env", [("g1", None), (None, "g1")])
def test_job_part_and_merge_refuse_another_robot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 robot: str | None, env: str | None) -> None:
    """The robot check comes before every other job-part / merge check (an otherwise empty split invocation)."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "invocation.json").write_text(json.dumps({"split": {}, **({"robot": robot} if robot else {})}))
    if env:
        monkeypatch.setenv("ROBOT", env)
    with pytest.raises(SystemExit, match=r"refusing --job-part: .* robot"):
        pr.main(["--job-part", str(out), "--job", "0", "--shards", "1", "--shard-env", "0=1:p"],
                job_left=lambda: pytest.fail("job_left called"))
    with pytest.raises(SystemExit, match=r"refusing --merge: .* robot"):
        pr.main(["--merge", str(out), "--ledger", str(tmp_path / "l.jsonl")])


# ----------------------------------------------------------------------------------- entry_runner / run_one hooks
def test_entry_runner_passes_run_kw_and_scene_meta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    got: list[dict[str, Any]] = []
    monkeypatch.setattr(rt, "run_one", lambda *a, **kw: got.append(kw) or {
        "setup": {"reached": True}, "outcome": "reported", "steps": 1, "score": {"pass": True, "label": None,
                                                                                   "label_why": None}})
    e = {"run_id": "r1", "task_id": "find-toolbox", "start_place": "hall", "seed": 1}
    pr.entry_runner(None, None, lambda s: None, {"scene": {"world": "closed"}}, robot="g1", place_objects=print)(
        e, tmp_path / "r1", {"invocation_id": "u"})
    pr.entry_runner(None, None, lambda s: None)(e, tmp_path / "r2", {"invocation_id": "u"})
    assert got[0]["extra_meta"] == {"invocation_id": "u", "scene": {"world": "closed"}}
    assert got[0]["robot"] == "g1" and got[0]["place_objects"] is print
    assert got[1]["extra_meta"] == {"invocation_id": "u"} and "robot" not in got[1] and "place_objects" not in got[1]


class HistTrace:
    """trace_factory stand-in for G1Trace: stop() adds history_samples."""

    def __init__(self, gt_pose: Any, path: Path) -> None:
        self.path = path

    def start(self) -> "HistTrace":
        return self

    def stop(self) -> dict[str, Any]:
        return {"samples": 4, "errors": 0, "last_error": None, "history_samples": 250}


def test_run_one_uses_the_g1_hooks(tmp_path: Path, world: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, SEEDED, FakeSeeded())
    monkeypatch.setattr(rt, "teleport", lambda *a: pytest.fail("gz teleport called"))
    placed: list[dict] = []

    def place(want: dict, log: dict) -> None:
        placed.append(want)
        log.update(models=len(want), sends=1, retries=[])
    ad = FakeAdapter("hall")
    rt.run_one(ad, ad.pose, TASK, ScriptedPlanner(GOOD, "g"), tmp_path / "r", tag="fake", robot="g1",
               place_objects=place, trace_factory=HistTrace, layout_seed=7, **world)
    run = json.loads((tmp_path / "r" / "run.json").read_text())
    assert len(placed) == 1 and run["teleport"] == {"models": 3, "sends": 1, "retries": []} and run["robot"] == "g1"
    assert run["gt_trace"] == {"samples": 4, "errors": 0, "last_error": None, "history_samples": 250}


# ------------------------------------------------------------------- frozen-world restore (task 012 H-P4 fix 1)
SPEC = ("toolbox_red_1", "cabinet_blue_1", "chair_orange_1", "chair_orange_2", "bed_blue_1")


@pytest.fixture
def frozen_world(world: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    """The fake world with distinct frozen poses (x, y, z, r, p, yaw; r != yaw) and realize's SPEC_OBJECTS on it."""
    t = json.loads(world["truth_path"].read_text())
    for k, o in enumerate(t["objects"]):
        o["pose"] = [k + 0.5, -k - 0.25, 0.01 * k, 0.3, 0.0, 0.1 * k + 0.2]
    world["truth_path"].write_text(json.dumps(t))
    monkeypatch.setattr(rt, "realizer", lambda: types.SimpleNamespace(SPEC_OBJECTS=SPEC))
    return {o["id"]: o for o in t["objects"]}


def placer(calls: list[dict], why: str | None = None) -> Any:
    def place(want: dict, log: dict) -> str | None:
        calls.append(want)
        log.update(models=len(want), sends=1, retries=[])
        return why
    return place


def test_frozen_run_restores_every_spec_object(tmp_path: Path, world: dict, frozen_world: dict,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "teleport", lambda *a: pytest.fail("gz teleport called"))
    calls: list[dict] = []
    ad = FakeAdapter("hall")
    res = rt.run_one(ad, ad.pose, TASK, ScriptedPlanner(GOOD, "g"), tmp_path / "r", tag="fake", robot="g1",
                     place_objects=lambda *a: pytest.fail("layout placer called"), restore_world=placer(calls), **world)
    want = {frozen_world[o]["model_name"]: (*frozen_world[o]["pose"][:3], frozen_world[o]["pose"][5]) for o in SPEC}
    assert calls == [want] and len(want) == len(SPEC)
    run = json.loads((tmp_path / "r" / "run.json").read_text())
    assert run["teleport"] == {"models": len(SPEC), "sends": 1, "retries": []}
    assert "layout_seed" not in run and "layout_spec" not in run and run["layout_sha256"] == LAYOUT_SHA(world)
    assert res["outcome"] == "reported" and res["score"]["checks"]["answer"] is True  # scored on the world's truth.json


def LAYOUT_SHA(world: dict) -> str:  # noqa: N802
    return json.loads(world["truth_path"].read_text())["layout_sha256"]


def test_layout_run_does_not_restore(tmp_path: Path, world: dict, frozen_world: dict,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, SEEDED, FakeSeeded())
    laid: list[dict] = []
    ad = FakeAdapter("hall")
    rt.run_one(ad, ad.pose, TASK, ScriptedPlanner(GOOD, "g"), tmp_path / "r", tag="fake", robot="g1",
               place_objects=placer(laid), restore_world=lambda *a: pytest.fail("restore on a layout run"),
               layout_seed=7, **world)
    assert len(laid) == 1 and json.loads((tmp_path / "r" / "run.json").read_text())["layout_seed"] == 7


def test_frozen_restore_miss_is_a_crash_before_any_step(tmp_path: Path, world: dict, frozen_world: dict) -> None:
    ad = FakeAdapter("hall")
    res = rt.run_one(ad, ad.pose, TASK, ScriptedPlanner(GOOD, "g"), tmp_path / "r", tag="fake", robot="g1",
                     restore_world=placer([], "chair_orange_1 at ...: 0.100 m"), **world)
    run = json.loads((tmp_path / "r" / "run.json").read_text())
    assert res["outcome"] == "crash" and res["steps"] == 0 and run["crash_origin"] == "layout_teleport"
    assert "frozen world" in run["crash"] and "chair_orange_1" in run["crash"] and run["setup"]["reached"] is None
