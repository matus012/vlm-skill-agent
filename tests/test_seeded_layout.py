"""Task 008 seeded layouts at run time (agent/INTERFACES.md section 10): a list entry's layout_seed reaches
adapters/run_task.run_one on the serial, sharded and cross-job paths; run_one teleports every movable with gz set_pose,
reads the poses back and fails a run whose movables did not land (label sim, no setup move, no planner call); the
scorer scores a seeded run with truth_for(run.json layout_seed, world) and an unseeded run exactly as before.
No sim: a fake `gz` (subprocess.run) and a fake sim/worlds/clinic/seeded.py module (never the real placement logic)."""
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import adapters.run_task as rt  # noqa: E402
import gz_cli  # noqa: E402
from adapters.hand_help import HandHelp  # noqa: E402
from agent import scorer, truth as truth_mod  # noqa: E402
from agent.fake_adapter import PLACES, FakeAdapter  # noqa: E402
from agent.planners import ScriptedPlanner  # noqa: E402
from agent.tasks import Task, TaskError  # noqa: E402
from probes import phase3_run as pr  # noqa: E402
from test_phase3_gates import LAYOUT, ROOMS, TRUTH  # noqa: E402

SEEDED = "sim.worlds.clinic.seeded"
TASK = Task("find-toolbox", "find", "Find the red toolbox.", {"object": "toolbox_red_1"})
GOOD = [{"name": "navigate_to", "args": {"place": "storage"}}, {"name": "look_around", "args": {}},
        {"name": "report", "args": {"answer": "storage", "photo": 1, "note": "seen"}}]
# the fake world's truth with frozen poses (x, y, z, r, p, yaw) and gz model names; toolbox z 0.05 shows where z is from
FROZEN = copy.deepcopy(TRUTH)
for _o in FROZEN["objects"]:
    _o |= {"model_name": f"m_{_o['id']}", "pose": [0.0, 0.0, 0.05 if _o["id"] == "toolbox_red_1" else 0.0, 0, 0, 0]}
PLACE = {"toolbox_red_1": (1.0, 2.0, 0.5), "chair_orange_1": (3.0, -1.0, -1.2), "chair_orange_2": (-2.0, 0.5, 3.0)}


def seeded_sha(seed: int) -> str:
    return hashlib.sha256(f"{LAYOUT}:{seed}".encode()).hexdigest()


class FakeSeeded(types.ModuleType):
    """sim/worlds/clinic/seeded.py's API (INTERFACES section 10); records its calls. toolbox_room: where truth_for
    puts the toolbox (the fake adapter always shows it in storage)."""

    class SeedRejected(Exception):
        pass

    def __init__(self) -> None:
        super().__init__(SEEDED)
        self.calls: list[tuple[Any, ...]] = []
        self.toolbox_room = "storage"

    def placements(self, seed: int) -> dict[str, tuple[float, float, float]]:
        self.calls.append(("placements", seed))
        return dict(PLACE)

    def truth_for(self, seed: int, world: str) -> dict[str, Any]:
        self.calls.append(("truth_for", seed, world))
        t = copy.deepcopy(FROZEN) | {"layout_sha256": seeded_sha(seed)}
        for o in t["objects"]:
            if o["id"] in PLACE:
                o["pose"] = [PLACE[o["id"]][0], PLACE[o["id"]][1], o["pose"][2], 0, 0, PLACE[o["id"]][2]]
            if o["id"] == "toolbox_red_1":
                o["room"] = self.toolbox_room
        return t


class FakeGz:
    """`gz service ... set_pose` and `gz topic -e ... pose/info` on a model pose table; everything else runs for real.
    off[name] = (dx, dz_yaw_deg): where a model lands relative to the request; ok=False: set_pose replies false;
    die: every pose/info read is killed by SIGSEGV; timeouts[name] = n: the first n set_pose calls of that model reply
    empty with gz's "Service call timed out" (rc 0, nothing moves)."""

    def __init__(self, real: Any) -> None:
        self.real, self.log, self.pose = real, [], {}
        self.off: dict[str, tuple[float, float]] = {}
        self.timeouts: dict[str, int] = {}
        self.ok, self.die = True, False

    def __call__(self, cmd: list[str], **kw: Any) -> Any:
        if cmd[0] != "gz":
            return self.real(cmd, **kw)
        self.log.append(cmd)
        if cmd[1] == "service":
            assert cmd[:-1] == rt.SET_POSE and kw.get("text"), cmd
            n = re.search(r'name: "(\w+)"', cmd[-1]).group(1)
            if self.timeouts.get(n, 0) > 0:
                self.timeouts[n] -= 1
                return subprocess.CompletedProcess(cmd, 0, "", "Service call timed out\n")
            dx, dyaw = self.off.get(n, (0.0, 0.0))
            yaw = 2 * math.atan2(float(re.search(r"z: (-?[\d.e-]+), w", cmd[-1]).group(1)),
                                 float(re.search(r"w: (-?[\d.e-]+)", cmd[-1]).group(1)))
            pos = [float(v) for v in re.search(r"position: \{x: (\S+), y: (\S+), z: (\S+)\}", cmd[-1]).groups()]
            self.pose[n] = (pos[0] + dx, pos[1], pos[2], yaw + math.radians(dyaw))
            return subprocess.CompletedProcess(cmd, 0, f"data: {'true' if self.ok else 'false'}\n", "")
        assert cmd[:5] == ["gz", "topic", "-e", "-n", "1"] and cmd[6] == rt.POSE_INFO, cmd
        if self.die:
            return subprocess.CompletedProcess(cmd, -11, "", "Segmentation fault")
        msg = {"header": {"stamp": {"sec": 1}}, "pose": [
            {"name": n, "position": {"x": x, "y": y, **({"z": z} if z else {})},  # gz JSON leaves out zero fields
             "orientation": {"z": math.sin(a / 2), "w": math.cos(a / 2)}} for n, (x, y, z, a) in self.pose.items()]
            + [{"name": "link", "position": {"x": 99.0}}]}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(msg) + "\n", "")

    def set_poses(self) -> list[str]:
        return [re.search(r'name: "(\w+)"', c[-1]).group(1) for c in self.log if c[1] == "service"]


@pytest.fixture(autouse=True)
def clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
    monkeypatch.delenv("SIM_WORLD", raising=False)


@pytest.fixture
def fake_seeded(monkeypatch: pytest.MonkeyPatch) -> FakeSeeded:
    m = FakeSeeded()
    monkeypatch.setitem(sys.modules, SEEDED, m)
    return m


@pytest.fixture
def gz(monkeypatch: pytest.MonkeyPatch) -> FakeGz:
    g = FakeGz(subprocess.run)
    monkeypatch.setattr(subprocess, "run", g)
    return g


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    t, p, m = tmp_path / "truth.json", tmp_path / "places.yaml", tmp_path / "map_knowledge.json"
    t.write_text(json.dumps(FROZEN))
    p.write_text(yaml.safe_dump({"places": {n: {"x": x, "y": y, "yaw": a} for n, (x, y, a) in PLACES.items()}}))
    m.write_text(json.dumps({"world": "fake", "layout_sha256": LAYOUT, "rooms": {r: {} for r in ROOMS}}))
    return {"truth_path": t, "places_path": p, "map_path": m}


class NoPlanner(ScriptedPlanner):
    def next_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        pytest.fail("the planner was called")


def one(world: dict, d: Path, ad: FakeAdapter, events: list[str] | None = None, planner: Any = None,
        **kw: Any) -> dict[str, Any]:
    if events is not None:
        go = ad.go_to
        ad.go_to = lambda *a: (events.append("go_to"), go(*a))[1]  # type: ignore[method-assign]
    return rt.run_one(ad, ad.pose, TASK, planner or ScriptedPlanner(GOOD, "g"), d, robot="fake", tag="fake",
                      hand_help=lambda: HandHelp(v_max_mps=1.0), **world, **kw)


def run_json(d: Path) -> dict[str, Any]:
    return json.loads((d / "run.json").read_text())


# ------------------------------------------------------------------------------------------------ run_one teleport
def test_seeded_run_teleports_then_reads_back_before_setup(tmp_path: Path, world: dict, fake_seeded: FakeSeeded,
                                                           gz: FakeGz) -> None:
    ad, events = FakeAdapter("hall"), []
    gz_log = gz.log
    real_go = ad.go_to
    ad.go_to = lambda *a: (events.append(f"go_to after {len(gz_log)} gz calls"), real_go(*a))[1]  # type: ignore
    r = one(world, tmp_path / "r", ad, layout_seed=7)
    # one set_pose per movable, in placements order, then one pose/info read, all before the setup move
    assert gz.set_poses() == [f"m_{oid}" for oid in PLACE]
    assert [c[1] for c in gz.log] == ["service"] * 3 + ["topic"] and events[0] == "go_to after 4 gz calls"
    req = gz.log[0][-1]
    assert req.startswith('name: "m_toolbox_red_1", position: {x: 1.0, y: 2.0, z: 0.05}')  # z: the frozen truth
    assert f"z: {math.sin(0.25)}, w: {math.cos(0.25)}" in req  # yaw 0.5 from placements
    run = run_json(tmp_path / "r")
    assert run["layout_seed"] == 7 and run["layout_sha256"] == seeded_sha(7) and run["world"] == "closed"
    assert run["gz_reads"]["reads"] == 1 and run["outcome"] == "reported" and r["score"]["pass"]
    assert ("truth_for", 7, "closed") in fake_seeded.calls


@pytest.mark.parametrize("off", [(0.021, 0.0), (0.0, 1.1), (0.0, -1.1)])
def test_readback_mismatch_fails_the_run_labelled_sim(tmp_path: Path, world: dict, fake_seeded: FakeSeeded,
                                                      gz: FakeGz, off: tuple[float, float]) -> None:
    gz.off["m_chair_orange_2"] = off
    ad, events = FakeAdapter("hall"), []
    r = one(world, tmp_path / "r", ad, events, planner=NoPlanner([], "never"), layout_seed=7)
    assert events == []  # no setup move
    run = run_json(tmp_path / "r")
    assert run["outcome"] == "crash" and run["crash_origin"] == "layout_teleport" and run["steps"] == 0
    assert "m_chair_orange_2" in run["crash"] and "m_toolbox_red_1" not in run["crash"]
    assert run["setup"]["reached"] is None and run["layout_seed"] == 7 and run["layout_sha256"] == seeded_sha(7)
    assert (tmp_path / "r/steps.jsonl").read_text() == ""
    assert r["outcome"] == "crash" and not r["score"]["pass"] and r["score"]["label"] == "sim"
    assert "layout_teleport" in r["score"]["label_why"]
    assert json.loads((tmp_path / "r/score.json").read_text())["label"] == "sim"


@pytest.mark.parametrize("off", [(0.019, 0.0), (0.0, 0.9), (-0.019, -0.9)])
def test_landing_within_2cm_1deg_runs(tmp_path: Path, world: dict, fake_seeded: FakeSeeded, gz: FakeGz,
                                      off: tuple[float, float]) -> None:
    gz.off["m_toolbox_red_1"] = off
    assert one(world, tmp_path / "r", FakeAdapter("hall"), layout_seed=7)["score"]["pass"]


def test_set_pose_refused_fails_before_the_readback(tmp_path: Path, world: dict, fake_seeded: FakeSeeded,
                                                    gz: FakeGz) -> None:
    gz.ok = False
    r = one(world, tmp_path / "r", FakeAdapter("hall"), planner=NoPlanner([], "never"), layout_seed=7)
    assert [c[1] for c in gz.log] == ["service"] and "data: false" in run_json(tmp_path / "r")["crash"]
    assert r["score"]["label"] == "sim"


def test_set_pose_timeout_is_sent_again_and_logged(tmp_path: Path, world: dict, fake_seeded: FakeSeeded, gz: FakeGz,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Deviation 20.2: an empty set_pose reply (gz timeout) is resent, at most twice, each resend logged in run.json."""
    monkeypatch.setattr(rt, "SET_POSE_WAIT_S", 0.0)
    gz.timeouts["m_chair_orange_2"] = 2
    r = one(world, tmp_path / "r", FakeAdapter("hall"), layout_seed=7)
    run = run_json(tmp_path / "r")
    assert gz.set_poses() == ["m_toolbox_red_1", "m_chair_orange_1"] + ["m_chair_orange_2"] * 3
    assert run["outcome"] == "reported" and r["score"]["pass"]
    tp = run["teleport"]
    assert tp["models"] == 3 and tp["sends"] == 5 and len(tp["retries"]) == 2
    assert all("m_chair_orange_2" in x and "timed out" in x for x in tp["retries"])


def test_set_pose_timeout_three_times_is_a_sim_crash(tmp_path: Path, world: dict, fake_seeded: FakeSeeded,
                                                     gz: FakeGz, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "SET_POSE_WAIT_S", 0.0)
    gz.timeouts["m_toolbox_red_1"] = 3
    r = one(world, tmp_path / "r", FakeAdapter("hall"), planner=NoPlanner([], "never"), layout_seed=7)
    run = run_json(tmp_path / "r")
    assert gz.set_poses() == ["m_toolbox_red_1"] * 3 and [c[1] for c in gz.log] == ["service"] * 3  # no read-back
    assert run["outcome"] == "crash" and run["crash_origin"] == "layout_teleport" and r["score"]["label"] == "sim"
    assert "send 3 of 3" in run["crash"] and "timed out" in run["crash"]
    assert run["teleport"]["sends"] == 3 and len(run["teleport"]["retries"]) == 2 and run["setup"]["skipped"]


def test_set_pose_refused_is_not_sent_again(tmp_path: Path, world: dict, fake_seeded: FakeSeeded,
                                            gz: FakeGz) -> None:
    gz.ok = False
    one(world, tmp_path / "r", FakeAdapter("hall"), planner=NoPlanner([], "never"), layout_seed=7)
    assert run_json(tmp_path / "r")["teleport"] == {"models": 3, "sends": 1, "retries": []}


def test_dead_readback_is_gz_read_labelled_sim(tmp_path: Path, world: dict, fake_seeded: FakeSeeded, gz: FakeGz,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gz_cli, "DEATHS", [])
    gz.die = True
    r = one(world, tmp_path / "r", FakeAdapter("hall"), planner=NoPlanner([], "never"), layout_seed=7)
    run = run_json(tmp_path / "r")
    assert run["crash_origin"] == "gz_read" and len(run["gz_reads"]["deaths"]) == 1 + gz_cli.MAX_RETRIES
    assert r["score"]["label"] == "sim"


def test_every_seeded_run_teleports(tmp_path: Path, world: dict, fake_seeded: FakeSeeded, gz: FakeGz) -> None:
    """A run after a seeded run on the same sim teleports again, though the sim already holds that placement."""
    ad = FakeAdapter("hall")
    one(world, tmp_path / "a", ad, layout_seed=7)
    one(world, tmp_path / "b", ad, layout_seed=7)
    assert gz.set_poses() == [f"m_{oid}" for oid in PLACE] * 2 and sum(c[1] == "topic" for c in gz.log) == 2


def test_task_invalid_on_the_seeded_layout_refused_before_gz(tmp_path: Path, world: dict, fake_seeded: FakeSeeded,
                                                             gz: FakeGz, monkeypatch: pytest.MonkeyPatch) -> None:
    good = fake_seeded.truth_for
    monkeypatch.setattr(fake_seeded, "truth_for", lambda s, w: good(s, w) | {"objects": []})
    with pytest.raises(TaskError):
        one(world, tmp_path / "r", FakeAdapter("hall"), layout_seed=7)
    assert gz.log == [] and not (tmp_path / "r").exists()


def test_unseeded_run_unchanged(tmp_path: Path, world: dict, gz: FakeGz, monkeypatch: pytest.MonkeyPatch) -> None:
    """No layout_seed: no gz call, seeded.py never imported, no layout_seed key, the frozen truth."""
    monkeypatch.delitem(sys.modules, SEEDED, raising=False)
    monkeypatch.setattr(truth_mod, "seeded", lambda: pytest.fail("seeded.py imported for an unseeded run"))
    r = one(world, tmp_path / "r", FakeAdapter("hall"))
    run = run_json(tmp_path / "r")
    assert gz.log == [] and "layout_seed" not in run and run["layout_sha256"] == LAYOUT and r["score"]["pass"]


# ------------------------------------------------------------------------------------------------------- scorer
def test_scorer_uses_the_seeded_truth_of_the_run(tmp_path: Path, world: dict, fake_seeded: FakeSeeded,
                                                 gz: FakeGz) -> None:
    d = tmp_path / "r"
    assert one(world, d, FakeAdapter("hall"), layout_seed=7)["score"]["pass"]
    fake_seeded.toolbox_room = "office"  # the same run against a layout where the toolbox is elsewhere
    fake_seeded.calls.clear()
    sc = scorer.score_run(d, places_path=world["places_path"])
    assert fake_seeded.calls == [("truth_for", 7, "closed")]
    assert sc["truth_answer"] == "office" and not sc["checks"]["answer"] and sc["label"] == "vlm"
    run = run_json(d)
    (d / "run.json").write_text(json.dumps(run | {"world": "cabinet_open"}))
    fake_seeded.calls.clear()
    scorer.score_run(d, places_path=world["places_path"])
    assert fake_seeded.calls == [("truth_for", 7, "cabinet_open")]
    (d / "run.json").write_text(json.dumps(run | {"layout_seed": 8}))  # the layout sha check stays
    with pytest.raises(ValueError, match="run layout"):
        scorer.score_run(d, places_path=world["places_path"])


def test_unseeded_run_scores_identically(tmp_path: Path, world: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """An old (unseeded) run: the default truth is its world's truth.json, as before 008; seeded.py never touched."""
    d = tmp_path / "r"
    one(world, d, FakeAdapter("hall"))
    monkeypatch.setitem(truth_mod.WORLD_DIRS, "closed", world["truth_path"].parent)
    monkeypatch.setattr(truth_mod, "seeded", lambda: pytest.fail("seeded.py used for an unseeded run"))
    saved = json.loads((d / "score.json").read_text())
    assert {**scorer.score_run(d, places_path=world["places_path"]), "sha256": saved["sha256"]} == saved
    assert scorer.score_run(d, truth_path=world["truth_path"], places_path=world["places_path"]) == \
        scorer.score_run(d, places_path=world["places_path"])


# ------------------------------------------------------------------------------------------------ list routing
SEEDS = {"d01": 11, "d02": 12, "d03": 13, "d04": 14}
DEV = ROOT / "tasks" / "phase3_dev.yaml"


def seeded_list(tmp: Path) -> Path:
    runs = [r | {"layout_seed": SEEDS[r["run_id"]]} for r in yaml.safe_load(DEV.read_text()) if r["run_id"] in SEEDS]
    p = tmp / "seeded.yaml"
    p.write_text(yaml.safe_dump(runs))
    return p


# Records what run_one got: the entry_runner path of phase3_run (the real sim_runner's) with run_one replaced.
RECORDER = r'''
def install(rt, pr):
    from pathlib import Path
    import json
    from agent.fake_adapter import FakeAdapter

    def run_one(ad, gt_pose, task, planner, d, **kw):
        Path(d).mkdir(parents=True)
        (Path(d) / "rec.json").write_text(json.dumps({"task": task.id, "planner": planner, **kw}))
        return {"setup": {"reached": True}, "outcome": "reported", "steps": 3,
                "score": {"pass": True, "label": None, "label_why": None}}
    rt.run_one = run_one
    ad = FakeAdapter("hall")
    return lambda: (pr.entry_runner(ad, ad.pose, lambda seed: seed), lambda: None, lambda: None)
'''
WORKER = r'''
import sys
sys.path.insert(0, sys.argv.pop(1)); sys.path.insert(0, sys.argv.pop(1))
import adapters.run_task as rt
from probes import phase3_run as pr
pr.git_state = lambda repo: {"git_commit": "c0ffee", "git_dirty": False}
''' + RECORDER + r'''
sys.exit(pr.main(sys.argv[1:], runner=install(rt, pr)))
'''


@pytest.fixture
def worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
    w = tmp_path / "seed_worker.py"
    w.write_text(WORKER)
    monkeypatch.setattr(pr, "WORKER", [sys.executable, str(w), str(ROOT), str(ROOT / "tests")])
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


def recs(out: Path) -> dict[str, dict[str, Any]]:
    return {d.name: json.loads((d / "rec.json").read_text()) for d in out.iterdir() if (d / "rec.json").is_file()}


def check_routed(out: Path, lst: Path) -> None:
    got = recs(out)
    assert sorted(got) == sorted(SEEDS)
    for e in pr.load_runs(lst):
        r = got[e["run_id"]]
        assert (r["layout_seed"], r["task"], r["start_place"], r["planner"]) == \
            (e["layout_seed"], e["task_id"], e["start_place"], e["seed"])
        assert r["extra_meta"]["run_id"] == e["run_id"]


def test_layout_seed_routed_serial(tmp_path: Path, worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    ns: dict[str, Any] = {}
    exec(RECORDER, ns)
    monkeypatch.setattr(rt, "run_one", rt.run_one)  # restored after the test
    runner = ns["install"](rt, pr)
    lst, out = seeded_list(tmp_path), tmp_path / "out"
    assert pr.main(["--list", str(lst), "--out", str(out), "--rehearsal"], runner=runner,
                   job_left=lambda: pytest.fail("job_left called")) == 0
    check_routed(out, lst)


def test_layout_seed_routed_sharded(tmp_path: Path, worker: None) -> None:
    lst, out = seeded_list(tmp_path), tmp_path / "out"
    assert pr.main(["--list", str(lst), "--out", str(out), "--rehearsal", "--shards", "2", "--shard-env", "0=40:s0",
                    "--shard-env", "1=41:s1"], runner=lambda: pytest.fail("the coordinator never drives a sim"),
                   job_left=lambda: pytest.fail("job_left called")) == 0
    check_routed(out, lst)


def test_layout_seed_routed_cross_job(tmp_path: Path, worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    lst, out, led = seeded_list(tmp_path), tmp_path / "out", tmp_path / "ledger.jsonl"

    def main(*a: str) -> int:
        return pr.main(["--list", str(lst), "--rehearsal", "--ledger", str(led), *a],
                       runner=lambda: pytest.fail("the coordinator never drives a sim"), job_left=lambda: 999.0)
    assert main("--out", str(out), "--split", "2:2") == 0
    for j in (0, 1):
        monkeypatch.setenv("SLURM_JOB_ID", f"10{j}")
        assert main("--job-part", str(out), "--job", str(j), "--shards", "2", "--shard-env", f"0=4{j}:j{j}s0",
                    "--shard-env", f"1=5{j}:j{j}s1") == 0
    check_routed(out, lst)


def test_load_runs_layout_seed(tmp_path: Path) -> None:
    lst = seeded_list(tmp_path)
    assert [e["layout_seed"] for e in pr.load_runs(lst)] == list(SEEDS.values())
    runs = yaml.safe_load(lst.read_text())
    for bad in ([runs[0] | {"layout_seed": True}, *runs[1:]], [runs[0] | {"layout_seed": "11"}, *runs[1:]],
                [{k: v for k, v in runs[0].items() if k != "layout_seed"}, *runs[1:]],  # some entries only
                [runs[0] | {"layout": 3}, *runs[1:]]):
        lst.write_text(yaml.safe_dump(bad))
        with pytest.raises(AssertionError):
            pr.load_runs(lst)
    assert all("layout_seed" not in e for e in pr.load_runs(DEV))  # old lists unchanged
