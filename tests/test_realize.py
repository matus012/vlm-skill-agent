"""Phase 8 layout specs (task 011, agent/INTERFACES.md section 12): sim/worlds/clinic/realize.py (realize,
truth_for_spec, spec_sha256, park poses), probes/vet_seeds.check_spec, the truth / scorer / gate side (run_truth with
layout_spec), adapters/run_task's teleport list and run.json keys, probes/phase3_run's spec entries.
SYNTHETIC specs only (built here by hand; the sealed / unsealed Phase 8 entries are never read)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "probes"))
sys.path.insert(0, str(ROOT / "sim" / "worlds" / "clinic"))
import adapters.run_task as rt  # noqa: E402
import realize as RZ  # noqa: E402
import seeded as S  # noqa: E402
import vet_seeds as V  # noqa: E402
from agent import scorer, truth as truth_mod  # noqa: E402
from agent.fake_adapter import FakeAdapter  # noqa: E402
from probes import phase3_run as pr  # noqa: E402
from probes import phase4_gates as p4  # noqa: E402
import test_seeded_layout as tsl  # noqa: E402
from test_seeded_layout import FROZEN, LAYOUT, NoPlanner, gz, one, run_json, world, worker  # noqa: E402,F401

SNAPSHOT = ROOT / "tests" / "seeded_snapshot_0_50.json"  # seeded.placements 0..50, taken before task 011's refactor


def spec(seed: int = 1, toolbox: str = "storage", cabinet: tuple[str, str] = ("storage", "closed"),
         bed: tuple[str, int] = ("ward", 3), chair: tuple[str, int] = ("reception", 2)) -> dict[str, Any]:
    """A synthetic spec in gen_phase8.render_pass's layout format."""
    objs = {RZ.TOOLBOX: {"room": toolbox}, RZ.CABINET: {"room": cabinet[0], "state": cabinet[1]}}
    objs |= {b: {"room": bed[0]} for b in RZ.BEDS[:bed[1]]}
    objs |= {c: {"room": chair[0]} for c in RZ.CHAIRS[:chair[1]]}
    return {"seed": seed, "objects": objs, "removed": [*RZ.BEDS[bed[1]:], *RZ.CHAIRS[chair[1]:]]}


FROZEN_CLOSED = {o["id"]: o for o in json.loads(S.world_truth_path("closed").read_text())["objects"]}


def frozen_pose(oid: str) -> tuple[float, float, float]:
    p = FROZEN_CLOSED[oid]["pose"]
    return p[0], p[1], p[5]


def in_zone(oid: str, room: str, pose: tuple[float, float, float]) -> bool:
    return any(x0 <= pose[0] <= x1 and y0 <= pose[1] <= y1 for x0, y0, x1, y1 in S.ZONES[room][S.CLASS_OF[oid]])


# ------------------------------------------------------------------------------------------------- determinism
def test_realize_deterministic_in_and_across_processes() -> None:
    s = spec(seed=7, toolbox="office", chair=("storage", 3))
    a = RZ.realize(s)
    assert RZ.realize(copy.deepcopy(s)) == a
    code = ("import json, sys; sys.path.insert(0, 'sim/worlds/clinic'); import realize as RZ\n"
            f"print(json.dumps(RZ.realize(json.loads({json.dumps(json.dumps(s))}))))")
    other = json.loads(subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                                      check=True).stdout)
    assert {k: tuple(v) for k, v in other["poses"].items()} == a["poses"] and other["parked"] == a["parked"]
    b = RZ.realize(spec(seed=8, toolbox="office", chair=("storage", 3)))
    assert b["poses"][RZ.TOOLBOX] != a["poses"][RZ.TOOLBOX]  # the seed drives the draw


def test_rng_is_sha256_of_realize_seed() -> None:
    """The first toolbox draw is the first draw of random.Random(int(sha256("realize:<seed>"))) (no global RNG)."""
    import random
    rng = random.Random(int(hashlib.sha256(b"realize:11").hexdigest(), 16))
    first = RZ._draw(rng, RZ.TOOLBOX, "storage")
    random.seed(999)
    assert S.alone_problems("toolbox", "storage", first) == []  # so the realizer keeps its first draw
    assert RZ.realize(spec(seed=11, toolbox="storage"))["poses"][RZ.TOOLBOX] == first


# ---------------------------------------------------------------------------------------------- section 12 rules
@pytest.mark.parametrize("state,world", [("open", "cabinet_open"), ("closed", "closed")])
def test_world_from_the_cabinet_state(state: str, world: str) -> None:
    assert RZ.realize(spec(cabinet=("storage", state)))["world"] == world


def test_toolbox_in_the_ward_is_parked() -> None:
    r = RZ.realize(spec(toolbox="ward", chair=("office", 2)))
    assert RZ.TOOLBOX in r["parked"] and RZ.TOOLBOX not in r["poses"]
    assert any(n.startswith(RZ.TOOLBOX) and "parked" in n for n in r["notes"])


@pytest.mark.parametrize("room", ["office", "reception", "storage"])
def test_toolbox_placed_in_its_room_on_the_rules(room: str) -> None:
    r = RZ.realize(spec(seed=3, toolbox=room, chair=("ward", 2), bed=("office" if room != "office" else "storage", 2)))
    p = r["poses"][RZ.TOOLBOX]
    assert S.room_of(*p[:2]) == room and in_zone(RZ.TOOLBOX, room, p)
    assert math.hypot(p[0] - S.PLACES[room]["x"], p[1] - S.PLACES[room]["y"]) >= S.TOOLBOX_MIN_DIST_M
    assert S.alone_problems("toolbox", room, p) == []


@pytest.mark.parametrize("room", ["ward", "office", "reception"])
def test_cabinet_outside_storage_is_parked(room: str) -> None:
    r = RZ.realize(spec(cabinet=(room, "open")))
    assert RZ.CABINET in r["parked"] and r["world"] == "cabinet_open"  # the state still picks the world


def test_cabinet_in_storage_and_beds_in_the_ward_stay_frozen() -> None:
    r = RZ.realize(spec(bed=("ward", 2), chair=("office", 2)))
    assert r["poses"][RZ.CABINET] == frozen_pose(RZ.CABINET)
    assert [r["poses"][b] for b in RZ.BEDS[:2]] == [frozen_pose(b) for b in RZ.BEDS[:2]]
    assert RZ.BEDS[2] in r["parked"] and f"{RZ.BEDS[2]}: removed: parked" in r["notes"]  # removed bed


@pytest.mark.parametrize("room", ["office", "reception", "storage"])
def test_beds_outside_the_ward_are_parked(room: str) -> None:
    r = RZ.realize(spec(bed=(room, 3), chair=("ward", 2)))
    assert set(RZ.BEDS) <= set(r["parked"])


def test_removed_chairs_parked_and_kept_row_in_its_room() -> None:
    r = RZ.realize(spec(seed=5, chair=("reception", 2)))
    assert set(RZ.CHAIRS[2:]) <= set(r["parked"])
    row = {c: r["poses"][c] for c in RZ.CHAIRS[:2]}
    assert all(S.room_of(*p[:2]) == "reception" and in_zone(c, "reception", p) for c, p in row.items())
    rooms = {c: "reception" for c in row} | {RZ.TOOLBOX: "storage"}
    assert S.problems(rooms, row | {RZ.TOOLBOX: r["poses"][RZ.TOOLBOX]}) == []  # the subset rules hold


@pytest.mark.parametrize("n", [3, 4])
def test_a_chair_row_that_cannot_fit_is_parked_whole(n: int) -> None:
    """The ward's chair zones (a 0.20 x 0.25 m rectangle and a point) hold at most 2 chairs 0.5 m apart."""
    r = RZ.realize(spec(chair=("ward", n), bed=("office", 2)))
    assert set(RZ.CHAIRS) <= set(r["parked"]) and not set(RZ.CHAIRS) & set(r["poses"])
    assert f"chair row of {n} in the ward: no fit in {RZ.DRAWS} draws: parked" in r["notes"]


def test_parked_list_is_every_spec_object_not_placed() -> None:
    r = RZ.realize(spec(toolbox="ward", cabinet=("office", "closed"), bed=("storage", 2), chair=("ward", 4)))
    assert r["parked"] == list(RZ.SPEC_OBJECTS) and r["poses"] == {}


def test_spec_is_checked() -> None:
    s = spec()
    for bad in (s | {"removed": s["removed"][1:]}, s | {"removed": [*s["removed"], RZ.BEDS[0]]},
                s | {"objects": s["objects"] | {RZ.CABINET: {"room": "storage", "state": "ajar"}}},
                s | {"objects": s["objects"] | {RZ.TOOLBOX: {"room": "hall"}}}):
        with pytest.raises(AssertionError):
            RZ.realize(bad)
    split = spec(chair=("office", 2))
    split["objects"][RZ.CHAIRS[1]] = {"room": "storage"}
    with pytest.raises(AssertionError):
        RZ.realize(split)


# ---------------------------------------------------------------------------------------------- truth and sha
def test_truth_for_spec_objects_and_sha() -> None:
    s = spec(seed=4, toolbox="office", cabinet=("storage", "open"), bed=("ward", 2), chair=("storage", 2))
    r, t = RZ.realize(s), RZ.truth_for_spec(s)
    world = json.loads(S.world_truth_path("cabinet_open").read_text())
    assert {o["id"] for o in t["objects"]} == {o["id"] for o in world["objects"]} - set(r["parked"])
    by = {o["id"]: o for o in t["objects"]}
    for oid, (x, y, yaw) in r["poses"].items():
        assert by[oid]["pose"] == [round(x, 4), round(y, 4), 0.0, 0.0, 0.0, round(yaw, 4)]
        assert by[oid]["room"] == S.room_of(x, y)
    assert by[RZ.TOOLBOX]["room"] == "office" and by[RZ.CABINET]["state"] == "open"
    canon = json.dumps({"seed": 4, "objects": s["objects"], "removed": s["removed"]}, sort_keys=True,
                       separators=(",", ":"))
    assert t["layout_sha256"] == hashlib.sha256(f"{world['layout_sha256']}:spec:{canon}".encode()).hexdigest()
    assert RZ.spec_sha256(s) == hashlib.sha256(canon.encode()).hexdigest()


def test_spec_sha_ignores_key_order_and_generator_extras() -> None:
    s = spec(seed=9)
    shuffled = {"removed": s["removed"], "objects": dict(reversed(s["objects"].items())), "seed": 9,
                "index": 3, "tasks": [{"id": "x"}]}
    assert RZ.spec_sha256(shuffled) == RZ.spec_sha256(s)
    assert RZ.spec_sha256(spec(seed=10)) != RZ.spec_sha256(s)
    assert RZ.truth_for_spec(spec(seed=10))["layout_sha256"] != RZ.truth_for_spec(s)["layout_sha256"]


def test_park_poses_fixed_distinct_and_far_outside() -> None:
    poses = {oid: RZ.park_pose(oid) for oid in RZ.SPEC_OBJECTS}
    assert len({p[:2] for p in poses.values()}) == len(poses)
    xs = [r["rect"][i] for r in S.LAYOUT["rooms"].values() for i in (0, 2)]
    for oid, (x, y, yaw) in poses.items():
        assert x > max(xs) + 20 and y == 40.0 and math.dist((x, y), (20, 20)) > 20
        assert yaw == FROZEN_CLOSED[oid]["pose"][5]


@pytest.mark.parametrize("world", ["closed", "cabinet_open"])
def test_parked_objects_are_outside_every_view(world: str) -> None:
    poses = {oid: RZ.park_pose(oid) for oid in RZ.SPEC_OBJECTS}
    labels = [S._labels(world)[oid] for oid in RZ.SPEC_OBJECTS]
    for place in S.PLACES:
        imgs = RZ.spec_photos(poses, world, place)
        assert all(int(np.isin(img, labels).sum()) == 0 for img in imgs), place


def test_spec_photos_with_frozen_poses_equal_seeded_photos() -> None:
    """Every spec object at its frozen pose = the frozen world: spec_photos == seeded.photos (movables frozen)."""
    poses = {oid: frozen_pose(oid) for oid in RZ.SPEC_OBJECTS}
    for place in ("storage", "hall"):
        for a, b in zip(RZ.spec_photos(poses, "closed", place),
                        S.photos({m: poses[m] for m in S.MOVABLES}, "closed", place)):
            assert np.array_equal(a, b)


# ---------------------------------------------------------------------------------------------- check_spec
@pytest.mark.parametrize("s", [spec(seed=2, toolbox="reception", chair=("office", 3)),
                               spec(seed=6, toolbox="storage", cabinet=("storage", "open"), chair=("storage", 2)),
                               spec(seed=12, toolbox="ward", bed=("reception", 2), chair=("ward", 2))],
                         ids=["office3", "storage2-open", "ward2"])
def test_check_spec_ok(s: dict) -> None:
    ok, bad = V.check_spec(s)
    assert ok and bad == [], bad


def test_check_spec_catches_a_hidden_chair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: two chairs on one pose (one hides the other) breaks G2 / G3."""
    s = spec(seed=2, toolbox="reception", chair=("office", 3))
    r = RZ.realize(s)
    r["poses"][RZ.CHAIRS[1]] = r["poses"][RZ.CHAIRS[0]]
    monkeypatch.setattr(RZ, "realize", lambda spec: copy.deepcopy(r))
    ok, bad = V.check_spec(s)
    assert not ok and any(RZ.CHAIRS[0] in b or RZ.CHAIRS[1] in b for b in bad), bad


# ---------------------------------------------------------------------------------------------- truth side
def test_run_truth_uses_truth_for_spec() -> None:
    s = spec(seed=4, toolbox="office")
    assert truth_mod.run_truth("closed", layout_spec=s) == RZ.truth_for_spec(s)
    with pytest.raises(AssertionError):
        truth_mod.run_truth("cabinet_open", layout_spec=s)  # the spec realizes in closed
    with pytest.raises(AssertionError):
        truth_mod.run_truth("closed", 3, s)


def test_phase4_expected_answers_from_the_spec() -> None:
    e = {"run_id": "x", "task_id": "find-toolbox", "start_place": "hall", "seed": 1,
         "layout_spec": spec(seed=4, toolbox="office")}
    assert p4.expected(e) == {"kind": "find", "answer": "office", "answer_room": "office"}


# ------------------------------------------------------------------------------- run_task: teleport list (no sim)
def test_layout_plan_teleports_every_spec_object() -> None:
    s = spec(seed=5, toolbox="ward", cabinet=("office", "closed"), bed=("ward", 2), chair=("reception", 3))
    frozen = json.loads(S.world_truth_path("closed").read_text())
    want, truth, keys, what = rt.layout_plan("closed", frozen, None, s)
    r = RZ.realize(s)
    assert list(want) == [FROZEN_CLOSED[o]["model_name"] for o in RZ.SPEC_OBJECTS]  # placed and parked, all of them
    for oid in RZ.SPEC_OBJECTS:
        x, y, yaw = r["poses"][oid] if oid in r["poses"] else RZ.park_pose(oid)
        assert want[FROZEN_CLOSED[oid]["model_name"]] == (x, y, FROZEN_CLOSED[oid]["pose"][2], yaw)
    assert r["parked"] == [RZ.TOOLBOX, RZ.CABINET, RZ.BEDS[2], RZ.CHAIRS[3]]
    assert truth == RZ.truth_for_spec(s)
    assert keys == {"layout_spec": RZ.canonical(s), "layout_spec_sha256": RZ.spec_sha256(s), "parked": r["parked"]}
    with pytest.raises(ValueError, match="refused"):
        rt.layout_plan("cabinet_open", frozen, None, s)
    with pytest.raises(AssertionError):
        rt.layout_plan("closed", frozen, 3, s)


class FakeRealizer(types.ModuleType):
    """realize.py's API on test_seeded_layout's fake world (5 objects): the toolbox placed, the cabinet parked."""
    SPEC_OBJECTS = ("toolbox_red_1", "cabinet_blue_1", "chair_orange_1", "chair_orange_2", "bed_blue_1")

    def __init__(self) -> None:
        super().__init__("sim.worlds.clinic.realize")
        self.world = "closed"

    def canonical(self, s: dict) -> dict:
        return {k: s[k] for k in ("seed", "objects", "removed")}

    def spec_sha256(self, s: dict) -> str:
        return hashlib.sha256(json.dumps(self.canonical(s), sort_keys=True).encode()).hexdigest()

    def realize(self, s: dict) -> dict:
        return {"world": self.world, "poses": {"toolbox_red_1": (1.0, 2.0, 0.5), "chair_orange_1": (3.0, -1.0, -1.2),
                                               "chair_orange_2": (-2.0, 0.5, 3.0), "bed_blue_1": (0.0, 0.0, 0.0)},
                "parked": ["cabinet_blue_1"], "notes": []}

    def park_pose(self, oid: str) -> tuple[float, float, float]:
        return 40.0 + 3 * self.SPEC_OBJECTS.index(oid), 40.0, 0.0

    def truth_for_spec(self, s: dict) -> dict:
        t = copy.deepcopy(FROZEN) | {"layout_sha256": hashlib.sha256(f"{LAYOUT}:spec:{s['seed']}".encode()).hexdigest()}
        t["objects"] = [o for o in t["objects"] if o["id"] != "cabinet_blue_1"]
        return t


@pytest.fixture
def fake_rz(monkeypatch: pytest.MonkeyPatch) -> FakeRealizer:
    m = FakeRealizer()
    monkeypatch.setattr(rt, "realizer", lambda: m)
    monkeypatch.setattr(truth_mod, "realizer", lambda: m)
    monkeypatch.setattr(rt, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
    monkeypatch.delenv("SIM_WORLD", raising=False)
    monkeypatch.setattr(truth_mod, "seeded", lambda: pytest.fail("seeded.py used for a spec run"))
    return m


FSPEC = {"seed": 77, "objects": {"toolbox_red_1": {"room": "storage"}}, "removed": []}


def test_spec_run_teleports_all_and_records_the_spec(tmp_path: Path, world: dict, fake_rz: FakeRealizer,
                                                     gz: tsl.FakeGz) -> None:
    r = one(world, tmp_path / "r", FakeAdapter("hall"), layout_spec=FSPEC)
    assert sorted(gz.set_poses()) == sorted(f"m_{o}" for o in FakeRealizer.SPEC_OBJECTS)
    assert gz.pose["m_cabinet_blue_1"][:2] == (43.0, 40.0)  # parked
    run = run_json(tmp_path / "r")
    assert run["layout_spec"] == FSPEC and run["layout_spec_sha256"] == fake_rz.spec_sha256(FSPEC)
    assert run["parked"] == ["cabinet_blue_1"] and "layout_seed" not in run
    assert run["layout_sha256"] == fake_rz.truth_for_spec(FSPEC)["layout_sha256"] and r["score"]["pass"]
    assert scorer.score_run(tmp_path / "r", places_path=world["places_path"])["pass"]  # scored by truth_for_spec


def test_spec_run_readback_miss_is_layout_teleport(tmp_path: Path, world: dict, fake_rz: FakeRealizer,
                                                   gz: tsl.FakeGz, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "READBACK_WAIT_S", 0.0)
    gz.off["m_bed_blue_1"] = (0.05, 0.0)  # a fixed model that did not land
    r = one(world, tmp_path / "r", FakeAdapter("hall"), planner=NoPlanner([], "never"), layout_spec=FSPEC)
    run = run_json(tmp_path / "r")
    assert r["outcome"] == "crash" and run["crash_origin"] == "layout_teleport" and r["score"]["label"] == "sim"
    assert "layout spec" in run["crash"] and run["layout_spec_sha256"] == fake_rz.spec_sha256(FSPEC)


def test_spec_in_another_world_refused_before_gz(tmp_path: Path, world: dict, fake_rz: FakeRealizer,
                                                 gz: tsl.FakeGz) -> None:
    fake_rz.world = "cabinet_open"
    with pytest.raises(ValueError, match="refused"):
        one(world, tmp_path / "r", FakeAdapter("hall"), layout_spec=FSPEC)
    assert gz.log == [] and not (tmp_path / "r").exists()


# ------------------------------------------------------------------------------- phase3_run: spec entries
DEV = ROOT / "tasks" / "phase3_dev.yaml"
SPECS = {rid: spec(seed=i + 1) for i, rid in enumerate(tsl.SEEDS)}


def spec_list(tmp: Path) -> Path:
    runs = [r | {"layout_spec": SPECS[r["run_id"]]} for r in yaml.safe_load(DEV.read_text()) if r["run_id"] in SPECS]
    p = tmp / "spec.yaml"
    p.write_text(yaml.safe_dump(runs))
    return p


def check_spec_routed(out: Path, lst: Path) -> None:
    got = tsl.recs(out)
    assert sorted(got) == sorted(SPECS)
    for e in pr.load_runs(lst):
        r = got[e["run_id"]]
        assert (r["layout_spec"], r["layout_seed"], r["task"], r["start_place"], r["planner"]) == \
            (e["layout_spec"], None, e["task_id"], e["start_place"], e["seed"])


def test_load_runs_layout_spec(tmp_path: Path) -> None:
    lst = spec_list(tmp_path)
    assert [e["layout_spec"] for e in pr.load_runs(lst)] == list(SPECS.values())
    runs = yaml.safe_load(lst.read_text())
    for bad in ([runs[0] | {"layout_spec": 5}, *runs[1:]],  # not a mapping
                [{k: v for k, v in runs[0].items() if k != "layout_spec"}, *runs[1:]],  # some entries only
                [runs[0] | {"layout_seed": 3}, *runs[1:]],  # both on one entry
                [*runs[:-1], {k: v for k, v in runs[-1].items() if k != "layout_spec"} | {"layout_seed": 3}]):  # mixed
        lst.write_text(yaml.safe_dump(bad))
        with pytest.raises(AssertionError):
            pr.load_runs(lst)


def test_layout_spec_routed_serial(tmp_path: Path, worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    ns: dict[str, Any] = {}
    exec(tsl.RECORDER, ns)
    monkeypatch.setattr(rt, "run_one", rt.run_one)  # restored after the test
    lst, out = spec_list(tmp_path), tmp_path / "out"
    assert pr.main(["--list", str(lst), "--out", str(out), "--rehearsal"], runner=ns["install"](rt, pr),
                   job_left=lambda: pytest.fail("job_left called")) == 0
    check_spec_routed(out, lst)


def test_layout_spec_routed_sharded(tmp_path: Path, worker: None) -> None:
    lst, out = spec_list(tmp_path), tmp_path / "out"
    assert pr.main(["--list", str(lst), "--out", str(out), "--rehearsal", "--shards", "2", "--shard-env", "0=40:s0",
                    "--shard-env", "1=41:s1"], runner=lambda: pytest.fail("the coordinator never drives a sim"),
                   job_left=lambda: pytest.fail("job_left called")) == 0
    check_spec_routed(out, lst)


def test_layout_spec_routed_cross_job(tmp_path: Path, worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    lst, out, led = spec_list(tmp_path), tmp_path / "out", tmp_path / "ledger.jsonl"

    def main(*a: str) -> int:
        return pr.main(["--list", str(lst), "--rehearsal", "--ledger", str(led), *a],
                       runner=lambda: pytest.fail("the coordinator never drives a sim"), job_left=lambda: 999.0)
    assert main("--out", str(out), "--split", "2:2") == 0
    for j in (0, 1):
        monkeypatch.setenv("SLURM_JOB_ID", f"10{j}")
        assert main("--job-part", str(out), "--job", str(j), "--shards", "2", "--shard-env", f"0=4{j}:j{j}s0",
                    "--shard-env", f"1=5{j}:j{j}s1") == 0
    check_spec_routed(out, lst)


# ------------------------------------------------------------------------------- seeded.py unchanged by 011
def test_seeded_placements_unchanged_for_seeds_0_to_50() -> None:
    snap = json.loads(SNAPSHOT.read_text())
    assert sorted(map(int, snap)) == list(range(51))
    for s in range(51):
        try:
            got = {"poses": {k: list(v) for k, v in S.placements(s).items()}}
        except S.SeedRejected as e:
            got = {"rejected": e.reasons}
        assert got == snap[str(s)], s
