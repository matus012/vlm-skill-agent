"""probes/phase5_gates.py (task 008 G1-G3 + measure), no sim, no VLM: the counted list run on the FakeAdapter through
adapters.run_task.run_one with each entry's layout_seed and world. A fake seeded module (never the real placement
code) puts the movables in the rooms tasks/phase5_seeds.json records for that seed, on two fake worlds (closed /
cabinet_open); the fake adapter shows each movable ahead of its room's place; teleport is faked as landed. The run
dirs are then edited to make each gate fail. phase3_gates.require_merged is replaced by a recorder. No run id or seed
is hard-coded: they come from the list (env P5_LISTS_DIR: another dir's phase5_counted.yaml, e.g. a pick_phase5 run)."""
import copy
import os
import hashlib
import json
import shutil
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
import agent.fake_adapter as fa  # noqa: E402
import agent.truth as at  # noqa: E402
from adapters.hand_help import HandHelp  # noqa: E402
from agent.runtime import MALFORMED_LIMIT  # noqa: E402
from agent.tasks import task_by_id  # noqa: E402
from probes import phase3_gates as p3  # noqa: E402
from probes import phase5_gates as g  # noqa: E402
from probes import pick_phase5 as pick  # noqa: E402
from probes.phase3_run import sha256_file  # noqa: E402
from test_phase3_gates import FakeVlm, edit_run, edit_steps  # noqa: E402

ROOMS = ["hall", "ward", "reception", "office", "storage"]
LAYOUTS = {"closed": "f" * 64, "cabinet_open": "e" * 64}
MOVABLE = {"toolbox_red_1": 10, "chair_orange_1": 20, "chair_orange_2": 21, "chair_orange_3": 22, "chair_orange_4": 23}
FIXED = [fa.Obj(30, 2.5, -5.5, (40, 60, 200)), fa.Obj(1, -1.0, 2.0, (60, 60, 230))]  # cabinet (storage, right), bed
COUNTED = Path(os.environ.get("P5_LISTS_DIR", ROOT / "tasks")) / "phase5_counted.yaml"
RUNS = yaml.safe_load(COUNTED.read_text())
BY_KIND = {k: [r["run_id"] for r in RUNS if task_by_id(r["task_id"]).kind == k] for k in g.TYPES}
WRONG = {k: set(v[:3]) for k, v in BY_KIND.items()}  # 7/10 per type pass
RIGHT = {k: [x for x in v if x not in WRONG[k]] for k, v in BY_KIND.items()}
TB = next(r["run_id"] for r in RUNS if r["task_id"] == "find-toolbox" and r["run_id"] in RIGHT["find"])
REJECTED = min(set(range(700)) - set(g.ok_seeds()))  # a seed the vetting rejected
DEV = COUNTED.parent / "phase5_dev.yaml"
SPOT_REAL = Path("runs/2026-11-05/008-d17-144468/p5_spot/spot.json")
DIFF = [{"kind": "check", "gate": "g2", "live": False, "raycast": True}]
INV = "inv-008"


def frozen(world: str) -> dict[str, Any]:
    objs = [{"id": oid, "class": oid.split("_")[0], "colour": oid.split("_")[1], "label": lab,
             "room": "storage" if oid.startswith("toolbox") else "reception", "state": None}
            for oid, lab in MOVABLE.items()]
    objs += [{"id": "cabinet_blue_1", "class": "cabinet", "colour": "blue", "label": 30, "room": "storage",
              "state": "open" if world == "cabinet_open" else "closed"},
             {"id": "bed_blue_1", "class": "bed", "colour": "blue", "label": 1, "room": "ward", "state": None}]
    for o in objs:
        o |= {"model_name": f"m_{o['id']}", "pose": [0.0, 0.0, 0.0, 0, 0, 0]}
    return {"layout_sha256": LAYOUTS[world], "rooms": {r: {} for r in ROOMS}, "objects": objs}


class FakeSeeded(types.ModuleType):
    """seeded.py's API: the rooms come from tasks/phase5_seeds.json (the record the lists were chosen from)."""

    def __init__(self) -> None:
        super().__init__("fake_seeded")
        self.ok = g.ok_seeds()

    def placements(self, seed: int) -> dict[str, tuple[float, float, float]]:
        return {oid: (1.0, 1.0, 0.0) for oid in self.ok[seed]["rooms"]}

    def truth_for(self, seed: int, world: str) -> dict[str, Any]:
        t = copy.deepcopy(frozen(world))
        t["layout_sha256"] = hashlib.sha256(f"{LAYOUTS[world]}:{seed}".encode()).hexdigest()
        for o in t["objects"]:
            o["room"] = self.ok[seed]["rooms"].get(o["id"], o["room"])
        return t


def objects_for(truth: dict[str, Any]) -> list[fa.Obj]:
    """The fake adapter's world for a seeded truth: each movable 1.2 m ahead of its room's place, side by side."""
    out, n = [], {r: 0 for r in ROOMS}
    for o in truth["objects"]:
        if o["id"] in MOVABLE:
            x, y, _ = fa.PLACES[o["room"]]
            out.append(fa.Obj(o["label"], x + 1.2, y + (-1.0, -0.5, 0.0, 0.5, 1.0)[n[o["room"]]], (200, 100, 30)))
            n[o["room"]] += 1
    return out + FIXED


def script(e: dict[str, Any], right: bool) -> list[dict[str, Any]]:
    ex = e["expected"]
    photo = 4 if task_by_id(e["task_id"]).target.get("object") == "cabinet_blue_1" else 1
    a = ex["answer"]
    answer = a if right else {"yes": "no", "no": "yes"}.get(a) or (str(int(a) + 1) if a.isdigit() else "hall")
    return [{"name": "navigate_to", "args": {"place": ex["answer_room"]}}, {"name": "look_around", "args": {}},
            {"name": "report", "args": {"answer": answer, "photo": photo, "note": "seen"}}]


def make_run(places: Path, d: Path, e: dict[str, Any], steps: list, fake: FakeSeeded, mp: pytest.MonkeyPatch) -> None:
    ad = fa.FakeAdapter("hall")
    mp.setenv("SIM_WORLD", e.get("world", "closed"))
    mp.setattr(fa, "OBJECTS", objects_for(fake.truth_for(e["layout_seed"], e.get("world", "closed"))))
    rt.run_one(ad, ad.pose, task_by_id(e["task_id"]), FakeVlm(steps, e["seed"]), d, robot="fake", tag="fake",
               places_path=places, hand_help=lambda: HandHelp(v_max_mps=1.0), start_place=e["start_place"],
               malformed_limit=MALFORMED_LIMIT, layout_seed=e["layout_seed"],
               extra_meta={"invocation_id": INV, "node": f"n{int(e['seed']) % 2}"})


def patch_world(mp: pytest.MonkeyPatch, dirs: dict[str, Path], fake: FakeSeeded) -> None:
    mp.setattr(at, "WORLD_DIRS", dict(dirs))
    mp.setattr(rt, "WORLD_DIRS", dict(dirs))
    mp.setattr(at, "seeded", lambda: fake)
    mp.setattr(rt, "seeded", lambda: fake)


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("p5")
    dirs = {}
    for w in LAYOUTS:
        (root / w).mkdir()
        (root / w / "truth.json").write_text(json.dumps(frozen(w)))
        (root / w / "map_knowledge.json").write_text(json.dumps({"world": "fake", "layout_sha256": LAYOUTS[w],
                                                                 "rooms": {r: {} for r in ROOMS}}))
        dirs[w] = root / w
    places = root / "places.yaml"
    places.write_text(yaml.safe_dump({"places": {n: {"x": x, "y": y, "yaw": a} for n, (x, y, a) in fa.PLACES.items()}}))
    out, fake = root / "out", FakeSeeded()
    with pytest.MonkeyPatch.context() as mp:
        patch_world(mp, dirs, fake)
        mp.setattr(rt, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
        mp.setattr(rt, "teleport", lambda want, log=None: None)  # landed
        mp.setattr(g, "COUNTED", COUNTED)
        entries = g.load_list(g.COUNTED)
        bad = set().union(*WRONG.values())
        for e in entries:
            make_run(places, out / e["run_id"], e, script(e, e["run_id"] not in bad), fake, mp)
    (out / "invocation.json").write_text(json.dumps({"invocation_id": INV, "list_sha256": sha256_file(COUNTED)}))
    spot = root / "spot"
    spot.mkdir()
    seeds = sorted(pick.spot_seeds())[:7]  # hpc/plans/p5_spot.sh: in no list; 6 agree, the 7th disagrees
    assert len(seeds) == 7
    (spot / "spot.json").write_text(json.dumps(
        {"seeds": {str(x): {"agree": i < 6, "diffs": [] if i < 6 else DIFF} for i, x in enumerate(seeds)},
         "agree_all": False}))
    return {"out": out, "entries": entries, "dirs": dirs, "fake": fake, "spot": spot}


@pytest.fixture
def case(base: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    out, spot = tmp_path / "out", tmp_path / "spot"
    shutil.copytree(base["out"], out)
    shutil.copytree(base["spot"], spot)
    patch_world(monkeypatch, base["dirs"], base["fake"])
    monkeypatch.setattr(g, "COUNTED", COUNTED)
    monkeypatch.setattr(g, "DEV", DEV)
    merged: list[Path] = []
    monkeypatch.setattr(p3, "require_merged", merged.append)
    return {**base, "out": out, "spot": spot, "entries": copy.deepcopy(base["entries"]), "merged": merged}


def spot_edit(c: dict, fn: Any) -> None:
    f = c["spot"] / "spot.json"
    s = json.loads(f.read_text())
    fn(s)
    f.write_text(json.dumps(s))


# ------------------------------------------------------------------------------------------------------------ G1
def test_runs_are_seeded_and_scored_on_their_layout(case: dict) -> None:
    run = json.loads((case["out"] / TB / "run.json").read_text())
    e = next(e for e in case["entries"] if e["run_id"] == TB)
    assert run["layout_seed"] == e["layout_seed"]
    assert run["layout_sha256"] == case["fake"].truth_for(e["layout_seed"], "closed")["layout_sha256"]


def test_g1_seven_of_ten_per_type(case: dict) -> None:
    v = g.g1_verdict(case["out"], case["entries"], True)
    assert v["pass"], (v["common_problems"], {k: t["passed"] for k, t in v["types"].items()})
    assert {k: t["passed"] for k, t in v["types"].items()} == {k: 7 for k in g.TYPES}
    assert case["merged"] == [case["out"]]


@pytest.mark.parametrize("kind", g.TYPES)
def test_one_more_failure_fails_that_type_only(case: dict, kind: str) -> None:
    shutil.rmtree(case["out"] / RIGHT[kind][-1])
    v = g.g1_verdict(case["out"], case["entries"], True)
    assert not v["pass"] and v["types"][kind]["passed"] == 6
    assert all(t["pass"] for k, t in v["types"].items() if k != kind)


def test_layout_seed_mismatch_fails_the_run(case: dict) -> None:
    e = next(e for e in case["entries"] if e["run_id"] == TB)
    e["layout_seed"] = e["layout_seed"] + 1
    row = next(r for r in g.g1_verdict(case["out"], case["entries"], True)["types"]["find"]["runs"]
               if r["run_id"] == TB)
    assert not row["pass"] and any("layout_seed" in p for p in row["problems"])


def test_scored_against_the_seeded_truth(case: dict) -> None:
    """Move the toolbox of TB's layout elsewhere in the (fake) seeded truth: its right answer becomes wrong."""
    e = next(e for e in case["entries"] if e["run_id"] == TB)
    fake, real = case["fake"], case["fake"].truth_for

    def moved(seed: int, world: str) -> dict:
        t = real(seed, world)
        if seed == e["layout_seed"]:
            next(o for o in t["objects"] if o["id"] == "toolbox_red_1")["room"] = "ward"
        return t
    fake.truth_for = moved
    try:
        v = g.g1_verdict(case["out"], case["entries"], True)
    finally:
        del fake.truth_for
    assert v["types"]["find"]["passed"] == 6 and not v["pass"]


@pytest.mark.parametrize("edit,needle", [
    (lambda out: edit_run(out / TB, git_dirty=True), "dirty"),
    (lambda out: edit_run(out / TB, system_prompt_sha256="0" * 64), "system_prompt_sha256"),
    (lambda out: edit_run(out / TB, invocation_id="other"), "invocation"),
    (lambda out: (out / "invocation.json").write_text(json.dumps({"invocation_id": INV, "list_sha256": "0"})),
     "list_sha256")])
def test_common_problems_fail_every_type(case: dict, edit: Any, needle: str) -> None:
    edit(case["out"])
    v = g.g1_verdict(case["out"], case["entries"], True)
    assert not v["pass"] and not any(t["pass"] for t in v["types"].values())
    assert any(needle in p for p in v["common_problems"]), v["common_problems"]


def test_not_counted_never_passes(case: dict) -> None:
    assert not g.g1_verdict(case["out"], case["entries"], False)["pass"]
    assert not g.g2_verdict(case["out"], case["entries"], False, {"returncode": 0})["pass"]
    assert not g.g3_verdict(case["out"], case["entries"], False, case["spot"])["pass"]


def test_every_gate_refuses_an_unmerged_invocation(case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(out: Path) -> None:
        raise SystemExit(f"{out}: merged.json ok false")
    monkeypatch.setattr(p3, "require_merged", refuse)
    for gate in (lambda: g.g1_verdict(case["out"], case["entries"], True),
                 lambda: g.g2_verdict(case["out"], case["entries"], True, {"returncode": 0}),
                 lambda: g.g3_verdict(case["out"], case["entries"], True, case["spot"]),
                 lambda: g.measure(case["out"], case["entries"]),
                 lambda: g.main(["g3", "--out", str(case["out"]), "--spot", str(case["spot"])])):
        with pytest.raises(SystemExit):
            gate()


# ------------------------------------------------------------------------------------------------------------ G2
def test_g2_passes_then_packet_coordinate_fails(case: dict) -> None:
    assert g.g2_verdict(case["out"], case["entries"], True, {"returncode": 0})["pass"]
    edit_steps(case["out"] / "p5c15", lambda r: r["packet"]["texts"].append("you are at (2.5, -3.9)"))
    v = g.g2_verdict(case["out"], case["entries"], True, {"returncode": 0})
    assert not v["pass"] and "p5c15" in v["failing_hits"]


def test_g2_log_rules_missing_run_and_no_ros(case: dict) -> None:
    sc = json.loads((case["out"] / "p5c01" / "score.json").read_text())
    (case["out"] / "p5c01" / "score.json").write_text(json.dumps({**sc, "label": None}))
    v = g.g2_verdict(case["out"], case["entries"], True, {"returncode": 0})
    assert not v["pass"] and "p5c01" in v["log_problems"]
    shutil.rmtree(case["out"] / "p5c02")
    assert g.g2_verdict(case["out"], case["entries"], True, {"returncode": 0})["missing_runs"] == ["p5c02"]


def test_g2_needs_the_no_answer_tests(case: dict) -> None:
    assert not g.g2_verdict(case["out"], case["entries"], True, {"returncode": 1})["pass"]
    assert p3.NO_ANSWER_TESTS == ("tests/test_agent_no_ros.py", "tests/test_truth_readers.py")


# ------------------------------------------------------------------------------------------------------------ G3
def test_g3_six_agree_one_disagreeing_seed_in_no_list_passes(case: dict) -> None:
    v = g.g3_verdict(case["out"], case["entries"], True, case["spot"])
    assert v["pass"], v["spot_problems"]
    assert len(v["layout_seeds"]) == 40 and len(v["spot_agree"]) == 6 and v["agree_all"] is False
    (bad,) = v["spot_disagree"]
    assert v["spot_disagree"][bad] == {"agree": False, "diffs": DIFF}  # the record verbatim
    assert g.g3_verdict(case["out"], case["entries"], True, case["spot"] / "spot.json")["pass"]  # file form


def mark_disagree(c: dict, seed: int) -> None:
    spot_edit(c, lambda s: s["seeds"].update({str(seed): {"agree": False, "diffs": DIFF}}))


def test_g3_disagreeing_seed_in_the_counted_list_fails(case: dict) -> None:
    mark_disagree(case, case["entries"][0]["layout_seed"])
    v = g.g3_verdict(case["out"], case["entries"], True, case["spot"])
    assert not v["pass"] and any("counted list" in p for p in v["spot_problems"])


def test_g3_disagreeing_seed_in_the_dev_list_fails(case: dict) -> None:
    mark_disagree(case, yaml.safe_load(DEV.read_text())[0]["layout_seed"])
    v = g.g3_verdict(case["out"], case["entries"], True, case["spot"])
    assert not v["pass"] and any("dev list" in p for p in v["spot_problems"])


@pytest.mark.parametrize("fn", [
    lambda s: s["seeds"][next(iter(s["seeds"]))].update(agree=False),  # 4 agree
    lambda s: s.update(seeds={k: v for k, v in list(s["seeds"].items())[:4]}),  # 3 agree
])
def test_g3_fewer_than_five_agreeing_fails(case: dict, fn: Any) -> None:
    spot_edit(case, lambda s: s["seeds"][list(s["seeds"])[1]].update(agree=False))  # 5 agree
    assert g.g3_verdict(case["out"], case["entries"], True, case["spot"])["pass"]
    spot_edit(case, fn)
    v = g.g3_verdict(case["out"], case["entries"], True, case["spot"])
    assert not v["pass"] and v["spot_problems"]


def test_g3_rejected_seed_does_not_count_as_agreeing(case: dict) -> None:
    spot_edit(case, lambda s: s.update(seeds={**{k: v for k, v in list(s["seeds"].items())[:4]},
                                              str(REJECTED): {"agree": True}}))
    assert not g.g3_verdict(case["out"], case["entries"], True, case["spot"])["pass"]


@pytest.mark.skipif(not SPOT_REAL.is_file(), reason="the PERUN spot-check evidence is not on this machine")
def test_g3_on_the_real_spot_check(case: dict) -> None:
    v = g.g3_verdict(case["out"], case["entries"], True, SPOT_REAL)
    assert v["pass"], v["spot_problems"]
    assert set(v["spot_disagree"]) == {int(s) for s, r in json.loads(SPOT_REAL.read_text())["seeds"].items()
                                       if r["agree"] is not True} and not v["spot_seeds_in_counted"]


def test_g3_missing_spot_and_not_ok_layout_seed_fail(case: dict) -> None:
    assert not g.g3_verdict(case["out"], case["entries"], True, None)["pass"]
    assert not g.g3_verdict(case["out"], case["entries"], True, case["spot"] / "nope.json")["pass"]
    case["entries"][0]["layout_seed"] = REJECTED
    v = g.g3_verdict(case["out"], case["entries"], True, case["spot"])
    assert not v["pass"] and v["not_ok"] == [REJECTED]


# ------------------------------------------------------------------------------------------------------- measure
def test_measure_splits_and_memory_signal(case: dict) -> None:
    slow = RUNS[4]["run_id"]
    edit_run(case["out"] / slow, sim_s=1.0, wall_s=10.0)
    # a Find run whose toolbox is not in storage; make its first move go to storage
    mem = next(e for e in case["entries"] if e["expected"]["kind"] == "find" and e["expected"]["memory_eligible"])
    assert mem["expected"]["memory_room"] == "storage" != mem["expected"]["answer_room"]
    edit_steps(case["out"] / mem["run_id"],
               lambda r: r["call"]["name"] == "navigate_to" and r["call"]["args"].update(place="storage"))
    m = g.measure(case["out"], case["entries"], job_gpu_h=4.0)
    assert sum(t["runs"] for t in m["per_type"].values()) == 40 and m["h200_h_per_run_job"] == 0.1
    assert slow in m["flagged_below_0.6"]
    cnt = [e for e in case["entries"] if e["expected"]["kind"] == "count"]
    assert m["count_by_photos"] == {p: {"runs": sum(e["expected"]["photos"] == p for e in cnt),
                                        "passed": sum(e["expected"]["photos"] == p and e["run_id"] not in WRONG["count"]
                                                      for e in cnt)} for p in ("one", "several")}
    ms = m["memory_signal"]
    eligible = sum(e["expected"]["memory_eligible"] for e in case["entries"])
    assert ms["eligible"] == eligible > 0 and ms["went"] == 1 and ms["per_type"]["find"]["went"] == 1
    assert ms["per_type"]["check_state"]["eligible"] == 0
    assert set(m["per_type"]["find"]["by_start"]) == {"hall", "answer_room", "other_room"}
    assert json.loads((case["out"] / "measure.json").read_text())["memory_signal"]["went"] == 1


def test_cli(case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p3, "no_ros_test", lambda: {"returncode": 0, "summary": "fake"})
    assert g.main(["all", "--out", str(case["out"]), "--spot", str(case["spot"]), "--rehearsal"]) == 1
    assert json.loads((case["out"] / "g3.json").read_text())["rehearsal"] is True
    assert g.main(["all", "--out", str(case["out"]), "--spot", str(case["spot"])]) == 0
    assert all((case["out"] / f"{n}.json").is_file() for n in ("g1", "g2", "g3", "measure"))
