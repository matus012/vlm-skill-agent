"""probes/phase4_gates.py (task 007 G1-G3 + measure) and the world-aware harness / scorer, no sim, no VLM: the counted
list run on the FakeAdapter through adapters.run_task.run_one with SIM_WORLD per entry, two fake worlds (closed /
cabinet_open, distinct layout sha256) swapped into agent.truth.WORLD_DIRS, then edited to make each gate fail.
phase3_gates.require_merged (owned by the cross-job split) is replaced by a fake that records its calls."""
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import adapters.run_task as rt  # noqa: E402
import agent.truth as at  # noqa: E402
from adapters.hand_help import HandHelp  # noqa: E402
from agent.fake_adapter import PLACES, FakeAdapter  # noqa: E402
from agent.runtime import MALFORMED_LIMIT  # noqa: E402
from agent.scorer import score_run  # noqa: E402
from agent.tasks import task_by_id  # noqa: E402
from probes import phase3_gates as p3  # noqa: E402
from probes import phase4_gates as g  # noqa: E402
from probes.phase3_run import sha256_file  # noqa: E402
from test_phase3_gates import ROOMS, TRUTH, FakeVlm, edit_run, edit_steps  # noqa: E402

LAYOUTS = {"closed": "f" * 64, "cabinet_open": "e" * 64}
WRONG = {"check_state": {"p4c01", "p4c07", "p4c09"}, "count": {"p4c11", "p4c16", "p4c18"},
         "conditional": {"p4c21", "p4c25", "p4c30"}, "find": {"p4c31", "p4c34", "p4c40"}}  # 7/10 per type
INV = "inv-007"


def fake_world(d: Path, world: str) -> Path:
    t = copy.deepcopy(TRUTH)
    t["layout_sha256"] = LAYOUTS[world]
    for o in t["objects"]:
        if o["id"] == "cabinet_blue_1":
            o["state"] = "open" if world == "cabinet_open" else "closed"
    d.mkdir(parents=True)
    (d / "truth.json").write_text(json.dumps(t))
    (d / "map_knowledge.json").write_text(json.dumps({"world": "fake", "layout_sha256": LAYOUTS[world],
                                                      "rooms": {r: {} for r in ROOMS}}))
    return d


def script(e: dict[str, Any], right: bool) -> list[dict[str, Any]]:
    ex = e["expected"]
    target = task_by_id(e["task_id"]).target
    photo = 4 if target.get("object") == "cabinet_blue_1" else 1  # fake adapter: cabinet to the right, rest ahead
    a = ex["answer"]  # a wrong answer stays in the kind's answer format (yes / no, integer, room)
    answer = a if right else {"yes": "no", "no": "yes"}.get(a) or (str(int(a) + 1) if a.isdigit() else "office")
    return [{"name": "navigate_to", "args": {"place": ex["answer_room"]}}, {"name": "look_around", "args": {}},
            {"name": "report", "args": {"answer": answer, "photo": photo, "note": "seen"}}]


def make_run(places: Path, d: Path, e: dict[str, Any], steps: list, mp: pytest.MonkeyPatch) -> Path:
    """One listed run through run_one: world from SIM_WORLD, truth + map from WORLD_DIRS (no truth_path given)."""
    ad = FakeAdapter("hall")
    mp.setenv("SIM_WORLD", g.world_of(e))
    rt.run_one(ad, ad.pose, task_by_id(e["task_id"]), FakeVlm(steps, e["seed"]), d, robot="fake", tag="fake",
               places_path=places, hand_help=lambda: HandHelp(v_max_mps=1.0), start_place=e["start_place"],
               malformed_limit=MALFORMED_LIMIT, extra_meta={"invocation_id": INV, "node": f"n{int(e['seed']) % 2}"})
    mp.delenv("SIM_WORLD")
    return d


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("p4")
    dirs = {w: fake_world(root / w, w) for w in LAYOUTS}
    places = root / "places.yaml"
    places.write_text(yaml.safe_dump({"places": {n: {"x": x, "y": y, "yaw": a} for n, (x, y, a) in PLACES.items()}}))
    out = root / "out"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(at, "WORLD_DIRS", dirs)
        mp.setattr(rt, "WORLD_DIRS", dirs)
        mp.setattr(rt, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
        entries = g.load_list(g.COUNTED)
        bad = set().union(*WRONG.values())
        for e in entries:
            make_run(places, out / e["run_id"], e, script(e, e["run_id"] not in bad), mp)
    (out / "invocation.json").write_text(json.dumps({"invocation_id": INV, "list_sha256": sha256_file(g.COUNTED)}))
    return {"out": out, "entries": entries, "dirs": dirs, "places": places}


@pytest.fixture
def case(base: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    out = tmp_path / "out"
    shutil.copytree(base["out"], out)
    monkeypatch.setattr(at, "WORLD_DIRS", dict(base["dirs"]))
    merged: list[Path] = []
    monkeypatch.setattr(p3, "require_merged", merged.append, raising=False)  # fake until/besides the real one
    return {**base, "out": out, "entries": copy.deepcopy(base["entries"]), "merged": merged}


# ------------------------------------------------------------------------------------------- harness + scorer
def test_run_json_records_world_and_scores_against_it(case: dict) -> None:
    run = json.loads((case["out"] / "p4c04" / "run.json").read_text())  # open? in cabinet_open -> yes
    assert run["world"] == "cabinet_open" and run["layout_sha256"] == LAYOUTS["cabinet_open"]
    assert score_run(case["out"] / "p4c04")["pass"]
    assert json.loads((case["out"] / "p4c02" / "run.json").read_text())["world"] == "closed"
    with pytest.raises(ValueError, match="layout"):  # the layout sha check stays
        score_run(case["out"] / "p4c04", truth_path=case["dirs"]["closed"] / "truth.json")


def test_run_without_world_scores_as_closed(case: dict) -> None:
    d = case["out"] / "p4c02"
    before = score_run(d)
    run = json.loads((d / "run.json").read_text())
    run.pop("world")
    (d / "run.json").write_text(json.dumps(run))
    assert score_run(d) == before == score_run(d, truth_path=case["dirs"]["closed"] / "truth.json")


def test_sim_world_refuses_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIM_WORLD", "ajar")
    with pytest.raises(SystemExit):
        rt.sim_world()
    monkeypatch.delenv("SIM_WORLD")
    assert rt.sim_world() == "closed"


def test_world_tag_per_world(monkeypatch: pytest.MonkeyPatch) -> None:
    assert rt.WORLD_TAG_KW["closed"] == {}  # closed keeps phase1-world-frozen* (tests/test_world_tag.py)
    kw = rt.WORLD_TAG_KW["cabinet_open"]
    assert all(f.startswith("sim/worlds/clinic/generated_cabinet_open/") for f in kw["files"])
    assert not kw["glob"].startswith("refs/tags/phase1-world-frozen")


# ------------------------------------------------------------------------------------------------------- G1 / G2
def test_g1_g2_seven_of_ten_per_type(case: dict) -> None:
    v1, v2 = g.g1_verdict(case["out"], case["entries"], True), g.g2_verdict(case["out"], case["entries"], True)
    assert v1["pass"] and v2["pass"], (v1["common_problems"], {k: t["passed"] for k, t in v1["types"].items()})
    assert {k: t["passed"] for k, t in v1["types"].items()} == {"check_state": 7, "count": 7, "conditional": 7}
    assert v2["types"]["find"]["passed"] == 7 and set(v1["types"]) == set(g.TYPES)
    assert case["merged"] == [case["out"], case["out"]]


@pytest.mark.parametrize("kind,run_id", [("check_state", "p4c04"), ("count", "p4c12"), ("conditional", "p4c22"),
                                         ("find", "p4c32")])
def test_one_more_failure_fails_that_type_only(case: dict, kind: str, run_id: str) -> None:
    shutil.rmtree(case["out"] / run_id)  # a missing run is a fail
    v = (g.g2_verdict if kind == "find" else g.g1_verdict)(case["out"], case["entries"], True)
    assert not v["pass"] and v["types"][kind]["passed"] == 6
    assert all(t["pass"] for k, t in v["types"].items() if k != kind)


def test_wrong_world_truth_flips_check_state(case: dict) -> None:
    """The cabinet_open runs pass only because they are scored against the open truth: swap the open world's truth
    for the closed one's states (same layout sha) and its 'open?' -> yes runs fail."""
    t = json.loads((case["dirs"]["closed"] / "truth.json").read_text())
    t["layout_sha256"] = LAYOUTS["cabinet_open"]
    d = case["out"].parent / "open_as_closed"
    d.mkdir()
    (d / "truth.json").write_text(json.dumps(t))
    at.WORLD_DIRS["cabinet_open"] = d
    v = g.g1_verdict(case["out"], case["entries"], True)
    assert v["types"]["check_state"]["passed"] < 7 and not v["pass"]


def test_world_mismatch_is_a_problem(case: dict) -> None:
    e = next(e for e in case["entries"] if e["run_id"] == "p4c04")
    e.pop("world")
    v = g.g1_verdict(case["out"], case["entries"], True)
    row = next(r for r in v["types"]["check_state"]["runs"] if r["run_id"] == "p4c04")
    assert not row["pass"] and any("world" in p for p in row["problems"])


def test_not_counted_never_passes(case: dict) -> None:
    assert not g.g1_verdict(case["out"], case["entries"], False)["pass"]
    assert not g.g2_verdict(case["out"], case["entries"], False)["pass"]
    assert not g.g3_verdict(case["out"], case["entries"], False, {"returncode": 0})["pass"]


@pytest.mark.parametrize("edit,needle", [
    (lambda out: edit_run(out / "p4c33", git_dirty=True), "dirty"),
    (lambda out: edit_run(out / "p4c33", system_prompt_sha256="0" * 64), "system_prompt_sha256"),
    (lambda out: edit_run(out / "p4c33", invocation_id="other"), "invocation"),
    (lambda out: (out / "invocation.json").write_text(json.dumps({"invocation_id": INV, "list_sha256": "0"})),
     "list_sha256")])
def test_common_problems_fail_every_type(case: dict, edit: Any, needle: str) -> None:
    edit(case["out"])
    v = g.g1_verdict(case["out"], case["entries"], True)
    assert not v["pass"] and not any(t["pass"] for t in v["types"].values())
    assert any(needle in p for p in v["common_problems"]), v["common_problems"]


def test_unmerged_invocation_refused_before_reading(case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(out: Path) -> None:
        raise SystemExit(f"{out}: merged.json ok false")
    monkeypatch.setattr(p3, "require_merged", refuse, raising=False)
    for gate in (lambda: g.g1_verdict(case["out"], case["entries"], True),
                 lambda: g.g2_verdict(case["out"], case["entries"], True),
                 lambda: g.g3_verdict(case["out"], case["entries"], True, {"returncode": 0})):
        with pytest.raises(SystemExit):
            gate()


def test_real_require_merged_signature_if_present() -> None:
    import inspect
    real = getattr(p3, "require_merged", None)
    if real is None:
        pytest.skip("phase3_gates.require_merged not in the tree yet (the gates call it by name; fakes above)")
    assert list(inspect.signature(real).parameters) == ["out"]


# ------------------------------------------------------------------------------------------------------------ G3
def test_g3_passes_then_coordinate_in_packet_fails(case: dict) -> None:
    assert g.g3_verdict(case["out"], case["entries"], True, {"returncode": 0})["pass"]
    edit_steps(case["out"] / "p4c15", lambda r: r["packet"]["texts"].append("you are at (2.5, -3.9)"))
    v = g.g3_verdict(case["out"], case["entries"], True, {"returncode": 0})
    assert not v["pass"] and "p4c15" in v["failing_hits"]


def test_g3_log_rules_and_missing_run(case: dict) -> None:
    sc = json.loads((case["out"] / "p4c01" / "score.json").read_text())
    (case["out"] / "p4c01" / "score.json").write_text(json.dumps({**sc, "label": None}))
    v = g.g3_verdict(case["out"], case["entries"], True, {"returncode": 0})
    assert not v["pass"] and "p4c01" in v["log_problems"]
    shutil.rmtree(case["out"] / "p4c02")
    assert g.g3_verdict(case["out"], case["entries"], True, {"returncode": 0})["missing_runs"] == ["p4c02"]
    assert not g.g3_verdict(case["out"], case["entries"], True, {"returncode": 1})["pass"]


# ------------------------------------------------------------------------------------------------------- measure
def test_measure_per_type_start_split_and_slow_flag(case: dict) -> None:
    edit_run(case["out"] / "p4c05", sim_s=1.0, wall_s=10.0)
    m = g.measure(case["out"], case["entries"], job_gpu_h=4.0)
    cs = m["per_type"]["check_state"]
    assert cs["runs"] == 10 and cs["passed"] == 7 and cs["failure_labels"] == {"vlm": 3}
    assert cs["by_start"] == {"hall": {"runs": 3, "passed": 1}, "answer_room": {"runs": 1, "passed": 1},
                              "other_room": {"runs": 6, "passed": 5}}
    assert sum(t["runs"] for t in m["per_type"].values()) == 40 and m["h200_h_per_run_job"] == 0.1
    assert "p4c05" in m["flagged_below_0.6"] and set(m["sim_wall_per_node"]) <= {"n0", "n1"}
    assert all(t["steps_per_run"]["n"] == 10 for t in m["per_type"].values())
    assert (case["out"] / "measure.json").is_file()


def test_cli_rehearsal_writes_and_fails(case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p3, "no_ros_test", lambda: {"returncode": 0, "summary": "fake"})
    assert g.main(["all", "--out", str(case["out"]), "--rehearsal"]) == 1
    assert json.loads((case["out"] / "g1.json").read_text())["rehearsal"] is True
    assert g.main(["g1", "--out", str(case["out"])]) == 0
