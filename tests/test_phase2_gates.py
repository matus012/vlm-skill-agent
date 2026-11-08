"""Gate verdict roll-ups of probes/phase2_gates.py (task 003 I), no sim: real run dirs from adapters/run_task.run_one on
the FakeAdapter (the fake world + synthetic truth of tests/test_scorer_e2e.py), then the gate checks on them."""
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import adapters.run_task as rt  # noqa: E402
from adapters.run_task import parse_place, run_one  # noqa: E402
from agent.fake_adapter import PLACES, FakeAdapter  # noqa: E402
from agent.planners import ScriptedPlanner  # noqa: E402
from agent.tasks import SCRIPTS_DIR, load_script  # noqa: E402
from agent.truth import derive, load_tasks, load_truth  # noqa: E402
from probes import phase2_gates as pg  # noqa: E402
from probes.replay import replay  # noqa: E402
from adapters.hand_help import HandHelp  # noqa: E402
from test_scorer_e2e import FAKE_PHOTO, FAKE_TRUTH  # noqa: E402

WALL = (1.45, -1.5)
CLEAN = {"git_commit": "c0ffee", "git_dirty": False}


@pytest.fixture(autouse=True)
def clean_git(monkeypatch: pytest.MonkeyPatch) -> dict:
    """run.json git state as on a counted run (the real tree is shared and often dirty); tests change it in place."""
    state = dict(CLEAN)
    monkeypatch.setattr(rt, "git_state", lambda repo: dict(state))
    monkeypatch.setattr(rt, "world_tag", lambda: "phase1-world-frozen-v2")  # world_tag() itself: test_world_tag.py
    return state


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    t, p = tmp_path / "truth.json", tmp_path / "places.yaml"
    m = tmp_path / "map_knowledge.json"
    t.write_text(json.dumps(FAKE_TRUTH))
    m.write_text(json.dumps({"world": "fake", "layout_sha256": FAKE_TRUTH["layout_sha256"], "rooms": FAKE_TRUTH["rooms"]}))
    p.write_text(yaml.safe_dump({"places": {n: {"x": x, "y": y, "yaw": a} for n, (x, y, a) in PLACES.items()}}))
    return {"truth_path": t, "places_path": p}


def rk(world: dict) -> dict:
    """run_one kwargs of the fake world: + its map_knowledge.json and a hand-help speed bound for the fake adapter,
    which drives any go_to (<= 10 m) in 10 s sim, faster than a TB4."""
    return {**world, "map_path": world["truth_path"].parent / "map_knowledge.json",
            "hand_help": lambda: HandHelp(v_max_mps=1.0)}


def fake_task(tid: str):  # noqa: ANN201
    real = {t.id: t for t in load_tasks()}[tid]
    answer, labels = derive(real.kind, real.target, FAKE_TRUTH)
    return type(real)(real.id, real.kind, real.prompt, real.target, answer, labels)


def fake_steps(steps: list) -> list:
    """Real script -> the fake world's photo numbers. Chair counts: the fake reception has 2 chairs (the clinic 4),
    one of them dead ahead of the place (photo 1)."""
    out = []
    for s in steps:
        if isinstance(s, dict) and s["name"] == "report":
            a = {**s["args"], "photo": FAKE_PHOTO.get(s["args"]["photo"], s["args"]["photo"])}
            if a["answer"] == "4":
                a["answer"], a["photo"] = "2", 1
            s = {**s, "args": a}
        out.append(s)
    return out


def run(world: dict, d: Path, tid: str, steps: list, sid: str = "t", ad: FakeAdapter | None = None,
        **kw) -> dict:  # noqa: ANN003
    ad = ad or FakeAdapter("ward", blocked=(WALL,))
    return run_one(ad, ad.pose, fake_task(tid), ScriptedPlanner(steps, sid), d, robot="fake", **rk(world), **kw)


def run_script(world: dict, d: Path, sid: str) -> dict:
    tid, steps = load_script(SCRIPTS_DIR / f"{sid}.yaml")
    return run(world, d, tid, fake_steps(steps), sid)


def edit_step(d: Path, i: int, **kv) -> None:  # noqa: ANN003
    lines = (d / "steps.jsonl").read_text().splitlines()
    rec = json.loads(lines[i])
    for k, v in kv.items():
        if "." in k:
            a, b = k.split(".")
            rec[a] = {**(rec[a] or {}), b: v}
        else:
            rec[k] = v
    lines[i] = json.dumps(rec)
    (d / "steps.jsonl").write_text("\n".join(lines) + "\n")


def edit_json(path: Path, **kv) -> None:  # noqa: ANN003
    d = json.loads(path.read_text())
    d.update(kv)
    path.write_text(json.dumps(d))


# ------------------------------------------------------------------------------------------------------ run_one
def test_run_one_setup_meta_and_score(tmp_path: Path, world: dict) -> None:
    r = run_script(world, tmp_path / "r", "find-toolbox")
    run_ = json.loads((tmp_path / "r/run.json").read_text())
    assert r["score"]["pass"] and run_["setup"]["start_place"] == "hall" and run_["setup"]["reached"] is True
    assert run_["setup"]["gt_err_m"] == pytest.approx(0.0) and run_["world_tag"] == "phase1-world-frozen-v2"
    assert run_["layout_sha256"] == FAKE_TRUTH["layout_sha256"] and run_["start_place"] == "hall"
    assert (r["git_commit"], r["git_dirty"]) == (run_["git_commit"], run_["git_dirty"]) == ("c0ffee", False)
    assert (tmp_path / "r/score.json").is_file()
    st = pg.steps_of(tmp_path / "r")
    assert st[0]["truth"]["place_err_m"] == pytest.approx(0.0) and st[1]["truth"]["place_err_m"] is None
    with pytest.raises(AssertionError):  # one run per dir
        run_script(world, tmp_path / "r", "find-toolbox")


def test_parse_place() -> None:
    assert parse_place("w:1.45,-1.5,0") == ("w", (1.45, -1.5, 0.0))


def test_run_one_setup_not_reached(tmp_path: Path, world: dict) -> None:
    ad = FakeAdapter("ward", blocked=(PLACES["hall"][:2],))
    r = run_one(ad, ad.pose, fake_task("find-toolbox"), ScriptedPlanner([], "x"), tmp_path / "r", robot="fake", **rk(world))
    assert r["setup"]["reached"] is False and r["setup"]["gt_err_m"] > 1.0
    ad = FakeAdapter("hall", blocked=(PLACES["hall"][:2],))  # Nav2 blocked while already standing on the place
    r = run_one(ad, ad.pose, fake_task("find-toolbox"), ScriptedPlanner([], "x"), tmp_path / "r2", robot="fake", **rk(world))
    assert r["setup"]["reached"] is False and r["setup"]["gt_err_m"] < 0.01


def test_run_one_setup_needs_gt_within_bound(tmp_path: Path, world: dict) -> None:
    """N2: Nav2 reached but ground truth 0.35 m off the hall place -> setup not reached; 0.25 m off -> reached."""
    for off, want in ((0.35, False), (0.25, True)):
        ad = FakeAdapter("ward")
        r = run_one(ad, lambda: (ad.x + off, ad.y, ad.yaw), fake_task("find-toolbox"), ScriptedPlanner([], "x"),
                    tmp_path / f"r{off}", robot="fake", **rk(world))
        assert r["setup"]["nav_status"] == "SUCCEEDED" and r["setup"]["reached"] is want, r["setup"]


# ------------------------------------------------------------------------------------------------------------ G1
def test_g1_plan_cycles_in_dev_yaml_order() -> None:
    tasks = load_tasks()
    plan = pg.g1_plan(tasks)
    assert len(plan) == pg.G1_TOTAL == 12
    for k in pg.KINDS:
        ids = [t.id for t in tasks if t.kind == k]
        assert [s for kk, s in plan if kk == k] == [ids[i % len(ids)] for i in range(3)]
        assert all((SCRIPTS_DIR / f"{s}.yaml").is_file() for kk, s in plan if kk == k)


def _g1_runs(n_pass: int, reached: bool = True) -> list[dict]:
    return [{"kind": k, "script": "s", "run_dir": "d", "outcome": "reported", "steps": 3, **CLEAN,
             "setup": {"reached": reached, "gt_err_m": 0.1}, "score": {"pass": i * 3 + j < n_pass}}
            for i, k in enumerate(pg.KINDS) for j in range(3)]


def test_g1_verdict() -> None:
    assert pg.g1_verdict(_g1_runs(12), True)["pass"] is True
    v = pg.g1_verdict(_g1_runs(11), True)
    assert v["pass"] is False and v["passed"] == 11 and v["per_kind"]["conditional"] == "2/3"
    assert pg.g1_verdict(_g1_runs(12), False)["pass"] is False      # --only subset never passes
    assert pg.g1_verdict(_g1_runs(12)[:11], True)["pass"] is False  # a missing run
    assert pg.g1_verdict(_g1_runs(12, reached=False), True)["pass"] is False  # not from the hall


def test_g1_extra_run_fails() -> None:
    assert pg.g1_verdict(_g1_runs(12) + _g1_runs(0)[:1], True)["pass"] is False  # 12 passes, 13 runs


def test_g1_git_pinned() -> None:
    """B4: one dirty run, or two commits, fails G1 even with 12/12 passes."""
    runs = _g1_runs(12)
    runs[5] = {**runs[5], "git_dirty": True}
    v = pg.g1_verdict(runs, True)
    assert v["pass"] is False and v["git_problems"] == ["1 run(s) on a dirty tree"]
    runs = _g1_runs(12)
    runs[0] = {**runs[0], "git_commit": "other"}
    assert pg.g1_verdict(runs, True)["pass"] is False


# ------------------------------------------------------------------------------------------------------------ G2
CONTROLS = sorted(p.name.removesuffix(".yaml") for p in SCRIPTS_DIR.glob("*.neg-*.yaml"))


@pytest.fixture
def g2_runs(tmp_path: Path, world: dict) -> dict[str, Path]:
    out = {}
    for sid in CONTROLS:
        out[sid] = tmp_path / "g2" / sid
        run_script(world, out[sid], sid)
    return out


def test_g2_all_controls_fail_for_their_reason(g2_runs: dict) -> None:
    v = pg.g2_verdict(g2_runs, True)
    assert v["pass"] is True, v["controls"]
    assert v["missing_rows"] == [] and {pg.control_name(s) for s in g2_runs} >= pg.G2_REQUIRED


def test_g2_control_failing_the_wrong_check(g2_runs: dict) -> None:
    by = {pg.control_name(s): d for s, d in g2_runs.items()}
    assert pg.g2_problems("no-proof", by["wrong-answer"])        # fails answer, not proof
    assert pg.g2_problems("wrong-answer", by["no-proof"])
    assert pg.g2_problems("13-steps", by["wrong-answer"])
    assert pg.g2_problems("bad-format", by["unknown-call"])       # other error texts
    assert pg.g2_problems("unknown-call", by["no-proof"])         # no error steps, does not pass
    v = pg.g2_verdict({"x.neg-no-proof": by["wrong-answer"], **{s: d for s, d in g2_runs.items()}}, True)
    assert v["pass"] is False and not v["controls"]["x.neg-no-proof"]["ok"]


def test_g2_unmapped_and_missing(g2_runs: dict) -> None:
    by = {pg.control_name(s): d for s, d in g2_runs.items()}
    assert pg.g2_problems("brand-new", by["unknown-call"]) == ["control 'brand-new' has no G2 mapping"]
    some = {s: d for s, d in g2_runs.items() if "13-steps" not in s}
    v = pg.g2_verdict(some, True)
    assert v["pass"] is False and v["missing_rows"] == ["13-steps"]
    assert pg.g2_verdict(g2_runs, False)["pass"] is False


def test_g2_13_steps_needs_step_limit(tmp_path: Path, world: dict) -> None:
    """A run that reports early (wrong answer) fails, but not on steps: the 13-steps control is not met."""
    d = tmp_path / "early"
    run(world, d, "count-chairs", [{"name": "navigate_to", "args": {"place": "reception"}},
                                   {"name": "report", "args": {"answer": "9", "photo": 1, "note": ""}}])
    assert pg.g2_problems("13-steps", d)


def test_g2_error_not_returned_is_caught(g2_runs: dict) -> None:
    d = g2_runs[next(s for s in g2_runs if "unknown-call" in s)]
    edit_step(d, 1, **{"packet.last_result": "start"})
    assert any("not returned" in p for p in pg.g2_problems("unknown-call", d))


def test_g2_fails_checks_label_and_outcome(tmp_path: Path, world: dict, g2_runs: dict) -> None:
    by = {pg.control_name(s): d for s, d in g2_runs.items()}
    d = tmp_path / "two"  # wrong answer AND no proof: not "exactly answer"
    run(world, d, "find-toolbox", [{"name": "navigate_to", "args": {"place": "storage"}}, {"name": "look_around"},
                                   {"name": "report", "args": {"answer": "office", "photo": 2, "note": ""}}])
    assert pg.g2_problems("wrong-answer", d)
    edit_json(by["wrong-answer"] / "score.json", label="nav")
    assert pg.g2_problems("wrong-answer", by["wrong-answer"]) == ["label nav != vlm"]
    edit_json(by["13-steps"] / "run.json", outcome="reported")
    assert pg.g2_problems("13-steps", by["13-steps"])


BAD = {"name": "report", "args": {"answer": "four", "photo": 1, "note": ""}}
NAV_LOOK = [{"name": "navigate_to", "args": {"place": "reception"}}, {"name": "look_around"}]


def test_g2_error_texts_count_and_final_report(tmp_path: Path, world: dict, g2_runs: dict) -> None:
    by = {pg.control_name(s): d for s, d in g2_runs.items()}
    assert pg.g2_problems("raw-invalid", by["unknown-call"])  # same count (2), other texts
    good = {"name": "report", "args": {"answer": "2", "photo": 1, "note": ""}}
    run(world, tmp_path / "twice", "count-chairs", NAV_LOOK + [BAD, BAD, good])
    assert pg.g2_problems("bad-format", tmp_path / "twice")  # right text, one error too many
    run(world, tmp_path / "wrong", "count-chairs", NAV_LOOK + [BAD, {**good, "args": {**good["args"], "answer": "3"}}])
    assert pg.g2_problems("bad-format", tmp_path / "wrong")  # error ok, but its final report is wrong
    run(world, tmp_path / "ok", "count-chairs", NAV_LOOK + [BAD, good])
    assert pg.g2_problems("bad-format", tmp_path / "ok") == []


def test_g2_git_pinned(g2_runs: dict) -> None:
    """B4: a dirty control run fails G2."""
    edit_json(next(iter(g2_runs.values())) / "run.json", git_dirty=True)
    v = pg.g2_verdict(g2_runs, True)
    assert v["pass"] is False and v["git_problems"]


# ------------------------------------------------------------------------------------------------------------ G3
def test_g3_wall_point_on_clinic_truth() -> None:
    t = load_truth()
    p = pg.g3_wall_point(t)
    assert p == pytest.approx(WALL)
    assert min(pg._seg_dist(p, *d) for d in pg.doorway_segments(t)) > 0.3


def _g3(world: dict, d: Path, steps: list, ad: FakeAdapter | None = None) -> dict:
    return run(world, d, "find-toolbox", steps, pg.G3_ID, ad=ad, extra_places={pg.G3_PLACE: (*WALL, 0.0)})


def test_g3_storage_wall_hall(tmp_path: Path, world: dict) -> None:
    d = tmp_path / "g3"
    _g3(world, d, pg.g3_script())
    v = pg.g3_verdict(d, WALL)
    assert v["pass"] is True, v["problems"]
    n = v["numbers"]
    assert n["blocked_nav_status"] == "ABORTED" and n["blocked_gt_err_m"] > 2.0 and n["hall_gt_move_m"] > 3.0
    assert n["hall_gt_err_m"] == pytest.approx(0.0) and n["blocked_nav_sim_s"] == 10.0
    assert "g3_wall" not in yaml.safe_load(world["places_path"].read_text())["places"]


def test_g3_wrong_script_shape(tmp_path: Path, world: dict) -> None:
    """The old (hall start, wall, hall) script: the hall step is a zero-length move -> refused by shape."""
    _g3(world, tmp_path / "a", pg.g3_script()[1:])
    assert "steps 1-3" in pg.g3_problems(tmp_path / "a")[0]


def test_g3_hall_step_must_move(tmp_path: Path, world: dict) -> None:
    """B1: a robot that never leaves the hall (storage step blocked too) reaches the hall by not moving -> FAIL."""
    ad = FakeAdapter("ward", blocked=(WALL, PLACES["storage"][:2]))
    d = tmp_path / "still"
    _g3(world, d, pg.g3_script(), ad=ad)
    p = pg.g3_problems(d)
    assert p == [f"hall step: GT moved 0.000 m, want >= {pg.G3_MIN_MOVE_M} m (a real move)"], p


def test_g3_blocked_but_gt_at_wall_point(tmp_path: Path, world: dict) -> None:
    """B1: 'failed: blocked' while ground truth is within 0.30 m of the wall point -> FAIL."""
    d = tmp_path / "g3"
    _g3(world, d, pg.g3_script())
    edit_step(d, 1, **{"truth.place_err_m": 0.29})
    assert any("GT 0.290 m from the wall point" in p for p in pg.g3_problems(d))


def test_g3_hall_end_within_bound(tmp_path: Path, world: dict) -> None:
    d = tmp_path / "g3"
    _g3(world, d, pg.g3_script())
    edit_step(d, 2, **{"truth.place_err_m": 0.31})
    assert any("from the hall place" in p for p in pg.g3_problems(d))
    edit_step(d, 2, **{"truth.place_err_m": 0.1, "result": "failed: blocked"})
    assert any("robot stuck" in p for p in pg.g3_problems(d))


def test_g3_time_is_nav_only(tmp_path: Path, world: dict) -> None:
    """B2: the limit is on skill_detail.nav_sim_s (+ 1 s hang guard), not on the step with its photo."""
    d = tmp_path / "g3"
    _g3(world, d, pg.g3_script())
    edit_step(d, 1, skill_sim_s=125.0, **{"skill_detail.nav_sim_s": 120.9})
    assert pg.g3_problems(d) == []  # a Nav2 timeout just past 120 s, then the photo: still a valid block
    edit_step(d, 1, **{"skill_detail.nav_sim_s": 121.1})
    assert any("nav_sim_s" in p for p in pg.g3_problems(d))
    edit_step(d, 1, skill_sim_s=5.0, **{"skill_detail.nav_sim_s": 10.0, "result": "reached"})
    assert any("want 'failed: blocked'" in p for p in pg.g3_problems(d))


def test_g3_git_pinned(tmp_path: Path, world: dict, clean_git: dict) -> None:
    clean_git["git_dirty"] = True
    _g3(world, tmp_path / "g3", pg.g3_script())
    v = pg.g3_verdict(tmp_path / "g3", WALL)
    assert v["pass"] is False and v["problems"] == ["1 run(s) on a dirty tree"]


def test_counted_run_refuses_dirty_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """B4: no --rehearsal on a dirty tree -> exit before any sim or file."""
    monkeypatch.setattr(pg, "git_state", lambda repo: {"git_commit": "x", "git_dirty": True})
    monkeypatch.setattr(sys, "argv", ["phase2_gates.py", "g4", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="dirty tree"):
        pg.main()
    assert not (tmp_path / "o").exists()


# ------------------------------------------------------------------------------------------------------ G4 / measure
def _three(world: dict, out: Path) -> list[Path]:
    run_script(world, out / "g1" / "00-find-toolbox", "find-toolbox")
    run_script(world, out / "g2" / "find-toolbox.neg-no-proof", "find-toolbox.neg-no-proof")
    _g3(world, out / "g3" / pg.G3_ID, pg.g3_script())
    return pg.run_dirs(out)


def test_g4_and_measure(tmp_path: Path, world: dict) -> None:
    out = tmp_path / "out"
    dirs = _three(world, out)
    assert len(dirs) == 3
    v = pg.g4_verdict(dirs, 3, **world)
    assert v["pass"] is True and all(r["pngs"] > 0 for r in v["runs"].values()), v
    m = pg.measure(dirs, out)
    assert (out / "measure.json").is_file() and m["hall_setup_gt_err_m"]["n"] == 3
    assert set(m["per_skill"]) == {"navigate_to", "look_around", "report"}
    assert pg.g4_verdict([], 0, **world)["pass"] is True and pg.g4_verdict(dirs, None, **world)["pass"] is False
    # tampered verdict -> replay differs
    sc = dirs[0] / "score.json"
    s = json.loads(sc.read_text())
    s["pass"] = not s["pass"]
    sc.write_text(json.dumps(s))
    assert pg.g4_verdict(dirs, 3, **world)["pass"] is False
    s["pass"] = not s["pass"]
    sc.write_text(json.dumps(s))
    # a missing PNG
    next((dirs[1] / "photos").glob("*_rgb.png")).unlink()
    v = pg.g4_verdict(dirs, 3, **world)
    assert v["pass"] is False and v["runs"][str(dirs[1])]["missing_pngs"]


def test_g4_run_count(tmp_path: Path, world: dict) -> None:
    """N4: the full run count is 12 + controls + 1; one run more or less fails G4."""
    dirs = _three(world, tmp_path / "out")
    assert pg.g4_verdict(dirs, 3, **world)["pass"] is True
    assert pg.g4_verdict(dirs, 4, **world)["pass"] is False
    assert pg.g4_verdict(dirs[:2], 3, **world)["pass"] is False


def test_g4_incomplete_record(tmp_path: Path, world: dict) -> None:
    d = tmp_path / "g1" / "x"
    run_script(world, d, "find-toolbox")
    run_ = json.loads((d / "run.json").read_text())
    del run_["setup"]["gt_err_m"]
    (d / "run.json").write_text(json.dumps(run_))
    assert pg.g4_verdict([d], 1, **world)["pass"] is False


# ------------------------------------------------------------------------------------------------- N3 file hashes
def test_replay_rehashes_files(tmp_path: Path, world: dict) -> None:
    """N3: score.json holds sha256 of run.json, steps.jsonl and the proof seg; replay fails if any changed."""
    d = tmp_path / "r"
    run_script(world, d, "find-toolbox")
    saved = json.loads((d / "score.json").read_text())["sha256"]
    seg = [f for f in saved if f.endswith("_seg.png")]
    assert set(saved) - set(seg) == {"run.json", "steps.jsonl"} and len(seg) == 1
    assert replay(d, **world) == (True, [])
    for f in ("steps.jsonl", seg[0]):
        orig = (d / f).read_bytes()
        (d / f).write_bytes(orig + b"\n" if f.endswith(".jsonl") else orig + b"\0")  # same verdict, other bytes
        same, diff = replay(d, **world)
        assert not same and len(diff) == 1 and diff[0].startswith(f"sha256 {f}: saved {saved[f]}"), diff
        (d / f).write_bytes(orig)
    assert replay(d, **world) == (True, [])
