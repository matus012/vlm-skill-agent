"""probes/phase3_gates.py (task 005 G1-G3 + Measure) and the run lists, no sim, no VLM: real run dirs from
adapters.run_task.run_one on the FakeAdapter with a VLM-shaped fake planner (reasoning, finish_reason, usage), scored by
agent.scorer against a synthetic truth matching the fake world, then edited to make each gate fail."""
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
import adapters.run_task as rt  # noqa: E402
from adapters.hand_help import HandHelp  # noqa: E402
from adapters.run_task import run_one  # noqa: E402
from agent.context import system_prompt, system_prompt_sha256  # noqa: E402
from agent.fake_adapter import PLACES, FakeAdapter  # noqa: E402
from agent.planners import ScriptedPlanner  # noqa: E402
from agent.runtime import MALFORMED_LIMIT  # noqa: E402
from agent.skills import load_places  # noqa: E402
from agent.tasks import load_tasks, task_by_id  # noqa: E402
from probes import phase3_gates as g  # noqa: E402
from probes import phase3_run as pr  # noqa: E402
from probes.phase3_run import COUNTED, load_runs  # noqa: E402

LAYOUT = "f" * 64
ROOMS = ["hall", "ward", "reception", "office", "storage"]
TRUTH = {"layout_sha256": LAYOUT, "rooms": {r: {} for r in ROOMS}, "objects": [
    {"id": "toolbox_red_1", "class": "toolbox", "colour": "red", "label": 10, "room": "storage", "state": None},
    {"id": "cabinet_blue_1", "class": "cabinet", "colour": "blue", "label": 30, "room": "storage", "state": "closed"},
    {"id": "chair_orange_1", "class": "chair", "colour": "orange", "label": 20, "room": "reception", "state": None},
    {"id": "chair_orange_2", "class": "chair", "colour": "orange", "label": 21, "room": "reception", "state": None},
    {"id": "bed_blue_1", "class": "bed", "colour": "blue", "label": 1, "room": "ward", "state": None}]}
# fake world: (room, look_around photo that shows the target) per Find prompt of the counted set
FAKE = {"find-toolbox": ("storage", 1), "find-cabinet": ("storage", 4), "find-beds-where": ("ward", 1),
        "find-chairs-plain": ("reception", 1)}
INFO = {"model": "fake-vlm", "thinking": True, "sampling": {"temperature": 1.0}, "max_tokens": 8192}


class FakeVlm:
    """A ScriptedPlanner dressed as the VLM: reasoning, finish_reason, usage."""

    def __init__(self, steps: list, seed: int) -> None:
        self.sp, self.name = ScriptedPlanner(steps, "fake"), "vlm:fake-vlm"
        self.info = {**INFO, "seed": seed}

    def next_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        return {**self.sp.next_call(messages, tools), "reasoning": "I should look first.", "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 900, "completion_tokens": 40}}


def correct(tid: str) -> list[dict[str, Any]]:
    room, photo = FAKE[tid]
    return [{"name": "navigate_to", "args": {"place": room}}, {"name": "look_around", "args": {}},
            {"name": "report", "args": {"answer": room, "photo": photo, "note": "seen"}}]


def wrong(tid: str) -> list[dict[str, Any]]:
    s = correct(tid)
    s[-1] = {**s[-1], "args": {**s[-1]["args"], "answer": "office"}}
    return s


def make_run(world: dict[str, Path], d: Path, entry: dict[str, Any], steps: list,
             meta: dict[str, Any] | None = None) -> Path:
    """One listed run through the real harness (adapters/run_task.run_one, start place + VLM malformed limit) on the
    fake adapter; git state as on a clean counted tree."""
    ad = FakeAdapter("hall")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rt, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
        run_one(ad, ad.pose, task_by_id(entry["task_id"]), FakeVlm(steps, entry["seed"]), d, robot="fake", tag="fake",
                map_path=world["truth_path"].parent / "map_knowledge.json", hand_help=lambda: HandHelp(v_max_mps=1.0),
                start_place=entry["start_place"], malformed_limit=MALFORMED_LIMIT, extra_meta=meta, **world)
    return d


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The counted list run once on the fake world through phase3_run.plan + run_list (invocation.json, run.json
    invocation_id, tmp ledger): c01-c07 correct, c08-c10 wrong answer (7/10)."""
    root = tmp_path_factory.mktemp("p3")
    world = {"truth_path": root / "truth.json", "places_path": root / "places.yaml"}
    world["truth_path"].write_text(json.dumps(TRUTH))
    world["places_path"].write_text(yaml.safe_dump({"places": {n: {"x": x, "y": y, "yaw": a}
                                                               for n, (x, y, a) in PLACES.items()}}))
    (root / "map_knowledge.json").write_text(json.dumps({"world": "fake", "layout_sha256": LAYOUT,
                                                         "rooms": {r: {} for r in ROOMS}}))
    out, ledger = root / "out", root / "ledger.jsonl"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pr, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
        entries, _, inv, kind = pr.plan(pr.parse(["--out", str(out)]))
    script = {e["run_id"]: correct(e["task_id"]) if i < 7 else wrong(e["task_id"]) for i, e in enumerate(entries)}
    pr.run_list(entries, out, inv, kind, lambda e, d, meta: make_run(world, d, e, script[e["run_id"]], meta),
                lambda: None, ledger)
    return {"out": out, "entries": entries, "world": world, "ledger": ledger, "inv": inv["invocation_id"]}


@pytest.fixture
def case(base: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    out, ledger = tmp_path / "out", tmp_path / "ledger.jsonl"
    shutil.copytree(base["out"], out)
    shutil.copy(base["ledger"], ledger)
    return {**base, "out": out, "ledger": ledger, "entries": copy.deepcopy(base["entries"])}


def g1(c: dict[str, Any], counted: bool = True) -> dict[str, Any]:
    return g.g1_verdict(c["out"], c["entries"], counted, c["ledger"], **c["world"])


def edit_run(d: Path, **kv: Any) -> None:
    run = json.loads((d / "run.json").read_text())
    run.update(kv)
    (d / "run.json").write_text(json.dumps(run))


def edit_steps(d: Path, fn: Any) -> None:
    recs = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines()]
    for r in recs:
        fn(r)
    (d / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")


# ------------------------------------------------------------------------------------------------------ run lists
def test_counted_set_shape() -> None:
    runs = load_runs(COUNTED)
    tasks = {t.id: t for t in load_tasks()}
    assert len(runs) == g.G1_RUNS == 10
    prompts = {tasks[r["task_id"]].prompt for r in runs}
    assert prompts == {"Find the red toolbox.", "Find the blue cabinet.", "Where are the beds?", "Find the chairs."}
    assert all(sum(tasks[r["task_id"]].prompt == p for r in runs) >= 2 for p in prompts)
    assert all(tasks[r["task_id"]].kind == "find" for r in runs)
    truth = json.loads((ROOT / "sim/worlds/clinic/generated/truth.json").read_text())
    room = {"find-toolbox": "storage", "find-cabinet": "storage", "find-beds-where": "ward",
            "find-chairs-plain": "reception"}
    for tid, r in room.items():  # the target rooms named in the task file, checked against truth.json by hand
        t = tasks[tid].target
        objs = [o for o in truth["objects"] if o["id"] == t.get("object") or o["class"] == t.get("class")]
        assert {o["room"] for o in objs} == {r}, tid
    starts = [(r["start_place"], room[r["task_id"]]) for r in runs]
    assert any(s == "hall" for s, _ in starts)
    assert any(s == t for s, t in starts)                       # starts in the target room
    assert any(s not in ("hall", t) for s, t in starts)         # starts in another room
    assert {r["seed"] for r in runs}.isdisjoint({r["seed"] for r in load_runs(ROOT / "tasks/phase3_dev.yaml")})
    assert {r["run_id"] for r in runs}.isdisjoint({r["run_id"] for r in load_runs(ROOT / "tasks/phase3_dev.yaml")})


@pytest.mark.parametrize("bad", [
    {"run_id": "c01", "task_id": "find-toolbox", "start_place": "hall", "seed": 2},           # duplicate run id
    {"run_id": "x", "task_id": "find-nothing", "start_place": "hall", "seed": 2},             # unknown task
    {"run_id": "x", "task_id": "find-toolbox", "start_place": "kitchen", "seed": 2},          # unknown place
    {"run_id": "x", "task_id": "find-toolbox", "start_place": "hall", "seed": "2"},           # seed not an int
    {"run_id": "x", "task_id": "find-toolbox", "start_place": "hall", "seed": 39827717},      # duplicate seed
    {"run_id": "x", "task_id": "find-toolbox", "start_place": "hall", "seed": 2, "note": 1},  # extra key
])
def test_load_runs_rejects(tmp_path: Path, bad: dict) -> None:
    p = tmp_path / "l.yaml"
    p.write_text(yaml.safe_dump(load_runs(COUNTED) + [bad]))
    with pytest.raises(AssertionError):
        load_runs(p)


# ------------------------------------------------------------------------------------------------------------ G1
def test_g1_seven_of_ten_passes(case: dict) -> None:
    v = g1(case)
    assert v["pass"] and v["passed"] == 7 and v["total"] == 10, v
    assert [r["label"] for r in v["runs"]][7:] == ["vlm"] * 3


def test_g1_six_fails(case: dict) -> None:
    make_run(case["world"], case["out"] / "c07x", case["entries"][6], wrong("find-beds-where"))
    shutil.rmtree(case["out"] / "c07")
    (case["out"] / "c07x").rename(case["out"] / "c07")
    v = g1(case)
    assert v["passed"] == 6 and not v["pass"]


def test_g1_missing_run_counts_as_fail(case: dict) -> None:
    shutil.rmtree(case["out"] / "c01")
    v = g1(case)
    assert v["passed"] == 6 and not v["pass"] and v["runs"][0]["problems"] == ["run dir missing"]


def test_g1_not_counted_or_short_list_never_passes(case: dict) -> None:
    assert not g1(case, counted=False)["pass"]
    case["entries"] = case["entries"][:7]
    assert g1(case)["passed"] == 7 and not g1(case)["pass"]


@pytest.mark.parametrize("edit,why", [
    ({"git_dirty": True}, "dirty"), ({"git_commit": "beef"}, "git_commit"),
    ({"system_prompt_sha256": "0" * 64}, "system_prompt_sha256")])
def test_g1_git_and_prompt_pinned(case: dict, edit: dict, why: str) -> None:
    edit_run(case["out"] / "c10", **edit)
    v = g1(case)
    assert not v["pass"] and any(why in p for p in v["git_problems"]), v["git_problems"]


@pytest.mark.parametrize("edit", [{"start_place": "office"}, {"planner_info": {**INFO, "seed": 1}},
                                  {"task": {"id": "find-cabinet", "kind": "find", "prompt": "Find the blue cabinet.",
                                            "target": {"object": "cabinet_blue_1"}}}])
def test_g1_run_must_match_its_entry(case: dict, edit: dict) -> None:
    edit_run(case["out"] / "c01", **edit)
    v = g1(case)
    assert not v["runs"][0]["pass"] and "list says" in v["runs"][0]["problems"][0] and not v["pass"]


def test_g1_setup_not_reached_does_not_count(case: dict) -> None:
    edit_run(case["out"] / "c02", setup={"start_place": "office", "reached": False, "gt_err_m": 0.5})
    v = g1(case)
    assert not v["runs"][1]["pass"] and not v["pass"]


def test_g1_saved_score_must_agree_with_rescore(case: dict) -> None:
    sc = json.loads((case["out"] / "c08/score.json").read_text())
    (case["out"] / "c08/score.json").write_text(json.dumps({**sc, "pass": True, "checks": {k: True for k in sc["checks"]}}))
    v = g1(case)
    assert not v["runs"][7]["pass"] and v["passed"] == 7 and "re-score" in v["runs"][7]["problems"][0]
    (case["out"] / "c03/score.json").unlink()
    assert g1(case)["passed"] == 6


# ------------------------------------------------------------------------------------------------------------ G2
def test_g2_complete_set_passes(case: dict) -> None:
    v = g.g2_verdict(case["out"], case["entries"], True)
    assert v["pass"], {k: r for k, r in v["runs"].items() if r["problems"]}
    assert not g.g2_verdict(case["out"], case["entries"], False)["pass"]


def _drop(path: str) -> Any:
    def fn(r: dict) -> None:
        a, b = path.split(".") if "." in path else (path, None)
        if b is None:
            r.pop(a)
        else:
            r[a].pop(b)
    return fn


def _photo(key: str, val: Any) -> Any:
    def fn(r: dict) -> None:
        r["packet"]["photos"][0][key] = val
    return fn


@pytest.mark.parametrize("fn,needle", [
    (_drop("packet.texts"), "packet.texts"),
    (_drop("packet.system_sha256"), "system_sha256"),
    (_photo("rgb_sha256", "0" * 64), "rgb_sha256"),
    (_photo("sent_sha256", None), "sent_sha256"),
    (lambda r: r["planner_out"].pop("reasoning"), "planner_out lacks"),
    (lambda r: r["planner_out"].update(usage=None), "finish_reason / usage"),
    (lambda r: r["planner_out"].update(finish_reason=None), "finish_reason / usage"),
    (lambda r: r["planner_out"].pop("content"), "planner_out lacks"),
    (lambda r: r["call"].update(raw="navigate_to"), "tool calls as received"),
    (lambda r: r.update(result=None) if r["check"] == "ok" else None, "skill result"),
    (_drop("planner_wall_s"), "timings"),
    (lambda r: r.update(wall_t1=None), "timings"),
    (lambda r: r["packet"].update(photos=[]), "no photos"),
])
def test_g2_step_record_gaps(case: dict, fn: Any, needle: str) -> None:
    edit_steps(case["out"] / "c04", fn)
    v = g.g2_verdict(case["out"], case["entries"], True)
    assert not v["pass"] and any(needle in p for p in v["runs"]["c04"]["problems"]), v["runs"]["c04"]


def test_g2_planner_error_step_needs_no_usage(case: dict) -> None:
    edit_steps(case["out"] / "c04", lambda r: r["planner_out"].update(error="APITimeoutError", usage=None,
                                                                       finish_reason=None, reasoning=None))
    assert g.g2_verdict(case["out"], case["entries"], True)["pass"]


@pytest.mark.parametrize("edit,needle", [
    ({"system_prompt": "changed"}, "system_prompt"), ({"tools": []}, "tools"),
    ({"planner_info": {k: v for k, v in INFO.items() if k != "model"} | {"seed": 1}}, "planner_info"),
    ({"planner_info": INFO}, "planner_info")])
def test_g2_run_json_gaps(case: dict, edit: dict, needle: str) -> None:
    edit_run(case["out"] / "c05", **edit)
    v = g.g2_verdict(case["out"], case["entries"], True)
    assert not v["pass"] and any(needle in p for p in v["runs"]["c05"]["problems"])


@pytest.mark.parametrize("label,why", [(None, "x"), ("vlm", None), ("vlm", "two\nlines"), ("planner", "x"),
                                       ("vlm", " ")])
def test_g2_failed_run_one_label_one_line(case: dict, label: Any, why: Any) -> None:
    p = case["out"] / "c09/score.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "label": label, "label_why": why}))
    v = g.g2_verdict(case["out"], case["entries"], True)
    assert not v["pass"] and "exactly one label" in v["runs"]["c09"]["problems"][-1]


def test_g2_passing_run_has_no_label_and_missing_run_fails(case: dict) -> None:
    p = case["out"] / "c01/score.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "label": "vlm"}))
    assert "failure label" in g.g2_verdict(case["out"], case["entries"], True)["runs"]["c01"]["problems"][-1]
    shutil.rmtree(case["out"] / "c02")
    assert g.g2_verdict(case["out"], case["entries"], True)["runs"]["c02"]["problems"] == ["run dir missing"]


def test_g2_incomplete_log(case: dict) -> None:
    (case["out"] / "c06" / "photos" / "0_0_rgb.png").unlink()
    v = g.g2_verdict(case["out"], case["entries"], True)
    assert not v["pass"] and any(p.startswith("check_complete") for p in v["runs"]["c06"]["problems"])


# ------------------------------------------------------------------------------------------------------------ G3
OK_NO_ROS = {"returncode": 0, "summary": "2 passed"}


def g3(c: dict, no_ros: dict = OK_NO_ROS, counted: bool = True) -> dict:
    return g.g3_verdict(c["out"], c["entries"], counted, no_ros)


def test_g3_clean_set_passes(case: dict) -> None:
    v = g3(case)
    assert v["pass"] and not v["hits"], v["hits"]
    assert not g3(case, {"returncode": 1, "summary": "1 failed"})["pass"]
    assert not g3(case, counted=False)["pass"]


@pytest.mark.parametrize("text", ["at 2.5, -3.9", "pose (2, -3)", "x=-2.50", "[1, 2]", "a TurtleBot 4", "create3 base",
                                  "oakd camera", "the G1 robot", "Unitree", "lidar scan", "the map", "map coordinates",
                                  "odom frame", "yaw 0"])
def test_g3_catches_leaks_in_step_texts(case: dict, text: str) -> None:
    edit_steps(case["out"] / "c03", lambda r: r["packet"]["texts"].__setitem__(0, r["packet"]["texts"][0] + " " + text)
               if r["step"] == 2 else None)
    v = g3(case)
    assert not v["pass"] and v["failing_hits"]["c03"][0]["where"] == "step 2 text 0", v["failing_hits"]


@pytest.mark.parametrize("where", ["history", "last_result"])
def test_g3_scans_history_and_last_result(case: dict, where: str) -> None:
    def fn(r: dict) -> None:
        if r["step"] == 3:
            if where == "history":
                r["packet"]["history"][0] += " at 1.25"
            else:
                r["packet"]["last_result"] += " (1, 2)"
    edit_steps(case["out"] / "c03", fn)
    assert not g3(case)["pass"]


@pytest.mark.parametrize("key,val", [("system_prompt", "You drive a turtlebot."),
                                     ("tools", [{"function": {"description": "go to x 1.5"}}])])
def test_g3_scans_prompt_and_tools(case: dict, key: str, val: Any) -> None:
    edit_run(case["out"] / "c05", **{key: val})
    v = g3(case)
    assert not v["pass"] and v["failing_hits"]["c05"][0]["where"] == key


def test_g3_echo_of_vlm_output_is_listed_not_failed(case: dict) -> None:
    """The VLM wrote "1.5" in a bad call; the error / history line repeating it is no leak."""
    def fn(r: dict) -> None:
        if r["step"] == 1:
            r["call"]["raw"] = [{"function": {"name": "report", "arguments": '{"photo": 1.5}'}}]
        if r["step"] == 2:
            r["packet"]["history"][0] += " report({\"photo\": 1.5})"
    edit_steps(case["out"] / "c03", fn)
    v = g3(case)
    assert v["pass"] and [h["hit"] for h in v["hits"]["c03"]] == ["1.5"] and v["hits"]["c03"][0]["echo"], v["hits"]
    edit_steps(case["out"] / "c03", lambda r: r["packet"]["history"].__setitem__(0, r["packet"]["history"][0] + " 2.25")
               if r["step"] == 3 else None)
    assert not g3(case)["pass"]


def test_g3_missing_run_fails(case: dict) -> None:
    shutil.rmtree(case["out"] / "c10")
    assert g3(case)["missing_runs"] == ["c10"] and not g3(case)["pass"]


def test_no_ros_test_runs() -> None:
    assert g.no_ros_test()["returncode"] == 0


# ------------------------------------------------------------------------------------------------------- Measure
def test_measure(case: dict) -> None:
    m = g.measure(case["out"], case["entries"])
    assert m["steps_per_run"]["n"] == 10 and m["steps_per_run"]["mean"] == 3
    assert m["vlm_latency_per_step_s"]["n"] == 30
    r = m["runs"][0]
    assert r["h200_h"] == pytest.approx(r["vlm_s"] / 3600) and (case["out"] / "measure.json").is_file()


def test_prompt_sha_recorded_matches_context(case: dict) -> None:
    run = json.loads((case["out"] / "c01/run.json").read_text())
    places = list(load_places(case["world"]["places_path"]))
    assert run["system_prompt"] == system_prompt(places) and run["system_prompt_sha256"] == system_prompt_sha256(places)
    assert run["planner_info"]["seed"] == case["entries"][0]["seed"] and run["malformed_limit"] == MALFORMED_LIMIT


def test_malformed_run_is_a_labelled_failure(case: dict) -> None:
    """Two malformed outputs in a row (fail_malformed) is a complete, labelled vlm failure for G1 / G2."""
    from agent.planners import tool_call
    shutil.rmtree(case["out"] / "c10")
    make_run(case["world"], case["out"] / "c10", case["entries"][9],
             [[tool_call("fly_to", {})], {"name": "navigate_to", "args": {"place": "kitchen"}}],
             {"invocation_id": case["inv"]})
    run = json.loads((case["out"] / "c10/run.json").read_text())
    assert run["outcome"] == "fail_malformed" and run["steps"] == 2
    assert g.g2_verdict(case["out"], case["entries"], True)["pass"]
    v = g1(case)
    assert v["pass"] and v["runs"][9]["label"] == "vlm" and "malformed" in v["runs"][9]["label_why"]


# ------------------------------------------------------------------------------------------- G1 one invocation (B1)
def ledger_add(c: dict, event: str, iid: str | None = None, sha: str = pr.COUNTED_SHA256) -> None:
    with c["ledger"].open("a") as f:
        f.write(json.dumps({"utc": "x", "event": event, "invocation_id": iid or c["inv"], "out": "o",
                            "list_sha256": sha, "git_commit": "c0ffee"}) + "\n")


def test_g1_invocation_written_by_the_runner(case: dict) -> None:
    inv = json.loads((case["out"] / "invocation.json").read_text())
    assert inv["list_sha256"] == pr.COUNTED_SHA256 and inv["git_commit"] == "c0ffee" and inv["counted"]
    assert inv["segments"][0]["entries_run"] == [f"c{i:02d}" for i in range(1, 11)] and inv["segments"][0]["end"] == "end"
    assert all(json.loads((case["out"] / e["run_id"] / "run.json").read_text())["invocation_id"] == case["inv"]
               for e in case["entries"])
    assert [json.loads(x)["event"] for x in case["ledger"].read_text().splitlines()] == ["start", "end"]
    assert g1(case)["invocation_problems"] == []


def test_g1_run_of_another_invocation_fails(case: dict) -> None:
    edit_run(case["out"] / "c04", invocation_id="other")
    v = g1(case)
    assert not v["pass"] and "c04" in v["invocation_problems"][0], v["invocation_problems"]


def test_g1_invocation_json_missing_or_other_list(case: dict) -> None:
    p = case["out"] / "invocation.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "list_sha256": "0" * 64}))
    assert not g1(case)["pass"] and "list_sha256" in g1(case)["invocation_problems"][0]
    p.unlink()
    assert not g1(case)["pass"] and g1(case)["invocation_problems"] == ["invocation.json missing"]


@pytest.mark.parametrize("extra,ok", [
    ([], True),
    ([("abort", None), ("resume", None), ("end", None)], False),        # a second end
    ([("start", "inv-2"), ("end", "inv-2")], False),                    # a second counted invocation
    ([("start", "inv-2")], False),                                      # a second one, even unfinished
    ([("start", "inv-dev", "f" * 64)], True),                           # another list's line is not this one
])
def test_g1_ledger_one_invocation(case: dict, extra: list, ok: bool) -> None:
    for e in extra:
        ledger_add(case, *e)
    v = g1(case)
    assert v["pass"] is ok, v["invocation_problems"]


@pytest.mark.parametrize("events,ok", [
    (["start", "abort", "resume", "end"], True), (["start", "resume", "resume", "end"], True),
    (["start"], False), (["start", "abort"], False), (["resume", "end"], False), (["start", "start", "end"], False),
    (["start", "end", "end"], False), (["end"], False), ([], False)])
def test_g1_ledger_event_order(case: dict, events: list, ok: bool) -> None:
    case["ledger"].write_text("")
    for e in events:
        ledger_add(case, e)
    assert g1(case)["pass"] is ok, g1(case)["invocation_problems"]


def test_g1_ledger_missing_fails_and_dev_list_skips_invocation(case: dict) -> None:
    case["ledger"].unlink()
    assert not g1(case)["pass"] and "ledger" in g1(case)["invocation_problems"][0]
    (case["out"] / "invocation.json").unlink()
    v = g1(case, counted=False)
    assert v["invocation_problems"] == [] and v["verdict_scope"].startswith("not the counted list")


# ------------------------------------------------------------------------------------------------ N6 / N4 / B2 / N5
def test_g1_setup_miss_is_labelled_nav(case: dict) -> None:
    edit_run(case["out"] / "c01", setup={"start_place": "hall", "reached": False, "gt_err_m": 0.734})
    v = g1(case)
    r = v["runs"][0]
    assert not r["pass"] and r["label"] == "nav" and r["label_why"] == "setup did not reach hall (GT 0.73 m)"
    assert v["setup_misses"] == ["c01"] and not v["pass"]


def test_g2_empty_reasoning_passes_and_is_listed(case: dict) -> None:
    edit_steps(case["out"] / "c04", lambda r: r["planner_out"].update(reasoning=None) if r["step"] == 2 else None)
    edit_steps(case["out"] / "c05", lambda r: r["planner_out"].update(reasoning="") if r["step"] == 1 else None)
    v = g.g2_verdict(case["out"], case["entries"], True)
    assert v["pass"] and v["empty_reasoning_steps"] == {"c04": [2], "c05": [1]}, v["empty_reasoning_steps"]


def test_measure_vlm_output_and_job_hours(case: dict) -> None:
    def fn(r: dict) -> None:
        if r["step"] == 1:
            r["planner_out"].update(finish_reason="length", usage={"completion_tokens": 8192})
        if r["step"] == 2:
            r["call"]["raw"] = []
            r["planner_out"].update(reasoning='ok <tool_call>{"name": "look_around"}')
    edit_steps(case["out"] / "c02", fn)
    m = g.measure(case["out"], case["entries"], job_gpu_h=2.5)
    vo = m["vlm_output"]
    assert vo["length_finishes"] == 1 and vo["length_steps"] == ["c02 step 1"]
    assert vo["toolcall_in_reasoning"] == 1 and vo["toolcall_in_reasoning_steps"] == ["c02 step 2"]
    assert vo["max_completion_tokens"] == 8192
    assert m["h200_h_per_run_job"] == pytest.approx(0.25) and "lower bound" in m["h200_h_per_run_note"]
    assert g.measure(case["out"], case["entries"])["h200_h_per_run_job"] is None
