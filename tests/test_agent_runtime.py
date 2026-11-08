import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from agent import context, runtime, skills  # noqa: E402
from agent.fake_adapter import PLACES, FakeAdapter  # noqa: E402
from agent.planners import Planner, ScriptedPlanner, VlmConfig, VlmPlanner, tool_call  # noqa: E402
from agent.runlog import check_complete  # noqa: E402
from agent.skills import SkillContext  # noqa: E402
from agent.tasks import Task  # noqa: E402
from mock_openai import MockServer, tool_call_msg  # noqa: E402
from mock_vlm import reasoning_step, scripted  # noqa: E402

ROOMS = ["ward", "reception", "office", "storage"]
META = {"robot": "fake", "layout_sha256": "0" * 64, "world_tag": "fake", "git_commit": "abc", "git_dirty": False,
        "start_place": "hall", "setup": {"start_place": "hall", "reached": True, "gt_err_m": 0.0}}


def task(kind: str = "find") -> Task:
    return Task(f"t-{kind}", kind, "Find the red toolbox.", {"object": "toolbox_red_1"})


def S(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args}


REPORT = S("report", answer="storage", photo=1, note="")


def go(planner: Planner, tmp: Path, kind: str = "find", ad: FakeAdapter | None = None,
       with_truth: bool = True) -> tuple[runtime.RunResult, FakeAdapter, list[dict[str, Any]]]:
    ad = ad or FakeAdapter("hall")
    res = runtime.run(task(kind), SkillContext(ad, PLACES, kind, ROOMS), planner, tmp / "run", META,
                      truth=ad.truth if with_truth else None)
    return res, ad, [json.loads(x) for x in (tmp / "run" / "steps.jsonl").read_text().splitlines()]


def test_scripted_find_end_to_end(tmp_path: Path) -> None:
    res, ad, log = go(ScriptedPlanner([S("navigate_to", place="storage"), S("look_around"),
                                       S("report", answer="storage", photo=1, note="toolbox")], "find-ok"), tmp_path)
    assert res.outcome == "reported" and res.steps == 3 and res.report["answer"] == "storage"
    assert [r["result"] for r in log] == ["reached", "4 photos", "reported"] and ad.stopped
    assert [r["check"] for r in log] == ["ok"] * 3
    assert log[2]["packet"]["history"] == ["1. navigate_to(storage) from hall -> reached", "2. look_around() in storage -> 4 photos"]
    assert [p["n"] for p in log[2]["packet"]["photos"]] == [1, 2, 3, 4]
    assert log[0]["truth"]["place_err_m"] == 0.0 and log[1]["truth"]["place_err_m"] is None
    assert check_complete(tmp_path / "run") == []
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["outcome"] == "reported" and run["planner"] == "scripted:find-ok" and run["steps"] == 3
    assert run["task"] == {"id": "t-find", "kind": "find", "prompt": "Find the red toolbox.",
                           "target": {"object": "toolbox_red_1"}} and run["sim_s"] > 0 and run["hand_help"] is False
    # scorer-ready: the reported photo's seg (resolved through the packet) has >= 1000 px of the toolbox label
    ph = {p["n"]: p for p in log[2]["packet"]["photos"]}[1]
    seg = np.asarray(Image.open(tmp_path / "run" / ph["seg"]))
    assert (seg == 10).sum() >= 1000 and Image.open(tmp_path / "run" / ph["rgb"]).mode == "RGB"


def test_errors_cost_a_step_and_never_reach_the_robot(tmp_path: Path) -> None:
    two = [tool_call("look_around", {}), tool_call("look_around", {})]
    script = [S("explore"), S("navigate_to", place="kitchen"), [tool_call("navigate_to", "{not json")],
              S("report", answer="garage", photo=1, note=""), two, [], REPORT]
    res, ad, log = go(ScriptedPlanner(script), tmp_path)
    checks = [r["check"] for r in log]
    assert checks[0].startswith("error: unknown skill 'explore'")
    assert checks[1].startswith("error: invalid navigate_to call at place")
    assert checks[2] == "error: arguments of navigate_to are not valid JSON"
    assert checks[3].startswith("error: invalid report call at answer")
    assert "exactly one tool" in checks[4] and "exactly one tool" in checks[5]
    assert res.outcome == "reported" and res.steps == 7
    assert all(r["result"] is None and r["call"]["name"] is None for r in log[:6])
    assert "go_to" not in ad.calls and "turn_to" not in ad.calls
    assert log[1]["packet"]["last_result"] == checks[0] and "kitchen" in log[2]["packet"]["history"][1]
    assert log[2]["call"]["raw"] == [tool_call("navigate_to", "{not json")]
    assert check_complete(tmp_path / "run") == []


def test_step_limit(tmp_path: Path) -> None:
    res, ad, log = go(ScriptedPlanner([S("look_around")] * 13), tmp_path)
    assert res.outcome == "fail_step_limit" and res.steps == 12 and len(log) == 12 and not ad.stopped
    assert check_complete(tmp_path / "run") == []


def test_report_on_step_12_counts(tmp_path: Path) -> None:
    res, _, _ = go(ScriptedPlanner([S("look_around")] * 11 + [REPORT]), tmp_path)
    assert res.outcome == "reported" and res.steps == 12


def test_script_exhausted_is_an_error_step(tmp_path: Path) -> None:
    res, _, log = go(ScriptedPlanner([S("look_around")]), tmp_path)
    assert log[1]["check"] == "error: planner call failed (script exhausted)"
    assert res.outcome == "fail_step_limit" and res.steps == 12


@pytest.mark.parametrize("fail_on,steps", [("turn_to", 1), ("go_to", 1), ("where", 1), ("photo", 0)])
def test_crash_is_logged_not_raised(tmp_path: Path, fail_on: str, steps: int) -> None:
    ad = FakeAdapter("hall", fail_on=fail_on)
    res, _, log = go(ScriptedPlanner([S("navigate_to", place="hall") if fail_on == "go_to" else S("look_around")]),
                     tmp_path, ad=ad)
    assert res.outcome == "crash" and res.steps == steps and len(log) == steps
    assert "RuntimeError: fake adapter: injected failure in " + fail_on in res.traceback
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["outcome"] == "crash" and "injected failure" in run["crash"] and run["end_utc"]
    if steps:
        assert "injected failure" in log[0]["traceback"]
    assert check_complete(tmp_path / "run") == []


def test_blocked_navigation(tmp_path: Path) -> None:
    ad = FakeAdapter("hall", blocked=(PLACES["storage"][:2],))
    res, ad, log = go(ScriptedPlanner([S("navigate_to", place="storage"), S("navigate_to", place="office"), REPORT]),
                      tmp_path, ad=ad)
    assert log[0]["result"] == "failed: blocked" and log[0]["skill_detail"]["nav_status"] == "ABORTED"
    assert len(log[0]["photos_out"]) == 1 and log[1]["packet"]["last_result"] == "failed: blocked"
    assert log[0]["truth"]["place_err_m"] > 3 and log[1]["result"] == "reached"


def test_without_truth_the_run_is_incomplete(tmp_path: Path) -> None:
    go(ScriptedPlanner([REPORT]), tmp_path, with_truth=False)
    assert check_complete(tmp_path / "run") == ["steps[0].truth.gt_pose_after: null"]


@pytest.mark.parametrize("kind,good,bad", [
    ("find", "storage", "garage"), ("conditional", "office", "yes"),
    ("check_state", "yes", "open"), ("check_state", "no", "3"),
    ("count", "3", "three"), ("count", "0", "03"), ("count", "2", "2.5")])
def test_answer_formats(tmp_path: Path, kind: str, good: str, bad: str) -> None:
    res, _, log = go(ScriptedPlanner([S("report", answer=bad, photo=1, note=""),
                                      S("report", answer=good, photo=1, note="")]), tmp_path, kind)
    assert log[0]["check"].startswith("error: invalid report call at answer")
    assert res.outcome == "reported" and res.steps == 2 and res.report["answer"] == good


def test_report_photo_must_exist_in_packet(tmp_path: Path) -> None:
    res, _, log = go(ScriptedPlanner([S("report", answer="ward", photo=2, note=""), REPORT]), tmp_path)
    assert "photo 2 does not exist" in log[0]["check"] and res.steps == 2


# --- VlmPlanner over the mock OpenAI server (Phase 3 path) ---

def vlm(srv: MockServer, thinking: bool = True) -> VlmPlanner:
    return VlmPlanner(VlmConfig(srv.base_url, "sekret", "mock", thinking=thinking))


@pytest.mark.parametrize("thinking", [True, False])
def test_vlm_thinking_flag_sampling_and_api_key_sent(tmp_path: Path, thinking: bool) -> None:
    srv = scripted(("report", {"answer": "ward", "photo": 1, "note": ""}))
    try:
        res, _, _ = go(vlm(srv, thinking), tmp_path)
    finally:
        srv.close()
    body, hdr = srv.requests[0], srv.headers[0]
    assert res.outcome == "reported"
    assert body["chat_template_kwargs"]["enable_thinking"] is thinking
    assert body["temperature"] == (1.0 if thinking else 0.7) and body["top_k"] == 20
    assert body["presence_penalty"] == (0.0 if thinking else 1.5)
    assert body["max_tokens"] == (8192 if thinking else 1024)
    assert {k.lower(): v for k, v in hdr.items()}["authorization"] == "Bearer sekret" and body["model"] == "mock"
    assert [t["function"]["name"] for t in body["tools"]] == ["navigate_to", "look_around", "report"]
    assert body["tools"][0]["function"]["parameters"]["properties"]["place"]["enum"] == list(PLACES)


def test_vlm_http_error_no_call_and_reasoning(tmp_path: Path) -> None:
    srv = MockServer([(500, None), (200, {"role": "assistant", "content": "I think storage."}),
                      reasoning_step("look_around", {}, "look first"),
                      (200, tool_call_msg("report", {"answer": "ward", "photo": 3, "note": "n"}))])
    try:
        res, _, log = go(vlm(srv), tmp_path)
    finally:
        srv.close()
    assert log[0]["planner_out"]["error"] and log[0]["check"].startswith("error: planner call failed")
    assert "exactly one tool" in log[1]["check"] and log[1]["planner_out"]["content"] == "I think storage."
    assert log[2]["planner_out"]["reasoning"] == "look first" and log[2]["planner_out"]["usage"]
    assert res.outcome == "reported" and res.steps == 4 and json.loads(
        (tmp_path / "run" / "run.json").read_text())["planner"] == "vlm:mock"
    raw = (tmp_path / "run" / "steps.jsonl").read_text()
    assert "base64" not in raw and "data:image" not in raw
    assert check_complete(tmp_path / "run") == []


def test_schema_single_source() -> None:
    tools = skills.build_tools(list(PLACES), "count")
    assert skills.schemas(tools)["report"] is tools[2]["function"]["parameters"]
    assert skills.is_valid_count("12") and not skills.is_valid_count("1 2")
    assert context.MAX_STEPS == 12 and math.isclose(skills.PLACE_BOUND_M, 0.30)


def test_history_lines_name_rooms_and_leak_no_coordinates(tmp_path: Path) -> None:
    import re
    import yaml
    names = set(yaml.safe_load((ROOT / "sim/worlds/clinic/places.yaml").read_text())) | set(PLACES)
    script = [S("navigate_to", place="storage"), S("look_around"), S("navigate_to", place="office"), S("look_around"),
              S("explore"), REPORT]
    _, _, log = go(ScriptedPlanner(script, "hist"), tmp_path)
    lines = log[-1]["packet"]["history"]
    assert lines[:4] == ["1. navigate_to(storage) from hall -> reached", "2. look_around() in storage -> 4 photos",
                         "3. navigate_to(office) from storage -> reached", "4. look_around() in office -> 4 photos"]
    assert len(lines) == 5 and lines[4].startswith("5. explore({}) in office -> error")
    for i, line in enumerate(lines[:4], 1):  # digits: the step number, and the "4 photos" count only
        m = re.fullmatch(r"(\d+)\. (navigate_to|look_around)\((\w*)\) (from|in) (\w+) -> (reached|\d+ photos)", line)
        assert m and int(m[1]) == i and m[5] in names and (m[3] in names or m[3] == "")
    assert not re.search(r"\d", lines[4].split(" -> ")[0][3:])  # the wrong call + room carry no digits
