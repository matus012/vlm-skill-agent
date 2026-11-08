"""Task 005 Phase B: malformed-output limit, the packet actually sent in the log, planner info + seed, run_one
start_place, the GT trace hook. No sim (FakeAdapter, mock OpenAI server)."""
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from adapters.hand_help import HandHelp  # noqa: E402
import adapters.run_task as rt  # noqa: E402
from adapters.run_task import run_one  # noqa: E402
from agent import runtime, scorer  # noqa: E402
from agent.context import system_prompt_sha256  # noqa: E402
from agent.fake_adapter import PLACES, FakeAdapter  # noqa: E402
from agent.planners import SAMPLING, ScriptedPlanner, VlmConfig, VlmPlanner, tool_call  # noqa: E402
from agent.skills import SkillContext  # noqa: E402
from agent.tasks import Task  # noqa: E402
from mock_openai import MockServer, tool_call_msg  # noqa: E402
from test_phase3_gates import TRUTH  # noqa: E402

ROOMS = ["hall", "ward", "reception", "office", "storage"]
META = {"robot": "fake", "layout_sha256": TRUTH["layout_sha256"], "world_tag": "fake", "git_commit": "abc",
        "git_dirty": False, "start_place": "hall", "setup": {"start_place": "hall", "reached": True, "gt_err_m": 0.0}}
TASK = Task("find-toolbox", "find", "Find the red toolbox.", {"object": "toolbox_red_1"})
GOOD = [{"name": "navigate_to", "args": {"place": "storage"}}, {"name": "look_around", "args": {}},
        {"name": "report", "args": {"answer": "storage", "photo": 1, "note": "seen"}}]
UNKNOWN = [tool_call("fly_to", {"place": "storage"})]
BAD_PLACE = {"name": "navigate_to", "args": {"place": "kitchen"}}


@pytest.fixture(autouse=True)
def clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """run.json git state as on a counted run (the shared tree is often dirty; copies are not repos)."""
    monkeypatch.setattr(rt, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    t, p, m = tmp_path / "truth.json", tmp_path / "places.yaml", tmp_path / "map_knowledge.json"
    t.write_text(json.dumps(TRUTH))
    p.write_text(yaml.safe_dump({"places": {n: {"x": x, "y": y, "yaw": a} for n, (x, y, a) in PLACES.items()}}))
    m.write_text(json.dumps({"world": "fake", "layout_sha256": TRUTH["layout_sha256"], "rooms": {r: {} for r in ROOMS}}))
    return {"truth_path": t, "places_path": p, "map_path": m}


def go(tmp: Path, planner: Any, limit: int | None = runtime.MALFORMED_LIMIT) -> tuple[runtime.RunResult, list[dict]]:
    ad = FakeAdapter("hall")
    res = runtime.run(TASK, SkillContext(ad, PLACES, "find", ROOMS), planner, tmp / "run", META, truth=ad.truth,
                      malformed_limit=limit)
    return res, [json.loads(x) for x in (tmp / "run" / "steps.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------------------------------- malformed outputs
def test_two_malformed_in_a_row_end_the_run(tmp_path: Path, world: dict) -> None:
    res, log = go(tmp_path, ScriptedPlanner([GOOD[0], UNKNOWN, BAD_PLACE, *GOOD[1:]]))
    assert res.outcome == "fail_malformed" and res.steps == 3 and len(log) == 3
    assert [r["check"] == "ok" for r in log] == [True, False, False]
    run = json.loads((tmp_path / "run/run.json").read_text())
    assert run["outcome"] == "fail_malformed" and run["malformed_limit"] == 2
    sc = scorer.score_run(tmp_path / "run", truth_path=world["truth_path"], places_path=world["places_path"])
    assert not sc["pass"] and sc["label"] == "vlm" and not sc["checks"]["steps"]
    assert sc["label_why"].startswith("2 malformed planner outputs in a row (steps 2, 3)") and "\n" not in sc["label_why"]
    assert "kitchen" in sc["label_why"]


def test_one_malformed_costs_a_step_and_goes_back(tmp_path: Path) -> None:
    res, log = go(tmp_path, ScriptedPlanner([UNKNOWN, GOOD[0], BAD_PLACE, *GOOD[1:]]))
    assert res.outcome == "reported" and res.steps == 5
    assert log[1]["packet"]["last_result"] == log[0]["check"] and log[1]["packet"]["history"][0].endswith(log[0]["check"])


def test_planner_call_failure_is_not_malformed_and_resets(tmp_path: Path) -> None:
    srv = MockServer([(200, tool_call_msg("fly_to", {})), (500, None), (200, tool_call_msg("fly_to", {})),
                      (200, tool_call_msg("navigate_to", {"place": "storage"})), (200, tool_call_msg("look_around", {})),
                      (200, tool_call_msg("report", {"answer": "storage", "photo": 1, "note": ""}))])
    try:
        res, log = go(tmp_path, VlmPlanner(VlmConfig(srv.base_url, "k", "mock", seed=7)))
    finally:
        srv.close()
    assert log[1]["planner_out"]["error"] and res.outcome == "reported" and res.steps == 6


def test_no_limit_keeps_phase2_behaviour(tmp_path: Path) -> None:
    res, _ = go(tmp_path, ScriptedPlanner([UNKNOWN, BAD_PLACE, *GOOD]), limit=None)
    assert res.outcome == "reported" and res.steps == 5
    assert json.loads((tmp_path / "run/run.json").read_text())["malformed_limit"] is None


# ------------------------------------------------------------------------------------------ packet actually sent
class Recorder(ScriptedPlanner):
    def __init__(self, steps: list) -> None:
        super().__init__(steps, "rec")
        self.sent: list[list[dict]] = []

    def next_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.sent.append(messages)
        return super().next_call(messages, tools)


def test_log_carries_the_packet_actually_sent(tmp_path: Path) -> None:
    p = Recorder(GOOD)
    _, log = go(tmp_path, p)
    run = json.loads((tmp_path / "run/run.json").read_text())
    assert run["system_prompt"] == p.sent[0][0]["content"]
    assert run["system_prompt_sha256"] == system_prompt_sha256(list(PLACES))
    assert [t["function"]["name"] for t in run["tools"]] == ["navigate_to", "look_around", "report"]
    assert run["planner_info"] == {"script": "rec"}
    for rec, msgs in zip(log, p.sent, strict=True):
        parts = msgs[1]["content"]
        pk = rec["packet"]
        assert pk["system_sha256"] == hashlib.sha256(msgs[0]["content"].encode()).hexdigest()
        assert pk["texts"] == [c["text"] for c in parts if c["type"] == "text"]
        urls = [c["image_url"]["url"] for c in parts if c["type"] == "image_url"]
        assert [ph["sent_sha256"] for ph in pk["photos"]] == [hashlib.sha256(u.encode()).hexdigest() for u in urls]
        for ph in pk["photos"]:
            assert ph["rgb_sha256"] == hashlib.sha256((tmp_path / "run" / ph["rgb"]).read_bytes()).hexdigest()
    assert len(log[2]["packet"]["photos"]) == 4
    raw = (tmp_path / "run/steps.jsonl").read_text()
    assert "base64" not in raw and "data:image" not in raw


# ---------------------------------------------------------------------------------------------- planner + seed
def test_vlm_seed_sent_and_info(tmp_path: Path) -> None:
    srv = MockServer([(200, tool_call_msg("report", {"answer": "hall", "photo": 1, "note": ""}))])
    try:
        go(tmp_path, VlmPlanner(VlmConfig(srv.base_url, "k", "mock", seed=1234)))
    finally:
        srv.close()
    assert srv.requests[0]["seed"] == 1234
    info = json.loads((tmp_path / "run/run.json").read_text())["planner_info"]
    assert info == {"model": "mock", "thinking": True, "sampling": SAMPLING[True], "max_tokens": 8192, "seed": 1234}


# -------------------------------------------------------------------------------------- run_one: start + trace
def one(world: dict, d: Path, ad: FakeAdapter, gt: Any, **kw: Any) -> dict[str, Any]:
    """hand_help bound 1 m/s: the fake adapter drives any go_to (<= 10 m) in 10 s sim (tests/test_phase2_gates.rk)."""
    return run_one(ad, gt, TASK, ScriptedPlanner(GOOD, "g"), d, robot="fake", tag="fake",
                   hand_help=lambda: HandHelp(v_max_mps=1.0), **world, **kw)


@pytest.mark.parametrize("start", ["office", "storage", "ward"])
def test_start_place(tmp_path: Path, world: dict, start: str) -> None:
    ad = FakeAdapter("hall")
    r = one(world, tmp_path / "r", ad, ad.pose, start_place=start)
    run = json.loads((tmp_path / "r/run.json").read_text())
    assert run["start_place"] == run["setup"]["start_place"] == start and r["setup"]["reached"] and r["score"]["pass"]
    log = [json.loads(x) for x in (tmp_path / "r/steps.jsonl").read_text().splitlines()]
    assert log[0]["packet"]["room"] == start


def test_start_place_gt_bound(tmp_path: Path, world: dict) -> None:
    ad = FakeAdapter("hall")
    r = one(world, tmp_path / "r", ad, lambda: (ad.x + 0.35, ad.y, ad.yaw), start_place="office")
    assert r["setup"]["nav_status"] == "SUCCEEDED" and not r["setup"]["reached"]


def test_gt_trace_samples_and_is_recorded(tmp_path: Path, world: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    import adapters.gt_trace as gtt
    monkeypatch.setattr(gtt, "PERIOD_S", 0.001)
    ad = FakeAdapter("hall")
    slow = ad.go_to
    monkeypatch.setattr(ad, "go_to", lambda *a: (time.sleep(0.05), slow(*a))[1])
    one(world, tmp_path / "r", ad, ad.pose)
    run = json.loads((tmp_path / "r/run.json").read_text())
    assert run["gt_trace"]["samples"] > 0 and run["gt_trace"]["errors"] == 0
    assert (tmp_path / "r/gt_trace.jsonl").is_file()


def test_gz_reads_recorded_per_run(tmp_path: Path, world: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """006 a recorded deviation: run.json gz_reads = this run's gz CLI reads, attempts and signal deaths (not earlier ones)."""
    import gz_cli
    monkeypatch.setattr(gz_cli, "DEATHS", [{"topic": "/before"}])
    monkeypatch.setattr(gz_cli, "STATS", {"reads": 5, "attempts": 6})
    ad = FakeAdapter("hall")

    def gt() -> tuple[float, float, float]:  # every GT read: 2 attempts, the first one dead
        gz_cli.STATS["reads"] += 1
        gz_cli.STATS["attempts"] += 2
        gz_cli.DEATHS.append({"topic": "/t", "attempt": 1, "signal": "SIGABRT"})
        return ad.pose()

    one(world, tmp_path / "r", ad, gt)
    g = json.loads((tmp_path / "r/run.json").read_text())["gz_reads"]
    assert g["reads"] >= 2 and g["attempts"] == 2 * g["reads"] and len(g["deaths"]) == g["reads"]
    assert all(d["signal"] == "SIGABRT" for d in g["deaths"])


@pytest.mark.parametrize("mode", ["raise", "hang"])
def test_gt_trace_failure_never_affects_the_run(tmp_path: Path, world: dict, mode: str,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    import adapters.gt_trace as gtt
    monkeypatch.setattr(gtt, "PERIOD_S", 0.001)
    monkeypatch.setattr(gtt, "STOP_WAIT_S", 0.05)
    ad0 = FakeAdapter("hall")
    base = one(world, tmp_path / "base", ad0, ad0.pose)
    ad = FakeAdapter("hall")
    release = threading.Event()

    def gt() -> tuple[float, float, float]:
        if threading.current_thread().name == "gt_trace":  # only the trace thread's calls fail
            if mode == "raise":
                raise RuntimeError("gz gone")
            release.wait(5)
        return ad.pose()

    r = one(world, tmp_path / "r", ad, gt)
    release.set()
    assert r["outcome"] == base["outcome"] == "reported" and r["steps"] == base["steps"]
    assert {k: r["score"][k] for k in ("pass", "checks", "label")} == {k: base["score"][k] for k in ("pass", "checks", "label")}
    tr = json.loads((tmp_path / "r/run.json").read_text())["gt_trace"]
    assert tr["samples"] == 0 and (tr["errors"] > 0 if mode == "raise" else tr["errors"] == 0)


# ------------------------------------------------------------------------- N3: the prompt says what the runtime does
def test_photos_stay_after_an_error_step_as_the_prompt_says(tmp_path: Path) -> None:
    """System prompt: photos come from the last navigate_to / look_around (start: one photo); a wrong call keeps them."""
    from agent.context import system_prompt
    sp = system_prompt(list(PLACES))
    assert "last navigate_to or look_around" in sp and "a wrong call does not change them" in sp
    _, log = go(tmp_path, ScriptedPlanner([UNKNOWN, GOOD[1], UNKNOWN, *GOOD[2:]]), limit=None)
    shown = [[p["rgb"] for p in r["packet"]["photos"]] for r in log]
    assert shown[0] == shown[1] == ["photos/0_0_rgb.png"]                 # start photo kept after the error step
    assert shown[2] == shown[3] == [f"photos/2_{k}_rgb.png" for k in range(4)]  # look_around photos kept likewise


def test_tool_descriptions_are_body_agnostic() -> None:
    from agent.skills import build_tools
    text = json.dumps(build_tools(list(PLACES), "find", ROOMS)).lower()
    assert "go to a named place" in text and not any(w in text for w in ("drive", "wheel", "robot", "camera"))
