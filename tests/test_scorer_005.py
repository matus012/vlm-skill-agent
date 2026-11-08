"""Scorer label rules from the 005 design review (agent/INTERFACES.md section 7): B2 no-call steps in the why, B3 VLM
endpoint error -> sim, N6 setup miss -> nav, precedence. Label / why only: pass and checks never change."""
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from agent import scorer  # noqa: E402
from test_scorer import TOOLBOX, make_run, world  # noqa: E402,F401

PROOF = {3: {40: 9100}}


def edit(d: Path, run: dict[str, Any] | None = None, step: Any = None) -> Path:
    if run:
        p = d / "run.json"
        p.write_text(json.dumps({**json.loads(p.read_text()), **run}))
    if step:
        recs = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines()]
        for r in recs:
            step(r)
        (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return d


def called(r: dict) -> None:
    """Give every step a received tool call (a VLM-shaped record), so only the steps a test empties are no-call."""
    r["call"]["raw"] = [{"function": {"name": r["call"]["name"] or "look_around", "arguments": "{}"}}]
    r["planner_out"].update(finish_reason="tool_calls", usage={"completion_tokens": 50}, reasoning="look")


def vlm_run(d: Path, answer: str = "office", **kw: Any) -> Path:
    return edit(make_run(d, TOOLBOX, answer, 3, PROOF, **kw), {"planner": "vlm:fake"}, called)


def score(d: Path, world: dict) -> dict[str, Any]:
    return scorer.score_run(d, **world)


# ------------------------------------------------------------------------------------------------------- B3
def test_endpoint_error_is_sim(tmp_path: Path, world: dict) -> None:
    d = vlm_run(tmp_path / "r", None, n_steps=12, outcome="fail_step_limit")
    base = score(d, world)
    assert base["label"] == "vlm"
    def fn(r: dict) -> None:
        if r["step"] in (4, 5):  # as VlmPlanner logs it: error set, no tool calls
            r["call"].update(raw=[], name=None, args=None)
            r["planner_out"].update(error="APIConnectionError: Connection error.\n x" + "y" * 200, reasoning=None,
                                    finish_reason=None, usage=None)
    edit(d, step=fn)
    sc = score(d, world)
    assert sc["label"] == "sim" and sc["label_why"] == ("VLM endpoint error at step 4: "
                                                        + ("APIConnectionError: Connection error. x" + "y" * 200)[:80])
    assert (sc["pass"], sc["checks"]) == (base["pass"], base["checks"])


def test_endpoint_error_beats_blocked_but_not_crash(tmp_path: Path, world: dict) -> None:
    d = vlm_run(tmp_path / "r", blocked_place="storage")
    assert score(d, world)["label"] == "nav"
    edit(d, step=lambda r: r["planner_out"].update(error="APITimeoutError: x") if r["step"] == 2 else None)
    assert score(d, world)["label"] == "sim"
    edit(d, {"outcome": "crash", "crash_origin": "skill"})
    assert score(d, world)["label"] == "skill"


def test_scripted_planner_error_is_not_an_endpoint_error(tmp_path: Path, world: dict) -> None:
    d = make_run(tmp_path / "r", TOOLBOX, "office", 3, PROOF)  # planner scripted:t
    edit(d, step=lambda r: r["planner_out"].update(error="script exhausted") if r["step"] == 2 else None)
    assert score(d, world)["label"] == "vlm"


def test_passing_run_with_an_endpoint_error_stays_a_pass(tmp_path: Path, world: dict) -> None:
    d = vlm_run(tmp_path / "r", "storage")
    edit(d, step=lambda r: r["planner_out"].update(error="APITimeoutError: x") if r["step"] == 1 else None)
    sc = score(d, world)
    assert sc["pass"] and sc["label"] is None and sc["label_why"] is None


# ------------------------------------------------------------------------------------------------------- N6
def test_setup_miss_is_nav_first(tmp_path: Path, world: dict) -> None:
    d = vlm_run(tmp_path / "r", blocked_place="storage")
    edit(d, {"setup": {"start_place": "office", "reached": False, "gt_err_m": 0.456}, "outcome": "crash",
             "crash_origin": "adapter"}, lambda r: r["planner_out"].update(error="APITimeoutError: x"))
    sc = score(d, world)
    assert sc["label"] == "nav" and sc["label_why"] == "setup did not reach office (GT 0.46 m)"


def test_setup_reached_or_absent_changes_nothing(tmp_path: Path, world: dict) -> None:
    d = vlm_run(tmp_path / "r")
    assert score(d, world)["label_why"] == "answer 'office' != truth 'storage'"
    run = json.loads((d / "run.json").read_text())
    del run["setup"]
    (d / "run.json").write_text(json.dumps(run))
    assert score(d, world)["label_why"] == "answer 'office' != truth 'storage'"


def test_passing_run_with_setup_miss_keeps_no_label(tmp_path: Path, world: dict) -> None:
    d = edit(vlm_run(tmp_path / "r", "storage"), {"setup": {"start_place": "hall", "reached": False, "gt_err_m": 1.0}})
    sc = score(d, world)
    assert sc["pass"] and sc["label"] is None  # G1 labels it (phase3_gates g1_run), the scorer's pass is four checks


# ------------------------------------------------------------------------------------------------------- B2
@pytest.mark.parametrize("reasoning,content,inside", [
    ("I will call <tool_call>{\"name\": \"look_around\"}", None, "yes"),
    ("then <function=navigate_to>", None, "yes"),
    ('{"name": "report", "arguments": {}}', None, "yes"),
    (None, '<tool_call>{"name":"report"}</tool_call>', "yes"),
    ("I think the toolbox is red and I should look around", None, "no"),
    ('{"name": "teleport"}', None, "no"),
])
def test_no_call_steps_named_in_the_why(tmp_path: Path, world: dict, reasoning: Any, content: Any,
                                        inside: str) -> None:
    def fn(r: dict) -> None:
        if r["step"] == 2:
            r["call"].update(raw=[], name=None, args=None)
            r["check"] = "error: call exactly one tool per step (got 0)"
            r["planner_out"].update(finish_reason="length", usage={"completion_tokens": 8192}, reasoning=reasoning,
                                    content=content)
    d = edit(vlm_run(tmp_path / "r"), step=fn)
    sc = score(d, world)
    assert sc["label"] == "vlm" and "\n" not in sc["label_why"]
    assert sc["label_why"] == ("answer 'office' != truth 'storage'; no-call steps: step 2 finish_reason length, "
                               f"completion_tokens 8192, tool call inside reasoning {inside}")


def test_fail_malformed_why_names_each_bad_step(tmp_path: Path, world: dict) -> None:
    def fn(r: dict) -> None:
        if r["step"] in (2, 3):
            r["call"].update(raw=[], name=None, args=None)
            r["check"] = "error: call exactly one tool per step (got 0)"
            r["planner_out"].update(finish_reason="length" if r["step"] == 2 else "stop", usage=None,
                                    reasoning="<tool_call>" if r["step"] == 3 else "hmm")
    d = edit(vlm_run(tmp_path / "r", None, outcome="fail_malformed"), {"malformed_limit": 2}, fn)
    why = score(d, world)["label_why"]
    assert why.startswith("2 malformed planner outputs in a row (steps 2, 3)")
    assert why.endswith("; no-call steps: step 2 finish_reason length, completion_tokens None, tool call inside "
                        "reasoning no, step 3 finish_reason stop, completion_tokens None, tool call inside reasoning "
                        "yes")


def test_bad_call_or_error_step_is_not_a_no_call_step(tmp_path: Path, world: dict) -> None:
    def fn(r: dict) -> None:
        if r["step"] == 2:
            r["call"]["raw"] = [{"function": {"name": "fly", "arguments": "{}"}}]
            r["planner_out"]["reasoning"] = "<tool_call>"
    d = edit(vlm_run(tmp_path / "r"), step=fn)
    assert score(d, world)["label_why"] == "answer 'office' != truth 'storage'"
