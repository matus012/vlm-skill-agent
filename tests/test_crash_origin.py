"""N1 (005 Step 0): a crash records its origin in run.json and the scorer labels it from that (INTERFACES 4, 7)."""
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from agent import runtime, scorer, skills  # noqa: E402
from agent.fake_adapter import PLACES, FakeAdapter  # noqa: E402
from agent.planners import ScriptedPlanner  # noqa: E402
from agent.runlog import RunLog  # noqa: E402
from agent.skills import SkillContext  # noqa: E402
from agent.tasks import Task  # noqa: E402
from test_scorer_e2e import FAKE_TRUTH, META  # noqa: E402

ROOMS = ["hall", "ward", "reception", "office", "storage"]
TASK = Task("t", "find", "Find the red toolbox.", {"object": "toolbox_red_1"})
STEPS = [{"name": "navigate_to", "args": {"place": "storage"}}, {"name": "look_around", "args": {}},
         {"name": "report", "args": {"answer": "storage", "photo": 1, "note": ""}}]


class BoomPlanner(ScriptedPlanner):
    def next_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        raise ConnectionError("planner: server gone")


def _boom(*a: Any, **k: Any) -> Any:
    raise KeyError("injected")


def crash(tmp: Path, ad: FakeAdapter | None = None, planner: ScriptedPlanner | None = None,
          truth: Any = None) -> tuple[runtime.RunResult, dict[str, Any]]:
    ad = ad or FakeAdapter("hall")
    res = runtime.run(TASK, SkillContext(ad, PLACES, "find", ROOMS), planner or ScriptedPlanner(STEPS, "c"),
                      tmp / "run", META, truth=truth or ad.truth)
    assert res.outcome == "crash", res
    return res, json.loads((tmp / "run" / "run.json").read_text())


def label(tmp: Path) -> str:
    t, p = tmp / "truth.json", tmp / "places.yaml"
    t.write_text(json.dumps(FAKE_TRUTH))
    p.write_text("places: {}\n")
    return scorer.score_run(tmp / "run", truth_path=t, places_path=p)["label"]


@pytest.mark.parametrize("method", ["go_to", "turn_to", "photo", "where"])
def test_adapter_crash_is_sim(tmp_path: Path, method: str) -> None:
    """Directly from the loop (photo / where) or from inside a skill (go_to / turn_to): adapter -> sim."""
    res, run = crash(tmp_path, FakeAdapter("hall", fail_on=method))
    assert run["crash_origin"] == res.crash_origin == "adapter" and label(tmp_path) == "sim"


def test_skill_crash_is_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skills, "look_around_start", _boom)  # a bug in skills.execute's own code path
    res, run = crash(tmp_path)
    assert run["crash_origin"] == "skill" and label(tmp_path) == "skill" and res.steps == 2


def test_planner_crash_is_vlm(tmp_path: Path) -> None:
    _, run = crash(tmp_path, planner=BoomPlanner(STEPS, "c"))
    assert run["crash_origin"] == "planner" and label(tmp_path) == "vlm"


def test_truth_hook_crash_is_scorer(tmp_path: Path) -> None:
    _, run = crash(tmp_path, truth=_boom)
    assert run["crash_origin"] == "truth_hook" and label(tmp_path) == "scorer"


def test_runtime_crash_is_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RunLog, "save_photos", _boom)  # the run log fails on the start photo
    _, run = crash(tmp_path)
    assert run["crash_origin"] == "runtime" and label(tmp_path) == "skill"


def test_old_run_without_origin_stays_sim(tmp_path: Path) -> None:
    crash(tmp_path, FakeAdapter("hall", fail_on="photo"))
    p = tmp_path / "run" / "run.json"
    run = json.loads(p.read_text())
    del run["crash_origin"]
    p.write_text(json.dumps(run))
    assert label(tmp_path) == "sim"


# ------------------------------------------------------------------------------- N2 (005 design review): agent code
def test_context_building_crash_is_agent_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import context
    monkeypatch.setattr(context, "data_url", _boom)  # build_messages (agent/context.py) fails
    _, run = crash(tmp_path)
    assert run["crash_origin"] == "agent" and label(tmp_path) == "skill"


def test_vlm_response_handling_crash_is_agent_code(tmp_path: Path) -> None:
    """The HTTP call returned; our handling of the response (resp.choices[0]) raises: agent code, not a VLM choice."""
    from types import SimpleNamespace

    from agent.planners import VlmConfig, VlmPlanner
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[], usage=None))))
    _, run = crash(tmp_path, planner=VlmPlanner(VlmConfig("http://x/v1", "k", "m"), client=client))
    assert run["crash_origin"] == "agent" and label(tmp_path) == "skill"


def test_vlm_http_failure_is_an_error_step_not_a_crash(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from agent.planners import VlmConfig, VlmPlanner

    def create(**kw: Any) -> Any:
        raise ConnectionError("server gone")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    res = runtime.run(TASK, SkillContext(FakeAdapter("hall"), PLACES, "find", ROOMS),
                      VlmPlanner(VlmConfig("http://x/v1", "k", "m"), client=client), tmp_path / "run", META,
                      malformed_limit=2)
    assert res.outcome == "fail_step_limit"


@pytest.mark.parametrize("where", ["truth", "go_to"])
def test_gz_read_death_is_sim(tmp_path: Path, where: str) -> None:
    """006 a recorded deviation: a gz CLI read dead after its retries (GzReadError, gz_read_failed) -> origin gz_read -> sim,
    from the truth hook (job 143786 M4 run 06: was `scorer`) or from inside the adapter."""
    sys.path.insert(0, str(ROOT / "probes"))
    from gz_cli import GzReadError

    def dead(*a: Any, **k: Any) -> Any:
        raise GzReadError(-6, ["gz", "topic"], "", "")
    ad = FakeAdapter("hall")
    if where == "go_to":
        ad.go_to = dead  # type: ignore[method-assign]
    res, run = crash(tmp_path, ad, truth=dead if where == "truth" else None)
    assert run["crash_origin"] == res.crash_origin == "gz_read" and label(tmp_path) == "sim"
