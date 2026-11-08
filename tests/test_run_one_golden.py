"""Task 012 P3 (design section 2): run_one on agent.fake_adapter with the default robot / hand_help / place_objects /
trace_factory gives a run.json equal to tests/fixtures/run_one_golden.json, recorded from the code BEFORE the robots
factory (HEAD 758ce21), minus times / uuids / commit / tmp paths / the wall-clock GT trace sample count.
Cases: the frozen world, a seeded layout that lands (fake gz) and one whose read-back misses (teleport_failed).
Regenerate (only on a deliberate TB4 change, named in the commit): GOLDEN_WRITE=1 uv run pytest tests/test_run_one_golden.py"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import adapters.run_task as rt  # noqa: E402
from agent.fake_adapter import FakeAdapter  # noqa: E402
from agent.planners import ScriptedPlanner  # noqa: E402
from test_seeded_layout import GOOD, SEEDED, TASK, FakeGz, FakeSeeded, world  # noqa: E402,F401

FIXTURE = ROOT / "tests" / "fixtures" / "run_one_golden.json"
VOLATILE = {"start_utc", "end_utc", "wall_s", "git_commit", "git_dirty", "invocation_id"}


def scrub(x: Any, tmp: str) -> Any:
    if isinstance(x, dict):
        return {k: "<volatile>" if k in VOLATILE else scrub(v, tmp) for k, v in x.items()}
    if isinstance(x, list):
        return [scrub(v, tmp) for v in x]
    return x.replace(tmp, "<tmp>") if isinstance(x, str) else x


def golden_run(tmp_path: Path, world: dict, case: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(rt, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
    monkeypatch.delenv("SIM_WORLD", raising=False)
    kw: dict[str, Any] = {}
    if case != "frozen":
        monkeypatch.setitem(sys.modules, SEEDED, FakeSeeded())
        gz = FakeGz(subprocess.run)
        if case == "seeded_miss":
            gz.off["m_chair_orange_2"] = (0.05, 0.0)
        monkeypatch.setattr(subprocess, "run", gz)
        monkeypatch.setattr(rt, "READBACK_WAIT_S", 0.0)
        kw["layout_seed"] = 7
    ad = FakeAdapter("hall")
    d = tmp_path / "r"
    rt.run_one(ad, ad.pose, TASK, ScriptedPlanner(GOOD, "g"), d, tag="fake", **world, **kw)
    run = json.loads((d / "run.json").read_text())
    if "gt_trace" in run:
        run["gt_trace"]["samples"] = "<volatile>"  # 2 Hz wall-clock thread
    return scrub(run, str(tmp_path))


CASES = ("frozen", "seeded", "seeded_miss")


@pytest.mark.parametrize("case", CASES)
def test_run_one_defaults_match_the_pre_factory_fixture(tmp_path: Path, world: dict, case: str,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    got = golden_run(tmp_path, world, case, monkeypatch)
    if os.environ.get("GOLDEN_WRITE"):
        FIXTURE.parent.mkdir(exist_ok=True)
        old = json.loads(FIXTURE.read_text()) if FIXTURE.is_file() else {}
        FIXTURE.write_text(json.dumps({**old, case: got}, indent=1, sort_keys=True) + "\n")
    assert got == json.loads(FIXTURE.read_text())[case]
