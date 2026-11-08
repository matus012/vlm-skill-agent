import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import runtime  # noqa: E402
from agent.fake_adapter import PLACES, FakeAdapter  # noqa: E402
from agent.planners import ScriptedPlanner  # noqa: E402
from agent.runlog import check_complete, git_state  # noqa: E402
from agent.skills import SkillContext  # noqa: E402
from agent.tasks import Task  # noqa: E402

META = {"robot": "fake", "layout_sha256": "0" * 64, "world_tag": "fake", "git_commit": "abc", "git_dirty": False,
        "start_place": "hall", "setup": {"start_place": "hall", "reached": True, "gt_err_m": 0.0}}


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    ad = FakeAdapter("hall")
    t = Task("t", "find", "Find the red toolbox.", {"object": "toolbox_red_1"})
    script = [{"name": "navigate_to", "args": {"place": "storage"}}, {"name": "look_around", "args": {}},
              {"name": "report", "args": {"answer": "storage", "photo": 1, "note": ""}}]
    runtime.run(t, SkillContext(ad, PLACES, "find", ["ward", "reception", "office", "storage"]),
                ScriptedPlanner(script), tmp_path / "run", META, truth=ad.truth)
    return tmp_path / "run"


def edit_json(p: Path, fn) -> None:  # type: ignore[no-untyped-def]
    d = json.loads(p.read_text())
    fn(d)
    p.write_text(json.dumps(d))


def edit_step(run_dir: Path, i: int, fn) -> None:  # type: ignore[no-untyped-def]
    p = run_dir / "steps.jsonl"
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    fn(lines[i])
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))


def test_complete_run_and_layout(run_dir: Path) -> None:
    assert check_complete(run_dir) == []
    names = sorted(f.name for f in (run_dir / "photos").iterdir())
    assert names == sorted(["0_0_rgb.png", "0_0_seg.png", "1_0_rgb.png", "1_0_seg.png"] +
                           [f"2_{k}_{s}.png" for k in range(4) for s in ("rgb", "seg")])


def test_missing_run_field(run_dir: Path) -> None:
    edit_json(run_dir / "run.json", lambda d: d.pop("world_tag"))
    assert check_complete(run_dir) == ["run.world_tag: missing"]


def test_missing_nested_and_null_fields(run_dir: Path) -> None:
    edit_json(run_dir / "run.json", lambda d: d["setup"].pop("gt_err_m"))
    edit_step(run_dir, 1, lambda r: r.pop("skill_wall_s"))
    edit_step(run_dir, 2, lambda r: r["packet"].update(room=None))
    assert check_complete(run_dir) == ["run.setup.gt_err_m: missing", "steps[1].skill_wall_s: missing",
                                       "steps[2].packet.room: null"]


def test_missing_png(run_dir: Path) -> None:
    (run_dir / "photos" / "2_3_seg.png").unlink()
    errs = check_complete(run_dir)
    assert errs == ["steps[1].photos_out[3].seg: file photos/2_3_seg.png missing",
                    "steps[2].packet.photos[3].seg: file photos/2_3_seg.png missing"]


def test_step_count_mismatch_and_missing_files(run_dir: Path, tmp_path: Path) -> None:
    edit_json(run_dir / "run.json", lambda d: d.update(steps=4))
    assert check_complete(run_dir) == ["run.steps = 4 but steps.jsonl has 3 lines"]
    assert check_complete(tmp_path / "nothing") == ["run.json: missing", "steps.jsonl: missing"]


def test_git_state() -> None:
    g = git_state(ROOT)
    assert len(g["git_commit"]) == 40 and isinstance(g["git_dirty"], bool)
