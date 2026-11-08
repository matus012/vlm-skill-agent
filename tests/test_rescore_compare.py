"""probes/rescore_compare.py on synthetic run dirs: same verdicts pass, any pass / checks / label difference or a
missing score.json fails, a changed why alone is listed, nothing is written into the run dirs."""
import functools
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from agent import scorer  # noqa: E402
from probes import rescore_compare as rc  # noqa: E402
from test_scorer import TOOLBOX, make_run, world  # noqa: E402,F401


@pytest.fixture
def block(tmp_path: Path, world: dict) -> Path:  # noqa: F811
    """A gate --out dir: r1 passes, r2 wrong answer, r3 no report; score.json written by the scorer."""
    out = tmp_path / "out"
    for name, answer, outcome in (("r1", "storage", "reported"), ("r2", "office", "reported"),
                                  ("r3", None, "fail_step_limit")):
        scorer.write_score(make_run(out / name, TOOLBOX, answer, 3, {3: {40: 9100}}, outcome=outcome), **world)
    (out / "g1.json").write_text("{}")  # gate files next to the run dirs are not run dirs
    return out


def tree_hash(d: Path) -> str:
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file())).hexdigest()


def edit_score(d: Path, **kv: object) -> None:
    p = d / "score.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), **kv}))


def test_same_verdicts_pass_and_nothing_written(block: Path, world: dict, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
                                                capsys: pytest.CaptureFixture) -> None:
    h = tree_hash(block)
    rows, n = rc.compare([block], **world)
    assert n == 0 and [Path(r["run"]).name for r in rows] == ["r1", "r2", "r3"] and all(r["same"] for r in rows)
    monkeypatch.setattr(rc, "score_run", functools.partial(scorer.score_run, **world))
    assert rc.main([str(block)]) == 0 and "3 runs, 0 differ" in capsys.readouterr().out
    assert tree_hash(block) == h


@pytest.mark.parametrize("run,kv", [("r2", {"label": "nav"}), ("r1", {"pass": False}),
                                    ("r2", {"checks": {"answer": True, "proof": True, "steps": True, "clean": True}})])
def test_any_verdict_difference_fails(block: Path, world: dict, monkeypatch: pytest.MonkeyPatch, run: str,  # noqa: F811
                                      kv: dict, capsys: pytest.CaptureFixture) -> None:
    edit_score(block / run, **kv)
    rows, n = rc.compare([block], **world)
    assert n == 1 and [r["same"] for r in rows] == [run != "r1", run != "r2", True]
    monkeypatch.setattr(rc, "score_run", functools.partial(scorer.score_run, **world))
    assert rc.main([str(block)]) == 1 and "DIFF" in capsys.readouterr().out


def test_missing_score_json_fails(block: Path, world: dict) -> None:  # noqa: F811
    (block / "r3/score.json").unlink()
    assert rc.compare([block], **world)[1] == 1


def test_why_change_alone_is_listed_not_failed(block: Path, world: dict) -> None:  # noqa: F811
    edit_score(block / "r2", label_why="old wording")
    rows, n = rc.compare([block], **world)
    assert n == 0 and [r["why_changed"] for r in rows] == [False, True, False]


def test_no_run_dirs_fails(tmp_path: Path) -> None:
    assert rc.main([str(tmp_path)]) == 1
