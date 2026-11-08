"""probes/phase2_gates.py stops a non-rehearsal invocation before the next sim run once the tree is dirty or on
another commit than at start (2026-11-04). No sim: fake adapter module, fake run_one, injected git_state."""
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probes import phase2_gates as pg  # noqa: E402

CLEAN, DIRTY = {"git_commit": "c0ffee", "git_dirty": False}, {"git_commit": "c0ffee", "git_dirty": True}
MOVED = {"git_commit": "beef", "git_dirty": False}


class Stop(Exception):
    pass


@pytest.fixture
def sim(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake adapters.turtlebot4 + run_one; run_one raises Stop on its 3rd call (ends a rehearsal run early)."""
    seen: dict[str, Any] = {"runs": [], "closed": False}

    class Ad:
        def close(self) -> None:
            seen["closed"] = True
    mod = types.ModuleType("adapters.turtlebot4")
    mod.TB4Adapter, mod.SimTruth = Ad, lambda: types.SimpleNamespace(gt_pose=None)
    monkeypatch.setitem(sys.modules, "adapters.turtlebot4", mod)

    def run_one(ad: Any, gt: Any, task: Any, planner: Any, d: Path, **kw: Any) -> dict[str, Any]:
        seen["runs"].append(d.name)
        if len(seen["runs"]) == 3:
            raise Stop
        return {"outcome": "reported", "score": {"pass": True, "checks": {}}, "setup": {"gt_err_m": 0.0}}
    monkeypatch.setattr(pg, "run_one", run_one)
    return seen


def run_main(tmp: Path, monkeypatch: pytest.MonkeyPatch, seq: list[dict[str, Any]], *args: str) -> None:
    left = list(seq)
    monkeypatch.setattr(pg, "git_state", lambda repo: left.pop(0) if len(left) > 1 else left[0])
    monkeypatch.setattr(sys, "argv", ["phase2_gates.py", "g1", "--out", str(tmp / "out"), *args])
    pg.main()


@pytest.mark.parametrize("later,needle", [(DIRTY, "tree dirty on commit c0ffee"),
                                          (MOVED, "tree on commit beef, not the start commit c0ffee")])
def test_stops_before_the_next_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sim: dict, later: dict,
                                   needle: str) -> None:
    with pytest.raises(SystemExit, match=f"stopping the invocation before the next run: {needle}") as e:
        run_main(tmp_path, monkeypatch, [CLEAN, CLEAN, later])  # start, guard before run 1, guard before run 2
    assert e.value.code != 0 and sim["runs"] == ["00-find-toolbox"] and sim["closed"]


def test_rehearsal_does_not_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sim: dict) -> None:
    with pytest.raises(Stop):
        run_main(tmp_path, monkeypatch, [DIRTY], "--rehearsal")
    assert len(sim["runs"]) == 3


def test_tree_moved() -> None:
    assert pg.tree_moved(CLEAN, CLEAN) is None
    assert pg.tree_moved(CLEAN, DIRTY) == "tree dirty on commit c0ffee"
    assert pg.tree_moved(CLEAN, MOVED) == "tree on commit beef, not the start commit c0ffee"
