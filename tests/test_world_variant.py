"""World variant cabinet_open (task 007, hpc/PERUN_SIM.md section 5): committed layout file, byte-identical
regeneration, and the only difference from the frozen closed world is cabinet_blue_1's state (+ the layout sha256)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLINIC = ROOT / "sim" / "worlds" / "clinic"
OPEN_LAYOUT = CLINIC / "layout_cabinet_open.yaml"
OPEN_DIR = CLINIC / "generated_cabinet_open"
FILES = ("clinic.sdf", "truth.json", "map_knowledge.json")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLINIC))
import test_world_generator as twg  # noqa: E402
from agent.truth import TRUTH_JSON, WORLD_DIRS, world_truth_path  # noqa: E402


def test_regeneration_byte_identical(tmp_path: Path) -> None:
    for k in range(2):
        d = tmp_path / str(k)
        subprocess.run([sys.executable, str(CLINIC / "generate.py"), str(OPEN_LAYOUT), "--out", str(d)],
                       check=True, capture_output=True)
        for f in FILES:
            assert (d / f).read_bytes() == (OPEN_DIR / f).read_bytes(), f"generated_cabinet_open/{f} is stale"
    assert sorted(p.name for p in OPEN_DIR.iterdir()) == sorted(p.name for p in (CLINIC / "generated").iterdir())


def test_layout_differs_in_the_cabinet_state_line_only() -> None:
    a, b = (CLINIC / "layout.yaml").read_text().splitlines(), OPEN_LAYOUT.read_text().splitlines()
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 1, diff
    x, y = diff[0]
    assert "id: cabinet_blue_1," in x and x.endswith("state: closed}")
    assert y.split("  #")[0] == x.replace("state: closed}", "state: open}")


def test_truth_differs_in_cabinet_state_and_layout_sha_only() -> None:
    closed, opened = (json.loads((d / "truth.json").read_text()) for d in (CLINIC / "generated", OPEN_DIR))
    assert closed["layout_sha256"] != opened["layout_sha256"]
    for o in closed["objects"]:
        if o["id"] == "cabinet_blue_1":
            assert o["state"] == "closed"
            o["state"] = "open"
    assert {**closed, "layout_sha256": opened["layout_sha256"]} == opened
    mc, mo = (json.loads((d / "map_knowledge.json").read_text()) for d in (CLINIC / "generated", OPEN_DIR))
    assert {**mc, "layout_sha256": mo["layout_sha256"]} == mo


def test_closed_sdf_header_unchanged_and_open_names_its_layout() -> None:
    assert "generate.py from layout.yaml (sha256" in (CLINIC / "generated" / "clinic.sdf").read_text()
    assert "generate.py from layout_cabinet_open.yaml (sha256" in (OPEN_DIR / "clinic.sdf").read_text()


def test_open_cabinet_inside_its_room(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(twg, "LAYOUT", OPEN_LAYOUT)
    twg.test_objects_inside_their_room()


def test_truth_path_per_world() -> None:
    assert world_truth_path() == TRUTH_JSON == CLINIC / "generated" / "truth.json"
    assert world_truth_path("cabinet_open") == OPEN_DIR / "truth.json"
    assert {"closed", "cabinet_open"} <= set(WORLD_DIRS)  # + the Phase 6 blocked_<room> (tests/test_blocked_worlds.py)
    with pytest.raises(KeyError):
        world_truth_path("ajar")
