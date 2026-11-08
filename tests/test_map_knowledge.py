"""map_knowledge.json (005 Step 0, N8; sim/INTERFACES.md section 2): rooms + doorways only, byte-deterministic,
committed, equal to truth.json's rooms block."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLINIC = ROOT / "sim" / "worlds" / "clinic"
sys.path.insert(0, str(ROOT))
from agent.world import load_doors, load_map, load_rooms  # noqa: E402


def test_deterministic_committed_and_other_outputs_unchanged(tmp_path: Path) -> None:
    outs = []
    for k in range(2):
        d = tmp_path / str(k)
        subprocess.run([sys.executable, str(CLINIC / "generate.py"), str(CLINIC / "layout.yaml"), "--out", str(d)],
                       check=True, capture_output=True)
        outs.append({f: (d / f).read_bytes() for f in ("clinic.sdf", "truth.json", "map_knowledge.json")})
    assert outs[0] == outs[1]
    for f, b in outs[0].items():
        assert b == (CLINIC / "generated" / f).read_bytes(), f"generated/{f} is stale"


def test_rooms_only_no_answers() -> None:
    m = json.loads((CLINIC / "generated" / "map_knowledge.json").read_text())
    truth = json.loads((CLINIC / "generated" / "truth.json").read_text())
    assert set(m) == {"world", "layout_sha256", "rooms"}
    assert m["rooms"] == truth["rooms"] and m["layout_sha256"] == truth["layout_sha256"]
    text = json.dumps(m)
    for o in truth["objects"]:  # no object id, class or state anywhere
        assert o["id"] not in text and f'"{o["class"]}"' not in text
    assert "open" not in text and "closed" not in text and "label" not in text


def test_loaders() -> None:
    rooms, doors = load_rooms(), load_doors()
    assert set(rooms) == {"hall", "ward", "office", "reception", "storage"}
    assert set(doors) == set(rooms) - {"hall"}
    assert load_map()["world"] == "clinic"


def test_load_map_refuses_extra_keys(tmp_path: Path) -> None:
    m = json.loads((CLINIC / "generated" / "map_knowledge.json").read_text())
    for bad in ({**m, "objects": []}, {**m, "rooms": {**m["rooms"], "ward": {**m["rooms"]["ward"], "state": "x"}}}):
        p = tmp_path / "m.json"
        p.write_text(json.dumps(bad))
        try:
            load_map(p)
        except AssertionError:
            continue
        raise AssertionError(f"load_map accepted {sorted(bad)}")
