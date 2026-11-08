"""Clinic world generator (task 002, sim/INTERFACES.md sections 1-2): determinism, truth.json <-> SDF agreement,
label ids. Run: uv run pytest tests/test_world_generator.py"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

CLINIC = Path(__file__).resolve().parents[1] / "sim" / "worlds" / "clinic"
LAYOUT = CLINIC / "layout.yaml"
sys.path.insert(0, str(CLINIC))
import generate  # noqa: E402

STRUCTURE = {"structure_floor": 241, "structure_walls": 240, "structure_ceiling": 242, "structure_trim": 243}


def _models(sdf: str) -> dict[str, tuple[list[float], list[int]]]:
    """model name -> (pose, label ids found anywhere inside the model)."""
    world = ET.fromstring(sdf).find("world")
    assert world is not None
    out: dict[str, tuple[list[float], list[int]]] = {}
    for m in world.findall("model"):
        name = m.get("name")
        assert name not in out, f"duplicate model {name}"
        pose = [float(v) for v in m.findtext("pose", "0 0 0 0 0 0").split()]
        labels = [int(p.findtext("label")) for p in m.iter("plugin") if p.get("name") == "gz::sim::systems::Label"]
        out[name] = (pose, labels)
    return out


@pytest.fixture(scope="module")
def generated() -> tuple[str, str]:
    return generate.generate(LAYOUT)


def test_deterministic_and_committed(tmp_path: Path) -> None:
    outs = []
    for k in range(2):
        d = tmp_path / str(k)
        subprocess.run([sys.executable, str(CLINIC / "generate.py"), str(LAYOUT), "--out", str(d)], check=True)
        outs.append(((d / "clinic.sdf").read_bytes(), (d / "truth.json").read_bytes()))
    assert outs[0] == outs[1]
    assert outs[0][0] == (CLINIC / "generated" / "clinic.sdf").read_bytes(), "generated/clinic.sdf is stale"
    assert outs[0][1] == (CLINIC / "generated" / "truth.json").read_bytes(), "generated/truth.json is stale"


def test_truth_matches_sdf(generated: tuple[str, str]) -> None:
    sdf, truth_s = generated
    truth = json.loads(truth_s)
    models = _models(sdf)
    objects = {o["model_name"]: o for o in truth["objects"]}
    assert set(models) - set(STRUCTURE) == set(objects), "SDF objects != truth.json objects"
    for name, o in objects.items():
        pose, labels = models[name]
        assert pose == pytest.approx(o["pose"], abs=1e-4), name
        assert labels == [o["label"]], name
    assert truth["layout_sha256"] == hashlib.sha256(LAYOUT.read_bytes()).hexdigest()
    assert truth["world"] == "clinic" and ET.fromstring(sdf).find("world").get("name") == "clinic"
    assert truth["structure_labels"] == {"wall": 240, "floor": 241, "ceiling": 242, "trim": 243, "other": 244}
    assert set(truth["rooms"]) == {"hall", "ward", "reception", "storage", "office"}


def test_labels_unique_and_in_range(generated: tuple[str, str]) -> None:
    sdf, truth_s = generated
    truth = json.loads(truth_s)
    labels = [o["label"] for o in truth["objects"]]
    assert len(labels) == len(set(labels))
    assert all(1 <= v <= 199 for v in labels)
    models = _models(sdf)
    for name, want in STRUCTURE.items():
        assert models[name][1] == [want], name
    for name, (_, lab) in models.items():
        assert len(lab) == 1, f"{name} has labels {lab}"


def test_layout_content(generated: tuple[str, str]) -> None:
    truth = json.loads(generated[1])
    objs = truth["objects"]
    by_class = lambda c: [o for o in objs if o["class"] == c]  # noqa: E731
    assert [o["colour"] for o in by_class("toolbox")] == ["red"]
    assert len(by_class("cabinet")) == 1 and by_class("cabinet")[0]["state"] in ("open", "closed")
    assert 3 <= len(by_class("chair")) <= 5 and all(o["room"] == "reception" for o in by_class("chair"))
    assert 2 <= len(by_class("bed")) <= 3 and all(o["room"] == "ward" for o in by_class("bed"))
    assert all(o["countable"] == (o["class"] in ("chair", "bed")) for o in objs)
    for room in ("ward", "reception", "storage", "office"):
        r = truth["rooms"][room]
        assert r["doorway"]["width"] >= 1.2
        xs = [p[0] for p in r["polygon"]]
        ys = [p[1] for p in r["polygon"]]
        for o in objs:
            if o["room"] == room:
                assert min(xs) < o["pose"][0] < max(xs) and min(ys) < o["pose"][1] < max(ys), o["id"]


def test_cabinet_both_states(tmp_path: Path) -> None:
    L = yaml.safe_load(LAYOUT.read_text())
    sdfs = {}
    for state in ("open", "closed"):
        for o in L["objects"]:
            if o["class"] == "cabinet":
                o["state"] = state
        p = tmp_path / f"{state}.yaml"
        p.write_text(yaml.safe_dump(L))
        sdf, truth = generate.generate(p)
        assert [o["state"] for o in json.loads(truth)["objects"] if o["class"] == "cabinet"] == [state]
        sdfs[state] = sdf
    assert sdfs["open"] != sdfs["closed"]
    with pytest.raises(ValueError):
        generate.parts_for("cabinet", "ajar")
    with pytest.raises(ValueError):
        generate.parts_for("chair", "open")


def test_objects_inside_their_room() -> None:
    """Every part of every object lies inside its room's interior (wall inner faces), so nothing pokes through a wall,
    and nothing hangs below the floor or above the ceiling."""
    import math

    L = yaml.safe_load(LAYOUT.read_text())
    t, cz = L["wall"]["thickness"] / 2, L["ceiling"]["z"]
    for o in L["objects"]:
        x0, y0, x1, y1 = L["rooms"][o["room"]]["rect"]
        x, y, yaw = o["pose"]
        c, s = math.cos(yaw), math.sin(yaw)
        for kind, size, (px, py, pz), _ in generate.parts_for(o["class"], o["state"]):
            hx, hy, hz = (size[0] / 2, size[1] / 2, size[2] / 2) if kind == "box" else \
                (size[0], size[0], size[1] / 2) if kind == "cyl" else (size, size, size)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    lx, ly = px + sx * hx, py + sy * hy
                    wx, wy = x + c * lx - s * ly, y + s * lx + c * ly
                    assert x0 + t - 1e-4 <= wx <= x1 - t + 1e-4 and y0 + t - 1e-4 <= wy <= y1 - t + 1e-4, \
                        f"{o['id']} part at ({wx:.3f},{wy:.3f}) outside {o['room']}"
            assert pz - hz >= -1e-6 and pz + hz <= cz, f"{o['id']} part z out of [0, ceiling]"


# 002 finding 13 (design ruling, task 003 step 0): no trick questions, so the toolbox is the only red box-shaped object in
# the world, in both cabinet states. "Box-shaped object" = a red box part whose every side is > 0.11 m: the trolley's old
# 0.20 x 0.30 x 0.12 bin and the open cabinet's 0.30 x 0.35 x 0.20 box count; a book (0.10 thick) and the water cooler's
# 4 cm tap do not.
RED_BOX_MIN_SIDE_M = 0.11


def red_box_parts(layout: dict) -> list[tuple[str, str, tuple]]:
    out = []
    for o in layout["objects"]:
        states = ("open", "closed") if o["class"] == "cabinet" else (None,)
        for st in states:
            for kind, size, _, colour in generate.parts_for(o["class"], st):
                c = o["colour"] if colour == "main" else colour
                if kind == "box" and c == "red" and min(size) > RED_BOX_MIN_SIDE_M:
                    out.append((o["id"], str(st), size))
    return out


def test_toolbox_is_the_only_red_box_shaped_object() -> None:
    found = red_box_parts(yaml.safe_load(LAYOUT.read_text()))
    assert found and {oid for oid, _, _ in found} == {"toolbox_red_1"}, found
