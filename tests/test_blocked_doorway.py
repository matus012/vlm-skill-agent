"""Blocked-doorway option (task 008 side lane): off-sim checks of sim/worlds/clinic/blocked.py + generate.py --blocked."""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLINIC = ROOT / "sim" / "worlds" / "clinic"
for p in (ROOT, ROOT / "probes", CLINIC):
    sys.path.insert(0, str(p))
import blocked as B  # noqa: E402
import generate as G  # noqa: E402
import raster_map as RM  # noqa: E402
import raycast_seg as R  # noqa: E402
import seeded as S  # noqa: E402

LAYOUT = CLINIC / "layout.yaml"
L = yaml.safe_load(LAYOUT.read_text())
PLACES = yaml.safe_load((CLINIC / "places.yaml").read_text())["places"]
ROOMS = ("ward", "office", "reception", "storage")


@pytest.fixture(scope="module")
def variants(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = {}
    for room in (None, *ROOMS):
        sdf, truth = G.generate(LAYOUT, room)
        d = tmp_path_factory.mktemp(room or "base")
        (d / "clinic.sdf").write_text(sdf)
        out[room] = (d / "clinic.sdf", json.loads(truth))
    return out


def reach(sdf: Path, radius_m: float) -> dict[str, bool]:
    """Flood fill from the hall place on the raster of `sdf`, occupied cells dilated by the robot radius (centre must
    stay >= radius_m from every obstacle cell edge, approximated by a disc kernel)."""
    _, occ = RM.build(sdf)
    k = int(math.ceil(radius_m / RM.RES))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    blocked = cv2.dilate(occ.astype(np.uint8), ker)
    h, w = blocked.shape
    seed = RM.cell_of(PLACES["hall"]["x"], PLACES["hall"]["y"])
    assert not blocked[seed], "hall place not free"
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(blocked.copy(), mask, (seed[1], seed[0]), 1, flags=4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
    seen = mask[1:-1, 1:-1] == 1
    return {n: bool(seen[RM.cell_of(p["x"], p["y"])]) for n, p in PLACES.items()}


@pytest.mark.parametrize("radius", [S.ROBOT_RADIUS_M, S.INFLATION_M])
def test_reachability(variants: dict, radius: float) -> None:
    assert all(reach(variants[None][0], radius).values()), reach(variants[None][0], radius)
    for room in ROOMS:
        r = reach(variants[room][0], radius)
        assert r[room] is False, (room, radius, r)
        assert all(v for n, v in r.items() if n != room), (room, radius, r)


def test_box_spec_per_room() -> None:
    for room in ROOMS:
        b = B.blocker(L, room)
        d = L["rooms"][room]["doorway"]
        assert b["label"] not in range(1, 200) and b["label"] not in G.STRUCTURE_LABELS.values()
        assert b["class"] == "blocker" and b["colour"] in G.COLOURS
        assert b["size"][0] == pytest.approx(d["width"] + B.JAMB_M) and b["size"][2] < L["door_height"]
        assert b["pose"][0] == pytest.approx(d["center"])
    with pytest.raises(ValueError):
        B.blocker(L, "hall")


def test_not_touching_phase5_zones() -> None:
    """Real footprint (seeded.footprint at seeded.pose_for's yaw) of every movable on a 0.05 m grid of every zone, clear of the box."""
    for room in ROOMS:
        b = B.blocker(L, room)
        bx0, bx1 = b["pose"][0] - b["size"][0] / 2, b["pose"][0] + b["size"][0] / 2
        by0, by1 = b["pose"][1] - b["size"][1] / 2, b["pose"][1] + b["size"][1] / 2
        box = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
        for r, classes in S.ZONES.items():
            for cls, rects in classes.items():
                for x0, y0, x1, y1 in rects:
                    for x in np.arange(x0, x1 + 1e-9, 0.05):
                        for y in np.arange(y0, y1 + 1e-9, 0.05):
                            pts = S.footprint(cls, None, x, y, S.pose_for(cls, r, x, y)[2])
                            assert S.gap(pts, box) > 0, (room, r, cls, x, y)


def box_pixels(sdf: Path, label: int, xyz: tuple, yaw: float) -> tuple[int, int, bool]:
    """(visible px, px of the box alone, touches image side)"""
    prims = R.load_prims(sdf)
    img = R.render(prims, xyz, yaw)
    alone = R.render([p for p in prims if p[0] == label], xyz, yaw)
    m = img == label
    return int(m.sum()), int((alone == label).sum()), bool(m[:, 0].any() or m[:, -1].any())


def test_visible_from_hall_place_and_approach(variants: dict) -> None:
    for room in ROOMS:
        sdf, truth = variants[room]
        b = truth["blocker"]
        best = max(box_pixels(sdf, b["label"], *S.camera("hall", k))[0] for k in range(4))
        assert best > 0, f"{room}: not in any hall photo"
        # approach: 1.2 m from the box face, in the hall, facing it
        d = L["rooms"][room]["doorway"]
        hall_side = -1.0 if d["wall"] == "south" else 1.0  # ward / office are north of the hall, reception / storage south
        x, y = b["pose"][0], b["pose"][1] + hall_side * (b["size"][1] / 2 + 1.2)  # 1.2 m from the box face
        yaw = math.pi / 2 if hall_side < 0 else -math.pi / 2
        vis, alone, edge = box_pixels(sdf, b["label"], (x, y, S.CAM_Z_M), yaw)
        assert alone > 0 and vis == alone and not edge, (room, vis, alone, edge)


def test_default_output_unchanged_and_option_adds_one_model(variants: dict) -> None:
    sdf0, truth0 = G.generate(LAYOUT)
    assert sdf0 == G.generate(LAYOUT, None)[0] == variants[None][0].read_text()
    assert (CLINIC / "generated" / "clinic.sdf").read_text() == sdf0
    assert (CLINIC / "generated" / "truth.json").read_text() == truth0
    for room in ROOMS:
        sdf, truth = variants[room][0].read_text(), variants[room][1]
        extra = [l for l in sdf.splitlines() if l not in sdf0.splitlines()]
        assert f'<model name="blocker_{room}">' in extra[0] and all("blocker" in l or l.strip().startswith(("<collision", "<visual", "</", "<static", "<pose", "<link", "<plugin")) for l in extra)
        t = dict(truth)
        t.pop("blocker")
        assert t == json.loads(truth0)
        assert G.map_knowledge(json.dumps(truth)) == G.map_knowledge(truth0)


def test_cli(tmp_path: Path) -> None:
    subprocess.run([sys.executable, str(CLINIC / "generate.py"), str(LAYOUT), "--blocked", "storage", "--out", str(tmp_path)],
                   check=True, capture_output=True)
    assert (tmp_path / "clinic.sdf").read_text() == G.generate(LAYOUT, "storage")[0]
    assert json.loads((tmp_path / "truth.json").read_text())["blocker"]["id"] == "blocker_storage"
