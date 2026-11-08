"""probes/raster_map.py (task 008): the movables-free raster map sim/maps/clinic_static_raster.* and its comparison with a
SLAM map. No sim. Run with -s to see the comparison numbers."""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import map_world_check as mwc  # noqa: E402
import raster_map as rm  # noqa: E402
from raycast_seg import load_prims  # noqa: E402

OUT = ROOT / "sim/maps/clinic_static_raster"
SLAM_002 = ROOT / "sim/maps/clinic.yaml"
ROBOT_RADIUS = 0.189  # turtlebot4_navigation nav2.yaml footprint: octagon, vertex radius 0.189 m
LABELS = {m.get("name"): int(m.find(".//plugin/label").text) for m in ET.parse(rm.SDF).getroot().iter("model")}


@pytest.fixture(scope="module")
def img() -> np.ndarray:
    data = OUT.with_suffix(".pgm").read_bytes()
    assert data.startswith(f"P5\n{rm.SIZE[0]} {rm.SIZE[1]}\n255\n".encode())
    return np.frombuffer(data[-rm.SIZE[0] * rm.SIZE[1]:], np.uint8).reshape(rm.SIZE[1], rm.SIZE[0])


def prims_of(names: tuple[str, ...]) -> list[tuple]:
    labs = {LABELS[n] for n in names}
    return [p for p in load_prims() if p[0] in labs]


def centres() -> tuple[np.ndarray, np.ndarray]:
    """World x, y of every cell centre, image order."""
    x = rm.ORIGIN[0] + rm.RES * (np.arange(rm.SIZE[0]) + 0.5)
    y = rm.ORIGIN[1] + rm.RES * (np.arange(rm.SIZE[1]) + 0.5)
    return np.meshgrid(x, y[::-1])


def test_regenerates_byte_identical(tmp_path: Path) -> None:
    im, _ = rm.build()
    rm.write(im, tmp_path / OUT.name, rm.SDF)
    for ext in (".pgm", ".yaml"):
        assert (tmp_path / OUT.name).with_suffix(ext).read_bytes() == OUT.with_suffix(ext).read_bytes(), ext


def test_yaml_frame_is_world() -> None:
    meta = yaml.safe_load(OUT.with_suffix(".yaml").read_text())
    assert meta["resolution"] == yaml.safe_load(SLAM_002.read_text())["resolution"] == rm.RES
    assert meta["mode"] == "trinary" and meta["origin"] == [*rm.ORIGIN, 0]
    for o in rm.ORIGIN:  # origin a whole number of cells from the world origin: cell edges on the world 0.05 grid
        assert abs(o / rm.RES - round(o / rm.RES)) < 1e-9


def test_trinary_values(img: np.ndarray) -> None:
    assert set(np.unique(img)) == {rm.OCC, rm.FREE, rm.UNKNOWN}
    assert (img[[0, -1]] == rm.UNKNOWN).all() and (img[:, [0, -1]] == rm.UNKNOWN).all()  # outside = unknown


def test_movables_absent(img: np.ndarray) -> None:
    for name in rm.MOVABLES:
        foot = rm.occupancy(rm.slices(prims_of((name,)), rm.LIDAR_Z))
        assert foot.sum() > 0, f"{name} not cut by the scan plane: test proves nothing"
        assert (img[foot] == rm.FREE).all(), name


def test_fixed_furniture_and_walls_present(img: np.ndarray) -> None:
    fixed = tuple(n for n in LABELS if n not in rm.MOVABLES)
    foot = rm.occupancy(rm.slices(prims_of(fixed), rm.LIDAR_Z))
    assert (img[foot] == rm.OCC).all() and (img == rm.OCC).sum() == foot.sum()


def test_places_free_with_robot_margin(img: np.ndarray) -> None:
    X, Y = centres()
    for name, p in yaml.safe_load(rm.PLACES.read_text())["places"].items():
        disc = np.hypot(X - p["x"], Y - p["y"]) <= ROBOT_RADIUS + rm.RES
        assert disc.sum() > 40 and (img[disc] == rm.FREE).all(), name  # free = flood-filled from the hall place


def test_doorways_open(img: np.ndarray) -> None:
    X, Y = centres()
    rooms = yaml.safe_load((ROOT / "sim/worlds/clinic/layout.yaml").read_text())["rooms"]
    for name, r in rooms.items():
        if "doorway" not in r:
            continue
        d = r["doorway"]
        wall_y = r["rect"][1] if d["wall"] == "south" else r["rect"][3]
        gap = (np.abs(X - d["center"]) < d["width"] / 2 - rm.RES / 2) & (np.abs(Y - wall_y) < 0.05 + rm.RES / 2)
        assert gap.sum() > 50 and (img[gap] == rm.FREE).all(), name


def test_against_002_slam_map() -> None:
    rep = rm.compare(SLAM_002)
    print(f"\n002 SLAM vs raster: SLAM occ {rep['slam_occupied_cells']}, raster visible {rep['raster_visible_cells']}; "
          f"SLAM->raster within 1 cell {rep['slam_with_raster_within_1']:.4f}, raster->SLAM "
          f"{rep['raster_with_slam_within_1']:.4f}\n slam_only {rep['slam_only_clusters']}\n raster_only (top 10) "
          f"{rep['raster_only_clusters'][:10]}")
    # every SLAM-occupied cell is within 1 cell of the full world's slice (static + movables at their frozen poses), so
    # the SLAM-only cells vs the static raster are the movables, and 002 did map them
    meta = yaml.safe_load(SLAM_002.read_text())
    slam = cv2.imread(str(ROOT / "sim/maps" / meta["image"]), cv2.IMREAD_GRAYSCALE)
    g = (tuple(meta["origin"][:2]), (slam.shape[1], slam.shape[0]), meta["resolution"])
    full_occ = rm.build(rm.SDF, (), *g)[0] == rm.OCC
    k = np.ones((3, 3), np.uint8)
    so = slam <= round((1 - meta["occupied_thresh"]) * 255)
    unexplained = so & ~cv2.dilate(full_occ.astype(np.uint8), k).astype(bool)
    assert unexplained.sum() == 0
    assert sum(c["cells"] for c in rep["slam_only_clusters"]) > 0


def test_002_slam_alignment_best_at_identity(tmp_path: Path) -> None:
    """Shifting the SLAM map by one cell in any direction lowers its agreement with the raster (map == world)."""
    meta = yaml.safe_load(SLAM_002.read_text())
    ident = rm.compare(SLAM_002)["slam_with_raster_within_1"]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]:
        m = dict(meta, image=str(ROOT / "sim/maps" / meta["image"]),
                 origin=[meta["origin"][0] + dx * meta["resolution"], meta["origin"][1] + dy * meta["resolution"], 0])
        (tmp_path / "s.yaml").write_text(yaml.safe_dump(m))
        assert rm.compare(tmp_path / "s.yaml")["slam_with_raster_within_1"] < ident, (dx, dy)


def test_g8_sweep_frozen_world_with_movables() -> None:
    """G8's sweep (map_world_check.sweep) on the raster with scans simulated in the frozen v2 world, movables included
    (the default world G0 runs in)."""
    grid = ((rm.ORIGIN[0], rm.ORIGIN[1]), (rm.SIZE[0] * 5, rm.SIZE[1] * 5), rm.RES / 5)
    full = rm.occupancy(rm.slices(load_prims(), rm.LIDAR_Z), *grid)
    m = mwc.load_map(OUT.with_suffix(".yaml"))
    for name, p in yaml.safe_load(rm.PLACES.read_text())["places"].items():
        s = mwc.sweep(m, rm.scan_endpoints(full, grid, p["x"], p["y"], p["yaw"]))
        print(name, round(s["hit_frac_identity"], 3), s["best_offset_dx_dy_dyawdeg"])
        assert s["pass"], (name, s)
