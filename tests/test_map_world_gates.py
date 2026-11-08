"""Pure helpers of probes/map_world_check.py (G8 sweep) and probes/phase1_gates.py (G1 plan, G9 summary). No sim."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
import map_world_check as mwc  # noqa: E402

pg = pytest.importorskip("phase1_gates")  # imports map_world_check + phase1_checks only (no ROS at import)


def room_map(tmp_path: Path, shift_cells: int = 0) -> Path:
    """4 x 3 m room, walls at x = 0 / 4 and y = 0 / 3 (world), drawn `shift_cells` cells to +x in the map. 0.05 m cells,
    origin (-1, -1)."""
    res, ox, oy, w, h = 0.05, -1.0, -1.0, 120, 100
    img = np.full((h, w), 254, np.uint8)

    def cell(x: float, y: float) -> tuple[int, int]:
        return int(math.floor((x - ox) / res)) + shift_cells, h - 1 - int(math.floor((y - oy) / res))

    for a, b in [((0, 0), (4, 0)), ((4, 0), (4, 3)), ((4, 3), (0, 3)), ((0, 3), (0, 0))]:
        cv2.line(img, cell(*a), cell(*b), 0, 1)
    cv2.imwrite(str(tmp_path / "m.pgm"), img)
    (tmp_path / "m.yaml").write_text("image: m.pgm\nmode: trinary\nresolution: 0.05\norigin: [-1.0, -1.0, 0]\n"
                                     "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    return tmp_path / "m.yaml"


def wall_points() -> np.ndarray:
    """What a lidar at (1.3, 1.1) would hit: points along the 4 walls (world frame)."""
    t = np.linspace(0.02, 0.98, 60)
    return np.concatenate([np.stack([4 * t, 0 * t], 1), np.stack([4 + 0 * t, 3 * t], 1),
                           np.stack([4 * t, 3 + 0 * t], 1), np.stack([0 * t, 3 * t], 1)]) + 0.01


def test_sweep_identity_passes(tmp_path: Path) -> None:
    s = mwc.sweep(mwc.load_map(room_map(tmp_path)), wall_points())
    assert s["pass"] and s["hit_frac_identity"] == 1.0
    assert s["hit_frac_at_0.3m_shift"] < 0.6


def test_sweep_detects_offset_map(tmp_path: Path) -> None:
    s = mwc.sweep(mwc.load_map(room_map(tmp_path, shift_cells=4)), wall_points())  # map is world + 0.20 m in x
    assert not s["pass"]
    assert 0.15 <= s["best_offset_dx_dy_dyawdeg"][0] <= 0.25  # +-1 cell dilation plateau, tie -> nearest identity


def test_scan_to_world_places_endpoints() -> None:
    r = np.array([1.0, 2.0, np.inf, 0.01])
    pts = mwc.scan_to_world(r, 0.0, math.pi / 2, 0.1, 10.0, (1.0, 1.0, math.pi / 2))
    assert pts == pytest.approx(np.array([[1.0, 2.0], [-1.0, 1.0]]))


def test_g1_starts_cycle_hall_first() -> None:
    assert pg.g1_starts("ward") == ["hall", "office", "reception", "storage"] * 2 + ["hall", "office"]


def test_g1_plan_counts_every_try_once() -> None:
    need = Counter((s, r) for r in pg.ROOMS for s in pg.g1_starts(r))
    want = need.copy()
    cur, counted, moves = "hall", Counter(), 0
    while sum(need.values()):
        tgt, c = pg.g1_plan_step(need, cur)
        assert tgt != cur
        if c:
            need[(cur, tgt)] -= 1
            counted[(cur, tgt)] += 1
        moves += 1
        cur = tgt
        assert moves < 100
    assert counted == want and sum(counted.values()) == 40


def test_g9_summary_counts_class_pixels(tmp_path: Path) -> None:
    truth = {"objects": [{"class": "chair", "label": 5}, {"class": "chair", "label": 6}, {"class": "bed", "label": 7}]}
    d = tmp_path / "p"
    d.mkdir()
    photos = []
    for k, (n5, n6, n7) in enumerate([(600, 500, 0), (999, 0, 2000), (1000, 0, 0), (0, 0, 0)]):
        img = np.zeros(4000, np.uint8)
        img[:n5] = 5
        img[n5:n5 + n6] = 6
        img[n5 + n6:n5 + n6 + n7] = 7
        cv2.imwrite(str(d / f"{k}_seg.png"), img.reshape(40, 100))
        photos.append({"k": k, "rgb": f"{k}_rgb.png", "seg": f"{k}_seg.png"})
    (d / "photos.json").write_text(json.dumps({"place": "p", "photos": photos}))
    s = pg.g9_summary(tmp_path, truth)
    assert s["chair"]["n_photos_ge_min"] == 2 and s["chair"]["best"] == [("p/0_rgb.png", 1100), ("p/2_rgb.png", 1000)]
    assert not s["chair"]["enough"] and s["bed"]["n_photos_ge_min"] == 1
