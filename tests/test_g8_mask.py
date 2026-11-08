"""G8 movables mask (task 008, a recorded deviation): probes/map_world_check.py sweep(cuts=...), movable_cuts, run() records.
Offline: scans simulated by raster_map.scan_endpoints on a 0.01 m slice of the world SDF. No sim."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import map_world_check as mwc  # noqa: E402
import raster_map as rm  # noqa: E402
from raycast_seg import load_prims  # noqa: E402

GRID = ((rm.ORIGIN[0], rm.ORIGIN[1]), (rm.SIZE[0] * 5, rm.SIZE[1] * 5), rm.RES / 5)
PLACES = yaml.safe_load(rm.PLACES.read_text())["places"]
FROZEN = {"chair_orange_1": [-4.075, -6.15, 0.0, 1.5708], "chair_orange_2": [-3.025, -6.15, 0.0, 1.5708],
          "chair_orange_3": [-1.975, -6.15, 0.0, 1.5708], "chair_orange_4": [-0.925, -6.15, 0.0, 1.5708],
          "toolbox_red_1": [0.4, -4.1, 0.0, 1.5708]}  # layout.yaml poses (frozen v2)
# offline stress (008 report): rng 0, trial 23, all 5 movables in storage; the one unmasked G8 failure of 80 sweeps
STRESS = {"chair_orange_1": [3.2627823773906504, -3.6999897756323796, 0.0, 0.8071304314107093],
          "chair_orange_2": [4.239833812780088, -5.4520508135345835, 0.0, -2.2002738169580005],
          "chair_orange_3": [0.8363697738753475, -5.863668307959846, 0.0, 0.21507986455674066],
          "chair_orange_4": [1.0292170685899975, -2.6484611075886444, 0.0, -2.999526500270101],
          "toolbox_red_1": [1.8659873658760784, -5.21971768367481, 0.0, -1.3708144367171982]}
# sha256 of json.dumps({place: sweep}, sort_keys=True) from the pre-008 sweep() (copy taken before the edit), 002 map,
# frozen-world scans: the unmasked path must stay byte-for-byte the same
GOLDEN_002 = "f8f50fcad9d920b8c33374f2097f41d44aafd4e7cd14b9ad52d49dc76c1a2e6e"
STATIC_CUTS = rm.slices([p for p in load_prims() if p[0] not in (6, 7, 8, 9, 13)], rm.LIDAR_Z)


def world_scan(movables: dict, place: str) -> np.ndarray:
    occ = rm.occupancy(STATIC_CUTS + mwc.movable_cuts(mwc.WORLD_DIR, movables), *GRID)
    p = PLACES[place]
    return rm.scan_endpoints(occ, GRID, p["x"], p["y"], p["yaw"])


def test_movable_cuts_at_frozen_poses_match_the_sdf() -> None:
    sdf_cuts = rm.slices([p for p in load_prims() if p[0] in (6, 7, 8, 9, 13)], rm.LIDAR_Z)
    got = mwc.movable_cuts(mwc.WORLD_DIR, FROZEN)
    assert len(got) == len(sdf_cuts) > 0
    for a, b in zip(sorted(got), sorted(sdf_cuts)):
        assert a[0] == b[0] and np.allclose(a[1:], b[1:], atol=1e-3)


def test_unmasked_sweep_unchanged_on_002_map() -> None:
    m = mwc.load_map(ROOT / "sim/maps/clinic.yaml")
    out = {n: mwc.sweep(m, world_scan(FROZEN, n)) for n in PLACES}
    assert hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest() == GOLDEN_002


def test_stress_case_fails_unmasked_passes_masked() -> None:
    m = mwc.load_map(ROOT / "sim/maps/clinic_static_raster.yaml")
    pts = world_scan(STRESS, "storage")
    plain = mwc.sweep(m, pts)
    masked = mwc.sweep(m, pts, mwc.movable_cuts(mwc.WORLD_DIR, STRESS))
    print(f"\nstress storage: unmasked {plain['hit_frac_identity']:.4f} best {plain['best_hit_frac']:.4f} at "
          f"{plain['best_offset_dx_dy_dyawdeg']}; masked {masked['n_masked']} pts, identity {masked['hit_frac_identity']:.4f}"
          f" best at {masked['best_offset_dx_dy_dyawdeg']}")
    assert not plain["pass"] and masked["pass"] and masked["n_masked"] > 0


def test_no_movables_in_view_masked_equals_unmasked() -> None:
    m = mwc.load_map(ROOT / "sim/maps/clinic_static_raster.yaml")
    pts = world_scan(FROZEN, "hall")  # frozen movables are in reception / storage, out of the hall place's sight
    masked = mwc.sweep(m, pts, mwc.movable_cuts(mwc.WORLD_DIR, FROZEN))
    assert masked.pop("n_masked") == 0 and masked == mwc.sweep(m, pts)


def test_near_cuts_distance() -> None:
    cuts = [("box", 0.0, 0.0, math.pi / 2, 0.4, 0.2), ("disc", 5.0, 0.0, 0.1)]
    pts = np.array([[0.0, 0.449], [0.0, 0.451], [0.249, 0.0], [0.251, 0.0], [5.149, 0.0], [5.151, 0.0]])
    assert mwc.near_cuts(pts, cuts, 0.05).tolist() == [True, False, True, False, True, False]


def test_default_mask_by_map_stem() -> None:
    assert mwc.default_mask(Path("sim/maps/clinic_static.yaml"))
    assert not mwc.default_mask(Path("sim/maps/clinic.yaml"))
    assert not mwc.default_mask(Path("sim/maps/clinic_static_raster.yaml"))


class FakeRobot:
    def __init__(self, pts: dict) -> None:
        self.pts, self.at = pts, None

    def navigate(self, goal: tuple) -> dict:
        self.at = next(n for n, p in PLACES.items() if (p["x"], p["y"]) == goal[:2])
        return {"status_name": "SUCCEEDED"}

    def spin_for_sim(self, s: float) -> None:
        pass

    def scan_world(self, gt: tuple) -> np.ndarray:
        return self.pts[self.at]

    def map_pose(self) -> tuple:
        raise RuntimeError("no tf")


@pytest.mark.parametrize("mask", [None, True, False])
def test_run_records_both_verdicts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mask: bool | None) -> None:
    at = ["storage", "ward", "office", "reception"]
    pts = {n: world_scan(STRESS, n) for n in at}
    robot = FakeRobot(pts)
    msg = lambda: {"header": {"stamp": {"sec": 1}}, "pose": [  # noqa: E731
        {"name": "turtlebot4", "position": {"x": PLACES[robot.at]["x"], "y": PLACES[robot.at]["y"]}},
        *({"name": k, "position": {"x": v[0], "y": v[1]}, "orientation": {"z": math.sin(v[3] / 2), "w": math.cos(v[3] / 2)}}
          for k, v in STRESS.items())]}
    monkeypatch.setattr(mwc, "pose_info", lambda world: msg())
    (tmp_path / "clinic_static.yaml").write_text((ROOT / "sim/maps/clinic_static_raster.yaml").read_text()
                                                 .replace("image: clinic_static_raster.pgm",
                                                          f"image: {ROOT / 'sim/maps/clinic_static_raster.pgm'}"))
    places = {n: (p["x"], p["y"], p["yaw"]) for n, p in PLACES.items()}
    res = mwc.run(robot, tmp_path / "clinic_static.yaml", places, at, "clinic", mask)
    assert res["masked"] is (mask is not False)  # None -> default: on for the clinic_static stem
    assert res["pass_unmasked"] is False and res["pass_masked"] is True
    assert res["pass"] is res["masked"]
    st = res["poses"]["storage"]
    assert set(st["movables_gt_xyzyaw"]) == set(STRESS)
    assert np.abs(mwc.parse_scan(st["scan_world_xy"]) - pts["storage"]).max() <= 0.0005 + 1e-9
    assert st["sweep_masked"]["n_masked"] > 0 and "n_masked" not in st["sweep"]
    json.dumps(res)  # g8.json-serialisable
