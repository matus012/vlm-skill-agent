#!/usr/bin/env python3
"""G8 map = world (task 002): the saved SLAM map's frame is Gazebo's world frame. Same test as 001 G2
(probes/nav_goal.py `scan_alignment`), generalised to any map and several poses.

    source sim/env.sh; python3 probes/map_world_check.py --out DIR [--map sim/maps/clinic.yaml] [--at ward,reception,...]

With the clinic sim + Nav2 running (AMCL or SLAM). For each place in --at: NavigateToPose there, stand still, read the
robot's Gazebo ground-truth pose (gz-transport, not ROS) and one fresh /scan; place the lidar endpoints in the world
with the GT pose (no AMCL involved) and count how many land on occupied map cells (+-1 cell). Sweep candidate
map<-world offsets dx, dy in +-0.30 m (half-cell steps) and dyaw in +-3 deg (0.5 deg steps).
Pass per pose = the sweep's best offset is identity within 1 cell (|dx|, |dy| <= resolution) and 1 deg (the gate text).
Reported, not gated: hit fraction at a 0.3 m shift and the grid median (a flat curve = no proof; read them).
G8 pass = every pose passes, >= 3 rooms.

Movables mask (task 008, a recorded deviation): the 008 map has no movables (toolbox_red_1, chair_orange_1..4), but the live
world does; their scan points miss the map and can tip the sweep's argmax (offline stress: 1 of 80 sweeps). Each pose
also reads the movables' GT poses (same pose/info message), slices their SDF prims (the sim's world dir, clinic.sdf
only, never the answer file) at the lidar height (probes/raster_map.py), and sweeps again without the endpoints within
1 map cell of a slice. Both sweeps are recorded; --mask-movables picks the deciding one (default: on iff the map stem is
clinic_static). The scan endpoints are saved (world xy, mm) so a G8 can be re-evaluated offline (parse_scan).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import yaml

import gz_cli
from raster_map import LIDAR_Z, MOVABLES, slices
from raycast_seg import load_prims

ROOT = Path(__file__).resolve().parents[1]
MODEL = "turtlebot4"
Pose2 = tuple[float, float, float]
SWEEP_XY = np.round(np.arange(-0.30, 0.3001, 0.025), 4)
SWEEP_YAW_DEG = np.round(np.arange(-3.0, 3.001, 0.5), 2)
MIN_ROOMS = 3  # task 002 G8 / brief: >= 3 poses in different rooms
NAV_SIM_LIMIT_S = 300.0  # positioning moves (not a gate number); G1 tries pass G1_MAX_SIM_S = 120 s sim instead
NAV_WALL_CAP_S = 800.0   # hang guard only (sim clock stalled), design review B2: >= 800 s wall
WORLD_DIR = ROOT / "sim/worlds/clinic/generated"  # default sim world dir (frozen v2, closed); only clinic.sdf is read
STATIC_MAP_STEM = "clinic_static"                  # the 008 movables-free map: mask on by default


def yaw_of(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def compose(a: Pose2, b: Pose2) -> Pose2:
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])


def load_map(map_yaml: Path) -> dict:
    """Occupancy map: occupied mask (dilated +-1 cell), resolution, origin (x, y; map yaw must be 0)."""
    meta = yaml.safe_load(map_yaml.read_text())
    assert abs(float(meta["origin"][2])) < 1e-9, "rotated map origin not supported"
    img = cv2.imread(str(map_yaml.parent / meta["image"]), cv2.IMREAD_GRAYSCALE)
    assert img is not None, map_yaml
    occ_thresh = int(round((1.0 - float(meta["occupied_thresh"])) * 255))  # trinary pgm: black = occupied
    occ = (img <= occ_thresh).astype(np.uint8)
    return {"occ": cv2.dilate(occ, np.ones((3, 3), np.uint8)), "res": float(meta["resolution"]),
            "ox": float(meta["origin"][0]), "oy": float(meta["origin"][1])}


def hit_fraction(m: dict, world_pts: np.ndarray, dx: float, dy: float, dyaw: float) -> float:
    """Share of world points that land on (dilated) occupied cells after map = R(dyaw) * world + (dx, dy)."""
    c, s = math.cos(dyaw), math.sin(dyaw)
    x = c * world_pts[:, 0] - s * world_pts[:, 1] + dx
    y = s * world_pts[:, 0] + c * world_pts[:, 1] + dy
    h, w = m["occ"].shape
    u = np.floor((x - m["ox"]) / m["res"]).astype(int)
    v = h - 1 - np.floor((y - m["oy"]) / m["res"]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return float(m["occ"][v[ok], u[ok]].sum() / len(world_pts))


def default_mask(map_yaml: Path) -> bool:
    return map_yaml.stem == STATIC_MAP_STEM


def movable_cuts(world_dir: Path, poses: dict[str, list[float]]) -> list[tuple]:
    """Lidar-height slices (raster_map.slices) of the movables' SDF prims placed at `poses` {name: [x, y, z, yaw]}."""
    sdf = world_dir / "clinic.sdf"
    names = {int(m.find(".//plugin/label").text): m.get("name") for m in ET.parse(sdf).getroot().iter("model")
             if m.find(".//plugin/label") is not None}
    prims = []
    for lab, _, _, kind, size, local in load_prims(sdf):
        if names[lab] in poses:
            x, y, z, yaw = poses[names[lab]]
            prims.append((lab, yaw, (x, y, z), kind, size, local))
    return slices(prims, LIDAR_Z)


def near_cuts(pts: np.ndarray, cuts: list[tuple], dist: float) -> np.ndarray:
    """bool per point: within `dist` of any cut (box: rotated rectangle, disc)."""
    near = np.zeros(len(pts), bool)
    for cut in cuts:
        dx, dy = pts[:, 0] - cut[1], pts[:, 1] - cut[2]
        if cut[0] == "disc":
            d = np.hypot(dx, dy) - cut[3]
        else:
            _, _, _, yaw, hx, hy = cut
            c, s = math.cos(yaw), math.sin(yaw)
            d = np.hypot(np.maximum(np.abs(c * dx + s * dy) - hx, 0), np.maximum(np.abs(-s * dx + c * dy) - hy, 0))
        near |= d <= dist
    return near


def sweep(m: dict, world_pts: np.ndarray, cuts: list[tuple] | None = None) -> dict:
    """G8 offset sweep. With `cuts` (movable_cuts), endpoints within 1 map cell of a cut are dropped first and the
    result adds n_masked; without, the result is the pre-008 one, unchanged."""
    if cuts is not None:
        drop = near_cuts(world_pts, cuts, m["res"])
        return {**sweep(m, world_pts[~drop]), "n_masked": int(drop.sum())}
    grid = [(dx, dy, dyaw) for dx in SWEEP_XY for dy in SWEEP_XY for dyaw in SWEEP_YAW_DEG]
    scores = np.array([hit_fraction(m, world_pts, dx, dy, math.radians(a)) for dx, dy, a in grid])
    best_score = scores.max()
    # ties: report the tied offset nearest identity (a plateau around identity is still identity)
    tied = [g for g, sc in zip(grid, scores) if sc >= best_score - 1e-12]
    best = min(tied, key=lambda g: (abs(g[0]) + abs(g[1]), abs(g[2])))
    ident = hit_fraction(m, world_pts, 0.0, 0.0, 0.0)
    shift = hit_fraction(m, world_pts, 0.30, 0.0, 0.0)
    ok = abs(best[0]) <= m["res"] + 1e-9 and abs(best[1]) <= m["res"] + 1e-9 and abs(best[2]) <= 1.0 + 1e-9
    return {"n_points": int(len(world_pts)), "hit_frac_identity": ident, "best_hit_frac": float(best_score),
            "best_offset_dx_dy_dyawdeg": [float(best[0]), float(best[1]), float(best[2])], "n_tied_best": len(tied),
            "tied_best_range_dx_dy_dyawdeg": [[float(min(g[i] for g in tied)), float(max(g[i] for g in tied))] for i in range(3)],
            "hit_frac_at_0.3m_shift": shift, "hit_frac_median_over_grid": float(np.median(scores)),
            "identity_above_0.3m_shift": bool(ident > shift), "pass": bool(ok)}


def scan_to_world(ranges: np.ndarray, angle_min: float, angle_inc: float, rmin: float, rmax: float,
                  lidar_in_world: Pose2) -> np.ndarray:
    a = angle_min + angle_inc * np.arange(len(ranges))
    ok = np.isfinite(ranges) & (ranges > rmin) & (ranges < rmax)
    ang = lidar_in_world[2] + a[ok]
    return np.stack([lidar_in_world[0] + ranges[ok] * np.cos(ang), lidar_in_world[1] + ranges[ok] * np.sin(ang)], 1)


# ---------------------------------------------------------------- live (ROS + Gazebo) helpers, shared by phase1_gates.py
def wait_result(done: Callable[[], bool], sim_now: Callable[[], float], spin: Callable[[], None], sim_limit_s: float,
                wall_cap_s: float, wall_now: Callable[[], float] = time.monotonic) -> tuple[str, float, float]:
    """Spin until done() -> 'done', or sim time since the call exceeds sim_limit_s -> 'sim_timeout', or wall time
    exceeds wall_cap_s -> 'wall_hang' (the hang guard: sim clock stalled). Returns (how, sim_s, wall_s)."""
    s0, w0 = sim_now(), wall_now()
    while not done():
        if sim_now() - s0 > sim_limit_s:
            return "sim_timeout", sim_now() - s0, wall_now() - w0
        if wall_now() - w0 > wall_cap_s:
            return "wall_hang", sim_now() - s0, wall_now() - w0
        spin()
    return "done", sim_now() - s0, wall_now() - w0


def pose_info(world: str) -> dict:
    out = gz_cli.echo_once(f"/world/{world}/pose/info", 20)  # signal deaths retried (a recorded deviation)
    return json.loads(out.splitlines()[0])


def model_pose(p: dict) -> list[float]:
    """[x, y, z, yaw] of one pose/info entry (proto JSON omits zero fields)."""
    pos, q = p.get("position", {}), p.get("orientation", {})
    return [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0),
            yaw_of(*(q.get(k, 0.0) for k in "xyz"), q.get("w", 1.0))]


def robot_pose(msg: dict) -> tuple[Pose2, float]:
    """Robot model world pose from a pose/info message, and its sim time."""
    x, y, _, yaw = model_pose(next(p for p in msg["pose"] if p.get("name") == MODEL))
    st = msg["header"]["stamp"]
    return (x, y, yaw), int(st.get("sec", 0)) + int(st.get("nsec", 0)) * 1e-9


def gt_movables(msg: dict) -> dict[str, list[float]]:
    return {p["name"]: model_pose(p) for p in msg["pose"] if p.get("name") in MOVABLES}


def gt_pose(world: str) -> tuple[Pose2, float]:
    """Robot model world pose from Gazebo pose/info, and sim time."""
    return robot_pose(pose_info(world))


def scan_text(pts: np.ndarray) -> str:
    """Scan endpoints as one compact string "x,y x,y ..." (world m, 1 mm)."""
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in pts)


def parse_scan(text: str) -> np.ndarray:
    return np.array([[float(v) for v in xy.split(",")] for xy in text.split()]).reshape(-1, 2)


def load_places(path: Path) -> dict[str, Pose2]:
    return {k: (float(v["x"]), float(v["y"]), float(v["yaw"])) for k, v in yaml.safe_load(path.read_text())["places"].items()}


def make_robot():  # noqa: ANN201  (ROS imports only when live)
    import rclpy
    from action_msgs.msg import GoalStatus
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformListener

    class Robot(Node):
        SUCCEEDED = GoalStatus.STATUS_SUCCEEDED

        def __init__(self) -> None:
            super().__init__("map115_probe", parameter_overrides=[Parameter("use_sim_time", value=True)])
            self.tf = Buffer()
            self.tf_listener = TransformListener(self.tf, self)
            self.scan: LaserScan | None = None
            self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), qos_profile_sensor_data)
            self.nav = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        def now_s(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def spin_for_sim(self, sim_s: float, wall_cap_s: float = 60.0) -> None:
            self.spin_until(lambda: self.now_s() > 0, 30.0)
            t0, end = self.now_s(), time.monotonic() + wall_cap_s
            while self.now_s() - t0 < sim_s and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)

        def spin_until(self, cond, wall_s: float) -> bool:  # noqa: ANN001
            end = time.monotonic() + wall_s
            while not cond():
                if time.monotonic() > end:
                    return False
                rclpy.spin_once(self, timeout_sec=0.05)
            return True

        def map_pose(self) -> Pose2:
            t = self.tf.lookup_transform("map", "base_link", Time())
            tr, q = t.transform.translation, t.transform.rotation
            return tr.x, tr.y, yaw_of(q.x, q.y, q.z, q.w)

        def navigate(self, goal: Pose2, sim_limit_s: float = NAV_SIM_LIMIT_S, wall_cap_s: float = NAV_WALL_CAP_S) -> dict:
            """NavigateToPose in the map frame. Waits for the result or until `sim_limit_s` of sim time since the goal
            was accepted (then cancels: SIM_TIMEOUT); `wall_cap_s` is only a hang guard (WALL_HANG). Records sim_s,
            wall_s and rtf = sim_s / wall_s; a gate verdict uses sim_s only."""
            assert self.nav.wait_for_server(timeout_sec=60.0), "no /navigate_to_pose"
            assert self.spin_until(lambda: self.now_s() > 0, 30.0), "no /clock"  # else sim_s counts from t = 0
            g = NavigateToPose.Goal()
            g.pose.header.frame_id = "map"
            g.pose.pose.position.x, g.pose.pose.position.y = goal[0], goal[1]
            g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(goal[2] / 2), math.cos(goal[2] / 2)
            fut = self.nav.send_goal_async(g)
            assert self.spin_until(fut.done, 30.0), "goal send timeout"
            h = fut.result()
            if not h.accepted:
                return {"status": -1, "status_name": "REJECTED", "sim_s": 0.0, "wall_s": 0.0, "rtf": None,
                        "sim_limit_s": sim_limit_s}
            res = h.get_result_async()
            how, sim_s, wall_s = wait_result(res.done, self.now_s, lambda: rclpy.spin_once(self, timeout_sec=0.05),
                                             sim_limit_s, wall_cap_s)
            rec = {"sim_s": sim_s, "wall_s": wall_s, "rtf": sim_s / wall_s if wall_s > 0 else None,
                   "sim_limit_s": sim_limit_s}
            if how != "done":
                cancel = h.cancel_goal_async()
                self.spin_until(lambda: cancel.done() and res.done(), 10.0)  # brief wait for the cancel to land
                return {"status": -2 if how == "sim_timeout" else -3, "status_name": how.upper(),
                        "cancel_acked": cancel.done(), "result_after_cancel": res.result().status if res.done() else None,
                        **rec}
            st = res.result().status
            return {"status": st, "status_name": {4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}.get(st, str(st)), **rec}

        def scan_world(self, gt: Pose2) -> np.ndarray:
            self.scan = None
            assert self.spin_until(lambda: self.scan is not None, 20.0), "no /scan"
            sc = self.scan
            t = self.tf.lookup_transform("base_link", sc.header.frame_id, Time())
            q = t.transform.rotation
            lidar = (t.transform.translation.x, t.transform.translation.y, yaw_of(q.x, q.y, q.z, q.w))
            return scan_to_world(np.asarray(sc.ranges), sc.angle_min, sc.angle_increment, sc.range_min, sc.range_max,
                                 compose(gt, lidar))

    rclpy.init()
    return Robot()


def check_here(robot, m: dict, world: str, world_dir: Path = WORLD_DIR) -> dict:  # noqa: ANN001
    """Scan alignment at the robot's current (still) pose, unmasked and with the movables masked."""
    robot.spin_for_sim(2.0)
    msg = pose_info(world)
    gt0, t0 = robot_pose(msg)
    movables = gt_movables(msg)
    pts = robot.scan_world(gt0)
    gt1, _ = gt_pose(world)
    moved = math.dist(gt0[:2], gt1[:2])
    assert moved < 0.01, f"robot moved {moved:.3f} m during the scan"
    try:
        loc = robot.map_pose()
        loc_res = {"map_pose": loc, "dxy_m": math.dist(loc[:2], gt0[:2]), "dyaw_deg": math.degrees(abs(wrap(loc[2] - gt0[2])))}
    except Exception as e:  # noqa: BLE001  (record only: no map->base_link TF is not a G8 failure)
        loc_res = {"error": str(e)}
    return {"gt_pose_world": gt0, "gt_sim_s": t0, "loc_vs_gt": loc_res, "movables_gt_xyzyaw": movables,
            "scan_world_xy": scan_text(pts), "sweep": sweep(m, pts),
            "sweep_masked": sweep(m, pts, movable_cuts(world_dir, movables))}


def run(robot, map_yaml: Path, places: dict[str, Pose2], at: list[str], world: str,  # noqa: ANN001
        mask_movables: bool | None = None, world_dir: Path = WORLD_DIR) -> dict:
    """G8 at the places `at`; `mask_movables` (None: default_mask(map_yaml)) picks the deciding sweep."""
    mask = default_mask(map_yaml) if mask_movables is None else mask_movables
    key = "sweep_masked" if mask else "sweep"
    m = load_map(map_yaml)
    poses = {}
    for name in at:
        nav = robot.navigate(places[name])
        poses[name] = {"place_map": places[name], "nav": nav, **check_here(robot, m, world, world_dir)}
        s = poses[name][key]
        print(f"G8 {name}: nav {nav['status_name']}  identity hits {s['hit_frac_identity']:.3f}  best {s['best_hit_frac']:.3f}"
              f" at {s['best_offset_dx_dy_dyawdeg']}  0.3 m shift {s['hit_frac_at_0.3m_shift']:.3f}  pass {s['pass']}"
              f"  masked {mask} ({poses[name]['sweep_masked']['n_masked']} pts near movables)", flush=True)
    rooms = [n for n in at if n != "hall"]
    verdict = {k: all(p[k]["pass"] for p in poses.values()) and len(rooms) >= MIN_ROOMS for k in ("sweep", "sweep_masked")}
    ok = verdict[key]
    return {"gate": "G8", "map": str(map_yaml), "resolution_m": m["res"],
            "method": "lidar endpoints placed with the Gazebo GT pose vs occupied map cells (+-1 cell), offset sweep "
                      f"dx,dy {SWEEP_XY[0]}..{SWEEP_XY[-1]} m step 0.025, dyaw {SWEEP_YAW_DEG[0]}..{SWEEP_YAW_DEG[-1]} deg "
                      "step 0.5; pass = best at identity within 1 cell / 1 deg",
            "transform_map_from_world": "identity" if ok else "NOT identity (see per-pose best offsets)",
            "rooms_checked": rooms, "poses": poses, "world_dir": str(world_dir), "masked": mask,
            "pass_unmasked": bool(verdict["sweep"]), "pass_masked": bool(verdict["sweep_masked"]), "pass": bool(ok)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--map", type=Path, default=ROOT / "sim/maps/clinic.yaml")
    ap.add_argument("--places", type=Path, default=ROOT / "sim/worlds/clinic/places.yaml")
    ap.add_argument("--at", default="ward,office,reception,storage,hall")
    ap.add_argument("--world", default="clinic")
    ap.add_argument("--world-dir", type=Path, default=WORLD_DIR, help="the sim's world dir (its clinic.sdf: movables)")
    ap.add_argument("--mask-movables", action=argparse.BooleanOptionalAction, default=None,
                    help=f"default: on iff the map stem is {STATIC_MAP_STEM}")
    a = ap.parse_args()
    robot = make_robot()
    res = run(robot, a.map, load_places(a.places), a.at.split(","), a.world, a.mask_movables, a.world_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "g8_map_world.json").write_text(json.dumps(res, indent=2))
    print("G8:", "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
