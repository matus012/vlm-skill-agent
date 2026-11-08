"""G2 nav: undock, send one NavigateToPose goal >= 2 m away, judge the end pose against Gazebo ground truth.

Run with the sim up (sim/launch_phase0.sh) and `source sim/env.sh`:
    python3 probes/nav_goal.py [--out DIR]
Exit 0 = PASS (result SUCCEEDED and |GT_end - goal| <= 0.30 m).

map <-> world transform. Hypothesis: identity. The robot spawns at world (0, 0, yaw 0) (spawn launch defaults,
read back from GT below) and sim/launch_phase0.sh seeds AMCL at map (0, 0, 0). AMCL then localises from the
laser against the stock warehouse map, so if the map frame were offset/rotated from the world, AMCL's pose would
diverge from GT once it updates. Proof: AMCL map pose vs GT world pose after undocking (start) and after the
>= 2 m drive (end) -- two points 2+ m apart pin translation and rotation. Both residuals are reported, plus a
transform estimated from the start pair (T_est = GT_start * map_start^-1) and its prediction error at the end.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from irobot_create_msgs.action import Undock
from irobot_create_msgs.msg import DockStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

WORLD, MODEL = "warehouse", "turtlebot4"
GOAL = (-2.8, 0.5, math.pi)  # map frame; free in warehouse.pgm (0.3 m around it all free), clear of the dock at +x
MIN_GOAL_DIST, GATE_M = 2.0, 0.30  # 001-phase0-pieces.md, G2
MAP_YAML = Path("/opt/ros/jazzy/share/turtlebot4_navigation/maps/warehouse.yaml")

Pose2 = tuple[float, float, float]


def yaw_of(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def gt_pose() -> tuple[Pose2, float]:
    """World-frame pose of the robot model straight from Gazebo (gz-transport, not ROS), and its sim time."""
    out = subprocess.run(["gz", "topic", "-e", "-n", "1", "-t", f"/world/{WORLD}/pose/info", "--json-output"],
                         check=True, capture_output=True, text=True, timeout=20).stdout
    msg = json.loads(out.splitlines()[0])
    p = next(p for p in msg["pose"] if p["name"] == MODEL)
    pos, q = p["position"], p["orientation"]
    st = msg["header"]["stamp"]
    return (pos.get("x", 0.0), pos.get("y", 0.0),
            yaw_of(q.get("x", 0.0), q.get("y", 0.0), q.get("z", 0.0), q.get("w", 1.0))), int(st["sec"]) + int(st.get("nsec", 0)) * 1e-9


def compose(a: Pose2, b: Pose2) -> Pose2:
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]


def inverse(a: Pose2) -> Pose2:
    c, s = math.cos(a[2]), math.sin(a[2])
    return -c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def residual(a: Pose2, b: Pose2) -> dict[str, float]:
    return {"dxy_m": math.hypot(a[0] - b[0], a[1] - b[1]), "dyaw_deg": math.degrees(abs(wrap(a[2] - b[2])))}


def scan_alignment(world_pts: np.ndarray) -> dict[str, object]:
    """Unit-free check of map == world that does not use AMCL: lidar endpoints placed with the GT pose land on
    occupied map cells. Sweep candidate map<-world offsets; identity must be the peak (a flat curve = no proof)."""
    meta = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in MAP_YAML.read_text().splitlines() if ":" in l)}
    res = float(meta["resolution"])
    ox, oy = (float(v) for v in meta["origin"].strip("[]").split(",")[:2])
    occ = cv2.dilate((cv2.imread(str(MAP_YAML.parent / meta["image"]), cv2.IMREAD_GRAYSCALE) < 50).astype(np.uint8),
                     np.ones((3, 3), np.uint8))  # occupied (black) +- 1 cell
    h, w = occ.shape

    def hits(dx: float, dy: float, dyaw: float) -> float:
        c, s = math.cos(dyaw), math.sin(dyaw)
        x = c * world_pts[:, 0] - s * world_pts[:, 1] + dx
        y = s * world_pts[:, 0] + c * world_pts[:, 1] + dy
        u = ((x - ox) / res).astype(int)
        v = h - 1 - ((y - oy) / res).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return float(occ[v[ok], u[ok]].sum() / len(world_pts))

    grid = [(dx, dy, dyaw) for dx in np.arange(-0.30, 0.301, 0.03) for dy in np.arange(-0.30, 0.301, 0.03)
            for dyaw in np.radians([-2, -1, 0, 1, 2])]
    scores = [hits(*g) for g in grid]
    best = int(np.argmax(scores))
    return {"n_points": len(world_pts), "hit_frac_identity": hits(0.0, 0.0, 0.0), "best_hit_frac": scores[best],
            "best_offset_dx_dy_dyawdeg": [round(grid[best][0], 3), round(grid[best][1], 3), round(math.degrees(grid[best][2]), 1)],
            "hit_frac_at_0.3m_shift": hits(0.3, 0.0, 0.0), "hit_frac_median_over_grid": float(np.median(scores))}


class Probe(Node):
    def __init__(self) -> None:
        super().__init__("nav_goal_probe", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.dock: DockStatus | None = None
        self.create_subscription(DockStatus, "/dock_status", lambda m: setattr(self, "dock", m), qos_profile_sensor_data)  # best-effort publisher
        self.scan: LaserScan | None = None
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), qos_profile_sensor_data)

    def scan_world_points(self, gt: Pose2) -> np.ndarray:
        """Endpoints of one fresh scan in the world frame, placed with the GT base pose (no AMCL involved)."""
        self.scan = None
        while self.scan is None:
            rclpy.spin_once(self, timeout_sec=1.0)
        sc = self.scan
        t = self.tf.lookup_transform("base_link", sc.header.frame_id, Time())
        q = t.transform.rotation
        lidar = (t.transform.translation.x, t.transform.translation.y, yaw_of(q.x, q.y, q.z, q.w))
        r = np.asarray(sc.ranges)
        a = sc.angle_min + sc.angle_increment * np.arange(len(r))
        ok = np.isfinite(r) & (r > sc.range_min) & (r < sc.range_max)
        pose = compose(gt, lidar)
        ang = pose[2] + a[ok]
        return np.stack([pose[0] + r[ok] * np.cos(ang), pose[1] + r[ok] * np.sin(ang)], axis=1)

    def wait(self, fut: rclpy.task.Future, timeout_s: float) -> object:
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout_s)
        assert fut.done(), f"timed out after {timeout_s} s"
        return fut.result()

    def spin_for(self, wall_s: float) -> None:
        end = time.monotonic() + wall_s
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def map_pose(self) -> Pose2:
        t = self.tf.lookup_transform("map", "base_link", Time())
        tr, q = t.transform.translation, t.transform.rotation
        return tr.x, tr.y, yaw_of(q.x, q.y, q.z, q.w)

    def act(self, typ: type, name: str, goal: object, timeout_s: float) -> tuple[int, object]:
        client = ActionClient(self, typ, name)
        assert client.wait_for_server(timeout_sec=30.0), f"no action server {name}"
        handle = self.wait(client.send_goal_async(goal), 30.0)
        assert handle.accepted, f"{name} goal rejected"
        res = self.wait(handle.get_result_async(), timeout_s)
        return res.status, res.result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / ".cache/vlm_skill_agent/runs/2026-11-03")
    args = ap.parse_args()
    rclpy.init()
    node = Probe()
    log: dict[str, object] = {"world": WORLD, "spawn_gt": gt_pose()[0]}

    while node.dock is None:
        rclpy.spin_once(node, timeout_sec=1.0)
    log["docked_at_start"] = node.dock.is_docked
    if node.dock.is_docked:
        status, res = node.act(Undock, "/undock", Undock.Goal(), 120.0)
        assert status == GoalStatus.STATUS_SUCCEEDED and not res.is_docked, (status, res)
    node.spin_for(3.0)  # let AMCL integrate the undock motion and TF catch up

    gt_start, t_start = gt_pose()
    map_start = node.map_pose()
    t_est = compose(gt_start, inverse(map_start))  # world <- map, estimated from the start pair
    goal_world = GOAL  # identity hypothesis
    dist = math.hypot(GOAL[0] - gt_start[0], GOAL[1] - gt_start[1])
    assert dist >= MIN_GOAL_DIST, f"goal only {dist:.2f} m from start"
    print(f"start GT {gt_start}  map {map_start}  goal {GOAL}  dist {dist:.2f} m", flush=True)

    g = NavigateToPose.Goal()
    g.pose.header.frame_id = "map"
    g.pose.pose.position.x, g.pose.pose.position.y = GOAL[0], GOAL[1]
    g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(GOAL[2] / 2), math.cos(GOAL[2] / 2)
    w0 = time.monotonic()
    status, _ = node.act(NavigateToPose, "/navigate_to_pose", g, 600.0)
    nav_wall_s = time.monotonic() - w0
    node.spin_for(2.0)  # settle
    gt_end, t_end = gt_pose()
    map_end = node.map_pose()
    scan_check = scan_alignment(node.scan_world_points(gt_end))
    assert math.dist(gt_pose()[0][:2], gt_end[:2]) < 0.01, "robot moved during scan check"
    node.destroy_node()
    rclpy.shutdown()

    error_m = math.hypot(gt_end[0] - goal_world[0], gt_end[1] - goal_world[1])
    log.update({
        "goal_map": GOAL, "goal_world": goal_world, "goal_dist_from_start_m": dist,
        "result_status": status, "result_status_name": {4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}.get(status, str(status)),
        "nav_wall_s": nav_wall_s, "nav_sim_s": t_end - t_start,
        "gt_start_world": gt_start, "gt_end_world": gt_end, "map_start": map_start, "map_end": map_end,
        "transform": {
            "method": "identity map==world (spawn at world origin, AMCL seeded at map origin); proven without AMCL by "
                      "lidar endpoints placed with the GT pose landing on occupied map cells (identity = peak of an "
                      "offset sweep), and cross-checked by AMCL-vs-GT residuals at start and end",
            "residual_start_identity": residual(gt_start, map_start),
            "residual_end_identity": residual(gt_end, map_end),
            "T_est_world_from_map": t_est,
            "residual_end_T_est": residual(gt_end, compose(t_est, map_end)),
            "scan_alignment_at_end_gt": scan_check,
        },
        "error_m": error_m,
        "error_m_with_T_est": math.hypot(*(a - b for a, b in zip(gt_end[:2], compose(t_est, GOAL)[:2]))),
        "end_yaw_err_deg": math.degrees(abs(wrap(gt_end[2] - GOAL[2]))),
        "gate_m": GATE_M,
    })
    log["pass"] = bool(status == GoalStatus.STATUS_SUCCEEDED and error_m <= GATE_M)
    print(json.dumps(log, indent=2))
    (args.out / "001-S-nav_goal.json").write_text(json.dumps(log, indent=2))
    print("G2 nav:", "PASS" if log["pass"] else "FAIL")
    return 0 if log["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
