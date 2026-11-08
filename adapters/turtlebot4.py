"""TurtleBot 4 adapter (agent/INTERFACES.md section 1): the only Stage 1 runtime code that imports ROS.

Reuses what Phase 1 proved:
- probes/photos.py `Camera` node, unchanged (absolute-heading P turn on /cmd_vel, odom settle, AMCL no-motion
  updates, frames stamped after the settle), `crop` by self_crop, `fused_labels` (depth not finite -> label 0), the
  seg bridge.
- probes/map_world_check.py `wait_result` (sim-time limit + wall hang guard) and `gt_pose` (SimTruth). `_Node.navigate`
  is map_world_check's `Robot.navigate` on the camera node: one node, so one TF buffer / clock that is spun all the
  time (a second node's TF buffer goes stale while the other one spins: 2026-11-04 smoke, ExtrapolationException).
"""
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from action_msgs.srv import CancelGoal
from nav2_msgs.action import NavigateToPose
from PIL import Image
from rclpy.action import ActionClient
from rclpy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import map_world_check as mwc  # noqa: E402
import photos  # noqa: E402

sys.path.insert(0, str(ROOT))
from agent.robot import Adapter, Frame, GoResult  # noqa: E402
from agent.world import load_rooms, room_of  # noqa: E402

ROBOT_CFG = ROOT / "config/robots/turtlebot4.yaml"
REACHED_NAV = "SUCCEEDED"


def go_status(nav_status_name: str) -> tuple[str, str]:
    """map_world_check.navigate status name -> (GoResult.status, GoResult.nav_status).
    SUCCEEDED -> reached; ABORTED / CANCELED / REJECTED / any other Nav2 status / SIM_TIMEOUT -> blocked.
    WALL_HANG (sim clock stalled) is not a navigation outcome: the caller raises."""
    assert nav_status_name != "WALL_HANG", "wall hang is a sim failure, not 'blocked'"
    nav = "TIMEOUT" if nav_status_name == "SIM_TIMEOUT" else nav_status_name
    return ("reached" if nav == REACHED_NAV else "blocked"), nav


def crop_frame(rgb: np.ndarray, seg_rgb8: np.ndarray, depth: np.ndarray, top: int, bottom: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """(cropped RGB, cropped label image) exactly as probes/photos.py saves them (label 0 where depth is not finite)."""
    lab, _ = photos.fused_labels(seg_rgb8, depth)
    return (np.ascontiguousarray(photos.crop(rgb, top, bottom)), np.ascontiguousarray(photos.crop(lab, top, bottom)))


class _Node(photos.Camera):
    """photos.Camera + the NavigateToPose client and a cancel-all client, in one node."""

    def __init__(self, frame: str, nav_action: str) -> None:
        super().__init__(frame)
        self.nav = ActionClient(self, NavigateToPose, nav_action)
        self.cancel_all = self.create_client(CancelGoal, nav_action + "/_action/cancel_goal")

    def wait(self, cond, wall_s: float) -> bool:  # noqa: ANN001  (= map_world_check Robot.spin_until)
        end = time.monotonic() + wall_s
        while not cond():
            if time.monotonic() > end:
                return False
            rclpy.spin_once(self, timeout_sec=0.05)
        return True

    def catch_up(self) -> None:
        """Process what queued while nobody spun (e.g. between skills), then wait for a map->base_link transform."""
        end = time.monotonic() + 0.2
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
        assert self.wait(lambda: self.tf.can_transform(self.frame, "base_link", Time()), 10.0), "no map TF"

    def navigate(self, goal: tuple[float, float, float], sim_limit_s: float, wall_cap_s: float) -> dict:
        """map_world_check Robot.navigate: goal in the map frame; waits for the result or `sim_limit_s` of sim time
        since acceptance (then cancels: SIM_TIMEOUT); `wall_cap_s` is only a hang guard (WALL_HANG)."""
        assert self.nav.wait_for_server(timeout_sec=60.0), "no /navigate_to_pose"
        assert self.wait(lambda: self.now_s() > 0, 30.0), "no /clock"
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = self.frame
        g.pose.pose.position.x, g.pose.pose.position.y = goal[0], goal[1]
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(goal[2] / 2), math.cos(goal[2] / 2)
        fut = self.nav.send_goal_async(g)
        assert self.wait(fut.done, 30.0), "goal send timeout"
        h = fut.result()
        if not h.accepted:
            return {"status_name": "REJECTED", "sim_s": 0.0, "wall_s": 0.0}
        res = h.get_result_async()
        how, sim_s, wall_s = mwc.wait_result(res.done, self.now_s, lambda: rclpy.spin_once(self, timeout_sec=0.05),
                                             sim_limit_s, wall_cap_s)
        if how != "done":
            cancel = h.cancel_goal_async()
            self.wait(lambda: cancel.done() and res.done(), 10.0)
            return {"status_name": how.upper(), "sim_s": sim_s, "wall_s": wall_s}
        st = res.result().status
        return {"status_name": {4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}.get(st, str(st)), "sim_s": sim_s,
                "wall_s": wall_s}


class SimTruth:
    """Evaluator side only (never reachable from agent/, so not an attribute of the adapter): Gazebo ground-truth
    robot pose, world frame == map frame."""

    def __init__(self, cfg_path: Path = ROBOT_CFG) -> None:
        self.world = yaml.safe_load(cfg_path.read_text())["adapter"]["world"]

    def gt_pose(self) -> tuple[float, float, float]:
        return mwc.gt_pose(self.world)[0]


class TB4Adapter(Adapter):
    def __init__(self, cfg_path: Path = ROBOT_CFG) -> None:
        cfg = yaml.safe_load(cfg_path.read_text())
        a = cfg["adapter"]
        t, f = a["topics"], a["frames"]
        assert (t["rgb"], t["depth"], t["seg"], f["camera"]) == (photos.RGB_T, photos.DEPTH_T, photos.SEG_T,
                                                                 photos.CAM_LINK), "config != reused Phase 1 code"
        self.top, self.bottom = cfg["self_crop"]["top_px"], cfg["self_crop"]["bottom_px"]
        self.map_frame, self.base = f["map"], f["base"]
        self.wall_cap_s, self.turn_rounds = float(a["go_to_wall_cap_s"]), int(a["turn_rounds"])
        self.rooms = load_rooms(ROOT / a["map_knowledge"])
        self.heading_target: float | None = None  # last turn_to target, cleared by any move
        self.bridge = photos.start_seg_bridge()
        rclpy.init()
        self.cam = _Node(self.map_frame, t["nav_action"])
        self.cam.spin_until(lambda: self.cam.tf.can_transform(self.map_frame, self.base, Time()), 60.0, "map TF")
        self.cam.frames_after(0.0)

    # ------------------------------------------------------------------ Adapter API
    def go_to(self, x: float, y: float, yaw: float | None, timeout_s: float) -> GoResult:
        if yaw is None:  # face the direction of travel
            px, py, _ = self.pose()
            yaw = math.atan2(y - py, x - px)
        self.heading_target = None
        r = self.cam.navigate((x, y, yaw), sim_limit_s=timeout_s, wall_cap_s=self.wall_cap_s)
        if r["status_name"] == "WALL_HANG":
            raise RuntimeError(f"sim clock stalled: no nav result after {r['wall_s']:.0f} s wall, {r['sim_s']:.1f} s sim")
        status, nav = go_status(r["status_name"])
        return GoResult(status, nav, float(r["sim_s"]), float(r["wall_s"]))

    def turn_to(self, yaw: float) -> float:
        """Absolute heading in the map frame; returns the final localization yaw error (rad, achieved - target)."""
        target = mwc.wrap(yaw)
        self.cam.catch_up()
        photos.face(self.cam, target, self.turn_rounds)  # + final turns to a residual <= 0.3 deg (006 M2 round 2)
        self.heading_target = target
        return mwc.wrap(self.cam.loc_yaw() - target)

    def photo(self) -> Frame:
        self.cam.catch_up()
        fr = self.cam.frames_after(self.cam.settle())
        rgb, lab = crop_frame(photos.to_array(fr["rgb"]), photos.to_array(fr["seg"]), photos.to_array(fr["depth"]),
                              self.top, self.bottom)
        meta = {"sim_time_s": photos.stamp(fr["rgb"]), "heading_target_rad": self.heading_target,
                "loc_pose": list(self.pose()), "crop": {"top_px": self.top, "bottom_px": self.bottom}}
        return Frame(Image.fromarray(rgb), lab, meta)

    def pose(self) -> tuple[float, float, float]:
        self.cam.catch_up()
        tr = self.cam.tf.lookup_transform(self.map_frame, self.base, Time()).transform
        q = tr.rotation
        return tr.translation.x, tr.translation.y, mwc.yaw_of(q.x, q.y, q.z, q.w)

    def where(self) -> str:
        x, y, _ = self.pose()
        return room_of(x, y, self.rooms)

    def stop(self) -> None:
        """Zero velocity and cancel every NavigateToPose goal (empty goal_info = all goals)."""
        for _ in range(5):
            self.cam.send(0.0, 0.0)
            rclpy.spin_once(self.cam, timeout_sec=0.02)
        assert self.cam.cancel_all.wait_for_service(timeout_sec=10.0), "no nav cancel service"
        fut = self.cam.cancel_all.call_async(CancelGoal.Request())
        assert self.cam.wait(fut.done, 10.0), "nav cancel timeout"
        self.heading_target = None

    def sim_time(self) -> float:
        self.cam.catch_up()
        return self.cam.now_s()

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self.cam.send(0.0, 0.0)
        self.cam.destroy_node()
        rclpy.shutdown()
        os.killpg(self.bridge.pid, signal.SIGTERM)  # ros2 run + its parameter_bridge child (as photos.py)
        self.bridge.wait(10)
