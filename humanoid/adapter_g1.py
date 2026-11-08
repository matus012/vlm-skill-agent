"""G1 adapter draft (side lane H, task 003): the Stage 1 adapter API (agent/INTERFACES.md section 1) on the MuJoCo G1.

Talks to a running `g1_node.py --clinic layout.yaml` over ROS (domain 18, run through humanoid/run_g1.sh) and drives
through humanoid/drive.py. Differences from the TB4 adapter, by design of this lane:
- No Nav2 and no map: `go_to` walks straight legs through doorway centres from map_knowledge.json (geometry.route()).
  No answers on the robot side (005 N8): rooms + doorways only, never the evaluator's object file.
  Each non-hall room has one doorway into the hall (layout.yaml), so a route is at most: the doorway's inside point,
  its hall-side point, the target room's hall-side point, its inside point, the goal. Doorway points sit geometry.DOOR_STEP_M
  off the door line, so the robot crosses every door square to the wall.
- No localization: `pose()` is the sim's ground-truth pelvis pose (/g1/odom), so pose() == ground truth.
- `self_crop` = 0 / 0: the level head camera never sees the G1's own body (Phase 9 self-view probe, 0 rows).
- `photo().seg` is the truth label image: the node's per-pixel MuJoCo body id -> scene.json `label_of_body`
  (objects 1..199 = the generator's object label, structure 240..243, robot / background 0), written by the MJCF
  writer at build time; the adapter reads only scene.json at runtime.
Phase 10 (012 design section 4): scene json + map are constructor args (one scene per world); the whole go_to (legs,
final turn, recentres) stays inside timeout_s, else blocked / TIMEOUT; a stalled leg -> blocked / STALLED; yaw None =
face the direction of travel; after every turn the robot walks back to its anchor if it drifted (recentre: the goal for
go_to's own final turn, else where the first standalone turn_to since the last go_to started, so a look_around ends
where it began: hand help's SLACK_M); every call is logged in `calls` (drift before / after, recentres). Drive raises RobotFell on a fall. Knobs: config/robots/g1.yaml `adapter`.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from PIL import Image

from agent.robot import Adapter, Frame, GoResult
from agent.world import load_doors, load_rooms, room_of
from humanoid.drive import AVOID_HALF_W_M, AVOID_LOOK_M, CFG, ODOM_HANG_S, SETTLE_S, STOP_TOL_M, Drive
from humanoid.geometry import HALL, blocked, route, scan_points, wrap
from humanoid.probes.common import FrameTap, image_to_array, stamp_s

A = CFG["adapter"]
RECENTRE_M = A["recentre_m"]
RECENTRE_STOP_M = A["recentre_stop_m"]
RECENTRE_ROUNDS = A["recentre_rounds"]
RECENTRE_OUT_M = A["recentre_out_m"]
RECENTRE_MIN_V = A["recentre_min_v"]
TURN_TIMEOUT_S = A["turn_timeout_s"]
FRAME_WAIT_S = A["frame_wait_s"]
log = logging.getLogger("g1_adapter")


class G1Adapter(Adapter):
    def __init__(self, scene_json: Path, map_path: Path) -> None:
        meta = json.loads(Path(scene_json).read_text())
        self.lut = np.asarray(meta["label_of_body"], np.uint8)
        self.rooms, self.doors = load_rooms(Path(map_path)), load_doors(Path(map_path))
        assert set(self.doors) == set(self.rooms) - {HALL}, "route() assumes one doorway per room into the hall"
        rclpy.init()
        self.drive = Drive("g1_adapter")
        self.tap = FrameTap(self.drive)
        self.drive.wait_pose()
        w0 = time.monotonic()
        while self.drive.scan is None:  # avoidance input (a recorded deviation): never walk blind
            assert time.monotonic() - w0 < ODOM_HANG_S, f"no /g1/scan within {ODOM_HANG_S:.0f} s wall"
            rclpy.spin_once(self.drive, timeout_sec=0.1)
        self.heading_target: float | None = None    # last turn_to target, cleared by any move
        self.settled = False                        # True right after a turn (it ends with a SETTLE_S hold)
        self.anchor: tuple[float, float] | None = None  # recentre point: the goal inside go_to; else the xy at the
        #                                                 first standalone turn_to since the last go_to
        self.calls: list[dict] = []                 # one entry per go_to / turn_to (legs, drift, recentres); log only

    # ------------------------------------------------------------------ Adapter API
    def go_to(self, x: float, y: float, yaw: float | None, timeout_s: float) -> GoResult:
        """Straight legs through doorway centres (route()), then the final turn (+ recentres), all within timeout_s
        sim: a leg that times out or a whole call over timeout_s -> blocked / TIMEOUT, a stalled leg -> blocked /
        STALLED. yaw None = face the direction of travel (as the TB4 adapter)."""
        self.heading_target, self.settled, self.anchor = None, False, None
        self.drive.sync()  # the pose may predate a layout reset (G1Scene.place): route from where the robot is now
        t0, w0 = self.sim_time(), time.monotonic()
        deadline = t0 + timeout_s
        p = self.drive.pose
        if yaw is None:
            yaw = math.atan2(y - p.y, x - p.x)
        call: dict = {"call": "go_to", "goal": [x, y, yaw], "legs": []}
        self.calls.append(call)

        def done(status: str, nav: str) -> GoResult:
            if status != "reached":  # stand still before returning: the caller's GT read is the rest pose (a stalled
                self.drive.hold(SETTLE_S)  # walk coasted 0.16 m after the return in p8d11, counted as hand help)
            r = GoResult(status, nav, self.sim_time() - t0, time.monotonic() - w0)
            call.update(status=status, nav_status=nav, sim_s=r.sim_s)
            log.info(json.dumps(call))
            return r

        for wx, wy in route(p.x, p.y, x, y, self.rooms, self.doors):
            leg = self.drive.goto(wx, wy, timeout_s=max(0.0, deadline - self.sim_time()), gate_m=STOP_TOL_M)
            call["legs"].append(leg)
            if leg["stalled"]:
                return done("blocked", "STALLED")
            if leg["timed_out"]:
                return done("blocked", "TIMEOUT")
        if self.sim_time() > deadline:
            return done("blocked", "TIMEOUT")
        self.anchor = (x, y)  # go_to's own final turn recentres to the goal (driving is allowed in this step)
        call["turn"] = self._turn(wrap(yaw), deadline)
        self.anchor = None    # the next standalone turn_to anchors where it starts (go_to may stop STOP_TOL_M short)
        if self.sim_time() > deadline:
            return done("blocked", "TIMEOUT")
        return done("reached", "SUCCEEDED")

    def turn_to(self, yaw: float) -> float:
        """Absolute heading (+ recentres); returns the settled error (rad, achieved - target)."""
        target = wrap(yaw)
        self.drive.sync()
        call = {"call": "turn_to", **self._turn(target, float("inf"))}
        self.calls.append(call)
        log.info(json.dumps(call))
        self.heading_target = target
        return wrap(self.drive.pose.yaw - target)

    def _turn(self, target: float, deadline: float) -> dict:
        """Drive.turn_to, then while |pose - anchor| > RECENTRE_M (at most RECENTRE_ROUNDS times): out-and-back to
        the anchor (RECENTRE_OUT_M ahead of it along the target heading, skipped when the scan is blocked ahead, then
        to the anchor with stop RECENTRE_STOP_M and speed >= RECENTRE_MIN_V: the policy ignores small moves, g1.yaml)
        and turn again. Each Drive call gets min(TURN_TIMEOUT_S, time to deadline)."""
        def budget() -> float:
            return max(0.0, min(TURN_TIMEOUT_S, deadline - self.sim_time()))

        if self.anchor is None:
            self.anchor = (self.drive.pose.x, self.drive.pose.y)
        ax, ay = self.anchor
        out: dict = {"anchor": [ax, ay], "turn": self.drive.turn_to(target, timeout_s=budget()), "recentres": []}
        out["drift_before_m"] = math.hypot(self.drive.pose.x - ax, self.drive.pose.y - ay)
        for _ in range(RECENTRE_ROUNDS):
            drift = math.hypot(self.drive.pose.x - ax, self.drive.pose.y - ay)
            if drift <= RECENTRE_M:
                break
            ox, oy = ax + RECENTRE_OUT_M * math.cos(target), ay + RECENTRE_OUT_M * math.sin(target)
            out_leg = None  # no out leg into an obstacle the scan sees within AVOID_LOOK_M ahead: only the way back
            if not self._blocked_ahead():
                out_leg = self.drive.goto(ox, oy, timeout_s=budget(), gate_m=STOP_TOL_M)
            leg = self.drive.goto(ax, ay, timeout_s=budget(), gate_m=RECENTRE_STOP_M, stop_m=RECENTRE_STOP_M,
                                  min_v=RECENTRE_MIN_V)
            after_leg = math.hypot(self.drive.pose.x - ax, self.drive.pose.y - ay)
            out["recentres"].append({"drift_m": drift, "out_leg": out_leg, "leg": leg, "after_leg_m": after_leg,
                                     "turn": self.drive.turn_to(target, timeout_s=budget()),
                                     "after_turn_m": math.hypot(self.drive.pose.x - ax, self.drive.pose.y - ay)})
        out["drift_m"] = math.hypot(self.drive.pose.x - ax, self.drive.pose.y - ay)  # after the recentres
        self.settled = True
        return out

    def _blocked_ahead(self) -> bool:
        """Does /g1/scan see anything in the drive's avoidance corridor (half-width AVOID_HALF_W_M, AVOID_LOOK_M
        long) straight ahead of the pelvis?"""
        sc = self.drive.scan
        pts = scan_points(sc.ranges, sc.angle_min, sc.angle_increment, sc.range_min, sc.range_max)
        return bool(blocked(pts, np.zeros(1), AVOID_LOOK_M, AVOID_HALF_W_M)[0])

    def photo(self) -> Frame:
        """First rgb + seg pair (same stamp) after the robot settled (SETTLE_S at zero command unless a turn just
        did that)."""
        if not self.settled:
            self.drive.hold(SETTLE_S)
            self.settled = True
        t_settled, w0 = self.sim_time(), time.monotonic()
        while True:
            rgb, seg = self.tap.rgb, self.tap.seg
            if rgb and seg and stamp_s(rgb) == stamp_s(seg) and stamp_s(rgb) >= t_settled:
                break
            if time.monotonic() - w0 > FRAME_WAIT_S:
                raise RuntimeError(f"no rgb + seg pair after sim t {t_settled:.2f} s within {FRAME_WAIT_S} s wall")
            rclpy.spin_once(self.drive, timeout_sec=0.1)
        meta = {"sim_time_s": stamp_s(rgb), "heading_target_rad": self.heading_target, "loc_pose": list(self.pose()),
                "crop": {"top_px": 0, "bottom_px": 0}}
        return Frame(Image.fromarray(image_to_array(rgb)), self.lut[image_to_array(seg)], meta)

    def pose(self) -> tuple[float, float, float]:
        p = self.drive.pose
        return p.x, p.y, p.yaw

    def where(self) -> str:
        x, y, _ = self.pose()
        return room_of(x, y, self.rooms)

    def stop(self) -> None:
        for _ in range(5):
            self.drive.send(0.0, 0.0, 0.0)
            rclpy.spin_once(self.drive, timeout_sec=0.02)
        self.heading_target, self.settled = None, False

    def sim_time(self) -> float:
        rclpy.spin_once(self.drive, timeout_sec=0.0)
        return self.drive.pose.t

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Idempotent: zero command, destroy the node, shut the default context down (last, after any other node on
        it is destroyed)."""
        if self.drive is None:
            return
        self.drive.send(0.0, 0.0, 0.0)
        self.drive.destroy_node()
        self.drive = None
        rclpy.shutdown()  # also uninstalls the signal handlers rclpy.init() installed (try_shutdown does not)
