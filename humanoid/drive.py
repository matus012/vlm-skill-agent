"""Waypoint + absolute-heading controller for the G1, on top of /cmd_vel, using the sim's exact pose (/g1/odom).

Library (`Drive`) for probes / the Phase 10 adapter, plus a CLI:
  humanoid/run_g1.sh humanoid/drive.py goto 2.0 1.0
  humanoid/run_g1.sh humanoid/drive.py heading 90
Time is sim time from the odom stamps, so timeouts mean sim seconds whatever the real-time factor.
Phase 10 (012 design section 4, a recorded deviation): `goto` stops on the stall rule and detours around what /g1/scan sees
(geometry.avoid_heading; scan only, never the layout); every odom sample below FALL_Z_M raises RobotFell.
Knobs: config/robots/g1.yaml `drive`.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS
from sensor_msgs.msg import LaserScan

from humanoid.geometry import (avoid_heading, fine_pulse, heading_cmd, scan_points, waypoint_cmd, wrap, xyzw_to_wxyz,
                               yaw_from_wxyz)

G1_CFG = Path(__file__).resolve().parents[1] / "config/robots/g1.yaml"
CFG = yaml.safe_load(G1_CFG.read_text())

STOP_TOL_M = 0.15      # controller's own arrival tolerance (tighter than the H1 gate's 0.30 m)
YAW_TOL_DEG = 1.0      # controller's own heading tolerance (tighter than the H2 gate's 3 deg)
SETTLE_S = 2.0         # zero command this long before a heading is measured
COARSE_DEG = 10.0      # above this error: continuous step-turn (until COARSE_STOP_DEG); below: correction pulses
COARSE_STOP_DEG = 3.0  # end of a coarse turn; the post-stop drift (~+-4 deg) is left to the pulses
ODOM_HANG_S = 30.0     # wall seconds without a /g1/odom sample -> the sim is dead or hung: raise (hang guard)
MIN_WZ = 0.6           # policy turns by stepping only above ~0.5 rad/s; below it twists and springs back (README)
STALL_M = CFG["drive"]["stall_m"]    # stall: closest distance not improved by STALL_M ...
STALL_S = CFG["drive"]["stall_s"]    # ... within STALL_S sim seconds
FALL_Z_M = CFG["drive"]["fall_z_m"]  # base z below this = fallen
AVOID_LOOK_M = CFG["drive"]["avoid"]["look_m"]
AVOID_HALF_W_M = CFG["drive"]["avoid"]["half_width_m"]


class RobotFell(RuntimeError):
    """Odom base z < FALL_Z_M. Raised out of the adapter, so the runtime records crash / origin adapter (label sim)."""


@dataclass
class Pose:
    t: float = -1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    q_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    min_z: float = float("inf")
    history: list[tuple[float, float, float, float, float]] = field(default_factory=list)  # t x y z yaw


class Drive(Node):
    def __init__(self, name: str = "g1_drive") -> None:
        super().__init__(name)
        self.pose = Pose()
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan: LaserScan | None = None  # latest /g1/scan (avoidance input)
        self.create_subscription(Odometry, "/g1/odom", self._on_odom, SENSOR_QOS)
        self.create_subscription(LaserScan, "/g1/scan", lambda m: setattr(self, "scan", m), SENSOR_QOS)

    def _on_odom(self, m: Odometry) -> None:
        p, o = m.pose.pose.position, m.pose.pose.orientation
        q = xyzw_to_wxyz((o.x, o.y, o.z, o.w))
        ps = self.pose
        ps.t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        ps.x, ps.y, ps.z, ps.q_wxyz, ps.yaw = p.x, p.y, p.z, q, yaw_from_wxyz(q)
        ps.min_z = min(ps.min_z, p.z)
        ps.history.append((ps.t, p.x, p.y, p.z, ps.yaw))

    def wait_pose(self) -> None:
        while self.pose.t < 0:
            rclpy.spin_once(self, timeout_sec=1.0)

    def sync(self) -> None:
        """Process queued callbacks until a new /g1/odom sample and a new /g1/scan have arrived. Nothing spins this node
        while G1Scene.place resets the robot (its own node), so until then pose and scan are the pre-reset ones (p8d11:
        go_to routed from the previous run's storage pose and walked from the hall spawn into reception)."""
        t0, scan0, w0 = self.pose.t, self.scan, time.monotonic()
        while self.pose.t == t0 or self.scan is scan0:
            rclpy.spin_once(self, timeout_sec=1.0)
            if time.monotonic() - w0 > ODOM_HANG_S:
                raise RuntimeError(f"no new /g1/odom + /g1/scan for {ODOM_HANG_S:.0f} s wall (sim t {t0:.2f} s)")

    def send(self, vx: float, vy: float, wz: float) -> None:
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
        self.pub.publish(t)

    def _tick(self) -> float:
        """Block until the next odom sample; return its sim time."""
        t0, w0 = self.pose.t, time.monotonic()
        while self.pose.t == t0:
            rclpy.spin_once(self, timeout_sec=1.0)
            if time.monotonic() - w0 > ODOM_HANG_S:
                raise RuntimeError(f"no /g1/odom for {ODOM_HANG_S:.0f} s wall (sim t {t0:.2f} s)")
        if self.pose.z < FALL_Z_M:
            self.send(0.0, 0.0, 0.0)
            raise RobotFell(f"base z {self.pose.z:.3f} m < {FALL_Z_M} m at sim t {self.pose.t:.2f} s "
                            f"({self.pose.x:.2f}, {self.pose.y:.2f})")
        return self.pose.t

    def hold(self, seconds: float) -> None:
        t_end = self._tick() + seconds
        while self._tick() < t_end:
            self.send(0.0, 0.0, 0.0)

    def _avoid(self, tx: float, ty: float, dist: float, side: int) -> tuple[tuple[float, float] | None, int]:
        """(point to steer at, detour side): the target, or AVOID_LOOK_M along the detour heading; None = no clear
        heading."""
        ps, sc = self.pose, self.scan
        if sc is None:
            return (tx, ty), 0
        pts = scan_points(sc.ranges, sc.angle_min, sc.angle_increment, sc.range_min, sc.range_max)
        bearing = wrap(math.atan2(ty - ps.y, tx - ps.x) - ps.yaw)
        h, side = avoid_heading(pts, bearing, dist, AVOID_LOOK_M, AVOID_HALF_W_M, side)
        if h is None or h == bearing:
            return (None if h is None else (tx, ty)), side
        return (ps.x + AVOID_LOOK_M * math.cos(ps.yaw + h), ps.y + AVOID_LOOK_M * math.sin(ps.yaw + h)), side

    def goto(self, tx: float, ty: float, timeout_s: float, gate_m: float, stop_m: float = STOP_TOL_M,
             min_v: float = 0.0) -> dict:
        """Drive to (tx, ty) until within `stop_m`, detouring around scan obstacles. Stops on timeout or on the stall
        rule (closest distance not improved by STALL_M in STALL_S sim). Returns sim time to first reach `gate_m`,
        closest distance, timeout / stall flags, detour ticks. `min_v` lifts a smaller (vx, vy) command to that speed
        (same direction): the policy's dead band for small moves, as MIN_WZ for turns."""
        t0 = self._tick()
        t_gate, best = None, float("inf")
        ref, t_ref = float("inf"), t0  # stall rule: last closest distance that improved by >= STALL_M, and when
        detours, side = 0, 0
        while True:
            t = self._tick()
            ps = self.pose
            dist = math.hypot(tx - ps.x, ty - ps.y)
            best = min(best, dist)
            if dist <= ref - STALL_M:
                ref, t_ref = dist, t
            if dist <= gate_m and t_gate is None:
                t_gate = t - t0
            stalled = t - t_ref > STALL_S
            if dist <= stop_m or t - t0 > timeout_s or stalled:
                break
            aim, side = self._avoid(tx, ty, dist, side)
            if aim is None:
                self.send(0.0, 0.0, 0.0)  # boxed in: stand; the stall rule ends the leg
                continue
            detours += aim != (tx, ty)
            vx, vy, wz = waypoint_cmd(ps.x, ps.y, ps.yaw, *aim)
            v = math.hypot(vx, vy)
            if 0.0 < v < min_v:
                vx, vy = vx * min_v / v, vy * min_v / v
            self.send(vx, vy, wz)
        self.send(0.0, 0.0, 0.0)
        return {"target": [tx, ty], "t_to_gate_s": t_gate, "closest_m": best, "final_dist_m": dist,
                "elapsed_s": t - t0, "timed_out": t - t0 > timeout_s, "stalled": stalled and dist > stop_m,
                "detour_ticks": detours}

    def _err_deg(self, target_yaw: float) -> float:
        return math.degrees(wrap(target_yaw - self.pose.yaw))

    def turn_to(self, target_yaw: float, timeout_s: float, max_rounds: int = 8) -> dict:
        """Turn to an absolute heading: coarse step-turn when |err| > COARSE_DEG, then settle-measure-pulse rounds
        (geometry.fine_pulse) until the error after a SETTLE_S settle is under YAW_TOL_DEG."""
        t0 = self._tick()
        pulses = []
        for rnd in range(1, max_rounds + 1):
            if abs(self._err_deg(target_yaw)) > COARSE_DEG:
                while abs(self._err_deg(target_yaw)) > COARSE_STOP_DEG and self.pose.t - t0 < timeout_s:
                    self._tick()
                    self.send(0.0, 0.0, heading_cmd(self.pose.yaw, target_yaw, min_wz=MIN_WZ))
            else:
                wz, dur = fine_pulse(self._err_deg(target_yaw))
                pulses.append([round(wz, 2), round(dur, 2)])
                t_end = self._tick() + dur
                while self._tick() < t_end:
                    self.send(0.0, 0.0, wz)
            self.hold(SETTLE_S)  # every round ends settled, so the last reading is the settled heading
            if abs(self._err_deg(target_yaw)) < YAW_TOL_DEG or self.pose.t - t0 > timeout_s:
                break
        return {"target_deg": math.degrees(target_yaw), "settled_err_deg": self._err_deg(target_yaw),
                "rounds": rnd, "fine_pulses": pulses, "elapsed_s": self.pose.t - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("goto")
    g.add_argument("x", type=float)
    g.add_argument("y", type=float)
    h = sub.add_parser("heading")
    h.add_argument("deg", type=float)
    args = ap.parse_args()
    rclpy.init()
    d = Drive()
    d.wait_pose()
    if args.cmd == "goto":
        print(d.goto(args.x, args.y, timeout_s=60.0, gate_m=0.30))
    else:
        print(d.turn_to(math.radians(args.deg), timeout_s=30.0))
    d.destroy_node()
    rclpy.shutdown()
    return None


if __name__ == "__main__":
    sys.exit(main())
