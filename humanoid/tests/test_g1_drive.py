"""Drive.goto / _tick on a fake pose + scan stream (no sim): stall rule, fall, stop_m, scan avoidance (012 section 4,
a recorded deviation). Mutants: tests/mutants_g1.py."""
from __future__ import annotations

import math

import numpy as np
import pytest
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import humanoid.drive as drive
from humanoid.drive import Drive, Pose, RobotFell
from humanoid.geometry import Limits, avoid_heading

Seg = tuple[tuple[float, float], tuple[float, float]]
BODY_M = 0.25  # fake world: the robot cannot get its centre closer than this to any segment (it is stopped there)
DT = 0.02      # 50 Hz odom


def box(cx: float, cy: float, half: float) -> list[Seg]:
    c = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
    return list(zip(c, c[1:] + c[:1]))


def seg_dist(px: float, py: float, s: Seg) -> float:
    (x1, y1), (x2, y2) = s
    dx, dy = x2 - x1, y2 - y1
    u = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - x1 - u * dx, py - y1 - u * dy)


class World:
    """Kinematic robot: applies the last body-frame command (clamped to the policy box) for DT per odom sample, stops
    at BODY_M from any segment, publishes a 360 x 1 deg LaserScan (0.1-4.0 m, pelvis-yaw frame) every 5th sample."""

    def __init__(self, x: float, y: float, yaw: float, segs: list[Seg], z_at: dict[float, float] | None = None) -> None:
        self.x, self.y, self.yaw, self.segs, self.t, self.n = x, y, yaw, segs, 0.0, 0
        self.z_at = z_at or {}
        self.cmd = (0.0, 0.0, 0.0)
        self.min_clear = float("inf")

    def clear(self, x: float, y: float) -> float:
        return min((seg_dist(x, y, s) for s in self.segs), default=float("inf"))

    def scan(self) -> LaserScan:
        m = LaserScan()
        m.angle_min, m.angle_increment, m.range_min, m.range_max = -math.pi, math.radians(1.0), 0.1, 4.0
        a = self.yaw + m.angle_min + m.angle_increment * np.arange(360)
        r = np.full(360, np.inf)
        dx, dy = np.cos(a), np.sin(a)
        for (x1, y1), (x2, y2) in self.segs:  # ray-segment intersection
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((x1 - self.x) * ey - (y1 - self.y) * ex) / den
                u = ((x1 - self.x) * dy - (y1 - self.y) * dx) / den
            hit = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
            r = np.where(hit & (t < r), t, r)
        m.ranges = [float(v) if v <= m.range_max else float("inf") for v in r]
        return m

    def step(self, d: Drive) -> None:
        lim = Limits()
        vx, vy, wz = (max(-lim.vx, min(lim.vx, self.cmd[0])), max(-lim.vy, min(lim.vy, self.cmd[1])),
                      max(-lim.wz, min(lim.wz, self.cmd[2])))
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        nx, ny = self.x + DT * (c * vx - s * vy), self.y + DT * (s * vx + c * vy)
        if self.clear(nx, ny) >= BODY_M:
            self.x, self.y = nx, ny
        self.yaw += DT * wz
        self.t += DT
        self.min_clear = min(self.min_clear, self.clear(self.x, self.y))
        m = Odometry()
        m.header.stamp.sec, m.header.stamp.nanosec = int(self.t), int(round((self.t % 1) * 1e9)) % 1_000_000_000
        m.pose.pose.position.x, m.pose.pose.position.y = self.x, self.y
        m.pose.pose.position.z = min((z for t, z in self.z_at.items() if self.t >= t), default=0.75)
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        d._on_odom(m)
        if self.n % 5 == 0 and self.segs:
            d.scan = self.scan()
        self.n += 1


@pytest.fixture
def make(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    def _make(w: World) -> Drive:
        d = Drive.__new__(Drive)  # no rclpy node: the World is the pose / scan stream
        d.pose, d.scan = Pose(), None
        d.send = lambda vx, vy, wz: setattr(w, "cmd", (vx, vy, wz))
        monkeypatch.setattr(drive.rclpy, "spin_once", lambda node, timeout_sec=None: w.step(node))
        w.step(d)
        return d
    return _make


def test_stall_rule_stops_a_stuck_leg(make) -> None:  # noqa: ANN001
    w = World(0.0, 0.0, 0.0, [])
    d = make(w)
    w.step = lambda node: _frozen(w, node)  # the robot cannot move at all
    leg = d.goto(3.0, 0.0, timeout_s=120.0, gate_m=0.15)
    assert leg["stalled"] and not leg["timed_out"], leg
    assert drive.STALL_S <= leg["elapsed_s"] <= drive.STALL_S + 0.1, leg


def _frozen(w: World, d: Drive) -> None:
    cmd = w.cmd
    w.cmd = (0.0, 0.0, 0.0)
    World.step(w, d)
    w.cmd = cmd


def test_free_leg_reaches_and_stop_m_is_honoured(make) -> None:  # noqa: ANN001
    d = make(World(0.0, 0.0, 0.0, []))
    leg = d.goto(0.3, 0.0, timeout_s=60.0, gate_m=0.05, stop_m=0.05)
    assert not leg["stalled"] and not leg["timed_out"] and leg["final_dist_m"] <= 0.05, leg
    assert leg["detour_ticks"] == 0


def test_min_v_lifts_small_commands(make) -> None:  # noqa: ANN001
    w = World(0.0, 0.0, 0.0, [])
    d = make(w)
    sent: list[float] = []
    d.send = lambda vx, vy, wz: (sent.append(math.hypot(vx, vy)), setattr(w, "cmd", (vx, vy, wz)))
    d.goto(0.1, 0.0, timeout_s=10.0, gate_m=0.05, stop_m=0.05, min_v=0.25)
    assert min(v for v in sent if v > 0) >= 0.25 - 1e-9


def test_fall_raises(make) -> None:  # noqa: ANN001
    d = make(World(0.0, 0.0, 0.0, [], z_at={1.0: 0.45}))
    with pytest.raises(RobotFell):
        d.goto(5.0, 0.0, timeout_s=60.0, gate_m=0.15)


def test_box_on_the_leg_detours_and_reaches(make) -> None:  # noqa: ANN001
    w = World(0.0, 0.0, 0.0, box(1.5, 0.0, 0.2))
    d = make(w)
    leg = d.goto(3.0, 0.0, timeout_s=120.0, gate_m=0.15)
    assert leg["final_dist_m"] <= drive.STOP_TOL_M and not leg["stalled"] and not leg["timed_out"], leg
    assert leg["detour_ticks"] > 0
    assert w.min_clear > BODY_M + 0.02, w.min_clear  # never pressed against the box


def test_doorway_posts_pass_straight(make) -> None:  # noqa: ANN001
    wall = [((1.5, -3.0), (1.5, -0.45)), ((1.5, 0.45), (1.5, 3.0))]  # 0.9 m doorway (narrowest the design ruling names)
    w = World(0.0, 0.05, 0.0, wall)
    d = make(w)
    leg = d.goto(3.0, 0.05, timeout_s=120.0, gate_m=0.15)
    assert leg["final_dist_m"] <= drive.STOP_TOL_M and leg["detour_ticks"] == 0, leg


def test_blocked_doorway_stalls_within_the_timeout(make) -> None:  # noqa: ANN001
    w = World(0.0, 0.0, 0.0, [((1.5, -20.0), (1.5, 20.0))])
    d = make(w)
    leg = d.goto(3.0, 0.0, timeout_s=120.0, gate_m=0.15)
    assert leg["stalled"] and not leg["timed_out"] and leg["elapsed_s"] < 120.0, leg
    assert w.min_clear > BODY_M + 0.02, w.min_clear


def test_avoid_heading_rules() -> None:
    pts = np.array([[0.4, 0.0]])  # a point straight ahead, inside look_m
    h, side = avoid_heading(pts, 0.0, 3.0, 0.5, 0.35)
    assert h > 0.0 and side == 1                                    # symmetric: left first
    h, side = avoid_heading(pts, 0.0, 3.0, 0.5, 0.35, side=-1)
    assert h < 0.0 and side == -1                                   # a started detour keeps its side
    assert avoid_heading(pts, 0.0, 0.3, 0.5, 0.35) == (0.0, 0)      # behind the goal point: ignored
    assert avoid_heading(np.array([[0.2, 0.45], [0.2, -0.45]]), 0.0, 3.0, 0.5, 0.35) == (0.0, 0)  # posts beside
    ring = np.array([[0.2 * math.cos(a), 0.2 * math.sin(a)] for a in np.radians(np.arange(0, 360, 5))])
    assert avoid_heading(ring, 0.0, 3.0, 0.5, 0.35)[0] is None      # boxed in


def test_sync_waits_for_new_odom_and_scan(make) -> None:  # noqa: ANN001
    """After a layout reset the held pose / scan are the pre-reset ones (p8d11): sync() returns only once a new odom
    sample and a new scan were processed (scans come every 5th odom sample here)."""
    w = World(0.0, 0.0, 0.0, box(3.0, 0.0, 0.2))
    d = make(w)
    stale = d.scan
    w.x, w.y = 2.0, 1.0  # the reset moved the robot; nothing processed yet
    d.sync()
    assert (d.pose.x, d.pose.y) == (2.0, 1.0) and d.scan is not stale
    assert w.n == 6  # make() took sample 0 (with its scan); sync ran samples 1-5, the next scan rides on 5
