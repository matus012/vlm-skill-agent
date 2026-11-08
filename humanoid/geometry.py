"""Pure math for the G1 lane: quaternions, angles, camera intrinsics, waypoint / heading control laws.

No ROS, no MuJoCo imports, so it is unit-testable with plain pytest. route() is the G1 adapter's doorway router.
Quaternion conventions: MuJoCo stores (w, x, y, z); ROS messages use (x, y, z, w).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from agent.world import Polygon, room_of

Quat = tuple[float, float, float, float]
HALL = "hall"
DOOR_STEP_M = 0.6  # G1 adapter: doorway approach points either side of the door line (doorways are 1.4 m wide)


def wxyz_to_xyzw(q: Quat) -> Quat:
    w, x, y, z = q
    return (x, y, z, w)


def xyzw_to_wxyz(q: Quat) -> Quat:
    x, y, z, w = q
    return (w, x, y, z)


def yaw_from_wxyz(q: Quat) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_angle_deg(a_wxyz: Quat, b_wxyz: Quat) -> float:
    """Smallest rotation angle between two unit quaternions (degrees)."""
    dot = abs(sum(p * q for p, q in zip(a_wxyz, b_wxyz)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def grew(before: float, after: float, min_change: float, angle: bool = False) -> tuple[float, bool]:
    """Signed change after - before (wrapped to [-pi, pi) for angles in radians) and whether it is >= min_change."""
    d = wrap(after - before) if angle else after - before
    return d, d >= min_change


def yaw_to_wxyz(yaw: float) -> Quat:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def fovy_for_hfov(hfov_rad: float, width: int, height: int) -> float:
    """MuJoCo `fovy` (vertical, degrees) that gives the requested horizontal FOV at width x height."""
    return math.degrees(2.0 * math.atan(math.tan(hfov_rad / 2.0) * height / width))


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) of a MuJoCo pinhole camera: square pixels, f from the vertical FOV."""
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return f, f, (width - 1) / 2.0, (height - 1) / 2.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class Limits:
    """Command box of AGILE's Velocity-G1-v0 CommandManager (agile/sim2mujoco/commands.py)."""
    vx: float = 0.5
    vy: float = 0.5
    wz: float = 1.0


def waypoint_cmd(x: float, y: float, yaw: float, tx: float, ty: float, lim: Limits = Limits(),
                 k_lin: float = 1.0, k_yaw: float = 1.5, face_dist: float = 0.4) -> tuple[float, float, float]:
    """Body-frame (vx, vy, wz) toward (tx, ty): holonomic P on the position error, and turn to face the
    target while it is farther than `face_dist` (near it, hold heading so the robot does not spin)."""
    dx, dy = tx - x, ty - y
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = c * dx + s * dy, -s * dx + c * dy
    vx, vy = k_lin * ex, k_lin * ey
    scale = max(abs(vx) / lim.vx, abs(vy) / lim.vy, 1.0)  # keep direction when saturating
    vx, vy = vx / scale, vy / scale
    wz = 0.0
    if math.hypot(dx, dy) > face_dist:
        wz = clamp(k_yaw * wrap(math.atan2(dy, dx) - yaw), -lim.wz, lim.wz)
    return vx, vy, wz


def fine_pulse(err_deg: float) -> tuple[float, float]:
    """(wz, seconds) of one yaw correction pulse for a settled error, from the dev step responses of AGILE
    Velocity-G1-v0 (README, "Turning"): below ~0.5 rad/s the policy mostly twists the pelvis and springs back, so
    small errors use a 0.4 rad/s pulse whose settled gain is ~0.3 deg + 1.1 deg/s; 4-10 deg use one 1.0 rad/s x
    0.3 s step-turn (~6 deg settled)."""
    s = math.copysign(1.0, err_deg)
    e = abs(err_deg)
    if e > 4.0:
        return s * 1.0, 0.3
    return s * 0.4, clamp((e - 0.3) / 1.1, 0.3, 3.0)


def heading_cmd(yaw: float, target_yaw: float, lim: Limits = Limits(), k_yaw: float = 1.5,
                min_wz: float = 0.0) -> float:
    """wz toward an absolute heading; `min_wz` lifts small commands over a policy dead band."""
    err = wrap(target_yaw - yaw)
    wz = clamp(k_yaw * err, -lim.wz, lim.wz)
    if 0.0 < abs(wz) < min_wz:
        wz = math.copysign(min_wz, wz)
    return wz


def door_points(poly: Polygon, door: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    """(inside, outside) points DOOR_STEP_M either side of a doorway centre on an axis-aligned room wall."""
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    cx, cy = door
    if min(abs(cy - min(ys)), abs(cy - max(ys))) < 1e-6:  # door in a wall along x: cross along y
        n = (0.0, math.copysign(1.0, (min(ys) + max(ys)) / 2 - cy))
    else:
        n = (math.copysign(1.0, (min(xs) + max(xs)) / 2 - cx), 0.0)
    return (cx + DOOR_STEP_M * n[0], cy + DOOR_STEP_M * n[1]), (cx - DOOR_STEP_M * n[0], cy - DOOR_STEP_M * n[1])


def route(x0: float, y0: float, x: float, y: float, rooms: dict[str, Polygon],
          doors: dict[str, tuple[float, float]]) -> list[tuple[float, float]]:
    """Waypoints from (x0, y0) to (x, y): out of the start room into the hall, into the goal room, to the goal."""
    a, b = room_of(x0, y0, rooms), room_of(x, y, rooms)
    pts: list[tuple[float, float]] = []
    if a != b:
        if a != HALL:
            pts += list(door_points(rooms[a], doors[a]))
        if b != HALL:
            pts += list(reversed(door_points(rooms[b], doors[b])))
    return pts + [(x, y)]


def scan_points(ranges: np.ndarray, angle_min: float, angle_inc: float, range_min: float, range_max: float
                ) -> np.ndarray:
    """(N, 2) body-frame obstacle points of a LaserScan (pelvis-yaw frame); returns outside [range_min, range_max]
    (inf / nan / no hit) are dropped."""
    r = np.asarray(ranges, float)
    a = angle_min + angle_inc * np.arange(r.size)
    ok = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    return np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], axis=1)


def blocked(pts: np.ndarray, headings: np.ndarray, length: np.ndarray | float, half_w: float) -> np.ndarray:
    """Per body-frame heading: does any point lie in the corridor 0 < forward <= length, |lateral| < half_w?
    Points beside or behind the robot never block, so doorway posts off the corridor do not either."""
    c, s = np.cos(headings)[:, None], np.sin(headings)[:, None]
    fwd = c * pts[:, 0] + s * pts[:, 1]
    lat = -s * pts[:, 0] + c * pts[:, 1]
    L = np.broadcast_to(np.asarray(length, float), headings.shape)[:, None]
    return ((fwd > 0.0) & (fwd <= L) & (np.abs(lat) < half_w)).any(axis=1)


def avoid_heading(pts: np.ndarray, bearing: float, goal_m: float, look_m: float, half_w: float, side: int = 0
                  ) -> tuple[float | None, int]:
    """(body-frame heading to walk, detour side) by a small deterministic VFH-lite rule. The goal bearing if its
    corridor (up to the goal or look_m) is clear -> side 0. Else the clear heading nearest to it in 1 deg steps; a
    detour keeps to the side it started on (side +1 left / -1 right; 0 = nearest, left first on a tie) so a symmetric
    obstacle cannot make it flip every tick; the other side only if the kept side is all blocked. None if every heading
    is blocked. The goal's own corridor stops at the goal, so an object behind the goal point does not divert the leg."""
    if not blocked(pts, np.array([bearing]), min(look_m, goal_m), half_w)[0]:
        return bearing, 0
    k = np.arange(1, 181)
    offs = np.radians(np.stack([k, -k], axis=1).ravel()) if side == 0 else \
        np.radians(np.concatenate([side * k, -side * k]))
    free = ~blocked(pts, bearing + offs, look_m, half_w)
    if not free.any():
        return None, side
    off = float(offs[np.argmax(free)])
    return wrap(bearing + off), (1 if off > 0 else -1)
