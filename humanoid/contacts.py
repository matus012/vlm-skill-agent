"""Robot-clinic contacts of the G1 (task 012, design ruling on H-P1 "collision"): pure mujoco + numpy, no ROS.

A collision is a MuJoCo contact (an entry of d.contact after a physics step) between a robot geom and a non-floor,
non-robot geom: placed movables (mocap bodies) or static clinic geometry (walls, door trims, fixed furniture), and the
floor itself for any robot part but a foot (design review: only foot-floor contacts are excluded; the G1's foot = its ankle pitch +
roll links, FOOT_SUFFIXES). Robot self-contacts are excluded. Counts are contact-steps: every such contact
present after a physics step counts once for that step.

Clearance: mj_geomDistance from each collidable pelvis geom to every collidable non-floor non-robot geom, capped at
`distmax` (None = nothing within the cap).
"""
from __future__ import annotations

import mujoco
import numpy as np

FLOOR_BODIES = ("floor", "clinic/structure_floor")  # humanoid/room.py build_scene / the clinic MJCF floor body
FOOT_SUFFIXES = ("ankle_roll_link", "ankle_pitch_link")  # G1 g1_29dof.xml: the foot assembly below the shin


def floor_geoms(m: mujoco.MjModel) -> np.ndarray:
    """bool[ngeom]: plane geoms and every geom of a floor body."""
    floor_b = np.array([m.body(b).name in FLOOR_BODIES for b in range(m.nbody)])
    return (m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) | floor_b[m.geom_bodyid]


class ContactCounter:
    """Cumulative robot-clinic contact-steps since construction, split mocap / static; scan(d) after each physics step."""

    def __init__(self, m: mujoco.MjModel, robot_body_ids: list[int], floor: np.ndarray | None = None,
                 foot_body_ids: list[int] | None = None) -> None:
        self.m = m
        self.robot = np.zeros(m.nbody, bool)
        self.robot[robot_body_ids] = True
        self.foot = np.zeros(m.nbody, bool)
        self.foot[[b for b in robot_body_ids if m.body(b).name.endswith(FOOT_SUFFIXES)]
                  if foot_body_ids is None else foot_body_ids] = True
        self.floor = floor_geoms(m) if floor is None else floor
        self.mocap_body = m.body_mocapid >= 0
        self.n_mocap = self.n_static = 0
        coll = (m.geom_contype | m.geom_conaffinity) != 0
        rg = self.robot[m.geom_bodyid]
        self.pelvis_geoms = [g for g in range(m.ngeom) if coll[g] and m.body(m.geom_bodyid[g]).name == "pelvis"]
        self.other_geoms = [g for g in range(m.ngeom) if coll[g] and not rg[g] and not self.floor[g]]

    def scan(self, d: mujoco.MjData) -> list[tuple[int, int, bool, float]]:
        """(robot body, other body, other is mocap, dist) of this step's counted contacts; adds them to the totals."""
        n = d.ncon
        if n == 0:
            return []
        g = d.contact.geom[:n]
        b = self.m.geom_bodyid[g]
        r = self.robot[b]
        rb_all = np.where(r[:, 0], b[:, 0], b[:, 1])
        keep = (r[:, 0] != r[:, 1]) & ~(self.floor[g].any(axis=1) & self.foot[rb_all])
        if not keep.any():
            return []
        out = []
        for i in np.flatnonzero(keep):
            rb, ob = (b[i, 0], b[i, 1]) if r[i, 0] else (b[i, 1], b[i, 0])
            mo = bool(self.mocap_body[ob])
            out.append((int(rb), int(ob), mo, float(d.contact.dist[i])))
            if mo:
                self.n_mocap += 1
            else:
                self.n_static += 1
        return out

    def clearance(self, d: mujoco.MjData, distmax: float) -> float | None:
        """Min mj_geomDistance pelvis -> any non-floor clinic geom (< 0 = penetration); None if none within distmax."""
        best = distmax
        for p in self.pelvis_geoms:
            for o in self.other_geoms:
                best = min(best, mujoco.mj_geomDistance(self.m, d, p, o, best, None))
        return None if best >= distmax else best
