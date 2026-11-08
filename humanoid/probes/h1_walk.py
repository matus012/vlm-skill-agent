"""H1 walk: 5 waypoints in the room, each 2-4 m from the previous, driven only through /cmd_vel by our controller.
Pass = ground-truth base within 0.30 m of each waypoint within 60 s sim time, base height never < 0.5 m. 5/5.
Needs a fresh g1_node (spawn at the room's spawn pose). Writes evidence sideH-H1-walk.json.
"""
from __future__ import annotations

import math

import rclpy

from humanoid.drive import Drive
from humanoid.probes.common import write_verdict

GATE_M, TIMEOUT_S, MIN_Z = 0.30, 60.0, 0.5   # spec side-H, gate H1
WAYPOINTS = [(1.5, -1.0), (1.5, 1.5), (-1.5, 1.8), (-3.0, -0.5), (0.5, -1.5)]  # spike_room.yaml, spawn (-1.5, -1)


def main() -> None:
    rclpy.init()
    d = Drive("h1_walk")
    d.wait_pose()
    d.hold(1.0)
    prev = (d.pose.x, d.pose.y)
    legs = []
    for wx, wy in WAYPOINTS:
        leg_m = math.hypot(wx - prev[0], wy - prev[1])
        r = d.goto(wx, wy, timeout_s=TIMEOUT_S, gate_m=GATE_M)
        r["leg_m"] = leg_m
        r["leg_in_2_4m"] = 2.0 <= leg_m <= 4.0
        r["pass"] = r["t_to_gate_s"] is not None and r["t_to_gate_s"] <= TIMEOUT_S
        legs.append(r)
        prev = (wx, wy)
    d.hold(1.0)
    n_ok = sum(r["pass"] and r["leg_in_2_4m"] for r in legs)
    no_fall = d.pose.min_z >= MIN_Z
    write_verdict("H1-walk", {
        "gate": "5 waypoints 2-4 m apart, base within 0.30 m within 60 s sim, base z never < 0.5 m",
        "legs": legs, "waypoints_passed": n_ok, "min_base_z_odom_50hz": d.pose.min_z,
        "odom_samples": len(d.pose.history), "sim_span_s": d.pose.history[-1][0] - d.pose.history[0][0],
        "verdict": "PASS" if n_ok == len(WAYPOINTS) and no_fall else "FAIL"})
    d.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
