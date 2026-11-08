"""H4 pose: the published pose (/g1/odom) equals the pelvis body pose read by a second access path (MuJoCo forward
kinematics, data.xpos / data.xquat of body `pelvis`), <= 1 cm and <= 0.5 deg, on 100 samples while walking; plus
direction checks: a +x /cmd_vel step makes the published x grow, a +z yaw rate makes the published yaw grow.
The node must run with --truth-log <jsonl> (per odom stamp: raw free-joint qpos[0:7] and the pelvis body pose).
The free-joint comparison (same doubles as the published pose, true by construction) is still recorded.
This probe checks directions from the spawn (yaw 0, so body +x = world +x), walks the robot, keeps 100 odom messages
taken while the base moves > 0.1 m/s, and joins them to the truth log by stamp. Writes sideH-H4-pose.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS
from nav_msgs.msg import Odometry

from humanoid.drive import Drive
from humanoid.geometry import grew, quat_angle_deg, xyzw_to_wxyz
from humanoid.probes.common import write_verdict

POS_TOL_M, ANG_TOL_DEG, N = 0.01, 0.5, 100   # spec side-H, gate H4
LEG = [(1.5, -1.0), (1.5, 1.5)]   # from spawn (-1.5, -1): forward 3 m, then a turn + 2.5 m
# Direction checks (task 004, set before the counted run): a 2 s step of vx 0.4 m/s and a 1.5 s step of wz 0.8 rad/s,
# then 1 s settle; the published x must grow >= 0.2 m and the published yaw >= 20 deg (ideal 0.8 m / 69 deg).
FWD_VX, FWD_S, FWD_MIN_M = 0.4, 2.0, 0.2
TURN_WZ, TURN_S, TURN_MIN_DEG = 0.8, 1.5, 20.0


def step_cmd(d: Drive, vx: float, wz: float, seconds: float) -> None:
    t_end = d._tick() + seconds
    while d._tick() < t_end:
        d.send(vx, 0.0, wz)
    d.hold(1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-log", type=Path, required=True)
    args = ap.parse_args()
    rclpy.init()
    d = Drive("h4_pose")
    walking: list[Odometry] = []

    def keep(m: Odometry) -> None:
        v = m.twist.twist.linear
        if math.hypot(v.x, v.y) > 0.1:
            walking.append(m)

    d.create_subscription(Odometry, "/g1/odom", keep, SENSOR_QOS)
    d.wait_pose()
    d.hold(1.0)
    x0 = d.pose.x
    step_cmd(d, FWD_VX, 0.0, FWD_S)
    fwd_dx, fwd_ok = grew(x0, d.pose.x, FWD_MIN_M)
    yaw0 = d.pose.yaw
    step_cmd(d, 0.0, TURN_WZ, TURN_S)
    turn_d, turn_ok = grew(yaw0, d.pose.yaw, math.radians(TURN_MIN_DEG), angle=True)
    for wx, wy in LEG:
        d.goto(wx, wy, timeout_s=60.0, gate_m=0.30)
    d.hold(0.5)
    time.sleep(2.0)  # let the node flush its truth log
    truth = {}
    for line in args.truth_log.read_text().splitlines():
        rec = json.loads(line)
        truth[rec["t_ns"]] = rec
    step = max(1, len(walking) // N)
    samples = walking[::step][:N]
    rows = []
    for m in samples:
        t_ns = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
        rec = truth[t_ns]
        p, o = m.pose.pose.position, m.pose.pose.orientation
        pub_xyz, pub_q = (p.x, p.y, p.z), xyzw_to_wxyz((o.x, o.y, o.z, o.w))
        rows.append({"t_ns": t_ns,
                     "body_pos_err_m": math.dist(pub_xyz, rec["body_xpos"]),
                     "body_ang_err_deg": quat_angle_deg(pub_q, tuple(rec["body_xquat"])),
                     "joint_pos_err_m": math.dist(pub_xyz, rec["qpos"][:3]),
                     "joint_ang_err_deg": quat_angle_deg(pub_q, tuple(rec["qpos"][3:7]))})
    worst = {k: max(r[k] for r in rows) for k in rows[0] if k != "t_ns"}
    passed = (len(rows) == N and worst["body_pos_err_m"] <= POS_TOL_M and worst["body_ang_err_deg"] <= ANG_TOL_DEG
              and fwd_ok and turn_ok)
    write_verdict("H4-pose", {
        "gate": "published pose == pelvis body pose by a second access path (xpos / xquat), <= 1 cm and <= 0.5 deg, "
                "100 samples while walking; +x cmd_vel grows x, +z yaw rate grows yaw",
        "n_samples": len(rows), "walking_msgs": len(walking),
        "max_body_pos_err_m": worst["body_pos_err_m"], "max_body_ang_err_deg": worst["body_ang_err_deg"],
        "max_joint_pos_err_m": worst["joint_pos_err_m"], "max_joint_ang_err_deg": worst["joint_ang_err_deg"],
        "direction": {"fwd_cmd": {"vx": FWD_VX, "s": FWD_S}, "fwd_dx_m": fwd_dx, "fwd_min_m": FWD_MIN_M, "fwd_ok": fwd_ok,
                      "turn_cmd": {"wz": TURN_WZ, "s": TURN_S}, "turn_dyaw_deg": math.degrees(turn_d),
                      "turn_min_deg": TURN_MIN_DEG, "turn_ok": turn_ok},
        "sample_speed_rule": "odom twist |v_xy| > 0.1 m/s", "samples": rows,
        "verdict": "PASS" if passed else "FAIL"})
    d.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
