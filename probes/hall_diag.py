#!/usr/bin/env python3
"""Task 003-L hall localization diagnostic: N hall arrivals, each from a room place (cycling ward, office, reception,
storage). Per arrival (room and hall): Nav2 status, sim time, Gazebo GT pose, AMCL pose (TF map->base_link and the last
/amcl_pose with its covariance), AMCL - GT split into dx (along the hall's long axis, map x) and dy (across, map y),
yaw error, and the GT distance to the place point. Logged after a 1 s sim settle.

    export ROS_DOMAIN_ID=.. GZ_PARTITION=..; source sim/env.sh
    python3 probes/hall_diag.py --out DIR [--arrivals 20] [--tag stock]

With the clinic sim + AMCL + Nav2 running (sim/launch_clinic.sh localization:=true nav2:=true [amcl:=<yaml>]).
Acceptance (design review, task 003 fix round 1): every hall arrival SUCCEEDED with GT < 0.20 m from the hall place point.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from pathlib import Path

import map_world_check as mwc

ROOT = Path(__file__).resolve().parents[1]
ROOMS = ["ward", "office", "reception", "storage"]
HALL_ACCEPT_M = 0.20  # design ruling, task 003 fix round 1: 20/20 hall arrivals with GT error < 0.20 m
SETTLE_SIM_S = 1.0


def loc_error(amcl: mwc.Pose2, gt: mwc.Pose2) -> dict:
    """AMCL - GT in the map frame: dx along the hall's long axis (x), dy across it (y), yaw error (deg, wrapped)."""
    dx, dy = amcl[0] - gt[0], amcl[1] - gt[1]
    return {"dx_along_m": dx, "dy_across_m": dy, "dxy_m": math.hypot(dx, dy),
            "dyaw_deg": math.degrees(mwc.wrap(amcl[2] - gt[2]))}


def summarize(rows: list[dict], accept_m: float = HALL_ACCEPT_M) -> dict:
    """Over arrival rows: max / mean |dx along|, |dy across|, |dyaw|; GT distance to the place; pass count
    (SUCCEEDED and GT < accept_m)."""
    if not rows:
        return {"n": 0}
    ax = [abs(r["err_tf"]["dx_along_m"]) for r in rows]
    ay = [abs(r["err_tf"]["dy_across_m"]) for r in rows]
    yw = [abs(r["err_tf"]["dyaw_deg"]) for r in rows]
    gd = [r["gt_dist_to_place_m"] for r in rows]
    ok = [r["status_name"] == "SUCCEEDED" and r["gt_dist_to_place_m"] < accept_m for r in rows]
    return {"n": len(rows), "n_succeeded": sum(r["status_name"] == "SUCCEEDED" for r in rows),
            "n_gt_below_accept": sum(ok), "accept_m": accept_m, "all_pass": all(ok),
            "along_abs_max_m": max(ax), "along_abs_mean_m": statistics.fmean(ax),
            "along_signed_mean_m": statistics.fmean(r["err_tf"]["dx_along_m"] for r in rows),
            "across_abs_max_m": max(ay), "across_abs_mean_m": statistics.fmean(ay),
            "across_signed_mean_m": statistics.fmean(r["err_tf"]["dy_across_m"] for r in rows),
            "dyaw_abs_max_deg": max(yw), "dyaw_abs_mean_deg": statistics.fmean(yw),
            "gt_dist_max_m": max(gd), "gt_dist_mean_m": statistics.fmean(gd),
            "gt_dists_sorted_m": sorted(round(d, 3) for d in gd)}


def attach_amcl_pose(robot) -> None:  # noqa: ANN001  (adds a /amcl_pose subscription to map_world_check's Robot)
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    robot.amcl_msg = None
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    robot.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", lambda m: setattr(robot, "amcl_msg", m), qos)


def amcl_msg_record(robot) -> dict | None:  # noqa: ANN001
    m = robot.amcl_msg
    if m is None:
        return None
    p, c = m.pose.pose, m.pose.covariance
    pose = (p.position.x, p.position.y, mwc.yaw_of(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w))
    return {"pose": pose, "stamp_s": m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
            "std_x_m": math.sqrt(max(c[0], 0.0)), "std_y_m": math.sqrt(max(c[7], 0.0)), "cov_xy": c[1],
            "std_yaw_deg": math.degrees(math.sqrt(max(c[35], 0.0)))}


def arrive(robot, name: str, goal: mwc.Pose2, kind: str) -> dict:  # noqa: ANN001
    nav = robot.navigate(goal)
    robot.spin_for_sim(SETTLE_SIM_S)
    gt, gt_t = mwc.gt_pose("clinic")
    tf = robot.map_pose()
    am = amcl_msg_record(robot)
    row = {"kind": kind, "place": name, "goal": goal, **nav, "sim_time_s": robot.now_s(), "gt_sim_time_s": gt_t,
           "gt": gt, "amcl_tf": tf, "err_tf": loc_error(tf, gt), "amcl_msg": am,
           "err_msg": loc_error(am["pose"], gt) if am else None,
           "amcl_msg_age_s": robot.now_s() - am["stamp_s"] if am else None,
           "gt_dist_to_place_m": math.dist(gt[:2], goal[:2]), "amcl_dist_to_place_m": math.dist(tf[:2], goal[:2]),
           "gt_minus_place_x_m": gt[0] - goal[0], "gt_minus_place_y_m": gt[1] - goal[1]}
    e = row["err_tf"]
    print(f"{kind:4s} {name:9s} {nav['status_name']:10s} sim {nav['sim_s']:5.1f}s  GT dist {row['gt_dist_to_place_m']:.3f}"
          f"  AMCL-GT along {e['dx_along_m']:+.3f} across {e['dy_across_m']:+.3f} yaw {e['dyaw_deg']:+.1f}"
          + (f"  std x {am['std_x_m']:.3f} y {am['std_y_m']:.3f} age {row['amcl_msg_age_s']:.1f}s" if am else ""),
          flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arrivals", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--places", type=Path, default=ROOT / "sim/worlds/clinic/places.yaml")
    a = ap.parse_args()
    places = mwc.load_places(a.places)
    robot = mwc.make_robot()
    attach_amcl_pose(robot)
    rows: list[dict] = []
    out = a.out / f"hall_diag{'_' + a.tag if a.tag else ''}.json"
    a.out.mkdir(parents=True, exist_ok=True)
    for k in range(a.arrivals):
        room = ROOMS[k % len(ROOMS)]
        rows.append({"arrival": k, **arrive(robot, room, places[room], "room")})
        rows.append({"arrival": k, "from": room, **arrive(robot, "hall", places["hall"], "hall")})
        hall = [r for r in rows if r["kind"] == "hall"]
        res = {"tag": a.tag, "written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
               "criterion": f"hall arrival passes = SUCCEEDED and GT < {HALL_ACCEPT_M} m from the hall place point; "
                            f"err = AMCL (TF map->base_link) - GT after {SETTLE_SIM_S} s sim settle; along = map x",
               "summary_hall": summarize(hall), "summary_room": summarize([r for r in rows if r["kind"] == "room"]),
               "rows": rows}
        out.write_text(json.dumps(res, indent=2, default=float))  # rewritten each arrival: a cut run keeps its rows
    s = res["summary_hall"]
    print(f"HALL {s['n_gt_below_accept']}/{s['n']} GT < {HALL_ACCEPT_M} m  along max {s['along_abs_max_m']:.3f} mean "
          f"{s['along_abs_mean_m']:.3f}  across max {s['across_abs_max_m']:.3f} mean {s['across_abs_mean_m']:.3f}  "
          f"GT dist max {s['gt_dist_max_m']:.3f}", flush=True)
    return 0 if s["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
