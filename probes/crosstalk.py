"""006 gate M5 (hpc/PERUN_SIM.md section 4): no cross-talk between N sims running at once in one job.

    python probes/crosstalk.py --out DIR --sim DOMAIN:PARTITION --sim DOMAIN:PARTITION [...] [--world clinic]

One process, all sims up, no Nav2 goal active. Sim A = the first --sim, B = every other one.
(a) ROS graph: per domain, `ros2 node list --no-daemon` holds each CORE_NODES name exactly once, no name twice, and the
    same other names (suffix-normalised) as every other domain. A duplicate or an extra = cross-talk.
(b) gz: sim A's robot model is teleported (set_pose in A's partition) to its own position turned 90 deg about z;
    every B's GT robot pose must not move (negative), A's must reach the target (positive control). A is then put back.
(c) ROS: a nonce is published on MARKER_TOPIC in A's domain; a subscriber in each B domain must not get it (negative),
    a subscriber in A's domain must (positive control). A private topic, so no running node reacts to it (a publish on
    /initialpose would re-initialise A's AMCL).
Plus RTF per sim from Gazebo sim time over RTF_WINDOW_S wall seconds (gate: >= RTF_MIN).
Writes DIR/m5.json.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photos  # noqa: E402
import seg_compare as sc  # noqa: E402

# One copy of each in every sim: the lifecycle-managed nodes and their managers, the costmaps, the robot / bridge nodes
# this project's launch adds. Read from a local-machine clinic launch (runs/2026-11-04/005-p3-counted/launch.log: the
# lifecycle managers' "Configuring <node>" lines and the per-process logger names).
CORE_NODES = tuple(f"/{n}" for n in (
    "amcl", "map_server", "lifecycle_manager_localization",
    "controller_server", "smoother_server", "planner_server", "route_server", "behavior_server", "velocity_smoother",
    "collision_monitor", "bt_navigator", "waypoint_follower", "docking_server", "lifecycle_manager_navigation",
    "global_costmap/global_costmap", "local_costmap/local_costmap",
    "robot_state_publisher", "turtlebot4_node", "motion_control",
    "camera_bridge", "lidar_bridge", "cmd_vel_bridge", "odom_base_tf_bridge", "pose_bridge"))
MARKER_TOPIC = "/m5_crosstalk_marker"
RTF_MIN = 0.7  # task 006 M5
SPIN_TIME_S = 10.0      # ros2 node list discovery time (instrument, not a gate number)
RTF_WINDOW_S = 10.0     # wall seconds per RTF sample (instrument)
MATCH_TIMEOUT_S = 30.0  # wait for the A publisher to match A's subscriber (instrument)
LISTEN_S = 3.0          # wall seconds of listening after the marker publishes (instrument)


def parse_sim(s: str) -> tuple[int, str]:
    d, p = s.split(":", 1)
    return int(d), p


def normalise(name: str) -> str:
    """Drop a per-process id suffix (e.g. transform_listener_impl_5a1b2c3d4e5f) so domains compare by role."""
    return re.sub(r"_[0-9a-f]{6,}$", "", name)


def node_check(names: dict[int, list[str]], core: tuple[str, ...] = CORE_NODES) -> dict[str, object]:
    """names: domain -> `ros2 node list` lines (duplicates kept)."""
    per: dict[str, object] = {}
    extras_by_dom = {d: sorted({normalise(n) for n in ns if n not in core}) for d, ns in names.items()}
    common = set.intersection(*(set(e) for e in extras_by_dom.values()))
    for d, ns in names.items():
        missing = [c for c in core if c not in ns]
        dup = sorted({n for n in ns if ns.count(n) > 1})
        extra = [e for e in extras_by_dom[d] if e not in common]
        per[str(d)] = {"n_nodes": len(ns), "missing_core": missing, "duplicated": dup, "extra_vs_other_domains": extra,
                       "pass": not (missing or dup or extra)}
    return {"domains": per, "pass": all(v["pass"] for v in per.values())}


def gz_check(a_before: sc.Pose3, a_target: sc.Pose3, a_after: sc.Pose3,
             b_before: dict[str, sc.Pose3], b_after: dict[str, sc.Pose3]) -> dict[str, object]:
    """Tolerances = the M2 read-back ones (sc.POS_TOL_M / sc.ROT_TOL_DEG)."""
    moved = sc.pose_err(a_before, a_target)
    reached = sc.pose_err(a_after, a_target)
    b = {k: sc.pose_err(b_before[k], b_after[k]) for k in b_before}
    pos = {"a_target_vs_before": moved, "a_after_vs_target": reached,
           "pass": (not sc.pose_ok(moved)) and sc.pose_ok(reached)}
    neg = {"b_after_vs_before": b, "pass": all(sc.pose_ok(e) for e in b.values())}
    return {"positive_control": pos, "negative": neg, "pass": bool(pos["pass"] and neg["pass"])}


def ros_check(got_a: list[str], got_b: dict[str, list[str]], nonce: str, a_subs_matched: int) -> dict[str, object]:
    pos = {"a_received": got_a.count(nonce), "a_publisher_subscription_count": a_subs_matched,
           "pass": nonce in got_a}
    neg = {"b_received": {k: v.count(nonce) for k, v in got_b.items()}, "pass": all(nonce not in v for v in got_b.values())}
    return {"topic": MARKER_TOPIC, "nonce": nonce, "positive_control": pos, "negative": neg,
            "pass": bool(pos["pass"] and neg["pass"])}


def overall(nodes: dict, gz: dict, ros: dict, rtf: dict, restore_ok: bool) -> bool:
    return bool(nodes["pass"] and gz["pass"] and ros["pass"] and rtf["pass"] and restore_ok)


@contextmanager
def partition(p: str) -> Iterator[None]:
    """photos.gz_poses / sc.set_pose call the gz CLI with os.environ: point it at one sim's partition."""
    old = os.environ.get("GZ_PARTITION")
    os.environ["GZ_PARTITION"] = p
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("GZ_PARTITION")
        else:
            os.environ["GZ_PARTITION"] = old


def robot_pose(world: str, p: str) -> tuple[sc.Pose3, float]:
    with partition(p):
        poses, t = photos.gz_poses(world)
    return poses[photos.MODEL], t


def wait_sim(world: str, p: str, t0: float, dt: float) -> None:
    while robot_pose(world, p)[1] < t0 + dt:
        time.sleep(0.1)


def node_list(domain: int) -> list[str]:
    out = subprocess.run(["ros2", "node", "list", "--no-daemon", "--spin-time", str(SPIN_TIME_S)], check=True,
                         capture_output=True, text=True, timeout=SPIN_TIME_S + 60,
                         env={**os.environ, "ROS_DOMAIN_ID": str(domain)}).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def rtf(world: str, p: str) -> float:
    _, t0 = robot_pose(world, p)
    w0 = time.monotonic()
    time.sleep(RTF_WINDOW_S)
    _, t1 = robot_pose(world, p)
    return (t1 - t0) / (time.monotonic() - w0)


def marker(dom_a: int, doms_b: list[int]) -> dict[str, object]:
    got: dict[int, list[str]] = {d: [] for d in [dom_a, *doms_b]}
    ctxs, execs, nodes = [], [], {}
    for d in got:
        ctx = rclpy.Context()
        rclpy.init(context=ctx, domain_id=d)
        n = rclpy.create_node(f"m5_crosstalk_{d}", context=ctx)
        n.create_subscription(String, MARKER_TOPIC, lambda m, d=d: got[d].append(m.data), 10)
        ex = SingleThreadedExecutor(context=ctx)
        ex.add_node(n)
        ctxs.append(ctx)
        execs.append(ex)
        nodes[d] = n

    def spin(wall_s: float, until=lambda: False) -> None:  # noqa: ANN001
        end = time.monotonic() + wall_s
        while time.monotonic() < end and not until():
            for ex in execs:
                ex.spin_once(timeout_sec=0.02)

    pub = nodes[dom_a].create_publisher(String, MARKER_TOPIC, 10)
    try:
        spin(MATCH_TIMEOUT_S, lambda: pub.get_subscription_count() >= 1)
        spin(1.0)  # let any (cross-talk) subscriber in a B domain match as well
        matched = pub.get_subscription_count()
        nonce = uuid.uuid4().hex
        for _ in range(10):
            pub.publish(String(data=nonce))
            spin(0.1)
        spin(LISTEN_S)
    finally:
        for n in nodes.values():
            n.destroy_node()
        for ctx in ctxs:
            rclpy.shutdown(context=ctx)
    return ros_check(got[dom_a], {str(d): got[d] for d in doms_b}, nonce, matched)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sim", action="append", type=parse_sim, required=True, help="ROS_DOMAIN_ID:GZ_PARTITION")
    ap.add_argument("--world", default="clinic")
    args = ap.parse_args()
    assert len(args.sim) >= 2, "need at least two sims"
    args.out.mkdir(parents=True, exist_ok=True)
    (dom_a, part_a), others = args.sim[0], args.sim[1:]

    lists = {d: node_list(d) for d, _ in args.sim}
    nodes = node_check(lists)
    print("(a) nodes:", "PASS" if nodes["pass"] else "FAIL", flush=True)

    rtfs = {p: rtf(args.world, p) for _, p in args.sim}
    rtf_res = {"per_partition": rtfs, "min": RTF_MIN, "window_wall_s": RTF_WINDOW_S,
               "pass": all(v >= RTF_MIN for v in rtfs.values())}
    print("RTF:", rtfs, flush=True)

    a_before, t0 = robot_pose(args.world, part_a)
    b_before = {p: robot_pose(args.world, p)[0] for _, p in others}
    quarter = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    a_target = (a_before[0].copy(), photos.qmul(quarter, a_before[1]))
    with partition(part_a):
        sc.set_pose(args.world, photos.MODEL, a_target)
    wait_sim(args.world, part_a, t0, photos.SETTLE_S)
    a_after, t1 = robot_pose(args.world, part_a)
    b_after = {p: robot_pose(args.world, p)[0] for _, p in others}
    gz = gz_check(a_before, a_target, a_after, b_before, b_after)
    with partition(part_a):
        sc.set_pose(args.world, photos.MODEL, a_before)
    wait_sim(args.world, part_a, t1, photos.SETTLE_S)
    a_back, _ = robot_pose(args.world, part_a)
    restore = {"err": sc.pose_err(a_back, a_before)}
    restore["pass"] = sc.pose_ok(restore["err"])
    print("(b) gz:", "PASS" if gz["pass"] else "FAIL", "restore", restore, flush=True)

    ros = marker(dom_a, [d for d, _ in others])
    print("(c) ros:", "PASS" if ros["pass"] else "FAIL", flush=True)

    res = {"gate": "006 M5", "sims": [{"ROS_DOMAIN_ID": d, "GZ_PARTITION": p} for d, p in args.sim],
           "a": {"ROS_DOMAIN_ID": dom_a, "GZ_PARTITION": part_a}, "core_nodes": list(CORE_NODES),
           "node_lists": {str(d): v for d, v in lists.items()}, "nodes": nodes, "rtf": rtf_res,
           "gz": {**gz, "a_before": sc.pose_dict(a_before), "a_target": sc.pose_dict(a_target),
                  "a_after": sc.pose_dict(a_after)},
           "restore_a": restore, "ros": ros, "pass": overall(nodes, gz, ros, rtf_res, restore["pass"])}
    (args.out / "m5.json").write_text(json.dumps(res, indent=2))
    print("M5:", "PASS" if res["pass"] else "FAIL", flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
