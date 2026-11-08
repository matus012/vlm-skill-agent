#!/usr/bin/env python3
"""Task 002 map step: with `sim/launch_clinic.sh slam:=true nav2:=true` up, prove the robot drives from spawn (dock
parked outside, no undock), tour the hall + all 4 rooms with NavigateToPose waypoints (map frame == world frame:
SLAM's map starts at the spawn pose, world origin yaw 0), then save the map through slam_toolbox's save_map service.

    source sim/env.sh; python3 probes/map_tour.py --out DIR [--save sim/maps/clinic]
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

from geometry_msgs.msg import TwistStamped

from map_world_check import gt_pose, make_robot

ROOT = Path(__file__).resolve().parents[1]
P2 = math.pi / 2
# hall -> storage -> hall -> office -> hall -> ward -> hall -> reception -> hall (doorways: storage x 3.6, office x 1.2,
# ward x -1.2, reception x -3.6; layout.yaml)
WAYPOINTS = [
    ("hall_e", (3.5, 0.0, -P2)), ("storage_door", (3.6, -2.4, -P2)), ("storage_c", (2.5, -3.9, math.pi)),
    ("storage_w", (1.6, -4.9, math.pi)), ("storage_out", (3.6, -2.2, P2)), ("hall_ne", (1.2, 0.4, P2)),
    ("office_door", (1.2, 2.5, P2)), ("office_c", (2.5, 3.8, 0.0)), ("office_ne", (3.4, 4.8, 0.0)),
    ("office_out", (1.2, 2.4, -P2)), ("hall_nw", (-1.2, 0.4, P2)), ("ward_door", (-1.2, 2.5, P2)),
    ("ward_c", (-2.5, 2.6, math.pi)), ("ward_w", (-3.6, 3.6, math.pi)), ("ward_out", (-1.2, 2.4, -P2)),
    ("hall_w", (-3.6, 0.0, -P2)), ("reception_door", (-3.6, -2.4, -P2)), ("reception_c", (-2.5, -3.9, 0.0)),
    ("reception_s", (-2.5, -4.8, 0.0)), ("reception_out", (-3.6, -2.2, P2)), ("hall", (0.0, 0.0, 0.0)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save", default=str(ROOT / "sim/maps/clinic"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    robot = make_robot()
    log: dict = {}
    # drive test: 0.1 m/s for 3 s sim, no undock
    cmd = robot.create_publisher(TwistStamped, "/cmd_vel", 10)
    gt0, t0 = gt_pose("clinic")
    robot.spin_for_sim(0.5)
    t_start = robot.now_s()
    while robot.now_s() - t_start < 3.0:
        m = TwistStamped()
        m.header.stamp = robot.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.twist.linear.x = 0.1
        cmd.publish(m)
        robot.spin_for_sim(0.05)
    cmd.publish(TwistStamped())
    robot.spin_for_sim(1.0)
    gt1, t1 = gt_pose("clinic")
    log["drive_test"] = {"gt_before": gt0, "gt_after": gt1, "moved_m": math.dist(gt0[:2], gt1[:2]),
                         "cmd": "0.1 m/s x 3 s sim, /cmd_vel TwistStamped, no undock", "sim_s": [t0, t1]}
    print("drive test:", log["drive_test"], flush=True)
    log["waypoints"] = []
    for name, wp in WAYPOINTS:
        r = robot.navigate(wp)
        gt, _ = gt_pose("clinic")
        r.update(name=name, goal=wp, gt_end=gt, err_m=math.dist(gt[:2], wp[:2]))
        log["waypoints"].append(r)
        print(f"{name:15s} {r['status_name']:10s} sim {r['sim_s']:6.1f} s  GT err {r['err_m']:.2f} m", flush=True)
    robot.spin_for_sim(3.0)
    cmdline = ["ros2", "service", "call", "/slam_toolbox/save_map", "slam_toolbox/srv/SaveMap", f"{{name: {{data: '{a.save}'}}}}"]
    log["save"] = subprocess.run(cmdline, capture_output=True, text=True, timeout=60).stdout
    print(log["save"])
    (a.out / "map_tour.json").write_text(json.dumps(log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
