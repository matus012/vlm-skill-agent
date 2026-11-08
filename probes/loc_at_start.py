#!/usr/bin/env python3
"""Task 002 item 5: AMCL localizes from the spawn pose through its own set_initial_pose / initial_pose params (no
/initialpose publish anywhere). Run right after `sim/launch_clinic.sh localization:=true nav2:=true` starts: waits for
map->base_link, then compares AMCL's map pose with the Gazebo GT pose at the first TF and after 5 s sim.

    source sim/env.sh; python3 probes/loc_at_start.py --out DIR
"""
import argparse
import json
import math
from pathlib import Path

from rclpy.time import Time

from map_world_check import gt_pose, make_robot, wrap


def sample(robot) -> dict:  # noqa: ANN001
    gt, t = gt_pose("clinic")
    m = robot.map_pose()
    return {"sim_s": t, "gt_world": gt, "amcl_map": m, "dxy_m": math.dist(gt[:2], m[:2]),
            "dyaw_deg": math.degrees(abs(wrap(gt[2] - m[2])))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    robot = make_robot()
    assert robot.spin_until(lambda: robot.tf.can_transform("map", "base_link", Time()), 300.0), "no map->base_link"
    first = sample(robot)
    robot.spin_for_sim(5.0)
    res = {"method": "no /initialpose publish; AMCL params set_initial_pose: true, initial_pose = spawn pose "
                     "(sim/launch_clinic.sh writes them)", "first_tf": first, "after_5s": sample(robot)}
    print(json.dumps(res, indent=2))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "loc_at_start.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
