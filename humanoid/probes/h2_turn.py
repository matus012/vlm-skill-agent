"""H2 turn: at one point, turn to absolute headings 0/90/180/270 deg; ground-truth yaw within +-3 deg of each after
settling (2 s sim at zero command), no fall (base z >= 0.5 m). 4/4.
Also saves one RGB + one segmentation frame per settled heading (H3 evidence; look at the RGB frames).
Order 90, 180, 270, 0 so every heading needs a 90 deg turn from the robot's spawn yaw 0.
Writes sideH-H2-turn.json and sideH-H3-rgb-<deg>.png / sideH-H3-seg-<deg>.png.
"""
from __future__ import annotations

import math

import imageio.v3 as iio
import rclpy

from humanoid.drive import Drive
from humanoid.geometry import wrap
from humanoid.probes.common import EVIDENCE, FrameTap, image_to_array, stamp_s, write_verdict

TOL_DEG, MIN_Z = 3.0, 0.5   # spec side-H, gate H2
HEADINGS_DEG = [90, 180, 270, 0]


def main() -> None:
    rclpy.init()
    d = Drive("h2_turn")
    tap = FrameTap(d)
    d.wait_pose()
    d.hold(1.0)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    rows = []
    for deg in HEADINGS_DEG:
        r = d.turn_to(math.radians(deg), timeout_s=30.0)
        t_settled = d.pose.t
        while tap.rgb is None or tap.seg is None or stamp_s(tap.rgb) < t_settled or stamp_s(tap.seg) < t_settled:
            d.send(0.0, 0.0, 0.0)
            rclpy.spin_once(d, timeout_sec=0.5)
        err = math.degrees(wrap(math.radians(deg) - d.pose.yaw))  # yaw at the frame grab, after settling
        rgb_path, seg_path = EVIDENCE / f"sideH-H3-rgb-{deg:03d}.png", EVIDENCE / f"sideH-H3-seg-{deg:03d}.png"
        iio.imwrite(rgb_path, image_to_array(tap.rgb))
        iio.imwrite(seg_path, image_to_array(tap.seg))
        r.update({"err_at_frame_deg": err, "pass": abs(r["settled_err_deg"]) <= TOL_DEG and abs(err) <= TOL_DEG,
                  "rgb": rgb_path.name, "seg": seg_path.name, "frame_stamp_s": stamp_s(tap.rgb)})
        rows.append(r)
    n_ok = sum(r["pass"] for r in rows)
    no_fall = d.pose.min_z >= MIN_Z
    write_verdict("H2-turn", {
        "gate": "absolute headings 0/90/180/270, yaw within +-3 deg after 2 s settle, no fall",
        "headings": rows, "headings_passed": n_ok, "min_base_z_odom_50hz": d.pose.min_z,
        "position_drift_m": math.hypot(d.pose.history[-1][1] - d.pose.history[0][1],
                                       d.pose.history[-1][2] - d.pose.history[0][2]),
        "verdict": "PASS" if n_ok == 4 and no_fall else "FAIL"})
    d.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
