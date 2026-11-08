"""Self-view (no bar): rows of the head camera where the G1's own body appears, from segmentation body ids, over
standing, walking and turning. Proposed self_crop = max over frames + 8 px margin (design notes crop rule), 0 if the
robot never appears. Writes sideH-selfview.json and sideH-selfview-worst.png (robot pixels white).
"""
from __future__ import annotations

import json
import math

import imageio.v3 as iio
import numpy as np
import rclpy
from sensor_msgs.msg import Image

from humanoid.drive import Drive
from humanoid.probes.common import EVIDENCE, SCENE_JSON, image_to_array, robot_rows, write_verdict

MARGIN_PX = 8   # design notes, self-view crop rule


def main() -> None:
    robot_ids = json.loads(SCENE_JSON.read_text())["robot_body_ids"]
    rclpy.init()
    d = Drive("selfview")
    phase = ["stand"]
    frames: list[tuple[str, dict, np.ndarray]] = []

    def on_seg(m: Image) -> None:
        seg = image_to_array(m)
        frames.append((phase[0], robot_rows(seg, robot_ids), seg))

    d.create_subscription(Image, "/head_rgb/segmentation", on_seg, 2)
    d.wait_pose()
    d.hold(3.0)
    phase[0] = "walk"
    d.goto(1.5, -1.0, timeout_s=60.0, gate_m=0.30)
    phase[0] = "turn"
    for deg in (90, 180, 270, 0):
        d.turn_to(math.radians(deg), timeout_s=30.0, max_rounds=1)
    phase[0] = "walk_sideways"
    d.goto(1.5, 0.5, timeout_s=60.0, gate_m=0.30)  # facing +x, target to the left: mostly vy
    d.hold(1.0)
    summary = {}
    for ph in ("stand", "walk", "turn", "walk_sideways"):
        rs = [r for p, r, _ in frames if p == ph]
        summary[ph] = {"frames": len(rs), "max_top_px": max(r["top_px"] for r in rs),
                       "max_bottom_px": max(r["bottom_px"] for r in rs),
                       "frames_with_robot": sum(r["rows_with_robot"] > 0 for r in rs),
                       "max_robot_pixel_frac": max(r["robot_pixel_frac"] for r in rs)}
    top = max(s["max_top_px"] for s in summary.values())
    bottom = max(s["max_bottom_px"] for s in summary.values())
    worst = max(frames, key=lambda f: f[1]["robot_pixel_frac"])
    iio.imwrite(EVIDENCE / "sideH-selfview-worst.png", (np.isin(worst[2], robot_ids) * 255).astype(np.uint8))
    write_verdict("selfview", {
        "measure": "rows with the G1's own body in the head camera (segmentation body ids), no bar",
        "robot_body_ids": robot_ids, "by_phase": summary, "max_top_px": top, "max_bottom_px": bottom,
        "proposed_self_crop": {"top_px": top + MARGIN_PX if top else 0, "bottom_px": bottom + MARGIN_PX if bottom else 0},
        "worst_frame": {"phase": worst[0], **worst[1]}})
    d.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
