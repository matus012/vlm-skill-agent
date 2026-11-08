"""H3 camera: 640x480 on a ROS topic, fx = 320 +- 0.5 (camera_info), segmentation frames available, one RGB frame per
heading saved (by h2_turn.py; a human/agent must still LOOK at them — this probe only checks they exist).
Listens 10 s wall to a running g1_node. Writes sideH-H3-camera.json.
"""
from __future__ import annotations

import time

import imageio.v3 as iio
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS
from sensor_msgs.msg import CameraInfo, Image

from humanoid.probes.common import EVIDENCE, image_to_array, stamp_s, write_verdict

W, H, FX, FX_TOL = 640, 480, 320.0, 0.5   # spec side-H, gate H3 (project camera decision 2026-11-03)
LISTEN_S = 10.0


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("h3_camera")
    got: dict[str, list] = {"rgb": [], "seg": [], "info": []}
    node.create_subscription(Image, "/head_rgb/image_raw", lambda m: got["rgb"].append((time.time(), m)), 2)
    node.create_subscription(Image, "/head_rgb/segmentation", lambda m: got["seg"].append((time.time(), m)), 2)
    node.create_subscription(CameraInfo, "/head_rgb/camera_info", lambda m: got["info"].append((time.time(), m)), SENSOR_QOS)
    t_end = time.time() + LISTEN_S
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    info = got["info"][-1][1]
    rgb = image_to_array(got["rgb"][-1][1])
    seg = image_to_array(got["seg"][-1][1])

    def rates(key: str) -> dict:
        wall = [w for w, _ in got[key]]
        sim = [stamp_s(m) for _, m in got[key]]
        return {"n": len(wall), "hz_wall": (len(wall) - 1) / (wall[-1] - wall[0]),
                "hz_sim": (len(sim) - 1) / (sim[-1] - sim[0])}

    saved = sorted(EVIDENCE.glob("sideH-H3-rgb-*.png"))
    saved_shapes = {p.name: list(iio.imread(p).shape) for p in saved}
    checks = {
        "rgb_640x480_rgb8": rgb.shape == (H, W, 3) and got["rgb"][-1][1].encoding == "rgb8",
        "camera_info_640x480": (info.width, info.height) == (W, H),
        "fx_320_pm_0.5": abs(info.k[0] - FX) <= FX_TOL,
        "segmentation_available": seg.shape == (H, W) and len(np.unique(seg)) > 1,
        "rgb_per_heading_saved": len(saved) == 4 and all(s == [H, W, 3] for s in saved_shapes.values()),
    }
    write_verdict("H3-camera", {
        "gate": "640x480 on ROS, fx 320+-0.5, segmentation available, one RGB per heading saved and looked at",
        "camera_info": {"width": info.width, "height": info.height, "fx": info.k[0], "fy": info.k[4],
                        "cx": info.k[2], "cy": info.k[5], "frame_id": info.header.frame_id},
        "topics": {"rgb": "/head_rgb/image_raw", "seg": "/head_rgb/segmentation (16UC1 body ids)",
                   "info": "/head_rgb/camera_info"},
        "rates": {k: rates(k) for k in got},
        "seg_body_ids_in_last_frame": np.unique(seg).tolist(),
        "rgb_frames_saved": saved_shapes,
        "looked_at": "not by this probe: see the side-H report (Read tool on each PNG)",
        "checks": {k: bool(v) for k, v in checks.items()},
        "verdict": "PASS" if all(checks.values()) else "FAIL"})
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
