"""G1 live: the running sim camera reports the project spec (design notes: Sim camera) and frames are saved.

Run with the sim up (sim/launch_phase0.sh) and `source sim/env.sh`:
    python3 probes/camera_check.py [--out DIR]
Exit 0 = PASS (640x480, fx = 320 +- 0.5).
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

NS = "/oakd/rgb/preview"
WIDTH, HEIGHT, HFOV = 640, 480, 1.5708  # design notes: Decisions -> Sim camera
FX_EXPECTED = (WIDTH / 2) / math.tan(HFOV / 2)  # 320.0 (to 1e-3 at HFOV=1.5708)
FX_TOL = 0.5


class Grab(Node):
    def __init__(self) -> None:
        super().__init__("camera_check")
        self.msgs: dict[str, CameraInfo | Image] = {}
        for key, typ, topic in [("info", CameraInfo, "camera_info"), ("rgb", Image, "image_raw"), ("depth", Image, "depth")]:
            self.create_subscription(typ, f"{NS}/{topic}", lambda m, k=key: self.msgs.setdefault(k, m), 10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / ".cache/vlm_skill_agent/runs/2026-11-03")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Grab()
    while len(node.msgs) < 3:
        rclpy.spin_once(node, timeout_sec=30.0)
    info, rgb, depth = node.msgs["info"], node.msgs["rgb"], node.msgs["depth"]
    node.destroy_node()
    rclpy.shutdown()

    assert rgb.encoding == "rgb8" and depth.encoding == "32FC1", (rgb.encoding, depth.encoding)
    img = np.frombuffer(rgb.data, np.uint8).reshape(rgb.height, rgb.step)[:, : rgb.width * 3].reshape(rgb.height, rgb.width, 3)
    d = np.frombuffer(depth.data, np.float32).reshape(depth.height, depth.step // 4)[:, : depth.width]
    cv2.imwrite(str(args.out / "001-S-rgb.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    finite = np.isfinite(d)
    cv2.imwrite(str(args.out / "001-S-depth_mm.png"), np.where(finite, np.clip(d * 1000, 0, 65535), 0).astype(np.uint16))
    lo, hi = (float(np.min(d[finite])), float(np.percentile(d[finite], 99))) if finite.any() else (0.0, 1.0)
    vis = cv2.applyColorMap((np.clip((np.where(finite, d, hi) - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~finite] = 0
    cv2.imwrite(str(args.out / "001-S-depth_vis.png"), vis)

    fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
    result = {
        "topic": f"{NS}/camera_info",
        "frame_id": info.header.frame_id,
        "width": info.width, "height": info.height,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "fx_expected": FX_EXPECTED, "fx_tol": FX_TOL,
        "hfov_from_fx_deg": math.degrees(2 * math.atan((info.width / 2) / fx)),
        "rgb": {"width": rgb.width, "height": rgb.height, "encoding": rgb.encoding},
        "depth": {"width": depth.width, "height": depth.height, "encoding": depth.encoding,
                  "finite_frac": float(finite.mean()), "min_m": lo, "p99_m": hi},
    }
    result["pass"] = bool(info.width == WIDTH and info.height == HEIGHT and abs(fx - FX_EXPECTED) <= FX_TOL)
    print(json.dumps(result, indent=2))
    (args.out / "001-S-camera_check.json").write_text(json.dumps(result, indent=2))
    print("G1 live:", "PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
