"""Shared bits for the side-H probes: evidence paths, verdict writing, camera-frame capture."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image

EVIDENCE = Path(os.environ.get("SIDEH_EVIDENCE", str(Path.home() / ".cache/vlm_skill_agent/runs/2026-11-04")))
SCENE_JSON = Path(os.environ.get("G1_SCENE_DIR", str(Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "g1_scene"))) / "scene.json"


def write_verdict(name: str, verdict: dict) -> Path:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verdict = {"probe": name, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **verdict}
    path = EVIDENCE / f"sideH-{name}.json"
    path.write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    return path


def image_to_array(m: Image) -> np.ndarray:
    dtype, ch = {"rgb8": (np.uint8, 3), "16UC1": (np.uint16, 1)}[m.encoding]
    a = np.frombuffer(bytes(m.data), dtype=dtype).reshape(m.height, m.step // np.dtype(dtype).itemsize)
    return a[:, : m.width * ch].reshape(m.height, m.width, ch).squeeze()


def stamp_s(m: Image) -> float:
    return m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec


class FrameTap:
    """Keeps the latest RGB and segmentation frames from the head camera."""

    def __init__(self, node: Node) -> None:
        self.rgb: Image | None = None
        self.seg: Image | None = None
        node.create_subscription(Image, "/head_rgb/image_raw", lambda m: setattr(self, "rgb", m), 2)
        node.create_subscription(Image, "/head_rgb/segmentation", lambda m: setattr(self, "seg", m), 2)


def robot_rows(seg: np.ndarray, robot_ids: list[int]) -> dict:
    """Rows of a body-id segmentation frame where the robot's own body appears."""
    mask = np.isin(seg, robot_ids)
    rows = np.flatnonzero(mask.any(axis=1))
    h = seg.shape[0]
    top = int(rows[rows < h // 2].max() + 1) if (rows < h // 2).any() else 0
    bottom = int(h - rows[rows >= h // 2].min()) if (rows >= h // 2).any() else 0
    return {"rows_with_robot": int(rows.size), "top_px": top, "bottom_px": bottom,
            "robot_pixel_frac": float(mask.mean())}
