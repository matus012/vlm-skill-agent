"""Pure parts of adapters/turtlebot4.py: Nav2 status -> GoResult status, frame crop + label fusion. No sim.
The module imports rclpy (it is the ROS adapter): run with `source sim/env.sh`, else skipped."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("rclpy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters import turtlebot4 as tb4  # noqa: E402


@pytest.mark.parametrize("name,status,nav", [("SUCCEEDED", "reached", "SUCCEEDED"), ("ABORTED", "blocked", "ABORTED"),
                                             ("CANCELED", "blocked", "CANCELED"), ("REJECTED", "blocked", "REJECTED"),
                                             ("SIM_TIMEOUT", "blocked", "TIMEOUT"), ("2", "blocked", "2")])
def test_go_status(name: str, status: str, nav: str) -> None:
    assert tb4.go_status(name) == (status, nav)


def test_wall_hang_is_not_blocked() -> None:
    with pytest.raises(AssertionError):
        tb4.go_status("WALL_HANG")


def test_crop_frame_shape_and_labels() -> None:
    crop = yaml.safe_load((ROOT / "config/robots/turtlebot4.yaml").read_text())["self_crop"]
    top, bottom = crop["top_px"], crop["bottom_px"]
    h, w = 480, 640
    rgb = np.arange(h * w * 3, dtype=np.uint32).reshape(h, w, 3).astype(np.uint8)
    seg = np.zeros((h, w, 3), np.uint8)
    seg[:] = 7                       # every pixel label 7 (R == G == B == label)
    seg[200:210, 100:110] = 42
    depth = np.ones((h, w), np.float32)
    depth[250:260, 300:320] = -np.inf  # robot self-view: no depth -> label 0
    rgb_c, lab_c = tb4.crop_frame(rgb, seg, depth, top, bottom)
    assert rgb_c.shape == (h - top - bottom, w, 3) and lab_c.shape == (h - top - bottom, w)
    assert np.array_equal(rgb_c, rgb[top: h - bottom])          # rows top..h-bottom, nothing else
    assert lab_c.dtype == np.uint8 and lab_c.flags["C_CONTIGUOUS"]
    assert (lab_c[200 - top: 210 - top, 100:110] == 42).all()
    assert (lab_c[250 - top: 260 - top, 300:320] == 0).all()
    assert int((lab_c == 7).sum()) == lab_c.size - 100 - 200
