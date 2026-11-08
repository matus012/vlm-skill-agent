"""H-P2 G7 negative control, live (a recorded deviation addendum): one g1_node --pitch-control on the local machine; the pitched
segmentation frame holds robot pixels, and the UNCHANGED G7 check (probes/phase1_checks.unlabelled_ok) fails on it
once robot ids go to label 0 through scene.json label_of_body. Level frames: 0 robot px (the reason for the control).
Takes the shared lock gz.lockdir (skipped if held); the node runs in its own process group."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes")]
LOCK = Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "gz.lockdir"
G1_XML = Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "ext/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"


@pytest.mark.skipif(not G1_XML.exists() or not Path("/opt/ros/jazzy").exists(), reason="needs the G1 model under $BUILD_DIR/ext (humanoid/setup.sh) and ROS 2 Jazzy")
def test_pitched_frames_have_robot_px_and_g7_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rclpy = pytest.importorskip("rclpy")
    import numpy as np
    from sensor_msgs.msg import Image

    import phase1_checks as pc
    from humanoid.probes.common import image_to_array
    monkeypatch.setenv("ROS_DOMAIN_ID", "18")
    try:
        LOCK.mkdir()
    except FileExistsError:
        pytest.skip(f"{LOCK} held by another run")
    proc = None
    try:
        proc = subprocess.Popen([str(ROOT / "humanoid/run_g1.sh"), str(ROOT / "humanoid/g1_node.py"), "--world", "closed",
                                 "--scene-dir", str(tmp_path / "scene"), "--rate", "1.0", "--pitch-control"],
                                stdout=(tmp_path / "node.log").open("w"), stderr=subprocess.STDOUT, start_new_session=True)
        rclpy.init()
        node = rclpy.create_node("pitch_tap")
        got: dict[str, list] = {"p": [], "l": []}
        node.create_subscription(Image, "/head_rgb_pitch/segmentation", lambda m: got["p"].append(image_to_array(m)), 2)
        node.create_subscription(Image, "/head_rgb/segmentation", lambda m: got["l"].append(image_to_array(m)), 2)
        end = time.monotonic() + 180
        while (len(got["p"]) < 3 or len(got["l"]) < 3) and time.monotonic() < end:
            assert proc.poll() is None, (tmp_path / "node.log").read_text()[-2000:]
            rclpy.spin_once(node, timeout_sec=0.5)
        assert len(got["p"]) >= 3 and len(got["l"]) >= 3, "no frames"
        sc = json.loads((tmp_path / "scene/scene.json").read_text())
        ids, lut = np.asarray(sc["robot_body_ids"]), np.asarray(sc["label_of_body"], np.uint8)
        pit, lvl = got["p"][-1], got["l"][-1]
        px = int(np.isin(pit, ids).sum())
        print(f"pitched frame: robot px {px} = {px / pit.size:.4f}; level frame robot px {int(np.isin(lvl, ids).sum())}; "
              f"unlabelled {pc.unlabelled_fraction(lut[pit]):.4f} (G7 bar {pc.G7_MAX_UNLABELLED_PERMILLE / 10} %)")
        assert pit.shape == (480, 640) and px > 0
        assert not pc.unlabelled_ok(lut[pit])  # the unchanged G7 check FAILS on the control frame
        node.destroy_node()
        rclpy.shutdown()
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        LOCK.rmdir()
