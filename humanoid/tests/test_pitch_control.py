"""H-P2 G7 negative control camera (a recorded deviation addendum): `head_rgb_pitch` exists only when asked for; the default
scene is byte-identical to before; the pitched camera sits at CAM_POS with the level quaternion tilted 47.6 deg down."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from humanoid.room import CAM_NAME, CAM_POS, PITCH_CAM_NAME, PITCH_DEG, build_clinic_scene  # noqa: E402
from humanoid.tests.test_scene_mocap import G1_XML, needs_g1  # noqa: E402

# sha256 of the default build_clinic_scene XML at HEAD 46e... (humanoid/room.py before this edit), per world
GOLDEN_XML = {"closed": "18ce7827ca6e378f7fe21cbdf3c8fa8b36f39ff418259023b538d8520bf0b1c5",
              "cabinet_open": "47e6042973f0f2269e79030ee303156e8acaa36e0ac1f04dfe07690d52cbf4c4"}


@needs_g1
@pytest.mark.parametrize("world", sorted(GOLDEN_XML))
def test_default_scene_is_unchanged(world: str, tmp_path: Path) -> None:
    xml = build_clinic_scene(world, tmp_path, G1_XML)
    assert hashlib.sha256(xml.read_bytes()).hexdigest() == GOLDEN_XML[world]
    m = mujoco.MjModel.from_xml_path(str(xml))
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, PITCH_CAM_NAME) == -1


@needs_g1
def test_pitched_camera_pose(tmp_path: Path) -> None:
    m = mujoco.MjModel.from_xml_path(str(build_clinic_scene("closed", tmp_path, G1_XML, pitch_control=True)))
    lvl, pit = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in (CAM_NAME, PITCH_CAM_NAME))
    assert pit >= 0 and m.cam_bodyid[pit] == m.cam_bodyid[lvl] and m.body(m.cam_bodyid[pit]).name == "torso_link"
    assert m.cam_pos[pit] == pytest.approx(CAM_POS) and m.cam_fovy[pit] == pytest.approx(m.cam_fovy[lvl])
    fwd = lambda i: -np.array(m.cam_mat0[i]).reshape(3, 3)[:, 2]  # camera looks along its -z; body-frame direction  # noqa: E731
    assert fwd(lvl) == pytest.approx([1, 0, 0], abs=1e-6)
    a = math.radians(PITCH_DEG)
    assert PITCH_DEG == 47.6 and fwd(pit) == pytest.approx([math.cos(a), 0, -math.sin(a)], abs=1e-6)
    up = np.array(m.cam_mat0[pit]).reshape(3, 3)[:, 1]
    assert up == pytest.approx([math.sin(a), 0, math.cos(a)], abs=1e-6)  # still upright (no roll)
