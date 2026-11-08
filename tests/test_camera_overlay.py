"""G1 static: the robot description the gz bringup builds carries the project camera (design notes: Sim camera)."""
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

if "AMENT_PREFIX_PATH" not in os.environ or shutil.which("xacro") is None:
    pytest.skip("ROS not sourced (source sim/env.sh)", allow_module_level=True)

from ament_index_python.packages import get_package_share_directory  # noqa: E402

# WS_LOCAL as in sim/env.sh (hpc/PERUN_SIM.md section 1): unset on the local machine, $PROJ/build on PERUN
OVERLAY = Path(os.environ.get("WS_LOCAL") or Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build"))) / "ros_ws/install/turtlebot4_description"
EXPECTED = ("<horizontal_fov>1.5708</horizontal_fov>", "<width>640</width>", "<height>480</height>")


def test_overlay_resolves() -> None:
    assert Path(get_package_share_directory("turtlebot4_description")).resolve() == (
        OVERLAY / "share/turtlebot4_description"
    ).resolve()


@pytest.mark.parametrize("model", ["standard", "lite"])
def test_full_robot_xacro_has_project_camera(model: str) -> None:
    # same xacro + args as turtlebot4_description/launch/robot_description.launch.py (used by turtlebot4_gz_bringup)
    xacro_file = Path(get_package_share_directory("turtlebot4_description")) / "urdf" / model / "turtlebot4.urdf.xacro"
    urdf = subprocess.run(
        ["xacro", str(xacro_file), "gazebo:=ignition", "namespace:="], check=True, capture_output=True, text=True
    ).stdout
    for line in EXPECTED:
        assert urdf.count(line) == 2, line  # rgbd_camera + seg_camera (task 002), checked per sensor below


def robot_urdf(model: str) -> str:
    xacro_file = Path(get_package_share_directory("turtlebot4_description")) / "urdf" / model / "turtlebot4.urdf.xacro"
    return subprocess.run(["xacro", str(xacro_file), "gazebo:=ignition", "namespace:="],
                          check=True, capture_output=True, text=True).stdout


def camera_spec(sensor: "ET.Element") -> tuple[str | None, ...]:
    c = sensor.find("camera")
    return tuple(c.findtext(p) for p in ("horizontal_fov", "image/width", "image/height", "clip/near", "clip/far"))


@pytest.mark.parametrize("model", ["standard", "lite"])
def test_segmentation_camera_matches_rgb(model: str) -> None:
    """Task 002: a semantic segmentation sensor on the RGB camera link, same fov / size / clip; RGB unchanged."""
    root = ET.fromstring(robot_urdf(model))
    sensors = {s.get("name"): (g.get("reference"), s) for g in root.iter("gazebo") for s in g.iter("sensor")}
    rgb_ref, rgb = sensors["rgbd_camera"]
    seg_ref, seg = sensors["seg_camera"]
    assert rgb.get("type") == "rgbd_camera" and seg.get("type") == "segmentation"
    assert rgb_ref == seg_ref == "oakd_rgb_camera_frame"
    assert seg.find("pose") is None and rgb.find("pose") is None  # both at the link origin
    assert camera_spec(rgb) == ("1.5708", "640", "480", "0.3", "100")  # RGB as decided (design notes Sim camera)
    assert camera_spec(seg) == camera_spec(rgb)
    assert seg.findtext("camera/segmentation_type") == "semantic"
    assert seg.findtext("topic") == "oakd/seg"
    assert rgb.findtext("camera/optical_frame_id") == "oakd_rgb_camera_optical_frame"
    assert rgb.findtext("update_rate") == seg.findtext("update_rate") == "5"  # sensor-load cut (2026-11-04)


@pytest.mark.parametrize("model", ["standard", "lite"])
def test_robot_description_is_a_yaml_string(model: str) -> None:
    """launch_ros yaml-parses parameter strings: a ': ' in an xacro comment made robot_description unparseable."""
    assert isinstance(yaml.safe_load(robot_urdf(model)), str)


@pytest.mark.parametrize("model", ["standard", "lite"])
def test_lidar_rate_cut(model: str) -> None:
    """Sensor-load cut (2026-11-04): rplidar 62 -> 10 Hz, samples unchanged (640)."""
    root = ET.fromstring(robot_urdf(model))
    lidar = next(s for g in root.iter("gazebo") for s in g.iter("sensor") if s.get("name") == "rplidar")
    assert float(lidar.findtext("update_rate")) == 10.0
    assert lidar.findtext("ray/scan/horizontal/samples") == "640" or lidar.findtext("lidar/scan/horizontal/samples") == "640"
