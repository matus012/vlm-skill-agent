"""Pure-math tests for the G1 lane (controller laws, pose conversion, camera intrinsics)."""
from __future__ import annotations

import math

import pytest

from humanoid.geometry import (
    Limits,
    fovy_for_hfov,
    grew,
    heading_cmd,
    intrinsics_from_fovy,
    quat_angle_deg,
    waypoint_cmd,
    wrap,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
    yaw_from_wxyz,
    yaw_to_wxyz,
)


def test_fovy_for_90_hfov_at_640x480_is_spec_value() -> None:
    assert fovy_for_hfov(math.pi / 2, 640, 480) == pytest.approx(73.7398, abs=1e-3)


def test_fx_is_320_for_spec_fovy() -> None:
    fx, fy, cx, cy = intrinsics_from_fovy(73.74, 640, 480)
    assert abs(fx - 320.0) <= 0.5 and fx == fy
    assert (cx, cy) == (319.5, 239.5)
    fx_exact, *_ = intrinsics_from_fovy(fovy_for_hfov(math.pi / 2, 640, 480), 640, 480)
    assert fx_exact == pytest.approx(320.0, abs=1e-9)


@pytest.mark.parametrize("yaw", [0.0, 0.5, -1.2, math.pi - 1e-6, -math.pi + 1e-3])
def test_yaw_roundtrip(yaw: float) -> None:
    assert yaw_from_wxyz(yaw_to_wxyz(yaw)) == pytest.approx(yaw, abs=1e-9)


def test_quat_order_roundtrip_and_angle() -> None:
    q = (0.9, 0.1, -0.3, 0.2)
    assert xyzw_to_wxyz(wxyz_to_xyzw(q)) == q
    assert wxyz_to_xyzw(q) == (0.1, -0.3, 0.2, 0.9)
    a, b = yaw_to_wxyz(0.0), yaw_to_wxyz(math.radians(10.0))
    assert quat_angle_deg(a, b) == pytest.approx(10.0, abs=1e-6)
    assert quat_angle_deg(b, tuple(-v for v in b)) == pytest.approx(0.0, abs=1e-6)  # q and -q are one rotation


def test_wrap() -> None:
    assert wrap(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
    assert wrap(0.1) == pytest.approx(0.1)


def test_waypoint_ahead_drives_forward_within_limits() -> None:
    vx, vy, wz = waypoint_cmd(0.0, 0.0, 0.0, 3.0, 0.0)
    assert vx == pytest.approx(Limits().vx) and vy == pytest.approx(0.0) and wz == pytest.approx(0.0)


def test_waypoint_is_body_frame_and_turns_toward_target() -> None:
    # Facing +y, target on world +x = on the robot's right: vy < 0, turn clockwise (wz < 0).
    vx, vy, wz = waypoint_cmd(0.0, 0.0, math.pi / 2, 2.0, 0.0)
    assert vx == pytest.approx(0.0, abs=1e-9) and vy < 0 and wz < 0
    assert abs(vy) <= Limits().vy and abs(wz) <= Limits().wz


def test_waypoint_saturation_keeps_direction() -> None:
    vx, vy, _ = waypoint_cmd(0.0, 0.0, 0.0, 4.0, 2.0)
    assert vy / vx == pytest.approx(0.5)
    assert max(abs(vx), abs(vy)) == pytest.approx(0.5)


def test_waypoint_no_spin_when_close() -> None:
    _, _, wz = waypoint_cmd(0.0, 0.0, 0.0, -0.2, 0.1)  # behind but within face_dist
    assert wz == 0.0


def test_heading_cmd_shortest_way_and_deadband_lift() -> None:
    assert heading_cmd(math.radians(170), math.radians(-170)) > 0  # +20 deg the short way, across +-pi
    assert heading_cmd(0.0, math.radians(2.0), min_wz=0.15) == pytest.approx(0.15)
    assert heading_cmd(0.0, math.radians(-2.0), min_wz=0.15) == pytest.approx(-0.15)
    assert heading_cmd(0.0, math.pi * 0.9) == pytest.approx(Limits().wz)
    assert heading_cmd(0.5, 0.5, min_wz=0.15) == 0.0


def test_fine_pulse_sign_and_size() -> None:
    from humanoid.geometry import fine_pulse
    assert fine_pulse(6.0) == (1.0, 0.3)
    assert fine_pulse(-6.0) == (-1.0, 0.3)
    wz, dur = fine_pulse(-2.5)
    assert wz == -0.4 and dur == pytest.approx(2.0)
    assert fine_pulse(0.2)[1] == 0.3   # never below the shortest pulse


def test_grew_signed_change_and_threshold() -> None:
    d, ok = grew(1.0, 1.3, 0.2)
    assert d == pytest.approx(0.3) and ok
    d, ok = grew(1.0, 1.1, 0.2)
    assert d == pytest.approx(0.1) and not ok
    d, ok = grew(1.0, 0.5, 0.2)  # moved the wrong way
    assert d == pytest.approx(-0.5) and not ok


def test_grew_wraps_angles_across_pi() -> None:
    d, ok = grew(math.radians(179.0), math.radians(-170.0), math.radians(10.0), angle=True)  # +11 deg through pi
    assert math.degrees(d) == pytest.approx(11.0) and ok
    d, ok = grew(math.radians(-170.0), math.radians(179.0), math.radians(10.0), angle=True)  # -11 deg
    assert math.degrees(d) == pytest.approx(-11.0) and not ok
