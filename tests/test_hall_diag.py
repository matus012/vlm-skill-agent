"""Pure helpers of probes/hall_diag.py (task 003-L): AMCL - GT split and the arrival summary. No sim."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
import hall_diag as hd  # noqa: E402


def test_loc_error_split_and_sign() -> None:
    e = hd.loc_error((3.0, -0.2, math.radians(10)), (2.7, -0.1, math.radians(-5)))
    assert e["dx_along_m"] == pytest.approx(0.3)
    assert e["dy_across_m"] == pytest.approx(-0.1)
    assert e["dxy_m"] == pytest.approx(math.hypot(0.3, 0.1))
    assert e["dyaw_deg"] == pytest.approx(15.0)


def test_loc_error_yaw_wraps() -> None:
    assert hd.loc_error((0, 0, math.radians(179)), (0, 0, math.radians(-179)))["dyaw_deg"] == pytest.approx(-2.0)


def row(status: str, gt_dist: float, dx: float = 0.0, dy: float = 0.0) -> dict:
    return {"status_name": status, "gt_dist_to_place_m": gt_dist,
            "err_tf": {"dx_along_m": dx, "dy_across_m": dy, "dyaw_deg": 1.0}}


def test_summarize_counts_strictly_below_and_succeeded_only() -> None:
    s = hd.summarize([row("SUCCEEDED", 0.05, -0.3, 0.02), row("SUCCEEDED", 0.20), row("ABORTED", 0.01, 0.1, -0.04)])
    assert s["n"] == 3 and s["n_succeeded"] == 2
    assert s["n_gt_below_accept"] == 1  # 0.20 is not < 0.20; ABORTED never passes
    assert not s["all_pass"]
    assert s["along_abs_max_m"] == pytest.approx(0.3)
    assert s["along_signed_mean_m"] == pytest.approx(-0.2 / 3)
    assert s["across_abs_max_m"] == pytest.approx(0.04)
    assert s["gt_dist_max_m"] == pytest.approx(0.20)


def test_summarize_all_pass() -> None:
    assert hd.summarize([row("SUCCEEDED", 0.19)] * 20)["all_pass"]
    assert hd.summarize([])["n"] == 0
