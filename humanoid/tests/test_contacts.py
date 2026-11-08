"""humanoid/contacts.py on a tiny MJCF: a robot standing on its foot counts nothing (a non-foot part on the floor does); pushed into a mocap box it counts
mocap contacts, into a static box static contacts; clearance from the pelvis geom."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from humanoid.contacts import ContactCounter, floor_geoms

XML = """<mujoco><worldbody>
  <body name="clinic/structure_floor"><geom type="box" size="5 5 0.05" pos="0 0 -0.05"/></body>
  <body name="pelvis" pos="0 0 0.25"><freejoint/><geom type="box" size="0.1 0.1 0.1"/>
    <body name="foot" pos="0 0 -0.175"><geom type="box" size="0.1 0.1 0.075"/></body>
  </body>
  <body name="clinic/chair" mocap="true" pos="0.5 0 0.1"><geom type="box" size="0.1 0.1 0.1"/></body>
  <body name="clinic/structure_walls" pos="-0.5 0 0.1"><geom type="box" size="0.1 0.1 0.1"/></body>
  <body name="clinic/structure_ceiling"><geom type="box" size="5 5 0.05" pos="0 0 3" contype="0" conaffinity="0"/></body>
</worldbody></mujoco>"""


def run(x: float, steps: int = 20, floor: np.ndarray | None = None, feet: list[str] = ("foot",)
        ) -> tuple[ContactCounter, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    c = ContactCounter(m, [m.body("pelvis").id, m.body("foot").id], floor, [m.body(f).id for f in feet])
    d.qpos[0] = x
    for _ in range(steps):
        mujoco.mj_step(m, d)
        c.scan(d)
    return c, d


def test_floor_only_counts_nothing() -> None:
    c, d = run(0.0)
    assert d.ncon > 0 and (c.n_mocap, c.n_static) == (0, 0)


def test_box_pushed_into_mocap_counts_mocap() -> None:
    c, _ = run(0.35)  # robot box [0.25, 0.45] overlaps the chair [0.4, 0.6]
    assert c.n_mocap > 0 and c.n_static == 0


def test_box_pushed_into_wall_counts_static() -> None:
    c, _ = run(-0.35)
    assert c.n_static > 0 and c.n_mocap == 0


def test_scan_reports_bodies() -> None:
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    c = ContactCounter(m, [m.body("pelvis").id, m.body("foot").id], foot_body_ids=[m.body("foot").id])
    d.qpos[0] = 0.35
    mujoco.mj_forward(m, d)
    rows = c.scan(d)
    assert rows and all(m.body(rb).name in ("pelvis", "foot") and m.body(ob).name == "clinic/chair" and mo and dist <= 0
                        for rb, ob, mo, dist in rows)


def test_floor_mask_is_the_floor_body() -> None:
    m = mujoco.MjModel.from_xml_string(XML)
    assert [m.body(m.geom_bodyid[g]).name for g in np.flatnonzero(floor_geoms(m))] == ["clinic/structure_floor"]


def test_floor_not_excluded_would_count() -> None:
    """The floor exclusion is what keeps the resting case at zero (the floor-not-excluded mutant counts)."""
    m = mujoco.MjModel.from_xml_string(XML)
    c, _ = run(0.0, floor=np.zeros(m.ngeom, bool))
    assert c.n_static > 0


def test_clearance() -> None:
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    c = ContactCounter(m, [m.body("pelvis").id, m.body("foot").id])
    mujoco.mj_forward(m, d)
    assert c.clearance(d, 4.0) == pytest.approx(0.3, abs=1e-6)  # box face 0.1 to chair / wall face 0.4: 0.3; floor ignored
    assert c.clearance(d, 0.25) is None
    assert m.body(m.geom_bodyid[c.other_geoms[0]]).name == "clinic/chair" and len(c.other_geoms) == 2


def test_non_foot_part_on_the_floor_counts() -> None:
    """design review: only foot-floor contacts are excluded; the same body declared not a foot touching the floor counts (static)."""
    c, _ = run(0.0, feet=[])
    assert c.n_static > 0 and c.n_mocap == 0


def test_g1_feet_by_name() -> None:
    from humanoid.contacts import FOOT_SUFFIXES
    assert FOOT_SUFFIXES == ("ankle_roll_link", "ankle_pitch_link")
