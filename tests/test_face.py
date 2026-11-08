"""probes/photos.face (006 M2 round 2): rounds of turn + settle + relocalize, then final turns WITHOUT relocalizing
until the control residual is <= TOL_RETURN (at most FINAL_STEPS). A fake node: turn_to lands within `land` rad of the
target, relocalize shifts the estimate by the next value of `jumps` (AMCL re-estimate)."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import photos  # noqa: E402

D = math.radians


class FakeNode:
    def __init__(self, jumps: list[float], land: float = D(0.1)) -> None:
        self.yaw, self.jumps, self.land = 0.0, list(jumps), land
        self.calls: list[str] = []

    def turn_to(self, target: float) -> None:
        self.calls.append("turn")
        self.yaw = target + self.land

    def settle(self) -> float:
        self.calls.append("settle")
        return float(len(self.calls))

    def relocalize(self) -> None:
        self.calls.append("reloc")
        self.yaw += self.jumps.pop(0) if self.jumps else 0.0

    def loc_yaw(self) -> float:
        return self.yaw


def test_good_first_round_no_final_turn() -> None:
    n = FakeNode([D(0.1)])
    r = photos.face(n, 0.5, 3)
    assert r["rounds"] == 1 and r["final_steps"] == 0 and n.calls == ["turn", "settle", "reloc"]
    assert abs(r["residual_rad"] - D(0.2)) < 1e-12


def test_jumps_every_round_then_final_turn_fixes_residual() -> None:
    """Old loop: 3 rounds, each relocalize jumps 0.8 deg, ends with a 0.9 deg residual (the 143786 storage case)."""
    n = FakeNode([D(0.8)] * 3)
    r = photos.face(n, 0.5, 3)
    assert r["rounds"] == 3 and r["final_steps"] == 1
    assert n.calls[-3:] == ["reloc", "turn", "settle"] and n.calls.count("reloc") == 3  # no relocalize after
    assert abs(r["residual_rad"]) <= photos.TOL_RETURN and r["settled_sim_s"] == len(n.calls)


def test_final_turns_bounded() -> None:
    n = FakeNode([D(0.8)] * 3, land=D(0.5))  # turn_to never gets inside 0.3 deg
    r = photos.face(n, 0.0, 3)
    assert r["final_steps"] == photos.FINAL_STEPS == 3 and abs(r["residual_rad"]) > photos.TOL_RETURN
    assert n.calls.count("turn") == 3 + 3 and n.calls.count("reloc") == 3


def test_rounds_respected() -> None:
    n = FakeNode([D(0.8)] * 5)
    assert photos.face(n, 0.0, 2)["rounds"] == 2 and n.calls.count("reloc") == 2
