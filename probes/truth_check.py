#!/usr/bin/env python3
"""G5 live half (task 002): every truth.json object exists in the running clinic sim at its stated pose and the sim has
no extra models. Reads Gazebo directly (`gz model --list`, `/world/<w>/pose/info`), not ROS.

    source sim/env.sh (same ROS_DOMAIN_ID / GZ_PARTITION as the sim); python3 probes/truth_check.py [--truth ...]

Allowed extra models: structure_* (walls, floor, ceiling, trims), the robot (turtlebot4) and its dock (standard_dock).
The dock is exempt only where sim/launch_clinic.sh parks it: (20, 20) +- 0.05 m (outside the building).
Pass = 0 missing, 0 extra, every position error <= 0.01 m and every yaw error <= 0.01 rad, dock parked, and the world
file the running `gz sim` loaded found (its folder + sha256 recorded). Exit 0 on pass, 1 on fail.
`--json PATH` also writes the dock pose, the sim's world folder and the clinic.sdf sha256 (phase1_gates.py g5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import gz_cli

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PREFIXES = ("structure_",)
ALLOWED_NAMES = {"turtlebot4", "standard_dock"}
DOCK_XY, DOCK_TOL_M = (20.0, 20.0), 0.05  # sim/launch_clinic.sh set_pose; tolerance: design review (002 G5 dock)


def gz_models() -> list[str]:
    out = subprocess.run(["gz", "model", "--list"], capture_output=True, text=True, check=True, timeout=30).stdout
    return sorted(line.strip()[2:].strip() for line in out.splitlines() if line.strip().startswith("- "))


def gz_poses(world: str) -> dict[str, list[dict]]:
    out = gz_cli.echo_once(f"/world/{world}/pose/info", 30)  # signal deaths retried (a recorded deviation)
    by_name: dict[str, list[dict]] = {}
    for p in json.loads(out)["pose"]:
        by_name.setdefault(p["name"], []).append(p)
    return by_name


def yaw_of(q: dict) -> float:
    x, y, z, w = (q.get(k, 0.0) for k in "xyzw")
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def dock_check(poses: dict[str, list[dict]]) -> tuple[list[float] | None, str | None]:
    """(dock position, problem or None): exactly one standard_dock, within DOCK_TOL_M of DOCK_XY (x, y)."""
    cands = poses.get("standard_dock", [])
    if len(cands) != 1:
        return None, f"standard_dock: {len(cands)} pose entries"
    pos = cands[0].get("position", {})
    xyz = [pos.get(k, 0.0) for k in "xyz"]
    d = math.dist(xyz[:2], DOCK_XY)
    return xyz, None if d <= DOCK_TOL_M else f"standard_dock at {xyz}, {d:.3f} m from parked {DOCK_XY} (tol {DOCK_TOL_M})"


def resolve_world(arg: str, cwd: Path, resource_path: str) -> Path | None:
    """Where gz sim finds a world file argument: as given (absolute or relative to its cwd), else the first
    GZ_SIM_RESOURCE_PATH entry holding it (sim/launch_clinic.sh puts worlddir first)."""
    for cand in [cwd / arg] + [Path(d) / arg for d in resource_path.split(":") if d]:
        if cand.is_file():
            return cand.resolve()
    return None


def cmd_argv(raw: bytes) -> list[str]:
    """argv from a /proc/<pid>/cmdline. `gz` is a ruby script that rewrites its process title, so the server's
    cmdline is one space-joined string plus NUL padding, not NUL-separated args: split on both."""
    return " ".join(a.decode() for a in raw.split(b"\0") if a).split()


def world_mismatch(sdf: Path | None, expected: Path) -> str | None:
    """None if the running sim loaded a file byte-identical to `expected` (the clinic.sdf next to the truth.json
    being checked), else the problem: a sim started from another world folder must not pass G5."""
    if sdf is None:
        return "world file of the running gz sim not found"
    if not expected.is_file():
        return f"expected world file {expected} missing"
    a, b = (hashlib.sha256(p.read_bytes()).hexdigest() for p in (sdf, expected))
    return None if a == b else f"sim loaded {sdf} (sha256 {a[:12]}) != {expected} (sha256 {b[:12]})"


def sim_world_file(partition: str) -> Path | None:
    """The .sdf loaded by the running `gz sim -s` server of this GZ_PARTITION (read from its /proc entry)."""
    found: list[Path | None] = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:  # processes vanish and other users' are unreadable while we scan
            argv = cmd_argv((d / "cmdline").read_bytes())
            env = dict(e.decode().split("=", 1) for e in (d / "environ").read_bytes().split(b"\0") if b"=" in e)
            cwd = Path(os.readlink(d / "cwd"))
        except OSError:
            continue
        if "sim" in argv and "-s" in argv and argv[-1].endswith(".sdf") and env.get("GZ_PARTITION") == partition:
            found.append(resolve_world(argv[-1], cwd, env.get("GZ_SIM_RESOURCE_PATH", "")))
    return found[0] if len(set(found)) == 1 else None  # a forked server shares argv / env: same file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=Path, default=ROOT / "sim/worlds/clinic/generated/truth.json")
    ap.add_argument("--tol-m", type=float, default=0.01)
    ap.add_argument("--tol-rad", type=float, default=0.01)
    ap.add_argument("--json", type=Path, default=None, help="also write dock pose + sim world file record here")
    a = ap.parse_args()
    truth = json.loads(a.truth.read_text())
    world = truth["world"]
    models = gz_models()
    poses = gz_poses(world)
    want = {o["model_name"]: o for o in truth["objects"]}
    missing = sorted(set(want) - set(models))
    extra = sorted(m for m in set(models) - set(want) if not m.startswith(ALLOWED_PREFIXES) and m not in ALLOWED_NAMES)
    allowed = sorted(m for m in set(models) - set(want) if m not in extra)
    rows, worst_m, worst_rad, bad = [], 0.0, 0.0, []
    for name in sorted(set(want) & set(models)):
        cands = poses.get(name, [])
        if len(cands) != 1:
            bad.append(f"{name}: {len(cands)} pose entries named like this")
            continue
        p = cands[0]
        pos = p.get("position", {})
        x, y, z = (pos.get(k, 0.0) for k in "xyz")
        tx, ty, tz, _, _, tyaw = want[name]["pose"]
        dm = math.dist((x, y, z), (tx, ty, tz))
        dyaw = abs(math.atan2(math.sin(yaw_of(p.get("orientation", {})) - tyaw), math.cos(yaw_of(p.get("orientation", {})) - tyaw)))
        worst_m, worst_rad = max(worst_m, dm), max(worst_rad, dyaw)
        ok = dm <= a.tol_m and dyaw <= a.tol_rad
        if not ok:
            bad.append(f"{name}: pos err {dm:.4f} m, yaw err {dyaw:.4f} rad")
        rows.append(f"  {'ok ' if ok else 'BAD'} {name:18s} sim=({x:.4f},{y:.4f},{z:.4f}) truth=({tx},{ty},{tz}) err={dm:.5f} m yaw_err={dyaw:.5f} rad")
    dock, dock_bad = dock_check(poses)
    if dock_bad:
        bad.append(dock_bad)
    sdf = sim_world_file(os.environ.get("GZ_PARTITION", ""))
    expected_sdf = a.truth.parent / "clinic.sdf"
    world_bad = world_mismatch(sdf, expected_sdf)
    if world_bad:
        bad.append(f"{world_bad} (GZ_PARTITION={os.environ.get('GZ_PARTITION')})")
    sdf_sha = hashlib.sha256(sdf.read_bytes()).hexdigest() if sdf else None
    passed = not missing and not extra and not bad
    print(f"truth_check world={world} truth={a.truth} layout_sha256={truth['layout_sha256']}")
    print(f"sim models: {len(models)}; truth objects: {len(want)}; allowed non-object models: {allowed}")
    print("\n".join(rows))
    print(f"missing (in truth, not in sim): {missing}")
    print(f"extra (in sim, not in truth, not allowed): {extra}")
    print(f"standard_dock at {dock}; sim world file {sdf} sha256 {sdf_sha}")
    print(f"pose problems: {bad}")
    print(f"worst position error {worst_m:.5f} m (tol {a.tol_m}); worst yaw error {worst_rad:.5f} rad (tol {a.tol_rad})")
    print(f"G5-live: {'PASS' if passed else 'FAIL'}")
    if a.json:
        a.json.write_text(json.dumps({"dock_pose": dock, "sim_world_dir": str(sdf.parent) if sdf else None,
                                      "sim_world_file": str(sdf) if sdf else None, "sim_clinic_sdf_sha256": sdf_sha,
                                      "expected_world_file": str(expected_sdf), "world_matches": world_bad is None,
                                      "pass": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
