#!/usr/bin/env python3
"""Mutation check of the G1 adapter / Drive rules (012 P2: recentre, stall, fall, timeout, scan avoidance) and of the
012 precondition fixes (run_one's G1 frozen-world restore; H-P3's 3-seed selection). Not collected by pytest.

    source sim/env.sh; $BUILD_DIR/.venv/bin/python tests/mutants_g1.py

Each mutant: copy the code the tests need to a temp dir (the 012-fix mutants: every git-tracked file but the Phase 8
test lists), apply one exact text edit (must match once), run that group's tests. Killed = the tests fail. Exit 0 iff the unmutated copy passes and every mutant is killed.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["humanoid/tests/test_g1_drive.py", "humanoid/tests/test_g1_adapter.py"]
COPY = ["agent", "humanoid/__init__.py", "humanoid/adapter_g1.py", "humanoid/drive.py", "humanoid/geometry.py",
        "humanoid/probes/__init__.py", "humanoid/probes/common.py", "humanoid/tests/__init__.py", *TESTS,
        "config/robots/g1.yaml", "sim/worlds/clinic/generated/map_knowledge.json", "pyproject.toml"]
A, D = "humanoid/adapter_g1.py", "humanoid/drive.py"
MUTANTS = [  # (name, file, old, new)
    ("recentre off", A, "            if drift <= RECENTRE_M:\n", "            if True:\n"),
    ("stall rule off (Drive)", D, "stalled = t - t_ref > STALL_S", "stalled = False"),
    ("stall not mapped to STALLED", A, '            if leg["stalled"]:\n', "            if False:\n"),
    ("fall off", D, "        if self.pose.z < FALL_Z_M:\n", "        if False:\n"),
    ("whole-call timeout off", A, '        if self.sim_time() > deadline:\n            return done("blocked", "TIMEOUT")\n'
     '        return done("reached"', '        return done("reached"'),
    ("avoidance off", D, "            aim, side = self._avoid(tx, ty, dist, side)\n", "            aim = (tx, ty)\n"),
    ("look_around anchored at the goal", A, "        self.anchor = None    # the next", "        pass  # the next"),
    ("min_v ignored", D, "            if 0.0 < v < min_v:\n", "            if False:\n"),
    ("recentre without out leg", A, "            out_leg = self.drive.goto(ox, oy, timeout_s=budget(), gate_m=STOP_TOL_M)\n",
     "            out_leg = {}\n"),
    ("stop_m ignored", D, "if dist <= stop_m or", "if dist <= STOP_TOL_M or"),
    ("sync without waiting for a scan", D, "        while self.pose.t == t0 or self.scan is scan0:\n",
     "        while self.pose.t == t0:\n"),
    ("go_to routes from the pre-reset pose", A, "        self.drive.sync()  # the pose may predate",
     "        # the pose may predate"),
    ("recentre out leg into a wall", A, "            if not self._blocked_ahead():\n", "            if True:\n"),
    ("blocked go_to returns walking", A, "                self.drive.hold(SETTLE_S)  # walk coasted",
     "                pass  # walk coasted"),
]


RT, RB, SP, G9 = "adapters/run_task.py", "adapters/robots.py", "humanoid/probes/g1_spot.py", "probes/g9_readable.py"
TESTS_FIX = ["tests/test_robots.py", "tests/test_g1_probes.py", "tests/test_run_one_golden.py"]
MUTANTS_FIX = [  # task 012 fixes 1 (frozen-world restore) and 2 (H-P3 photos from 3 different seeds)
    ("frozen run not restored", RT, "if laid else restore_world\n", "if laid else None\n"),
    ("restore yaw from roll", RT, 'frozen[oid]["pose"][5])', 'frozen[oid]["pose"][3])'),
    ("g1 without restore_world", RB, '"restore_world": scene.place,', ""),
    ("restore miss scored on the default world", RT, "places_path, None if laid else truth_path)", "places_path, None)"),
    ("failed spot seed 693 in the pool", SP, "dev_seeds() - pick_phase6.EXCLUDED", "dev_seeds()"),
    ("pick ignores photos in hand", SP, 'n = len({c["seed"] for c in candidates(out, done, cls, room, state)})', "n = 0"),
    ("pick redraws tried seeds", SP, '{int(d.name) for d in out.glob("*") if d.name.isdigit()} | set(exclude)',
     "set(exclude)"),
    ("select passes on fewer seeds", SP, " and len(used[g]) >= N_PHOTOS for g, s", " for g, s"),
    ("fixed candidates ignore the state", SP, 'and o.get("state") == state]', "]"),
    ("selection groups without state", G9, '"state": s[0].get("state") if s else None', '"state": None'),
    ("verdict ignores the seed count", SP, " and len(used.get(g, [])) >= N_PHOTOS\n", "\n"),
]


def run(tmp: Path, tests: list[str] = TESTS) -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests], cwd=tmp,
                       capture_output=True, text=True)
    return r.returncode == 0


def copy_tracked(tmp: Path) -> None:
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    for rel in files.decode().split("\0"):
        if rel and not rel.startswith("tasks/phase8_pass") and (ROOT / rel).is_file():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, tmp / rel)


def copy(tmp: Path) -> None:
    for rel in COPY:
        src, dst = ROOT / rel, tmp / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main() -> int:
    ok = True
    for copier, tests, mutants in ((copy, TESTS, MUTANTS), (copy_tracked, TESTS_FIX, MUTANTS_FIX)):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            copier(tmp)
            base = run(tmp, tests)
            print(f"{'PASS' if base else 'FAIL'}  unmutated ({len(tests)} test files)")
            ok &= base
            for name, rel, old, new in mutants:
                f = tmp / rel
                src = f.read_text()
                assert src.count(old) == 1, f"{name}: edit matches {src.count(old)} times"
                f.write_text(src.replace(old, new))
                killed = not run(tmp, tests)
                f.write_text(src)
                print(f"{'killed' if killed else 'SURVIVED'}  {name}")
                ok &= killed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
