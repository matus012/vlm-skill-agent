#!/usr/bin/env python3
"""Mutation check of the phase 1 gate verdict code (design review B1). Not collected by pytest (no test_ prefix).

    $BUILD_DIR/.venv/bin/python tests/mutants_phase1_gates.py

Each mutant: copy probes/ + tests/ to a temp dir, apply one exact text edit (must match once), run the gate tests.
Killed = the tests fail. Exit 0 iff the unmutated copy passes and every mutant is killed.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_phase1_gates.py", "tests/test_phase1_checks.py", "tests/test_map_world_gates.py"]
G, C, M, T = "probes/phase1_gates.py", "probes/phase1_checks.py", "probes/map_world_check.py", "probes/truth_check.py"
MUTANTS = [  # (name, file, old, new)
    ("G7 neg_ok = True", G, 'neg_ok = any(not v["check_passes"] for v in neg.values())', "neg_ok = True"),
    ("G1 drop err <= G1_MAX_ERR_M", G, " and err <= G1_MAX_ERR_M\n", "\n"),
    ("G1 drop sim_s <= G1_MAX_SIM_S", G, '\n                               and r["sim_s"] <= G1_MAX_SIM_S)', ")"),
    ("G1 n >= 39 instead of 40", G, "n == G1_TOTAL}", "n >= 39}"),
    ("photo set: drop 4-photo assert", C, "bad = [] if len(ph) == len(HEADINGS_DEG) else "
                                          '[f"{len(ph)} photos, need {len(HEADINGS_DEG)}"]', "bad = []"),
    ("G2 drop yaw roll-up over 5 places", G, ' and all(v["yaw_pass"] for v in res.values())', ""),
    ("G2 drop yaw from check_place g2_pass", C, ' and res["yaw_pass"]', ""),
    ("G2 yaw_ok always true", C, "return abs(err_deg) <= G2_MAX_YAW_DEG", "return True"),
    ("G1 drop start-pose check", G, 'bool(rec["valid_start"] and ', "bool("),
    ("G1 counted tries not limited to 120 s sim", G, "sim_limit_s=G1_MAX_SIM_S if counted else mwc.NAV_SIM_LIMIT_S",
     "sim_limit_s=mwc.NAV_SIM_LIMIT_S"),
    ("photos: drop GT error check", G, " and err <= PHOTO_MAX_ERR_M", ""),
    ("photos: drop nav SUCCEEDED check", G, 'navs[-1]["status"] == robot.SUCCEEDED and err', "err"),
    ("photos: pass always true", G, '"pass": all(r["at_place"] for r in runs.values())', '"pass": True'),
    ("checks: never refuse", G, "    if bad:  # refuse", "    if False:  # refuse"),
    ("checks: ignore at_place", G, "if at.get(name) is not True:", "if False:"),
    ("navigate: no sim timeout", M, "if sim_now() - s0 > sim_limit_s:", "if False:"),
    ("G5 dock: no tolerance", T, "None if d <= DOCK_TOL_M else", "None if True else"),
]


def run_tests(tree: Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS], cwd=tree,
                       capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout.strip().splitlines() or [""])[-1]


def main() -> int:
    rows, ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, rel, old, new) in enumerate([("(none: unmutated)", None, None, None)] + MUTANTS):
            tree = Path(tmp) / f"m{i}"
            for d in ("probes", "tests"):
                shutil.copytree(ROOT / d, tree / d, ignore=shutil.ignore_patterns("__pycache__"))
            if rel:
                f = tree / rel
                s = f.read_text()
                assert s.count(old) == 1, f"mutant '{name}': pattern found {s.count(old)} times in {rel}"
                f.write_text(s.replace(old, new))
            rc, last = run_tests(tree)
            verdict = ("baseline PASS" if rc == 0 else "baseline FAIL") if not rel else ("KILLED" if rc else "SURVIVED")
            ok &= (rc == 0) if not rel else (rc != 0)
            rows.append(f"{verdict:13s} | {name:42s} | {rel or '-':26s} | {last}")
            print(rows[-1], flush=True)
    print(f"mutants: {len(MUTANTS)}, killed {sum(r.startswith('KILLED') for r in rows)}, "
          f"survived {sum(r.startswith('SURVIVED') for r in rows)}; {'OK' if ok else 'NOT OK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
