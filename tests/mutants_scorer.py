#!/usr/bin/env python3
"""Mutation check of the scorer + task truth (task 003 C). Not collected by pytest (no test_ prefix).

    $BUILD_DIR/.venv/bin/python tests/mutants_scorer.py

Each mutant: copy the code the tests need to a temp dir, apply one exact text edit (must match once), run the scorer
tests. Killed = the tests fail. Exit 0 iff the unmutated copy passes and every mutant is killed.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_scorer.py", "tests/test_task_scripts.py", "tests/test_scorer_e2e.py", "tests/test_crash_origin.py"]
COPY = ["agent", "tests", "tasks", "probes/replay.py", "sim/worlds/clinic/places.yaml",
        "sim/worlds/clinic/generated/truth.json"]
S, T = "agent/scorer.py", "agent/truth.py"
MUTANTS = [  # (name, file, old, new)
    ("check 1 answer always true", S, '"answer": report is not None and report["answer"] == truth_answer',
     '"answer": report is not None'),
    ("check 2 proof always true", S, '"proof": proof_px >= PROOF_MIN_PX', '"proof": True'),
    ("check 3 steps always true", S, '"steps": run["outcome"] == "reported" and n_steps <= MAX_STEPS',
     '"steps": True'),
    ("check 4 clean always true", S, '"clean": run["outcome"] != "crash" and not run.get("hand_help", False)',
     '"clean": True'),
    ("proof off-by-one: > 1000", S, "proof_px >= PROOF_MIN_PX", "proof_px > PROOF_MIN_PX"),
    ("proof off-by-one: threshold 999", S, "PROOF_MIN_PX = 1000", "PROOF_MIN_PX = 999"),
    ("steps off-by-one: < 12", S, "n_steps <= MAX_STEPS", "n_steps < MAX_STEPS"),
    ("steps off-by-one: limit 13", S, "MAX_STEPS = 12", "MAX_STEPS = 13"),
    ("steps ignores outcome", S, 'run["outcome"] == "reported" and n_steps', "n_steps"),
    ("clean ignores hand help", S, ' and not run.get("hand_help", False)', ""),
    ("clean ignores crash", S, '"clean": run["outcome"] != "crash" and ', '"clean": '),
    ("proof = sum over labels, not one label", S, "return px[best], best", "return sum(px.values()), best"),
    ("proof from the rgb file, not seg", S, 'return hit[0]["seg"] if hit', 'return hit[0]["rgb"] if hit'),
    ("proof photo number off by one", S, 'p["n"] == report["photo"]', 'p["n"] == report["photo"] + 1'),
    ("label source: list index, not truth label", T, 'tuple(o["label"] for o in objs)',
     'tuple(truth["objects"].index(o) + 1 for o in objs)'),
    ("label source: count proof from any room", T,
     'objs = [o for o in truth["objects"] if o["class"] == target["class"] and o["room"] == target["room"]]',
     'objs = [o for o in truth["objects"] if o["class"] == target["class"]]'),
    ("check_state yes/no flipped", T, '("yes" if obj["state"] == target["state"] else "no")',
     '("no" if obj["state"] == target["state"] else "yes")'),
    ("conditional validity not checked", T, "if room not in (a, b):", "if False:"),
    ("class truth ignores split rooms", T, "if len(rooms_of) != 1:", "if not rooms_of:"),
    ("class target ignores colour", T, 'target.get("colour", o["colour"]) == o["colour"]', "True"),
    ("label: adapter crash -> vlm", S, '"adapter": "sim"', '"adapter": "vlm"'),
    ("label: pre-005 crash -> vlm", S, 'if origin else "sim"', 'if origin else "vlm"'),
    ("label: blocked ignores reachability", S, '\n                and s["call"]["args"]["place"] in reachable', ""),
    ("label: nav never fires", S, "        elif blocked:", "        elif False:"),
]


def run_tests(tree: Path) -> tuple[int, str]:
    tests = [t for t in TESTS if (tree / t).exists()]
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests], cwd=tree,
                       capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout.strip().splitlines() or [""])[-1]


def main() -> int:
    rows, ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, rel, old, new) in enumerate([("(none: unmutated)", None, None, None)] + MUTANTS):
            tree = Path(tmp) / f"m{i}"
            for c in COPY:
                src, dst = ROOT / c, tree / c
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
                else:
                    shutil.copy2(src, dst)
            if rel:
                f = tree / rel
                s = f.read_text()
                assert s.count(old) == 1, f"mutant '{name}': pattern found {s.count(old)} times in {rel}"
                f.write_text(s.replace(old, new))
            rc, last = run_tests(tree)
            verdict = ("baseline PASS" if rc == 0 else "baseline FAIL") if not rel else ("KILLED" if rc else "SURVIVED")
            ok &= (rc == 0) if not rel else (rc != 0)
            rows.append(f"{verdict:13s} {name}  [{last}]")
            print(rows[-1], flush=True)
    killed = sum(r.startswith("KILLED") for r in rows)
    print(f"{killed}/{len(MUTANTS)} mutants killed; {'OK' if ok else 'NOT OK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
