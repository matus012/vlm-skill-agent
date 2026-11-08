#!/usr/bin/env python3
"""Task 010 G2 mutation check of tests/test_interface_freeze.py: a schema, the system prompt and a packet field are each
mutated in a temp copy of the repo code; the freeze test must fail on every mutant and pass unmutated. Not collected
by pytest (no test_ prefix).

    source sim/env.sh; $BUILD_DIR/.venv/bin/python tests/mutants_freeze.py

Exit 0 iff the baseline passes and all 3 mutants are killed.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = ["agent", "sim/worlds/clinic", "probes", "tests"]
TEST = ["tests/test_interface_freeze.py"]
MUTANTS = [  # (name, file, old, new); each old must match exactly once
    ("schema: navigate_to description", "agent/skills.py",
     '"Go to a named place; returns a new photo."', '"Go to a named place; returns a photo."'),
    ("prompt: one word in the system prompt", "agent/context.py",
     "running out is a failure", "running out is a fail"),
    ("packet: a new Packet field", "agent/context.py",
     "    step: int  # 1-based, the step about to be taken\n",
     "    step: int  # 1-based, the step about to be taken\n    hint: str = \"\"\n"),
]


def copy_tree(dst: Path) -> None:
    for c in COPY:
        src = ROOT / c
        shutil.copytree(src, dst / c, ignore=shutil.ignore_patterns("__pycache__", "generated*")
                        if c != "sim/worlds/clinic" else shutil.ignore_patterns("__pycache__"))


def run_tests(d: Path) -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TEST], cwd=d,
                       capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as t:
        base = Path(t) / "base"
        copy_tree(base)
        passed = run_tests(base)
        print(f"baseline: {'pass' if passed else 'FAIL'}")
        ok &= passed
        for i, (name, f, old, new) in enumerate(MUTANTS):
            d = Path(t) / f"m{i}"
            copy_tree(d)
            p = d / f
            s = p.read_text()
            assert s.count(old) == 1, f"{name}: match text found {s.count(old)} times in {f}"
            p.write_text(s.replace(old, new))
            killed = not run_tests(d)
            print(f"{'killed' if killed else 'SURVIVED'}: {name}")
            ok &= killed
    print("mutants_freeze:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
