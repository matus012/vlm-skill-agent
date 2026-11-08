#!/usr/bin/env python3
"""Mutation check of 005 Step 0 (N8 truth readers + map_knowledge, N1 crash origin, N7 hand help, N5 world tag).
Not collected by pytest (no test_ prefix).

    source sim/env.sh; $BUILD_DIR/.venv/bin/python tests/mutants_s0.py

Each mutant: copy the repo code to a temp dir (not a git repo), apply one exact text edit (must match once), run the
tests named for it. Killed = they fail. Exit 0 iff every unmutated test set passes and every mutant is killed.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = ["agent", "adapters", "humanoid", "sim", "tasks", "probes", "hpc", "tests", "config"]
TR, MK = ["tests/test_truth_readers.py"], ["tests/test_map_knowledge.py"]
CO, HH = ["tests/test_crash_origin.py"], ["tests/test_hand_help.py"]
WT = ["tests/test_world_tag.py", "-k", "not real_repo"]  # the copy is no git repo; the real-repo case runs in the suite
READ = '\nTRUTH = __import__("json").loads(open("sim/worlds/clinic/generated/truth.json").read())\n'
MUTANTS = [  # (name, file, old, new, tests)
    ("N8: agent/skills.py reads truth.json", "agent/skills.py", "\nNAV_TIMEOUT_S", READ + "NAV_TIMEOUT_S", TR),
    ("N8: adapters/turtlebot4.py reads truth.json", "adapters/turtlebot4.py", '\nROBOT_CFG', READ + "ROBOT_CFG", TR),
    ("N8: humanoid/adapter_g1.py reads truth.json", "humanoid/adapter_g1.py", "\nTURN_TIMEOUT_S", READ + "TURN_TIMEOUT_S", TR),
    ("N8: agent/runtime.py imports agent.truth", "agent/runtime.py", "from agent.tasks import Task",
     "from agent.tasks import Task\nfrom agent.truth import derive", TR),
    ("N8: agent/world.py loads TRUTH_JSON", "agent/world.py", 'MAP_KNOWLEDGE = ROOT', 'TRUTH_JSON = MAP_KNOWLEDGE = ROOT', TR),
    ("N8: map_knowledge keeps objects", "sim/worlds/clinic/generate.py", '("world", "layout_sha256", "rooms")',
     '("world", "layout_sha256", "rooms", "objects")', MK),
    ("N8: map_knowledge not written", "sim/worlds/clinic/generate.py",
     '    (a.out / "map_knowledge.json").write_text(map_knowledge(truth))\n', "", MK),
    ("N8: load_map accepts extra keys", "agent/world.py", 'assert set(m) == {"world", "layout_sha256", "rooms"}',
     "assert True", MK),
    ("N1: no adapter detection", "agent/runtime.py", 'if any(f.f_locals.get("self") is ad for f in below):',
     "if False:", CO),
    ("N1: planner seen as runtime", "agent/runtime.py", 'if top.f_locals.get("self") is planner:', "if False:",
     CO),
    ("N1: skill seen as runtime", "agent/runtime.py", 'return "skill" if Path', 'return "runtime" if Path', CO),
    ("N1: truth hook not detected", "agent/runtime.py", "if truth is not None and top.f_code is truth.__code__:",
     "if False:", CO),
    ("N1: origin not written to run.json", "agent/runtime.py", ', "crash_origin": res.crash_origin', "", CO),
    ("N1: scorer ignores origin (all sim)", "agent/scorer.py", 'label = CRASH_LABEL[origin] if origin else "sim"',
     'label = "sim"', CO),
    ("N1: skill crash -> sim", "agent/scorer.py", '"skill": "skill"', '"skill": "sim"', CO),
    ("N1: planner crash -> skill", "agent/scorer.py", '"planner": "vlm"', '"planner": "skill"', CO),
    ("N7: no slack -> any drift is help", "adapters/hand_help.py", "bound = self.slack + (", "bound = 0.0 + (", HH),
    ("N7: drive bound on every step", "adapters/hand_help.py", "if drove else 0.0", "", HH),
    ("N7: no speed bound on navigate_to", "adapters/hand_help.py", "(self.v_max * max(dt, 0.0) if drove else 0.0)",
     "1e9", HH),
    ("N7: initialpose ignored", "adapters/hand_help.py", "        if n_msgs:\n", "        if False:\n", HH),
    ("N7: echo count off (lines, not '---')", "adapters/hand_help.py", 'if line.strip() == "---"', "", HH),
    ("N7: harness does not write the verdict", "adapters/run_task.py", "**hh.verdict(), \"gt_trace\"", "\"gt_trace\"", HH),
    ("N7: hook never feeds GT", "adapters/run_task.py", "        hh.gt(n_step[0], ad.sim_time()", "        0 and hh.gt(n_step[0], ad.sim_time()", HH),
    ("N5: oldest matching tag", "adapters/run_task.py", '"--sort=-creatordate"', '"--sort=creatordate"', WT),
    ("N5: no hash check (newest tag)", "adapters/run_task.py",
     '        if [hashlib.sha256(git("show", f"{tag}:{f}")).hexdigest() for f in files] == disk:',
     "        if True:", WT),
    ("N5: no match -> newest tag, no refusal", "adapters/run_task.py", '    raise RuntimeError(f"no tag', '    return tags[0]\n    raise RuntimeError(f"no tag', WT),
]


def run_tests(tree: Path, tests: list[str]) -> tuple[int, str]:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests], cwd=tree,
                       capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout.strip().splitlines() or [""])[-1]


def main() -> int:
    rows, ok = [], True
    base = [(f"(none: unmutated) {' '.join(t)}", None, None, None, t) for t in (TR, MK, CO, HH, WT)]
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, rel, old, new, tests) in enumerate(base + MUTANTS):
            tree = Path(tmp) / f"m{i}"
            for c in COPY:
                shutil.copytree(ROOT / c, tree / c, ignore=shutil.ignore_patterns("__pycache__", "generated_*"))
            if rel:
                f = tree / rel
                s = f.read_text()
                assert s.count(old) == 1, f"mutant '{name}': pattern found {s.count(old)} times in {rel}"
                f.write_text(s.replace(old, new))
            rc, last = run_tests(tree, tests)
            verdict = ("baseline PASS" if rc == 0 else "baseline FAIL") if not rel else ("KILLED" if rc else "SURVIVED")
            ok &= (rc == 0) if not rel else (rc != 0)
            rows.append(f"{verdict:13s} {name}  [{last}]")
            print(rows[-1], flush=True)
            shutil.rmtree(tree)
    killed = sum(r.startswith("KILLED") for r in rows)
    print(f"{killed}/{len(MUTANTS)} mutants killed; {'OK' if ok else 'NOT OK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
