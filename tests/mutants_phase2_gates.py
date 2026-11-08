#!/usr/bin/env python3
"""Mutation check of the phase 2 gate verdicts (task 003 I). Not collected by pytest (no test_ prefix).

    source sim/env.sh; $BUILD_DIR/.venv/bin/python tests/mutants_phase2_gates.py

Each mutant: copy the code the tests need to a temp dir, apply one exact text edit (must match once), run
tests/test_phase2_gates.py. Killed = the tests fail. Exit 0 iff the unmutated copy passes and every mutant is killed.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_phase2_gates.py"]
COPY = ["agent", "adapters/__init__.py", "adapters/run_task.py", "adapters/hand_help.py", "tests/test_phase2_gates.py",
        "tests/test_scorer_e2e.py", "tasks", "probes/replay.py", "probes/phase2_gates.py",
        "sim/worlds/clinic/places.yaml", "sim/worlds/clinic/generated/truth.json"]
G = "probes/phase2_gates.py"
MUTANTS = [  # (name, file, old, new)
    ("G1: 11/12 passes", G, "sum(ok) == G1_TOTAL and not gp}", "sum(ok) >= G1_TOTAL - 1 and not gp}"),
    ("G1: setup not required", G, 'bool(r["setup"]["reached"] and r["score"]["pass"])', 'bool(r["score"]["pass"])'),
    ("G1: run count not checked", G, "full and len(runs) == G1_TOTAL and", "full and"),
    ("G1: subset may pass", G, '"pass": full and len(runs)', '"pass": len(runs)'),
    ("G1: plan does not cycle", G, "by_kind[k][i % len(by_kind[k])]", "by_kind[k][0]"),
    ("G2: failed set is a superset, not equal", G, "failed != G2_FAILS[name]", "not failed >= G2_FAILS[name]"),
    ("G2: label not checked", G, '        if score["label"] != "vlm":', '        if False:'),
    ("G2: 13-steps outcome not checked", G, 'name == "13-steps" and (', 'False and ('),
    ("G2: unmapped control passes", G, 'return [f"control {name!r} has no G2 mapping"]', "return []"),
    ("G2: missing rows ignored", G, "full and not missing and", "full and"),
    ("G2: error texts not matched", G, "w in s[\"check\"] and s[\"check\"].startswith(\"error:\")", "True"),
    ("G2: error count not matched", G, "len(errs) != len(want) or ", ""),
    ("G2: error-continue need not pass", G, 'run["outcome"] != "reported" or not score["pass"]',
     'run["outcome"] != "reported"'),
    ("G2: error not returned to caller", G, 'nxt[0]["packet"]["last_result"] != s["check"]', "False"),
    ("G3: any result counts as blocked", G, 'if n["blocked_result"] != "failed: blocked":', "if False:"),
    ("G3: wall point at the door", G, "(a + b) / 2, cy)", "(cx, cy)"),
    # design review 2026-11-04: B1, B2, B4, N2, N3, N4
    ("B1: hall step need not move", G, 'if not n["hall_gt_move_m"] >= G3_MIN_MOVE_M:', "if False:"),
    ("B1: blocked step GT not checked", G, 'if not n["blocked_gt_err_m"] > PLACE_BOUND_M:', "if False:"),
    ("B1: hall step GT end not checked", G, 'if not n["hall_gt_err_m"] <= PLACE_BOUND_M:', "if False:"),
    ("B1: hall step 'reached' not checked", G, 'if n["hall_result"] != "reached":', "if False:"),
    ("B1: script starts at the hall (no storage leg)", G, 'G3_PLACES = ("storage", G3_PLACE, "hall")',
     'G3_PLACES = (G3_PLACE, "hall")'),
    ("B2: time = whole step incl. photo", G, '"blocked_nav_sim_s": (b.get("skill_detail") or {}).get("nav_sim_s")',
     '"blocked_nav_sim_s": b["skill_sim_s"]'),
    ("B2: hang guard 10 s", G, "HANG_GUARD_S = 1.0", "HANG_GUARD_S = 10.0"),
    ("B4: counted run may start dirty", G, 'if not a.rehearsal and git_state(ROOT)["git_dirty"]:', "if False:"),
    ("B4: dirty runs ignored", G, 'if any(s["git_dirty"] for s in states) else []', "if False else []"),
    ("B4: commit disagreement ignored", G, "if len(commits) > 1 else []", "if False else []"),
    ("B4: G1 ignores git", G, "sum(ok) == G1_TOTAL and not gp}", "sum(ok) == G1_TOTAL}"),
    ("B4: G2 ignores git", G, "full and not missing and not gp and", "full and not missing and"),
    ("B4: G3 ignores git", G, "p = g3_problems(run_dir) + git_problems([git_of(run_dir)])", "p = g3_problems(run_dir)"),
    ("N2: setup ignores GT", "adapters/run_task.py", '"reached": go.status == "reached" and err <= PLACE_BOUND_M',
     '"reached": go.status == "reached"'),
    ("N2: setup ignores Nav2", "adapters/run_task.py", '"reached": go.status == "reached" and err <= PLACE_BOUND_M',
     '"reached": err <= PLACE_BOUND_M'),
    ("N3: replay ignores hashes", "probes/replay.py", 'if saved.get("sha256", {}).get(f) != h]', "if False]"),
    ("N3: proof seg not hashed", "agent/scorer.py", '*([seg] if seg else [])', ""),
    ("N4: G4 run count not checked", G, '"pass": len(per) == expected and all(', '"pass": all('),
    ("G4: replay ignored", G, 'and r["replay_same"]', ""),
    ("G4: completeness ignored", G, 'not r["incomplete"] and ', ""),
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
