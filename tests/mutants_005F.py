#!/usr/bin/env python3
"""Mutation check of the 005 lead-review fixes (B1 counted invocation + ledger, B2 no-call why + measure, B3 endpoint
check + endpoint error label + job time left, N2 crash origin, N3 prompt, N4 reasoning key, N5 job GPU-h, N6 setup
miss, rescore_compare). Not collected by pytest (no test_ prefix).

    source sim/env.sh; $BUILD_DIR/.venv/bin/python tests/mutants_005F.py

Each mutant: copy the repo code to a temp dir (not a git repo), apply one exact text edit (must match once), run the
tests named for it. Killed = they fail. Exit 0 iff every unmutated test set passes and every mutant is killed.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# sim/worlds/clinic whole: 007/008 truth per world (generated_cabinet_open) and per layout seed (seeded.py) live there
COPY = ["agent", "adapters", "config", "sim/worlds/clinic", "sim/maps", "tasks", "probes", "tests"]
RUN, GATES, RT = ["tests/test_phase3_run.py"], ["tests/test_phase3_gates.py"], ["tests/test_phase3_runtime.py"]
SC = ["tests/test_scorer_005.py", "tests/test_crash_origin.py", "tests/test_scorer.py", "tests/test_scorer_e2e.py"]
CO, RC = ["tests/test_crash_origin.py"], ["tests/test_rescore_compare.py"]
TG = ["tests/test_phase2_tree_guard.py"]
R, G, S, RTM = "probes/phase3_run.py", "probes/phase3_gates.py", "agent/scorer.py", "agent/runtime.py"
MUTANTS = [  # (name, file, old, new, tests)
    # B1 runner
    # re-anchored 008 (task 007 made is_counted a COUNTED_LISTS lookup): same logic, the by-hash arm of is_counted
    ("B1: counted list not detected by hash", R, "\n            or sha256_file(path) in COUNTED_LISTS.values())", ")", RUN),
    ("B1: pinned list hash wrong", R, 'COUNTED_SHA256 = "c757', 'COUNTED_SHA256 = "0757', RUN),
    ("B1: --rehearsal allowed on counted", R, "if counted and (a.rehearsal or a.only):", "if counted and a.only:", RUN),
    ("B1: --only allowed on counted", R, "if counted and (a.rehearsal or a.only):", "if counted and a.rehearsal:", RUN),
    ("B1: existing --out allowed", R, "if counted and a.out.exists():", "if False:", RUN),
    ("B1: resume re-runs existing dirs", R, 'if kind == "resume" and (out / e["run_id"]).exists():', "if False:", RUN),
    ("B1: resume list / commit not checked", R,
     'if (inv["list_sha256"], inv["git_commit"]) != (sha256_file(a.list), git["git_commit"]):', "if False:", RUN),
    ("B1: resume after end allowed", R, 'if inv["segments"] and inv["segments"][-1]["end"] == "end":', "if False:", RUN),
    ("B1: resume gets a new uuid", R, '        return runs, a.resume, inv, "resume"',
     '        return runs, a.resume, {**inv, "invocation_id": str(uuid.uuid4())}, "resume"', RUN),
    ("B1: invocation id not in run.json", R, 'run_meta(e, None, None, {"invocation_id": inv["invocation_id"]})',
     "run_meta(e, None, None, {})", RUN),
    ("B1: no ledger start line", R, "        ledger_line(ledger, inv, kind)\n", "        pass\n", RUN),
    # re-anchored 008 (task 006 added coordinator / merge copies): anchored to the serial segment's finally block
    ("B1: ledger never says end", R, '"end" if seg["end"] == "end" else "abort", reason=seg["end"],\n                        entries_run=seg["entries_run"])\n    return results',
     '"abort", reason=seg["end"],\n                        entries_run=seg["entries_run"])\n    return results', RUN),
    # re-anchored 008: serial segment only (the coordinator's finally has write_inv(out, inv, name))
    ("B1: runner death swallowed", R, '[:300]\n        raise\n    finally:\n        seg["end_utc"] = utc()\n        write_inv(out, inv)\n', '[:300]\n    finally:\n        seg["end_utc"] = utc()\n        write_inv(out, inv)\n', RUN),
    ("B1: death reason dropped", R, 'seg["end"] = f"abort: {type(ex).__name__}: {ex}"[:300]\n        raise\n    finally:\n        seg["end_utc"] = utc()\n        write_inv(out, inv)\n',
     'pass\n        raise\n    finally:\n        seg["end_utc"] = utc()\n        write_inv(out, inv)\n', RUN),
    ("B1: dev list writes the ledger", R, 'ledger = (a.ledger or default_ledger()) if inv["counted"] else None',
     "ledger = a.ledger or default_ledger()", RUN),
    ("B1: extra_meta dropped by run_one", "adapters/run_task.py", ', **(extra_meta or {})}', "}", GATES),
    # B3 runner
    # re-anchored 008: serial segment (the worker has the same call at another indent, with `end =`)
    ("B3: endpoint not checked", R, '            down = check_endpoint()\n            if down is not None:\n                seg["end"]',
     '            down = None\n            if down is not None:\n                seg["end"]', RUN),
    ("B3: endpoint without the key", R, 'headers={"Authorization": f"Bearer {api_key}"}', "headers={}", RUN),
    ("B3: endpoint error read as up", R, 'return f"{type(e).__name__}: {e}"[:200]', "return None", RUN),
    ("B3: job left ignored", R, "if a.min_job_left_min > 0 and (left := job_left()) < a.min_job_left_min:",
     "if False:", RUN),
    ("B3: job left off by one (<=)", R, "(left := job_left()) < a.min_job_left_min", "(left := job_left()) <= a.min_job_left_min",
     RUN),
    ("B3: squeue days ignored", R, "int(days or 0) * 1440 + ", "", RUN),
    ("B3: squeue seconds as minutes", R, "h * 60 + m + s / 60", "h * 60 + m + s", RUN),
    ("B3: squeue of no job", R, 'f"squeue -h -j {job} -o %L"', 'f"squeue -h -o %L"', RUN),
    # tree dirtied / moved during an invocation (2026-11-04)
    ("TREE: phase3 tree not re-checked", R, "            moved = check_tree()", "            moved = None", RUN),
    ("TREE: phase3 check never fires", R, "return None if a.rehearsal else tree_moved(start, git_state(ROOT))",
     "return None", RUN),
    ("TREE: phase3 rehearsal also stops", R, "return None if a.rehearsal else", "return None if False else", RUN),
    ("TREE: phase2 guard off", "probes/phase2_gates.py",
     "if not a.rehearsal and (why := tree_moved(start, git_state(ROOT))):", "if False:", TG),
    ("TREE: phase2 rehearsal also stops", "probes/phase2_gates.py", "if not a.rehearsal and (why :=", "if (why :=", TG),
    ("TREE: dirty not seen", "probes/phase2_gates.py", '    if now["git_dirty"]:\n', "    if False:\n", TG + RUN),
    ("TREE: commit move not seen", "probes/phase2_gates.py", 'return None if now["git_commit"] == start["git_commit"] else (',
     "return None if True else (", TG + RUN),
    ("TREE: phase2 g1 loop unguarded", "probes/phase2_gates.py",
     '                    tid, steps = load_script(SCRIPTS_DIR / f"{sid}.yaml")\n                    guard()\n                    r = run_one(ad, st.gt_pose, tasks[tid], ScriptedPlanner(steps, sid), a.out / "g1"',
     '                    tid, steps = load_script(SCRIPTS_DIR / f"{sid}.yaml")\n                    r = run_one(ad, st.gt_pose, tasks[tid], ScriptedPlanner(steps, sid), a.out / "g1"', TG),
    # B1 gates
    ("G1: invocation not checked", G, "ip = invocation_problems(out, runs, ledger or default_ledger()) if counted else []",
     "ip = []", GATES),
    ("G1: invocation problems not failing", G, "and not gp and not ip}", "and not gp}", GATES),
    ("G1: list sha not checked", G, 'if inv.get("list_sha256") != COUNTED_SHA256:', "if False:", GATES),
    ("G1: run invocation ids not checked", G, "    if other:\n", "    if False:\n", GATES),
    ("G1: other invocations allowed", G, "if ids != [iid]:", "if iid not in ids:", GATES),
    ("G1: ledger start not required", G, 'if kinds[:1] != ["start"] or ', "if ", GATES),
    ("G1: ledger end not required", G, ' or kinds[-1:] != ["end"]', "", GATES),
    ("G1: ledger middle not checked", G, ' or set(kinds[1:-1]) - {"resume", "abort"}', "", GATES),
    ("G1: other lists' ledger lines counted", G, ' if e["list_sha256"] == COUNTED_SHA256]', "]", GATES),
    ("G1: missing invocation.json passes", G, 'return ["invocation.json missing"]', "return []", GATES),
    # N6
    ("N6: G1 row keeps the scorer label", G, '''label, why = "nav", f"setup did not reach {run['start_place']} (GT''',
     '''why = f"setup did not reach {run['start_place']} (GT''', GATES),
    ("N6: setup misses not reported", G, 'if any(x.startswith("setup did not reach") for x in r["problems"])]',
     "if False]", GATES),
    ("N6: scorer ignores setup miss", S, 'if setup.get("reached") is False:', "if False:", SC),
    ("N6: setup miss after crash", S, 'if setup.get("reached") is False:',
     'if setup.get("reached") is False and run["outcome"] != "crash":', SC),
    ("N6: GT distance rounding", S, "(GT {setup['gt_err_m']:.2f} m)", "(GT {setup['gt_err_m']:.1f} m)", SC),
    # N4
    ("N4: reasoning must be non-empty again", G, 'if po["finish_reason"] is None or po["usage"] is None:',
     'if po["finish_reason"] is None or po["usage"] is None or not po["reasoning"]:', GATES),
    ("N4: empty reasoning not listed", G, 'and not s["planner_out"].get("reasoning")]', "and False]", GATES),
    # B2 measure + N5
    ("B2: length finishes not counted", G, '.get("finish_reason") == "length"]', '.get("finish_reason") == "stop"]',
     GATES),
    ("B2: toolcall-in-reasoning inverted", G, "for s in no_call_steps(steps) if toolcall_in_reasoning(s)]",
     "for s in no_call_steps(steps) if not toolcall_in_reasoning(s)]", GATES),
    ("B2: max tokens of the first step only", G, "max(tokens, default=None)", "max(tokens[:1], default=None)", GATES),
    ("N5: job GPU-h not divided", G, "job_gpu_h / len(rows)", "job_gpu_h", GATES),
    # B2 / B3 scorer
    ("B3: endpoint error ignored", S, "        elif endpoint:  # B3", "        elif False:  # B3", SC),
    ("B3: scripted errors count as endpoint", S, 'if not str(run.get("planner")).startswith("vlm:"):', "if False:", SC),
    ("B3: blocked beats endpoint", S, "        elif endpoint:  # B3", "        elif endpoint and not blocked:  # B3", SC),
    ("B3: endpoint beats crash", S, '        elif run["outcome"] == "crash":',
     '        elif run["outcome"] == "crash" and not endpoint:', SC),
    ("B3: why truncated at 60", S, "split())[:80]}", "split())[:60]}", SC),
    ("B2: no-call suffix missing", S, "            why += no_call_why(no_call_steps(steps))", "            pass", SC),
    ("B2: error steps are no-call steps", S, 's["planner_out"].get("error") is None', "True", SC),
    ("B2: bad calls are no-call steps", S, '\n            and s["call"]["raw"] == []]', "]", SC),
    ("B2: <tool_call> not detected", S, "r'<tool_call>|<function=|", "r'<function=|", SC),
    ("B2: <function= not detected", S, "r'<tool_call>|<function=|", "r'<tool_call>|", SC),
    ("B2: any name counts", S, r'''|"name"\s*:\s*"(?:navigate_to|look_around|report)"')''', r'''|"name"')''', SC),
    ("B2: content not scanned", S, " {po.get('content') or ''}", "", SC),
    ("B2: prompt tokens reported", S, ".get('completion_tokens')}", ".get('prompt_tokens')}", SC),
    # N2
    ("N2: agent code under the planner -> planner", RTM, 'return "agent" if any(', 'return "planner" if any(', CO),
    ("N2: context.py -> planner", RTM, '    if Path(top.f_code.co_filename).name == "context.py":\n        return "agent"',
     '    if Path(top.f_code.co_filename).name == "context.py":\n        return "planner"', CO),
    ("N2: AGENT_DIR wrong", RTM, "AGENT_DIR = Path(__file__).resolve().parent", "AGENT_DIR = Path(__file__).resolve()",
     CO),
    ("N2: agent label vlm", S, '"agent": "skill"', '"agent": "vlm"', CO),
    # N3
    ("N3: prompt says photos only from the last action", "agent/context.py", "a wrong call does not change them",
     "earlier photos are not shown again", RT),
    ("N3: navigate_to says drive", "agent/skills.py", '"Go to a named place', '"Drive to a named place', RT),
    # rescore_compare
    ("RC: label not compared", "probes/rescore_compare.py", 'KEYS = ("pass", "checks", "label")',
     'KEYS = ("pass", "checks")', RC),
    ("RC: checks not compared", "probes/rescore_compare.py", 'KEYS = ("pass", "checks", "label")',
     'KEYS = ("pass", "label")', RC),
    ("RC: exit 0 on a difference", "probes/rescore_compare.py", "return 1 if n_diff or not rows else 0",
     "return 1 if not rows else 0", RC),
    ("RC: exit 0 on no runs", "probes/rescore_compare.py", "return 1 if n_diff or not rows else 0",
     "return 1 if n_diff else 0", RC),
]


def run_tests(tree: Path, tests: list[str]) -> tuple[int, str]:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests], cwd=tree,
                       capture_output=True, text=True, timeout=900)
    return p.returncode, (p.stdout.strip().splitlines() or [""])[-1]


def copy_tree(tree: Path) -> None:
    for c in COPY:
        src, dst = ROOT / c, tree / c
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        else:
            shutil.copy2(src, dst)


def main() -> int:
    ok, killed = True, 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base"
        copy_tree(base)
        for tests in sorted({tuple(m[4]) for m in MUTANTS}):
            rc, last = run_tests(base, list(tests))
            ok &= rc == 0
            print(f"{'baseline PASS' if rc == 0 else 'baseline FAIL'} {' '.join(tests)}  [{last}]", flush=True)
        for i, (name, rel, old, new, tests) in enumerate(MUTANTS):
            tree = Path(tmp) / f"m{i}"
            shutil.copytree(base, tree)
            f = tree / rel
            s = f.read_text()
            assert s.count(old) == 1, f"mutant '{name}': pattern found {s.count(old)} times in {rel}"
            f.write_text(s.replace(old, new))
            rc, last = run_tests(tree, tests)
            killed += rc != 0
            ok &= rc != 0
            print(f"{'KILLED' if rc else 'SURVIVED':9s} {name}  [{last}]", flush=True)
            shutil.rmtree(tree)
    print(f"{killed}/{len(MUTANTS)} mutants killed; {'OK' if ok else 'NOT OK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
