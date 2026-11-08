#!/usr/bin/env python3
"""Mutation check of the phase 3 gate verdicts, run-list loader and the Phase B runtime / harness changes (task 005 P3). Not collected by pytest.

    source sim/env.sh; $BUILD_DIR/.venv/bin/python tests/mutants_phase3_gates.py

Each mutant: copy the code the tests need to a temp dir, apply one exact text edit (must match once), run
tests/test_phase3_gates.py + tests/test_phase3_runtime.py. Killed = the tests fail. Exit 0 iff the unmutated copy passes and every mutant is killed.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_phase3_gates.py", "tests/test_phase3_runtime.py"]
COPY = ["agent", "adapters", "probes/replay.py", "probes/phase2_gates.py", "probes/phase3_gates.py",
        "probes/phase3_run.py", "tasks", "sim/worlds/clinic/places.yaml", "sim/worlds/clinic/generated",
        "tests/test_phase3_gates.py", "tests/test_agent_no_ros.py",
        "tests/test_phase3_runtime.py", "tests/mock_openai.py"]
G = "probes/phase3_gates.py"
R = "probes/phase3_run.py"
MUTANTS = [  # (name, file, old, new)
    ("G1: 6/10 passes", G, "n >= G1_NEED and not gp and", "n >= G1_NEED - 1 and not gp and"),
    ("G1: list length not checked", G, '"pass": counted and len(entries) == G1_RUNS and n', '"pass": counted and n'),
    ("G1: not-counted may pass", G, '"pass": counted and len(entries) == G1_RUNS and n', '"pass": len(entries) == G1_RUNS and n'),
    ("G1: git ignored", G, "n >= G1_NEED and not gp and", "n >= G1_NEED and"),
    ("G1: prompt sha ignored", G, "\n          + prompt_problems(list(runs.values())))", ")"),
    ("G1: entry not matched", G, "p, label, why = entry_problems(entry, run), ", "p, label, why = [], "),
    ("G1: seed not matched", G, 'want = (entry["task_id"], entry["start_place"], entry["seed"])',
     'want = (entry["task_id"], entry["start_place"], got[2])'),
    ("G1: setup ignored", G, 'if not run["setup"]["reached"]:', "if False:"),
    ("G1: saved score trusted", G, 'elif saved["pass"] != fresh["pass"] or saved["checks"] != fresh["checks"]:',
     "elif False:"),
    ("G1: score.json may be missing", G, 'p.append("score.json missing")', "pass"),
    ("G1: fresh score ignored", G, '"pass": bool(fresh["pass"]) and not p}', '"pass": not p}'),
    ("G2: system sha not checked", G, 'if pk.get("system_sha256") != run.get("system_prompt_sha256"):', "if False:"),
    ("G2: texts not required", G, "if not (isinstance(texts, list)", "if False and (isinstance(texts, list)"),
    ("G2: photos may be empty", G, "    if not photos:", "    if False:"),
    ("G2: rgb sha not compared", G, ' or ph.get("rgb_sha256") != sha256_file(f)', ""),
    ("G2: sent sha not required", G, 'if not re.fullmatch(r"[0-9a-f]{64}", str(ph.get("sent_sha256"))):', "if False:"),
    ("G2: reasoning key not required (N4)", G, '{"content", "reasoning", "finish_reason", "usage", "error"}',
     '{"content", "finish_reason", "usage", "error"}'),
    ("G2: usage not required", G, 'if po["finish_reason"] is None or po["usage"] is None:',
     'if po["finish_reason"] is None:'),
    ("G2: finish_reason not required", G, 'if po["finish_reason"] is None or po["usage"] is None:',
     'if po["usage"] is None:'),
    ("G2: planner_out keys not required", G, 'or set(po) < {"content", "reasoning", "finish_reason", "usage", "error"}',
     ""),
    ("G2: raw call not required", G, 'if not isinstance((s.get("call") or {}).get("raw"), list):', "if False:"),
    ("G2: skill result not required", G, '(s.get("check") == "ok" and s["result"] is None and not crashed)', "False"),
    ("G2: timings not required", G, "if missing and not crashed:", "if False:"),
    ("G2: check_complete ignored", G, 'p = [f"check_complete: {e}" for e in check_complete(d)]', "p = []"),
    ("G2: prompt text not hashed", G, 'hashlib.sha256(sp.encode()).hexdigest() != run.get("system_prompt_sha256")',
     "False"),
    ("G2: tools not required", G, 'if not isinstance(run.get("tools"), list) or not run["tools"]:', "if False:"),
    ("G2: planner_info not required", G, "if any(info.get(k) is None for k in PLANNER_INFO):", "if False:"),
    ("G2: any label", G, 'sc.get("label") in LABELS and ', ""),
    ("G2: multi-line why", G, ' and "\\n" not in why', ""),
    ("G2: blank why", G, " and why.strip()", ""),
    ("G2: passing run may carry a label", G,
     'if sc["pass"] and (sc.get("label") is not None or why is not None):', "if False:"),
    ("G2: not-counted may pass", G, '"pass": counted and len(entries) == G1_RUNS and all(not v', '"pass": all(not v'),
    ("G3: decimals not scanned", G, 'POSE_RES = [re.compile(r"-?\\d+\\.\\d+"), ', "POSE_RES = ["),
    ("G3: pairs not scanned", G, ', re.compile(r"[(\\[]\\s*-?\\d+(?:\\.\\d+)?\\s*,\\s*-?\\d+(?:\\.\\d+)?")]', "]"),
    ("G3: G1 word not scanned", G, "|\\bg1\\b", ""),
    ("G3: case-sensitive words", G, "humanoid\", re.I)", "humanoid\")"),
    ("G3: everything is an echo", G, '"echo": h.lower() in echoes}', '"echo": True}'),
    ("G3: history not scanned", G, '*(pk.get("history") or []), ', ""),
    ("G3: last result not scanned", G, ', str(pk.get("last_result"))]', "]"),
    ("G3: tools not scanned", G, 'found += [{"where": "tools", **h} for h in scan(json.dumps(run.get("tools")), "")]',
     "pass"),
    ("G3: prompt not scanned", G, 'scan(str(run.get("system_prompt")), "")', 'scan("", "")'),
    ("G3: no-ROS test ignored", G, ' and no_ros["returncode"] == 0', ""),
    ("G3: missing runs ignored", G, "and not missing and", "and"),
    ("G3: not-counted may pass", G, '"pass": (counted and len(entries) == G1_RUNS and not missing',
     '"pass": (len(entries) == G1_RUNS and not missing'),
    ("Measure: H200-h in minutes", G, '"h200_h": sum(pw) / 3600}', '"h200_h": sum(pw) / 60}'),
    ("list: duplicates allowed", R, 'assert len({r[k] for r in runs}) == len(runs), f"{path}: duplicate {k}"', "pass"),
    ("list: str seed allowed", R, 'assert isinstance(r["seed"], int) and not isinstance(r["seed"], bool), r', "pass"),
    ("list: extra keys allowed", R, 'assert set(r) - {"world"} == KEYS, f"{r}: keys', 'assert True, f"{r}: keys'),
    ("list: unknown task / place allowed", R, 'assert r["task_id"] in tasks and r["start_place"] in places, r', "pass"),
    # Phase B (runtime / planners / harness / scorer)
    ("B: malformed limit off", "agent/runtime.py", "if malformed_limit is not None and malformed >= malformed_limit:",
     "if False:"),
    ("B: planner failure counted as malformed", "agent/runtime.py",
     'malformed = 0 if out["error"] else malformed + 1', "malformed = malformed + 1"),
    ("B: valid call does not reset", "agent/runtime.py", 'rec["check"], malformed = "ok", 0', 'rec["check"] = "ok"'),
    ("B: limit 3", "agent/runtime.py", "MALFORMED_LIMIT = 2", "MALFORMED_LIMIT = 3"),
    ("B: texts not logged", "agent/runtime.py", '"texts": [c["text"] for c in parts if c["type"] == "text"]}',
     '"texts": [c["text"] for c in parts if c["type"] == "text"][:1]}'),
    ("B: rgb sha of the path", "agent/runtime.py", 'sha256((log.dir / e["rgb"]).read_bytes())', 'sha256(e["rgb"].encode())'),
    ("B: sent sha of the file", "agent/runtime.py", '"sent_sha256": sha256(u.encode())',
     '"sent_sha256": sha256((log.dir / e["rgb"]).read_bytes())'),
    ("B: seed not sent", "agent/planners.py", ' seed=cfg.seed,', ""),
    ("B: seed not in info", "agent/planners.py", '"seed": cfg.seed}', '"seed": None}'),
    ("B: start place ignored", "adapters/run_task.py", "hx, hy, hyaw = places[start_place]",
     "hx, hy, hyaw = places[START_PLACE]"),
    ("B: setup logs the hall", "adapters/run_task.py", 'setup = {"start_place": start_place,',
     'setup = {"start_place": START_PLACE,'),
    ("B: malformed limit not passed", "adapters/run_task.py", "truth=truth_hook, malformed_limit=malformed_limit)",
     "truth=truth_hook)"),
    ("B: gt_trace not recorded", "adapters/run_task.py", ', "gt_trace": gt_trace,', ","),
    ("B: gz_reads not recorded", "adapters/run_task.py", '"gz_reads": gz_cli.since(gz0)}', "}"),
    ("B: gz_reads from process start", "adapters/run_task.py", "gz_cli.since(gz0)", "gz_cli.since((0, 0, 0))"),
    ("B: fail_malformed why generic", "agent/scorer.py", 'if run["outcome"] == "fail_malformed":', "if False:"),
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
