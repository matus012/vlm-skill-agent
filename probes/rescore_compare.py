#!/usr/bin/env python3
"""Regression for scorer-only changes (task 005 "Counted-run rules"): re-score every run dir with the current scorer, in
memory (nothing is written into the run dirs), and compare with its saved score.json. pass, checks and label must be
identical; a changed label_why is listed, not failed. Exit 1 on any difference (a missing score.json is one).

    $BUILD_DIR/.venv/bin/python probes/rescore_compare.py DIR [DIR ...]

DIR = a directory whose subdirectories with a run.json are run dirs (e.g. a phase2 / phase3 gate --out).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.scorer import score_run  # noqa: E402

KEYS = ("pass", "checks", "label")


def compare(dirs: list[Path], **score_kw: Path) -> tuple[list[dict], int]:
    """(one row per run dir, number of runs whose pass / checks / label differ)."""
    rows = []
    for d in sorted(r.parent for top in dirs for r in Path(top).glob("*/run.json")):
        f = d / "score.json"
        saved = json.loads(f.read_text()) if f.is_file() else {}
        fresh = score_run(d, **score_kw)
        rows.append({"run": str(d), "saved": {k: saved.get(k) for k in KEYS}, "fresh": {k: fresh[k] for k in KEYS},
                     "same": all(saved.get(k) == fresh[k] for k in KEYS),
                     "why_changed": saved.get("label_why") != fresh["label_why"], "fresh_why": fresh["label_why"]})
    return rows, sum(not r["same"] for r in rows)


def main(argv: list[str]) -> int:
    rows, n_diff = compare([Path(a) for a in argv])
    for r in rows:
        s, f = r["saved"], r["fresh"]
        print(f"{'same' if r['same'] else 'DIFF':4s}  {r['run']}  saved pass {s['pass']!s:5s} label {s['label']!s:6s}  "
              f"fresh pass {f['pass']!s:5s} label {f['label']!s:6s}"
              + (f"  checks {s['checks']} -> {f['checks']}" if s["checks"] != f["checks"] else "")
              + (f"  why now: {r['fresh_why']}" if r["why_changed"] else ""))
    print(f"{len(rows)} runs, {n_diff} differ in pass / checks / label")
    return 1 if n_diff or not rows else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
