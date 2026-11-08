#!/usr/bin/env python3
"""Task 013: the one-entry (or few) rehearsal list for the showcase video, copied line-for-line from the counted
tasks/phase8_pass2.yaml (same run_id / seed / layout_spec, so the planner seed matches the counted run).
    python3 probes/make_video_list.py [--ids p8p2c001]  ->  tasks/phase8_video.yaml
No tasks file is written: the entry's task_id already resolves from tasks/phase8_pass2_tasks.yaml, and a second
phase8_*_tasks.yaml holding the same id would make agent.tasks.all_tasks() raise on duplicates (agent/ is frozen).
The output is not a counted list (its sha256 is not in phase3_run.COUNTED_LISTS). Prompts are never printed."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC, TASKS, OUT = (ROOT / "tasks" / n for n in ("phase8_pass2.yaml", "phase8_pass2_tasks.yaml", "phase8_video.yaml"))


def build(ids: list[str], src: Path = SRC, tasks: Path = TASKS) -> str:
    lines = {m.group(1): ln for ln in src.read_text().splitlines()
             if (m := re.match(r"- \{run_id: (\w+),", ln))}
    unknown = [i for i in ids if i not in lines]
    if unknown:
        sys.exit(f"unknown run ids {unknown} (not in {src.name})")
    known = {t["id"] for t in yaml.safe_load(tasks.read_text())}
    for i in ids:
        tid = yaml.safe_load(lines[i])[0]["task_id"]
        assert tid in known, f"{i}: task {tid} not in {tasks.name}"
    return (f"# Task 013 showcase-video rehearsal list: entries {', '.join(ids)} copied line-for-line from {src.name}.\n"
            "# NOT a counted run: a re-run of the same entry for the video (same run_id and seed). Written by "
            "probes/make_video_list.py.\n" + "".join(lines[i] + "\n" for i in ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="p8p2c001", help="comma-separated run ids of tasks/phase8_pass2.yaml")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    ids = a.ids.split(",")
    a.out.write_text(build(ids))
    from probes.phase3_run import is_counted, load_runs
    assert not is_counted(a.out), f"{a.out} is a counted list"
    print(f"wrote {a.out}: {len(load_runs(a.out))} entries {ids}")


if __name__ == "__main__":
    main()
