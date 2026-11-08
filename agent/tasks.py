"""Dev tasks (INTERFACES section 6): load the prompts of tasks/dev.yaml. Agent side: no answers, no world truth.

A run list's task_id resolves in TASK_FILES order (task 008, 009): tasks/dev.yaml, then tasks/phase5_tasks.yaml
(Phase 5 prompts for seeded layouts), then tasks/phase6_tasks.yaml (Phase 6 prompts, blocked doorways), then every tasks/phase8_*_tasks.yaml
(task 011: Phase 8 pass prompts and the synthetic dev prompts, written by probes/build_phase8.py). load_tasks()
still returns dev.yaml alone (Phase 2 G1, the task scripts).

Truth answers, proof labels and task validation against the world live in agent/truth.py (scorer side). No ROS imports.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEV_YAML = ROOT / "tasks" / "dev.yaml"
PHASE5_YAML = ROOT / "tasks" / "phase5_tasks.yaml"
PHASE6_YAML = ROOT / "tasks" / "phase6_tasks.yaml"
PHASE8_YAML = ROOT / "tasks" / "phase8_pass1_tasks.yaml"
# task_id resolution order (008, 009, 011): Phase 8 = every tasks/phase8_*_tasks.yaml that exists (pass lists and the
# synthetic dev list), sorted by name; their ids never overlap (p8-<pass>-<i>-<kind>; dev uses a pass number > 3)
TASK_FILES = (DEV_YAML, PHASE5_YAML, PHASE6_YAML) + tuple(sorted((ROOT / "tasks").glob("phase8_*_tasks.yaml")))
SCRIPTS_DIR = ROOT / "tasks" / "scripts"
KINDS = ("find", "check_state", "count", "conditional")


class TaskError(ValueError):
    pass


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    prompt: str
    target: dict[str, Any]

    def run_json(self) -> dict[str, Any]:
        """The `task` block of run.json (INTERFACES section 5)."""
        return {"id": self.id, "kind": self.kind, "prompt": self.prompt, "target": self.target}


def load_tasks(path: Path = DEV_YAML) -> list[Task]:
    raw = yaml.safe_load(Path(path).read_text())
    ids = [t["id"] for t in raw]
    if len(set(ids)) != len(ids):
        raise TaskError(f"duplicate task ids in {path}")
    out = []
    for t in raw:
        if set(t) != {"id", "kind", "prompt", "target"}:
            raise TaskError(f"task {t.get('id')!r}: keys {sorted(t)}")
        if t["kind"] not in KINDS:
            raise TaskError(f"unknown kind {t['kind']!r}")
        out.append(Task(t["id"], t["kind"], t["prompt"], dict(t["target"])))
    return out


def all_tasks() -> list[Task]:
    """Every task a run list may name: TASK_FILES in order; an id in two files raises (so dev.yaml always wins)."""
    out = [t for p in TASK_FILES for t in load_tasks(p)]
    if len({t.id for t in out}) != len(out):
        raise TaskError(f"task ids repeated across {[p.name for p in TASK_FILES]}")
    return out


def task_by_id(task_id: str, path: Path | None = None) -> Task:
    """The task `task_id` of `path`, or with no path of all_tasks() (TASK_FILES in order)."""
    return {t.id: t for t in (all_tasks() if path is None else load_tasks(path))}[task_id]


def load_script(path: Path) -> tuple[str, list[Any]]:
    """(task id, ScriptedPlanner steps) of tasks/scripts/<task id>[.<variant>].yaml. A step is {name, args} or a raw
    tool_calls list (negative controls)."""
    return Path(path).name.split(".")[0], yaml.safe_load(Path(path).read_text())
