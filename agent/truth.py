"""Scorer side (INTERFACES section 6): truth answers and proof labels derived by code from truth.json, and task
validation against the world. Never imported by anything the runtime imports (tests/test_truth_readers.py).

Truth answers and proof labels are never typed: they come from truth.json here. No ROS imports.
"""
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from agent.tasks import DEV_YAML, KINDS, ROOT, Task, TaskError, load_tasks as load_prompts

# World variants (hpc/PERUN_SIM.md section 5): name -> generated dir (clinic.sdf, truth.json, map_knowledge.json).
CLINIC = ROOT / "sim" / "worlds" / "clinic"
# Phase 6 (INTERFACES section 11): blocked_<room> = closed + the blocker box in that room's doorway (truth "blocker").
BLOCKED_ROOMS = ("storage", "reception", "office", "ward")
WORLD_DIRS = {"closed": CLINIC / "generated", "cabinet_open": CLINIC / "generated_cabinet_open",
              **{f"blocked_{r}": CLINIC / f"generated_blocked_{r}" for r in BLOCKED_ROOMS}}
DEFAULT_WORLD = "closed"  # a run.json / list entry without `world` (all runs before 007)
TRUTH_JSON = WORLD_DIRS[DEFAULT_WORLD] / "truth.json"
TARGET_CLASSES = ("toolbox", "chair", "bed", "cabinet")  # G9-passing classes (design notes: Task targets)
STATES = ("open", "closed")
TARGET_KEYS = {"find": {"object"}, "check_state": {"object", "state"}, "count": {"class", "room"},
               "conditional": {"object", "if_not_in", "else_room"}}
# find / conditional may name a class (optionally a colour) instead of one object (INTERFACES section 6)
WHO_KEYS = ({"object"}, {"class"}, {"class", "colour"})


@dataclass(frozen=True)
class TruthTask(Task):
    truth_answer: str              # what report.answer must equal
    proof_labels: tuple[int, ...]  # seg label ids that count as proof (>= 1000 px of one of them)

    def prompt_only(self) -> Task:
        """The agent-side Task (no answers), what the runtime gets."""
        return Task(self.id, self.kind, self.prompt, self.target)


def world_truth_path(world: str = DEFAULT_WORLD) -> Path:
    """truth.json of a world variant; KeyError on an unknown name."""
    return WORLD_DIRS[world] / "truth.json"


def load_truth(path: Path = TRUTH_JSON) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def seeded() -> ModuleType:
    """sim/worlds/clinic/seeded.py (INTERFACES section 10): placements(seed), truth_for(seed, world), SeedRejected.
    Imported only here, on the scorer side, and only for a seeded run (never by the agent side)."""
    return importlib.import_module("sim.worlds.clinic.seeded")


def realizer() -> ModuleType:
    """sim/worlds/clinic/realize.py (INTERFACES section 12): realize(spec), truth_for_spec(spec), spec_sha256(spec).
    Imported only here, on the scorer side, and only for a layout-spec run (never by the agent side)."""
    return importlib.import_module("sim.worlds.clinic.realize")


def run_truth(world: str = DEFAULT_WORLD, layout_seed: int | None = None,
              layout_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """The truth of a run: its world's truth.json, or with a layout seed that seeded layout's (seeded.truth_for), or with
    a Phase 8 layout spec that spec's (realize.truth_for_spec; the spec's world must be `world`). A blocked world's
    truth carries its top-level "blocker" entry either way (both read that world's truth.json)."""
    if layout_spec is not None:
        assert layout_seed is None, "a run has a layout seed or a layout spec, not both"
        got = realizer().realize(layout_spec)["world"]
        assert got == world, f"layout spec realizes in {got!r}, the run's world is {world!r}"
        return realizer().truth_for_spec(layout_spec)
    return load_truth(world_truth_path(world)) if layout_seed is None else seeded().truth_for(layout_seed, world)


def _object(truth: dict[str, Any], oid: str) -> dict[str, Any]:
    hits = [o for o in truth["objects"] if o["id"] == oid]
    if len(hits) != 1:
        raise TaskError(f"object {oid!r} found {len(hits)} times in truth")
    if hits[0]["class"] not in TARGET_CLASSES:
        raise TaskError(f"object {oid!r} is a {hits[0]['class']}, not a target class {TARGET_CLASSES}")
    return hits[0]


def _located(target: dict[str, Any], truth: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    """(room, proof labels) of an object target, or of a class target: the one room holding all its instances."""
    if "object" in target:
        obj = _object(truth, target["object"])
        return obj["room"], (obj["label"],)
    if target["class"] not in TARGET_CLASSES:
        raise TaskError(f"class {target['class']!r} is not a target class {TARGET_CLASSES}")
    inst = [o for o in truth["objects"]
            if o["class"] == target["class"] and target.get("colour", o["colour"]) == o["colour"]]
    rooms_of = sorted({o["room"] for o in inst})
    if len(rooms_of) != 1:
        raise TaskError(f"{target} instances are in {len(rooms_of)} rooms {rooms_of}; need exactly one")
    return rooms_of[0], tuple(o["label"] for o in inst)


def derive(kind: str, target: dict[str, Any], truth: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    """(truth answer, proof labels) for one task. Raises TaskError if the task is invalid on this layout."""
    if kind not in KINDS:
        raise TaskError(f"unknown kind {kind!r}")
    keys = set(target)
    if kind in ("find", "conditional"):  # swap a class (+ colour) target for "object" before the key check
        rest = TARGET_KEYS[kind] - {"object"}
        keys = (rest | {"object"}) if keys - rest in WHO_KEYS and rest <= keys else keys
    if keys != TARGET_KEYS[kind]:
        raise TaskError(f"{kind} target keys {sorted(target)} != {sorted(TARGET_KEYS[kind])}")
    rooms = set(truth["rooms"])
    if kind == "count":
        if target["class"] not in TARGET_CLASSES:
            raise TaskError(f"class {target['class']!r} is not a target class {TARGET_CLASSES}")
        if target["room"] not in rooms:
            raise TaskError(f"unknown room {target['room']!r}")
        objs = [o for o in truth["objects"] if o["class"] == target["class"] and o["room"] == target["room"]]
        if not objs:  # no object -> no proof photo can exist; the brief leaves out "objects that aren't there"
            raise TaskError(f"no {target['class']} in {target['room']}: count 0 has no proof")
        return str(len(objs)), tuple(o["label"] for o in objs)
    if kind == "find":
        return _located(target, truth)
    if kind == "check_state":
        obj = _object(truth, target["object"])
        if obj["state"] is None:
            raise TaskError(f"object {obj['id']!r} has no state")
        if target["state"] not in STATES:
            raise TaskError(f"state {target['state']!r} not in {STATES}")
        return ("yes" if obj["state"] == target["state"] else "no"), (obj["label"],)
    a, b = target["if_not_in"], target["else_room"]  # conditional
    if a == b or not {a, b} <= rooms:
        raise TaskError(f"conditional rooms {a!r}, {b!r} must be two different rooms")
    room, labels = _located({k: v for k, v in target.items() if k not in ("if_not_in", "else_room")}, truth)
    if room not in (a, b):
        raise TaskError(f"target is in {room}, neither if_not_in {a!r} nor else_room {b!r}")
    return room, labels


def load_tasks(path: Path = DEV_YAML, truth_path: Path = TRUTH_JSON) -> list[TruthTask]:
    """Every dev task with its truth; raises TaskError if any task is invalid on this layout."""
    truth = load_truth(truth_path)
    return [TruthTask(t.id, t.kind, t.prompt, t.target, *derive(t.kind, t.target, truth)) for t in load_prompts(path)]


def task_by_id(task_id: str, path: Path = DEV_YAML, truth_path: Path = TRUTH_JSON) -> TruthTask:
    return {t.id: t for t in load_tasks(path, truth_path)}[task_id]
