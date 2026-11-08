#!/usr/bin/env python3
"""Task 011 list builder (Phase 8, INTERFACES section 12): the pass-N counted list from the sealed generator's output.

    $BUILD_DIR/.venv/bin/python probes/build_phase8.py --pass N --pass-file F --fresh-seeds S \
        --gen G --out tasks/

Stream = F's layouts in order, then S's fresh seeds in order, rendered by the UNCHANGED generator G (render_pass on F's
seeds + S's seeds, so the ids continue p8-N-26, ...). G re-rendering F's own seeds must give F byte for byte. Each
layout carries 4 entries (one per kind); per entry in stream order the validity filter (a recorded deviation, section 12):
  (a) the target's (class, spec room) is G9-passing AND realize(spec) placed every target object (none parked), AND
      the task derives on truth_for_spec(spec);
  (b) check_spec(spec) is ok (ray-caster G2-G4 at every place);
  (c) neither the layout seed (spec seed) nor the planner seed (sha256(entry id) mod 2^31) is a dev / counted seed of
      Phases 3-6 (every seed / layout_seed of tasks/phase[3-6]_counted.yaml, phase[3-6]_dev.yaml, phase3_*.yaml) or an
      008 spot seed (probes/pick_phase6: pick_phase5.spot_seeds() | FIRST_SPOT | EXCLUDED);
  (d) the start rule: per kind, in accepted-entry order, cycle hall / other room / answer room, at most 2 answer-room
      starts per kind (later answer-room slots take an other room); other room = a places.yaml room that is neither the
      answer room nor (conditional) one of its two named rooms, rotating through G's ROOMS (the first such room at or
      after the kind's pointer, which then moves past it); no such place -> drop.
An entry is dropped by its first failing rule. Valid entries are taken until each kind has 25; a kind already full
skips its entries; the stream stops being consumed once all four are full. Running out first is an error.
Answer room (agent.truth.derive on truth_for_spec): find / conditional = the answer, count = the counted room,
check_state = the cabinet's room.
Writes <out>/phase8_pass<N>_tasks.yaml, <out>/phase8_pass<N>.yaml (100 entries, phase5 list format with layout_spec and
world = realize(spec)["world"]), <out>/phase8_pass<N>_drops.json. Prints the drop report (counts and sha256s only):
never a prompt, seed or room of an entry.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.tasks import KINDS, TaskError  # noqa: E402
from agent.truth import derive  # noqa: E402
from probes import pick_phase6  # noqa: E402

PER_KIND = 25  # task 011 / brief Phase 8: 25 runs per type
ANSWER_STARTS_MAX = 2  # section 12 start rule: at most 2 answer-room starts per kind
# Section 12 (a): G9-passing (class, room); chair: any room (filled from G.ROOMS at run time).
G9_ROOMS = {"toolbox": {"reception", "storage"}, "cabinet": {"storage"}, "bed": {"ward"}}
CABINET = "cabinet_blue_1"
PLACES_YAML = ROOT / "sim" / "worlds" / "clinic" / "places.yaml"
FRESH_PINS = ROOT / "tasks" / "phase8_fresh_seeds.sha256"
RULES = ("a_g9_placement", "b_check_spec", "c_seed_collision", "d_start_rule")
CYCLE = ("hall", "other_room", "answer_room")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_gen(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_phase8", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def realizer() -> SimpleNamespace:
    """realize / truth_for_spec from sim/worlds/clinic/realize.py (truth side, imported only here) and check_spec from
    probes/vet_seeds.py (the ray-caster check, filter rule b)."""
    r = importlib.import_module("sim.worlds.clinic.realize")
    vet = importlib.import_module("probes.vet_seeds")
    return SimpleNamespace(realize=r.realize, truth_for_spec=r.truth_for_spec, spec_sha256=r.spec_sha256,
                           check_spec=vet.check_spec)


def planner_seed(entry_id: str) -> int:
    return int(hashlib.sha256(entry_id.encode()).hexdigest(), 16) % 2**31


def used_seeds(tasks_dir: Path = ROOT / "tasks") -> set[int]:
    """Every dev / counted seed of Phases 3-6 (seed and layout_seed of the run lists) + 008's spot seeds."""
    files = {*tasks_dir.glob("phase[3-6]_counted.yaml"), *tasks_dir.glob("phase[3-6]_dev.yaml"),
             *tasks_dir.glob("phase3_*.yaml")}
    seeds = {e[k] for f in files for e in yaml.safe_load(f.read_text()) for k in ("seed", "layout_seed") if k in e}
    return seeds | pick_phase6.p5.spot_seeds() | pick_phase6.FIRST_SPOT | pick_phase6.EXCLUDED


def read_fresh(path: Path) -> list[int]:
    """One seed per line (comments / blanks skipped). If tasks/phase8_fresh_seeds.sha256 pins this file name, its
    sha256 and count must match."""
    seeds = [int(x) for x in Path(path).read_text().split("\n") if x.strip() and not x.lstrip().startswith("#")]
    for line in FRESH_PINS.read_text().splitlines() if FRESH_PINS.is_file() else []:
        parts = line.split()
        if not line.startswith("#") and len(parts) == 3 and parts[2] == Path(path).name:
            if (parts[0], int(parts[1])) != (sha256(path), len(seeds)):
                sys.exit(f"{path.name}: sha256 / count differ from the pin in {FRESH_PINS.name}")
    return seeds


def stream(gen: ModuleType, n: int, pass_file: Path, fresh: list[int]) -> tuple[list[dict[str, Any]], int]:
    """(layouts of the whole stream, number from F). F's seeds come from F's `seed:` fields; G must re-render F."""
    doc = yaml.safe_load(Path(pass_file).read_text())
    if doc["pass"] != n:
        sys.exit(f"pass file is pass {doc['pass']}, --pass {n}")
    own = [lay["seed"] for lay in doc["layouts"]]
    if gen.render_pass(n, own).encode() != Path(pass_file).read_bytes():
        sys.exit("the generator does not re-render the pass file byte for byte")
    layouts = yaml.safe_load(gen.render_pass(n, own + fresh))["layouts"]
    if layouts[:len(own)] != doc["layouts"]:
        sys.exit("stream prefix differs from the pass file's layouts")
    return layouts, len(own)


def spec_of(layout: dict[str, Any]) -> dict[str, Any]:
    return {k: layout[k] for k in ("seed", "objects", "removed")}


def cls_of(oid: str) -> str:
    return oid.split("_")[0]


def target_ids(task: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    t = task["target"]
    if task["kind"] == "check_state" or "object" in t:
        return [t.get("object", CABINET)]
    return [o for o in spec["objects"] if cls_of(o) == t["class"]]


def answer_room(task: dict[str, Any], truth: dict[str, Any]) -> str:
    answer = derive(task["kind"], task["target"], truth)[0]
    if task["kind"] == "check_state":
        return derive("find", {"object": task["target"]["object"]}, truth)[0]
    return task["target"]["room"] if task["kind"] == "count" else answer


def rule_a(task: dict[str, Any], spec: dict[str, Any], real: dict[str, Any], truth: dict[str, Any],
           rooms: tuple[str, ...]) -> bool:
    """(a): G9-passing (class, spec room), every target object placed (none parked), the task derives on the truth."""
    ids = target_ids(task, spec)
    if not ids or set(ids) & set(real["parked"]):
        return False
    room = {spec["objects"][o]["room"] for o in ids}
    if len(room) != 1 or not room <= G9_ROOMS.get(cls_of(ids[0]), set(rooms)):
        return False
    try:
        answer_room(task, truth)
    except TaskError:
        return False
    return True


def build(gen: ModuleType, R: Any, n: int, pass_file: Path, fresh_file: Path, gen_file: Path,
          used: set[int], places: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """(accepted tasks in stream order, list entries, drop report)."""
    fresh = read_fresh(fresh_file)
    layouts, n_own = stream(gen, n, pass_file, fresh)
    rooms = tuple(gen.ROOMS)
    kept: dict[str, list] = {k: [] for k in KINDS}
    drops = {r: {k: 0 for k in KINDS} for r in RULES}
    examined, skipped = {k: 0 for k in KINDS}, {k: 0 for k in KINDS}
    n_answer, ptr = {k: 0 for k in KINDS}, {k: 0 for k in KINDS}
    tasks, entries, consumed = [], [], 0
    for lay in layouts:
        if all(len(v) == PER_KIND for v in kept.values()):
            break
        consumed += 1
        spec = spec_of(lay)
        real, truth = R.realize(spec), R.truth_for_spec(spec)
        spec_ok = R.check_spec(spec)[0]
        for task in lay["tasks"]:
            k = task["kind"]
            if len(kept[k]) == PER_KIND:
                skipped[k] += 1
                continue
            examined[k] += 1
            pseed = planner_seed(task["id"])
            if not rule_a(task, spec, real, truth, rooms):
                rule = "a_g9_placement"
            elif not spec_ok:
                rule = "b_check_spec"
            elif spec["seed"] in used or pseed in used:
                rule = "c_seed_collision"
            else:
                rule = None
                ans = answer_room(task, truth)
                slot = CYCLE[len(kept[k]) % 3]
                if slot == "answer_room" and n_answer[k] >= ANSWER_STARTS_MAX:
                    slot = "other_room"
                if slot == "hall":
                    start = "hall"
                elif slot == "answer_room":
                    start = ans
                else:
                    named = {task["target"].get(x) for x in ("if_not_in", "else_room")}
                    cand = [i for i in range(len(rooms)) if rooms[i] in places and rooms[i] not in named | {ans}]
                    nxt = min(cand, key=lambda i: (i - ptr[k]) % len(rooms)) if cand else None
                    start = rooms[nxt] if nxt is not None else None
                if start not in places:
                    rule = "d_start_rule"
            if rule:
                drops[rule][k] += 1
                continue
            if slot == "answer_room":
                n_answer[k] += 1
            if slot == "other_room":
                ptr[k] = (nxt + 1) % len(rooms)
            kept[k].append(slot)
            tasks.append(task)
            entries.append({"run_id": f"p8p{n}c{len(entries) + 1:03d}", "task_id": task["id"], "start_place": start,
                            "seed": pseed, "world": real["world"], "layout_spec": spec})
    full = {k: len(v) for k, v in kept.items()}
    if any(v < PER_KIND for v in full.values()):
        sys.exit(f"stream ran out after {consumed} layouts: kept per kind {full}, need {PER_KIND} each")
    if len({e["seed"] for e in entries}) != len(entries):
        sys.exit("two accepted entries share a planner seed")
    report = {"pass": n, "pass_file_sha256": sha256(pass_file), "fresh_seeds_sha256": sha256(fresh_file),
              "generator_sha256": sha256(gen_file), "layouts_in_pass_file": n_own, "fresh_seeds_available": len(fresh),
              "layouts_consumed": consumed, "fresh_seeds_used": max(0, consumed - n_own),
              "entries_examined": examined, "entries_skipped_kind_full": skipped, "drops": drops,
              "kept": full, "yield": {k: full[k] / examined[k] for k in KINDS},
              "starts": {k: {s: kept[k].count(s) for s in CYCLE} for k in KINDS}}
    return tasks, entries, report


def header(report: dict[str, Any]) -> str:
    return (f"# pass file sha256 {report['pass_file_sha256']}\n# fresh seeds sha256 {report['fresh_seeds_sha256']}\n"
            f"# generator sha256 {report['generator_sha256']}\n")


def write(out: Path, n: int, gen: ModuleType, tasks: list[dict[str, Any]], entries: list[dict[str, Any]],
          report: dict[str, Any]) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"tasks": out / f"phase8_pass{n}_tasks.yaml", "list": out / f"phase8_pass{n}.yaml",
             "drops": out / f"phase8_pass{n}_drops.json"}
    paths["tasks"].write_text(
        f"# Phase 8 pass {n} test prompts (task 011): the tasks of the valid entries of the sealed stream, in stream\n"
        "# order, rendered by the unchanged generator (dev.yaml format). Written by probes/build_phase8.py.\n"
        + header(report) + "".join(gen.entry(t)[2:] + "\n" for t in tasks))
    paths["list"].write_text(
        f"# Task 011 Phase 8 pass {n} counted list: {len(entries)} entries, {PER_KIND} per kind, in stream order,\n"
        "# each used once. Entry {run_id, task_id, start_place, seed, world, layout_spec} (INTERFACES section 12):\n"
        "# seed = sha256(task id) mod 2^31, world = realize(layout_spec)['world']; task_id resolves in\n"
        f"# tasks/phase8_pass{n}_tasks.yaml. Written by probes/build_phase8.py (module doc = the rule). True answers\n"
        "# come from truth_for_spec(layout_spec).\n" + header(report)
        + "".join("- {" + ", ".join(f"{k}: {json.dumps(e[k]) if k == 'layout_spec' else e[k]}" for k in e) + "}\n"
                  for e in entries))
    report |= {"tasks_sha256": sha256(paths["tasks"]), "list_sha256": sha256(paths["list"])}
    paths["drops"].write_text(json.dumps(report, indent=2) + "\n")
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="n", type=int, required=True)
    ap.add_argument("--pass-file", type=Path, required=True)
    ap.add_argument("--fresh-seeds", type=Path, required=True)
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    gen = load_gen(a.gen)
    places = set(yaml.safe_load(PLACES_YAML.read_text())["places"])
    tasks, entries, report = build(gen, realizer(), a.n, a.pass_file, a.fresh_seeds, a.gen, used_seeds(), places)
    write(a.out, a.n, gen, tasks, entries, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
