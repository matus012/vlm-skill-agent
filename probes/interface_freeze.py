#!/usr/bin/env python3
"""Task 010 (Phase 7): the frozen skill interface, as data. `current()` reads every frozen item from the code as it is
now; `tests/interface_freeze.json` holds the values frozen at tag `skill-interface-frozen-v1`, and
`tests/test_interface_freeze.py` fails if any item differs. Human-readable list: `agent/INTERFACE_FREEZE.md`.

    python3 probes/interface_freeze.py            # print current() as JSON
    python3 probes/interface_freeze.py --write    # (re)write tests/interface_freeze.json: only at a freeze / an design review-ruled change

Frozen items: the tool schemas for each task kind (full JSON, places from places.yaml, rooms from the closed world's
map_knowledge.json, as adapters/run_task.py builds them), the system prompt sha256, the Packet fields, the packet text
(build_messages on a fixed example, image data replaced by a marker), the runtime rules (step limit, malformed limit,
nav timeout, place bound, count pattern, task kinds), the VLM sampling + max_tokens, and the robot adapter API (abstract
methods with signatures, Frame / Photo / GoResult fields). No ROS, no sim.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import context, planners, robot, runtime, skills  # noqa: E402
from agent.truth import WORLD_DIRS  # noqa: E402
from agent.world import load_map  # noqa: E402

FIXTURE = ROOT / "tests" / "interface_freeze.json"
PLACES_YAML = ROOT / "sim" / "worlds" / "clinic" / "places.yaml"


def tname(t: Any) -> str:
    """A field type as text, the same whether the module uses postponed annotations (a string) or not (a class)."""
    return t.__name__ if isinstance(t, type) else str(t)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def example_packet_text() -> list[dict[str, Any]]:
    """build_messages on a fixed two-photo packet; image URLs replaced so only the format is frozen."""
    img = Image.new("RGB", (8, 8))
    p = context.Packet(photos=[robot.Photo(1, img), robot.Photo(2, img)], room="hall",
                       places=["hall", "ward", "office", "reception", "storage"],
                       history=["1. look_around() in hall -> 4 photos"], last_result="4 photos", task="Find the X.",
                       step=2)
    msgs = context.build_messages(p)
    for part in msgs[1]["content"]:
        if part["type"] == "image_url":
            part["image_url"]["url"] = "<image>"
    return msgs


def current() -> dict[str, Any]:
    places = list(skills.load_places(PLACES_YAML))
    rooms = list(load_map(WORLD_DIRS["closed"] / "map_knowledge.json")["rooms"])
    adapter = {n: str(inspect.signature(f)) for n, f in sorted(vars(robot.Adapter).items())
               if getattr(f, "__isabstractmethod__", False)}
    return {
        "tool_schemas": {k: skills.build_tools(places, k, rooms) for k in skills.TASK_KINDS},
        "system_prompt_sha256": context.system_prompt_sha256(places),
        "packet_fields": [[f.name, tname(f.type)] for f in dataclasses.fields(context.Packet)],
        "packet_text_sha256": sha(json.dumps(example_packet_text(), sort_keys=True)),
        "runtime_rules": {"MAX_STEPS": context.MAX_STEPS, "MALFORMED_LIMIT": runtime.MALFORMED_LIMIT,
                          "NAV_TIMEOUT_S": skills.NAV_TIMEOUT_S, "PLACE_BOUND_M": skills.PLACE_BOUND_M,
                          "COUNT_PATTERN": skills.COUNT_PATTERN, "TASK_KINDS": list(skills.TASK_KINDS)},
        "vlm": {"SAMPLING": {str(k): v for k, v in planners.SAMPLING.items()},
                "MAX_TOKENS": {str(k): v for k, v in planners.MAX_TOKENS.items()}},
        "adapter_api": {"abstract_methods": adapter,
                        **{c.__name__: [[f.name, tname(f.type)] for f in dataclasses.fields(c)]
                           for c in (robot.Frame, robot.Photo, robot.GoResult)}},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"write {FIXTURE.relative_to(ROOT)}")
    a = ap.parse_args()
    cur = json.dumps(current(), indent=1, sort_keys=True) + "\n"
    if a.write:
        FIXTURE.write_text(cur)
        print(f"wrote {FIXTURE} sha256 {sha(cur)}")
    else:
        print(cur, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
