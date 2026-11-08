"""Task 010 G2 / 011 a recorded deviation: the skill interface frozen at tag skill-interface-frozen-v2 (v1 + a recorded deviation's
system-prompt sentence; agent/INTERFACE_FREEZE.md). Fails if any frozen
item (tool schemas, system prompt sha256, packet fields / text, runtime rules, VLM sampling, robot adapter API) differs
from tests/interface_freeze.json. Changing one is a deviation (design ruling + design notes row), then `probes/interface_freeze.py
--write` and a new tag. Mutation check: tests/mutants_freeze.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import interface_freeze as fz  # noqa: E402

FROZEN = json.loads(fz.FIXTURE.read_text())
CURRENT = fz.current()


def test_same_items() -> None:
    assert sorted(CURRENT) == sorted(FROZEN)


@pytest.mark.parametrize("item", sorted(FROZEN))
def test_item_unchanged(item: str) -> None:
    assert CURRENT[item] == FROZEN[item], f"frozen interface item {item!r} changed (agent/INTERFACE_FREEZE.md)"


def test_prompt_hash_is_the_counted_one() -> None:
    # frozen-v2 (a recorded deviation): the system prompt of 011 pass 2 on; v1 531d7c88 = Phases 3-6 + 011 pass 1
    assert FROZEN["system_prompt_sha256"].startswith("f8374e72")
