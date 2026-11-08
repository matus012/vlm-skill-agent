"""The 3 brief tools as OpenAI function-tool schemas (single source for probes and later phases)."""
from typing import Any

PLACES: list[str] = ["ward", "reception", "storage", "office"]  # dummy names, Phase 0 only


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


TOOLS: list[dict[str, Any]] = [
    _tool("navigate_to", "Drive to a named place; returns a new photo.",
          {"place": {"type": "string", "enum": PLACES}}, ["place"]),
    _tool("look_around", "Turn on the spot and take four numbered photos.", {}, []),
    _tool("report", "Give the final answer and end the run.",
          {"answer": {"type": "string"}, "photo": {"type": "integer"}, "note": {"type": "string"}},
          ["answer", "photo", "note"]),
]

SCHEMAS: dict[str, dict[str, Any]] = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
