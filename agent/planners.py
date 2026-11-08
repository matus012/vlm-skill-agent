"""Planners (INTERFACES section 3). The runtime calls `next_call(messages, tools)` and never knows which one it has."""
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI

# Qwen3.8-27B model card (hpc/VLLM_VERSION.md), same as probes/vlm_probe.py.
SAMPLING = {True: {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0, "top_k": 20},
            False: {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5, "top_k": 20}}
MAX_TOKENS = {True: 8192, False: 1024}


class Planner(Protocol):
    name: str  # run.json "planner": "scripted:<script id>" | "vlm:<model>"
    info: dict[str, Any]  # run.json "planner_info" (INTERFACES section 3)

    def next_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """{"error", "tool_calls", "content", "reasoning", "finish_reason", "usage"}"""
        ...


def _out(tool_calls: list[dict[str, Any]], error: str | None = None, **kw: Any) -> dict[str, Any]:
    return {"error": error, "tool_calls": tool_calls, "content": kw.get("content"), "reasoning": kw.get("reasoning"),
            "finish_reason": kw.get("finish_reason"), "usage": kw.get("usage")}


@dataclass(frozen=True)
class VlmConfig:
    base_url: str
    api_key: str
    model: str
    thinking: bool = True  # design notes: thinking ON from Phase 3
    timeout: float = 300
    seed: int | None = None  # per run (task 005 counted / dev lists); sent as the request's `seed`

    @classmethod
    def from_env(cls) -> "VlmConfig":
        return cls(os.environ["VLM_BASE_URL"], os.environ["VLM_API_KEY"], os.environ["VLM_MODEL"])


def call_vlm(client: OpenAI, cfg: VlmConfig, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
    """One VLM call. An HTTP/network error becomes {"error": ...} (a logged failed step)."""
    sp = SAMPLING[cfg.thinking]
    try:
        resp = client.chat.completions.create(
            model=cfg.model, messages=messages, tools=tools, tool_choice="auto", max_tokens=MAX_TOKENS[cfg.thinking],
            temperature=sp["temperature"], top_p=sp["top_p"], presence_penalty=sp["presence_penalty"], seed=cfg.seed,
            extra_body={"top_k": sp["top_k"], "chat_template_kwargs": {"enable_thinking": cfg.thinking}})
    except Exception as e:  # the only try/except: network boundary
        return _out([], f"{type(e).__name__}: {e}"[:300])
    m = resp.choices[0].message
    extra = m.model_extra or {}
    return _out([tc.model_dump() for tc in (m.tool_calls or [])], content=m.content,
                finish_reason=resp.choices[0].finish_reason,
                reasoning=extra.get("reasoning") or extra.get("reasoning_content"),
                usage=resp.usage.model_dump() if resp.usage else None)


class VlmPlanner:
    def __init__(self, cfg: VlmConfig, client: OpenAI | None = None) -> None:
        self.cfg, self.name = cfg, f"vlm:{cfg.model}"
        self.info = {"model": cfg.model, "thinking": cfg.thinking, "sampling": SAMPLING[cfg.thinking],
                     "max_tokens": MAX_TOKENS[cfg.thinking], "seed": cfg.seed}
        self.client = client or OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout, max_retries=0)

    def next_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        return call_vlm(self.client, self.cfg, messages, tools)


def tool_call(name: str, args: Any, call_id: str = "s0") -> dict[str, Any]:
    """OpenAI-style tool call; `args` a dict (JSON-encoded here) or a raw string (sent as is, e.g. bad JSON)."""
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


class ScriptedPlanner:
    """Plays the planner from a script, one entry per step, ignoring the packet. An entry is {"name", "args"} or a raw
    tool_calls list (negative controls: unknown skill, unknown place, bad format, bad JSON, several calls).
    Past the end of the script it returns no call with error "script exhausted" (an error step, not a crash)."""

    def __init__(self, steps: list[dict[str, Any] | list[dict[str, Any]]], script_id: str = "inline") -> None:
        self.steps, self.i, self.name = steps, 0, f"scripted:{script_id}"
        self.info = {"script": script_id}

    def next_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if self.i >= len(self.steps):
            return _out([], "script exhausted")
        s = self.steps[self.i]
        self.i += 1
        return _out(s if isinstance(s, list) else [tool_call(s["name"], s.get("args", {}), f"s{self.i}")])
