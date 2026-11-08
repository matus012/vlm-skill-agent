"""Scripted VLM for agent tests: an OpenAI-compatible mock server fed with simple (skill, args) steps."""
from typing import Any

from mock_openai import MockServer, tool_call_msg


def scripted(*steps: tuple[str, Any]) -> MockServer:
    """Each step = (tool name, args dict or raw string). Cycles when exhausted."""
    return MockServer([(200, tool_call_msg(n, a)) for n, a in steps])


def reasoning_step(name: str, args: Any, reasoning: str = "thinking...") -> tuple[int, dict[str, Any]]:
    return 200, {**tool_call_msg(name, args), "reasoning": reasoning}
