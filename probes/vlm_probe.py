"""Phase 0 VLM probe: one photo + the 3 brief tools -> does the model return a schema-valid tool call?

Exit 0 only if >= --require calls are valid (G3: --n 3 --require 3; G4: --n 10 --require 9).
"""
import argparse
import base64
import io
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import jsonschema
from openai import OpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from phase0_tools import PLACES, SCHEMAS, TOOLS  # noqa: E402

SYSTEM = ("You are a small wheeled robot in a clinic. You see photo 1, taken by your camera. "
          "You can only act by calling tools. Pick exactly one tool.")
# Sampling per the Qwen3.8-27B model card (hpc/VLLM_VERSION.md): thinking vs non-thinking mode.
SAMPLING = {True: {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0, "top_k": 20},
            False: {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5, "top_k": 20}}
MAX_TOKENS = {True: 8192, False: 1024}  # thinking needs room before the tool call; truncation shows as finish_reason=length
TASK = f"Task: find out which room has a red toolbox. Known places: {', '.join(PLACES)}. Call exactly one tool now."


def image_data_url(path: Path, max_side: int = 640) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def verdict(tool_calls: list[dict[str, Any]]) -> tuple[bool, str]:
    """valid = >= 1 tool call, name in the 3 tools, arguments parse as JSON and validate against that tool's schema."""
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name")
        if name not in SCHEMAS:
            continue
        try:
            jsonschema.validate(json.loads(fn.get("arguments")), SCHEMAS[name])
        except (ValueError, TypeError, jsonschema.ValidationError):
            continue
        return True, name
    return False, "no valid tool call"


def one_call(client: OpenAI, model: str, url: str, thinking: bool, max_tokens: int) -> dict[str, Any]:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": TASK}]}]
    rec: dict[str, Any] = {"status": None, "latency_s": None, "tool_calls": [], "finish_reason": None,
                           "usage": None, "content": None, "reasoning_chars": None, "valid": False, "why": ""}
    sp = SAMPLING[thinking]
    t0 = time.perf_counter()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model, messages=msgs, tools=TOOLS, tool_choice="auto", max_tokens=max_tokens,
            temperature=sp["temperature"], top_p=sp["top_p"], presence_penalty=sp["presence_penalty"],
            extra_body={"top_k": sp["top_k"], "chat_template_kwargs": {"enable_thinking": thinking}})
        rec["status"] = raw.status_code
        resp = raw.parse()
        ch = resp.choices[0]
        rec["finish_reason"] = ch.finish_reason
        rec["tool_calls"] = [tc.model_dump() for tc in (ch.message.tool_calls or [])]
        rec["usage"] = resp.usage.model_dump() if resp.usage else None
        rec["content"] = (ch.message.content or "")[:500]
        extra = ch.message.model_extra or {}
        reasoning = extra.get("reasoning") or extra.get("reasoning_content") or ""
        rec["reasoning_chars"] = len(reasoning)
        rec["valid"], rec["why"] = verdict(rec["tool_calls"])
    except Exception as e:  # network/HTTP error becomes a recorded failed call
        rec["status"] = getattr(e, "status_code", None) or 0
        rec["why"] = f"{type(e).__name__}: {e}"[:300]
    rec["latency_s"] = time.perf_counter() - t0
    return rec


def percentile(xs: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(xs)
    return s[math.ceil(q * len(s)) - 1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--api-key", default=None, help="default: env VLM_API_KEY, else EMPTY")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--out", type=Path, default=Path("probe_out.jsonl"))
    ap.add_argument("--require", type=int, required=True, help="min valid calls for exit 0")
    ap.add_argument("--max-tokens", type=int, default=None, help="default: 8192 thinking on, 1024 off")
    a = ap.parse_args(argv)

    key = a.api_key or os.environ.get("VLM_API_KEY") or "EMPTY"
    client = OpenAI(base_url=a.base_url, api_key=key, timeout=300, max_retries=0)
    url = image_data_url(a.image)
    think = a.thinking == "on"
    max_tokens = a.max_tokens or MAX_TOKENS[think]
    recs = [one_call(client, a.model, url, think, max_tokens) for _ in range(a.n)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as f:
        for i, r in enumerate(recs):
            f.write(json.dumps({"i": i, "model": a.model, "base_url": a.base_url, "thinking": a.thinking, "image": str(a.image),
                                "max_tokens": max_tokens, **r}) + "\n")
    lat = [r["latency_s"] for r in recs]
    nv = sum(r["valid"] for r in recs)
    print(f"thinking={a.thinking} valid {nv}/{a.n}  median {statistics.median(lat):.2f}s  "
          f"p90 {percentile(lat, 0.9):.2f}s  (require {a.require})")
    return 0 if nv >= a.require else 1


if __name__ == "__main__":
    sys.exit(main())
