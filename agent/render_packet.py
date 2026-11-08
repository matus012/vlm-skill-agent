"""Render one packet (the exact messages + tools the planner gets) to a self-contained HTML file, for inspection.
No VLM calls. `python -m agent.render_packet OUT.html` renders an example from the fake adapter (step 3 of a find
task, after navigate_to(storage) and look_around())."""
import html
import json
import sys
from pathlib import Path
from typing import Any

from agent import skills
from agent.context import Packet, build_messages, history_line
from agent.fake_adapter import PLACES, FakeAdapter
from agent.skills import SkillContext


def render(messages: list[dict[str, Any]], tools: list[dict[str, Any]], out: Path) -> None:
    parts = []
    for m in messages:
        parts.append(f"<h2>{html.escape(m['role'])}</h2>")
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for c in content:
            if c["type"] == "text":
                parts.append(f"<pre>{html.escape(c['text'])}</pre>")
            else:
                parts.append(f"<img src=\"{c['image_url']['url']}\">")
    parts.append(f"<h2>tools</h2><pre>{html.escape(json.dumps(tools, indent=1))}</pre>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("<!doctype html><meta charset=utf-8><title>Packet</title><style>body{font:14px sans-serif;"
                   "max-width:900px;margin:1em auto;padding:0 16px}pre{white-space:pre-wrap;background:#f3f3f3;"
                   "padding:6px}img{max-width:100%;border:1px solid #888}</style>\n" + "\n".join(parts))


def example(out: Path) -> None:
    rooms = ["ward", "reception", "office", "storage"]
    ad = FakeAdapter("hall")
    ctx = SkillContext(ad, PLACES, "find", rooms)
    r1 = skills.execute("navigate_to", {"place": "storage"}, ctx)
    r2 = skills.execute("look_around", {}, ctx)
    hist = [history_line(1, "navigate_to(storage)", r1.text, "hall"), history_line(2, "look_around()", r2.text, "storage")]
    p = Packet(r2.photos, ad.where(), list(PLACES), hist, r2.text, "Find the red toolbox.", 3)
    render(build_messages(p), skills.build_tools(list(PLACES), "find", rooms), out)


if __name__ == "__main__":
    example(Path(sys.argv[1]))
