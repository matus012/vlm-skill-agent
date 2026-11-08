"""The loop (INTERFACES section 4): packet -> planner -> check -> skill -> log -> repeat.
Ends on a valid report, after MAX_STEPS (fail_step_limit), after `malformed_limit` malformed planner outputs in a row
(fail_malformed), or on any exception (crash, logged, never raised). Each step logs the packet actually sent: its text
parts, the system prompt sha256 and per photo the rgb file's sha256 + the sent data URL's sha256 (no base64)."""
import hashlib
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent import skills
from agent.context import MAX_STEPS, Packet, build_messages, history_line, system_prompt
from agent.planners import Planner
from agent.robot import Photo
from agent.runlog import RunLog, utc_now
from agent.skills import SkillContext
from agent.tasks import Task

# Evaluator hook from the harness: place name (navigate_to) or None -> {"gt_pose_after": [x, y, yaw],
# "place_err_m": float | None}. The agent never imports anything that provides it.
Truth = Callable[[str | None], dict[str, Any]]
NO_TRUTH = {"gt_pose_after": None, "place_err_m": None}
PLANNER_KEYS = ("content", "reasoning", "finish_reason", "usage", "error")
AGENT_DIR = Path(__file__).resolve().parent
MALFORMED_LIMIT = 2  # task 005 design: two malformed outputs in a row end a VLM run (INTERFACES section 4)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass
class RunResult:
    outcome: str  # "reported" | "fail_step_limit" | "fail_malformed" | "crash"
    steps: int
    report: dict[str, Any] | None
    run_dir: Path
    traceback: str | None = None
    crash_origin: str | None = None


def crash_origin(tb: Any, ad: Any, planner: Planner, truth: Truth | None) -> str:
    """INTERFACES section 4: planner | agent | skill | adapter | truth_hook | runtime, from the frame right below run()
    (what the loop called), whether any deeper frame is a method of the adapter, and (under planner.next_call) whether
    any deeper frame is agent/ code."""
    below = [f for f, _ in traceback.walk_tb(tb)][1:]
    if not below:
        return "runtime"  # raised in run() itself
    top = below[0]
    if truth is not None and top.f_code is truth.__code__:
        return "truth_hook"
    if any(f.f_locals.get("self") is ad for f in below):
        return "adapter"
    if top.f_locals.get("self") is planner:  # N2: our own code under next_call is agent code, not the model call
        return "agent" if any(Path(f.f_code.co_filename).resolve().parent == AGENT_DIR for f in below) else "planner"
    if Path(top.f_code.co_filename).name == "context.py":
        return "agent"
    return "skill" if Path(top.f_code.co_filename) == Path(skills.__file__) else "runtime"


def _call_text(tool_calls: list[dict[str, Any]]) -> str:
    """History text of an invalid call: the first call as received, or "(invalid call)" when there is none."""
    fn = tool_calls[0].get("function") if tool_calls and isinstance(tool_calls[0], dict) else None
    return f"{fn.get('name')}({fn.get('arguments')})" if isinstance(fn, dict) else "(invalid call)"


def run(task: Task, ctx: SkillContext, planner: Planner, run_dir: Path, meta: dict[str, Any],
        truth: Truth | None = None, malformed_limit: int | None = None) -> RunResult:
    """meta: robot, layout_sha256, world_tag, git_commit, git_dirty, setup, start_place, hand_help (harness side).
    malformed_limit: that many malformed outputs in a row end the run (None: never; scripted Phase 2 controls)."""
    ad = ctx.adapter
    tools = skills.build_tools(list(ctx.places), task.kind, ctx.rooms)
    sp = system_prompt(list(ctx.places))
    log = RunLog(run_dir, {"task": task.run_json(), "planner": planner.name, "planner_info": planner.info,
                           "system_prompt": sp, "system_prompt_sha256": sha256(sp.encode()), "tools": tools,
                           "malformed_limit": malformed_limit, "hand_help": False, **meta,
                           "start_utc": utc_now(), "end_utc": None, "outcome": None, "steps": 0, "report": None,
                           "sim_s": None, "wall_s": None})
    res = RunResult("fail_step_limit", 0, None, Path(run_dir))
    wall0, sim0, sim_last = time.perf_counter(), None, None
    rec: dict[str, Any] | None = None
    try:
        sim0 = sim_last = ad.sim_time()
        f = ad.photo()
        photos = [Photo(1, f.rgb, f)]
        shown = log.save_photos(0, photos)
        history: list[str] = []
        last, malformed = "start", 0
        for step in range(1, MAX_STEPS + 1):
            rec = {"step": step, "packet": None, "call": {"raw": None, "name": None, "args": None}, "check": None,
                   "result": None, "photos_out": [], "sim_t0": None, "sim_t1": None, "wall_t0": time.time(),
                   "wall_t1": None, "skill_sim_s": 0.0, "skill_wall_s": 0.0, "truth": dict(NO_TRUTH),
                   "planner_out": None, "planner_wall_s": None}
            rec["sim_t0"] = sim_last = ad.sim_time()
            packet = Packet(photos, ad.where(), list(ctx.places), list(history), last, task.prompt, step)
            msgs = build_messages(packet)
            parts = msgs[1]["content"]
            urls = [c["image_url"]["url"] for c in parts if c["type"] == "image_url"]
            rec["packet"] = {"room": packet.room, "history": packet.history, "last_result": last,
                             "photos": [{**{k: e[k] for k in ("n", "rgb", "seg")},
                                         "rgb_sha256": sha256((log.dir / e["rgb"]).read_bytes()),
                                         "sent_sha256": sha256(u.encode())} for e, u in zip(shown, urls, strict=True)],
                             "system_sha256": sha256(msgs[0]["content"].encode()),
                             "texts": [c["text"] for c in parts if c["type"] == "text"]}
            t = time.perf_counter()
            out = planner.next_call(msgs, tools)
            rec["planner_wall_s"] = time.perf_counter() - t
            rec["planner_out"] = {k: out.get(k) for k in PLANNER_KEYS}
            rec["call"]["raw"] = out["tool_calls"]
            parsed = (f"error: planner call failed ({out['error']})" if out["error"]
                      else skills.parse_call(out["tool_calls"], tools, len(photos)))
            sr = None
            if isinstance(parsed, str):
                rec["check"], call = parsed, _call_text(out["tool_calls"])
                place = None
                malformed = 0 if out["error"] else malformed + 1  # a planner-call failure is not a malformed output
            else:
                name, args = parsed
                rec["call"].update(name=name, args=args)
                rec["check"], malformed = "ok", 0
                call = f"navigate_to({args['place']})" if name == "navigate_to" else f"{name}()"
                place = args["place"] if name == "navigate_to" else None
                t, s = time.perf_counter(), ad.sim_time()
                sr = skills.execute(name, args, ctx)
                rec["skill_wall_s"], rec["skill_sim_s"] = time.perf_counter() - t, ad.sim_time() - s
                rec["result"], rec["skill_detail"] = sr.text, sr.detail
                rec["photos_out"] = saved = log.save_photos(step, sr.photos)
            if truth is not None:
                rec["truth"] = truth(place)
            rec["sim_t1"] = sim_last = ad.sim_time()
            rec["wall_t1"] = time.time()
            log.step(rec)
            rec, res.steps = None, step
            if sr is not None and sr.done:
                res.outcome, res.report = "reported", parsed[1]
                break
            if malformed_limit is not None and malformed >= malformed_limit:
                res.outcome = "fail_malformed"
                break
            last = sr.text if sr is not None else parsed
            history.append(history_line(step, call, last, packet.room))
            if sr is not None and sr.photos:
                photos, shown = sr.photos, saved
    except Exception as e:  # crash: anything raised by a skill, the adapter, the planner or the truth hook
        res.outcome, res.traceback = "crash", traceback.format_exc()
        res.crash_origin = ("gz_read" if getattr(e, "gz_read_failed", False)  # probes/gz_cli.GzReadError (dev. 13)
                            else crash_origin(e.__traceback__, ad, planner, truth))
        if rec is not None:
            rec["traceback"] = res.traceback
            log.step(rec)
            res.steps = rec["step"]
    sim_s = (sim_last - sim0) if sim0 is not None else 0.0
    log.finish({"end_utc": utc_now(), "outcome": res.outcome, "steps": res.steps, "report": res.report,
                "sim_s": sim_s, "wall_s": time.perf_counter() - wall0,
                **({"crash": res.traceback, "crash_origin": res.crash_origin} if res.traceback else {})})
    return res
