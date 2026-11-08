"""Skills (INTERFACES section 2): tool schemas (one source, sent to the VLM and used to validate) and execution
on an Adapter. No ROS here."""
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from agent.robot import Adapter, Photo

TASK_KINDS = ("find", "check_state", "count", "conditional")
# Answer format per task type (brief: Tasks). Find/Conditional -> room name, Check state -> yes/no, Count -> integer.
COUNT_PATTERN = r"^(0|[1-9][0-9]*)$"
NAV_TIMEOUT_S = 120.0  # sim seconds, the G1 limit (INTERFACES section 1)
PLACE_BOUND_M = 0.30   # "at a place" radius (Phase 1 place bound; look_around start-heading rule)


def load_places(path: Path) -> dict[str, tuple[float, float, float]]:
    """places.yaml -> {name: (x, y, yaw)}, map frame."""
    raw = yaml.safe_load(Path(path).read_text())["places"]
    return {n: (float(p["x"]), float(p["y"]), float(p["yaw"])) for n, p in raw.items()}


@dataclass
class SkillContext:
    adapter: Adapter
    places: dict[str, tuple[float, float, float]]
    kind: str          # task kind, picks the report answer format
    rooms: list[str]   # valid room-name answers (find / conditional)

    def place_at(self) -> str | None:
        """The place whose point is within PLACE_BOUND_M of the localization pose (nearest), else None."""
        x, y, _ = self.adapter.pose()
        d, name = min((math.hypot(px - x, py - y), n) for n, (px, py, _) in self.places.items())
        return name if d <= PLACE_BOUND_M else None


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


def answer_schema(kind: str, rooms: list[str]) -> dict[str, Any]:
    if kind in ("find", "conditional"):
        return {"type": "string", "enum": rooms, "description": "room name"}
    if kind == "check_state":
        return {"type": "string", "enum": ["yes", "no"]}
    if kind == "count":
        return {"type": "string", "pattern": COUNT_PATTERN, "description": "a whole number, digits only"}
    raise ValueError(f"unknown task kind {kind!r}")


def build_tools(places: list[str], kind: str, rooms: list[str] | None = None) -> list[dict[str, Any]]:
    """The run's tool list. `places` = enum for navigate_to; `rooms` = valid room-name answers (default: places)."""
    return [
        _tool("navigate_to", "Go to a named place; returns a new photo.",
              {"place": {"type": "string", "enum": places}}, ["place"]),
        _tool("look_around", "Turn on the spot and take four numbered photos.", {}, []),
        _tool("report", "Give the final answer and end the run. photo = number of the photo (in the current "
              "message) that shows the evidence. note = free text, never scored.",
              {"answer": answer_schema(kind, rooms or places), "photo": {"type": "integer", "minimum": 1},
               "note": {"type": "string"}}, ["answer", "photo", "note"]),
    ]


def schemas(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {t["function"]["name"]: t["function"]["parameters"] for t in tools}


def parse_call(tool_calls: list[dict[str, Any]], tools: list[dict[str, Any]],
               shown_photos: int) -> tuple[str, dict[str, Any]] | str:
    """Validate the planner's tool call. Returns (name, args) or an error string to send back to the caller."""
    if len(tool_calls) != 1:
        return f"error: call exactly one tool per step (got {len(tool_calls)})"
    fn = tool_calls[0].get("function") or {}
    name, sch = fn.get("name"), schemas(tools)
    if name not in sch:
        return f"error: unknown skill {name!r}; available: {', '.join(sch)}"
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except (ValueError, TypeError):
        return f"error: arguments of {name} are not valid JSON"
    try:
        jsonschema.validate(args, sch[name])
    except jsonschema.ValidationError as e:
        where = "/".join(str(p) for p in e.absolute_path) or "arguments"
        return f"error: invalid {name} call at {where}: {e.message}"
    if name == "report" and args["photo"] > shown_photos:
        return f"error: photo {args['photo']} does not exist; the current message shows photos 1..{shown_photos}"
    return name, args


@dataclass
class SkillResult:
    text: str  # goes into history / last result
    photos: list[Photo] = field(default_factory=list)
    done: bool = False
    detail: dict[str, Any] = field(default_factory=dict)  # for the log only (nav status, turn errors)


def wrap(a: float) -> float:
    """Angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def look_around_start(ctx: SkillContext) -> float:
    """INTERFACES section 2: the yaw of the place the robot is at (within PLACE_BOUND_M), else the localization yaw."""
    place = ctx.place_at()
    return ctx.places[place][2] if place else ctx.adapter.pose()[2]


def execute(name: str, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
    ad = ctx.adapter
    if name == "navigate_to":
        x, y, yaw = ctx.places[args["place"]]
        go = ad.go_to(x, y, yaw, NAV_TIMEOUT_S)
        lx, ly, _ = ad.pose()
        loc_dist = math.hypot(lx - x, ly - y)
        # INTERFACES section 2: Nav2 may SUCCEED short of an unreachable goal (planner tolerance 0.5 m)
        reached = go.status == "reached" and loc_dist <= PLACE_BOUND_M
        f = ad.photo()  # also after a block: the view where the robot stopped
        return SkillResult("reached" if reached else "failed: blocked", [Photo(1, f.rgb, f)],
                           detail={"nav_status": go.nav_status, "nav_sim_s": go.sim_s, "nav_wall_s": go.wall_s,
                                   "loc_dist_m": loc_dist})
    if name == "look_around":
        start, photos, errs = look_around_start(ctx), [], []
        for k in range(4):
            errs.append(ad.turn_to(wrap(start + k * math.pi / 2)))
            f = ad.photo()
            photos.append(Photo(k + 1, f.rgb, f))
        return SkillResult(f"{len(photos)} photos", photos, detail={"start_yaw": start, "turn_err_rad": errs})
    ad.stop()  # report
    return SkillResult("reported", done=True)


def is_valid_count(s: str) -> bool:
    return re.fullmatch(COUNT_PATTERN, s) is not None
