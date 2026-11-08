"""Pure fake Adapter for tests: a small rectangle-room world with places and objects. Frames carry a synthetic
segmentation label image (same crop, uint8) so the scorer can be tested end to end. Not used in any real run."""
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from agent.robot import Adapter, Frame, GoResult

W, H = 320, 240
HALF_FOV = math.pi / 4  # 90 deg HFOV, as the sim camera
ROOMS = {"ward": (-5.0, 0.0, 0.0, 6.0), "reception": (-5.0, -6.0, 0.0, 0.0), "office": (0.0, 2.0, 5.0, 6.0),
         "storage": (0.0, -6.0, 5.0, -2.0), "hall": (0.0, -2.0, 5.0, 2.0)}  # xmin, ymin, xmax, ymax
PLACES = {"hall": (2.75, -0.1, 0.7854), "ward": (-2.5, 2.0, 0.0), "office": (2.5, 3.8, 0.0),
          "reception": (-2.5, -3.4, 0.0), "storage": (2.5, -3.9, 0.0)}


@dataclass(frozen=True)
class Obj:
    label: int
    x: float
    y: float
    colour: tuple[int, int, int]


OBJECTS = [Obj(10, 4.0, -3.9, (220, 30, 30)),     # red toolbox, storage, dead ahead of the storage place
           Obj(30, 2.5, -5.5, (40, 60, 200)),     # cabinet, storage, to the right
           Obj(20, -1.0, -3.4, (120, 80, 40)),    # chair, reception, ahead
           Obj(21, -2.5, -1.5, (120, 80, 40)),    # chair, reception, to the left
           Obj(1, -1.0, 2.0, (60, 60, 230))]      # bed, ward, ahead


def room_at(x: float, y: float) -> str:
    for name, (x0, y0, x1, y1) in ROOMS.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return "unknown"


class FakeAdapter(Adapter):
    """start: place name. blocked: goal points (x, y) that go_to refuses (robot stays). arrive_yaw_err: localization
    yaw after go_to = goal yaw + this (so the look_around start rule is visible). fail_on: a method name that raises
    RuntimeError (crash tests). short: (gx, gy, metres): go_to that goal SUCCEEDS but ends that far west of it (Nav2's
    planner tolerance snapping an unreachable goal to the nearest free cell)."""

    def __init__(self, start: str = "hall", blocked: tuple[tuple[float, float], ...] = (),
                 arrive_yaw_err: float = 0.2, fail_on: str | None = None,
                 short: tuple[tuple[float, float, float], ...] = ()) -> None:
        self.x, self.y, yaw = PLACES[start]
        self.yaw = yaw + arrive_yaw_err
        self.blocked, self.arrive_yaw_err, self.fail_on, self.short = blocked, arrive_yaw_err, fail_on, short
        self.t, self.heading_target, self.calls = 0.0, None, []
        self.stopped = False

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if name == self.fail_on:
            raise RuntimeError(f"fake adapter: injected failure in {name}")

    def go_to(self, x: float, y: float, yaw: float | None, timeout_s: float) -> GoResult:
        self._call("go_to")
        self.t += 10.0
        self.heading_target = None
        if any(math.hypot(x - bx, y - by) < 0.05 for bx, by in self.blocked):
            return GoResult("blocked", "ABORTED", 10.0, 0.0)
        off = next((m for gx, gy, m in self.short if math.hypot(x - gx, y - gy) < 0.05), 0.0)
        self.x, self.y = x - off, y
        self.yaw = (self.yaw if yaw is None else yaw) + self.arrive_yaw_err
        return GoResult("reached", "SUCCEEDED", 10.0, 0.0)

    def turn_to(self, yaw: float) -> float:
        self._call("turn_to")
        self.t += 1.0
        self.yaw = self.heading_target = yaw
        return 0.0

    def photo(self) -> Frame:
        self._call("photo")
        self.t += 0.1
        rgb = Image.new("RGB", (W, H), (210, 210, 200))
        seg = Image.new("L", (W, H), 0)
        dr, ds = ImageDraw.Draw(rgb), ImageDraw.Draw(seg)
        room = room_at(self.x, self.y)
        seen = [(math.hypot(o.x - self.x, o.y - self.y), o) for o in OBJECTS if room_at(o.x, o.y) == room]
        for d, o in sorted(seen, key=lambda t: -t[0]):  # far first, near ones drawn on top
            rel = math.atan2(math.sin(math.atan2(o.y - self.y, o.x - self.x) - self.yaw),
                             math.cos(math.atan2(o.y - self.y, o.x - self.x) - self.yaw))
            if abs(rel) >= HALF_FOV:
                continue
            cx, h = W / 2 - rel / HALF_FOV * W / 2, min(H - 20, 160 / max(d, 0.5))
            box = (cx - 0.35 * h, H / 2 - h / 2, cx + 0.35 * h, H / 2 + h / 2)
            dr.rectangle(box, fill=o.colour)
            ds.rectangle(box, fill=o.label)
        return Frame(rgb, np.asarray(seg, dtype=np.uint8).copy(),
                     {"sim_time_s": self.t, "heading_target_rad": self.heading_target,
                      "loc_pose": [self.x, self.y, self.yaw], "crop": {"top_px": 0, "bottom_px": 0}})

    def pose(self) -> tuple[float, float, float]:
        self._call("pose")
        return self.x, self.y, self.yaw

    def where(self) -> str:
        self._call("where")
        return room_at(self.x, self.y)

    def stop(self) -> None:
        self._call("stop")
        self.stopped = True

    def sim_time(self) -> float:
        return self.t

    def truth(self, place: str | None) -> dict[str, object]:
        """The harness-side truth hook for runtime.run (here localization is exact, so gt = pose)."""
        err = math.hypot(PLACES[place][0] - self.x, PLACES[place][1] - self.y) if place else None
        return {"gt_pose_after": [self.x, self.y, self.yaw], "place_err_m": err}
