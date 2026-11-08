#!/usr/bin/env python3
"""4-panel demo video from one run dir (brief T10, task 005 side lane). Offline: no sim, no ROS, never touches a run.

    nice -n 19 $BUILD_DIR/.venv-g1/bin/python probes/demo_recorder.py RUN_DIR --out FILE.mp4 [--gt-trace]

(.venv-g1 has mujoco 3.17; the main venv must keep ROS's system numpy 1.x, so mujoco is not added there.)
Panels: sim view (MuJoCo re-render of the frozen clinic, from layout.yaml via generate_mjcf.py, ceiling dropped, robot
= its real body, TurtleBot 4 meshes or Unitree G1 MJCF), map (sim/maps/clinic, path, places, current goal), the photos the VLM was shown at
the step (packet photos; the reported proof photo ringed at the end), and the step text (call, check, result, VLM
reasoning excerpt; final score.json verbatim). Every frame carries a speed badge: skill time at x--act-speed, the
planner's wait squeezed to --think-show-s (factor shown), holds marked PAUSED.
Poses: --gt-trace reads RUN_DIR/gt_trace.jsonl (adapters/gt_trace.py, wall clock mapped to sim time through the step
stamps). Without it (development only): recorded anchors (setup GT pose, GT after every step, localization pose at
every photo) joined by A* on the map, and the panels say "interpolated".
Writes FILE.mp4 (h264) and FILE_contact.png (6 frames) plus FILE_f<i>.png full-size frames.
Seeded / spec layouts (task 011): the sim view is the run's world variant (closed / cabinet_open) with every object moved
to the run's truth pose (agent.truth.run_truth of run.json world + layout_seed / layout_spec) and parked objects left
out; that truth's layout_sha256 must equal the run's. The map panel is sim/maps/clinic_static for seeded / spec runs
(Phase 5+ navigate on it), else sim/maps/clinic; --map overrides. Blocked worlds (Phase 6) are drawn without the box.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.context import MAX_STEPS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import robot_bodies as rb  # noqa: E402

W, H, CAP = 1280, 720, 26          # frame, caption bar height per panel
FPS = 15
ACT_SPEED = 8.0                    # sim seconds per video second while a skill runs
THINK_SHOW_S = 2.0                 # planner wait longer than this is squeezed to it
MIN_STEP_S = 3.0                   # reading time per step: a shorter step is followed by a PAUSED hold
TITLE_S, END_S = 3.0, 6.0
MAP_YAML = ROOT / "sim/maps/clinic.yaml"
STATIC_MAP_YAML = ROOT / "sim/maps/clinic_static.yaml"  # Phase 5+ (008): seeded / spec runs navigate on it
LAYOUT_OF_WORLD = {"cabinet_open": ROOT / "sim/worlds/clinic/layout_cabinet_open.yaml"}  # else layout.yaml
LAYOUT = ROOT / "sim/worlds/clinic/layout.yaml"
PLACES = ROOT / "sim/worlds/clinic/places.yaml"
TRUTH = ROOT / "sim/worlds/clinic/generated/truth.json"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


# ---------------------------------------------------------------- pure parts (tests/test_demo_recorder.py)
@dataclass(frozen=True)
class Segment:
    kind: str      # title | think | act | hold | end
    step: int      # index into steps (-1 for title)
    sim_a: float
    sim_b: float
    show_s: float  # video seconds

    @property
    def speed(self) -> float | None:  # sim s per video s; None = paused
        return None if self.sim_b <= self.sim_a else (self.sim_b - self.sim_a) / self.show_s


def timeline(steps: list[dict], act_speed: float = ACT_SPEED, think_show_s: float = THINK_SHOW_S,
             title_s: float = TITLE_S, end_s: float = END_S) -> list[Segment]:
    prev = steps[0]["sim_t0"] - steps[0].get("planner_wall_s", 0.0)  # first planner call: wall s (no sim stamp)
    segs = [Segment("title", -1, prev, prev, title_s)]
    for i, s in enumerate(steps):
        gap, act = s["sim_t0"] - prev, s["sim_t1"] - s["sim_t0"]
        shown = 0.0
        if gap > 0:
            shown = min(gap, think_show_s)
            segs.append(Segment("think", i, prev, s["sim_t0"], shown))
        segs.append(Segment("act", i, s["sim_t0"], s["sim_t1"], act / act_speed))
        shown += act / act_speed
        if shown < MIN_STEP_S:
            segs.append(Segment("hold", i, s["sim_t1"], s["sim_t1"], MIN_STEP_S - shown))
        prev = s["sim_t1"]
    segs.append(Segment("end", len(steps) - 1, prev, prev, end_s))
    return segs


def frame_times(segs: list[Segment], fps: int = FPS) -> list[tuple[float, Segment]]:
    """One (sim time, segment) per video frame; frame k sits at video time k / fps."""
    out: list[tuple[float, Segment]] = []
    edges = np.cumsum([s.show_s for s in segs])
    for k in range(int(round(edges[-1] * fps))):
        v = k / fps
        j = int(np.searchsorted(edges, v, side="right"))
        seg = segs[min(j, len(segs) - 1)]
        t0 = edges[j - 1] if j > 0 else 0.0
        a = min(1.0, (v - t0) / seg.show_s)
        out.append((seg.sim_a + a * (seg.sim_b - seg.sim_a), seg))
    return out


def speed_label(seg: Segment) -> str:
    sp = seg.speed
    if sp is None:
        return "PAUSED"
    tag = ("x%.0f" % sp if sp >= 10 or abs(sp - round(sp)) < 0.15 else "x%.1f" % sp) if sp >= 1.5 else "x1"
    return f"{tag} (planner wait {seg.sim_b - seg.sim_a:.1f} s)" if seg.kind == "think" else f"{tag} sim time"


def layout(w: int = W, h: int = H) -> dict[str, tuple[int, int, int, int]]:
    """Panel boxes (x0, y0, x1, y1): 2 x 2 tiling of the frame."""
    hw, hh = w // 2, h // 2
    return {"sim": (0, 0, hw, hh), "map": (hw, 0, w, hh), "photos": (0, hh, hw, h), "text": (hw, hh, w, h)}


def wall_to_sim(steps: list[dict], wall: np.ndarray) -> np.ndarray:
    """Piecewise-linear map from steps.jsonl wall stamps to sim stamps (clamped outside the run)."""
    w = np.array([v for s in steps for v in (s["wall_t0"], s["wall_t1"])])
    t = np.array([v for s in steps for v in (s["sim_t0"], s["sim_t1"])])
    return np.interp(wall, w, t)


def trim(text: str | None, n: int = 420) -> str | None:
    if not text:
        return None
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + " ..."


def call_str(call: dict) -> str:
    if not call.get("name"):
        return f"(invalid call) {json.dumps(call.get('raw'))[:120]}"
    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in (call.get("args") or {}).items())
    return f"{call['name']}({args})"


# ---------------------------------------------------------------- poses
class Grid:
    def __init__(self, map_yaml: Path = MAP_YAML) -> None:
        self.name = str(Path(map_yaml).with_suffix("").relative_to(ROOT)) if Path(map_yaml).is_absolute() else str(map_yaml)
        m = yaml.safe_load(map_yaml.read_text())
        self.img = np.asarray(Image.open(map_yaml.parent / m["image"]).convert("L"))
        self.res, (self.ox, self.oy) = float(m["resolution"]), m["origin"][:2]
        self.h, self.w = self.img.shape

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.ox) / self.res, self.h - (y - self.oy) / self.res

    def astar(self, a: tuple[float, float], b: tuple[float, float], inflate: int = 4) -> list[tuple[float, float]]:
        free = self.img > 250
        occ = ~free
        for _ in range(inflate):
            occ = occ | np.roll(occ, 1, 0) | np.roll(occ, -1, 0) | np.roll(occ, 1, 1) | np.roll(occ, -1, 1)
        (ca, ra), (cb, rb) = (tuple(int(round(v)) for v in self.to_px(*p)) for p in (a, b))
        occ[ra, ca] = occ[rb, cb] = False
        dist, came, pq = {(ra, ca): 0.0}, {}, [(0.0, (ra, ca))]
        while pq:
            _, cur = heapq.heappop(pq)
            if cur == (rb, cb):
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (cur[0] + dr, cur[1] + dc)
                if not (0 <= n[0] < self.h and 0 <= n[1] < self.w) or occ[n]:
                    continue
                d = dist[cur] + math.hypot(dr, dc)
                if d < dist.get(n, 1e18):
                    dist[n], came[n] = d, cur
                    heapq.heappush(pq, (d + math.hypot(n[0] - rb, n[1] - cb), n))
        path, cur = [], (rb, cb)
        while cur in came:
            path.append(cur)
            cur = came[cur]
        path = [(ra, ca)] + path[::-1]
        xy = np.array([(self.ox + (c + 0.5) * self.res, self.oy + (self.h - r - 0.5) * self.res) for r, c in path])
        k = 5  # smooth the 8-connected staircase
        if len(xy) > 2 * k:
            xy[k:-k] = np.array([xy[i - k:i + k + 1].mean(0) for i in range(k, len(xy) - k)])
        xy[0], xy[-1] = a, b
        return [tuple(p) for p in xy]


def track_from_trace(run_dir: Path, steps: list[dict]) -> np.ndarray:
    rows = [json.loads(l) for l in (run_dir / "gt_trace.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "x" in r]
    t = wall_to_sim(steps, np.array([r["wall_t"] for r in rows]))
    return np.array([[ti, r["x"], r["y"], r["yaw"]] for ti, r in zip(t, rows)])


def track_interpolated(run: dict, steps: list[dict], grid: Grid) -> np.ndarray:
    anchors = [(steps[0]["sim_t0"], *run["setup"]["gt_pose"])]
    for s in steps:
        anchors.append((s["sim_t0"], *anchors[-1][1:]))
        anchors += [(p["meta"]["sim_time_s"], *p["meta"]["loc_pose"]) for p in s["photos_out"]]
        anchors.append((s["sim_t1"], *s["truth"]["gt_pose_after"]))
    rows: list[tuple[float, float, float, float]] = [anchors[0]]
    for (ta, xa, ya, qa), (tb, xb, yb, qb) in zip(anchors, anchors[1:]):
        if math.dist((xa, ya), (xb, yb)) < 0.3:
            rows.append((tb, xb, yb, qb))
            continue
        pts = np.array(grid.astar((xa, ya), (xb, yb)))
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        for j in range(1, len(pts)):
            q = qb if j == len(pts) - 1 else math.atan2(*(pts[j] - pts[j - 1])[::-1])
            rows.append((ta + (tb - ta) * seg[j] / seg[-1], pts[j][0], pts[j][1], q))
    return np.array(sorted(rows))


def pose_at(track: np.ndarray, t: float) -> tuple[float, float, float]:
    yaw = np.unwrap(track[:, 3])
    return tuple(float(np.interp(t, track[:, 0], track[:, i])) for i in (1, 2)) + (float(np.interp(t, track[:, 0], yaw)),)


# ---------------------------------------------------------------- panels
def fonts() -> dict[str, ImageFont.FreeTypeFont]:
    return {"cap": ImageFont.truetype(FONT, 14), "b": ImageFont.truetype(FONT_B, 18), "t": ImageFont.truetype(FONT, 16),
            "m": ImageFont.truetype(FONT_M, 15), "badge": ImageFont.truetype(FONT_B, 22),
            "big": ImageFont.truetype(FONT_B, 28), "lbl": ImageFont.truetype(FONT_B, 13)}


def caption(img: Image.Image, text: str, f: dict) -> None:
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, img.width, CAP), fill=(30, 30, 30))
    d.text((8, 5), text, font=f["cap"], fill=(235, 235, 235))


def wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        if cur and font.getlength(cur + " " + w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


class MapPanel:
    def __init__(self, grid: Grid, places: dict, rooms: dict, track: np.ndarray, interp: bool, f: dict) -> None:
        self.g, self.places, self.track, self.f = grid, places, track, f
        self.s = (H // 2 - CAP - 8) / grid.h
        base = Image.fromarray(grid.img).convert("RGB")
        self.base = base.resize((round(grid.w * self.s), round(grid.h * self.s)), Image.NEAREST)
        d = ImageDraw.Draw(self.base)
        for name, r in rooms.items():
            poly = np.array(r["polygon"])
            xy = self.px(poly[:, 0].min() + 0.25, poly[:, 1].max() - 0.2)
            d.rectangle(d.textbbox(xy, name, font=f["lbl"]), fill=(255, 255, 255))
            d.text(xy, name, font=f["lbl"], fill=(70, 70, 170))
        self.interp = interp

    def px(self, x: float, y: float) -> tuple[float, float]:
        c, r = self.g.to_px(x, y)
        return c * self.s, r * self.s

    def render(self, t: float, goal: str | None) -> Image.Image:
        img = Image.new("RGB", (W // 2, H // 2), (245, 245, 245))
        m = self.base.copy()
        d = ImageDraw.Draw(m)
        for name, (x, y, _) in self.places.items():
            cx, cy = self.px(x, y)
            ring = (230, 120, 0) if name == goal else (120, 120, 120)
            rr = 9 if name == goal else 4
            d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=ring, width=3 if name == goal else 1)
        past = self.track[self.track[:, 0] <= t]
        pts = [self.px(x, y) for x, y in past[:, 1:3]] + [self.px(*pose_at(self.track, t)[:2])]
        if len(pts) > 1:
            d.line(pts, fill=(0, 150, 60), width=3)
        x, y, q = pose_at(self.track, t)
        cx, cy = self.px(x, y)
        tri = [(cx + 11 * math.cos(-q), cy + 11 * math.sin(-q)), (cx + 7 * math.cos(-q + 2.4), cy + 7 * math.sin(-q + 2.4)),
               (cx + 7 * math.cos(-q - 2.4), cy + 7 * math.sin(-q - 2.4))]
        d.polygon(tri, fill=(200, 30, 30))
        img.paste(m, (8, CAP + 4))
        dd = ImageDraw.Draw(img)
        lx, ly = m.width + 24, CAP + 14
        legend = [((0, 150, 60), "robot path " + ("(interpolated, dev)" if self.interp else "(recorded GT)")),
                  ((200, 30, 30), "robot now (GT)" if not self.interp else "robot now (interpolated)"),
                  ((230, 120, 0), "goal of navigate_to"), ((120, 120, 120), "places (navigate_to targets)")]
        for col, txt in legend:
            dd.rectangle((lx, ly + 4, lx + 14, ly + 14), fill=col)
            dd.text((lx + 22, ly), txt, font=self.f["cap"], fill=(30, 30, 30))
            ly += 24
        for ln in wrap("Evaluator view: the VLM never sees this map, its coordinates or the path.", self.f["cap"],
                       W // 2 - lx - 10):
            dd.text((lx, ly + 10), ln, font=self.f["cap"], fill=(110, 110, 110))
            ly += 18
        caption(img, f"Map ({self.g.name}, the Nav2 map) with the robot path", self.f)
        return img


def photo_panel(run_dir: Path, photos: list[dict], step: int, proof: int | None, f: dict) -> Image.Image:
    img = Image.new("RGB", (W // 2, H // 2), (20, 20, 20))
    n = len(photos)
    cols = 1 if n == 1 else 2
    rows = 1 if n <= 2 else 2
    aw, ah = W // 2 - 8, H // 2 - CAP - 6
    cw, ch = aw // cols, ah // rows
    d = ImageDraw.Draw(img)
    for i, p in enumerate(photos):
        ph = Image.open(run_dir / p["rgb"]).convert("RGB")
        s = min((cw - 6) / ph.width, (ch - 6) / ph.height)
        ph = ph.resize((round(ph.width * s), round(ph.height * s)), Image.BILINEAR)
        x = 4 + (i % cols) * cw + (cw - ph.width) // 2
        y = CAP + 3 + (i // cols) * ch + (ch - ph.height) // 2
        img.paste(ph, (x, y))
        if proof == p["n"]:
            d.rectangle((x - 3, y - 3, x + ph.width + 2, y + ph.height + 2), outline=(255, 210, 0), width=5)
        d.rectangle((x, y, x + 26, y + 24), fill=(0, 0, 0))
        d.text((x + 13, y + 12), str(p["n"]), font=f["b"], fill=(255, 255, 255), anchor="mm")
    cap = f"Photos the VLM was shown at step {step + 1} ({n} photo{'s' if n > 1 else ''})"
    if proof is not None:
        cap = f"Step {step + 1} photos; ringed = photo {proof}, reported as proof"
    caption(img, cap, f)
    return img


def text_panel(run: dict, steps: list[dict], score: dict, seg: Segment, t: float, f: dict) -> Image.Image:
    img = Image.new("RGB", (W // 2, H // 2), (250, 250, 250))
    d = ImageDraw.Draw(img)
    x, y, tw = 14, CAP + 8, W // 2 - 28

    def put(txt: str, font: str = "t", col: tuple = (20, 20, 20), gap: int = 4, max_lines: int = 99) -> None:
        nonlocal y
        for ln in wrap(txt, f[font], tw)[:max_lines]:
            d.text((x, y), ln, font=f[font], fill=col)
            y += f[font].size + gap

    put(f'Task: "{run["task"]["prompt"]}"', "b")
    put(f'planner {run["planner"]}  |  robot {run["robot"]}', "cap", (90, 90, 90))
    y += 6
    if seg.kind == "title":
        put("Start: robot set at the hall by the evaluator (not a step).")
        put(f"Up to {MAX_STEPS} steps; one skill call per step.")
    elif seg.kind == "end":
        rep = run.get("report") or {}
        put(f'Run end: {run["outcome"]}, {run["steps"]} steps, sim {run["sim_s"]:.0f} s, wall {run["wall_s"]:.0f} s')
        put(f'Answer: {rep.get("answer")}  (photo {rep.get("photo")})  |  truth: {score["truth_answer"]}', "b")
        put(f'Note (never scored): "{trim(rep.get("note"), 160)}"', "cap", (60, 60, 60), max_lines=2)
        put("score.json: " + "  ".join(f'{k} {"PASS" if v else "FAIL"}' for k, v in score["checks"].items()), "m")
        verdict = "PASS" if score["pass"] else f'FAIL  label: {score["label"]}'
        y += 4
        put(verdict, "big", (0, 140, 50) if score["pass"] else (200, 30, 30))
        if score.get("label_why"):
            put(str(score["label_why"]), "cap", max_lines=3)
    else:
        s = steps[seg.step]
        phase = {"think": "planner choosing", "act": "skill running", "hold": "step done"}[seg.kind]
        put(f'Step {s["step"]}/{MAX_STEPS}  -  {phase}', "b")
        put("Call: " + (call_str(s["call"]) if seg.kind != "think" else "..."), "m")
        if seg.kind != "think":
            ok = s["check"] == "ok"
            put(f'Check: {s["check"]}', "m", (20, 20, 20) if ok else (200, 30, 30), max_lines=2)
            done = t >= s["sim_t1"] - 1e-6
            put(f'Result: {s["result"] if done else "..."}', "m")
        po = s.get("planner_out") or {}
        why = trim(po.get("reasoning")) or trim(po.get("content"))
        y += 4
        if why:
            put("VLM reasoning (excerpt):", "cap", (90, 90, 90))
            put(why, "cap", (40, 40, 40), 2, max_lines=7)
        elif run["planner"].startswith("scripted"):
            put("(scripted planner: no VLM reasoning)", "cap", (120, 120, 120))
    caption(img, "Agent log (steps.jsonl) and score (score.json), verbatim", f)
    return img


class SimView:
    """MuJoCo re-render of the clinic with the robot's real body at the recorded poses. Mesa software EGL (the laptop's
    GPU EGL hangs; MUJOCO_EGL_DEVICE_ID=2), never the NVIDIA GPU that Gazebo renders on. Only mj_forward runs.
    turtlebot4: rigid mocap body from the real TurtleBot 4 meshes (probes/robot_bodies.py build-tb4).
    g1: the articulated Unitree G1 MJCF (collisions off); joint angles from RUN_DIR/qpos.jsonl when it exists, else the
    base pose from gt_history.jsonl in a default stance."""

    def __init__(self, run: dict, track: np.ndarray, interp: bool, f: dict, run_dir: Path) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
        import mujoco
        self.robot = run.get("robot", "turtlebot4")
        sys.path.insert(0, str(ROOT / "sim/worlds/clinic"))
        from generate_mjcf import generate_mjcf
        from agent.truth import load_truth, run_truth, world_truth_path
        world = run.get("world", "closed")
        xml, labels = generate_mjcf(LAYOUT_OF_WORLD.get(world, LAYOUT))
        assert json.loads(labels)["layout_sha256"] == load_truth(world_truth_path(world))["layout_sha256"], \
            f"the layout file != the {world} world"
        truth = run_truth(world, run.get("layout_seed"), run.get("layout_spec"))
        assert truth["layout_sha256"] == run["layout_sha256"], "the rebuilt truth != the run's layout"
        root = ET.fromstring(xml)
        wb = root.find("worldbody")
        poses = {o["id"]: o["pose"] for o in truth["objects"]}
        movable = {o["id"] for o in load_truth(world_truth_path(world))["objects"]}  # every object; structure is not
        for body in list(wb.findall("body")):
            name = body.get("name")
            if name not in movable:
                continue  # structure
            if name not in poses:  # parked in this run: outside the building, left out of the view
                wb.remove(body)
                continue
            x, y, *_, yaw = poses[name]
            z = float(body.get("pos").split()[2])
            body.set("pos", f"{x:.4f} {y:.4f} {z:.4f}")
            body.set("quat", f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}")
        wb.remove(next(b for b in wb.findall("body") if b.get("name") == "structure_ceiling"))
        root.insert(0, ET.fromstring('<visual><global offwidth="1280" offheight="720"/>'
                                     '<headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45"/>'
                                     '<quality shadowsize="0"/></visual>'))
        self.mj = mujoco
        self.track, self.f, self.interp = track, f, interp
        self.qlog: tuple[np.ndarray, np.ndarray] | None = None
        self.hist: np.ndarray | None = None
        if self.robot == "g1":
            self._build_g1(root, run_dir)
        else:
            frag, assets = rb.load_tb4()
            asset = root.find("asset")
            if asset is None:
                asset = ET.SubElement(root, "asset")
            asset.extend(list(frag.find("asset")))
            wb.append(frag.find("worldbody/body"))
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
            self.what = "TurtleBot 4 meshes at recorded GT poses"
        self.data = mujoco.MjData(self.model)
        self.r = mujoco.Renderer(self.model, H // 2 - CAP, W // 2)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 90.0, -72.0, 4.8  # 013: -60 let hall walls block the view
        self.look: np.ndarray | None = None

    def _build_g1(self, root: ET.Element, run_dir: Path) -> None:
        mj = self.mj
        spec = mj.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        g1 = mj.MjSpec.from_file(str(rb.G1_XML))
        g1.meshdir = str((rb.G1_XML.parent / "meshes").resolve())
        for g in g1.geoms:
            g.contype, g.conaffinity = 0, 0
        spec.attach(g1, prefix="g1_", frame=spec.worldbody.add_frame())
        self.model = spec.compile()
        ref = mj.MjModel.from_xml_path(str(rb.G1_XML))
        names = [mj.mj_id2name(ref, mj.mjtObj.mjOBJ_JOINT, i) for i in range(ref.njnt)]
        adr = [int(self.model.joint("g1_" + n).qposadr[0]) for n in names]
        assert adr[1:] == list(range(adr[0] + 7, adr[0] + ref.nq)), "attached G1 qpos is not contiguous"
        self.adr, self.nq = adr[0], ref.nq
        qp = run_dir / "qpos.jsonl"
        if qp.exists():
            head, t, q = rb.read_qpos(qp)
            assert head["nq"] == ref.nq, f"{qp}: nq {head['nq']} != the g1_29dof model's {ref.nq}"
            self.qlog = (t, q)
            self.what = "Unitree G1 MJCF, joints replayed from this run's qpos log"
        else:
            self.hist = rb.read_history(run_dir / "gt_history.jsonl")
            key = ref.key_qpos[0] if ref.nkey else None
            pose = (("hip_pitch", -0.48), ("knee", 0.96), ("ankle_pitch", -0.48))  # flat soles at the logged pelvis z ~0.72
            bent = {int(ref.joint(f"{s}_{j}_joint").qposadr[0]): v for s in ("left", "right") for j, v in pose}
            self.stance, _ = rb.choose_stance(key, ref.qpos0, bent)
            self.what = f"Unitree G1 MJCF at GT base pose, legs not replayed (no joint log)"

    def _pose(self, t: float) -> tuple[float, float, float]:
        d = self.data
        if self.robot != "g1":
            x, y, q = pose_at(self.track, t)
            d.mocap_pos[0] = (x, y, 0.0)
            d.mocap_quat[0] = (math.cos(q / 2), 0, 0, math.sin(q / 2))
            return x, y, q
        if self.qlog is not None:
            qv = rb.interp_qpos(*self.qlog, t)
            d.qpos[self.adr:self.adr + self.nq] = qv
            w, qx, qy, qz = qv[3:7]
            return float(qv[0]), float(qv[1]), math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        x, y, z, yaw = rb.history_pose_at(self.hist, t)
        qv = self.stance.copy()
        qv[:7] = (x, y, z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        d.qpos[self.adr:self.adr + self.nq] = qv
        return x, y, yaw

    def render(self, t: float, badge: str) -> Image.Image:
        x, y, q = self._pose(t)
        self.mj.mj_forward(self.model, self.data)
        p = np.array([x, y, 0.3])
        self.look = p if self.look is None else self.look + 0.12 * (p - self.look)  # fixed azimuth, smoothed follow
        self.cam.lookat[:] = self.look
        self.r.update_scene(self.data, camera=self.cam)
        img = Image.new("RGB", (W // 2, H // 2))
        img.paste(Image.fromarray(self.r.render()), (0, CAP))
        what = self.what.replace("recorded GT", "interpolated (dev)") if self.interp else self.what
        caption(img, f"Sim view ({self.robot}): MuJoCo{' re-render' if self.robot != 'g1' else ''}; {what}", self.f)  # < 624 px
        d = ImageDraw.Draw(img)
        bw = self.f["badge"].getlength(badge) + 20
        col = (110, 110, 110) if badge == "PAUSED" else (220, 90, 0)
        d.rectangle((W // 2 - bw - 8, CAP + 8, W // 2 - 8, CAP + 42), fill=col)
        d.text((W // 2 - bw + 2, CAP + 12), badge, font=self.f["badge"], fill=(255, 255, 255))
        return img


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gt-trace", action="store_true", help="use RUN_DIR/gt_trace.jsonl (else interpolate, dev)")
    ap.add_argument("--map", type=Path, help="map yaml (default: clinic_static for seeded / spec runs, else clinic)")
    ap.add_argument("--act-speed", type=float, default=ACT_SPEED, help="sim s per video s while a skill runs")
    ap.add_argument("--think-show-s", type=float, default=THINK_SHOW_S, help="planner waits longer than this are squeezed to it")
    ap.add_argument("--title-s", type=float, default=TITLE_S, help="title hold, video s (013 showcase: shorter)")
    ap.add_argument("--end-s", type=float, default=END_S, help="final score hold, video s")
    a = ap.parse_args()
    rd = a.run_dir
    run = json.loads((rd / "run.json").read_text())
    steps = [json.loads(l) for l in (rd / "steps.jsonl").read_text().splitlines()]
    score = json.loads((rd / "score.json").read_text())
    seeded = run.get("layout_seed") is not None or run.get("layout_spec") is not None
    grid, f = Grid(a.map or (STATIC_MAP_YAML if seeded else MAP_YAML)), fonts()
    track = track_from_trace(rd, steps) if a.gt_trace else track_interpolated(run, steps, grid)
    places = {k: (v["x"], v["y"], v["yaw"]) for k, v in yaml.safe_load(PLACES.read_text())["places"].items()}
    mp = MapPanel(grid, places, json.loads(TRUTH.read_text())["rooms"], track, not a.gt_trace, f)
    sv = SimView(run, track, not a.gt_trace, f, rd)
    boxes = layout()
    segs = timeline(steps, a.act_speed, a.think_show_s, a.title_s, a.end_s)
    ft = frame_times(segs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                           "-preset", "medium", "-threads", "2", str(a.out)], stdin=subprocess.PIPE)
    n = len(ft)
    pick = [round(i * (n - 1) / 5) for i in range(6)]
    keep: list[Image.Image] = []
    for k, (t, seg) in enumerate(ft):
        i = max(seg.step, 0)
        s = steps[i]
        proof = (run.get("report") or {}).get("photo") if seg.kind == "end" else None
        goal = (s["call"].get("args") or {}).get("place") if s["call"]["name"] == "navigate_to" and seg.kind in (
            "act", "hold") else None
        frame = Image.new("RGB", (W, H))
        frame.paste(sv.render(t, speed_label(seg)), boxes["sim"][:2])
        frame.paste(mp.render(t, goal), boxes["map"][:2])
        frame.paste(photo_panel(rd, s["packet"]["photos"], i, proof, f), boxes["photos"][:2])
        frame.paste(text_panel(run, steps, score, seg, t, f), boxes["text"][:2])
        ImageDraw.Draw(frame).line((0, H // 2, W, H // 2), fill=(0, 0, 0), width=2)
        ImageDraw.Draw(frame).line((W // 2, 0, W // 2, H), fill=(0, 0, 0), width=2)
        ff.stdin.write(frame.tobytes())
        if k in pick:
            keep.append(frame)
            frame.save(a.out.with_name(f"{a.out.stem}_f{len(keep)}.png"))
    ff.stdin.close()
    assert ff.wait() == 0, "ffmpeg failed"
    sheet = Image.new("RGB", (W, H * 3 // 2))
    for j, fr in enumerate(keep):
        sheet.paste(fr.resize((W // 2, H // 2)), ((j % 2) * W // 2, (j // 2) * H // 2))
    sheet.save(a.out.with_name(f"{a.out.stem}_contact.png"))
    print(json.dumps({"out": str(a.out), "frames": n, "video_s": round(n / FPS, 2), "pick": pick,
                      "segments": [(s.kind, s.step, round(s.sim_b - s.sim_a, 2), round(s.show_s, 2)) for s in segs],
                      "poses": "gt_trace" if a.gt_trace else "interpolated"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
