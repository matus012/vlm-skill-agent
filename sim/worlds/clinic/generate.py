#!/usr/bin/env python3
"""Clinic world generator (task 002, sim/INTERFACES.md section 2).

    python3 sim/worlds/clinic/generate.py sim/worlds/clinic/layout.yaml --out sim/worlds/clinic/generated

Writes <out>/clinic.sdf, <out>/truth.json and <out>/map_knowledge.json (rooms + doorways only: the robot side's map
knowledge, no objects, no states; 005 Step 0 N8). Deterministic: same layout.yaml -> byte-identical output
(no timestamps, fixed float formatting, sorted JSON keys). Every object is a static model built from SDF
primitives (part library below) and carries a gz Label plugin with a unique id 1..199; floor / walls /
ceiling / door lintels are labelled 241 / 240 / 242 / 243. The robot is spawned later and never labelled.
Phase 5 adds --seed here; not now.

World variants (hpc/PERUN_SIM.md section 5): each is a committed layout file, generated into its own dir.
    closed       layout.yaml               -> generated/              (tag phase1-world-frozen-v2)
    cabinet_open layout_cabinet_open.yaml  -> generated_cabinet_open/ (layout.yaml with only cabinet_blue_1 open;
                 tests/test_world_variant.py proves the one-line diff and byte-identical regeneration)
    python3 sim/worlds/clinic/generate.py sim/worlds/clinic/layout_cabinet_open.yaml --out sim/worlds/clinic/generated_cabinet_open
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

STRUCTURE_LABELS: dict[str, int] = {"wall": 240, "floor": 241, "ceiling": 242, "trim": 243, "other": 244}

# RGB 0..1. "main" in a part means the object's own colour from layout.yaml.
COLOURS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.08, 0.06), "blue": (0.15, 0.35, 0.8), "green": (0.2, 0.6, 0.25), "white": (0.93, 0.93, 0.93),
    "grey": (0.55, 0.57, 0.6), "orange": (0.95, 0.5, 0.1), "teal": (0.1, 0.5, 0.55), "brown": (0.6, 0.42, 0.25),
    "wood": (0.62, 0.43, 0.25), "metal": (0.7, 0.72, 0.75), "black": (0.08, 0.08, 0.08),
    "darkgrey": (0.25, 0.25, 0.27), "lightwood": (0.82, 0.68, 0.5), "terracotta": (0.72, 0.36, 0.22),
    "leaf": (0.15, 0.5, 0.15), "lightblue": (0.55, 0.78, 0.95), "cardboard": (0.72, 0.56, 0.36),
    "mattress": (0.96, 0.96, 0.96), "yellow": (0.95, 0.8, 0.1),
    "wall": (0.88, 0.86, 0.8), "ceiling": (0.95, 0.95, 0.95), "trim": (0.45, 0.3, 0.18),
    "floor_hall": (0.62, 0.62, 0.6), "floor_ward": (0.7, 0.78, 0.82), "floor_office": (0.6, 0.5, 0.4),
    "floor_reception": (0.78, 0.72, 0.6), "floor_storage": (0.5, 0.52, 0.5),
}

Part = tuple  # ("box", (sx, sy, sz), (x, y, z), colour) | ("cyl", (r, h), (x, y, z), colour) | ("sph", r, (x, y, z), colour)


def _legs(sx: float, sy: float, h: float, w: float, colour: str) -> list[Part]:
    """Four square legs of width w and height h at the corners of an sx * sy footprint (floor at z = 0)."""
    return [("box", (w, w, h), (ix * (sx - w) / 2, iy * (sy - w) / 2, h / 2), colour) for ix in (-1, 1) for iy in (-1, 1)]


def _bed() -> list[Part]:  # hospital bed, local +x = foot end (open, blanket over it), head at -x; ~1.95 x 0.95
    return [("box", (1.6, 0.7, 0.3), (0, 0, 0.17), "darkgrey"),
            ("box", (1.9, 0.92, 0.1), (0, 0, 0.37), "metal"),
            ("box", (1.86, 0.88, 0.2), (0, 0, 0.52), "mattress"),
            ("box", (1.35, 0.94, 0.2), (0.3, 0, 0.57), "main"),
            ("box", (0.4, 0.62, 0.14), (-0.65, 0, 0.69), "white"),
            ("box", (0.05, 0.95, 0.75), (-0.955, 0, 0.775), "metal"),
            ("box", (0.04, 0.95, 0.04), (0.955, 0, 0.44), "darkgrey")] + [
        ("sph", 0.05, (ix * 0.8, iy * 0.38, 0.05), "black") for ix in (-1, 1) for iy in (-1, 1)]


def _chair() -> list[Part]:  # faces +x; 0.5 x 0.5
    return (_legs(0.46, 0.46, 0.44, 0.04, "darkgrey")
            + [("box", (0.5, 0.5, 0.06), (0, 0, 0.47), "main"),
               ("box", (0.05, 0.5, 0.5), (-0.225, 0, 0.75), "main")])


def _toolbox() -> list[Part]:  # portable tool box 0.8 x 0.4 x ~0.6, long side along local x
    return [("box", (0.8, 0.4, 0.36), (0, 0, 0.18), "main"),
            ("box", (0.82, 0.42, 0.1), (0, 0, 0.41), "main"),
            ("box", (0.83, 0.43, 0.03), (0, 0, 0.36), "black"),
            ("box", (0.06, 0.02, 0.08), (-0.25, 0.215, 0.3), "metal"),
            ("box", (0.06, 0.02, 0.08), (0.25, 0.215, 0.3), "metal"),
            ("box", (0.05, 0.05, 0.12), (-0.22, 0, 0.52), "black"),
            ("box", (0.05, 0.05, 0.12), (0.22, 0, 0.52), "black"),
            ("box", (0.5, 0.06, 0.05), (0, 0, 0.6), "black")]


def _cabinet(state: str) -> list[Part]:  # front +x; 0.5 deep, 1.0 wide, 1.9 tall; two doors hinged at the sides
    d, w, h, t = 0.5, 1.2, 1.8, 0.03
    parts: list[Part] = [("box", (t, w, h), (-d / 2 + t / 2, 0, h / 2), "darkgrey"),
                         ("box", (d, t, h), (0, -w / 2 + t / 2, h / 2), "main"),
                         ("box", (d, t, h), (0, w / 2 - t / 2, h / 2), "main"),
                         ("box", (d, w, t), (0, 0, h - t / 2), "main"),
                         ("box", (d, w, 0.1), (0, 0, 0.05), "black")]
    parts += [("box", (d - t, w - 2 * t, 0.02), (t / 2, 0, z), "main") for z in (0.55, 0.95, 1.35)]
    door_w = w / 2
    if state == "closed":
        for s in (-1, 1):
            parts.append(("box", (0.02, door_w - 0.02, h - 0.12), (d / 2 + 0.01, s * door_w / 2, h / 2 + 0.04), "main"))
            parts.append(("box", (0.03, 0.03, 0.3), (d / 2 + 0.035, s * 0.07, 1.0), "metal"))
        parts.append(("box", (0.01, 0.02, h - 0.12), (d / 2 + 0.005, 0, h / 2 + 0.04), "black"))
    elif state == "open":  # doors swung 90 deg out, hinge at the outer edges
        for s in (-1, 1):
            parts.append(("box", (door_w - 0.02, 0.02, h - 0.12), (d / 2 + door_w / 2, s * (w / 2 - 0.01), h / 2 + 0.04), "main"))
        # contents visible only when open: coloured boxes on the shelves
        parts += [("box", (0.3, 0.4, 0.25), (0, -0.25, 0.685), "yellow"), ("box", (0.3, 0.35, 0.3), (0, 0.25, 1.11), "orange"),
                  ("box", (0.3, 0.45, 0.3), (0, 0.2, 0.25), "cardboard"), ("box", (0.3, 0.35, 0.2), (0, -0.25, 1.46), "lightblue"),
                  ("box", (0.3, 0.3, 0.25), (0, 0.25, 0.685), "white")]
    else:
        raise ValueError(f"cabinet state must be open|closed, got {state!r}")
    return parts


def _desk() -> list[Part]:  # user side +x; 0.7 deep, 1.4 wide; monitor at the back facing +x
    return (_legs(0.7, 1.4, 0.72, 0.05, "darkgrey")
            + [("box", (0.7, 1.4, 0.04), (0, 0, 0.74), "main"),
               ("box", (0.04, 0.6, 0.38), (-0.2, 0, 1.03), "black"),
               ("box", (0.06, 0.06, 0.1), (-0.22, 0, 0.81), "black"),
               ("box", (0.18, 0.45, 0.02), (0.1, 0, 0.77), "darkgrey")])


def _bookshelf() -> list[Part]:  # front +x; 0.35 deep, 1.1 wide, 1.9 tall, 4 rows of books
    d, w, h, t = 0.35, 1.1, 1.9, 0.03
    parts: list[Part] = [("box", (t, w, h), (-d / 2 + t / 2, 0, h / 2), "main"),
                         ("box", (d, t, h), (0, -w / 2 + t / 2, h / 2), "main"),
                         ("box", (d, t, h), (0, w / 2 - t / 2, h / 2), "main"),
                         ("box", (d, w, t), (0, 0, h - t / 2), "main")]
    book_colours = ("red", "blue", "yellow", "green", "white", "orange", "teal", "darkgrey")
    for row, z in enumerate((0.05, 0.5, 0.95, 1.4)):
        parts.append(("box", (d, w, t), (0, 0, z - t / 2 + 0.03), "main"))
        for k in range(8):
            bh = 0.28 + 0.03 * ((k + row) % 3)
            parts.append(("box", (0.25, 0.1, bh), (0.0, -w / 2 + 0.1 + k * 0.125, z + 0.03 + bh / 2), book_colours[(k + 3 * row) % 8]))
    return parts


def _plant() -> list[Part]:  # pot + foliage, ~1.3 tall
    return [("cyl", (0.2, 0.42), (0, 0, 0.21), "terracotta"),
            ("cyl", (0.03, 0.35), (0, 0, 0.55), "brown"),
            ("sph", 0.38, (0, 0, 0.95), "leaf"),
            ("sph", 0.25, (0.15, 0.1, 1.2), "main"),
            ("sph", 0.22, (-0.12, -0.12, 1.15), "main")]


def _sofa() -> list[Part]:  # faces +x; 0.9 deep, 1.9 wide
    return [("box", (0.9, 1.9, 0.42), (0, 0, 0.21), "main"),
            ("box", (0.2, 1.9, 0.5), (-0.35, 0, 0.67), "main"),
            ("box", (0.9, 0.18, 0.25), (0, -0.86, 0.545), "main"),
            ("box", (0.9, 0.18, 0.25), (0, 0.86, 0.545), "main"),
            ("box", (0.65, 0.76, 0.1), (0.1, -0.39, 0.47), "lightblue"),
            ("box", (0.65, 0.76, 0.1), (0.1, 0.39, 0.47), "lightblue")]


def _trolley() -> list[Part]:  # medical cart 0.5 x 0.8, 3 trays, 1.0 tall
    parts: list[Part] = [("cyl", (0.02, 0.95), (ix * 0.22, iy * 0.37, 0.53), "metal") for ix in (-1, 1) for iy in (-1, 1)]
    parts += [("box", (0.5, 0.8, 0.03), (0, 0, z), "main") for z in (0.2, 0.55, 0.95)]
    parts += [("box", (0.48, 0.78, 0.25), (0, 0, 0.35), "lightblue"),
              ("box", (0.2, 0.3, 0.12), (0, -0.15, 1.025), "yellow")]  # sharps bin; red = toolbox look-alike (002 f13)
    parts += [("sph", 0.05, (ix * 0.22, iy * 0.37, 0.05), "black") for ix in (-1, 1) for iy in (-1, 1)]
    return parts


def _privacy_screen() -> list[Part]:  # 3 fabric panels in a frame along y, 1.8 wide, 1.75 tall
    parts: list[Part] = [("box", (0.02, 0.58, 1.4), (0, (k - 1) * 0.6, 1.0), "main") for k in range(3)]
    parts += [("cyl", (0.02, 1.75), (0, y, 0.875), "metal") for y in (-0.9, -0.3, 0.3, 0.9)]
    parts += [("box", (0.4, 0.04, 0.03), (0, y, 0.03), "metal") for y in (-0.9, -0.3, 0.3, 0.9)]
    return parts


def _reception_counter() -> list[Part]:  # staff side -x, customer side +x; 0.7 deep, 2.0 long, 1.1 tall
    return [("box", (0.6, 2.0, 1.05), (0, 0, 0.525), "main"),
            ("box", (0.75, 2.1, 0.05), (0.05, 0, 1.075), "lightwood"),
            ("box", (0.02, 1.6, 0.25), (0.31, 0, 0.75), "white"),
            ("box", (0.04, 0.45, 0.3), (-0.15, -0.4, 1.25), "black")]


def _water_cooler() -> list[Part]:  # front +x
    return [("box", (0.35, 0.35, 1.0), (0, 0, 0.5), "white"),
            ("cyl", (0.14, 0.45), (0, 0, 1.235), "main"),
            ("box", (0.04, 0.08, 0.06), (0.19, -0.07, 0.85), "red"),
            ("box", (0.04, 0.08, 0.06), (0.19, 0.07, 0.85), "blue")]


def _shelf_rack() -> list[Part]:  # front +x; metal rack 0.5 deep, 1.3 wide, 1.9 tall, 4 shelves with cartons
    parts: list[Part] = [("box", (0.04, 0.04, 1.9), (ix * 0.23, iy * 0.63, 0.95), "main") for ix in (-1, 1) for iy in (-1, 1)]
    for k, z in enumerate((0.15, 0.65, 1.15, 1.65)):
        parts.append(("box", (0.5, 1.3, 0.03), (0, 0, z), "main"))
        parts.append(("box", (0.4, 0.5, 0.3), (0, -0.33, z + 0.165), "cardboard"))
        if k % 2 == 0:
            parts.append(("box", (0.4, 0.45, 0.25), (0, 0.33, z + 0.14), "white"))
    return parts


def _boxes_stack() -> list[Part]:  # three cardboard cartons, footprint x in [-0.3, 1.05]
    return [("box", (0.6, 0.6, 0.45), (0, 0, 0.225), "main"),
            ("box", (0.5, 0.5, 0.4), (0.03, -0.02, 0.65), "main"),
            ("box", (0.6, 0.6, 0.45), (0.75, 0, 0.225), "main")]


def parts_for(cls: str, state: str | None) -> list[Part]:
    """Part list of one object class (local frame: origin on the floor at the footprint centre)."""
    if cls == "cabinet":
        return _cabinet(state)  # type: ignore[arg-type]
    if state is not None:
        raise ValueError(f"class {cls} has no states, got state={state!r}")
    return {"bed": _bed, "chair": _chair, "toolbox": _toolbox, "desk": _desk, "bookshelf": _bookshelf,
            "plant": _plant, "sofa": _sofa, "trolley": _trolley, "privacy_screen": _privacy_screen,
            "reception_counter": _reception_counter, "water_cooler": _water_cooler, "shelf_rack": _shelf_rack,
            "boxes_stack": _boxes_stack}[cls]()


# ---------------------------------------------------------------------------------------------------------- SDF
def f(v: float) -> str:
    """Fixed float format (determinism): 4 decimals, trailing zeros stripped, no -0."""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def fs(*vals: float) -> str:
    return " ".join(f(v) for v in vals)


def material(colour: str) -> str:
    r, g, b = COLOURS[colour]
    return (f"<material><ambient>{fs(r, g, b, 1)}</ambient><diffuse>{fs(r, g, b, 1)}</diffuse>"
            f"<specular>0.1 0.1 0.1 1</specular></material>")


def geometry(kind: str, size: object) -> str:
    if kind == "box":
        return f"<geometry><box><size>{fs(*size)}</size></box></geometry>"  # type: ignore[misc]
    if kind == "cyl":
        r, h = size  # type: ignore[misc]
        return f"<geometry><cylinder><radius>{f(r)}</radius><length>{f(h)}</length></cylinder></geometry>"
    if kind == "sph":
        return f"<geometry><sphere><radius>{f(size)}</radius></sphere></geometry>"  # type: ignore[arg-type]
    raise ValueError(kind)


def label_plugin(label: int) -> str:
    return f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{label}</label></plugin>'


def static_model(name: str, pose: tuple[float, ...], parts: list[Part], main: str, label: int) -> str:
    """One static model, one link, a visual + collision per part, model-level label."""
    lines = [f'    <model name="{name}">', "      <static>true</static>", f"      <pose>{fs(*pose)}</pose>",
             '      <link name="link">']
    for i, (kind, size, pos, colour) in enumerate(parts):
        geo = geometry(kind, size)
        c = main if colour == "main" else colour
        lines.append(f'        <collision name="c{i}"><pose>{fs(*pos, 0, 0, 0)}</pose>{geo}</collision>')
        lines.append(f'        <visual name="v{i}"><pose>{fs(*pos, 0, 0, 0)}</pose>{geo}{material(c)}</visual>')
    lines += ["      </link>", f"      {label_plugin(label)}", "    </model>"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- structure
def doorway_of(room: dict) -> tuple[str, float, float, float] | None:
    """(axis, line coordinate, gap lo, gap hi) of a room's doorway; axis 'y' = wall along x at y = line."""
    d = room.get("doorway")
    if d is None:
        return None
    x0, y0, x1, y1 = room["rect"]
    line = {"south": ("y", y0), "north": ("y", y1), "west": ("x", x0), "east": ("x", x1)}[d["wall"]]
    return line[0], line[1], d["center"] - d["width"] / 2, d["center"] + d["width"] / 2


def wall_segments(rooms: dict) -> list[tuple[str, float, float, float]]:
    """Merge every room edge per line, cut the doorways out. Returns (axis, line, lo, hi), sorted."""
    lines: dict[tuple[str, float], list[list[float]]] = {}
    for room in rooms.values():
        x0, y0, x1, y1 = room["rect"]
        for key, lo, hi in ((("y", y0), x0, x1), (("y", y1), x0, x1), (("x", x0), y0, y1), (("x", x1), y0, y1)):
            lines.setdefault(key, []).append([lo, hi])
    gaps: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for room in rooms.values():
        d = doorway_of(room)
        if d:
            gaps.setdefault((d[0], d[1]), []).append((d[2], d[3]))
    segs: list[tuple[str, float, float, float]] = []
    for key in sorted(lines):
        merged: list[list[float]] = []
        for lo, hi in sorted(lines[key]):
            if merged and lo <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        for lo, hi in merged:
            pieces = [(lo, hi)]
            for glo, ghi in gaps.get(key, []):
                pieces = [p for a, b in pieces for p in ((a, min(b, glo)), (max(a, ghi), b)) if p[1] - p[0] > 1e-9]
            segs += [(key[0], key[1], a, b) for a, b in pieces]
    return segs


def structure_models(L: dict) -> list[str]:
    rooms, t, wh = L["rooms"], L["wall"]["thickness"], L["wall"]["height"]
    xs = [r["rect"][i] for r in rooms.values() for i in (0, 2)]
    ys = [r["rect"][i] for r in rooms.values() for i in (1, 3)]
    bx0, bx1, by0, by1 = min(xs) - t / 2, max(xs) + t / 2, min(ys) - t / 2, max(ys) + t / 2
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    floor_parts: list[Part] = []
    for name in sorted(rooms):
        x0, y0, x1, y1 = rooms[name]["rect"]
        floor_parts.append(("box", (x1 - x0, y1 - y0, 0.1), ((x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy, -0.05), rooms[name]["floor"]))
    wall_parts: list[Part] = []
    for axis, line, lo, hi in wall_segments(rooms):
        ext_lo = t / 2 if not _is_gap_edge(rooms, axis, line, lo) else 0.0
        ext_hi = t / 2 if not _is_gap_edge(rooms, axis, line, hi) else 0.0
        a, b = lo - ext_lo, hi + ext_hi
        if axis == "y":
            wall_parts.append(("box", (b - a, t, wh), ((a + b) / 2 - cx, line - cy, wh / 2), "wall"))
        else:
            wall_parts.append(("box", (t, b - a, wh), (line - cx, (a + b) / 2 - cy, wh / 2), "wall"))
    trim_parts: list[Part] = []
    dh = L["door_height"]
    for name in sorted(rooms):
        d = doorway_of(rooms[name])
        if d is None:
            continue
        axis, line, lo, hi = d
        h = L["ceiling"]["z"] - dh
        if axis == "y":
            trim_parts.append(("box", (hi - lo, t + 0.04, h), ((lo + hi) / 2 - cx, line - cy, dh + h / 2), "trim"))
        else:
            trim_parts.append(("box", (t + 0.04, hi - lo, h), (line - cx, (lo + hi) / 2 - cy, dh + h / 2), "trim"))
    cz, ct = L["ceiling"]["z"], L["ceiling"]["thickness"]
    ceiling_parts: list[Part] = [("box", (bx1 - bx0, by1 - by0, ct), (0, 0, cz + ct / 2), "ceiling")]
    pose = (cx, cy, 0, 0, 0, 0)
    return [static_model("structure_floor", pose, floor_parts, "", STRUCTURE_LABELS["floor"]),
            static_model("structure_walls", pose, wall_parts, "", STRUCTURE_LABELS["wall"]),
            static_model("structure_trim", pose, trim_parts, "", STRUCTURE_LABELS["trim"]),
            static_model("structure_ceiling", pose, ceiling_parts, "", STRUCTURE_LABELS["ceiling"])]


def _is_gap_edge(rooms: dict, axis: str, line: float, v: float) -> bool:
    return any(d is not None and d[0] == axis and abs(d[1] - line) < 1e-9 and (abs(d[2] - v) < 1e-9 or abs(d[3] - v) < 1e-9)
               for d in (doorway_of(r) for r in rooms.values()))


def light(i: int, x: float, y: float, z: float) -> str:
    return (f'    <light type="point" name="lamp_{i}"><pose>{fs(x, y, z, 0, 0, 0)}</pose><cast_shadows>false</cast_shadows>'
            "<diffuse>0.8 0.8 0.78 1</diffuse><specular>0.1 0.1 0.1 1</specular>"
            "<attenuation><range>8</range><constant>0.4</constant><linear>0.08</linear><quadratic>0.02</quadratic></attenuation>"
            "</light>")


# -------------------------------------------------------------------------------------------------------- main
def generate(layout_path: Path, blocked: str | None = None) -> tuple[str, str]:
    """Return (sdf text, truth.json text) for a layout file. blocked: room whose doorway gets the blocker box (blocked.py)."""
    raw = layout_path.read_bytes()
    L = yaml.safe_load(raw)
    world, rooms = L["world"], L["rooms"]
    countable = set(L["countable_classes"])
    ids = [o["id"] for o in L["objects"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate object id")
    if len(ids) > 199:
        raise ValueError("more than 199 objects")
    objects_truth: list[dict] = []
    models: list[str] = []
    for label, o in enumerate(L["objects"], start=1):
        if o["room"] not in rooms:
            raise ValueError(f"{o['id']}: unknown room {o['room']}")
        x, y, yaw = (float(v) for v in o["pose"])
        pose = (x, y, 0.0, 0.0, 0.0, yaw)
        models.append(static_model(o["id"], pose, parts_for(o["class"], o["state"]), o["colour"], label))
        objects_truth.append({"id": o["id"], "class": o["class"], "colour": o["colour"], "room": o["room"],
                              "pose": [round(v, 4) for v in pose], "state": o["state"],
                              "countable": o["class"] in countable, "label": label, "model_name": o["id"]})
    extra: dict = {}
    if blocked is not None:
        import blocked as blocked_mod  # sibling module, only needed for the option
        b = blocked_mod.blocker(L, blocked)
        models.append(static_model(b["id"], tuple(b["pose"]), blocked_mod.parts(b), b["colour"], b["label"]))
        extra["blocker"] = {**b, "pose": [round(v, 4) for v in b["pose"]]}
    rooms_truth: dict[str, dict] = {}
    for name, r in rooms.items():
        x0, y0, x1, y1 = r["rect"]
        entry: dict = {"polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}
        d = doorway_of(r)
        if d is not None:
            axis, line, lo, hi = d
            c = (lo + hi) / 2
            entry["doorway"] = {"center": [round(v, 4) for v in ((c, line) if axis == "y" else (line, c))],
                                "width": r["doorway"]["width"]}
        rooms_truth[name] = entry
    truth = {"world": world, "layout_sha256": hashlib.sha256(raw).hexdigest(), "rooms": rooms_truth,
             "objects": objects_truth, "structure_labels": STRUCTURE_LABELS, **extra}
    lz = L["ceiling"]["z"] - 0.2
    sdf = "\n".join([
        '<?xml version="1.0"?>',
        f"<!-- GENERATED by sim/worlds/clinic/generate.py from {layout_path.name} (sha256 {truth['layout_sha256']}). Do not edit. -->",
        '<sdf version="1.9">',
        f'  <world name="{world}">',
        '    <physics type="ode"><max_step_size>0.003</max_step_size><real_time_factor>1.0</real_time_factor></physics>',
        '    <plugin name="gz::sim::systems::Physics" filename="gz-sim-physics-system"/>',
        '    <plugin name="gz::sim::systems::UserCommands" filename="gz-sim-user-commands-system"/>',
        '    <plugin name="gz::sim::systems::SceneBroadcaster" filename="gz-sim-scene-broadcaster-system"/>',
        '    <plugin name="gz::sim::systems::Contact" filename="gz-sim-contact-system"/>',
        "    <!-- no world-level Sensors system: the robot brings one; gz runs with the render-engine-server ogre2 flag -->",
        "    <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.3 0.7 0.9 1</background><shadows>false</shadows><grid>false</grid></scene>",
        *[light(i, x, y, lz) for i, (x, y) in enumerate(L["lights"])],
        *structure_models(L),
        *models,
        "  </world>",
        "</sdf>",
        "",
    ])
    return sdf, json.dumps(truth, indent=1, sort_keys=True) + "\n"


def map_knowledge(truth_text: str) -> str:
    """map_knowledge.json text: world, layout_sha256 and the rooms block (polygon + doorway) of truth.json, nothing else."""
    t = json.loads(truth_text)
    return json.dumps({k: t[k] for k in ("world", "layout_sha256", "rooms")}, indent=1, sort_keys=True) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("layout", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocked", metavar="ROOM", help="add the blocker box in this room's doorway (blocked.py)")
    a = ap.parse_args()
    sdf, truth = generate(a.layout, a.blocked)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "clinic.sdf").write_text(sdf)
    (a.out / "truth.json").write_text(truth)
    (a.out / "map_knowledge.json").write_text(map_knowledge(truth))
    print(f"wrote {a.out / 'clinic.sdf'}, {a.out / 'truth.json'} and {a.out / 'map_knowledge.json'}")


if __name__ == "__main__":
    main()
