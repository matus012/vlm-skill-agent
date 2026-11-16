"""Figures for the explainer PDF, drawn with matplotlib (the docs venv). Every number drawn here is a number the text
quotes from a report; the Wilson intervals are computed from the counts. `make_all(out_dir)` writes one PNG per name."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge

_TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for _f in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
    fm.fontManager.addfont(str(_TTF / _f))
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

INK, GREY, BLUE, ORANGE, RED, LIGHT = "#1f2933", "#6b7785", "#2563a8", "#c8641a", "#b3261e", "#eef2f7"
TEAL, GREEN, YELLOW, WOOD = "#1f8a8a", "#3f8f4f", "#e0b000", "#a9825a"
TYPES = ["Find", "Check state", "Count", "Conditional"]
# counts out of 25 (reports 011 and 012)
TB4_R1, TB4_R2, G1 = [21, 14, 23, 23], [24, 20, 24, 23], [24, 20, 24, 25]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc=LIGHT, ec=INK, fs=8, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, p, q, color=INK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ bodies
def bodies(out: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, title, cam_h, body in [(axs[0], "TurtleBot 4 (wheels, Gazebo)", 0.45, "wheels"),
                                   (axs[1], "Unitree G1 (legs, MuJoCo)", 1.2, "legs")]:
        ax.set_xlim(0, 6); ax.set_ylim(0, 2.2); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=9, fontweight="bold", color=INK)
        ax.plot([0, 6], [0, 0], color=INK, lw=1.5)
        ax.add_patch(Rectangle((4.5, 0), 0.9, 0.55, fc="#9fc5e8", ec=INK, lw=0.8))      # cabinet-like box
        ax.add_patch(Rectangle((3.6, 0), 0.5, 0.45, fc="#f0b27a", ec=INK, lw=0.8))      # chair-like box
        if body == "wheels":
            ax.add_patch(Rectangle((0.5, 0.08), 0.75, 0.32, fc=GREY, ec=INK, lw=0.8))
            ax.add_patch(Rectangle((0.7, 0.4), 0.3, 0.08, fc=GREY, ec=INK, lw=0.8))
            for cx in (0.65, 1.1):
                ax.add_patch(Circle((cx, 0.08), 0.08, fc=INK))
            cx0, cy0 = 0.85, cam_h
        else:
            ax.add_patch(Rectangle((0.78, 0), 0.12, 0.7, fc=GREY, ec=INK, lw=0.8))       # legs
            ax.add_patch(Rectangle((1.0, 0), 0.12, 0.7, fc=GREY, ec=INK, lw=0.8))
            ax.add_patch(Rectangle((0.72, 0.7), 0.46, 0.38, fc=GREY, ec=INK, lw=0.8))   # torso
            ax.add_patch(Circle((0.95, 1.2), 0.1, fc=GREY, ec=INK, lw=0.8))             # head
            cx0, cy0 = 1.0, cam_h
        ax.add_patch(Polygon([(cx0, cy0), (5.9, cy0 + 0.9 * 1.0), (5.9, max(cy0 - 1.1, 0.02))], fc=ORANGE, alpha=0.18, ec=ORANGE, lw=0.8))
        ax.plot([cx0], [cy0], "o", color=ORANGE, ms=4)
        ax.text(cx0 + 0.1, cy0 + 0.17, "camera", fontsize=7, color=ORANGE)
    fig.text(0.5, -0.03, "Same camera specification: 90° horizontal view, 640 × 480. Different height, gait and motion code. Sketch, not to scale.",
             ha="center", fontsize=7.5, color=GREY)
    fig.tight_layout()
    _save(fig, out, "bodies")


# ------------------------------------------------------------------ run timeline
def run(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 6.2); ax.axis("off")
    _box(ax, 0.1, 5.15, 14.2, 0.85, "Task: \"Unless the orange seats are in the ward, search the storage room.\"   Start: hall. "
         "Start photo: 1.", fc="#fdf4e7", ec=ORANGE, fs=7.2)
    rows = [("Step 1", "VLM calls\nnavigate_to(ward)", "runtime checks, robot drives to the ward\n-> \"reached\" + 1 photo"),
            ("Step 2", "VLM calls\nlook_around()", "robot turns 4 × 90°\n-> \"4 photos\", orange seats in photo 1"),
            ("Step 3", "VLM calls\nreport(\"ward\", photo 1)", "run ends -> script scores it against\nsimulator truth: PASS (answer, proof, steps, clean)")]
    for i, (s, a, b) in enumerate(rows):
        y = 3.55 - i * 1.55
        ax.text(0.1, y + 0.55, s, fontsize=8.5, fontweight="bold", color=INK, va="center")
        _box(ax, 1.4, y, 4.6, 1.1, a, fc="#dce8f6", ec=BLUE, fs=8)
        _arrow(ax, (6.0, y + 0.55), (6.9, y + 0.55))
        _box(ax, 6.9, y, 7.4, 1.1, b, fs=8)
    ax.text(14.3, 0.0, "run used 3 steps, limit 12", fontsize=7.5, color=GREY, ha="right")
    _save(fig, out, "run")


# ------------------------------------------------------------------ floor plan
def plan(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_aspect("equal"); ax.set_xlim(-5.4, 5.4); ax.set_ylim(-7.0, 7.0); ax.axis("off")
    rooms = {"WARD": (-5, 1.5, 0, 6.5), "OFFICE": (0, 1.5, 5, 6.5), "HALL": (-5, -1.5, 5, 1.5),
             "RECEPTION": (-5, -6.5, 0, -1.5), "STORAGE": (0, -6.5, 5, -1.5)}
    fills = {"WARD": "#e9f0fa", "OFFICE": "#eef6ee", "HALL": "#f6f6f6", "RECEPTION": "#fbf1e6", "STORAGE": "#f1eef8"}
    for n, (x0, y0, x1, y1) in rooms.items():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fills[n], ec=INK, lw=1.6))
        lx, ly, ha, va = {"WARD": (x0 + 0.1, y0 + 0.12, "left", "bottom"), "OFFICE": (x0 + 0.1, y0 + 0.12, "left", "bottom"),
                          "HALL": ((x0 + x1) / 2 - 1.5, 0.9, "center", "top"), "RECEPTION": (x0 + 0.1, y1 - 0.12, "left", "top"),
                          "STORAGE": (x1 - 0.1, y1 - 0.12, "right", "top")}[n]
        ax.text(lx, ly, n, ha=ha, va=va, fontsize=7.5, fontweight="bold", color=GREY)
    # doorways (gaps drawn over the walls)
    for xc, y in [(-1.2, 1.5), (1.2, 1.5), (-3.6, -1.5), (3.6, -1.5)]:
        ax.plot([xc - 0.7, xc + 0.7], [y, y], color="white", lw=3.2, solid_capstyle="butt", zorder=3)
    def rect(x, y, w, h, c, label=None, fs=5.5):
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=c, ec=INK, lw=0.6, zorder=4))
        if label:
            ax.text(x, y, label, fontsize=fs, ha="center", va="center", color="white" if c in (BLUE, TEAL, RED, GREEN) else INK, zorder=5)
    for x in (-4.0, -2.5, -1.0):
        rect(x, 5.46, 0.9, 1.9, BLUE, "bed")
    rect(-4.6, 3.0, 0.5, 0.8, "#dddddd"); rect(-0.45, 3.4, 0.12, 1.0, GREEN)
    rect(0.5, 4.0, 0.8, 1.8, TEAL, "sofa"); rect(2.5, 6.1, 0.7, 1.6, WOOD); rect(4.77, 4.0, 0.3, 1.5, WOOD)
    ax.add_patch(Circle((4.5, 6.0), 0.2, fc=GREEN, ec=INK, lw=0.5, zorder=4))
    for x in (-4.075, -3.025, -1.975, -0.925):
        rect(x, -6.15, 0.5, 0.5, ORANGE)
    rect(-4.55, -4.3, 0.6, 1.6, WOOD); rect(-0.3, -4.2, 0.4, 0.4, "#9fc5e8"); ax.add_patch(Circle((-0.5, -2.0), 0.2, fc=GREEN, ec=INK, lw=0.5, zorder=4))
    rect(0.4, -4.1, 0.25, 0.4, RED); rect(2.5, -6.19, 1.0, 0.5, BLUE, "cabinet"); rect(4.65, -4.3, 0.5, 1.5, "#bbbbbb"); rect(0.8, -2.0, 0.6, 0.6, "#c9a97a")
    places = {"hall": (2.75, -0.1), "ward": (-2.5, 2.0), "office": (2.5, 3.8), "reception": (-2.5, -3.4), "storage": (2.5, -3.9)}
    for n, (x, y) in places.items():
        ax.plot(x, y, marker="*", color=RED, ms=9, mec=INK, mew=0.5, zorder=6)
        ax.text(x + 0.18, y + 0.18, n, fontsize=6.5, color=RED, fontweight="bold", zorder=6)
    ax.add_patch(Rectangle((3.35, -1.35), 0.5, 0.35, fc=YELLOW, ec=INK, lw=0.6, zorder=5))
    ax.annotate("blocker box (some runs)", (3.6, -1.15), (-0.4, -0.95), fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY), zorder=7)
    ax.text(-5.3, -7.0, "red star: the five places the VLM can name.  Blue: beds, cabinet.  Orange: chairs.  Red: toolbox.  Teal: sofa.\n"
            "Walls have doorways. Hall 10 × 3 m, rooms 5 × 5 m. Movable: toolbox and chairs (positions shown are the fixed layout).",
            fontsize=6.2, color=GREY, va="top")
    _save(fig, out, "plan")


# ------------------------------------------------------------------ camera coverage
def camera(out: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    cols = [BLUE, ORANGE, GREEN, "#8e44ad"]
    for ax, hfov, title in [(axs[0], 71.6, "stock camera: 71.6° × 4"), (axs[1], 90.0, "project camera: 90° × 4")]:
        ax.set_aspect("equal"); ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.axis("off"); ax.set_title(title, fontsize=8, color=INK)
        ax.add_patch(Circle((0, 0), 1.0, fc="none", ec=GREY, lw=0.6, ls=":"))
        for k in range(4):
            c = 90 * k
            ax.add_patch(Wedge((0, 0), 1.0, c - hfov / 2, c + hfov / 2, fc=cols[k], alpha=0.35, ec=cols[k], lw=0.8))
        if hfov < 90:
            for k in range(4):
                c = 45 + 90 * k
                ax.add_patch(Wedge((0, 0), 1.0, c - 9.2, c + 9.2, fc="white", ec=RED, lw=0.8, hatch="////", alpha=1.0))
            ax.text(0, -1.2, "blind wedges of 18.4° between photos", ha="center", fontsize=6.5, color=RED)
        else:
            ax.text(0, -1.2, "tiles the circle; seams at 45°, 135° ...", ha="center", fontsize=6.5, color=INK)
        ax.plot([0], [0], "o", color=INK, ms=3)
    ax = axs[2]
    ax.set_xlim(0, 6.4); ax.set_ylim(-0.2, 4.8); ax.set_aspect("equal"); ax.axis("off"); ax.set_title("self-view crop", fontsize=8, color=INK)
    ax.add_patch(Rectangle((0, 0), 6.4, 4.8, fc="#cfe3f6", ec=INK, lw=0.8))
    ax.add_patch(Rectangle((0, 4.0), 6.4, 0.8, fc=RED, alpha=0.45, ec="none")); ax.add_patch(Rectangle((0, 0), 6.4, 0.6, fc=RED, alpha=0.45, ec="none"))
    ax.text(3.2, 4.4, "cropped: robot's top plate", ha="center", va="center", fontsize=6.3)
    ax.text(3.2, 0.3, "cropped: robot's body", ha="center", va="center", fontsize=6.3)
    ax.text(3.2, 2.3, "what the VLM gets\n(640 px wide)", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    _save(fig, out, "camera")


# ------------------------------------------------------------------ check-state breakdown
def checkstate(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    rows = [("Round 1 (prompt v1)", [(14, "#2e7d32", "pass 14"), (7, RED, "VLM took a look-alike\nfor the cabinet 7"), (4, GREY, "simulator\ncrash 4")]),
            ("Round 2 (prompt v2)", [(20, "#2e7d32", "pass 20"), (5, RED, "look-alike 5")])]
    for i, (lab, segs) in enumerate(rows):
        x = 0
        for n, c, t in segs:
            ax.barh(i, n, left=x, color=c, edgecolor="white", height=0.62)
            ax.text(x + n / 2, i, t, ha="center", va="center", color="white", fontsize=7, fontweight="bold")
            x += n
    ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 25); ax.set_xlabel("check-state tasks (25 per round)", fontsize=8); ax.tick_params(labelsize=7.5)
    _save(fig, out, "checkstate")


# ------------------------------------------------------------------ results
def results(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    series = [("TurtleBot 4, round 1", TB4_R1, "#b8c4d2"), ("TurtleBot 4, final", TB4_R2, BLUE), ("Unitree G1", G1, ORANGE)]
    w = 0.26
    for j, (lab, ks, c) in enumerate(series):
        xs = [i + (j - 1) * w for i in range(4)]
        ax.bar(xs, ks, width=w, color=c, label=lab, edgecolor="white")
        for x, k in zip(xs, ks):
            lo, hi = wilson(k, 25)
            ax.plot([x, x], [lo * 25, hi * 25], color=INK, lw=0.9)
            ax.text(x, hi * 25 + 0.5, str(k), ha="center", fontsize=7)
    ax.axhline(18, color=RED, lw=1.2, ls="--", label="bar: 18 of 25")
    ax.set_xticks(range(4)); ax.set_xticklabels(TYPES, fontsize=8.5)
    ax.set_ylim(0, 31); ax.set_ylabel("runs passed of 25", fontsize=8); ax.tick_params(axis="y", labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=4, columnspacing=1.2, handlelength=1.4)
    ax.set_title("Whiskers: Wilson 95 % interval. Round 1 and the final TurtleBot round used different sealed sets.", fontsize=7.5, color=GREY, loc="left")
    _save(fig, out, "results")


# ------------------------------------------------------------------ misses
def misses(out: Path) -> None:
    cats = ["Look-alike taken for\nthe blue cabinet", "Amber chairs refused\nas \"orange\"", "Chairs miscounted", "A find task,\nnot classified"]
    tb4, g1 = [5, 3, 1, 0], [5, 0, 1, 1]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ys = range(len(cats))
    ax.barh([y - 0.19 for y in ys], tb4, height=0.36, color=BLUE, label="TurtleBot 4, 9 misses")
    ax.barh([y + 0.19 for y in ys], g1, height=0.36, color=ORANGE, label="Unitree G1, 7 misses")
    for y, a, b in zip(ys, tb4, g1):
        ax.text(a + 0.08, y - 0.19, str(a), va="center", fontsize=7.5); ax.text(b + 0.08, y + 0.19, str(b), va="center", fontsize=7.5)
    ax.set_yticks(list(ys)); ax.set_yticklabels(cats, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlim(0, 6.2); ax.set_xlabel("failed runs of 100 (all labelled vlm)", fontsize=8); ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    _save(fig, out, "misses")


# ------------------------------------------------------------------ compute
def compute(out: Path) -> None:
    pts = [("cluster move", 10.52), ("phase 4", 14.87), ("phase 5", 22.72), ("phase 6", 27.29), ("Stage 1 test", 42.19), ("humanoid test", 49.7), ("showcase", 49.8)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.bar([p[0] for p in pts], [p[1] for p in pts], color=BLUE, width=0.6)
    for i, (_, v) in enumerate(pts):
        ax.text(i, v + 2, f"{v:g}", ha="center", fontsize=7.5)
    ax.axhline(120, color=RED, ls="--", lw=1.2); ax.text(6.4, 122, "approved cap: 120", color=RED, fontsize=7.5, ha="right")
    ax.axhspan(90, 115, color=GREY, alpha=0.15); ax.text(-0.4, 100, "brief's estimate: 90 to 115", fontsize=7.5, color=GREY, va="center")
    ax.set_ylim(0, 135); ax.set_ylabel("H200 GPU-hours, running total", fontsize=8); ax.tick_params(labelsize=7.5)
    _save(fig, out, "compute")


# ------------------------------------------------------------------ system diagram (ported from the showcase write-up)
def system(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    fs = 11
    ax.set_xlim(0, 15); ax.set_ylim(0, 5.2); ax.axis("off")
    y, h = 2.6, 1.3
    _box(ax, 0.1, y, 2.0, h, "Camera\nphotos", fs=fs)
    _box(ax, 2.7, y, 2.3, h, "VLM planner\n(Qwen3.8-27B)", fc="#dce8f6", ec=BLUE, fs=fs, bold=True)
    _box(ax, 5.6, y, 2.2, h, "One skill call\nnavigate_to,\nlook_around, report", fs=fs - 1)
    _box(ax, 8.4, y, 2.0, h, "Runtime\ncheck", fs=fs)
    _box(ax, 11.0, y, 1.8, h, "Robot\nadapter", fs=fs)
    _box(ax, 13.3, y + 0.75, 1.65, 0.75, "TurtleBot 4\nGazebo", fc="#fbe9d9", ec=ORANGE, fs=fs - 1)
    _box(ax, 13.3, y - 0.05, 1.65, 0.75, "Unitree G1\nMuJoCo", fc="#fbe9d9", ec=ORANGE, fs=fs - 1)
    cy = y + h / 2
    for a, b in [(2.1, 2.7), (5.0, 5.6), (7.8, 8.4), (10.4, 11.0)]:
        _arrow(ax, (a, cy), (b, cy))
    _arrow(ax, (12.8, cy + 0.2), (13.3, y + 1.12)); _arrow(ax, (12.8, cy - 0.2), (13.3, y + 0.32))
    ax.plot([14.1, 14.1, 1.1], [y - 0.05, 1.75, 1.75], color=GREY, lw=1.4, ls="--")
    _arrow(ax, (1.1, 1.75), (1.1, y), color=GREY)
    ax.text(7.6, 1.5, "robot moves, takes new photos", ha="center", va="center", fontsize=fs - 1, color=GREY)
    ax.text(8.4, y + h + 0.35, "rejects unknown skills, bad places, bad answers: the error goes back as a step",
            ha="center", fontsize=fs - 2, color=GREY, style="italic")
    ax.add_patch(FancyBboxPatch((1.5, 0.15), 10.0, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#fdecea", ec=RED, lw=1.2))
    ax.text(6.5, 0.55, "The VLM never sees: maps, lidar, coordinates, joint angles, robot type",
            ha="center", va="center", fontsize=fs, color=RED, fontweight="bold")
    ax.text(12.0, 0.55, "Every step is logged;\na script scores the run\nagainst simulator truth", ha="left", va="center", fontsize=fs - 1, color=GREY)
    fig.tight_layout(pad=0.3)
    _save(fig, out, "system")


def make_all(out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    for f in (system, bodies, run, plan, camera, checkstate, results, misses, compute):
        f(out)
    return {p.stem: p for p in out.glob("*.png")}


if __name__ == "__main__":
    import sys
    make_all(Path(sys.argv[1] if len(sys.argv) > 1 else "figs_preview"))
