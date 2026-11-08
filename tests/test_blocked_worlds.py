"""Phase 6 blocked world variants (task 009, agent/INTERFACES.md section 11): generated dirs, truth side (WORLD_DIRS,
seeded.truth_for, run_truth, WORLD_TAG_KW), the scorer's reachable places, and the P3 ray-cast check (vet_seeds)."""
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLINIC = ROOT / "sim" / "worlds" / "clinic"
for p in (ROOT, ROOT / "probes", CLINIC):
    sys.path.insert(0, str(p))
import blocked as B  # noqa: E402
import generate as G  # noqa: E402
import phase1_checks as pc  # noqa: E402
import seeded as S  # noqa: E402
import vet_seeds as V  # noqa: E402
from adapters import run_task as rt  # noqa: E402
from agent import scorer  # noqa: E402
from agent.truth import BLOCKED_ROOMS, WORLD_DIRS, run_truth, world_truth_path  # noqa: E402
from test_scorer import SYN_TRUTH, TOOLBOX, make_run, world  # noqa: E402,F401  (world: fixture)

LAYOUT = CLINIC / "layout.yaml"
CLOSED = CLINIC / "generated"
WORLDS = [f"blocked_{r}" for r in BLOCKED_ROOMS]
SEEDS = [e["seed"] for e in json.loads((ROOT / "tasks" / "phase5_seeds.json").read_text())["ok"]][:3]


def test_blocked_rooms_are_the_rooms_with_a_place() -> None:
    places = yaml.safe_load(scorer.PLACES_YAML.read_text())["places"]
    assert set(BLOCKED_ROOMS) == set(S.ROOMS) and set(BLOCKED_ROOMS) <= set(places)


@pytest.mark.parametrize("room", BLOCKED_ROOMS)
def test_variant_dir_regenerates_byte_identical(room: str) -> None:
    d = WORLD_DIRS[f"blocked_{room}"]
    assert d == CLINIC / f"generated_blocked_{room}"
    assert sorted(p.name for p in d.iterdir()) == sorted(p.name for p in (CLINIC / "generated_cabinet_open").iterdir())
    sdf, truth = G.generate(LAYOUT, room)
    assert (d / "clinic.sdf").read_text() == sdf, f"generated_blocked_{room}/clinic.sdf is stale"
    assert (d / "truth.json").read_text() == truth, f"generated_blocked_{room}/truth.json is stale"
    assert (d / "map_knowledge.json").read_text() == G.map_knowledge(truth)


@pytest.mark.parametrize("room", BLOCKED_ROOMS)
def test_variant_is_closed_plus_the_blocker(room: str) -> None:
    d = WORLD_DIRS[f"blocked_{room}"]
    assert (d / "map_knowledge.json").read_bytes() == (CLOSED / "map_knowledge.json").read_bytes()
    t, c = json.loads((d / "truth.json").read_text()), json.loads((CLOSED / "truth.json").read_text())
    assert t["objects"] == c["objects"] and {k: v for k, v in t.items() if k != "blocker"} == c
    b = B.blocker(yaml.safe_load(LAYOUT.read_text()), room)
    assert t["blocker"] == {**b, "pose": [round(v, 4) for v in b["pose"]]}
    assert t["blocker"]["room"] == room and t["blocker"]["label"] == B.LABEL
    assert all(o["label"] != B.LABEL for o in t["objects"])


def test_truth_side_knows_the_variants() -> None:
    assert {"closed", "cabinet_open", *WORLDS} == set(WORLD_DIRS) == set(rt.WORLD_TAG_KW)
    assert set(S.WORLDS) == set(WORLD_DIRS) and S.PLACEMENT_WORLDS == ("closed", "cabinet_open")
    for w in WORLDS:
        kw = rt.WORLD_TAG_KW[w]
        assert kw["glob"] == "refs/tags/phase6-world-blocked*"
        assert kw["files"] == (f"sim/worlds/clinic/generated_{w}/clinic.sdf", f"sim/worlds/clinic/generated_{w}/truth.json")
        assert all((ROOT / f).is_file() for f in kw["files"])


@pytest.mark.parametrize("world", WORLDS)
def test_truth_for_and_run_truth_carry_the_blocker(world: str) -> None:
    blocker = json.loads(world_truth_path(world).read_text())["blocker"]
    assert run_truth(world)["blocker"] == blocker
    for seed in SEEDS[:2]:
        t, c = S.truth_for(seed, world), S.truth_for(seed, "closed")
        assert t["blocker"] == blocker and t["objects"] == c["objects"]
        sha = json.loads(world_truth_path(world).read_text())["layout_sha256"]
        assert t["layout_sha256"] == hashlib.sha256(f"{sha}:{seed}".encode()).hexdigest()
        assert run_truth(world, seed) == t
    assert "blocker" not in run_truth("closed", SEEDS[0])


def test_blocked_room_place_is_not_reachable(tmp_path: Path, world: dict) -> None:
    """failed: blocked on the blocked room's place is a result (vlm label here), on another room still nav."""
    truth = world["truth_path"]
    truth.write_text(json.dumps({**SYN_TRUTH, "blocker": {"id": "blocker_office", "room": "office", "label": 220}}))
    kw = {"n_steps": 12, "outcome": "fail_step_limit"}
    sc = scorer.score_run(make_run(tmp_path / "a", TOOLBOX, None, blocked_place="office", **kw), **world)
    assert sc["label"] == "vlm"
    sc = scorer.score_run(make_run(tmp_path / "b", TOOLBOX, None, blocked_place="storage", **kw), **world)
    assert sc["label"] == "nav" and "storage" in sc["label_why"]


def test_score_run_reads_the_blocker_from_the_run_world(tmp_path: Path) -> None:
    """No truth_path: the run.json world's truth (agent.truth.run_truth) reaches _blocked_reachable."""
    closed = run_truth("closed")
    toolbox = next(o for o in closed["objects"] if o["id"] == "toolbox_red_1")
    other = next(r for r in BLOCKED_ROOMS if r != toolbox["room"])
    labels = {}
    for w, place in ((f"blocked_{other}", other), ("closed", other), (f"blocked_{other}", toolbox["room"])):
        d = make_run(tmp_path / f"{w}_{place}", TOOLBOX, None, n_steps=12, outcome="fail_step_limit",
                     blocked_place=place)
        run = json.loads((d / "run.json").read_text())
        (d / "run.json").write_text(json.dumps(run | {"world": w, "layout_sha256": closed["layout_sha256"]}))
        labels[(w, place)] = scorer.score_run(d)["label"]
    assert labels == {(f"blocked_{other}", other): "vlm", ("closed", other): "nav",
                      (f"blocked_{other}", toolbox["room"]): "nav"}


@pytest.mark.parametrize("world", WORLDS)
def test_p3_check_world(world: str) -> None:
    """G2-G4 at every reachable place for listed seeds; the blocker (label 220) shows up yet breaks no check, and hides
    no own-room object pixel at a reachable place compared with the closed world."""
    room = world.removeprefix("blocked_")
    for seed in SEEDS:
        rec = V.check_world(seed, world, skip_blocked=True)
        assert rec["ok"], rec
        assert rec["skipped_place"] == room and set(rec["places"]) == set(S.PLACES) - {room}
        assert rec["blocker_px"]["hall"] > 0  # the box is in the hall photos
        poses = S.placements(seed)
        truth = S.truth_with(poses, "closed")
        for place in rec["places"]:
            a, b = S.photos(poses, world, place), S.photos(poses, "closed", place)
            for o in truth["objects"]:
                if o["room"] == place:
                    assert pc.pixel_counts(a, o["label"]) == pc.pixel_counts(b, o["label"]), (seed, place, o["id"])
    assert V.check_world(SEEDS[0], world, skip_blocked=False)["skipped_place"] is None


def test_p3_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "p3.json"
    monkeypatch.setattr(sys, "argv", ["vet_seeds.py", "--worlds", "blocked_storage,blocked_ward", "--check-seeds",
                                      str(SEEDS[0]), "--skip-blocked", "--out", str(out), "--jobs", "2"])
    with pytest.raises(SystemExit) as e:
        V.main()
    assert e.value.code == 0
    d = json.loads(out.read_text())
    assert d["not_ok"] == [] and [(r["seed"], r["world"]) for r in d["records"]] == \
        [(SEEDS[0], "blocked_storage"), (SEEDS[0], "blocked_ward")]
