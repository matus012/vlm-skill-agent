"""Pure parts of the task 012 G1 precondition probes (humanoid/probes/{ready,hp1_reach,g1_pairs,g1_spot,g1_vet_list,
r1_checks}.py): no ROS, no MuJoCo, synthetic lists only (never the test list)."""
import json
import math
from collections import Counter
from pathlib import Path

import pytest
import yaml

from humanoid.probes import g1_pairs, g1_spot, g1_vet_list as vl, hp1_reach as hp1, r1_checks, ready

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "tasks/phase8_dev.yaml"


def good_ready() -> dict:
    return {"odom_samples": 60, "layout_echo": True, "rtf": 1.0, "base_z": 0.78, "rgb_seg_pair": True,
            "sim_world": "closed", "scene_world": "closed"}


@pytest.mark.parametrize("key,bad", [("odom_samples", 49), ("layout_echo", False), ("rtf", 0.89), ("rtf", 1.11),
                                     ("rtf", None), ("base_z", 0.59), ("base_z", None), ("rgb_seg_pair", False),
                                     ("scene_world", "cabinet_open")])
def test_ready_bar(key: str, bad: object) -> None:
    assert ready.ready_problems(good_ready()) == []
    assert len(ready.ready_problems({**good_ready(), key: bad})) == 1


def test_params_bar() -> None:
    k = [320.0, 0.0, 319.5, 0.0, 320.0, 239.5, 0.0, 0.0, 1.0]
    cmd = "/opt/venv-g1/bin/python humanoid/g1_node.py --world closed --rate 1.0 --torch-threads 2 --truth-log x "
    assert ready.params_problems(k, cmd, "a", "a") == []
    assert ready.params_problems([319.9999726081529, 0, 319.5, 0, 319.9999726081529, *k[5:]], cmd, "a", "a") == []
    assert ready.params_problems([319.9, *k[1:]], cmd, "a", "a") == ["camera_info fx 319.9 (want 320.0)"]
    assert len(ready.params_problems(k, cmd, "a", "b")) == 1
    assert len(ready.params_problems(k, cmd.replace("--rate 1.0", "--rate 0"), "a", "a")) == 1
    assert len(ready.params_problems(k, cmd.replace("--torch-threads 2", "--torch-threads 4"), "a", "a")) == 1
    assert ready.params_problems(None, cmd, "a", "a") == ["no camera_info"]


def test_hp1_seeds_are_vetted_dev_seeds_only() -> None:
    pool = hp1.dev_seeds()
    counted = {e.get("layout_seed") for e in yaml.safe_load((ROOT / "tasks/phase5_counted.yaml").read_text())}
    assert {474, 434, 196} <= pool and not pool & counted
    assert len(pool) == 50  # ok seeds x rule-c set minus counted, 2026-11-06


def try_rec(**kw: object) -> dict:
    return {"valid_start": True, "status": "reached", "err_m": 0.1, "sim_s": 30.0, "min_z": 0.75, "stalled": False,
            "collision": False, **kw}


@pytest.mark.parametrize("kw", [{"valid_start": False}, {"status": "blocked"}, {"status": "fell"}, {"err_m": 0.31},
                                {"sim_s": 120.1}, {"min_z": 0.49}, {"min_z": None}, {"stalled": True},
                                {"collision": True}])
def test_hp1_try_bar(kw: dict) -> None:
    assert hp1.try_pass(try_rec(), 0.5) and not hp1.try_pass(try_rec(**kw), 0.5)


def test_hp1_clearance() -> None:
    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    rows = [{"x": 2.0, "y": 0.5}, {"x": 1.3, "y": 0.5}]
    assert hp1.clearance(rows, [sq]) == pytest.approx(0.3)
    assert hp1.clearance([{"x": 0.5, "y": 0.5}], [sq]) == 0.0
    assert hp1.clearance(rows, []) is None and hp1.clearance([], [sq]) is None
    assert len(hp1.footprints(474)) == 5 and hp1.footprints(None) == []


def contact_rec(mocap: int = 0, static: int = 0, foot: float | None = 1.0) -> dict:
    return {"contacts_mocap": mocap, "contacts_static": static, "footprint_clearance_m": foot}


@pytest.mark.parametrize("rec, contact, contact_mocap, clearance", [
    (contact_rec(), False, False, False),
    (contact_rec(mocap=3), True, True, False),           # a movable touched while the proxy says clear
    (contact_rec(static=2), True, False, False),          # a wall / doorframe touch
    (contact_rec(foot=0.2), False, False, True),          # proxy violation without any contact
    (contact_rec(foot=None), False, False, False),        # closed world: no footprints
    (contact_rec(mocap=1, static=1, foot=0.34), True, True, True)])
def test_hp1_collision_defs(rec: dict, contact: bool, contact_mocap: bool, clearance: bool) -> None:
    got = [hp1.collided(rec, how) for how in hp1.COLLISION_DEFS]
    assert hp1.COLLISION_DEFS == ("contact", "contact_mocap", "clearance") and got == [contact, contact_mocap, clearance]
    for how, want in zip(hp1.COLLISION_DEFS, got):
        assert hp1.try_pass(try_rec(collision=hp1.collided(rec, how)), 0.5) is (not want)
    with pytest.raises(AssertionError):
        hp1.collided(rec, "proxy")


def test_hp1_contact_delta() -> None:
    def msg(t: float, nm: int, ns: int, cl: float | None, last: list | None = None) -> dict:
        return {"t_sim": t, "n_mocap_total": nm, "n_static_total": ns, "clearance_m": cl, "last": last or []}
    hit = {"robot_body": "left_knee_link", "other_body": "clinic/chair_orange_1", "t_sim": 10.25, "dist": -0.001}
    old = {**hit, "other_body": "clinic/structure_walls", "t_sim": 4.0}
    msgs = [msg(9.9, 5, 7, 0.01), msg(10.0, 5, 7, 0.5), msg(10.1, 5, 7, 0.4, [old]), msg(10.3, 9, 7, -0.01, [old, hit]),
            msg(10.4, 9, 8, None, [old, hit]), msg(10.5, 12, 8, 0.02)]
    d = hp1.contact_delta(msgs, 10.05, 10.35)
    assert d == {"contacts_mocap": 4, "contacts_static": 1, "contacts_seen": ["clinic/chair_orange_1"],
                 "clearance_m": -0.01}  # base 10.0 (last <= t0), end 10.4 (first >= t1); the 4.0 s wall touch is old
    assert hp1.contact_delta(msgs, 10.0, 10.3) == {**d, "contacts_static": 0}


def test_hp1_merge(tmp_path: Path) -> None:
    for tag in ("closed", "seed474"):
        for room in hp1.ROOMS:
            tries = [{"pass": True, **try_rec(), "fell": None} for _ in range(10)]
            if tag == "seed474" and room == "ward":
                tries[3] = {**tries[3], "pass": False, "collision": True}
            (tmp_path / f"hp1_{tag}_{room}.json").write_text(json.dumps(
                {"per_room": f"{sum(t['pass'] for t in tries)}/10", "history_file": "h", "tries": tries}))
    v = hp1.merge(tmp_path, [474], True)
    assert not v["pass"] and v["layouts"]["closed"]["pass"] and not v["layouts"]["seed474"]["pass"]
    assert v["fail_reasons"] == {"collision": 1} and len(v["tries"]) == 80
    v = hp1.merge(tmp_path, [474, 434], True)
    assert not v["pass"] and len(v["missing_parts"]) == 4


def test_g1_pairs_on_the_dev_list() -> None:
    entries = yaml.safe_load(DEV.read_text())
    tasks = {t["id"]: t for t in yaml.safe_load(g1_pairs.tasks_of(DEV).read_text())}
    c = g1_pairs.pairs(entries, tasks)
    assert c[("bed", "ward", None)] == 6 and c[("cabinet", "storage", "open")] == 2
    assert c[("cabinet", "storage", "closed")] == 1 and c[("toolbox", "storage", None)] == 2
    assert sum(c.values()) == 26 and g1_pairs.outside(c) == 0
    assert g1_pairs.outside(Counter({("toolbox", "ward", None): 2, ("chair", "ward", None): 1})) == 2


SPOT_LABEL = {"bed_blue_1": 3, "cabinet_blue_1": 4, "toolbox_red_1": 13, "chair_orange_1": 6}


def spot_seed(out: Path, seed: int, world: str, objs: dict[str, str], shots: dict[str, list[dict[str, int]]]) -> None:
    """A synthetic g1_spot live dir: objs id -> room (+ the bed in the ward and the cabinet in storage, in `world`'s
    state), shots room -> per photo k {id: px} (label images as phase1_checks.load_place reads them)."""
    from PIL import Image
    cls = {"bed_blue_1": "bed", "cabinet_blue_1": "cabinet", "toolbox_red_1": "toolbox", "chair_orange_1": "chair"}
    objs = {"bed_blue_1": "ward", "cabinet_blue_1": "storage", **objs}
    state = {"cabinet_blue_1": "open" if world == "cabinet_open" else "closed"}
    truth = {"objects": [{"id": i, "class": cls[i], "room": r, "label": SPOT_LABEL[i], "state": state.get(i),
                          "pose": [g1_spot.S.PLACES[r]["x"] + 1, g1_spot.S.PLACES[r]["y"], 0, 0, 0, 0]}
                         for i, r in objs.items()]}
    (out / str(seed)).mkdir(parents=True)
    (out / str(seed) / "truth.json").write_text(json.dumps(truth))
    for room, v in shots.items():
        d = out / str(seed) / "photos" / room
        d.mkdir(parents=True)
        for k, px in enumerate(v):
            seg = bytearray(64 * 64)
            pos = 0
            for oid, n in px.items():
                seg[pos:pos + n] = bytes([SPOT_LABEL[oid]]) * n
                pos += n
            Image.frombytes("L", (64, 64), bytes(seg)).save(d / f"{k}_seg.png")
            Image.new("RGB", (64, 64)).save(d / f"{k}_rgb.png")
        (d / "photos.json").write_text(json.dumps({"place": room, "photos": [
            {"k": k, "rgb": f"{k}_rgb.png", "seg": f"{k}_seg.png"} for k in range(len(v))]}))


SPOT_PAIRS = {"pairs": [{"class": "toolbox", "room": "reception", "state": None, "n": 1},
                        {"class": "bed", "room": "ward", "state": None, "n": 1},
                        {"class": "cabinet", "room": "storage", "state": None, "n": 1}]}


def test_g1_spot_pool_drops_failed_spot_seeds() -> None:
    pool = g1_spot.pool()
    assert 693 not in pool and g1_spot.pick_phase6.EXCLUDED and not g1_spot.pick_phase6.EXCLUDED & set(pool)
    assert set(pool) == hp1.dev_seeds() - {693} and pool == sorted(pool, reverse=True)


def test_g1_spot_pick_draws_until_three_seeds(tmp_path: Path) -> None:
    pairs = {"pairs": [{"class": "chair", "room": "reception", "state": None, "n": 1},
                       {"class": "toolbox", "room": "storage", "state": None, "n": 1},
                       {"class": "bed", "room": "ward", "state": None, "n": 1}]}
    ok, pool = g1_spot.sc.ok_seeds(), g1_spot.pool()
    got = g1_spot.pick(pairs, "closed", pool, ok, tmp_path)
    assert got[:3] == pool[:3] and set(got) <= set(pool) and len(set(got)) == len(got)
    for cls, room in (("chair", "reception"), ("toolbox", "storage")):
        assert sum(g1_spot.has(ok[s], cls, room) for s in got) >= 3
    opn = g1_spot.pick(pairs, "cabinet_open", pool, ok, tmp_path, frozenset(got))
    assert opn == [s for s in pool if s not in got][:3]  # disjoint from the closed seeds (one dir per seed)
    # a seed already run counts only with a real candidate photo: bed photos >= MIN_PX in 2 seeds -> 1 more seed
    for s, px in ((pool[0], 5000), (pool[1], 5000), (pool[2], 999)):
        spot_seed(tmp_path, s, "closed", {}, {"ward": [{"bed_blue_1": px}], "storage": [{"cabinet_blue_1": 4000}]})
    assert g1_spot.pick({"pairs": []}, "closed", pool, ok, tmp_path) == [pool[3]]
    assert g1_spot.pick({"pairs": []}, "closed", pool, ok, tmp_path, frozenset({pool[3]})) == [pool[4]]


def test_g1_spot_select_three_different_seeds(tmp_path: Path) -> None:
    # toolbox@reception: seed 1 has 3 photos, seed 2 one; fixed targets in seeds 1, 2 (closed) and 3 (open)
    tb3 = [{"toolbox_red_1": 2000}, {"toolbox_red_1": 3000}, {"toolbox_red_1": 4000}]
    spot_seed(tmp_path, 1, "closed", {"toolbox_red_1": "reception"},
              {"reception": tb3, "ward": [{"bed_blue_1": 3000}], "storage": [{"cabinet_blue_1": 3000}]})
    spot_seed(tmp_path, 2, "closed", {"toolbox_red_1": "reception"},
              {"reception": [{"toolbox_red_1": 1500}], "ward": [{"bed_blue_1": 3000}], "storage": [{"cabinet_blue_1": 3000}]})
    spot_seed(tmp_path, 3, "cabinet_open", {}, {"ward": [{"bed_blue_1": 3000}], "storage": [{"cabinet_blue_1": 3000}]})
    assert g1_spot.select(SPOT_PAIRS, tmp_path) == 1  # 3 toolbox photos, but from 2 seeds: short
    used = json.loads((tmp_path / "g9_seeds.json").read_text())
    assert used == {"bed": [1, 2, 3], "cabinet:closed": [1, 2], "cabinet:open": [3], "toolbox@reception": [1, 2]}
    sel = json.loads((tmp_path / "g9_selection.json").read_text())
    assert [c["state"] for c in sel["cabinet:open"]] == ["open"] and sel["bed"][0]["class"] == "bed"
    for s in (4, 5, 6):  # every fixed group now has 3 seeds; the toolbox still 3 photos from only 2 seeds
        spot_seed(tmp_path, s, "closed" if s == 4 else "cabinet_open", {}, {"storage": [{"cabinet_blue_1": 3000}]})
    assert g1_spot.select(SPOT_PAIRS, tmp_path) == 1
    assert len(json.loads((tmp_path / "g9_selection.json").read_text())["toolbox@reception"]) == 3
    spot_seed(tmp_path, 7, "closed", {"toolbox_red_1": "reception"}, {"reception": [{"toolbox_red_1": 1200}]})
    assert g1_spot.select(SPOT_PAIRS, tmp_path) == 0
    used = json.loads((tmp_path / "g9_seeds.json").read_text())
    assert used == {"bed": [1, 2, 3], "cabinet:closed": [1, 2, 4], "cabinet:open": [3, 5, 6],
                    "toolbox@reception": [1, 2, 7]}
    groups = g1_spot.g9.load_selection(tmp_path / "g9_selection.json")[1]
    assert groups["cabinet:open"] == {"class": "cabinet", "state": "open", "gated": True}  # the state call is made
    assert groups["toolbox@reception"] == {"class": "toolbox", "state": None, "gated": True}


def test_g1_spot_groups_and_verdict(tmp_path: Path) -> None:
    assert g1_spot.groups_for({"class": "cabinet", "room": "storage", "state": None}) == ["cabinet:open", "cabinet:closed"]
    assert g1_spot.groups_for({"class": "chair", "room": "ward", "state": None}) == ["chair@ward"]
    assert g1_spot.groups_for({"class": "bed", "room": "ward", "state": None}) == ["bed"]
    pairs = {"pairs": [{"class": "chair", "room": "ward", "state": None, "n": 2},
                       {"class": "cabinet", "room": "storage", "state": "open", "n": 1},
                       {"class": "bed", "room": "ward", "state": None, "n": 1}]}
    (tmp_path / "g9_sel").mkdir()
    (tmp_path / "g9_sel" / "g9_summary.json").write_text(json.dumps({"groups": {
        "chair@ward": {"pass": True}, "cabinet:open": {"pass": False}, "bed": {"pass": True}}}))
    (tmp_path / "g9_seeds.json").write_text(json.dumps({"chair@ward": [1, 2, 3], "cabinet:open": [4, 5, 6],
                                                         "bed": [1, 2]}))
    assert g1_spot.verdict(pairs, tmp_path, True) == 1
    v = json.loads((tmp_path / "hp3.json").read_text())
    assert [r["pass"] for r in v["pairs"]] == [True, False, False]  # bed: names pass but only 2 seeds
    assert v["pairs"][0]["seeds"] == {"chair@ward": [1, 2, 3]}


def counts(**px: int) -> list[int]:
    c = [0] * 256
    for k, v in px.items():
        c[int(k[1:])] = v
    c[0] = 640 * 480 - sum(px.values())
    return c


def test_vet_view_fails() -> None:
    truth = {"objects": [{"label": 5, "room": "ward", "countable": True}, {"label": 6, "room": "office",
                                                                           "countable": True}]}
    full = 640 * 480
    ok = {"ward": [counts(l5=2000, l240=full - 2000)] + [counts(l240=full)] * 3, "hall": [counts(l240=full)] * 4}
    assert vl.view_fails(truth, ok) == set()
    assert vl.view_fails(truth, {**ok, "ward": [counts(l5=900, l240=full - 900)] + ok["ward"][1:]}) == {"g2"}
    assert vl.view_fails(truth, {**ok, "ward": [counts(l5=2000, l240=full - 2000), counts(l5=300, l240=full - 300)]
                                 + ok["ward"][2:]}) == {"g3"}
    assert vl.view_fails(truth, {**ok, "ward": ok["ward"][:3] + [counts(l6=200, l240=full - 200)]}) == {"g4"}
    assert vl.view_fails(truth, {**ok, "hall": [counts(l240=full - 308)] * 4}) == {"g7"}
    assert vl.view_fails(truth, {**ok, "hall": [counts(l240=full - 307)] * 4}) == set()


def test_vet_legs_and_hosting(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = yaml.safe_load(DEV.read_text())[4]["layout_spec"]
    legs = vl.legs("closed")
    assert len(legs) > 20 and not vl.leg_hit(spec, legs)
    (a, b) = legs[0]
    r = vl.rz.realize(spec)
    on_leg = {**r, "poses": {**r["poses"], "chair_orange_1": ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.0)}}
    monkeypatch.setattr(vl.rz, "realize", lambda s: on_leg)
    assert vl.leg_hit(spec, legs)
    frozen = {o["id"]: o for o in json.loads(vl.S.world_truth_path("closed").read_text())["objects"]}
    bodies = {vl.PREFIX + frozen[o]["model_name"] for o in on_leg["poses"]}
    assert vl.unhosted("closed", spec, bodies) == 0
    assert vl.unhosted("closed", spec, bodies - {vl.PREFIX + frozen["chair_orange_1"]["model_name"]}) == 1


def test_vet_want_matches_run_task_layout_plan() -> None:
    from adapters.run_task import layout_plan
    from agent.truth import load_truth, world_truth_path
    for e in yaml.safe_load(DEV.read_text())[:5]:
        w = e["world"]
        assert vl.want_of(w, e["layout_spec"]) == layout_plan(w, load_truth(world_truth_path(w)), None,
                                                              e["layout_spec"])[0]


def test_r1_log_advance() -> None:
    text = ("[INFO] [1000.000] [g1_sim]: G1 up\n[INFO] [1010.500] [g1_sim]: sim 10.0 s, RTF 0.95, base z 0.78\n"
            "[INFO] [1020.000] [g1_sim]: sim 20.0 s, RTF 0.97, base z 0.78\n"
            "[INFO] [1200.000] [g1_sim]: sim 30.0 s, RTF 0.40, base z 0.78\n")
    assert r1_checks.log_advance(text) == {"lines": 3, "max_wall_gap_s": 180.0, "last_sim_s": 30.0, "last_rtf": 0.40}
    assert r1_checks.log_advance("")["max_wall_gap_s"] is None
    assert math.isclose(r1_checks.log_advance(text)["max_wall_gap_s"], 180.0)


def test_frozen_want_is_every_spec_object_at_its_frozen_pose() -> None:
    from humanoid.probes import hp2_photos
    from probes.spot_check import want_for
    fw = hp2_photos.frozen_want("closed")
    truth = {o["model_name"]: o for o in json.loads(vl.S.world_truth_path("closed").read_text())["objects"]}
    assert len(fw) == len(vl.rz.SPEC_OBJECTS) == 9
    assert all(fw[n] == (truth[n]["pose"][0], truth[n]["pose"][1], truth[n]["pose"][2], truth[n]["pose"][5]) for n in fw)
    seeded = want_for(474, "closed")
    assert set(seeded) < set(fw) and {**fw, **seeded}.keys() == fw.keys()


def test_hp1_contact_delta_base_is_the_first_message_at_t0() -> None:
    """145170: a probe's first move started before any /g1/contacts message; t0 is now taken from a received message,
    so the base exists even when it is the very first one."""
    msgs = [{"t_sim": 5.0, "n_mocap_total": 0, "n_static_total": 0, "clearance_m": 1.0, "last": []},
            {"t_sim": 5.1, "n_mocap_total": 2, "n_static_total": 0, "clearance_m": 0.2, "last": []}]
    d = hp1.contact_delta(msgs, msgs[0]["t_sim"], 5.1)
    assert d["contacts_mocap"] == 2 and d["contacts_static"] == 0
