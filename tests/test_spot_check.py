"""probes/spot_check.py (task 008 live spot check + G9 selection) and g9_readable --selection, with fakes: no gz, no ROS,
no VLM. The live path runs on ray-cast frames written as photos.py output (teleport / photos / checks faked)."""
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
sys.path.insert(0, str(ROOT / "tests"))
import g9_readable as g9  # noqa: E402
import phase1_gates as pg  # noqa: E402
import spot_check as sc  # noqa: E402
from mock_openai import MockServer  # noqa: E402

import adapters.run_task as rt  # noqa: E402  (sys.path: spot_check put ROOT first)
import map_world_check as mwc  # noqa: E402

S = sc.S
OK = sorted(sc.ok_seeds())
SEED, SEED2, WORLD = OK[-1], OK[-2], "closed"  # never hard-coded: the ok set is re-vetted (a recorded deviation)
GATES = ("g2", "g3", "g4", "g7")


@pytest.fixture(scope="module")
def pred() -> sc.Images:
    return sc.predicted(SEED, WORLD)


def copy(imgs: sc.Images) -> sc.Images:
    return {p: [i.copy() for i in v] for p, v in imgs.items()}


# ---------------------------------------------------------------------------------------------------- comparison
def test_identical_frames_agree_and_vetting_matches(pred: sc.Images) -> None:
    truth = S.truth_for(SEED, WORLD)
    assert sc.compare(copy(pred), pred, truth) == []
    assert sc.vetting_diffs(pred, truth, sc.ok_seeds()[SEED]) == []


def test_planted_disagreement_moved_chair(pred: sc.Images) -> None:
    """A chair's pixels moved to another photo of its room: presence diffs in both photos and a photo diff."""
    truth = S.truth_for(SEED, WORLD)
    chair = next(o for o in truth["objects"] if o["class"] == "chair" and o["room"] == "storage")
    live = copy(pred)
    k = sc.photo_of(live["storage"], chair["label"])
    j = (k + 2) % 4
    live["storage"][k][live["storage"][k] == chair["label"]] = 0
    live["storage"][j][:40, :50] = chair["label"]  # 2000 px
    d = sc.compare(live, pred, truth)
    assert {"kind": "photo", "room": "storage", "object": chair["id"], "live_k": j, "raycast_k": k} in d
    pres = {(x["k"], x["live_px"] >= 1000) for x in d if x["kind"] == "presence" and x["object"] == chair["id"]}
    assert pres == {(k, False), (j, True)}


def test_presence_threshold_and_missing_place(pred: sc.Images) -> None:
    truth = S.truth_for(SEED, WORLD)
    bed = next(o for o in truth["objects"] if o["id"] == "bed_blue_1")
    live = copy(pred)
    k = int(np.argmax(sc.pc.pixel_counts(live["ward"], bed["label"])))
    idx = np.flatnonzero(live["ward"][k] == bed["label"])
    assert idx.size > 1000
    live["ward"][k].flat[idx[999:]] = 0            # 999 px left: below G2_MIN_PX -> a presence diff
    del live["hall"]
    d = sc.compare(live, pred, truth)
    assert {"kind": "no_photos", "place": "hall"} in d
    assert [x for x in d if x["kind"] == "presence"] == [
        {"kind": "presence", "place": "ward", "k": k, "object": "bed_blue_1", "live_px": 999, "raycast_px": idx.size}]
    live["ward"][k].flat[idx[999]] = bed["label"]  # exactly 1000 px: present on both sides
    assert not [x for x in sc.compare(live, pred, truth) if x["kind"] == "presence"]


def test_vetting_drift_is_a_diff(pred: sc.Images) -> None:
    truth = S.truth_for(SEED, WORLD)
    vetted = json.loads(json.dumps(sc.ok_seeds()[SEED]))
    room, m = next(iter(vetted["chair_photos"].items()))
    cid = next(iter(m))
    m[cid] = (m[cid] + 1) % 4
    assert [x["kind"] for x in sc.vetting_diffs(pred, truth, vetted)] == ["vetting"]


# ------------------------------------------------------------------------------------------------- live, faked
def write_photos(sdir: Path, imgs: sc.Images) -> None:
    for place, v in imgs.items():
        d = sdir / "photos" / place
        d.mkdir(parents=True)
        for k, img in enumerate(v):
            cv2.imwrite(str(d / f"{k}_seg.png"), img)
            cv2.imwrite(str(d / f"{k}_rgb.png"), np.full((*img.shape, 3), 60 + 40 * k, np.uint8))
        (d / "photos.json").write_text(json.dumps({"place": place, "photos": [
            {"k": k, "rgb": f"{k}_rgb.png", "seg": f"{k}_seg.png"} for k in range(4)]}))


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch, pred: sc.Images) -> dict[str, Any]:
    """Fakes: teleport lands (records the request), photos = the ray-cast frames (optionally edited), checks pass."""
    st: dict[str, Any] = {"teleports": [], "edit": None, "failed_gates": (), "photos_raise": set()}
    monkeypatch.setattr(sc, "predicted", lambda seed, world: pred)
    # SEED2 reuses SEED's frames (the fake photos are SEED's ray-cast, so they agree with themselves); its vetting record
    # is SEED's too, and the vetting guard (tested above) is off for it
    monkeypatch.setattr(sc, "ok_seeds", lambda: {SEED: VETTED[SEED], SEED2: VETTED[SEED]})
    vd = sc.vetting_diffs
    monkeypatch.setattr(sc, "vetting_diffs", lambda p, truth, v: vd(p, truth, v) if truth == S.truth_for(SEED, WORLD) else [])
    monkeypatch.setattr(rt, "teleport", lambda want: st["teleports"].append(want) or None)

    def photos(robot: Any, places: dict, sdir: Path, rehearsal: bool) -> dict:
        if sdir.name in st["photos_raise"]:
            raise AssertionError("photos.py rc 1")
        imgs = copy(pred)
        if st["edit"]:
            st["edit"](imgs)
        write_photos(sdir, imgs)
        return {"pass": True}

    monkeypatch.setattr(pg, "gate_photos", photos)
    monkeypatch.setattr(pg, "gate_checks", lambda sdir, truth, places, reh: [
        {"gate": g, "pass": g not in st["failed_gates"]} for g in GATES])
    monkeypatch.setattr(mwc, "make_robot", lambda: object())
    monkeypatch.setenv("SIM_WORLD", WORLD)
    return st


VETTED = {SEED: sc.ok_seeds()[SEED]}


def test_live_agree_writes_truth_and_part(live: dict[str, Any], tmp_path: Path) -> None:
    assert sc.main(["--seeds", str(SEED), "--out", str(tmp_path)]) == 0
    part = json.loads((tmp_path / "spot_closed.json").read_text())
    rec = part["seeds"][str(SEED)]
    assert rec == {"world": "closed", "checks": {"photos": True, **{g: True for g in GATES}}, "agree": True, "diffs": []}
    assert json.loads((tmp_path / str(SEED) / "truth.json").read_text()) == S.truth_for(SEED, WORLD)
    # the teleport request is run_task's form: gz model name -> (x, y, frozen z, yaw) of every movable
    (want,) = live["teleports"]
    frozen = {o["id"]: o for o in json.loads(S.world_truth_path(WORLD).read_text())["objects"]}
    assert want == {frozen[o]["model_name"]: (x, y, frozen[o]["pose"][2], yaw) for o, (x, y, yaw) in S.placements(SEED).items()}


def test_live_planted_disagreement_and_failed_check(live: dict[str, Any], tmp_path: Path) -> None:
    truth = S.truth_for(SEED, WORLD)
    tb = next(o for o in truth["objects"] if o["id"] == S.TOOLBOX)

    def hide_toolbox(imgs: sc.Images) -> None:
        for img in imgs[tb["room"]]:
            img[img == tb["label"]] = 0
    live["edit"], live["failed_gates"] = hide_toolbox, ("g2",)
    assert sc.main(["--seeds", str(SEED), "--out", str(tmp_path)]) == 1
    rec = json.loads((tmp_path / "spot_closed.json").read_text())["seeds"][str(SEED)]
    kinds = {(x["kind"], x.get("object") or x.get("gate")) for x in rec["diffs"]}
    assert ("check", "g2") in kinds and ("presence", S.TOOLBOX) in kinds and ("photo", S.TOOLBOX) in kinds
    assert rec["agree"] is False and rec["checks"]["g2"] is False


def test_live_refuses_unvetted_seed(live: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not ok seeds"):
        sc.main(["--seeds", f"{SEED},{min(set(range(SEED)) - set(OK))}", "--out", str(tmp_path)])
    assert not live["teleports"]


def test_live_teleport_miss_and_crash_are_recorded(live: dict[str, Any], tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    live["photos_raise"] = {str(SEED)}
    assert sc.main(["--seeds", f"{SEED},{SEED2}", "--out", str(tmp_path)]) == 1  # SEED crashes, SEED2 still runs
    seeds = json.loads((tmp_path / "spot_closed.json").read_text())["seeds"]
    assert seeds[str(SEED)]["diffs"][0]["kind"] == "error" and "photos.py rc 1" in seeds[str(SEED)]["diffs"][0]["why"]
    assert seeds[str(SEED2)]["agree"] is True
    monkeypatch.setattr(rt, "teleport", lambda want: "chair_orange_1 at ...: 0.100 m")
    assert sc.main(["--seeds", str(SEED), "--out", str(tmp_path / "t")]) == 1
    rec = json.loads((tmp_path / "t" / "spot_closed.json").read_text())["seeds"][str(SEED)]
    assert rec == {"world": "closed", "checks": {}, "agree": False,
                   "diffs": [{"kind": "teleport", "why": "chair_orange_1 at ...: 0.100 m"}]}
    assert not (tmp_path / "t" / str(SEED) / "truth.json").exists()


# ------------------------------------------------------------------------------------------------------- merge
def part(out: Path, world: str, seeds: dict[str, bool]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"spot_{world}.json").write_text(json.dumps({"seeds": {s: sc.record(world, {g: True for g in GATES},
                                                                               [] if ok else [{"kind": "x"}])
                                                                  for s, ok in seeds.items()}}))


def test_merge_shape_and_agree_all(tmp_path: Path) -> None:
    part(tmp_path, "closed", {"1": True, "2": True})
    part(tmp_path, "cabinet_open", {"3": True})
    assert sc.main(["--merge", str(tmp_path)]) == 0
    s = json.loads((tmp_path / "spot.json").read_text())
    assert set(s) == {"seeds", "agree_all", "n_seeds"} and s["agree_all"] is True and s["n_seeds"] == 3
    assert set(s["seeds"]["3"]) == {"world", "checks", "agree", "diffs"} and s["seeds"]["3"]["world"] == "cabinet_open"
    part(tmp_path, "cabinet_open", {"3": False})
    assert sc.main(["--merge", str(tmp_path)]) == 1
    assert json.loads((tmp_path / "spot.json").read_text())["agree_all"] is False
    part(tmp_path, "cabinet_open", {"2": True})
    with pytest.raises(AssertionError, match="two parts"):
        sc.merge(tmp_path)
    (tmp_path / "empty").mkdir()
    assert sc.merge(tmp_path / "empty") == {"seeds": {}, "agree_all": False, "n_seeds": 0}  # no part: never agree_all


# ------------------------------------------------------------------------------------------------ G9 selection
LABEL = {"toolbox_red_1": 13, "chair_orange_1": 6, "chair_orange_2": 7, "chair_orange_3": 8, "chair_orange_4": 9}
W, H = 640, 377


def place_xy(room: str, d: float) -> list[float]:
    return [S.PLACES[room]["x"] + d, S.PLACES[room]["y"], 0, 0, 0, 0]


def make_seed(out: Path, seed: str, objs: dict[str, tuple[str, float]], shots: dict[str, list[dict[str, int]]]) -> None:
    """objs: id -> (room, distance from the room's place); shots: room -> per photo k {id: px}."""
    truth = {"objects": [{"id": i, "class": "toolbox" if i.startswith("toolbox") else "chair", "room": r,
                          "label": LABEL[i], "pose": place_xy(r, d), "countable": i.startswith("chair")}
                         for i, (r, d) in objs.items()]
             + [{"id": "chair_fixed_9", "class": "chair", "room": "office", "label": 30, "pose": place_xy("office", 1),
                 "countable": True}]}
    (out / seed).mkdir(parents=True)
    (out / seed / "truth.json").write_text(json.dumps(truth))
    for room, v in shots.items():
        d = out / seed / "photos" / room
        d.mkdir(parents=True)
        for k, px in enumerate(v):
            seg = bytearray(W * H)
            pos = 0
            for oid, n in px.items():
                seg[pos:pos + n] = bytes([LABEL.get(oid, 30)]) * n
                pos += n
            Image.frombytes("L", (W, H), bytes(seg)).save(d / f"{k}_seg.png")
            Image.new("RGB", (W, H), (int(seed) % 256, 20 * k, 100)).save(d / f"{k}_rgb.png")
        (d / "photos.json").write_text(json.dumps({"place": room, "photos": [
            {"k": k, "rgb": f"{k}_rgb.png", "seg": f"{k}_seg.png"} for k in range(len(v))]}))


@pytest.fixture
def spot(tmp_path: Path) -> Path:
    out = tmp_path / "spot"
    tw = {"reception": [{"toolbox_red_1": 5000}, {}, {}, {}]}
    make_seed(out, "10", {"toolbox_red_1": ("reception", 1.0)}, tw)
    make_seed(out, "11", {"toolbox_red_1": ("reception", 2.0)}, {"reception": [{}, {"toolbox_red_1": 1000}, {}, {}]})
    make_seed(out, "12", {"toolbox_red_1": ("reception", 1.1)}, {"reception": [{}, {}, {"toolbox_red_1": 4000}, {}]})
    make_seed(out, "13", {"toolbox_red_1": ("reception", 3.0)}, {"reception": [{}, {}, {}, {"toolbox_red_1": 999}]})
    make_seed(out, "14", {"toolbox_red_1": ("reception", 2.5)}, {"reception": [{"toolbox_red_1": 9000}, {}, {}, {}]})  # disagrees
    # chairs in office: seed 20 has two chairs (two photos), seed 21 one; the fixed chair never counts
    make_seed(out, "20", {"chair_orange_1": ("office", 1.0), "chair_orange_2": ("office", 2.0)},
              {"office": [{"chair_orange_1": 3000, "chair_fixed_9": 9000}, {"chair_orange_2": 2000, "chair_orange_1": 2500},
                          {"chair_fixed_9": 9000}, {}]})
    make_seed(out, "21", {"chair_orange_3": ("office", 1.5)}, {"office": [{}, {}, {"chair_orange_3": 1500}, {}]})
    seeds = {s: sc.record("closed", {}, [] if s != "14" else [{"kind": "x"}]) for s in
             ("10", "11", "12", "13", "14", "20", "21")}
    (out / "spot.json").write_text(json.dumps({"seeds": seeds, "agree_all": False, "n_seeds": len(seeds)}))
    return out


def test_g9_candidates_rules(spot: Path) -> None:
    agree = ["10", "11", "12", "13", "20", "21"]
    tb = {c["photo"]: c for c in sc.g9_candidates(spot, agree, "toolbox", "reception")}
    assert sorted(tb) == ["10/reception/0", "11/reception/1", "12/reception/2"]   # 1000 px counts, 999 does not; 14 not passed in
    assert tb["11/reception/1"]["pixels"] == 1000 and tb["11/reception/1"]["distance_m"] == 2.0
    assert tb["10/reception/0"]["rgb"] == "10/photos/reception/0_rgb.png" and tb["10/reception/0"]["group"] == "toolbox@reception"
    ch = {c["photo"]: c for c in sc.g9_candidates(spot, agree, "chair", "office")}
    assert sorted(ch) == ["20/office/0", "20/office/1", "21/office/2"]  # the fixed chair's photo 2 is no candidate
    assert ch["20/office/1"]["instance"] == "chair_orange_1" and ch["20/office/1"]["pixels"] == 2500  # most px wins


def test_g9_pick_prefers_new_seed_then_distance_spread() -> None:
    def c(seed: int, k: int, d: float, px: int) -> dict[str, Any]:
        return {"seed": seed, "photo": f"{seed}/r/{k}", "distance_m": d, "pixels": px}
    pool = [c(1, 0, 1.0, 9000), c(1, 1, 3.0, 8000), c(2, 0, 1.1, 7000), c(3, 0, 2.9, 1000), c(4, 0, 2.0, 5000)]
    got = [x["photo"] for x in sc.g9_pick(pool)]
    # first: most px; then new seeds only, the one farthest in distance (2.9 m) beats more px; then 2.0 m (gap 0.9)
    assert got == ["1/r/0", "3/r/0", "4/r/0"]
    assert [x["photo"] for x in sc.g9_pick(pool[:2])] == ["1/r/0", "1/r/1"]  # one seed only: its photos still fill


def test_g9_select_writes_selection_and_flags_short(spot: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert sc.main(["--g9-select", str(spot)]) == 1                 # other pairs have no photos here
    sel = json.loads((spot / "g9_selection.json").read_text())
    assert set(sel) == {f"{c}@{r}" for c, r in sc.NEW_PAIRS}
    assert sorted(x["seed"] for x in sel["toolbox@reception"]) == [10, 11, 12]  # 14 disagreed: never used
    assert len(sel["chair@office"]) == 3 and sel["chair@storage"] == []
    assert "SHORT" in capsys.readouterr().out


# -------------------------------------------------------------------------------------- g9_readable --selection
def test_g9_readable_selection_two_of_three(spot: Path, tmp_path: Path) -> None:
    sc.g9_select(spot)
    sel = {g: s for g, s in json.loads((spot / "g9_selection.json").read_text()).items() if s}
    (spot / "sel.json").write_text(json.dumps(sel))
    tb = [x["rgb"] for x in sel["toolbox@reception"]]
    colour = {Image.open(spot / p).getpixel((5, 5)): p for p in tb}

    def resp(miss: int):  # noqa: ANN202
        def f(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            import base64
            import io
            url = body["messages"][0]["content"][0]["image_url"]["url"]
            px = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB").getpixel((5, 5))
            near = colour[min(colour, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, px)))]  # JPEG-safe
            hit = near not in tb[:miss]
            return 200, {"role": "assistant", "content": "A red toolbox, orange chairs." if hit else "Orange chairs.",
                         "reasoning": "r"}
        return f

    for miss, rc_want in ((1, 0), (2, 1)):
        srv = MockServer([], responder=resp(miss))
        try:
            rc = g9.main(["--selection", str(spot / "sel.json"), "--out", str(tmp_path / f"m{miss}"),
                          "--base-url", srv.base_url, "--api-key", "EMPTY"])
        finally:
            srv.close()
        s = json.loads((tmp_path / f"m{miss}" / "g9_summary.json").read_text())
        assert rc == rc_want and s["groups"]["toolbox@reception"]["name_hits"] == 3 - miss
        assert s["groups"]["toolbox@reception"]["gated"] and s["groups"]["chair@office"]["pass"]
        assert s["calls"] == len({x["photo"] for v in sel.values() for x in v})  # one list call per photo, no state


def test_g9_readable_selection_xor_photos(spot: Path, tmp_path: Path) -> None:
    for argv in (["--out", str(tmp_path)],
                 ["--selection", str(spot / "spot.json"), "--photos", str(spot), "--truth", str(spot), "--out", str(tmp_path)]):
        with pytest.raises(SystemExit):
            g9.main(argv)


# ------------------------------------------------------------------------------------------------- plan seeds
def test_plan_seeds_cover_every_pair() -> None:
    plan = (ROOT / "hpc/plans/p5_spot.sh").read_text()
    lists = re.findall(r"sim_run (\d) \$PY probes/spot_check.py --seeds ([\d,]+)", plan)
    assert [i for i, _ in lists] == ["0", "1"] and "w1=cabinet_open map=clinic_static" in plan
    seeds = [int(s) for _, v in lists for s in v.split(",")]
    ok = sc.ok_seeds()
    counted = {int(x) for x in re.findall(r"layout_seed:\s*(\d+)", (ROOT / "tasks/phase5_counted.yaml").read_text())}
    assert len(set(seeds)) == len(seeds) >= 5 and all(s in ok for s in seeds) and not set(seeds) & counted
    # the seeds were taken from the top 12 of (ok - counted) against the counted list of their day (sha 2c205659, ad2c0ac);
    # 008's single re-pick (62a4775) then changed the counted list, so that window is history, not an invariant
    tb = [ok[s]["toolbox_room"] for s in seeds]
    assert tb.count("office") >= 3 and tb.count("reception") >= 3 and "storage" in tb   # G9: 3 photos per new pair
    assert all(any(ok[s]["chairs_per_room"][r] for s in seeds) for r in S.ROOMS)
    assert {c for c, _ in sc.NEW_PAIRS} == {"toolbox", "chair"} and ("toolbox", "ward") not in sc.NEW_PAIRS
    assert sum(ok[s]["any_room_spans_2_photos"] for s in seeds) >= 5
