"""probes/g9_readable.py against synthetic photo dirs (photos.py layout) and the mock OpenAI server."""
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
sys.path.insert(0, str(ROOT / "tests"))
import g9_readable as g9  # noqa: E402
from mock_openai import MockServer  # noqa: E402

W, H = 640, 377  # cropped frame size seen in 002-C photos
TARGETS = yaml.safe_load(g9.TARGETS.read_text())["classes"]
OBJS = [("toolbox_red_1", "toolbox", 1, None), ("cabinet_blue_1", "cabinet", 2, "closed"),
        ("chair_orange_1", "chair", 3, None), ("chair_orange_2", "chair", 4, None),
        ("bed_blue_1", "bed", 5, None), ("bed_blue_2", "bed", 6, None), ("plant_green_1", "plant", 7, None)]
LAB = {i: lab for i, _, lab, _ in OBJS}
# place -> per photo k: {object id: pixels}. Every gated group gets >= 3 candidate photos per set.
PHOTOS = {
    "storage": [{"toolbox_red_1": 5000, "cabinet_blue_1": 8000}, {"toolbox_red_1": 3000, "cabinet_blue_1": 2000},
                {"cabinet_blue_1": 1500, "toolbox_red_1": 999}, {"toolbox_red_1": 1000}],
    "reception": [{"chair_orange_1": 9000}, {"chair_orange_1": 7000, "chair_orange_2": 2000},
                  {"chair_orange_2": 1200, "plant_green_1": 4000}, {}],
    "ward": [{"bed_blue_1": 6000, "bed_blue_2": 6500}, {"bed_blue_1": 4000}, {"bed_blue_2": 3000}, {"plant_green_1": 50}],
}
COLOURS: dict[tuple[int, int, int], str] = {}


def make_set(root: Path, name: str, cabinet_state: str, si: int) -> tuple[Path, Path]:
    pdir = root / name
    objs = [{"id": i, "class": c, "colour": "x", "room": "r", "pose": [0] * 6,
             "state": cabinet_state if c == "cabinet" else s, "countable": c in ("chair", "bed"), "label": lab,
             "model_name": i} for i, c, lab, s in OBJS]
    truth = root / f"{name}_truth.json"
    truth.write_text(json.dumps({"world": "t", "objects": objs}))
    for pi, (place, shots) in enumerate(PHOTOS.items()):
        d = pdir / place
        d.mkdir(parents=True)
        meta = []
        for k, px in enumerate(shots):
            seg = bytearray(W * H)
            pos = 0
            for oid, n in px.items():
                seg[pos:pos + n] = bytes([LAB[oid]]) * n
                pos += n
            Image.frombytes("L", (W, H), bytes(seg)).save(d / f"{k}_seg.png")
            col = (40 + 60 * si, 30 + 50 * pi, 30 + 55 * k)
            COLOURS[col] = f"{si}:{name}/{place}/{k}"
            Image.new("RGB", (W, H), col).save(d / f"{k}_rgb.png")
            meta.append({"k": k, "rgb": f"{k}_rgb.png", "seg": f"{k}_seg.png"})
        (d / "photos.json").write_text(json.dumps({"place": place, "photos": meta}))
    return pdir, truth


@pytest.fixture
def sets(tmp_path: Path) -> list[tuple[Path, Path]]:
    return [make_set(tmp_path, "closed", "closed", 0), make_set(tmp_path, "open", "open", 1)]


def photo_of(body: dict[str, Any]) -> str:
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    px = im.getpixel((W // 2, H // 2))
    return COLOURS[min(COLOURS, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, px)))]


GOOD_LIST = "A red toolbox, a blue cabinet, orange chairs, blue hospital beds, a green plant."


def responder(lists: dict[str, str] | None = None, states: dict[str, str] | None = None, reasoning: str = "r" * 50):
    """Default: every list answer names everything; state answers say the true state of the photo's dir."""
    def f(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        p = photo_of(body)
        if body["messages"][0]["content"][1]["text"] == g9.STATE_PROMPT:
            text = (states or {}).get(p, "The cabinet is " + ("closed." if p.startswith("0:") else "open."))
        else:
            text = (lists or {}).get(p, GOOD_LIST)
        return 200, {"role": "assistant", "content": text, "reasoning": reasoning}
    return f


def run(sets: list[tuple[Path, Path]], out: Path, srv: MockServer | None = None, extra: list[str] = ()) -> int:
    argv = [x for d, t in sets for x in ("--photos", str(d), "--truth", str(t))] + ["--out", str(out)]
    if srv is not None:
        argv += ["--base-url", srv.base_url]
    return g9.main(argv + list(extra))


def selection(sets: list[tuple[Path, Path]]) -> dict[str, list[dict[str, Any]]]:
    return {g: g9.select(c) for g, c in g9.candidates(sets, TARGETS).items()}


# ---------- selection ----------
def test_min_px_rule(sets: list[tuple[Path, Path]]) -> None:
    c = g9.candidates(sets, TARGETS)
    tb = {x["photo"]: x["pixels"] for x in c["toolbox"]}
    assert tb["0:closed/storage/3"] == 1000                       # exactly 1000 counts
    assert "0:closed/storage/2" not in tb                         # 999 does not
    assert all(x["pixels"] >= g9.MIN_PX for v in c.values() for x in v)
    assert not [x for x in c["plant"] if x["place"] == "ward"]   # 50 px plant never a candidate


def test_three_distinct_photos_prefer_new_instance(sets: list[tuple[Path, Path]]) -> None:
    s = selection(sets)
    for g in ("toolbox", "chair", "bed", "cabinet:open", "cabinet:closed"):
        assert len(s[g]) == 3 and len({x["photo"] for x in s[g]}) == 3, g
    # chair: biggest first (chair_1, 9000 px), then the other instance beats a bigger chair_1 photo
    assert [x["instance"] for x in s["chair"][:2]] == ["chair_orange_1", "chair_orange_2"]
    # a photo with two beds is one candidate, with the larger instance
    beds = [x for x in g9.candidates(sets, TARGETS)["bed"] if x["photo"] == "0:closed/ward/0"]
    assert len(beds) == 1 and beds[0]["instance"] == "bed_blue_2" and beds[0]["pixels"] == 6500
    # every group: 3 distinct viewpoints (place, k), whatever the world variant (design review 002 B4)
    for g, v in s.items():
        assert len({(x["place"], x["k"]) for x in v}) == len(v), g


def test_same_view_in_two_worlds_is_one_view() -> None:
    """An open- and a closed-world shot of the same (place, k) never both count (design review 002 B4)."""
    def c(s: int, place: str, k: int, px: int) -> dict[str, Any]:
        return {"instance": "toolbox_red_1", "set": s, "place": place, "k": k, "pixels": px, "photo": f"{s}:{place}/{k}"}
    pool = [c(0, "storage_m25", 2, 16076), c(1, "storage_m25", 2, 16062), c(0, "storage_p25", 2, 12174),
            c(1, "storage_p25", 2, 12000), c(0, "storage", 2, 9000)]
    got = g9.select(pool)
    assert [(x["place"], x["k"]) for x in got] == [("storage_m25", 2), ("storage_p25", 2), ("storage", 2)]
    assert len(g9.select(pool[:2])) == 1                          # one view only -> 1 photo, so the gate cannot pass


def test_cabinet_states_split_by_dir(sets: list[tuple[Path, Path]]) -> None:
    s = selection(sets)
    assert {x["photo"].split(":")[0] for x in s["cabinet:closed"]} == {"0"}
    assert {x["photo"].split(":")[0] for x in s["cabinet:open"]} == {"1"}
    assert all(x["state"] == "open" for x in s["cabinet:open"])


def test_select_only(sets: list[tuple[Path, Path]], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(sets, tmp_path / "o", extra=["--select-only"]) == 0
    assert "selection OK" in capsys.readouterr().out
    sheet = Image.open(tmp_path / "o" / "g9_sheet_cabinet_open.png")
    assert sheet.size == (3 * W, H + 40)
    assert not (tmp_path / "o" / "g9_calls.jsonl").exists()
    # only the closed dir: cabinet:open has no photo -> selection short -> exit 1
    assert run(sets[:1], tmp_path / "o2", extra=["--select-only"]) == 1
    assert "SHORT: ['cabinet:open']" in capsys.readouterr().out


def test_truth_class_missing_from_targets_fails_fast(sets: list[tuple[Path, Path]]) -> None:
    t = dict(TARGETS)
    del t["plant"]
    with pytest.raises(AssertionError, match="plant"):
        g9.candidates(sets, t)


def test_paired_args(sets: list[tuple[Path, Path]], tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        g9.main(["--photos", str(sets[0][0]), "--photos", str(sets[1][0]), "--truth", str(sets[0][1]),
                 "--out", str(tmp_path), "--select-only"])


# ---------- matcher ----------
@pytest.mark.parametrize("cls,text,hit", [
    ("toolbox", "A red toolbox.", True), ("toolbox", "TOOLBOX", True), ("toolbox", "a red Tool Box", True),
    ("toolbox", "a tool-box", True), ("toolbox", "two toolboxes", True), ("toolbox", "a red tool chest", True),
    ("toolbox", "a red box", False), ("toolbox", "toolboxing", False),
    ("cabinet", "a blue Cabinet", True), ("cabinet", "filing cabinets", True), ("cabinet", "a cupboard", True),
    ("cabinet", "cabinetry", False),
    ("chair", "orange chairs", True), ("chair", "an armchair", True), ("chair", "a wheelchair", False),
    ("chair", "a chairman", False), ("chair", "seats", False),
    ("bed", "hospital beds", True), ("bed", "a Bed.", True), ("bed", "a cot", True), ("bed", "a bedside table", False),
    ("bed", "embedded", False), ("bed", "a gurney", False),
])
def test_matcher(cls: str, text: str, hit: bool) -> None:
    assert bool(g9.synonym_regex(TARGETS[cls]["synonyms"]).search(text)) is hit


@pytest.mark.parametrize("text,want", [
    ("The cabinet is closed.", "closed"), ("Open.", "open"), ("It is OPEN", "open"), ("CLOSED", "closed"),
    ("Neither: it is ajar.", None), ("", None), ("reopened", None), ("enclosed", None),
    # a recorded deviation: first state word wins (run-1 answers name the other state in the explanation)
    ("It is open, not closed.", "open"), ("It is open or closed.", "open"),
    ("The cabinet is **open**. ... If it were closed, you'd only see a flat front.", "open"),
    ("The cabinet is **closed**. ... no swung‑open doors, no gap.", "closed"),
    # negation flips (design ruling)
    ("The cabinet is not open.", "closed"), ("It is not closed; the doors stand open.", "open"),
    ("The cabinet is not **open**.", "closed"), ("Not  closed.", "open"), ("NOT OPEN", "closed"),
])
def test_parse_state(text: str, want: str | None) -> None:
    assert g9.parse_state(text) == want


# ---------- verdict math ----------
@pytest.mark.parametrize("args,ok", [
    ((True, 3, 2, None), True), ((True, 3, 3, None), True), ((True, 3, 1, None), False), ((True, 2, 2, None), False),
    ((True, 3, 2, 3), True), ((True, 3, 2, 2), False), ((True, 3, 1, 3), False), ((False, 0, 0, None), True),
])
def test_group_pass(args: tuple[bool, int, int, int | None], ok: bool) -> None:
    assert g9.group_pass(*args) is ok


def test_thresholds_pinned() -> None:
    assert (g9.MIN_PX, g9.N_PHOTOS, g9.NAME_MIN, g9.STATE_MIN) == (1000, 3, 2, 3)


# ---------- end to end against the mock ----------
def summary(out: Path) -> dict[str, Any]:
    return json.loads((out / "g9_summary.json").read_text())


def rows(out: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (out / "g9_calls.jsonl").read_text().splitlines()]


def e2e(sets: list[tuple[Path, Path]], out: Path, resp: Any, key: str | None = None,
        extra: list[str] = ()) -> tuple[int, MockServer]:
    srv = MockServer([], responder=resp, api_key=key)
    try:
        return run(sets, out, srv, extra), srv
    finally:
        srv.close()


def test_all_good_passes_and_requests_are_clean(sets: list[tuple[Path, Path]], tmp_path: Path) -> None:
    rc, srv = e2e(sets, tmp_path / "o", responder(), extra=["--api-key", "EMPTY"])
    s = summary(tmp_path / "o")
    assert rc == 0 and s["pass"]
    assert s["groups"]["cabinet:open"]["state_hits"] == 3 and s["groups"]["cabinet:closed"]["state_hits"] == 3
    n_photos = len({x["photo"] for v in selection(sets).values() for x in v})
    assert s["calls"] == len(srv.requests) == n_photos + 6           # one list call per photo + 3 + 3 state calls
    words = {w for t in TARGETS.values() for syn in t["synonyms"] for w in syn.split()} | {"red", "blue", "orange"}
    for b in srv.requests:
        assert b["chat_template_kwargs"] == {"enable_thinking": True} and b["max_tokens"] == 8192
        assert (b["temperature"], b["top_p"], b["top_k"], b["presence_penalty"]) == (1.0, 0.95, 20, 0.0)
        text = b["messages"][0]["content"][1]["text"]
        assert text in (g9.LIST_PROMPT, g9.STATE_PROMPT) and len(b["messages"]) == 1
        if text == g9.LIST_PROMPT:
            assert not set(text.lower().rstrip(".").split()) & words   # no hint of the answer
    r = rows(tmp_path / "o")
    assert {"photo", "class", "pixels", "prompt", "content", "reasoning_chars", "latency_s", "matched"} <= set(r[0])
    assert all(x["reasoning_chars"] == 50 for x in r)


def test_workers_do_not_change_results(sets: list[tuple[Path, Path]], tmp_path: Path) -> None:
    base = responder()

    def slow(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        time.sleep(sum(map(ord, photo_of(body))) % 7 * 0.03)      # per-photo latency, so completion order != call order
        return base(body)

    res = []
    for w in (1, 4):
        rc, _ = e2e(sets, tmp_path / f"w{w}", slow, extra=["--api-key", "EMPTY", "--workers", str(w)])
        r = [{k: v for k, v in x.items() if k != "latency_s"} for x in rows(tmp_path / f"w{w}")]
        s = summary(tmp_path / f"w{w}")
        assert s.pop("workers") == w
        s.pop("latency_s"), s.pop("base_url")                       # port differs per server
        res.append((rc, r, s))
    assert res[0] == res[1] and [x["call"] for x in res[0][1]] == sorted(x["call"] for x in res[0][1])


def test_name_two_of_three_passes_one_of_three_fails(sets: list[tuple[Path, Path]], tmp_path: Path) -> None:
    beds = [x["photo"] for x in selection(sets)["bed"]]
    no_bed = "A red toolbox, a blue cabinet, orange chairs, a green plant."
    rc, _ = e2e(sets, tmp_path / "a", responder(lists={beds[0]: no_bed}))
    assert rc == 0 and summary(tmp_path / "a")["groups"]["bed"]["name_hits"] == 2
    rc, _ = e2e(sets, tmp_path / "b", responder(lists={beds[0]: no_bed, beds[1]: no_bed}))
    s = summary(tmp_path / "b")
    assert rc == 1 and s["groups"]["bed"]["name_hits"] == 1 and not s["groups"]["bed"]["pass"]


def test_informative_class_never_gates(sets: list[tuple[Path, Path]], tmp_path: Path) -> None:
    rc, _ = e2e(sets, tmp_path / "o", responder(lists={p: GOOD_LIST.replace(", a green plant", "")
                                                       for p in COLOURS.values()}))
    s = summary(tmp_path / "o")
    assert rc == 0 and s["groups"]["plant"]["name_hits"] == 0 and s["groups"]["plant"]["pass"]


def test_word_only_in_reasoning_does_not_count(sets: list[tuple[Path, Path]], tmp_path: Path) -> None:
    lists = {p: "Some furniture." for p in COLOURS.values()}
    rc, _ = e2e(sets, tmp_path / "o", responder(lists=lists, reasoning=GOOD_LIST))
    assert rc == 1 and summary(tmp_path / "o")["groups"]["toolbox"]["name_hits"] == 0


@pytest.mark.parametrize("bad", ["The cabinet is open.", "It is open or closed.", "I cannot tell."])
def test_cabinet_state_must_be_three_of_three(sets: list[tuple[Path, Path]], tmp_path: Path, bad: str) -> None:
    p = selection(sets)["cabinet:closed"][1]["photo"]
    rc, _ = e2e(sets, tmp_path / "o", responder(states={p: bad}))
    g = summary(tmp_path / "o")["groups"]
    assert rc == 1 and g["cabinet:closed"]["state_hits"] == 2 and g["cabinet:open"]["pass"]
    st = [r for r in rows(tmp_path / "o") if r["kind"] == "state" and r["photo"] == p][0]
    assert st["expected"] == "closed" and st["parsed"] == g9.parse_state(bad) and not st["matched"]


def test_state_only_asks_just_the_state_items(sets: list[tuple[Path, Path]], tmp_path: Path) -> None:
    """G9 confirmation run (a recorded deviation): only the 6 cabinet-state calls; verdict = 3/3 per state."""
    rc, _ = e2e(sets, tmp_path / "ok", responder(), extra=["--state-only"])
    s, r = summary(tmp_path / "ok"), rows(tmp_path / "ok")
    assert rc == 0 and s["state_only"] and s["calls"] == 6 and set(s["groups"]) == {"cabinet:open", "cabinet:closed"}
    assert all(x["kind"] == "state" and x["prompt"] == g9.STATE_PROMPT for x in r) and len(r) == 6
    assert all(g["name_hits"] is None and g["state_hits"] == 3 for g in s["groups"].values())
    p = selection(sets)["cabinet:open"][0]["photo"]
    rc, _ = e2e(sets, tmp_path / "bad", responder(states={p: "The cabinet is not open."}), extra=["--state-only"])
    g = summary(tmp_path / "bad")["groups"]
    assert rc == 1 and g["cabinet:open"]["state_hits"] == 2 and not g["cabinet:open"]["pass"]


def test_missing_open_dir_fails(sets: list[tuple[Path, Path]], tmp_path: Path) -> None:
    rc, _ = e2e(sets[:1], tmp_path / "o", responder())
    g = summary(tmp_path / "o")["groups"]["cabinet:open"]
    assert rc == 1 and g["n_photos"] == 0 and not g["pass"]


def test_api_key(sets: list[tuple[Path, Path]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(g9, "LINK_ENV", tmp_path / "none.env")
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    rc, srv = e2e(sets, tmp_path / "ok", responder(), key="k1", extra=["--api-key", "k1"])
    assert rc == 0 and {k.lower(): v for k, v in srv.headers[0].items()}["authorization"] == "Bearer k1"
    monkeypatch.setenv("VLM_API_KEY", "k1")                       # env default
    assert e2e(sets, tmp_path / "env", responder(), key="k1")[0] == 0
    monkeypatch.delenv("VLM_API_KEY")
    for name, extra in [("wrong", ["--api-key", "k2"]), ("missing", [])]:
        rc, srv = e2e(sets, tmp_path / name, responder(), key="k1", extra=extra)
        r = rows(tmp_path / name)
        assert rc == 1 and not srv.requests and r and all(x["status"] == 401 and not x["matched"] for x in r), name


def test_api_key_from_link_env(tmp_path: Path) -> None:
    f = tmp_path / "vlm_link.env"
    f.write_text("BASE_URL=http://127.0.0.1:8000/v1\nJOBID=1\nFWD_PID=2\nAPI_KEY=abc\nKEYTMP=\n")
    assert g9.link_env(f) == {"BASE_URL": "http://127.0.0.1:8000/v1", "JOBID": "1", "FWD_PID": "2",
                              "API_KEY": "abc", "KEYTMP": ""}
    assert g9.link_env(tmp_path / "absent.env") == {}
