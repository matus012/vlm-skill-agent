import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import context  # noqa: E402
from agent.context import Packet  # noqa: E402
from agent import skills  # noqa: E402
from agent.fake_adapter import PLACES, FakeAdapter  # noqa: E402
from agent.skills import SkillContext  # noqa: E402

def packet() -> Packet:
    ad = FakeAdapter("ward")
    photos = skills.execute("look_around", {}, SkillContext(ad, PLACES, "find", ["ward"])).photos
    return Packet(photos, ad.where(), ["ward", "storage"], ["1. navigate_to(ward) from hall -> reached"],
                  "reached", "Find the red toolbox.", 2)


def test_packet_has_only_allowed_fields() -> None:
    assert set(Packet.__dataclass_fields__) == {"photos", "room", "places", "history", "last_result", "task", "step"}


def test_messages_shape_and_content() -> None:
    msgs = context.build_messages(packet())
    assert [m["role"] for m in msgs] == ["system", "user"]
    parts = msgs[1]["content"]
    assert sum(p["type"] == "image_url" for p in parts) == 4
    assert all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in parts if p["type"] == "image_url")
    text = "\n".join(p["text"] for p in parts if p["type"] == "text")
    for needle in ["Step 2 of 12", "Current room: ward", "ward, storage", "1. navigate_to(ward) from hall -> reached",
                   "Last result: reached", "Find the red toolbox.", "Photo 1:", "Photo 4:"]:
        assert needle in text


def test_messages_ignore_photo_frames() -> None:
    """Packet.photos[].frame carries seg + loc_pose for the run log; the messages must not depend on it (lead N6)."""
    import dataclasses
    p = packet()
    bare = dataclasses.replace(p, photos=[dataclasses.replace(ph, frame=None) for ph in p.photos])
    assert all(ph.frame is not None and ph.frame.seg is not None for ph in p.photos)
    assert context.build_messages(p) == context.build_messages(bare)


def test_no_coordinates_or_robot_identity() -> None:
    msgs = context.build_messages(packet())
    text = msgs[0]["content"] + "\n".join(p["text"] for p in msgs[1]["content"] if p["type"] == "text")
    assert not re.search(r"\d+\.\d+", text)  # no decimals
    assert not re.search(r"[(\[]\s*-?\d+\s*,\s*-?\d+", text)  # no (x, y) pairs
    assert not re.search(r"turtle|create3|unitree|g1\b|humanoid|lidar|occupancy|odom|yaw|joint|nav2|ros", text, re.I)
    # every digit in the text is a step/photo number or inside the known step counter
    assert set(re.findall(r"\d+", text)) <= {"1", "2", "3", "4", "12"}


def test_render_packet_example(tmp_path: Path) -> None:
    from agent import render_packet
    out = tmp_path / "p.html"
    render_packet.example(out)
    s = out.read_text()
    assert s.count("<img src=\"data:image/jpeg;base64,") == 4 and "Current room: storage" in s
    assert "2. look_around() in storage -&gt; 4 photos" in s and "&quot;report&quot;" in s


# ---------- system prompt (task 005) ----------
SITE = ["hall", "ward", "office", "reception", "storage"]


def test_system_prompt_is_the_system_message_and_lists_the_packet_places() -> None:
    p = packet()
    assert context.build_messages(p)[0]["content"] == context.system_prompt(p.places)
    sp = context.system_prompt(SITE)
    assert "hall, ward, office, reception, storage" in sp
    assert "four rooms" not in sp and not re.search(r"\b(two|three|four|five|six|\d+)\s+rooms\b", sp, re.I)
    assert "pharmacy" in context.system_prompt(SITE + ["pharmacy"])  # nothing about the site is hard-coded


def test_system_prompt_covers_the_rules() -> None:
    sp = context.system_prompt(SITE)
    for needle in ["navigate_to(place)", "look_around()", "report(answer, photo, note)", "exactly one tool",
                   "at most 12 steps", "Report only what you saw", "numbered from 1", "room name", "yes or no",
                   "whole number in digits", "two wrong calls in a row"]:
        assert needle in sp, needle


def test_system_prompt_is_body_agnostic() -> None:
    sp = context.system_prompt(SITE)
    assert not re.search(r"turtle|create3|oakd|unitree|\bg1\b|humanoid|lidar|laser|wheel|leg|odom|yaw|joint|nav2|"
                         r"\bros\b|map|coordinat|metre|meter|clinic", sp, re.I)
    assert set(re.findall(r"\d+", sp)) <= {"1", "12"}


def test_system_prompt_sha256_is_stable() -> None:
    import hashlib
    h = context.system_prompt_sha256(SITE)
    assert h == hashlib.sha256(context.system_prompt(SITE).encode()).hexdigest() == context.system_prompt_sha256(SITE)
    assert h != context.system_prompt_sha256(SITE[::-1])


# ---------- a recorded deviation: history line names the room ----------
V1_SHA = "531d7c884f90c566775cbe6ea14e2ae9eba113bff870c2d6c46a2b03b055677c"  # Phases 3-6 + 011 pass 1 (frozen-v1)
DEV20 = (" Things of a similar colour can look alike: answer only once a photo clearly shows the very object the task "
         "names; if you are not sure, look in the places you have not seen yet.")


def test_system_prompt_sha256_unchanged_by_deviation_12() -> None:
    # v1 (unchanged by a recorded deviation) = today's prompt without a recorded deviation's sentence
    import hashlib
    sp = context.system_prompt(list(PLACES))
    assert sp.count(DEV20) == 1 and sp.endswith("Look before you decide." + DEV20)
    assert hashlib.sha256(sp.replace(DEV20, "").encode()).hexdigest() == V1_SHA


def test_system_prompt_sha256_deviation_20() -> None:
    # frozen-v2: the only change to the system prompt since 2907c0b (007) is a recorded deviation's one sentence
    assert context.system_prompt_sha256(list(PLACES)) == "f8374e72f1b2fa018936f47d4c93533404b64a79f415cf5b5e5805e4f7e16a7a"


def test_history_line_rule() -> None:
    hl = context.history_line
    assert hl(2, "navigate_to(office)", "reached", "hall") == "2. navigate_to(office) from hall -> reached"
    assert hl(3, "look_around()", "4 photos", "storage") == "3. look_around() in storage -> 4 photos"
    assert hl(5, "report(...)", "reported", "storage") == "5. report(...) in storage -> reported"
    assert hl(1, "explore()", "error: unknown skill 'explore'", "hall") == "1. explore() in hall -> error: unknown skill 'explore'"
