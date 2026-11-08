"""Context builder. build_messages puts only what the VLM may see into the messages: photo RGB, room name, place names,
history, last result, task, step counter. Packet.photos[].frame also carries the seg image and loc_pose for the run log;
build_messages never reads them (tests/test_agent_context.py), so lidar/map/coordinates/robot identity never reach the VLM.
The system prompt is system_prompt(places): the place list comes from places.yaml via Packet.places."""
import base64
import hashlib
import io
from dataclasses import dataclass

from PIL import Image

from agent.robot import Photo

MAX_STEPS = 12


def system_prompt(places: list[str]) -> str:
    """The system prompt (task 005: written once, short, body-agnostic). `places` = the places.yaml names, in order;
    nothing about the world is hard-coded here."""
    return "\n".join([
        "You control a robot in a building and see only through its camera. Do the task in the user message.",
        f"Places you can go to, each a room of the building: {', '.join(places)}.",
        "Each step, call exactly one tool:",
        "- navigate_to(place): go to a place; returns one photo from where you end up.",
        "- look_around(): turn on the spot; returns four photos in different directions.",
        "- report(answer, photo, note): give the final answer; this ends the task.",
        "Photos are numbered from 1 in each message. They come from your last navigate_to or look_around (at the "
        "start: one photo from where you are); a wrong call does not change them. Older photos are not shown again.",
        "Answer format: a find or where task -> one room name from the list above; a yes/no question -> yes or no; "
        "a how-many question -> a whole number in digits.",
        "Report only what you saw in a photo. The photo number in report must be a photo in the current message "
        "that shows the evidence.",
        f"You have at most {MAX_STEPS} steps; running out is a failure. A wrong call (no tool, several tools, an "
        "unknown place, a wrong answer format) comes back as an error and still costs a step; two wrong calls in a "
        "row end the task as a failure.",
        "Look before you decide. Things of a similar colour can look alike: answer only once a photo clearly shows the "
        "very object the task names; if you are not sure, look in the places you have not seen yet.",  # a recorded deviation
    ])


def system_prompt_sha256(places: list[str]) -> str:
    """sha256 of system_prompt(places) (UTF-8); run.json and the gate report record it."""
    return hashlib.sha256(system_prompt(places).encode()).hexdigest()


@dataclass(frozen=True)
class Packet:
    photos: list[Photo]  # 1..4, numbered
    room: str
    places: list[str]
    history: list[str]  # one line per past step
    last_result: str
    task: str
    step: int  # 1-based, the step about to be taken


def history_line(step: int, call: str, result: str, room: str) -> str:
    """Deviation 12: `room` = where() at the start of the step. navigate_to says "from <room>", every other call
    (incl. wrong ones) "in <room>"."""
    where = f"from {room}" if call.startswith("navigate_to(") else f"in {room}"
    return f"{step}. {call} {where} -> {result}"


def data_url(img: Image.Image, max_side: int = 640) -> str:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_messages(p: Packet) -> list[dict]:
    text = "\n".join([
        f"Task: {p.task}",
        f"Step {p.step} of {MAX_STEPS}.",
        f"Current room: {p.room}",
        f"Places you can navigate to: {', '.join(p.places)}",
        "History:" + ("" if p.history else " (none)"), *p.history,
        f"Last result: {p.last_result}",
        f"The photos below are numbered {p.photos[0].number} to {p.photos[-1].number}.",
    ])
    content: list[dict] = [{"type": "text", "text": text}]
    for ph in p.photos:
        content.append({"type": "text", "text": f"Photo {ph.number}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url(ph.image)}})
    return [{"role": "system", "content": system_prompt(p.places)}, {"role": "user", "content": content}]
