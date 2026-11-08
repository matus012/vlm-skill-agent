# Skill interface freeze — tag `skill-interface-frozen-v2` (task 011, a recorded deviation; v1 = task 010, Phase 7)

Frozen for Phases 8–10: the VLM sees exactly this interface. A change to any item below is a deviation: design ruling,
`design notes` row, regression run, then `probes/interface_freeze.py --write` and a new tag (`-v3`). Machine copy:
`tests/interface_freeze.json`; `tests/test_interface_freeze.py` fails on any difference (mutation-checked by
`tests/mutants_freeze.py`: schema, prompt, packet field, 3/3 killed).

**v2 (a recorded deviation, 011 pass 1 → pass 2):** the only change to v1 is one sentence appended to the system prompt's last
line: "Things of a similar colour can look alike: answer only once a photo clearly shows the very object the task
names; if you are not sure, look in the places you have not seen yet." Why: in 011 pass 1, check_state failed 7 of 7
times when the VLM did not reach the cabinet's room. Each time it took another blue/teal object for the cabinet (proof
px 0); all 14 runs that reached the cabinet passed. New prompt sha256 `f8374e72…`; the packet-format hash changes only
through the system message (with the sentence removed, both hashes equal v1's: `tests/test_agent_context.py`). Tools,
Packet fields, runtime rules, sampling and the adapter API are unchanged. v1 (`531d7c88…`) = every counted run of
Phases 3–6 and 011 pass 1.

**Decision behind the freeze (brief, Phase 7):** the proof photo passed the scorer's proof check in **60 of 60** counted
runs of Phases 5 + 6 (008: 40, 009: 20; denominator = runs whose proof check was reached, i.e. a report naming a photo;
re-scored from the raw run dirs, 0 differences to the saved scores). 100 % ≥ 90 % → **no `inspect` skill**. Proof pixels
min 6574 (bar 1000), median 11866 (`probes/proof_stats.py --rescore`; task 010 report).

**Provenance:** the frozen code files (`agent/context.py`, `skills.py`, `robot.py`, `planners.py`, `runtime.py`) last
changed in 2907c0b (007 a recorded deviation, history lines name the room) and are identical at the counted commits of 007
(549c11a), 008 (62a4775) and 009 (806f671).

## Tools (`agent/skills.build_tools`, one list per task kind)
| tool | arguments | returns |
|---|---|---|
| `navigate_to(place)` | `place`: enum of places.yaml places (hall, ward, office, reception, storage) | `reached` + 1 photo, or `failed: blocked` + 1 photo |
| `look_around()` | none | 4 numbered photos (90° apart) |
| `report(answer, photo, note)` | `answer` by kind: find / conditional = room enum (map_knowledge rooms), check_state = `yes`/`no`, count = `^(0\|[1-9][0-9]*)$`; `photo` ≥ 1 (a number in the current message); `note` free text, never scored | ends the run |

All schemas `additionalProperties: false`, all arguments required. Full JSON: `tests/interface_freeze.json` →
`tool_schemas`.

## System prompt
`agent/context.system_prompt(places)`, sha256 **`f8374e72f1b2fa018936f47d4c93533404b64a79f415cf5b5e5805e4f7e16a7a`**
(v2) for the places.yaml places; v1 `531d7c884f90c566775cbe6ea14e2ae9eba113bff870c2d6c46a2b03b055677c` (Phases 3–6, 011
pass 1). Every run records its sha in run.json.

## Packet (`agent/context.Packet`, `build_messages`)
- Fields: `photos` (1..4, numbered), `room`, `places`, `history` (one line per past step), `last_result`, `task`,
  `step`.
- User message text, in order: `Task: …`, `Step N of 12.`, `Current room: …`, `Places you can navigate to: …`,
  `History:` + lines (`N. <call> from|in <room> -> <result>`), `Last result: …`,
  `The photos below are numbered a to b.`, then `Photo k:` + image per photo (longest side ≤ 640 px).
- Format hash of a fixed example (images masked): `packet_text_sha256` in the fixture.
- Never in a packet: lidar, maps, coordinates, joint angles, robot type (`tests/test_agent_no_ros.py`, G3 rule scans).

## Runtime rules (`agent/runtime.py`, `agent/skills.py`)
- At most **12 steps** (`MAX_STEPS`); running out = `fail_step_limit`.
- **2 malformed planner outputs in a row** end a VLM run (`MALFORMED_LIMIT`).
- Errors are results, not crashes: a wrong call (no tool, several tools, unknown skill / place, bad arguments) becomes
  an error step the VLM sees as its last result.
- `navigate_to`: Nav2 goal with a **120 s sim** timeout (`NAV_TIMEOUT_S`); `reached` only with SUCCEEDED and the
  localization pose within **0.30 m** (`PLACE_BOUND_M`) of the place point, else `failed: blocked`.
- Task kinds: find, check_state, count, conditional.

## VLM call (`agent/planners.py`)
- Thinking on: temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0.0, **max_tokens 8192**.
- Thinking off: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, max_tokens 1024.
- Model and endpoint come from the environment (`VLM_MODEL` / `VLM_BASE_URL`), as recorded in run.json.

## Robot adapter API (`agent/robot.py`, the only thing a robot backend implements)
- `Adapter`, abstract:
  - `go_to(x, y, yaw | None, timeout_s) -> GoResult`
  - `turn_to(yaw) -> float`
  - `photo() -> Frame`
  - `pose() -> (x, y, yaw)`
  - `where() -> str`
  - `stop() -> None`
  - `sim_time() -> float`
- Data types:
  - `GoResult(status, nav_status, sim_s, wall_s)`
  - `Frame(rgb, seg | None, meta)`
  - `Photo(number, image, frame | None)`
- The TurtleBot 4 adapter is `adapters/turtlebot4.py`; Phase 9's humanoid adapter implements the same API.
