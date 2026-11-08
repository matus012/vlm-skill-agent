# agent/INTERFACES.md — Phase 2 contracts (task 003)

Frozen at the start of 003 so the adapter, the agent core, the scorer and the dev tasks can be built in parallel.
Change only by editing this file first, in its own commit. Nothing here is the frozen *skill interface* (that freezes
at the end of Phase 7); this is the build contract for Phase 2.

## 0. Packages and the ROS rule
- `agent/` — no ROS imports, ever (`tests/test_agent_context.py` + a new import test cover every module):
  `robot.py` (Adapter ABC, Photo, results), `skills.py`, `planners.py`, `runtime.py`, `context.py`, `world.py`,
  `tasks.py`, `scorer.py`, `runlog.py`, `fake_adapter.py` (tests).
- `adapters/` — the only Stage 1 code that imports ROS (besides `ros_ws/` and the Phase 0/1 `probes/`):
  `adapters/turtlebot4.py` (TB4Adapter + SimTruth), `adapters/run_task.py` (CLI: one task, one planner, one run dir).
- Robot config: `config/robots/turtlebot4.yaml` (camera, `self_crop`, frames, topics, timeouts) — one file per robot.
- Gate runner: `probes/phase2_gates.py`. Replay: `probes/replay.py <run_dir>`.
- Dev tasks: `tasks/dev.yaml`; scripts for the ScriptedPlanner: `tasks/scripts/*.yaml`.
- No answers on the agent side (005 Step 0, N8): room geometry comes only from
  `sim/worlds/clinic/generated/map_knowledge.json` (sim/INTERFACES.md §2: rooms + doorways, no objects, no states).
  `truth.json` is read only by `agent/truth.py` (truth answers, proof labels, task validation), `agent/scorer.py`,
  the harness `adapters/run_task.py` (setup + score), instruments (`probes/`, `tasks/derive_photos.py`,
  `humanoid/probes/`) and tests. Nothing the runtime imports reads it (`tests/test_truth_readers.py`).

## 1. Adapter (`agent/robot.py`, implemented by `adapters/turtlebot4.py`)
All poses are map frame (= Gazebo world frame, G8), metres / radians. The VLM never sees any of them.
```python
@dataclass(frozen=True)
class Frame:            # one camera capture, already cropped by the robot's self_crop
    rgb: PIL.Image.Image            # cropped RGB, what the VLM sees
    seg: numpy.ndarray | None       # cropped label image (uint8/uint16 label id per pixel), same crop; scorer only
    meta: dict                      # sim_time_s, heading_target_rad|None, loc_pose [x,y,yaw], crop {top_px,bottom_px}

class Adapter(ABC):
    def go_to(self, x: float, y: float, yaw: float | None, timeout_s: float) -> GoResult   # Nav2 goal, waits
    def turn_to(self, yaw: float) -> float          # absolute heading (localization), returns final error (rad)
    def photo(self) -> Frame                       # frame stamped after the robot settled
    def pose(self) -> tuple[float, float, float]   # localization (AMCL) x, y, yaw
    def where(self) -> str                         # room name of pose() via agent.world.room_of (map_knowledge.json polygons)
    def stop(self) -> None                         # zero velocity, cancel any goal
    def sim_time(self) -> float

@dataclass(frozen=True)
class GoResult:
    status: str        # "reached" | "blocked"   (Nav2 ABORTED/CANCELED/REJECTED or timeout -> "blocked")
    nav_status: str    # raw Nav2 status name or "TIMEOUT"
    sim_s: float; wall_s: float
```
- `go_to` timeout is **sim** seconds (120 s, the G1 limit) with a wall hang guard, as `map_world_check.navigate`.
- `SimTruth` (adapters only, evaluator side, never reachable from `agent/`): `gt_pose() -> (x, y, yaw)` from Gazebo.
  The harness logs it after every skill; the planner never gets it.

## 2. Skills (`agent/skills.py`, no ROS) — what the planner can call
| call | does | result text | photos returned |
|---|---|---|---|
| `navigate_to(place)` | `places.yaml` x, y, yaw → `adapter.go_to(x, y, yaw, 120)` | `reached` / `failed: blocked` (see below) | 1 (`photo()` after arrival), number 1 |
| `look_around()` | 4 × (`turn_to(start + k·90°)`, `photo()`), k = 0..3 | `4 photos` | 4, numbered k + 1 = 1..4 |
| `report(answer, photo, note)` | `stop()`, end the run | `reported` | — |
- **`reached` means at the place (003, adapter smoke):** `reached` only if `go_to` returned `reached` **and** the
  localization pose is within 0.30 m (the place bound) of the place point; otherwise `failed: blocked`. Nav2's
  planner `tolerance: 0.5` planned to the nearest free cell, so a goal inside a wall came back SUCCEEDED with the robot
  0.43 m away. Since a recorded deviation the planner tolerance is 0.0 (sim/INTERFACES.md §5; Phase 1 G1 re-counted), so such a
  goal ABORTs; the 0.30 m check stays as a second guard (design review: both stay).
- **Photo numbering:** 1..n per packet (brief: "1–4 numbered photos"); heading index k = 0..3 maps to photo k + 1.
- **look_around start heading (design ruling 2026-11-04):** if the localization pose is within 0.30 m (the place bound) of
  a place point, that place's yaw; otherwise the current localization yaw.
  This reproduces Phase 1's proven photo set (photos at place yaw + k·90°, G2/G3/G7); Nav2 alone ends within its
  0.25 rad yaw tolerance, too loose for the G3 seams.
- **report format check:** answer schema per task type (room name enum / `yes`|`no` / digits). A wrong format is an
  `error: …` back to the caller (costs a step, not scored as an answer).
- Unknown skill or place, bad JSON, several calls → `error: …` back to the caller, logged, run continues.

## 3. Planner (`agent/planners.py`)
```python
class Planner(Protocol):
    def next_call(self, messages: list[dict], tools: list[dict]) -> dict
        # {"error": str|None, "tool_calls": [OpenAI-style tool call dicts], "content": str|None,
        #  "reasoning": str|None, "finish_reason": str|None, "usage": dict|None}
```
- `VlmPlanner(cfg)` = today's `runtime.call_vlm` (Phase 3). `ScriptedPlanner(steps)` returns the scripted tool calls in
  order, one per step, ignoring the packet (it may carry raw invalid calls for negative controls). Runtime never
  knows which one it has.
- Every planner has `name` (run.json `planner`) and `info: dict` (run.json `planner_info`, 005): VlmPlanner
  `{model, thinking, sampling, max_tokens, seed}`, ScriptedPlanner `{script}`. `VlmConfig.seed` (int | None, per run)
  is sent as the request's `seed` (vLLM honours it); sampling stays the model card's thinking-mode set (`SAMPLING[True]`).

## 4. Runtime (`agent/runtime.py`)
`run(task: Task, skills_ctx: SkillContext, planner: Planner, run_dir: Path, meta: dict, truth=None,
malformed_limit: int | None = None) -> RunResult`.
Loop as today (12-step cap, error = a step, history lines), via `planner.next_call`. Any exception inside a skill or the
adapter ends the run with `outcome = "crash"` (logged with traceback), never propagates.
`outcome ∈ {"reported", "fail_step_limit", "fail_malformed", "crash"}`.
- **Packet history line (a recorded deviation):** `history_line(step, call, result, room)` = `N. <call> from <room> -> <result>` for
  `navigate_to(...)`, `N. <call> in <room> -> <result>` for every other call (wrong calls and errors keep their wording
  plus the room). `room` = `packet.room` of that step, i.e. `adapter.where()` at step start; room names only, no new
  information source. E.g. `2. navigate_to(office) from hall -> reached`, `3. look_around() in storage -> 4 photos`.
- **Malformed outputs (005):** a step whose check is a `parse_call` error (no call / several calls / bad JSON / unknown
  skill or place / wrong answer format / photo out of range) is a malformed output: it costs a step and its error text
  goes back as the step's result (as before). `malformed_limit` of them in a row end the run with
  `fail_malformed` (after the step is logged). A planner-call failure (HTTP / network, `planner_out.error`) is an error
  step but not a malformed output; it resets the count. The harness passes `malformed_limit = 2`
  (`runtime.MALFORMED_LIMIT`, task 005 design) for the VLM planner and `None` (no limit) for scripted runs: the Phase 2
  G2 controls `unknown-call` / `raw-invalid` make two errors in a row on purpose and must still reach their report.
- run.json gets from the runtime (005): `system_prompt` (the text sent), `system_prompt_sha256`, `tools` (the schemas
  sent), `planner_info` (§3), `malformed_limit`.
A crash records its origin in `run.json` `crash_origin` (005 Step 0, N1), from the traceback frame right below
`runtime.run` (what the loop called) and, under a skill, whether any deeper frame is a method of the adapter:
`planner` (raised in planner.next_call's model-call path: no frame below next_call is agent/ code),
`agent` (005 N2: our own agent/ code on the planner side — context building in agent/context.py, or anything raised
under an agent/ planner's next_call, e.g. VlmPlanner's response handling after the HTTP call; VlmPlanner's HTTP call
itself never raises, it returns `error`), `skill` (agent/skills.py code itself), `adapter` (any adapter method,
called directly by the loop or from a skill: ROS / sim / Nav2 client failures, the WALL_HANG stall),
`truth_hook` (the harness's evaluator hook), `runtime` (the loop or the run log itself), `layout_teleport` (008: a seeded
layout's set_pose / read-back did not land, before setup; INTERFACES section 10), `gz_read` (006 a recorded deviation:
any exception with `gz_read_failed` = probes/gz_cli.GzReadError, a `gz topic` CLI read killed by a signal on all of
its 1 + 2 attempts, wherever it was called; checked before the frame rule).

## 5. Run directory (`agent/runlog.py`) — the replayable record
```
<run_dir>/run.json        metadata + summary (written at start, rewritten at the end)
<run_dir>/steps.jsonl     one line per step
<run_dir>/photos/<step>_<k>_rgb.png, <step>_<k>_seg.png   (seg = cropped label image, lossless)
<run_dir>/score.json      written by the scorer
```
`run.json`: `task {id, kind, prompt, target}`, `planner` ("scripted:<script id>" | "vlm:<model>"), `robot`,
`layout_sha256`, `world_tag`, `git_commit`, `git_dirty`, `start_place`, `start_utc`, `end_utc`, `outcome`, `steps`,
`report {answer, photo, note}|null`, `sim_s`, `wall_s`, `setup {start_place, reached, gt_err_m}`, `hand_help` (false
unless a human touched the run; set by the harness, `adapters/hand_help.py`, with `hand_help_why: [str]`: a GT
position jump the robot cannot drive between two consecutive samples, or any `/initialpose` message during the run;
the setup move before step 1 is excepted), `crash_origin` (crash only, §4), `invocation_id` (005 B1: only in runs
written by `probes/phase3_run.py`, the uuid of its `invocation.json`; `run_one(..., extra_meta=None)` adds nothing, so
Phase 2 run.json is unchanged).
`steps.jsonl` line: `step`, `packet {room, history, last_result, photos:[{n, rgb, seg}]}`,
`call {raw: <tool_calls as received>, name|null, args|null}`, `check` ("ok" | the error text), `result` (skill
result text or null), `photos_out:[{n, rgb, seg, meta}]`, `sim_t0`, `sim_t1`, `wall_t0`, `wall_t1`,
`skill_sim_s`, `skill_wall_s`, `truth {gt_pose_after, place_err_m (navigate_to only)}` (evaluator fields),
`planner_out {content, reasoning, finish_reason, usage, error}`.
Complete = every key above present (null allowed only where marked); every referenced PNG exists.
Added in 005 (the packet actually sent; not part of `check_complete`, so Phase 2 run dirs stay complete; checked by
`probes/phase3_gates.py` G2): `packet.system_sha256` (sha256 of the system message text), `packet.texts` (every text
part of the user message, in order), `packet.photos[].rgb_sha256` (sha256 of the rgb PNG file) and `sent_sha256`
(sha256 of the image data URL string sent), `planner_wall_s` (wall time of `next_call`). No base64 in the log.
Evaluator side (harness, never read by `agent/`): `gt_trace.jsonl` (adapters/gt_trace.py, GT pose at 2 Hz for the demo
recorder) and run.json `gt_trace {samples, errors, last_error}`; neither is required for completeness.

## 6. Tasks (`tasks/dev.yaml`, `agent/tasks.py`)
```yaml
- {id: find-toolbox,  kind: find,        prompt: "Find the red toolbox.",                 target: {object: toolbox_red_1}}
- {id: state-cabinet, kind: check_state, prompt: "Is the storage cabinet open?",          target: {object: cabinet_blue_1, state: open}}
- {id: count-chairs,  kind: count,       prompt: "How many chairs are in reception?",     target: {class: chair, room: reception}}
- {id: cond-toolbox,  kind: conditional, prompt: "If the toolbox isn't in storage, check the office.",
   target: {object: toolbox_red_1, if_not_in: storage, else_room: office}}
```
`agent/tasks.py` loads prompts only (`Task {id, kind, prompt, target}`, no answers). Truth by code from
`truth.json` in `agent/truth.py` (never typed): find → the object's room; check_state → `yes` iff the object's state
equals the asked state; count → number of objects of the class in the room; conditional → the object's room, which
must be `if_not_in` or `else_room` (else the task is invalid and loading fails). Proof objects: find / check_state /
conditional → the target object's label; count → any object of the class in the room. ≥ 2 prompts per type, targets
only from G9-passing classes (toolbox, chair, bed, cabinet + state).
Class targets (find / conditional): `target: {class: bed}` (optional `colour`) instead of `object`. The object's room is
then the one room holding **all** instances of the class (and colour); instances split across rooms → loading fails.
Proof = any instance of the class with ≥ 1000 px (as count). Find / conditional answers must span more than one room,
so a constant-answer planner cannot pass them (lead, 2026-11-04).

## 7. Scorer (`agent/scorer.py`) — pass only if all four hold (brief)
1. `answer` equals the truth answer. 2. The reported photo's cropped seg has ≥ 1000 px of a proof label.
3. steps ≤ 12 (outcome `reported`). 4. `outcome != "crash"` and no hand help (`run.json` `hand_help` false/absent).
`score.json`: `{pass, checks: {answer, proof, steps, clean}, truth_answer, proof_px, label, label_why}`.
Label (failed runs only, exactly one): `vlm` (planner chose wrong: wrong answer, no proof, step limit, bad format),
`skill` (a skill returned wrong data), `nav` (navigate_to blocked on a reachable place), `sim` (crash in sim / adapter),
`scorer` (scorer / evaluator bug). Auto: crash → by `crash_origin` (§4): `adapter` → `sim`, `skill` → `skill`,
`runtime` → `skill` (agent-side code, not the VLM), `agent` → `skill` (005 N2), `planner` → `vlm`,
`truth_hook` → `scorer`, `gz_read` → `sim` (006 a recorded deviation), `layout_teleport` → `sim` (008: a seeded layout did not land), absent (pre-005 runs) → `sim`; blocked on a reachable place → `nav`; else → `vlm` (005:
outcome `fail_malformed` → `vlm`, why names the two steps and the last error). A crash is never `nav`: Nav2 failures
are results (`blocked`), not exceptions.
Precedence for a failed run (005 design review N6, B3), first match wins:
1. setup missed its start place (`run.json setup.reached` false) → `nav`, why `setup did not reach <place> (GT x.xx m)`;
2. crash → by `crash_origin` as above;
3. VLM endpoint error (B3): a `vlm:` planner run with any step whose `planner_out.error` is not null (VlmPlanner sets
   `error` only when the HTTP call itself fails: connection, timeout, HTTP status) → `sim`, why
   `VLM endpoint error at step N: <first 80 chars>` (an instrument failure, not a VLM choice);
4. blocked on a reachable place → `nav`; 5. else `vlm`.
No-call steps (B2): a step with `planner_out.error` null and no tool call. Every failed run's why ends with
`; no-call steps: step N finish_reason F, completion_tokens T, tool call inside reasoning yes|no, …` when it has
any (`tool call inside reasoning` = `<tool_call>`, `<function=` or `"name": "<skill>"` in the reasoning or content).
These rules change only `label` / `label_why`, never `pass` or `checks` (regression: `probes/rescore_compare.py`).

## 8. Harness and gates
- `world_tag` in run.json (N5): the newest `phase1-world-frozen*` git tag whose committed
  `generated/clinic.sdf` and `generated/truth.json` hash equal to the files on disk (`adapters/run_task.world_tag`);
  no such tag → the harness refuses to run. (The freeze manifests live in evidence kept outside the repo, not in the repo; the
  tag's own tree is the in-repo record of the same hashes.)
- Every run starts at its start place, default the hall (setup move, not a step, logged in `run.json.setup`; end
  distance logged, 002 finding 14). `run_one(..., start_place=<any places.yaml place>)` (005): the same rule,
  reached = Nav2 reached and GT within 0.30 m of that place. `adapters/run_task.py --start-place P --planner vlm
  --seed N`.
- Phase 3: `probes/phase3_run.py --list tasks/phase3_counted.yaml --out DIR` (one sim session, VLM planner, seed and
  start per entry), then `probes/phase3_gates.py {g1,g2,g3,measure,all} --out DIR` (no sim).
  Counted-run rules (005 design review B1, B3; task file "Counted-run rules"):
  - The counted list is `tasks/phase3_counted.yaml`, recognised by path or by its sha256 (`phase3_run.COUNTED_SHA256`,
    the committed blob). For it the runner refuses `--rehearsal`, `--only` and an `--out` that exists.
  - Every invocation writes `OUT/invocation.json` first: `invocation_id` (uuid4), `argv`, `list`, `list_sha256`,
    `git_commit`, `segments: [{kind: start|resume, start_utc, end_utc, entries_run, end}]`. Every run.json of the
    invocation carries `invocation_id`.
  - Counted invocations append one JSON line per event (`start`, `resume`, `end`, `abort` with a reason) to a ledger
    outside the repo, `<ws>/.cache/vlm_skill_agent/runs/phase3_counted_ledger.jsonl` (`--ledger` overrides).
  - `--resume OUT` (same list, same commit, same invocation id): runs only the entries with no run dir in OUT. A run
    dir that exists, finished or partial, is never re-run or deleted; a partial one is scored as it stands.
  - Before each entry: `GET {VLM_BASE_URL}/models` with the key, 10 s timeout. Down → ledger `abort` ("endpoint
    down"), the invocation stops, that entry is never started (then `--resume`).
  - Before each entry (and in `probes/phase2_gates.py` before each g1 / g2 / g3 run), without `--rehearsal`: the tree
    must still be clean and on the invocation's start commit (2026-11-04). Else the invocation stops before that
    run, prints why, exits non-zero; phase3_run writes ledger `abort` ("tree dirty on commit …" / "tree on commit …,
    not the start commit …").
  - `--min-job-left-min N` (default 0 = no check): before the first entry, the vLLM job's remaining wall
    (`squeue -h -j $JOBID -o %L` over ssh, JOBID from `$BUILD_DIR/vlm_link.env`) must be ≥ N min.
  - G1 (counted) also requires: `OUT/invocation.json` with `list_sha256 == COUNTED_SHA256`; all 10 run.json carry its
    `invocation_id`; the ledger holds exactly one counted invocation, this one, as `start`, then only `resume` /
    `abort`, then one `end`.
- `probes/phase2_gates.py {g1,g2,g3,g4,g5,all} --out DIR` with the sim running (one Gazebo, gz lock, INTERFACES §6).
- Per-run and per-skill sim / wall time are summarised by the gate runner (no bar).

## World variants (007, hpc/PERUN_SIM.md section 5)
`agent/truth.py` lists the world dirs (`closed` → `sim/worlds/clinic/generated/`, `cabinet_open` →
`generated_cabinet_open/`, generated from `layout_cabinet_open.yaml`, tag `phase4-world-cabinet-open-v1`).
`adapters/run_task.run_one` takes the world from env `SIM_WORLD` (default closed), uses that world's truth.json and
map_knowledge.json, and writes `world` to run.json. `agent/scorer.score_run(run_dir)` with no `truth_path` scores against
the run.json `world` (missing = closed) and still requires the run's layout sha256 to equal that truth's. Run lists may
carry `world: cabinet_open`; true answers always come from that world's truth.

## 10. Phase 5 seeded layouts (task 008; frozen before 008's build)
- **Movables:** `toolbox_red_1` (any of the 4 rooms) and `chair_orange_1..4` (any rooms, 0–4 per room, the total
  fixed at 4: the frozen worlds' SDFs stay unchanged). Beds, cabinet and all other furniture stay where layout.yaml
  puts them. The cabinet state stays per sim world (`closed` / `cabinet_open`, section "World variants").
- **Placement = f(layout seed):** `sim/worlds/clinic/seeded.py` (harness / truth side: never loaded by the agent side or the robot adapters; `agent/truth.py` imports it lazily for the scorer, `tests/test_truth_readers.py` enforces it):
  - `placements(seed) -> {object_id: (x, y, yaw)}` for every movable, deterministic.
  - Rules: per-room zones fully visible from the room's place, away from the 90° photo seams, not visible through any
    doorway from another place; ≥ 0.5 m floor gaps between countable objects; inside the room, clear of fixed
    furniture and the robot's place footprint. A seed breaking a rule raises `SeedRejected`.
  - `truth_for(seed, world) -> dict`: the world's truth.json with the movables' poses and rooms replaced and
    `layout_sha256` = sha256 of (world layout sha, seed).
  - `count_photos(seed, room, cls)`: which of the 4 look_around photos hold each instance.
- **Vetting (off-sim):** `probes/vet_seeds.py` uses the ray-caster (`probes/raycast_seg.py` with per-model pose
  overrides) to check, per seed and world, G2 coverage, G3 counted-once and G4 no-leakage at every place. It writes
  `tasks/phase5_seeds.json`. A seed that fails is never listed.
- **Run time:** a list entry may carry `layout_seed` (int). `adapters/run_task.py`:
  - before the run, teleports every movable with gz `set_pose` (`/world/clinic/set_pose`), then reads the GT poses
    back (gz_cli; up to 3 read-backs 1 s apart, for a pose message still in flight) and requires ≤ 2 cm / 1° each,
    else the run fails labelled `sim` (crash_origin `layout_teleport`, no setup move, no VLM step);
  - uses `truth_for(seed, world)` for everything truth-related;
  - records `layout_seed` and the truth's `layout_sha256` in run.json.
  - The scorer and gates score with `truth_for(run.json layout_seed, world)`; no `layout_seed` = the frozen world, as
    before.
- **Map:** Phase 5+ navigate on `sim/maps/clinic_static.{yaml,pgm}`: static structure + fixed furniture only, no
  movables. Nav2's obstacle layer sees the movables live. The frozen `clinic.yaml` stays for re-runs of Phases 1–4.
  `hpc/sim_job.sbatch` arg `map=<name>` selects it. G0 = Phase 1 G1 40/40 + G8 on it, before any VLM run.

## 11. Phase 6 blocked doorways (task 009; frozen before 009's build)
- **Block = world variant**, one per blocked room: `blocked_<room>` for room in {storage, reception, office, ward}, cabinet
  closed, generated by `generate.py sim/worlds/clinic/layout.yaml --blocked <room> --out
  sim/worlds/clinic/generated_blocked_<room>/` (the `blocker_<room>` box of `blocked.py`, label 220, class `blocker`,
  not in truth `objects`; truth.json top-level `"blocker": {id, room, pose, size, label}`; map_knowledge.json identical to
  closed). Tag `phase6-world-blocked-v1` (all four dirs). The map stays `clinic_static` (the block is not in it; Nav2's
  obstacle layer sees it live). A sim's world is set per sim (`sim_job.sbatch w<i>=blocked_<room>`), as in 007/008.
- **Truth side:** `agent/truth.WORLD_DIRS`, `adapters/run_task.WORLD_TAG_KW` and `seeded.WORLDS` know the four variants.
  `seeded.truth_for(seed, blocked_<room>)` = closed's truth_for with the movables placed, plus that world's `"blocker"`
  entry; its `layout_sha256` = sha256 of (that world's layout sha, seed). `run_truth(world, seed)` carries `"blocker"`.
- **Scorer:** a run's reachable places = places.yaml minus the place of `truth["blocker"]["room"]` (if any).
  `navigate_to(<blocked room>) -> failed: blocked` is then a result, not a `nav` failure label. Nothing else changes.
- **No change** to prompts, packet, skills, runtime, Nav2 params or the map (task 009 non-goals).
- **Lists:** an entry's `world` carries the block (`blocked_<room>`); the answer is never in the blocked room; no start in
  the blocked room. Blocked set = entries whose world starts with `blocked_`; guard = `closed` / `cabinet_open`.
  `tasks/phase6_counted.yaml` (10 blocked + 10 guard, one invocation, pinned in `COUNTED_LISTS` before any Phase 6 VLM
  output), `tasks/phase6_dev.yaml`, new prompts in `tasks/phase6_tasks.yaml` (resolved after dev.yaml and
  phase5_tasks.yaml).
- **Preconditions (before any VLM run):** P1 `navigate_to(<blocked room>)` -> `failed: blocked` 10/10 per blocked
  layout from cycling reachable starts, each ≤ 120 s sim, then `navigate_to(hall)` reached; P2 10 navigations per
  blocked layout to the other places pass the Phase 1 G1 rule (SUCCEEDED, GT ≤ 0.30 m, ≤ 120 s sim, valid start);
  P3 ray-caster G2–G4 at every reachable place for every counted seed in its (blocked) world.

## 12. Phase 8 layout specs (task 011, a recorded deviation; frozen before 011's build)
- **Spec** = one layout of the sealed generator's output (`gen_phase8.render_pass`): `seed` (31-bit), `objects`
  {`toolbox_red_1`: {room}, `cabinet_blue_1`: {room, state}, kept `bed_blue_i` / `chair_orange_i`: {room} (each row in one
  room)}, `removed` [ids]. The generator is used UNCHANGED. Test entries are never read by a person; everything below is
  built and unit-tested on synthetic specs only.
- **Realizer** `sim/worlds/clinic/realize.py` (truth side, like seeded.py): `realize(spec) -> {"world", "poses", "parked"}`,
  deterministic (RNG from sha256("realize:" + seed)):
  - world = `cabinet_open` if the cabinet's state is open, else `closed`;
  - toolbox -> its spec room using 008's zones and rules (≥ 1.9 m, clear of furniture), else parked (the ward has no zone);
  - kept chairs -> all n in the spec chair room using 008's zones and rules (gaps, seams, doorway views, reachability);
    if the full row does not fit, the whole kept row is parked;
  - beds and cabinet are fixed: they stay at their frozen poses if the spec room is the frozen one (beds: ward, cabinet:
    storage), else they are parked; removed beds / chairs are parked;
  - parked = teleported outside the building (z from the frozen truth), and absent from the truth's objects.
  - `truth_for_spec(spec)` = the world's truth.json with poses replaced and parked objects removed;
    `layout_sha256` = sha256(world sha + ":spec:" + canonical JSON of the spec).
- **Validity filter (a recorded deviation), per entry:**
  - (a) the target's (class, spec room) is G9-passing (toolbox: reception, storage; cabinet: storage; bed: ward; chair:
    any room) AND the realizer placed it exactly as the spec says (not parked, the whole row);
  - (b) the realized layout passes the ray-caster G2–G4 checks at every place;
  - (c) its layout seed or planner seed collides with no dev / counted seed of Phases 3–6;
  - (d) the start rule holds. Dropped entries are replaced by the next entries of the stream.
- **Stream:** pass n = sealed set n's layouts in order, then fresh 31-bit seeds drawn by a blind helper (`secrets`, sha256
  committed before any realization), in order, through the unchanged generator (ids `p8-<n>-<i>`), until each kind has 25
  valid entries.
- **Starts and planner seeds (design ruling):** per kind, in entry order, cycle hall / other room / answer room; at most 2
  answer-room starts per kind (later answer-room slots become an other room); never a non-existent place. Planner seed
  = sha256(entry id) mod 2^31. No blocked doorways in Phase 8.
- **Run time:** a list entry may carry `layout_spec` (instead of `layout_seed`). `adapters/run_task.py` teleports the
  placed movables and every parked object (beds / cabinet included) with the 008 read-back rule. Truth, scorer and gates
  use `truth_for_spec`; run.json records `layout_spec` and its sha.
- **Lists:** `tasks/phase8_pass<n>_tasks.yaml` (the generated tasks of the valid entries) + `tasks/phase8_pass<n>.yaml`
  (100 entries, 25 per kind), pinned in `COUNTED_LISTS` before any run.
