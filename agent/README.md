# agent/

Planner + skills + runtime + run log. Build contract: `INTERFACES.md` (Phase 2, task 003). No ROS imports here,
ever: `tests/test_agent_no_ros.py` scans every `agent/*.py` by AST (incl. `__import__` / `import_module` by literal
name). Robot-specific code lives in `adapters/`.

## Modules
| file | what |
|---|---|
| `robot.py` | `Adapter` ABC (go_to, turn_to, photo, pose, where, stop, sim_time), `Frame` (cropped rgb + seg + meta), `GoResult`, `Photo` (packet number + image + source frame) |
| `skills.py` | tool schemas (one source: sent to the planner and used to validate), `parse_call`, `SkillContext`, `execute`, `load_places` |
| `planners.py` | `Planner` protocol, `VlmPlanner` (OpenAI-compatible; sampling / thinking / max_tokens per model card), `ScriptedPlanner` |
| `runtime.py` | `run(task, ctx, planner, run_dir, meta, truth=None) -> RunResult` |
| `runlog.py` | run directory writer (`RunLog`), `check_complete(run_dir)` (gate G4), `git_state(repo)` |
| `context.py` | `Packet` (only what the VLM may see) -> messages (images as JPEG data URLs) |
| `render_packet.py` | one packet -> self-contained HTML, for inspection (`python -m agent.render_packet OUT.html`) |
| `fake_adapter.py` | pure fake `Adapter` for tests: rectangle rooms, places, objects with seg labels, blocked points, injected failures |
| `world.py`, `tasks.py`, `scorer.py` | room polygons, dev tasks + truth, scorer (other 003 workers) |

## The loop (`runtime.run`)
1. Start photo (step 0). Packet = numbered photos, `adapter.where()`, place names, history lines, last result, task, step.
2. `planner.next_call(messages, tools)`; the runtime never knows which planner it has.
3. `skills.parse_call`: exactly one call, known skill, JSON, schema (place enum, answer format per kind), photo exists.
4. Invalid / planner error -> `check = "error: ..."`, becomes the next last result. **Costs a step.** Never reaches the adapter.
5. Valid -> `skills.execute` on the adapter. `navigate_to` -> `go_to(x, y, yaw, 120)` + 1 photo (also after a block);
   `look_around` -> 4 × (`turn_to(start + k·90°)`, `photo()`), photos 1..4; `report` -> `stop()`, end.
6. Optional `truth(place|None)` hook from the harness fills `truth {gt_pose_after, place_err_m}` after every step.
7. Ends: valid `report` -> `reported`; step 12 done -> `fail_step_limit`; any exception (skill, adapter, planner,
   truth hook) -> `crash`, traceback in the step line and `run.json.crash`, never raised.

## Decisions (where the contract is silent)
- **look_around start heading** (INTERFACES §2, design ruling): place yaw if `adapter.pose()` is within 0.30 m of a place
  point (nearest place), else localization yaw. Read once, before the first turn. Targets wrapped to (-π, π].
- **Photo files** `photos/<step>_<k>_{rgb,seg}.png`, k = photo number − 1; step 0 = the start photo. Packet photos
  point at the files of the step that produced them (no copies). Paths are relative to the run dir.
- `check_complete`: every key of INTERFACES §5 present and non-null except the marked nullable ones (`report`,
  `call.name/args`, `result`, `truth.place_err_m`, all of `planner_out`); every referenced PNG exists; `run.steps`
  equals the number of lines. A crash step (has `traceback`) may carry nulls. A run without a truth hook is
  incomplete (`gt_pose_after` null) by design: gate runs must log truth.
- Extra keys beyond the contract: step `planner_wall_s`, `skill_detail` (nav status, look_around start yaw and turn
  errors), crash `traceback`; run `crash`.
- `sim_t0/sim_t1`, `wall_t0/wall_t1` bound the whole step (planner + skill); `skill_*_s` the skill alone.
  `wall_t*` are Unix epoch seconds.
- `ScriptedPlanner` past its end returns "script exhausted" (an error step), not a crash.
- History line: `<step>. <skill>(<place>) -> <result or error>`; an invalid call shows as received, e.g.
  `navigate_to({"place": "kitchen"})`.
- `report` answer schema per kind: room enum (find, conditional), `yes`/`no` (check_state), digits string (count).

## Tests
`source sim/env.sh; $BUILD_DIR/.venv/bin/python -m pytest -q`. Agent tests: `tests/test_agent_*.py`.
VlmPlanner tests run over `tests/mock_openai.py` / `tests/mock_vlm.py`.
