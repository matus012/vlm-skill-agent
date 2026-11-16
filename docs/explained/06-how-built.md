# How it was built

> **In plain words.** The project was built in twelve numbered phases, 0 to 11, and each phase added exactly one
> new thing. A phase had a pass mark that was written down before it ran, and the next phase did not start until
> the mark was met. The point is blame: when something fails, the one new thing is the first suspect.

## 6.1 The phases

The table gives, for each phase, what was new, the gate that had to be passed and what was measured. Gates were set
in the project brief before any run, and none was changed to make a result pass (section 7). The whole build took
about five calendar days, from 2026-11-03 to 2026-11-08.

| Phase | What was new | Gate (written first) | Result |
|---|---|---|---|
| 0 | Nothing connected: the simulator with the 90° camera, the VLM on a GPU, and the link between them, each alone | all three work; the VLM returns a valid tool call | passed: valid tool call 3/3, 10/10 through the link, robot stopped 0.105 m from a goal† |
| 1 | The clinic world, its map, the named places, the true answers | robot reaches every place 10/10 times; four photos show the whole room | passed: 40/40† reaches, every object at least 4,463† pixels, nine checks |
| 2 | Adapter, three skills, scorer, run log. A hand-written script plays the VLM | a correct script passes all four kinds; a wrong answer fails | passed: 12/12 scripts, 7/7 planted mistakes caught |
| 3 | Context builder, agent runtime, the VLM chooses skills (Find only) | at least 7 of 10 runs | passed: 10/10† Find runs, 105/105† model replies were valid tool calls |
| (sim) | Simulators moved onto the cluster (not a numbered phase) | same checks pass there; speed bar 0.7 of real time | passed: 40/40 and 12/12 twice; 0.709 and 0.740 |
| 4 | Check state, count and conditional tasks | at least 7 of 10 per kind | passed: 10/10† per kind, 40† runs |
| 5 | Seeded layouts, so the answer moves between runs | at least 7 of 10 per kind | passed: 10/10 per kind, 40 runs |
| 6 | A blocked doorway in some runs | at least 7 of 10, no crash | passed: 10/10, 0 crashes, 8/10 hit `failed: blocked` |
| 7 | Nothing, unless the numbers ask for an `inspect` skill | proof photo passes in at least 90 % of the phase 5 and 6 runs | 60/60 so no `inspect`; the skill interface frozen |
| 8 | Test prompts and seeds on the frozen system: 25 runs per kind, 100 in all | at least 18/25 per kind | round 1 81/100, check state 14/25 missed; one fix; round 2 91/100 passed |
| 9 | Unitree G1 in MuJoCo with the pretrained walking policy | walks to a waypoint in a room and gives camera images, or stop after 3 sessions | go: 5/5† waypoints, 4/4† headings, 1† session |
| 10 | G1 adapter with the same API | at least 18/25 per kind, same tasks; 0 lines changed in the agent | passed in G1 round 1: 93/100, 0 lines |
| 11 | Nothing technical: video, clean repository, write-up, slides | every number matches the audited reports, by script | done; this document extends it |

Phase 9 ran in parallel with phases 1 to 3, as a separate lane, because the humanoid was the biggest unknown and shares
nothing with the first stage until phase 10. The order in the table is the order of the numbers, not of the calendar.

## 6.2 Why one new thing per phase

This is the central discipline. Suppose phase 3 had connected a new world, a new robot, a planner, the context builder
and the model at once, and half the runs failed. Any of five things could be responsible. Because the phases were
cut so that only one thing changes, a failure points at it.

Three examples show how this paid off:

- **Phase 2 existed to clear the body of blame.** With a hand-written script playing the VLM, a correct script had to
  pass all four kinds and every planted mistake had to fail on the check named for it. Only then could a later wrong
  answer be put down to the VLM. In the process the ground-truth checks exposed two hidden faults (a planner tolerance
  that reported a goal inside a wall as reached, and stale position updates in the hall), which section 10.1 describes.
- **Phase 4 had a second new thing, and it was handled openly.** In phase 3 the VLM forgot which rooms it had already
  searched, because the history line said what was done and not where. Phase 4 added the room name to each history
  line (a recorded deviation) and used the phase 3 Find tasks as a regression guard, which passed 10/10†.
- **The prompt did not move.** The system prompt was written once, in phase 3, and stayed byte-identical through
  phase 4, phase 5, phase 6 and round 1 of phase 8. When round 1 failed on check state, the only change
  allowed was one sentence, and it was recorded as a new version (section 8.2).

## 6.3 Lanes, and when rules were allowed to bend

At the start the phases were a strict chain. Two changes, both logged, made it faster without losing attribution:

- **Two lanes before phase 3** (a recorded deviation). The simulation lane (phases 0a, 1, 2) and the model lane (phases 0b, 0c)
  share no component, so they could run in parallel, and phase 3 started only when both passed.
- **The humanoid spike was moved early** (a recorded deviation), as a third lane, for the reason above. Phase 10 still waited
  for phases 8 and 9.

Code for a later phase could be written early, for example the runtime and context builder against a fake robot, but
it counted towards no gate until its own phase ran.

## 6.4 Kill rules

The brief and the project notes contained rules for stopping, fixed before any number existed:

- any phase gate: three diagnosed fix rounds that still fail stop the project at that phase;
- the Stage 1 test: at most three rounds; a round below 70 % on any kind allows one diagnosed fix, then a fresh sealed set;
  a third failing round stops Stage 1 and reports as it stands;
- the humanoid spike: not working after three sessions means no-go and Stage 1 ships alone;
- the humanoid test: the same rule as Stage 1 with two rounds.

Three of these rules came close to firing: round 1 of the Stage 1 test missed on one kind and spent its one
allowed fix; a re-count in phase 1 used one of three fix rounds (section 10.1); and the cluster speed bar passed only
on its third and last round (section 10.2). None fired as a stop.
