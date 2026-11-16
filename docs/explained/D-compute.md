# Compute used

> **In plain words.** The whole project used 49.8 of the 120 GPU-hours it was allowed. Almost all of it went to running the
> VLM and the simulators on the cluster. The Stage 1 test cost the most, 14.90 hours, because it ran twice; the humanoid test passed on its
> round 1 and cost about 7.5.

## D.1 Budget

The brief estimated 90 to 115 H200 GPU-hours for the whole project and proposed a cap of 120. That cap was approved, together
with 50 CPU-hours for CPU-only jobs, as a soft limit; an overrun of up to 50 %† was allowed if needed, with the reason in the report.
The cap was never approached.

A cluster job is billed for its wall-clock time on the GPU even when the GPU is idle, so the model server ran only when a
session needed it. Phases 1 and 2 used no GPU time at all.

## D.2 H200-hours by task

| Report | What | H200-hours used | Allowed for the task |
|---|---|---|---|
| 001 | Phase 0, each piece alone | 0.09† | 2 |
| 002 | Phase 1, the clinic world (readability check) | 0.11† | 0.3 |
| 003 | Phase 2, executor without the VLM | 0 | (none planned) |
| 004 | Phase 9, humanoid spike | 0 | (none planned) |
| 005 | Phase 3, the VLM on Find tasks | 0.63† + 1.55† (two jobs) | not recorded |
| 006 | Simulators on the cluster | 8.14† | 9 |
| 007 | Phase 4, all task types | 4.35† | 12 |
| 008 | Phase 5, moving objects | 7.85 | 15 |
| 009 | Phase 6, blocked doorways | 4.57 | 10 |
| 010 | Phase 7, the `inspect` decision and the freeze | 0 | (none planned) |
| 011 | Phase 8, the Stage 1 test (two rounds) | 14.90 | 20 |
| 012 | Phase 10, the humanoid test | about 7.5 | 25 |
| 013 | Phase 11, the showcase (one video re-run) | 0.064† | 2 |
| 014 | Cleaning the repository | 0 | (none planned) |
| | **Total** | **about 49.8 of 120†** | |

Running totals reported along the way: 10.52† after the cluster move, 14.87† after phase 4, 22.72 after phase 5, 27.29 after phase 6,
42.19 after the Stage 1 test and about 49.7 after the humanoid test.

![Running total of H200 GPU-hours at the end of each stage, against the approved cap.](fig:compute)

## D.3 Cost per run

The cost of a counted run is the whole job's time divided by its runs, including start-up of the model server:

| Phase | H200-hours per counted run | Runs per hour |
|---|---|---|
| 3 (one job, ten runs) | 0.155† | not measured |
| 4 | 0.049† | about 42† |
| 5 | 0.055† | not recorded |
| 8, round 1 | 0.059 | 47 |
| 8, round 2 | 0.067 | 39 |

The model's own time per step is a small part: the lower bound from the planner's wall time alone was 0.004† H200-hours per run. The rest
is the simulator and its set-up.

## D.4 CPU-hours

CPU-only jobs: none were used for the simulation. Image builds and preparation ran on a login node or in jobs recorded as 0 CPU-hours of
the allowed 50. The project lead ruled on 2026-11-04 that the CPUs bundled with a GPU job are not counted against the 50 CPU-hours;
they are logged as requested, bundled. For example, the first GPU job of phase 3 requested 16 CPUs, which by the ledger's count alone put it
over the task's 5 CPU-hour allowance. The final project report gives about 3† CPU-hours used of the 50.

## D.5 Laptop

The laptop hosted the early build (phases 0 to 3) and the chats. The largest session was the phase 1 counted run, 72† minutes, within about
8.6† laptop hours. From phase 4 the laptop ran no counted experiments; it rendered videos and hosted the chats.
