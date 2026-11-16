# Stage 2 results: the Unitree G1 humanoid

> **In plain words.** The same agent, with the same prompt and the same code, was put in a walking humanoid in a
> different simulator and given the same 100 tasks. It got 93 right, two more than the wheeled robot. A comparison of the
> agent's files with the frozen version shows 0 changed lines. The robot never fell and never got stuck. One planned
> check failed beforehand, and it was run anyway, with that decision on the record.

## 9.1 What had to be built, and what did not

Everything below the adapter line was new: a MuJoCo version of the clinic, a node that runs the walking policy and
publishes camera images and poses, a driver that walks to a point and turns to a heading, the G1 adapter, a contact
logger, and cluster images and job scripts for all of it. Everything above the adapter line had to stay as it was.

The test of that is a swap check: a `git diff` of the agent, the world generators, the interface files, the freeze test
and the serving files against the version 2 tag must print nothing. It printed nothing: **0 lines**. The test is run
over a fixed list of paths, written before the work began.

## 9.2 The walking policy

The G1 walks with NVIDIA's AGILE velocity policy (`Velocity-G1-v0`), a public pretrained network in ONNX format under the
Apache-2.0 licence. It takes a velocity command, forward speed, sideways speed and turn rate, and produces joint targets. It
is run in a plain Python MuJoCo loop and wrapped in a ROS node that listens on `/cmd_vel` and publishes the camera images,
a segmentation image and the exact pose.

The route chosen in the spike was not the one the brief first named. The brief expected NVIDIA's Isaac ROS container path,
but that uses ROS 2 Lyrical, which is only a lower-support platform on Ubuntu 24.04, needs a large amount of disk and has
unclear messaging with Jazzy, the ROS version the rest of the project uses. The plain MuJoCo runner avoided that. This
was a logged deviation (a recorded deviation).

The spike gate was four checks, all passed on a counted run: it walks to 5 of 5† waypoints, 2.5 to 3.6 m away, within
0.30 m; it turns to 4 of 4† headings within 3 degrees (the worst was 0.69°†); the camera is 640 by 480 with a focal length
of 320 pixels; and the published pose equals the pelvis pose computed from the joints, and moves in the commanded direction.

## 9.3 The preconditions

Before any test task ran on the G1, four preconditions were checked on the G1 itself. Their job is to avoid blaming the VLM
for a problem of the body.

| Check | Bar | Result |
|---|---|---|
| H-P1 reach | 10 walks per room, within 0.30 m, within 120 s, no fall, no collision, on the closed world and 3 development layouts | passed: 160/160 walks, 0 contacts of any kind, 0 falls, worst error 0.132 m |
| H-P2 photos | the coverage, counted-once, no-leakage and clean-frame checks at the G1 head camera | passed at all five places; the control (camera pitched 47.6° down) fails on 20 of 20 frames, as it must |
| H-P3 readable | the VLM must name each target at least 2 of 3 times | **failed** for the toolbox in reception only: 2 of 5 combined |
| H-P4 executor | scripted run of the four kinds | passed: 12/12 |

A "collision" here is a physical contact, logged on every physics step, between any part of the robot and anything that is not the
floor. An earlier stand-in, a 0.35 m clearance, was replaced by this counter after it was found to be a proxy, and a deliberate
walk into a chair is the sanity control: it registered 66 contact steps.

## 9.4 The failed precondition, and why the test ran anyway

H-P3 failed. From the G1's head height of about 1.2 m, the VLM reads the red toolbox in reception as a "red box" or a
"trunk", not a toolbox. The geometry showed no cropping, so a camera tilt was not a diagnosed fix.

The question was what to do. The precondition was a prediction: some test tasks about the toolbox in reception might be lost. The
alternatives were to drop those entries, which would give a better looking score on an easier list, or to run the identical
100-task list and count any such loss as a real property of the humanoid. The project lead decided on 2026-11-06 to run
it and count it. This is recorded as a recorded deviation and marked as a change that the lead authorised, with no bar moved.

What happened: all 3 of 3 tasks about the toolbox in reception passed in the counted run. A likely reason is that a task
lets the agent choose where to look, so it has more than one view, while the naming check looks at a single fixed photo. With
n = 3 this is a lean, not a result, and the limit stays on record.

## 9.5 The result

| Kind | TurtleBot 4 | G1 | Wilson 95 % CI for the G1 | Bar |
|---|---|---|---|---|
| Find | 24/25 | 24/25 | 0.80 to 0.99 | 18 |
| Check state | 20/25 | 20/25 | 0.61 to 0.91 | 18 |
| Count | 24/25 | 24/25 | 0.80 to 0.99 | 18 |
| Conditional | 23/25 | 25/25 | 0.87 to 1.00 | 18 |
| All | 91/100 | 93/100 | | |

The G1 was in the final test once, in its round 1. A G1 round 2 was not needed because no kind was below 18. The run
list is the same file as TurtleBot round 2 (same hash, 100 entries, none dropped), so this is a like-for-like
comparison with one body changed.

- **Falls: 0. Stalls: 0.** A fall (the body drops below 0.5 m) would be a crash and a failed run; a stall would be a failed run
  too, and neither can be removed from the count. All 7 misses are labelled `vlm`, so no run was lost to a collision either.
- **The swap diff is 0 lines.** The freeze test passed, and the commit of all counted runs holds the same interface as the tag.
- **No leakage.** The scan of all 420 G1 packets found 0 hits, including "g1", "unitree" and "humanoid".
- **No hand help** in any run.

## 9.6 Same score, same mistakes, different runs

| Kind | Both passed | TurtleBot only | G1 only | Neither | Agreement |
|---|---|---|---|---|---|
| Find | 23 | 1 | 1 | 0 | 92 % agree |
| Check state | 17 | 3 | 3 | 2 | 76 % agree |
| Count | 23 | 1 | 1 | 0 | 92 % agree |
| Conditional | 23 | 0 | 2 | 0 | 92 % agree |
| All | 86 | 5 | 7 | 2 | 88 % agree |

The two bodies agree on 88 % of the tasks: both right or both wrong. They miss largely different entries: only 2
check-state misses are shared. That fits a sampling-level error, where different photos lead to different reasoning, and does
not fit a fixed property of those layouts.

The 7 G1 misses are the same kind as the TurtleBot's. Five are the look-alike cabinet, with 0 cabinet pixels in the proof photo.
One is a miscount, with the chairs in view. One is a find. None was a walking failure.

The amber-chair mistakes did not occur on the G1. The 3 TurtleBot misses where chairs were refused as "orange" all passed on the
G1, and the naming checks on G1 photos always said "orange". The head camera looks down on the seats from 1.2 m, so they fill less of
the frame under the lamp. That fits, but it was not tested, and with 3 runs it is a lean.

## 9.7 What the G1 needed below the agent

Rehearsals and development runs found several bugs, none in the agent. All were fixed in the adapter or the instrument:

- the neural-network runtime pinned one thread per machine core and starved the simulator; it was limited to one thread and re-pinned;
- an abort when the program exited, from a thread that was not joined;
- a stale pose after a layout reset: the robot routed from where the previous run had ended; the adapter now waits for a fresh pose and
  scan (the one real bug the rehearsals found, caught on a development run);
- the movable objects were not restored between runs of the same world;
- a race at the first move, before any contact message had arrived.

The adapter also gained obstacle sensing: a simulated planar range scan about 0.4 m above the floor that makes the robot stop and walk around
furniture. It is built from the simulator's rays and never from the truth file or from object positions. A test checks that the G1 adapter
does not read the truth file.

Turn drift is a G1-specific problem. In the first clinic smoke test, arrival in the hall was 0.236† m off from turn drift. After each
`look_around` turn the adapter recentres on its start of the set, at most 0.08 m net over four turns. The worst measured repeat-set drift
is 0.087 m, and it was never exercised in a counted run: 0 of 100 runs on either robot did two `look_around` calls in a row.

## 9.8 The side-by-side demonstration

The project's video includes the two bodies running the same task. By rule it is the first test entry that passed on both bodies: p8p2c001,
"How many chairs are there in the office?", from the hall. Both went to the office, looked around and answered 4, the true count. The
TurtleBot took 98 s of simulated time and the G1 took 90 s. The G1's counted run logged only its base pose, so the video uses a labelled
re-run with full joint logging. That re-run failed: it answered 3, truth 4. The caption shows it as it is, beside the counted result, which passed.
