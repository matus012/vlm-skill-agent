# What went wrong, and what was learned

> **In plain words.** Most of the work was not the VLM. It was the simulator, the robot's idea of where it is, the
> speed of the experiments, and the world itself. Each problem below is written the same way: what we saw, what
> it turned out to be, what we changed, and where the evidence is. The gates were designed to catch exactly this kind of
> thing, and several of them did.

## 10.1 Localisation: the robot thought it had arrived

**Symptom.** After the planner tolerance was set to zero (so that goals inside walls fail, section 4.2), the re-count of the
phase 1 reach test came in at 39† of 40. In the failing try, the setup move to the hall ended 0.362† m from the hall point, so the start was
invalid.

**Diagnosis.** The ground-truth check had asked for the robot's real position, not Nav2's word. The robot's position estimate (AMCL) is
updated only after the robot has moved a set distance or turned a set angle: by default 0.25 m† or 0.2 rad†. The last creep and final
turn of a move fall under those limits, so Nav2 judged arrival on a stale estimate. In the long hall the error ran up to 0.21† m
along the hall's axis.

**Fix.** Update after 0.05 m† or 0.1 rad†. The acceptance bar (arrivals under 0.20 m by ground truth) was set before testing.

**Evidence.** 20† of 20† hall arrivals within the bar, worst 0.151† m; then the phase 1 reach test re-counted at 40/40†. Report 003
(a recorded deviation). An earlier, related fix is in report 002: Nav2's stock goal tolerance of 0.25 m left too little room under the 0.30 m
place bound, because the robot arrives from one side and stops about 0.23 m short, so the tolerance was cut to 0.10 m.

**Lesson.** A navigation success flag is not a measurement. The gates compared against the simulator's ground truth, and found a fault that
the system itself did not report.

## 10.2 Simulation speed: one laptop was too slow

**Symptom.** The laptop ran 14 to 17† experiments an hour, at 0.82 to 0.86† of real time per simulator. The test lists needed hundreds of
runs and a model call per step.

**Diagnosis.** On the cluster a single simulator ran slower, about 0.78† of real time, because the cluster's CPUs are capped at 2.4 GHz†,
about half the laptop's boost clock, and Gazebo mostly runs one busy thread. Packing more simulators into one job made each slower:
about 0.55† of real time with 4, and 0.33† with 8, even with cores of their own. The shared GPU's graphics is the likely cause. It was not proved.

**Fix.** Give each simulator its own cores, move the agent, the gate scripts and the model server onto other cores, and run two simulators
per job. Scale by running two or three jobs at the same time. That gives about 50† runs an hour, 3 times† the laptop.

**Evidence.** Report 006. The speed bar of 0.7 of real time per simulator was met only with two per job: 0.709 and 0.740, on the third and
last allowed fix round. With the model sharing the GPU the figure is 0.663. In the counted runs it was 0.6 to 0.7† in phase 4 and lower in
later phases, which only stretched wall time; the gates count simulated time. The ruling that one failed job (a pinning bug) was not counted was
made after its result and is flagged as such in the deviation log (a recorded deviation).

**Lesson.** The largest cost was never the model. Measuring the batch before it burns, and writing the bar before the data, kept the move
to the cluster honest.

## 10.3 The toolbox was readable in some rooms and not in others

**Symptom.** The brief's example task is "Find the red toolbox". In the first naming check (G9, phase 5) the VLM called the toolbox a
"briefcase" or "suitcase" in the ward (0 of 3), "red cabinet on the table" or "red sofa" in the office (1 of 3), but a toolbox in reception (2 of 3).

**Diagnosis.** The label followed the room's context. An isolated red box is a "toolbox" in storage or reception, but a "red box" or "suitcase" in
the ward and office, even standing alone and 1.9 m or more from the camera. Moving it clear of furniture did not help in the office (second check:
office 0 of 3).

**Fix.** The toolbox is only an *answer* in reception or storage. In the ward the find-tasks ask about the beds. The toolbox still stands in the
office as a background object in some layouts (14 of the 40† counted layouts of phase 5), where the task asks about something else. The
counted list was re-picked once, mechanically, after the check.

**Evidence.** Report 008 (a recorded deviation, logged before any counted output).

**Lesson.** The pre-registered rule "targets only from placements that were readable" failed, and the response was to change the placement, not the
rule. A VLM's name for an object depends on what surrounds it.

## 10.4 The cabinet and the sofa

**Symptom.** In round 1 of the TurtleBot test, check state scored 14 of 25.

**Diagnosis.** In 7 of 7 VLM failures, the robot never saw the real cabinet in storage and answered about a teal or blue look-alike, mostly the
office sofa. All 14 runs that reached the cabinet passed. The first sign was a single dress-rehearsal miss in phase 3, which read the teal sofa as a
"low blue cabinet".

**Fix.** One generic prompt sentence (section 8.3). It improved check state to 20 of 25 on a new set but did not end the error: 5 of the 9
remaining TurtleBot misses and 5 of the 7 G1 misses are the same mistake.

**Evidence.** Report 011 and report 012.

**Lesson.** A small fix helps in proportion to how much of the failure is "not looking" and not "looking and seeing wrong". Here the sentence made
the VLM look in more places, and the remaining misses are where it still answers from a look-alike.

## 10.5 The simulator's set-pose calls timed out

**Symptom.** In round 1 of the TurtleBot test, 10 runs crashed in the layout teleport: the simulator's set-pose call returned no reply. Four of those
were check-state runs, which is why that kind looked worse than it was.

**Diagnosis.** A service timeout in the simulator, leaning towards later runs in a simulator's sequence. It is an instrument failure, labelled
`sim`, and not a VLM mistake. It still counted as a failed run.

**Fix.** Resend a set-pose that times out, at most twice, and log each resend. If it still fails, it is a crash as before.

**Evidence.** Report 011: in round 2, 8 retries and 0 crashes. A related instrument fault was found earlier: the simulator's command-line
tool died with a signal in 15 of 20† runs on the cluster, although never on the laptop. Retrying up to twice and logging it reduced failed reads to
zero in more than 43,000† attempts (a recorded deviation, report 006).

**Lesson.** An instrument that fails silently turns into a score. Every retry is logged, and a failure that persists is still a failed run.

## 10.6 The humanoid: pose, collisions and turn drift

**Symptom.** The first G1 smoke test in the clinic reached all places, but the hall arrival was 0.236† m off by ground truth, from drift in its turns. Early
walks had no sense of furniture, and in the dress rehearsal the robot routed from where the previous run had ended.

**Diagnosis.** Three separate causes: legs do not turn precisely; the first G1 had no obstacle sensing, so straight legs could clip furniture (a real gap in
the body, not in the agent); and after a layout reset the adapter reused a stale position.

**Fix.** The adapter recentres after each `look_around` turn (at most 0.08 m net per set). It has a simulated range scan for stop-and-detour. It waits for a
fresh pose and scan after every reset. A collision is now a real physical contact from the simulator, counted on every step, replacing a clearance
proxy. A deliberate walk into a chair is the control and registered contacts.

**Evidence.** Report 012: 160 of 160 walks without a contact, then 0 falls and 0 stalls in the 100 counted runs.

**Lesson.** Body problems belong to the adapter. The clean boundary is why they could be fixed without touching the agent, and why the swap check could
still show 0 changed lines.

## 10.7 Smaller findings worth keeping

- **The VLM forgot where it had been.** In phase 3 a run used 10 of its 12 steps because the history line said what was done but not where. History lines now name
  the room, and the longest later runs use at most 6 steps in phase 4 (mean 3.3†).
- **A step can use up its token allowance.** One count step used 8187 of the 8192 tokens allowed and still returned its call. Raising the limit would have changed the
  frozen interface, so it was left alone. In the final test 3 steps did hit the limit, each survivable.
- **Counting across photos is the weak spot.** There were over-merges of two neighbouring chairs in phase 6 (one run), phase 8 (one TurtleBot run) and a development
  run. No hint was added to the prompt, since that would have been a second change.
- **Sims can fail to start.** 2 of 27† bringup attempts got stuck in one job in phase 6. A job part that never started is resubmitted plainly, and the
  resume option correctly refuses it.
- **Neighbouring jobs can cross-talk.** Two sims on adjacent ROS domains discovered each other and shared a clock, and two jobs on one node did the same. Each job now has a
  block of domains.
