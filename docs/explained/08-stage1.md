# Stage 1 results: the TurtleBot 4

> **In plain words.** On the first sealed test the robot missed the bar on one kind of task, check state, and passed
> the other three. We looked at every failure, found one clear cause, and added one sentence to the prompt. On a second,
> different sealed test every kind passed: 91 of 100 tasks right. Nine misses remain, and each is explained below.

## 8.1 The test

Each round used 100 tasks: 25 for each of find, check state, count and conditional. The bar was 18 of 25 per kind (at
least 70 %). A task passes only if all four scorer checks pass (section 5.3). The rounds used different sealed sets,
built by the same procedure (section 7.3). The numbers of fair layouts that the procedure had to examine to fill the lists
are in the report; for round 2, 25 sealed layouts plus 68 fresh ones were consumed, and of the starts, 9 were in the
hall, 14 in another room and 2 in the answer room per kind.

## 8.2 Round 1: one kind missed the bar

![Runs passed out of 25, by kind. Whiskers are Wilson 95 % intervals; the dashed line is the bar of 18 of 25. Round 1 and round 2 (final) used different sealed sets.](fig:results)

| Kind | Round 1 (set 1, prompt v1) | Wilson 95 % CI | Round 2 (set 2, prompt v2) | Wilson 95 % CI | Bar |
|---|---|---|---|---|---|
| Find | 21/25 (84 %) | 0.65 to 0.94 | 24/25 (96 %) | 0.80 to 0.99 | 18 |
| Check state | 14/25 (56 %) | 0.37 to 0.73 | 20/25 (80 %) | 0.61 to 0.91 | 18 |
| Count | 23/25 (92 %) | 0.75 to 0.98 | 24/25 (96 %) | 0.80 to 0.99 | 18 |
| Conditional | 23/25 (92 %) | 0.75 to 0.98 | 23/25 (92 %) | 0.75 to 0.98 | 18 |
| All | 81/100 | | 91/100 | | |

A Wilson interval is the range of true pass rates that would plausibly produce the count we saw. With 25 runs it is wide.

Round 1 failed on check state alone: 14 of 25. The reports diagnosed it from the runs themselves, not from guesses.

![Check-state tasks in round 1 and round 2. Round 1 lost runs to look-alikes and to simulator crashes; in round 2 five look-alike misses remain.](fig:checkstate)

- **7 of the 11 failures were a clear VLM mistake.** In all 7 the robot never saw the real cabinet, which stands in
  storage, and answered about a different blue or teal object, usually the office sofa. The proof photo showed 0 cabinet
  pixels each time.
- **The split was clean.** When the robot did reach storage, 14 of 14 runs passed. When it did not, 0 of 7 did.
- **The other 4 failures were crashes** in the layout teleport (the simulator's set-pose service timed out), recorded
  as simulator failures, not as VLM mistakes. In all, 10 runs of round 1 crashed that way (section 10.5).

## 8.3 The one fix

The rule for Stage 1 was: below 70 % on any kind, one diagnosed fix, a new freeze, then the next sealed set. The fix was
a single generic sentence added to the end of the system prompt:

> "Things of a similar colour can look alike: answer only once a photo clearly shows the very object the task names; if
> you are not sure, look in the places you have not seen yet."

It names no room, no object and no world. The change made the frozen interface version 2, a new tag, and a test shows
that version 1 is exactly version 2 without that sentence.

Two instrument fixes were made at the same time. They changed how the test is run, not the agent. A set-pose call with an
empty reply or a timeout is now resent up to twice, each resend logged. And the gate that checks logs accepts a record of a
skipped teleport as complete, as an earlier gate had since phase 5. The fixes were decided before the second set was opened.

## 8.4 Round 2 (final): every kind passes

Round 2 ran on set 2 with the version 2 agent. Every kind is at or above 18 of 25.

- Check state rose from 14 to 20 of 25. That is not a controlled comparison: the sets differ, and the intervals overlap
  (0.37 to 0.73 against 0.61 to 0.91).
- By start place, the pass counts were: hall 33/36, another room 50/56, the answer room 8/8.
- The set-pose retry worked. Eight retries happened in total and 0 runs crashed, against 10 crashes in round 1.
- All 9 failures were labelled `vlm`. None was `sim`, `nav` or `skill`.
- Checking state now takes more steps: the mean rose from 3.4 to 4.72, because the agent looks in more places before it answers.

| Kind | Steps per run | Median VLM time per step | Median wall time per run | Sim speed (fraction of real time) |
|---|---|---|---|---|
| Find | mean 6.68, median 6, max 12 | 2.59 s | 475 s | 0.63 |
| Check state | mean 4.72, median 4, max 8 | 2.34 s | 310 s | 0.62 |
| Count | mean 3.0, median 3, max 4 | 1.88 s | 221 s | 0.57 |
| Conditional | mean 4.08, median 4, max 12 | 4.16 s | 258 s | 0.60 |

The runs are well under the 12-step limit on average, and no kind came close to being limited by it, apart from some
find runs that searched all five places.

The model's output is longer since version 2: the 99th percentile of tokens in one step rose from 3959 to 7269, and 3 steps
reached the 8192-token limit (two count runs and one conditional). None of those hid a tool call inside the reasoning.
Each came back as "call exactly one tool per step (got 0)", cost one step, and both count runs still passed. The decision was
to leave the limit unchanged, so the humanoid test would be run on the very same interface.

## 8.5 Every miss, classified

![The 9 TurtleBot misses and the 7 G1 misses of the final test, by cause. Every one was a wrong answer by the VLM.](fig:misses)

| Run | Kind | What happened | Class |
|---|---|---|---|
| p8p2c003 | check state | answered about a look-alike in the hall without going to storage; 0 cabinet pixels | look-alike cabinet |
| p8p2c052 | check state | "the blue (teal) cabinet in the office"; answer right by chance, proof photo 0 pixels | look-alike cabinet |
| p8p2c083 | check state | a teal look-alike "with its top open"; 0 cabinet pixels | look-alike cabinet |
| p8p2c094 | check state | went to the office, not storage; 0 cabinet pixels | look-alike cabinet |
| p8p2c095 | check state | "the teal cabinet in the office"; answer right by chance, proof 0 pixels | look-alike cabinet |
| p8p2c067 | count | answered 2, truth 3: merged the chair at the edge of photo 4 with one in photo 3 | over-merged across photos |
| p8p2c018 | find | its office look showed 2 chairs but it read them as "golden" or "amber", not orange, then guessed | amber chairs refused |
| p8p2c072 | conditional | its office photos held all 4 chairs; it said "no orange chairs" | amber chairs refused |
| p8p2c087 | conditional | storage photos held 4 chairs; it said "not there"; one step reached the token limit | amber chairs refused |

Reading the table:

1. **The look-alike cabinet is the main remaining miss: 5 of the 9.** The robot answers about a teal object (the office
   sofa, a hall object) with the real cabinet never in a photo. The one fix helped (14 to 20), but did not end it.
2. **Amber chairs, a new failure.** In 3 runs the orange chairs were in plain view (7000 to 12000 pixels each), and the VLM
   said they were not there. In the model's own reasoning the colour reads as hex values that it called "closer to a golden
   colour". Tasks that say only "chairs" are not affected. This is a reading of the reasoning text. Nothing was changed or
   tested for it.
3. **One over-merge**, the same kind as two earlier cases in the development phases: two neighbouring chairs on adjacent
   walls are taken for one chair seen twice.
