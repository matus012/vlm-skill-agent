# Task types and scoring

> **In plain words.** There are four kinds of question, each with one answer format. A run is only counted as a
> success if the answer is right, if the photo the robot points to really shows the object, if it took 12 steps
> or fewer, and if nothing crashed. The "right answer" is never typed in by hand: it is read from the simulator's
> own world description.

## 5.1 The four kinds

| Kind | What the task asks | Answer | Truth comes from |
|---|---|---|---|
| Find | where an object or a set of objects is | a room name | the room that holds the object (or all instances of the class) |
| Check state | whether the blue cabinet is open or closed | yes / no | whether the cabinet's state equals the state the task asks about |
| Count | how many instances of a class are in a room | a whole number | the number of such objects in that room |
| Conditional | "if X is not in room A, check room B" | a room name | the room that holds X, which is by construction room A or room B |

Task wording varies on purpose, so the VLM cannot lean on one sentence. The test list uses phrasings such as
"Track down the patient beds", "Locate the orange seats", "Is the blue cabinet standing open?", "Count the chairs in
the ward", and "Should the orange chairs not be in storage, check reception instead." In the blocked-doorway phase the
tasks also include "The red toolbox is in storage or reception. Find it."

Find and conditional answers must span more than one room across the list, so a planner that always says the same room
cannot pass. Only four classes may be the target of a task: toolbox, chair, bed and cabinet, because only those were
named correctly by the VLM in a naming check on the plain-shaped world (section 10.3). Plants, sofas, water coolers and
privacy screens are furnishings and distractors, never targets.

## 5.2 The proof photo

When it reports, the VLM must name the photo that shows the evidence. The simulator also renders, for every photo, a
segmentation image in which each pixel carries the label of the object it shows. This is how the scorer knows what
was really in view. The proof check passes if the named photo shows at least 1000 pixels of the target (for a count: of
any chair in the room). The smallest proof in phases 5 and 6 had 6574 pixels.

That check is what stops a lucky guess from passing. A VLM that says "no" about a cabinet it never saw, and happens to be
right, fails the proof check, because its photo contains 0 cabinet pixels.

## 5.3 The four checks

A run passes only if all four hold:

1. **Answer.** The answer equals the truth answer.
2. **Proof.** The photo named in the report shows at least 1000 pixels of the target.
3. **Steps.** The run ended with a report within 12 steps. Reaching the 12-step limit is a fail.
4. **Clean.** There was no crash and no hand help (nobody repositioned the robot).

The scorer is a plain script that reads the run directory and the world's truth file. It is mutation-tested, and the
rule that it recomputes the same verdict from the saved raw files was enforced throughout: any run could be re-scored
later and had to give the same answer.

## 5.4 Why a run fails: one label each

Every failed run gets exactly one label, so a failure is attributed to the part that caused it:

| Label | Meaning |
|---|---|
| `vlm` | the planner chose wrong: wrong answer, no proof, step limit, a bad format |
| `skill` | a skill returned wrong data, or the agent's own code broke |
| `nav` | `navigate_to` was blocked on a place that is reachable |
| `sim` | a crash in the simulator or the adapter, or the language-model server was unreachable |
| `scorer` | a bug in the scorer or the evaluator |

A crash is labelled by where the exception came from, not by who was running. A language-model server error is labelled
`sim` because it is an instrument failure, not a choice of the VLM.

## 5.5 Not the same as a good score

Two weaknesses of the scoring are known and recorded. First, the count proof passes with 1000 pixels of any chair,
not necessarily the ones that were counted. The scorer was frozen, so it was not changed; no counted result depends on
it as far as the totals show. Second, a bar on a sealed list of 25 per kind is coarse: one run is 4 points (section 11).
