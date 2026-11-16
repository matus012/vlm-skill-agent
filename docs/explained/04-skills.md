# The skills

> **In plain words.** The VLM has exactly three buttons: go to a named place, turn on the spot and take four
> photos, and give the final answer. Each button has a fixed meaning, a fixed result text and a defined way of
> failing. There is no fourth button for looking closer, because measurement said it was not needed.

## 4.1 The three skills

| Skill | What the code does | What comes back |
|---|---|---|
| `navigate_to(place)` | looks up the place's point in the place file, sends a navigation goal, waits | `reached` or `failed: blocked`, and one new photo |
| `look_around()` | turns to four headings 90° apart, one photo at each | `4 photos`, numbered 1 to 4 |
| `report(answer, photo, note)` | stops the robot and saves everything | ends the run |

Every argument is required and the schemas reject extra fields. The tool definitions sent to the VLM are in
appendix B.

## 4.2 navigate_to

**What it does.** The place name is one of hall, ward, office, reception, storage. The code turns the name into
a point and a heading and asks the adapter to go there with a time limit of 120 simulated seconds.

**What it returns.** `reached`, or `failed: blocked`, plus one photo taken after arrival (or after giving up).

**How it fails.** `reached` is claimed only if the navigation said it succeeded and the robot's own position estimate
is within 0.30 m of the place point. Both checks are needed. The first Nav2 setup had a planner tolerance of 0.5 m,
and a goal placed inside a wall came back as a success with the robot 0.4† m away, so the planner tolerance was set to
zero and the 0.30 m rule stayed as a second guard. Anything else (an aborted goal, a rejected goal, the 120 second
timeout) is `failed: blocked`. A name that is not a place is not a navigation failure but a malformed call (4.5).

Typical cost: about 28† simulated seconds on the TurtleBot.

## 4.3 look_around

**What it does.** The robot turns to its start heading, takes a photo, turns 90°, takes another, and so on for
four photos. The start heading is the place's own yaw if the robot is within 0.30 m of a place, else its current
heading. This reproduces the exact photo set the world was validated with.

**What it returns.** The text `4 photos` and the four photos, numbered 1 to 4 in turning order. Because the four
photos tile the circle, each wall of a room appears in exactly one of them.

**How it fails.** It does not return an error as such. The failure is perceptual: a seam can split an object
across two photos, and the VLM may then count it twice or merge two neighbouring objects into one (section 8.5).
On the humanoid a turn can drift, so the adapter recentres after each turn.

Typical cost: about 66† simulated seconds, the most expensive skill.

## 4.4 report

**What it does.** The call has three fields: `answer`, `photo` and `note`. The answer is checked against a format that
depends on the task kind:

- find and conditional: one of the room names;
- check state: `yes` or `no`;
- count: digits only, a whole number.

`photo` must be a photo that is in the current message, so a report always points at an image the VLM could really
see. `note` is free text for the demo and is never scored.

**How it fails.** A report in the wrong format is not accepted. It comes back as an error, costs a step, and is not
scored as an answer. A report is also not enough on its own to pass: the scorer separately checks that the photo
named really shows the object (section 5.3).

## 4.5 Wrong calls

The runtime treats a bad call as a normal event, not a crash. An unknown skill, an unknown place, no tool call, several
tool calls, invalid JSON, or a photo number that does not exist all give an error text that goes back to the VLM as
the result of that step. It costs a step. Two such errors in a row end the run as a failure. In 7 of 7 deliberate
test cases during the build, each planted mistake failed on exactly the check named for it, and a mistaken call
did not stop a run that went on to a correct report.

## 4.6 Why there is no `inspect` skill

The brief allowed a fourth skill, `inspect(photo, side)`, that would give one close-up photo, but only if the numbers
asked for it. The rule was: measure how often the proof photo passes in the runs of phases 5 and 6, and add `inspect`
only if that rate is below 90 %.

The proof photo passed in all 60 of the counted runs that reached a report, with the smallest proof showing
6574 object pixels against a bar of 1000. That is 100 %, so there is no `inspect`. The measurement was recomputed
from the raw run directories, not from tables, and it matched the saved scores in every run.

## 4.7 What a skill cannot do

The skills do not grasp, open doors, explore unknown buildings, recall earlier runs or detect objects with a separate
detector. Those were left out on purpose (the brief lists them), to keep the claim narrow: a VLM that looks and
decides, with nothing else helping.

## 4.8 The interface freeze

At the end of phase 7 the skill interface was frozen: the three tool schemas, the system prompt, the packet fields and
text format, the runtime rules (12 steps, 2 malformed in a row, the 120 second limit, the 0.30 m bound), the sampling
and the adapter API. Each is stored in a machine-readable copy, and a test fails on any difference. A mutation
test showed the test itself works: 3 of 3† planted changes (a schema, the prompt, a packet field) were caught.

The freeze was lifted exactly once, to add one sentence to the prompt after round 1 of the TurtleBot test.
That produced version 2, and the humanoid ran against version 2 with a check that nothing differs from it (section 7.5).
