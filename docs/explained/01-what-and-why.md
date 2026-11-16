# What this project is, and why

> **In plain words.** A robot looks at photos, a language model that can also read images decides what the
> robot should do next, and a small program makes sure the model only asks for things the robot can really
> do. We built that loop twice, once for a wheeled robot and once for a walking humanoid, and checked that
> the same model, the same prompt and the same code can drive both.

## 1.1 The question

Can one vision-language model drive different robot bodies from photos alone?

A vision-language model (VLM) is a neural network that reads pictures and text together and answers in
text. Here the VLM is used as a planner: at every step it is shown the latest camera photos and the task, and
it must choose one action. Nothing is trained or fine-tuned. The model is used as it comes.

The reason this is interesting is the usual cost of changing a robot. Robot software is normally written
for one body: its sensors, its motion planner, its map. If the planner only ever sees photos, place names and
the results of a few named skills, then the body can be swapped behind the skills and the planner should not
notice. That is the claim the project tests, with numbers, in simulation.

Three rules shaped everything else:

- The tasks are ordinary, not hard, and none is a trick question. Each one needs the VLM to look and decide,
  otherwise a fixed script would do.
- There is no training of any kind.
- The final demonstration is the same prompt and the same agent on two bodies, side by side.

## 1.2 The two bodies

![The two bodies. Same camera specification, very different height and motion; everything that is specific to a body lives in the robot adapter.](fig:bodies)

**Stage 1: TurtleBot 4.** A small wheeled robot, simulated in Gazebo Harmonic with ROS 2 Jazzy. Navigation is
Nav2 with a map made by SLAM Toolbox and localisation by AMCL. This is the body the system was built and tuned
on.

**Stage 2: Unitree G1.** A walking humanoid, simulated in MuJoCo. It walks with NVIDIA's pretrained AGILE
velocity policy, which we use without any change or training. Its camera sits at head height, about 1.2 m up,
so it sees the furniture from above. It has legs, so it can fall, and it turns and stops differently from
a wheeled base.

Both bodies carry the same camera specification: 90 degrees of horizontal field of view and 640 by 480 pixels.
Everything that is specific to a body lives in one small piece of code called the robot adapter (section 2).

## 1.3 The four kinds of task

Each task kind has one fixed answer format, so a script can score it.

| Kind | Example | Answer |
|---|---|---|
| Find | "Find the red toolbox." | a room name |
| Check state | "Is the blue cabinet shut?" | yes or no |
| Count | "How many chairs are there in the office?" | a whole number |
| Conditional | "Unless the orange seats are in the ward, search the storage room." | a room name |

The world is a small simulated clinic: a hall, and four rooms called ward, office, reception and storage
(section 3). Between runs the objects move, so the right answer changes from task to task.

## 1.4 The results on one page

Each round used 100 tasks that the prompt had never been tuned on, 25 per kind, with a pass bar of 18
out of 25 for every kind. A run passes only if its answer is right, its proof photo shows the object, it used
12 steps or fewer, and nothing crashed (section 5).

| Kind | TurtleBot 4, round 1 | TurtleBot 4, round 2 (final) | Unitree G1 | Bar |
|---|---|---|---|---|
| Find | 21/25 | 24/25 | 24/25 | at least 18/25 |
| Check state | 14/25 | 20/25 | 20/25 | at least 18/25 |
| Count | 23/25 | 24/25 | 24/25 | at least 18/25 |
| Conditional | 23/25 | 23/25 | 25/25 | at least 18/25 |
| All | 81/100 | 91/100 | 93/100 | |

**Terminology.** A round (the reports say "pass") is one full run of a sealed 100-task list. Round 1 used prompt v1 on set 1;
round 2 (the final one) used prompt v2 on set 2.

Three things to take from the table:

- TurtleBot round 1 missed the bar on one kind, check state (14/25). One diagnosed fix, a single
  sentence added to the prompt, was allowed. Round 2, on a different sealed test set, passed every
  kind (section 8).
- The G1 humanoid ran the same 100 tasks that TurtleBot round 2 (final) had run.
  It scored 93/100. Changing the body
  changed 0 lines of the agent: the program, the skills, the prompt and the packet builder are identical
  (section 9).
- On the G1 no run was lost to a fall or a stall: all 7 misses were wrong answers by the VLM, and 5 of them
  were one mistake, taking a look-alike object for the blue cabinet.

The whole project used 49.8† of the 120 H200 GPU-hours that were approved (appendix D). It is simulation
only, with simple boxy furniture and 25 runs per kind, so one run is worth 4 points of a kind's score
(section 11).

*A dagger (†), as in 49.8† above, marks a number from a task report that was not independently audited; every
other number comes from an audited report (section 1.5).*

## 1.5 How to read this document

Every section starts with a short "in plain words" box and then gives the detail. Section 2 to 5 describe the
finished system. Section 6 and 7 explain how it was built and how the evaluation was kept honest, which is
as much of the result as the scores are. Section 8 to 10 give the results and what went wrong. Section 11 is
the list of limits and section 12 shows how to run it yourself. The appendices hold the full prompt, the
skill interface, the log of every rule change, the compute that was used and an index of the reports.

Numbers marked with a dagger (†) come from a task report that was not independently audited. All other
numbers come from reports that an independent audit checked against the saved evidence, and the headline
results were also compared with the raw run files by a script. "Report 011" and similar names refer to the
numbered task reports listed in appendix E.

**Words used.** *Skill*: one named action the VLM may call. *Packet*: everything the VLM is shown in one step.
*Adapter*: the code that turns a skill into motion on one particular robot body. *Run*: one task from the first
photo to the report, or to the 12-step limit. *Round*: one full run of a sealed test list (the reports call it a pass). *Sealed*: written
and fixed before it is looked at, then used once.
