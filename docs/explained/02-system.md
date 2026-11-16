# The system

> **In plain words.** Four parts take turns. The planner (the VLM) looks and chooses one skill. The skill
> caller checks the choice, runs it, and writes it down. The robot adapter turns the skill into motion on a
> particular body. Then new photos come back and the loop repeats until the VLM reports an answer or runs out
> of its 12 steps. The VLM is kept blind to everything except photos, place names and its own history.

## 2.1 The loop

![The system. The VLM sees photos and picks one skill; a runtime checks and runs it; the adapter drives the body, and new photos come back. The VLM never sees maps, lidar, coordinates, joint angles or the robot type.](fig:system)

The brief that defined the project describes three roles:

- **Planner.** The VLM picks the next skill, one step at a time. It is Qwen3.8-27B, served by vLLM on one
  H200 GPU of a GPU cluster (PERUN, H200 GPUs). Thinking mode is on, so the model reasons before it answers.
  It is asked for one tool call per step.
- **Caller.** The agent runtime checks the call against the skill interface, runs it, logs it, and returns
  the result to the planner. A wrong skill name, an unknown place or a malformed answer does not crash
  anything. It comes back to the VLM as an error message and still costs one step.
- **Executor.** The skill code asks the robot adapter for motion, the adapter drives the body, and the body
  moves and takes photos.

Two rules keep the parts apart. All robot-specific code stays in the adapter. And the agent and the skills
contain no ROS imports at all: a test scans the source to enforce it. That is why the same agent can be
started on two very different simulators.

## 2.2 One run, step by step

![One real run from the final test (run p8p2c005): three steps, answer "ward", scored by a script against the simulator's truth.](fig:run)

A run is a short story. A real one from the final test: the task was "Unless the orange seats are in the
ward, search the storage room", the robot started in the hall, and the answer was "ward".

1. A simulator loads the world with a seeded object layout and the robot is placed at its start. The true answer
   is computed from the world file and kept away from the agent.
2. The context builder makes the first packet: the task, one photo, the current room, the place names, the
   step counter.
3. The VLM picks one skill. Here it chose `navigate_to(ward)`.
4. The runtime checks it and runs it. The robot drives to the ward and returns one photo.
5. The new packet carries the new photo and a history line, "navigate_to(ward) from hall -> reached". The VLM
   chose `look_around()`, got four photos, saw the orange seats, and called
   `report(answer="ward", photo=1, note=...)`.
6. A script scores the run against the simulator's truth. That run passed in 3 steps.

The run ends on `report`, or at 12 steps, which counts as a fail.

## 2.3 What the VLM sees, and what it never sees

This is the most important design rule, because it is what makes the body swap honest.

**It sees:**

- 1 to 4 numbered photos (the latest `navigate_to` photo, or the four `look_around` photos), shrunk so the longest
  side is at most 640 pixels;
- the name of the room the robot is in, and the list of place names;
- a one-line history of its earlier steps, each naming the room where the step started;
- the result of the last skill, the task text, and the step counter ("Step n of 12").

**It never sees:** lidar, the occupancy map, any coordinates, joint angles, or which robot it is on. Tests scan
the logged packets of every run for such words and numbers. For the 100 humanoid runs that scan found 0 hits,
including the words "g1", "unitree" and "humanoid".

The only room geometry that agent and robot code may read is a small file with the rooms' outlines and doorways.
Object positions and answers are in a separate truth file that only the scorer and the logging code read. A test
checks this split by looking at which modules import which.

## 2.4 The robot adapter

The adapter is the single seam between the agent and a body. It has seven methods and nothing else crosses it:

| Method | What it does |
|---|---|
| `go_to(x, y, yaw, timeout)` | drive or walk to a point; returns reached or blocked |
| `turn_to(yaw)` | turn to an absolute compass heading and report the remaining error |
| `photo()` | one camera frame, already cropped as described in section 3.3 |
| `pose()` | the robot's own estimate of where it is (never given to the VLM) |
| `where()` | the name of the room that estimate falls in |
| `stop()` | zero velocity, cancel any goal |
| `sim_time()` | the simulator clock |

The TurtleBot adapter sends Nav2 goals and uses AMCL for its position. The G1 adapter walks legs with the
pretrained policy, uses the simulator's exact pose in place of SLAM, and adds a simulated range scan so it can
stop and walk around furniture. Neither of those choices reaches the agent.

## 2.5 The model call

The planner sends the system prompt (appendix A), the packet and the three tool definitions (appendix B) to the
vLLM server through its OpenAI-compatible interface. Sampling is the one the model card recommends for thinking
mode: temperature 1.0, top-p 0.95, top-k 20. The reply may use up to 8192 tokens, and each run has its own
random seed. Two malformed replies in a row end the run as a failure. In TurtleBot round 2 (final) the median step took between 1.9 and 4.2 seconds, depending on the kind of task,
and a few long deliberations took more than 140 seconds.

## 2.6 Where it all runs

Early work used a laptop (Gazebo, Nav2 and the agent) with the VLM on the cluster over an SSH tunnel. From the
fourth phase on, the simulators moved to the cluster too, packed into container images: one GPU job holds
a vLLM server and two simulators, and two or three such jobs run at the same time (section 10.2). The humanoid
stack has its own image. Nothing about that changes what the agent sees.
