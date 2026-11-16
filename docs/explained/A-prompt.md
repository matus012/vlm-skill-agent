# The full prompt

> **In plain words.** This is the exact text that tells the VLM what it is and what it may do. It is the same for both
> robots, and it is written without any word about a body, a map or a coordinate.

## A.1 The system prompt (version 2, frozen)

The place list in the second line is filled in from the place file; in the clinic it reads
"hall, ward, office, reception, storage". The last sentence is the one added after round 1 of the TurtleBot test; version 1
is this text without it.

```text
You control a robot in a building and see only through its camera. Do the task in the user message.
Places you can go to, each a room of the building: hall, ward, office, reception, storage.
Each step, call exactly one tool:
- navigate_to(place): go to a place; returns one photo from where you end up.
- look_around(): turn on the spot; returns four photos in different directions.
- report(answer, photo, note): give the final answer; this ends the task.
Photos are numbered from 1 in each message. They come from your last navigate_to or look_around (at the start: one photo from where you are); a wrong call does not change them. Older photos are not shown again.
Answer format: a find or where task -> one room name from the list above; a yes/no question -> yes or no; a how-many question -> a whole number in digits.
Report only what you saw in a photo. The photo number in report must be a photo in the current message that shows the evidence.
You have at most 12 steps; running out is a failure. A wrong call (no tool, several tools, an unknown place, a wrong answer format) comes back as an error and still costs a step; two wrong calls in a row end the task as a failure.
Look before you decide. Things of a similar colour can look alike: answer only once a photo clearly shows the very object the task names; if you are not sure, look in the places you have not seen yet.
```

(Long lines are wrapped by the page; each line above is one line of the real prompt.) Its SHA-256 for the clinic's places is
`f8374e72f1b2fa018936f47d4c93533404b64a79f415cf5b5e5805e4f7e16a7a`, and every run records the hash it used. Version 1 had hash
`531d7c884f90c566775cbe6ea14e2ae9eba113bff870c2d6c46a2b03b055677c`.

## A.2 The message the VLM gets at each step

After the system prompt, the user message is plain text followed by the numbered photos, each a JPEG of at most 640 pixels on its longest side:

```text
Task: <the task text>
Step <n> of 12.
Current room: <room name>
Places you can navigate to: hall, ward, office, reception, storage
History:
1. navigate_to(ward) from hall -> reached
2. look_around() in ward -> 4 photos
Last result: 4 photos
The photos below are numbered 1 to 4.
Photo 1:
<image>
Photo 2:
<image>
...
```

At the first step the history reads "History: (none)" and one photo is shown. A history line says `from` the starting room for `navigate_to`
and `in` the room for every other call, so the VLM can see where each look-around happened. Wrong calls keep their own wording and the same
room naming. Room names are the only location information; there are no coordinates.

## A.3 How the model is called

| Setting | Value |
|---|---|
| Model | Qwen3.8-27B, served by vLLM 0.32.0 |
| Mode | thinking on (the model reasons before it answers) |
| Sampling | temperature 1.0, top-p 0.95, top-k 20, presence penalty 0.0 |
| Maximum reply | 8192 tokens |
| Seed | one per run, derived from the task id |
| Tools | the three tool definitions in appendix B, `tool_choice` automatic |
| Malformed replies in a row that end a run | 2 |
