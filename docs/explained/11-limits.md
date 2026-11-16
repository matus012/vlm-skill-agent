# Limits

> **In plain words.** This is a careful experiment in a simulator with simple furniture. It shows that one VLM and one
> prompt can drive two different bodies at about the same level, with the failures understood. It does not show that this
> works on a real robot, in a real building, or with every VLM.

## 11.1 What the results cannot say

- **Simulation only.** Nothing was run on a real robot. Cameras, lighting, motor noise, floors and doors in the real world
  are all different.
- **Simple assets.** The world is made of boxes, cylinders and spheres in four rooms and a hall. The VLM was asked about
  objects that stand clearly visible in the open. Real clutter, textures and occlusion are untested. The world was
  shaped on purpose to be fair to the checks, with objects arranged so that every one is visible from the room's place, so
  it is easier than a building nobody arranged.
- **A known building.** The agent is given the names of the places, and the skill code knows how to get there. Unknown buildings, doors
  that open, grasping, and objects that are not there at all were left out on purpose.
- **Small samples.** There are 25 runs per kind, so one run is 4 points of a kind's score. The intervals are wide: for
  check state, 20 of 25 is anywhere from 61 to 91 % at 95 % confidence. Differences between the two robots of one or two tasks
  are noise. Each task was run once with one random seed, so run-to-run variation of the VLM is not separately measured.
- **The prompt fix is not a controlled comparison.** Check state rose from 14 to 20 of 25, but on a different set of tasks. The
  intervals overlap, and 5 of the remaining misses are the same mistake the fix was written for.
- **One VLM.** Only Qwen3.8-27B, in thinking mode, with one sampling setting, was evaluated. The backup model named in the brief was
  never needed. Other models may fail in other ways, and the hex-colour reading of the chairs may be specific to this one.

## 11.2 Known weaknesses that were not fixed

- **Look-alike objects.** The teal sofa and other teal objects are taken for the blue cabinet when storage is never visited (5 TurtleBot
  and 5 G1 misses in the final test).
- **Colour naming.** Amber-looking chairs were refused as "orange" in 3 TurtleBot runs. They did not recur on the G1, but that is a small
  number.
- **Counting across photos.** Two neighbouring chairs on adjacent walls are sometimes merged into one chair seen twice (one run in the final
  test, one in phase 6).
- **The toolbox.** It is a valid answer only in reception and storage, because the VLM does not name it in the office or the ward. From the
  G1's head height it named the toolbox in reception only 2 of 5 times in the naming check, although all 3 such test tasks passed.
- **No blocked doorways in the final test.** That behaviour was measured only in phase 6, in 20 counted runs, with 10 of 10 blocked runs passing.
- **The scorer's count proof** passes with any chair's pixels, not necessarily the counted ones. The scorer was frozen, so it was left.
- **The G1 limits.** It walks at low speed with a pretrained policy in a room with little clutter. It is not a manipulation robot. Two
  back-to-back turn sets can drift by up to 0.087 m, a case that never occurred in a counted run.
- **Speed.** A simulator ran at 0.4 to 0.7 of real time, and the median wall time per run was between 221 and 475 seconds. This is an
  experiment, not a real-time controller.

## 11.3 Things that could mislead a reader

- The "G1 beats the TurtleBot, 93 against 91" is two tasks apart on 100, inside the noise.
- The video clip of the G1 in the side-by-side is a labelled re-run, and that re-run failed (3 against 4). The counted run of that
  task passed.
- Numbers marked † have not been audited by an independent check. They are accurate to their source report, as checked by script, but
  nobody re-derived them from raw evidence.
- The project lead made certain decisions, such as running the humanoid test despite a failed precondition. They are listed in the
  deviation log (appendix C) and are not hidden.
