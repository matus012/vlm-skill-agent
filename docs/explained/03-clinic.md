# The simulated clinic

> **In plain words.** The test world is a small clinic with a long hall and four rooms. It is built from
> simple boxes and cylinders, on purpose, so that the right answer to every question is known exactly. Between
> runs the red toolbox and the chairs are moved by a random seed, so the VLM cannot learn where things are.
> In some runs a doorway is blocked, so a room cannot be reached at all.

## 3.1 The building

![Plan of the clinic: a hall and four rooms. Stars mark the five named places. Object positions are those of the fixed layout; the toolbox and the chairs move between runs.](fig:plan)

The clinic is a hall 10 m long and 3 m wide with four rooms of 5 m by 5 m opening off it: the **ward** and
the **office** on the north side, **reception** and **storage** on the south side. The doorways are staggered
(the ward and office doors are at the inner ends of the hall, the reception and storage doors at its outer ends)
so that from a room's viewing spot the robot cannot see countable objects of another room through a door.

Each room has one named place, a point from which one turn on the spot shows the whole room. The VLM can
`navigate_to` five places: hall, ward, office, reception and storage. The point for each place was chosen
so that every object in the room lies inside one of the four photos of a `look_around`.

The world has 20† labelled objects:

| Room | Furniture and objects |
|---|---|
| Ward | three blue beds in a row, a trolley with a yellow bin, a green privacy screen |
| Office | a desk with a monitor, a bookshelf, a teal sofa, a plant |
| Reception | four orange chairs in a row, a counter, a water cooler, a plant |
| Storage | the red toolbox, the blue cabinet, a metal rack with boxes, a stack of boxes |

Everything is a primitive shape. The honest consequence is that "furniture" is crude: a sofa is a teal box
with a back. Section 8 shows that this matters, because the VLM sometimes takes the teal sofa for the blue cabinet.

## 3.2 What the camera sees

![Why the camera was replaced, and the self-view crop. Four turns of the stock camera leave blind wedges (left); four turns of 90 degrees tile the circle (middle); rows that show the robot's own body are cropped before the VLM sees a frame (right).](fig:camera)

A camera that sees 90 degrees needs four photos to cover a full turn. The stock TurtleBot 4 simulation camera is
narrower, 71.6° at 320 by 240 pixels. Four turns of 71.6° cover only 286.5°†, which leaves a blind wedge of 18.4°†
between every pair of photos, so a `look_around` could not show a whole room. The project therefore replaced the
camera description with its own copy (a small overlay package, never an edit of the installed one) at 90° and
640 by 480. The G1's camera has the same specification.

Four photos of exactly 90° tile the circle with no overlap, so any heading error would leave a gap or show
an object twice at a seam. Two measures handle that. `look_around` turns to absolute headings (the place's yaw
plus 0, 90, 180 and 270 degrees, from the localisation), not four relative turns, so errors do not add up. And
the world was checked before it was frozen: every object must be well visible in one photo, countable objects
must appear in one photo only, and nothing from another room may show through a door.

## 3.3 The self-view crop

The wide camera also sees the robot itself. The TurtleBot's top plate and front body sit closer than the camera's
0.3 m† near clip, and they were drawn as flat sky-blue patches covering 6.5 to 10 %† of the frame at the top
and 3 to 5.5 %† at the bottom. In the first test the model described "a small blue object (possibly a vehicle or
cart)" in the foreground: it had invented an object from the robot's own body.

The fix was to crop fixed rows off the top and bottom of every photo before the VLM sees it. The crop is measured
once per robot, from many poses, plus a margin, and it lives in the adapter. It is also a rule of the rig that every
check from then on is run on the cropped frame, that is, on what the VLM really gets. The crop costs some
vertical field of view (from about 74° to about 60°†) and none of the horizontal 90°.

Cropping was preferred to lowering the near clip, which would show the robot's body to the VLM and so reveal which
robot it is. The G1's head camera sees none of its own body, so its crop is zero. To be sure the crop checks were
not vacuous, the project required that the same check must fail on an uncropped frame (a negative control) and it
did.

## 3.4 Moving things around: seeded layouts

If the toolbox were always in storage, the VLM could answer "storage" without looking. So from phase 5 on, a
seed decides where the movable objects stand: the red toolbox and the four orange chairs. The same seed always
gives the same layout.

A layout is only accepted if it is fair. Movable objects have to stand where they are fully visible from the room's
viewing spot, clear of furniture outlines, and wholly inside one photo, so that counting is not a trick question.
A ray-casting renderer checked each candidate seed offline (306† of the seeds 0 to 699 passed), and live spot
checks in the simulator confirmed a sample. A seed that breaks a rule is rejected, never quietly redrawn.

At run time the objects are moved by teleporting them in the simulator, and read back: they must land within 2 cm
and 1 degree, or the run is recorded as a simulator failure.

The cabinet is the fixed object of "check state" tasks. It exists in two worlds, one with the door closed and one
with it open. A cabinet is used, not a room door, so that the state question never changes navigation.

One naming limit also shaped the layouts. The VLM calls an isolated red box a "toolbox" in reception and storage,
but a "briefcase", "suitcase" or "red cabinet" in the office and the ward. So the toolbox is only ever the
*answer* in reception or storage; in the ward the find-tasks ask about the beds (section 10.3).

## 3.5 Blocked doorways

Nav2 steers around small obstacles by itself, so a box in the middle of the hall changes nothing. To test what the
VLM does when a room cannot be reached, a box is placed in a doorway (on the hall side) so that the room is
truly unreachable. The planner then has no path and gives up quickly, in 14 to 51 simulated seconds. The skill
returns `failed: blocked` together with a photo, and the VLM has to choose again. The ground-truth checker knows
that the blocked room cannot be reached and drops it from the places a robot may be expected to visit.

A blocker is not a countable object. It has its own label and is yellow so that a VLM which looks in the doorway
can see why it failed. Blocked doorways were tested in phase 6 only; the final test lists have none (section 11).

## 3.6 The world on the humanoid

The G1 lives in a MuJoCo copy of the same clinic, written by a generator from the same layout file, with the same
segmentation labels. The spec objects are movable bodies that the same seed layouts reposition. The G1's camera
is a head camera with a vertical field of view of 73.74°† that gives the same 90° horizontal view at 640 by 480.
