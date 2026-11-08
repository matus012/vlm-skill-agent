# sim/INTERFACES.md — contracts between the Phase 1 pieces (task 002)

Frozen at the start of 002 so the world, the camera instrument and the gate checks can be built in parallel.
Change only by editing this file first, in its own commit.

## 1. Segmentation labels (Gazebo label system, `gz::sim::systems::Label`)
- One **unique** label id per object instance, `1..199`, assigned by the world generator, stored in `truth.json`.
- Structure: walls `240`, floor `241`, ceiling `242`, door frames / trims `243`, other static structure `244`.
- `0` = unlabelled = background (sky / clear colour) **or the robot itself**. The robot is never labelled.
- (002-C, measured) The segmentation camera sees *through* the robot's own body (inside the 0.3 m near clip), while the
  RGB camera paints it as background colour, exactly where the rgbd depth is `-inf`. So `probes/photos.py` sets every
  pixel with depth `-inf` to `0` in the saved label PNG: label `0` in a **saved** frame = background or robot, i.e.
  what the VLM sees as non-world.
- G7 "unlabelled pixel" = label `0` in the cropped, saved segmentation frame.
- The `standard_dock` (spawned by the stock launch) carries no label: it must never be in view at a place.
  `sim/launch_clinic.sh` moves it to (20, 20), outside every room, after spawn (launch aborts if the move fails);
  G5 exempts it only when its live pose is asserted there (design notes a recorded deviation).
- Segmentation camera: semantic type, co-located with `oakd_rgb_camera_frame`, same HFOV / width / height / clip as
  the RGB camera (`ros_ws/src/turtlebot4_description/urdf/sensors/oakd.urdf.xacro` macro params). The label id is
  read from the frame as documented in `probes/photos.py` (which channel holds it).

## 2. `sim/worlds/clinic/layout.yaml` → generator → `clinic.sdf` + `truth.json` + `map_knowledge.json`
- Generator: `sim/worlds/clinic/generate.py layout.yaml --out <dir>` (deterministic: same input → byte-identical
  output; Phase 5 will add `--seed`, not now).
- `truth.json`:
  ```json
  {"world": "clinic", "layout_sha256": "<sha256 of layout.yaml>",
   "rooms": {"ward": {"polygon": [[x, y], ...], "doorway": {"center": [x, y], "width": 1.4}}, ..., "hall": {...}},
   "objects": [{"id": "toolbox_red_1", "class": "toolbox", "colour": "red", "room": "storage",
                "pose": [x, y, z, roll, pitch, yaw], "state": null | "open" | "closed",
                "countable": false, "label": 7, "model_name": "<gz model name in the world>"}],
   "structure_labels": {"wall": 240, "floor": 241, "ceiling": 242, "trim": 243, "other": 244}}
  ```
- World frame = Gazebo world frame; poses are world-frame metres / radians.
- `countable: true` for every class a Count task may ask about (at least chairs, beds).
- `map_knowledge.json` (005 Step 0, design review N8): the generator also writes
  `generated/map_knowledge.json` = `{"world", "layout_sha256", "rooms": {name: {"polygon", "doorway": {"center",
  "width"}}}}`: the `rooms` block of `truth.json` (all rooms incl. the hall; the hall has no `doorway`), and nothing
  else: no objects, no states, no labels. Byte-deterministic like the other two outputs. Robot / agent code
  (`agent/`, `adapters/turtlebot4.py`, `humanoid/` adapter + router) reads only this file for room geometry. Only the
  scorer side (`agent/truth.py`, `agent/scorer.py`, the harness `adapters/run_task.py`) and instruments (`probes/`,
  `tasks/derive_photos.py`, `humanoid/probes/`, tests) may open `truth.json`; `tests/test_truth_readers.py` enforces it.

## 3. Robot config `config/robots/turtlebot4.yaml`
```yaml
robot: turtlebot4
camera: {hfov_rad: 1.5708, width: 640, height: 480}
self_crop: {top_px: <int>, bottom_px: <int>}   # measured (max over columns, poses) + 8 px margin
self_crop_evidence: <evidence file name>
```
Cropped frame = rows `[top_px, 480 - bottom_px)`, all 640 columns. Applied identically to RGB and segmentation.

## 4. Photo script `probes/photos.py`
- `python3 probes/photos.py --place <name> --out <dir> [--headings 0,90,180,270] [--yaw0 DEG]` with the sim +
  Nav2/AMCL running.
- Turns to **absolute** headings `start_yaw + k·90°`. `start_yaw` = `--yaw0` (the gate runner passes the place's yaw
  from `places.yaml`, so photos are at place-yaw + k·90° however the robot arrived); without it, read once from
  localization (`map`→`base_link`). Settles, then saves per photo `k`:
  `<dir>/<place>/<k>_rgb.png` (cropped RGB), `<k>_seg.png` (cropped label image, label id per pixel, lossless),
  `<k>_seg_full.png` (the same label image uncropped: G7's negative control only, never shown to the VLM),
  and one `<dir>/<place>/photos.json`: per photo target heading, ground-truth camera pose (world frame, from
  Gazebo), ground-truth yaw error vs target, crop used, sim time.
- Frames: camera + segmentation run at 5 Hz sim time (sensor-load cut, a recorded deviation); each photo uses frames stamped
  after the settle.
- Gate runner (`probes/phase1_gates.py`): photos of a place count only if the last navigation to it SUCCEEDED and
  the ground-truth pose is within 0.30 m of the place point right before the photos (`at_place` + error recorded);
  each place has exactly 4 photos at place-yaw + 0/90/180/270°; G2/G3/G4/G7 refuse to run on a place that fails this.
- Gate checks read only these files plus `truth.json`.

## 5. Map + places
- `sim/maps/clinic.{yaml,pgm}` (SLAM Toolbox), `sim/worlds/clinic/places.yaml`:
  `places: {ward: {x, y, yaw}, reception: {...}, storage: {...}, office: {...}, hall: {...}}` in the `map` frame.
- `map` frame = Gazebo world frame (G8: identity is the best offset at every place), so places and truth poses share
  coordinates with no transform.
- The hall place is not the spawn point: it is in the east half at yaw 45°, where no countable object is in sight
  and the robot sees itself (G7's negative control is exercised there).
- Sensor rates (a recorded deviation): lidar 10 Hz, 640 samples; camera + segmentation 5 Hz; no point-cloud bridge
  (`sim/launch/bridge_115.launch.py`).
- AMCL initial pose via its own params (`set_initial_pose: true`, `initial_pose: {x, y, yaw}`), not a one-shot
  `/initialpose` publish.
- Nav2 goal tolerance (002 rehearsal 3, measured): `controller_server` `general_goal_checker.xy_goal_tolerance`
  = **0.10 m** (stock turtlebot4 0.25 m), yaw tolerance and everything else stock. With 0.25 m the robot stops on the
  approach side of office / storage (+0.23 m mean in x) and AMCL's few cm tip the true distance past the 0.30 m place
  bound (G1, `at_place`). `sim/launch_clinic.sh` writes the copy and passes it as `params_file:=`; the 0.30 m bound
  is unchanged.
- Nav2 planner tolerance (003, a recorded deviation, 2026-11-04): `planner_server` `GridBased.tolerance` = **0.0** (stock
  NavFn 0.5 m). With 0.5 m a goal inside a wall is snapped to the nearest reachable cell and comes back SUCCEEDED
  0.36–0.48 m away; with 0.0 a goal in a lethal cell fails at once (ABORTED). Every place point is in a free cell.
  Re-counted: Phase 1 G1 on this setting + world v2 (003 report).
- AMCL update gates (003 fix round 1, 2026-11-04): `amcl` `update_min_d` = **0.05 m**, `update_min_a` = **0.1 rad**
  (stock turtlebot4 0.25 m / 0.2 rad); every other AMCL param stock, map unchanged. With the stock gates the filter
  skips the last < 25 cm creep and the final turn, so its estimate is stale where Nav2 judges the goal: at the hall
  (10 × 3 m corridor) AMCL − GT along x reached 0.207 m (mean |.| 0.058, signed −0.052) vs 0.094 across, and GT ended
  up to 0.276 m from the hall place (19/20 < 0.20 m). With 0.05 / 0.1: along max 0.045 (mean 0.018), across max
  0.028, GT ≤ 0.148 m, 20/20 < 0.20 m (`probes/hall_diag.py`, 20 hall arrivals cycling ward / office / reception /
  storage; real-time factor unchanged, 0.89). The values live in `config/robots/turtlebot4_amcl.yaml` (flat
  `{param: value}`); `sim/launch_clinic.sh amcl:=<yaml>` merges any such file into the stock AMCL params copy (an
  unknown key fails the launch), default = that config file.

## 6. One Gazebo at a time (2026-11-04: the local machine GPU is serialised)
- Before starting any sim: `until mkdir $BUILD_DIR/gz.lockdir 2>/dev/null; do sleep 15; done`, then write
  your name + UTC time to `gz.lockdir/owner`. Release with `rm -rf $BUILD_DIR/gz.lockdir` once your sim's
  process group is dead. Hold it ≤ ~20 min per session. Never remove a lock whose owner isn't you.
- A counted gate run (G1–G8 in one go, ~45–55 min) holds the lock for its whole length with no other sim running
  (2026-11-04); the ~20 min limit is for rehearsals and lanes that share the GPU.
- Each worker still sets its own `ROS_DOMAIN_ID` and `GZ_PARTITION` before `source sim/env.sh`, starts its sim with
  `setsid`, and stops it with `kill -- -<pgid>`; never `sim/stop.sh` or broad `pkill` while others work.
