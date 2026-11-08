# humanoid/ — Unitree G1 in MuJoCo on ROS 2 Jazzy (Phase 9 spike, side lane H)

A G1 walks with NVIDIA WBC-AGILE's pretrained `Velocity-G1-v0` policy (LEAPP ONNX) inside a room, driven through
`/cmd_vel`, and publishes a 90° head camera plus its ground-truth pose. Spec:
`the Phase 9 humanoid spike spec (not included)`. Nothing here imports `agent/`, and `agent/` never imports this.

## Setup (the local machine, once)
```bash
humanoid/setup.sh      # pinned clones -> $BUILD_DIR/ext, venv -> $BUILD_DIR/.venv-g1
```
Never `uv sync` AGILE's pyproject (it pulls Isaac Sim + torch cu128). Licences: `LICENSES.md`.

## Run
```bash
setsid humanoid/run_g1.sh humanoid/g1_node.py &          # sim node, real time, ROS_DOMAIN_ID 18
humanoid/run_g1.sh humanoid/drive.py goto 1.5 -1.0        # waypoint (world metres)
humanoid/run_g1.sh humanoid/drive.py heading 90           # absolute heading (deg)
setsid humanoid/run_g1.sh humanoid/g1_node.py --clinic sim/worlds/clinic/layout.yaml &   # the clinic, spawn hall (0,0,0)
humanoid/probes/clinic_smoke.sh storage reception         # adapter smoke under gz.lockdir (task 003)
kill -TERM -<pid>                                          # stop your own process group only
```
`run_g1.sh` sets `ROS_DOMAIN_ID=18` (override with the env var), sources `sim/env.sh` (Jazzy + the FastDDS
localhost profile — ufw drops multicast), `MUJOCO_GL=egl`, and `PYTHONPATH=<repo>:<AGILE clone>`.

| topic | type | notes |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | in: `linear.x`, `linear.y`, `angular.z`, clamped by AGILE's CommandManager to ±0.5 m/s, ±0.5 m/s, ±1.0 rad/s; zeroed after 0.5 s sim without a message |
| `/g1/odom` | `nav_msgs/Odometry` | ground truth from the free joint, `world` → `pelvis`, 50 Hz; also on `/tf`. **Not for the agent** (brief rule) |
| `/head_rgb/image_raw` | `sensor_msgs/Image` rgb8 640×480 | 10 Hz sim |
| `/head_rgb/camera_info` | `sensor_msgs/CameraInfo` | fx = fy = 320, cx 319.5, cy 239.5 (`fovy` 73.74° = 90° HFOV) |
| `/head_rgb/segmentation` | `sensor_msgs/Image` 16UC1 | MuJoCo body id per pixel (0 = world); names in `$BUILD_DIR/g1_scene/scene.json` |
| `/clock` | `rosgraph_msgs/Clock` | sim time; every stamp above is sim time |

QoS: `/clock`, `/g1/odom`, `/tf`, `camera_info` use the **sensor-data QoS (best effort)**; the two images are
**reliable, depth 2**. Why: a reliable `/clock` publisher hung forever inside FastDDS `publish()` once in dev (py-spy:
main thread in `Publisher.publish`, 0 % CPU; most likely after a participant on domain 18 vanished — a second, stray
g1 node was also alive then), so the small 50 Hz topics went best effort. Best-effort 900 KB images then arrived at only
1–2 Hz (fragments dropped; the shared FastDDS profile's shared-memory segment is smaller than a frame), so the images
went back to reliable (~10 Hz). Subscribers to the best-effort topics must use best effort too (`ros2 topic echo`
adapts; RViz / tf2 listeners need their QoS set). The hang did not recur in the rehearsals, but its root cause is not
proven — watch for a node whose log stops advancing.

## Layout
- `g1_node.py` — the one process that owns MuJoCo: AGILE's own `MuJocoSimulation` + `LeappPolicyController`
  (unchanged, physics 500 Hz, policy 50 Hz as `scripts/sim2mujoco_eval.py`), rclpy spin on a side thread for
  `/cmd_vel`, all rendering on the main thread (EGL). `--rate 0` runs unpaced; `--truth-log` writes raw free-joint
  qpos per odom stamp (H4); `--stats` writes RTF / min base height / timings on exit.
- `room.py` + `rooms/spike_room.yaml` — generic room loader: walls as segments, objects as primitives, lights, spawn →
  one scene MJCF (G1 + room + `head_rgb` camera). Phase 10 seam: feed it the clinic from 002's `layout.yaml`.
- Clinic (task 003): `sim/worlds/clinic/generate_mjcf.py` translates generate.py's own SDF of `layout.yaml` into MJCF
  (same parts, poses, colours, walls, doorways, lintels, floors, ceiling, lamps), plus `clinic_labels.json` (body name ->
  truth label). `room.build_clinic_scene()` attaches it under the prefix `clinic/` and writes `label_of_body` (body id ->
  truth label, 0 = robot / world) into `scene.json`. `/head_rgb/segmentation` still carries body ids; the adapter
  converts them.
- `adapter_g1.py` — `G1Adapter`, the Stage 1 adapter API (`agent.robot.Adapter`): `go_to` = straight legs through doorway
  centres (`geometry.route`, from truth.json rooms; no Nav2) then `turn_to(yaw)`; `photo()` -> `Frame` with the truth label
  image and `self_crop` 0 / 0; `pose()` = sim ground truth (no localization); `where()` = `agent.world.room_of`.
- `drive.py` — waypoint + absolute-heading controller on `/cmd_vel`, from the exact sim pose (no SLAM for the G1).
- `geometry.py` — pure math (quaternions, intrinsics, control laws); `tests/` — pytest for it.
- `probes/` — H1–H4 + self-view + load, each writing `sideH-*.json` to the output dir; `probes/rehearsal.sh` runs
  them all under the shared `gz.lockdir` GPU lock.

## Decisions (side-H session 1)
- **Camera mount:** `head_rgb` on `torso_link` at (0.10, 0, 0.43) m, looking level along the torso's +x. That is the
  G1's own head camera spot (Unitree URDF `d435_link` 0.0576, 0.0175, 0.4299), centred and moved forward of the face
  shell, but level instead of the D435's 47.6° down-pitch, so `look_around` sees rooms the way Stage 1's camera does.
  Lens ≈ 1.2 m above the floor. With it level the G1 never sees its own body (self-view probe: 0 rows), so
  `self_crop` = 0 / 0.
- **Turning:** measured in dev (step responses at RTF 1): below ~0.5 rad/s the policy mostly twists the pelvis and
  springs back after the command stops (0.3 rad/s for 3 s gave +6° that relaxed to +2°); at ≥ 0.6 rad/s it really
  step-turns. So `turn_to` step-turns at ≥ 0.6 rad/s until within 3°, settles 2 s at zero command, and applies short
  correction pulses (`geometry.fine_pulse`), each followed by a 2 s settle, until the settled error is < 1°.
- **Waypoints:** holonomic P control in the body frame (vx, vy) + turn to face the target beyond 0.4 m; stop at
  0.15 m (the H1 gate is 0.30 m). Reaches ~0.5 m/s, the policy's command limit.
