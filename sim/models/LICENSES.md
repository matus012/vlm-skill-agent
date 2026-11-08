# Assets used by the project's sim worlds — one line per asset

The clinic world (task 002) uses **no third-party assets**: every object, wall, floor and ceiling is an SDF primitive
(box / cylinder / sphere) composed by `sim/worlds/clinic/generate.py` and written into
`sim/worlds/clinic/generated/clinic.sdf`. No meshes, textures or Fuel models are referenced, so nothing is downloaded
at launch and nothing outside this repo's own licence applies.

Why primitives: the cached Fuel models on the local machine (`~/.gz/fuel`) carry no licence in their `model.config`, and the brief
says "no unclear licences — if unsure, use a primitive".

| asset (class in generate.py) | source | licence |
|---|---|---|
| bed, chair, toolbox, cabinet (open / closed), desk, bookshelf, plant, sofa, trolley, privacy_screen, reception_counter, water_cooler, shelf_rack, boxes_stack | own work, primitives in `sim/worlds/clinic/generate.py` | same as this repository |
| walls, floor, ceiling, door lintels (`structure_*` models) | own work, `sim/worlds/clinic/generate.py` | same as this repository |
| TurtleBot 4 + dock (spawned, not part of the world) | `ros-jazzy-turtlebot4-description` 2.1.1 / `irobot_create_description` (apt), project overlay in `ros_ws/` | Apache-2.0 (upstream packages) |

VLM recognition, judged by eye from the renders in task 002 (Phase 3 measures it): bed, chair, sofa, bookshelf, desk,
plant, water cooler, reception counter and shelf rack read clearly. Doubtful: `trolley` (reads as a generic cart),
`privacy_screen` (green panels, could read as a partition or curtain), `toolbox` (a red box with a handle; reads as a
red box or case), `cabinet` closed (a blue metal wardrobe; could read as a locker).
