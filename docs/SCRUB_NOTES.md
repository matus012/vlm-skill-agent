# Scrub notes

This snapshot was scrubbed of machine names, personal paths and internal workflow notes. A few files were left
byte-identical on purpose, because their exact bytes are checked: a layout file's hash is the world's `layout_sha256`
(in `truth.json` and in every run record), and a task list's sha256 is pinned in `probes/phase3_run.py` or recorded in
the runs made from it. Changing a comment there would break those checks. The words left in them are internal
workflow notes only (no names, emails, paths or keys). One line per file:

## Pinned data kept byte-identical

- `sim/worlds/clinic/layout.yaml`: "L1" (a design-review date in a comment); layout hash.
- `sim/worlds/clinic/layout_cabinet_open.yaml`: "L1" (the same comment); layout hash.
- `sim/worlds/clinic/layout_nomovables.yaml`: "L1" (two comments); layout hash.
- `tasks/dev.yaml`: "context.md" (a pointer to an internal notes file, not shipped); task-list hash in run records.
- `tasks/phase5_counted.yaml`: "L1", "deviation 17" (why the list was re-picked); counted list, sha256 pinned.
- `tasks/phase5_tasks.yaml`: "L1", "Deviation 17" (a fallback rule's origin); task file used by the counted list.
- `tasks/phase8_regress.yaml`: "deviation 20" (why the regression list exists); task-list hash in run records.
- `tasks/phase8_regress_tasks.yaml`: "deviation 20" (the same note); task file of that list.

## Vendored upstream attribution kept

- `ros_ws/src/turtlebot4_description/` (`package.xml`, `CHANGELOG.rst`, `launch/robot_description.launch.py`):
  the upstream Clearpath maintainers' names and emails, kept as Apache-2.0 attribution. Full licence text:
  `ros_ws/src/turtlebot4_description/LICENSE-APACHE-2.0`.

## Kept on purpose

- "PERUN" is the name of the GPU cluster the evaluation ran on; the `hpc/` scripts and notes keep it. No account
  names, user names or project paths of that cluster remain.
