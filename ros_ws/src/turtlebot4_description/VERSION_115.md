# turtlebot4_description overlay (project 115)

- Source: https://github.com/turtlebot/turtlebot4 tag `2.1.1`, commit `7fd29fb420e906f3aca4a904adb54b69b11c7c00`,
  directory `turtlebot4_description/` (copied as-is, incl. CMakeLists.txt and CHANGELOG.rst).
- Matches installed `ros-jazzy-turtlebot4-description 2.1.1-1noble.20261006.041007`: `diff -r` of urdf/, meshes/,
  launch/ and package.xml against `/opt/ros/jazzy/share/turtlebot4_description` is empty
  (checked 2026-11-03).
- Only change: `urdf/sensors/oakd.urdf.xacro` — `horizontal_fov`, `width`, `height` are macro params
  defaulting to 1.5708 / 640 / 480 (stock hard-coded 1.25 / 320 / 240). Depth shares the `rgbd_camera`
  sensor; near clip unchanged (0.3 m). (The project's simulated-camera decision.)
- `meshes/` is not in git (29 MB, byte-identical): `sim/build.sh` copies it from the installed 2.1.1 package
  (version checked with `dpkg-query`) before `colcon build`.
- Change 2 (task 002, 2026-11-04): `urdf/sensors/oakd.urdf.xacro` gets a second sensor `seg_camera`
  (`type="segmentation"`, semantic) on the same link `oakd_rgb_camera_frame`, no extra pose, same macro params for
  `horizontal_fov` / `width` / `height` and the same clip (0.3 / 100 m) as `rgbd_camera`; 10 Hz; gz topic
  `oakd/seg` (-> `/oakd/seg/labels_map`, label id in every channel of an rgb8 image). Instrument only (ground-truth
  visibility, sim/INTERFACES.md section 1). `rgbd_camera` unchanged. Needs the ogre2 render engine on the server
  (`gz sim --render-engine-server ogre2`), because create3's Sensors plugin asks for ogre 1.x, which has no
  segmentation camera. No `: ` in xacro comments: launch_ros yaml-parses `robot_description`.
