# hpc/SIM_VERSIONS.md — clinic-sim image (PERUN) vs the local machine

Task 006 M1. Image `$PROJ/images/sim.sif` (`PROJ=$PROJ`), built from `hpc/sim.def`
by `hpc/build_sim.sh` on the login node (`apptainer build --fakeroot`, apptainer 1.4.5).

- **sha256** `c704832825b7103feb7a3c176cf305611d6c39721834dc84aa8b5d1e196d51ef`, 1 203 175 424 B, built
  2026-11-04 23:07:51 CET (build ~3 min).
- Base `docker://ros:jazzy` (Ubuntu 24.04, Python 3.12.3), pulled at build time; apt from packages.ros.org +
  archive.ubuntu.com on 2026-11-04. uv 0.13.6 (same as the local machine). Gazebo Sim 8.15.0 on both.
- Image-side raw list: `$PROJ/images/sim.versions.txt` (rewritten by every `build_sim.sh` run).
- local side: `dpkg-query -W` and `uv pip list` on the local machine, 2026-11-04.

**Result: every relevant apt package and every venv package has the same version on both.** The only difference is
`ros-jazzy-turtlebot4-desktop`, not installed in the image on purpose (RViz/GUI meta package; no GUI on PERUN, nothing in
the sim launch uses it). Outside the table, the local machine also has `ros-jazzy-desktop` (demos, rqt, turtlesim, teleop, joy,
pcl/image_geometry); none are referenced by the project or the launch trees it includes. `ros2 pkg list`: the image's
379 packages are all on the local machine too (the other 67 on the local machine are those desktop extras).

## apt (ROS / Gazebo / tools)
| package | local | image | |
|---|---|---|---|
| procps | 2:4.0.4-4ubuntu3.3 | 2:4.0.4-4ubuntu3.3 | same |
| python3-colcon-common-extensions | 0.3.0-100 | 0.3.0-100 | same |
| python3-numpy | 1:1.26.4+ds-6ubuntu1 | 1:1.26.4+ds-6ubuntu1 | same |
| python3-opencv | 4.6.0+dfsg-13.1ubuntu1 | 4.6.0+dfsg-13.1ubuntu1 | same |
| python3-yaml | 6.0.1-2build2 | 6.0.1-2build2 | same |
| ros-jazzy-cv-bridge | 4.1.0-1noble.20260911.143610 | 4.1.0-1noble.20260911.143610 | same |
| ros-jazzy-fastrtps | 2.14.7-1noble.20260911.035053 | 2.14.7-1noble.20260911.035053 | same |
| ros-jazzy-gz-cmake-vendor | 0.0.12-1noble.20260618.173318 | 0.0.12-1noble.20260618.173318 | same |
| ros-jazzy-gz-common-vendor | 0.0.10-1noble.20260905.033536 | 0.0.10-1noble.20260905.033536 | same |
| ros-jazzy-gz-dartsim-vendor | 0.0.3-1noble.20260225.232143 | 0.0.3-1noble.20260225.232143 | same |
| ros-jazzy-gz-fuel-tools-vendor | 0.0.6-1noble.20260905.065120 | 0.0.6-1noble.20260905.065120 | same |
| ros-jazzy-gz-gui-vendor | 0.0.5-1noble.20260905.071619 | 0.0.5-1noble.20260905.071619 | same |
| ros-jazzy-gz-math-vendor | 0.0.11-1noble.20260905.031649 | 0.0.11-1noble.20260905.031649 | same |
| ros-jazzy-gz-msgs-vendor | 0.0.8-1noble.20260905.033532 | 0.0.8-1noble.20260905.033532 | same |
| ros-jazzy-gz-ogre-next-vendor | 0.0.5-1noble.20260225.232146 | 0.0.5-1noble.20260225.232146 | same |
| ros-jazzy-gz-physics-vendor | 0.0.9-1noble.20260905.065116 | 0.0.9-1noble.20260905.065116 | same |
| ros-jazzy-gz-plugin-vendor | 0.0.5-1noble.20260820.145930 | 0.0.5-1noble.20260820.145930 | same |
| ros-jazzy-gz-rendering-vendor | 0.0.7-1noble.20260905.065113 | 0.0.7-1noble.20260905.065113 | same |
| ros-jazzy-gz-sensors-vendor | 0.0.6-1noble.20260905.071631 | 0.0.6-1noble.20260905.071631 | same |
| ros-jazzy-gz-sim-vendor | 0.0.13-1noble.20260905.073557 | 0.0.13-1noble.20260905.073557 | same |
| ros-jazzy-gz-tools-vendor | 0.0.8-1noble.20260820.143431 | 0.0.8-1noble.20260820.143431 | same |
| ros-jazzy-gz-transport-vendor | 0.0.9-1noble.20260905.065154 | 0.0.9-1noble.20260905.065154 | same |
| ros-jazzy-gz-utils-vendor | 0.0.5-1noble.20260618.173807 | 0.0.5-1noble.20260618.173807 | same |
| ros-jazzy-irobot-create-common-bringup | 3.0.4-1noble.20261006.040506 | 3.0.4-1noble.20261006.040506 | same |
| ros-jazzy-irobot-create-control | 3.0.4-1noble.20261006.035448 | 3.0.4-1noble.20261006.035448 | same |
| ros-jazzy-irobot-create-description | 3.0.4-1noble.20261006.040347 | 3.0.4-1noble.20261006.040347 | same |
| ros-jazzy-irobot-create-gz-bringup | 3.0.4-1noble.20261006.041857 | 3.0.4-1noble.20261006.041857 | same |
| ros-jazzy-irobot-create-gz-plugins | 3.0.4-1noble.20261006.043654 | 3.0.4-1noble.20261006.043654 | same |
| ros-jazzy-irobot-create-gz-toolbox | 3.0.4-1noble.20260915.183633 | 3.0.4-1noble.20260915.183633 | same |
| ros-jazzy-irobot-create-msgs | 3.0.0-2noble.20260911.114505 | 3.0.0-2noble.20260911.114505 | same |
| ros-jazzy-irobot-create-nodes | 3.0.4-1noble.20260915.181312 | 3.0.4-1noble.20260915.181312 | same |
| ros-jazzy-irobot-create-toolbox | 3.0.4-1noble.20260911.134511 | 3.0.4-1noble.20260911.134511 | same |
| ros-jazzy-nav2-bringup | 1.3.13-1noble.20261006.042503 | 1.3.13-1noble.20261006.042503 | same |
| ros-jazzy-navigation2 | 1.3.13-1noble.20260921.033624 | 1.3.13-1noble.20260921.033624 | same |
| ros-jazzy-rmw-fastrtps-cpp | 8.4.4-1noble.20260911.100940 | 8.4.4-1noble.20260911.100940 | same |
| ros-jazzy-ros-base | 0.11.0-1noble.20261006.035407 | 0.11.0-1noble.20261006.035407 | same |
| ros-jazzy-ros-core | 0.11.0-1noble.20261006.035319 | 0.11.0-1noble.20261006.035319 | same |
| ros-jazzy-ros-gz | 1.0.24-1noble.20261006.041805 | 1.0.24-1noble.20261006.041805 | same |
| ros-jazzy-ros-gz-bridge | 1.0.24-1noble.20260920.234644 | 1.0.24-1noble.20260920.234644 | same |
| ros-jazzy-ros-gz-image | 1.0.24-1noble.20260921.023911 | 1.0.24-1noble.20260921.023911 | same |
| ros-jazzy-ros-gz-interfaces | 1.0.24-1noble.20260911.110630 | 1.0.24-1noble.20260911.110630 | same |
| ros-jazzy-ros-gz-sim | 1.0.24-1noble.20261006.035056 | 1.0.24-1noble.20261006.035056 | same |
| ros-jazzy-ros-gz-sim-demos | 1.0.24-1noble.20261006.040806 | 1.0.24-1noble.20261006.040806 | same |
| ros-jazzy-slam-toolbox | 2.8.5-1noble.20260915.204151 | 2.8.5-1noble.20260915.204151 | same |
| ros-jazzy-turtlebot4-description | 2.1.1-1noble.20261006.041007 | 2.1.1-1noble.20261006.041007 | same |
| ros-jazzy-turtlebot4-desktop | 2.0.1-1noble.20261006.044258 | MISSING | **DIFF** |
| ros-jazzy-turtlebot4-gz-bringup | 2.0.2-1noble.20261006.045255 | 2.0.2-1noble.20261006.045255 | same |
| ros-jazzy-turtlebot4-gz-gui-plugins | 2.0.2-1noble.20260905.074348 | 2.0.2-1noble.20260905.074348 | same |
| ros-jazzy-turtlebot4-gz-toolbox | 2.0.2-1noble.20260912.162937 | 2.0.2-1noble.20260912.162937 | same |
| ros-jazzy-turtlebot4-msgs | 2.1.1-1noble.20260911.095127 | 2.1.1-1noble.20260911.095127 | same |
| ros-jazzy-turtlebot4-navigation | 2.1.1-1noble.20261006.044110 | 2.1.1-1noble.20261006.044110 | same |
| ros-jazzy-turtlebot4-node | 2.1.1-1noble.20260912.162937 | 2.1.1-1noble.20260912.162937 | same |
| ros-jazzy-turtlebot4-simulator | 2.0.2-1noble.20261006.050425 | 2.0.2-1noble.20261006.050425 | same |
| ros-jazzy-turtlebot4-viz | 2.0.1-1noble.20261006.043247 | 2.0.1-1noble.20261006.043247 | same |
| ros-jazzy-xacro | 2.1.1-1noble.20260519.011123 | 2.1.1-1noble.20260519.011123 | same |

## /opt/venv (uv sync --frozen from uv.lock; the local machine = `$BUILD_DIR/.venv`)
Both are venvs with `--system-site-packages` on `/usr/bin/python3` 3.12.3 (rclpy, cv2 4.6.0 from apt).

| package | local | image | |
|---|---|---|---|
| annotated-types | 0.8.0 | 0.8.0 | same |
| anyio | 4.16.0 | 4.16.0 | same |
| attrs | 26.1.0 | 26.1.0 | same |
| h11 | 0.16.0 | 0.16.0 | same |
| httpcore2 | 2.16.0 | 2.16.0 | same |
| httpx2 | 2.16.0 | 2.16.0 | same |
| idna | 3.20 | 3.20 | same |
| iniconfig | 2.3.1 | 2.3.1 | same |
| jiter | 0.18.0 | 0.18.0 | same |
| jsonschema | 4.26.0 | 4.26.0 | same |
| jsonschema-specifications | 2025.9.1 | 2025.9.1 | same |
| numpy | 1.26.4 | 1.26.4 | same |
| openai | 3.44.0 | 3.44.0 | same |
| packaging | 26.3 | 26.3 | same |
| pillow | 13.0.0 | 13.0.0 | same |
| pluggy | 1.7.0 | 1.7.0 | same |
| pydantic | 2.14.2 | 2.14.2 | same |
| pydantic-core | 2.50.2 | 2.50.2 | same |
| pygments | 2.21.0 | 2.21.0 | same |
| pytest | 9.2.0 | 9.2.0 | same |
| pyyaml | 6.0.3 | 6.0.3 | same |
| referencing | 0.37.0 | 0.37.0 | same |
| rpds-py | 2026.9.1 | 2026.9.1 | same |
| sniffio | 1.3.1 | 1.3.1 | same |
| truststore | 0.10.5 | 0.10.5 | same |
| typing-extensions | 4.16.0 | 4.16.0 | same |
| typing-inspection | 0.4.4 | 0.4.4 | same |
