#!/usr/bin/env bash
# Build the project colcon overlay; build/install/log live off NTFS in $WS_LOCAL (default $BUILD_DIR).
# The overlay's meshes/ (29 MB, byte-identical to upstream 2.1.1) are not in git: copied here from the installed
# ros-jazzy-turtlebot4-description, after checking it is exactly the 2.1.1 the overlay was copied from.
set -eo pipefail  # no -u: ROS setup.bash reads unset vars
source /opt/ros/jazzy/setup.bash
L="${WS_LOCAL:-${BUILD_DIR:-$HOME/vlm_skill_agent_build}}/ros_ws"  # WS_LOCAL: hpc/PERUN_SIM.md section 1
cd "$(dirname "$0")/.."
ver=$(dpkg-query -W -f='${Version}' ros-jazzy-turtlebot4-description)
[[ $ver == 2.1.1-* ]] || { echo "need ros-jazzy-turtlebot4-description 2.1.1, found $ver"; exit 1; }
mkdir -p ros_ws/src/turtlebot4_description/meshes
cp -u /opt/ros/jazzy/share/turtlebot4_description/meshes/*.dae ros_ws/src/turtlebot4_description/meshes/
colcon --log-base "$L/log" build --base-paths ros_ws/src --build-base "$L/build" --install-base "$L/install" \
    --allow-overriding turtlebot4_description
