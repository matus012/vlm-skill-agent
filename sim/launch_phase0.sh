#!/usr/bin/env bash
# Phase 0a sim: headless Gazebo (server only, no GUI) + TurtleBot 4 (project camera overlay)
# + AMCL on the stock warehouse map + Nav2. Equivalent to
#   ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:=warehouse localization:=true nav2:=true
# except the gz GUI is not started (turtlebot4's sim.launch.py hard-codes gz_args, so -s cannot be passed through).
# Usage: sim/launch_phase0.sh   (foreground; stop with sim/stop.sh)
set -eo pipefail  # no -u: ROS setup.bash reads unset vars
source "$(dirname "$0")/env.sh"
WORLD=warehouse
share() { ros2 pkg prefix "$1"; }
# same GZ_SIM_RESOURCE_PATH as turtlebot4_gz_bringup/launch/sim.launch.py (overlay description first via ament)
export GZ_SIM_RESOURCE_PATH="$(share turtlebot4_gz_bringup)/share/turtlebot4_gz_bringup/worlds:$(share irobot_create_gz_bringup)/share/irobot_create_gz_bringup/worlds:$(share turtlebot4_description)/share:$(share irobot_create_description)/share"

trap 'kill 0' EXIT
# The robot model loads the Sensors system with ogre (1.x), which renders through GLX on $DISPLAY;
# PRIME offload puts that on the RTX 4060 instead of the Intel iGPU (Mesa).
# ogre2: the overlay carries a segmentation camera (002), which ogre 1.x (create3's Sensors default) cannot create.
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia gz sim -s -r --headless-rendering --render-engine-server ogre2 -v 3 "$WORLD.sdf" &
# /clock bridge as in sim.launch.py
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock &
ros2 launch turtlebot4_gz_bringup turtlebot4_spawn.launch.py world:="$WORLD" \
  localization:=true nav2:=true \
  map:="$(share turtlebot4_navigation)/share/turtlebot4_navigation/maps/$WORLD.yaml" &
# AMCL starts without a pose (Nav2 then waits forever for map->odom). The robot spawns at world (0,0,0);
# seed AMCL there (stock warehouse map: map origin == world origin, proven against GT by probes/nav_goal.py).
ros2 topic pub --once -w 1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: map}, pose: {pose: {orientation: {w: 1.0}}, covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.07]}}" &
wait
