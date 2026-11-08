#!/usr/bin/env bash
# Task 002 camera-instrument dev sim: headless Gazebo with the closed labelled test room sim/worlds/test_seg
# + TurtleBot 4 (project overlay incl. the segmentation camera), AMCL on a map generated from the
# world SDF (test_seg_map.*: map == world), initial pose from AMCL params; no Nav2.
# The server renders with ogre2 (--render-engine-server): create3.urdf.xacro loads the Sensors system with ogre 1.x,
# which has no segmentation camera; the CLI flag overrides that SDF setting. world:= makes the stock bridge use
# /world/test_seg/... topics.
# Same pattern as sim/launch_phase0.sh. Run it under setsid and stop it with kill -- -<pgid> (INTERFACES section 6):
#   export ROS_DOMAIN_ID=16 GZ_PARTITION=cam115; setsid bash sim/launch_test_seg.sh > log 2>&1 &
set -eo pipefail  # no -u: ROS setup.bash reads unset vars
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/env.sh"
share() { ros2 pkg prefix "$1"; }
export GZ_SIM_RESOURCE_PATH="$here/worlds/test_seg:$(share turtlebot4_gz_bringup)/share/turtlebot4_gz_bringup/worlds:$(share irobot_create_gz_bringup)/share/irobot_create_gz_bringup/worlds:$(share turtlebot4_description)/share:$(share irobot_create_description)/share"

trap 'kill 0' EXIT
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia gz sim -s -r --headless-rendering --render-engine-server ogre2 -v 3 test_seg.sdf &
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock &
ros2 launch turtlebot4_gz_bringup turtlebot4_spawn.launch.py world:=test_seg localization:=false nav2:=false &
ros2 launch turtlebot4_navigation localization.launch.py use_sim_time:=true \
  map:="$here/worlds/test_seg/test_seg_map.yaml" params:="$here/worlds/test_seg/localization.yaml" &
wait
