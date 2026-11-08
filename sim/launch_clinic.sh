#!/usr/bin/env bash
# Phase 1 clinic sim: headless Gazebo (server only) with sim/worlds/clinic/generated/clinic.sdf + TurtleBot 4
# (project overlay camera) spawned in the hall. Same pattern as sim/launch_phase0.sh.
# Nav2 / localization / SLAM are OFF by default. Args (key:=value, any order):
#   x:= y:= yaw:=        spawn pose in the world frame (default hall 0 0 0)
#   localization:=true   AMCL on map:=<yaml> (default sim/maps/clinic.yaml); initial pose = the spawn pose,
#                        set through AMCL's own set_initial_pose / initial_pose params (INTERFACES section 5)
#   amcl:=<yaml>         flat {param: value} merged into the stock AMCL params (default config/robots/turtlebot4_amcl.yaml)
#   nav2:=true  slam:=true  map:=<yaml>
#   worlddir:=<dir>      dir holding clinic.sdf (default sim/worlds/clinic/generated; e.g. a cabinet-open variant)
# Run under setsid and stop with kill -- -<pgid> (INTERFACES section 6), holding the gz lock:
#   export ROS_DOMAIN_ID=17 GZ_PARTITION=world115; setsid bash sim/launch_clinic.sh > log 2>&1 &
set -eo pipefail  # no -u: ROS setup.bash reads unset vars
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/env.sh"
declare -A A=([x]=0.0 [y]=0.0 [yaw]=0.0 [localization]=false [nav2]=false [slam]=false [map]="$here/maps/clinic.yaml" [worlddir]="$here/worlds/clinic/generated" [amcl]="$here/../config/robots/turtlebot4_amcl.yaml")
for kv in "$@"; do
  k="${kv%%:=*}"; [[ "$kv" == *":="* && -v "A[$k]" ]] || { echo "unknown arg: $kv" >&2; exit 2; }
  A[$k]="${kv#*:=}"
done
share() { ros2 pkg prefix "$1"; }
export GZ_SIM_RESOURCE_PATH="${A[worlddir]}:$(share turtlebot4_gz_bringup)/share/turtlebot4_gz_bringup/worlds:$(share irobot_create_gz_bringup)/share/irobot_create_gz_bringup/worlds:$(share turtlebot4_description)/share:$(share irobot_create_description)/share"

extra=()
if [[ "${A[localization]}" == true ]]; then
  [[ -f "${A[map]}" ]] || { echo "map not found: ${A[map]}" >&2; exit 2; }
  run="${WS_LOCAL:-${BUILD_DIR:-$HOME/vlm_skill_agent_build}}/run"; mkdir -p "$run"
  params="$run/localization_clinic_${GZ_PARTITION}_${ROS_DOMAIN_ID}.yaml"  # unique per job + sim (PERUN_SIM.md §1)
  [[ -f "${A[amcl]}" ]] || { echo "amcl params not found: ${A[amcl]}" >&2; exit 2; }
  python3 - "$(share turtlebot4_navigation)/share/turtlebot4_navigation/config/localization.yaml" "$params" \
    "${A[amcl]}" "${A[x]}" "${A[y]}" "${A[yaw]}" <<'EOF'
import sys, yaml
src, dst, amcl, x, y, yaw = sys.argv[1], sys.argv[2], sys.argv[3], *map(float, sys.argv[4:7])
p = yaml.safe_load(open(src))
over = yaml.safe_load(open(amcl)) or {}
ros = p["amcl"]["ros__parameters"]
unknown = sorted(set(over) - set(ros))
assert not unknown, f"amcl:= keys not in the stock AMCL params: {unknown}"  # a typo would be silently ignored by AMCL
ros.update(over)
print(f"amcl overrides from {amcl}: {over}")
ros.update(set_initial_pose=True, initial_pose={"x": x, "y": y, "z": 0.0, "yaw": yaw})
yaml.safe_dump(p, open(dst, "w"), sort_keys=True)
EOF
  extra+=(params:="$params")
fi
if [[ "${A[nav2]}" == true ]]; then
  # stock nav2.yaml with xy_goal_tolerance 0.10 m (0.25 m ends past the 0.30 m place bound) and planner tolerance 0.0
  # (0.5 m snaps a goal inside a wall to a reachable cell; a recorded deviation), INTERFACES section 5
  run="${WS_LOCAL:-${BUILD_DIR:-$HOME/vlm_skill_agent_build}}/run"; mkdir -p "$run"
  nav2_params="$run/nav2_clinic_${GZ_PARTITION}_${ROS_DOMAIN_ID}.yaml"
  python3 - "$(share turtlebot4_navigation)/share/turtlebot4_navigation/config/nav2.yaml" "$nav2_params" <<'EOF'
import sys, yaml
src, dst = sys.argv[1:3]
p = yaml.safe_load(open(src))
p["controller_server"]["ros__parameters"]["general_goal_checker"]["xy_goal_tolerance"] = 0.10
p["planner_server"]["ros__parameters"]["GridBased"]["tolerance"] = 0.0
yaml.safe_dump(p, open(dst, "w"), sort_keys=True)
EOF
  extra+=(params_file:="$nav2_params")
fi

trap 'kill 0' EXIT
# --render-engine-server ogre2 overrides create3's Sensors <render_engine>ogre</render_engine>: ogre 1.x has no
# segmentation camera and gz segfaults at spawn. Renders through GLX on $DISPLAY; PRIME offload -> RTX 4060.
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia gz sim -s -r --headless-rendering --render-engine-server ogre2 -v 3 clinic.sdf &
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock &
# world:=clinic: the stock bridge otherwise listens on /world/warehouse/... and /oakd/rgb/preview stays silent
# spawn_115 = the stock turtlebot4_spawn.launch.py with a trimmed bridge (no point cloud / HMI; sim/launch/bridge_115.launch.py)
ros2 launch "$here/launch/spawn_115.launch.py" world:=clinic \
  x:="${A[x]}" y:="${A[y]}" yaw:="${A[yaw]}" \
  localization:="${A[localization]}" nav2:="${A[nav2]}" slam:="${A[slam]}" map:="${A[map]}" "${extra[@]}" &
# The stock spawn always adds standard_dock next to the robot; it has no seg label (G7) and is not part of the clinic
# (INTERFACES section 1). Park it outside the building once it exists, so the robot starts undocked in the hall.
# A failure here kills the whole launch (set -e in the subshell + kill 0 below): no silent dock in view.
(
  for _ in $(seq 120); do gz model --list 2>/dev/null | grep -q -- "- standard_dock$" && break; sleep 1; done
  reply="$(gz service -s /world/clinic/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 5000 \
    --req 'name: "standard_dock", position: {x: 20, y: 20, z: 0}')"
  [[ "$reply" == *"data: true"* ]] || { echo "DOCK MOVE FAILED: '$reply'" >&2; kill 0; }
  echo "dock moved out of the building: $reply"
) &
wait
