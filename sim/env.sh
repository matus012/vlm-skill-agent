# Source (do not execute): ROS Jazzy + project overlay. Usage: source sim/env.sh
source /opt/ros/jazzy/setup.bash
# WS_LOCAL: per-machine build / run dir (hpc/PERUN_SIM.md section 1); default = $BUILD_DIR, i.e. ~/vlm_skill_agent_build
source "${WS_LOCAL:-${BUILD_DIR:-$HOME/vlm_skill_agent_build}}/ros_ws/install/setup.bash"
# Overridable so two sims can run side by side (different ROS_DOMAIN_ID + GZ_PARTITION); default 15. Keep the domain IDs
# of sims running at once >= 2 apart: fastdds_localhost.xml announces to participant IDs 0..127 (256 ports) and domains
# are 250 ports apart, so adjacent domains discover each other and share /clock (006, PERUN job 143783).
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-15}
export GZ_PARTITION=${GZ_PARTITION:-sim115}
# Localhost-only discovery. ufw (INPUT DROP) drops looped-back multicast, so SUBNET discovery is unreliable,
# and the stock LOCALHOST mode never discovered the late Nav2 nodes (~60 participants). The XML profile does
# localhost unicast discovery over 128 participant ports; SYSTEM_DEFAULT stops rmw overriding it.
export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT
export FASTRTPS_DEFAULT_PROFILES_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fastdds_localhost.xml"
# gz-transport multicast discovery is dropped by ufw as well; pin it to loopback
export GZ_IP=127.0.0.1
