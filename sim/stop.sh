#!/usr/bin/env bash
# Stop everything sim/launch_phase0.sh started (anchored patterns so the caller's own shell never matches).
pkill -f '^bash .*sim/launch_phase0\.sh'
pkill -f '^gz sim'
pkill -f '^/usr/bin/python3 /opt/ros/jazzy/bin/ros2'
pkill -f '^/opt/ros/jazzy/lib/'
sleep 3
pkill -9 -f '^gz sim'; pkill -9 -f '^/opt/ros/jazzy/lib/'
pkill -f "^/usr/bin/python3 -c from ros2cli.daemon"  # ros2 daemon
# SIGKILLed FastDDS processes leave locked /dev/shm ports; the next launch then fails service discovery
source /opt/ros/jazzy/setup.bash; fastdds shm clean >/dev/null
