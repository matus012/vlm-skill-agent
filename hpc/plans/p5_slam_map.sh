# 008 (2026-11-05): the Phase 5+ Nav2 map comes from a SLAM Toolbox tour, as in 002 (probes/map_tour.py: 21 waypoints
# through the hall and every room, then slam_toolbox save_map), of the movables-free world (generated_nomovables: the
# frozen v2 layout minus toolbox_red_1 + chair_orange_1..4), so the map holds no movable object. Map frame == world frame
# (SLAM starts at the spawn pose, world origin, yaw 0). The saved map is fetched to the local machine, committed as
# sim/maps/clinic_static.{yaml,pgm} and tagged; G0 then runs on it.
#   sbatch --time=00:45:00 --cpus-per-task=32 --mem=200G hpc/sim_job.sbatch hpc/plans/p5_slam_map.sh 1 slam w0=nomovables
PY=/opt/venv/bin/python
mkdir -p "$RUN/slam"
sim_run 0 $PY probes/map_tour.py --out "$RUN/slam" --save "$RUN/slam/clinic_static" || echo "map tour rc=$?"
ls -la "$RUN/slam" > "$RUN/slam/ls.txt" 2>&1
