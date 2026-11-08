# 013: the showcase-video rehearsal of ONE counted test entry (p8p2c001, world closed) on the G1 with full joint logging
# (g1_node --qpos-log via the `qpos` job arg). NOT the counted run: labelled "re-run for this video". Needs tasks/phase8_video.yaml
# (probes/make_video_list.py). One sim; vLLM startup ~5-8 min + one run ~2-3 min.
#   cd $PROJ/repo && sbatch --time=00:40:00 --cpus-per-task=32 --mem=128G hpc/g1_job.sbatch hpc/plans/g1_video.sh 1 vlm qpos
# Result: $RUN/video/p8p2c001/ (run.json with robot g1, ... + qpos.jsonl). ROBOT=g1 comes from sim_vars, as for every G1 plan.
pin_snapshot "sims ready, idle"
sim_run 0 /opt/venv/bin/python probes/phase3_run.py --list tasks/phase8_video.yaml --out "$RUN/video" --rehearsal --shards 1 \
    --shard-env "0=$(sim_dom 0):$(sim_part 0):$(sim_world 0)" || echo "video run rc=$?"
cp "$RUN/s0/qpos.jsonl" "$RUN/video/p8p2c001/qpos.jsonl" || echo "qpos copy failed"
