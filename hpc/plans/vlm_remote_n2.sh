# 007 Step 0b (2026-11-05; 006 finding 15): N = 2 sims, vLLM in another GPU job (sim_job ... vlm=<serve job>), the
# 005 dev entries d01-d06 sharded over the 2 sims exactly as in 143819's N=2 dev list (k % 2: d01 d03 d05 on sim 0),
# where vLLM shared the sims' H200 and per-run sim/wall was 0.617-0.725 (median 0.663). Adopt "vLLM on its own GPU"
# only if VLM-active per-run sim/wall >= 0.7 and runs per H200-hour >= 0.75x the N=2 single-GPU layout.
#   sbatch --time=00:40:00 --cpus-per-task=64 --mem=400G hpc/sim_job.sbatch hpc/plans/vlm_remote_n2.sh 2 vlm=<serve job>
PY=/opt/venv/bin/python
ENVS=(); for i in 0 1; do ENVS+=(--shard-env "$i=$(sim_dom $i):$(sim_part $i)"); done
pin_snapshot "sims ready, idle"
date -Is > "$RUN/throughput_t0.txt"
( sleep 420; pin_snapshot "dev entries, +420 s" ) & SNAP=$!
sim_run 0 $PY probes/phase3_run.py --list tasks/phase3_dev.yaml --out "$RUN/n2r/runs" --rehearsal \
    --only d01,d02,d03,d04,d05,d06 --shards 2 "${ENVS[@]}" || echo "N2R rc=$?"
date -Is > "$RUN/throughput_t1.txt"
kill "$SNAP" 2>/dev/null
