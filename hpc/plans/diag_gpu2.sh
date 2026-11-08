# NOTE (e7f62ee, 006 M5 round 3): since then sim_run commands run on the client cpus (the job's cpus no sim holds) and
# /proc/self Cpus_allowed_list in a plan = the client cpus; results recorded for this plan used the older layout.
# 006 M5 round 2 diagnosis, no gate verdicts. Job 143808 (64 cpus, 1 H200, dedicated cpus per sim): RTF 0.78 alone,
# 0.60 / 0.69 at N = 2, 0.55-0.62 at N = 4, while our processes used ~22 of 64 cpus. Question: is it the shared GPU?
# 2 H200, sims round-robin over them (sim_lib sim_gpu + sim_hide: job 143811 showed CUDA_VISIBLE_DEVICES does not steer
# EGL, all 4 rendered on one GPU): N = 4 = 2 per GPU, then sims 2, 3 stopped = 1 per GPU.
# Each sim pinned to its own cpus. No vLLM.
#   sbatch --time=00:25:00 --gres=gpu:2 --cpus-per-task=64 --mem=400G hpc/sim_job.sbatch hpc/plans/diag_gpu2.sh 4
D=$RUN/diag; mkdir -p "$D"
PY=/opt/venv/bin/python
# shellcheck source=hpc/plans/diag_lib.sh
source "$REPO/hpc/plans/diag_lib.sh"
{ echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; grep Cpus_allowed_list /proc/self/status; nvidia-smi topo -m
  nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv; nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv
  nvidia-smi; } > "$D/host.txt" 2>&1
for i in 0 1 2 3; do pin $i 4; done; sleep 5
snap g4 & rtf_all g4 0 1 2 3; wait
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >> "$D/summary.txt"
sim_stop 2; sim_stop 3; sleep 10
for i in 0 1; do pin $i 2; done; sleep 5
snap g2 & rtf_all g2 0 1; wait
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >> "$D/summary.txt"
