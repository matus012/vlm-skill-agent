# NOTE (e7f62ee, 006 M5 round 3): since then sim_run commands run on the client cpus (the job's cpus no sim holds) and
# /proc/self Cpus_allowed_list in a plan = the client cpus; results recorded for this plan used the older layout.
# 006 M5 round 2 diagnosis, no gate verdicts. Job 143806 (32 cpus, no vLLM): RTF 0.44-0.45 per sim at N = 4, 0.79 for
# sim 0 alone; GPU busy only ~38 % at N = 4 (186 MiB per sim, NVIDIA EGL), no gz thread at 100 %. Question: is it the
# job's CPU share? With 64 cpus: N = 4 unpinned, N = 4 each sim pinned to its own 16 cpus (taskset -a on all its
# processes; behaviour unchanged), N = 2 pinned 32 each, N = 1. No vLLM (143786 with vLLM: the same 0.46).
#   sbatch --time=00:25:00 --cpus-per-task=64 --mem=400G hpc/sim_job.sbatch hpc/plans/diag_scale.sh 4
D=$RUN/diag; mkdir -p "$D"
PY=/opt/venv/bin/python
{ grep Cpus_allowed_list /proc/self/status; echo "nproc $(nproc)"; lscpu | grep -E "Model name|MHz|NUMA node[0-9]"; nvidia-smi topo -m; } > "$D/host.txt" 2>&1
# shellcheck source=hpc/plans/diag_lib.sh
source "$REPO/hpc/plans/diag_lib.sh"
snap a4 & rtf_all a4 0 1 2 3; wait
for i in 0 1 2 3; do pin $i 4; done; sleep 5
snap p4 & rtf_all p4 0 1 2 3; wait
sim_stop 2; sim_stop 3; sleep 10
for i in 0 1; do pin $i 2; done; sleep 5
snap p2 & rtf_all p2 0 1; wait
sim_stop 1; sleep 10
pin 0 1; sleep 5
snap p1 & rtf_all p1 0; wait
