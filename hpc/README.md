## VLM link (Phase 0b/0c)

Needs Pritunl up on the local machine and (H200 hours approved) the image `$PROJ/images/vllm.sif` + weights `$PROJ/models/Qwen3.8-27B`.
Run from `<repo>`. `$PROJ=$PROJ`.

```bash
# 1. up: submits serve_vllm.sbatch (gpu_short, 1 H200, wall default 02:00:00), waits for the ready file,
#    opens the ssh -L forward, writes $BUILD_DIR/vlm_link.env (0600: BASE_URL, JOBID, FWD_PID, API_KEY,
#    KEYTMP). The endpoint needs the per-job key (see "API key" below); vlm_up checks /v1/models answers 200 with it
#    and 401 without it.
#    Flags: --wall HH:MM:SS (default 02:00:00) --port LOCALPORT (default 8000) --job JOBID (attach to a running job).
hpc/vlm_up.sh                       # or: hpc/vlm_up.sh --wall 01:30:00 --port 8000   /   hpc/vlm_up.sh --job <JOBID>
source $BUILD_DIR/vlm_link.env; export VLM_API_KEY=$API_KEY   # probes read the key from env
ssh -o BatchMode=yes $LOGIN_NODE "grep -E 'healthy|MiB' $PROJ/logs/vlm115_${JOBID}.out"   # load time + GPU memory

# 2. G3 (0b): 3 of 3 valid, thinking on and off; any 640x480 photo (lane S frame when it exists)
uv run probes/vlm_probe.py --image <frame.png> --base-url "$BASE_URL" --n 3 --require 3 --thinking on  --out <evidence>/G3-think-on.jsonl
uv run probes/vlm_probe.py --image <frame.png> --base-url "$BASE_URL" --n 3 --require 3 --thinking off --out <evidence>/G3-think-off.jsonl

# 3. G4 (0c): 10 calls through the forward (the forward IS the base URL above), >= 9 valid; median/p90 are printed
uv run probes/vlm_probe.py --image <frame.png> --base-url "$BASE_URL" --n 10 --require 9 --thinking <on|off> --out <evidence>/G4.jsonl

# 4. down: kills the forward and scancels the job. Do this the moment 0b+0c are measured.
hpc/vlm_down.sh
ssh -o BatchMode=yes $LOGIN_NODE "squeue -h --me"   # empty = job gone (sacct is dead)
```

Log the job in `hpc/JOBS.md` (one row: date UTC, job id, `gpu_short`, "0b+0c vLLM serve", wall = from job start to
scancel, GPU-h = wall hours x 1, CPU-h = wall x 16, result).
The sbatch never uses `--export`; the wall is passed as `sbatch --time=`.

## API key (task 002; design notes: vLLM endpoint)
- `vlm_up.sh` makes a fresh key per job (`openssl rand -hex 16`) and writes it over ssh **stdin** into a 0600 file
  `$PROJ/runs/serve/pending_<utc>_<pid>.key`; sbatch gets only that path (positional arg; never `--export`).
- The job moves it to `$PROJ/runs/serve/<jobid>.key` (0600, checked) and passes it as `APPTAINERENV_VLLM_API_KEY`
  -> `VLLM_API_KEY` inside the container, which `vllm serve` treats like `--api-key` (v0.32.0
  `vllm/entrypoints/serve/middleware/register.py`) without putting it in `ps` on a shared node.
- vLLM v0.32.0 guards only `/v1 /v2 /inference /cohere` (`middleware/authenticate.py`): `/health` needs no key, so
  the job's health poll is unchanged. Clients send `Authorization: Bearer <key>` (OpenAI client `api_key=`).
- Clients: `probes/vlm_probe.py --api-key` (default env `VLM_API_KEY`, else `EMPTY`); `probes/g9_readable.py`
  (default env `VLM_API_KEY`, else `API_KEY` from the link env file). Avoid `--api-key <key>` on the command line.
- `vlm_down.sh` removes `<jobid>.key` and the pending file on PERUN together with the scancel. `vlm_up.sh --job`
  attaches by reading `<jobid>.key` over ssh.
