# Reproduce it

> **In plain words.** The pure-Python part, including all the tests that need no simulator, runs on any machine in a few
> minutes. The simulators need Linux, ROS and Gazebo (for the TurtleBot) or MuJoCo (for the G1). A full list needs a GPU for the model
> and, to be practical, a cluster. The commands below come from the public repository's README.

The repository is called `vlm-skill-agent`. Everything here assumes it is cloned and that you are in its root directory.

## 12.1 Check the tests (no ROS, no GPU)

```bash
uv sync
uv run pytest -q
```

Tests that need ROS, Gazebo, MuJoCo or a cluster skip themselves or say so in their description. This suite checks, among other things, that
the agent imports no ROS, that agent code never reads the truth file, that the interface matches its freeze, and the scorer and gate code
with their planted-bug tests.

## 12.2 Run one task on the TurtleBot 4

You need Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, `ros-jazzy-turtlebot4-description` 2.1.1, Nav2, SLAM Toolbox and a GPU for rendering.

```bash
bash sim/build.sh && source sim/env.sh       # builds the camera overlay under $BUILD_DIR (default ~/vlm_skill_agent_build)
export ROS_DOMAIN_ID=17 GZ_PARTITION=world115
setsid bash sim/launch_clinic.sh localization:=true nav2:=true > /tmp/clinic.log 2>&1 &
python3 adapters/run_task.py --task find-toolbox --planner scripted --out runs/demo \
        --script tasks/scripts/find-toolbox.yaml
sim/stop.sh
```

The scripted planner plays a hand-written solution: it exercises the body, skills and scorer without any model, as in phase 2. The files
`probes/camera_check.py` and `probes/nav_goal.py` are the first checks to run on a new machine.

## 12.3 Run it with the VLM

Serve the model with vLLM (version 0.32.0 was used). The weights are about 56 GB in BF16; one 141 GB GPU was used.

```bash
vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 1 --max-model-len 32768 \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
export VLM_BASE_URL=http://localhost:8000/v1 VLM_API_KEY=EMPTY VLM_MODEL=Qwen/Qwen3.8-27B
python3 probes/vlm_probe.py --help           # check the endpoint with one photo
python3 adapters/run_task.py --task find-toolbox --planner vlm --seed 1 --out runs/vlm_demo
```

## 12.4 Run the test list

The 100-task list used for the results is `tasks/phase8_pass2.yaml`:

```bash
python3 probes/phase3_run.py --list tasks/phase8_pass2.yaml --out runs/pass2
```

This needs the simulator and the endpoint up, and it takes hours. The cluster scripts in `hpc/` run several simulators per job; they expect
`PROJ`, `LOGIN_NODE` and the placeholders `<PERUN_ACCOUNT>` and `<PROJECT_DIR>` in the `#SBATCH` lines to be filled in for your cluster, and
`hpc/PERUN_SIM.md` describes the layout. The gates are scored by `probes/phase8_gates.py`. The sealed generator that produced the lists is not part of
the public repository, so the lists are provided as they were run.

## 12.5 Run the same agent on the G1

```bash
humanoid/setup.sh                              # pinned clones of WBC-AGILE and unitree_mujoco, a CPU-torch venv
setsid humanoid/run_g1.sh humanoid/g1_node.py --clinic sim/worlds/clinic/layout.yaml &
ROBOT=g1 python3 adapters/run_task.py --task find-toolbox --planner vlm --out runs/g1_demo
```

On a cluster the G1 stack is packed into a container image (`hpc/build_sim.sh` then `hpc/build_g1.sh`, definition `hpc/g1.def`) and run through
`hpc/g1_job.sbatch`. `humanoid/README.md` lists topics and details. The agent is unchanged: only `ROBOT=g1` selects the other adapter.

## 12.6 Rebuild this document

The text and the figures live in `docs/explained/`. The one command is in `docs/explained/README.md`.
