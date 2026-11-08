# Licences

## This project's code

MIT: see [`LICENSE`](LICENSE).

Files that start from an upstream file keep the upstream licence for that part (see the table below).

## Third-party material

### In this repository (vendored or derived)

| what | where | source | licence |
|---|---|---|---|
| `turtlebot4_description` (URDF/xacro, launch, CMake, changelog; upstream 2.1.1, commit `7fd29fb420e906f3aca4a904adb54b69b11c7c00`), with a camera-parameter change and a second segmentation sensor in `urdf/sensors/oakd.urdf.xacro` (see `VERSION_115.md`). The 29 MB `meshes/` are not in the repo: `sim/build.sh` copies them from the installed apt package. | `ros_ws/src/turtlebot4_description` | https://github.com/turtlebot/turtlebot4 (tag 2.1.1), Clearpath Robotics | Apache-2.0 (`package.xml`: "Apache 2.0"); full text in `ros_ws/src/turtlebot4_description/LICENSE-APACHE-2.0` |
| `sim/launch/spawn_115.launch.py`, `sim/launch/bridge_115.launch.py`: derived from the stock `turtlebot4_gz_bringup` launch files, with a reduced sensor load | `sim/launch/` | https://github.com/turtlebot/turtlebot4_simulator | Apache-2.0 (Clearpath Robotics header kept in the files) |

Everything else in the simulated clinic (walls, furniture, objects) is built from SDF primitives by
`sim/worlds/clinic/generate.py`: no meshes, textures or downloaded models.

### Used, not vendored (fetched or installed at set-up time)

| what | how it is used | source | licence |
|---|---|---|---|
| `irobot_create_description` (TurtleBot 4 base and dock) | spawned by the TurtleBot 4 simulation, from the apt package `ros-jazzy-turtlebot4-description` dependencies | https://github.com/iRobotEducation/create3_sim | BSD (as stated in the installed package's `package.xml`) |
| ROS 2 Jazzy, Nav2, SLAM Toolbox, ros_gz | runtime, installed from apt | https://docs.ros.org/en/jazzy/ | Apache-2.0 / BSD (per package) |
| Gazebo Harmonic | simulator, installed from apt | https://gazebosim.org | Apache-2.0 |
| unitree_mujoco G1 model (`unitree_robots/g1/g1_29dof.xml` + meshes), commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` | fetched by `humanoid/setup.sh` and `hpc/build_g1.sh`; the clinic room and camera are added to a generated scene | https://github.com/unitreerobotics/unitree_mujoco | BSD-3-Clause |
| NVIDIA WBC-AGILE, commit `f5c388ebc7c1b056de891e3edb8f47693a50d5d3`, including the `Velocity-G1-v0` LEAPP/ONNX walking policy (`agile/data/policy/velocity_g1/leapp/Velocity-G1-v0/`, Git LFS) | fetched by `humanoid/setup.sh`; imported unchanged | https://github.com/nvidia-isaac/WBC-AGILE | Apache-2.0 (most of the code and the policy bundle; the README of the repo lists BSD-3-Clause for `third_party/rsl_rl/patches/`, which is not used here) |
| MuJoCo 3.17.0, LEAPP 0.5.2, ONNX Runtime 1.31.1, PyTorch (CPU) 2.11.0 | G1 simulation and policy inference | PyPI / download.pytorch.org | Apache-2.0, Apache-2.0, MIT, BSD-3-Clause |
| vLLM v0.32.0 | serves the VLM through an OpenAI-compatible API (image `vllm/vllm-openai:v0.32.0`) | https://github.com/vllm-project/vllm | Apache-2.0 |
| Python packages in `pyproject.toml` / `uv.lock` (jsonschema, numpy, openai, pillow, pyyaml, pytest) | agent and tests | PyPI | MIT, BSD, Apache-2.0, HPND (each under its own licence) |

Details for the G1 stack are also in `humanoid/LICENSES.md`.

### Model weights (not included)

| what | source | licence |
|---|---|---|
| `Qwen/Qwen3.8-27B` (the VLM) | https://huggingface.co/Qwen/Qwen3.8-27B | Apache-2.0 (the licence field of the model card, checked 2026-11-06) |

No weights, container images, videos or camera frames are stored in this repository, except the README's demo GIF
(`media/demo.gif`, cut from this project's own showcase video).
