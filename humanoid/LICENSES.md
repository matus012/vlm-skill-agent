# Third-party material used by humanoid/

Nothing below is vendored into this repo. Clones, weights and wheels live in `$BUILD_DIR/`
(`ext/`, `.venv-g1/`) and are fetched by `setup.sh` at the pinned versions.

| what | source | version / commit | licence | how we use it |
|---|---|---|---|---|
| WBC-AGILE (`agile/sim2mujoco/*`, LEAPP loader) | https://github.com/nvidia-isaac/WBC-AGILE | `f5c388ebc7c1b056de891e3edb8f47693a50d5d3` | Apache-2.0 (all but `third_party/rsl_rl/patches/`, BSD-3, unused) — `LICENCE` | imported unchanged from the clone via `PYTHONPATH` |
| `Velocity-G1-v0` policy bundle (ONNX, yaml, initial values; Git LFS) | same repo, `agile/data/policy/velocity_g1/leapp/Velocity-G1-v0/` | same commit; ONNX sha256 `4c92c5a64d1220ab02b042e77bdd69bcd2c0310590755c3dd53b5bde229d26e4` | Apache-2.0 (repo licence) | the walking policy |
| unitree_mujoco G1 model (`unitree_robots/g1/g1_29dof.xml` + meshes) | https://github.com/unitreerobotics/unitree_mujoco | `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` | BSD-3-Clause — `LICENSE` | robot MJCF; `room.py` adds the room + `head_rgb` camera into a generated scene in `$BUILD_DIR/g1_scene/` |
| MuJoCo | PyPI `mujoco` | 3.17.0 | Apache-2.0 | physics + EGL rendering |
| LEAPP | PyPI `leapp` | 0.5.2 (pin from the bundle yaml) | Apache-2.0 | policy pipeline runtime |
| ONNX Runtime | PyPI `onnxruntime` | 1.31.1 | MIT | ONNX inference |
| PyTorch (CPU) | download.pytorch.org/whl/cpu | 2.11.0+cpu | BSD-3-Clause | tensors in AGILE's sim2mujoco |
| ROS 2 Jazzy (rclpy, msgs) | apt, /opt/ros/jazzy | system | Apache-2.0 | ROS interface |
| numpy, scipy, pandas, matplotlib, seaborn, plotly, jinja2, tqdm, imageio, pyyaml, pytest | PyPI | see `setup.sh` | BSD / MIT / PSF-style permissive | imported by AGILE's package `__init__`, PNG writing, tests |
