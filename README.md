# Reinforcement learning for the Microban climbing robot (Rhoban) - Rock Climbing Robots hackathon 2026

This project was my first experience with Reinforcement Learning.  
I set up a GPU training environment (mjlab, on my RTX 5070) for the hackathon's climbing robot, tested the simulation
and spotted its flaws (lost contacts, limbs passing through the wall, the holds or the other leg), then created and
trained my own tasks using what I learned during the hackathon.  
I relied on an AI (Claude) to assemble this environment from three existing repositories (robot model, BAM servo
model, official simulator) that did not work together as-is, to fix the collision flaws I had spotted, and to learn
how PPO and the structure of a task work from an example.

https://github.com/user-attachments/assets/de18ed70-e726-46a8-8a15-fd353b5d9e58

## Repositories used

| Repository | What we take from it |
|---|---|
| [Rhoban/microban_model](https://github.com/Rhoban/microban_model) | the MJCF model of the climbing robot (`robot_climb.xml`, ball-shaped hands), uv dependency |
| [Rhoban/bam](https://github.com/Rhoban/bam) | the XL330 servos simulated on GPU (`bam.mjlab`), uv dependency, same commit as the official simulator |
| [rock-climbing-robots/simulator](https://github.com/rock-climbing-robots/simulator) | the official hackathon simulator: walls, hold geometry and hanging posture, read as data files (`simulator/` folder of the team repository) |

In addition: [mujocolab/mjlab](https://github.com/mujocolab/mjlab) (the RL framework, installed by BAM) and
[Rhoban/mjlab_microban](https://github.com/Rhoban/mjlab_microban), from which the task structure and the PPO
settings were taken (its robot is the old walking Microban, without balls).

## Installation

You need an NVIDIA GPU (MuJoCo Warp only works with CUDA) and Git LFS (robot meshes).
The official simulator is included as a submodule (we read the walls, the hold geometry and the hanging posture from
it): clone with `--recursive`.

```bash
sudo apt install -y git-lfs && git lfs install
git clone --recursive git@github.com:LeonardBett/climbing-robot-rl.git
cd climbing-robot-rl
uv sync --python 3.12
uv run check-scene --num-envs 64 --seconds 3
```

Already cloned without `--recursive`: `git submodule update --init`.

`check-scene` checks the starting pose (IK error, no contact between limbs), then simulates a few seconds with a zero
action: all 4 balls should be hooked in 100% of the environments, with no `nefc overflow`.
Use `--task <task>` to test another task.

## Tasks

| Task | Goal |
|---|---|
| `Climb-Hang-Microban` (Claude's example) | stay hanging from the 4 starting holds |
| `Climb-LeftFoot-Microban` (Claude's example) | left foot from (21, 2) to (18, 7), up and to the left, the other 3 limbs staying hooked |
| `Climb-BothFeet-Microban` (my own work) | both feet climb up one rung (up-left and up-right), both hands staying hooked |

The hold directly above a foot, (21, 7), is out of reach: the feet climb diagonally (zigzag).

```bash
# watch a task without a policy
uv run play Climb-BothFeet-Microban --agent zero --viewer viser --num-envs 4

# train (6144 robots fit in the 12 GB of an RTX 5070)
uv run train Climb-BothFeet-Microban --env.scene.num-envs 6144

# resume training from a checkpoint
uv run train Climb-BothFeet-Microban --env.scene.num-envs 6144 \
  --agent.resume True --agent.load-run <run folder> --agent.load-checkpoint model_<N>.pt

# watch the result of the training (both feet climbing one rung)
uv run play Climb-BothFeet-Microban --checkpoint-file models/climb_both_feet/model_200.pt --viewer viser --num-envs 4
```

## Organization

| File | Role |
|---|---|
| `src/climbing_rl/wall.py` | reads the wall (simulator JSON), socket positions |
| `src/climbing_rl/robot.py` | climbing MJCF, box collisions, BAM servos, hanging pose |
| `src/climbing_rl/scene.py` | adds the wall, the holds and the floor to the scene |
| `src/climbing_rl/tasks/mdp.py` | climbing-specific functions: observations, rewards, terminations |
| `src/climbing_rl/tasks/climb_env_cfg.py` | full task configuration + PPO settings |
| `src/climbing_rl/tasks/__init__.py` | task registration |
| `src/climbing_rl/check_scene.py` | scene check without training |

The first four files describe the physical world and rarely change anymore. My work on the tasks is done in
`climb_env_cfg.py` and `mdp.py`.

## Fidelity to the real robot

- Servos: BAM, the m6 model of the XL330 identified on a test bench by Rhoban (same settings as the official
  simulator, with a little randomization).
- Collisions: for GPU speed, the 269 body meshes are replaced by boxes measured on those meshes, which collide with
  the wall, the holds, the floor and the other limbs. Holds close to the moving limb keep their full collision meshes.
- Known differences: 4 ms time step instead of 2 ms, MuJoCo Warp instead of CPU MuJoCo. A policy must therefore be
  re-checked in the official simulator before going to the real robot.

## Versions

mjlab 1.3 + MuJoCo 3.7 + Warp 1.12, imposed by BAM (`better-actuator-models[mjlab]`). This project therefore cannot
import `climbing_sim` (MuJoCo ≥ 3.11): we read its data files instead. `wandb` is pinned to 0.22.3 (recent versions
break rsl_rl 5.0.1).

README written with the help of Claude.
