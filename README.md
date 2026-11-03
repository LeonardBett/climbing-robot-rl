# Reinforcement learning for the Microban climbing robot (Rhoban) - Rock Climbing Robots hackathon 2026

This project was my first experience with Reinforcement Learning.  
I set up a GPU training environment (mjlab, on my RTX 5070) for the hackathon's climbing robot, tested the simulation
and spotted its flaws (lost contacts, limbs passing through the wall, the holds or the other leg), then created and
trained my own tasks using what I learned during the hackathon, up to a policy that climbs the whole N1 wall in
simulation.  
I relied on an AI (Claude) to assemble this environment from three existing repositories (robot model, BAM servo
model, official simulator) that did not work together as-is, to fix the collision flaws I had spotted, to learn how
PPO and the structure of a task work from an example.

https://github.com/user-attachments/assets/44066fdb-e204-49ff-a39f-2f81f943e0cf

*The trained `Climb-N1Route-Microban` policy climbing the whole N1 wall (real-time video, about 20 s).*  
The policy climbs faster than the real servos allow (~0.4 s per move): it exploits the simulation and would need
speed penalties and a check in the official simulator before going to the real robot.

## Repositories used

| Repository | What we take from it |
|---|---|
| [Rhoban/microban_model](https://github.com/Rhoban/microban_model) | the MJCF model of the climbing robot (`robot_climb.xml`, ball-shaped hands), uv dependency |
| [Rhoban/bam](https://github.com/Rhoban/bam) | the XL330 servos simulated on GPU (`bam.mjlab`), uv dependency, same commit as the official simulator |
| [rock-climbing-robots/simulator](https://github.com/rock-climbing-robots/simulator) | the official hackathon simulator: walls, hold geometry and hanging posture, read as data files (`simulator/` submodule) |

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

| Task | Goal | Result |
|---|---|---|
| `Climb-Hang-Microban` (Claude's example) | stay hanging from the 4 starting holds | works |
| `Climb-LeftFoot-Microban` (Claude's example) | left foot from (21, 2) to (18, 7), up and to the left, the other 3 limbs staying hooked | works |
| `Climb-BothFeet-Microban` (my own work) | both feet climb up one rung (up-left and up-right), both hands staying hooked | works (~200 iterations) |
| `Climb-N1-Microban` (my own work) | climb N1 freely: any hold of the next rung, rewards on height records | **failed**, kept as a lesson |
| `Climb-N1Route-Microban` (my own work) | climb the whole N1 wall following a hard-coded route, one limb at a time | **climbs the whole wall** (~1000 iterations) |

The hold directly above a foot, (21, 7), is out of reach: the feet climb diagonally (zigzag).

```bash
# watch a task without a policy
uv run play Climb-N1Route-Microban --agent zero --viewer viser --num-envs 4

# train (4096 robots: with collision meshes on every hold, more robots no longer increases the throughput)
uv run train Climb-N1Route-Microban --env.scene.num-envs 4096

# resume training from a checkpoint
uv run train Climb-N1Route-Microban --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run <run folder> --agent.load-checkpoint model_<N>.pt

# watch the result of the training
uv run play Climb-N1Route-Microban --checkpoint-file models/climb_n1_route/model_<N>.pt --viewer viser --num-envs 1
```

## What I learned from the N1 tasks

**`Climb-N1-Microban` (failed).** The robot was free to choose any hold of the next rung, and was rewarded for its
height records. Three runs, three ways for the robot to cheat the rewards:
1. Guidance by proximity to the nearest higher hold: the right hand stayed stuck *under* an unreachable hold and
   collected the reward at every step.
2. Guidance by distance records (no more farming): nothing was rewarded for staying hooked anymore, so falling cost
   nothing and the robot let go of everything.
3. With a fall penalty: the robot held on, but only took the easy points (the right hand) and never moved its feet.

**`Climb-N1Route-Microban` (works).** Same recipe as `BothFeet`, chained: a hard-coded route of moves on N1 (a
pattern over 2 rungs, repeated up to the top), one active limb at a time, known to the policy as an observation.
- Rewards: the 3 other limbs staying in their holds (at every step), Gaussian guidance of the active limb towards
  its target (a narrow one for precision, a wide one to guide from far away), and a one-time bonus when the move is
  validated (ball held 0.1 s in the target hold).
- Terminations: one of the 3 other limbs more than 3 cm from its hold, or the trunk 5 cm below its record.
- Training: 4 moves learned in ~500 iterations, then the left hand got stuck (too far from its target for the narrow
  Gaussian): resumed from iteration 500 with the wide Gaussian added, and the robot climbed the whole wall at ~1000.

General lessons: watch the episode terminations and the viewer, not only the rewards; a reward given at every step
can be farmed, a reward given only for progress removes the reason to hold on; and a fixed, reachable target is far
easier to learn than a free choice.

## Organization

| File | Role |
|---|---|
| `src/climbing_rl/wall.py` | reads the wall (simulator JSON), socket positions |
| `src/climbing_rl/robot.py` | climbing MJCF, box collisions, BAM servos, hanging pose |
| `src/climbing_rl/scene.py` | adds the wall, the holds and the floor to the scene |
| `src/climbing_rl/tasks/mdp.py` | shared functions and fixed-target tasks (Hang, LeftFoot, BothFeet) |
| `src/climbing_rl/tasks/records_mdp.py` | `Climb-N1` task: height records (failed attempt) |
| `src/climbing_rl/tasks/route_mdp.py` | `Climb-N1Route` task: hard-coded route, active limb, rewards and terminations |
| `src/climbing_rl/tasks/climb_env_cfg.py` | full task configurations + PPO settings |
| `src/climbing_rl/tasks/__init__.py` | task registration |
| `src/climbing_rl/check_scene.py` | scene check without training |

The first four files describe the physical world and rarely change anymore. My work on the tasks is done in
`climb_env_cfg.py` and the `*mdp.py` files.

## Fidelity to the real robot

- Servos: BAM, the m6 model of the XL330 identified on a test bench by Rhoban (same settings as the official
  simulator, with a little randomization).
- Collisions: for GPU speed, the 269 body meshes are replaced by boxes measured on those meshes, which collide with
  the wall, the holds, the floor and the other limbs. The holds keep their full collision meshes (only those close
  to the moving limb for the one-move tasks, all of them for the N1 tasks).
- Known differences: 4 ms time step instead of 2 ms, MuJoCo Warp instead of CPU MuJoCo. A policy must therefore be
  re-checked in the official simulator before going to the real robot.
- Known limit of the `Climb-N1Route` policy: it moves too fast (~0.4 s per move) and its training ended with a high
  exploration noise. Before any real-robot test, it would need a short fine-tuning with a joint velocity penalty
  (`joint_vel_l2`), a stronger `action_rate` penalty and a lower entropy coefficient.

## Versions

mjlab 1.3 + MuJoCo 3.7 + Warp 1.12, imposed by BAM (`better-actuator-models[mjlab]`). This project therefore cannot
import `climbing_sim` (MuJoCo ≥ 3.11): we read its data files instead. `wandb` is pinned to 0.22.3 (recent versions
break rsl_rl 5.0.1).

README written with the help of Claude.
