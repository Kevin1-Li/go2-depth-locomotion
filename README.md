# Go2 Depth-Perception Locomotion (Isaac Lab)

A Go2 locomotion research project based on [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) and Isaac Lab. This snapshot adds a privileged terrain teacher and a depth/proprioception student trained by teacher-student distillation.

## Current student policy

- Depth input: 32 × 48 RayCasterCamera depth map.
- Depth encoder: 4 × 4 patch tokens (96 tokens, 16 depth values each), positional embeddings, and cross-attention with current proprioception as the query.
- Temporal encoder: 21 frames of proprioceptive history encoded by an LSTM (hidden size 128).
- Fusion: temporal and depth-conditioned features are fused into a 128-dimensional latent; the actor receives that latent and current proprioceptive state and outputs 12 joint actions.
- Distillation: frozen teacher; representation latent MSE plus action MSE. Student training is behavior cloning, without PPO updates.

The complete source snapshot is provided as [`unitree_locomotion_pkg_source.zip`](unitree_locomotion_pkg_source.zip). Extract it to recover the original `source/`, `scripts/`, `assets/`, and `doc/` directory layout.

## Reported simulation result

The 2026-10-12 project note reports 98.4% survival over 192 evaluation episodes for the 10,000-iteration student run, with mean planar velocity error 0.117 m/s. These are Isaac Lab simulation results with generated geometric depth; they do not establish real-robot or Sim2Real performance. Check `doc/UPDATE_2026-10-12.md` inside the archive for protocol, per-terrain results, caveats, and reproduction commands.

## Environment

The training snapshot was run with Isaac Sim 5.1, Isaac Lab 2.3.1, and `rsl_rl` 5.4.2. Isaac Sim/Isaac Lab are external dependencies and are not included here. Start with the [upstream installation guide](https://github.com/unitreerobotics/unitree_rl_lab#installation), then follow the project-specific notes in the archive's `doc/UPDATE_2026-10-12.md`.

Some robot-description paths in this snapshot may still refer to the original training machine and need to be updated for your local installation. Check the extracted `source/unitree_rl_lab/unitree_rl_lab/assets/robots/` before running.

## Repository contents

- `unitree_locomotion_pkg_source.zip`: full source tree with the original directory structure, including project code, training/evaluation tools, robot assets, and project notes.

Trained checkpoints, TensorBoard logs, and raw evaluation outputs are not included in the archive.
