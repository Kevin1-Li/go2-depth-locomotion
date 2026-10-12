#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Attention visualisation for the cross-attention Student.
#
# Loads a ``cross_attention`` Student checkpoint (auto-detected from metadata; refuses a
# ``concat`` checkpoint), runs a short headless deployment rollout and, every
# ``--sample_every`` steps, dumps PNGs that overlay the 8x8 cross-attention weights
# (B, heads, queries, 64) onto the corresponding 64x64 depth frame.
#
# These heatmaps only show WHERE the queries attend over the 8x8 spatial token grid; they
# are not a causal explanation of terrain understanding or footstep selection.
#
# Usage:
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/visualize_attention.py --headless \
#       --checkpoint logs/rsl_rl/unitree_go2_student/<run>/student_model_2999.pt \
#       --num_envs 8 --steps 200 --sample_every 50 --out_dir logs/attn_viz

import argparse
import os

import matplotlib

matplotlib.use("Agg")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Cross-attention Student attention visualisation.")
parser.add_argument("--checkpoint", type=str, required=True, help="Student checkpoint path.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Depth-Student", help="Task id.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--steps", type=int, default=200, help="Number of control steps.")
parser.add_argument("--sample_every", type=int, default=50, help="Dump cadence (steps).")
parser.add_argument("--max_envs_per_step", type=int, default=2, help="Envs to render per sample step.")
parser.add_argument("--out_dir", type=str, default="logs/attn_viz", help="Output directory.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.models import load_student_policy  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

STUDENT_PROPRIO_KEYS = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
]


def _upsample(heat: np.ndarray, size: int) -> np.ndarray:
    h, w = heat.shape
    return np.kron(heat, np.ones((size // h, size // w), dtype=heat.dtype))


def render(attn: np.ndarray, depth: np.ndarray, out_path: str, step: int, env_idx: int) -> None:
    heads, queries, tokens = attn.shape
    grid = int(round(np.sqrt(tokens)))
    heat = attn.reshape(heads, queries, grid, grid)  # (H, Q, 8, 8)

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.35)

    # Left: depth image with the mean (over heads & queries) attention overlaid.
    mean_attn = heat.mean(axis=(0, 1))  # (8, 8)
    overlay = _upsample(mean_attn, depth.shape[0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(depth, cmap="gray")
    ax0.imshow(overlay, cmap="hot", alpha=0.5)
    ax0.set_title(f"depth + mean attn (step {step}, env {env_idx})")
    ax0.axis("off")

    # Right: per-head x per-query 8x8 heatmaps.
    ax1 = fig.add_subplot(gs[0, 1])
    canvas = np.zeros((heads * grid, queries * grid))
    for h in range(heads):
        for q in range(queries):
            canvas[h * grid : (h + 1) * grid, q * grid : (q + 1) * grid] = heat[h, q]
    im = ax1.imshow(canvas, cmap="hot", interpolation="nearest")
    ax1.set_xticks(np.arange(queries) * grid + grid / 2, [f"q{q}" for q in range(queries)])
    ax1.set_yticks(np.arange(heads) * grid + grid / 2, [f"h{h}" for h in range(heads)])
    ax1.set_title("attention (rows=heads, cols=queries)")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    policy = load_student_policy(args_cli.checkpoint, device)
    if policy.fusion_type != "cross_attention":
        raise ValueError(
            f"Checkpoint fusion_type={policy.fusion_type!r}; attention visualisation requires "
            f"'cross_attention'. Refusing to run."
        )
    policy.eval()

    os.makedirs(args_cli.out_dir, exist_ok=True)
    saved: list[str] = []

    obs, _ = env.reset()
    with torch.inference_mode():
        for step in range(1, args_cli.steps + 1):
            s = obs["student"]
            proprio = torch.cat([s[k] for k in STUDENT_PROPRIO_KEYS], dim=-1)
            depth = s["depth"]
            actions, attn = policy(depth, proprio, return_attention=True)  # (B,heads,4,64)

            if step % args_cli.sample_every == 0:
                for env_idx in range(min(args_cli.max_envs_per_step, attn.shape[0])):
                    a = attn[env_idx].cpu().numpy()  # (heads, 4, 64)
                    d = depth[env_idx, 0].cpu().numpy()  # (64, 64)
                    out_path = os.path.join(args_cli.out_dir, f"attn_step{step:05d}_env{env_idx}.png")
                    render(a, d, out_path, step, env_idx)
                    saved.append(out_path)

            obs, reward, term, trunc, extras = env.step(actions)
            if step % 100 == 0:
                print(f"[step {step}/{args_cli.steps}] saved {len(saved)} PNGs", flush=True)

    print(f"[INFO] wrote {len(saved)} attention PNGs to {os.path.abspath(args_cli.out_dir)}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    simulation_app.close()
