# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Diagnosis of the depth-perception Student's staircase failure.
#
# Goal (task A): quantify how far the Student's action deviates from the frozen
# privileged Teacher's action, *broken down by terrain type*, and dump visual depth
# samples so we can judge whether the stairs are even discernible in the 64x64 depth
# image and how large the near-field blind spot is.
#
# Self-contained (does NOT import rsl_rl), mirrors train_distill.py / eval_three_way.py.
#
# The environment is the Student *play* config, which still points at the OLD (ANYmal-
# scale) big-stair ``PLAY_TERRAIN_CFG`` -- exactly the config under which the Student
# fails, so this diagnoses the failure mode before the geometry fix is applied.
#
# Protocol:
#   * task  : Unitree-Go2-Depth-Student (play_env_cfg_entry_point -> StudentRobotPlayEnvCfg)
#   * seed 42, headless, num_envs 32
#   * the Student acts (deployment roll-out); on every state we ALSO query the frozen
#     Teacher (232-D privileged observation) and accumulate per-terrain action MSE
#   * >=1500 control steps
#   * depth samples: every 100 steps, for the two stair terrains (pyramid_stairs /
#     pyramid_stairs_inv) dump the 1x64x64 depth obs as PNG (imshow + colorbar) plus
#     per-image stats (min/mean/max, near <0.5 m fraction, far-clipped fraction)
#
# Usage:
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/diag_student_stairs.py --headless \
#       --num_envs 32 --seed 42 --steps 1500

import argparse
import os

import matplotlib

matplotlib.use("Agg")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Student staircase failure diagnosis.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Depth-Student", help="Task id.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--steps", type=int, default=1500, help="Number of control steps.")
parser.add_argument("--sample_every", type=int, default=100, help="Depth sample cadence (steps).")
parser.add_argument("--out_dir", type=str, default="logs/diag_depth", help="Directory for depth PNGs.")
parser.add_argument(
    "--student_ckpt",
    type=str,
    default="logs/rsl_rl/unitree_go2_student/2026-10-09_01-00-08_v1/student_model_2999.pt",
    help="Student checkpoint.",
)
parser.add_argument(
    "--teacher_ckpt",
    type=str,
    default="logs/rsl_rl/unitree_go2_teacher/2026-10-07_15-40-51/model_29999.pt",
    help="Teacher checkpoint.",
)
parser.add_argument(
    "--samples_per_type",
    type=int,
    default=2,
    help="Depth images to dump per stair terrain per sample step.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.models import DepthStudentPolicy, TeacherMLPPolicy  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402
from unitree_rl_lab.tasks.locomotion.robots.go2.teacher_env_cfg import PLAY_TERRAIN_CFG  # noqa: E402

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

TERRAIN_SHORT = {
    "pyramid_stairs": "stairs_down",
    "pyramid_stairs_inv": "stairs_up",
    "boxes": "boxes",
    "hf_pyramid_slope": "slope",
    "random_rough": "rough",
}

DEPTH_FAR_CLIP = 10.0  # metres; the student depth obs is normalised by this.


def terrain_names() -> list[str]:
    """Column -> sub-terrain name mapping (identical to eval_three_way.py)."""
    names = list(PLAY_TERRAIN_CFG.sub_terrains.keys())
    proportions = np.array([c.proportion for c in PLAY_TERRAIN_CFG.sub_terrains.values()], dtype=np.float64)
    proportions = proportions / proportions.sum()
    num_cols = PLAY_TERRAIN_CFG.num_cols
    mapping = []
    for col in range(num_cols):
        sub = int(np.min(np.where(col / num_cols + 0.001 < np.cumsum(proportions))[0]))
        mapping.append(names[sub])
    return mapping


def load_student(path: str, device: str) -> DepthStudentPolicy:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    student = DepthStudentPolicy(depth_channels=1, depth_height=64, depth_width=64, proprio_dim=45, action_dim=12)
    student.load_state_dict(ckpt["student_state_dict"], strict=True)
    student.to(device).eval()
    return student


def load_teacher(path: str, device: str) -> TeacherMLPPolicy:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    teacher = TeacherMLPPolicy(input_dim=232, hidden_dims=[512, 256, 128], action_dim=12, activation="elu")
    teacher.load_actor_state_dict(ckpt["actor_state_dict"])
    teacher.to(device).eval()
    return teacher


def depth_stats(depth_norm: torch.Tensor) -> dict:
    """Depth statistics in metres (depth_norm in [0, 1] = [0, DEPTH_FAR_CLIP] m)."""
    depth_m = depth_norm * DEPTH_FAR_CLIP
    near_frac = (depth_m < 0.5).float().mean().item()
    far_frac = (depth_norm > 0.999).float().mean().item()
    return {
        "min_m": float(depth_m.min().item()),
        "mean_m": float(depth_m.mean().item()),
        "max_m": float(depth_m.max().item()),
        "near_lt_0p5m_frac": near_frac,
        "far_clipped_frac": far_frac,
    }


def save_depth_png(depth_norm: torch.Tensor, out_dir: str, step: int, terrain_short: str, env_idx: int, status: str):
    depth_m = (depth_norm * DEPTH_FAR_CLIP).cpu().numpy()
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    im = ax.imshow(depth_m, cmap="turbo", vmin=0.0, vmax=DEPTH_FAR_CLIP, aspect="auto")
    fig.colorbar(im, ax=ax, label="depth (m)")
    ax.set_title(f"step {step} | {terrain_short} | env {env_idx} | {status}")
    ax.set_xlabel("pixel col (width)")
    ax.set_ylabel("pixel row (height)")
    fname = os.path.join(out_dir, f"depth_step{step:05d}_{terrain_short}_env{env_idx}_{status}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    student = load_student(args_cli.student_ckpt, device)
    teacher = load_teacher(args_cli.teacher_ckpt, device)
    print(f"[INFO] student={args_cli.student_ckpt}", flush=True)
    print(f"[INFO] teacher={args_cli.teacher_ckpt}", flush=True)
    print(f"[INFO] num_envs={env.unwrapped.num_envs} device={device}", flush=True)

    terrain_types = env.unwrapped.scene.terrain.terrain_types
    col_names = terrain_names()
    n_terrains = len(col_names)
    print(f"[INFO] terrain columns: {col_names}", flush=True)
    print(f"[INFO] terrain_types per env: {terrain_types.tolist()}", flush=True)

    reward_manager = env.unwrapped.reward_manager
    track_idx = reward_manager._term_names.index("track_lin_vel_xy")

    # per-terrain accumulators
    mse_sum = [0.0] * n_terrains
    mse_count = [0] * n_terrains
    step_count = [0] * n_terrains
    reward_sum = [0.0] * n_terrains
    track_sum = [0.0] * n_terrains

    # depth sample bookkeeping
    os.makedirs(args_cli.out_dir, exist_ok=True)
    saved_pngs: list[str] = []
    depth_stats_log: list[dict] = []

    stair_types = [0, 1]  # pyramid_stairs, pyramid_stairs_inv

    obs, _ = env.reset()
    last_term = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=device)
    last_trunc = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=device)

    t_start = time.time()
    with torch.inference_mode():
        for step in range(1, args_cli.steps + 1):
            # ---- compute both actions on the current state ----
            s = obs["student"]
            proprio = torch.cat([s[k] for k in STUDENT_PROPRIO_KEYS], dim=-1)
            depth = s["depth"]
            student_a = student(depth, proprio)
            teacher_a = teacher(obs["teacher"])

            mse = ((student_a - teacher_a) ** 2).mean(dim=-1)  # (num_envs,)

            for t in range(n_terrains):
                mask = terrain_types == t
                cnt = int(mask.sum().item())
                if cnt:
                    mse_sum[t] += float(mse[mask].sum().item())
                    mse_count[t] += cnt

            # ---- depth sampling (post-step obs, using the *current* depth) ----
            if step % args_cli.sample_every == 0:
                for t in stair_types:
                    idxs = (terrain_types == t).nonzero(as_tuple=False).flatten()
                    short = TERRAIN_SHORT.get(col_names[t], col_names[t])
                    picked = idxs[: args_cli.samples_per_type]
                    for env_idx in picked:
                        env_idx_i = int(env_idx.item())
                        d = obs["student"]["depth"][env_idx_i, 0]  # (64, 64)
                        if last_term[env_idx_i]:
                            status = "dead"
                        elif last_trunc[env_idx_i]:
                            status = "timeout"
                        else:
                            status = "alive"
                        st = depth_stats(d)
                        fname = save_depth_png(d, args_cli.out_dir, step, short, env_idx_i, status)
                        saved_pngs.append(fname)
                        depth_stats_log.append(
                            {"step": step, "terrain": short, "env": env_idx_i, "status": status, **st}
                        )

            # ---- step the environment with the student (deployment scenario) ----
            obs, reward, term, trunc, extras = env.step(student_a)
            last_term = term
            last_trunc = trunc

            for t in range(n_terrains):
                mask = terrain_types == t
                cnt = int(mask.sum().item())
                if cnt:
                    step_count[t] += cnt
                    reward_sum[t] += float(reward[mask].sum().item())
                    track_sum[t] += float(reward_manager._step_reward[mask, track_idx].sum().item())

            if step % 500 == 0:
                print(f"[step {step}/{args_cli.steps}] elapsed {time.time() - t_start:.1f}s", flush=True)

    elapsed = time.time() - t_start

    # ---- report ----
    print("\n==== Per-terrain student-vs-teacher action MSE (mean over env-steps) ====", flush=True)
    header = "| terrain      | MSE (x1e-2) | env-steps | mean reward | track_lin_vel_xy |"
    print(header, flush=True)
    print("|" + "---|" * 5, flush=True)
    table_rows = []
    for t in range(n_terrains):
        short = TERRAIN_SHORT.get(col_names[t], col_names[t])
        mse_mean = (mse_sum[t] / mse_count[t]) if mse_count[t] else float("nan")
        r_mean = reward_sum[t] / step_count[t] if step_count[t] else float("nan")
        tr_mean = track_sum[t] / step_count[t] if step_count[t] else float("nan")
        row = f"| {short:<12} | {mse_mean * 100:10.4f} | {step_count[t]:>9} | {r_mean:11.2f} | {tr_mean:16.3f} |"
        print(row, flush=True)
        table_rows.append(
            {
                "terrain": short,
                "terrain_name": col_names[t],
                "action_mse": mse_mean,
                "env_steps": step_count[t],
                "mean_reward": r_mean,
                "track_lin_vel_xy": tr_mean,
            }
        )

    # ---- depth stats summary ----
    print("\n==== Depth sample statistics (stairs envs) ====", flush=True)
    for entry in depth_stats_log:
        print(
            f"[step {entry['step']:>5}] {entry['terrain']:<11} env {entry['env']:<2} {entry['status']:<7} "
            f"min={entry['min_m']:.2f}m mean={entry['mean_m']:.2f}m max={entry['max_m']:.2f}m "
            f"near<0.5m={entry['near_lt_0p5m_frac']:.3f} far_clipped={entry['far_clipped_frac']:.3f}",
            flush=True,
        )

    print(f"\n[INFO] saved {len(saved_pngs)} depth PNGs to {os.path.abspath(args_cli.out_dir)}", flush=True)

    # ---- persist results ----
    result = {
        "seed": args_cli.seed,
        "num_envs": env.unwrapped.num_envs,
        "steps": args_cli.steps,
        "terrain_columns": col_names,
        "per_terrain": table_rows,
        "depth_samples": depth_stats_log,
        "elapsed_s": elapsed,
    }
    out_json = os.path.join(args_cli.out_dir, "diag_student_stairs_result.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] wrote {os.path.abspath(out_json)}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    simulation_app.close()
