# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher -> Student behaviour-cloning (MSE) distillation for the Go2 depth-perception task.

The converged privileged Teacher (a frozen MLP that consumes a 232-D privileged observation)
is used purely as an action labeler. A depth-perception Student (CNN depth encoder + proprio
branch) is trained on-policy with pure behaviour-cloning: the Student acts in the environment
to collect states, the Teacher labels each state, and the Student regresses the Teacher's
deterministic action via MSE.

This script is self-contained: it does not rely on rsl-rl's runner/algorithm classes (the
installed rsl-rl is 2.3.4 which has no Distillation/CNN model), only on Isaac Lab for the
environment.

Usage:
  # Teacher validity gate check (must pass before launching distillation)
  python.sh scripts/rsl_rl/train_distill.py --headless --task Unitree-Go2-Depth-Student --num_envs 16 --gate

  # distillation training (background, see run instructions)
  python.sh scripts/rsl_rl/train_distill.py --headless --task Unitree-Go2-Depth-Student --num_envs 32
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teacher-Student distillation training for Go2 depth perception.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Depth-Student", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=3000, help="Number of distillation iterations.")
parser.add_argument("--num_steps_per_env", type=int, default=120, help="Steps per env per iteration.")
parser.add_argument("--num_learning_epochs", type=int, default=2, help="Updates per collected batch.")
parser.add_argument("--num_mini_batches", type=int, default=4, help="Mini-batches per update.")
parser.add_argument("--learning_rate", type=float, default=1.0e-3, help="Student learning rate.")
parser.add_argument("--save_interval", type=int, default=100, help="Iterations between checkpoints.")
parser.add_argument("--experiment_name", type=str, default="unitree_go2_student", help="Experiment folder name.")
parser.add_argument("--run_name", type=str, default="", help="Run name suffix.")
parser.add_argument(
    "--fusion_type",
    type=str,
    default="concat",
    choices=["concat", "cross_attention", "patch_ca", "lstm_patch_ca"],
    help="Student depth/proprio fusion: 'concat' (original CNN+MLP concat), "
         "'cross_attention' (legacy 4-query spatial-token CA), 'patch_ca' (single-query "
         "patch CA, no LSTM), or 'lstm_patch_ca' (21-frame LSTM + patch CA, target).",
)
parser.add_argument("--lambda_repr", type=float, default=1.0, help="Weight on the representation (latent) distillation loss.")
parser.add_argument("--lambda_action", type=float, default=1.0, help="Weight on the action distillation loss.")
parser.add_argument(
    "--teacher_ckpt",
    type=str,
    default="logs/rsl_rl/unitree_go2_teacher/2026-10-10_15-00-10/model_19999.pt",
    help="Path to the frozen Teacher checkpoint (rsl-rl 5.x actor_state_dict layout).",
)
parser.add_argument("--gate", action="store_true", default=False, help="Run Teacher validity gate check and exit.")
parser.add_argument("--gate_steps", type=int, default=300, help="Number of steps for the gate check.")
parser.add_argument("--mem_track", action="store_true", default=False, help="Track total GPU memory via pynvml.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[STEP] AppLauncher ready", flush=True)

import os
import time
import math
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.models import DepthStudentPolicy, TeacherMLPPolicy
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

PROPRIO_KEYS = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
]


def extract_student(obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the ``student`` observation group into (depth, proprio).

    With ``history_length=21`` the proprio terms are ``(num_envs, 21, dim)``; they are
    concatenated along the last dim to ``(num_envs, 21, 45)``. For non-LSTM fusions only the
    current frame (last history entry) is kept, matching the ``patch_ca``/``concat`` API.
    """
    student = obs["student"]
    proprio_hist = torch.cat([student[k] for k in PROPRIO_KEYS], dim=-1)
    depth = student["depth"]
    if args_cli.fusion_type == "lstm_patch_ca":
        return depth, proprio_hist
    return depth, proprio_hist[:, -1, :]


class MemTracker:
    """Tracks the peak total GPU memory usage via NVML (fallback: torch allocations)."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.peak_total = None
        self._nvml = None
        if enabled:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml = pynvml
            except Exception as exc:  # noqa: BLE001
                print(f"[MemTracker] pynvml unavailable ({exc}); falling back to torch metrics.")

    def sample(self) -> dict:
        if self._nvml is not None:
            info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            total_mb = info.used / 1024**2
            self.peak_total = max(self.peak_total or 0, total_mb)
            return {"total_mb": total_mb, "peak_mb": self.peak_total}
        return {
            "total_mb": None,
            "peak_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        }


def load_teacher(path: str, device: str) -> TeacherMLPPolicy:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "actor_state_dict" in ckpt:
        actor = ckpt["actor_state_dict"]
    elif "model_state_dict" in ckpt:
        raise ValueError("Checkpoint is rsl-rl 2.x (model_state_dict); expected rsl-rl 5.x actor_state_dict layout.")
    else:
        raise KeyError(f"Unknown checkpoint layout; keys: {list(ckpt.keys())}")
    teacher = TeacherMLPPolicy(input_dim=232, hidden_dims=[512, 256, 128], action_dim=12, activation="elu")
    teacher.load_actor_state_dict(actor)
    teacher.to(device).eval()
    return teacher


def build_student(device: str) -> DepthStudentPolicy:
    student = DepthStudentPolicy(
        depth_channels=1,
        depth_height=32,
        depth_width=48,
        proprio_dim=45,
        action_dim=12,
        fusion_type=args_cli.fusion_type,
    )
    student.to(device)
    return student


def run_gate(env, teacher: TeacherMLPPolicy, gate_steps: int) -> dict:
    """Run the frozen Teacher as the actor in the Student env and report tracking metrics."""
    reward_manager = env.unwrapped.reward_manager
    track_idx = reward_manager._term_names.index("track_lin_vel_xy")

    obs, _ = env.reset()
    teacher_obs = obs["teacher"]

    total_rewards = []
    track_rewards = []
    terminated = 0
    with torch.inference_mode():
        for _ in range(gate_steps):
            actions = teacher.act(teacher_obs)
            obs, reward, term, trunc, extras = env.step(actions)
            teacher_obs = obs["teacher"]
            total_rewards.append(reward.mean().item())
            track_rewards.append(reward_manager._step_reward[:, track_idx].mean().item())
            terminated += int((term | trunc).sum().item())

    stats = {
        "mean_total_reward": sum(total_rewards) / len(total_rewards),
        "mean_track_lin_vel_xy": sum(track_rewards) / len(track_rewards),
        "terminated_steps": terminated,
        "num_envs": env.unwrapped.num_envs,
        "gate_steps": gate_steps,
    }
    return stats


def train(env, teacher: TeacherMLPPolicy, student: DepthStudentPolicy, device: str):
    num_envs = env.unwrapped.num_envs
    num_steps = args_cli.num_steps_per_env
    optimizer = torch.optim.Adam(student.parameters(), lr=args_cli.learning_rate)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name:
        log_dir += f"_{args_cli.run_name}"
    log_dir = os.path.join(log_root, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
    print(f"[INFO] Logging to {log_dir}")

    mem = MemTracker(args_cli.mem_track)

    obs, _ = env.reset()
    student.train()

    reward_manager = env.unwrapped.reward_manager
    track_idx = reward_manager._term_names.index("track_lin_vel_xy")

    for it in range(args_cli.max_iterations):
        start = time.time()
        depths = []
        proprios = []
        teacher_obss = []
        rewbuffer = []
        trackbuffer = []

        with torch.inference_mode():
            for _ in range(num_steps):
                depth, proprio = extract_student(obs)
                actions = student.act(depth, proprio)
                teacher_obss.append(obs["teacher"])
                depths.append(depth)
                proprios.append(proprio)
                obs, reward, term, trunc, extras = env.step(actions)
                rewbuffer.append(reward.mean().item())
                trackbuffer.append(reward_manager._step_reward[:, track_idx].mean().item())

        depth_batch = torch.cat(depths, dim=0)
        proprio_batch = torch.cat(proprios, dim=0)
        teacher_batch = torch.cat(teacher_obss, dim=0)

        dual = args_cli.fusion_type in ("patch_ca", "lstm_patch_ca")
        with torch.inference_mode():
            if dual:
                teacher_z, teacher_a = teacher.forward_with_latent(teacher_batch)
                teacher_z = teacher_z.detach()
                teacher_a = teacher_a.detach()
            else:
                teacher_a = teacher(teacher_batch).detach()

        total_samples = depth_batch.shape[0]
        perm = torch.randperm(total_samples, device=device)
        chunk = max(1, total_samples // args_cli.num_mini_batches)

        repr_losses = []
        action_losses = []
        total_losses = []
        for _ in range(args_cli.num_learning_epochs):
            for i in range(0, total_samples, chunk):
                idx = perm[i : i + chunk]
                if dual:
                    z_s, a_s = student.forward_with_latent(depth_batch[idx], proprio_batch[idx])
                    loss_repr = F.mse_loss(z_s, teacher_z[idx])
                    loss_action = F.mse_loss(a_s, teacher_a[idx])
                    loss = args_cli.lambda_repr * loss_repr + args_cli.lambda_action * loss_action
                    repr_losses.append(loss_repr.item())
                else:
                    a_s = student(depth_batch[idx], proprio_batch[idx])
                    loss_action = F.mse_loss(a_s, teacher_a[idx])
                    loss = loss_action
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                action_losses.append(loss_action.item())
                total_losses.append(loss.item())

        behavior_loss = sum(total_losses) / len(total_losses)
        repr_loss = (sum(repr_losses) / len(repr_losses)) if repr_losses else float("nan")
        action_loss = sum(action_losses) / len(action_losses)
        mean_reward = sum(rewbuffer) / len(rewbuffer)
        mean_track = sum(trackbuffer) / len(trackbuffer)

        mem_info = mem.sample()
        elapsed = time.time() - start

        writer.add_scalar("Loss/behavior_mse", behavior_loss, it)
        writer.add_scalar("Loss/repr_mse", repr_loss, it)
        writer.add_scalar("Loss/action_mse", action_loss, it)
        writer.add_scalar("Train/mean_reward", mean_reward, it)
        writer.add_scalar("Train/track_lin_vel_xy", mean_track, it)
        writer.add_scalar("Perf/iteration_time", elapsed, it)

        print(
            f"[iter {it:5d}/{args_cli.max_iterations}] repr_mse={repr_loss:.6f} "
            f"action_mse={action_loss:.6f} total={behavior_loss:.6f} "
            f"track_lin_vel_xy={mean_track:5.2f} mean_reward={mean_reward:6.2f} "
            f"fps={int(num_envs * num_steps / elapsed)} "
            f"gpu_total={mem_info['total_mb'] if mem_info['total_mb'] else 'n/a'}MiB "
            f"peak={mem_info['peak_mb']:.0f}MiB"
        )

        if (it + 1) % args_cli.save_interval == 0:
            torch.save(
                {
                    "student_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter": it,
                    "fusion_type": student.fusion_type,
                    "model_config": student.to_config(),
                    "lambda_repr": args_cli.lambda_repr,
                    "lambda_action": args_cli.lambda_action,
                },
                os.path.join(log_dir, f"student_model_{it}.pt"),
            )

    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": args_cli.max_iterations - 1,
            "fusion_type": student.fusion_type,
            "model_config": student.to_config(),
            "lambda_repr": args_cli.lambda_repr,
            "lambda_action": args_cli.lambda_action,
        },
        os.path.join(log_dir, f"student_model_{args_cli.max_iterations - 1}.pt"),
    )
    writer.close()
    return log_dir


def main():
    print("[STEP] main() started", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    print("[STEP] env_cfg parsed", flush=True)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[STEP] env created", flush=True)

    device = env.unwrapped.device
    teacher = load_teacher(args_cli.teacher_ckpt, device)
    print(f"[INFO] Loaded Teacher from {args_cli.teacher_ckpt}", flush=True)
    print(f"[INFO] num_envs={env.unwrapped.num_envs}, device={device}")

    obs, _ = env.reset()
    print("[INFO] observation groups:")
    for gname, gval in obs.items():
        if isinstance(gval, dict):
            shapes = {k: tuple(v.shape) for k, v in gval.items()}
            print(f"  '{gname}' (dict): {shapes}")
        else:
            print(f"  '{gname}': {tuple(gval.shape)}")

    if args_cli.gate:
        stats = run_gate(env, teacher, args_cli.gate_steps)
        print("==== Teacher validity gate ====")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        passed = stats["mean_track_lin_vel_xy"] > 1.0
        print(f"==== GATE {'PASSED' if passed else 'FAILED'} (mean_track_lin_vel_xy > 1.0) ====")
        env.close()
        return

    student = build_student(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[INFO] Student fusion_type={student.fusion_type} params={n_params}")
    log_dir = train(env, teacher, student, device)
    print(f"[INFO] Training complete. Logs at {log_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    simulation_app.close()
