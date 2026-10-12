#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# GPU smoke test for ONE Student fusion arm.
#
# 1. Builds the Student env (small: ``--num_envs``, default 4) + frozen Teacher v4.
# 2. Runs a few distillation iterations (pure BC MSE) to confirm shapes / no NaN/Inf and
#    that gradients flow end-to-end on GPU.
# 3. Times a single Student forward at batch 1 and batch 64 (mean + p95), reports parameter
#    count and peak GPU memory.
#
# Run once per arm (serially, only after the Teacher training has finished and the GPU is
# otherwise idle):
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/smoke_ca.py --headless --fusion_type concat
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/smoke_ca.py --headless --fusion_type cross_attention

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GPU smoke test for one Student fusion arm.")
parser.add_argument("--fusion_type", type=str, required=True, choices=["concat", "cross_attention"])
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--timing_iters", type=int, default=200)
parser.add_argument("--warmup_iters", type=int, default=20)
parser.add_argument(
    "--teacher_ckpt",
    type=str,
    default="logs/rsl_rl/unitree_go2_teacher/2026-10-10_15-00-10/model_19999.pt",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.models import DepthStudentPolicy, TeacherMLPPolicy  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

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


def extract_student(obs):
    s = obs["student"]
    return s["depth"], torch.cat([s[k] for k in PROPRIO_KEYS], dim=-1)


def load_teacher(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint has no 'actor_state_dict'; keys: {list(ckpt.keys())}")
    t = TeacherMLPPolicy(232, [512, 256, 128], 12, "elu")
    t.load_actor_state_dict(ckpt["actor_state_dict"])
    return t.to(device).eval()


def gpu_mem_mb():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2
    except Exception:  # noqa: BLE001
        return float(torch.cuda.max_memory_allocated() / 1024**2)


def time_forward(student, device, batch):
    depth = torch.randn(batch, 1, 64, 64, device=device)
    proprio = torch.randn(batch, 45, device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    student.eval()
    with torch.inference_mode():
        for _ in range(args_cli.warmup_iters):
            student(depth, proprio)
        torch.cuda.synchronize()
        lat = []
        for _ in range(args_cli.timing_iters):
            start.record()
            student(depth, proprio)
            end.record()
            torch.cuda.synchronize()
            lat.append(start.elapsed_time(end))
    lat_sorted = sorted(lat)
    mean = sum(lat) / len(lat)
    p95 = lat_sorted[int(0.95 * (len(lat) - 1))]
    return mean, p95


def main():
    ft = args_cli.fusion_type
    env_cfg = parse_env_cfg("Unitree-Go2-Depth-Student", device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    env = gym.make("Unitree-Go2-Depth-Student", cfg=env_cfg)
    device = env.unwrapped.device

    teacher = load_teacher(args_cli.teacher_ckpt, device)
    student = DepthStudentPolicy(1, 64, 64, 45, 12, fusion_type=ft).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[smoke] fusion_type={ft} params={n_params} num_envs={env.unwrapped.num_envs}", flush=True)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    obs, _ = env.reset()
    student.train()
    nan_seen = False
    for it in range(args_cli.iterations):
        depths, proprios, teacher_obss = [], [], []
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                depth, proprio = extract_student(obs)
                actions = student.act(depth, proprio)
                depths.append(depth)
                proprios.append(proprio)
                teacher_obss.append(obs["teacher"])
                obs, _, _, _, _ = env.step(actions)
        db = torch.cat(depths, 0)
        pb = torch.cat(proprios, 0)
        tb = torch.cat(teacher_obss, 0)
        with torch.inference_mode():
            teacher_mean = teacher(tb).detach()
        loss = F.mse_loss(student(db, pb), teacher_mean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        finite = bool(torch.isfinite(loss).item())
        nan_seen = nan_seen or not finite
        print(f"[smoke] iter {it}: loss={loss.item():.6f} finite={finite}", flush=True)

    env.close()
    torch.cuda.empty_cache()

    mean1, p95_1 = time_forward(student, device, 1)
    mean64, p95_64 = time_forward(student, device, 64)
    mem = gpu_mem_mb()

    print("==== smoke result ====", flush=True)
    print(f"fusion_type={ft}", flush=True)
    print(f"num_params={n_params}", flush=True)
    print(f"loss_nan_seen={nan_seen}", flush=True)
    print(f"forward_batch1_mean_ms={mean1:.3f} p95_ms={p95_1:.3f}", flush=True)
    print(f"forward_batch64_mean_ms={mean64:.3f} p95_ms={p95_64:.3f}", flush=True)
    print(f"peak_gpu_mb={mem:.1f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    simulation_app.close()
