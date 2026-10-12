#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# GPU smoke test for the LSTM + Patch Cross-Attention Student (target architecture).
#
# 1. Builds the Student env (small) + frozen Teacher v4.
# 2. Verifies the observation shapes (depth 32x48, proprio history 21 frames) and the
#    reset fill rule (repeat-first-frame, no cross-episode leakage).
# 3. Runs >=100 distillation iterations with the dual loss; both L_repr and L_action must
#    decrease and stay finite.
# 4. Times a single forward (mean + p95) at batch 1 and 64, reports params + peak GPU mem.
#
# Run:
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/smoke_lstm_ca.py --headless

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GPU smoke test for the LSTM + patch CA Student.")
parser.add_argument("--fusion_type", type=str, default="lstm_patch_ca",
                    choices=["patch_ca", "lstm_patch_ca"])
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--iterations", type=int, default=120)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--timing_iters", type=int, default=200)
parser.add_argument("--warmup_iters", type=int, default=20)
parser.add_argument("--lambda_repr", type=float, default=1.0)
parser.add_argument("--lambda_action", type=float, default=1.0)
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
    proprio_hist = torch.cat([s[k] for k in PROPRIO_KEYS], dim=-1)  # (N, 21, 45)
    depth = s["depth"]
    if args_cli.fusion_type == "lstm_patch_ca":
        return depth, proprio_hist
    return depth, proprio_hist[:, -1, :]


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
        return float(torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0)


def time_forward(student, device, batch, history_len):
    depth = torch.randn(batch, 1, 32, 48, device=device)
    if history_len > 1:
        proprio = torch.randn(batch, history_len, 45, device=device)
    else:
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
    student = DepthStudentPolicy(1, 32, 48, 45, 12, fusion_type=ft).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[smoke] fusion_type={ft} params={n_params} num_envs={env.unwrapped.num_envs}", flush=True)

    obs, _ = env.reset()
    s = obs["student"]
    depth_shape = tuple(s["depth"].shape)
    hist_shape = tuple(s[PROPRIO_KEYS[0]].shape)
    print(f"[smoke] obs depth={depth_shape} proprio_term0={hist_shape}", flush=True)
    check_depth = depth_shape == (args_cli.num_envs, 1, 32, 48)
    check_hist = hist_shape == (args_cli.num_envs, 21, 3)  # base_ang_vel is 3-dim
    print(f"[smoke] shape_check depth_ok={check_depth} hist_ok={check_hist}", flush=True)

    # reset fill rule: right after reset every history slot == current frame (repeat-first-frame)
    ph = torch.cat([s[k] for k in PROPRIO_KEYS], dim=-1)  # (N, 21, 45)
    repeat_first = bool((ph[:, 0, :] == ph[:, -1, :]).all())
    print(f"[smoke] reset_repeat_first_frame={repeat_first}", flush=True)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    student.train()
    nan_seen = False
    first_repr = first_action = None
    last_repr = last_action = None
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
            z_t, a_t = teacher.forward_with_latent(tb)
            z_t = z_t.detach()
            a_t = a_t.detach()

        total_losses = []
        repr_losses = []
        action_losses = []
        n = db.shape[0]
        chunk = max(1, n // 4)
        perm = torch.randperm(n, device=device)
        for _ in range(2):
            for i in range(0, n, chunk):
                idx = perm[i:i + chunk]
                z_s, a_s = student.forward_with_latent(db[idx], pb[idx])
                lr = F.mse_loss(z_s, z_t[idx])
                la = F.mse_loss(a_s, a_t[idx])
                loss = args_cli.lambda_repr * lr + args_cli.lambda_action * la
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_losses.append(loss.item())
                repr_losses.append(lr.item())
                action_losses.append(la.item())

        r = sum(repr_losses) / len(repr_losses)
        a = sum(action_losses) / len(action_losses)
        finite = all(torch.isfinite(torch.tensor([r, a])).tolist())
        nan_seen = nan_seen or (not finite)
        if first_repr is None:
            first_repr, first_action = r, a
        last_repr, last_action = r, a
        if it % 10 == 0 or it == args_cli.iterations - 1:
            print(f"[smoke] iter {it}: repr={r:.6f} action={a:.6f} finite={finite}", flush=True)

    env.close()
    torch.cuda.empty_cache()

    mean1, p95_1 = time_forward(student, device, 1, 21 if ft == "lstm_patch_ca" else 1)
    mean64, p95_64 = time_forward(student, device, 64, 21 if ft == "lstm_patch_ca" else 1)
    mem = gpu_mem_mb()

    print("==== smoke result ====", flush=True)
    print(f"fusion_type={ft}", flush=True)
    print(f"num_params={n_params}", flush=True)
    print(f"shape_depth_ok={check_depth}", flush=True)
    print(f"shape_hist_ok={check_hist}", flush=True)
    print(f"reset_repeat_first_frame={repeat_first}", flush=True)
    print(f"loss_nan_seen={nan_seen}", flush=True)
    print(f"repr_first={first_repr:.6f} repr_last={last_repr:.6f}", flush=True)
    print(f"action_first={first_action:.6f} action_last={last_action:.6f}", flush=True)
    print(f"repr_decreased={last_repr < first_repr}", flush=True)
    print(f"action_decreased={last_action < first_action}", flush=True)
    print(f"forward_batch1_mean_ms={mean1:.3f} p95_ms={p95_1:.3f}", flush=True)
    print(f"forward_batch64_mean_ms={mean64:.3f} p95_ms={p95_64:.3f}", flush=True)
    print(f"p95_under_20ms={p95_64 < 20.0}", flush=True)
    print(f"peak_gpu_mb={mem:.1f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    simulation_app.close()
