# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Three-way same-condition quantitative evaluation of the Go2 policies:
#   * blind   : blind-walk baseline (45-D proprio) on the complex play terrain
#   * teacher : privileged Teacher (45-D proprio + 187-D height_scan) on the same terrain
#   * student : depth-perception Student (45-D proprio + 64x64 depth) on the same terrain
#
# All three are replayed headless on the IDENTICAL shared ``PLAY_TERRAIN_CFG`` (via each
# task's ``play_env_cfg_entry_point``), so terrain, spawn levels and command distribution
# match across models. Protocol (fixed for fairness / reproducibility):
#   * seed = 42, num_envs = 64, deterministic mean actions (no sampling noise)
#   * command ranges = ``limit_ranges`` (full-scale random), corruption off (play mode)
#   * 3 rounds x max_episode_length (1000 steps = 20 s), unified ``env.reset()`` per round
#
# Each invocation evaluates ONE model (one Isaac Lab environment per process; a single
# process cannot reliably spin up several ManagerBasedRLEnv instances back-to-back).
# Results are written to a per-model JSON, then merged into ``logs/eval_three_way_results.md``
# by ``--merge``.
#
# Usage (run serially, one GPU task at a time):
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/eval_three_way.py --headless --model blind
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/eval_three_way.py --headless --model teacher
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/eval_three_way.py --headless --model student
#   ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/eval_three_way.py --merge

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Three-way same-condition Go2 policy evaluation.")
parser.add_argument("--model", type=str, default=None, choices=["blind", "teacher", "student"],
                    help="Which model to evaluate (required unless --merge).")
parser.add_argument("--merge", action="store_true", default=False,
                    help="Merge the per-model JSON results into the markdown table and exit.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--rounds", type=int, default=3, help="Number of full episode rounds per model.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--max_gpu_gb", type=float, default=6.5, help="GPU memory red line (informational).")
parser.add_argument("--out_dir", type=str, default="logs", help="Output directory for result JSON/MD.")
parser.add_argument("--tag", type=str, default="", help="Suffix appended to result filenames (e.g. 'v2').")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Override the default checkpoint path for the model being evaluated.")
parser.add_argument("--fusion_type", type=str, default=None,
                    choices=["concat", "cross_attention", "patch_ca", "lstm_patch_ca"],
                    help="Optional fusion_type to validate against the checkpoint metadata (student only).")
parser.add_argument("--teacher_ckpt", type=str,
                    default="logs/rsl_rl/unitree_go2_teacher/2026-10-10_15-00-10/model_19999.pt",
                    help="Teacher checkpoint used for the student-vs-teacher action MSE (student eval only).")
parser.add_argument("--out_json", type=str, default=None,
                    help="Explicit output JSON path (overrides the default per-model filename).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.models import DepthStudentPolicy, TeacherMLPPolicy, load_student_policy  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402
from unitree_rl_lab.tasks.locomotion.robots.go2.teacher_env_cfg import PLAY_TERRAIN_CFG  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

MODEL_REGISTRY = {
    "blind": {
        "task": "Unitree-Go2-Velocity-BlindEval",
        "obs_group": "policy",
        "input_dim": 45,
        "default_ckpt": "logs/rsl_rl/unitree_go2_velocity/2026-09-22_18-57-19/model_49999.pt",
    },
    "teacher": {
        "task": "Unitree-Go2-Velocity-Teacher",
        "obs_group": "policy",
        "input_dim": 232,
        "default_ckpt": "logs/rsl_rl/unitree_go2_teacher/2026-10-10_15-00-10/model_19999.pt",
    },
    "student": {
        "task": "Unitree-Go2-Depth-Student",
        "obs_group": "student",
        "input_dim": 45,
        "default_ckpt": "logs/rsl_rl/unitree_go2_student_lstm_ca/2026-10-11_20-45-12_lstm_patch_ca_long_s42/student_model_9999.pt",
    },
}

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


def column_to_terrain_name():
    names = list(PLAY_TERRAIN_CFG.sub_terrains.keys())
    proportions = np.array([c.proportion for c in PLAY_TERRAIN_CFG.sub_terrains.values()], dtype=np.float64)
    proportions = proportions / proportions.sum()
    num_cols = PLAY_TERRAIN_CFG.num_cols
    mapping = []
    for col in range(num_cols):
        sub = int(np.min(np.where(col / num_cols + 0.001 < np.cumsum(proportions))[0]))
        mapping.append(names[sub])
    return mapping


def build_policy(model: str, device: str) -> torch.nn.Module:
    if model in ("blind", "teacher"):
        return TeacherMLPPolicy(MODEL_REGISTRY[model]["input_dim"], [512, 256, 128], 12, "elu")
    return DepthStudentPolicy(1, 32, 48, 45, 12)


def load_policy(model: str, ckpt_path: str, device: str) -> torch.nn.Module:
    if model == "student":
        policy = load_student_policy(ckpt_path, device, fusion_type=args_cli.fusion_type)
        policy.eval()
        return policy
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy = build_policy(model, device)
    policy.load_actor_state_dict(ckpt["actor_state_dict"])
    policy.to(device).eval()
    return policy


def policy_act(policy: torch.nn.Module, model: str, obs: dict) -> torch.Tensor:
    if model == "student":
        s = obs["student"]
        proprio_hist = torch.cat([s[k] for k in STUDENT_PROPRIO_KEYS], dim=-1)
        if policy.fusion_type == "lstm_patch_ca":
            proprio = proprio_hist  # (num_envs, 21, 45)
        else:
            proprio = proprio_hist[:, -1, :]  # (num_envs, 45)
        return policy(s["depth"], proprio)
    return policy(obs[MODEL_REGISTRY[model]["obs_group"]])


def gpu_mem_mb() -> float:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2
    except Exception:  # noqa: BLE001
        return float(torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0)


def load_teacher(path: str, device: str) -> TeacherMLPPolicy:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint has no 'actor_state_dict'; keys: {list(ckpt.keys())}")
    teacher = TeacherMLPPolicy(input_dim=232, hidden_dims=[512, 256, 128], action_dim=12, activation="elu")
    teacher.load_actor_state_dict(ckpt["actor_state_dict"])
    teacher.to(device).eval()
    return teacher


def evaluate_model(model: str, num_envs: int, rounds: int, seed: int) -> dict:
    meta = MODEL_REGISTRY[model]
    ckpt_path = args_cli.checkpoint or meta["default_ckpt"]
    env_cfg = parse_env_cfg(meta["task"], device=args_cli.device, num_envs=num_envs,
                            entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = seed
    env = gym.make(meta["task"], cfg=env_cfg)
    device = env.unwrapped.device
    policy = load_policy(model, ckpt_path, device)
    n_params = sum(p.numel() for p in policy.parameters())

    # Frozen Teacher for the student-vs-teacher action MSE comparison.
    teacher = None
    if model == "student":
        teacher = load_teacher(args_cli.teacher_ckpt, device)

    reward_manager = env.unwrapped.reward_manager
    termination_manager = env.unwrapped.termination_manager
    track_idx = reward_manager._term_names.index("track_lin_vel_xy")
    robot = env.unwrapped.scene["robot"]

    terrain_types = env.unwrapped.scene.terrain.terrain_types
    col_names = column_to_terrain_name()
    n_terrains = len(col_names)

    max_episode_length = env.unwrapped.max_episode_length

    total_eps = 0
    survived = 0
    died_bo = 0
    died_bc = 0
    per_terrain_total = [0] * n_terrains
    per_terrain_survived = [0] * n_terrains

    err_xy_sum = 0.0
    err_yaw_sum = 0.0
    track_sum = 0.0
    total_steps = 0

    # teacher action MSE accumulators (student eval only)
    teacher_mse_sum = 0.0
    teacher_mse_count = 0
    per_terrain_mse_sum = [0.0] * n_terrains
    per_terrain_mse_count = [0] * n_terrains

    # inference latency (forward wall time in ms)
    latencies: list[float] = []
    use_cuda_events = str(device).startswith("cuda")
    if use_cuda_events:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

    t_start = time.time()
    with torch.inference_mode():
        for rnd in range(rounds):
            obs, _ = env.reset()
            for _ in range(max_episode_length):
                if use_cuda_events:
                    start_ev.record()
                actions = policy_act(policy, model, obs)
                if use_cuda_events:
                    end_ev.record()
                    torch.cuda.synchronize()
                    latencies.append(start_ev.elapsed_time(end_ev))

                if teacher is not None:
                    teacher_a = teacher(obs["teacher"])
                    mse = ((actions - teacher_a) ** 2).mean(dim=-1)  # (num_envs,)
                    teacher_mse_sum += float(mse.sum().item())
                    teacher_mse_count += int(mse.shape[0])
                    for t in range(n_terrains):
                        mask = terrain_types == t
                        cnt = int(mask.sum().item())
                        if cnt:
                            per_terrain_mse_sum[t] += float(mse[mask].sum().item())
                            per_terrain_mse_count[t] += cnt

                obs, reward, term, trunc, extras = env.step(actions)

                cmd = env.unwrapped.command_manager.get_command("base_velocity")
                vel_xy = robot.data.root_lin_vel_b[:, :2]
                ang_z = robot.data.root_ang_vel_b[:, 2]
                err_xy_sum += torch.norm(cmd[:, :2] - vel_xy, dim=-1).sum().item()
                err_yaw_sum += torch.abs(cmd[:, 2] - ang_z).sum().item()
                track_sum += reward_manager._step_reward[:, track_idx].mean().item()
                total_steps += 1

                ended = term | trunc
                bo = termination_manager.get_term("bad_orientation")
                bc = termination_manager.get_term("base_contact")
                if int(ended.sum()):
                    total_eps += int(ended.sum())
                    survived += int((trunc & ~term).sum())
                    died_bo += int((term & bo).sum())
                    died_bc += int((term & ~bo & bc).sum())
                    for t in range(n_terrains):
                        mask = terrain_types == t
                        per_terrain_total[t] += int((ended & mask).sum())
                        per_terrain_survived[t] += int((trunc & ~term & mask).sum())

            print(f"[{model}] round {rnd + 1}/{rounds} done (elapsed {time.time() - t_start:.1f}s)", flush=True)

    elapsed = time.time() - t_start
    n_envs = env.unwrapped.num_envs
    peak_mem = gpu_mem_mb()

    def _lat_pct(p: float) -> float:
        if not latencies:
            return float("nan")
        return float(np.percentile(latencies, p))

    denom_steps = max(1, total_steps * n_envs)
    stats = {
        "task": meta["task"],
        "checkpoint": ckpt_path,
        "fusion_type": getattr(policy, "fusion_type", None),
        "num_params": n_params,
        "num_envs": n_envs,
        "rounds": rounds,
        "max_episode_length": max_episode_length,
        "total_episodes": total_eps,
        "time_out_pct": 100.0 * survived / max(1, total_eps),
        "bad_orientation_pct": 100.0 * died_bo / max(1, total_eps),
        "base_contact_pct": 100.0 * died_bc / max(1, total_eps),
        "error_vel_xy": err_xy_sum / denom_steps,
        "error_vel_yaw": err_yaw_sum / denom_steps,
        "track_lin_vel_xy": track_sum / max(1, total_steps),
        "teacher_action_mse": (teacher_mse_sum / max(1, teacher_mse_count)) if teacher is not None else None,
        "teacher_ckpt": args_cli.teacher_ckpt if teacher is not None else None,
        "forward_latency_mean_ms": float(np.mean(latencies)) if latencies else None,
        "forward_latency_p95_ms": _lat_pct(95),
        "forward_latency_p99_ms": _lat_pct(99),
        "peak_gpu_mb": peak_mem,
        "elapsed_s": elapsed,
        "per_terrain": {},
    }
    for t, name in enumerate(col_names):
        short = TERRAIN_SHORT.get(name, name)
        tot = per_terrain_total[t]
        surv = per_terrain_survived[t]
        per_t = {
            "name": name,
            "episodes": tot,
            "time_out_pct": 100.0 * surv / max(1, tot),
        }
        if teacher is not None:
            per_t["teacher_action_mse"] = per_terrain_mse_sum[t] / max(1, per_terrain_mse_count[t])
        stats["per_terrain"][short] = per_t

    env.close()
    return stats


def out_json_path(model: str) -> str:
    if args_cli.out_json:
        return args_cli.out_json
    tag = f"_{args_cli.tag}" if args_cli.tag else ""
    return os.path.join(args_cli.out_dir, f"eval_three_way_results{tag}_{model}.json")


def run_single(model: str):
    print(f"==== evaluating {model} ====", flush=True)
    stats = evaluate_model(model, args_cli.num_envs, args_cli.rounds, args_cli.seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_json_path(model))), exist_ok=True)
    with open(out_json_path(model), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[DONE] {model} -> {out_json_path(model)}", flush=True)
    print(f"[RESULT] {model}: time_out={stats['time_out_pct']:.1f}% "
          f"bad_ori={stats['bad_orientation_pct']:.1f}% base_contact={stats['base_contact_pct']:.1f}% "
          f"err_xy={stats['error_vel_xy']:.3f} err_yaw={stats['error_vel_yaw']:.3f} "
          f"track={stats['track_lin_vel_xy']:.3f} "
          f"teacher_mse={stats['teacher_action_mse']} "
          f"lat_ms={stats['forward_latency_mean_ms']}", flush=True)


def merge_results():
    models = ["blind", "teacher", "student"]
    all_stats = {}
    for model in models:
        with open(out_json_path(model)) as f:
            all_stats[model] = json.load(f)

    ter = ["stairs_down", "stairs_up", "boxes", "slope", "rough"]
    lines = []
    lines.append("# Three-way same-condition evaluation (blind / Teacher / Student)")
    lines.append("")
    lines.append(f"- seed={args_cli.seed}, num_envs={all_stats['blind']['num_envs']}, "
                 f"rounds={all_stats['blind']['rounds']}, "
                 f"max_episode_length={all_stats['blind']['max_episode_length']} steps (20 s)")
    lines.append(f"- terrain: {', '.join(ter)} (shared PLAY_TERRAIN_CFG)")
    lines.append(f"- generated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("| model | time_out% | bad_orientation% | base_contact% | error_vel_xy | error_vel_yaw | track_lin_vel_xy |")
    lines.append("|---|---|---|---|---|---|---|")
    for model in models:
        s = all_stats[model]
        lines.append(f"| {model} | {s['time_out_pct']:.1f} | {s['bad_orientation_pct']:.1f} | "
                     f"{s['base_contact_pct']:.1f} | {s['error_vel_xy']:.3f} | {s['error_vel_yaw']:.3f} | "
                     f"{s['track_lin_vel_xy']:.3f} |")
    lines.append("")
    lines.append("## Per-terrain survival (time_out%)")
    lines.append("")
    lines.append("| model | " + " | ".join(ter) + " |")
    lines.append("|" + "---|" * (len(ter) + 1))
    for model in models:
        s = all_stats[model]
        cells = [f"{s['per_terrain'][t]['time_out_pct']:.1f}" for t in ter]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```")
    tag_args = f" --tag {args_cli.tag}" if args_cli.tag else ""
    lines.append(f"~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/eval_three_way.py --headless --model blind{tag_args}")
    lines.append(f"~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/eval_three_way.py --headless --model teacher{tag_args}")
    lines.append(f"~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/eval_three_way.py --headless --model student{tag_args}")
    lines.append(f"~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/eval_three_way.py --merge{tag_args}")
    lines.append("```")
    lines.append("")
    md = "\n".join(lines) + "\n"
    tag = f"_{args_cli.tag}" if args_cli.tag else ""
    out_md = os.path.join(args_cli.out_dir, f"eval_three_way_results{tag}.md")
    with open(out_md, "w") as f:
        f.write(md)
    print(md)
    print(f"[INFO] wrote {out_md}")


def main():
    if args_cli.merge:
        merge_results()
        return
    if args_cli.model is None:
        raise SystemExit("Either --model {blind|teacher|student} or --merge is required.")
    run_single(args_cli.model)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    simulation_app.close()
