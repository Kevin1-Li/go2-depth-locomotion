#!/usr/bin/env python3
"""Merge per-run ``logs/ca_exp/results_{arm}_{seed}.json`` into ``summary.md``.

Cross-seed mean +- std is reported per arm. Per-terrain survival (time_out%) is listed for
the 5 shared PLAY_TERRAIN_CFG columns (stairs_down / stairs_up / boxes / slope / rough; the
play terrain has no flat tier).

Usage (system python3, no Isaac Lab needed):
    python3 scripts/rsl_rl/merge_ca_results.py --out_dir logs/ca_exp
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics

TERRAINS = ["stairs_down", "stairs_up", "boxes", "slope", "rough"]


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _fmt(v: float, nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return "-"
    return f"{v:.{nd}f}"


def load_results(out_dir: str) -> dict[str, dict]:
    results = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "results_*.json"))):
        base = os.path.basename(path)
        m = re.match(r"results_(concat|xattn)_(\d+)\.json", base)
        if not m:
            continue
        arm, seed = m.group(1), m.group(2)
        with open(path) as f:
            results[f"{arm}_{seed}"] = json.load(f)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="logs/ca_exp")
    args = parser.parse_args()

    results = load_results(args.out_dir)
    if not results:
        print("[merge_ca_results] no results found; nothing to merge.")
        return

    arms = sorted({k.rsplit("_", 1)[0] for k in results})
    lines = ["# Cross-Attention vs Concat (CNN baseline) — fixed same-condition evaluation", ""]
    lines.append("- Teacher: v4 `model_19999.pt` (frozen behaviour-cloning labeler)")
    lines.append("- Student env: `Unitree-Go2-Depth-Student` play config (shared `PLAY_TERRAIN_CFG`)")
    lines.append("- Protocol: deterministic mean action, seed=42, 64 envs x 3 rounds, "
                 "max_episode_length=1000 steps")
    lines.append("- Terrain columns: stairs_down / stairs_up / boxes / slope / rough (no flat tier)")
    lines.append("")
    lines.append("## Per-run results")
    lines.append("")
    lines.append("| arm | seed | time_out% | bad_ori% | base_contact% | error_vel_xy | error_vel_yaw | "
                 "teacher_mse | lat_mean_ms | lat_p95_ms | lat_p99_ms | params | peak_gpu_mb |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(results):
        s = results[key]
        arm, seed = key.rsplit("_", 1)
        lines.append(
            f"| {arm} | {seed} | {_fmt(s.get('time_out_pct'))} | {_fmt(s.get('bad_orientation_pct'))} | "
            f"{_fmt(s.get('base_contact_pct'))} | {_fmt(s.get('error_vel_xy'), 3)} | "
            f"{_fmt(s.get('error_vel_yaw'), 3)} | {_fmt(s.get('teacher_action_mse'), 4)} | "
            f"{_fmt(s.get('forward_latency_mean_ms'), 2)} | {_fmt(s.get('forward_latency_p95_ms'), 2)} | "
            f"{_fmt(s.get('forward_latency_p99_ms'), 2)} | {s.get('num_params')} | "
            f"{_fmt(s.get('peak_gpu_mb'), 0)} |"
        )
    lines.append("")

    lines.append("## Per-terrain survival (time_out%)")
    lines.append("")
    lines.append("| arm | seed | " + " | ".join(TERRAINS) + " |")
    lines.append("|" + "---|" * (len(TERRAINS) + 2))
    for key in sorted(results):
        s = results[key]
        arm, seed = key.rsplit("_", 1)
        pt = s.get("per_terrain", {})
        cells = [_fmt(pt.get(t, {}).get("time_out_pct")) for t in TERRAINS]
        lines.append(f"| {arm} | {seed} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-terrain teacher action MSE")
    lines.append("")
    lines.append("| arm | seed | " + " | ".join(TERRAINS) + " |")
    lines.append("|" + "---|" * (len(TERRAINS) + 2))
    for key in sorted(results):
        s = results[key]
        arm, seed = key.rsplit("_", 1)
        pt = s.get("per_terrain", {})
        cells = [_fmt(pt.get(t, {}).get("teacher_action_mse"), 3) for t in TERRAINS]
        lines.append(f"| {arm} | {seed} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Cross-seed mean ± std")
    lines.append("")
    lines.append("| arm | time_out% | bad_ori% | base_contact% | error_vel_xy | error_vel_yaw | "
                 "teacher_mse | lat_mean_ms |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm in arms:
        keys = [f"{arm}_{s}" for s in sorted({k.rsplit('_', 1)[1] for k in results if k.startswith(arm + '_')})]
        seeds = [results[k] for k in keys]

        def ms(metric):
            vals = [s.get(metric) for s in seeds if s.get(metric) is not None]
            m, sd = _mean_std(vals)
            return f"{_fmt(m)} ± {_fmt(sd)}"

        lines.append(
            f"| {arm} | {ms('time_out_pct')} | {ms('bad_orientation_pct')} | {ms('base_contact_pct')} | "
            f"{ms('error_vel_xy')} | {ms('error_vel_yaw')} | {ms('teacher_action_mse')} | "
            f"{ms('forward_latency_mean_ms')} |"
        )
    lines.append("")

    md = "\n".join(lines) + "\n"
    out_md = os.path.join(args.out_dir, "summary.md")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    print(md)
    print(f"[merge_ca_results] wrote {os.path.abspath(out_md)}")


if __name__ == "__main__":
    main()
