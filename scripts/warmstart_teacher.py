#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Warm-start the Teacher from a converged blind-walk checkpoint by zero-padding the
# actor/critic first-layer input weights to the new (privileged) observation dimension.
#
# Blind-walk obs dims (from logs/rsl_rl/unitree_go2_velocity/.../model_49999.pt):
#   actor  mlp.0.weight : (512, 45)
#   critic mlp.0.weight : (512, 60)
# Teacher obs dims (height_scan appended, H rays):
#   actor  mlp.0.weight : (512, 45 + H)
#   critic mlp.0.weight : (512, 60 + H)
#
# The new H columns are zero-initialised. Optimizer state is discarded (a fresh
# zero-momentum state matching the new parameter shapes is written so that the
# rsl-rl 5.4.2 ``runner.load`` default path still succeeds), and ``iter`` is reset to 0.

import argparse
from pathlib import Path

import torch


def expand_first_layer(old_weight: torch.Tensor, new_in_dim: int) -> torch.Tensor:
    """Copy the first ``old_in_dim`` columns and zero-pad the new trailing columns."""
    old_in_dim = old_weight.shape[1]
    assert new_in_dim >= old_in_dim, f"new_in_dim {new_in_dim} < old_in_dim {old_in_dim}"
    new_weight = torch.zeros(old_weight.shape[0], new_in_dim, dtype=old_weight.dtype)
    new_weight[:, :old_in_dim] = old_weight
    return new_weight


def build_fresh_optimizer_state(old_optimizer: dict, mapping: dict[tuple, tuple]) -> dict:
    """Rebuild a zeroed optimizer state whose momentum tensors match the new param shapes.

    Args:
        old_optimizer: the original ``optimizer_state_dict``.
        mapping: dict mapping old tensor shape -> new tensor shape for the params whose
            shape changed (the first layers). Unchanged shapes are kept as-is.

    Returns:
        A fresh optimizer state dict (``state`` + ``param_groups``) with zero momentum.
    """
    fresh_state = {}
    for idx, old_entry in old_optimizer["state"].items():
        exp_avg = old_entry["exp_avg"]
        new_shape = mapping.get(tuple(exp_avg.shape), tuple(exp_avg.shape))
        fresh_state[idx] = {
            "step": torch.tensor(0.0, dtype=torch.float32),
            "exp_avg": torch.zeros(new_shape, dtype=exp_avg.dtype),
            "exp_avg_sq": torch.zeros(new_shape, dtype=exp_avg.dtype),
        }
    return {"state": fresh_state, "param_groups": old_optimizer["param_groups"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start Teacher from blind-walk checkpoint.")
    parser.add_argument(
        "--input",
        type=str,
        default="logs/rsl_rl/unitree_go2_velocity/2026-09-22_18-57-19/model_49999.pt",
        help="Path to the converged blind-walk checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/rsl_rl/unitree_go2_teacher/warmstart/teacher_warmstart.pt",
        help="Output path for the warm-started Teacher checkpoint.",
    )
    parser.add_argument(
        "--height-dim",
        type=int,
        default=187,
        help="Number of height_scan rays H (GridPattern resolution 0.1, size [1.6, 1.0] -> 187).",
    )
    args = parser.parse_args()

    H = args.height_dim
    ckpt_path = Path(args.input)
    out_path = Path(args.output)

    print(f"[warmstart] loading blind-walk checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[warmstart] checkpoint keys: {list(ckpt.keys())}")

    actor = ckpt["actor_state_dict"]
    critic = ckpt["critic_state_dict"]

    actor_w = actor["mlp.0.weight"]
    critic_w = critic["mlp.0.weight"]
    old_actor_in = actor_w.shape[1]
    old_critic_in = critic_w.shape[1]
    print(f"[warmstart] old actor  mlp.0.weight shape: {tuple(actor_w.shape)}")
    print(f"[warmstart] old critic mlp.0.weight shape: {tuple(critic_w.shape)}")

    # sanity checks (blind-walk Go2 dims)
    assert old_actor_in == 45, f"expected actor input dim 45, got {old_actor_in}"
    assert old_critic_in == 60, f"expected critic input dim 60, got {old_critic_in}"

    new_actor_in = old_actor_in + H
    new_critic_in = old_critic_in + H
    print(f"[warmstart] H = {H} -> new actor input dim {new_actor_in}, new critic input dim {new_critic_in}")

    # expand first-layer input weights (zero-pad the trailing H columns)
    actor["mlp.0.weight"] = expand_first_layer(actor_w, new_actor_in)
    critic["mlp.0.weight"] = expand_first_layer(critic_w, new_critic_in)
    print(f"[warmstart] new actor  mlp.0.weight shape: {tuple(actor['mlp.0.weight'].shape)}")
    print(f"[warmstart] new critic mlp.0.weight shape: {tuple(critic['mlp.0.weight'].shape)}")

    # fresh (zeroed) optimizer state matching the new parameter shapes
    shape_mapping = {
        tuple(actor_w.shape): (actor_w.shape[0], new_actor_in),
        tuple(critic_w.shape): (critic_w.shape[0], new_critic_in),
    }
    optimizer = build_fresh_optimizer_state(ckpt["optimizer_state_dict"], shape_mapping)
    print(f"[warmstart] optimizer state: {len(optimizer['state'])} params, momentum zeroed")

    # write the warmstart checkpoint in the rsl-rl 5.4.2 layout
    out_dict = {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "optimizer_state_dict": optimizer,
        "iter": 0,
        "infos": None,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, out_path)
    print(f"[warmstart] saved warm-started checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
