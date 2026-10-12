# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class BlindEvalPPORunnerCfg(BasePPORunnerCfg):
    """PPO runner configuration for the blind-walk eval task.

    Identical to ``BasePPORunnerCfg`` (rsl-rl 5.4.2 separate actor/critic API, MLP
    512-256-128) except ``experiment_name`` is pinned to ``unitree_go2_velocity`` so
    play.py resolves checkpoints from the blind-walk training logs.
    """

    experiment_name: str = "unitree_go2_velocity"
