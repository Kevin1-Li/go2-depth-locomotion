# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class TeacherPPORunnerCfg(BasePPORunnerCfg):
    """PPO runner configuration for the privileged Teacher.

    Identical to the blind-walk ``BasePPORunnerCfg`` (rsl-rl 5.4.2 separate actor/critic
    API) except for the experiment folder name and the default iteration budget. The
    first-layer *input* dimension is widened automatically at runtime by ``MLPModel``
    (only ``hidden_dims`` is declared here), so no network changes are required to
    absorb the extra ``height_scan`` observation columns.
    """

    experiment_name: str = "unitree_go2_teacher"
    max_iterations: int = 30000
