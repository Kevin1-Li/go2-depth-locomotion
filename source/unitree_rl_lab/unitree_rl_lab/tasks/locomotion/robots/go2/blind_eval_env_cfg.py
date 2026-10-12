# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Blind-walk evaluation variant for the "blind-walk vs Teacher" side-by-side replay.
#
# Observation / action / reward / termination / command / event / curriculum are
# INHERITED UNCHANGED from the baseline ``velocity_env_cfg.py`` (policy 45-dim,
# critic 60-dim), so the blind-walk ``model_49999.pt`` checkpoint loads directly
# without any observation-dimension mismatch. The only difference is the terrain:
# the baseline flat terrain is replaced by the shared complex ``PLAY_TERRAIN_CFG``
# that the Teacher play config also references.

from isaaclab.utils import configclass

from .velocity_env_cfg import RobotEnvCfg
from .teacher_env_cfg import PLAY_TERRAIN_CFG, PLAY_INIT_TERRAIN_LEVEL


@configclass
class BlindEvalEnvCfg(RobotEnvCfg):
    """Blind-walk environment replayed on the shared complex play terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = PLAY_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = PLAY_INIT_TERRAIN_LEVEL


@configclass
class BlindEvalPlayEnvCfg(BlindEvalEnvCfg):
    """Play (evaluation) configuration for the blind-walk baseline."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
