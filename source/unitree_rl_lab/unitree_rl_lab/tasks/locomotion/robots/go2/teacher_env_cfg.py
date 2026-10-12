# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Teacher (privileged) variant of the Go2 velocity-tracking task for the
# "blind-walk warmstart Teacher -> depth Student distillation" pipeline.
#
# Differences vs. the baseline ``velocity_env_cfg.py`` (which is left untouched):
#   1. Complex terrain: flat + random_rough + slope(+inv) + boxes + stairs(+inv),
#      with flat proportion kept >= 0.2 so the warmstarted blind-walk policy has
#      an easy transition surface. difficulty_range (0.0, 1.0) and the existing
#      terrain-levels curriculum (terrain_levels_vel) remain active.
#   2. Privileged GT height-map observation (``height_scan``) appended to BOTH the
#      actor (policy) and critic observation groups. At deployment this privileged
#      signal is replaced by the Student's learned depth estimation.

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    RewardsCfg,
    TerminationsCfg,
)

##
# Complex terrain generator for the Teacher.
##
TEACHER_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.2, 1.0),
    use_cache=False,
    sub_terrains={
        # keep flat >= 0.2 for a smooth warmstart transition from the blind-walk policy
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        # Go2-scaled rough terrain (noise amplitude kept small for a 0.4 m tall robot)
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        # Go2-scaled boxes (0.025-0.1 m, down from the 0.05-0.2 m default for larger robots)
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.025, 0.1), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.03, 0.12),
            step_width=0.25,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.03, 0.12),
            step_width=0.25,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


##
# Shared play/eval terrain.
#
# This generator is referenced by BOTH the Teacher play config (below) and the
# blind-walk ``BlindEval`` task, so the two models are replayed on IDENTICAL
# terrain for a side-by-side visual comparison.
#
# Design notes:
#   * Only the visually informative obstacle types are kept (stairs down / stairs up /
#     boxes / slope / rough), proportions split evenly so all five show up when num_cols=5.
#   * difficulty_range is narrowed to (0.4, 0.9) so every spawn row is already in
#     the medium-high difficulty band (no mostly-flat default rows).
#   * curriculum=True orders rows by difficulty (row 0 easiest -> row 5 hardest),
#     which is what maps ``max_init_terrain_level`` to spawn difficulty.
#   * num_rows/num_cols are small (play doesn't need a large training grid).
##
PLAY_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=6,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.4, 0.9),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.03, 0.12),
            step_width=0.25,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.03, 0.12),
            step_width=0.25,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.025, 0.1), platform_width=2.0
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
    },
)

# Spawn envs across all difficulty rows (num_rows=6 => levels 0..5). Because the
# difficulty_range is already (0.4, 0.9), every row is medium-high difficulty, so
# envs always spawn on representative complex terrain (never a flat default row).
PLAY_INIT_TERRAIN_LEVEL = 5


@configclass
class TeacherRobotSceneCfg(InteractiveSceneCfg):
    """Scene for the Teacher: baseline scene but with the complex terrain generator.

    The height scanner is kept exactly as in the baseline (GridPattern resolution 0.1,
    size [1.6, 1.0] => 187 rays). A 1.6 m forward lookahead at the Go2 command ceiling
    of 1.0 m/s gives ~80 control steps of anticipation, which is ample for the stair /
    slope / box obstacles here; keeping the original footprint also keeps the warmstart
    height-scan dimension modest (H=187) and minimises rollout memory on the 8 GB GPU.
    """

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TEACHER_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class TeacherObservationsCfg:
    """Observations for the Teacher: blind-walk terms + privileged height_scan.

    ``height_scan`` is appended LAST in both groups, so the H new columns land at the
    end of each observation vector — this is exactly where the warmstart zero-padding
    script appends its new (zero-initialised) columns.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the actor (policy) group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # privileged GT height map (no noise: this is the Teacher's clean ground-truth
        # signal that the Student will later regress against)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for the critic (value) group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # privileged GT height map (same as actor, no noise)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class TeacherCurriculumCfg:
    """Curriculum terms for the Teacher.

    Same command-tracking curriculum as the baseline, but the terrain-level term is replaced
    by :func:`terrain_levels_vel_forward`, which only lets forward-dominant episodes promote or
    demote. This keeps turning/standing commands from dragging well-tracked robots back down.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_forward)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class TeacherRobotEnvCfg(ManagerBasedRLEnvCfg):
    """Teacher environment configuration (privileged height map + complex terrain)."""

    # Scene settings
    scene: TeacherRobotSceneCfg = TeacherRobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: TeacherCurriculumCfg = TeacherCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # commands: start at full speed range (same as Student/BlindEval). The warmstart
        # blind-walk policy is already full-speed competent, and the "start slow"
        # lin_vel_cmd_levels bootstrap otherwise stalls terrain promotion when the
        # difficulty floor is raised above 0.0 (net displacement stays sub-metre).
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class TeacherRobotPlayEnvCfg(TeacherRobotEnvCfg):
    """Play (evaluation) configuration for the Teacher.

    Swaps the terrain generator to the shared ``PLAY_TERRAIN_CFG`` so the Teacher is
    replayed on the exact same complex terrain as the ``BlindEval`` task.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator = PLAY_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = PLAY_INIT_TERRAIN_LEVEL
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
