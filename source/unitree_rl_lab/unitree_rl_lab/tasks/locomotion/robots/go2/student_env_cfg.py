# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Depth-perception Student environment for the Teacher -> Student distillation pipeline.
#
# The world / terrain / events / terminations / commands are inherited verbatim from
# ``TeacherRobotEnvCfg``. On top of that we (1) mount a pinhole ``RayCasterCamera`` depth
# sensor on the robot head, and (2) re-organise the observations into two groups:
#
#   * ``teacher``  : the exact privileged policy observation the converged Teacher consumed
#                    (45-D proprio + 187-D height_scan = 232-D), used only to query the
#                    frozen Teacher for behaviour-cloning labels.
#   * ``student``  : the deployable observation (45-D proprio with a 21-frame history window
#                    + 32x48 depth image), used as the Student policy's input.

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

from .teacher_env_cfg import (
    PLAY_INIT_TERRAIN_LEVEL,
    PLAY_TERRAIN_CFG,
    TeacherRobotEnvCfg,
    TeacherRobotSceneCfg,
)

##
# Depth camera (team real-robot convention: horizontal FOV 120 deg, vertical FOV ~60 deg,
# far clamp 10 m). The pinhole pattern is parameterised by focal length + apertures, where
#   fov = 2 * atan(aperture / (2 * focal_length)).
##

DEPTH_CAMERA_FOCAL_LENGTH = 24.0
DEPTH_CAMERA_WIDTH = 48
DEPTH_CAMERA_HEIGHT = 32
DEPTH_CAMERA_HFOV_DEG = 120.0
DEPTH_CAMERA_VFOV_DEG = 60.0
DEPTH_CAMERA_FAR_CLIP = 10.0
DEPTH_CAMERA_PITCH_DOWN_DEG = 35.0

# Ordered proprio history window (frames) for the LSTM branch of the ``lstm_patch_ca``
# Student. Applied per-term via ``history_length`` so the depth image (same group) stays
# single-frame. ``flatten_history_dim=False`` keeps each term as (num_envs, T, dim).
PROPRIO_HISTORY_LENGTH = 21


def _aperture_for_fov(fov_deg: float) -> float:
    return 2.0 * DEPTH_CAMERA_FOCAL_LENGTH * math.tan(math.radians(fov_deg) / 2.0)


def _pitch_down_quat(pitch_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(pitch_deg) / 2.0
    return (math.cos(half), 0.0, math.sin(half), 0.0)


@configclass
class StudentRobotSceneCfg(TeacherRobotSceneCfg):
    """Teacher scene plus a head-mounted pinhole depth camera.

    The camera is attached to the robot ``base`` (the articulation root, which is the same
    prim the existing ``height_scanner`` mounts on, so view resolution is guaranteed) and
    offset forward to the head position ``(0.285, 0, 0.01)`` from ``go2_description.urdf``.
    It is pitched ~35 deg downward so that, with a 60 deg vertical FOV, it observes the
    ground from ~5 deg to ~65 deg below the horizon: enough near-field for foot placement
    and enough forward lookahead for the ~1 m/s command ceiling.
    """

    depth_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.285, 0.0, 0.01),
            rot=_pitch_down_quat(DEPTH_CAMERA_PITCH_DOWN_DEG),
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        max_distance=DEPTH_CAMERA_FAR_CLIP,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=DEPTH_CAMERA_FOCAL_LENGTH,
            horizontal_aperture=_aperture_for_fov(DEPTH_CAMERA_HFOV_DEG),
            vertical_aperture=_aperture_for_fov(DEPTH_CAMERA_VFOV_DEG),
            width=DEPTH_CAMERA_WIDTH,
            height=DEPTH_CAMERA_HEIGHT,
        ),
    )


@configclass
class StudentObservationsCfg:
    """Observations for the Student: a privileged ``teacher`` group and a deployable
    ``student`` group.

    The ``teacher`` group is a strict replica of ``TeacherObservationsCfg.PolicyCfg``
    (identical term order, scales, clips and noise parameterisation => identical 232-D
    vector) but with corruption disabled so the frozen Teacher produces clean,
    deterministic behaviour-cloning labels. The ``student`` group carries the same noisy
    45-D proprio terms plus the normalised depth image (concatenation disabled so the image
    stays separate from the vector terms).
    """

    @configclass
    class TeacherGroupCfg(ObsGroup):
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
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StudentGroupCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=PROPRIO_HISTORY_LENGTH, flatten_history_dim=False,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY_LENGTH, flatten_history_dim=False,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"},
            history_length=PROPRIO_HISTORY_LENGTH, flatten_history_dim=False,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=PROPRIO_HISTORY_LENGTH, flatten_history_dim=False,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=PROPRIO_HISTORY_LENGTH, flatten_history_dim=False,
        )
        last_action = ObsTerm(
            func=mdp.last_action, clip=(-100, 100),
            history_length=PROPRIO_HISTORY_LENGTH, flatten_history_dim=False,
        )
        depth = ObsTerm(
            func=mdp.depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "far_clip": DEPTH_CAMERA_FAR_CLIP},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    teacher: TeacherGroupCfg = TeacherGroupCfg()
    student: StudentGroupCfg = StudentGroupCfg()


@configclass
class StudentRobotEnvCfg(TeacherRobotEnvCfg):
    """Depth-perception Student environment configuration.

    Identical to the Teacher environment except for the depth camera and the re-organised
    observation groups. All MDP components (actions, commands, rewards, terminations,
    events, curriculum) are inherited unchanged so the Student distills against the same
    world the Teacher mastered.
    """

    scene: StudentRobotSceneCfg = StudentRobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_camera.update_period = self.decimation * self.sim.dt


@configclass
class StudentRobotPlayEnvCfg(StudentRobotEnvCfg):
    """Play (evaluation) configuration for the Student.

    Swaps the terrain generator to the shared ``PLAY_TERRAIN_CFG`` so the Student is
    replayed on the exact same complex terrain as the Teacher and ``BlindEval`` tasks.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator = PLAY_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = PLAY_INIT_TERRAIN_LEVEL
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
