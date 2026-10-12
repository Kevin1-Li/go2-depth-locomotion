# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class DistillationStudentTeacherCfg:
    """Student-teacher network configuration (rsl-rl >= 5.0.0 distillation API)."""

    class_name: str = "StudentTeacher"
    init_noise_std: float = 0.5
    noise_std_type: str = "scalar"
    student_obs_normalization: bool = False
    teacher_obs_normalization: bool = False
    student_hidden_dims: list[int] = [256, 128]
    teacher_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"


@configclass
class DistillationAlgorithmCfg:
    """Distillation (pure behaviour-cloning MSE) algorithm configuration (rsl-rl >= 5.0.0)."""

    class_name: str = "Distillation"
    num_learning_epochs: int = 2
    learning_rate: float = 1.0e-3
    gradient_length: int = 15
    max_grad_norm: float | None = 1.0
    optimizer: str = "adam"
    loss_type: str = "mse"


@configclass
class DistillationRunnerCfg:
    """Distillation runner configuration (rsl-rl >= 5.0.0 API).

    Mirrors the anymal_d distillation example but with the Go2-specific experiment name,
    iteration budget and observation-group mapping. The ``teacher`` group maps to the
    privileged 232-D group and the ``policy`` group maps to the deployable depth+proprio
    group of the ``Unitree-Go2-Depth-Student`` task.
    """

    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 120
    max_iterations: int = 3000
    save_interval: int = 100
    experiment_name: str = "unitree_go2_student"
    run_name: str = ""
    logger: str = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float | None = None
    obs_groups: dict[str, list[str]] = {"policy": ["student"], "teacher": ["teacher"]}
    policy: DistillationStudentTeacherCfg = DistillationStudentTeacherCfg()
    algorithm: DistillationAlgorithmCfg = DistillationAlgorithmCfg()
